#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .month_utils import MONTHS, T
from .model import LSTMChangeDetector, infer_arch_from_state_dict


def load_scaler_stats(path: Path):
    arr = np.load(path)
    mean = arr.get("mean")
    scale = arr.get("scale")
    monthly_mean = arr.get("monthly_mean")
    monthly_scale = arr.get("monthly_scale")
    if mean is None or scale is None:
        raise ValueError(f"scaler_stats.npz missing mean/scale: {path}")
    scale = scale.astype(np.float64)
    scale[scale == 0.0] = 1.0
    if monthly_mean is not None and monthly_scale is not None:
        ms = monthly_scale.astype(np.float64)
        ms[ms == 0.0] = 1.0
        return mean.astype(np.float64), scale, monthly_mean.astype(np.float64), ms
    return mean.astype(np.float64), scale, None, None


def normalize_features(X: np.ndarray, mean: np.ndarray, scale: np.ndarray, monthly_mean=None, monthly_scale=None) -> np.ndarray:
    if monthly_mean is not None and monthly_scale is not None and monthly_mean.shape == X.shape:
        Z = (X - monthly_mean) / monthly_scale
    else:
        Z = (X - mean) / scale
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z.astype(np.float32)


def parse_args():
    ap = argparse.ArgumentParser("Infer weakly-supervised monthly probabilities for all months")
    ap.add_argument("--features_csv", type=Path, required=True)
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--scaler_path", type=Path, required=True)
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument(
        "--group_cols",
        type=str,
        default="site_name",
        help="Comma-separated columns that uniquely identify an instance (default: site_name). For global grids use: site_name,grid_id",
    )
    ap.add_argument(
        "--meta_cols",
        type=str,
        default="",
        help="Comma-separated columns to copy (first value per group) into output (e.g., lon,lat).",
    )
    ap.add_argument("--mode", type=str, default="weakly_supervised")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    def _split_cols(s: str) -> list[str]:
        return [c.strip() for c in str(s).split(",") if c.strip()]

    group_cols = _split_cols(args.group_cols)
    if not group_cols:
        group_cols = ["site_name"]
    meta_cols = _split_cols(args.meta_cols)

    df = pd.read_csv(args.features_csv)
    if "month" not in df.columns:
        raise ValueError("features_csv must contain a 'month' column")
    for c in group_cols:
        if c not in df.columns:
            raise ValueError(f"features_csv missing required group column: {c}")
    for c in meta_cols:
        if c not in df.columns:
            raise ValueError(f"features_csv missing requested meta column: {c}")

    df["month"] = df["month"].astype(str).str.replace("-", "_", regex=False)
    df["month"] = df["month"].str.extract(r"(20\d{2}_\d{2})", expand=False)
    df = df[df["month"].isin(MONTHS)].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering to 2017_01..2024_12")

    feat_cols = sorted([c for c in df.columns if c.startswith("f")], key=lambda x: int(x[1:]))
    if not feat_cols:
        raise ValueError("No feature columns (f*) found")

    # Load model state dict and infer architecture
    sd = torch.load(args.model_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError("model_path must contain a state_dict")

    arch = infer_arch_from_state_dict(sd)
    if args.verbose:
        print(f"[info] inferred arch: {arch}")

    needed_F = arch["input_dim"]
    if len(feat_cols) < needed_F:
        for i in range(len(feat_cols), needed_F):
            df[f"f{i}"] = 0.0
        feat_cols = feat_cols + [f"f{i}" for i in range(len(feat_cols), needed_F)]
    elif len(feat_cols) > needed_F:
        feat_cols = feat_cols[:needed_F]

    mean, scale, monthly_mean, monthly_scale = load_scaler_stats(args.scaler_path)
    if mean.shape[0] != needed_F:
        raise ValueError(f"Scaler dimension {mean.shape[0]} != model input_dim {needed_F}")

    # monthly stats alignment
    if monthly_mean is not None and monthly_mean.shape != (T, needed_F):
        if args.verbose:
            print("[warn] monthly_mean shape mismatch; ignoring monthly stats")
        monthly_mean = None
        monthly_scale = None

    model = LSTMChangeDetector(
        input_dim=arch["input_dim"],
        enc_hidden=arch["enc_hidden"],
        lstm_hidden=arch["lstm_hidden"],
        lstm_layers=arch["lstm_layers"],
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    grp = df.groupby(group_cols, sort=False)
    keys = list(grp.groups.keys())

    rows = []
    with torch.inference_mode():
        batch_keys = []
        batch_meta = []
        batch_X = []

        def flush():
            if not batch_keys:
                return
            X = np.stack(batch_X, axis=0)  # (B,T,F)
            inp = torch.tensor(X, dtype=torch.float32, device=device)
            t_logits = model.forward_time_logits(inp).detach().cpu().numpy()  # (B,T)
            probs = 1.0 / (1.0 + np.exp(-t_logits))
            for bi, key in enumerate(batch_keys):
                row = {}
                if len(group_cols) == 1:
                    row[group_cols[0]] = key
                else:
                    for ci, c in enumerate(group_cols):
                        row[c] = key[ci]
                for mc, mv in batch_meta[bi].items():
                    row[mc] = mv
                row["mode"] = args.mode
                for mi, mm in enumerate(MONTHS):
                    row[mm] = float(probs[bi, mi])
                rows.append(row)
            batch_keys.clear(); batch_meta.clear(); batch_X.clear()

        for key in tqdm(keys, desc="infer instances", dynamic_ncols=True):
            sdf = grp.get_group(key)
            meta = {c: sdf.iloc[0][c] for c in meta_cols} if meta_cols else {}
            mdf = sdf.set_index("month").reindex(MONTHS)
            X = mdf[feat_cols].to_numpy(dtype=np.float64)
            mask = ~np.isfinite(X)
            if monthly_mean is not None:
                X = np.where(mask, monthly_mean, X)
            else:
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            Z = normalize_features(X, mean, scale, monthly_mean, monthly_scale)
            batch_keys.append(key)
            batch_meta.append(meta)
            batch_X.append(Z)
            if len(batch_keys) >= int(args.batch_size):
                flush()
        flush()

    out_cols = group_cols + meta_cols + ["mode"] + MONTHS
    out_df = pd.DataFrame(rows, columns=out_cols)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"[ok] wrote {args.output_csv} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
