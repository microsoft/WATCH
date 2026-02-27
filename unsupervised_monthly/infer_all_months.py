#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Infer unsupervised monthly probabilities for all months.

This is a deployment-style exporter for a trained unsupervised ensemble
(`unsup_models.pt` + `scaler_stats.npz`).

It reads a unified features CSV with rows:
  - month (YYYY_MM)
  - f0..fF
  - one or more instance identifier columns (default: site_name)

For global grids, use group columns like: site_name,grid_id and optionally
preserve lon/lat via meta columns.

Outputs a single CSV with one row per instance and month probability columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .datasets import MONTHS, T, load_scaler_npz
from .models import MaskedAutoencoder, NextMonthForecaster, TemporalTransformer
from .utils import knn_density, normalize_scores, robust_zscore


def _split_cols(s: str) -> list[str]:
    return [c.strip() for c in str(s).split(",") if c.strip()]


def _ensure_month_str(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("-", "_", regex=False)
    s = s.str.extract(r"(\d{4}_\d{2})", expand=False)
    return s


def _load_models(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("unsup_models.pt must be a checkpoint dict")

    input_dim = int(ckpt["input_dim"])
    d_model = int(ckpt.get("d_model", 256))
    depth = int(ckpt.get("depth", 4))
    nhead = int(ckpt.get("nhead", 8))
    ff = int(ckpt.get("ff", 512))
    dropout = float(ckpt.get("dropout", 0.1))
    mask_ratio = float(ckpt.get("mask_ratio", 0.3))

    mae = MaskedAutoencoder(
        input_dim,
        d_model=d_model,
        depth=depth,
        nhead=nhead,
        ff=ff,
        dropout=dropout,
        mask_ratio=mask_ratio,
    ).to(device)
    mae.load_state_dict(ckpt["mae"], strict=True)
    mae.eval()

    fore = NextMonthForecaster(
        input_dim,
        d_model=d_model,
        depth=max(1, depth - 1),
        nhead=nhead,
        ff=ff,
        dropout=dropout,
    ).to(device)
    fore.load_state_dict(ckpt["fore"], strict=True)
    fore.eval()

    trunk = TemporalTransformer(
        input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=depth,
        dim_feedforward=ff,
        dropout=dropout,
    ).to(device)
    trunk.load_state_dict(ckpt["trunk"], strict=True)
    trunk.eval()

    proj = torch.nn.Sequential(
        torch.nn.Linear(d_model, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
    ).to(device)
    proj.load_state_dict(ckpt["proj"], strict=True)
    proj.eval()

    return input_dim, mae, fore, trunk, proj


def parse_args():
    ap = argparse.ArgumentParser("Infer unsupervised monthly probabilities for all months")
    ap.add_argument("--features_csv", type=Path, required=True)
    ap.add_argument("--trained_model", type=Path, required=True)
    ap.add_argument("--scaler_path", type=Path, required=True)
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument(
        "--group_cols",
        type=str,
        default="site_name",
        help="Comma-separated instance key columns (default: site_name). For global grids use: site_name,grid_id",
    )
    ap.add_argument(
        "--meta_cols",
        type=str,
        default="",
        help="Comma-separated columns to copy (first value per group) into output (e.g., lon,lat).",
    )

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)

    ap.add_argument("--k_density", type=int, default=5)
    ap.add_argument("--alpha_rec", type=float, default=0.6)
    ap.add_argument("--alpha_fore", type=float, default=0.3)
    ap.add_argument("--alpha_novel", type=float, default=0.4)
    ap.add_argument("--rw_window", type=int, default=12)
    ap.add_argument("--negative_z_ok", action="store_true")

    ap.add_argument("--robust_scaler", action="store_true")
    ap.add_argument("--disable_per_month_feature_norm", action="store_true")

    ap.add_argument(
        "--score_norm_method",
        type=str,
        default="sigmoid",
        choices=["none", "minmax", "sigmoid", "softmax"],
        help="Convert fused scores to probability-like values per instance.",
    )
    ap.add_argument("--score_norm_temperature", type=float, default=1.0)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    group_cols = _split_cols(args.group_cols)
    if not group_cols:
        group_cols = ["site_name"]
    meta_cols = _split_cols(args.meta_cols)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    df = pd.read_csv(args.features_csv)
    if "month" not in df.columns:
        raise ValueError("features_csv must contain a 'month' column")
    for c in group_cols:
        if c not in df.columns:
            raise ValueError(f"features_csv missing required group column: {c}")
    for c in meta_cols:
        if c not in df.columns:
            raise ValueError(f"features_csv missing requested meta column: {c}")

    df["month"] = _ensure_month_str(df["month"])
    df = df[df["month"].isin(MONTHS)].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering to 2017_01..2024_12")

    feat_cols = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    if not feat_cols:
        raise ValueError("No feature columns (f*) found")

    # Load model + scaler
    input_dim, mae, fore, trunk, _proj = _load_models(args.trained_model, device=device)
    if len(feat_cols) < input_dim:
        for i in range(len(feat_cols), input_dim):
            df[f"f{i}"] = 0.0
        feat_cols = feat_cols + [f"f{i}" for i in range(len(feat_cols), input_dim)]
    elif len(feat_cols) > input_dim:
        feat_cols = feat_cols[:input_dim]

    stats = load_scaler_npz(str(args.scaler_path), input_dim)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    month_mean = stats.get("month_mean")
    month_std = stats.get("month_std")
    if args.disable_per_month_feature_norm:
        month_mean = None
        month_std = None

    if args.robust_scaler and ("feat_median" in stats and "feat_mad" in stats):
        mean = stats["feat_median"].astype(np.float32)
        mad = stats["feat_mad"].astype(np.float32)
        mad[mad == 0] = 1.0
        std = (mad * 1.4826).astype(np.float32)

    if month_mean is not None and month_std is not None and month_mean.shape != (T, input_dim):
        month_mean = None
        month_std = None

    grp = df.groupby(group_cols, sort=False)
    keys = list(grp.groups.keys())

    def build_matrix(sdf: pd.DataFrame) -> np.ndarray:
        mat = np.full((T, input_dim), np.nan, dtype=np.float32)
        for _, r in sdf.iterrows():
            mm = r.get("month")
            if not isinstance(mm, str) or mm not in MONTHS:
                continue
            ti = MONTHS.index(mm)
            mat[ti, :] = r[feat_cols].to_numpy(dtype=np.float32, copy=False)

        idx = np.arange(T)
        for j in range(input_dim):
            col = mat[:, j]
            mask = np.isfinite(col)
            if mask.any():
                mat[:, j] = np.interp(idx, idx[mask], col[mask]).astype(np.float32)
            else:
                mat[:, j] = mean[j]

        if month_mean is not None and month_std is not None:
            mmu = month_mean.astype(np.float32)
            msd = month_std.astype(np.float32)
            msd[msd == 0] = 1.0
            mat = (mat - mmu) / msd

        mat = (mat - mean) / std
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return mat

    rows: list[dict] = []

    batch_keys: list = []
    batch_meta: list[dict] = []
    batch_X: list[np.ndarray] = []

    @torch.inference_mode()
    def flush():
        nonlocal batch_keys, batch_meta, batch_X
        if not batch_keys:
            return
        X = np.stack(batch_X, axis=0)  # (B,T,F)
        xt = torch.tensor(X, dtype=torch.float32, device=device)

        # Reconstruction error
        _, recon, _ = mae(xt)
        rec_err = ((recon - xt) ** 2).mean(dim=-1).detach().cpu().numpy()  # (B,T)

        # Forecast error
        _, pred = fore(xt)
        target = xt[:, 1:, :]
        err_t = ((pred - target) ** 2).mean(dim=-1)  # (B,T-1)
        fore_err = torch.zeros((xt.size(0), T), device=xt.device)
        if err_t.numel() > 0:
            fore_err[:, :-1] = err_t
            fore_err[:, -1] = err_t[:, -1]
        fore_err = fore_err.detach().cpu().numpy()

        # Latent novelty via kNN density over time
        z = trunk(xt)  # (B,T,d)
        nov_list = []
        for bi in range(z.size(0)):
            dens = knn_density(z[bi], k=int(args.k_density)).detach().cpu().numpy()
            nov = (dens - dens.min()) / (dens.max() - dens.min() + 1e-8)
            nov = 1.0 - nov
            nov_list.append(nov.astype(np.float32))
        novelty = np.stack(nov_list, axis=0)

        # Robust rolling z and fuse
        for bi in range(len(batch_keys)):
            rec_rw = robust_zscore(rec_err[bi], window=int(args.rw_window), positive_only=(not args.negative_z_ok))
            fore_rw = robust_zscore(fore_err[bi], window=int(args.rw_window), positive_only=(not args.negative_z_ok))
            nov_rw = robust_zscore(novelty[bi], window=int(args.rw_window), positive_only=(not args.negative_z_ok))
            fused = (
                float(args.alpha_rec) * rec_rw
                + float(args.alpha_fore) * fore_rw
                + float(args.alpha_novel) * nov_rw
            ).astype(np.float32)

            probs = normalize_scores(
                fused,
                method=str(args.score_norm_method),
                temperature=float(args.score_norm_temperature),
            ).astype(np.float32)

            row: dict = {}
            key = batch_keys[bi]
            if len(group_cols) == 1:
                row[group_cols[0]] = key
            else:
                for ci, c in enumerate(group_cols):
                    row[c] = key[ci]
            for mc, mv in batch_meta[bi].items():
                row[mc] = mv
            row["mode"] = "learned_unsupervised"
            for mi, mm in enumerate(MONTHS):
                row[mm] = float(probs[mi])
            rows.append(row)

        batch_keys = []
        batch_meta = []
        batch_X = []

    for key in tqdm(keys, desc="prepare+infer", dynamic_ncols=True):
        sdf = grp.get_group(key)
        meta = {c: sdf.iloc[0][c] for c in meta_cols} if meta_cols else {}
        X = build_matrix(sdf)
        batch_keys.append(key)
        batch_meta.append(meta)
        batch_X.append(X)
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
