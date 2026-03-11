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
from .utils import knn_density, causal_knn_density, delta_signal, normalize_scores, robust_zscore, cusum_change_score, population_knn_novelty


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
    causal = bool(ckpt.get("causal", False))

    mae = MaskedAutoencoder(
        input_dim,
        d_model=d_model,
        depth=depth,
        nhead=nhead,
        ff=ff,
        dropout=dropout,
        mask_ratio=mask_ratio,
        causal=causal,
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
        causal=causal,
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
        causal=causal,
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

    ap.add_argument("--k_prior", type=int, default=3,
                    help="Causal reference window: number of previous months used by --use_causal_knn and --use_delta_scores.")
    ap.add_argument("--use_causal_knn", action="store_true",
                    help="Replace symmetric KNN density with causal (past-only) KNN density.")
    ap.add_argument("--use_delta_scores", action="store_true",
                    help="Replace raw rec/fore errors with delta vs. rolling median of previous k_prior months.")

    ap.add_argument("--use_pop_knn", action="store_true",
                    help="Add cross-site population KNN novelty as a 4th fusion signal (requires two passes).")
    ap.add_argument("--k_pop", type=int, default=10,
                    help="Number of nearest population neighbours for cross-site KNN (default: 10).")
    ap.add_argument("--alpha_pop", type=float, default=0.4,
                    help="Weight for cross-site population KNN novelty signal in fusion (default: 0.4).")
    ap.add_argument("--delta_pop_knn", action="store_true",
                    help="Apply delta_signal to population KNN scores (change in population distance vs. recent months) "
                         "instead of absolute distance, reducing bias from geographically unusual sites.")

    ap.add_argument("--use_cusum", action="store_true",
                    help="Apply one-sided CUSUM to the fused score before normalization to reward persistent change.")
    ap.add_argument("--cusum_drift", type=float, default=0.5,
                    help="CUSUM drift parameter (slack per step). Larger = less sensitive to small sustained changes.")

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

    # --- Pass 0 (optional): collect all latents for cross-site population KNN ---
    pop_novelty_dict: dict = {}  # key -> (T,) float32 array
    if args.use_pop_knn:
        all_pop_keys: list = []
        all_pop_Z: list[np.ndarray] = []

        @torch.inference_mode()
        def _collect_latents(b_keys, b_X):
            xt = torch.tensor(np.stack(b_X, axis=0), dtype=torch.float32, device=device)
            z = trunk(xt).detach().cpu().numpy()  # (B, T, d)
            for i, k in enumerate(b_keys):
                all_pop_keys.append(k)
                all_pop_Z.append(z[i])

        _batch_keys_p0: list = []
        _batch_X_p0: list = []
        for key_p0, sdf_p0 in tqdm(grp, desc="pass0 latents", dynamic_ncols=True):
            if isinstance(key_p0, tuple) and len(group_cols) == 1:
                key_p0 = key_p0[0]
            _batch_keys_p0.append(key_p0)
            _batch_X_p0.append(build_matrix(sdf_p0))
            if len(_batch_keys_p0) >= int(args.batch_size):
                _collect_latents(_batch_keys_p0, _batch_X_p0)
                _batch_keys_p0, _batch_X_p0 = [], []
        if _batch_keys_p0:
            _collect_latents(_batch_keys_p0, _batch_X_p0)

        all_Z_arr = np.stack(all_pop_Z, axis=0)  # (N, T, d)
        if args.verbose:
            print(f"[pop_knn] computing {args.k_pop}-NN over {all_Z_arr.shape[0]} sites × {T} months")
        pop_scores_arr = population_knn_novelty(all_Z_arr, k=int(args.k_pop))  # (N, T)
        for i, k in enumerate(all_pop_keys):
            pop_novelty_dict[k] = pop_scores_arr[i]

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
            if args.use_causal_knn:
                z_np = z[bi].detach().cpu().numpy()  # (T, d)
                dens = causal_knn_density(z_np, k=int(args.k_density), k_prior=int(args.k_prior))
            else:
                dens = knn_density(z[bi], k=int(args.k_density)).detach().cpu().numpy()
            nov = (dens - dens.min()) / (dens.max() - dens.min() + 1e-8)
            nov = 1.0 - nov
            nov_list.append(nov.astype(np.float32))
        novelty = np.stack(nov_list, axis=0)

        # Robust rolling z and fuse
        # When using CUSUM, allow negative z-scores so the CUSUM can reset during
        # normal months. Positive-clipping would make fused >= 0 always, causing the
        # CUSUM to grow monotonically (never resetting) and peak at end-of-series.
        positive_only_z = (not args.negative_z_ok) and (not args.use_cusum)
        for bi in range(len(batch_keys)):
            rec_i = delta_signal(rec_err[bi], k_prior=int(args.k_prior)) if args.use_delta_scores else rec_err[bi]
            fore_i = delta_signal(fore_err[bi], k_prior=int(args.k_prior)) if args.use_delta_scores else fore_err[bi]
            rec_rw = robust_zscore(rec_i, window=int(args.rw_window), positive_only=positive_only_z)
            fore_rw = robust_zscore(fore_i, window=int(args.rw_window), positive_only=positive_only_z)
            nov_rw = robust_zscore(novelty[bi], window=int(args.rw_window), positive_only=positive_only_z)
            fused = (
                float(args.alpha_rec) * rec_rw
                + float(args.alpha_fore) * fore_rw
                + float(args.alpha_novel) * nov_rw
            ).astype(np.float32)

            if args.use_pop_knn:
                key_bi = batch_keys[bi]
                pop_nov = pop_novelty_dict.get(key_bi)
                if pop_nov is not None:
                    if args.delta_pop_knn:
                        pop_nov = delta_signal(pop_nov, k_prior=int(args.k_prior))
                    pop_rw = robust_zscore(pop_nov, window=int(args.rw_window), positive_only=positive_only_z)
                    fused = fused + float(args.alpha_pop) * pop_rw

            if args.use_cusum:
                fused = cusum_change_score(fused, drift=float(args.cusum_drift))

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

    for key, sdf in tqdm(grp, desc="prepare+infer", dynamic_ncols=True):
        if isinstance(key, tuple) and len(group_cols) == 1:
            key = key[0]
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
