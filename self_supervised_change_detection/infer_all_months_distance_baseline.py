#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Infer distance-baseline monthly probabilities for all months.

This exports a single per-instance row with month columns 2017_01..2024_12.

For each instance (group), it:
- builds a (T,F) feature matrix from a unified features CSV
- applies the *trained* scaler stats (Afghanistan-trained) from scaler_stats.npz
- computes a distance-baseline score per month: distance(current, median(prev 1..K))
- converts scores to probability-like values using the same normalization family
  used elsewhere in the project (sigmoid/minmax/softmax)

For global grids, use group columns: site_name,grid_id and meta columns lon,lat.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .datasets import MONTHS, T, load_scaler_npz
from .utils import normalize_scores, robust_zscore


def _split_cols(s: str) -> list[str]:
    return [c.strip() for c in str(s).split(",") if c.strip()]


def _ensure_month_str(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("-", "_", regex=False)
    s = s.str.extract(r"(\d{4}_\d{2})", expand=False)
    return s


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = float(np.sqrt(np.sum(a * a)))
    nb = float(np.sqrt(np.sum(b * b)))
    if na < eps or nb < eps:
        return 0.0
    return float(1.0 - (np.dot(a, b) / (na * nb)))


def parse_args():
    ap = argparse.ArgumentParser("Infer distance-baseline monthly probabilities for all months")
    ap.add_argument("--features_csv", type=Path, required=True)
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

    ap.add_argument("--k_prior", type=int, default=3, help="Number of previous months used for the baseline (default: 3).")
    ap.add_argument("--distance", type=str, default="l2", choices=["l2", "cosine"])
    ap.add_argument("--robust_scaler", action="store_true")
    ap.add_argument("--disable_per_month_feature_norm", action="store_true")
    ap.add_argument("--rw_window", type=int, default=12, help="Rolling window size for robust z-score calibration (default: 12). Set to 0 to disable.")
    ap.add_argument("--negative_z_ok", action="store_true", help="Allow negative z-scores (default: clip to positive).")

    ap.add_argument(
        "--score_norm_method",
        type=str,
        default="sigmoid",
        choices=["none", "minmax", "sigmoid", "softmax"],
        help="Convert baseline scores to probability-like values per instance.",
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

    stats = load_scaler_npz(str(args.scaler_path), len(feat_cols))
    input_dim = int(stats["mean"].shape[0])

    if len(feat_cols) < input_dim:
        for i in range(len(feat_cols), input_dim):
            df[f"f{i}"] = 0.0
        feat_cols = feat_cols + [f"f{i}" for i in range(len(feat_cols), input_dim)]
    elif len(feat_cols) > input_dim:
        feat_cols = feat_cols[:input_dim]

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

    rows: list[dict] = []
    idx = np.arange(T)
    k_prior = max(1, int(args.k_prior))

    dist_fn = _l2 if args.distance == "l2" else _cosine

    for key, sdf in tqdm(grp, desc="prepare+infer", dynamic_ncols=True):
        if isinstance(key, tuple) and len(group_cols) == 1:
            key = key[0]
        meta = {c: sdf.iloc[0][c] for c in meta_cols} if meta_cols else {}

        mat = np.full((T, input_dim), np.nan, dtype=np.float32)
        for _, r in sdf.iterrows():
            mm = r.get("month")
            if not isinstance(mm, str) or mm not in MONTHS:
                continue
            ti = MONTHS.index(mm)
            mat[ti, :] = r[feat_cols].to_numpy(dtype=np.float32, copy=False)

        # Interpolate missing months per feature; fall back to global mean if entire series missing.
        for j in range(input_dim):
            col = mat[:, j]
            mask = np.isfinite(col)
            if mask.any():
                mat[:, j] = np.interp(idx, idx[mask], col[mask]).astype(np.float32)
            else:
                mat[:, j] = mean[j]

        # Normalize using trained scaler stats (Afghanistan-trained).
        if month_mean is not None and month_std is not None:
            mmu = month_mean.astype(np.float32)
            msd = month_std.astype(np.float32)
            msd[msd == 0] = 1.0
            mat = (mat - mmu) / msd

        std2 = std.copy()
        std2[std2 == 0] = 1.0
        mat = (mat - mean) / std2
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        scores = np.zeros((T,), dtype=np.float32)
        for t in range(T):
            if t == 0:
                scores[t] = 0.0
                continue
            a = max(0, t - k_prior)
            baseline = np.median(mat[a:t, :], axis=0)
            scores[t] = float(dist_fn(mat[t, :], baseline))

        if int(args.rw_window) > 0:
            scores = robust_zscore(scores, window=int(args.rw_window), positive_only=(not args.negative_z_ok))
        probs = normalize_scores(scores, method=args.score_norm_method, temperature=float(args.score_norm_temperature))
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)

        row = {}
        if len(group_cols) == 1:
            row[group_cols[0]] = key
        else:
            for ci, c in enumerate(group_cols):
                row[c] = key[ci]
        for mc, mv in meta.items():
            row[mc] = mv
        row["mode"] = "distance_baseline"
        for mi, mm in enumerate(MONTHS):
            row[mm] = float(probs[mi])
        rows.append(row)



    out_cols = group_cols + meta_cols + ["mode"] + MONTHS
    out_df = pd.DataFrame(rows, columns=out_cols)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"[ok] wrote {args.output_csv} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
