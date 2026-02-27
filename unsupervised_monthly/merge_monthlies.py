#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Merge per-month CSVs into a single aggregated matrix.

Canonical modes:
- distance_baseline
- learned_unsupervised

Per-month inputs (canonical filenames):
- distance_baseline: monthly_distance_baseline_<embedding>_<YYYY>_<MM>_<split>.csv
- learned_unsupervised: monthly_learned_unsupervised_<YYYY>_<MM>_<split>.csv

Backwards compatibility:
- Also accepts legacy per-month filenames with monthly_baseline_* and monthly_unsupervised_*
- Also accepts legacy mode args: baseline -> distance_baseline, unsupervised -> learned_unsupervised

Writes `unsup_month_scores_<split>_<mode>.csv` with columns [site_name, mode, YYYY_MM...]

Usage:
    python -m unsupervised_monthly.merge_monthlies \
        --out_dir unsupervised_monthly/model_runs/handcrafted \
        --mode distance_baseline --split all --remove
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np

from .mode_utils import CANONICAL_MODES, is_legacy_mode, normalize_mode


def parse_args():
    ap = argparse.ArgumentParser("Merge monthly CSVs into aggregated matrix")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory containing monthly_<mode>_*_<split>.csv files")
    ap.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=list(CANONICAL_MODES) + ["baseline", "unsupervised"],
        help="Which per-month files to merge (canonical: distance_baseline|learned_unsupervised).",
    )
    ap.add_argument("--split", type=str, default="all", help="Split suffix used in filenames (e.g., all, train, val, test)")
    ap.add_argument("--suffix", type=str, default="", help="Optional suffix to append before .csv in output filename, e.g., _new")
    ap.add_argument("--normalize", action="store_true", help="Normalize per-site month series to [0,1] (e.g., sigmoid/minmax/softmax)")
    ap.add_argument("--norm_method", type=str, default="none", choices=["none","minmax","sigmoid","softmax"], help="Normalization method for --normalize")
    ap.add_argument("--norm_temperature", type=float, default=1.0, help="Temperature for sigmoid/softmax normalization")
    ap.add_argument("--remove", action="store_true", help="Remove per-month files after merging")
    return ap.parse_args()


def month_key(m: str):
    try:
        y = int(m[:4]); mm = int(m[5:7])
        return (y, mm)
    except Exception:
        return (9999, 99)


def merge_monthlies(out_dir: str, mode: str, split: str, remove: bool, suffix: str = "", normalize: bool = False, norm_method: str = "none", norm_temperature: float = 1.0):
    outp = Path(out_dir)
    if not outp.exists():
        raise FileNotFoundError(f"Missing out_dir: {out_dir}")

    if is_legacy_mode(mode):
        print(f"[warn] Legacy mode '{mode}' is deprecated; use '{normalize_mode(mode)}' instead")
    mode = normalize_mode(mode)

    if mode not in CANONICAL_MODES:
        raise ValueError(f"Unknown mode: {mode}. Supported: {list(CANONICAL_MODES)}")

    # Patterns
    if mode == "distance_baseline":
        # Accept any embedding token between '<mode>' and the date.
        pats = [
            re.compile(r"^monthly_distance_baseline_.*_([0-9]{4})_([0-9]{2})_" + re.escape(split) + r"\.csv$"),
            # Legacy
            re.compile(r"^monthly_baseline_.*_([0-9]{4})_([0-9]{2})_" + re.escape(split) + r"\.csv$"),
        ]
    else:  # learned_unsupervised
        pats = [
            re.compile(r"^monthly_learned_unsupervised_([0-9]{4})_([0-9]{2})_" + re.escape(split) + r"\.csv$"),
            # Legacy
            re.compile(r"^monthly_unsupervised_([0-9]{4})_([0-9]{2})_" + re.escape(split) + r"\.csv$"),
        ]

    files = []
    for name in os.listdir(out_dir):
        if any(p.match(name) for p in pats):
            files.append(name)
    if not files:
        print(f"[merge] No per-month files found for mode={mode} split={split} in {out_dir}")
        return

    files = sorted(files)
    print(f"[merge] Found {len(files)} monthly files")

    # Build aggregated frame
    agg = None
    months = []
    for fname in files:
        fp = outp / fname
        df = pd.read_csv(fp)
        # Expect columns: site_name, month, (score or fused_score), optional mode
        if "site_name" not in df.columns or "month" not in df.columns:
            raise ValueError(f"Invalid monthly file (missing site_name/month): {fp}")
        # Choose value column
        val_col = "fused_score" if "fused_score" in df.columns else ("score" if "score" in df.columns else None)
        if val_col is None:
            raise ValueError(f"Invalid monthly file (missing score columns): {fp}")
        mstr = str(df["month"].iloc[0])
        months.append(mstr)
        ser = df.set_index("site_name")[val_col].astype(np.float32)
        if agg is None:
            agg = pd.DataFrame(index=ser.index)
            agg["mode"] = mode
        # Align index union
        union_idx = agg.index.union(ser.index)
        agg = agg.reindex(union_idx)
        ser = ser.reindex(union_idx)
        agg[mstr] = ser.values

    # Order columns by month
    months = sorted(set(months), key=month_key)
    agg = agg.reset_index().rename(columns={"index": "site_name"})
    cols = ["site_name","mode"] + months
    agg = agg.reindex(columns=cols)

    # Reduce residual NaNs by carry-forward along months (row-wise)
    if agg is not None and len(months) > 0:
        # Forward-fill from previous months across the row
        mm_df = agg[months]
        mm_df = mm_df.ffill(axis=1)
        # For first month (e.g., 2017_01), if still NaN, copy from month[1] (e.g., 2017_02)
        try:
            first = months[0]
            if len(months) > 1:
                second = months[1]
                mm_df[first] = mm_df[first].where(~mm_df[first].isna(), mm_df[second])
        except Exception:
            pass
        agg[months] = mm_df

    # Optionally normalize per-site vectors across month columns
    if normalize and agg is not None and len(months) > 0:
        # Local copy to avoid importing torch via utils
        def normalize_scores(x, method: str = "none", temperature: float = 1.0, eps: float = 1e-6):
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            if method == "none":
                return x
            if method == "minmax":
                mn, mx = np.nanmin(x), np.nanmax(x)
                den = (mx - mn)
                if not np.isfinite(den) or den < eps:
                    return np.zeros_like(x)
                return (x - mn) / (den + eps)
            if method == "sigmoid":
                mu = np.nanmean(x); sd = np.nanstd(x)
                sd = sd if (np.isfinite(sd) and sd > 0) else 1.0
                z = (x - mu) / (sd * max(temperature, eps))
                return 1.0 / (1.0 + np.exp(-z))
            if method == "softmax":
                t = max(temperature, eps); y = x / t; y = y - np.nanmax(y)
                e = np.exp(y); den = np.nansum(e)
                if not np.isfinite(den) or den < eps:
                    n = x.size; return np.full_like(x, 1.0 / max(n, 1))
                return e / (den + eps)
            return x
        # Apply row-wise
        # Keep site_name/mode intact
        site_col = agg["site_name"].values
        mode_col = agg["mode"].values if "mode" in agg.columns else None
        M = agg[months].to_numpy(dtype=np.float32)
        out = np.zeros_like(M, dtype=np.float32)
        for i in range(M.shape[0]):
            out[i, :] = normalize_scores(M[i, :], method=norm_method, temperature=norm_temperature)
        agg[months] = out

    # Write aggregated file
    suffix = suffix or ""
    out_file = outp / f"unsup_month_scores_{split}_{mode}{suffix}.csv"
    agg.to_csv(out_file, index=False)
    print(f"[merge] Wrote aggregated: {out_file}")

    # Remove per-month files if requested
    if remove:
        for fname in files:
            try:
                (outp / fname).unlink()
            except Exception as e:
                print(f"[warn] Could not remove {fname}: {e}")
        print(f"[merge] Removed {len(files)} monthly files")


def main():
    args = parse_args()
    merge_monthlies(args.out_dir, args.mode, args.split, args.remove, args.suffix, args.normalize, args.norm_method, args.norm_temperature)


if __name__ == "__main__":
    main()
