#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Monthly deployment-style change detection.
For a target month t (YYYY_MM), compute Baseline(t) as the median of available
prior months {t-1, t-2, t-3} per site (with carry-forward imputation), then
Change(t) = distance(Emb(t), Baseline(t)).

Inputs: yearly CSVs under planet_mosaics_final_4bands/features_new_with_mask/
 naming: features_<embedding>_<year>_masked.csv

Outputs:
- Per-month CSV: monthly_change_<embedding>_<YYYY>_<MM>_<split>.csv
    - With `--output_components`, includes columns: `t_score`, `cgz_score`, `cgs_score`, `fused_score`.
- Aggregated matrix CSV in output_dir: unsup_month_scores_<split>.csv
    with month-named columns (96) storing the fused score per site (one row per `site_name`).

Usage examples:
  Single month:
    python -m self_supervised_change_detection.monthly_inference \
      --embedding dinov3 --year 2019 --month 02 \
      --split test --groundtruth_csv path/to/gt.csv \
      --output_dir ./self_supervised_change_detection_outputs

  Batch (see run_monthly_batch.py) processes Jan 2017..Dec 2024.
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

from .month_utils import MONTHS, MONTH_TO_INDEX, prev_months, normalize_month_str
from .utils import normalize_scores
from .distances import l2_distance, cosine_distance

DEFAULT_FEATURES_DIR = "planet_mosaics_final_4bands/features_new_with_mask"

# Resolve package directory for robust default paths regardless of CWD
PKG_DIR = Path(__file__).resolve().parent

# Optional: reuse scaler stats loader from local monthly package
try:
    from .datasets import load_scaler_npz
except Exception:
    load_scaler_npz = None

def _feature_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith('f') and c[1:].isdigit()], key=lambda x: int(x[1:]))

def _load_year(embedding: str, year: int, features_dir: str) -> pd.DataFrame | None:
    fp = Path(features_dir) / f"features_{embedding}_{year}_masked.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    if 'month' not in df.columns or 'site_name' not in df.columns:
        raise ValueError(f"Missing required columns in {fp}")
    return df

def _load_unified(features_unified_csv: str) -> pd.DataFrame:
    fp = Path(features_unified_csv)
    if not fp.exists():
        raise FileNotFoundError(f"Unified features CSV not found: {fp}")
    df = pd.read_csv(fp)
    if 'month' not in df.columns or 'site_name' not in df.columns:
        raise ValueError(f"Missing required columns in {fp}")
    return df

def _sites_from_groundtruth(gt_csv: str | None, split: str) -> List[str]:
    if gt_csv and os.path.exists(gt_csv):
        gdf = pd.read_csv(gt_csv)
        scol = 'split' if 'split' in gdf.columns else None
        if scol is None:
            return sorted(gdf['site_name'].unique().tolist())
        if split in ('train','val','test'):
            return sorted(gdf[gdf[scol] == split]['site_name'].unique().tolist())
        elif split == 'all':
            return sorted(gdf['site_name'].unique().tolist())
        elif split == 'all_looted':
            if 'looted' in gdf.columns:
                try:
                    mask = gdf['looted'].astype(int) == 1
                except Exception:
                    mask = gdf['looted'] == 1
                return sorted(gdf[mask]['site_name'].unique().tolist())
            else:
                raise ValueError("'all_looted' requested but 'looted' column missing in groundtruth")
        else:
            raise ValueError("split must be one of train/val/test/all/all_looted")
    return []

def _carry_forward(site: str, y: int, m: int, df_by_year: Dict[int, pd.DataFrame], fcols: List[str]) -> np.ndarray | None:
    # Search backwards within loaded yearly data for the most recent prior available month for this site
    yy, mm = y, m
    for _ in range(60):  # up to 5 years back
        mm -= 1
        if mm == 0:
            yy -= 1
            mm = 12
        if yy < 2017:
            break
        d = df_by_year.get(yy)
        if d is None:
            continue
        row = d[(d['site_name'] == site) & (d['month'] == f"{yy}_{mm:02d}")]
        if len(row) == 1:
            return row.iloc[0][fcols].to_numpy(dtype=np.float32)
    return None

def _get_month_vec(site: str, y: int, m: int, df_by_year: Dict[int, pd.DataFrame], fcols: List[str]) -> np.ndarray | None:
    d = df_by_year.get(y)
    if d is None:
        return _carry_forward(site, y, m, df_by_year, fcols)
    row = d[(d['site_name'] == site) & (d['month'] == f"{y}_{m:02d}")]
    if len(row) == 1:
        return row.iloc[0][fcols].to_numpy(dtype=np.float32)
    # Impute via carry-forward
    return _carry_forward(site, y, m, df_by_year, fcols)

def _carry_forward_unified(site: str, t_index: int, df_unified: pd.DataFrame, fcols: List[str]) -> np.ndarray | None:
    for idx in range(t_index - 1, max(-1, t_index - 60), -1):
        if idx < 0:
            break
        mn = MONTHS[idx]
        row = df_unified[(df_unified['site_name'] == site) & (df_unified['month'] == mn)]
        if len(row) == 1:
            return row.iloc[0][fcols].to_numpy(dtype=np.float32)
    return None

def _get_month_vec_unified(site: str, y: int, m: int, df_unified: pd.DataFrame, fcols: List[str]) -> np.ndarray | None:
    mn = f"{y}_{m:02d}"
    row = df_unified[(df_unified['site_name'] == site) & (df_unified['month'] == mn)]
    if len(row) == 1:
        return row.iloc[0][fcols].to_numpy(dtype=np.float32)
    # Impute via carry-forward within unified CSV
    t_index = MONTHS.index(mn) if mn in MONTHS else None
    if t_index is None:
        return None
    return _carry_forward_unified(site, t_index, df_unified, fcols)

def _apply_normalization(x: np.ndarray, t_index: int, m_index: int, stats: dict | None,
                         per_month_feature_norm: bool, robust: bool) -> np.ndarray:
    """
    Apply two-stage normalization similar to unsupervised pipeline:
      1) calendar-month normalization (12xF or TxF) if available and enabled
      2) global feature standardization (mean/std or robust median/MAD)
    x: (F,)
    t_index: 0..95 month index (YYYY_MM across 2017..2024)
    m_index: 0..11 calendar month index
    """
    if stats is None:
        return x.astype(np.float32, copy=False)
    # Global stats
    mean = None; std = None
    if robust and ('feat_median' in stats and 'feat_mad' in stats):
        mean = stats.get('feat_median')
        mad = stats.get('feat_mad')
        if mad is not None:
            mad = mad.astype(np.float32)
            mad[mad == 0] = 1.0
            std = (mad * 1.4826).astype(np.float32)
    if mean is None or std is None:
        mean = stats.get('mean') or stats.get('feat_mean')
        std = stats.get('std') or stats.get('feat_std')
    if mean is None or std is None:
        mean = np.zeros_like(x, dtype=np.float32)
        std = np.ones_like(x, dtype=np.float32)

    y = x.astype(np.float32)
    # Per-calendar-month normalization
    if per_month_feature_norm:
        mm = None; ms = None
        # Accept either 96xF month stats or 12xF calendar stats
        if 'month_mean' in stats and 'month_std' in stats:
            mm = stats['month_mean']; ms = stats['month_std']
            if isinstance(mm, np.ndarray) and mm.shape[0] == len(MONTHS):
                y = (y - mm[t_index]) / (ms[t_index] + 1e-8)
            else:
                mm = None; ms = None
        if mm is None or ms is None:
            pm_mean = stats.get('per_month_mean'); pm_std = stats.get('per_month_std')
            if isinstance(pm_mean, np.ndarray) and pm_mean.shape[0] == 12:
                y = (y - pm_mean[m_index]) / (pm_std[m_index] + 1e-8)
    # Global standardization
    y = (y - mean) / (std + 1e-8)
    return y

def compute_monthly_change(embedding: str, year: int, month: int, features_dir: str,
                           sites: List[str], distance: str = 'l2',
                           scaler_path: str | None = None,
                           features_unified_csv: str | None = None,
                           per_month_feature_norm: bool = True,
                           robust: bool = False,
                           intra_site_grid_calibration: bool = False,
                           month_of_year_seasonal: bool = False,
                           enable_cross_grid: bool = False,
                           w_fuse_temporal: float = 1.0,
                           w_fuse_cgz: float = 1.0,
                           w_fuse_cgs: float = 0.5,
                           output_components: bool = False,
                           aggregated_scores_path: str | None = None,
                           rolling_window: int = 6) -> pd.DataFrame:
    # Choose data source: unified CSV or per-year masked directory
    df_unified: pd.DataFrame | None = None
    df_by_year: Dict[int, pd.DataFrame] = {}
    ref_df: pd.DataFrame | None = None
    if features_unified_csv:
        df_unified = _load_unified(features_unified_csv)
        ref_df = df_unified
        if not sites:
            sites = sorted(df_unified['site_name'].unique().tolist())
    else:
        # Load target year and potential previous year
        df_cur = _load_year(embedding, year, features_dir)
        df_prev = _load_year(embedding, year-1, features_dir) if month <= 3 else None
        if df_cur is not None:
            df_by_year[year] = df_cur
        if df_prev is not None:
            df_by_year[year-1] = df_prev
        # If sites not provided, derive from available features
        if not sites:
            pool = []
            for d in df_by_year.values():
                pool.extend(d['site_name'].unique().tolist())
            sites = sorted(list(set(pool)))
        ref_df = df_cur if df_cur is not None else df_prev
    # Feature columns from chosen df
    if ref_df is None:
        raise FileNotFoundError(f"No feature CSVs found for embedding={embedding} around {year}_{month:02d}")
    fcols = _feature_cols(ref_df)
    # Prepare baseline months list (up to 3)
    prior = prev_months(year, month, k=3)
    # Load scaler stats if available
    stats = None
    if load_scaler_npz is not None and scaler_path:
        try:
            stats = load_scaler_npz(scaler_path, len(fcols))
        except Exception as e:
            print(f"[warn] Failed to load scaler stats from {scaler_path}: {e}")
    # Compute change per site (one row per site_name)
    rows = []
    for s in sites:
        if df_unified is not None:
            xt = _get_month_vec_unified(s, year, month, df_unified, fcols)
        else:
            xt = _get_month_vec(s, year, month, df_by_year, fcols)
        if xt is None:
            # No target and no prior; skip or set NaN
            rows.append((s, f"{year}_{month:02d}", np.nan))
            continue
        # Apply normalization on target
        t_index = MONTHS.index(f"{year}_{month:02d}")
        m_index = (month - 1)
        xt_n = _apply_normalization(xt, t_index, m_index, stats, per_month_feature_norm, robust)
        bs_list = []
        for (yy, mm) in prior:
            if df_unified is not None:
                xb = _get_month_vec_unified(s, yy, mm, df_unified, fcols)
            else:
                xb = _get_month_vec(s, yy, mm, df_by_year, fcols)
            if xb is not None:
                b_t_index = MONTHS.index(f"{yy}_{mm:02d}")
                b_m_index = (mm - 1)
                xb_n = _apply_normalization(xb, b_t_index, b_m_index, stats, per_month_feature_norm, robust)
                bs_list.append(xb_n)
        if len(bs_list) == 0:
            # No baseline available; change undefined
            if output_components:
                rows.append((s, f"{year}_{month:02d}", np.nan, np.nan, np.nan, np.nan))
            else:
                rows.append((s, f"{year}_{month:02d}", np.nan))
            continue
        baseline = np.median(np.stack(bs_list, axis=0), axis=0)
        if distance == 'cosine':
            t_score = cosine_distance(xt_n, baseline)
        else:
            t_score = l2_distance(xt_n, baseline)

        # Optional month-of-year seasonal normalization across sites for current calendar month
        # Deferred until we have all sites; will apply after loop.
        rows.append((s, f"{year}_{month:02d}", float(t_score)))

    # Build base dataframe
    df_scores = pd.DataFrame(rows, columns=['site_name','month','t_score'])

    # Month-of-year seasonal normalization: robust z across sites for current month
    if month_of_year_seasonal:
        vals = df_scores['t_score'].to_numpy(dtype=np.float32)
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        scale = (mad * 1.4826) if mad > 0 else (np.nanstd(vals) + 1e-8)
        df_scores['t_score'] = (df_scores['t_score'] - med) / (scale + 1e-8)
        # Keep only positive anomalies
        df_scores['t_score'] = df_scores['t_score'].clip(lower=0.0)

    # Rolling robust z-score per site over previous months using aggregated scores if available
    if aggregated_scores_path and os.path.exists(aggregated_scores_path) and rolling_window > 0:
        try:
            agg = pd.read_csv(aggregated_scores_path)
            cur_month = f"{year}_{month:02d}"
            # Determine previous W months names
            pm_list = [f"{yy}_{mm:02d}" for (yy, mm) in prev_months(year, month, k=rolling_window)]
            for i, row in df_scores.iterrows():
                sname = row['site_name']
                base_row = agg[agg['site_name'] == sname]
                if len(base_row) == 1:
                    prev_vals = []
                    for mn in pm_list:
                        if mn in base_row.columns:
                            val = base_row.iloc[0][mn]
                            if not pd.isna(val):
                                prev_vals.append(float(val))
                    if len(prev_vals) >= 3:
                        prev_vals = np.array(prev_vals, dtype=np.float32)
                        med = np.nanmedian(prev_vals)
                        mad = np.nanmedian(np.abs(prev_vals - med))
                        scale = (mad * 1.4826) if mad > 0 else (np.nanstd(prev_vals) + 1e-8)
                        z = (row['t_score'] - med) / (scale + 1e-8)
                        df_scores.at[i, 't_score'] = max(0.0, float(z))
        except Exception as e:
            print(f"[warn] rolling z failed: {e}")

    # Intra-site grid calibration: robust z across grids for the same site and month.
    # If features are one row per site (no grids), this is effectively a no-op.
    if intra_site_grid_calibration:
        # Group by site_name; robust z within the group
        groups = df_scores.groupby('site_name')
        cgz = []
        for sname, g in groups:
            v = g['t_score'].to_numpy(dtype=np.float32)
            med = np.nanmedian(v)
            mad = np.nanmedian(np.abs(v - med))
            scale = (mad * 1.4826) if mad > 0 else (np.nanstd(v) + 1e-8)
            z = (v - med) / (scale + 1e-8)
            z = np.clip(z, 0.0, None)
            cgz.extend(z.tolist())
        df_scores['cgz_score'] = cgz
    else:
        df_scores['cgz_score'] = 0.0

    # Cross-grid softmax per site (spatial saliency across grids)
    if enable_cross_grid:
        cgs_vals = []
        for _, g in df_scores.groupby('site_name'):
            x = g['t_score'].to_numpy(dtype=np.float32)
            # Stable softmax
            x = x - np.max(x)
            ex = np.exp(x)
            sm = ex / (np.sum(ex) + 1e-8)
            cgs_vals.extend(sm.tolist())
        df_scores['cgs_score'] = cgs_vals
    else:
        df_scores['cgs_score'] = 0.0

    # Fused score
    df_scores['fused_score'] = (
        w_fuse_temporal * df_scores['t_score'].astype(np.float32) +
        w_fuse_cgz * df_scores['cgz_score'].astype(np.float32) +
        w_fuse_cgs * df_scores['cgs_score'].astype(np.float32)
    ).astype(np.float32)

    # Return per-site row with required columns
    if output_components:
        return df_scores[['site_name','month','t_score','cgz_score','cgs_score','fused_score']]
    else:
        return df_scores[['site_name','month','fused_score']].rename(columns={'fused_score':'score'})

def update_aggregated_scores(out_dir: str, split: str, month_str: str, df_scores: pd.DataFrame):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    base = Path(out_dir) / f"unsup_month_scores_{split}.csv"
    # Initialize or load aggregated matrix
    if base.exists():
        agg = pd.read_csv(base)
    else:
        agg = pd.DataFrame({'site_name': df_scores['site_name'].tolist()})
        # Preserve mode column if provided
        if 'mode' in df_scores.columns:
            # If mode varies per site, take the provided value; else a single mode value is fine
            modes = df_scores.set_index('site_name')['mode']
            agg = agg.set_index('site_name')
            agg['mode'] = modes
            agg = agg.reset_index()
    # Ensure all MONTHS columns exist
    for m in MONTHS:
        if m not in agg.columns:
            agg[m] = np.nan
    # Set index to site_name for assignment
    agg = agg.set_index('site_name')
    # Ensure all current sites exist
    for s in df_scores['site_name']:
        if s not in agg.index:
            agg.loc[s, :] = np.nan
    # Preserve/assign mode if present
    if 'mode' in df_scores.columns:
        cur_mode = df_scores.set_index('site_name')['mode']
        agg.loc[cur_mode.index, 'mode'] = cur_mode.values
    # Assign current month scores
    cur_series = df_scores.set_index('site_name')['score']
    agg.loc[cur_series.index, month_str] = cur_series.values
    # Reorder columns to canonical MONTHS order and write out
    # Preserve mode column if present
    re_cols = MONTHS
    if 'mode' in agg.columns:
        re_cols = ['mode'] + MONTHS
    agg = agg.reindex(columns=re_cols)
    agg = agg.reset_index()
    agg.to_csv(base, index=False)

    # Additionally, write mode-specific aggregated file if mode available
    if 'mode' in df_scores.columns:
        mode_val = None
        try:
            unique_modes = sorted(set(df_scores['mode'].astype(str).unique().tolist()))
            mode_val = unique_modes[0] if len(unique_modes) == 1 else None
        except Exception:
            mode_val = None
        if mode_val:
            base_mode = Path(out_dir) / f"unsup_month_scores_{split}_{mode_val}.csv"
            agg.to_csv(base_mode, index=False)

def save_monthly_output(out_dir: str, embedding: str, year: int, month: int, split: str, df_scores: pd.DataFrame, mode: str = 'distance_baseline'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Add mode column for clarity
    if 'mode' not in df_scores.columns:
        df_scores = df_scores.copy()
        df_scores.insert(1, 'mode', mode)
    # Name per-mode file distinctly
    basename = f"monthly_{mode}_{embedding}_{year}_{month:02d}_{split}.csv" if mode else f"monthly_{embedding}_{year}_{month:02d}_{split}.csv"
    fp = Path(out_dir) / basename
    df_scores.to_csv(fp, index=False)
    return str(fp)

def parse_args():
    ap = argparse.ArgumentParser("Monthly change detection")
    ap.add_argument('--mode', type=str, default='baseline', choices=['baseline','unsup'], help='Inference mode: baseline distance or unsupervised ensemble')
    ap.add_argument('--embedding', type=str, required=True, help='Embedding type (e.g., dinov3, georsclip, prithvi-eo-2.0)')
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--month', type=int, required=True)
    ap.add_argument('--features_dir', type=str, default=DEFAULT_FEATURES_DIR)
    ap.add_argument('--features_csv', type=str, default=None, help='Unified monthly features CSV (required for --mode unsup)')
    ap.add_argument('--features_unified_csv', type=str, default=None, help='Unified monthly features CSV for baseline mode (masked/global)')
    ap.add_argument('--groundtruth_csv', type=str, default=None)
    ap.add_argument('--split', type=str, default='all', choices=['train','val','test','all','all_looted'])
    ap.add_argument('--distance', type=str, default='l2', choices=['l2','cosine'])
    ap.add_argument('--scaler_path', type=str, default=None, help='Scaler stats .npz for normalization (defaults to self_supervised_change_detection/model_runs/<embedding>/scaler_stats.npz)')
    ap.add_argument('--trained_model', type=str, default=None, help='Path to unsup_models.pt (defaults to self_supervised_change_detection/model_runs/<embedding>/unsup_models.pt)')
    ap.add_argument('--model_runs_dir', type=str, default='self_supervised_change_detection/model_runs', help='Base directory for model runs and scaler stats')
    ap.add_argument('--overwrite_models', action='store_true', help='Force retraining and overwrite existing `.pt` and `.npz` artifacts')
    ap.add_argument('--robust', action='store_true', help='Use robust (median/MAD) scaling if stats available')
    ap.add_argument('--disable_per_month_feature_norm', action='store_true', help='Disable seasonal per-month normalization stage even if stats available')
    ap.add_argument('--output_dir', type=str, default=None, help='Defaults to self_supervised_change_detection/model_runs/<embedding>')
    # Sophistication/adaptation flags
    ap.add_argument('--intra_site_grid_calibration', action='store_true', help='Robust (median/MAD) per-month cross-grid z (positive part) within site')
    ap.add_argument('--month_of_year_seasonal', action='store_true', help='Global month-of-year median/MAD normalization of temporal base across sites')
    ap.add_argument('--enable_cross_grid', action='store_true', help='Compute cross-grid softmax per site and fuse')
    ap.add_argument('--w_fuse_temporal', type=float, default=1.0, help='Fusion weight for temporal base component')
    ap.add_argument('--w_fuse_cgz', type=float, default=1.0, help='Fusion weight for intra-site robust z component')
    ap.add_argument('--w_fuse_cgs', type=float, default=0.5, help='Fusion weight for cross-grid softmax component')
    ap.add_argument('--output_components', action='store_true', help='Export component columns (t_*, cgz_*, cgs_*, fused_*) in per-month CSV')
    ap.add_argument('--rolling_window', type=int, default=6, help='Rolling window size (months) for per-site temporal robust z using aggregated scores if available')
    # Probability normalization for ensemble mode
    ap.add_argument('--score_norm_method', type=str, default='sigmoid', choices=['none','minmax','sigmoid','softmax'])
    ap.add_argument('--score_norm_temperature', type=float, default=1.0)
    ap.add_argument('--export_probabilities', action='store_true')
    ap.add_argument('--merge', action='store_true', help='After updating aggregated matrix, remove the per-month CSV')
    # Baseline probability-like normalization at generation time (to mirror unsupervised export)
    ap.add_argument('--baseline_norm_method', type=str, default='none', choices=['none','minmax','sigmoid','softmax'], help='Normalize baseline per-site series to [0,1]')
    ap.add_argument('--baseline_norm_temperature', type=float, default=1.0, help='Temperature for baseline normalization')
    ap.add_argument('--baseline_export_probabilities', action='store_true', help='Write normalized probability-like score for baseline mode')
    return ap.parse_args()

def main():
    args = parse_args()
    # Determine site list from groundtruth split if provided
    sites = _sites_from_groundtruth(args.groundtruth_csv, args.split)
    month_str = f"{args.year}_{args.month:02d}"
    # Resolve default output directory to model_runs/<embedding> (absolute based on package dir if relative)
    base_runs_dir = Path(args.model_runs_dir)
    if not base_runs_dir.is_absolute():
        # Resolve relative to repo root to avoid double 'self_supervised_change_detection/' nesting
        repo_root = PKG_DIR.parent
        base_runs_dir = repo_root / base_runs_dir
    default_runs_dir = str((base_runs_dir / args.embedding).resolve())
    os.makedirs(default_runs_dir, exist_ok=True)
    if not args.output_dir:
        args.output_dir = default_runs_dir
    if args.mode == 'unsup':
        # Delegate to ensemble exporter
        # Ensure features_csv provided
        if not args.features_csv:
            raise ValueError("--features_csv is required for --mode unsup")
        # Resolve default model/scaler paths under self_supervised_change_detection/model_runs/<embedding>
        model_dir = default_runs_dir
        trained_model = args.trained_model or os.path.join(model_dir, 'unsup_models.pt')
        scaler_path = args.scaler_path or os.path.join(model_dir, 'scaler_stats.npz')
        need_train = args.overwrite_models or (not os.path.exists(trained_model)) or (not os.path.exists(scaler_path))
        if need_train:
            # Train models and compute scaler via wrapper
            cmd = [
                sys.executable, '-m', 'self_supervised_change_detection.train_unsup',
                '--features_csv', args.features_csv,
                '--output_dir', model_dir,
                '--scaler_path', scaler_path,
            ]
            if args.groundtruth_csv:
                cmd.extend(['--groundtruth_csv', args.groundtruth_csv])
            if args.robust:
                cmd.append('--robust_scaler')
            if args.disable_per_month_feature_norm:
                cmd.append('--disable_per_month_feature_norm')
            if args.overwrite_models:
                cmd.append('--overwrite_scaler')
            print('Training ensemble (models + scaler):', ' '.join(cmd))
            subprocess.run(cmd, check=True)
            # After training, ensure paths are set
        import argparse as _ap
        from .monthly_unsup_inference import export_single_month_unsup
        # Build a namespace of arguments expected by the ensemble exporter
        ens_args = _ap.Namespace(
            features_csv=args.features_csv,
            groundtruth_csv=args.groundtruth_csv,
            split_col='split',
            split=args.split,
            scaler_path=scaler_path,
            trained_model=trained_model,
            output_dir=args.output_dir,
            device='cuda',
            merge=args.merge,
            year=args.year,
            month=args.month,
            k_density=5,
            alpha_rec=0.6,
            alpha_fore=0.3,
            alpha_novel=0.4,
            rw_window=12,
            global_spike_q=0.98,
            calendar_population='preserved',
            no_calendar_calibration=False,
            negative_z_ok=False,
            intra_site_grid_calibration=args.intra_site_grid_calibration,
            month_of_year_seasonal=args.month_of_year_seasonal,
            enable_cross_grid=args.enable_cross_grid,
            w_fuse_temporal=args.w_fuse_temporal,
            w_fuse_cgz=args.w_fuse_cgz,
            w_fuse_cgs=args.w_fuse_cgs,
            output_components=args.output_components,
            robust_scaler=args.robust,
            disable_per_month_feature_norm=args.disable_per_month_feature_norm,
            score_norm_method=args.score_norm_method,
            score_norm_temperature=args.score_norm_temperature,
            export_probabilities=args.export_probabilities,
        )
        export_single_month_unsup(ens_args)
        print(f"[done] Month {month_str} exported via ensemble to {args.output_dir}")
        return

    # Path to aggregated scores (for rolling z)
    agg_path = os.path.join(args.output_dir, f"unsup_month_scores_{args.split}.csv")
    df = compute_monthly_change(
        args.embedding, args.year, args.month, args.features_dir, sites,
        distance=args.distance,
        scaler_path=args.scaler_path,
        features_unified_csv=args.features_unified_csv,
        per_month_feature_norm=(not args.disable_per_month_feature_norm),
        robust=args.robust,
        intra_site_grid_calibration=args.intra_site_grid_calibration,
        month_of_year_seasonal=args.month_of_year_seasonal,
        enable_cross_grid=args.enable_cross_grid,
        w_fuse_temporal=args.w_fuse_temporal,
        w_fuse_cgz=args.w_fuse_cgz,
        w_fuse_cgs=args.w_fuse_cgs,
        output_components=args.output_components,
        aggregated_scores_path=agg_path,
        rolling_window=args.rolling_window,
    )
    # Optional baseline normalization to probability-like score using per-site series (from aggregated if available)
    if (args.baseline_norm_method != 'none') or args.baseline_export_probabilities:
        agg_for_norm = None
        agg_path = os.path.join(args.output_dir, f"unsup_month_scores_{args.split}.csv")
        if os.path.exists(agg_path):
            try:
                agg_for_norm = pd.read_csv(agg_path).set_index('site_name')
            except Exception as _e:
                agg_for_norm = None
        cur_month = f"{args.year}_{args.month:02d}"
        # Collect value per site for current month (fused if present else score)
        if 'fused_score' in df.columns:
            cur_vals = df.set_index('site_name')['fused_score'].astype(np.float32)
        else:
            cur_vals = df.set_index('site_name')['score'].astype(np.float32)
        probs = []
        for sname, cur in cur_vals.items():
            vec = []
            if agg_for_norm is not None and sname in agg_for_norm.index:
                # Build chronological vector of prior months
                row = agg_for_norm.loc[sname]
                series_vals = []
                for m in MONTHS:
                    if m == cur_month:
                        break
                    if m in agg_for_norm.columns:
                        v = row.get(m)
                        try:
                            v = float(v)
                        except Exception:
                            v = np.nan
                        if not np.isnan(v):
                            series_vals.append(v)
                vec = series_vals
            # Append current value and normalize
            vec = np.array(list(vec) + [float(cur)], dtype=np.float32)
            pvec = normalize_scores(vec, method=args.baseline_norm_method, temperature=args.baseline_norm_temperature)
            probs.append((sname, float(pvec[-1]) if pvec.size > 0 else 0.5))
        prob_series = pd.Series({k:v for k,v in probs}, name='prob', dtype=np.float32)
        # Overwrite df with probability-like output
        df = df.copy()
        if 'fused_score' in df.columns:
            df['fused_score'] = df['site_name'].map(prob_series).astype(np.float32)
        else:
            df['score'] = df['site_name'].map(prob_series).astype(np.float32)

    pm_fp = save_monthly_output(args.output_dir, args.embedding, args.year, args.month, args.split, df, mode='distance_baseline')
    # When output_components enabled, aggregated scores should use fused_score (already normalized if requested)
    df_for_agg = df[['site_name','month']].copy()
    if 'fused_score' in df.columns:
        df_for_agg['score'] = df['fused_score'].astype(np.float32)
    else:
        df_for_agg['score'] = df['score'].astype(np.float32)
    df_for_agg['mode'] = 'distance_baseline'
    update_aggregated_scores(args.output_dir, args.split, month_str, df_for_agg)
    # Optionally remove per-month file after aggregation
    if args.merge:
        try:
            os.remove(pm_fp)
        except Exception as _e_rm:
            print(f"[warn] could not remove per-month file {pm_fp}: {_e_rm}")
    print(f"[done] Month {month_str} computed for {len(df)} sites; outputs written to {args.output_dir}")

if __name__ == '__main__':
    main()
