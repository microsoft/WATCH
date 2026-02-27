#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

# Batch runner for Jan 2017..Dec 2024

def run_batch(embedding: str, features_dir: str, groundtruth_csv: str | None, split: str, distance: str, output_dir: str,
              scaler_path: str | None = None, robust: bool = False, disable_per_month_feature_norm: bool = False,
              intra_site_grid_calibration: bool = False, month_of_year_seasonal: bool = False, enable_cross_grid: bool = False,
              w_fuse_temporal: float = 1.0, w_fuse_cgz: float = 1.0, w_fuse_cgs: float = 0.5,
              output_components: bool = False, rolling_window: int = 6,
              use_unsup_model: bool = False, trained_model: str | None = None,
              features_unified_csv: str | None = None,
              score_norm_method: str = "sigmoid", score_norm_temperature: float = 1.0,
              export_probabilities: bool = False,
              baseline_norm_method: str = "none", baseline_norm_temperature: float = 1.0,
              baseline_export_probabilities: bool = False,
              year_start: int = 2017, year_end: int = 2024):
    for y in range(year_start, year_end + 1):
        for m in range(1, 13):
            if use_unsup_model:
                src_features = features_unified_csv if features_unified_csv else features_dir
                # Use unified monthly_inference with --mode unsup to auto-train if needed
                cmd = [
                    sys.executable, '-m', 'unsupervised_monthly.monthly_inference',
                    '--mode', 'unsup',
                    '--embedding', embedding,
                    '--features_csv', src_features,
                    '--year', str(y),
                    '--month', f"{m:02d}",
                    '--split', split,
                    '--output_dir', output_dir,
                ]
                if groundtruth_csv:
                    cmd.extend(['--groundtruth_csv', groundtruth_csv])
                if scaler_path:
                    cmd.extend(['--scaler_path', scaler_path])
                if trained_model:
                    cmd.extend(['--trained_model', trained_model])
                if robust:
                    cmd.append('--robust')
                if disable_per_month_feature_norm:
                    cmd.append('--disable_per_month_feature_norm')
                if intra_site_grid_calibration:
                    cmd.append('--intra_site_grid_calibration')
                if month_of_year_seasonal:
                    cmd.append('--month_of_year_seasonal')
                if enable_cross_grid:
                    cmd.append('--enable_cross_grid')
                if output_components:
                    cmd.append('--output_components')
                # Fusion weights
                cmd.extend(['--w_fuse_temporal', str(w_fuse_temporal), '--w_fuse_cgz', str(w_fuse_cgz), '--w_fuse_cgs', str(w_fuse_cgs)])
                if export_probabilities:
                    cmd.append('--export_probabilities')
                    cmd.extend(['--score_norm_method', score_norm_method, '--score_norm_temperature', str(score_norm_temperature)])
            else:
                cmd = [
                    sys.executable, '-m', 'unsupervised_monthly.monthly_inference',
                    '--embedding', embedding,
                    '--year', str(y),
                    '--month', f"{m:02d}",
                    '--split', split,
                    '--distance', distance,
                    '--output_dir', output_dir,
                ]
                if features_unified_csv:
                    cmd.extend(['--features_unified_csv', features_unified_csv])
                else:
                    cmd.extend(['--features_dir', features_dir])
                if groundtruth_csv:
                    cmd.extend(['--groundtruth_csv', groundtruth_csv])
                if scaler_path:
                    cmd.extend(['--scaler_path', scaler_path])
                if robust:
                    cmd.append('--robust')
                if disable_per_month_feature_norm:
                    cmd.append('--disable_per_month_feature_norm')
                if intra_site_grid_calibration:
                    cmd.append('--intra_site_grid_calibration')
                if month_of_year_seasonal:
                    cmd.append('--month_of_year_seasonal')
                if enable_cross_grid:
                    cmd.append('--enable_cross_grid')
                if output_components:
                    cmd.append('--output_components')
                if rolling_window and rolling_window > 0:
                    cmd.extend(['--rolling_window', str(rolling_window)])
                # Fusion weights
                cmd.extend(['--w_fuse_temporal', str(w_fuse_temporal), '--w_fuse_cgz', str(w_fuse_cgz), '--w_fuse_cgs', str(w_fuse_cgs)])
                # Baseline normalization flags
                if baseline_norm_method and baseline_norm_method != 'none':
                    cmd.extend(['--baseline_norm_method', baseline_norm_method, '--baseline_norm_temperature', str(baseline_norm_temperature)])
                if baseline_export_probabilities:
                    cmd.append('--baseline_export_probabilities')
            print('Running:', ' '.join(cmd))
            subprocess.run(cmd, check=True)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser('Batch monthly change detection runner')
    ap.add_argument('--embedding', required=True)
    ap.add_argument('--features_dir', default='planet_mosaics_final_4bands/features_new_with_mask')
    ap.add_argument('--features_unified_csv', default=None, help='Unified CSV for baseline mode (masked/global)')
    ap.add_argument('--groundtruth_csv', default=None)
    ap.add_argument('--split', default='all', choices=['train','val','test','all','all_looted'])
    ap.add_argument('--distance', default='l2', choices=['l2','cosine'])
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--scaler_path', default=None)
    ap.add_argument('--robust', action='store_true')
    ap.add_argument('--disable_per_month_feature_norm', action='store_true')
    ap.add_argument('--intra_site_grid_calibration', action='store_true')
    ap.add_argument('--month_of_year_seasonal', action='store_true')
    ap.add_argument('--enable_cross_grid', action='store_true')
    ap.add_argument('--w_fuse_temporal', type=float, default=1.0)
    ap.add_argument('--w_fuse_cgz', type=float, default=1.0)
    ap.add_argument('--w_fuse_cgs', type=float, default=0.5)
    ap.add_argument('--output_components', action='store_true')
    ap.add_argument('--rolling_window', type=int, default=6)
    ap.add_argument('--year_start', type=int, default=2017)
    ap.add_argument('--year_end', type=int, default=2024)
    # Ensemble-based monthly export
    ap.add_argument('--use_unsup_model', action='store_true')
    ap.add_argument('--trained_model', type=str, default=None)
    ap.add_argument('--score_norm_method', type=str, default='sigmoid', choices=['none','minmax','sigmoid','softmax'])
    ap.add_argument('--score_norm_temperature', type=float, default=1.0)
    ap.add_argument('--export_probabilities', action='store_true')
    # Baseline normalization at generation
    ap.add_argument('--baseline_norm_method', type=str, default='none', choices=['none','minmax','sigmoid','softmax'])
    ap.add_argument('--baseline_norm_temperature', type=float, default=1.0)
    ap.add_argument('--baseline_export_probabilities', action='store_true')
    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_batch(
        args.embedding, args.features_dir, args.groundtruth_csv, args.split, args.distance, args.output_dir,
        scaler_path=args.scaler_path, robust=args.robust, disable_per_month_feature_norm=args.disable_per_month_feature_norm,
        intra_site_grid_calibration=args.intra_site_grid_calibration, month_of_year_seasonal=args.month_of_year_seasonal,
        enable_cross_grid=args.enable_cross_grid, w_fuse_temporal=args.w_fuse_temporal, w_fuse_cgz=args.w_fuse_cgz,
        w_fuse_cgs=args.w_fuse_cgs, output_components=args.output_components, rolling_window=args.rolling_window,
        use_unsup_model=args.use_unsup_model, trained_model=args.trained_model, features_unified_csv=args.features_unified_csv,
        score_norm_method=args.score_norm_method, score_norm_temperature=args.score_norm_temperature,
        export_probabilities=args.export_probabilities,
        baseline_norm_method=args.baseline_norm_method, baseline_norm_temperature=args.baseline_norm_temperature,
        baseline_export_probabilities=args.baseline_export_probabilities,
        year_start=args.year_start, year_end=args.year_end,
    )
