#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Convenience wrapper to run the training/export pipeline using the LOCAL
`self_supervised_change_detection/main_train_unsupervised.py` entry (vendorized trainer).
Use this to produce `unsup_models.pt` and `scaler_stats.npz` for a given features CSV.
All CLI arguments are forwarded transparently to the local script.

Example:
  python -m self_supervised_change_detection.train_unsup \
    --features_csv ../planet_mosaics_final_4bands/features_dinov3_monthly_masked.csv \
    --groundtruth_csv ../planet_mosaics_final_4bands/ground_truth_split_balanced_aux.csv \
    --output_dir self_supervised_change_detection/model_runs/dinov3 \
    --scaler_path self_supervised_change_detection/model_runs/dinov3/scaler_stats.npz \
    --epochs 60 --batch_size 16 --export_only
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
import subprocess

def main():
  here = Path(__file__).resolve().parent
  cmd = [sys.executable, '-m', 'self_supervised_change_detection.main_train_unsupervised'] + sys.argv[1:]
  print('Forwarding to local trainer (module):', ' '.join(cmd))
  env = os.environ.copy()
  # Ensure this package is importable when running as a module
  pkg_path = str(here.parent)  # add repo root to path
  existing = env.get('PYTHONPATH', '')
  env['PYTHONPATH'] = (pkg_path + ((':' + existing) if existing else '')).strip(':')
  subprocess.run(cmd, check=True, env=env)

if __name__ == '__main__':
    main()
