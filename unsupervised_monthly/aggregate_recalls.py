#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Aggregate recall@k across embeddings into margin-columns matrices.

Reads metrics CSVs produced by evaluate_unified_monthlies under
    <results_dir>/<embedding>/<embedding>_<mode>_<split>_metrics.csv

Writes summary CSVs into <results_dir>:
    - recall_{all|test}_{mode}.csv for mode in {distance_baseline, learned_unsupervised, weakly_supervised}
    - plus directional variants: *_positive.csv and *_negative.csv

Each file has rows=embeddings and columns=margin_0..margin_6 with values recall_at_k.
Missing entries become NaN.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

from .mode_utils import CANONICAL_MODES, MODE_ALIASES, is_legacy_mode, normalize_mode


def _find_metrics_file(emb_dir: Path, emb: str, mode: str, split: str, variant: str) -> Path | None:
    """Try canonical and legacy mode names to find a metrics CSV."""
    # Collect candidate mode names: canonical first, then any legacy aliases
    candidates = [mode]
    for alias, canonical in MODE_ALIASES.items():
        if canonical == mode and alias != mode:
            candidates.append(alias)
    for m in candidates:
        if variant:
            p = emb_dir / f"{emb}_{m}_{split}_metrics_{variant}.csv"
        else:
            p = emb_dir / f"{emb}_{m}_{split}_metrics.csv"
        if p.exists():
            return p
    return None


def collect_recall_matrix(results_root: Path, split: str, mode: str, max_margin: int = 6, variant: str = "") -> pd.DataFrame:
    rows = {}
    for emb_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        emb = emb_dir.name
        metrics_path = _find_metrics_file(emb_dir, emb, mode, split, variant)
        if metrics_path is None:
            continue
        try:
            df = pd.read_csv(metrics_path)
            # Expect 'margin' and 'recall_at_k'
            out_row = {}
            for m in range(max_margin + 1):
                v = df.loc[df['margin'] == m, 'recall_at_k']
                out_row[f"margin_{m}"] = float(v.iloc[0]) if len(v) > 0 else float('nan')
            rows[emb] = out_row
        except Exception as e:
            print(f"[warn] Failed reading {metrics_path}: {e}")
    if not rows:
        return pd.DataFrame()
    mat = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    mat.index.name = 'embedding'
    return mat


def main():
    import argparse
    ap = argparse.ArgumentParser('Aggregate recall@k matrices from metrics CSVs')
    ap.add_argument('--results_dir', type=str, default=None, help='Directory containing per-embedding metrics (defaults to unsupervised_monthly/results)')
    ap.add_argument(
        '--modes',
        type=str,
        default=None,
        help='Comma-separated list of modes to aggregate (e.g., weakly_supervised). Defaults to all supported modes.',
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    results_root = Path(args.results_dir) if args.results_dir else (root / 'results')
    results_root.mkdir(parents=True, exist_ok=True)

    supported_modes = list(CANONICAL_MODES)
    if args.modes:
        modes = [m.strip() for m in args.modes.split(',') if m.strip()]
        # Allow legacy aliases and normalize to canonical.
        norm_modes = []
        for m in modes:
            if is_legacy_mode(m):
                print(f"[warn] Legacy mode '{m}' is deprecated; use '{normalize_mode(m)}' instead")
            norm_modes.append(normalize_mode(m) or m)
        # De-dupe while preserving order.
        modes = []
        for m in norm_modes:
            if m not in modes:
                modes.append(m)
        unknown = sorted(set(modes) - set(supported_modes))
        if unknown:
            raise SystemExit(f"Unknown mode(s): {unknown}. Supported: {supported_modes}")
    else:
        modes = supported_modes
    variants = ["", "positive", "negative"]
    specs = []
    for split in ("all", "test"):
        for mode in modes:
            for variant in variants:
                specs.append((split, mode, variant))

    for split, mode, variant in specs:
        mat = collect_recall_matrix(results_root, split=split, mode=mode, max_margin=6, variant=variant)
        if variant:
            out = results_root / f"recall_{split}_{mode}_{variant}.csv"
        else:
            out = results_root / f"recall_{split}_{mode}.csv"
        mat.to_csv(out)
        print(f"[write] {out} ({mat.shape[0]}x{mat.shape[1] if not mat.empty else 0})")


if __name__ == '__main__':
    main()
