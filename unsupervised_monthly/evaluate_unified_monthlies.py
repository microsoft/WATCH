#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Evaluate unified monthly score matrices produced by merge_monthlies.

Canonical modes:
- distance_baseline
- learned_unsupervised
- weakly_supervised

Inputs: aggregated CSV like unsup_month_scores_all_distance_baseline_new.csv or
unsup_month_scores_all_learned_unsupervised_new.csv
Behavior:
- Filters rows by split using ground truth CSV (e.g., split == 'test' or all rows)
- Computes top-k ranking per site and metrics across margins 0..max_margin
- Optionally computes directional metrics (positive=future-only, negative=past-only)
- Writes metrics files into a results directory named by embedding, mode, and split

Output files (in results_dir):
- <embedding>_<mode>_<split>_metrics.csv
- <embedding>_<mode>_<split>_metrics_positive.csv (if --directional)
- <embedding>_<mode>_<split>_metrics_negative.csv (if --directional)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .month_utils import MONTHS, normalize_month_str
from .mode_utils import infer_mode_from_filename, is_legacy_mode, normalize_mode


def parse_args():
    ap = argparse.ArgumentParser("Evaluate unified monthly matrices")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scores_csv", type=str, help="Path to unified scores CSV (unsup_month_scores_all_<mode>*.csv or unsup_month_scores_<split>.csv)")
    src.add_argument("--output_dir", type=str, help="Directory containing unsup_month_scores_<split>.csv (legacy per-split export)")
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test","all","all_looted"], help="Split to evaluate")
    ap.add_argument("--groundtruth_csv", type=str, required=True, help="Ground truth CSV with site_name, split, looted/looted_month")
    ap.add_argument("--results_dir", type=str, default="", help="Directory to write results (defaults to parent of scores file)")
    ap.add_argument("--embedding", type=str, default=None, help="Embedding/model name for output naming (auto from path if omitted)")
    ap.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=[
            "distance_baseline",
            "learned_unsupervised",
            "weakly_supervised",
            # legacy aliases
            "baseline",
            "unsupervised",
        ],
        help="Mode for output naming (auto from filename if omitted). Canonical: distance_baseline|learned_unsupervised|weakly_supervised.",
    )
    ap.add_argument("--top_k", type=int, default=24)
    ap.add_argument("--max_margin", type=int, default=6)
    ap.add_argument("--directional", action="store_true", help="Also compute directional (+/-) metrics")
    # Stratified-by-year selection: take top_k/8 per year (2017..2024)
    ap.add_argument("--stratified_year", action="store_true", default=True, help="Select months stratified by year: take roughly top_k/8 per year")
    ap.add_argument("--no-stratified_year", dest="stratified_year", action="store_false", help="Disable stratified-by-year selection (use global top-k)")
    # Limit evaluation to a year range (inclusive). If omitted, uses all months present.
    ap.add_argument("--year_start", type=int, default=None, help="Earliest year to include (e.g., 2017)")
    ap.add_argument("--year_end", type=int, default=None, help="Latest year to include (e.g., 2020)")
    return ap.parse_args()


def _infer_embedding_and_mode(scores_csv: str, provided_embedding: str | None, provided_mode: str | None) -> Tuple[str, str]:
    # Infer embedding from parent dir name under model_runs/<embedding>
    p = Path(scores_csv).resolve()
    embedding = provided_embedding
    mode = provided_mode
    try:
        if embedding is None:
            # Expect .../model_runs/<embedding>/unsup_month_scores_...
            embedding = p.parent.name
        if mode is None:
            mode = infer_mode_from_filename(p.name)
    except Exception:
        pass
    embedding = embedding or "unknown"
    if is_legacy_mode(mode or ""):
        print(f"[warn] Legacy mode '{mode}' is deprecated; use '{normalize_mode(mode)}' instead")
    mode = normalize_mode(mode) or "unknown"
    return embedding, mode


def _select_score_matrix(df: pd.DataFrame) -> tuple[np.ndarray, List[str]]:
    # Prefer explicit month-named columns matching MONTHS (full span)
    cols = [m for m in MONTHS if m in df.columns]
    if len(cols) == len(MONTHS):
        return df[cols].to_numpy(dtype=np.float32), cols
    # Fallback: probability columns p_0..p_95
    pcols = [f"p_{i}" for i in range(len(MONTHS)) if f"p_{i}" in df.columns]
    if len(pcols) == len(MONTHS):
        return df[pcols].to_numpy(dtype=np.float32), pcols
    # Fallback 2: score columns s_0.. with softmax per row
    scols = [c for c in df.columns if c.startswith("s_")]
    if scols:
        raw = df[scols[:len(MONTHS)]].to_numpy(dtype=np.float32)
        raw = raw - np.nanmax(raw, axis=1, keepdims=True)
        e = np.exp(np.nan_to_num(raw))
        den = np.nansum(e, axis=1, keepdims=True) + 1e-8
        return e / den, scols[:len(MONTHS)]
    # If none found, try auto-detect numeric month-like columns (subset of MONTHS)
    mcols = [m for m in MONTHS if m in df.columns]
    if mcols:
        return df[mcols].to_numpy(dtype=np.float32), mcols
    raise ValueError("Could not identify score/probability columns in unified CSV")


def _filter_sites_by_split(scores_df: pd.DataFrame, gt: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all":
        # Keep all rows present in scores CSV
        return scores_df
    if split == "all_looted":
        if "looted" in gt.columns:
            keep = set(gt[gt["looted"] == 1].index)
            return scores_df[scores_df["site_name"].isin(keep)].reset_index(drop=True)
        return scores_df
    if "split" in gt.columns:
        keep = set(gt[gt["split"] == split].index)
        return scores_df[scores_df["site_name"].isin(keep)].reset_index(drop=True)
    # If split info missing, return as-is (caller will compute metrics only for sites with known looted_month)
    return scores_df


def _rank_and_labels(scores_df: pd.DataFrame, probs: np.ndarray, gt: pd.DataFrame, top_k: int, stratified_year: bool, months_used: List[str]):
    rows = []  # (site, true_idx, true_change_month, topk_indices, full_ranking)
    # Build month index mapping for this matrix (handles subsets of MONTHS)
    month_to_idx = {m: i for i, m in enumerate(months_used)}
    # Group indices by contiguous 12-month years based on order in months_used
    n_months = probs.shape[1]
    n_years = max(1, int(np.ceil(n_months / 12)))
    year_blocks = []
    for y in range(n_years):
        start = y * 12
        end = min((y + 1) * 12, n_months)
        if start < end:
            year_blocks.append(list(range(start, end)))
    for i, r in scores_df.iterrows():
        site = r["site_name"]
        scores = probs[i]
        tmp = scores.copy()
        tmp[~np.isfinite(tmp)] = -np.inf
        ranking = np.argsort(-tmp).tolist()
        if stratified_year:
            # Distribute top_k across 8 years as evenly as possible
            base = max(0, top_k // len(year_blocks))
            rem = max(0, top_k - base * len(year_blocks))
            # For remainder allocation, score the next-best month per year
            per_year = []
            for y, idxs in enumerate(year_blocks):
                year_scores = tmp[idxs]
                order = np.argsort(-year_scores)
                # Keep up to base now; extra will be handled after
                take = min(base, len(order))
                chosen = [idxs[j] for j in order[:take]]
                # record the next-best candidate if any
                next_score = year_scores[order[take]] if take < len(order) else -np.inf
                per_year.append({"year": y, "idxs": chosen, "order": order, "idxs_full": idxs, "next_score": float(next_score)})
            # Allocate remainder to years with highest next_score
            if rem > 0:
                # sort by next_score descending
                alloc_order = sorted(range(len(year_blocks)), key=lambda y: per_year[y]["next_score"], reverse=True)
                for k in range(rem):
                    y = alloc_order[k % len(year_blocks)]
                    info = per_year[y]
                    take_pos = len(info["idxs"])  # next position
                    if take_pos < len(info["order"]):
                        info["idxs"].append(info["idxs_full"][info["order"][take_pos]])
                # ensure stored back (dicts are by ref already)
            topk = []
            for info in per_year:
                topk.extend(info["idxs"])
            # If due to NaNs we still have fewer than top_k, pad from global ranking
            if len(topk) < top_k:
                extra = [m for m in ranking if m not in set(topk)][: (top_k - len(topk))]
                topk.extend(extra)
            else:
                topk = topk[:top_k]
        else:
            topk = ranking[:top_k]
        true_idx = -1
        true_change_month = "NA"
        if site in gt.index and "looted_month" in gt.columns:
            lm = normalize_month_str(gt.loc[site, "looted_month"])
            true_change_month = lm
            if lm in month_to_idx:
                true_idx = month_to_idx[lm]
        rows.append((site, true_idx, true_change_month, topk, ranking))
    return rows


def _within_margin(true_idx: int, pred_list: List[int], m: int) -> int:
    if true_idx < 0:
        return 0
    return int(any(abs(true_idx - p) <= m for p in pred_list))


def _within_margin_pos(true_idx: int, pred_list: List[int], m: int) -> int:
    if true_idx < 0:
        return 0
    return int(any((p - true_idx) >= 0 and (p - true_idx) <= m for p in pred_list))


def _within_margin_neg(true_idx: int, pred_list: List[int], m: int) -> int:
    if true_idx < 0:
        return 0
    return int(any((true_idx - p) >= 0 and (true_idx - p) <= m for p in pred_list))


def _compute_metrics(rows, probs: np.ndarray | None, top_k: int, max_margin: int, directional: bool = False):
    from sklearn.metrics import roc_auc_score
    known = [r for r in rows if r[1] >= 0]
    # Symmetric margins
    metrics = []
    for m in range(max_margin + 1):
        hits = sum(_within_margin(ti, tk, m) for (_s, ti, _lm, tk, _rk) in known)
        total = len(known)
        precision_at_k = hits / (total * top_k) if total > 0 else np.nan
        recall_at_k = hits / total if total > 0 else np.nan
        f1_at_k = (2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)) if (precision_at_k + recall_at_k) > 0 else 0.0
        auroc = np.nan
        if probs is not None and total > 0:
            try:
                y_true, y_score = [], []
                for i, (_s, ti, _lm, _tk, _rk) in enumerate(known):
                    if ti < 0:
                        continue
                    pv = probs[i]
                    for j, p in enumerate(pv):
                        if not np.isfinite(p):
                            continue
                        y_score.append(float(p))
                        y_true.append(1 if abs(j - ti) <= m else 0)
                if 1 in y_true and 0 in y_true:
                    auroc = roc_auc_score(y_true, y_score)
            except Exception:
                pass
        metrics.append({"margin": m, "acc": float(np.mean([_within_margin(ti, tk, m) for (_s, ti, _lm, tk, _rk) in known])) if total>0 else np.nan,
                        "precision_at_k": precision_at_k, "recall_at_k": recall_at_k, "f1_at_k": f1_at_k, "auroc": auroc})

    if not directional:
        return pd.DataFrame(metrics)

    # Directional margins
    metrics_pos, metrics_neg = [], []
    for m in range(max_margin + 1):
        # Future-only
        hits_p = sum(_within_margin_pos(ti, tk, m) for (_s, ti, _lm, tk, _rk) in known)
        total = len(known)
        precision_p = hits_p / (total * top_k) if total > 0 else np.nan
        recall_p = hits_p / total if total > 0 else np.nan
        f1_p = (2 * precision_p * recall_p / (precision_p + recall_p)) if (precision_p + recall_p) > 0 else 0.0
        auroc_p = np.nan
        if probs is not None and total > 0:
            try:
                y_true, y_score = [], []
                for i, (_s, ti, _lm, _tk, _rk) in enumerate(known):
                    if ti < 0:
                        continue
                    pv = probs[i]
                    for j, p in enumerate(pv):
                        if not np.isfinite(p):
                            continue
                        y_score.append(float(p))
                        y_true.append(1 if (j - ti) >= 0 and (j - ti) <= m else 0)
                if 1 in y_true and 0 in y_true:
                    from sklearn.metrics import roc_auc_score
                    auroc_p = roc_auc_score(y_true, y_score)
            except Exception:
                pass
        metrics_pos.append({"margin": m, "acc": float(np.mean([_within_margin_pos(ti, tk, m) for (_s, ti, _lm, tk, _rk) in known])) if total>0 else np.nan,
                            "precision_at_k": precision_p, "recall_at_k": recall_p, "f1_at_k": f1_p, "auroc": auroc_p})

        # Past-only
        hits_n = sum(_within_margin_neg(ti, tk, m) for (_s, ti, _lm, tk, _rk) in known)
        precision_n = hits_n / (total * top_k) if total > 0 else np.nan
        recall_n = hits_n / total if total > 0 else np.nan
        f1_n = (2 * precision_n * recall_n / (precision_n + recall_n)) if (precision_n + recall_n) > 0 else 0.0
        auroc_n = np.nan
        if probs is not None and total > 0:
            try:
                y_true, y_score = [], []
                for i, (_s, ti, _lm, _tk, _rk) in enumerate(known):
                    if ti < 0:
                        continue
                    pv = probs[i]
                    for j, p in enumerate(pv):
                        if not np.isfinite(p):
                            continue
                        y_score.append(float(p))
                        y_true.append(1 if (ti - j) >= 0 and (ti - j) <= m else 0)
                if 1 in y_true and 0 in y_true:
                    from sklearn.metrics import roc_auc_score
                    auroc_n = roc_auc_score(y_true, y_score)
            except Exception:
                pass
        metrics_neg.append({"margin": m, "acc": float(np.mean([_within_margin_neg(ti, tk, m) for (_s, ti, _lm, tk, _rk) in known])) if total>0 else np.nan,
                            "precision_at_k": precision_n, "recall_at_k": recall_n, "f1_at_k": f1_n, "auroc": auroc_n})

    return (pd.DataFrame(metrics), pd.DataFrame(metrics_pos), pd.DataFrame(metrics_neg))


def run_evaluation(scores_csv: Path, split: str, groundtruth_csv: str, results_dir: Path | None,
                   embedding: str | None, mode: str | None, top_k: int, max_margin: int, directional: bool,
                   stratified_year: bool, year_start: int | None = None, year_end: int | None = None):
    if not scores_csv.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv}")
    emb, md = _infer_embedding_and_mode(str(scores_csv), embedding, mode)

    df = pd.read_csv(scores_csv)
    if "site_name" not in df.columns:
        raise ValueError("scores_csv must contain 'site_name'")

    gt = pd.read_csv(groundtruth_csv)
    if "site_name" not in gt.columns:
        raise ValueError("groundtruth_csv must contain 'site_name'")
    if "looted_month" in gt.columns:
        gt["looted_month"] = gt["looted_month"].apply(normalize_month_str)
    gt = gt.set_index("site_name")

    df_split = _filter_sites_by_split(df, gt, split)
    probs, months_used = _select_score_matrix(df_split)
    # Filter by requested year range
    if (year_start is not None) or (year_end is not None):
        ys = year_start if year_start is not None else 0
        ye = year_end if year_end is not None else 9999
        keep_idx = [i for i, m in enumerate(months_used) if (int(m[:4]) >= ys and int(m[:4]) <= ye)]
        if len(keep_idx) == 0:
            raise ValueError(f"No months within requested year range {ys}-{ye} in {scores_csv}")
        months_used = [months_used[i] for i in keep_idx]
        probs = probs[:, keep_idx]
    rows = _rank_and_labels(df_split, probs, gt, top_k, stratified_year, months_used)
    result = _compute_metrics(rows, probs, top_k, max_margin, directional=directional)

    out_dir = results_dir if results_dir is not None and str(results_dir) != "" else scores_csv.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{emb}_{md}_{split}"

    def _round_df(mdf: pd.DataFrame) -> pd.DataFrame:
        out = mdf.copy()
        for c in out.columns:
            if c != "margin":
                out[c] = out[c].astype(float).round(3)
        return out

    if isinstance(result, tuple):
        m_sym, m_pos, m_neg = result
        _round_df(m_sym).to_csv(out_dir / f"{base}_metrics.csv", index=False)
        _round_df(m_pos).to_csv(out_dir / f"{base}_metrics_positive.csv", index=False)
        _round_df(m_neg).to_csv(out_dir / f"{base}_metrics_negative.csv", index=False)
    else:
        _round_df(result).to_csv(out_dir / f"{base}_metrics.csv", index=False)

    print(f"[done] {emb}/{md} split={split} metrics written under {out_dir}")


def main():
    args = parse_args()
    if is_legacy_mode(args.mode or ""):
        print(f"[warn] Legacy mode '{args.mode}' is deprecated; use '{normalize_mode(args.mode)}' instead")
    args.mode = normalize_mode(args.mode)
    # Resolve scores_csv from either direct file or output_dir+split
    if args.scores_csv:
        scores_csv = Path(args.scores_csv)
    else:
        scores_csv = Path(args.output_dir) / f"unsup_month_scores_{args.split}.csv"
    results_dir = Path(args.results_dir) if args.results_dir else None
    run_evaluation(scores_csv, args.split, args.groundtruth_csv, results_dir,
                   args.embedding, args.mode, args.top_k, args.max_margin, args.directional,
                   args.stratified_year, args.year_start, args.year_end)


if __name__ == "__main__":
    main()
