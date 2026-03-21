#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Export a single merged per-site inference table for each embedding.

Creates, for each embedding, a CSV with columns:
  site_name, known_month_of_change, 2017_01, ..., 2024_12

It covers:
- temporal_embedding_distance (TED mode)
- self_supervised_change_detection (SSCD mode)
- weakly_supervised (weakly_supervised mode)

The source "scores" CSVs already contain per-month probabilities, but are stored
under the model_runs folders. This script materializes a single merged artifact
under the corresponding results/<embedding>/ folders.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd

MONTH_RE = re.compile(r"^(\d{4})_(\d{2})$")


def month_columns(year_start: int, year_end: int) -> List[str]:
    cols: List[str] = []
    for y in range(year_start, year_end + 1):
        for m in range(1, 13):
            cols.append(f"{y}_{m:02d}")
    return cols


def normalize_known_month(val: object, year_start: int, year_end: int) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    m = MONTH_RE.match(s)
    if not m:
        return ""
    y = int(m.group(1))
    mm = int(m.group(2))
    if y < year_start or y > year_end:
        return ""
    if mm < 1 or mm > 12:
        return ""
    return s


def load_known_month_map(groundtruth_csv: str, year_start: int, year_end: int) -> pd.DataFrame:
    gt = pd.read_csv(groundtruth_csv)
    if "site_name" not in gt.columns:
        raise ValueError(f"groundtruth_csv missing site_name: {groundtruth_csv}")

    looted_col = "looted" if "looted" in gt.columns else None
    looted_month_col = "looted_month" if "looted_month" in gt.columns else None
    if looted_month_col is None:
        raise ValueError(f"groundtruth_csv missing looted_month: {groundtruth_csv}")

    out = gt[["site_name", looted_month_col]].copy()
    out["known_month_of_change"] = out[looted_month_col].apply(
        lambda v: normalize_known_month(v, year_start, year_end)
    )

    if looted_col is not None:
        # Only keep known months for confirmed looted sites.
        out.loc[gt[looted_col].astype(int) != 1, "known_month_of_change"] = ""

    return out[["site_name", "known_month_of_change"]]


@dataclass(frozen=True)
class ExportJob:
    embedding: str
    mode: str
    scores_csv: str
    out_csv: str


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def select_existing(*paths: str) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def infer_ted_jobs(
    repo_root: str, year_start: int, year_end: int
) -> List[ExportJob]:
    """Temporal Embedding Distance (TED) jobs."""
    model_runs_dir = os.path.join(repo_root, "temporal_embedding_distance", "model_runs")
    results_dir = os.path.join(repo_root, "temporal_embedding_distance", "results")
    # Fallback to old directory name for backward compat
    if not os.path.isdir(model_runs_dir):
        model_runs_dir = os.path.join(repo_root, "self_supervised_change_detection", "model_runs")
        results_dir = os.path.join(repo_root, "self_supervised_change_detection", "results")
    if not os.path.isdir(model_runs_dir):
        return []

    jobs: List[ExportJob] = []
    for embedding in sorted(os.listdir(model_runs_dir)):
        emb_dir = os.path.join(model_runs_dir, embedding)
        if not os.path.isdir(emb_dir):
            continue

        # TED canonical with legacy fallbacks
        baseline_src = select_existing(
            os.path.join(emb_dir, "unsup_month_scores_all_temporal_embedding_distance.csv"),
            os.path.join(emb_dir, "unsup_month_scores_all_distance_baseline_new.csv"),
            os.path.join(emb_dir, "unsup_month_scores_all_distance_baseline.csv"),
            os.path.join(emb_dir, "unsup_month_scores_all_baseline_new.csv"),
            os.path.join(emb_dir, "unsup_month_scores_all_baseline.csv"),
        )
        if baseline_src:
            jobs.append(
                ExportJob(
                    embedding=embedding,
                    mode="temporal_embedding_distance",
                    scores_csv=baseline_src,
                    out_csv=os.path.join(
                        results_dir, embedding, "inference_all_months_temporal_embedding_distance.csv"
                    ),
                )
            )

    return jobs


def infer_sscd_jobs(
    repo_root: str, year_start: int, year_end: int
) -> List[ExportJob]:
    """Self-Supervised Change Detection (SSCD) jobs."""
    model_runs_dir = os.path.join(repo_root, "self_supervised_change_detection", "model_runs")
    results_dir = os.path.join(repo_root, "self_supervised_change_detection", "results")
    if not os.path.isdir(model_runs_dir):
        return []

    jobs: List[ExportJob] = []
    for embedding in sorted(os.listdir(model_runs_dir)):
        emb_dir = os.path.join(model_runs_dir, embedding)
        if not os.path.isdir(emb_dir):
            continue

        # SSCD canonical with legacy fallbacks
        unsup_src = select_existing(
            os.path.join(emb_dir, "unsup_month_scores_all_self_supervised_change_detection_new.csv"),
            os.path.join(emb_dir, "unsup_month_scores_all_self_supervised_change_detection.csv"),
            os.path.join(emb_dir, "unsup_month_scores_all_learned_unsupervised_new.csv"),
            os.path.join(emb_dir, "unsup_month_scores_all_learned_unsupervised.csv"),
            os.path.join(emb_dir, "unsup_month_scores_all_unsupervised_new.csv"),
            os.path.join(emb_dir, "unsup_month_scores_all_unsupervised.csv"),
        )
        if unsup_src:
            jobs.append(
                ExportJob(
                    embedding=embedding,
                    mode="self_supervised_change_detection",
                    scores_csv=unsup_src,
                    out_csv=os.path.join(
                        results_dir, embedding, "inference_all_months_self_supervised_change_detection.csv"
                    ),
                )
            )

    return jobs


def infer_weakly_supervised_monthly_jobs(repo_root: str) -> List[ExportJob]:
    model_runs_dir = os.path.join(repo_root, "weakly_supervised", "model_runs")
    results_dir = os.path.join(repo_root, "weakly_supervised", "results")
    if not os.path.isdir(model_runs_dir):
        return []

    jobs: List[ExportJob] = []
    for embedding in sorted(os.listdir(model_runs_dir)):
        emb_dir = os.path.join(model_runs_dir, embedding)
        if not os.path.isdir(emb_dir):
            continue

        src = os.path.join(emb_dir, "unsup_month_scores_all_weakly_supervised.csv")
        if os.path.exists(src):
            jobs.append(
                ExportJob(
                    embedding=embedding,
                    mode="weakly_supervised",
                    scores_csv=src,
                    out_csv=os.path.join(
                        results_dir,
                        embedding,
                        "inference_all_months_weakly_supervised.csv",
                    ),
                )
            )

    return jobs


def export_one(
    job: ExportJob,
    known_month_df: pd.DataFrame,
    months: List[str],
) -> Tuple[int, List[str]]:
    df = pd.read_csv(job.scores_csv)
    if "site_name" not in df.columns:
        raise ValueError(f"scores_csv missing site_name: {job.scores_csv}")

    missing_months = [c for c in months if c not in df.columns]
    if missing_months:
        # Some older exports cover only a subset of years. We still materialize
        # the full 2017_01..2024_12 table and leave missing months as NaN.
        for c in missing_months:
            df[c] = pd.NA

    out = df[["site_name"] + months].copy()
    out = out.merge(known_month_df, on="site_name", how="left")
    out["known_month_of_change"] = out["known_month_of_change"].fillna("")

    # Column order: site_name, known_month_of_change, months...
    out = out[["site_name", "known_month_of_change"] + months]

    ensure_parent_dir(job.out_csv)
    out.to_csv(job.out_csv, index=False)
    return len(out), list(out.columns)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--groundtruth_csv",
        type=str,
        default="planet_mosaics_final_4bands/ground_truth_split_balanced_aux.csv",
    )
    ap.add_argument("--year_start", type=int, default=2017)
    ap.add_argument("--year_end", type=int, default=2024)
    ap.add_argument(
        "--repo_root",
        type=str,
        default=".",
        help="Path to repo root (default: current directory)",
    )
    ap.add_argument(
        "--only_embedding",
        type=str,
        default="",
        help="If set, only export for this embedding ID (e.g., clip).",
    )
    ap.add_argument(
        "--pipelines",
        type=str,
        default="all",
        choices=[
            "all",
            "temporal_embedding_distance",
            "self_supervised_change_detection",
            "weakly_supervised",
            # legacy aliases
            "unsupervised_monthly",
            "weakly_supervised_monthly",
        ],
        help="Which pipeline(s) to export (default: all).",
    )
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    gt_path = os.path.join(repo_root, args.groundtruth_csv)
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"groundtruth_csv not found: {gt_path}")

    months = month_columns(args.year_start, args.year_end)
    known_month_df = load_known_month_map(gt_path, args.year_start, args.year_end)

    jobs: List[ExportJob] = []
    if args.pipelines in {"all", "temporal_embedding_distance", "unsupervised_monthly"}:
        jobs.extend(infer_ted_jobs(repo_root, args.year_start, args.year_end))
    if args.pipelines in {"all", "self_supervised_change_detection"}:
        jobs.extend(infer_sscd_jobs(repo_root, args.year_start, args.year_end))
    if args.pipelines in {"all", "weakly_supervised", "weakly_supervised_monthly"}:
        jobs.extend(infer_weakly_supervised_monthly_jobs(repo_root))

    if args.only_embedding:
        jobs = [j for j in jobs if j.embedding == args.only_embedding]

    if not jobs:
        print("[warn] No export jobs found (no model_runs folders or no unified scores CSVs).")
        return 0

    print(f"[info] exporting merged monthly inference tables: jobs={len(jobs)}")

    failures = 0
    for job in jobs:
        try:
            nrows, cols = export_one(job, known_month_df, months)
            print(
                f"[ok] {job.embedding:<16} {job.mode:<16} -> {os.path.relpath(job.out_csv, repo_root)} (rows={nrows}, cols={len(cols)})"
            )
        except Exception as e:
            failures += 1
            print(
                f"[error] {job.embedding:<16} {job.mode:<16} from {os.path.relpath(job.scores_csv, repo_root)}: {e}"
            )

    if failures:
        print(f"[warn] completed with failures={failures}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
