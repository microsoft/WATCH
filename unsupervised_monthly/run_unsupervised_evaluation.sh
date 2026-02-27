#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Evaluate unified monthly aggregated results for each embedding and split
# Uses unsupervised_monthly/evaluate_unified_monthlies.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ground truth and paths
GROUNDTRUTH_CSV="${ROOT_DIR}/planet_mosaics_final_4bands/ground_truth_split_balanced_aux.csv"
# Allow env overrides for base directories
MODEL_RUNS_DIR="${MODEL_RUNS_DIR:-${SCRIPT_DIR}/model_runs}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"

TOP_K=${TOP_K:-12}
MAX_MARGIN=${MAX_MARGIN:-6}
DIRECTIONAL=${DIRECTIONAL_MARGINS:-1}  # default on
# Stratification control: 1=on (default), 0=off -> pass --no-stratified_year
STRATIFIED_YEAR=${STRATIFIED_YEAR:-1}
# Year range filter (optional). Set YEAR_START/YEAR_END env to limit months.
YEAR_START=${YEAR_START:-}
YEAR_END=${YEAR_END:-}

if [[ ! -f "${GROUNDTRUTH_CSV}" ]]; then
  echo "[ERROR] Ground truth CSV missing: ${GROUNDTRUTH_CSV}" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

usage(){ cat <<EOF
Usage: bash ${0##*/} [--models a,b,c] [--no-directional]
Env:
  TOP_K (default ${TOP_K})
  MAX_MARGIN (default ${MAX_MARGIN})
  DIRECTIONAL_MARGINS=0 to disable directional metrics
EOF
};

MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) shift; IFS=',' read -r -a MODELS <<<"$1"; shift;;
    --no-directional) DIRECTIONAL=0; shift;;
    --help|-h) usage; exit 0;;
    *) echo "[warn] Unknown arg: $1"; shift;;
  esac
done

# Discover embeddings if none specified
if [[ ${#MODELS[@]} -eq 0 ]]; then
  mapfile -t MODELS < <(ls -1 "${MODEL_RUNS_DIR}" | grep -vE "^\.|__pycache__" || true)
fi

echo "[info] Evaluating embeddings: ${MODELS[*]}"

PY_EVAL_MODULE="unsupervised_monthly.evaluate_unified_monthlies"

for EMB in "${MODELS[@]}"; do
  EMB_DIR="${MODEL_RUNS_DIR}/${EMB}"
  [[ -d "$EMB_DIR" ]] || { echo "[skip] ${EMB}: directory missing"; continue; }

  # Prefer canonical *_new.csv if present, else fallback to legacy names
  declare -A MODE_TO_FILE
  if [[ -f "${EMB_DIR}/unsup_month_scores_all_distance_baseline_new.csv" ]]; then
    MODE_TO_FILE[distance_baseline]="${EMB_DIR}/unsup_month_scores_all_distance_baseline_new.csv"
  elif [[ -f "${EMB_DIR}/unsup_month_scores_all_distance_baseline.csv" ]]; then
    MODE_TO_FILE[distance_baseline]="${EMB_DIR}/unsup_month_scores_all_distance_baseline.csv"
  elif [[ -f "${EMB_DIR}/unsup_month_scores_all_baseline_new.csv" ]]; then
    MODE_TO_FILE[distance_baseline]="${EMB_DIR}/unsup_month_scores_all_baseline_new.csv"
  elif [[ -f "${EMB_DIR}/unsup_month_scores_all_baseline.csv" ]]; then
    MODE_TO_FILE[distance_baseline]="${EMB_DIR}/unsup_month_scores_all_baseline.csv"
  fi
  if [[ -f "${EMB_DIR}/unsup_month_scores_all_learned_unsupervised_new.csv" ]]; then
    MODE_TO_FILE[learned_unsupervised]="${EMB_DIR}/unsup_month_scores_all_learned_unsupervised_new.csv"
  elif [[ -f "${EMB_DIR}/unsup_month_scores_all_learned_unsupervised.csv" ]]; then
    MODE_TO_FILE[learned_unsupervised]="${EMB_DIR}/unsup_month_scores_all_learned_unsupervised.csv"
  elif [[ -f "${EMB_DIR}/unsup_month_scores_all_unsupervised_new.csv" ]]; then
    MODE_TO_FILE[learned_unsupervised]="${EMB_DIR}/unsup_month_scores_all_unsupervised_new.csv"
  elif [[ -f "${EMB_DIR}/unsup_month_scores_all_unsupervised.csv" ]]; then
    MODE_TO_FILE[learned_unsupervised]="${EMB_DIR}/unsup_month_scores_all_unsupervised.csv"
  fi

  if [[ ${#MODE_TO_FILE[@]} -eq 0 ]]; then
    echo "[warn] No unified CSVs found for ${EMB}; expected one of unsup_month_scores_all_{distance_baseline,learned_unsupervised}[_new].csv (or legacy {baseline,unsupervised})"
    continue
  fi

  for MODE in "${!MODE_TO_FILE[@]}"; do
    CSV="${MODE_TO_FILE[$MODE]}"
    for SPLIT in test all; do
      echo "[eval] ${EMB} (${MODE}) split=${SPLIT}"
      python -u -m "$PY_EVAL_MODULE" \
        --scores_csv "$CSV" \
        --split "$SPLIT" \
        --groundtruth_csv "$GROUNDTRUTH_CSV" \
        --results_dir "$RESULTS_DIR/$EMB" \
        --embedding "$EMB" \
        --mode "$MODE" \
        --top_k "$TOP_K" \
        --max_margin "$MAX_MARGIN" \
        $([[ "$DIRECTIONAL" == 1 ]] && echo "--directional" || true) \
        $([[ "$STRATIFIED_YEAR" == 0 ]] && echo "--no-stratified_year" || true) \
        $([[ -n "$YEAR_START" ]] && echo "--year_start $YEAR_START" || true) \
        $([[ -n "$YEAR_END" ]] && echo "--year_end $YEAR_END" || true)
    done
  done

  # Materialize a single merged per-site table (site_name, known_month_of_change, 2017_01..2024_12)
  # under unsupervised_monthly/results/<embedding>/.
  python -u "${ROOT_DIR}/export_merged_monthly_inference_tables.py" \
    --repo_root "${ROOT_DIR}" \
    --pipelines unsupervised_monthly \
    --only_embedding "${EMB}" \
    --year_start 2017 --year_end 2024
done

echo "[done] Results written to ${RESULTS_DIR}"
