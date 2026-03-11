#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail

# Baseline pipeline: start monthly generation in tmux per embedding, merge to unified CSVs,
# evaluate (test+all), and aggregate recall matrices.
# Usage:
#   bash unsupervised_monthly/run_baseline_pipeline.sh start [EMBEDDINGS...]
#   bash unsupervised_monthly/run_baseline_pipeline.sh merge [EMBEDDINGS...]
#   bash unsupervised_monthly/run_baseline_pipeline.sh evaluate [EMBEDDINGS...]
#   bash unsupervised_monthly/run_baseline_pipeline.sh aggregate
#   bash unsupervised_monthly/run_baseline_pipeline.sh all [EMBEDDINGS...]
#
# Env overrides:
#   FEATURES_DIR_YEARLY (default planet_mosaics_final_4bands/features_new_with_mask)
#   FEATURES_UNIFIED_BASE (default planet_mosaics_final_4bands/features_unified_new_with_mask)
#   GROUNDTRUTH_CSV (default planet_mosaics_final_4bands/ground_truth_split_balanced_aux.csv)
#   SPLIT (default all)
#   BASELINE_NORM_METHOD (default sigmoid)
#   BASELINE_NORM_TEMPERATURE (default 1.0)
#   BASELINE_EXPORT_PROB (default 1)
#   TOP_K (default 24)
#   MAX_MARGIN (default 6)
#   DIRECTIONAL_MARGINS (default 1)
#   STRATIFIED_YEAR (default 1; set 0 to pass --no-stratified_year)
#   YEAR_START/YEAR_END (optional; e.g., 2017 and 2020)

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$REPO_ROOT"

FEATURES_DIR_YEARLY="${FEATURES_DIR_YEARLY:-planet_mosaics_final_4bands/features_new_with_mask}"
FEATURES_UNIFIED_BASE="${FEATURES_UNIFIED_BASE:-planet_mosaics_final_4bands/features_unified_new_with_mask}"
GROUNDTRUTH_CSV="${GROUNDTRUTH_CSV:-planet_mosaics_final_4bands/ground_truth_split_balanced_aux.csv}"
SPLIT="${SPLIT:-all}"
BASELINE_NORM_METHOD="${BASELINE_NORM_METHOD:-sigmoid}"
BASELINE_NORM_TEMPERATURE="${BASELINE_NORM_TEMPERATURE:-1.0}"
BASELINE_EXPORT_PROB="${BASELINE_EXPORT_PROB:-1}"
TOP_K="${TOP_K:-24}"
MAX_MARGIN="${MAX_MARGIN:-6}"
DIRECTIONAL_MARGINS="${DIRECTIONAL_MARGINS:-1}"
STRATIFIED_YEAR="${STRATIFIED_YEAR:-1}"
YEAR_START="${YEAR_START:-}"
YEAR_END="${YEAR_END:-}"

# Embedding map: labels to ids
declare -A EMBED_MAP=(
  [CLIP]="clip"
  [GEORSCLIP]="georsclip"
  [HANDCRAFTED]="handcrafted"
  [PRITHIVI]="prithvi-eo-2.0"
  [SATLASPRETRAIN]="satlaspretrain"
  [SATMAE]="satmae"
  [DINOV3]="dinov3"
  [SATCLIP]="satclip"
)

DEFAULT_LABELS=(CLIP GEORSCLIP HANDCRAFTED PRITHIVI SATLASPRETRAIN SATMAE DINOV3 SATCLIP)

unified_csv_for() {
  local embed_id="$1"
  echo "${FEATURES_UNIFIED_BASE}/features_${embed_id}_2017_2024_masked.csv"
}

ensure_session() {
  local name="$1"; shift
  local cmd="$*"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "[skip] tmux session '$name' already exists"
  else
    echo "[start] tmux session '$name'"
    tmux new-session -d -s "$name" -c "$REPO_ROOT" "bash -lc '$cmd'"
  fi
}

start_baseline() {
  local labels=("$@"); [[ ${#labels[@]} -eq 0 ]] && labels=("${DEFAULT_LABELS[@]}")
  for label in "${labels[@]}"; do
    local embed_id="${EMBED_MAP[$label]}"; [[ -z "$embed_id" ]] && { echo "[warn] Unknown label $label"; continue; }
    local unified_csv; unified_csv="$(unified_csv_for "$embed_id")"
    local out_dir="unsupervised_monthly/model_runs/${embed_id}"
    local gt_opt=""; [[ -n "$GROUNDTRUTH_CSV" ]] && gt_opt="--groundtruth_csv $GROUNDTRUTH_CSV"
    local norm_opts=""; [[ -n "$BASELINE_NORM_METHOD" ]] && norm_opts="--baseline_norm_method $BASELINE_NORM_METHOD --baseline_norm_temperature $BASELINE_NORM_TEMPERATURE"
    local prob_opt=""; [[ "$BASELINE_EXPORT_PROB" == "1" ]] && prob_opt="--baseline_export_probabilities"
    local cmd="source change_detect/bin/activate && python -u -m unsupervised_monthly.run_monthly_batch \
      --embedding ${embed_id} \
      --features_dir ${FEATURES_DIR_YEARLY} \
      --features_unified_csv ${unified_csv} \
      --split ${SPLIT} \
      --distance l2 \
      --output_dir ${out_dir} \
      ${gt_opt} \
      --rolling_window 6 \
      ${norm_opts} ${prob_opt}"
    ensure_session "${embed_id}_distance_baseline" "$cmd"
  done
}

merge_baseline() {
  local labels=("$@"); [[ ${#labels[@]} -eq 0 ]] && labels=("${DEFAULT_LABELS[@]}")
  for label in "${labels[@]}"; do
    local embed_id="${EMBED_MAP[$label]}"; [[ -z "$embed_id" ]] && { echo "[warn] Unknown label $label"; continue; }
    local out_dir="unsupervised_monthly/model_runs/${embed_id}"
    python -u -m unsupervised_monthly.merge_monthlies --out_dir "$out_dir" --mode distance_baseline --split "$SPLIT" --suffix _new
  done
}

evaluate_baseline() {
  local labels=("$@"); [[ ${#labels[@]} -eq 0 ]] && labels=("${DEFAULT_LABELS[@]}")
  local splits=(test all)
  for label in "${labels[@]}"; do
    local embed_id="${EMBED_MAP[$label]}"; [[ -z "$embed_id" ]] && { echo "[warn] Unknown label $label"; continue; }
    local csv="unsupervised_monthly/model_runs/${embed_id}/unsup_month_scores_all_distance_baseline_new.csv"
    [[ -f "$csv" ]] || { echo "[skip] ${embed_id}: unified baseline CSV missing"; continue; }
    for SPL in "${splits[@]}"; do
      python -u -m unsupervised_monthly.evaluate_unified_monthlies \
        --scores_csv "$csv" \
        --split "$SPL" \
        --groundtruth_csv "$GROUNDTRUTH_CSV" \
        --results_dir "unsupervised_monthly/results/${embed_id}" \
        --embedding "$embed_id" \
        --mode distance_baseline \
        --top_k "$TOP_K" \
        --max_margin "$MAX_MARGIN" \
        $([[ "$DIRECTIONAL_MARGINS" == 1 ]] && echo "--directional" || true) \
        $([[ "$STRATIFIED_YEAR" == 0 ]] && echo "--no-stratified_year" || true) \
        $([[ -n "$YEAR_START" ]] && echo "--year_start $YEAR_START" || true) \
        $([[ -n "$YEAR_END" ]] && echo "--year_end $YEAR_END" || true)
    done
  done
}

aggregate_recalls() {
  python -u unsupervised_monthly/aggregate_recalls.py
}

case "${1:-}" in
  start)
    shift; start_baseline "$@";;
  merge)
    shift; merge_baseline "$@";;
  evaluate)
    shift; evaluate_baseline "$@";;
  aggregate)
    aggregate_recalls;;
  all)
    shift; merge_baseline "$@"; evaluate_baseline "$@"; aggregate_recalls;;
  *)
    cat <<EOF
Usage: bash unsupervised_monthly/run_baseline_pipeline.sh <start|merge|evaluate|aggregate|all> [EMBEDDINGS...]
Examples:
  bash unsupervised_monthly/run_baseline_pipeline.sh start PRITHIVI DINOV3
  bash unsupervised_monthly/run_baseline_pipeline.sh merge PRITHIVI DINOV3
  TOP_K=12 STRATIFIED_YEAR=0 YEAR_START=2017 YEAR_END=2020 \
    bash unsupervised_monthly/run_baseline_pipeline.sh evaluate PRITHIVI DINOV3
  bash unsupervised_monthly/run_baseline_pipeline.sh aggregate
EOF
    exit 1;;
esac
