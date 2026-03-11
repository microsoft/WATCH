#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail

# TED (Temporal Embedding Distance) pipeline: start monthly generation in tmux per embedding,
# evaluate (test+all), and aggregate recall matrices.
# Usage:
#   bash self_supervised_change_detection/run_baseline_pipeline.sh start [EMBEDDINGS...]
#   bash self_supervised_change_detection/run_baseline_pipeline.sh evaluate [EMBEDDINGS...]
#   bash self_supervised_change_detection/run_baseline_pipeline.sh aggregate
#   bash self_supervised_change_detection/run_baseline_pipeline.sh all [EMBEDDINGS...]
#
# Env overrides:
#   FEATURES_DIR_YEARLY (default planet_mosaics_final_4bands/features_new_with_mask)
#   FEATURES_UNIFIED_BASE (default planet_mosaics_final_4bands/features_unified_new_with_mask)
#   GROUNDTRUTH_CSV (default planet_mosaics_final_4bands/ground_truth_split_balanced_aux.csv)
#   SPLIT (default all)
#   BASELINE_NORM_METHOD (default sigmoid)
#   BASELINE_NORM_TEMPERATURE (default 1.0)
#   BASELINE_EXPORT_PROB (default 1)
#   BASELINE_RW_WINDOW (default 12; rolling window for per-site robust z-score calibration)
#   TOP_K (default 12)
#   MAX_MARGIN (default 6)
#   DIRECTIONAL_MARGINS (default 1)
#   STRATIFIED_YEAR (default 1; set 0 to pass --no-stratified_year)
#   YEAR_START/YEAR_END (default 2017/2020)

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$REPO_ROOT"

FEATURES_DIR_YEARLY="${FEATURES_DIR_YEARLY:-planet_mosaics_final_4bands/features_new_with_mask}"
FEATURES_UNIFIED_BASE="${FEATURES_UNIFIED_BASE:-planet_mosaics_final_4bands/features_unified_new_with_mask}"
GROUNDTRUTH_CSV="${GROUNDTRUTH_CSV:-planet_mosaics_final_4bands/ground_truth_split_balanced_aux.csv}"
SPLIT="${SPLIT:-all}"
BASELINE_NORM_METHOD="${BASELINE_NORM_METHOD:-sigmoid}"
BASELINE_NORM_TEMPERATURE="${BASELINE_NORM_TEMPERATURE:-1.0}"
BASELINE_EXPORT_PROB="${BASELINE_EXPORT_PROB:-1}"
BASELINE_RW_WINDOW="${BASELINE_RW_WINDOW:-12}"
TOP_K="${TOP_K:-12}"
MAX_MARGIN="${MAX_MARGIN:-6}"
DIRECTIONAL_MARGINS="${DIRECTIONAL_MARGINS:-1}"
STRATIFIED_YEAR="${STRATIFIED_YEAR:-1}"
YEAR_START="${YEAR_START:-2017}"
YEAR_END="${YEAR_END:-2020}"
MODEL_RUNS_DIR="${MODEL_RUNS_DIR:-temporal_embedding_distance/model_runs}"
RESULTS_DIR="${RESULTS_DIR:-temporal_embedding_distance/results}"

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

DEFAULT_LABELS=(CLIP GEORSCLIP HANDCRAFTED PRITHIVI SATLASPRETRAIN SATMAE DINOV3)

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
    local out_dir="${MODEL_RUNS_DIR}/${embed_id}"
    local scaler_path="self_supervised_change_detection/model_runs/${embed_id}/scaler_stats.npz"
    local out_csv="${out_dir}/unsup_month_scores_${SPLIT}_temporal_embedding_distance.csv"
    mkdir -p "$out_dir"
    local cmd="source change_detect/bin/activate && python -u -m temporal_embedding_distance.infer_all_months \
      --features_csv ${unified_csv} \
      --scaler_path ${scaler_path} \
      --output_csv ${out_csv} \
      --distance l2 \
      --rw_window ${BASELINE_RW_WINDOW} \
      --score_norm_method ${BASELINE_NORM_METHOD} \
      --score_norm_temperature ${BASELINE_NORM_TEMPERATURE}"
    ensure_session "${embed_id}_temporal_embedding_distance" "$cmd"
  done
}

evaluate_baseline() {
  local labels=("$@"); [[ ${#labels[@]} -eq 0 ]] && labels=("${DEFAULT_LABELS[@]}")
  local splits=(test all)
  for label in "${labels[@]}"; do
    local embed_id="${EMBED_MAP[$label]}"; [[ -z "$embed_id" ]] && { echo "[warn] Unknown label $label"; continue; }
    local csv="${MODEL_RUNS_DIR}/${embed_id}/unsup_month_scores_all_temporal_embedding_distance.csv"
    # Fallback to old filename for backward compat
    [[ -f "$csv" ]] || csv="${MODEL_RUNS_DIR}/${embed_id}/unsup_month_scores_all_distance_baseline_new.csv"
    [[ -f "$csv" ]] || { echo "[skip] ${embed_id}: unified TED CSV missing"; continue; }
    local res_dir="${RESULTS_DIR}/${embed_id}"
    mkdir -p "$res_dir"
    for SPL in "${splits[@]}"; do
      python -u -m self_supervised_change_detection.evaluate_unified_monthlies \
        --scores_csv "$csv" \
        --split "$SPL" \
        --groundtruth_csv "$GROUNDTRUTH_CSV" \
        --results_dir "$res_dir" \
        --embedding "$embed_id" \
        --mode temporal_embedding_distance \
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
  python -u -m self_supervised_change_detection.aggregate_recalls \
    --model_runs_dir "$MODEL_RUNS_DIR" \
    --results_dir "$RESULTS_DIR"
}

case "${1:-}" in
  start)
    shift; start_baseline "$@";;
  evaluate)
    shift; evaluate_baseline "$@";;
  aggregate)
    aggregate_recalls;;
  all)
    # infer_all_months writes unified CSV directly, no merge step needed
    shift
    local_labels=("$@")
    start_baseline "${local_labels[@]}"
    evaluate_baseline "${local_labels[@]}"
    aggregate_recalls;;
  *)
    cat <<EOF
Usage: bash self_supervised_change_detection/run_baseline_pipeline.sh <start|evaluate|aggregate|all> [EMBEDDINGS...]
Examples:
  bash self_supervised_change_detection/run_baseline_pipeline.sh start PRITHIVI DINOV3
  TOP_K=12 STRATIFIED_YEAR=0 YEAR_START=2017 YEAR_END=2020 \
    bash self_supervised_change_detection/run_baseline_pipeline.sh evaluate PRITHIVI DINOV3
  bash self_supervised_change_detection/run_baseline_pipeline.sh aggregate
EOF
    exit 1;;
esac
