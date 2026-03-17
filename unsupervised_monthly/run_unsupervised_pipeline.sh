#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail

# Unsupervised pipeline: start monthly generation in tmux per embedding, merge to unified CSVs,
# evaluate (test+all), and aggregate recall matrices.
# Usage:
#   bash unsupervised_monthly/run_unsupervised_pipeline.sh start [EMBEDDINGS...]
#   bash unsupervised_monthly/run_unsupervised_pipeline.sh merge [EMBEDDINGS...]
#   bash unsupervised_monthly/run_unsupervised_pipeline.sh evaluate [EMBEDDINGS...]
#   bash unsupervised_monthly/run_unsupervised_pipeline.sh aggregate
#   bash unsupervised_monthly/run_unsupervised_pipeline.sh all [EMBEDDINGS...]
#
# Env overrides:
#   FEATURES_UNIFIED_BASE (default planet_mosaics_final_4bands/features_unified_new_with_mask)
#   GROUNDTRUTH_CSV (default planet_mosaics_final_4bands/ground_truth_split_balanced_aux.csv)
#   SPLIT (default all)
#   SCORE_NORM_METHOD (default sigmoid)
#   SCORE_NORM_TEMPERATURE (default 1.0)
#   EXPORT_PROB (default 1)
#   TOP_K (default 12)
#   MAX_MARGIN (default 6)
#   DIRECTIONAL_MARGINS (default 1)
#   STRATIFIED_YEAR (default 1; set 0 to pass --no-stratified_year)
#   YEAR_START/YEAR_END (default 2017/2020)

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$REPO_ROOT"

# Paths and env overrides
FEATURES_UNIFIED_BASE="${FEATURES_UNIFIED_BASE:-planet_mosaics_final_4bands/features_unified_new_with_mask}"
GROUNDTRUTH_CSV="${GROUNDTRUTH_CSV:-planet_mosaics_final_4bands/ground_truth_split_balanced_aux.csv}"
SPLIT="${SPLIT:-all}"
SCORE_NORM_METHOD="${SCORE_NORM_METHOD:-sigmoid}"
SCORE_NORM_TEMPERATURE="${SCORE_NORM_TEMPERATURE:-1.0}"
EXPORT_PROB="${EXPORT_PROB:-1}"
TOP_K="${TOP_K:-12}"
MAX_MARGIN="${MAX_MARGIN:-6}"
DIRECTIONAL_MARGINS="${DIRECTIONAL_MARGINS:-1}"
STRATIFIED_YEAR="${STRATIFIED_YEAR:-1}"
YEAR_START="${YEAR_START:-2017}"
YEAR_END="${YEAR_END:-2020}"
# Custom model runs and results directories
MODEL_RUNS_DIR="${MODEL_RUNS_DIR:-unsupervised_monthly/model_runs}"
RESULTS_DIR="${RESULTS_DIR:-unsupervised_monthly/results}"
mkdir -p "$MODEL_RUNS_DIR" "$RESULTS_DIR"

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
  local masked="${FEATURES_UNIFIED_BASE}/features_${embed_id}_2017_2024_masked.csv"
  local plain="${FEATURES_UNIFIED_BASE}/features_${embed_id}_2017_2024.csv"
  if [[ -f "$masked" ]]; then
    echo "$masked"
  elif [[ -f "$plain" ]]; then
    echo "$plain"
  else
    # Fallback to plain naming even if not present (downstream will error clearly)
    echo "$plain"
  fi
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

start_unsup() {
  local labels=("$@"); [[ ${#labels[@]} -eq 0 ]] && labels=("${DEFAULT_LABELS[@]}")
  for label in "${labels[@]}"; do
    local embed_id="${EMBED_MAP[$label]}"; [[ -z "$embed_id" ]] && { echo "[warn] Unknown label $label"; continue; }
    local unified_csv; unified_csv="$(unified_csv_for "$embed_id")"
    local out_dir="${MODEL_RUNS_DIR}/${embed_id}"
    local gt_opt=""; [[ -n "$GROUNDTRUTH_CSV" ]] && gt_opt="--groundtruth_csv $GROUNDTRUTH_CSV"
    local prob_opt=""; [[ "$EXPORT_PROB" == "1" ]] && prob_opt="--export_probabilities"
    local cmd="source change_detect/bin/activate && python -u -m unsupervised_monthly.run_monthly_batch \
      --embedding ${embed_id} \
      --features_dir ${unified_csv} \
      --split ${SPLIT} \
      --output_dir ${out_dir} \
      --use_unsup_model \
      ${gt_opt} \
      ${prob_opt} \
      --score_norm_method ${SCORE_NORM_METHOD} --score_norm_temperature ${SCORE_NORM_TEMPERATURE}"
    ensure_session "${embed_id}_learned_unsupervised" "$cmd"
  done
}

merge_unsup() {
  local labels=("$@"); [[ ${#labels[@]} -eq 0 ]] && labels=("${DEFAULT_LABELS[@]}")
  for label in "${labels[@]}"; do
    local embed_id="${EMBED_MAP[$label]}"; [[ -z "$embed_id" ]] && { echo "[warn] Unknown label $label"; continue; }
    local out_dir="${MODEL_RUNS_DIR}/${embed_id}"
    python -u -m unsupervised_monthly.merge_monthlies --out_dir "$out_dir" --mode learned_unsupervised --split "$SPLIT" --suffix _new
  done
}

evaluate_unsup() {
  local labels=("$@"); [[ ${#labels[@]} -eq 0 ]] && labels=("${DEFAULT_LABELS[@]}")
  local splits=(test all)
  for label in "${labels[@]}"; do
    local embed_id="${EMBED_MAP[$label]}"; [[ -z "$embed_id" ]] && { echo "[warn] Unknown label $label"; continue; }
    local csv="${MODEL_RUNS_DIR}/${embed_id}/unsup_month_scores_all_learned_unsupervised_new.csv"
    [[ -f "$csv" ]] || { echo "[skip] ${embed_id}: unified unsupervised CSV missing"; continue; }
    for SPL in "${splits[@]}"; do
      python -u -m unsupervised_monthly.evaluate_unified_monthlies \
        --scores_csv "$csv" \
        --split "$SPL" \
        --groundtruth_csv "$GROUNDTRUTH_CSV" \
        --results_dir "${RESULTS_DIR}/${embed_id}" \
        --embedding "$embed_id" \
        --mode learned_unsupervised \
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
    shift; start_unsup "$@";;
  merge)
    shift; merge_unsup "$@";;
  evaluate)
    shift; evaluate_unsup "$@";;
  aggregate)
    aggregate_recalls;;
  all)
    shift; merge_unsup "$@"; evaluate_unsup "$@"; aggregate_recalls;;
  *)
    cat <<EOF
Usage: bash unsupervised_monthly/run_unsupervised_pipeline.sh <start|merge|evaluate|aggregate|all> [EMBEDDINGS...]
Examples:
  bash unsupervised_monthly/run_unsupervised_pipeline.sh start PRITHIVI DINOV3
  bash unsupervised_monthly/run_unsupervised_pipeline.sh merge PRITHIVI DINOV3
  TOP_K=12 STRATIFIED_YEAR=0 YEAR_START=2017 YEAR_END=2020 \
    bash unsupervised_monthly/run_unsupervised_pipeline.sh evaluate PRITHIVI DINOV3
  bash unsupervised_monthly/run_unsupervised_pipeline.sh aggregate
EOF
    exit 1;;
esac
