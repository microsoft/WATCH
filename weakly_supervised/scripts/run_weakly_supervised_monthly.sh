#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail

###############################################################################
# Weakly-supervised monthly runner
#
# - One tmux session per embedding (clip, dinov3, georsclip, handcrafted,
#   prithvi-eo-2.0, satlaspretrain, satmae)
# - Trains using month labels up to LABEL_END_MONTH (default: 2020_12)
# - Runs inference for the full window 2017_01..2024_12
# - Evaluates with the existing monthly evaluator (margins 0..6, plus
#   directional positive/negative) for splits: test and all
# - Writes outputs under weakly_supervised/
###############################################################################

REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
cd "$REPO_ROOT"

FEATURES_UNIFIED_BASE="${FEATURES_UNIFIED_BASE:-planet_mosaics_final_4bands/features_unified_new_with_mask}"
GROUNDTRUTH_CSV="${GROUNDTRUTH_CSV:-planet_mosaics_final_4bands/ground_truth_split_balanced_aux.csv}"
SPLIT_COL="${SPLIT_COL:-split}"
LABEL_END_MONTH="${LABEL_END_MONTH:-2020_12}"
LABEL_WINDOW="${LABEL_WINDOW:-1}"
LABEL_SMOOTH_TYPE="${LABEL_SMOOTH_TYPE:-gauss}"

# Training knobs (defaults chosen for heavy class imbalance: only ~38 known-month positives)
EPOCHS="${EPOCHS:-50}"
# If POS_WEIGHT <= 0, training auto-computes based on effective positive mass.
POS_WEIGHT="${POS_WEIGHT:-0}"
MAX_POS_WEIGHT="${MAX_POS_WEIGHT:-200}"
# Optional: oversample known-month positive sites by this factor (>1 enables sampler)
OVERSAMPLE_POS_SITES="${OVERSAMPLE_POS_SITES:-0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}"
NUM_WORKERS="${NUM_WORKERS:-0}"

# Output locations (requested)
MODEL_RUNS_DIR="${MODEL_RUNS_DIR:-weakly_supervised/model_runs}"
RESULTS_DIR="${RESULTS_DIR:-weakly_supervised/results}"
mkdir -p "$MODEL_RUNS_DIR" "$RESULTS_DIR"

# Evaluation knobs
TOP_K="${TOP_K:-12}"
MAX_MARGIN="${MAX_MARGIN:-6}"
YEAR_START="${YEAR_START:-2017}"
YEAR_END="${YEAR_END:-2020}"

declare -A EMBED_MAP=(
  [CLIP]="clip"
  [DINOV3]="dinov3"
  [GEORSCLIP]="georsclip"
  [HANDCRAFTED]="handcrafted"
  [PRITHIVI]="prithvi-eo-2.0"
  [SATLASPRETRAIN]="satlaspretrain"
  [SATMAE]="satmae"
)

DEFAULT_LABELS=(CLIP DINOV3 GEORSCLIP HANDCRAFTED PRITHIVI SATLASPRETRAIN SATMAE)

unified_csv_for() {
  local embed_id="$1"
  local masked="${FEATURES_UNIFIED_BASE}/features_${embed_id}_2017_2024_masked.csv"
  local plain="${FEATURES_UNIFIED_BASE}/features_${embed_id}_2017_2024.csv"
  if [[ -f "$masked" ]]; then
    echo "$masked"
  else
    echo "$plain"
  fi
}

ensure_session() {
  local name="$1"; shift
  local cmd="$1"
  # tmux session names cannot contain '.' on some setups; normalize to safe chars.
  name="${name//[^A-Za-z0-9_-]/_}"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "[skip] tmux session '$name' already exists"
  else
    echo "[start] tmux session '$name'"
    tmux new-session -d -s "$name" -c "$REPO_ROOT" "bash \"$cmd\""
  fi
}

start_sessions() {
  local labels=("$@"); [[ ${#labels[@]} -eq 0 ]] && labels=("${DEFAULT_LABELS[@]}")

  for label in "${labels[@]}"; do
    local embed_id="${EMBED_MAP[$label]:-}"
    if [[ -z "$embed_id" ]]; then
      echo "[warn] Unknown embedding label: $label"
      continue
    fi

    local features_csv
    features_csv="$(unified_csv_for "$embed_id")"
    if [[ ! -f "$features_csv" ]]; then
      echo "[skip] ${embed_id}: missing features CSV: $features_csv"
      continue
    fi

    local out_dir="${MODEL_RUNS_DIR}/${embed_id}"
    local res_dir="${RESULTS_DIR}/${embed_id}"
    mkdir -p "$out_dir" "$res_dir"

    # Optional GPU assignment per session
    local device="${DEVICE:-cuda}"
    local gpu_id="${GPU_ID:-0}"
    local keep_open="${TMUX_KEEP_OPEN:-1}"

    local run_file="${out_dir}/run_in_tmux.sh"
    local log_file="${out_dir}/tmux.log"

    cat >"$run_file" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd "$REPO_ROOT"

exec > >(tee -a "$log_file") 2>&1

echo "[info] embedding=${embed_id}"
echo "[info] features_csv=${features_csv}"
echo "[info] model_runs=${out_dir}"
echo "[info] results=${res_dir}"
echo "[info] device=${device} gpu_id=${gpu_id}"
echo "[info] label_end_month=${LABEL_END_MONTH}"
echo "[info] epochs=${EPOCHS} pos_weight=${POS_WEIGHT} max_pos_weight=${MAX_POS_WEIGHT} oversample_pos_sites=${OVERSAMPLE_POS_SITES} early_stop_patience=${EARLY_STOP_PATIENCE}"
echo "[info] label_window=${LABEL_WINDOW} label_smooth_type=${LABEL_SMOOTH_TYPE}"
echo "[info] num_workers=${NUM_WORKERS}"

source change_detect/bin/activate

python -u -m weakly_supervised.train \
  --features_csv "${features_csv}" \
  --groundtruth_csv "${GROUNDTRUTH_CSV}" \
  --split_col "${SPLIT_COL}" \
  --label_end_month "${LABEL_END_MONTH}" \
  --output_dir "${out_dir}" \
  --label_window "${LABEL_WINDOW}" \
  --label_smooth_type "${LABEL_SMOOTH_TYPE}" \
  --epochs "${EPOCHS}" \
  --pos_weight "${POS_WEIGHT}" \
  --max_pos_weight "${MAX_POS_WEIGHT}" \
  --oversample_pos_sites "${OVERSAMPLE_POS_SITES}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${device}" --gpu_id "${gpu_id}"
rc=\$?
if [[ \$rc -ne 0 ]]; then
  echo "[error] train failed rc=\$rc"
  [[ "${keep_open}" == "1" ]] && exec bash
  exit \$rc
fi

python -u -m weakly_supervised.infer \
  --features_csv "${features_csv}" \
  --model_path "${out_dir}/model.pt" \
  --scaler_path "${out_dir}/scaler_stats.npz" \
  --output_csv "${out_dir}/unsup_month_scores_all_weakly_supervised.csv" \
  --mode weakly_supervised \
  --device "${device}" --gpu_id "${gpu_id}"
rc=\$?
if [[ \$rc -ne 0 ]]; then
  echo "[error] infer failed rc=\$rc"
  [[ "${keep_open}" == "1" ]] && exec bash
  exit \$rc
fi

for SPL in test all; do
  python -u -m self_supervised_change_detection.evaluate_unified_monthlies \
    --scores_csv "${out_dir}/unsup_month_scores_all_weakly_supervised.csv" \
    --split "\${SPL}" \
    --groundtruth_csv "${GROUNDTRUTH_CSV}" \
    --results_dir "${res_dir}" \
    --embedding "${embed_id}" \
    --mode weakly_supervised \
    --top_k "${TOP_K}" \
    --max_margin "${MAX_MARGIN}" \
    --directional \
    --no-stratified_year \
    --year_start "${YEAR_START}" --year_end "${YEAR_END}"
  rc=\$?
  if [[ \$rc -ne 0 ]]; then
    echo "[error] evaluate failed split=\${SPL} rc=\$rc"
    [[ "${keep_open}" == "1" ]] && exec bash
    exit \$rc
  fi
done

python -u "${REPO_ROOT}/export_merged_monthly_inference_tables.py" \
  --repo_root "${REPO_ROOT}" \
  --pipelines weakly_supervised \
  --only_embedding "${embed_id}" \
  --year_start 2017 --year_end 2024
rc=\$?
if [[ \$rc -ne 0 ]]; then
  echo "[error] export merged inference failed rc=\$rc"
  [[ "${keep_open}" == "1" ]] && exec bash
  exit \$rc
fi

echo "[done] ${embed_id}"
[[ "${keep_open}" == "1" ]] && exec bash
EOF
    chmod +x "$run_file"

    # Optional GPU assignment per session
    local device="${DEVICE:-cuda}"
    local gpu_id="${GPU_ID:-0}"
    local keep_open="${TMUX_KEEP_OPEN:-1}"

    local sess="weakly_${embed_id}"
    ensure_session "$sess" "$run_file"
  done
}

aggregate_results() {
  source change_detect/bin/activate
  python -u -m self_supervised_change_detection.aggregate_recalls \
    --results_dir "$RESULTS_DIR" --modes weakly_supervised
}

case "${1:-}" in
  start)
    shift
    start_sessions "$@"
    ;;
  aggregate)
    shift
    aggregate_results
    ;;
  *)
    cat <<EOF
Usage:
  bash weakly_supervised/scripts/run_weakly_supervised_monthly.sh start [EMBEDDINGS...]
  bash weakly_supervised/scripts/run_weakly_supervised_monthly.sh aggregate

Examples:
  bash weakly_supervised/scripts/run_weakly_supervised_monthly.sh start
  bash weakly_supervised/scripts/run_weakly_supervised_monthly.sh start DINOV3 SATMAE

Embeddings (labels): ${DEFAULT_LABELS[*]}

Outputs:
  weakly_supervised/model_runs/<embedding>/model.pt, scaler_stats.npz, unsup_month_scores_all_weakly_supervised.csv
  weakly_supervised/results/<embedding>/<embedding>_weakly_supervised_{test|all}_metrics{,_positive,_negative}.csv
  weakly_supervised/results/recall_{test|all}_weakly_supervised{,_positive,_negative}.csv
EOF
    exit 1
    ;;
esac
