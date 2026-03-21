#!/usr/bin/env bash
# Watches the 3 parallel satlaspretrain sessions and re-runs merge/evaluate/aggregate
# once all months are complete.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
LOG="$REPO_ROOT/satlaspretrain_finalize.log"
exec > >(tee -a "$LOG") 2>&1

source change_detect/bin/activate

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== satlaspretrain_finalize started ==="
log "Watching for satlaspretrain parallel sessions to complete..."

while tmux ls 2>/dev/null | grep -qP "satlaspretrain.*learned_unsupervised"; do
  n=$(tmux ls 2>/dev/null | grep -cP "satlaspretrain.*learned_unsupervised" || true)
  log "  $n session(s) still running..."
  sleep 30
done

log "All parallel sessions done. Checking monthly file count..."
n_done=$(ls self_supervised_change_detection/model_runs/satlaspretrain/monthly_self_supervised_change_detection_*_all.csv 2>/dev/null | wc -l)
log "  $n_done monthly files present (expected 96)"

log "Running merge..."
bash self_supervised_change_detection/run_unsupervised_pipeline.sh merge SATLASPRETRAIN

log "Running evaluate..."
bash self_supervised_change_detection/run_unsupervised_pipeline.sh evaluate SATLASPRETRAIN

log "Running aggregate..."
python -u -m self_supervised_change_detection.aggregate_recalls

log "=== satlaspretrain finalize complete ==="
log "Results written to self_supervised_change_detection/results/"
