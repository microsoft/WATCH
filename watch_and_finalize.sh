#!/usr/bin/env bash
# watch_and_finalize.sh
# Watches the three pipeline tmux session groups and automatically runs
# evaluate + aggregate once each group's sessions exit.
#
# Runs itself inside a tmux session (auto_finalize) via:
#   bash watch_and_finalize.sh
#
# Log: watch_and_finalize.log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

SELF="$(realpath "$0")"
SESSION="auto_finalize"
LOG="$REPO_ROOT/watch_and_finalize.log"

# If not inside the target tmux session, re-launch ourselves inside one.
if [[ "${TMUX:-}" == "" ]] || ! tmux display-message -p '#S' 2>/dev/null | grep -qx "$SESSION"; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[info] Session '$SESSION' already exists. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
  fi
  echo "[info] Launching watcher inside tmux session '$SESSION'..."
  tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" \
    "bash -lc 'bash $SELF --inside-tmux 2>&1 | tee -a $LOG; exec bash'"
  echo "[info] Watcher running in tmux session '$SESSION'. Log: $LOG"
  exit 0
fi

# ---- Everything below runs inside the tmux session ----

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== watch_and_finalize started ==="
log "Repo: $REPO_ROOT"
log "Log:  $LOG"

# Activate venv once for aggregate calls at the end
source change_detect/bin/activate

# Poll interval in seconds
POLL=30

wait_sessions_gone() {
  local pattern="$1"
  local desc="$2"
  log "Waiting for '$desc' sessions (pattern: $pattern)..."
  while tmux ls 2>/dev/null | grep -qP "$pattern"; do
    local n
    n=$(tmux ls 2>/dev/null | grep -cP "$pattern" || true)
    log "  $n '$desc' session(s) still running..."
    sleep "$POLL"
  done
  log "  All '$desc' sessions finished."
}

###############################################################################
# Group 1: Distance baseline  →  evaluate  →  aggregate (shared at end)
###############################################################################
baseline_worker() {
  log "[baseline] Starting worker..."
  wait_sessions_gone "_distance_baseline" "distance_baseline"
  log "[baseline] Running evaluate..."
  bash self_supervised_change_detection/run_baseline_pipeline.sh evaluate \
    CLIP GEORSCLIP HANDCRAFTED PRITHIVI SATLASPRETRAIN SATMAE DINOV3
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "[baseline] ERROR: evaluate exited with rc=$rc"
  else
    log "[baseline] Evaluate complete."
  fi
  touch /tmp/_baseline_eval_done
}

###############################################################################
# Group 2: Learned unsupervised  →  merge  →  evaluate  →  aggregate (at end)
###############################################################################
unsup_worker() {
  log "[unsup] Starting worker..."
  wait_sessions_gone "_learned_unsupervised" "learned_unsupervised"
  log "[unsup] Running merge..."
  bash self_supervised_change_detection/run_unsupervised_pipeline.sh merge \
    CLIP GEORSCLIP HANDCRAFTED PRITHIVI SATLASPRETRAIN SATMAE DINOV3
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "[unsup] ERROR: merge exited with rc=$rc"
    touch /tmp/_unsup_eval_done
    return
  fi
  log "[unsup] Merge complete. Running evaluate..."
  bash self_supervised_change_detection/run_unsupervised_pipeline.sh evaluate \
    CLIP GEORSCLIP HANDCRAFTED PRITHIVI SATLASPRETRAIN SATMAE DINOV3
  rc=$?
  if [[ $rc -ne 0 ]]; then
    log "[unsup] ERROR: evaluate exited with rc=$rc"
  else
    log "[unsup] Evaluate complete."
  fi
  touch /tmp/_unsup_eval_done
}

###############################################################################
# Group 3: Weakly supervised  →  aggregate only (evaluate runs inside each session)
###############################################################################
weakly_worker() {
  log "[weakly] Starting worker..."
  wait_sessions_gone "^weakly_" "weakly_supervised"
  log "[weakly] Running aggregate..."
  bash weakly_supervised/scripts/run_weakly_supervised_monthly.sh aggregate
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "[weakly] ERROR: aggregate exited with rc=$rc"
  else
    log "[weakly] Aggregate complete."
  fi
  touch /tmp/_weakly_agg_done
}

###############################################################################
# Launch all three workers in parallel background subshells
###############################################################################
rm -f /tmp/_baseline_eval_done /tmp/_unsup_eval_done /tmp/_weakly_agg_done

baseline_worker &
unsup_worker    &
weakly_worker   &

wait  # wait for all three workers to finish

log "=== All three pipeline workers finished. Running final aggregates... ==="

log "[aggregate] Running baseline aggregate..."
bash self_supervised_change_detection/run_baseline_pipeline.sh aggregate
log "[aggregate] Running unsupervised aggregate..."
bash self_supervised_change_detection/run_unsupervised_pipeline.sh aggregate

log "=== ALL DONE ==="
log "Results:"
log "  self_supervised_change_detection/results/"
log "  weakly_supervised/results/"
