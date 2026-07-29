#!/bin/bash
# Sequential enumeration driver for the hub-ordering ablation (Logs/053).
#
# The `debug` QoS allows exactly ONE job per user — MaxJobsPU=1 AND MaxSubmitJobsPU=1 — so a
# --dependency chain cannot even be submitted: the second sbatch is rejected outright. This driver
# therefore does the queueing itself: submit one slice, wait for it to leave the queue, submit the
# next.
#
# Before each submission it re-runs plan_arms.py, which (a) drops hubs the finished slices already
# covered — including hubs shared with a later arm — and (b) refits its per-depth cost model on the
# timings just measured. That matters: the seed model is fitted on the incumbent's shallow, high-flow
# hubs, while the reverse/random arms are mostly depth-3 hubs we have barely measured. If a slice
# dies (almost always the 2h wall clock), the driver halves --slice-seconds and re-plans, so the
# retry is genuinely smaller rather than the same job again.
#
# Idempotent: re-running picks up wherever it stopped (finished slices are cache, not work).
#
#   nohup bash experiments/lsd_hubs/hub_order/chain.sh > /dev/null 2>&1 &
#   tail -f $SCRATCH/rgfn_runs/lsdflow/hub_order/chain.log
#
# Env: OUT_ROOT, SLICE_SECONDS (initial), MAX_STEPS, POLL (seconds).

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-$SCRATCH/rgfn_runs/lsdflow/hub_order}
SLICE_SECONDS=${SLICE_SECONDS:-3600}
MIN_SLICE_SECONDS=${MIN_SLICE_SECONDS:-600}
MAX_STEPS=${MAX_STEPS:-40}
POLL=${POLL:-60}
PY=${PY:-python3}          # the driver itself is pure stdlib; no conda env needed on the login node
LOG="$OUT_ROOT/chain.log"

mkdir -p "$OUT_ROOT/enum"
cd "$REPO" || exit 2

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Block until this user has no job in the debug partition (MaxSubmitJobsPU=1 rejects otherwise).
wait_for_debug_slot() {
  while :; do
    n=$(squeue -u "$USER" -h -p debug 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] && return 0
    sleep "$POLL"
  done
}

log "=== chain start (OUT_ROOT=$OUT_ROOT, SLICE_SECONDS=$SLICE_SECONDS, REPO=$REPO) ==="

step=0
while [ "$step" -lt "$MAX_STEPS" ]; do
  # Re-plan: recompute every arm's remaining work against the current cache + timings.
  if ! $PY experiments/lsd_hubs/hub_order/plan_arms.py \
        --out-root "$OUT_ROOT" --slice-seconds "$SLICE_SECONDS" >> "$LOG" 2>&1; then
    log "FATAL plan_arms.py failed; aborting"; exit 1
  fi

  NEXT=$($PY - "$OUT_ROOT" <<'PY'
import json, sys
m = json.load(open(f"{sys.argv[1]}/manifest.json"))
for arm in m["arms"]:
    if arm["slices"]:
        s = arm["slices"][0]
        print(arm["name"], s["file"], s["n_hubs"], s["est_seconds"])
        break
PY
)
  read -r ARM SLICE NHUBS EST <<< "$NEXT"
  if [ -z "${ARM:-}" ]; then
    log "=== all arms fully enumerated; nothing left to submit ==="
    break
  fi

  step=$((step + 1))
  OUT_DIR="$OUT_ROOT/enum/step$(printf '%02d' "$step")_${ARM}"
  if [ -f "$OUT_DIR/enum_children.json" ]; then   # paranoia: never overwrite a good slice
    log "step $step: $OUT_DIR already complete, skipping name"; continue
  fi
  log "step $step: $ARM slice ($NHUBS hubs, est ${EST}s) -> $OUT_DIR"

  wait_for_debug_slot
  JID=$(HUBS_FILE="$SLICE" OUT_DIR="$OUT_DIR" REPO="$REPO" \
        sbatch --parsable experiments/lsd_hubs/hub_order/submit_slice.sh 2>>"$LOG")
  if [ -z "$JID" ]; then
    log "FATAL sbatch rejected the submission; aborting"; exit 1
  fi
  log "step $step: job $JID submitted; waiting"

  while squeue -j "$JID" -h 2>/dev/null | grep -q .; do sleep "$POLL"; done

  if [ -f "$OUT_DIR/enum_children.json" ]; then
    log "step $step: job $JID OK ($ARM, $NHUBS hubs)"
  else
    log "step $step: job $JID FAILED (no enum_children.json) — likely the 2h wall clock"
    SLICE_SECONDS=$((SLICE_SECONDS / 2))
    if [ "$SLICE_SECONDS" -lt "$MIN_SLICE_SECONDS" ]; then
      log "FATAL slice budget fell below ${MIN_SLICE_SECONDS}s; a single hub may not fit. Aborting."
      exit 1
    fi
    log "retrying with SLICE_SECONDS=$SLICE_SECONDS"
  fi
done

log "=== enumeration phase done; merging arms ==="
$PY experiments/lsd_hubs/hub_order/merge_enum.py --out-root "$OUT_ROOT" >> "$LOG" 2>&1 \
  && log "merge OK" || log "merge FAILED (see $LOG)"
log "=== chain finished; next: experiments/lsd_hubs/hub_order/run_arms.sh ==="
