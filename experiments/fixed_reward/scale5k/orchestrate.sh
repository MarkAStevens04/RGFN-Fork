#!/bin/bash
# orchestrate.sh — cap-aware driver for the EXTRA seeds (43,44) of the 16-cell matrix (Logs/030).
#
# RUN ON A LOGIN NODE. Balam forbids sbatch from a compute job and the `normal` QOS caps
# 60 SUBMITTED / 30 RUNNING jobs per user (shared across ALL your jobs), so the 32 extra cells
# (2 seeds x 16) cannot be pre-chained all at once like a single-seed matrix. This script is
# IDEMPOTENT and cap-aware: each invocation
#   1. classifies every target cell as DONE (candidates.csv exists), ACTIVE (a job with its tag
#      is running/pending), or IDLE (neither);
#   2. launches IDLE cells as short afterany mini-chains (MINI links each) — DOCKING cells first
#      (they are the multi-week long pole), surrogate cells fill the rest — up to the *live*
#      submit budget (cap - headroom - my current submitted jobs);
#   3. no-ops cells that are already active or complete.
# Re-run it periodically (on check-ins, or from a cron): a cell whose mini-chain has run out
# (both links TIMEOUT'd without finishing) falls back to IDLE and gets topped up on the next run,
# resuming from its last checkpoint. Cells that didn't fit under the cap launch as slots free.
#
# Each cell is tagged in its SLURM job name as  c5_<gen>_<system>_s<seed>  (e.g. c5_rgfn_clpp_s43)
# so `squeue -o %j` is the single source of truth for "is this cell active?" — no state file.
# seed-42 jobs use different names (fr5k_*), so they are counted in the budget but never managed
# here (they keep their own pre-submitted chains).
#
#   bash experiments/fixed_reward/scale5k/orchestrate.sh [--dry-run]
set -uo pipefail
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
D="$HOME/projects/RGFN_Fork/RGFN-Fork/experiments/fixed_reward/scale5k"
FR_ROOT="$SCRATCH/rgfn_runs/experiments/fixed_reward"
SEEDS=(43 44)
GENS=(rgfn fraggfn rxnflow scent)
CAP=${CAP:-60}; HEADROOM=${HEADROOM:-2}; MINI=${MINI:-2}
# Campaign submits at NORMAL priority (NICE=0) so it competes for the user's FULL fair share. A high
# --nice was tried (2026-07-24) then reverted (2026-07-27): it deprioritized the campaign below ALL
# other users (not just the user's own jobs), so it under-used the user's share whenever they had
# nothing queued. To deprioritize on demand instead, run with NICE=10000000 (PreemptMode=OFF here, so
# nice only reorders the PENDING queue and never preempts a running job).
NICE=${NICE:-0}

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "FATAL: run orchestrate.sh from a LOGIN node, not inside a job (Balam forbids sbatch there)."; exit 1
fi

CUR=$(squeue -u "$USER" -h -o "%i" 2>/dev/null | wc -l)
BUDGET=$(( CAP - HEADROOM - CUR ))
ACTIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null)   # full (untruncated) job names = our tags

launch_cell(){ # tag script args...
  local tag=$1; shift; local script=$1; shift
  local dep="" jid k
  for k in $(seq 1 "$MINI"); do
    if [ -z "$dep" ]; then jid=$(sbatch --parsable --nice="$NICE" -J "$tag" "$script" "$@")
    else jid=$(sbatch --parsable --nice="$NICE" -J "$tag" --dependency=afterany:"$dep" "$script" "$@"); fi
    if [ -z "$jid" ]; then echo "    ERROR: sbatch failed for $tag"; return 1; fi
    dep=$jid
  done
  echo "    launched $tag  ($MINI links, tail=$dep)  [$script $*]"
}

# (seed:gen) pairs to manage: seeds 43,44 x all 4 generators (32 cells), PLUS seed 42 x FragGFN
# only (4 cells) — the seed-42 FragGFN cells finished at the WRONG max_nodes=9 (Logs/046: oversized
# MW -> reward collapse), were archived to seed42_maxnodes9/, and are regenerated here at the
# paper-faithful max_nodes=6 so all three seeds' FragGFN row is comparable. RGFN/RxnFlow/SCENT keep
# their existing seed-42 runs (unaffected by the FragGFN cap).
declare -a PAIRS=()
for seed in "${SEEDS[@]}"; do for gen in "${GENS[@]}"; do PAIRS+=("$seed:$gen"); done; done
PAIRS+=("42:fraggfn")
# seed-42 RGFN 6td3+clpp stalled 2026-07-25 (their fr5k chains were cancelled to free weekend slots +
# nothing requeued them). Manage them here so they can't fall through again; the done seed-42 RGFN
# cells (seh, drd2) are auto-skipped via the candidates.csv check.
PAIRS+=("42:rgfn")

declare -a DOCK=() SURR=()
DONE=0; ACT=0
for pair in "${PAIRS[@]}"; do
  seed=${pair%%:*}; gen=${pair##*:}
  for sys in 6td3 clpp seh drd2; do
    tag="c5_${gen}_${sys}_s${seed}"
    complete="$FR_ROOT/${gen}_${sys}_5k/seed${seed}/fixed_reward/candidates/candidates.csv"
    if [ -f "$complete" ]; then DONE=$((DONE+1)); continue; fi
    if grep -qxF "$tag" <<<"$ACTIVE"; then ACT=$((ACT+1)); continue; fi
    if [ "$gen" = rgfn ]; then item="$tag|$D/submit_rgfn.sh|$sys $seed"
    else item="$tag|$D/submit_baseline.sh|$gen $sys $seed"; fi
    if [ "$sys" = 6td3 ] || [ "$sys" = clpp ]; then DOCK+=("$item"); else SURR+=("$item"); fi
  done
done

IDLE=$(( ${#DOCK[@]} + ${#SURR[@]} ))
TOTAL=$(( ${#PAIRS[@]} * 4 ))
echo "=== orchestrate seeds ${SEEDS[*]} (all gens) + seed42 FragGFN-at-6 @ $(date '+%F %T') ==="
echo "cells: done=$DONE active=$ACT idle=$IDLE / $TOTAL   |   my submitted=$CUR, budget=$BUDGET links (MINI=$MINI)"
[ "$DRY" = 1 ] && echo "(dry-run: nothing submitted)"

launched=0
for item in "${DOCK[@]:-}" "${SURR[@]:-}"; do
  [ -z "$item" ] && continue
  IFS='|' read -r tag script argstr <<<"$item"
  read -ra args <<<"$argstr"
  if [ "$BUDGET" -lt "$MINI" ]; then
    echo "  budget exhausted -> $tag (and the remaining idle cells) wait for the next run"; break
  fi
  if [ "$DRY" = 1 ]; then echo "    WOULD launch $tag ($MINI links)  [$script ${args[*]}]"
  else launch_cell "$tag" "$script" "${args[@]}" || continue; fi
  BUDGET=$(( BUDGET - MINI )); launched=$((launched+1))
done

echo "=== launched $launched idle cell(s) this run; $DONE/$TOTAL cells complete ==="
[ "$DONE" -eq "$TOTAL" ] && echo "CAMPAIGN_COMPLETE (seeds ${SEEDS[*]}) — stop the orchestrator loop"
exit 0
