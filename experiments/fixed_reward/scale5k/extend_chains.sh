#!/bin/bash
# extend_chains.sh — deepen the DOCKING cells' afterany chains to fill spare submit-cap slots.
#
# Complements orchestrate.sh (which LAUNCHES idle cells with short mini-chains). This one takes the
# cells that are already active and, round-robin LONGEST-POLE-FIRST, appends afterany links to their
# chain tails until the live submit budget (CAP - HEADROOM - my current jobs) is spent. It targets
# ONLY the docking cells (6TD3/ClpP), because those are the ones that actually churn through many
# 3-day links (~4-17 days); the surrogate cells (sEH/DRD2) finish inside one link, so a deep chain
# on them would just be no-op links — launch those with orchestrate.sh instead. Links use NICE
# (default 0 = normal priority; set NICE=10000000 to deprioritize) and resume-or-no-op via the submit scripts.
#
#   bash experiments/fixed_reward/scale5k/extend_chains.sh [--dry-run]
set -uo pipefail
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
D="$HOME/projects/RGFN_Fork/RGFN-Fork/experiments/fixed_reward/scale5k"
FR="$SCRATCH/rgfn_runs/experiments/fixed_reward"
CAP=${CAP:-60}; HEADROOM=${HEADROOM:-2}; NICE=${NICE:-0}   # 0 = normal priority (full fair share); set 10000000 to deprioritize

if [ -n "${SLURM_JOB_ID:-}" ]; then echo "FATAL: run from a LOGIN node."; exit 1; fi

# docking cells, LONGEST-pole first (so extra rounds deepen the slowest cells):
#   RGFN 6TD3 (~17d) > RGFN ClpP (~7d) > baseline 6TD3 (~4-5d) > baseline ClpP (~3d)
CELLS=(
  "rgfn 6td3 43" "rgfn 6td3 44"
  "rgfn clpp 43" "rgfn clpp 44"
  "scent 6td3 43" "scent 6td3 44" "fraggfn 6td3 43" "fraggfn 6td3 44" "fraggfn 6td3 42" "rxnflow 6td3 43" "rxnflow 6td3 44"
  "scent clpp 43" "scent clpp 44" "fraggfn clpp 43" "fraggfn clpp 44" "fraggfn clpp 42" "rxnflow clpp 43" "rxnflow clpp 44"
)

tail_of(){ # tag -> highest (latest) jobid whose name == tag, in R or PD; empty if none active
  squeue -u "$USER" -h -t R,PD -o "%i|%j" | awk -F'|' -v t="$1" '{gsub(/ /,"",$2)} $2==t{print $1}' | sort -n | tail -1
}

CUR=$(squeue -u "$USER" -h -o "%i" | wc -l)
BUDGET=$(( CAP - HEADROOM - CUR ))
echo "=== extend docking chains @ $(date '+%F %T') ==="
echo "submitted=$CUR/$CAP  ->  budget=$BUDGET links (HEADROOM=$HEADROOM, NICE=$NICE)"
[ "$DRY" = 1 ] && echo "(dry-run: nothing submitted)"

# seed each cell's current chain tail once, then update locally as we append (so links truly chain)
declare -A TAIL SKIP
for c in "${CELLS[@]}"; do
  read -r gen sys seed <<<"$c"; tag="c5_${gen}_${sys}_s${seed}"
  if [ -f "$FR/${gen}_${sys}_5k/seed${seed}/fixed_reward/candidates/candidates.csv" ]; then SKIP[$tag]=done; continue; fi
  t=$(tail_of "$tag"); [ -z "$t" ] && { SKIP[$tag]=inactive; continue; }
  TAIL[$tag]=$t
done

added=0
while [ "$BUDGET" -gt 0 ]; do
  progressed=0
  for c in "${CELLS[@]}"; do
    [ "$BUDGET" -le 0 ] && break
    read -r gen sys seed <<<"$c"; tag="c5_${gen}_${sys}_s${seed}"
    [ -n "${SKIP[$tag]:-}" ] && continue
    dep=${TAIL[$tag]}
    if [ "$DRY" = 1 ]; then
      echo "  +link $tag  (afterany:$dep)"; jid="DRY$RANDOM"
    else
      if [ "$gen" = rgfn ]; then jid=$(sbatch --parsable --nice="$NICE" -J "$tag" --dependency=afterany:"$dep" "$D/submit_rgfn.sh" "$sys" "$seed")
      else jid=$(sbatch --parsable --nice="$NICE" -J "$tag" --dependency=afterany:"$dep" "$D/submit_baseline.sh" "$gen" "$sys" "$seed"); fi
      [ -z "$jid" ] && { echo "  sbatch FAILED for $tag"; SKIP[$tag]=failed; continue; }
      echo "  +link $tag  $jid (afterany:$dep)"
    fi
    TAIL[$tag]=$jid; BUDGET=$((BUDGET-1)); added=$((added+1)); progressed=1
  done
  [ "$progressed" -eq 0 ] && break   # every cell skipped -> stop
done

set +u   # empty associative-array expansion trips `set -u` on this bash version
echo "=== added=$added link(s) ($([ "$DRY" = 1 ] && echo dry-run || echo submitted)); skipped ${#SKIP[@]} cell(s) ==="
for k in "${!SKIP[@]}"; do echo "    skip $k (${SKIP[$k]})"; done | sort
set -u
