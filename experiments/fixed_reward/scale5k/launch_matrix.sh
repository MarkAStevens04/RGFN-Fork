#!/bin/bash
# launch_matrix.sh — fire the full 16-cell publication-scale matrix (campaign Logs/030).
#
# RUN ON A LOGIN NODE (launch_chain.sh submits the SLURM dependency chains; Balam forbids
# sbatch from compute). Run ONLY after the smokes pass. One seed (42) to start.
#
#   bash experiments/fixed_reward/scale5k/launch_matrix.sh [seed]
#
# Each cell is an auto-requeue CHAIN of N 3-day links (each resumes from the last checkpoint,
# no-ops once candidates.csv exists). At cap 4 / 5000 iters: docking cells (6td3, clpp) ~8-12
# days -> DOCK_LINKS 3-day links; surrogate cells (seh, drd2) ~3-5 days -> SURR_LINKS.
#
# JOB-CAP CONSTRAINT: the `normal` QOS caps 60 SUBMITTED jobs / 30 RUNNING per user. With 8
# docking cells + 8 surrogate cells, DOCK_LINKS=5 + SURR_LINKS=2 = 56 jobs (fits). Only the 16
# chain heads are dependency-free, so at most ~16 run at once (< 30). A pre-flight check below
# aborts (rather than half-submitting) if the total + already-queued jobs would exceed the cap.
set -uo pipefail
SEED=${1:-42}
D="$HOME/projects/RGFN_Fork/RGFN-Fork/experiments/fixed_reward/scale5k"
LC="$D/launch_chain.sh"
DOCK_LINKS=${DOCK_LINKS:-5}   # 3-day links per docking cell (~15 days; run ~8-12 days)
SURR_LINKS=${SURR_LINKS:-2}   # 3-day links per surrogate cell (~6 days; run ~3-5 days)
CAP=${CAP:-60}                # normal-QOS MaxSubmitJobsPerUser

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "FATAL: run launch_matrix.sh from a LOGIN node, not inside a job."; exit 1
fi

# Pre-flight: make sure the whole matrix fits under the submit cap (don't half-submit).
TOTAL=$(( 8 * DOCK_LINKS + 8 * SURR_LINKS ))
CURRENT=$(squeue -u "$USER" -h -o "%i" 2>/dev/null | wc -l)
echo "pre-flight: submitting $TOTAL jobs (8 docking x $DOCK_LINKS + 8 surrogate x $SURR_LINKS); "
echo "            already queued: $CURRENT; QOS cap: $CAP"
if [ $(( TOTAL + CURRENT )) -gt "$CAP" ]; then
  echo "FATAL: $TOTAL + $CURRENT = $(( TOTAL + CURRENT )) > cap $CAP. Lower DOCK_LINKS/SURR_LINKS"
  echo "       or wait for queued jobs to clear, then re-run."; exit 2
fi

echo "### RGFN (in-process docking; native resume) ###"
bash "$LC" "$DOCK_LINKS" "$D/submit_rgfn.sh" 6td3 "$SEED"
bash "$LC" "$DOCK_LINKS" "$D/submit_rgfn.sh" clpp "$SEED"
bash "$LC" "$SURR_LINKS" "$D/submit_rgfn.sh" seh  "$SEED"
bash "$LC" "$SURR_LINKS" "$D/submit_rgfn.sh" drd2 "$SEED"

for GEN in fraggfn rxnflow scent; do
  echo "### $GEN (docking via persistent server on the docking cells) ###"
  bash "$LC" "$DOCK_LINKS" "$D/submit_baseline.sh" "$GEN" 6td3 "$SEED"
  bash "$LC" "$DOCK_LINKS" "$D/submit_baseline.sh" "$GEN" clpp "$SEED"
  bash "$LC" "$SURR_LINKS" "$D/submit_baseline.sh" "$GEN" seh  "$SEED"
  bash "$LC" "$SURR_LINKS" "$D/submit_baseline.sh" "$GEN" drd2 "$SEED"
done

echo "### 16 cells submitted (seed $SEED, $TOTAL jobs). Monitor: squeue -u $USER ###"
