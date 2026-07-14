#!/bin/bash
# launch_chain.sh — RUN ON A LOGIN NODE. Submit a dependency chain of N jobs for one run.
#
# Balam forbids `sbatch` from inside a compute job (a compute job cannot resubmit itself), so
# the auto-requeue is a chain pre-submitted from the login node: N links, each depending
# `afterany` on the previous (so a link runs whether the prior finished, timed out, or failed).
# Every link runs the SAME per-job submit script, which resumes from the last checkpoint and
# exits immediately if the run's completion marker already exists — so once the run finishes,
# the remaining links are ~instant no-ops. N just needs to be >= the number of 3-day walls the
# run needs (pick generously; unused links cost seconds).
#
#   bash experiments/fixed_reward/scale5k/launch_chain.sh <N> <submit_script> [script args...]
# e.g.
#   bash .../launch_chain.sh 8 experiments/fixed_reward/scale5k/submit_rgfn.sh 6td3 42
set -uo pipefail
N=${1:?usage: launch_chain.sh <num_links> <submit_script> [args...]}
shift
SCRIPT=${1:?usage: launch_chain.sh <num_links> <submit_script> [args...]}
shift
ARGS=("$@")

if [ "$(hostname)" != "${SLURM_SUBMIT_HOST:-$(hostname)}" ] || [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "WARNING: looks like you're inside a job. Run launch_chain.sh from a LOGIN node." >&2
fi

DEP=""
for i in $(seq 1 "$N"); do
  if [ -z "$DEP" ]; then
    JID=$(sbatch --parsable "$SCRIPT" "${ARGS[@]}")
  else
    JID=$(sbatch --parsable --dependency=afterany:"$DEP" "$SCRIPT" "${ARGS[@]}")
  fi
  if [ -z "$JID" ]; then echo "FATAL: sbatch failed at link $i"; exit 1; fi
  echo "  link $i/$N -> job $JID (dep=${DEP:-none})  [$SCRIPT ${ARGS[*]}]"
  DEP=$JID
done
echo "chain of $N links submitted for: $SCRIPT ${ARGS[*]}"
