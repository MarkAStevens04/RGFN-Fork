#!/bin/bash
#SBATCH --job-name=fr_chain
#SBATCH --time=3-00:00:00
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Run SEVERAL (generator, system, seed) cells sequentially inside ONE job.
#
# WHY CHAIN. One job per cell meant 30 jobs for this campaign, on an account three other agents
# share, and every cell queued separately -- so a cell that finished handed its GPU back and the
# next one waited again. A chain holds the GPU and works straight through. It also means a weekend
# needs far fewer things to go right unattended.
#
# A THIN WRAPPER ON PURPOSE: it calls submit_baseline.sh rather than duplicating any of its logic,
# so the OpenCL gate, the persistent docking server, the env setup and the resume/no-op behaviour
# live in exactly one file. (Its #SBATCH lines are comments to bash, so invoking it is safe.)
#
# SAFE TO RE-SUBMIT. submit_baseline.sh no-ops on a cell whose candidates.csv already exists, so a
# chain killed by walltime can simply be resubmitted: finished cells are skipped and the one that
# was interrupted restarts. A failing cell does not stop the rest; the exit status is non-zero if
# any failed.
#
# Usage -- CELLS is a space-separated list of GEN:SYSTEM:SEED triples:
#   CELLS="reinvent:clpp:42 saturn:clpp:42" \
#       sbatch experiments/fixed_reward/scale5k/submit_baseline_chain.sh

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
CELLS=${CELLS:?set CELLS to a list of GEN:SYSTEM:SEED triples}

# Snapshot the callee. bash reads a script incrementally and remembers a BYTE OFFSET, so editing
# submit_baseline.sh while this job sits inside a multi-hour cell would make it resume into garbage
# (that killed job 74318). SLURM's own snapshot protects THIS file, not the one it invokes.
CELL_SCRIPT=$(mktemp /tmp/fr_baseline.XXXXXX.sh)
cp experiments/fixed_reward/scale5k/submit_baseline.sh "$CELL_SCRIPT"
trap 'rm -f "$CELL_SCRIPT"' EXIT

echo "host=$(hostname)  CELLS=$CELLS"; nvidia-smi -L || true
FAILED=""
for CELL in $CELLS; do
    GEN=${CELL%%:*}; REST=${CELL#*:}; SYS=${REST%%:*}; SEED=${REST##*:}
    echo ""
    echo "############ CELL $GEN $SYS seed=$SEED  ($(date '+%F %H:%M')) ############"
    bash "$CELL_SCRIPT" "$GEN" "$SYS" "$SEED"
    rc=$?
    if [ "$rc" -eq 0 ]; then echo "CELL $CELL OK"; else echo "CELL $CELL FAILED rc=$rc" >&2; FAILED="$FAILED $CELL"; fi
done

echo ""
if [ -n "$FAILED" ]; then echo "FAILED CELLS:$FAILED" >&2; exit 1; fi
echo "ALL CELLS OK: $CELLS"
