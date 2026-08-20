#!/bin/bash
#SBATCH --job-name=comp_routes_chain
#SBATCH --partition=compute
#SBATCH --time=12:00:00                    # ~3.7 h of MultiAiZ per cell + minutes of greedy
#SBATCH --gpus-per-node=1                  # AiZynth's expansion policy is a neural net
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Run submit_competitor_routes.sh for SEVERAL cells sequentially in one job.
#
# A THIN WRAPPER ON PURPOSE. It invokes the single-cell script with `bash` rather than duplicating any
# of its logic, so the protocol that must not drift between cells lives in exactly one file. (Its
# #SBATCH lines are comments to bash, so running it directly is safe.)
#
# WHY CHAIN AT ALL. Route discovery is ~3.7 h per cell and there are 11 cells left; submitting them
# individually would take 11 of the account's 60 job slots on a cluster three other agents share.
# Discovery is also idempotent — `submit_multiaiz_discover.sh` skips a pool whose artifact already
# exists — so a chain that dies partway costs only the cell it was working on, and re-submitting
# resumes rather than restarts.
#
# A FAILING CELL DOES NOT STOP THE REST. Each cell's outcome is printed and the exit status is
# non-zero if any failed, so a single unroutable pool cannot silently swallow the whole batch.
#
# Usage — CELLS is a space-separated list of GENERATOR:TARGET:SEED triples:
#   CELLS="reinvent:seh:43 reinvent:seh:44 reinvent:drd2:42" \
#       sbatch experiments/lsd_hubs/campaign/submit_competitor_routes_chain.sh
#
# POOL selects the pool construction — naive (default) or pruned. See docs/RESEARCH_CONTEXT.md,
# "The two pools and the two numbers". The two variants write to DIFFERENT tags, so their MultiAiZ
# caches never collide and both can be chained independently.
#
# Both selection arms run by default (RUN_SB=1 in the single-cell script), because SB is the arm the
# two-pool design reports both numbers from. Set RUN_SB=0 for greedy only. NOTE that MILP tractability
# is POOL-DEPENDENT, not budget-dependent: S3-GFN's network solved R=100 in 57 s `Optimal`, while
# REINVENT's sEH network had not returned R=50 after 10 min. Always check `time_capped` before reading
# an SB row as a converged optimum — a capped row is a LOWER bound on the competitor.

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

CELLS=${CELLS:?set CELLS to a list of GENERATOR:TARGET:SEED triples}
FR_ROOT=${FR_ROOT:-$SCRATCH/rgfn_runs/experiments/fixed_reward}

echo "host=$(hostname)"; nvidia-smi -L
echo "CELLS=$CELLS"

# SNAPSHOT the per-cell script before running it. bash reads a script incrementally and remembers a
# BYTE OFFSET, so editing the file while a job sits inside its 3.7 h MultiAiZ call moves everything
# past that offset -- and when the call returns, bash executes whatever bytes now live there. That
# killed job 74318 five hours in with `line 106: —: command not found`, after its route discovery had
# already succeeded. Running a private copy makes the live file safe to edit while cells are in
# flight. (SLURM's own submit-time snapshot protects THIS file, not the one it invokes.)
CELL_SCRIPT=$(mktemp /tmp/competitor_routes.XXXXXX.sh)
cp experiments/lsd_hubs/campaign/submit_competitor_routes.sh "$CELL_SCRIPT"
trap 'rm -f "$CELL_SCRIPT"' EXIT
echo "cell script snapshot: $CELL_SCRIPT"

FAILED=""
for CELL in $CELLS; do
    GEN=${CELL%%:*}; REST=${CELL#*:}; TGT=${REST%%:*}; SD=${REST##*:}
    RUN_DIR="$FR_ROOT/${GEN}_${TGT}/seed${SD}"
    echo ""
    echo "############ CELL $CELL ############"
    if [ ! -s "$RUN_DIR/fixed_reward/candidates/candidates.csv" ]; then
        echo "FAILED $CELL — no candidates at $RUN_DIR" >&2; FAILED="$FAILED $CELL"; continue
    fi
    GENERATOR="$GEN" TARGET="$TGT" SEED="$SD" RUN_DIR="$RUN_DIR" POOL="${POOL:-naive}" \
        bash "$CELL_SCRIPT"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "CELL $CELL OK"
    elif [ "$rc" -eq 2 ]; then
        # exit 2 is the saturation gate: a real pool-size finding, not an infrastructure failure.
        echo "CELL $CELL GATED (pool cannot reach the mode target) — reported, not retried" >&2
        FAILED="$FAILED ${CELL}(gated)"
    else
        echo "CELL $CELL FAILED rc=$rc" >&2; FAILED="$FAILED $CELL"
    fi
done

echo ""
if [ -n "$FAILED" ]; then echo "FAILED/GATED CELLS:$FAILED" >&2; exit 1; fi
echo "ALL CELLS OK: $CELLS"
