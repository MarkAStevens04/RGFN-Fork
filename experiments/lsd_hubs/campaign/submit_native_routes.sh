#!/bin/bash
#SBATCH --job-name=native_routes
#SBATCH --partition=compute
#SBATCH --time=02:00:00                    # no MultiAiZ here: pool + two frontier solves, minutes
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# STAGE 3 FOR A GENERATOR THAT ALREADY CARRIES ITS OWN ROUTES.
#
# submit_competitor_routes.sh is for ROUTE-LESS entrants: it spends 6-12 h planning routes with
# MultiAiZ because those generators emit SMILES with no recipe. SynFormer does not need that step --
# it builds molecules from catalogue blocks by construction and writes a routes.jsonl whose every
# leaf is already purchasable. sparrow_select_frontier.py has had `--route-source external` for
# exactly this since the SynFormer work landed; what was missing was a launcher, so the header of
# submit_competitor_routes.sh called the path "Phase 2 and deliberately absent". This is that path.
#
# WHAT DIFFERS FROM THE ROUTE-LESS PIPELINE, AND WHY IT IS STILL LIKE-FOR-LIKE:
#   * No discovery step. The routes come free with the molecule, so there is no MultiAiZ cost and no
#     `discovery_timing.json`. When compute is compared across generators this ZERO is the honest
#     number, not a missing value -- planning cost SynFormer nothing because it never plans.
#   * No --snapshot. `native` (RGFN/SCENT) routes are shallow and need recipe expansion; `external`
#     routes are already deep, every leaf a catalogue block. Passing --snapshot here would be wrong.
#   * The pool is built from the BUDGET-FAITHFUL candidates. SynFormer is exempt from Stage 2 -- a GA
#     population cannot be upsampled (`pool = have[:n_samples]` is a slice of accumulated elites, so
#     "more molecules" means more GA generations, which is more TRAINING). Its cells are therefore
#     pool-limited BY CONSTRUCTION and must be reported that way, never as a cost result.
#
# Everything else is deliberately identical to the route-less path -- same gate from targets.py, same
# tau, same two selectors, same output layout -- because a cross-generator table must compare
# generators, not procedures.
#
# Usage -- CELLS is a space-separated list of GENERATOR:TARGET:SEED triples:
#   CELLS="synformer:clpp:42 synformer:clpp:43" \
#       sbatch experiments/lsd_hubs/campaign/submit_native_routes.sh
#
#   POOL=naive|pruned   which pool construction (default naive)
#   RUN_SB=0            greedy only; SB can be added later at no re-planning cost

set -uo pipefail
REPO_DIR=${REPO_DIR:-$HOME/projects/RGFN_Fork/RGFN-Fork}
cd "$REPO_DIR"

CELLS=${CELLS:?set CELLS to a list of GENERATOR:TARGET:SEED triples}
N=${N:-500}
CUTOFF=${CUTOFF:-0.5}
TARGET_MODES=${TARGET_MODES:-100}
RXN_BUDGET=${RXN_BUDGET:-100}
MIN_MODES=${MIN_MODES:-10}
BUDGETS=${BUDGETS:-50,100,150,200,300,400,500,600,800,1000}
MODE_POINTS=${MODE_POINTS:-5,10,15,20,25,50,75,100,125,150}   # finer than the route-less default:
# a pool-limited cell delivering <25 modes produces NO row on the 25,50,... ladder and vanishes from
# the table entirely. These cells are pool-limited by construction, so the low rungs are the point.
POOL=${POOL:-naive}
FR_ROOT=${FR_ROOT:-$SCRATCH/rgfn_runs/experiments/fixed_reward}
POOL_ROOT=$SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools
RES_ROOT=$SCRATCH/rgfn_runs/lsdflow_sparrow/results

case "$POOL" in
    naive)  POOL_FLAG="" ;;
    pruned) POOL_FLAG="--pruned --cutoff $CUTOFF" ;;
    *) echo "FATAL: POOL must be naive or pruned, got '$POOL'" >&2; exit 1 ;;
esac

export PYTHONUNBUFFERED=1
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
echo "host=$(hostname)  CELLS=$CELLS  POOL=$POOL  N=$N"

FAILED=""
for CELL in $CELLS; do
    GEN=${CELL%%:*}; REST=${CELL#*:}; TGT=${REST%%:*}; SD=${REST##*:}
    RUN_DIR="$FR_ROOT/${GEN}_${TGT}/seed${SD}"
    CANDS="$RUN_DIR/fixed_reward/candidates/candidates.csv"
    ROUTES="$RUN_DIR/fixed_reward/candidates/routes.jsonl"
    TAG=${GEN}_${TGT}_seed${SD}$([ "$POOL" = pruned ] && echo "_pruned" || echo "")
    echo ""; echo "############ CELL $CELL ($(date '+%F %H:%M')) ############"

    [ -s "$CANDS" ]  || { echo "FAILED $CELL — no candidates at $CANDS" >&2; FAILED="$FAILED $CELL"; continue; }
    # The routes artifact is what makes this path valid at all. Without it the cell belongs in the
    # route-less pipeline, and silently falling back would price an EMPTY library as Optimal.
    [ -s "$ROUTES" ] || { echo "FAILED $CELL — no routes.jsonl at $ROUTES; this generator is NOT route-carrying, use submit_competitor_routes.sh" >&2; FAILED="$FAILED $CELL"; continue; }

    read -r GATE DIR <<EOF
$(conda run --no-capture-output -n rgfn python - "$TGT" <<'PYGATE'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "experiments" / "lsd_hubs" / "matrix16"))
from targets import get_target
t = get_target(sys.argv[1])
print(t.mode_reward_threshold, "higher" if t.higher_is_better else "lower")
PYGATE
)
EOF
    [ -n "${GATE:-}" ] && [ -n "${DIR:-}" ] || { echo "FAILED $CELL — no gate for '$TGT'" >&2; FAILED="$FAILED $CELL"; continue; }
    if [ "$DIR" = lower ]; then DIR_FLAGS="--lower-is-better"; HIB=false; else DIR_FLAGS=""; HIB=true; fi
    echo "  gate: $GATE ($DIR is better) — from matrix16/targets.py"
    echo "  routes: $ROUTES (native, no MultiAiZ — planning cost is ZERO for this generator)"

    echo "=== [0/2] mode-saturation pre-flight ==="
    conda run --no-capture-output -n rgfn python experiments/lsd_hubs/campaign/mode_saturation.py \
        --candidates "$CANDS" --gate "$GATE" --cutoff "$CUTOFF" --target-modes "$TARGET_MODES" \
        --rxn-budget "$RXN_BUDGET" --min-modes "$MIN_MODES" --pool "$POOL" $DIR_FLAGS \
        --sizes "50,100,250,${N}" --tag "$TAG" --out-dir "$RES_ROOT/${TAG}_saturation" || {
            echo "GATED $CELL — fewer than $MIN_MODES modes; a POOL-SIZE result, reported not retried" >&2
            FAILED="$FAILED ${CELL}(gated)"; continue; }

    echo "=== [1/2] pool ==="
    conda run --no-capture-output -n rgfn python experiments/lsd_hubs/campaign/build_s3gfn_pools.py \
        --candidates "$CANDS" --out-root "$POOL_ROOT" --tag "$TAG" --sizes "$N" \
        --gate "$GATE" $DIR_FLAGS $POOL_FLAG || { echo "FAILED $CELL — pool build" >&2; FAILED="$FAILED $CELL"; continue; }
    POOL_DIR="$POOL_ROOT/${TAG}_N${N}"
    if [ ! -s "$POOL_DIR/pool.smi" ]; then
        ACTUAL=$(ls -d "$POOL_ROOT/${TAG}"_N[0-9]* 2>/dev/null | head -1)
        [ -n "$ACTUAL" ] && [ -s "$ACTUAL/pool.smi" ] && { echo "  pool-limited: requested N=$N, using $(basename "$ACTUAL")"; POOL_DIR="$ACTUAL"; }
    fi
    [ -s "$POOL_DIR/pool.smi" ] || { echo "FAILED $CELL — pool not built" >&2; FAILED="$FAILED $CELL"; continue; }
    NACT=$(wc -l < "$POOL_DIR/pool.smi")
    echo "  pool: $POOL_DIR ($NACT molecules)"

    echo "=== [2/2] frontiers (greedy first, then SB) ==="
    RC=0
    conda run --no-capture-output -n rgfn python \
        experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
        --routes "$ROUTES" --pool "$POOL_DIR/pool_scores.csv" --route-source external \
        --selection greedy --gate "$GATE" --higher-is-better "$HIB" \
        --cutoff "$CUTOFF" --mode-points "$MODE_POINTS" \
        --out-dir "$RES_ROOT/${TAG}_greedy_N${NACT}" \
        --tag "${TAG}_external_greedy" || RC=1

    if [ "${RUN_SB:-1}" = "1" ]; then
        conda run --no-capture-output -n rgfn python \
            experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
            --routes "$ROUTES" --pool "$POOL_DIR/pool_scores.csv" --route-source external \
            --gate "$GATE" --higher-is-better "$HIB" --cutoff "$CUTOFF" --budgets "$BUDGETS" \
            --max-seconds "${SB_MAX_SECONDS:-1800}" \
            --out-dir "$RES_ROOT/${TAG}_select_N${NACT}" \
            --tag "${TAG}_external_select_N${NACT}" || RC=1
    fi

    if [ "$RC" -eq 0 ]; then echo "CELL $CELL OK"; else echo "CELL $CELL FAILED rc=$RC" >&2; FAILED="$FAILED $CELL"; fi
done

echo ""
if [ -n "$FAILED" ]; then echo "FAILED/GATED CELLS:$FAILED" >&2; exit 1; fi
echo "ALL CELLS OK: $CELLS"
