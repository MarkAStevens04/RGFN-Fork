#!/bin/bash
#SBATCH --job-name=cmode_rt
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Route ONE catalogue-distinct pool with MultiAiZ, then read both selection arms off those routes.
# The pool already exists (submit_cmode_pools.sh); this is the expensive half.
#
# 12 h because MultiAiZ is ~2.25 h for a 500-molecule pool at ZINC and the tau=0.4 pool reaches
# further down the reward ranking, so its molecules are less catalogue-like and its searches run
# longer. Route discovery is resumable (the artifact is skipped if present), so an overrun costs the
# frontier only.
#
# GREEDY FIRST, SB OPTIONAL — same reasoning as submit_competitor_routes.sh: the greedy arm prices a
# FIXED mode set so its MILPs are small and reliably Optimal, while the SB arm's solve time is wildly
# non-monotonic in the budget. Running greedy first means a walltime kill loses the arm we cannot use
# rather than the one we can.
#
# Submit:  POOL_DIR=<multiaiz_pools/...cmode05_N500> TAU=0.5 sbatch .../submit_cmode_routes.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export XDG_CACHE_HOME=$SCRATCH/.cache MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

POOL_DIR=${POOL_DIR:?set POOL_DIR}
TAU=${TAU:?set TAU, and it must match the cutoff the pool was built at}
GATE=${GATE:-5.68}
RES=$SCRATCH/rgfn_runs/lsdflow_sparrow/results
TAG=$(basename "$POOL_DIR")

# The pool records the cutoff it was built at; selecting at a different tau would report a diversity
# the pool was never constructed for.
PTAU=$(python -c "import json;print(json.load(open('$POOL_DIR/pool_meta.json'))['cutoff'])")
[ "$PTAU" = "$TAU" ] || { echo "FATAL: pool built at cutoff $PTAU but TAU=$TAU" >&2; exit 1; }
echo "host=$(hostname) pool=$TAG tau=$TAU n=$(wc -l < "$POOL_DIR/pool.smi")"

echo "=== [1/2] MultiAiZ discovery ==="
POOL_DIR="$POOL_DIR" bash experiments/lsd_hubs/campaign/submit_multiaiz_discover.sh || exit 1
ROUTES="$POOL_DIR/multiaiz_routes.json"
[ -s "$ROUTES" ] || { echo "FATAL: no routes at $ROUTES" >&2; exit 1; }

echo "=== [2/2] selection arms ==="
source ~/bin/rgfn-smoke-env.sh      # sparrow_select_frontier imports glue -> dgl -> needs the helper
RC=0
python experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
    --routes "$ROUTES" --pool "$POOL_DIR/pool_scores.csv" --route-source multiaiz \
    --selection greedy --gate "$GATE" --higher-is-better true --cutoff "$TAU" \
    --mode-points "25,50,75,100,125,150" \
    --out-dir "$RES/${TAG}_greedy" --tag "${TAG}_greedy" || RC=1
python experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
    --routes "$ROUTES" --pool "$POOL_DIR/pool_scores.csv" --route-source multiaiz \
    --gate "$GATE" --higher-is-better true --cutoff "$TAU" --budgets 50,100,150 \
    --max-seconds 1800 \
    --out-dir "$RES/${TAG}_select" --tag "${TAG}_select" || RC=1
echo "CMODE-ROUTES rc=$RC"
exit "$RC"
