#!/bin/bash
#SBATCH --job-name=bc_sb_scale
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# BC-SB (Best-Candidate SPARROW-Batching) — the naive-chemist competitor, scaled up.
#
# THE ARM. Take the N highest-reward DISTINCT candidates the reaction-GFN sampled, hand SPARROW
# their by-construction routes, and let it batch them down under a reaction budget. No enumeration,
# no flow, no diversity filter -- what someone would do if they simply ran SPARROW on their best
# molecules. It is the arm LSD-Flow has to beat to show that enumerating off hubs contributes
# something sampling alone does not.
#
# WHY THIS IS A JOB AND NOT A LOGIN-NODE RUN. The MILP is superlinear in pool size: ~1 s per budget
# point at N=500, >110 s at N=2000. The login node kills anything over 3600 s of CPU, so the scale
# probe has to run on compute. Each N is attempted in increasing order and every outcome is
# recorded, so a pool that fails to solve is a DATAPOINT ("SPARROW could not batch 10k targets in
# the time limit"), not a lost run.
#
# TWO CORRECTNESS REQUIREMENTS baked into the driver, both learned the hard way:
#   * pool dedup -- a records.csv row is a sampling EVENT, not a candidate; the GFlowNet re-samples
#     the same molecule many times (top-500 rows = 115 distinct molecules). Without dedup the arm
#     is handed a fraction of the pool it is supposed to get.
#   * recipe expansion -- native routes are shallow, so promoted fragments must be expanded into
#     their builds or SPARROW BUYS what count-once BUILDS (the Logs/049 DRD2 62.7% failure).
#
# SMOKE ON debug FIRST -- ALWAYS. Every knob is an env override precisely so a 2-minute compute-node
# smoke can exercise the whole path before a 2 h job queues behind Balam's other tenants. This is not
# optional hygiene: the first submission of this script "COMPLETED" in 12 s on all three seeds because
# $HOME is read-only on compute, and a login-node test cannot catch that class of bug by construction.
#
#   SIZES=100 BUDGETS=25,50 MILP_CAP=120 TAG=bc_sb_smoke \
#     sbatch -p debug --time=00:20:00 experiments/lsd_hubs/campaign/submit_bc_sb_scale.sh
#
# Usage:  RUN=<native run dir> SNAP=<fragments_N.json> TAG=<tag> \
#           sbatch -p compute --time=02:00:00 experiments/lsd_hubs/campaign/submit_bc_sb_scale.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

RUN=${RUN:-/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_native_70974}
SNAP=${SNAP:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json}
TAG=${TAG:-bc_sb_seh_seed42}
GATE=${GATE:-7.0}
CUTOFF=${CUTOFF:-0.5}
SIZES=${SIZES:-"500 1000 2000 5000 10000 21000"}
BUDGETS=${BUDGETS:-"50,100,125,150,200,300"}
MILP_CAP=${MILP_CAP:-900}          # per-MILP ceiling; a hit shows up as a non-Optimal status
# $HOME IS READ-ONLY ON BALAM COMPUTE NODES. Writing results into the repo works on the login node
# and dies with EACCES on compute (jobs 72509-72511 "COMPLETED" in 12 s having done nothing). Results
# go to $SCRATCH; sync back to the repo from a login node afterwards.
OUT_ROOT=${OUT_ROOT:-/scratch/markymoo/rgfn_runs/lsdflow_sparrow/bc_sb}
# ROUTE_SOURCE selects the ARM, because both arms are the same sweep over a different pool:
#   native = BC-SB        -- the N best sampled candidates, routes looked up in routes.json
#   enum   = BC-Enum-SB   -- the enumerated CHILDREN of a hub set. Children are one reaction past a
#            hub and are absent from routes.json (29/500 present), so their routes are ASSEMBLED as
#            hub-prefix + the child's own reaction; ENUM_JSON and HUB_ROUTES are then required and
#            --pool is derived rather than supplied.
ROUTE_SOURCE=${ROUTE_SOURCE:-native}
ENUM_JSON=${ENUM_JSON:-}
HUB_ROUTES=${HUB_ROUTES:-}
if [ "$ROUTE_SOURCE" = "enum" ]; then
    [ -n "$ENUM_JSON" ] && [ -n "$HUB_ROUTES" ] || { echo "FATAL: enum needs ENUM_JSON + HUB_ROUTES" >&2; exit 1; }
    SRC_ARGS=(--routes "$ENUM_JSON" --route-source enum --hub-routes "$HUB_ROUTES")
else
    SRC_ARGS=(--routes "$RUN/sample/routes.json" --route-source native --pool "$RUN/sample/records.csv")
fi

export PYTHONUNBUFFERED=1
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/rgfn/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"

FIRST_N=$(echo $SIZES | awk '{print $1}')
echo "host=$(hostname)  RUN=$RUN  TAG=$TAG  sizes=$SIZES  out=$OUT_ROOT"
for N in $SIZES; do
    OUT="$OUT_ROOT/${TAG}_N${N}"
    if [ -s "$OUT/select_frontier.csv" ]; then
        echo "=== N=$N already done -> SKIP ==="; continue
    fi
    echo "=== N=$N ==="
    START=$(date +%s)
    python experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
        "${SRC_ARGS[@]}" --snapshot "$SNAP" --top-n "$N" \
        --gate "$GATE" --cutoff "$CUTOFF" \
        --out-dir "$OUT" --tag "${TAG}_N${N}" \
        --budgets "$BUDGETS" --max-seconds "$MILP_CAP"
    RC=$?
    echo "  N=$N rc=$RC wall=$(( $(date +%s) - START ))s"
    # Stop climbing once a size fails -- but do NOT dress an infrastructure error up as a science
    # result. A failure at the SMALLEST size cannot be a solver ceiling (N=500 solves in ~30 s on a
    # login node), so it is a bug in the run, and saying "practical ceiling" there would have us
    # reporting a limitation that does not exist.
    if [ "$RC" -ne 0 ]; then
        if [ "$N" = "$FIRST_N" ]; then
            echo "  N=$N FAILED at the smallest size -- this is a RUN ERROR, not a solver ceiling."
            echo "  Check the .err; do not report a scaling limit from this."
            exit 1
        fi
        echo "  N=$N did not complete -- larger pools can only be worse. Practical ceiling is the"
        echo "  largest size above that finished with an Optimal status."
        break
    fi
done
echo "=== DONE $TAG ==="
