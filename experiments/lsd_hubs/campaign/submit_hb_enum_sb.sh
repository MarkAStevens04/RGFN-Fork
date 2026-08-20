#!/bin/bash
#SBATCH --job-name=hb_enum_sb
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# HB-Enum-SB — SPARROW-Batching run on HUB-BATCHING's OWN enumerated candidates.
#
# THE QUESTION. BC-Enum-SB hands SPARROW the children of hubs picked from the top-K best CANDIDATES.
# This arm hands it the children of the hub set OUR method actually enumerates. Same optimizer, same
# budget, same gate -- the only thing that changes is WHERE the candidates came from. It answers:
#   "does SPARROW do better on candidates drawn from high-flow hubs than on candidates from
#    naively (reward-)picked hubs?"
# Expectation is a modest gain, not a dramatic one: reward-picked hubs are themselves already fairly
# high-flow, so the two candidate sets are not disjoint in character (Logs/053 -- flow picks the
# neighbourhood, not the rank).
#
# WHY IT IS ALSO THE FAIR "SAME MOLECULES, DIFFERENT CHOOSER" TEST. Earlier write-ups described
# BC-Enum-SB that way. That was WRONG: its hubs are reward-picked (Logs/059 -- "64 reward-picked
# hubs") and overlap our own seed-42 hub set by only 35/64, so it sees a genuinely different
# candidate set. THIS arm is the one where the candidate set really is ours.
#
# NO NEW ENUMERATION. It reuses the matrix16 `scent_seh` enumerations -- 200 hubs, ~440-490k
# children each, one per seed -- which are the same artifacts our own hub-batching arm is scored
# from. Only the MILP is new.
#
# DO NOT POINT THIS AT `campaign_enum_seh_70363`. That artifact predates the per-child `reaction`
# fix (2026-07-15) and stores NONE, so every child's route stops at its hub and SPARROW returns the
# empty library as trivially Optimal -- measured: 2,000 targets -> 239 compound nodes, 0 selected,
# status `Optimal`. `load_enum_pool` now aborts on it, but the right artifact is the matrix16 one.
#
# ONE CONFOUND TO STATE, NOT HIDE. These enumerations used n_hubs=200 / top_k=1000 while BC-Enum-SB
# used 64 / 100, so the arms differ in the WIDTH of the net as well as in how hubs were ranked. A
# gain here is therefore "flow-picked hubs from a wider net", not "flow alone". Narrowing that needs
# a 64-hub flow enumeration, which does not exist yet.
#
# SMOKE FIRST -- always:
#   SIZES=2000 BUDGETS=100 MILP_CAP=300 TAG=hbenum_smoke \
#     sbatch -p debug --time=00:25:00 experiments/lsd_hubs/campaign/submit_hb_enum_sb.sh
#
# Usage:  sbatch -p compute --time=16:00:00 experiments/lsd_hubs/campaign/submit_hb_enum_sb.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

# SEED picks the matrix16 cell; every one carries its own paired sample/routes.json.
SEED=${SEED:-42}
case "$SEED" in
    42) CELL=matrix16/scent_seh
        SNAP_DEF=/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json ;;
    *)  CELL=matrix16_seed${SEED}/scent_seh
        SNAP_DEF=/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh_5k/seed${SEED}/additional_fragments/fragments_4000.json ;;
esac
BASE=${BASE:-/scratch/markymoo/rgfn_runs/lsdflow/$CELL}
ENUM=${ENUM:-$BASE/enum/enum_children.json}
HUB_ROUTES=${HUB_ROUTES:-$BASE/sample/routes.json}
SNAP=${SNAP:-$SNAP_DEF}
TAG=${TAG:-hbenumR100_seed${SEED}_L1}
GATE=${GATE:-7.0}
CUTOFF=${CUTOFF:-0.5}
# 50000 matches BC-Enum-SB's `enumR100_*_N50000` exactly, so pool size is NOT a free variable
# between the two arms.
SIZES=${SIZES:-50000}
BUDGETS=${BUDGETS:-"50,100,150"}
LAMBDA_DIV=${LAMBDA_DIV:-1.0}
MILP_CAP=${MILP_CAP:-7200}
OUT_ROOT=${OUT_ROOT:-/scratch/markymoo/rgfn_runs/lsdflow_sparrow/hb_enum_sb}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$SCRATCH/.cache/matplotlib"
export TRITON_CACHE_DIR="$SCRATCH/.cache/triton"
export HF_HOME="$SCRATCH/.cache/hf"
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$HF_HOME"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/rgfn/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"

for P in "$ENUM" "$HUB_ROUTES" "$SNAP"; do
    [ -s "$P" ] || { echo "FATAL: missing input $P" >&2; exit 1; }
done
echo "host=$(hostname)  SEED=$SEED  TAG=$TAG  sizes=$SIZES  lambda=$LAMBDA_DIV"
echo "  ENUM=$ENUM"; echo "  HUB_ROUTES=$HUB_ROUTES"
FIRST_N=$(echo $SIZES | awk '{print $1}')
for N in $SIZES; do
    OUT="$OUT_ROOT/${TAG}_N${N}"
    if [ -s "$OUT/select_frontier.csv" ]; then
        echo "=== N=$N already done -> SKIP ==="; continue
    fi
    echo "=== N=$N ==="
    START=$(date +%s)
    python experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
        --routes "$ENUM" --route-source enum --hub-routes "$HUB_ROUTES" \
        --snapshot "$SNAP" --top-n "$N" \
        --gate "$GATE" --cutoff "$CUTOFF" \
        --out-dir "$OUT" --tag "${TAG}_N${N}" \
        --budgets "$BUDGETS" --max-seconds "$MILP_CAP" \
        --lambda-div "$LAMBDA_DIV"
    RC=$?
    echo "  N=$N rc=$RC wall=$(( $(date +%s) - START ))s"
    if [ "$RC" -ne 0 ]; then
        if [ "$N" = "$FIRST_N" ]; then
            echo "  N=$N FAILED at the smallest size -- RUN ERROR, not a solver ceiling. Check the .err."
            exit 1
        fi
        echo "  N=$N did not complete; larger pools can only be worse. Stopping."
        break
    fi
done
echo "=== DONE $TAG ==="
