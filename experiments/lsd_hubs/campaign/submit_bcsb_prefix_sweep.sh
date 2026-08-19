#!/bin/bash
#SBATCH --job-name=bcsb_prefix
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# BC-SB vs SAMPLING BUDGET -- closing the candidate-count confound (Logs/062 Result 8).
#
# THE OBJECTION. Hub-batching delivers ~3.4x more distinct molecules than BC-SB at a fixed
# 100-reaction budget, but it also SAW more candidates: 74,606 reward-gen calls against BC-SB's
# 29,997. For BC-Enum-SB the objection is already dead (it saw 131,474 -- 1.8x OURS -- and still
# lost), but for BC-SB it is live, and a reviewer will raise it.
#
# WHY A PREFIX IS A VALID BUDGET SLICE, AND COSTS NO ORACLE CALLS. records.csv is a post-hoc sample
# from a FROZEN checkpoint, not a training trace: mean reward is flat across the file (7.489 / 7.495
# / 7.505 over the first, middle and last 5k rows). So the rows are i.i.d. draws from one fixed
# policy, and the first k rows are exactly "what if we had only sampled k times" -- with no
# training-progress confound, and without re-running the generator. Every molecule in the prefix was
# already scored, so the sweep spends MILP time only.
#
# Deliberately NO --top-n cap. The whole point is that the sampling budget k is the only variable;
# capping the pool as well would confound "how many we sampled" with "how many we handed the
# solver". The pool is therefore every distinct molecule above the gate within the prefix.
#
# GATE 7.0, NOT the new headline 5.0. This experiment interrogates the numbers in Logs/062, which
# are all at 7.0; re-running it at 5.0 would answer a different question than the one asked.
#
# SMOKE FIRST -- always (a $HOME-read-only bug once made three of these "COMPLETE" in 12 s):
#   PREFIXES="2000 5000" BUDGETS=100 MILP_CAP=300 TAG=prefix_smoke SEED=43 \
#     sbatch -p debug --time=00:25:00 experiments/lsd_hubs/campaign/submit_bcsb_prefix_sweep.sh
#
# Usage:  SEED=43 sbatch -p compute --time=08:00:00 \
#           experiments/lsd_hubs/campaign/submit_bcsb_prefix_sweep.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

SEED=${SEED:-43}
RUN=${RUN:-/scratch/markymoo/rgfn_runs/lsdflow/t45_seh_seed${SEED}}
SNAP=${SNAP:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json}
TAG=${TAG:-bcsb_prefix_seed${SEED}}
GATE=${GATE:-7.0}
CUTOFF=${CUTOFF:-0.5}
LAMBDA_DIV=${LAMBDA_DIV:-1.0}
BUDGETS=${BUDGETS:-100}
PREFIXES=${PREFIXES:-"2000 5000 10000 20000 30000"}
MILP_CAP=${MILP_CAP:-3600}
# $HOME is READ-ONLY on Balam compute nodes -- results go to $SCRATCH (see submit_bc_sb_scale.sh).
OUT_ROOT=${OUT_ROOT:-/scratch/markymoo/rgfn_runs/lsdflow_sparrow/bc_sb_prefix}
WORK=${WORK:-/scratch/markymoo/rgfn_runs/lsdflow_sparrow/_prefix_pools}

export PYTHONUNBUFFERED=1
# Libraries that default their caches to $HOME crash on a read-only compute node.
export MPLCONFIGDIR="$SCRATCH/.cache/matplotlib"
export TRITON_CACHE_DIR="$SCRATCH/.cache/triton"
export HF_HOME="$SCRATCH/.cache/hf"
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$HF_HOME" "$WORK"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/rgfn/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"

POOL="$RUN/sample/records.csv"
[ -s "$POOL" ] || { echo "FATAL: no records.csv at $POOL" >&2; exit 1; }
TOTAL=$(( $(wc -l < "$POOL") - 1 ))
echo "host=$(hostname)  SEED=$SEED  RUN=$RUN  events=$TOTAL  prefixes=$PREFIXES  gate=$GATE"

FIRST_K=$(echo $PREFIXES | awk '{print $1}')
for K in $PREFIXES; do
    OUT="$OUT_ROOT/${TAG}_k${K}"
    if [ -s "$OUT/select_frontier.csv" ]; then
        echo "=== k=$K already done -> SKIP ==="; continue
    fi
    # A prefix longer than the file is the full file -- record it as such rather than silently
    # re-running the same point under a different label.
    KEFF=$K; [ "$K" -gt "$TOTAL" ] && KEFF=$TOTAL
    PFX="$WORK/${TAG}_k${K}.csv"
    head -n $(( KEFF + 1 )) "$POOL" > "$PFX"
    echo "=== k=$K (effective $KEFF of $TOTAL events) ==="
    START=$(date +%s)
    python experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
        --pool "$PFX" --routes "$RUN/sample/routes.json" --route-source native \
        --snapshot "$SNAP" --top-n 0 \
        --gate "$GATE" --cutoff "$CUTOFF" \
        --out-dir "$OUT" --tag "${TAG}_k${K}" \
        --budgets "$BUDGETS" --max-seconds "$MILP_CAP" \
        --lambda-div "$LAMBDA_DIV"
    RC=$?
    echo "  k=$K rc=$RC wall=$(( $(date +%s) - START ))s"
    # A failure at the SMALLEST prefix cannot be a scaling limit (that pool is tiny) -- it is a bug
    # in the run, and calling it a ceiling would invent a limitation that does not exist.
    if [ "$RC" -ne 0 ]; then
        if [ "$K" = "$FIRST_K" ]; then
            echo "  k=$K FAILED at the smallest prefix -- RUN ERROR, not a solver ceiling. Check the .err."
            exit 1
        fi
        echo "  k=$K did not complete; larger prefixes can only be slower. Stopping."
        break
    fi
done
echo "=== DONE $TAG ==="
