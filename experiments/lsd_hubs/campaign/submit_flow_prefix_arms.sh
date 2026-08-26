#!/bin/bash
#SBATCH --job-name=flowprefix
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# BOTH ARMS ON ONE FROZEN CANDIDATE FILE -- the flow-prefix pool.
#
# THE QUESTION (user framing, 2026-08-26): "how well does SPARROW do if we gave it the exact same
# candidates as hub-batching?"  Every earlier attempt at this failed on the candidate set, not the
# solver: BC-Enum-SB's hubs are reward-picked (35/64 overlap with ours), and the first HB-Enum-SB run
# read a live scratch path that another job rewrote mid-flight. So the pool here is a frozen file with
# a recorded md5, built by make_flow_prefix_subsample.py, and BOTH arms read that one file.
#
# WHY A FLOW PREFIX AND NOT A RANDOM SUBSAMPLE. A uniform random draw thins every hub by the same
# factor (measured, gate 7.0 seed 43: median 423 qualifying children per hub -> ~71), and per-hub
# density is exactly what hub-batching exploits -- it amortises one hub prefix over many children. So
# random subsampling would attack the mechanism under test. The prefix instead takes the pool our arm
# actually REACHES: hubs in flow-descending order, all children kept, until ~20k qualifying. At R=100
# hub-batching touches ~2 hubs, so a 5-6 hub prefix is already generous relative to what we spend.
#
# GATE 5.0 -- the paper's sEH gate (`targets.py` mode_reward_threshold). NOTE the earlier 50k and
# full-pool HB-Enum-SB timeouts were measured at GATE 7.0, whose qualifying pool is 3.6x SMALLER
# (125k vs 455k on seed 43). Those runs are therefore NOT the same-gate bracket; they bound this one
# only a fortiori (a solver that times out on the smaller pool times out on the larger). That is why
# the 50k tier is re-run here at 5.0 rather than quoted from the 7.0 result.
#
# THREE ARMS, one pool:
#   hb      hub-batching (ours)          run_campaign.py, count-once cost model
#   sb      SPARROW-Batching             its MILP selects under a hard reaction budget
#   greedy  reward-ranked tau-diverse    selection is greedy, SPARROW only PRICES it -- the
#           greedy, SPARROW-priced       baseline's STRONGEST config (diversity-aware)
#
# A TimeLimit row is a LOWER bound on the competitor, i.e. it flatters us, and must be labelled one.
# CBC reports `Optimal` for whatever it holds when the wall stops it, so status alone is not enough --
# read solve_s against MILP_CAP. The timeout is itself a reportable result here.
#
# Usage:
#   sbatch experiments/lsd_hubs/campaign/submit_flow_prefix_arms.sh          # TIER/SEED from env
#   TIER=flow20k SEED=43 sbatch experiments/lsd_hubs/campaign/submit_flow_prefix_arms.sh
#   TIER=full SEED=43 MILP_CAP=21600 sbatch .../submit_flow_prefix_arms.sh
set -uo pipefail
cd /home/markymoo/projects/RGFN_Fork/RGFN-Fork

TGT=${TGT:-seh}
SEED=${SEED:-43}
TIER=${TIER:-flow20k}
GATE=${GATE:-5.0}
CUTOFF=${CUTOFF:-0.5}
SIMILARITY=${SIMILARITY:-0.5}
LAMBDA_DIV=${LAMBDA_DIV:-1.0}
BUDGET_REACTIONS=${BUDGET_REACTIONS:-100}
BUDGET_MODES=${BUDGET_MODES:-300}

case "$TIER" in
    flow20k) POOL_DEF=/scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_subsample_flow20k/${TGT}_seed${SEED}
             BUDGETS_DEF="50,100,150"; CAP_DEF=7200 ;;
    flow50k) POOL_DEF=/scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_subsample_flow50k/${TGT}_seed${SEED}
             BUDGETS_DEF="100";         CAP_DEF=21600 ;;
    full)    POOL_DEF=/scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_snapshot_20260820/${TGT}_seed${SEED}
             BUDGETS_DEF="100";         CAP_DEF=21600 ;;
    *) echo "FATAL: unknown TIER=$TIER (flow20k|flow50k|full)" >&2; exit 1 ;;
esac
POOL=${POOL:-$POOL_DEF}
BUDGETS=${BUDGETS:-$BUDGETS_DEF}
MILP_CAP=${MILP_CAP:-$CAP_DEF}
GAP_REL=${GAP_REL:-}
# --mode-points at 1-MODE granularity. The default grid (25,50,75,...) cannot be read at a fixed
# reaction budget at all: around R=100 it jumps 25 modes/70 rxns to 50 modes/128, so nothing lands
# on the budget and the arm can only be bounded to a 2x-wide interval -- the exact defect commit
# 2564000 fixed for the s3gfn cell. Priced per mode instead, so R=100 reads off exactly. Pricing a
# FIXED selection is a much smaller MILP than selecting one, hence its own shorter cap.
MODE_POINTS=${MODE_POINTS:-$(seq -s, 1 150)}
GREEDY_CAP=${GREEDY_CAP:-1800}

ENUM="$POOL/enum_children.json"
HUB_ROUTES="$POOL/routes.json"
# The campaign reads records.csv + compositions.json for the cost table from the SAMPLE dir; only the
# hub_batching arm is quoted from this run, and it reads candidates solely from --enum-children.
SAMPLE_DIR=${SAMPLE_DIR:-/scratch/markymoo/rgfn_runs/lsdflow/matrix16_seed${SEED}/scent_${TGT}/sample}
SNAP=${SNAP:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_${TGT}_5k/seed${SEED}/additional_fragments/fragments_4000.json}

TAG=${TAG:-flowpre_${TGT}_seed${SEED}_${TIER}}
OUT_ROOT=${OUT_ROOT:-/scratch/markymoo/rgfn_runs/lsdflow_sparrow/flow_prefix}
OUT="$OUT_ROOT/$TAG"
mkdir -p "$OUT"

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$SCRATCH/.cache/matplotlib"
export TRITON_CACHE_DIR="$SCRATCH/.cache/triton"
export HF_HOME="$SCRATCH/.cache/hf"
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$HF_HOME"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/rgfn/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"

for P in "$ENUM" "$HUB_ROUTES" "$SNAP" "$SAMPLE_DIR/records.csv"; do
    [ -s "$P" ] || { echo "FATAL: missing input $P" >&2; exit 1; }
done

echo "host=$(hostname)  TGT=$TGT SEED=$SEED TIER=$TIER gate=$GATE tau=$CUTOFF cap=${MILP_CAP}s"
echo "  POOL=$POOL"
echo "  pool md5: $(cat "$POOL/enum_children.md5" 2>/dev/null || md5sum "$ENUM")"
[ -s "$POOL/subsample_manifest.json" ] && python -c "
import json,sys; m=json.load(open('$POOL/subsample_manifest.json'))
print('  manifest: hubs %s/%s | %s qualifying (%.1f%% of pool) | %s children | gate %s'%(
  m['hubs_kept'],m['hubs_total'],f\"{m['qualifying_kept']:,}\",
  100*m['qualifying_fraction_of_pool'],f\"{m['children_kept']:,}\",m['gate']))"

# ---- arm 1: hub-batching (ours), count-once ------------------------------------------------------
# SCENT convention from run_cell_campaign.sh: free_frag + pre-select-K=20 (Logs/037). Kept identical
# so this run differs from the matrix cell in the POOL alone, not the policy.
echo "=== arm hb (hub-batching) ==="
S=$(date +%s)
python experiments/lsd_hubs/campaign/run_campaign.py \
    --analysis-dir "$SAMPLE_DIR" --enum-children "$ENUM" --snapshot "$SNAP" \
    --reward-threshold "$GATE" --higher-is-better true \
    --similarity "$SIMILARITY" \
    --budget-reactions "$BUDGET_REACTIONS" --budget-modes "$BUDGET_MODES" \
    --child-policy free_frag --prebuild-k 20 \
    --tag "${TAG}_hb" --out-dir "$OUT/hb"
echo "  arm hb rc=$? wall=$(( $(date +%s) - S ))s"

# ---- arm 2: SPARROW-Batching (its MILP selects) --------------------------------------------------
echo "=== arm sb (SPARROW-Batching) ==="
S=$(date +%s)
python experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
    --routes "$ENUM" --route-source enum --hub-routes "$HUB_ROUTES" \
    --snapshot "$SNAP" --top-n 0 \
    --gate "$GATE" --cutoff "$CUTOFF" \
    --selection sparrow --budgets "$BUDGETS" \
    --lambda-div "$LAMBDA_DIV" --max-seconds "$MILP_CAP" \
    ${GAP_REL:+--gap-rel "$GAP_REL"} \
    --out-dir "$OUT/sb" --tag "${TAG}_sb"
echo "  arm sb rc=$? wall=$(( $(date +%s) - S ))s"

# ---- arm 3: greedy selection, SPARROW pricing (baseline's strongest) -----------------------------
echo "=== arm greedy (tau-diverse greedy, SPARROW-priced) ==="
S=$(date +%s)
python experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
    --routes "$ENUM" --route-source enum --hub-routes "$HUB_ROUTES" \
    --snapshot "$SNAP" --top-n 0 \
    --gate "$GATE" --cutoff "$CUTOFF" \
    --selection greedy --mode-points "$MODE_POINTS" --max-seconds "$GREEDY_CAP" \
    --out-dir "$OUT/greedy" --tag "${TAG}_greedy"
echo "  arm greedy rc=$? wall=$(( $(date +%s) - S ))s"

echo "=== DONE $TAG -> $OUT ==="
