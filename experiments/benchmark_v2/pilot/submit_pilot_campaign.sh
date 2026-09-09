#!/bin/bash
#SBATCH --job-name=pilotB_campaign
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# ARM-A VIABILITY PILOT, stage 3 — enumerate the arm-A hub set and price it.
#
# WHAT THIS TURNS FROM PREDICTION INTO MEASUREMENT. Stage 2 showed the arm-A flow field is well
# separated (top10-median 4.28 nats against v1's 3.76) and far better evidenced (median 101 estimates
# per walked hub against 10) -- so it is NOT noise. But it ranks almost exclusively DEPTH-3 hubs
# ({2:3, 3:37} in the top 40) where the 320,000-call field ranks depth-1 ({0:2, 1:31, 2:7}). A depth-3
# hub costs 3 reactions before its first child, against 1 for depth-1 and 0 for a bought depth-0
# scaffold, so the prediction is a materially worse reactions/mode. This measures it.
#
# WHY ONLY 64 HUBS. At R=100 the campaign walks ~5. 64 is a generous envelope and keeps the
# enumeration affordable; the hub set is flow-descending so the first 64 are the ones a 100-reaction
# budget could ever reach.
#
# ALSO THE FIRST REAL EXERCISE OF `depth_mix`. Agent C could not run run_campaign end to end (the
# sandbox refuses the env helper), so its depth-mix block is unit-tested by AST extraction, never
# executed. This job is where it either populates or does not -- checked explicitly at the end
# rather than assumed.
#
# Usage:
#   HUBS=<...>/pool_all/hubs.csv CKPT=<...>/last_gfn.pt OUT=<...>/campaign_all \
#     sbatch experiments/benchmark_v2/pilot/submit_pilot_campaign.sh
set -uo pipefail

GEN=${GEN:-scent}
SYSTEM=${SYSTEM:-seh}
SEED=${SEED:-42}
HUBS=${HUBS:?set HUBS to a hubs.csv}
CKPT=${CKPT:?set CKPT to the arm-A checkpoint}
SAMPLE=${SAMPLE:?set SAMPLE to the stage-2 sample dir}
OUT=${OUT:?set OUT}
GUIDANCE=${GUIDANCE:-}
N_HUBS=${N_HUBS:-64}
GATE=${GATE:-5.68}
ENUM_MAX=${ENUM_MAX:-4000}

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "FATAL: cannot cd to '$REPO'"; exit 1; }
[ -f experiments/lsd_hubs/campaign/run_campaign.py ] || { echo "FATAL: not an RGFN-Fork checkout"; exit 1; }
[ -f "$HUBS" ] || { echo "FATAL: no hubs at $HUBS"; exit 1; }

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export XDG_CACHE_HOME=$SCRATCH/.cache/xdg
mkdir -p "$OUT/enum" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

head -n $((N_HUBS + 1)) "$HUBS" > "$OUT/hubs_head.csv"
echo "host=$(hostname) gen=$GEN hubs=$(( $(wc -l < "$OUT/hubs_head.csv") - 1 )) gate=$GATE"
nvidia-smi -L || true

echo "=== [1/2] ENUMERATE ==="
T0=$(date +%s)
GARG=(); [ -n "$GUIDANCE" ] && GARG=(--guidance "$GUIDANCE")
if [ -z "$(ls "$(dirname "$(dirname "$CKPT")")"/additional_fragments/fragments_*.json 2>/dev/null)" ]; then
    echo "[pilot-campaign] no additional_fragments snapshot -> --no-freeze (418 base library)"
    GARG+=(--no-freeze)
fi
conda run --no-capture-output -n "$GEN" python validation/lsdflow/adapters/workers/${GEN}_worker.py \
    --mode enumerate --config validation/configs/scent_${SYSTEM}_fixed_5k.gin \
    --checkpoint "$CKPT" --reward-name "$SYSTEM" --model-name "$GEN" --seed "$SEED" \
    --hubs-file "$OUT/hubs_head.csv" --enum-max-children "$ENUM_MAX" \
    --run-dir "$OUT/rundir" --out-dir "$OUT/enum" "${GARG[@]}"
RC=$?
ENUM_S=$(( $(date +%s) - T0 ))
echo "[pilot-campaign] enumerate exit=$RC wall=${ENUM_S}s"
[ "$RC" -ne 0 ] && exit $RC

echo "=== [2/2] CAMPAIGN (--pool all hub set, free_frag, prebuild-k 0) ==="
T1=$(date +%s)
/home/markymoo/miniconda3/envs/rgfn/bin/python experiments/lsd_hubs/campaign/run_campaign.py \
    --analysis-dir "$SAMPLE" --enum-children "$OUT/enum/enum_children.json" \
    --reward-threshold "$GATE" --similarity 0.5 --higher-is-better true \
    --budget-reactions 100 --budget-modes 300 \
    --child-policy free_frag --prebuild-k 0 \
    --enum-timings "$OUT/enum/enum_timings.json" \
    --tag "pilot_armA_${GEN}_${SYSTEM}" --out-dir "$OUT/campaign"
RC=$?
echo "[pilot-campaign] campaign exit=$RC wall=$(( $(date +%s) - T1 ))s  enumerate=${ENUM_S}s"

# C's depth_mix block has never been EXECUTED (its sandbox refused the env helper). Say plainly
# whether it populated, rather than leaving the next reader to discover it did not.
echo "=== depth_mix populated? (agent C's block, first real exercise) ==="
/home/markymoo/miniconda3/envs/rgfn/bin/python - "$OUT/campaign/summary.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"  CANNOT READ summary.json: {exc}"); raise SystemExit(0)
for k in ("hub_pool", "min_hub_depth", "max_hub_depth", "min_synth_depth",
          "walked_depth_hist", "n_promoted_fragments"):
    print(f"  {k:<22} = {d.get(k, '<ABSENT>')}")
for arm in ("hub_batching", "best_candidate"):
    a = d.get(arm) or {}
    print(f"  {arm}.depth_mix     = {a.get('depth_mix', '<ABSENT>')}")
PY
exit $RC
