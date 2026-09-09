#!/bin/bash
#SBATCH --job-name=pilotB_hubs
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# ARM-A VIABILITY PILOT, stage 2 — sample an arm-A checkpoint and rank its hubs.
#
# THE VIABILITY QUESTION IS ANSWERABLE WITHOUT ENUMERATION. Arm A gives our reaction-GFNs ~100-157
# gradient steps, and hub-batching reads the trained FLOW FIELD. So the question is whether that
# field carries structure at all: how many hubs the walk can even see, and whether their flow scores
# are separated or flat. Both come from SAMPLE + PICK_HUBS -- records.csv and the log-term flow
# estimate. Enumeration (the expensive stage) only converts that ranking into modes, and is not
# needed to decide go/no-go.
#
# `--pool all` per runbook 2.2: the flow field does the selecting, with no reward pre-filter. Under
# that pool `--n-hubs` is the ONLY cap, so it is passed explicitly rather than left to a default.
#
# Usage:
#   CKPT=<arm_a.pt> GEN=rgfn OUT=$SCRATCH/rgfn_runs/v2_pilot/hubs/rgfn_armA \
#     sbatch experiments/benchmark_v2/pilot/submit_pilot_hubs.sh
set -uo pipefail

GEN=${GEN:?set GEN=rgfn|scent}
CKPT=${CKPT:?set CKPT to the arm-A checkpoint}
OUT=${OUT:?set OUT to an output dir}
SYSTEM=${SYSTEM:-seh}
SEED=${SEED:-42}
N_TRAJ=${N_TRAJ:-30000}
N_HUBS=${N_HUBS:-200}
GUIDANCE=${GUIDANCE:-}

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "FATAL: cannot cd to '$REPO'"; exit 1; }
[ -f experiments/lsd_hubs/campaign/pick_hubs.py ] || { echo "FATAL: not an RGFN-Fork checkout"; exit 1; }
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint at $CKPT"; exit 1; }

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export XDG_CACHE_HOME=$SCRATCH/.cache/xdg
mkdir -p "$OUT/sample" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

case "$GEN" in
  rgfn)  ENV=rgfn;  WORKER=validation/lsdflow/adapters/workers/rgfn_worker.py
         CFG=configs/glue/fixed_reward_seh_proxy_stdlib_5k.gin ;;
  scent) ENV=scent; WORKER=validation/lsdflow/adapters/workers/scent_worker.py
         CFG=validation/configs/scent_${SYSTEM}_fixed_5k.gin ;;
  *) echo "FATAL: GEN must be rgfn|scent"; exit 2 ;;
esac

echo "host=$(hostname) gen=$GEN ckpt=$CKPT n_traj=$N_TRAJ n_hubs=$N_HUBS"
nvidia-smi -L || true

echo "=== [1/2] SAMPLE ==="
T0=$(date +%s)
GARG=(); [ -n "$GUIDANCE" ] && GARG=(--guidance "$GUIDANCE")
# SCENT freezes to the highest-N additional_fragments/fragments_<N>.json by default. At ARM A no
# snapshot exists at all -- the promotion schedule (every_n_iterations=1000 x 10) first fires at
# iteration 1,000 = 64,000 oracle calls, 6.4x the arm-A budget -- so the honest representation of
# an arm-A SCENT is the 418-fragment base library. Pass --no-freeze rather than letting the
# default resolve to nothing silently.
if [ "$GEN" = scent ] && [ -z "$(ls "$(dirname "$(dirname "$CKPT")")"/additional_fragments/fragments_*.json 2>/dev/null)" ]; then
    echo "[pilot-hubs] NOTE: no additional_fragments snapshot for this checkpoint -> --no-freeze (418 base library)"
    GARG+=(--no-freeze)
fi
conda run --no-capture-output -n "$ENV" python "$WORKER" \
    --mode sample --config "$CFG" --checkpoint "$CKPT" \
    --reward-name "$SYSTEM" --model-name "$GEN" \
    --n-trajectories "$N_TRAJ" --seed "$SEED" \
    --run-dir "$OUT/rundir" --out-dir "$OUT/sample" "${GARG[@]}"
RC=$?
echo "[pilot-hubs] sample exit=$RC wall=$(( $(date +%s) - T0 ))s"
[ "$RC" -ne 0 ] && exit $RC

echo "=== [2/2] PICK_HUBS (--pool all, the runbook 2.2 standard) ==="
/home/markymoo/miniconda3/envs/rgfn/bin/python experiments/lsd_hubs/campaign/pick_hubs.py \
    --records "$OUT/sample/records.csv" --out "$OUT/hubs_all.csv" \
    --pool all --order flow_desc --n-hubs "$N_HUBS" --higher-is-better true
echo "[pilot-hubs] pick_hubs(all) exit=$?"

# The v1 pool as a contrast, so the two are read off the SAME arm-A sample.
/home/markymoo/miniconda3/envs/rgfn/bin/python experiments/lsd_hubs/campaign/pick_hubs.py \
    --records "$OUT/sample/records.csv" --out "$OUT/hubs_topk.csv" \
    --pool topk_candidates --top-k-candidates 1000 --order flow_desc \
    --n-hubs "$N_HUBS" --higher-is-better true
echo "[pilot-hubs] pick_hubs(topk) exit=$?"
echo "DONE -> $OUT"
