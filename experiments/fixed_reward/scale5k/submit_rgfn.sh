#!/bin/bash
#SBATCH --job-name=fr5k_rgfn
#SBATCH --time=3-00:00:00
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# One LINK of an RGFN publication-scale fixed-reward run (campaign Logs/030). Submit a CHAIN of
# these from a login node via launch_chain.sh (Balam forbids sbatch from a compute job, so each
# link is pre-submitted with an afterany dependency instead of self-resubmitting):
#
#   bash experiments/fixed_reward/scale5k/launch_chain.sh 8 \
#        experiments/fixed_reward/scale5k/submit_rgfn.sh <6td3|clpp|seh|drd2> [seed]
#
# Each link: exit early if the run already finished (candidates.csv exists); else resume from
# the last checkpoint (or start fresh) and train toward n_iterations, then sample + emit. RGFN
# checkpoints every valid_every_n_iterations, so a walltime kill loses at most that many iters;
# the next chain link resumes from the last checkpoint. Set N_ITERS_OVERRIDE for a smoke test.
set -uo pipefail
SYSTEM=${1:?usage: submit_rgfn.sh <6td3|clpp|seh|drd2> [seed]}
SEED=${2:-42}
REPO="$HOME/projects/RGFN_Fork/RGFN-Fork"
cd "$REPO"

case "$SYSTEM" in
  6td3) CFG=configs/glue/fixed_reward_6td3_5k.gin;              DOCKING=1 ;;
  clpp) CFG=configs/glue/fixed_reward_clpp_5k.gin;              DOCKING=1 ;;
  seh)  CFG=configs/glue/fixed_reward_seh_proxy_stdlib_5k.gin;  DOCKING=0 ;;
  drd2) CFG=configs/glue/fixed_reward_drd2_stdlib_5k.gin;       DOCKING=0 ;;
  *) echo "FATAL: unknown SYSTEM '$SYSTEM' (want 6td3|clpp|seh|drd2)"; exit 2 ;;
esac

export WANDB_PROJECT=rgfn WANDB_MODE=offline
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb WANDB_DIR=$SCRATCH/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch PIP_CACHE_DIR=$SCRATCH/.cache/pip
FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$FR_ROOT_DIR"

RUN_NAME="fixed_reward/rgfn_${SYSTEM}_5k/seed${SEED}"
RUN_DIR="$FR_ROOT_DIR/$RUN_NAME"
COMPLETE="$RUN_DIR/fixed_reward/candidates/candidates.csv"
mkdir -p "$RUN_DIR"

# Chain link no-op: the run already finished in an earlier link.
if [ -f "$COMPLETE" ]; then
  echo "[chain] run already complete ($COMPLETE); this link is a no-op."; exit 0
fi

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname) system=$SYSTEM seed=$SEED cfg=$CFG docking=$DOCKING"; nvidia-smi -L || true

# GPU-docking systems need a healthy QuickVina2-GPU OpenCL context (a wedged node gives
# all-NaN docks, Logs/013). If dead, exit non-zero so the next chain link tries another node.
if [ "$DOCKING" = 1 ]; then
  HC=$SCRATCH/vina_gpu/opencl_healthcheck
  HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
  if ! grep -q "clCreateContext err=0" <<<"$HC_OUT"; then
    echo "FATAL: OpenCL dead on $(hostname); exiting so the next chain link retries elsewhere."
    echo "$HC_OUT"; exit 42
  fi
  echo "OpenCL health OK on $(hostname)"
fi

# Resume from the last checkpoint if this run dir already has one. The forward policy's
# lazily-populated *_cache buffers are absent from a fresh model and break the STRICT
# load_state_dict, so strip them into a cleaned checkpoint first (Logs/021).
CKPT="$RUN_DIR/train/checkpoints/last_gfn.pt"
RESUME_ARGS=()
if [ -f "$CKPT" ]; then
  CLEAN="${CKPT%.pt}.resumeclean.pt"
  python - "$CKPT" "$CLEAN" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu")
for k in [k for k in list(d["model"]) if k.endswith("_cache")]:
    d["model"].pop(k)
torch.save(d, sys.argv[2])
print(f"[resume] cleaned checkpoint -> resume from iter {int(d['metrics']['epoch']) + 1}", flush=True)
PY
  RESUME_ARGS=(--resume-from "$CLEAN")
  echo "[chain] resuming from $CKPT"
else
  echo "[chain] fresh start (no checkpoint in $RUN_DIR)"
fi

# Optional smoke-test overrides (n_iterations and/or the emitted sample count).
ITER_ARGS=()
[ -n "${N_ITERS_OVERRIDE:-}" ] && ITER_ARGS+=(--gin-binding "Trainer.n_iterations=${N_ITERS_OVERRIDE}")
[ -n "${N_SAMPLES_OVERRIDE:-}" ] && ITER_ARGS+=(--gin-binding "FixedRewardPipeline.n_samples=${N_SAMPLES_OVERRIDE}")

python scripts/fixed_reward.py \
    --cfg "$CFG" --seed "$SEED" --root-dir "$FR_ROOT_DIR" --run-name "$RUN_NAME" \
    "${RESUME_ARGS[@]}" "${ITER_ARGS[@]}"
RC=$?
if [ -f "$COMPLETE" ]; then
  echo "[chain] $SYSTEM seed=$SEED COMPLETE (candidates emitted)."
else
  echo "[chain] $SYSTEM seed=$SEED link ended (exit $RC) without completion; next link will resume."
fi
exit $RC
