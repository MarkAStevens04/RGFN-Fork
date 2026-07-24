#!/bin/bash
#SBATCH --job-name=al_scent_seh_lsdflow
#SBATCH --time=13:00:00                         # 5 warm-started cycles: SCENT train + enum (reward-gen) + docking; margin
#SBATCH --partition=compute
#SBATCH --exclude=balam008,balam009
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# SCENT + sEH LSD-Flow active-learning arm (docs/AL_PIPELINE_ARCHITECTURE.md). One arm per job via
# env vars; SMOKE=1 uses the 1-cycle smoke config. hub_batching warm-starts from the rich sEH
# checkpoint (pre-select-K=20 bites); best_candidate/random need no warm-start.
#
#   ARM=hub_batching   SMOKE=1 sbatch experiments/active_learning/scent_seh/submit_scent_seh_lsdflow.sh
#   ARM=best_candidate         sbatch experiments/active_learning/scent_seh/submit_scent_seh_lsdflow.sh
#   ARM=random                 sbatch experiments/active_learning/scent_seh/submit_scent_seh_lsdflow.sh
#
# Two-env design: the loop + SCENT GFN run in `scent`; each round shells to the `rgfn` env for the
# glue selector (select_acquisition) AND the docking bridge (score_batch). One CUDA-11.8 covers both.

set -uo pipefail
ARM="${ARM:-hub_batching}"
SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

case "$ARM" in policy|random|hub_batching|best_candidate) ;; *) echo "FATAL: bad ARM=$ARM"; exit 2 ;; esac
CFG=validation/configs/scent_seh_lsdflow.gin
[ "$SMOKE" = "1" ] && CFG=validation/configs/scent_seh_lsdflow_smoke.gin

# Warm-start checkpoint (hub_batching only; the rich sEH library that makes pre-select-K=20 bite).
WARM_CKPT=/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/train/checkpoints/last_gfn.pt

export WANDB_PROJECT=rgfn WANDB_MODE=offline
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb WANDB_CONFIG_DIR=$SCRATCH/.config/wandb
export WANDB_DATA_DIR=$SCRATCH/.cache/wandb WANDB_DIR=$SCRATCH/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch PIP_CACHE_DIR=$SCRATCH/.cache/pip
AL_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$AL_ROOT_DIR"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname) ARM=$ARM SEED=$SEED SMOKE=$SMOKE CFG=$CFG"; nvidia-smi -L

# OpenCL health gate (docking sEH via QuickVina2-GPU; wedged nodes silently make 0 poses, Logs/013).
HC=$SCRATCH/vina_gpu/opencl_healthcheck
if [ ! -x "$HC" ]; then echo "FATAL: healthcheck $HC missing (build on login, Logs/013)."; exit 43; fi
if ! CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1 | grep -q "clCreateContext err=0"; then
    echo "FATAL: NVIDIA OpenCL FAILS on $(hostname) -- add to --exclude and resubmit."; exit 42
fi
echo "OpenCL health OK on $(hostname)"

# Warm-start ALL arms from the same sEH checkpoint: the comparison must hold the generator state
# fixed (same trained policy + frozen 2018-frag library) and vary ONLY the acquisition. (The env/policy
# also *needs* the freeze — the warm-started policy was trained on the 2018-frag library, so without it
# the 418-base env mismatches the policy embedding.)
WARM_ARG=(--warm-start-checkpoint "$WARM_CKPT")

conda activate scent
python validation/generators/scent/run_scent_al.py \
        --cfg "$CFG" \
        --seed-csv experiments/active_learning/seh/seed_seh.csv \
        --acquisition "$ARM" \
        --seed "$SEED" \
        --root-dir "$AL_ROOT_DIR" \
        "${WARM_ARG[@]}"
