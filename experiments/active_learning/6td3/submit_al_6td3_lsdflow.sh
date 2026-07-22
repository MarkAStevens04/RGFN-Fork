#!/bin/bash
#SBATCH --job-name=al_6td3_lsdflow
#SBATCH --time=13:00:00                        # hub_batching est. ~8-9h (measured: ~13s/iter train, ~20 children/s enum, smoke 71025); margin for docking + best_candidate/random arms are faster
#SBATCH --partition=compute                    # 1-GPU job -> regular partition
#SBATCH --gpus-per-node=1
#SBATCH --exclude=balam008,balam009            # OpenCL-wedged / degraded nodes (Logs/013/014)
# Absolute $SCRATCH log paths: $HOME is read-only on compute nodes (Logs/012).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# LSD-Flow active-learning arm on the 6TD3 GPU differential-docking oracle.
# ONE arm per job (the three arms are independent runs of the SAME config/oracle/seed
# — only --acquisition differs). Select the arm + seed with env vars:
#
#   ARM=hub_batching  SEED=42  sbatch experiments/active_learning/6td3/submit_al_6td3_lsdflow.sh
#   ARM=best_candidate SEED=42 sbatch experiments/active_learning/6td3/submit_al_6td3_lsdflow.sh
#   ARM=random        SEED=42  sbatch experiments/active_learning/6td3/submit_al_6td3_lsdflow.sh
#
# (or use launch_lsdflow_6td3.sh to queue all three at once). Each run writes its own
# arm+seed-tagged run dir; the top-k-vs-oracle-calls curve is assembled afterwards with
# validation/harness/acquisition_curve.py over the three oracle_calls.csv traces.

set -uo pipefail
ARM="${ARM:-hub_batching}"
SEED="${SEED:-42}"
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

case "$ARM" in
    hub_batching|best_candidate|random|policy) ;;
    *) echo "FATAL: unknown ARM=$ARM"; exit 2 ;;
esac

# --- Caches + run outputs: redirect everything away from the read-only $HOME ---
export WANDB_PROJECT=rgfn
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export WANDB_CONFIG_DIR=$SCRATCH/.config/wandb
export WANDB_DATA_DIR=$SCRATCH/.cache/wandb
export WANDB_DIR=$SCRATCH/wandb
export WANDB_MODE=offline
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch
export PIP_CACHE_DIR=$SCRATCH/.cache/pip

AL_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" \
        "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$AL_ROOT_DIR"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname) ARM=$ARM SEED=$SEED"; nvidia-smi -L

# --- OpenCL health gate (necessary: wedged nodes silently make 0 poses, Logs/013). ---
HC=$SCRATCH/vina_gpu/opencl_healthcheck
if [ ! -x "$HC" ]; then echo "FATAL: healthcheck $HC missing (build on login, Logs/013)."; exit 43; fi
HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
if ! grep -q "clCreateContext err=0" <<<"$HC_OUT"; then
    echo "FATAL: NVIDIA OpenCL clCreateContext FAILS on $(hostname) -- bad node."; echo "$HC_OUT"
    echo "       Add '$(hostname)' to --exclude and resubmit."; exit 42
fi
echo "OpenCL health OK on $(hostname)"

# --- Pre-flight dock gate (sufficient: balam009 passes OpenCL yet makes 0 poses, Logs/014). ---
python experiments/active_learning/6td3/preflight_dock.py
PF=$?
if [ "$PF" -ne 0 ]; then
    echo "FATAL: pre-flight docking failed on $(hostname) (exit $PF) -- add to --exclude and resubmit."; exit "$PF"
fi

python scripts/active_learning.py \
        --cfg configs/glue/active_learning_6td3_lsdflow.gin \
        --acquisition "$ARM" \
        --seed "$SEED" \
        --root-dir "$AL_ROOT_DIR"
