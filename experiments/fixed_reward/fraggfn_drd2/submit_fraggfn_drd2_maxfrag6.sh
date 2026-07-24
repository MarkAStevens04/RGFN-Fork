#!/bin/bash
#SBATCH --job-name=fr_fraggfn_drd2_mf6
#SBATCH --time=04:00:00                        # single-shot 5000-step GFN run (no docking); smoke did 1500 in ~20m
#SBATCH --partition=compute                    # 1-GPU job -> regular partition
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# FragGFN DRD2 retrain at the PAPER's fragment cap (max_frags=6) — recovers the collapse
# (Logs/019: mean 0.03) to the paper's ~0.9. Root cause: our matrix used FragGFN's native
# max_nodes=9 -> MW ~664 oversized molecules the drug-sized DRD2 SVM scores ~0; the paper
# (Koziarski 2024, App. B.2) capped FGFN at 6. Everything else already matched the paper
# (reward=exp(48*proxy), fixed beta, bengio2021flow lib, TDC DRD2 oracle). Smoke (1500 steps,
# login A100) confirmed: mean DRD2 0.908, 93% modes, MW 385. This run is the full 5000-step
# production cell -> experiments/fixed_reward/fraggfn_drd2_maxfrag6/seed42.
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export WANDB_PROJECT=rgfn
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export WANDB_CONFIG_DIR=$SCRATCH/.config/wandb
export WANDB_DATA_DIR=$SCRATCH/.cache/wandb
export WANDB_DIR=$SCRATCH/wandb
export WANDB_MODE=offline
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" \
        "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$FR_ROOT_DIR"
# CUDA-11.8 serves BOTH envs (fraggfn torch cu118; rgfn glue/dgl cu118 for the ingest step).
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L
conda activate fraggfn
python validation/generators/fraggfn/run_fraggfn_fixed.py \
        --cfg validation/configs/fraggfn_drd2_maxfrag6.yaml \
        --seed 42 \
        --root-dir "$FR_ROOT_DIR"
