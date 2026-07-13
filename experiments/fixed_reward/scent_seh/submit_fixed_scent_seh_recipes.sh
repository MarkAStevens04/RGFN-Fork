#!/bin/bash
#SBATCH --job-name=fr_scent_seh_recipes
#SBATCH --time=24:00:00                       # 5000-iter single-shot GFN run: ~10-14h (matches the entry-024 patched re-run, job 70066)
#SBATCH --partition=compute                   # 1-GPU job -> regular partition
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# SCENT sEH fixed-reward re-run WITH synthesis-recipe logging (Logs/027, LSD-Flow exact cost).
# Identical to submit_fixed_scent_seh.sh (5000-iter, frozen @SehMoleculeProxy, guidance sidecar
# for P_B recovery per entry 024) EXCEPT it passes --log-recipes: our recipe_logging.py
# monkeypatches DynamicLibrary to record each promoted dynamic-library fragment's min-reaction
# SYNTHESIS ROUTE (ordered reaction steps: reaction + reactants + product) into the
# additional_fragments/fragments_<N>.json snapshot (alongside the existing chosen_smiles /
# min_num_reactions / $-cost).
#
# WHY: the LSD-Flow reactions-per-mode / amortization metric must charge each promoted fragment's
# own build cost EXACTLY ONCE per costed library, INCLUDING nested (fragment-built-from-fragment)
# cost — which needs each fragment's actual recipe (only observable during training; once promoted
# the model uses it atomically). The same routes feed a future chemist-facing "synthesize these
# intermediates" view. The SCENT clone is untouched (monkeypatch in our adapter layer).
#
# Supersedes the recipe-less 5000-iter run (2026-07-07_17-16-09) as the SCENT sEH anchor for the
# LSD-Flow analysis. Guidance sidecar (guidance_models.pt) is emitted as usual (P_B recoverable).
#
# Submit with:  sbatch experiments/fixed_reward/scent_seh/submit_fixed_scent_seh_recipes.sh

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

# CUDA-11.8 serves BOTH envs (scent torch/dgl cu118; rgfn glue/dgl cu118 for ingest).
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L

conda activate scent
python validation/generators/scent/run_scent_fixed.py \
        --cfg validation/configs/scent_seh_fixed.gin \
        --seed 42 \
        --root-dir "$FR_ROOT_DIR" \
        --log-recipes
