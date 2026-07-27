#!/bin/bash
#SBATCH --job-name=campaign_enum_drd2
#SBATCH --time=12:00:00                        # enumerate ~50 hubs x up to 12k children each + reward-score; generous headroom
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Target-generality (T4.3): the DRD2 analogue of submit_scent_seh_enum.sh — the exhaustive hub
# enumeration that feeds the DRD2 reactions-per-mode frontier (hub-batching vs best-candidate).
# Same recipe as sEH; only the target changes. Inputs are the 2026-07-10_17-28-06 SCENT DRD2 batch
# (checkpoint + P_B sidecar + fragments_4000 snapshot) that pairs with the scent_drd2_70190 flow
# analysis. DRD2 "mode" bar = 0.5 activity probability (matrix16 targets.py, Logs/028) — applied at
# the frontier step, not here. Runs in the `scent` env.
#
# Submit:  sbatch experiments/lsd_hubs/campaign/submit_scent_drd2_enum.sh
# Then (fast CPU frontier):
#   sweep_campaign.py --analysis-dir <ANALYSIS> --enum-children <OUT>/enum_children.json \
#       --snapshot <SNAP> --reward-threshold 0.5 --evaluator count_once --child-policy free_frag ...

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

ANALYSIS=${ANALYSIS:-/scratch/markymoo/rgfn_runs/lsdflow/scent_drd2_70190}
CKPT=${CKPT:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_drd2/2026-07-10_17-28-06/train/checkpoints/last_gfn.pt}
CFG=${CFG:-validation/configs/scent_drd2_fixed.gin}
K=${K:-100}
N_HUBS=${N_HUBS:-50}
ENUM_MAX=${ENUM_MAX:-12000}
OUT_DIR=${OUT_DIR:-$SCRATCH/rgfn_runs/lsdflow/campaign_enum_drd2_${SLURM_JOB_ID}}

export WANDB_MODE=offline
export TORCH_HOME=$SCRATCH/.cache/torch
export HF_HOME=$SCRATCH/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR" "$TORCH_HOME" "$HF_HOME"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate scent
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/scent/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"

echo "host=$(hostname)"; nvidia-smi -L
echo "ANALYSIS=$ANALYSIS  CKPT=$CKPT  K=$K  N_HUBS=$N_HUBS  ENUM_MAX=$ENUM_MAX  OUT_DIR=$OUT_DIR"

# 1) pick + rank the hub set (pure CPU, csv only)
python experiments/lsd_hubs/campaign/pick_hubs.py \
    --records "$ANALYSIS/records.csv" --out "$OUT_DIR/hubs.csv" \
    --top-k-candidates "$K" --n-hubs "$N_HUBS"

# 2) enumerate + reward-score their children on the frozen full library (scent env worker)
python validation/lsdflow/adapters/workers/scent_worker.py --mode enumerate \
    --config "$CFG" --checkpoint "$CKPT" \
    --hubs-file "$OUT_DIR/hubs.csv" --enum-max-children "$ENUM_MAX" \
    --out-dir "$OUT_DIR"

echo "DONE campaign_enum_drd2 -> $OUT_DIR (enum_children.json)"
