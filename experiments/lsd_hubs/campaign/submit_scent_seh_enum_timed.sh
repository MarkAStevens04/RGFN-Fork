#!/bin/bash
#SBATCH --job-name=enum_timed_seh
#SBATCH --time=02:00:00                        # debug max; each ~145k-child slice ~1.2h + setup
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Timed re-enumeration of ONE hub slice (Logs/039): re-run the exact canonical enumeration
# (campaign_enum_seh_70363) for a subset of its 200 ranked hubs, WITH the worker's per-hub
# compute-time instrumentation on, to measure real enumeration / reward-gen / flow-extract
# wall-clock. Only enum_timings.json is harvested (the enum_children it also writes are byte-for-byte
# the cached ones; the campaign selection reuses the canonical cache). Split into 6 slices so each
# fits the 2h debug limit; run in parallel, then merge_enum_timings.py unions the per-hub timings.
#
# Submit one slice:   sbatch experiments/lsd_hubs/campaign/submit_scent_seh_enum_timed.sh 0
# All six:            for s in 0 1 2 3 4 5; do sbatch experiments/lsd_hubs/campaign/submit_scent_seh_enum_timed.sh $s; done

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

SLICE=${1:-${SLICE:-0}}
TIMED=${TIMED:-/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_timed}
CKPT=${CKPT:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/train/checkpoints/last_gfn.pt}
CFG=${CFG:-validation/configs/scent_seh_fixed.gin}
ENUM_MAX=${ENUM_MAX:-12000}                    # matches the canonical 70363 run (no truncation)
HUBS_FILE="$TIMED/slices/hubs_slice${SLICE}.csv"
OUT_DIR="$TIMED/slice${SLICE}"

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
echo "SLICE=$SLICE  HUBS_FILE=$HUBS_FILE  ENUM_MAX=$ENUM_MAX  OUT_DIR=$OUT_DIR"
test -f "$HUBS_FILE" || { echo "MISSING hubs slice file $HUBS_FILE"; exit 1; }

python validation/lsdflow/adapters/workers/scent_worker.py --mode enumerate \
    --config "$CFG" --checkpoint "$CKPT" \
    --hubs-file "$HUBS_FILE" --enum-max-children "$ENUM_MAX" \
    --run-dir "$OUT_DIR/run" --out-dir "$OUT_DIR"

echo "DONE enum_timed slice $SLICE -> $OUT_DIR/enum_timings.json"
