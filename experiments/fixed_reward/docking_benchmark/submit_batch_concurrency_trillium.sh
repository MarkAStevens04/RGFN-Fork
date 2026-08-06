#!/bin/bash
#SBATCH --job-name=trig_dock_bench
#SBATCH --time=01:45:00
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
#
# TRILLIUM (grillium, H100 80GB) port of
# experiments/fixed_reward/docking_benchmark/submit_batch_concurrency.sh
# so the Logs/036 A100 numbers can be compared like-for-like on an H100.
#
# The ONLY differences from the Balam script (no repo code is touched):
#   * no `module load cuda/11.8.0` (does not exist on Trillium) -> use
#     ~/bin/rgfn-smoke-env.sh, which points LD_LIBRARY_PATH at torch's bundled
#     CUDA libs instead. Cluster-agnostic by construction.
#   * no `--exclude=balam008`
#   * partition `compute` rather than `debug` (debug QOS here is MaxJobsPU=1)
#
# Same benchmark script, same molecule pool, same batch sizes, same oracles.

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork" || exit 1

OUT_DIR=$SCRATCH/rgfn_runs/docking_benchmark/trillium_${SLURM_JOB_ID:-manual}
mkdir -p "$OUT_DIR"

source ~/bin/rgfn-smoke-env.sh          # conda activate rgfn + LD_LIBRARY_PATH + $GNINA
export TORCH_HOME=$SCRATCH/.cache/torch HF_HOME=$SCRATCH/.cache/huggingface
export PYTHONUNBUFFERED=1
echo "host=$(hostname)  out=$OUT_DIR"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,persistence_mode --format=csv

# --- OpenCL health gate: prove QuickVina2-GPU can create a context on THIS node. ---
HC=$SCRATCH/vina_gpu/opencl_healthcheck
HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
if ! grep -q "clCreateContext err=0" <<<"$HC_OUT"; then
    echo "FATAL: NVIDIA OpenCL clCreateContext FAILS on $(hostname) -- bad node."
    echo "$HC_OUT"; exit 42
fi
echo "OpenCL health OK on $(hostname)"

python experiments/fixed_reward/docking_benchmark/bench_batch_concurrency.py \
        --targets clpp 6td3 \
        --smiles-csv experiments/active_learning/6td3/seed_6td3.csv \
        --n 200 --batch-sizes 25 50 100 200 \
        --concurrency-n 96 --repeats 2 \
        --out-dir "$OUT_DIR"
echo "[trillium bench] done -> $OUT_DIR"
