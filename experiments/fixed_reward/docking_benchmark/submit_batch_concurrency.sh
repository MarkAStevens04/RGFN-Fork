#!/bin/bash
#SBATCH --job-name=fr_batch_bench
#SBATCH --time=02:00:00                        # ~30-45 min of docking; debug cap is 2h
#SBATCH --partition=debug                      # short benchmark -> debug (NOT the main compute partition)
#SBATCH --exclude=balam008                     # balam008 OpenCL wedged (Logs/013/014): QV2-GPU -> all no_pose
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Logs/036 -- QuickVina2-GPU docking batch-size sweep + 1-vs-2 concurrent-process test
# on ONE A100. Part A: s/mol vs batch size (25/50/100/200) for ClpP + 6TD3 live oracles.
# Part B: aggregate throughput of one QV2 process vs two concurrent processes on the same
# GPU (with nvidia-smi utilisation). See bench_batch_concurrency.py.
#
# Submit with:  sbatch experiments/fixed_reward/docking_benchmark/submit_batch_concurrency.sh

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

OUT_DIR=$SCRATCH/rgfn_runs/docking_benchmark/${SLURM_JOB_ID:-manual}
mkdir -p "$OUT_DIR"

module load cuda/11.8.0                 # dgl graphbolt AND QuickVina2-GPU OpenCL runtime
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

# QuickVina2-GPU-2.1 boost runtime libs + gnina launcher.
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
export GNINA=/scratch/markymoo/gnina/run_gnina.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname)  out=$OUT_DIR"; nvidia-smi -L

# --- OpenCL health gate: prove QuickVina2-GPU can create a context on THIS node. ---
HC=$SCRATCH/vina_gpu/opencl_healthcheck
if [ ! -x "$HC" ]; then
    echo "FATAL: healthcheck binary $HC missing -- build it on the login node (see Logs/013)."
    exit 43
fi
HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
if ! grep -q "clCreateContext err=0" <<<"$HC_OUT"; then
    echo "FATAL: NVIDIA OpenCL clCreateContext FAILS on $(hostname) -- bad node."
    echo "$HC_OUT"
    echo "       Add '$(hostname)' to the #SBATCH --exclude list and resubmit."
    exit 42
fi
echo "OpenCL health OK on $(hostname)"

python experiments/fixed_reward/docking_benchmark/bench_batch_concurrency.py \
        --targets clpp 6td3 \
        --smiles-csv experiments/active_learning/6td3/seed_6td3.csv \
        --n 200 --batch-sizes 25 50 100 200 \
        --concurrency-n 96 --repeats 2 \
        --out-dir "$OUT_DIR"
echo "[submit_batch_concurrency] done -> $OUT_DIR"
