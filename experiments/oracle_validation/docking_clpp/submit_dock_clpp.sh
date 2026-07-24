#!/bin/bash
#SBATCH --job-name=dock_clpp
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --exclude=balam008
#SBATCH --chdir=/scratch/markymoo/rgfn_runs
#SBATCH --output=dock_clpp-%j.out
#SBATCH --error=dock_clpp-%j.err

# Dock the ClpP actives + matched decoys against human ClpP (7UVU) with the SAME
# QuickVina2-GPU oracle the 16-cell ClpP cell trains on (DockingClpPOracle), to
# calibrate that cell's mode threshold. Single GPU is plenty: ~200 actives + ~200
# decoys, docking_batch_size=200 (one QV2 process per set, Logs/036 3.3x speed-up).
# Per Balam docs: do NOT request cpus/mem; exclude the OpenCL-wedged node balam008.
set -euo pipefail

REPO=/home/markymoo/projects/RGFN_Fork/RGFN-Fork
DOCK=$REPO/experiments/oracle_validation/docking_clpp
cd "$REPO"

# rgfn env + QuickVina2-GPU boost libs (LD_LIBRARY_PATH) + GNINA (CLAUDE.md: the helper
# for the GPU docking oracle; works unchanged off the login node — shared FS).
source ~/bin/rgfn-smoke-env.sh

echo "node=$(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

# one process per set (fresh CPU budget each); they share the one GPU sequentially.
python "$DOCK/dock_sets.py" --set actives
python "$DOCK/dock_sets.py" --set matched

echo "===== ClpP discrimination + threshold ====="
python "$DOCK/benchmark_clpp_docking.py"
echo "results in $DOCK"
