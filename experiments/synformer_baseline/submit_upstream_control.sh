#!/bin/bash
#SBATCH --job-name=sf_control
#SBATCH --time=02:00:00
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# CONTROL RUN — SynFormer's own GraphGA-SF loop, none of our divergences. See upstream_control.py.
#
# Submit (debug is 2 h and near-instant, and this is a diagnostic, not a campaign):
#   sbatch experiments/synformer_baseline/submit_upstream_control.sh
#   GENERATIONS=20 sbatch -p compute -t 06:00:00 .../submit_upstream_control.sh   # longer run

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

GENERATIONS=${GENERATIONS:-12}
# One divergence per variable; default is pure upstream. See upstream_control.py.
WORKERS=${WORKERS:-1}
ROUTES=${ROUTES:-0}                       # 1 = apply the route patch AND assert the column is filled
TORCH_IN_PARENT=${TORCH_IN_PARENT:-0}     # 1 = build the sEH proxy before forking (fork-after-torch)
ENV_PREFIX=${SYNFORMER_ENV_PREFIX:-/scratch/markymoo/conda_envs/synformer}

EXTRA=()
[ "$ROUTES" = "1" ] && EXTRA+=(--routes)
[ "$TORCH_IN_PARENT" = "1" ] && EXTRA+=(--torch-in-parent)

# Compute nodes cannot write $HOME; anything that caches there kills the job minutes in.
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton
export MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export HF_HOME=$SCRATCH/.cache/hf
export XDG_CACHE_HOME=$SCRATCH/.cache
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$HF_HOME"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)"; nvidia-smi -L
echo "generations=$GENERATIONS workers=$WORKERS routes=$ROUTES torch_in_parent=$TORCH_IN_PARENT"
echo "env=$ENV_PREFIX"
echo "clone=external/synformer @ $(git -C external/synformer rev-parse HEAD 2>/dev/null || echo '?')"

source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda run --no-capture-output -p "$ENV_PREFIX" \
    python experiments/synformer_baseline/upstream_control.py \
        --generations "$GENERATIONS" --workers "$WORKERS" "${EXTRA[@]}"
rc=$?
echo "CONTROL rc=$rc"
exit $rc
