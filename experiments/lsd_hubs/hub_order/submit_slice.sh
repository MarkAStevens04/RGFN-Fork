#!/bin/bash
#SBATCH --job-name=hubord_enum
#SBATCH --time=02:00:00                        # debug max; slices are packed to ~1h estimated work
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# ONE hub slice of the hub-ordering ablation (Logs/053): exhaustively enumerate + reward-score the
# one-reaction children of the hubs in $HUBS_FILE, on the frozen full SCENT library, and write the
# usual worker outputs into $OUT_DIR. Hub sets come from plan_arms.py; the slice's hubs.csv is copied
# into $OUT_DIR so the dir self-describes as a cache source for later arms (merge_enum.py /
# plan_arms.py --auto-cache both key off hubs.csv + enum_children.json).
#
# The worker writes ONLY after finishing every hub in the slice, so an overrun loses the slice —
# that is why plan_arms.py packs to roughly half the 2h limit.
#
#   HUBS_FILE=... OUT_DIR=... sbatch experiments/lsd_hubs/hub_order/submit_slice.sh
#
# Normally driven by chain.sh (debug QoS allows exactly ONE job per user at a time).

set -uo pipefail

# Resolve the repo from this script's location so it works from a worktree without editing paths.
# $SLURM_SUBMIT_DIR FIRST: under sbatch the batch script is COPIED to a spool dir, so BASH_SOURCE
# points at /var/spool/... and deriving the repo from it yields REPO=/var/spool -- which fails as
# "python: can't open file '/var/spool/validation/.../scent_worker.py'" six seconds in (jobs
# 74447-74450). The matrix16 launchers already resolve it this way; this script predates that fix and
# only worked because it had been launched in ways that happened to leave CWD in the repo.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}
[ -f "$REPO/validation/lsdflow/adapters/workers/scent_worker.py" ] || {
    echo "ERROR: REPO=$REPO is not the repo root (no scent_worker.py). Set REPO= explicitly."; exit 2; }
cd "$REPO" || exit 2

HUBS_FILE=${HUBS_FILE:?set HUBS_FILE=<slice csv>}
OUT_DIR=${OUT_DIR:?set OUT_DIR=<enumeration output dir>}
CKPT=${CKPT:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/train/checkpoints/last_gfn.pt}
CFG=${CFG:-validation/configs/scent_seh_fixed.gin}
ENUM_MAX=${ENUM_MAX:-12000}                    # matches the canonical 70363 run (no truncation)

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
echo "REPO=$REPO"
echo "HUBS_FILE=$HUBS_FILE ($(($(wc -l < "$HUBS_FILE") - 1)) hubs)  ENUM_MAX=$ENUM_MAX  OUT_DIR=$OUT_DIR"
test -f "$HUBS_FILE" || { echo "MISSING hubs slice file $HUBS_FILE"; exit 1; }
cp "$HUBS_FILE" "$OUT_DIR/hubs.csv"

python validation/lsdflow/adapters/workers/scent_worker.py --mode enumerate \
    --config "$CFG" --checkpoint "$CKPT" \
    --hubs-file "$HUBS_FILE" --enum-max-children "$ENUM_MAX" \
    --run-dir "$OUT_DIR/run" --out-dir "$OUT_DIR"
RC=$?

if [ -f "$OUT_DIR/enum_children.json" ]; then
  echo "DONE hubord slice -> $OUT_DIR (enum_children.json + enum_timings.json)"
else
  echo "FAILED hubord slice (exit $RC): no enum_children.json in $OUT_DIR"
fi
exit $RC
