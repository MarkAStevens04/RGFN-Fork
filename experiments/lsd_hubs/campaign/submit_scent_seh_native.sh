#!/bin/bash
#SBATCH --job-name=scent_seh_native
#SBATCH --time=02:00:00                         # RESUMABLE: skips sample/pick if present, so a re-run on an existing OUT_DIR is enum-only (~40 min). Fresh full run (30k sample + enum) needs ~9h — raise --time then. Sized to fit before the 2026-07-21 04:00 maintenance for the enum-only re-run.
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Full-scale SCENT sEH re-run WITH native routes — the CHECK-1 (native-route SPARROW) anchor.
#
# The campaign anchor (scent_seh_70189 sample + campaign_enum_seh_70363) predates two worker
# features and so CANNOT support native-route SPARROW: its sampling has no routes.json, and its
# enumeration has no per-child `reaction` field. Count-once works there (uses compositions + recipe),
# but native-route SPARROW needs the by-construction routes. The CURRENT scent_worker writes BOTH
# automatically, so this re-run reproduces the anchor scale (30k traj, 200 hubs) as a complete
# routes.json pipeline — exactly recon_smoke_70526's shape, at headline scale.
#
# Output (mirrors recon_smoke): <run>/sample/{records.csv,compositions.json,routes.json,...} +
# <run>/enum/{enum_children.json (with `reaction`),...}. Feeds reconcile_t15.py (full-coverage
# best-candidate too) and native-route SPARROW on the frontier (--route-source native).
#
# Sequential in one job (sample -> pick_hubs -> enumerate); the sample persists BEFORE enumeration,
# so a timeout still leaves routes.json (best-candidate native routes) — resume just the enum then.
#
# Submit:  sbatch experiments/lsd_hubs/campaign/submit_scent_seh_native.sh
# Override: N_TRAJ=30000 N_HUBS=200 ENUM_MAX=12000 K=100 sbatch ...

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

CKPT=${CKPT:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/train/checkpoints/last_gfn.pt}
CFG=${CFG:-validation/configs/scent_seh_fixed.gin}
N_TRAJ=${N_TRAJ:-30000}
SAMPLE_BATCH=${SAMPLE_BATCH:-200}
K=${K:-100}                                      # top-K candidates -> their parent hubs
N_HUBS=${N_HUBS:-200}                            # match campaign_enum_seh_70363 (200-hub)
ENUM_MAX=${ENUM_MAX:-12000}                      # per-hub cap on the frozen 2,018-frag library
REWARD=${REWARD:-seh}
OUT_DIR=${OUT_DIR:-$SCRATCH/rgfn_runs/lsdflow/scent_seh_native_${SLURM_JOB_ID}}
SAMPLE_DIR="$OUT_DIR/sample"
ENUM_DIR="$OUT_DIR/enum"

export TORCH_HOME=$SCRATCH/.cache/torch
export HF_HOME=$SCRATCH/.cache/huggingface
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb
export PYTHONUNBUFFERED=1
mkdir -p "$TORCH_HOME" "$HF_HOME" "$WANDB_DIR" "$SAMPLE_DIR" "$ENUM_DIR"

module load cuda/11.8.0                          # dgl graphbolt CUDA-11.8 runtime (compute node)
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate scent
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/scent/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"

echo "host=$(hostname)"; nvidia-smi -L
echo "CKPT=$CKPT"
echo "N_TRAJ=$N_TRAJ K=$K N_HUBS=$N_HUBS ENUM_MAX=$ENUM_MAX OUT_DIR=$OUT_DIR"

# --- 1. Sample -> records.csv + compositions.json + routes.json (native routes). RESUMABLE. -------
if [ -s "$SAMPLE_DIR/routes.json" ] && [ -s "$SAMPLE_DIR/records.csv" ]; then
    echo "=== [1/3] sample: routes.json + records.csv already present in $SAMPLE_DIR — SKIP ==="
else
    echo "=== [1/3] sample N=$N_TRAJ -> routes.json ==="
    python validation/lsdflow/adapters/workers/scent_worker.py --mode sample \
        --config "$CFG" --checkpoint "$CKPT" \
        --n-trajectories "$N_TRAJ" --batch-size "$SAMPLE_BATCH" \
        --reward-name "$REWARD" --out-dir "$SAMPLE_DIR"
    if [ ! -s "$SAMPLE_DIR/routes.json" ] || [ ! -s "$SAMPLE_DIR/records.csv" ]; then
        echo "[native] FATAL: sample step did not produce routes.json + records.csv — aborting." >&2
        exit 1
    fi
fi

# --- 2. Pick + rank the hub set (pure CPU; rgfn env has pick_hubs' deps but scent works too). RESUMABLE. --
if [ -s "$OUT_DIR/hubs.csv" ]; then
    echo "=== [2/3] pick_hubs: hubs.csv already present — SKIP ==="
else
    echo "=== [2/3] pick_hubs -> $N_HUBS hubs ==="
    python experiments/lsd_hubs/campaign/pick_hubs.py \
        --records "$SAMPLE_DIR/records.csv" --out "$OUT_DIR/hubs.csv" \
        --top-k-candidates "$K" --n-hubs "$N_HUBS"
fi

# --- 3. Enumerate their children -> enum_children.json WITH the per-child `reaction` field. --------
# ALWAYS (re-)run: this is the step the fix in scent_worker.py (ReactionStateEarlyTerminal guard, the
# crash that left enum_children.json missing in job 70974) targets.
echo "=== [3/3] enumerate $N_HUBS hubs -> enum_children.json (reaction field) ==="
python validation/lsdflow/adapters/workers/scent_worker.py --mode enumerate \
    --config "$CFG" --checkpoint "$CKPT" \
    --hubs-file "$OUT_DIR/hubs.csv" --enum-max-children "$ENUM_MAX" \
    --out-dir "$ENUM_DIR"
if [ ! -s "$ENUM_DIR/enum_children.json" ]; then
    echo "[native] FATAL: enumerate step did not write enum_children.json — see errors above." >&2
    exit 1
fi

echo ""
echo "DONE scent_seh_native -> $OUT_DIR"
echo "  sample: $SAMPLE_DIR/{records.csv,compositions.json,routes.json}"
echo "  enum:   $ENUM_DIR/enum_children.json (with reaction field)"
echo "Next: reconcile_t15.py --recon-dir $OUT_DIR (full-coverage) + native-route SPARROW headline."
