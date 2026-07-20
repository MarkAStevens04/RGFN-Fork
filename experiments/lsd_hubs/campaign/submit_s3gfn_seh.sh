#!/bin/bash
#SBATCH --job-name=s3gfn_seh
#SBATCH --time=06:00:00                          # de-risk ~3 s/step -> 5000 steps ~4.5h incl. model load + evals + pool + ingest; 6h buffer. SIZED TO FIT before the 2026-07-21 04:00 maintenance reservation (submit before ~21:30).
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# S3-GFN FIXED-REWARD full run on sEH — the marquee NON-reaction baseline (LSD-Flow T3.1, Logs/040).
# Trains S3-GFN's SMILES GFlowNet (GP-MolFormer + soft synthesizability) ONCE against OUR frozen sEH
# proxy (injected in place of native get_scores), then samples a pool and emits a standard candidate
# dataset with has_route=0 (routes recovered post-hoc via AiZynth->SPARROW in T3.2).
#
# Synthesizability env = zincfrag_hb105 (ZINCFrag public blocks + hb.txt/105 templates); see the
# decision in experiments/lsd_hubs/campaign/build_s3gfn_retro_env.sh + validation/configs/
# s3gfn_seh_fixed.yaml. De-risk (200 steps): synth_ratio 4.7%->93.8%, pool 96% synthesizable.
#
# Submit:  sbatch experiments/lsd_hubs/campaign/submit_s3gfn_seh.sh
# Override: STEPS=5000 N_SAMPLES=2000 SEED=42 sbatch ...

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

CFG=${CFG:-validation/configs/s3gfn_seh_fixed.yaml}
STEPS=${STEPS:-5000}
N_SAMPLES=${N_SAMPLES:-2000}
SEED=${SEED:-42}
RETRO_ENV=${RETRO_ENV:-zincfrag_hb105}
OUT_DIR=${OUT_DIR:-$SCRATCH/rgfn_runs/experiments/fixed_reward/s3gfn_seh/${SLURM_JOB_ID}}

export TORCH_HOME=$SCRATCH/.cache/torch
export HF_HOME=$SCRATCH/.cache/huggingface           # pre-fetched GP-MolFormer weights (setup_s3gfn.sh)
export PYTHONUNBUFFERED=1
mkdir -p "$TORCH_HOME" "$HF_HOME" "$OUT_DIR"

# dgl/graphbolt CUDA-11.8 runtime for the INGEST child (rgfn env imports glue->rgfn->dgl). Safe for
# the s3gfn parent: cuda 11.8 exposes .11 sonames, torch cu121 loads its bundled .12 libs via RUNPATH.
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate s3gfn

# The ingest child (conda run -n rgfn) needs rgfn's torch-bundled CUDA libs (dgl) on LD_LIBRARY_PATH,
# but we must NOT put them on the s3gfn parent's path (torch cu121). run_s3gfn_fixed.py sets
# LD_LIBRARY_PATH for the ingest subprocess ONLY, from this var. Include the cuda module path too.
_RGFN_SP=$(conda run -n rgfn python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
_RGFN_NVLIBS=$(find "$_RGFN_SP/nvidia" -name lib -type d 2>/dev/null | paste -sd:)
export RGFN_INGEST_LD_LIBRARY_PATH="${_RGFN_NVLIBS}:$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}"

echo "host=$(hostname)"; nvidia-smi -L
echo "CFG=$CFG STEPS=$STEPS N_SAMPLES=$N_SAMPLES SEED=$SEED RETRO_ENV=$RETRO_ENV"
echo "OUT_DIR=$OUT_DIR"

python validation/generators/s3gfn/run_s3gfn_fixed.py \
    --cfg "$CFG" \
    --run-dir "$OUT_DIR" \
    --seed "$SEED" \
    --n-train-steps "$STEPS" \
    --n-samples "$N_SAMPLES" \
    --retro-env "$RETRO_ENV"

echo ""
echo "DONE s3gfn_seh -> $OUT_DIR"
echo "  candidates: $OUT_DIR/fixed_reward/candidates/{candidates.csv,manifest.json}  (has_route=0)"
echo "  pool:       $OUT_DIR/fixed_reward/pairs.csv"
echo "Next (T3.2): route the pool via AiZynth->SPARROW + place on the reactions-per-mode frontier."
