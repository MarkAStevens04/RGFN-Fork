#!/bin/bash
#SBATCH --job-name=lsdflow_seh_enum
#SBATCH --time=12:00:00                        # RGFN env rollout is CPU-bound on glue_standard_v1 (112 rxns, Logs/020): ~0.3-0.5 s/traj -> ~3-4 h for 30k traj, + deep-hub enumeration (a backward disconnection per child). 12 h is generous headroom.
#SBATCH --partition=compute                    # 1-GPU job -> regular (non-full-node) partition, like the fixed_reward runs
#SBATCH --gpus-per-node=1
# Absolute $SCRATCH log paths: $HOME is read-only on compute nodes (Logs/012).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# LSD-Flow phase-2: fresh STEREO-KEYED DAG + deep-hub exhaustive enumeration on the trained
# RGFN sEH stdlib model (Logs/025). A fresh run (not --from-records) reconstructs hubs from the
# LIVE stereo SMILES in the DAG, so depth>=1 (stereo-bearing) hubs enumerate faithfully — the
# committed 10k run couldn't (its records.csv predates the stereo columns), which is why its
# deeper hubs skipped. This job lands the reactions-per-mode cost-amortization result on the
# INTERIOR (depth 1-2) hubs where the saving actually applies (a depth-0 fragment costs 0
# reactions -> diversity but no saving; depth-3 children hit the max-reaction boundary).
#
# GFN INFERENCE ONLY — no docking. Needs a compute node (not the login node) because a large
# sample exceeds the login CPU-time cap (Logs/011, SIGXCPU) and wants uninterrupted wall time.
#
# Submit:  sbatch validation/lsdflow/submit_seh_enum.sh
# Override knobs: N_TRAJ=50000 ENUM_HUBS=8 sbatch validation/lsdflow/submit_seh_enum.sh
#
# After it finishes, copy the SMALL artifacts back into the repo from the login node:
#   cp $SCRATCH/rgfn_runs/lsdflow/<run>/{meta.json,report.json,acquisitions.csv,enumeration.json,hub_summary.csv} \
#      validation/lsdflow/results/seh_rgfn_enum/
# (leave the large records.csv / enumerated_records.csv / graph.gpickle on scratch).

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

# Tunables (env-overridable).
N_TRAJ=${N_TRAJ:-30000}
SAMPLE_BATCH=${SAMPLE_BATCH:-200}
ENUM_HUBS=${ENUM_HUBS:-6}
ENUM_MAX=${ENUM_MAX:-4000}
CKPT=${CKPT:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/seh_proxy_stdlib/2026-07-02_14-59-53/train/checkpoints/last_gfn.pt}
CFG=${CFG:-configs/glue/fixed_reward_seh_proxy_stdlib.gin}
OUT_DIR=${OUT_DIR:-$SCRATCH/rgfn_runs/lsdflow/seh_stdlib_${SLURM_JOB_ID}}

# Keep all framework caches on $SCRATCH ($HOME is read-only on compute nodes).
export TORCH_HOME=$SCRATCH/.cache/torch
export HF_HOME=$SCRATCH/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$TORCH_HOME" "$HF_HOME" "$OUT_DIR"

module load cuda/11.8.0                         # dgl graphbolt CUDA-11.8 runtime (compute node)
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

echo "host=$(hostname)"; nvidia-smi -L
echo "N_TRAJ=$N_TRAJ ENUM_HUBS=$ENUM_HUBS ENUM_MAX=$ENUM_MAX"
echo "CKPT=$CKPT"
echo "OUT_DIR=$OUT_DIR"

# Fresh sample -> persist DAG (with stereo keys) BEFORE enumeration -> enumerate interior +
# top-terminating-flow hubs from the live stereo SMILES.
python -m validation.lsdflow.harness.run \
    --checkpoint "$CKPT" \
    --config-path "$CFG" \
    --n-trajectories "$N_TRAJ" \
    --sample-batch-size "$SAMPLE_BATCH" \
    --enumerate-top-hubs "$ENUM_HUBS" \
    --enumerate-max-children "$ENUM_MAX" \
    --run-id "seh_stdlib_${SLURM_JOB_ID}_enum" \
    --out-dir "$OUT_DIR"

echo "DONE lsdflow_seh_enum -> $OUT_DIR"
