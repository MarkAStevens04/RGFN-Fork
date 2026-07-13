#!/bin/bash
#SBATCH --job-name=lsdflow_scent_drd2
#SBATCH --time=12:00:00                        # 30k traj ~3-4h + enumeration on the frozen 2,018-frag library (heavy, ~10x RGFN's 418-lib). DAG is persisted BEFORE enumeration, so a timeout still leaves the sampled analysis + compositions.json.
#SBATCH --partition=compute                    # 1-GPU job -> regular (non-full-node) partition
#SBATCH --gpus-per-node=1
# Absolute $SCRATCH log paths: $HOME is read-only on compute nodes (Logs/012).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# LSD-Flow phase-3 (proposal §10.3): post-hoc hub extraction on the trained, cost-aware SCENT
# sEH model — the same sampled-flow vertical slice the RGFN anchor runs (Logs/025), now for the
# cost-aware peer. This is the plumbing for the RGFN-vs-SCENT hub-coincidence study (§8, the
# paper's spine): the SCENT DAG produced here is what gets compared to the RGFN DAG and to
# SCENT's own promoted intermediates (chosen_smiles / additional_fragments/fragments_4000.json).
#
# CROSS-ENV: the harness runs in the `rgfn` env (imports glue); the SCENT sampler + enumerator run
# in the `scent` env via a subprocess the SCENTAdapter spawns (SCENT's package is also named `rgfn`,
# so no co-import). The adapter loads the trained backward policy P_B from the guidance_models.pt
# sidecar beside the checkpoint (entry 024) so the recovered flow is cost-aware for free (§5).
#
# FROZEN FULL LIBRARY (faithful full SCENT, entry 027): the SCENTAdapter freezes the dynamic
# library to its final promoted-fragment snapshot (fragments_<N>.json beside the checkpoint) for
# BOTH sampling and enumeration, so the model runs with its full trained vocabulary (418 base +
# ~1,600 promoted), not the 418-base restriction a bare checkpoint build leaves it in.
#
# CHECKPOINT: use the recipe-logging re-run (experiments/fixed_reward/scent_drd2/submit_fixed_scent_seh_recipes.sh)
# whose fragments_<N>.json carries per-fragment synthesis ROUTES (exact nested cost + chemist
# routes). Any patched (sidecar-bearing) 5,000-iter sEH run works for the flow analysis itself;
# the recipe re-run is required only for the nested cost model. Pre-fix runs are flagged
# BACKWARD_POLICY_NOT_SAVED.txt and their P_B is irrecoverable.
#
# ENUMERATION (entry 027): with the frozen 2,018-fragment library a hub's true one-reaction
# neighborhood is large (~4k for a depth-0 fragment, ~10x the RGFN 418-lib case), so ENUM_MAX must
# be generous for exhaustive recovery (validated: depth-0 hub 3,975 paths, 2/2 sampled recovered
# at cap 12k). sEH reward is free to enumerate.
#
# GFN INFERENCE ONLY — no docking. Compute node (not login): a large sample + enumeration exceeds
# the login CPU-time cap (Logs/011, SIGXCPU) and wants uninterrupted wall time.
#
# Submit:  sbatch validation/lsdflow/submit_scent_seh.sh
# Override: CKPT=<recipe-rerun>/train/checkpoints/last_gfn.pt N_TRAJ=30000 ENUM_HUBS=8 ENUM_MAX=12000 sbatch ...
#
# After it finishes, copy the SMALL artifacts back into the repo from the login node:
#   cp $SCRATCH/rgfn_runs/lsdflow/<run>/{meta.json,report.json,acquisitions.csv,hub_summary.csv,enumeration.json} \
#      validation/lsdflow/results/scent_drd2_pilot/
# (leave the large records.csv / enumerated_records.csv / graph.gpickle on scratch).

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

# Tunables (env-overridable).
N_TRAJ=${N_TRAJ:-30000}
SAMPLE_BATCH=${SAMPLE_BATCH:-200}
MODE_SIM=${MODE_SIM:-0.7}
MODE_REWARD=${MODE_REWARD:-0.5}                 # DRD2 "active" bar (TDC 0-1 proxy; NOT the sEH 7.0). DRD2 saturates near 1.0 so this barely filters — Logs/028.
ENUM_HUBS=${ENUM_HUBS:-8}                        # enumerate children of this many selected hubs (0=off)
ENUM_MAX=${ENUM_MAX:-12000}                      # per-hub cap; generous for the frozen 2,018-frag library
# CKPT: 'auto' (default) discovers the newest scent_drd2 run's checkpoint — after the recipe re-run
# (run this as a --dependency=afterok:<rerun> job, so the newest run IS the recipe re-run). Set CKPT
# explicitly to pin a specific run.
CKPT=${CKPT:-auto}
if [ "$CKPT" = "auto" ]; then
    CKPT=$(ls -td "$SCRATCH"/rgfn_runs/experiments/fixed_reward/scent_drd2/*/train/checkpoints/last_gfn.pt 2>/dev/null | head -1)
    echo "[auto] discovered newest scent_drd2 checkpoint: ${CKPT:-<NONE FOUND>}"
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then echo "ERROR: no scent_drd2 checkpoint found; set CKPT=..."; exit 1; fi
CFG=${CFG:-validation/configs/scent_drd2_fixed.gin}
OUT_DIR=${OUT_DIR:-$SCRATCH/rgfn_runs/lsdflow/scent_drd2_${SLURM_JOB_ID}}

# Keep all framework caches on $SCRATCH ($HOME is read-only on compute nodes). WANDB offline so
# the scent-env worker (imports fixed_reward -> wandb) never blocks on the network.
export TORCH_HOME=$SCRATCH/.cache/torch
export HF_HOME=$SCRATCH/.cache/huggingface
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export PYTHONUNBUFFERED=1
mkdir -p "$TORCH_HOME" "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$OUT_DIR"

module load cuda/11.8.0                         # dgl graphbolt CUDA-11.8 runtime (compute node)
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn                             # the HARNESS env; the SCENT worker self-activates `scent`

echo "host=$(hostname)"; nvidia-smi -L
echo "N_TRAJ=$N_TRAJ MODE_SIM=$MODE_SIM MODE_REWARD=$MODE_REWARD ENUM_HUBS=$ENUM_HUBS ENUM_MAX=$ENUM_MAX"
echo "CKPT=$CKPT"
echo "OUT_DIR=$OUT_DIR"

python -m validation.lsdflow.harness.run \
    --model scent \
    --checkpoint "$CKPT" \
    --config-path "$CFG" \
    --reward-name drd2 \
    --n-trajectories "$N_TRAJ" \
    --sample-batch-size "$SAMPLE_BATCH" \
    --mode-similarity "$MODE_SIM" \
    --mode-reward-threshold "$MODE_REWARD" \
    --enumerate-top-hubs "$ENUM_HUBS" \
    --enumerate-max-children "$ENUM_MAX" \
    --run-id "scent_drd2_${SLURM_JOB_ID}" \
    --out-dir "$OUT_DIR"

echo "DONE lsdflow_scent_drd2 -> $OUT_DIR"
