#!/bin/bash
#SBATCH --job-name=sparrow_headline_timed
#SBATCH --time=01:00:00                          # SPARROW MILP sweep over cached routes (~15 min at 13:37); 1h buffer. Fits before the 2026-07-21 04:00 maintenance.
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# THE SAME HEADLINE SWEEP AS submit_sparrow_headline_timed.sh, BUT AT --prebuild-k 0.
#
# WHY IT HAD TO BE RE-RUN. The primary readout is now a fixed budget of 100 REACTIONS (CLAUDE.md,
# "THE BENCHMARK'S PRIMARY READOUT"), and pre-select-K is the one knob whose sign FLIPS with the
# budget. K charges reactions UP FRONT to pre-synthesize the top-K fragments; at a 300-mode target
# that amortizes and wins, but at a 100-reaction budget K=100 spends essentially the whole budget
# before a single mode is delivered. Measured on the committed sweeps at tau=0.5:
#
#     free_frag K=0    -> 82 modes at 100 reactions   (count-once)
#     free_frag K=100  -> 27 modes at 100 reactions   (this sweep's predecessor, SPARROW)
#     naive            -> 33-34
#     best_candidate   -> 26-29
#
# So the published SPARROW headline was run at the WRONG operating point for the new axis. This is
# Logs/056's "pre-select-K break-even" finding reappearing where it now matters most.
#
# STILL from_scratch, NOT native — and that is a known gap, not a choice. `sweep_campaign.py` calls
# `evaluate_ordering(...)` WITHOUT its `routes=` kwarg, so `LibrarySet.routes` is always None and
# `--route-source native` would hand SPARROW an empty native dict. Feeding it is not a one-liner: the
# native routes are SHALLOW and must be recipe-expanded from the fragment snapshot first, or SPARROW
# BUYS what count-once BUILDS (the Logs/049 62.7% error). That is why Logs/056's native 131 came from
# reconcile_t15.py rather than from a sweep. Until that is wired, the runnable SPARROW numbers on the
# reaction axis are from-scratch — the regime where we LOSE — so read them as a floor, not a headline.
#
# T2.2 — from-scratch SPARROW HEADLINE + compute frontier for the reaction-GFN entrant (SCENT sEH),
# now with the TIMED route cache (job 70976) so the compute frontier carries the REAL from-scratch
# AiZynth route-finding time (the earlier 13:37 run used the untimed cache -> route_finding_s = 0).
#
# INDEPENDENT of the SCENT native enum re-run: this uses the CAMPAIGN 200-hub enum
# (campaign_enum_seh_70363, done) + the timed route cache (done). CPU-only to consume (cached routes
# -> no AiZynth calls; SparrowEvaluator shells to the sparrow env for the CBC MILP). Runs the driver
# under the rgfn env (imports glue -> dgl needs the CUDA libs). $HOME is read-only on compute, so
# results go to $SCRATCH via --out-dir; sync back to the repo from a login node afterward.
#
# Submit:  sbatch experiments/lsd_hubs/campaign/submit_sparrow_headline_timed.sh
# After:   cp -r $OUT_DIR/scent_seh_sparrow_headline <repo>/experiments/lsd_hubs/campaign/results/

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

ANALYSIS=${ANALYSIS:-/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189}
ENUM=${ENUM:-/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json}
SNAPSHOT=${SNAPSHOT:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json}
CACHE=${CACHE:-/scratch/markymoo/rgfn_runs/lsdflow_sparrow/scent_seh/routecache_zinc_uspto_timed.json}
OUT_DIR=${OUT_DIR:-/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results}
TAG=${TAG:-scent_seh_sparrow_headline_k0}
THR=${THR:-7.0}

export TORCH_HOME=$SCRATCH/.cache/torch HF_HOME=$SCRATCH/.cache/huggingface PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR" "$TORCH_HOME"

# rgfn env + dgl/graphbolt CUDA libs (sweep imports glue -> rgfn -> dgl). Mirrors rgfn-smoke-env.sh.
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/rgfn/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}"

echo "host=$(hostname)"; nvidia-smi -L
echo "ANALYSIS=$ANALYSIS"; echo "ENUM=$ENUM"; echo "CACHE=$CACHE"; echo "OUT_DIR=$OUT_DIR/$TAG"

python experiments/lsd_hubs/campaign/sweep_campaign.py \
    --analysis-dir "$ANALYSIS" \
    --enum-children "$ENUM" \
    --snapshot "$SNAPSHOT" \
    --reward-threshold "$THR" \
    --tag "$TAG" \
    --out-dir "$OUT_DIR" \
    --evaluator sparrow \
    --route-source from_scratch \
    --snapshot-schedule geometric \
    --child-policy free_frag \
    --prebuild-k 0 \
    --n-hubs 200 \
    --sparrow-cache "$CACHE"

echo ""
echo "DONE sparrow_headline_timed -> $OUT_DIR/$TAG"
echo "  compute frontier (real route-finding time) in sweep_summary.json['compute_time']"
echo "Sync back:  cp -r $OUT_DIR/$TAG \$REPO/experiments/lsd_hubs/campaign/results/"
