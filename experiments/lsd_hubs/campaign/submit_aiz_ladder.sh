#!/bin/bash
#SBATCH --job-name=aiz_ladder
#SBATCH --partition=debug
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# CATALOGUE LADDER, one cell per job. See aiz_stock_ladder.py for the science.
#
# WHY debug AND WHY ONE CELL PER JOB. The compute queue is backlogged (19 pending as of 2026-09-04)
# and debug is empty, but debug caps at 2 h and its QoS allows exactly ONE job per user -- so
# `--dependency` chains are REJECTED AT SUBMIT and these must be submitted one at a time as each
# finishes. The work is sized to fit: ~100 molecules x 4 rungs at the production budget.
#
# --gpus-per-node=1 ON A CPU JOB IS DELIBERATE: Balam refuses CPU-only jobs at submit. Both existing
# AiZynth/MultiAiZ submit scripts do the same; this is not a copy-paste error.
#
# RESUMABLE: aiz_stock_ladder.py skips (stock, molecule) pairs already in the JSONL, so a walltime
# kill is repaired by resubmitting this script unchanged.
#
# Submit:  CELL=s3gfn_seh_seed42_stage2_pruned_N500 sbatch experiments/lsd_hubs/campaign/submit_aiz_ladder.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

# Compute nodes have a read-only $HOME and these libraries default their caches there.
export XDG_CACHE_HOME=$SCRATCH/.cache MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export HOME_CACHE_GUARD=1 PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

CELL=${CELL:?set CELL to a multiaiz_pools cell directory name}
LADDER=$SCRATCH/rgfn_runs/lsdflow_sparrow/ladder
SMI=${SMI:-$LADDER/pools/${CELL}.smi}
OUT=${OUT:-$LADDER/${CELL}_ladder.jsonl}
# Rungs MOST IMPORTANT FIRST — a walltime kill then leaves complete rates for the ones that matter.
STOCKS=${STOCKS:-zinc,rgfnlib,zincfrag,zinc_rand418}

[ -s "$SMI" ] || { echo "FATAL: no molecule list at $SMI" >&2; exit 1; }
echo "host=$(hostname)  cell=$CELL  n=$(wc -l < "$SMI")  rungs=$STOCKS"

source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate aizynth

python experiments/lsd_hubs/campaign/aiz_stock_ladder.py \
    --smi "$SMI" --out "$OUT" --config "$LADDER/config_ladder.yml" --stocks "$STOCKS"
RC=$?
echo "LADDER rc=$RC  out=$OUT"
exit "$RC"
