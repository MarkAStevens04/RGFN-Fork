#!/bin/bash
#SBATCH --job-name=known_depth
#SBATCH --partition=compute
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
# HOW DEEP ARE MOLECULES PEOPLE ACTUALLY MAKE? Routes known molecular glues and known ChEMBL actives
# against ZINC at the SAME budget as every other depth measurement in this campaign.
#
# WHY IT MATTERS. The depth constraint reads as artificial -- "a method that wins in a case nobody
# cares about." This tests that directly: if real glues and real actives sit at depth >= 2-3 from a
# 17.4M-compound catalogue, then depth is not an imposed handicap, it is where the molecules the field
# actually makes already live, and the benchmark's one-step optimum is the artifact instead.
# Sample is 100 per set, fixed seed, plain AiZynth, iter=100/time=60/depth=6 -- identical to the
# catalogue ladder so the numbers sit side by side.
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export XDG_CACHE_HOME=$SCRATCH/.cache MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
D=$SCRATCH/rgfn_runs/lsdflow_sparrow/ladder/known
LAD=$SCRATCH/rgfn_runs/lsdflow_sparrow/ladder
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate aizynth
RC=0
for S in ${SETS:-glues_crbn_gspt1 glues_ddb1_cdk12 actives_seh actives_clpp actives_drd2}; do
  [ -s "$D/$S.smi" ] || { echo "skip $S (no input)"; continue; }
  echo ""; echo "########## $S ($(wc -l < "$D/$S.smi") molecules) ##########"
  python experiments/lsd_hubs/campaign/aiz_stock_ladder.py \
      --smi "$D/$S.smi" --out "$D/${S}_zinc.jsonl" --config "$LAD/config_ladder.yml" \
      --stocks zinc --nproc 24 || RC=1
done
echo "KNOWN-DEPTH rc=$RC"; exit "$RC"
