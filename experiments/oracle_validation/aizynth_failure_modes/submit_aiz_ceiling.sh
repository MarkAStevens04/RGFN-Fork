#!/bin/bash
#SBATCH --job-name=aiz_ceiling
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
# 047 HIGH-BUDGET CEILING: full 4,749-molecule SCENT sEH mode union routed with ZINC+SMALL blocks at
# 10x search budget (iter=1000/time=300/depth=9). Upper bound to sit beside the production number
# (48.7%->73.8%, Logs/047 job 71436). Same script as the headline, env-parametrized budget +
# incremental JSONL (wall-clock safe). ZINC hdf5 untouched (union via config_rgfnlib_flat.yml).
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export PYTHONUNBUFFERED=1
export AIZ_CONFIG=/scratch/markymoo/rgfn_runs/lsdflow_sparrow/config_rgfnlib_flat.yml
export AIZ_IT=1000 AIZ_TL=300 AIZ_MT=9
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate aizynth
DIR=/scratch/markymoo/rgfn_runs/lsdflow_sparrow/scent_rawpool
python experiments/oracle_validation/aizynth_failure_modes/aiz_fullpool.py \
    "$DIR/full_union.smi" "$DIR/control300.smi" "$DIR/aiz_fullpool_ceiling.json" "${SLURM_CPUS_ON_NODE:-32}"
