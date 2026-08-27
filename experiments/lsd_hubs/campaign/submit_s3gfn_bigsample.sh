#!/bin/bash
#SBATCH --job-name=s3_big_pipe
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# END-TO-END for one S3-GFN cell on an ENLARGED pool: sample more from the frozen checkpoint ->
# ingest candidates -> MultiAiZ retrosynthesis -> greedy AND SPARROW at a fine mode ladder.
#
# WHY. A 65-molecule above-gate pool caps SPARROW at 65 modes, so beating it with hub-batching's 82
# proves nothing about selection. This gives the baseline the biggest pool its own trained policy can
# supply. Training budget is untouched -- the policy is frozen; the extra cost is EVAL-phase oracle
# calls, recorded in bigsample_meta.json and required alongside any number from these pools.
#
# Submit:  CELLS="s3gfn_seh/seed42 s3gfn_seh/seed44" sbatch run_bigsample_pipeline.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch
export XDG_CACHE_HOME=$SCRATCH/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$HF_HOME" "$TORCH_HOME"
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi -L
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

FINE="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,110,125,150"
RC=0
for CELL in $CELLS; do                       # CELL is like s3gfn_seh/seed43
  GENTGT=${CELL%%/*}; SEEDDIR=${CELL##*/}
  TGT=${GENTGT#s3gfn_}; SD=${SEEDDIR#seed}
  SRC="$SCRATCH/rgfn_runs/experiments/fixed_reward/$CELL"
  BIG="${SRC}_bigsample"
  echo ""; echo "################ $CELL ################"

  if [ ! -s "$BIG/fixed_reward/candidates/candidates.csv" ]; then
    conda run --no-capture-output -n s3gfn python \
      experiments/lsd_hubs/campaign/s3gfn_sample_more.py --run-dir "$SRC" \
      --target-above-gate "${TARGET_ABOVE:-500}" --gate "${GATE:-7.0}" \
      --max-samples "${MAX_SAMPLES:-40000}" --round-size "${ROUND:-4000}"
    rc=$?; [ "$rc" -ne 0 ] && { echo "SAMPLE FAILED $CELL rc=$rc" >&2; RC=1; continue; }
  else
    echo "  candidates already present at $BIG — skipping sampling"
  fi

  # Both pool variants, both selection arms, fine ladder. TAG_SUFFIX keeps these in their own
  # namespace so a _bigsample result can never be mistaken for the budget-faithful cell.
  for POOLV in naive pruned; do
    GENERATOR=s3gfn TARGET="$TGT" SEED="$SD" RUN_DIR="$BIG" POOL="$POOLV" \
      TAG_SUFFIX="_big" MODE_POINTS="$FINE" RUN_SB=1 \
      bash experiments/lsd_hubs/campaign/submit_competitor_routes.sh
    rc=$?; [ "$rc" -ne 0 ] && { echo "ROUTES FAILED $CELL/$POOLV rc=$rc" >&2; RC=1; }
  done
done
echo ""; echo "PIPELINE rc=$RC"
exit "$RC"
