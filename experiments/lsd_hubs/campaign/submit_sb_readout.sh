#!/bin/bash
#SBATCH --job-name=sb_readout
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1                  # not used by CBC; Balam REJECTS a job without it
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# SPARROW-Batching readout over pools whose MultiAiZ routes are ALREADY CACHED.
#
# WHY THIS EXISTS SEPARATELY FROM submit_competitor_routes.sh. That script is the full
# gate -> pool -> MultiAiZ -> select pipeline. Once routes are cached, the only remaining work is a
# CBC solve, and re-entering the routing script to reach it means re-running the gate and pool steps
# and depending on a routes artifact that must already be there. This runs just the solve, over a
# LIST of already-routed pools, so a night of cached-route readouts is one job rather than one per
# cell. The solve is pure CPU and never touches the GPU — but the #SBATCH line above still requests
# one, because Balam rejects at submit time any job that does not ask for 1 or 4 ("Jobs on Balam must
# use --gpus-per-node"). Do not remove it to "free" the GPU; the job will not submit.
#
# WHY IT IS NOT A LOGIN-NODE LOOP. Balam login enforces `ulimit -t 3600`, so a CBC process pinned at
# ~100% CPU is SIGXCPU'd at about an hour of wall-clock. A REINVENT sEH R=50 solve had already burned
# 16 min of CPU without returning, so these solves CANNOT run on login. (Learned by watching one
# start down that road on 2026-08-19.)
#
# BUDGETS default to the primary readout alone (100 reactions) plus 50 as a cheap lower anchor.
# Deliberately NOT the full 10-point ladder: the middle budgets are where the MILP wall lives
# (S3-GFN R=200/300 both hit a 1800 s cap) and they are not what the paper reports.
#
# ROUTE SOURCE. Defaults to the MultiAiZ artifact each pool dir carries. The route-BEARING entrants
# (SynFormer, TANGO) instead ship their own routes.jsonl, and TANGO ships two — one per route arm —
# so the file is named rather than assumed. Point ROUTES_NAME at a file INSIDE each pool dir, which
# keeps this a single loop over POOLS and makes "same pool, different route source" the natural way
# to run an arm comparison:
#   POOLS="tango_seh_seed42_N500 tango_seh_seed42_pruned_N218" \
#       ROUTE_SOURCE=external ROUTES_NAME=routes_arm2.jsonl TAG_SUFFIX=ARM2 \
#       sbatch experiments/lsd_hubs/campaign/submit_sb_readout.sh
#
# Usage — POOLS is a space-separated list of pool directory NAMES under multiaiz_pools/:
#   POOLS="reinvent_seh_seed42_N500 reinvent_seh_seed43_N500" \
#       sbatch experiments/lsd_hubs/campaign/submit_sb_readout.sh
#
# Each pool is independent: one that fails or runs long does not stop the others.

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

POOLS=${POOLS:?set POOLS to a list of pool directory names under multiaiz_pools/}
POOL_ROOT=$SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools
RES_ROOT=$SCRATCH/rgfn_runs/lsdflow_sparrow/results
GATE=${GATE:-7.0}
CUTOFF=${CUTOFF:-0.5}
BUDGETS=${BUDGETS:-50,100}
MAX_SECONDS=${MAX_SECONDS:-3600}
ROUTE_SOURCE=${ROUTE_SOURCE:-multiaiz}
ROUTES_NAME=${ROUTES_NAME:-multiaiz_routes.json}
TAG_SUFFIX=${TAG_SUFFIX:-SB}

# Compute nodes cannot write $HOME. Anything that caches there must be redirected or the job dies
# minutes in with a PermissionError (job 73617).
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton
export MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export HF_HOME=$SCRATCH/.cache/hf
export XDG_CACHE_HOME=$SCRATCH/.cache
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$HF_HOME"
export PYTHONUNBUFFERED=1
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

echo "host=$(hostname)  POOLS=$POOLS"
echo "budgets=$BUDGETS  max_seconds=$MAX_SECONDS  gate=$GATE  cutoff=$CUTOFF"
echo "route_source=$ROUTE_SOURCE  routes_name=$ROUTES_NAME  tag_suffix=$TAG_SUFFIX"

FAILED=""
for P in $POOLS; do
    D="$POOL_ROOT/$P"
    echo ""
    echo "############ $P ############"
    if [ ! -s "$D/$ROUTES_NAME" ]; then
        echo "SKIP $P — no cached routes at $D/$ROUTES_NAME" >&2
        FAILED="$FAILED $P(noroutes)"; continue
    fi
    # DRD2 pools use a different gate; infer it rather than making the caller remember.
    G="$GATE"; case "$P" in *_drd2_*) G=${DRD2_GATE:-0.5} ;; esac
    OUT="$RES_ROOT/${P}_${TAG_SUFFIX}_R${BUDGETS//,/_}"
    conda run --no-capture-output -n rgfn python \
        experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
        --routes "$D/$ROUTES_NAME" --pool "$D/pool_scores.csv" --route-source "$ROUTE_SOURCE" \
        --gate "$G" --cutoff "$CUTOFF" --budgets "$BUDGETS" --max-seconds "$MAX_SECONDS" \
        --out-dir "$OUT" --tag "${P}_${TAG_SUFFIX}"
    rc=$?
    if [ "$rc" -eq 0 ]; then echo "$P OK -> $OUT"; else echo "$P FAILED rc=$rc" >&2; FAILED="$FAILED $P"; fi
done

echo ""
if [ -n "$FAILED" ]; then echo "FAILED:$FAILED" >&2; exit 1; fi
echo "ALL POOLS OK: $POOLS"
