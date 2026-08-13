#!/bin/bash
#SBATCH --job-name=s3gfn_rep_routes
#SBATCH --partition=compute
#SBATCH --time=05:00:00                    # MultiAiZ N=500 measured 2.25 h at seed 42; frontiers are minutes
#SBATCH --gpus-per-node=1                  # AiZynth's expansion policy is a neural net
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# EVERYTHING DOWNSTREAM OF ONE S3-GFN TRAINING RUN, for the competitor REPLICATES (seeds 43/44).
#
# WHY THIS EXISTS. The competitor side of the S3-GFN head-to-head (Logs/056, figure F1) is the only
# arm in the benchmark with no error bar: ours is n=3, theirs was a single training run plus a single
# MultiAiZ planning run. Adding seeds means repeating THREE stages per seed, and the stages must be
# byte-identical in protocol to seed 42 or the "band" would mix procedures instead of measuring
# seed variance. This script hard-codes that protocol so it cannot drift between seeds:
#
#   1. POOL      build_s3gfn_pools.py, nested top-N by reward, DEDUP by SMILES  (N=500, as seed 42)
#   2. ROUTES    submit_multiaiz_discover.sh, INVOKED AS A SUBROUTINE rather than copied, so the
#                discovery parameters (stock=zinc, uspto expansion/filter, n_iters=5, all routes
#                emitted, stereo stripped) and the timing sidecar cannot diverge from seed 42's
#   3. FRONTIER  sparrow_select_frontier.py TWICE: SPARROW-Batching (it selects) and the
#                diversity-aware greedy (it only prices) -- the baseline's strongest configuration
#
# THE CACHE KEY IS THE POOL, NOT THE MOLECULE. MultiAiZ is set-based, so a route depends on what
# else was planned alongside it. Each seed therefore gets its OWN pool directory and its own routes
# artifact; discovery is skipped only if that seed's own artifact already exists.
#
# Budgets/mode-points below are seed 42's exact ladders (from its *_frontier_summary.json), so the
# replicate curves are read at the same x-positions rather than at whatever a default produced.
#
# Usage:  SEED=43 RUN_DIR=<the training run dir> sbatch experiments/lsd_hubs/campaign/submit_s3gfn_replicate_routes.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

SEED=${SEED:?set SEED (43 or 44)}
RUN_DIR=${RUN_DIR:?set RUN_DIR to the s3gfn training run dir (contains fixed_reward/candidates)}
N=${N:-500}
GATE=${GATE:-7.0}
CUTOFF=${CUTOFF:-0.5}
BUDGETS=${BUDGETS:-50,100,150,200,300,400,500,600,800,1000}   # seed 42's SB ladder
MODE_POINTS=${MODE_POINTS:-25,50,75,100,125,150}              # seed 42's greedy ladder

CANDS="$RUN_DIR/fixed_reward/candidates/candidates.csv"
POOL_ROOT=$SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools
TAG=s3gfn_seh_seed${SEED}
POOL_DIR="$POOL_ROOT/${TAG}_N${N}"
# Results go to $SCRATCH: $HOME is READ-ONLY on Balam compute nodes, and a job writing into the repo
# "COMPLETES" in seconds with exit 0 having done nothing (the Logs/059 failure).
RES_ROOT=$SCRATCH/rgfn_runs/lsdflow_sparrow/results

export PYTHONUNBUFFERED=1
# No `module load cuda` / LD_LIBRARY_PATH here, unlike submit_s3gfn_frontier.sh: checked that
# sparrow_select_frontier.py's import surface (validation.lsdflow.eval.{network,route_recovery} +
# metrics.diversity) does NOT pull glue->rgfn->dgl, so the rgfn env imports clean. Adding the module
# would also leak CUDA paths into the aizynth child in step 2.
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

echo "host=$(hostname)  SEED=$SEED  N=$N"
echo "RUN_DIR=$RUN_DIR"
[ -s "$CANDS" ] || { echo "FATAL: no candidates.csv at $CANDS" >&2; exit 1; }

# ---- 1. pool -----------------------------------------------------------------------------------
echo "=== [1/3] pool ==="
conda run --no-capture-output -n rgfn python experiments/lsd_hubs/campaign/build_s3gfn_pools.py \
    --candidates "$CANDS" --out-root "$POOL_ROOT" --tag "$TAG" --sizes "$N" || exit 1
[ -s "$POOL_DIR/pool.smi" ] || { echo "FATAL: pool not built at $POOL_DIR" >&2; exit 1; }

# ---- 2. routes ---------------------------------------------------------------------------------
# Subroutine call, NOT a copy: same discovery parameters and same timing sidecar as seed 42. Its
# conda activate stays in this child shell.
echo "=== [2/3] MultiAiZ discovery ==="
POOL_DIR="$POOL_DIR" bash experiments/lsd_hubs/campaign/submit_multiaiz_discover.sh || exit 1
ROUTES="$POOL_DIR/multiaiz_routes.json"
[ -s "$ROUTES" ] || { echo "FATAL: no routes artifact at $ROUTES" >&2; exit 1; }

# ---- 3. frontiers ------------------------------------------------------------------------------
# Both selectors, because quoting only the one SPARROW loses on would invite the objection that
# SPARROW was judged on diversity, which it does not optimize (see sparrow_select_frontier.py).
echo "=== [3/3] frontiers (SPARROW-Batching, then diversity-aware greedy) ==="
RC=0
conda run --no-capture-output -n rgfn python \
    experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
    --routes "$ROUTES" --pool "$POOL_DIR/pool_scores.csv" --route-source multiaiz \
    --gate "$GATE" --cutoff "$CUTOFF" --budgets "$BUDGETS" \
    --out-dir "$RES_ROOT/${TAG}_select_N${N}" \
    --tag "${TAG}_multiaiz_select_N${N}" || RC=1

conda run --no-capture-output -n rgfn python \
    experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
    --routes "$ROUTES" --pool "$POOL_DIR/pool_scores.csv" --route-source multiaiz \
    --selection greedy --gate "$GATE" --cutoff "$CUTOFF" --mode-points "$MODE_POINTS" \
    --out-dir "$RES_ROOT/${TAG}_greedy_N${N}" \
    --tag "${TAG}_multiaiz_greedy" || RC=1

echo ""
echo "DONE seed=$SEED rc=$RC"
echo "  pool   : $POOL_DIR/{pool.smi,pool_scores.csv}"
echo "  routes : $ROUTES  (+ discovery_timing.json)"
echo "  SB     : $RES_ROOT/${TAG}_select_N${N}/select_frontier.csv"
echo "  greedy : $RES_ROOT/${TAG}_greedy_N${N}/greedy_frontier.csv"
exit "$RC"
