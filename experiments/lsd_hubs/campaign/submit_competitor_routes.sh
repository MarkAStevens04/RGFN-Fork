#!/bin/bash
#SBATCH --job-name=comp_routes
#SBATCH --partition=compute
#SBATCH --time=07:00:00                    # MultiAiZ N=500 measured 3.7 h on REINVENT (26.4 s/target,
                                           # vs 16 s/target on S3-GFN); greedy is minutes. Give it room.
#SBATCH --gpus-per-node=1                  # AiZynth's expansion policy is a neural net
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# EVERYTHING DOWNSTREAM OF ONE ROUTE-LESS GENERATOR RUN — for ANY route-less entrant.
#
# Generalized from submit_s3gfn_replicate_routes.sh (Logs/061), which hard-coded S3-GFN. The protocol
# is identical for every route-less baseline (S3-GFN, REINVENT 4, Saturn), and it MUST be identical,
# or a cross-generator table would be comparing procedures instead of generators. So there is one
# script and the generator is an argument:
#
#   0. PRE-FLIGHT  mode_saturation.py — does this pool even contain the deliverable?  ** A GATE. **
#   1. POOL        build_s3gfn_pools.py, nested top-N by reward, DEDUP by SMILES (N=500)
#   2. ROUTES      submit_multiaiz_discover.sh, INVOKED AS A SUBROUTINE rather than copied, so the
#                  discovery parameters (stock=zinc, uspto expansion/filter, n_iters=5, all routes
#                  emitted, stereo stripped) cannot diverge between generators or seeds
#   3. FRONTIER    sparrow_select_frontier.py TWICE: SPARROW-Batching (it selects) and the
#                  diversity-aware greedy (it only prices) — the baseline's strongest configuration
#
# WHY STEP 0 IS A GATE AND NOT A REPORT. Step 2 costs ~2.25 h per pool and its cache key is the POOL,
# never the molecule (MultiAiZ is set-based: a route depends on what else was planned alongside it).
# Spending that on a pool that cannot reach 100 modes produces a frontier that stops early, which
# reads downstream as a cost result when it is a pool-size result. Catch it in seconds instead.
#
# THE CATALOGUE IS THE COMPETITOR'S OWN, ON PURPOSE. Stock is plain ZINC for every baseline here —
# the catalogue a chemist would actually hold for a ZINC-native SMILES generator — and it is far
# larger than our 418 blocks. That is the CHARITABLE setting, not a handicap: purchasable leaves are
# free, so a broad catalogue means the baseline needs less batching to be cheap. See
# submit_multiaiz_discover.sh for the same decision recorded at the discovery step.
#
# Usage:
#   GENERATOR=reinvent TARGET=seh SEED=42 RUN_DIR=<the training run dir> \
#       sbatch experiments/lsd_hubs/campaign/submit_competitor_routes.sh
#
# NOT for reaction-aware generators (SynFormer). Those carry native routes, skip MultiAiZ entirely,
# and price through `sparrow_select_frontier.py --route-source external`. That path is Phase 2 and
# is deliberately absent here rather than present and untested.

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

GENERATOR=${GENERATOR:?set GENERATOR (reinvent | saturn | s3gfn)}
TARGET=${TARGET:?set TARGET (seh | drd2)}
SEED=${SEED:-42}
RUN_DIR=${RUN_DIR:?set RUN_DIR to the generator run dir (contains fixed_reward/candidates)}
N=${N:-500}

# Per-target reward gate — value AND direction. These are the project's calibrated bars
# (docs/paper_planning/lsd-flow-iclr-evidence-handoff.md section 3); they are NOT free parameters.
case "$TARGET" in
    seh)  GATE=${GATE:-7.0} ;;
    drd2) GATE=${GATE:-0.5} ;;
    *)    echo "FATAL: unknown TARGET '$TARGET' (expected seh or drd2). A docking target needs the" >&2
          echo "       lower-is-better gate handling, which this script does not carry." >&2; exit 1 ;;
esac
CUTOFF=${CUTOFF:-0.5}
TARGET_MODES=${TARGET_MODES:-100}
BUDGETS=${BUDGETS:-50,100,150,200,300,400,500,600,800,1000}   # seed 42's SB ladder (Logs/061)
MODE_POINTS=${MODE_POINTS:-25,50,75,100,125,150}              # seed 42's greedy ladder

CANDS="$RUN_DIR/fixed_reward/candidates/candidates.csv"
POOL_ROOT=$SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools
TAG=${GENERATOR}_${TARGET}_seed${SEED}
POOL_DIR="$POOL_ROOT/${TAG}_N${N}"
# Results to $SCRATCH: $HOME is READ-ONLY on Balam compute nodes, and a job writing into the repo
# "COMPLETES" in seconds with exit 0 having done nothing (the Logs/059 failure).
RES_ROOT=$SCRATCH/rgfn_runs/lsdflow_sparrow/results

export PYTHONUNBUFFERED=1
# No `module load cuda` / LD_LIBRARY_PATH here: sparrow_select_frontier.py's import surface
# (validation.lsdflow.eval.{network,route_recovery} + metrics.diversity) does NOT pull glue->rgfn->dgl,
# so the rgfn env imports clean, and adding the module would leak CUDA paths into the aizynth child.
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

echo "host=$(hostname)  GENERATOR=$GENERATOR TARGET=$TARGET SEED=$SEED N=$N GATE=$GATE"
echo "RUN_DIR=$RUN_DIR"
[ -s "$CANDS" ] || { echo "FATAL: no candidates.csv at $CANDS" >&2; exit 1; }

# ---- 0. pre-flight gate -------------------------------------------------------------------------
echo "=== [0/3] mode-saturation pre-flight (gate) ==="
conda run --no-capture-output -n rgfn python experiments/lsd_hubs/campaign/mode_saturation.py \
    --candidates "$CANDS" --gate "$GATE" --cutoff "$CUTOFF" --target-modes "$TARGET_MODES" \
    --sizes "50,100,250,${N}" --tag "$TAG" --out-dir "$RES_ROOT/${TAG}_saturation" || {
        echo "" >&2
        echo "ABORT: $TAG cannot deliver $TARGET_MODES modes from $N candidates." >&2
        echo "  Not spending ~2.25 h of route discovery on it. This is a POOL-SIZE result and must" >&2
        echo "  be reported as one — see $RES_ROOT/${TAG}_saturation/summary.json." >&2
        exit 2; }

# ---- 1. pool ------------------------------------------------------------------------------------
echo "=== [1/3] pool ==="
conda run --no-capture-output -n rgfn python experiments/lsd_hubs/campaign/build_s3gfn_pools.py \
    --candidates "$CANDS" --out-root "$POOL_ROOT" --tag "$TAG" --sizes "$N" || exit 1
[ -s "$POOL_DIR/pool.smi" ] || { echo "FATAL: pool not built at $POOL_DIR" >&2; exit 1; }

# ---- 2. routes ----------------------------------------------------------------------------------
# Subroutine call, NOT a copy: same discovery parameters and same timing sidecar as every other
# entrant. Its conda activate stays in this child shell. Resumable — skips if the artifact exists.
echo "=== [2/3] MultiAiZ discovery ==="
POOL_DIR="$POOL_DIR" bash experiments/lsd_hubs/campaign/submit_multiaiz_discover.sh || exit 1
ROUTES="$POOL_DIR/multiaiz_routes.json"
[ -s "$ROUTES" ] || { echo "FATAL: no routes artifact at $ROUTES" >&2; exit 1; }

# ---- 3. frontiers -------------------------------------------------------------------------------
# BOTH selectors. Quoting only the one SPARROW loses on would invite the objection that SPARROW was
# judged on diversity, which it does not optimize (see sparrow_select_frontier.py).
# GREEDY FIRST, AND SB IS OPTIONAL. Both from measurement, not taste (job 73610):
#
#  * The GREEDY arm prices a FIXED set of N modes, so its MILPs are small and every solve so far has
#    come back `Optimal` in seconds. It is also the arm the conservative headline ratio is quoted
#    against, and the only one whose column schema is identical across all the S3-GFN seeds on disk.
#  * The SB arm solves a SELECTION MILP over the whole merged network (~8,500 route entries, ~19k
#    variables) and its solve time is wildly non-monotonic in the budget: on S3-GFN at
#    --max-seconds 1800, budgets 50/100/400/1000 solved in 6-57 s while 200 and 300 BOTH hit the
#    1800 s wall. Rows that hit the wall are lower bounds the frontier script refuses to certify.
#
# Job 73610 ran SB first, spent 1.3 h producing only capped rows, hit its walltime, and wrote NO csv
# at all — the frontier only writes after every budget point. Running greedy first means a walltime
# kill costs the arm we cannot use rather than the one we can.
echo "=== [3/3] frontiers (diversity-aware greedy FIRST, then optional SPARROW-Batching) ==="
RC=0
conda run --no-capture-output -n rgfn python \
    experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
    --routes "$ROUTES" --pool "$POOL_DIR/pool_scores.csv" --route-source multiaiz \
    --selection greedy --gate "$GATE" --cutoff "$CUTOFF" --mode-points "$MODE_POINTS" \
    --out-dir "$RES_ROOT/${TAG}_greedy_N${N}" \
    --tag "${TAG}_multiaiz_greedy" || RC=1

if [ "${RUN_SB:-0}" = "1" ]; then
    conda run --no-capture-output -n rgfn python \
        experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
        --routes "$ROUTES" --pool "$POOL_DIR/pool_scores.csv" --route-source multiaiz \
        --gate "$GATE" --cutoff "$CUTOFF" --budgets "$BUDGETS" \
        --max-seconds "${SB_MAX_SECONDS:-1800}" \
        --out-dir "$RES_ROOT/${TAG}_select_N${N}" \
        --tag "${TAG}_multiaiz_select_N${N}" || RC=1
else
    echo "  SB skipped (RUN_SB=0). Re-run with RUN_SB=1 once the routes artifact is cached —"
    echo "  discovery is skipped on a re-run, so SB costs only its own solve time."
fi

echo ""
echo "DONE $TAG rc=$RC"
echo "  saturation : $RES_ROOT/${TAG}_saturation/summary.json"
echo "  pool       : $POOL_DIR/{pool.smi,pool_scores.csv}"
echo "  routes     : $ROUTES  (+ discovery_timing.json)"
echo "  SB         : $RES_ROOT/${TAG}_select_N${N}/select_frontier.csv"
echo "  greedy     : $RES_ROOT/${TAG}_greedy_N${N}/greedy_frontier.csv"
exit "$RC"
