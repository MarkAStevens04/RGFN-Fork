#!/bin/bash
#SBATCH --job-name=reprice
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=1                  # CBC is CPU-only, but Balam refuses a CPU-only request
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# RE-PRICE CELLS FROM CACHED ROUTES, CHANGING ONE SOLVER KNOB AND NOTHING ELSE.
#
# Two arms, two defects, one mechanism:
#
#   ARM=sb      re-solve a TIME-CAPPED SPARROW-Batching row at the primary readout (R=100)
#   ARM=greedy  re-price a COARSE-LADDER greedy arm on a dense mode ladder
#
# ---------------------------------------------------------------------------------------------
# WHY ARM=sb.
#
# WHY THIS EXISTS. A `TimeLimit` row is a LOWER BOUND on the competitor, not an optimum, and
# CLAUDE.md requires it be read as solver-truncated rather than as a property of the generator.
# Across the campaign 14 of 87 R=100 rows are capped, and they are not distributed evenly — on sEH
# they fall entirely on REINVENT:
#
#     pool     S3-GFN (ours)                REINVENT 4 (competitor)
#     naive    56 / 38 / 49   Optimal       61 / 47 / 83   Optimal      <- competitor WINS, certified
#     pruned   98 / 100 / 100 Optimal       80 / 70 / 92   TimeLimit    <- we "win", but vs. a
#                                                                          truncated competitor
#
# So the direction of the sEH result FLIPS between the two pools, and the pool we win on is exactly
# the pool where CBC ran out of time on the competitor. Worse, our pruned rows sit at the arithmetic
# ceiling — used=100 reactions for 98/100/100 modes is 1 reaction per mode, which cannot be beaten —
# so the entire question is whether REINVENT also reaches that ceiling when the solver is allowed to
# finish. Publishing 100-vs-70 without answering that is quoting a number we know is understated,
# and any reviewer with CBC and an afternoon can overturn it.
#
# ---------------------------------------------------------------------------------------------
# WHY ARM=greedy. The greedy arm is priced on the MODE axis (ask for M modes, learn the cost) while
# the primary readout is the REACTION axis (spend R=100, learn the modes). Converting means taking
# the largest rung whose used_rxns <= 100, so the answer is quantized to the ladder spacing and the
# UNSPENT budget at that rung is the size of the understatement. On the default 25/50/75/... ladder,
# 59 of 96 cells are coarse, the median one is quoted from a rung leaving 33 of 100 reactions
# unspent, and 10 cells have NO readable number at all because even 25 modes costs more than 100.
# That is the CERTIFIED arm -- the one this script's sibling comment calls "the arm the conservative
# headline ratio is quoted against" -- understated everywhere and unevenly, which distorts
# cross-generator comparison on top of the absolute numbers.
#
# The dense ladder is a strict SUPERSET of the coarse one (25/50/75/100/125/150 all appear in it),
# so re-pricing in place loses no information, and sparrow_select_frontier.py writes its CSV only
# after every rung completes -- a killed run leaves the old file intact rather than a truncated one.
# Densify ALL cells in a comparison or none: a 3-seed band mixing a 25-rung cell with a 2-rung cell
# reads as a seed effect, which is exactly how a shared-tree metric change once cost us a result.
#
# ---------------------------------------------------------------------------------------------
# WHY IT CALLS THE FRONTIER DIRECTLY instead of submit_competitor_routes{,_chain}.sh:
#   * The routes are already cached. The chain re-runs the pre-flight gate, the pool build and
#     MultiAiZ discovery (~3.7 h/cell on REINVENT) to reach a solve that costs minutes.
#   * The chain's out-dir is derived from TAG, so restricting --budgets to 100 would OVERWRITE the
#     existing ten-budget select_frontier.csv with a one-row file and destroy the other nine points
#     that the two-knob surfaces read. The sb arm writes to its own _longsolve dir instead.
#   * THE CHAIN REBUILDS THE POOL, and build_s3gfn_pools.py has since gained the NaN gate fix. A
#     rebuilt pool is no longer the pool the cached routes were planned for (20 of 178 pools held
#     NaN-scored molecules), so re-pricing through the chain would silently change the pool AND the
#     ladder at once and leave a cell whose two arms were solved on different molecule sets. Reading
#     the cached pool_scores.csv directly is what makes this a re-price rather than a re-run.
#
# THE TWO SOLVER LEVERS, AND WHY BOTH. sparrow_select_frontier.py's own --gap-rel help records the
# earlier gap experiment: relaxing CBC's gap did NOT achieve convergence on its own, but left the
# objective stable to 0.5% across 1e-7..1e-2, so it COMPOSES with a longer wall-clock rather than
# replacing it. Hence gap 1e-2 AND a 6 h cap, not one or the other. SPARROW hardcodes 1e-7 — seven
# digits of optimality on a problem whose answer we report in whole molecules.
#
# READING THE RESULT. Three outcomes, all publishable, none of them "nothing happened":
#   * Optimal, ~100 modes  -> the pruned-pool sEH win DISAPPEARS; both hit the 1 rxn/mode ceiling and
#                             the honest claim is a tie there, with the naive pool (competitor wins)
#                             the only certified separation. This is the outcome that saves us.
#   * Optimal, still < ~98 -> the win is REAL and now certified, and the headline gets stronger.
#   * TimeLimit again      -> still a lower bound; report it as solver-truncated and quote the naive
#                             pool, which is fully certified on both sides.
#
# Usage:  GENERATOR=reinvent TARGET=seh SEED=42 POOL=pruned ARM=sb     sbatch submit_reprice_cached.sh
#         GENERATOR=reinvent TARGET=seh SEED=42 POOL=pruned ARM=greedy sbatch submit_reprice_cached.sh
#         CELLS="reinvent:seh:42 s3gfn:seh:42" POOL=pruned ARM=greedy  sbatch submit_reprice_cached.sh
set -u

ARM=${ARM:-sb}
case "$ARM" in sb|greedy) ;; *) echo "FATAL: ARM must be sb or greedy, got '$ARM'" >&2; exit 1 ;; esac
POOL=${POOL:-pruned}
N=${N:-500}
BUDGETS=${BUDGETS:-100}                    # the PRIMARY readout only; the other nine points already
                                           # solved fine and live in the untouched _select_N500 dir
SB_MAX_SECONDS=${SB_MAX_SECONDS:-21600}    # 6 h vs the 1800 s that these rows hit
GAP_REL=${GAP_REL:-0.01}
CUTOFF=${CUTOFF:-0.5}
# `-` not `:-`, deliberately: a route-carrying generator has NO Stage 2 and so takes an EMPTY suffix,
# and `${TAG_SUFFIX:-_stage2}` would silently rewrite that empty string back to `_stage2` and then
# look for a pool that never existed. Caught by the guard below on the first SynFormer smoke.
TAG_SUFFIX=${TAG_SUFFIX-_stage2}
# The dense ladder. Superset of the 25/50/75/100/125/150 default, with the low rungs that let a
# pool-limited cell report a number at all -- FragGFN cells routing 10-23 molecules had EMPTY
# frontiers purely because the first rung was 25.
MODE_POINTS=${MODE_POINTS:-2,5,10,15,20,25,30,40,50,60,75,100,125,150}

# CELLS batches "gen:target:seed[:tagsuffix]" into one allocation, because a greedy re-price costs
# ~20 s/cell against cached routes and one job per cell would be all queue and no work. A single
# cell may still be given as GENERATOR/TARGET/SEED.
CELLS=${CELLS:-}
if [ -z "$CELLS" ]; then
    CELLS="${GENERATOR:?set GENERATOR or CELLS}:${TARGET:?set TARGET}:${SEED:?set SEED}"
fi

# REPO_DIR, not a hard-coded cd: three launchers in this campaign hard-coded the shared checkout and
# one of them silently ran stale code for a whole cell (docs/REFACTOR_LOG.md, "a wrapper can
# snapshot a stale callee"). Default to the tree this script actually lives in.
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "$REPO_DIR" || { echo "FATAL: cannot cd to REPO_DIR=$REPO_DIR" >&2; exit 1; }

export PYTHONUNBUFFERED=1
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

POOL_ROOT=$SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools
RES_ROOT=$SCRATCH/rgfn_runs/lsdflow_sparrow/results
# ROUTE-CARRYING generators (SynFormer) keep their routes with the generator's own output, not in the
# pool directory, and the frontier reads them as `external`. They are re-priced HERE rather than
# through submit_native_routes.sh for the same reason the chain is bypassed: that launcher rebuilds
# the pool, and build_s3gfn_pools.py has since gained the NaN gate, so a rebuild would change the
# ladder AND the molecule set in one step. TAG_SUFFIX must be "" for these -- they never had Stage 2.
ROUTE_SOURCE=${ROUTE_SOURCE:-multiaiz}
FR_ROOT=${FR_ROOT:-$SCRATCH/rgfn_runs/experiments/fixed_reward}   # same default as submit_native_routes.sh

# Gate resolved from matrix16/targets.py, never a default here — same reasoning, and the same code,
# as submit_competitor_routes.sh. A stale bar would silently re-price against a different question.
# Memoised: the conda spin-up costs ~5 s and a batch re-uses two or three targets.
declare -A GATE_OF DIR_OF
resolve_gate() {
    local tgt=$1
    [ -n "${GATE_OF[$tgt]:-}" ] && return 0
    local out
    out=$(conda run --no-capture-output -n rgfn python \
          experiments/lsd_hubs/campaign/_resolve_gate.py "$tgt" 2>/dev/null | tail -1)
    GATE_OF[$tgt]=$(echo "$out" | awk '{print $2}')
    DIR_OF[$tgt]=$(echo "$out" | awk '{print $3}')
    [ -n "${GATE_OF[$tgt]}" ] && [ -n "${DIR_OF[$tgt]}" ]
}

echo "host=$(hostname)  ARM=$ARM  POOL=$POOL  N=$N"
[ "$ARM" = sb ] && echo "  solver : --budgets $BUDGETS --max-seconds $SB_MAX_SECONDS --gap-rel $GAP_REL" \
                || echo "  ladder : $MODE_POINTS"
echo ""

RC=0
NDONE=0
NSKIP=0
for CELL in $CELLS; do
    IFS=: read -r GEN TGT SD SFX <<<"$CELL"
    SFX=${SFX:-$TAG_SUFFIX}
    if ! resolve_gate "$TGT"; then
        echo "SKIP $CELL — could not resolve a gate for TARGET='$TGT' from matrix16/targets.py" >&2
        NSKIP=$((NSKIP + 1)); RC=1; continue
    fi
    GATE=${GATE_OF[$TGT]}
    [ "${DIR_OF[$TGT]}" = lower ] && HIB=false || HIB=true

    TAG=${GEN}_${TGT}_seed${SD}${SFX}$([ "$POOL" = pruned ] && echo "_pruned" || echo "")
    POOL_DIR="$POOL_ROOT/${TAG}_N${N}"
    # A pool-limited cell writes _N<actual>, not _N$N — resolve it rather than dying on a path that
    # was never going to exist (the same fix submit_competitor_routes.sh carries).
    if [ ! -s "$POOL_DIR/pool_scores.csv" ]; then
        ACTUAL=$(ls -d "$POOL_ROOT/${TAG}"_N[0-9]* 2>/dev/null | head -1)
        [ -n "$ACTUAL" ] && POOL_DIR="$ACTUAL"
    fi
    if [ "$ROUTE_SOURCE" = multiaiz ]; then
        ROUTES="$POOL_DIR/multiaiz_routes.json"
    else
        ROUTES="$FR_ROOT/${GEN}_${TGT}/seed${SD}/fixed_reward/candidates/routes.jsonl"
    fi

    # REFUSE rather than rediscover. If the cache is missing, the cheap re-price this script promises
    # would silently become a multi-hour MultiAiZ run under a walltime that is not sized for it.
    if [ ! -s "$ROUTES" ] || [ ! -s "$POOL_DIR/pool_scores.csv" ]; then
        echo "SKIP $TAG — no cached routes/pool ($ROUTES / $POOL_DIR) — run the chain first, do not rediscover here" >&2
        NSKIP=$((NSKIP + 1)); RC=1; continue
    fi

    echo "=== $TAG  (gate $GATE, ${DIR_OF[$TGT]} is better) ==="
    if [ "$ARM" = sb ]; then
        OUT="$RES_ROOT/${TAG}_select_N${N}_longsolve"
        PREV="$RES_ROOT/${TAG}_select_N${N}/select_frontier.csv"
        [ -s "$PREV" ] && awk -F, -v b="$BUDGETS" '$2==b{printf "  was: R=%s modes_kept=%s status=%s capped=%s solve=%ss\n",$2,$6,$17,$18,$19}' "$PREV"
        conda run --no-capture-output -n rgfn python \
            experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
            --routes "$ROUTES" --pool "$POOL_DIR/pool_scores.csv" --route-source "$ROUTE_SOURCE" \
            --gate "$GATE" --higher-is-better "$HIB" --cutoff "$CUTOFF" --budgets "$BUDGETS" \
            --max-seconds "$SB_MAX_SECONDS" --gap-rel "$GAP_REL" \
            --out-dir "$OUT" --tag "${TAG}_longsolve" || RC=1
        [ -s "$OUT/select_frontier.csv" ] && awk -F, 'NR>1{printf "  now: R=%s modes_kept=%s cost_kept=%s status=%s capped=%s solve=%ss\n",$2,$6,$7,$17,$18,$19}' "$OUT/select_frontier.csv"
    else
        # In place: the dense ladder is a strict superset of the coarse one, and the frontier writes
        # its CSV only after every rung, so a killed run leaves the old file rather than a stub.
        OUT="$RES_ROOT/${TAG}_greedy_N${N}"
        [ -s "$OUT/greedy_frontier.csv" ] && awk -F, 'NR>1 && $2<=100{m=$1;u=$2} END{if(m)printf "  was: %s modes @ %s rxn (coarse, %s rxn of 100 unspent)\n",m,u,100-u; else print "  was: no rung fits R=100"}' "$OUT/greedy_frontier.csv"
        conda run --no-capture-output -n rgfn python \
            experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
            --routes "$ROUTES" --pool "$POOL_DIR/pool_scores.csv" --route-source "$ROUTE_SOURCE" \
            --selection greedy --gate "$GATE" --higher-is-better "$HIB" \
            --cutoff "$CUTOFF" --mode-points "$MODE_POINTS" \
            --out-dir "$OUT" --tag "${TAG}_multiaiz_greedy" || RC=1
        [ -s "$OUT/greedy_frontier.csv" ] && awk -F, 'NR>1 && $2<=100{m=$1;u=$2} END{if(m)printf "  now: %s modes @ %s rxn (dense, %s rxn of 100 unspent)\n",m,u,100-u; else print "  now: no rung fits R=100"}' "$OUT/greedy_frontier.csv"
    fi
    NDONE=$((NDONE + 1))
    echo ""
done

echo "DONE arm=$ARM pool=$POOL  repriced=$NDONE skipped=$NSKIP rc=$RC"
exit $RC
