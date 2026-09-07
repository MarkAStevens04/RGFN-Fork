#!/bin/bash
#SBATCH --job-name=zinc_axis
#SBATCH --partition=compute
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# THE CLEAN X-AXIS: hub-batching's deliverable spec restated in the COMPETITOR's units.
#
# WHY SLURM AND NOT THE LOGIN NODE. The driver is an hours-long loop of campaign runs and parallel
# retrosynthesis. Long login-node work is killed (a per-process CPU ceiling plus session teardown --
# an earlier `nohup` run of this died silently at 5,000 records with the log still showing a healthy
# ETA). Everything that runs for more than a few minutes belongs here.
#
# ALL FIVE RUNGS IN ONE JOB, ASCENDING. The routed-depth cache is shared across rungs and grows
# monotonically, so depth 1 (which rejects only unroutables) converges fast and seeds the cache for
# the stricter rungs, each of which then re-routes far less. Ascending also means a walltime kill
# leaves COMPLETE rungs at the loose end rather than partial ones everywhere -- and depth 5 is the
# rung most likely to churn, so it is last.
#
# BOTH UNROUTABLE POLICIES, drop FIRST. `drop` (the conservative reading) does the routing; `deep`
# re-interprets the SAME cache -- an unroutable molecule is already judged, so `deep` needs almost no
# new searches. Reporting the pair is the point: dropping unroutables removes exactly the molecules
# furthest from the competitor's catalogue, so neither reading alone is honest.
#
# --gpus-per-node=1 on a CPU job is deliberate: Balam refuses CPU-only jobs at submit.
#
# Submit:  SEED=42 sbatch experiments/lsd_hubs/campaign/submit_zinc_axis.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export XDG_CACHE_HOME=$SCRATCH/.cache MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # 32 workers x 64 BLAS threads
export PYTHONUNBUFFERED=1                                          # exhausts the thread limit
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# MAX_ROUNDS IS NOT COSMETIC. An unconverged `drop` rung still holds molecules whose ZINC depth is
# unknown, and unjudged molecules PASS the filter -- so a capped `drop` count is an UPPER BOUND on the
# conservative reading, which makes the band open at the bottom rather than merely wide. Job 75749 hit
# the 40-round cap on all five `drop` rungs (2/3/10/17/42 unjudged). Raising the cap on a warm cache
# costs only the marginal routing.
SEED=${SEED:?set SEED (42|43|44)}
NPROC=${NPROC:-32}
DEPTHS=${DEPTHS:-"1 2 3 4 5"}
POLICIES=${POLICIES:-"drop deep"}
SUF=""; [ "$SEED" != "42" ] && SUF="_seed${SEED}"
TARGET=${TARGET:-seh}
S=$SCRATCH/rgfn_runs/lsdflow/matrix16${SUF}/scent_${TARGET}
SNAP=$(ls $SCRATCH/rgfn_runs/experiments/fixed_reward/scent_${TARGET}*/seed${SEED}/additional_fragments/fragments_*.json 2>/dev/null | sort -t_ -k2 -n | tail -1)
LAD=$SCRATCH/rgfn_runs/lsdflow_sparrow/ladder
CACHE=$LAD/zinc_depth_scent_${TARGET}_s${SEED}.jsonl
# seed 42's cache was started on the login node before this script existed; keep using it.
[ "$TARGET" = "seh" ] && [ "$SEED" = "42" ] && [ -s "$LAD/zinc_depth_scent_seh.jsonl" ] && CACHE=$LAD/zinc_depth_scent_seh.jsonl

read -r GATE HIB <<EOF
$(source ~/bin/rgfn-smoke-env.sh >/dev/null 2>&1; python - "$TARGET" <<'PYG'
import sys; sys.path.insert(0,"experiments/lsd_hubs/matrix16")
from targets import TARGETS
t=TARGETS[sys.argv[1]]; print(t.mode_reward_threshold, "true" if t.higher_is_better else "false")
PYG
)
EOF
[ -n "${GATE:-}" ] || { echo "FATAL: no gate for $TARGET" >&2; exit 1; }
echo "host=$(hostname) target=$TARGET seed=$SEED gate=$GATE hib=$HIB nproc=$NPROC depths='$DEPTHS' policies='$POLICIES'"
echo "snapshot=$SNAP"
echo "cache=$CACHE ($( [ -s "$CACHE" ] && wc -l < "$CACHE" || echo 0 ) already routed)"
[ -s "$S/enum/enum_children.json" ] || { echo "FATAL: no enumeration at $S" >&2; exit 1; }

# `rgfn-smoke-env.sh`, NOT a bare `conda activate rgfn`. run_campaign.py imports `glue`, which pulls
# in dgl, and dgl needs the torch-bundled CUDA libs on LD_LIBRARY_PATH -- without them the import dies
# with "Cannot load Graphbolt C++ library". Job 75746 failed all ten configurations in 52 s this way.
# The login-node runs never showed it because the helper was always sourced there by hand.
# The aizynth routing child inherits this LD_LIBRARY_PATH and is unaffected (proven: 1,255 molecules
# routed successfully under exactly this environment on the login node).
source ~/bin/rgfn-smoke-env.sh
RC=0
for POL in $POLICIES; do
  for D in $DEPTHS; do
    echo ""; echo "################ seed $SEED  min-zinc-depth $D  unroutable=$POL ################"
    python experiments/lsd_hubs/campaign/zinc_axis_driver.py \
      --analysis-dir "$S/sample" --enum-children "$S/enum/enum_children.json" \
      ${SNAP:+--snapshot "$SNAP"} \
      --cache "$CACHE" --config "$LAD/config_ladder.yml" \
      --gate "$GATE" --higher-is-better "$HIB" --similarity 0.5 \
      --min-zinc-depth "$D" --unroutable "$POL" \
      --out-dir "$SCRATCH/rgfn_runs/lsdflow_sparrow/results/zinc_axis_${TARGET}_seed${SEED}_d${D}_${POL}" \
      --nproc "$NPROC" --max-rounds "${MAX_ROUNDS:-40}" \
      --aizynth-python /home/markymoo/miniconda3/envs/aizynth/bin/python
    rc=$?; [ "$rc" -ne 0 ] && { echo "FAILED d=$D pol=$POL rc=$rc" >&2; RC=1; }
  done
done
echo ""; echo "ZINC-AXIS rc=$RC  cache now $(wc -l < "$CACHE") routed"
exit "$RC"
