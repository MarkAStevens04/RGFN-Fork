#!/bin/bash
#SBATCH --job-name=maiz_discover
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# STANDALONE MultiAiZ route DISCOVERY over a fixed candidate pool -> a reusable routes artifact.
#
# WHY THIS EXISTS (the plan-once/price-many split). `validation/lsdflow/eval/multiaiz.py` runs
# MultiAiZ inside `score()` on every call (subprocess.run is unconditional; the `out_path.exists()`
# check on the next line is a failure guard, not a cache), and it keys its snapshot dir by CALL
# COUNTER rather than pool content — so every re-price re-pays the ~14 h discovery. This script
# performs discovery ONCE for a pool and persists `multiaiz_routes.json`, so SPARROW parameters
# (objective, budget, selection mode, tau) can be varied afterwards for free.
#
# CACHE KEY IS THE POOL, NOT THE MOLECULE. MultiAiZ is set-based: it runs AiZynth over the whole
# target set for n_iters cycles, appending discovered intermediates to stock so targets converge on
# shared chemistry. Routes therefore depend on pool COMPOSITION — the artifact is valid only for the
# exact pool that produced it, and a per-molecule cache would be unsound. Change the pool => re-run.
#
# Pools come from build_s3gfn_pools.py (nested top-N by reward). Running several N gives the
# route-planning COST-SCALING curve — the axis where the baseline pays and a reaction-grounded
# generator pays nothing (reported, not matched; see the three-axis accounting).
#
# STOCK = plain ZINC by design (decision 2026-08-04): S3-GFN is ZINC-native, so ZINC is the
# catalogue a chemist would actually stock. Re-running with the ZINC∪418-block merged stock (the
# Logs/047 stock) is a planned EXTENSION — it removes a catalogue-access asymmetry vs our native
# routes, and is expected to make the baseline stronger.
#
# Usage:  POOL_DIR=<dir with pool.smi> sbatch -p compute --time=<wall> \
#           experiments/lsd_hubs/campaign/submit_multiaiz_discover.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

POOL_DIR=${POOL_DIR:?set POOL_DIR to a directory containing pool.smi}
CONFIG=${CONFIG:-data/models/aizynthfinder/config.yml}
STOCK=${STOCK:-zinc}
EXPANSION=${EXPANSION:-uspto}
FILTER=${FILTER:-uspto}
N_ITERS=${N_ITERS:-5}                 # MultiAiZ cycles (paper value)
MAX_ROUTES=${MAX_ROUTES:-0}           # candidate routes/target emitted (0 = all; SPARROW picks)
STRIP_STEREO=${STRIP_STEREO:-true}

POOL_SMI="$POOL_DIR/pool.smi"
OUT_JSON="$POOL_DIR/multiaiz_routes.json"
WORK="$POOL_DIR/multiaiz_run"
[ -s "$POOL_SMI" ] || { echo "FATAL: no pool.smi at $POOL_SMI" >&2; exit 1; }

# RESUMABLE: discovery is the expensive step; never redo it if the artifact is already there.
if [ -s "$OUT_JSON" ]; then
    echo "=== routes already present, SKIP discovery: $OUT_JSON ==="
    exit 0
fi

export PYTHONUNBUFFERED=1
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate aizynth                # MultiAiZ installs --no-deps into the aizynth env (Logs/043)

N=$(wc -l < "$POOL_SMI")
echo "host=$(hostname)  pool=$POOL_SMI  N=$N  stock=$STOCK  n_iters=$N_ITERS"
START=$(date +%s)

python validation/lsdflow/adapters/workers/multiaiz_worker.py \
    --pool-smi "$POOL_SMI" \
    --out "$OUT_JSON" \
    --config "$CONFIG" \
    --stock "$STOCK" \
    --expansion "$EXPANSION" \
    --filter "$FILTER" \
    --n-iters "$N_ITERS" \
    --work-dir "$WORK" \
    --strip-stereo "$STRIP_STEREO" \
    --max-routes-per-target "$MAX_ROUTES"
RC=$?
END=$(date +%s)
ELAPSED=$((END-START))

# Timing sidecar = the scaling-curve datapoint. Written even on failure so a timeout is visible
# as a datapoint ("N=500 exceeded the wall") rather than silently missing.
python - "$POOL_DIR" "$N" "$ELAPSED" "$RC" "$N_ITERS" "$STOCK" <<'PY'
import json, sys
d, n, el, rc, it, stock = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
json.dump({"n_targets": n, "discovery_seconds": el, "returncode": rc,
           "n_iters": it, "stock": stock, "seconds_per_target": (el / n if n else None)},
          open(f"{d}/discovery_timing.json", "w"), indent=2)
print(f"[timing] N={n} elapsed={el}s ({el/3600:.2f}h) rc={rc} -> {d}/discovery_timing.json")
PY

[ "$RC" -eq 0 ] || { echo "FATAL: multiaiz_worker exited $RC" >&2; exit "$RC"; }
echo "=== DONE N=$N in ${ELAPSED}s -> $OUT_JSON ==="
