#!/bin/bash
#SBATCH --job-name=s3gfn_routes
#SBATCH --time=04:00:00                         # ~2000 pool molecules x up to 60s AiZynth each, parallel over the node's cores; generous headroom. Fits before the 2026-07-21 04:00 maintenance if the S3-GFN run finishes by ~00:00.
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1                        # Balam schedules by GPU count (balam-slurm memory); AiZynth is CPU-ONLY, the GPU is unused.
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# From-scratch route recovery for the S3-GFN entrant's pool (LSD-Flow T3.2 prerequisite).
#
# S3-GFN emits molecules with NO shared-route structure (has_route=0), so — exactly like a real lab
# faced with a SMILES-only generator — the whole pool must be routed post-hoc before it can be
# priced. This job routes the S3-GFN candidate pool ONCE (AiZynth, standard USPTO/ZINC — the SAME
# chemistry as the SCENT from-scratch headline, so the two curves overlay fairly) into a persistent
# RouteCache; the T3.2 frontier's SPARROW MILPs then read the cache with no AiZynth. The recovery
# records per-molecule search_time, so this cache serves BOTH pricing and the compute frontier.
#
# Simpler than submit_lsdflow_routes.sh (the reaction-GFN one): there is no hub/enum union to dump —
# the molecules ARE the candidate pool. synthesizability.py --recover-routes reads candidates.csv's
# 'smiles' column directly.
#
# Fire AFTER the S3-GFN run lands its pool (job 71007). Either submit with a dependency:
#     sbatch --dependency=afterok:71007 experiments/lsd_hubs/campaign/submit_s3gfn_routes.sh
# or run manually once you've confirmed <RUN_DIR>/fixed_reward/candidates/candidates.csv exists.
# Override: RUN_DIR=<s3gfn run dir> TIME_LIMIT=90 sbatch ...

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

RUN_DIR=${RUN_DIR:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/s3gfn_seh/71007}
CANDS=${CANDS:-$RUN_DIR/fixed_reward/candidates/candidates.csv}
AICONFIG=${AICONFIG:-data/models/aizynthfinder/config.yml}   # standard USPTO/ZINC (headline chemistry)
STOCK=${STOCK:-zinc}
EXPANSION=${EXPANSION:-uspto}
FILTER=${FILTER:-uspto}
TIME_LIMIT=${TIME_LIMIT:-60}                                  # per-molecule AiZynth search seconds
TAG=${TAG:-s3gfn_seh}
OUT_DIR=${OUT_DIR:-$SCRATCH/rgfn_runs/lsdflow_sparrow/${TAG}}
ROUTES_JSONL=${ROUTES_JSONL:-$OUT_DIR/pool_routes.jsonl}
CACHE=${CACHE:-$OUT_DIR/routecache_${STOCK}_${EXPANSION}.json}   # chemistry-specific persistent cache
NPROC=${NPROC:-${SLURM_CPUS_ON_NODE:-$(nproc)}}

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
mkdir -p "$OUT_DIR"
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

echo "host=$(hostname)  nproc=$NPROC  time_limit=${TIME_LIMIT}s  chemistry=${STOCK}/${EXPANSION}"
echo "CANDS=$CANDS"; echo "CACHE=$CACHE"

# HARD GUARD: the pool must exist (the S3-GFN run must have finished + ingested).
if [ ! -s "$CANDS" ]; then
    echo "[s3gfn-routes] FATAL: candidate pool not found ($CANDS). Run the S3-GFN job first." >&2
    exit 1
fi

# --- 1. Recover a route per molecule via AiZynth (aizynth env; CPU-only, dgl-free). ------
# --recover-routes reads the 'smiles' column of candidates.csv directly. Records search_time per
# molecule (serves the T2.2-style compute axis for S3-GFN too).
conda activate aizynth
python validation/harness/synthesizability.py \
    --recover-routes "$CANDS" --routes-out "$ROUTES_JSONL" \
    --config "$AICONFIG" --stock "$STOCK" --expansion "$EXPANSION" --filter "$FILTER" \
    --time-limit "$TIME_LIMIT" --nproc "$NPROC"
if [ ! -s "$ROUTES_JSONL" ]; then
    echo "[s3gfn-routes] FATAL: AiZynth produced no routes ($ROUTES_JSONL missing/empty) — aborting." >&2
    exit 1
fi

# --- 2. Fold the JSONL into the persistent RouteCache the SparrowEvaluator reads. ---------------
# eval modules (route_recovery/network) are RDKit+stdlib, dgl-free -> run in the aizynth env.
FOLD_PY="$(mktemp --suffix=.py)"   # temp file, NOT `python - <<HEREDOC` under conda run (stdin gotcha)
cat > "$FOLD_PY" <<'PY'
import json, sys, os
sys.path.insert(0, os.environ["REPO_ROOT"])  # temp script lives in /tmp -> put the repo on sys.path
from validation.lsdflow.eval.route_recovery import RouteCache
from validation.lsdflow.eval.network import canonical
jsonl, cache_path = sys.argv[1], sys.argv[2]
cache = RouteCache(cache_path)
n = solved = 0
for line in open(jsonl):
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    smi = rec.get("smiles")
    if not smi:
        continue
    canon = canonical(smi, strip_stereo=True)
    if not canon:
        continue
    route = rec.get("route")
    is_solved = bool(rec.get("solved")) and route is not None
    cache.put(canon, is_solved, route, rec.get("search_time"))
    n += 1
    solved += int(is_solved)
cache.save()
print(f"[s3gfn-routes] folded {n} routes into cache ({solved} solved, "
      f"{100.0*solved/max(n,1):.1f}%) -> {cache_path}")
PY
REPO_ROOT="$PWD" python "$FOLD_PY" "$ROUTES_JSONL" "$CACHE"
rm -f "$FOLD_PY"

echo ""
echo "DONE s3gfn_routes -> $CACHE"
echo "Next (T3.2): sweep_campaign.py --pool-csv $CANDS --evaluator sparrow --route-source from_scratch"
echo "             --sparrow-cache $CACHE  (best-candidate only; overlay on the SCENT headline axes)."
