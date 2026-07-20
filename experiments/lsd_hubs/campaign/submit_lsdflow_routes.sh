#!/bin/bash
#SBATCH --job-name=lsdflow_routes
#SBATCH --time=12:00:00                        # ~4.5k novel molecules x up to 60s AiZynth each, parallel over the node's cores; generous headroom
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1                       # Balam allocates via GPU count; AiZynth is CPU-ONLY (the GPU is unused) — see note
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# From-scratch route recovery for the LSD-Flow SPARROW evaluator (T1.4/T2.1 prerequisite).
#
# The from-scratch SPARROW headline re-routes (AiZynth, standard USPTO/ZINC) every molecule the
# frontier prices. This job routes that whole set ONCE into a persistent route cache
# (validation/lsdflow/eval/route_recovery.py); the fast per-snapshot MILPs then read the cache with
# no AiZynth. Step 1 dumps the exact molecule union (both strategies x all cutoffs x the headline hub
# configs); step 2 routes it in the `aizynth` env.
#
# CPU-ONLY: AiZynth needs no GPU. We request 1 GPU only because Balam schedules by GPU count (per the
# balam-slurm memory: never request cpu/mem); the GPU goes unused. --nproc is taken from the node's
# actual core count at runtime. To grab a whole node's cores instead, use compute_full_node.
#
# Submit:  sbatch experiments/lsd_hubs/campaign/submit_lsdflow_routes.sh
# Override: EN=<200-hub enum_children.json> PREBUILD_K=100 TIME_LIMIT=90 sbatch ...

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

ANALYSIS=${ANALYSIS:-/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189}
EN=${EN:-/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json}  # 200-hub (widest coverage)
SNAP=${SNAP:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json}
THRESHOLD=${THRESHOLD:-7.0}
PREBUILD_K=${PREBUILD_K:-100}
AICONFIG=${AICONFIG:-data/models/aizynthfinder/config.yml}   # standard USPTO/ZINC (headline chemistry)
STOCK=${STOCK:-zinc}
EXPANSION=${EXPANSION:-uspto}
FILTER=${FILTER:-uspto}
TIME_LIMIT=${TIME_LIMIT:-60}                                  # per-molecule AiZynth search seconds
TAG=${TAG:-scent_seh}
OUT_DIR=${OUT_DIR:-$SCRATCH/rgfn_runs/lsdflow_sparrow/${TAG}}
SMI=${SMI:-$OUT_DIR/frontier_smiles.smi}
CACHE=${CACHE:-$OUT_DIR/routecache_${STOCK}_${EXPANSION}.json}   # chemistry-specific persistent cache
NPROC=${NPROC:-${SLURM_CPUS_ON_NODE:-$(nproc)}}

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
mkdir -p "$OUT_DIR"
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

echo "host=$(hostname)  nproc=$NPROC  time_limit=${TIME_LIMIT}s  chemistry=${STOCK}/${EXPANSION}"
echo "ANALYSIS=$ANALYSIS  EN=$EN  PREBUILD_K=$PREBUILD_K"
echo "SMI=$SMI  CACHE=$CACHE"

# --- 1. Dump the exact molecule union to route (pure CPU; runs in the rgfn env). ------
# The dump imports glue.samplers.lsdflow, whose import chain pulls dgl (graphbolt C++), so the rgfn
# env needs cuda/11.8.0 + the torch-bundled CUDA libs on LD_LIBRARY_PATH even though this step uses
# NO GPU (balam-dgl-cuda-module). Mirrors submit_scent_seh_enum.sh's proven env setup.
module load cuda/11.8.0
conda activate rgfn
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/rgfn/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"
python experiments/lsd_hubs/campaign/dump_frontier_smiles.py \
    --analysis-dir "$ANALYSIS" --enum-children "$EN" --snapshot "$SNAP" \
    --reward-threshold "$THRESHOLD" --configs best_candidate naive preselectk \
    --prebuild-k "$PREBUILD_K" --out "$SMI"
conda deactivate
# HARD GUARD: no .smi -> the dump failed; stop now (don't sail past into a no-op AiZynth run).
if [ ! -s "$SMI" ]; then
    echo "[routes] FATAL: dump step produced no SMILES ($SMI missing/empty) — aborting." >&2
    exit 1
fi
echo "[routes] dumped $(wc -l < "$SMI") molecules to route."

# --- 2. Recover a route per molecule via AiZynth (aizynth env). ------
# Called directly (not through route_recovery.py's conda-run bridge) since we are ALREADY on a compute
# node; aizynth is CPU-only + dgl-free, so no cuda module / LD_LIBRARY_PATH needed here.
conda activate aizynth
ROUTES_JSONL="$OUT_DIR/frontier_routes.jsonl"
python validation/harness/synthesizability.py \
    --recover-routes "$SMI" --routes-out "$ROUTES_JSONL" \
    --config "$AICONFIG" --stock "$STOCK" --expansion "$EXPANSION" --filter "$FILTER" \
    --time-limit "$TIME_LIMIT" --nproc "$NPROC"
if [ ! -s "$ROUTES_JSONL" ]; then
    echo "[routes] FATAL: AiZynth produced no routes ($ROUTES_JSONL missing/empty) — aborting." >&2
    exit 1
fi

# --- 3. Fold the JSONL into the persistent RouteCache the SparrowEvaluator reads. ---------------
# The eval modules (route_recovery/network) are RDKit+stdlib, dgl-free -> run in the aizynth env
# (has rdkit); no need to re-enter the rgfn/dgl env.
python - "$ROUTES_JSONL" "$CACHE" <<'PY'
import json, sys
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
    c = canonical(rec.get("smiles"))
    if c is None:
        continue
    cache.put(c, bool(rec.get("solved")), rec.get("route"))
    n += 1; solved += int(bool(rec.get("solved")))
cache.save()
print(f"[routes] folded {n} routes into cache ({solved} solved, {solved/max(n,1):.1%}) -> {cache_path}")
PY
conda deactivate
echo "[routes] done. SparrowEvaluator will read $CACHE (set --sparrow-cache to it)."
