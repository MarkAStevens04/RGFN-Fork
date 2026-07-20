#!/bin/bash
#SBATCH --job-name=lsdflow_routes_timed
#SBATCH --time=04:00:00                          # ~4.7k mols x up to 60s AiZynth, 32-way parallel ~2h; 4h fits before the 2026-07-21 04:00 maintenance if started by ~midnight.
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1                         # Balam schedules by GPU count; AiZynth is CPU-only (GPU unused).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Timed route recovery — recover an AiZynth route (WITH per-molecule search_time) for an EXISTING
# SMILES list into a persistent RouteCache. Reuses a pre-dumped frontier .smi (no GPU/dgl dump step),
# so it is CPU-only. The per-molecule search_time (instrumented in synthesizability.py) makes the
# from-scratch route-finding COMPUTE attributable per method for the LSD-Flow compute frontier (T2.2)
# with no further re-run — instrument once, upfront.
#
# A fresh --cache (default: routecache_<stock>_<expansion>_timed.json) is REQUIRED to actually
# re-search: recovery here is a direct batch CLI over the whole .smi, but we fold into a fresh cache
# so it doesn't clobber the untimed one the from-scratch T2.1 headline already used.
#
# Submit:  sbatch experiments/lsd_hubs/campaign/submit_route_recovery.sh
# Override: SMI=<...>.smi CACHE=<...>.json TIME_LIMIT=90 STOCK=zinc EXPANSION=uspto sbatch ...

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

OUT=${OUT:-$SCRATCH/rgfn_runs/lsdflow_sparrow/scent_seh}
SMI=${SMI:-$OUT/frontier_smiles.smi}
STOCK=${STOCK:-zinc}
EXPANSION=${EXPANSION:-uspto}
FILTER=${FILTER:-uspto}
AICONFIG=${AICONFIG:-data/models/aizynthfinder/config.yml}
TIME_LIMIT=${TIME_LIMIT:-60}
CACHE=${CACHE:-$OUT/routecache_${STOCK}_${EXPANSION}_timed.json}
ROUTES_JSONL=${ROUTES_JSONL:-$OUT/frontier_routes_timed.jsonl}
NPROC=${NPROC:-${SLURM_CPUS_ON_NODE:-$(nproc)}}

export PYTHONUNBUFFERED=1
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

echo "host=$(hostname)  nproc=$NPROC  time_limit=${TIME_LIMIT}s  chemistry=${STOCK}/${EXPANSION}"
echo "SMI=$SMI  CACHE=$CACHE"
if [ ! -s "$SMI" ]; then echo "FATAL: no SMILES at $SMI (run the dump first)" >&2; exit 1; fi
echo "recovering $(wc -l < "$SMI") molecules"

# --- recover routes WITH per-molecule search_time (aizynth env, CPU) ---
conda activate aizynth
python validation/harness/synthesizability.py \
    --recover-routes "$SMI" --routes-out "$ROUTES_JSONL" \
    --config "$AICONFIG" --stock "$STOCK" --expansion "$EXPANSION" --filter "$FILTER" \
    --time-limit "$TIME_LIMIT" --nproc "$NPROC"
if [ ! -s "$ROUTES_JSONL" ]; then echo "FATAL: no routes produced" >&2; exit 1; fi

# --- fold into the persistent (timed) RouteCache (route_recovery/network are dgl-free -> aizynth env) ---
python - "$ROUTES_JSONL" "$CACHE" <<'PY'
import json, sys
from validation.lsdflow.eval.route_recovery import RouteCache
from validation.lsdflow.eval.network import canonical
jsonl, cache_path = sys.argv[1], sys.argv[2]
cache = RouteCache(cache_path)
n = solved = timed = 0
for line in open(jsonl):
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    c = canonical(rec.get("smiles"))
    if c is None:
        continue
    cache.put(c, bool(rec.get("solved")), rec.get("route"), rec.get("search_time"))
    n += 1; solved += int(bool(rec.get("solved"))); timed += int(rec.get("search_time") is not None)
cache.save()
tot = cache.total_search_time([canonical(json.loads(l)["smiles"]) for l in open(jsonl) if l.strip()])
print(f"[routes] folded {n} ({solved} solved, {timed} timed, total_search={tot:.0f}s) -> {cache_path}")
PY
conda deactivate
echo "[routes] done -> $CACHE (with per-molecule search_time for the T2.2 compute frontier)."
