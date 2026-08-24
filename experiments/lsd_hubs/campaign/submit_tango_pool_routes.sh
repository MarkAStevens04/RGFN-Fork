#!/bin/bash
#SBATCH --job-name=tango_pool_routes
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# ARM 2: retrosynthesise a TANGO cell's EMITTED POOL, so every priced molecule has a route.
#
# WHY THIS IS NEEDED AT ALL. TANGO's generator is a SMILES language model (Saturn's Mamba agent): it
# writes strings, it does not build molecules from reactions, so nothing about generation produces a
# route. Routes exist only for molecules syntheseus was asked to score DURING TRAINING. Our protocol
# emits the pool as a fresh sample from the trained final policy, and measured on tango_seh seed 42
# only 548 of those 2,000 molecules (27.4%, canonical) were ever searched. The rest have no route,
# not because the search failed but because nobody ran it.
#
# So this is not re-doing work: it asks the retrosynthesis question about molecules nobody asked it
# about. It uses TANGO's OWN planner and inventory, so the routes are the ones TANGO's reward was
# defined against -- self-consistent in a way MultiAiZ would not be.
#
# ARM 1, for contrast, prices only the 548 already-routed molecules. Cheaper and honest, but a much
# smaller pool. Both are run; see docs/RESEARCH_CONTEXT.md.
#
# Usage:
#   TARGET=seh SEED=42 sbatch experiments/lsd_hubs/campaign/submit_tango_pool_routes.sh

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
TARGET=${TARGET:?set TARGET (seh | drd2 | clpp)}
SEED=${SEED:-42}
RUN_DIR=$SCRATCH/rgfn_runs/experiments/fixed_reward/tango_${TARGET}/seed${SEED}
POOL_CSV="$RUN_DIR/fixed_reward/candidates/candidates.csv"
[ -s "$POOL_CSV" ] || { echo "FATAL: no candidates at $POOL_CSV" >&2; exit 1; }

STOCK=${STOCK:-/scratch/markymoo/tango/authors_stock/frag-reac-zinc-stock.smi}
MODEL=${MODEL:-MEGAN}                 # the authors' reaction model
TIME_LIMIT=${TIME_LIMIT:-180}         # the authors' per-molecule cap
NTOP=${NTOP:-5}                       # our measured deviation from their 1; see docs/PATCHES.md
WORK=$RUN_DIR/pool_syntheseus
OUT="$WORK/results"

# $HOME is read-only on compute nodes, and syntheseus resolves its checkpoint through this cache.
export SYNTHESEUS_CACHE_DIR=${SYNTHESEUS_CACHE_DIR:-/scratch/markymoo/tango/syntheseus_cache}
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export XDG_CACHE_HOME=$SCRATCH/.cache
# MEGAN's loader spawns per-core BLAS threads and segfaults against the 1024-proc rlimit on a
# 64-core node; it needs no parallel BLAS to run inference.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$WORK" "$SYNTHESEUS_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

echo "host=$(hostname)  tango_${TARGET} seed${SEED}  model=$MODEL ntop=$NTOP tl=${TIME_LIMIT}s"

# targets = the emitted pool, one SMILES per line
awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="smiles") c=i; next} {print $c}' "$POOL_CSV" > "$WORK/targets.smi"
# MAX_TARGETS truncates the pool for a smoke; unset means the whole thing.
if [ -n "${MAX_TARGETS:-}" ]; then
    head -"$MAX_TARGETS" "$WORK/targets.smi" > "$WORK/targets.trunc" && mv "$WORK/targets.trunc" "$WORK/targets.smi"
fi
echo "targets: $(wc -l < "$WORK/targets.smi")"

# route_convert reads pool ORDER from <run_dir>/fixed_reward/candidates/candidates.csv, so point
# $WORK at the real one rather than letting it fall back to sorted-by-SMILES.
ln -sfn "$RUN_DIR/fixed_reward" "$WORK/fixed_reward"

cat > "$WORK/config.yml" <<EOF
inventory_smiles_file: $STOCK
search_targets_file: $WORK/targets.smi
model_class: $MODEL
time_limit_s: $TIME_LIMIT
num_top_results: $NTOP
results_dir: $OUT
save_graph: false
EOF

# MEGAN writes a bare `logs/` into the CWD, so run from a scratch dir (it landed in the repo once).
mkdir -p "$WORK/cwd" && cd "$WORK/cwd"
conda run --no-capture-output -n syntheseus syntheseus search --config "$WORK/config.yml"
RC=$?
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
[ "$RC" -eq 0 ] || { echo "FATAL: syntheseus search exited $RC" >&2; exit "$RC"; }

# The search writes <MODEL>_<timestamp>/<i>/, which is the same shape route_convert already walks
# once the per-molecule dirs are reachable as output_*/mol_*/.
SRC=$(ls -d "$OUT"/${MODEL}_* 2>/dev/null | head -1)
[ -n "$SRC" ] || { echo "FATAL: no $MODEL output under $OUT" >&2; exit 1; }
mkdir -p "$WORK/syntheseus_results/output_0"
for d in "$SRC"/*/; do
    b=$(basename "$d"); case "$b" in ''|*[!0-9]*) continue ;; esac
    ln -sfn "$d" "$WORK/syntheseus_results/output_0/mol_$b"
done

conda run --no-capture-output -n syntheseus python \
    validation/generators/saturn/route_convert.py "$WORK" \
    "$RUN_DIR/fixed_reward/routes_pool.jsonl"
echo "DONE -> $RUN_DIR/fixed_reward/routes_pool.jsonl"
