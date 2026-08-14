#!/bin/bash
#SBATCH --job-name=dock_noise
#SBATCH --time=03:00:00
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
#
# How reproducible is one docking score? Dock the SAME molecules K times and measure the spread.
#
# WHY THIS IS WORTH GPU TIME. Every number in Logs/058 -- the 2.43-4.43x edges, the 300-mode
# libraries, the gate sweeps -- rests on a single docking call per molecule. While auditing
# scent_clpp we found that molecules docked in two different processes DISAGREE: of 11,482 children
# docked more than once, 9.6% differed, median 0.70 and p90 2.20 kcal/mol. QuickVina2-GPU's pose
# search is stochastic. That estimate is opportunistic though -- those duplicates are whatever
# happened to be reachable from two hubs, so the sample is not controlled and K is only 2.
#
# This measures it properly: a STRATIFIED probe set (half within 0.5 kcal/mol of the -8.0 bar, where
# a flip changes whether a molecule qualifies as a mode; half spread across the range, to test
# whether noise depends on score) docked K times.
#
# ONE PROCESS PER REPLICATE, deliberately. DockingBridgeReward and CachedProxyBase both cache per
# canonical SMILES, so K repeats inside one process would return the SAME cached value K times and
# report zero noise -- a convincing, completely wrong answer. Separate processes are what make these
# independent draws.
#
# Consumes:  probe.smi (built from scent_clpp's enumeration; see the log entry)
# Produces:  rep<k>.csv per replicate -> analyze_replicates.py
set -uo pipefail

K=${K:-5}
ORACLE=${ORACLE:-docking_clpp}
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || exit 1
OUT=${OUT:-$SCRATCH/rgfn_runs/dock_noise}
PROBE=${PROBE:-$REPO/experiments/lsd_hubs/dock_noise/probe.smi}
mkdir -p "$OUT"

# Source the canonical helper -- do NOT hand-roll this. QuickVina2-GPU links against
# libboost_{program_options,system,filesystem}.so.1.83.0, which live in $SCRATCH/vina_gpu/boost/lib
# and NOT in the conda env; the helper puts that on LD_LIBRARY_PATH and exports GNINA. The first
# version of this script built LD_LIBRARY_PATH from the nvidia libs alone, so `ldd` on the binary had
# 3 "not found" entries, the docker could not start on ANY node, and the resulting all-nan output was
# misread as a degraded GPU (job 73370 -- balam006 was blameless). submit_docking_cell.sh sources the
# same helper, which is exactly why production docking works and this did not.
module load cuda/11.8.0 2>/dev/null || true
source ~/bin/rgfn-smoke-env.sh || { echo "ERROR: rgfn-smoke-env.sh failed"; exit 1; }
export TORCH_HOME=$SCRATCH/.cache/torch HF_HOME=$SCRATCH/.cache/huggingface

# Prove the toolchain resolves BEFORE burning a GPU allocation on it: 3 missing boost libs here is
# the difference between a 2-second exit and an hour of all-nan results that reads as a bad node.
QV_BIN=$(find "${SCRATCH}/vina_gpu" -maxdepth 3 -type f -name 'QuickVina2-GPU*' -perm -u+x 2>/dev/null | head -1)
if [ -n "$QV_BIN" ]; then
    MISSING=$(ldd "$QV_BIN" 2>&1 | grep -c 'not found')
    echo "[dock-noise] QuickVina2-GPU: $QV_BIN ($MISSING unresolved shared libs)"
    [ "$MISSING" = 0 ] || { echo "ERROR: $MISSING unresolved libs -- LD_LIBRARY_PATH is wrong, not the node"; exit 1; }
fi

echo "=== dock-noise: $(wc -l < "$PROBE") molecules x K=$K replicates, oracle=$ORACLE"
echo "    host=$(hostname)"; nvidia-smi -L 2>/dev/null | head -1

# Gate the node before spending hours: a wedged GPU returns all-failures that read as bad chemistry.
python scripts/preflight_dock.py --oracle "$ORACLE" || { echo "ERROR: preflight failed"; exit 42; }

for k in $(seq 1 "$K"); do
    echo "=== replicate $k/$K ==="
    python scripts/score_batch.py --oracle "$ORACLE" \
        --in "$PROBE" --out "$OUT/rep${k}.csv" \
        --oracle-arg docking_batch_size=200 \
        || { echo "ERROR: replicate $k failed"; exit 1; }
done

echo "=== DONE -> $OUT (rep1..rep${K}.csv) ==="
