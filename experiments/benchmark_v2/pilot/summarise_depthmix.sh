#!/bin/bash
# Roll the three depth arms up into one table. Thin wrapper over depth_mix.py so the join logic
# lives in exactly one place.
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || exit 1
PY=/home/markymoo/miniconda3/envs/rgfn/bin/python
H=${H:-/scratch/markymoo/rgfn_runs/lsdflow/matrix16/scent_seh/enum/hubs.csv}
OUT=${OUT:-/scratch/markymoo/rgfn_runs/v2_pilot/depthmix}
RES=experiments/benchmark_v2/pilot/results
mkdir -p "$RES"
for arm in base d2 d3; do
    "$PY" experiments/benchmark_v2/pilot/depth_mix.py \
        --run-dir "$OUT/$arm" --hubs "$H" --budget 100 \
        --json-out "$RES/depthmix_$arm.json" > /dev/null 2>&1
done
"$PY" experiments/benchmark_v2/pilot/depth_mix_table.py --results "$RES"
