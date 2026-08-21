#!/bin/bash
#SBATCH --job-name=resample_check
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Does re-sampling a cell REPRODUCE its existing sample? Answer before spending the re-enumeration.
#
# WHY THIS QUESTION IS WORTH A JOB. RxnFlow and RGFN cells have an EMPTY routes.json -- the sample
# stage never materialised routes (rxnflow_worker.py:282 writes `{}`), so neither competitor arm can
# price them. The fix is a re-sample, not a re-train, because the checkpoints are frozen. But a
# re-sample is only cheap if it lands on the SAME molecules: the hubs are the parents of the top-K
# candidates, so different molecules mean different hubs, which invalidates every enumeration built
# on them. Enumeration is the expensive stage, not sampling -- measured ~19 GPU-h per RGFN surrogate
# cell and ~104 GPU-h per RGFN docking cell, against ~1-2 h to sample one.
#
# So: ~2 GPU-h here decides whether the RxnFlow+RGFN route recovery costs ~65 GPU-h (hubs reproduce,
# existing enumerations stand) or ~500+ (they do not).
#
# IT WRITES TO A SCRATCH DIR, NEVER THE CELL. submit_cell.sh resolves SAMPLE_DIR from the manifest via
# `eval`, which clobbers any env override, and the manifest is NOT seed-aware
# (SCRATCH_ROOT / tag, no seed) -- so running its sample stage would OVERWRITE a finished sample in
# the seed-42 tree. That is the failure that once wiped a completed cell. This script calls the
# worker directly with --out-dir pointing elsewhere, so the live artifacts are read-only here.
#
# PYTHONHASHSEED IS LOAD-BEARING (2026-08-21): --seed alone gave an RGFN sample 377 vs 387 routes
# sharing 8 keys, because per-process dict/set iteration order feeds action-space construction and an
# identical RNG draw over a differently-ordered list picks a different action. With
# PYTHONHASHSEED=0 two runs were 730/730 byte-identical. The ORIGINAL samples almost certainly ran
# WITHOUT it, so reproduction is exactly what is in question -- do not assume it either way.
#
# Usage:  GEN=rxnflow TGT=seh sbatch -p compute --time=04:00:00 \
#           experiments/lsd_hubs/matrix16/submit_resample_check.sh
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || exit 1

GEN=${GEN:?set GEN=rxnflow|rgfn}
TGT=${TGT:?set TGT=seh|drd2|clpp}
N_TRAJ=${N_TRAJ:-30000}
SAMPLE_BATCH=${SAMPLE_BATCH:-200}
OUT_ROOT=${OUT_ROOT:-/scratch/markymoo/rgfn_runs/lsdflow_resample_check}

CONDA_SH=/home/markymoo/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"; conda activate base
SPEC="$(python experiments/lsd_hubs/matrix16/manifest.py --emit "$GEN" "$TGT")" || exit 1
eval "$SPEC"
LIVE_SAMPLE="$SAMPLE_DIR"                      # capture BEFORE we redirect; this stays read-only
OUT="$OUT_ROOT/${CELL_TAG}"                    # where the CHECK writes
mkdir -p "$OUT"

export WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0                        # see header -- load-bearing, not hygiene
export TORCH_HOME=$SCRATCH/.cache/torch HF_HOME=$SCRATCH/.cache/huggingface
export MPLCONFIGDIR=$SCRATCH/.cache/matplotlib TRITON_CACHE_DIR=$SCRATCH/.cache/triton
mkdir -p "$TORCH_HOME" "$HF_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"

module load cuda/11.8.0 2>/dev/null || true
conda activate "$CONDA_ENV"
[ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ] || { echo "FATAL: env is '$CONDA_DEFAULT_ENV', need '$CONDA_ENV'" >&2; exit 1; }

echo "host=$(hostname)  cell=$CELL_TAG  seed=$SEED  PYTHONHASHSEED=$PYTHONHASHSEED"
echo "  live (read-only): $LIVE_SAMPLE"
echo "  check writes to : $OUT"
[ -s "$LIVE_SAMPLE/records.csv" ] || { echo "FATAL: no existing sample to compare against" >&2; exit 1; }

python "$WORKER" --mode sample \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" ${GUIDANCE:+--guidance "$GUIDANCE"} \
    --reward-name "$REWARD_NAME" --n-trajectories "$N_TRAJ" --batch-size "$SAMPLE_BATCH" \
    --device auto --run-dir "$OUT/run" --out-dir "$OUT" --seed "$SEED" \
    || { echo "ERROR: sample failed"; exit 1; }

echo "=== COMPARISON ==="
python - "$LIVE_SAMPLE" "$OUT" <<'PY'
import csv, json, sys
from pathlib import Path
live, new = Path(sys.argv[1]), Path(sys.argv[2])

def mols(p):
    with open(p / "records.csv", newline="") as fh:
        return [r["child_key"] for r in csv.DictReader(fh)]

a, b = mols(live), mols(new)
sa, sb = set(a), set(b)
print(f"  records       live={len(a):,}  new={len(b):,}")
print(f"  distinct      live={len(sa):,}  new={len(sb):,}")
print(f"  set overlap   {len(sa & sb):,}  ({len(sa & sb) / max(len(sa), 1):.1%} of live)")
print(f"  ORDER identical: {a == b}")
# The decisive test is the HUBS, since those are what enumerations are keyed on. hub_key is in the
# records, so the hub POOL can be compared without re-running pick_hubs.
def hubs(p):
    with open(p / "records.csv", newline="") as fh:
        return {r["hub_key"] for r in csv.DictReader(fh)}
ha, hb = hubs(live), hubs(new)
print(f"  hub pool      live={len(ha):,}  new={len(hb):,}  overlap={len(ha & hb):,} "
      f"({len(ha & hb) / max(len(ha), 1):.1%})")
nr = len(json.loads((new / 'routes.json').read_text())) if (new / 'routes.json').exists() else 0
print(f"  routes.json in the NEW sample: {nr:,} entries "
      f"({'POPULATED — the fix works' if nr else 'STILL EMPTY — re-sample alone is not enough'})")
verdict = ("REPRODUCES — existing enumerations remain valid" if a == b else
           "DIVERGES — enumerations built on the old hubs are invalidated")
print(f"  VERDICT: {verdict}")
PY
echo "=== DONE $CELL_TAG ==="
