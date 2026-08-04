#!/bin/bash
#SBATCH --job-name=m16_dock
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
# $HOME is read-only on compute nodes (Logs/012) -> absolute $SCRATCH log paths.
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
#
# Enumerate one DOCKING matrix cell: every one-reaction child of a hub slice, scored by the real
# GPU docking oracle instead of a fast surrogate.
#
# Shape of the problem (measured, Logs/036 Part A at batch 200 / 1 QuickVina2-GPU process):
# ClpP 0.398 s/mol, 6TD3 0.632 s/mol -- and a second concurrent QV2 process buys nothing because
# the GPU is already saturated. At ~200k children per cell that is 22-69 GPU-hours SERIAL, so this
# script slices hubs across jobs. Slicing buys wall-clock, not total compute.
#
# Three things it does that a surrogate cell does not need:
#
#  1. ONE persistent docking server per job. Constructing the oracle (receptors + gnina +
#     QuickVina2-GPU context) costs ~35-44 s; the per-batch `score_batch.py` fallback would pay
#     that thousands of times. The worker's DockingBridgeReward finds it via RGFN_DOCK_SOCKET, and
#     _docking.require_socket() makes a MISSING socket a hard error rather than a slow success.
#  2. A live probe THROUGH the socket before the real work. A node can pass the OpenCL health
#     probe and still pose nothing (balam009, Logs/013/014) -- and a wedged docker does not crash,
#     it returns a full set of failures that reads as bad chemistry. Probing through the server
#     (rather than running scripts/preflight_dock.py, which would construct a SECOND oracle and
#     double VRAM) validates the exact path the run depends on, at no extra construction cost.
#  3. Two processes share one GPU: the server's docker and the generator's own model. The bridge
#     already calls torch.cuda.empty_cache() before each dock so the model's allocator does not
#     starve the docking subprocess (Logs/014).
#
# Usage:  sbatch [--time=...] submit_docking_cell.sh <generator> <target> [SLICE_IDX] [N_SLICES]
#         SLICE_IDX is 1-based; omit both for all hubs in one job.
# Knobs:  ENUM_MAX (4000, keep it to stay comparable with the surrogate cells) DEVICE (auto)
#         HUBS_FILE (default: the cell's own enum/hubs.csv)
set -uo pipefail

GEN=${1:?usage: submit_docking_cell.sh <generator> <target> [slice_idx] [n_slices]}
TGT=${2:?usage: submit_docking_cell.sh <generator> <target> [slice_idx] [n_slices]}
SLICE_IDX=${3:-1}
N_SLICES=${4:-1}
ENUM_MAX=${ENUM_MAX:-4000}
DEVICE=${DEVICE:-auto}
STAGE=${STAGE:-all}
N_TRAJ=${N_TRAJ:-30000}
N_HUBS=${N_HUBS:-200}
TOPK=${TOPK:-1000}          # caps the distinct-hub count; must be >= a few x N_HUBS (Logs/031)
SAMPLE_BATCH=${SAMPLE_BATCH:-200}

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "ERROR: cannot cd to repo root '$REPO'"; exit 1; }
[ -f experiments/lsd_hubs/matrix16/manifest.py ] || {
    echo "ERROR: not at repo root (no matrix16/manifest.py under $REPO)"; exit 1; }

# Conda BEFORE the manifest emit: a bare SLURM shell has no `python` on PATH.
source /home/markymoo/miniconda3/etc/profile.d/conda.sh || { echo "ERROR: no conda.sh"; exit 1; }
conda activate base
eval "$(python experiments/lsd_hubs/matrix16/manifest.py --emit "$GEN" "$TGT")" || {
    echo "ERROR: manifest emit failed for $GEN/$TGT"; exit 1; }
[ "$STATUS" = "ready" ] || { echo "ERROR: cell $CELL_TAG is '$STATUS', not ready"; exit 1; }
[ "$REWARD_TYPE" = "docking" ] || {
    echo "ERROR: $CELL_TAG is a '$REWARD_TYPE' cell — use submit_cell.sh for surrogate targets"; exit 1; }

# Oracle NAME from targets.py (declared beside the gate); oracle ARGS from the cell's own training
# config, which is the only record of what the checkpoint was actually trained against.
read -r ORACLE ORACLE_ARGS < <(python - "$TGT" "$CONFIG" <<'PY'
import sys, shlex
sys.path.insert(0, "experiments/lsd_hubs/matrix16")
from targets import get_target
tgt, cfg_path = sys.argv[1], sys.argv[2]
oracle = get_target(tgt).oracle_name
args = {}
try:
    from omegaconf import OmegaConf
    args = dict((OmegaConf.load(cfg_path).get("reward", {}) or {}).get("oracle_args", {}) or {})
except Exception:
    pass
print(oracle, " ".join(f"--oracle-arg {k}={v}" for k, v in args.items()))
PY
) || { echo "ERROR: could not resolve the oracle for $TGT"; exit 1; }
[ -n "$ORACLE" ] || { echo "ERROR: empty oracle for $TGT"; exit 1; }

# Force the measured-optimal QuickVina2 batch (Logs/036 Part A: batch 200 / one process is 3.3x
# faster per molecule than batch 25, with batch-INVARIANT scores). DockingClpPOracle already
# inherits 200, but the 6TD3 differential oracle still defaults to 25 -- without this override the
# 6TD3 cells would run ~3x slower for no reason. Skipped if the cell's config already pins it.
DOCK_BATCH=${DOCK_BATCH:-200}
case "$ORACLE_ARGS" in
    *docking_batch_size*) : ;;
    *) ORACLE_ARGS="$ORACLE_ARGS --oracle-arg docking_batch_size=$DOCK_BATCH" ;;
esac

RUN="$ENUM_DIR/slice${SLICE_IDX}of${N_SLICES}"
mkdir -p "$RUN"
SOCK="$RUN/dock.sock"
SRC_HUBS=${HUBS_FILE:-$ENUM_DIR/hubs.csv}

# Auto-skip a finished stage so a requeue never redoes docking already paid for -- worth far more
# here than for a surrogate cell, where a redo costs minutes rather than GPU-hours.
RESUME=${RESUME:-1}
if [ "$RESUME" = 1 ] && [ "$STAGE" = all ] && [ -s "$SAMPLE_DIR/records.csv" ]; then
    echo "=== [$CELL_TAG] SKIP sample — $SAMPLE_DIR/records.csv exists. RESUME=0 to redo. ==="
    STAGE=enum
fi
if [ "$STAGE" = enum ]; then
    [ -f "$SRC_HUBS" ] || { echo "ERROR: no hubs file $SRC_HUBS — run the sample stage first"; exit 1; }
fi

# Disjoint round-robin hub slice (enum stage only; for STAGE=all it runs after pick_hubs).
# Round-robin rather than contiguous blocks so each slice sees a comparable mix of hub fan-outs.
_slice_hubs() {
python - "$SRC_HUBS" "$RUN/hubs.csv" "$SLICE_IDX" "$N_SLICES" <<'PY'
import csv, sys
src, dst, idx, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = list(csv.DictReader(open(src)))
mine = [r for i, r in enumerate(rows) if i % n == (idx - 1)]
with open(dst, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(mine)
print(f"[dock-cell] slice {idx}/{n}: {len(mine)}/{len(rows)} hubs")
PY
}

echo "=== [$CELL_TAG] slice $SLICE_IDX/$N_SLICES  oracle=$ORACLE $ORACLE_ARGS  enum_max=$ENUM_MAX ==="
echo "=== node=$(hostname)  run=$RUN ==="

# ---- 1. persistent docking server (rgfn env; owns the GPU docker) --------------------------------
conda activate rgfn
source ~/bin/rgfn-smoke-env.sh >/dev/null 2>&1 || true   # LD_LIBRARY_PATH for QV2-GPU + gnina
# Remember rgfn's interpreter: the cleanup trap fires AFTER we switch to the generator's env, and
# that env cannot import glue (no gin) -- without this the clean shutdown degrades to a kill and we
# lose the server's utilization stats.
RGFN_PY="$(command -v python)"
python -m glue.oracles.docking_server --oracle "$ORACLE" \
    --socket "$SOCK" --stats "$RUN/dock_server_stats.json" $ORACLE_ARGS \
    > "$RUN/dock_server.log" 2>&1 &
SERVER_PID=$!
echo "[dock-cell] docking server pid=$SERVER_PID -> $RUN/dock_server.log"

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[dock-cell] shutting down docking server"
        "$RGFN_PY" - "$SOCK" <<'PY' || kill "$SERVER_PID" 2>/dev/null
import sys
sys.path.insert(0, ".")
from glue.oracles.docking_server import DockingServerClient
try:
    print("[dock-cell] final server stats:", DockingServerClient(sys.argv[1]).stats())
    DockingServerClient(sys.argv[1]).shutdown()
except Exception as exc:
    print(f"[dock-cell] clean shutdown failed ({exc}); killing")
    raise SystemExit(1)
PY
        sleep 3; kill -0 "$SERVER_PID" 2>/dev/null && kill "$SERVER_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

# ---- 2. probe THROUGH the socket: is this node's docker actually usable? -------------------------
python - "$SOCK" "$ORACLE" <<'PY' || { echo "ERROR: docking preflight failed on $(hostname)"; exit 42; }
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from glue.oracles.docking_server import DockingServerClient
from preflight_dock import PROBES

sock, oracle = sys.argv[1], sys.argv[2]
c = DockingServerClient(sock)
if not c.wait_until_ready(timeout=900):
    print("[preflight] FATAL: docking server never became ready", flush=True)
    raise SystemExit(42)
probes = PROBES.get(oracle, PROBES["_default"])
labels, _ = c.dock(probes)
ok = sum(1 for v in labels if v is not None and v == v)
print(f"[preflight] {ok}/{len(probes)} probes scored via the socket: {labels}", flush=True)
if ok == 0:
    print(
        "[preflight] FATAL: the docker scored NOTHING on this node — degraded for docking even if "
        "the OpenCL probe passes (balam009, Logs/013/014). Add it to '#SBATCH --exclude'.",
        flush=True,
    )
    raise SystemExit(42)
PY

# ---- 3. enumerate this hub slice in the GENERATOR's env ------------------------------------------
conda activate "$CONDA_ENV"
ENV_PREFIX="/home/markymoo/miniconda3/envs/$CONDA_ENV"
export PATH="$ENV_PREFIX/bin:$PATH"
PY_REAL="$(command -v python || true)"
case "$PY_REAL" in
    "$ENV_PREFIX"/*) : ;;
    *) echo "ERROR: python resolved to '${PY_REAL:-<none>}', not '$ENV_PREFIX/bin/python'."; exit 1 ;;
esac
export RGFN_DOCK_SOCKET="$SOCK"

GUIDANCE_ARG=(); [ -n "${GUIDANCE:-}" ] && GUIDANCE_ARG=(--guidance "$GUIDANCE")
SAMPLE_DIR_ARG=(); case "$GENERATOR" in fraggfn) SAMPLE_DIR_ARG=(--sample-dir "$SAMPLE_DIR") ;; esac

if [ "$STAGE" = sample ] || [ "$STAGE" = all ]; then
    echo "=== [$CELL_TAG] SAMPLE ($N_TRAJ traj, every terminal docked) -> $SAMPLE_DIR ==="
    python "$WORKER" --mode sample \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" "${GUIDANCE_ARG[@]}" \
        --reward-name "$REWARD_NAME" --n-trajectories "$N_TRAJ" --batch-size "$SAMPLE_BATCH" \
        --device "$DEVICE" --run-dir "$RUN/run" --out-dir "$SAMPLE_DIR" \
        || { echo "ERROR: docking sample failed"; exit 1; }

    echo "=== [$CELL_TAG] PICK_HUBS (top-$N_HUBS) -> $ENUM_DIR/hubs.csv ==="
    python experiments/lsd_hubs/campaign/pick_hubs.py \
        --records "$SAMPLE_DIR/records.csv" --out "$ENUM_DIR/hubs.csv" \
        --top-k-candidates "$TOPK" --n-hubs "$N_HUBS" --higher-is-better "$HIGHER_IS_BETTER" \
        || { echo "ERROR: pick_hubs failed"; exit 1; }
fi

if [ "$STAGE" = enum ] || [ "$STAGE" = all ]; then
    _slice_hubs || { echo "ERROR: hub slicing failed"; exit 1; }
    echo "=== [$CELL_TAG] ENUMERATE via docking oracle -> $RUN ==="
    python "$WORKER" --mode enumerate \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" "${GUIDANCE_ARG[@]}" "${SAMPLE_DIR_ARG[@]}" \
        --reward-name "$REWARD_NAME" --hubs-file "$RUN/hubs.csv" \
        --enum-max-children "$ENUM_MAX" --device "$DEVICE" \
        --run-dir "$RUN/run" --out-dir "$RUN" \
        || { echo "ERROR: docking enumerate failed"; exit 1; }
fi

echo "=== DONE [$CELL_TAG] stage=$STAGE slice $SLICE_IDX/$N_SLICES -> $RUN ==="
