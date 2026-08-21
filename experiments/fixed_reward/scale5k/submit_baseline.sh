#!/bin/bash
#SBATCH --job-name=fr5k_base
#SBATCH --time=3-00:00:00
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# One LINK of a baseline (FragGFN / RxnFlow / SCENT) publication-scale fixed-reward run
# (campaign Logs/030). Submit a CHAIN of these from a login node via launch_chain.sh:
#
#   bash experiments/fixed_reward/scale5k/launch_chain.sh 6 \
#        experiments/fixed_reward/scale5k/submit_baseline.sh <fraggfn|rxnflow|scent> <6td3|clpp|seh|drd2> [seed]
#
# For the two DOCKING systems (6td3, clpp) this launches the persistent docking server in the
# rgfn env (kills the ~40 s/step cross-env subprocess spawn at 5,000 steps) and points the
# baseline's docking bridge at it via RGFN_DOCK_SOCKET; the surrogate systems (seh, drd2) need
# no server. Each link resumes from the baseline's last checkpoint and no-ops once
# candidates.csv exists. Set N_ITERS_OVERRIDE / N_SAMPLES_OVERRIDE for a smoke test.
set -uo pipefail
GEN=${1:?usage: submit_baseline.sh <fraggfn|rxnflow|scent> <6td3|clpp|seh|drd2> [seed]}
SYSTEM=${2:?usage: submit_baseline.sh <fraggfn|rxnflow|scent> <6td3|clpp|seh|drd2> [seed]}
SEED=${3:-42}
REPO="$HOME/projects/RGFN_Fork/RGFN-Fork"
cd "$REPO"

# --- map (GEN, SYSTEM) -> config + docking flag + oracle + env + runner --------------------
DOCKING=0; ORACLE=""
case "$SYSTEM" in
  6td3) DOCKING=1; ORACLE=docking_6td3_gpu ;;
  clpp) DOCKING=1; ORACLE=docking_clpp ;;
  seh|drd2) DOCKING=0 ;;
  *) echo "FATAL: unknown SYSTEM '$SYSTEM'"; exit 2 ;;
esac
case "$GEN" in
  fraggfn) ENVNAME=fraggfn; RUNNER=validation/generators/fraggfn/run_fraggfn_fixed.py
    case "$SYSTEM" in
      6td3) CFG=validation/configs/fraggfn_6td3_docking_fixed_5k.yaml ;;
      clpp) CFG=validation/configs/fraggfn_clpp_docking_fixed_5k.yaml ;;
      seh)  CFG=validation/configs/fraggfn_seh_fixed_5k.yaml ;;
      drd2) CFG=validation/configs/fraggfn_drd2_fixed_5k.yaml ;;
    esac ;;
  rxnflow) ENVNAME=rxnflow; RUNNER=validation/generators/rxnflow/run_rxnflow_fixed.py
    case "$SYSTEM" in
      6td3) CFG=validation/configs/rxnflow_6td3_docking_fixed_5k.yaml ;;
      clpp) CFG=validation/configs/rxnflow_clpp_docking_fixed_5k.yaml ;;
      seh)  CFG=validation/configs/rxnflow_seh_fixed_stdlib_5k.yaml ;;
      drd2) CFG=validation/configs/rxnflow_drd2_fixed_stdlib_5k.yaml ;;
    esac ;;
  scent) ENVNAME=scent; RUNNER=validation/generators/scent/run_scent_fixed.py
    case "$SYSTEM" in
      6td3) CFG=validation/configs/scent_6td3_fixed_5k.gin ;;
      clpp) CFG=validation/configs/scent_clpp_fixed_5k.gin ;;
      seh)  CFG=validation/configs/scent_seh_fixed_5k.gin ;;
      drd2) CFG=validation/configs/scent_drd2_fixed_5k.gin ;;
    esac ;;
  # --- the five EXTERNAL entrants added 2026-08-21 ------------------------------------------
  # They reuse this script rather than getting their own because everything hard here is theirs
  # too: the OpenCL health gate, the persistent docking server, the RGFN_DOCK_SOCKET handoff, and
  # the resume/no-op logic. They run at their AUTHORS' budgets (set in their configs), NOT the 5k
  # step count this file was originally written for, so no --n-train-steps is passed for them.
  reinvent) ENVNAME=reinvent4; RUNNER=validation/generators/reinvent/run_reinvent_fixed.py
    CFG=validation/configs/reinvent_${SYSTEM}_fixed.yaml ;;
  saturn) ENVNAME=saturn; RUNNER=validation/generators/saturn/run_saturn_fixed.py
    CFG=validation/configs/saturn_${SYSTEM}_fixed.yaml ;;
  tango) ENVNAME=saturn; RUNNER=validation/generators/saturn/run_saturn_fixed.py
    # TANGO is Saturn's agent with an extra oracle component, so same env and same runner.
    CFG=validation/configs/tango_${SYSTEM}_fixed.yaml ;;
  synformer) ENVNAME=synformer; RUNNER=validation/generators/synformer/run_synformer_fixed.py
    CFG=validation/configs/synformer_${SYSTEM}_fixed.yaml ;;
  s3gfn) ENVNAME=s3gfn; RUNNER=validation/generators/s3gfn/run_s3gfn_fixed.py
    CFG=validation/configs/s3gfn_${SYSTEM}_fixed.yaml ;;
  *) echo "FATAL: unknown GEN '$GEN' (want fraggfn|rxnflow|scent|reinvent|saturn|tango|synformer|s3gfn)"; exit 2 ;;
esac
[ -f "$CFG" ] || { echo "FATAL: no config at $CFG"; exit 2; }

export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch
FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$FR_ROOT_DIR"

# The `_5k` suffix belongs to the original three, whose campaign was defined by its 5,000-step
# budget. The external entrants run at their authors' budgets, so tagging them `_5k` would name
# them after a number that is not theirs.
case "$GEN" in
  reinvent|saturn|tango|synformer|s3gfn) RUN_DIR="$FR_ROOT_DIR/fixed_reward/${GEN}_${SYSTEM}/seed${SEED}" ;;
  *) RUN_DIR="$FR_ROOT_DIR/fixed_reward/${GEN}_${SYSTEM}_5k/seed${SEED}" ;;
esac
COMPLETE="$RUN_DIR/fixed_reward/candidates/candidates.csv"
mkdir -p "$RUN_DIR"
if [ -f "$COMPLETE" ]; then
  echo "[chain] $GEN $SYSTEM seed=$SEED already complete ($COMPLETE); no-op."; exit 0
fi

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
echo "host=$(hostname) gen=$GEN system=$SYSTEM seed=$SEED docking=$DOCKING cfg=$CFG"; nvidia-smi -L || true

# --- docking cells: OpenCL gate + launch the persistent docking server (rgfn env) ----------
SERVER_PID=""
if [ "$DOCKING" = 1 ]; then
  # docking libs (shared by the OpenCL check + the server subprocess)
  export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}
  export GNINA=/scratch/markymoo/gnina/run_gnina.sh
  HC=$SCRATCH/vina_gpu/opencl_healthcheck
  HC_OUT=$(CUDA_VISIBLE_DEVICES=0 "$HC" 2>&1)
  if ! grep -q "clCreateContext err=0" <<<"$HC_OUT"; then
    echo "FATAL: OpenCL dead on $(hostname); exit so the next chain link retries elsewhere."
    echo "$HC_OUT"; exit 42
  fi
  echo "OpenCL health OK on $(hostname)"
  # node-local socket (AF_UNIX ~108-byte cap; shared by all procs of this job on this node)
  export RGFN_DOCK_SOCKET=/tmp/rgfn_dock_${SLURM_JOB_ID:-$$}.sock
  echo "[server] launching persistent docking server (oracle=$ORACLE) on $RGFN_DOCK_SOCKET"
  ( source "$HOME/bin/rgfn-smoke-env.sh" >/dev/null 2>&1
    exec python -m glue.oracles.docking_server \
        --oracle "$ORACLE" --socket "$RGFN_DOCK_SOCKET" \
        --stats "$RUN_DIR/dock_server_stats.json" ) &
  SERVER_PID=$!
  # stats are flushed per-request, so a hard kill on exit still leaves the latest utilization.
  trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null' EXIT
fi

# --- smoke-test overrides (optional) -------------------------------------------------------
# The "how much training" flag is NOT the same across runners -- passing the wrong one is an
# argparse error (exit 2) nine seconds into the job, which is how this was found:
#   fraggfn/rxnflow/s3gfn --n-train-steps | reinvent --max-steps | saturn/tango/synformer --budget
# (saturn and synformer count ORACLE CALLS, not steps, so N_ITERS_OVERRIDE means calls for them.)
OVR=()
if [ -n "${N_ITERS_OVERRIDE:-}" ]; then
  case "$GEN" in
    reinvent)                 OVR+=(--max-steps "$N_ITERS_OVERRIDE") ;;
    saturn|tango|synformer)   OVR+=(--budget    "$N_ITERS_OVERRIDE") ;;
    *)                        OVR+=(--n-train-steps "$N_ITERS_OVERRIDE") ;;
  esac
fi
[ -n "${N_SAMPLES_OVERRIDE:-}" ] && OVR+=(--n-samples "$N_SAMPLES_OVERRIDE")

# --- run the baseline training (its env), resuming from RUN_DIR's checkpoint if present -----
if [ "$GEN" = scent ]; then
  # SCENT also needs --root-dir (for the run_name relative path); its override flags differ.
  SOVR=()
  [ -n "${N_ITERS_OVERRIDE:-}" ] && SOVR+=(--n-iterations "$N_ITERS_OVERRIDE")
  [ -n "${N_SAMPLES_OVERRIDE:-}" ] && SOVR+=(--n-samples "$N_SAMPLES_OVERRIDE")
  # --log-recipes: log every promoted dynamic-library fragment's synthesis route into the
  # fragments_<N>.json snapshot, so LSD-Flow can charge nested fragment builds EXACTLY instead of
  # falling back to min_num_reactions (entry 027/028). Only observable during training, so a run
  # without it can never be fixed after the fact. Redundant with the runner's default (on since
  # 2026-07-29) — passed explicitly so this script states the campaign's intent on its own.
  conda run --no-capture-output -n "$ENVNAME" python "$RUNNER" \
      --cfg "$CFG" --seed "$SEED" --root-dir "$FR_ROOT_DIR" --run-dir "$RUN_DIR" \
      --log-recipes "${SOVR[@]}"
else
  conda run --no-capture-output -n "$ENVNAME" python "$RUNNER" \
      --cfg "$CFG" --seed "$SEED" --run-dir "$RUN_DIR" "${OVR[@]}"
fi
RC=$?

if [ -f "$COMPLETE" ]; then
  echo "[chain] $GEN $SYSTEM seed=$SEED COMPLETE (candidates emitted)."
else
  echo "[chain] $GEN $SYSTEM seed=$SEED link ended (exit $RC) without completion; next link resumes."
fi
exit $RC
