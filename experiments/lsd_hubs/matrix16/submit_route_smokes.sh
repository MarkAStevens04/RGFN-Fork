#!/bin/bash
#SBATCH --job-name=route_smokes
#SBATCH --time=01:45:00
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
# The three route-contract checks the LOGIN node could not finish: it enforces `ulimit -t 3600` on CPU
# time, which SIGXCPU-killed the rxnflow smoke, the rgfn enumerate end-to-end, and the determinism test
# mid-run (entry: balam-login-cpu-timelimit). debug turns each into minutes on a GPU.
#
# All three in ONE job on purpose: the debug QoS allows exactly one job per user (MaxJobsPU=1), so a
# --dependency chain is REJECTED AT SUBMIT here. Sequential inside one allocation is the only shape
# that works.
#
#   1. RXNFLOW ROUTE EMISSION -- the point of this job. The emitter shipped UNSMOKED; it is the last
#      unverified piece of the route contract.
#   2. RGFN ENUMERATE end-to-end -- the exact path 8 queued rgfn_6td3 slices will run, including the
#      new _routes.validate_enum_reactions call. Verified against real artifacts already, but never as
#      a live worker invocation.
#   3. RGFN SEED DETERMINISM under PYTHONHASHSEED=0 -- tests the one hypothesis left standing for why
#      --seed alone gave 377 vs 387 routes: per-process set/dict iteration order feeding action-space
#      construction, so an identical RNG draw picks a different action.
set -uo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-/home/markymoo/projects/RGFN_Fork/RGFN-Fork}}
cd "$REPO" || { echo "FATAL: cannot cd to REPO=$REPO"; exit 1; }
[ -f validation/lsdflow/adapters/workers/_routes.py ] || {
    echo "FATAL: REPO=$REPO is not the repo (no _routes.py). \$SLURM_SUBMIT_DIR was '${SLURM_SUBMIT_DIR:-}'"; exit 1; }

# $HOME is READ-ONLY on compute nodes and several libs default their caches there.
OUT=/scratch/markymoo/rgfn_runs/route_smokes/${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"
export MPLCONFIGDIR="$OUT/.mpl" TRITON_CACHE_DIR="$OUT/.triton" HF_HOME="$OUT/.hf"
export XDG_CACHE_HOME="$OUT/.cache"
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"

echo "=== route smokes  node=$(hostname)  out=$OUT ==="
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

# ---------------------------------------------------------------- 1. rxnflow route emission
echo; echo "######## [1/3] rxnflow route emission (the unsmoked one) ########"
(
  conda activate rxnflow || exit 90
  export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/rxnflow/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"
  python validation/lsdflow/adapters/workers/rxnflow_worker.py --mode sample \
    --config validation/configs/rxnflow_seh_fixed_stdlib_5k.yaml \
    --checkpoint /scratch/markymoo/rgfn_runs/experiments/fixed_reward/rxnflow_seh_5k/seed42/checkpoints/last_gfn.pt \
    --reward-name seh --n-trajectories 200 --batch-size 50 --seed 42 \
    --run-dir "$OUT/rxn_run" --out-dir "$OUT/rxnflow_sample"
) 2>&1 | tail -30
echo "[1/3] exit=${PIPESTATUS[0]}"

# ---------------------------------------------------------------- 2. rgfn enumerate end-to-end
echo; echo "######## [2/3] rgfn ENUMERATE end-to-end (path of the 8 queued slices) ########"
head -4 /scratch/markymoo/rgfn_runs/lsdflow/matrix16/rgfn_seh/enum/hubs.csv > "$OUT/hubs3.csv"
(
  source ~/bin/rgfn-smoke-env.sh
  python validation/lsdflow/adapters/workers/rgfn_worker.py --mode enumerate \
    --config configs/glue/fixed_reward_seh_proxy_stdlib_5k.gin \
    --checkpoint /scratch/markymoo/rgfn_runs/experiments/fixed_reward/rgfn_seh_5k/seed42/train/checkpoints/last_gfn.pt \
    --reward-name seh --hubs-file "$OUT/hubs3.csv" --enum-max-children 500 \
    --run-dir "$OUT/enum_run" --out-dir "$OUT/rgfn_enum"
) 2>&1 | grep -vE 'Failed embedding|Bad Conformer' | tail -20
echo "[2/3] exit=${PIPESTATUS[0]}"

# ---------------------------------------------------------------- 3. determinism under PYTHONHASHSEED
echo; echo "######## [3/3] rgfn --seed 42 x2 with PYTHONHASHSEED=0 ########"
for r in A B; do
  (
    source ~/bin/rgfn-smoke-env.sh
    export PYTHONHASHSEED=0
    python validation/lsdflow/adapters/workers/rgfn_worker.py --mode sample \
      --config configs/glue/fixed_reward_seh_proxy_stdlib_5k.gin \
      --checkpoint /scratch/markymoo/rgfn_runs/experiments/fixed_reward/rgfn_seh_5k/seed42/train/checkpoints/last_gfn.pt \
      --reward-name seh --n-trajectories 200 --batch-size 50 --seed 42 \
      --run-dir "$OUT/det_run_$r" --out-dir "$OUT/det_$r"
  ) > "$OUT/det_$r.log" 2>&1
  echo "  run $r exit=$?  $(grep -o '[0-9]* routes emitted' "$OUT/det_$r.log" | head -1)"
done

# ---------------------------------------------------------------- verdicts
echo; echo "######## VERDICTS ########"
python - "$OUT" <<'PYV'
import csv, json, sys
from pathlib import Path
O = Path(sys.argv[1])

def jload(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None

print("-- [1] rxnflow route emission --")
r = jload(O/"rxnflow_sample/routes.json"); st = jload(O/"rxnflow_sample/route_status.json")
if r is None:
    print("   FAIL: no routes.json (see [1/3] output above)")
else:
    print(f"   n_routes={len(r):,}   route_status={st}")
    if r:
        # hub keys the campaign will look up must be present, and steps must chain
        recs = list(csv.DictReader(open(O/"rxnflow_sample/records.csv")))
        hk = {x["hub_key"] for x in recs if x.get("hub_key")}
        hit = len(hk & set(r))
        print(f"   hub_keys routed: {hit}/{len(hk)} = {100*hit/max(len(hk),1):.1f}%")
        bad = [k for k,v in list(r.items())[:300] if not all(
            {"reaction","reactants","input","product"} <= set(s) for s in v.get("steps",[]))]
        print(f"   canonical-schema violations in first 300: {len(bad)}")
        ch = tot = 0
        for v in list(r.values())[:400]:
            for a,b in zip(v.get("steps",[]), v.get("steps",[])[1:]):
                tot += 1; ch += (a["product"] == b["input"])
        print(f"   step-chain integrity: {ch}/{tot}")
        import collections
        print(f"   depth histogram: {dict(sorted(collections.Counter(v['num_reactions'] for v in r.values()).items()))}")

print("-- [2] rgfn enumerate --")
e = jload(O/"rgfn_enum/enum_children.json"); st2 = jload(O/"rgfn_enum/route_status.json")
if e is None:
    print("   FAIL: no enum_children.json")
else:
    hubs = e.get("hubs", [])
    nc = sum(len(h.get("children",[]) or []) for h in hubs)
    nr = sum(1 for h in hubs for c in (h.get("children",[]) or []) if c.get("reaction"))
    print(f"   hubs={len(hubs)} children={nc:,} with_reaction={nr:,} ({100*nr/max(nc,1):.1f}%)")
    print(f"   route_status={st2}")

print("-- [3] determinism under PYTHONHASHSEED=0 --")
a, b = jload(O/"det_A/routes.json"), jload(O/"det_B/routes.json")
if not a or not b:
    print("   INCONCLUSIVE: one run produced no routes")
else:
    ka, kb = set(a), set(b)
    print(f"   run A {len(a):,} routes | run B {len(b):,} routes | shared {len(ka&kb):,}/{len(ka|kb):,}")
    if ka == kb:
        same = sum(1 for k in ka if json.dumps(a[k],sort_keys=True)==json.dumps(b[k],sort_keys=True))
        print(f"   identical content {same}/{len(ka)} -> "
              + ("REPRODUCIBLE with PYTHONHASHSEED=0" if same==len(ka) else "keys match, content differs"))
    else:
        print("   NOT reproducible even with PYTHONHASHSEED=0 -> the cause is elsewhere")
PYV
echo; echo "=== done; artifacts under $OUT ==="
