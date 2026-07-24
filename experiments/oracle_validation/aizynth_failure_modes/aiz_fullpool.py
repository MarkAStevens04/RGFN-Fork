#!/usr/bin/env python
"""Full-pool headline: AiZynth solve rate over the ENTIRE 4749-molecule SCENT sEH mode union
with the reaction-GFN's 418 building blocks added to stock (stereo-agnostic), at production budget.

Baseline (ZINC only, same config) = 2314/4749 = 48.7% (from the from-scratch timed cache).
This routes all 4749 under stock=[zinc, rgfnlib] and a 300-molecule ZINC-only control (to confirm
the 48.7% baseline reproduces on this node/config). Production budget = iter=100 / time=60 / depth=6,
matching how the 48.7% was measured, so the delta is the pure stock effect.
"""
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

CONFIG = os.environ.get(
    "AIZ_CONFIG", "/scratch/markymoo/rgfn_runs/lsdflow_sparrow/config_rgfnlib_flat.yml"
)
STOCKS = {"zinc": ["zinc"], "union": ["zinc", "rgfnlib"]}
# budget knobs (env-overridable): production = 100/60/6 (matches the 48.7%/73.8% headline);
# the high-budget "ceiling" run uses AIZ_IT=1000 AIZ_TL=300 AIZ_MT=9.
IT = int(os.environ.get("AIZ_IT", "100"))
TL = int(os.environ.get("AIZ_TL", "60"))
MT = int(os.environ.get("AIZ_MT", "6"))
_F = None


def _init():
    global _F
    from aizynthfinder.aizynthfinder import AiZynthFinder

    _F = AiZynthFinder(configfile=CONFIG)
    _F.expansion_policy.select("uspto")
    _F.filter_policy.select("uspto")
    _F.config.search.iteration_limit = IT
    _F.config.search.time_limit = TL
    _F.config.search.max_transforms = MT


def _route(task):
    smi, stock_key = task
    try:
        _F.stock.select(STOCKS[stock_key])
        _F.target_smiles = smi
        _F.tree_search()
        ss = _F.search_stats
        return {
            "smiles": smi,
            "stock": stock_key,
            "solved": ss.get("first_solution_iteration") is not None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "smiles": smi,
            "stock": stock_key,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }


def main():
    union_smi, ctrl_smi, out = sys.argv[1], sys.argv[2], sys.argv[3]
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    union = [ln.strip() for ln in open(union_smi) if ln.strip()]
    ctrl = [ln.strip() for ln in open(ctrl_smi) if ln.strip()]
    tasks = [(s, "union") for s in union] + [(s, "zinc") for s in ctrl]
    print(
        f"[fullpool] union={len(union)} (zinc+blocks) + control={len(ctrl)} (zinc) "
        f"= {len(tasks)} routes | nproc={nproc} | budget iter={IT}/time={TL}/depth={MT}",
        flush=True,
    )
    t0 = time.time()
    # incremental JSONL write: a wall-clock kill keeps every completed route (compute-node safe)
    jsonl = out + "l"
    results = []
    done = 0
    with open(jsonl, "w") as fh, Pool(nproc, initializer=_init) as pool:
        for r in pool.imap_unordered(_route, tasks, chunksize=4):
            results.append(r)
            fh.write(json.dumps(r) + "\n")
            done += 1
            if done % 500 == 0:
                fh.flush()
                print(f"[fullpool] {done}/{len(tasks)} routed ({time.time()-t0:.0f}s)", flush=True)
    json.dump(results, open(out, "w"), indent=0)
    print(f"[fullpool] wrote {len(results)} -> {out} in {time.time()-t0:.0f}s", flush=True)

    u = [r for r in results if r["stock"] == "union" and "error" not in r]
    z = [r for r in results if r["stock"] == "zinc" and "error" not in r]
    us = sum(r["solved"] for r in u)
    zs = sum(r["solved"] for r in z)
    errs = sum(1 for r in results if "error" in r)
    print("\n================ FULL-POOL HEADLINE ================")
    print(f"  errors: {errs}")
    print(f"  ZINC baseline (cache, all 4749):        2314/4749 = 48.7%")
    print(
        f"  ZINC control (this run, {len(z)}):            {zs}/{len(z)} = {100*zs/max(len(z),1):.1f}%  "
        f"(sanity: should ~48.7%)"
    )
    print(
        f"  ZINC + 418 blocks (this run, all {len(u)}): {us}/{len(u)} = {100*us/max(len(u),1):.1f}%"
    )
    print(
        f"  => stock effect: 48.7% -> {100*us/max(len(u),1):.1f}%  "
        f"(+{100*us/max(len(u),1)-48.7:.1f} pts) at production budget"
    )


if __name__ == "__main__":
    main()
