#!/usr/bin/env python
"""Does adding the reaction-GFN's OWN 418 building blocks to the stock rescue failures?

Hypothesis (c) test, done fairly: the reaction-GFN (RGFN/SCENT) assembles molecules from a
fixed 418-fragment "small" library (external/scent/data/small == data/libraries/glue_standard_v1).
Those blocks are its assumed starting materials, but most are NOT in the ZINC catalogue (they carry
reactive handles, e.g. BrCc1ccccn1). So AiZynth, judging synthesizability against ZINC only, can
disconnect a molecule ~all the way yet fail to *terminate* because the natural stopping point is a
reaction-GFN block ZINC doesn't sell.

This routes the 50 known-failures under a 2x2 of {stock} x {budget}:
  stock  = zinc            (catalogue only, the baseline)
         | zinc+rgfnlib    (UNION: catalogue + the 418 blocks, as a NEW library; ZINC hdf5 untouched)
  budget = current (iter=100 time=60 depth=6)  |  high (iter=1000 time=300 depth=9)

The expansion policy is still USPTO (untouched) — we are NOT teaching AiZynth the reaction-GFN's
reactions, only letting its retrosynthesis TERMINATE at the reaction-GFN's blocks when it happens
to reach one. Delta(union - zinc) = how many failures were "stock-limited on the block set".
"""
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

CONFIG = os.environ.get(
    "AIZ_CONFIG", "/scratch/markymoo/rgfn_runs/lsdflow_sparrow/config_rgfnlib.yml"
)
STOCKS = {"zinc": ["zinc"], "union": ["zinc", "rgfnlib"]}
BUDGETS = {"current": (100, 60, 6), "high": (1000, 300, 9)}
_F = None


def _init():
    global _F
    from aizynthfinder.aizynthfinder import AiZynthFinder

    _F = AiZynthFinder(configfile=CONFIG)
    _F.expansion_policy.select("uspto")
    _F.filter_policy.select("uspto")


def _route(task):
    smi, stock_key, bud_key = task
    it, tl, mt = BUDGETS[bud_key]
    try:
        _F.stock.select(STOCKS[stock_key])
        _F.target_smiles = smi
        _F.config.search.iteration_limit = it
        _F.config.search.time_limit = tl
        _F.config.search.max_transforms = mt
        _F.tree_search()
        _F.build_routes()
        ss = _F.search_stats
        return {
            "smiles": smi,
            "stock": stock_key,
            "budget": bud_key,
            "solved": ss.get("first_solution_iteration") is not None,
            "iters": ss.get("iterations"),
            "time": round(float(ss.get("time", 0.0)), 2),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "smiles": smi,
            "stock": stock_key,
            "budget": bud_key,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }


def main():
    smi_file, out = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    smis = [ln.strip() for ln in open(smi_file) if ln.strip()]
    tasks = [(s, sk, bk) for s in smis for sk in STOCKS for bk in BUDGETS]
    print(
        f"[stock] {len(smis)} mols x {len(STOCKS)} stocks x {len(BUDGETS)} budgets "
        f"= {len(tasks)} routes | nproc={nproc}",
        flush=True,
    )
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        results = pool.map(_route, tasks)
    json.dump(results, open(out, "w"), indent=0)
    print(f"[stock] wrote {len(results)} -> {out} in {time.time()-t0:.0f}s", flush=True)

    idx = {}
    for r in results:
        if "error" not in r:
            idx[(r["smiles"], r["stock"], r["budget"])] = r["solved"]
    print("\n================ 2x2 SOLVE TABLE (of 50 known-failures) ================")
    print(f"{'budget':<10}{'zinc':>10}{'zinc+blocks':>14}{'Δ (blocks)':>12}")
    for bk in BUDGETS:
        z = sum(1 for s in smis if idx.get((s, "zinc", bk)))
        u = sum(1 for s in smis if idx.get((s, "union", bk)))
        print(f"{bk:<10}{z:>10}{u:>14}{u - z:>+12}")
    # who did the blocks rescue (at each budget)?
    for bk in BUDGETS:
        rescued = [s for s in smis if (not idx.get((s, "zinc", bk))) and idx.get((s, "union", bk))]
        print(f"\n  blocks rescued @ {bk}: {len(rescued)}")
        for s in rescued[:8]:
            print(f"    {s}")


if __name__ == "__main__":
    main()
