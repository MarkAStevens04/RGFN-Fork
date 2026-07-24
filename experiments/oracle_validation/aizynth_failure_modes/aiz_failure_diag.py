#!/usr/bin/env python
"""AiZynthFinder failure-mode diagnostic (read-only, no benchmark changes).

WHY: our SCENT sEH from-scratch AiZynth solve rate (best-candidate 51.3% / hub 61.0%,
top-300+diversity) sits ~20 pts below the paper's top-100 0.773. The research report
(Logs, 2026-07-23) lays out four distinct failure modes with separable signatures:
  (a) BUDGET   — a route exists but wasn't reached: hit iteration/time/depth cap, best
                 partial route has HIGH fraction-in-stock (~1, one open leaf), common
                 templates. Fix = more iterations/time/depth.
  (b) OOD      — USPTO policy proposes nothing: search RAN DRY early (iters << cap),
                 small tree, LOW fraction-in-stock, low template-occurrence. Budget won't help.
  (c) STOCK    — reaches buyable-looking fragments not in ZINC (frac<1 at leaves).
  (d) FILTER   — QuickKerasFilter pruned the only disconnection.

This routes a sample of CURRENTLY-FAILED molecules under two budgets and records, per
molecule, the artifacts that separate (a) from (b): did it hit the iteration cap, how
in-stock is the best partial route, how deep, how common are its templates, and — the
decisive test — does it SOLVE when given a much larger budget.
"""
import json
import sys
import time
import traceback
from multiprocessing import Pool

CONFIG = "data/models/aizynthfinder/config.yml"
ARMS = {
    # name: (iteration_limit, time_limit, max_transforms)
    "current": (100, 60, 6),
    "high": (1000, 300, 9),
}

_F = None
_SCORERS = None


def _init():
    global _F, _SCORERS
    from aizynthfinder.aizynthfinder import AiZynthFinder
    from aizynthfinder.context.scoring import (
        AverageTemplateOccurrenceScorer,
        FractionInStockScorer,
        MaxTransformScorer,
        NumberOfReactionsScorer,
        StateScorer,
    )

    _F = AiZynthFinder(configfile=CONFIG)
    _F.stock.select("zinc")
    _F.expansion_policy.select("uspto")
    _F.filter_policy.select("uspto")
    _SCORERS = {
        "frac_in_stock": FractionInStockScorer(_F.config),
        "n_rxn": NumberOfReactionsScorer(_F.config),
        "max_depth": MaxTransformScorer(_F.config),
        "tmpl_occ": AverageTemplateOccurrenceScorer(_F.config),
        "state": StateScorer(_F.config),
    }


def _route(task):
    smi, arm = task
    it, tl, mt = ARMS[arm]
    try:
        _F.target_smiles = smi
        _F.config.search.iteration_limit = it
        _F.config.search.time_limit = tl
        _F.config.search.max_transforms = mt
        _F.tree_search()
        _F.build_routes()
        ss = _F.search_stats
        rts = _F.routes.reaction_trees
        solved = ss.get("first_solution_iteration") is not None
        best = {}
        if rts:
            rt = rts[0]  # top-ranked partial (by state score)
            for n, s in _SCORERS.items():
                try:
                    best[n] = round(float(s(rt)), 4)
                except Exception:
                    best[n] = None
        return {
            "smiles": smi,
            "arm": arm,
            "solved": bool(solved),
            "iters": ss.get("iterations"),
            "iter_limit": it,
            "hit_iter_cap": ss.get("iterations", 0) >= it,
            "time": round(float(ss.get("time", 0.0)), 2),
            "n_routes": len(rts),
            "best": best,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "smiles": smi,
            "arm": arm,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }


def main():
    smi_file = sys.argv[1]
    out = sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    smis = [ln.strip() for ln in open(smi_file) if ln.strip()]
    tasks = [(s, arm) for s in smis for arm in ARMS]
    print(
        f"[diag] {len(smis)} molecules x {len(ARMS)} arms = {len(tasks)} routes | nproc={nproc}",
        flush=True,
    )
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        results = pool.map(_route, tasks)
    json.dump(results, open(out, "w"), indent=0)
    print(f"[diag] wrote {len(results)} -> {out} in {time.time()-t0:.0f}s", flush=True)

    # ---- summary ----------------------------------------------------------------
    by = {}
    for r in results:
        by.setdefault(r["arm"], []).append(r)

    def frac_bucket(v):
        if v is None:
            return "no_route"
        if v >= 0.999:
            return "1.0(solvable-leaf)"
        if v >= 0.75:
            return ">=0.75"
        if v >= 0.5:
            return "0.5-0.75"
        return "<0.5"

    print("\n================ SUMMARY ================")
    for arm in ARMS:
        rs = [r for r in by.get(arm, []) if "error" not in r]
        errs = [r for r in by.get(arm, []) if "error" in r]
        n = len(rs)
        solved = sum(r["solved"] for r in rs)
        print(
            f"\n--- arm={arm}  (iter={ARMS[arm][0]} time={ARMS[arm][1]}s depth={ARMS[arm][2]}) ---"
        )
        print(
            f"  routed={n}  errors={len(errs)}  SOLVED={solved}/{n} = {100.0*solved/max(n,1):.1f}%"
        )
        cap = sum(1 for r in rs if r.get("hit_iter_cap"))
        drylow = sum(
            1
            for r in rs
            if not r.get("hit_iter_cap") and (r["best"].get("frac_in_stock") or 0) < 0.5
        )
        print(
            f"  hit_iteration_cap: {cap}/{n}  |  ran-dry-early & frac<0.5 (OOD-like): {drylow}/{n}"
        )
        # fraction-in-stock distribution of the BEST partial route (unsolved only)
        fb = {}
        depth_at_cap = 0
        for r in rs:
            if r["solved"]:
                continue
            fb[frac_bucket(r["best"].get("frac_in_stock"))] = (
                fb.get(frac_bucket(r["best"].get("frac_in_stock")), 0) + 1
            )
            if (r["best"].get("max_depth") or 0) >= ARMS[arm][2]:
                depth_at_cap += 1
        print(f"  UNSOLVED best-route frac-in-stock: {dict(sorted(fb.items()))}")
        print(f"  UNSOLVED hitting depth cap (>= {ARMS[arm][2]}): {depth_at_cap}")

    # decisive: of molecules that FAILED at current budget, how many SOLVE at high budget?
    cur = {r["smiles"]: r for r in by.get("current", []) if "error" not in r}
    hi = {r["smiles"]: r for r in by.get("high", []) if "error" not in r}
    common = [s for s in cur if s in hi]
    recovered = [s for s in common if (not cur[s]["solved"]) and hi[s]["solved"]]
    still = [s for s in common if (not cur[s]["solved"]) and (not hi[s]["solved"])]
    print("\n================ DECISIVE ================")
    print(
        f"  molecules failed@current: {sum(1 for s in common if not cur[s]['solved'])}/{len(common)}"
    )
    print(f"  ...RECOVERED with high budget (=> budget-limited, mode a): {len(recovered)}")
    print(f"  ...STILL fail at high budget (=> genuine OOD/stock/filter, b/c/d): {len(still)}")
    # of the still-failing, how in-stock did they get at high budget? (near-1 => stock/filter; low => OOD)
    still_fb = {}
    for s in still:
        still_fb[frac_bucket(hi[s]["best"].get("frac_in_stock"))] = (
            still_fb.get(frac_bucket(hi[s]["best"].get("frac_in_stock")), 0) + 1
        )
    print(f"  STILL-failing best frac-in-stock @high: {dict(sorted(still_fb.items()))}")


if __name__ == "__main__":
    main()
