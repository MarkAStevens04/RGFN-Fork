#!/usr/bin/env python
"""Follow-up: isolate WHY the stubborn high-budget failures won't close their last leaves.

The 2-arm diagnostic showed: 0/50 are OOD; 18% recover with 10x budget; ~82% reach >=75%
in-stock but can't close the residual 1-2 leaves even at iter=1000/depth=9. This run separates
the residual cause on the SAME 50 molecules, all at high iteration budget:

  arm 'filter_on'   : iter=1000 time=300 depth=9 filter=uspto   (baseline; reproduce ~18%)
  arm 'filter_off'  : iter=1000 time=300 depth=9 filter=NONE    (isolate (d) QuickKerasFilter)
  arm 'depth12'     : iter=1000 time=300 depth=12 filter=uspto  (isolate residual depth)

For every UNSOLVED best route it also dumps the not-in-stock ("open") leaves with heavy-atom
counts: small open leaves => (c) stock too small (should be buyable); large/complex open leaves
=> the residual needs deeper/other disconnection the USPTO policy won't make (fragment-level OOD).
"""
import json
import sys
import time
import traceback
from multiprocessing import Pool

CONFIG = "data/models/aizynthfinder/config.yml"
ARMS = {
    "filter_on": (1000, 300, 9, "uspto"),
    "filter_off": (1000, 300, 9, "none"),
    "depth12": (1000, 300, 12, "uspto"),
}
_F = None


def _init():
    global _F
    from aizynthfinder.aizynthfinder import AiZynthFinder

    _F = AiZynthFinder(configfile=CONFIG)
    _F.stock.select("zinc")
    _F.expansion_policy.select("uspto")


def _hac(smi):
    from rdkit import Chem

    m = Chem.MolFromSmiles(smi) if smi else None
    return m.GetNumHeavyAtoms() if m is not None else None


def _route(task):
    smi, arm = task
    it, tl, mt, filt = ARMS[arm]
    try:
        _F.filter_policy.deselect()  # reset any prior selection (worker is reused across arms)
        if filt and filt.lower() != "none":
            _F.filter_policy.select(filt)
        _F.target_smiles = smi
        _F.config.search.iteration_limit = it
        _F.config.search.time_limit = tl
        _F.config.search.max_transforms = mt
        _F.tree_search()
        _F.build_routes()
        ss = _F.search_stats
        rts = _F.routes.reaction_trees
        solved = ss.get("first_solution_iteration") is not None
        open_leaves = []
        if rts and not solved:
            rt = rts[0]
            for m in rt.leafs():
                if m not in _F.stock:
                    open_leaves.append({"smi": m.smiles, "hac": _hac(m.smiles)})
        return {
            "smiles": smi,
            "arm": arm,
            "solved": bool(solved),
            "iters": ss.get("iterations"),
            "hit_iter_cap": ss.get("iterations", 0) >= it,
            "time": round(float(ss.get("time", 0.0)), 2),
            "n_routes": len(rts),
            "n_open_leaves": len(open_leaves),
            "open_leaves": open_leaves,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "smiles": smi,
            "arm": arm,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }


def main():
    smi_file, out = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    smis = [ln.strip() for ln in open(smi_file) if ln.strip()]
    tasks = [(s, arm) for s in smis for arm in ARMS]
    print(
        f"[resid] {len(smis)} molecules x {len(ARMS)} arms = {len(tasks)} routes | nproc={nproc}",
        flush=True,
    )
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        results = pool.map(_route, tasks)
    json.dump(results, open(out, "w"), indent=0)
    print(f"[resid] wrote {len(results)} -> {out} in {time.time()-t0:.0f}s", flush=True)

    by = {}
    for r in results:
        by.setdefault(r["arm"], []).append(r)
    print("\n================ SUMMARY ================")
    for arm in ARMS:
        rs = [r for r in by.get(arm, []) if "error" not in r]
        errs = [r for r in by.get(arm, []) if "error" in r]
        n = len(rs)
        solved = sum(r["solved"] for r in rs)
        it, tl, mt, filt = ARMS[arm]
        print(f"\n--- {arm} (iter={it} depth={mt} filter={filt}) ---")
        print(f"  routed={n} errors={len(errs)} SOLVED={solved}/{n} = {100.0*solved/max(n,1):.1f}%")
        # open-leaf size distribution over unsolved
        nol = [r["n_open_leaves"] for r in rs if not r["solved"]]
        if nol:
            import statistics as st

            print(
                f"  unsolved n_open_leaves: mean={st.mean(nol):.2f} "
                f"min={min(nol)} max={max(nol)} "
                f"(==1: {sum(1 for x in nol if x==1)}, ==2: {sum(1 for x in nol if x==2)}, "
                f">=3: {sum(1 for x in nol if x>=3)})"
            )
        hacs = [lf["hac"] for r in rs if not r["solved"] for lf in r["open_leaves"] if lf["hac"]]
        if hacs:
            import statistics as st

            small = sum(1 for h in hacs if h <= 8)
            print(
                f"  open-leaf heavy-atom count: n={len(hacs)} mean={st.mean(hacs):.1f} "
                f"median={st.median(hacs):.0f} min={min(hacs)} max={max(hacs)} "
                f"| <=8 HA (small/buyable-ish): {small} ({100*small/len(hacs):.0f}%)"
            )

    # filter contribution: molecules solved by turning the filter OFF (holding budget)
    fon = {r["smiles"]: r for r in by.get("filter_on", []) if "error" not in r}
    foff = {r["smiles"]: r for r in by.get("filter_off", []) if "error" not in r}
    d12 = {r["smiles"]: r for r in by.get("depth12", []) if "error" not in r}
    common = [s for s in fon if s in foff]
    filt_gain = [s for s in common if (not fon[s]["solved"]) and foff[s]["solved"]]
    depth_gain = [s for s in fon if s in d12 and (not fon[s]["solved"]) and d12[s]["solved"]]
    print("\n================ RESIDUAL CAUSE ================")
    print(f"  failed@filter_on: {sum(1 for s in common if not fon[s]['solved'])}/{len(common)}")
    print(f"  ...solved by FILTER OFF (=> (d) filter pruned): {len(filt_gain)}")
    print(f"  ...solved by DEPTH 12  (=> residual depth):     {len(depth_gain)}")
    print(
        "  (remaining stubborn = (c) stock too small / fragment-level policy-OOD; "
        "see open-leaf HAC above: small=stock, large=needs disconnection)"
    )


if __name__ == "__main__":
    main()
