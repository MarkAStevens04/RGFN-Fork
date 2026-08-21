#!/usr/bin/env python
"""Which cell-seeds can feed the SPARROW competitor arm? One command, no archaeology.

WHY THIS EXISTS. "Can we run the competitor arm on cell X?" took an agent a full session to answer,
by hand, across three scratch trees -- and the answer was mostly no. The facts needed are cheap and
local (is routes.json populated? do enumerated children carry reactions? can the run's promoted
fragments be expanded?), so there is no reason for that question to ever be expensive again. Run this
before planning competitor work, not after.

THREE requirements, not two, and the third bites hardest because it is invisible. A generator with a
DYNAMIC LIBRARY (SCENT alone) attaches promoted fragments in one shallow step, so its routes must be
recipe-expanded from the run's `fragments_<N>.json` or SPARROW BUYS what count-once BUILDS (the 62.7%
reconcile of Logs/049). Two ways that fails:
  * the snapshot has no ``smiles_to_route`` at all -- recipe logging was off for runs launched
    2026-07-14..07-26, which is every scent_clpp seed. NOT repairable: a fragment's route is
    observable only while it is still being BUILT, so this needs a re-TRAIN, not a re-sample.
  * the snapshot belongs to a DIFFERENT model. It is then internally complete, so a coverage check on
    its own ``chosen_smiles`` reads 100% while the run's fragments go unexpanded -- measured 53% on
    matrix16/scent_seh paired with the dated scent_seh snapshot, with a green light on top.

Reads only artifacts, never a model, so it is instant and safe to run against live runs.

    python experiments/lsd_hubs/matrix16/check_route_readiness.py            # all trees
    python experiments/lsd_hubs/matrix16/check_route_readiness.py --seed 42
    python experiments/lsd_hubs/matrix16/check_route_readiness.py --json     # machine-readable

Exit 1 if any route-bearing cell is missing routes, so it can gate a submit script.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

SCRATCH = Path("/scratch/markymoo/rgfn_runs/lsdflow")
TREES = {42: "matrix16", 43: "matrix16_seed43", 44: "matrix16_seed44"}

# Mirrors _routes.ROUTE_CONTRACT. Duplicated deliberately: this script must run in ANY env (it is a
# stdlib-only reporter, often invoked from a login shell with no worker path set up), and importing
# the workers package would drag in their sys.path convention for no benefit. If the contract changes,
# change it in both places -- there are exactly two.
NOT_ROUTE_BEARING = {"fraggfn"}
NA_REASON = "attachments, not reactions (docs/LSD_FLOW_PROPOSAL.md L274) -- empty is correct"


def _n_routes(sample_dir: Path):
    p = sample_dir / "routes.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return len(d) if hasattr(d, "__len__") else None
    except Exception:
        return None


def _enum_rxn_coverage(enum_dir: Path):
    """(n_children, n_with_reaction, n_hubs_seen) unioned over slices and any top-level file."""
    paths = sorted(glob.glob(str(enum_dir / "slice*of*" / "enum_children.json")))
    top = enum_dir / "enum_children.json"
    if top.exists():
        paths.append(str(top))
    if not paths:
        return None
    n_child = n_rxn = 0
    hubs_seen = set()
    for p in paths:
        try:
            for h in json.loads(Path(p).read_text()).get("hubs", []):
                hubs_seen.add(h.get("hub_input") or h.get("hub_key"))
                for c in h.get("children", []) or []:
                    n_child += 1
                    if c.get("reaction"):
                        n_rxn += 1
        except Exception:
            continue
    return n_child, n_rxn, len(hubs_seen)


def _n_hubs_wanted(enum_dir: Path):
    p = enum_dir / "hubs.csv"
    if not p.exists():
        return None
    try:
        return sum(1 for _ in csv.DictReader(open(p)))
    except Exception:
        return None


FIXED = Path("/scratch/markymoo/rgfn_runs/experiments/fixed_reward")
MIN_RECIPE_COV = 0.95  # matches sparrow_select_frontier's own abort threshold


def _run_id(path: str) -> str:
    """The ``<run>/<sub>`` segment of a .../fixed_reward/<run>/<sub>/... path, or ''."""
    parts = Path(path).parts
    if "fixed_reward" in parts:
        i = parts.index("fixed_reward")
        return "/".join(parts[i + 1 : i + 3])
    return ""


def _has_key(path: Path, needle: bytes = b'"smiles_to_route"') -> bool:
    """Is the key present? Chunked byte scan, because these snapshots are 100 MB+ and the key is
    appended LAST -- so neither a full ``json.loads`` nor a prefix read will do. Mirrors
    ``run_scent_fixed.py::_has_routes``; overlap by len(needle)-1 so a match spanning a chunk
    boundary is still found. This is what keeps the script instant."""
    tail = b""
    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(1 << 20):
                if needle in tail + chunk:
                    return True
                tail = chunk[-(len(needle) - 1) :]
    except Exception:
        return False
    return False


def _recipe_health(cell_dir: Path):
    """(coverage, snapshot) for the cell's OWN training snapshot, or None if it has no dynamic library.

    The snapshot is resolved FROM the cell's recorded checkpoint, so it is the run's own by
    construction -- which is why coverage of ``chosen_smiles`` is the right question here. (The other
    failure mode, being handed a DIFFERENT model's snapshot, cannot arise from this direction; that
    one is the caller's problem and ``sparrow_select_frontier`` aborts on it.)

    Coverage is only computed when the key is present, so the common failure (absent entirely) costs
    a byte scan rather than parsing 100 MB of JSON.
    """
    meta = cell_dir / "enum" / "meta.json"
    if not meta.exists():
        meta = cell_dir / "sample" / "meta.json"
    if not meta.exists():
        return None
    try:
        ckpt = json.loads(meta.read_text()).get("checkpoint", "") or ""
    except Exception:
        return None
    run_id = _run_id(ckpt)
    if not run_id:
        return None
    snaps = sorted(glob.glob(str(FIXED / run_id / "additional_fragments" / "fragments_*.json")))
    if not snaps:
        return None  # no dynamic library -> nothing to expand, correct for non-SCENT
    snap = Path(snaps[-1])
    if not _has_key(snap):
        return 0.0, str(snap)          # the 07-14..07-26 logging window; no parse needed
    try:
        d = json.loads(snap.read_text())
    except Exception:
        return None
    chosen, routes = d.get("chosen_smiles") or [], set(d.get("smiles_to_route") or {})
    if not chosen:
        return None
    return sum(1 for x in chosen if x in routes) / len(chosen), str(snap)


def scan(seeds):
    rows = []
    for seed in seeds:
        tree = SCRATCH / TREES[seed]
        if not tree.exists():
            continue
        for cell_dir in sorted(tree.iterdir()):
            if not cell_dir.is_dir():
                continue
            gen = cell_dir.name.split("_")[0]
            sample, enum = cell_dir / "sample", cell_dir / "enum"
            if not sample.exists() and not enum.exists():
                continue
            nr = _n_routes(sample)
            cov = _enum_rxn_coverage(enum)
            want = _n_hubs_wanted(enum)
            status = json.loads((sample / "route_status.json").read_text()) \
                if (sample / "route_status.json").exists() else None
            rec = _recipe_health(cell_dir)
            row = {
                "seed": seed,
                "cell": cell_dir.name,
                "generator": gen,
                "route_bearing": gen not in NOT_ROUTE_BEARING,
                "n_routes": nr,
                "enum_children": cov[0] if cov else None,
                "enum_with_reaction": cov[1] if cov else None,
                "enum_hubs": cov[2] if cov else None,
                "enum_hubs_wanted": want,
                "declared_status": (status or {}).get("state"),
                # None = no dynamic library, so no expansion needed (correct for non-SCENT)
                "recipe_coverage": None if rec is None else round(rec[0], 4),
                "recipe_snapshot": None if rec is None else rec[1],
            }
            # SPARROW-ready = a full route exists for every enumerated child: the hub prefix comes
            # from routes.json, the final step from children[].reaction, and (for SCENT) the promoted
            # fragments in them must be recipe-expandable. Missing ANY of the three blocks pricing.
            #
            # REPORT EVERY BLOCKER, NOT THE FIRST. This used to be an elif chain, and the ordering
            # silently hid a real defect for hours: "NO ROUTES" was tested first, so for any cell
            # lacking routes -- i.e. EVERY rgfn and rxnflow cell -- the reaction check below was never
            # reached. That produced the confident and wrong claim "100% reaction coverage matrix-wide",
            # while rgfn_clpp@s43 (332,056 children) and rgfn_6td3@s43 (477,554 children) sat COMPLETE
            # at 200/200 hubs with `"reaction": []` on every child and no repair in flight. Both are
            # ~104 GPU-h docking enumerations, so the cost of not noticing was high. A readiness tool
            # that reports one blocker at a time invites exactly this: you fix the named problem, re-run,
            # and discover the next one -- or worse, you read the absence of a message as an all-clear.
            blockers = []
            if gen in NOT_ROUTE_BEARING:
                row["verdict"] = "n/a (control)"
            else:
                if not nr:
                    blockers.append("NO ROUTES (re-sample)")
                if cov and cov[0] and cov[1] == 0:
                    blockers.append(f"NO REACTIONS 0/{cov[0]:,} (re-enum)")
                elif cov and cov[0] and cov[1] < cov[0]:
                    blockers.append(f"partial reactions {cov[1]:,}/{cov[0]:,} (re-enum)")
                if rec is not None and rec[0] < MIN_RECIPE_COV:
                    blockers.append(f"NO RECIPES {rec[0]:.0%} (re-TRAIN)")
                if not cov or not cov[0]:
                    blockers.append("no enumeration yet")
                row["verdict"] = " + ".join(blockers) if blockers else "READY"
            row["blockers"] = blockers
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rows = scan(a.seed)
    if a.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"{'seed':>4}  {'cell':<15} {'routes':>9} {'children':>10} {'rxn%':>6}"
              f" {'recipes':>8}  verdict")
        print("-" * 86)
        for r in rows:
            pct = ""
            if r["enum_children"]:
                pct = f"{100*r['enum_with_reaction']/r['enum_children']:.0f}%"
            rcp = "n/a" if r["recipe_coverage"] is None else f"{100*r['recipe_coverage']:.0f}%"
            print(f"{r['seed']:>4}  {r['cell']:<15} {str(r['n_routes'] or '-'):>9} "
                  f"{str(r['enum_children'] or '-'):>10} {pct:>6} {rcp:>8}  {r['verdict']}")
        bad = [r for r in rows if r["route_bearing"] and not r["n_routes"]]
        ready = [r for r in rows if r["verdict"] == "READY"]
        print("-" * 86)
        print(f"  {len(ready)} SPARROW-ready | {len(bad)} route-bearing cell-seeds with NO routes"
              f" | {len([r for r in rows if not r['route_bearing']])} control (n/a: {NA_REASON})")
        norec = [r for r in rows if str(r["verdict"]).startswith("NO RECIPES")]
        if norec:
            print(f"\n  {len(norec)} cell-seed(s) have routes AND reactions but cannot be priced: the"
                  "\n  snapshot cannot expand the promoted fragments they are built from. This needs a"
                  "\n  RE-TRAIN with --log-recipes (default ON since 2026-07-29) -- a fragment's route is"
                  "\n  observable only while it is being built, so no re-sample or re-enum recovers it:"
                  + "".join(f"\n    seed {r['seed']} {r['cell']}" for r in norec))
        if bad:
            print("\n  Cells with no routes need a RE-SAMPLE -- the trajectory is not recoverable from"
                  "\n  existing artifacts (compositions.json keeps only num_reactions). Re-running the"
                  "\n  ENUMERATION alone does not help: routes.json is written by the SAMPLE stage.")
    return 1 if any(r["route_bearing"] and not r["n_routes"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
