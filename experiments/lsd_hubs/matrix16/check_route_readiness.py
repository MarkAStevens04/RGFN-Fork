#!/usr/bin/env python
"""Which cell-seeds can feed the SPARROW competitor arm? One command, no archaeology.

WHY THIS EXISTS. "Can we run the competitor arm on cell X?" took an agent a full session to answer,
by hand, across three scratch trees -- and the answer was mostly no. The facts needed are cheap and
local (is routes.json populated? do enumerated children carry reactions?), so there is no reason for
that question to ever be expensive again. Run this before planning competitor work, not after.

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
            }
            # SPARROW-ready = a full route exists for every enumerated child: the hub prefix comes
            # from routes.json, the final step from children[].reaction. Missing EITHER half means the
            # library cannot be priced, so both are required and reported separately.
            if gen in NOT_ROUTE_BEARING:
                row["verdict"] = "n/a (control)"
            elif not nr:
                row["verdict"] = "NO ROUTES (needs re-sample)"
            elif cov and cov[0] and cov[1] < cov[0]:
                row["verdict"] = f"partial reactions {cov[1]}/{cov[0]}"
            elif cov and cov[0]:
                row["verdict"] = "READY"
            else:
                row["verdict"] = "routes ok, no enumeration yet"
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
        print(f"{'seed':>4}  {'cell':<15} {'routes':>9} {'children':>10} {'rxn%':>6}  verdict")
        print("-" * 74)
        for r in rows:
            pct = ""
            if r["enum_children"]:
                pct = f"{100*r['enum_with_reaction']/r['enum_children']:.0f}%"
            print(f"{r['seed']:>4}  {r['cell']:<15} {str(r['n_routes'] or '-'):>9} "
                  f"{str(r['enum_children'] or '-'):>10} {pct:>6}  {r['verdict']}")
        bad = [r for r in rows if r["route_bearing"] and not r["n_routes"]]
        ready = [r for r in rows if r["verdict"] == "READY"]
        print("-" * 74)
        print(f"  {len(ready)} SPARROW-ready | {len(bad)} route-bearing cell-seeds with NO routes"
              f" | {len([r for r in rows if not r['route_bearing']])} control (n/a: {NA_REASON})")
        if bad:
            print("\n  Cells with no routes need a RE-SAMPLE -- the trajectory is not recoverable from"
                  "\n  existing artifacts (compositions.json keeps only num_reactions). Re-running the"
                  "\n  ENUMERATION alone does not help: routes.json is written by the SAMPLE stage.")
    return 1 if any(r["route_bearing"] and not r["n_routes"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
