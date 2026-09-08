#!/usr/bin/env python
"""Do library-less reaction-GFNs hub at exactly (reaction cap - 1)?

THE CLAIM UNDER TEST. On one cell (sEH seed 42) the flow field's top-ranked hubs sit at depth
(cap - 1) for the two generators with NO dynamic library, and much shallower for the one that has
one:

    rgfn     cap 4  -> depth 3   one step from terminal
    rxnflow  cap 3  -> depth 2   one step from terminal
    scent    cap 4  -> depth 1   three steps from terminal

If that is structural rather than one cell's accident, it says something sharp: a dynamic library is
what lets a reaction-GFN hub EARLY, which is the only place there is anything to amortise. Without
one, reward mass concentrates one step before the end -- the worst possible place to batch from.

This runs `pick_hubs --pool all` over every cell whose sample survives and prints the top-40 depth
histogram per cell, so the claim is tested on the grid rather than asserted from a single cell.
`pick_hubs` is pure stdlib and takes ~0.2 s, so the whole grid costs seconds.

EACH CELL GETS ITS OWN OUTPUT DIRECTORY. `pick_hubs` writes `hub_scores.csv` and
`pick_hubs_timing.json` to a FIXED name in the parent of `--out`, so two runs sharing a parent
silently clobber each other's provenance -- which is how a stale hub_scores.csv is produced.

FragGFN is reported but is NOT a test of the claim: its moves are fragment attachments, not
reactions, so its "depth" is not a reaction count and it has no reaction cap in the same sense.

Usage:
  python experiments/benchmark_v2/pilot/hub_depth_grid.py --out-root /scratch/.../hub_depth_grid
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PICK = REPO / "experiments" / "lsd_hubs" / "campaign" / "pick_hubs.py"
TREES = {
    "/scratch/markymoo/rgfn_runs/lsdflow/matrix16": 42,
    "/scratch/markymoo/rgfn_runs/lsdflow/matrix16_seed43": 43,
    "/scratch/markymoo/rgfn_runs/lsdflow/matrix16_seed44": 44,
}
# Per-generator reaction cap. The claim is about depth RELATIVE to this, so it is the load-bearing
# constant here -- sourced from each generator's own config, not guessed.
CAPS = {"rgfn": 4, "rxnflow": 3, "scent": 4}
HAS_LIBRARY = {"scent"}
# Docking targets gate on raw energy (lower better); surrogates on the proxy value.
LOWER_IS_BETTER = {"6td3", "clpp"}


def run_pick(records: Path, out: Path, higher: bool) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(PICK), "--records", str(records), "--out", str(out),
           "--pool", "all", "--order", "flow_desc", "--n-hubs", "200",
           "--higher-is-better", "true" if higher else "false"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pick_hubs failed for {records}: {r.stderr[-400:]}")
    return out.parent / "hub_scores.csv"


def top_depths(scores: Path, n: int = 40):
    with open(scores) as fh:
        rows = [(int(r["rank"]), int(r["depth"])) for r in csv.DictReader(fh)]
    rows.sort()
    return dict(sorted(Counter(d for _, d in rows[:n]).items())), len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    results = []
    for tree, seed in TREES.items():
        for sample in sorted(Path(tree).glob("*/sample/records.csv")):
            cell = sample.parent.parent.name
            gen = cell.rsplit("_", 1)[0] if "_" in cell else cell
            tgt = cell.rsplit("_", 1)[-1]
            if "cap9" in cell:  # a deliberate sensitivity variant, not a matrix cell
                continue
            out = a.out_root / f"{cell}_seed{seed}" / "hubs.csv"
            try:
                scores = run_pick(sample, out, higher=tgt not in LOWER_IS_BETTER)
                hist, n = top_depths(scores, a.top_n)
            except Exception as exc:  # noqa: BLE001
                results.append({"cell": cell, "seed": seed, "error": str(exc)[:120]})
                continue
            cap = CAPS.get(gen)
            deepest = max(hist) if hist else None
            mode_depth = max(hist, key=hist.get) if hist else None
            results.append({
                "cell": cell, "generator": gen, "target": tgt, "seed": seed,
                "cap": cap, "has_library": gen in HAS_LIBRARY,
                "n_ranked": n, "top_depth_hist": hist,
                "modal_depth": mode_depth,
                "modal_is_cap_minus_1": (cap is not None and mode_depth == cap - 1),
            })

    print(f"{'cell':<16} {'seed':>4} {'cap':>4} {'lib':>4} {'modal':>6} {'=cap-1':>7}  top{a.top_n} depth histogram")
    print("-" * 96)
    for r in sorted(results, key=lambda x: (x.get("generator", ""), x.get("target", ""), x["seed"])):
        if "error" in r:
            print(f"{r['cell']:<16} {r['seed']:>4}  ERROR {r['error']}")
            continue
        print(f"{r['cell']:<16} {r['seed']:>4} {str(r['cap']):>4} {('yes' if r['has_library'] else 'no'):>4} "
              f"{str(r['modal_depth']):>6} {str(r['modal_is_cap_minus_1']):>7}  {r['top_depth_hist']}")

    testable = [r for r in results if "error" not in r and r.get("cap")]
    lib_less = [r for r in testable if not r["has_library"]]
    with_lib = [r for r in testable if r["has_library"]]
    print()
    if lib_less:
        hit = sum(1 for r in lib_less if r["modal_is_cap_minus_1"])
        print(f"library-less cells hubbing at modal depth == cap-1: {hit}/{len(lib_less)}")
    if with_lib:
        hit = sum(1 for r in with_lib if r["modal_is_cap_minus_1"])
        print(f"library-bearing (SCENT) cells at cap-1:             {hit}/{len(with_lib)}")
    print("FragGFN is excluded from the test: its moves are attachments, not reactions.")

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
