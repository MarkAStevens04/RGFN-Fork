#!/usr/bin/env python
"""Reconcile mode-vs-median hub depth, and report CONCENTRATION beside the central value.

WHY. Two agents read the same cells and published different depths for `scent_clpp` and
`scent_6td3` -- 3 from a mode, 2 from a median. Both were computed correctly; they are different
statistics of a distribution that is not concentrated enough for the difference to be cosmetic.

AND THAT IS THE ACTUAL POINT. "Modal depth 3" reads the same whether the mode holds 75% of the top
40 or 37%, and those are completely different claims about where a flow field puts its hubs. The
(cap - 1) statement is a claim about CONCENTRATION, so the share has to travel with it -- a 37% mode
is a spread distribution being described by its largest bin, not a field that hubs at cap - 1.

Prints mode, mode share, median, and whether the two agree, so a disagreement is visible rather than
resolved by whichever statistic was reached for first.

Usage:
  python experiments/benchmark_v2/pilot/reconcile_depth.py \
      --grid experiments/benchmark_v2/pilot/results/hub_depth_grid.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# A mode holding less than this much of the top-40 is not describing a concentrated field.
CONCENTRATED = 0.50


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    rows = json.loads(a.grid.read_text())
    out = []
    print(f"{'cell':<14} {'seed':>4} {'cap':>4} {'mode':>5} {'share':>6} {'median':>7} "
          f"{'agree':>6} {'concentrated':>13}  histogram")
    print("-" * 104)
    for r in sorted(rows, key=lambda x: (x.get("generator", ""), x.get("target", ""), x["seed"])):
        if "error" in r or r.get("generator") == "fraggfn":
            continue
        hist = {int(k): v for k, v in r["top_depth_hist"].items()}
        vals = [d for d, c in sorted(hist.items()) for _ in range(c)]
        if not vals:
            continue
        mode = max(hist, key=hist.get)
        share = hist[mode] / len(vals)
        med = statistics.median(vals)
        rec = {
            "cell": r["cell"], "seed": r["seed"], "generator": r.get("generator"),
            "target": r.get("target"), "cap": r.get("cap"),
            "modal_depth": mode, "modal_share": round(share, 3),
            "median_depth": med, "mode_median_agree": mode == med,
            "concentrated": share >= CONCENTRATED,
            "hist": r["top_depth_hist"],
        }
        out.append(rec)
        print(f"{r['cell']:<14} {r['seed']:>4} {str(r.get('cap')):>4} {mode:>5} {share:>5.0%} "
              f"{med:>7} {str(mode == med):>6} {str(share >= CONCENTRATED):>13}  {r['top_depth_hist']}")

    print()
    disagree = [r for r in out if not r["mode_median_agree"]]
    spread = [r for r in out if not r["concentrated"]]
    def name(rs):
        return ", ".join("{}/{}".format(r["cell"], r["seed"]) for r in rs) or "none"

    print("mode != median on {}/{} cells: {}".format(len(disagree), len(out), name(disagree)))
    print("NOT concentrated (mode < {:.0%} of top-40) on {}/{}: {}".format(
        CONCENTRATED, len(spread), len(out), name(spread)))
    print()
    print("A cell in the second list should NOT be described by its modal depth: the field is spread,")
    print("not hubbing at one depth. The (cap-1) claim only means something for concentrated cells.")

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
