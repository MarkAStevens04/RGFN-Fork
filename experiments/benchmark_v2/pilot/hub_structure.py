#!/usr/bin/env python
"""Does an arm-A checkpoint's flow field carry usable hub structure, or is it noise?

THE VIABILITY QUESTION. Arm A gives a reaction-GFN ~100-157 gradient steps. Hub-batching reads the
trained FLOW FIELD, so the question is not "did training finish" but "is the ranking that field
produces separated enough to act on". Two things decide that and both come from pick_hubs' own
sidecar, before any enumeration is paid for:

  * POPULATION -- how many hubs the walk can even see, and how many estimates back each one. A hub
    scored from one observed child is a point estimate; the campaign walks ~5 hubs at R=100, so what
    matters is whether the TOP of the ranking is well-evidenced, not the tail.
  * SEPARATION -- whether log-flow actually discriminates. A field that has learned nothing gives a
    flat ranking, and "flow-descending" then means nothing. The honest test is the spread across the
    ranks the walk will really touch, against the spread of the pool as a whole.

Compares as many hub_scores.csv files as you pass, so an arm-A field can be read beside a v1
(320,000-call) one on the same axis.

Usage:
  python experiments/benchmark_v2/pilot/hub_structure.py --label armA:<dir>/hub_scores.csv \
                                                          --label v1:<dir>/hub_scores.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def load(path: Path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        try:
            out.append((int(r["rank"]), float(r["log_flow"]), int(r["depth"]), int(r["n_estimates"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def quant(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def summarise(name: str, rows) -> dict:
    flows = [f for _, f, _, _ in rows]
    top = rows[:10]
    walk = rows[:40]  # the campaign walks ~5 hubs at R=100; 40 is a generous envelope
    d = {
        "label": name,
        "n_hubs_ranked": len(rows),
        "log_flow": {
            "max": round(max(flows), 3) if flows else None,
            "p50": round(quant(flows, 0.5), 3) if flows else None,
            "min": round(min(flows), 3) if flows else None,
        },
        # SEPARATION: how far the top of the ranking sits above the median, in units of the pool's
        # own spread. A flat (unlearned) field puts this near zero.
        "top10_minus_median_nats": round(
            (sum(f for _, f, _, _ in top) / len(top)) - quant(flows, 0.5), 3
        ) if top and flows else None,
        "rank0_minus_rank40_nats": round(rows[0][1] - rows[min(39, len(rows) - 1)][1], 3) if rows else None,
        "depth_hist_top200": dict(sorted(Counter(dep for _, _, dep, _ in rows).items())),
        "depth_hist_top40": dict(sorted(Counter(dep for _, _, dep, _ in walk).items())),
        "n_estimates_top10_median": quant([n for _, _, _, n in top], 0.5),
        "n_estimates_top40_median": quant([n for _, _, _, n in walk], 0.5),
        "n_estimates_all_median": quant([n for _, _, _, n in rows], 0.5),
        "frac_top40_single_estimate": round(
            sum(1 for _, _, _, n in walk if n <= 1) / len(walk), 3
        ) if walk else None,
    }
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", action="append", required=True,
                    help="NAME:/path/to/hub_scores.csv (repeatable)")
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    reports = []
    for spec in a.label:
        name, _, path = spec.partition(":")
        p = Path(path)
        if not p.exists():
            print(f"[hub_structure] MISSING {name}: {p}")
            continue
        reports.append(summarise(name, load(p)))

    print(f"{'label':<10} {'ranked':>7} {'max lF':>9} {'p50 lF':>9} "
          f"{'top10-med':>10} {'r0-r40':>8} {'est@top40':>10} {'1-est@top40':>12} {'depth@top40'}")
    print("-" * 112)
    for r in reports:
        print(f"{r['label']:<10} {r['n_hubs_ranked']:>7} {str(r['log_flow']['max']):>9} "
              f"{str(r['log_flow']['p50']):>9} {str(r['top10_minus_median_nats']):>10} "
              f"{str(r['rank0_minus_rank40_nats']):>8} {str(r['n_estimates_top40_median']):>10} "
              f"{str(r['frac_top40_single_estimate']):>12} {r['depth_hist_top40']}")
    print()
    print("top10-med / r0-r40 are in NATS of log-flow: near zero means the ranking is flat and")
    print("'flow-descending' is not selecting anything. 1-est@top40 is the fraction of walked hubs")
    print("resting on a SINGLE observed child, i.e. an unreplicated point estimate.")

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
