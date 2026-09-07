#!/usr/bin/env python
"""Render the depth-mix arms as one table. Reads the per-arm JSON depth_mix.py wrote."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = [
    ("base", "--min-synth-depth 0 (everything)"),
    ("d2", "--min-synth-depth 2 (drops buy-and-couple)"),
    ("d3", "--min-synth-depth 3 (stricter)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    a = ap.parse_args()

    print(f"{'arm':<6} {'msd':>4} | {'HB modes@100':>12} {'hubs':>5} {'hub depths':>16} "
          f"{'modes by depth':>20} {'depth-0 share':>13} | {'BC modes@100':>12} {'BC unjoined':>11}")
    print("-" * 122)
    for name, _label in ARMS:
        p = a.results / f"depthmix_{name}.json"
        if not p.exists():
            print(f"{name:<6} {'-':>4} | (missing {p.name})")
            continue
        d = json.loads(p.read_text())
        h = d.get("hub_batching", {})
        b = d.get("best_candidate", {})
        share = h.get("share_modes_on_depth0")
        print(
            f"{name:<6} {str(d.get('min_synth_depth')):>4} | "
            f"{h.get('modes_at_budget', '-'):>12} {h.get('distinct_hubs_walked', '-'):>5} "
            f"{str(h.get('hub_depth_of_walked_hubs', {})):>16} "
            f"{str(h.get('modes_by_hub_depth', {})):>20} "
            f"{(f'{share:.1%}' if share is not None else '-'):>13} | "
            f"{b.get('modes_at_budget', '-'):>12} {b.get('unjoined_hubs', '-'):>11}"
        )
    print()
    base = a.results / "depthmix_base.json"
    if base.exists():
        d = json.loads(base.read_text())
        print(f"full enumerated hub set ({d.get('n_hubs_in_hubs_csv')} hubs), depth histogram: "
              f"{d.get('depth_of_full_hub_set')}")
        print("NOTE: this is the v1 REWARD-PRE-FILTERED hub set (TOPK=1000), not `--pool all`, and it")
        print("      runs on a 320,000-call checkpoint. It answers the DEPTH question, not viability.")


if __name__ == "__main__":
    main()
