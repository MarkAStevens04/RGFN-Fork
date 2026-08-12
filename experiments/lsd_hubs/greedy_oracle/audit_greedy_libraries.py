#!/usr/bin/env python
"""Severe test: is the greedy's cheaper library actually as good, or is it gaming the metric?

The greedy optimises (new modes)/(marginal reactions), and "new mode" is decided by a **greedy
sphere-exclusion** filter fed in acceptance order. That objective is order-dependent, so an optimiser
free to choose the order could in principle inflate its mode count without delivering a genuinely more
diverse library — exactly the numerator/denominator gaming the metric's threat model warns about.

Two independent checks on the delivered libraries themselves (not on the counts):

* **Reward-blind diversity.** Re-cluster each arm's library with Butina, which picks cluster centres
  by neighbourhood density and never reads the reward or the acceptance order. If the greedy's edge
  were an ordering artifact, its Butina cluster count would fall short of its nominal mode count by
  more than the flow walk's does. (Same instrument and the same 1e-9 threshold nudge as Logs/054, so
  the numbers are comparable to it.)
* **Reward parity.** Median and 10th-percentile reward of the delivered library. Reactions saved by
  delivering weaker molecules would not be a saving.

Both are computed on the FIRST `--modes` accepted molecules of each arm, so the libraries are
matched in size (Logs/051's set-size constraint) and the comparison is like-for-like.

    python experiments/lsd_hubs/greedy_oracle/audit_greedy_libraries.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from validation.lsdflow.metrics.diversity import count_butina_clusters  # noqa: E402

ARMS = ("best_candidate", "hub_batching", "greedy_oracle")
LABEL = {"best_candidate": "best-cand", "hub_batching": "flow", "greedy_oracle": "greedy"}


def _library(curve: Path, arm: str, n: int):
    rows = []
    with open(curve) as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == arm:
                rows.append((r["smiles"], float(r["reward"])))
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--modes", type=int, default=300, help="matched library size to audit")
    ap.add_argument("--cutoff", type=float, default=0.5)
    a = ap.parse_args()
    root = Path(a.results)

    out_rows = []
    print(f"\nMatched-library audit at n={a.modes}, Butina cutoff {a.cutoff} (reward-blind)\n")
    print(
        f"{'cell':<15} {'arm':<10} {'n':>4} {'butina':>7} {'b/n':>6} {'med reward':>11} {'p10 reward':>11}"
    )
    print("-" * 70)
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        curve = cell_dir / "curve.csv"
        summ = cell_dir / "summary.json"
        if not curve.exists() or not summ.exists():
            continue
        s = json.load(open(summ))
        hib = s["higher_is_better"]
        for arm in ARMS:
            lib = _library(curve, arm, a.modes)
            if not lib:
                continue
            smiles = [x for x, _ in lib]
            rewards = [r for _, r in lib]
            # Logs/054's boundary correction: the greedy filter rejects sim > c while Butina merges at
            # sim >= c, and a library built at cutoff c piles pairs onto it, so nudge by 1e-9.
            nb = count_butina_clusters(smiles, similarity_threshold=a.cutoff + 1e-9)
            srt = sorted(rewards, reverse=hib)
            p10 = srt[int(0.9 * (len(srt) - 1))]
            row = {
                "cell": cell_dir.name,
                "arm": arm,
                "n": len(lib),
                "butina_clusters": nb,
                "butina_per_molecule": round(nb / len(lib), 4),
                "median_reward": round(statistics.median(rewards), 3),
                "p10_reward": round(p10, 3),
            }
            out_rows.append(row)
            print(
                f"{cell_dir.name:<15} {LABEL[arm]:<10} {len(lib):>4} {nb:>7} "
                f"{nb/len(lib):>6.3f} {row['median_reward']:>11.3f} {row['p10_reward']:>11.3f}"
            )
        print("-" * 70)

    out = root / "library_audit.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    by_arm = {}
    for r in out_rows:
        by_arm.setdefault(r["arm"], []).append(r)
    print("\nAcross cells (median):")
    for arm in ARMS:
        rs = by_arm.get(arm, [])
        if not rs:
            continue
        print(
            f"  {LABEL[arm]:<10} butina/molecule {statistics.median(r['butina_per_molecule'] for r in rs):.3f}"
            f"   (n={len(rs)} cells)"
        )
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
