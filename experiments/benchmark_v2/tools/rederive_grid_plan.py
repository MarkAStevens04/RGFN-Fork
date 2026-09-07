#!/usr/bin/env python
"""Rewrite `grid.csv`'s ``train_plan`` from what is on disk, not from what we remembered.

RUN THIS RATHER THAN EDITING grid.csv BY HAND. The column was first written on 2026-08-28 from a
one-moment inventory, and disk overtook it within days: FragGFN was retrained at the normalized
budget, SynFormer went 1 of 9 cells to 9 of 9, and S3-GFN's traces were truncated by a runner
re-invocation. Cells are still landing, so any hand-maintained plan is stale the day after it is
written. This script is idempotent and cheap — re-run it before every planning decision.

WHAT IT SETS. ``train_plan`` becomes one of four values, and the distinction between the middle two
is the one that costs real money if collapsed:

  copy      the cell is clean at the arm-A budget and travels as-is.
  resample  the POOL is missing but the checkpoint and trace survive. One sampling pass from the
            frozen policy — minutes. Calling this "generate" buys a re-train nobody needs.
  attention copyable, but DEGRADED: its trace is gone or short, so stage 2 loses the free-pool
            harvest and must sample instead. A COST, not a correctness failure — the cell is still
            usable and must not be quietly dropped.
  generate  genuinely needs a training run: no artifacts, over budget, or never run at all.

Phase-2 (6TD3-B) cells are always ``generate`` — no generator has ever been run against that reward.
The three reaction-GFNs are always ``generate`` — the budget standardisation superseded every v1
cell, and none of them emits a trace yet (runbook §7.2).

    python experiments/benchmark_v2/tools/rederive_grid_plan.py            # rewrite grid.csv
    python experiments/benchmark_v2/tools/rederive_grid_plan.py --dry-run  # show the diff only
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID = HERE.parent / "grid.csv"
sys.path.insert(0, str(HERE))
from inventory_v1_cells import (  # noqa: E402
    COMPETITORS,
    REACTION_GFNS,
    SEEDS,
    TARGETS,
    inspect,
)

PHASE2 = "6td3b"


def derive() -> dict:
    """(generator, target, seed) -> (train_plan, note)."""
    plan = {}
    for gen in COMPETITORS:
        for tgt in TARGETS:
            for seed in SEEDS:
                r = inspect(gen, tgt, seed)
                plan[(gen, tgt, seed)] = (r["verdict"], r["why"])
    return plan


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--grid", type=Path, default=GRID)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.grid)))
    plan = derive()

    changed = []
    for row in rows:
        gen, tgt, seed = row["generator"], row["target"], int(row["seed"])
        old = row["train_plan"]
        if tgt == PHASE2:
            new, why = "generate", "6TD3-B has never been run for any generator"
        elif gen in REACTION_GFNS:
            new, why = "generate", (
                "budget standardisation supersedes every v1 cell; "
                "no trace emitted yet (runbook 7.2)"
            )
        else:
            new, why = plan.get((gen, tgt, seed), ("generate", "not inspected"))
        if old != new:
            changed.append((gen, tgt, seed, old, new, why))
        row["train_plan"], row["note"] = new, why

    for gen, tgt, seed, old, new, why in changed:
        print(f"  {gen:<10} {tgt:<6} s{seed}  {old:>8} -> {new:<9} {why}")
    print(f"\n{len(changed)} of {len(rows)} rows change.")
    tally = {}
    for row in rows:
        tally[row["train_plan"]] = tally.get(row["train_plan"], 0) + 1
    print("new plan:", ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    if a.dry_run:
        print("\n--dry-run: grid.csv not written")
        return 0
    with open(a.grid, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.grid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
