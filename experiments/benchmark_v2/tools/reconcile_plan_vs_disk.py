#!/usr/bin/env python
"""Reconcile `grid.csv`'s PLAN against what the v1 source tree actually holds.

WHO OWNS WHAT, BECAUSE THIS SCRIPT DELIBERATELY WRITES NOTHING. ``grid.csv`` is a BUILD PRODUCT of
``build_grid.py``, which carries the taxonomy and has a ``--check`` mode proving the committed file
still matches its definition -- so hand-editing it (an earlier version of this script did exactly
that) breaks ``--check`` and puts two sources of truth in the tree. ``build_grid.py`` states the
division plainly: *"``train_plan`` IS A PLAN, NOT AN OBSERVATION ... Re-derive it against reality
before relying on it -- the copy-forward step owns reconciling them."* This script is that
reconciliation. It prints a diff and a patch; a human moves the plan corrections into
``build_grid.py`` and regenerates.

AND IT IS NOT ``manifest.py`` EITHER. That reports live status of the **v2** tree -- has this cell
been copied in, does its ``trace.csv`` reach the budget, is it frozen. This one surveys the **v1
source** the copy would read FROM, before anything has been copied. Two different objects; the
overlap is zero and the ordering matters (survey the source, then copy, then let manifest watch it).

THE FINDING THAT MAKES THIS WORTH RUNNING RATHER THAN ASSUMING. Nine cells that every earlier
inventory reported as having no training history in fact have a complete one -- see
``inventory_v1_cells.best_trace``. Re-invoking a runner truncates ``trace.csv`` to its 59-byte
header while the real trace survives as ``trace.csv.1``. That has a direct consequence for the copy
step and it is easy to get wrong:

    THE COPY MUST RESOLVE THE REAL TRACE AND LAND IT AS ``trace.csv`` IN v2.

Copying the file *named* ``trace.csv`` would carry the stub, ``manifest.py`` would then correctly
report ``no-trace`` for a cell whose history we still hold, and Stage 2 would pay to re-sample a pool
it could have harvested for free (``saturn_clpp/s42``: 500 modes from history alone, 28 GPU-h -> 0).

    python experiments/benchmark_v2/tools/reconcile_plan_vs_disk.py
    python experiments/benchmark_v2/tools/reconcile_plan_vs_disk.py --patch   # build_grid.py edit
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


def survey() -> dict:
    """(generator, target, seed) -> inspection record, for the phase-1 competitor cells."""
    return {
        (g, t, s): inspect(g, t, s)
        for g in COMPETITORS
        for t in TARGETS
        for s in SEEDS
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grid", type=Path, default=GRID)
    ap.add_argument(
        "--patch",
        action="store_true",
        help="print the per-generator plan corrections to apply in build_grid.py",
    )
    a = ap.parse_args()

    if not a.grid.is_file():
        sys.exit(
            f"no grid at {a.grid}. It is a build product -- generate it first:\n"
            f"  python experiments/benchmark_v2/tools/build_grid.py"
        )

    rows = list(csv.DictReader(open(a.grid)))
    obs = survey()

    print("PLAN vs DISK -- phase-1 competitor cells (the only ones a copy can read from)\n")
    print(f"{'cell':<26} {'plan':>9}  {'disk':<9} evidence")
    print("-" * 104)
    agree = disagree = 0
    for row in rows:
        gen, tgt, seed = row["generator"], row["target"], int(row["seed"])
        if tgt == PHASE2 or gen in REACTION_GFNS:
            continue  # nothing on disk to reconcile against: never run / superseded by the budget
        r = obs.get((gen, tgt, seed))
        if r is None:
            continue
        plan, disk = row.get("train_plan", "?"), r["verdict"]
        same = plan == disk
        agree += same
        disagree += not same
        if not same:
            print(f"{gen}/{tgt}/s{seed:<18} {plan:>9}  {disk:<9} {r['why']}")
    print(f"\n{agree} agree, {disagree} disagree.")

    if disagree and not a.patch:
        print("\nRe-run with --patch for the build_grid.py corrections. Do NOT edit grid.csv:")
        print("it is generated, and a hand-edit fails `build_grid.py --check`.")

    if a.patch:
        print("\n--- corrections for build_grid.py's plan table -------------------------------")
        for gen in COMPETITORS:
            verdicts = {}
            for tgt in TARGETS:
                for seed in SEEDS:
                    v = obs[(gen, tgt, seed)]["verdict"]
                    verdicts[v] = verdicts.get(v, 0) + 1
            dominant = max(verdicts, key=verdicts.get)
            detail = ", ".join(f"{v}={n}" for v, n in sorted(verdicts.items()))
            print(f"  {gen:<10} phase-1 plan -> {dominant:<9}  ({detail})")
        print(
            "\nWhere a generator is not unanimous the exceptions are per-cell and belong in the\n"
            "note column, not the plan: a plan is what we intend for the generator, and the\n"
            "per-cell reality is what this script and manifest.py report."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
