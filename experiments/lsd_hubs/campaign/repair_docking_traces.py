#!/usr/bin/env python
"""Repair docking trace.csv files that recorded the SHAPED value instead of raw Vina.

WHAT WENT WRONG. `trace_from_saturn` reads `glue_surrogate_raw_values` and `trace_from_reinvent`
reads `<name> (raw)`. Both are "raw" from the ORACLE COMPONENT's point of view -- correct for the
sEH/DRD2 surrogates, wrong for docking, where the component returns `clip(-vina/norm, 0, inf)`, the
value the GFN trains on. Measured 2026-08-28: nine ClpP traces (REINVENT / Saturn / TANGO x3 seeds)
held positive 0..17 values with a median near 10, and NOT ONE row cleared the -8.0 gate. Every one of
those cells' training histories therefore looked empty -- which matters now that Stage 2 harvests the
training history as a free starting pool.

NO RE-DOCKING IS NEEDED. The transform is invertible where it is not clipped: `raw = -value * norm`
for `value > 0`. Verified against candidates.csv, which carries both columns: 1984/2000 rows satisfy
`raw == -score` exactly, and all 16 exceptions are `score == 0`. So this is a few seconds of
arithmetic per cell rather than ~14 GPU-hours of re-docking each (~125 GPU-hours across nine cells).

`value == 0` IS CENSORED and is written as NaN, not 0.0. The clip floor means raw was >= 0 or the
dock failed; either way the molecule cannot clear a negative gate, and a fabricated 0.0 would look
like a real measurement sitting just above the bar.

S3-GFN and SynFormer traces are ALREADY CORRECT (their adapters call `provider.raw_scores()`), so the
detector below refuses to touch a trace whose values are already mostly negative. Run with --check to
report without writing.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics as st
from pathlib import Path

FIELDS = ["n_scored", "n_distinct", "phase", "step", "smiles", "raw_score", "elapsed_s"]


def classify(path: Path):
    """(n_rows, median, verdict) -- 'shaped', 'raw', or 'unknown'."""
    vals = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                vals.append(float(r["raw_score"]))
            except (TypeError, ValueError, KeyError):
                continue
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return 0, None, "unknown"
    med = st.median(vals)
    # A docking trace of real Vina energies is mostly NEGATIVE. Positive median => shaped.
    return len(vals), med, ("shaped" if med > 0 else "raw")


def repair(path: Path, norm: float, dry: bool) -> dict:
    n, med, verdict = classify(path)
    out = dict(path=str(path), rows=n, median=med, verdict=verdict, changed=0, censored=0)
    if verdict != "shaped":
        return out
    if dry:
        return out
    rows = list(csv.DictReader(open(path, newline="")))
    shutil.copy2(path, path.with_suffix(".csv.preunshape"))  # keep the original, always
    changed = censored = 0
    for r in rows:
        try:
            v = float(r["raw_score"])
        except (TypeError, ValueError):
            continue
        if v > 0.0:
            r["raw_score"] = repr(-v * norm)
            changed += 1
        else:
            r["raw_score"] = "nan"  # clip-censored: NOT a real 0.0
            censored += 1
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    out.update(changed=changed, censored=censored)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="/scratch/markymoo/rgfn_runs/experiments/fixed_reward")
    ap.add_argument("--target", default="clpp", help="only docking targets need this")
    ap.add_argument("--norm", type=float, default=1.0, help="reward.norm from the cell's config")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    root = Path(a.root)
    paths = sorted(root.glob(f"*_{a.target}/seed*/trace.csv"))
    print(
        f"  {'cell':<26} {'rows':>7} {'median':>9} {'verdict':>8}  {'changed':>8} {'censored':>9}"
    )
    tot = 0
    for p in paths:
        r = repair(p, a.norm, a.check)
        cell = f"{p.parent.parent.name}/{p.parent.name}"
        med = "-" if r["median"] is None else f"{r['median']:.2f}"
        print(
            f"  {cell:<26} {r['rows']:>7} {med:>9} {r['verdict']:>8}  "
            f"{r['changed']:>8} {r['censored']:>9}"
        )
        tot += 1 if r["changed"] else 0
    print(
        f"\n  {'CHECK ONLY — nothing written' if a.check else f'repaired {tot} trace(s); originals kept as *.csv.preunshape'}"
    )


if __name__ == "__main__":
    main()
