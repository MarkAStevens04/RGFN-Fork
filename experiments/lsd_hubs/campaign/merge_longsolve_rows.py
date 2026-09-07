#!/usr/bin/env python
"""Merge re-solved R=100 rows from `<tag>_select_N500_longsolve/` back into `<tag>_select_N500/`.

WHY THIS EXISTS. Re-solving a time-capped SPARROW row writes to a SEPARATE directory so the original
ten-budget CSV survives (submit_reprice_cached.sh, ARM=sb). That protects the other nine budget
points but leaves TWO sources of truth for the PRIMARY readout: the main CSV still carries the stale
`TimeLimit` row at R=100 while only the sibling has the certified one. This is not hypothetical --
an audit script written in the same session globbed `*_select_N*`, matched both directories, and
re-reported three already-certified rows as still capped. Any reader that does not know to prefer
`_longsolve` quotes the lower bound.

WHAT IT DOES. For each cell with a `_longsolve` CSV, replaces the matching `budget_rxns` row in the
main CSV, and appends a `gap_rel` column recording the relative MIP gap each row was solved at.
That column is the point: `solve_s` dropping from ~1800 to <10 beside `milp_status=Optimal` is the
only existing in-file trace of a re-solve, and NOTHING in the schema records that the new rows are
certified to 1% while the untouched ones are at SPARROW's hardcoded 1e-7. Appending at the END keeps
both `csv.DictReader` and the positional `awk` readers used across this campaign working unchanged.

WHAT IT REFUSES TO DO.
  * It will not merge a row that is still `time_capped` -- replacing one lower bound with another
    gains nothing and would hide that the cell is still uncertified.
  * It will not merge a row whose mode count DISAGREES with the capped row unless --allow-change is
    given. Every re-solve measured so far returned the identical count, so a changed count means
    something other than the gap moved and deserves a human look, not a silent overwrite.

A later SB re-run through the chain will overwrite a merged row with a fresh 1e-7 solve, which may
cap again. That is correct behaviour, not a regression -- and the `gap_rel` column is what keeps the
difference visible when it happens.
"""
import argparse
import csv
import os
import shutil
from pathlib import Path

SPARROW_DEFAULT_GAP = "1e-07"


def rows_of(path):
    with path.open() as f:
        r = csv.DictReader(f)
        return list(r), (r.fieldnames or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--gap-rel", default="0.01",
                    help="gap the _longsolve rows were solved at; recorded, not applied")
    ap.add_argument("--allow-change", action="store_true",
                    help="merge even when the re-solved mode count differs from the capped one")
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry-run")
    a = ap.parse_args()

    root = Path(a.results_root or (Path(os.environ["SCRATCH"]) / "rgfn_runs" / "lsdflow_sparrow"
                                   / "results"))
    merged = skipped = 0
    for ls in sorted(root.glob("*_select_N*_longsolve")):
        main_dir = root / ls.name[: -len("_longsolve")]
        src, dst = ls / "select_frontier.csv", main_dir / "select_frontier.csv"
        if not src.exists() or not dst.exists():
            print(f"SKIP {ls.name} — missing CSV on one side")
            skipped += 1
            continue

        new_rows, _ = rows_of(src)
        cur_rows, cur_fields = rows_of(dst)
        by_budget = {r.get("budget_rxns"): r for r in new_rows}

        changes = []
        for i, r in enumerate(cur_rows):
            nr = by_budget.get(r.get("budget_rxns"))
            if nr is None:
                continue
            if str(nr.get("time_capped")) == "True":
                print(f"  {main_dir.name} R={r['budget_rxns']}: re-solve STILL CAPPED, not merging")
                continue
            old_m, new_m = r.get("n_modes_kept"), nr.get("n_modes_kept")
            if old_m != new_m and not a.allow_change:
                print(f"  {main_dir.name} R={r['budget_rxns']}: modes {old_m} -> {new_m} CHANGED; "
                      f"refusing without --allow-change")
                continue
            changes.append((i, nr, old_m, new_m, r.get("solve_s"), nr.get("solve_s")))

        if not changes:
            skipped += 1
            continue

        for i, nr, om, nm, os_, ns in changes:
            print(f"  {main_dir.name} R={nr['budget_rxns']}: {om} modes, "
                  f"{cur_rows[i].get('milp_status')} {float(os_ or 0):.0f}s -> "
                  f"{nm} modes, {nr.get('milp_status')} {float(ns or 0):.1f}s")
            cur_rows[i] = nr

        fields = cur_fields if "gap_rel" in cur_fields else cur_fields + ["gap_rel"]
        merged_idx = {i for i, *_ in changes}
        for i, r in enumerate(cur_rows):
            r.setdefault("gap_rel", "")
            if not r["gap_rel"]:
                r["gap_rel"] = a.gap_rel if i in merged_idx else SPARROW_DEFAULT_GAP

        if a.apply:
            bak = dst.with_suffix(".csv.pre_merge")
            if not bak.exists():
                shutil.copy2(dst, bak)
            with dst.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(cur_rows)
        merged += 1

    verb = "merged" if a.apply else "would merge"
    print(f"\n{verb} {merged} cells; {skipped} skipped")
    if not a.apply:
        print("dry run — pass --apply to write (originals kept as select_frontier.csv.pre_merge)")


if __name__ == "__main__":
    main()
