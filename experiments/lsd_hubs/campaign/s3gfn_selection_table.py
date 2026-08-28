#!/usr/bin/env python
"""Greedy vs SPARROW-Batching on the SAME retrosynthesis, at a fixed REACTION budget.

WHAT THIS ANSWERS. One generator's pool is routed once by MultiAiZ, and then two different selectors
spend the same budget on those same routes: a diversity-aware GREEDY selection (SPARROW prices what
greedy picked) and SPARROW-Batching (SPARROW both picks and prices). The question is the project's
headline one -- "I have R reactions, how many distinct high-reward molecules do I get?" -- asked of
the two selectors rather than of two generators.

WHY THE TWO ARMS NEED CARE TO COMPARE. They are recorded on DIFFERENT axes. SPARROW sweeps a reaction
budget and reports modes; greedy sweeps a mode target and reports reactions. Reading them off the same
row number would compare a 25-mode greedy point against a 50-reaction SPARROW point. This inverts the
greedy ladder instead: the greedy answer at R is the LARGEST mode point whose used_rxns fits in R.
That is exactly how a chemist reads it, and it is the same ordering both arms already produce.

FOUR NUMBERS PER CELL, and all four are load-bearing (convention agreed 2026-08-27):

  modes_at_R     what the selector delivered inside the budget
  modes_makeable how many of the pool's modes can actually be synthesized from the route network.
                 `routed` only means "has route entries"; MultiAiZ promotes its own discovered
                 intermediates to stock, so a routed target can still have no path to real ZINC.
                 Quoting modes_at_R without this overstates a pool with poor route coverage.
  pool_size      s3gfn's above-gate pools are TINY (sEH pruned: 89/28/60 molecules). A reader who
                 sees "N modes at R reactions" and assumes a 500-molecule pool is being misled.
  used_rxns      separates budget-binding from pool-exhausted. Only budget-binding cells are
                 like-for-like; a pool-exhausted cell must never be scored as a cost win or loss.

STOP REASON is derived, not assumed:
  budget-binding   used_rxns is within `--slack` of R and candidates remained
  pool-exhausted   the selector ran out of qualifying candidates with budget unspent
  solver-truncated SPARROW's MILP hit its wall clock (time_capped) -- a LOWER bound on the
                   competitor, i.e. it flatters us, and must not be read as a converged optimum

Run:
    python experiments/lsd_hubs/campaign/s3gfn_selection_table.py --budget 100
    python experiments/lsd_hubs/campaign/s3gfn_selection_table.py --generator saturn --budget 100
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow")
POOLS = ROOT / "multiaiz_pools"
RESULTS = ROOT / "results"


def _rows(path: Path):
    try:
        return list(csv.DictReader(open(path)))
    except OSError:
        return []


def _num(v):
    """CSV cells are strings; empty/None means the solve produced no value, which is not zero."""
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def pool_info(tag: str):
    """(pool_dir, pool_size, pool_limited) for a cell tag, or (None, None, None)."""
    cands = sorted(POOLS.glob(f"{tag}_N[0-9]*"))
    cands = [d for d in cands if "OLDBUDGET" not in d.name]
    if not cands:
        return None, None, None
    d = cands[0]
    size = (
        len([l for l in open(d / "pool.smi") if l.strip()]) if (d / "pool.smi").exists() else None
    )
    limited = None
    if (d / "pool_meta.json").exists():
        try:
            limited = json.load(open(d / "pool_meta.json")).get("pool_limited")
        except (OSError, json.JSONDecodeError):
            pass
    return d, size, limited


def greedy_at_budget(tag: str, budget: float):
    """Largest greedy mode point whose used_rxns fits the budget, plus what the arm could see.

    Returns (modes, used_rxns, modes_makeable, n_points_priced). modes_makeable comes from the
    frontier's own rows: the arm skips mode points above the synthesizable count, so the largest
    priced point is a LOWER bound on it, and that is the honest thing to report.
    """
    rows = _rows(RESULTS / f"{tag}_greedy_N500" / "greedy_frontier.csv")
    priced = [r for r in rows if _num(r.get("used_rxns")) is not None]
    if not priced:
        return None, None, None, 0
    fits = [r for r in priced if _num(r["used_rxns"]) <= budget]
    best = max(fits, key=lambda r: int(r["n_modes"])) if fits else None
    largest = max(int(r["n_modes"]) for r in priced)
    if best is None:
        return 0, 0, largest, len(priced)

    # GRANULARITY WARNING. Greedy is swept on a MODE ladder, so at a reaction budget it can only
    # report a rung. If the next rung overshoots R slightly, the reported number understates greedy
    # badly -- measured on s3gfn_clpp_seed43_pruned: m=25 costs 52 rxns and m=50 costs 101, so at
    # R=100 this returns 25 when the true answer is ~49. That is a 2x understatement of the
    # competitor's strongest arm, in a table whose whole point is greedy vs SPARROW, so it is
    # flagged rather than silently returned. Re-run the arm with a finer --mode-points to remove it.
    nxt = [r for r in priced if _num(r["used_rxns"]) > budget]
    if nxt:
        n0 = min(nxt, key=lambda r: _num(r["used_rxns"]))
        over = _num(n0["used_rxns"]) - budget
        # Fire on the MODE gap, not the reaction overshoot. A fine ladder legitimately has its next
        # rung just past the budget -- clpp_43_pruned on a 5-mode ladder gives 45@95 and 50@101, so
        # 45 is the answer to within 5 modes and there is nothing to warn about. The artefact is a
        # COARSE ladder: 25 modes of uncertainty is half the answer, which is what the original
        # 25-rung sweep produced. Warning on overshoot alone cried wolf on exactly the runs that
        # had just been fixed, which would train the reader to ignore the flag.
        mode_gap = int(n0["n_modes"]) - int(best["n_modes"])
        if over <= 0.25 * budget and mode_gap > 10:
            print(
                f"    ! {tag}: next greedy rung is m={n0['n_modes']} at {_num(n0['used_rxns']):.0f} "
                f"rxns, only {over:.0f} over R={budget:.0f} and {mode_gap} modes above the reported "
                f"{best['n_modes']}. LADDER ARTEFACT — re-run with finer --mode-points."
            )
    return int(best["n_modes"]), int(_num(best["used_rxns"])), largest, len(priced)


def sparrow_at_budget(tag: str, budget: float):
    """SPARROW row at this budget: (modes, used, cost_kept, status, capped, n_selected)."""
    rows = _rows(RESULTS / f"{tag}_select_N500" / "select_frontier.csv")
    hit = [r for r in rows if _num(r.get("budget_rxns")) == budget]
    if not hit:
        return (None,) * 6
    r = hit[0]
    modes = _num(r.get("n_modes_kept"))
    if modes is None:  # older runs wrote n_modes; never mix the two in one table
        modes = _num(r.get("n_modes"))
    return (
        int(modes) if modes is not None else None,
        _num(r.get("used_rxns")),
        _num(r.get("cost_kept_rxns")),
        r.get("milp_status"),
        str(r.get("time_capped")).lower() == "true",
        _num(r.get("n_selected")),
    )


def stop_reason(used, budget, slack, capped, exhausted_hint, cost_kept=None, n_selected=None):
    """Classify the SB row, using the used-vs-cost_kept GAP rather than trusting used_rxns.

    The SB arm does not merely leave the budget unspent when it exhausts the pool: its reported
    used_rxns INFLATES, climbing with the budget while the true cost of the identical selected set
    stays flat (measured 2026-08-20 on Saturn's pruned sEH cell -- 65 molecules priced at 247
    reactions while used_rxns read 300->387 across R=300..1000). So `used >= budget - slack` alone
    calls an exhausted cell budget-binding, which is exactly the misreading the convention warns
    about: it implies the cell could have spent R and chose not to.

    The gap IS the detector. When cost_kept is far below used, the selection is pool-exhausted no
    matter what used_rxns says, and only cost_kept may be quoted.
    """
    if capped:
        return "solver-truncated"
    if used is None:
        return "-"
    spent = used >= budget - slack
    lopsided = cost_kept is not None and used > 0 and cost_kept < 0.5 * used
    if lopsided and spent and n_selected and n_selected >= 0.5 * used:
        # NOT exhaustion. SPARROW bought a full budget's worth of molecules and they collapsed to few
        # distinct modes -- measured here at R=100: used=100, n_selected=100, n_modes_kept=15,
        # mode_rate 0.15, mean_pairwise_sim 0.48. That is the diversity blind spot (cheap syntheses
        # come from shared intermediates, which come from similar molecules), and it is a RESULT, not
        # a caveat. Calling it pool-exhausted would bury the finding and wrongly excuse the number.
        return "redundant"
    if lopsided:
        return "pool-exhausted*"  # gap with FEW selected: the SB used_rxns inflation on an
        #                           exhausted set, where only cost_kept may be quoted
    if spent:
        return "budget-binding"
    return "pool-exhausted" if exhausted_hint else "under-budget"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--generator", default="s3gfn")
    ap.add_argument(
        "--budget", type=float, default=100.0, help="the fixed REACTION budget (primary readout)"
    )
    ap.add_argument(
        "--slack", type=float, default=5.0, help="how close to R still counts as budget-binding"
    )
    ap.add_argument("--out", default="", help="also write the table as CSV here")
    a = ap.parse_args()

    # DISCOVER the cells from disk rather than enumerating a fixed list. The fixed list silently
    # omitted every *_big cell -- the enlarged-pool runs that exist precisely because the
    # budget-faithful sEH pools were too small to compare -- while still printing their stale
    # small-pool namesakes. A reader would have seen "seh_seed43: 25 modes" and had no way to know
    # the 100-mode enlarged result was sitting on disk unlisted.
    seen = set()
    for d in sorted(RESULTS.glob(f"{a.generator}_*_select_N*")):
        seen.add(d.name.rsplit("_select_N", 1)[0])
    for d in sorted(RESULTS.glob(f"{a.generator}_*_greedy_N*")):
        seen.add(d.name.rsplit("_greedy_N", 1)[0])
    cells = sorted(seen)
    if not cells:  # nothing on disk yet: fall back to the canonical grid so the header still prints
        cells = [
            f"{a.generator}_{t}_seed{s}{v}"
            for t in ("seh", "drd2", "clpp")
            for s in (42, 43, 44)
            for v in ("", "_pruned")
        ]

    hdr = ("cell", "pool", "makeable", "G_modes", "G_rxns", "S_modes", "S_used", "S_cost", "S_stop")
    print(
        f"\n  {a.generator} — greedy vs SPARROW on the SAME routes, at R={a.budget:.0f} reactions\n"
    )
    print("  %-30s %5s %8s | %7s %6s | %7s %6s %6s %-16s" % hdr)
    print("  " + "-" * 108)

    out_rows = []
    for tag in cells:
        pdir, psize, plimited = pool_info(tag)
        gm, gr, gmake, gpts = greedy_at_budget(tag, a.budget)
        sm, su, sc, sstat, scap, ssel = sparrow_at_budget(tag, a.budget)
        if pdir is None and gm is None and sm is None:
            continue
        sstop = stop_reason(
            su, a.budget, a.slack, scap, bool(plimited), cost_kept=sc, n_selected=ssel
        )
        print(
            "  %-30s %5s %8s | %7s %6s | %7s %6s %6s %-16s"
            % (
                tag.replace(f"{a.generator}_", ""),
                psize if psize is not None else "-",
                gmake if gmake is not None else "-",
                gm if gm is not None else "-",
                gr if gr is not None else "-",
                sm if sm is not None else "-",
                int(su) if su is not None else "-",
                int(sc) if sc is not None else "-",
                sstop,
            )
        )
        out_rows.append(dict(zip(hdr, (tag, psize, gmake, gm, gr, sm, su, sc, sstop))))

    print("\n  G_* = greedy (largest mode point fitting the budget)   S_* = SPARROW-Batching")
    print(
        "  makeable = largest greedy mode point actually priced; a LOWER bound on synthesizable modes"
    )
    print("  Quote cost_kept_rxns (S_cost), not used_rxns, outside the budget-binding regime.")
    print(
        "  solver-truncated rows are LOWER bounds on the competitor and must not be read as optima.\n"
    )

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(hdr))
            w.writeheader()
            w.writerows(out_rows)
        print(f"  wrote {a.out}\n")


if __name__ == "__main__":
    main()
