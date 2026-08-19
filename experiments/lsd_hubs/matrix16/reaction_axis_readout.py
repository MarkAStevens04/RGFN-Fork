#!/usr/bin/env python
"""Re-read the 16-cell matrix on the PRIMARY axis: modes at a fixed REACTION budget.

Every cell in the matrix was written up at "300 modes, N reactions". `CLAUDE.md` (decided
2026-08-17) makes the headline stopping condition the other way round -- "I have 100 reactions, how
many distinct high-reward molecules do I get?" -- with reactions-for-300-modes kept as the secondary
readout. Both are read off the SAME greedy ordering, so this is a reporting change and needs no new
compute: the per-step curves already carry cum_reactions, cum_modes and cum_reward_gen_calls.

WHY THIS IS NOT JUST argmax(cum_modes | cum_reactions <= R). A reaction budget "never yields an
unreportable cell", but that is only true if you say what the number MEANS in each cell, because a
mode count at R is comparable across arms ONLY when the budget actually bound. Three cases, and the
CSV labels every row with one of them:

  budget-binding    the next mode would not fit in the remaining budget. The like-for-like case.
                    You can rarely land exactly on R, so this is "the next step would exceed R",
                    never "used == R".
  pool-exhausted    the arm ran out of qualifying candidates before spending the budget. The budget
                    goes UNSPENT, so quoting "modes at R" implies it could have spent R and chose
                    not to -- which is false. Never counts as a win or a loss on cost.
  mode-capped       the run stopped at its 300-mode budget while reactions were still available.
                    The MODE cap bound, not the reaction budget, so this is a lower bound on what
                    the arm would deliver at R and must not be read as "this is all it could do".

`n_modes_available` is reported beside the count so a short cell can be diagnosed: it separates
COLLAPSE (few distinct molecules exist at all) from mere REDUNDANCY.

6TD3 is EXCLUDED by default (--include-6td3 to override): its -2.0 gate rests on warhead-matched
decoys -- every decoy carries the CR8-like purine, but MW is only range-bounded (250-650) and no
other property is matched -- so the bar is not yet defensible as a hit threshold. Logs/007 shows the
differential survives MW matching (0.95 -> 0.87) where absolute score does not (0.89 -> 0.65), so the
oracle is sound; it is the CUTOFF that needs a property-matched decoy set. Parked 2026-08-18.

Run:  conda run -n rgfn python experiments/lsd_hubs/matrix16/reaction_axis_readout.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from targets import TARGETS  # noqa: E402  -- the authoritative gate table

ARMS = ("hub_batching", "best_candidate")
SEED_DIRS = {"results": 42, "results_seed43": 43, "results_seed44": 44}
# Deliberate sensitivity-sweep variants, not cells of the matrix proper.
VARIANT_MARKERS = ("thr", "cap9", "naive")


def read_curve(path):
    with open(path) as fh:
        rows = [
            (int(r["cum_reactions"]), int(r["cum_modes"]), int(r["cum_reward_gen_calls"] or 0))
            for r in csv.DictReader(fh)
        ]
    return rows


def read_at_budget(rows, budget, mode_cap):
    """Modes at a fixed reaction budget, plus the label that says whether the budget bound."""
    if not rows:
        return None
    kept = [r for r in rows if r[0] <= budget]
    if not kept:
        # not even the first mode fits in the budget
        return dict(modes=0, used=0, calls=0, status="budget-binding", modes_available=rows[-1][1])
    used, modes, calls = kept[-1]
    total_rxn, total_modes = rows[-1][0], rows[-1][1]
    if total_modes >= mode_cap and total_rxn <= budget:
        status = "mode-capped"  # hit the 300-mode budget with reactions to spare
    elif total_rxn <= budget:
        status = "pool-exhausted"  # ran out of candidates without spending the budget
    else:
        status = "budget-binding"  # the next mode would not fit
    return dict(modes=modes, used=used, calls=calls, status=status, modes_available=total_modes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrix-root",
        default=str(Path(__file__).resolve().parent),
        help="dir holding results/, results_seed43/, results_seed44/",
    )
    ap.add_argument("--budgets", default="50,100,150,200")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--include-6td3",
        action="store_true",
        help="include 6TD3 despite its un-calibrated gate (see module docstring)",
    )
    a = ap.parse_args()

    root = Path(a.matrix_root)
    budgets = [int(x) for x in a.budgets.split(",")]
    out = Path(a.out_dir) if a.out_dir else root / "results" / "reaction_axis"
    out.mkdir(parents=True, exist_ok=True)

    records, skipped = [], []
    for sd, seed in SEED_DIRS.items():
        for cell in sorted((root / sd).glob("*/summary.json")) if (root / sd).is_dir() else []:
            d = json.loads(cell.read_text())
            tag = d.get("tag", "")
            cdir = cell.parent
            if any(m in tag for m in VARIANT_MARKERS):
                continue
            target = next((t for t in ("6td3", "clpp", "drd2", "seh") if tag.endswith(t)), None)
            if target is None:
                continue
            if target == "6td3" and not a.include_6td3:
                skipped.append(f"{sd}/{tag}")
                continue
            headline = TARGETS[target].mode_reward_threshold
            run_gate = d.get("reward_threshold")
            gate_ok = run_gate is not None and abs(float(run_gate) - headline) < 1e-9
            mode_cap = int(d.get("budget_modes") or 300)
            generator = tag.rsplit("_", 1)[0]

            for arm in ARMS:
                cp = cdir / f"curve_{arm}.csv"
                if not cp.exists():
                    continue
                rows = read_curve(cp)
                for R in budgets:
                    r = read_at_budget(rows, R, mode_cap)
                    if r is None:
                        continue
                    records.append(
                        dict(
                            seed=seed,
                            generator=generator,
                            target=target,
                            cell=tag,
                            arm=arm,
                            gate=headline,
                            gate_matches_run=gate_ok,
                            budget_rxns=R,
                            modes_at_R=r["modes"],
                            used_rxns=r["used"],
                            reward_gen_calls=r["calls"],
                            status=r["status"],
                            n_modes_available=r["modes_available"],
                            rxn_per_mode=round(r["used"] / r["modes"], 3) if r["modes"] else "",
                        )
                    )

    if not records:
        sys.exit("no cells found -- check --matrix-root")

    fields = list(records[0])
    with open(out / "reaction_axis.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    # ---- headline table: the ratio between our two strategies at each budget --------------------
    ratios = []
    for R in budgets:
        for gen, tgt, seed in sorted({(r["generator"], r["target"], r["seed"]) for r in records}):
            got = {
                r["arm"]: r
                for r in records
                if r["budget_rxns"] == R
                and r["generator"] == gen
                and r["target"] == tgt
                and r["seed"] == seed
            }
            if set(ARMS) - set(got):
                continue
            hb, bc = got["hub_batching"], got["best_candidate"]
            ratios.append(
                dict(
                    budget_rxns=R,
                    generator=gen,
                    target=tgt,
                    seed=seed,
                    hub_batching=hb["modes_at_R"],
                    best_candidate=bc["modes_at_R"],
                    ratio=round(hb["modes_at_R"] / bc["modes_at_R"], 3) if bc["modes_at_R"] else "",
                    hb_status=hb["status"],
                    bc_status=bc["status"],
                    comparable=(hb["status"] == "budget-binding" == bc["status"]),
                )
            )
    with open(out / "reaction_axis_ratio.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ratios[0]))
        w.writeheader()
        w.writerows(ratios)

    # ---- console summary -----------------------------------------------------------------------
    print(
        f"cells read: {len({(r['seed'], r['cell']) for r in records})}   "
        f"rows: {len(records)}   -> {out}/reaction_axis.csv"
    )
    if skipped:
        print(f"EXCLUDED (un-calibrated gate): {', '.join(sorted(set(skipped)))}")
    bad = [r for r in records if not r["gate_matches_run"]]
    print(
        f"gate mismatches: {len(bad)}"
        + (" <-- INVESTIGATE" if bad else " (all cells at the headline gate)")
    )

    for R in budgets:
        sel = [r for r in ratios if r["budget_rxns"] == R]
        if not sel:
            continue
        comp = [r for r in sel if r["comparable"]]
        vals = [r["ratio"] for r in comp if r["ratio"] != ""]
        print(f"\n--- R = {R} reactions ---   {len(comp)}/{len(sel)} strictly comparable")
        print(f"{'cell':<18}{'seed':>5}{'hub':>6}{'bc':>6}{'ratio':>7}   status")
        for r in sorted(sel, key=lambda x: (x["target"], x["generator"], x["seed"])):
            flag = "" if r["comparable"] else f"  [{r['hb_status']}/{r['bc_status']}]"
            print(
                f"{r['generator']+'_'+r['target']:<18}{r['seed']:>5}"
                f"{r['hub_batching']:>6}{r['best_candidate']:>6}{str(r['ratio']):>7}{flag}"
            )
        if vals:
            vals_sorted = sorted(vals)
            med = vals_sorted[len(vals_sorted) // 2]
            print(
                f"   median ratio over strictly-comparable cells: {med:.2f}x  "
                f"(min {min(vals):.2f}, max {max(vals):.2f})"
            )


if __name__ == "__main__":
    main()
