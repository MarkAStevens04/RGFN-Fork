#!/usr/bin/env python
"""Cross-generator diversity sweep: modes found at a fixed reaction budget, vs the Tanimoto cutoff.

The canonical LSD-Flow Pareto panel (`campaign/results/scent_seh_1kx200/pareto.png`) sweeps the
diversity cutoff for **one** generator and asks how many diverse hits each strategy assembles inside a
fixed synthesis budget. That panel is the clearest single statement of the method's value — but it has
only ever been drawn for SCENT, which leaves open whether the shape is a property of hub-batching or of
SCENT. This driver redraws it with EVERY generator that has an sEH enumeration on disk, hub-batching
and best-candidate for each.

Cost: none beyond CPU. It re-scores the already-enumerated children at each cutoff, loading each
cell's enumeration ONCE and looping the cutoffs in-process (same trick as ``gate_curve.py``), so a
full 13-cutoff x 4-generator panel is a login-node job rather than a cluster submission.

Axis conventions are inherited from the existing panel so the two are visually comparable:
  * x runs 0.9 -> 0.3 (INVERTED), i.e. left-to-right = "similar modes -> diverse modes";
  * y = modes discovered within ``--budget-reactions`` (default 100), higher is better.
Because x is inverted, the ideal direction in SCREEN space is up-and-RIGHT (more diverse, more modes).

Cells with no enumeration are reported and skipped, never silently dropped — a missing generator must
be visible as missing. ``fraggfn`` is excluded by default: its "reactions" are fragment attachments,
not synthesis steps (it is the non-reaction control), so plotting it on a shared reactions axis invites
a false comparison. ``--include-fraggfn`` opts in for the control panel.

Usage:
    python experiments/lsd_hubs/matrix16/tau_curve_all_generators.py
    python experiments/lsd_hubs/matrix16/tau_curve_all_generators.py --target seh --cutoffs 0.9:0.3:-0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(HERE), str(REPO / "experiments" / "lsd_hubs" / "campaign")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_campaign as RC  # noqa: E402
from manifest import select  # noqa: E402

# Per-generator colour, stable across every figure in the paper.
GEN_COLOR = {
    "scent": "#2a9d8f",
    "rxnflow": "#e76f51",
    "rgfn": "#4361ee",
    "fraggfn": "#8a4fbf",
}
# Hub-batching solid + filled; best-candidate dashed + hollow. One glance separates method from model.
STRAT_STYLE = {
    "hub_batching": dict(ls="-", marker="o", ms=5, lw=1.9),
    "best_candidate": dict(ls="--", marker="o", ms=5, lw=1.4, mfc="white"),
}


def parse_cutoffs(spec: str) -> list:
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        out, v = [], lo
        while (step > 0 and v <= hi + 1e-9) or (step < 0 and v >= hi - 1e-9):
            out.append(round(v, 6))
            v += step
        return out
    return [float(x) for x in spec.split(",")]


def _snapshot_for(cell):
    """SCENT's promoted-fragment recipe snapshot (nested cost model); '' for the other generators."""
    if cell.generator != "scent":
        return ""
    root = Path(cell.checkpoint).parents[2] if len(Path(cell.checkpoint).parents) > 2 else None
    if root is None:
        return ""
    hits = sorted(
        root.glob("**/additional_fragments/fragments_*.json"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return str(hits[-1]) if hits else ""


def _enum_status(cell, min_coverage: float):
    """``None`` if the cell's enumeration is usable, else a short reason string.

    Existence of ``enum_children.json`` is NOT sufficient. A timed-out or partially-flushed
    enumeration leaves a valid file covering only the hubs that finished — ``rgfn_seh`` currently
    holds 10 of its 200 hubs from an old smoke, and silently plotting it would understate that
    generator by 20x while looking perfectly healthy. So require the enumeration to cover
    ``min_coverage`` of the hubs the cell actually selected (``hubs.csv``)."""
    enum_path = cell.enum_dir / "enum_children.json"
    hubs_path = cell.enum_dir / "hubs.csv"
    if not enum_path.exists():
        return "not enumerated"
    try:
        n_enum = len(json.load(open(enum_path)).get("hubs", []))
    except Exception:  # noqa: BLE001
        return "unreadable enum_children.json"
    if not hubs_path.exists():
        return None  # nothing to compare against; trust the file
    n_want = sum(1 for _ in open(hubs_path)) - 1
    if n_want > 0 and n_enum < min_coverage * n_want:
        return f"PARTIAL enumeration {n_enum}/{n_want} hubs — stale, re-run"
    return None


def sweep_cell(cell, cutoffs, a):
    """One cell's curve: both strategies at every cutoff, enumeration loaded once."""
    enum_path = cell.enum_dir / "enum_children.json"
    hib = cell.target.higher_is_better
    cands, comps = RC._load_candidates(cell.sample_dir, hib)
    enum_hubs = RC._load_enumerated_hubs(enum_path, comps)
    snap = _snapshot_for(cell)
    cost_table = RC.load_cost_table_from_snapshot(json.load(open(snap)) if snap else {})
    # Child policy follows the committed per-generator convention (run_cell_campaign.sh): SCENT uses
    # its hero free_frag + pre-select-K=20; the baselines have no dynamic library so free_frag is
    # degenerate for them and naive `reward` is the honest comparison.
    policy_name = "free_frag" if cell.generator == "scent" else "reward"
    prebuild_k = a.scent_prebuild_k if cell.generator == "scent" else 0
    child_policy = RC.make_child_policy(policy_name)
    gate = cell.target.mode_reward_threshold if a.gate is None else a.gate

    prebuilt = None
    if prebuild_k > 0:
        ranked = RC.rank_fragments(
            enum_hubs, cost_table, gate, method="build_score", higher_is_better=hib
        )
        prebuilt = {f for f, _ in ranked[:prebuild_k]}

    print(
        f"[tau] {cell.tag}: {len(cands)} candidates, {len(enum_hubs)} hubs, "
        f"{sum(len(h.children) for h in enum_hubs)} children | gate {gate} | policy {policy_name}"
        + (f" K={prebuild_k}" if prebuild_k else "")
    )
    rows = []
    for tau in cutoffs:
        common = dict(
            target=f"{cell.tag}@tau{tau}",
            reward_threshold=gate,
            similarity=tau,
            higher_is_better=hib,
        )
        budget = ("modes", a.budget_modes)
        bc, _ = RC.run_timed(
            RC.build_strategy("best_candidate", cands, cost_table, comps, **common), budget
        )
        hb, _ = RC.run_timed(
            RC.build_strategy(
                "hub_batching",
                enum_hubs,
                cost_table,
                comps,
                child_policy=child_policy,
                prebuilt_fragments=prebuilt,
                **common,
            ),
            budget,
        )
        rb = RC._readouts(bc, a.budget_reactions, a.budget_modes)
        rh = RC._readouts(hb, a.budget_reactions, a.budget_modes)
        k1 = f"case1_modes_at_{a.budget_reactions}rxn"
        rows.append(
            {
                "cell": cell.tag,
                "generator": cell.generator,
                "child_policy": policy_name,
                "prebuild_k": prebuild_k,
                "gate": gate,
                "tau": tau,
                "best_modes_at_budget": rb[k1],
                "hub_modes_at_budget": rh[k1],
                "best_total_modes": rb["total_modes"],
                "hub_total_modes": rh["total_modes"],
                "best_rxn_per_mode": rb["reactions_per_mode"],
                "hub_rxn_per_mode": rh["reactions_per_mode"],
            }
        )
        print(
            f"[tau]   tau={tau:<5} modes@{a.budget_reactions}rxn: "
            f"best {rb[k1]:>3} | hub {rh[k1]:>3}",
            flush=True,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="seh")
    ap.add_argument("--cutoffs", default="0.9:0.3:-0.05", help="lo:hi:step or v1,v2,...")
    ap.add_argument("--budget-reactions", type=int, default=100, help="the FIXED synthesis budget")
    ap.add_argument("--budget-modes", type=int, default=300)
    ap.add_argument("--gate", type=float, default=None, help="override the target's hit bar")
    ap.add_argument("--scent-prebuild-k", type=int, default=20)
    ap.add_argument("--include-fraggfn", action="store_true")
    ap.add_argument(
        "--min-hub-coverage",
        type=float,
        default=0.9,
        help="reject a cell whose enumeration covers less than this fraction of its hubs.csv "
        "(guards against plotting a timed-out/partial enumeration as if it were complete)",
    )
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()

    cutoffs = parse_cutoffs(a.cutoffs)
    cells, missing = [], []
    for c in select(targets=[a.target]):
        if c.generator == "fraggfn" and not a.include_fraggfn:
            continue
        why = _enum_status(c, a.min_hub_coverage)
        if why is None:
            cells.append(c)
        else:
            missing.append(f"{c.tag} ({why})")
    if not cells:
        raise SystemExit(f"[tau] no {a.target} cell has a usable enumeration yet")
    if missing:
        print(f"[tau] EXCLUDED from the panel: {'; '.join(missing)}")

    rows = []
    for c in cells:
        rows.extend(sweep_cell(c, cutoffs, a))

    out = Path(a.out_dir) if a.out_dir else HERE / "results" / "tau_curve_all" / a.target
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tau_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(
        {
            "target": a.target,
            "budget_reactions": a.budget_reactions,
            "budget_modes": a.budget_modes,
            "cutoffs": cutoffs,
            "cells": [c.tag for c in cells],
            "not_enumerated": missing,
            "points": rows,
        },
        open(out / "tau_curve.json", "w"),
        indent=2,
    )
    _plot(out / "tau_curve_all_generators.png", rows, cells, missing, a)
    print(f"\n[tau] wrote {out}/tau_curve_all_generators.png (+ csv/json)")


def _plot(path: Path, rows, cells, missing, a) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[tau] plot skipped ({exc})")
        return
    from validation.lsdflow.plot_style import ideal_arrow

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for cell in cells:
        sub = [r for r in rows if r["cell"] == cell.tag]
        xs = [r["tau"] for r in sub]
        col = GEN_COLOR.get(cell.generator, "#555555")
        for skey, ykey in (
            ("hub_batching", "hub_modes_at_budget"),
            ("best_candidate", "best_modes_at_budget"),
        ):
            ax.plot(
                xs,
                [r[ykey] for r in sub],
                color=col,
                label=f"{cell.generator} — {skey.replace('_', '-')}",
                **STRAT_STYLE[skey],
            )
    ax.invert_xaxis()  # 0.9 -> 0.3 so left-to-right reads "similar -> diverse" (matches Logs/029)
    ax.axvline(0.5, ls="--", lw=1.1, color="#777")
    # Anchor the label in DATA x but AXES y, so it sits just under the frame regardless of ylim.
    ax.annotate(
        " default 0.5",
        xy=(0.5, 0.985),
        xycoords=("data", "axes fraction"),
        rotation=90,
        va="top",
        ha="left",
        fontsize=7.5,
        color="#777",
    )
    ax.set_xlabel("Diversity cutoff (Tanimoto similarity)      similar modes → diverse modes")
    ax.set_ylabel(f"modes discovered within {a.budget_reactions} reactions")
    gate = rows[0]["gate"]
    ax.set_title(
        f"{a.target} — diverse hits per fixed synthesis budget, all generators\n"
        f"gate {gate} · {a.budget_reactions}-reaction budget · 200-hub enumeration reused\n"
        f"solid = hub-batching, dashed = best-candidate",
        fontsize=10,
    )
    ax.grid(ls=":", alpha=0.45)
    ax.legend(fontsize=8, ncol=2, framealpha=0.92, loc="lower left")
    # x is INVERTED, so "more diverse" is to the RIGHT: ideal = up-and-right in screen space. The
    # top-right corner is empty here (every curve decays toward it), so the arrow sits clear of data.
    ideal_arrow(ax, "up-right", loc="upper right")
    if missing:
        fig.text(
            0.5,
            0.005,
            "excluded (no usable enumeration): " + "; ".join(missing),
            ha="center",
            fontsize=7.5,
            style="italic",
            color="#a33",
        )
    fig.tight_layout(rect=(0, 0.03 if missing else 0, 1, 1))
    fig.savefig(path, dpi=130)
    print(f"[tau] wrote {path}")


if __name__ == "__main__":
    main()
