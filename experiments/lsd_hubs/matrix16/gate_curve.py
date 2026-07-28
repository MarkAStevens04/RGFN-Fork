#!/usr/bin/env python
"""Fine-grained reward-gate curve for one matrix cell — hub-batching vs best-candidate.

WHY a second sweep tool next to ``gate_sweep.sh``: that script walks each target's coarse
``threshold_variants`` (sEH 5/6/7) by shelling out to ``run_campaign.py`` once per gate. Every one of
those invocations re-reads and re-parses the cell's ``enum_children.json`` — 232k children / ~180 MB
for rxnflow_seh — so a 9-point curve pays that cost 9 times for identical data. This driver loads the
candidates + enumerated hubs **once** and loops the gates in-process, which is the whole reason a
fine sweep is a minutes-long login-node job instead of a cluster submission.

It re-uses ``run_campaign``'s own ``build_strategy`` / ``run_timed`` / ``_readouts``, so a point here
is bit-identical to the corresponding ``run_campaign.py --reward-threshold <g>`` run. No enumeration,
no GPU, no model: purely a re-scoring of already-measured child rewards at a different hit bar.

Two panels (the readouts that actually answer "is the gate starving this cell?"):
  1. gate -> total modes reached, with the 300-mode target line. A cell that cannot reach 300 is
     POOL-LIMITED at that bar, and its reactions/mode is then measured over a smaller library.
  2. gate -> reactions/mode for both strategies (+ the hub-batching edge).

Usage:
    python experiments/lsd_hubs/matrix16/gate_curve.py rxnflow seh --gates 4:8:0.5
    python experiments/lsd_hubs/matrix16/gate_curve.py scent seh --gates 5,6,7 --child-policy free_frag --prebuild-k 20

Output: ``results/gate_curve/<cell>/{gate_curve.csv,gate_curve.png,gate_curve.json}``
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

import run_campaign as RC  # noqa: E402  (experiments/lsd_hubs/campaign/run_campaign.py)
from manifest import get_cell  # noqa: E402


def parse_gates(spec: str) -> list:
    """``lo:hi:step`` (inclusive) or a comma-separated explicit list."""
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        out, v = [], lo
        while v <= hi + 1e-9:
            out.append(round(v, 6))
            v += step
        return out
    return [float(x) for x in spec.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("generator")
    ap.add_argument("target")
    ap.add_argument("--gates", default="4:8:0.5", help="lo:hi:step (inclusive) or v1,v2,...")
    ap.add_argument("--similarity", type=float, default=0.5)
    ap.add_argument("--budget-modes", type=int, default=300)
    ap.add_argument("--budget-reactions", type=int, default=100)
    ap.add_argument("--child-policy", default="reward", choices=["reward", "free_frag"])
    ap.add_argument("--prebuild-k", type=int, default=0)
    ap.add_argument("--rank-by", default="build_score")
    ap.add_argument("--snapshot", default="", help="SCENT fragments_<N>.json (nested cost model)")
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()

    cell = get_cell(a.generator, a.target)
    gates = parse_gates(a.gates)
    enum_path = cell.enum_dir / "enum_children.json"
    if not enum_path.exists():
        raise SystemExit(f"[gate_curve] no enumeration for {cell.tag}: {enum_path}")

    hib = cell.target.higher_is_better
    # ---- load ONCE (the point of this script) -------------------------------
    cands, comps = RC._load_candidates(cell.sample_dir, hib)
    enum_hubs = RC._load_enumerated_hubs(enum_path, comps)
    snapshot = json.load(open(a.snapshot)) if a.snapshot else {}
    cost_table = RC.load_cost_table_from_snapshot(snapshot)
    child_policy = RC.make_child_policy(a.child_policy)
    print(
        f"[gate_curve] {cell.tag}: {len(cands)} candidates, {len(enum_hubs)} enumerated hubs, "
        f"{sum(len(h.children) for h in enum_hubs)} children — loaded once for {len(gates)} gates"
    )

    rows = []
    for g in gates:
        # pre-select-K is gate-dependent (fragments are ranked against the bar), so it is re-ranked
        # per gate rather than hoisted out of the loop.
        prebuilt = None
        if a.prebuild_k > 0:
            ranked = RC.rank_fragments(
                enum_hubs, cost_table, g, method=a.rank_by, higher_is_better=hib
            )
            prebuilt = {f for f, _ in ranked[: a.prebuild_k]}
        common = dict(
            target=f"{cell.tag}@{g}",
            reward_threshold=g,
            similarity=a.similarity,
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
        edge = (
            round(rb["reactions_per_mode"] / rh["reactions_per_mode"], 3)
            if rb["reactions_per_mode"] and rh["reactions_per_mode"]
            else None
        )
        rows.append(
            {
                "gate": g,
                "best_modes": rb["total_modes"],
                "best_rxn_per_mode": rb["reactions_per_mode"],
                "best_stop_reason": rb["stop_reason"],
                "hub_modes": rh["total_modes"],
                "hub_rxn_per_mode": rh["reactions_per_mode"],
                "hub_stop_reason": rh["stop_reason"],
                "edge": edge,
                "hub_hubs_used": rh["distinct_hubs_used"],
                "hub_reward_gen_calls": rh["total_reward_gen_calls"],
                # A cell that stops before the mode budget is POOL-LIMITED: its reactions/mode is
                # then measured over a smaller library and is not comparable to a 300-mode point.
                "pool_limited": rh["total_modes"] < a.budget_modes,
            }
        )
        print(
            f"[gate_curve]   gate {g:>5}: best {rb['total_modes']:>3} modes @ "
            f"{rb['reactions_per_mode']} r/m | hub {rh['total_modes']:>3} modes @ "
            f"{rh['reactions_per_mode']} r/m ({rh['distinct_hubs_used']} hubs) -> {edge}x",
            flush=True,
        )

    out = Path(a.out_dir) if a.out_dir else HERE / "results" / "gate_curve" / cell.tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "gate_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(
        {
            "cell": cell.tag,
            "child_policy": a.child_policy,
            "prebuild_k": a.prebuild_k,
            "similarity": a.similarity,
            "budget_modes": a.budget_modes,
            "higher_is_better": hib,
            "n_enumerated_hubs": len(enum_hubs),
            "points": rows,
        },
        open(out / "gate_curve.json", "w"),
        indent=2,
    )
    _plot(out / "gate_curve.png", rows, cell, a)
    print(f"\n[gate_curve] wrote {out}/gate_curve.{{csv,json,png}}")


def _plot(path: Path, rows, cell, a) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[gate_curve] plot skipped ({exc})")
        return
    from validation.lsdflow.plot_style import ideal_arrow

    gates = [r["gate"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    BC, HB = "#b23a48", "#2a9d8f"

    # ---- Panel 1: modes reached vs gate ------------------------------------
    ax1.plot(gates, [r["best_modes"] for r in rows], "o-", color=BC, label="best-candidate")
    ax1.plot(gates, [r["hub_modes"] for r in rows], "s-", color=HB, label="hub-batching")
    ax1.axhline(
        a.budget_modes, ls="--", lw=1.2, color="#555", label=f"target ({a.budget_modes} modes)"
    )
    # Shade where hub-batching can no longer fill the library — past here reactions/mode is measured
    # over a smaller set, so the right panel's numbers are not 300-mode-comparable.
    lim = [r["gate"] for r in rows if r["pool_limited"]]
    if lim:
        ax1.axvspan(
            min(lim) - 1e-9,
            max(gates),
            color="#c0392b",
            alpha=0.07,
            label="pool-limited (< target)",
        )
        ax2.axvspan(min(lim) - 1e-9, max(gates), color="#c0392b", alpha=0.07)
    ax1.set_xlabel(f"{cell.target_name} reward gate (hit bar)")
    ax1.set_ylabel(f"modes reached (budget {a.budget_modes})")
    ax1.set_title("Can the pool still fill the library?")
    ax1.set_ylim(0, a.budget_modes * 1.12)
    ax1.grid(ls=":", alpha=0.45)
    ax1.legend(fontsize=8)
    ideal_arrow(ax1, "up", label="ideal (more modes)", loc="upper center")

    # ---- Panel 2: reactions/mode vs gate -----------------------------------
    ax2.plot(gates, [r["best_rxn_per_mode"] for r in rows], "o-", color=BC, label="best-candidate")
    ax2.plot(gates, [r["hub_rxn_per_mode"] for r in rows], "s-", color=HB, label="hub-batching")
    for r in rows:  # annotate the edge so the two panels can be read together
        if r["edge"] and r["hub_rxn_per_mode"]:
            ax2.annotate(
                f"{r['edge']:g}×",
                (r["gate"], r["hub_rxn_per_mode"]),
                textcoords="offset points",
                xytext=(0, -13),
                ha="center",
                fontsize=7.5,
                color=HB,
            )
    ax2.set_xlabel(f"{cell.target_name} reward gate (hit bar)")
    ax2.set_ylabel("reactions per mode  (lower = cheaper library)")
    ax2.set_title("Synthesis cost per distinct mode")
    ax2.set_ylim(0, None)
    ax2.grid(ls=":", alpha=0.45)
    ax2.legend(fontsize=8)
    ideal_arrow(ax2, "down", label="ideal (cheaper)", loc="upper right")

    pol = a.child_policy + (f" K={a.prebuild_k}" if a.prebuild_k else "")
    fig.suptitle(
        f"{cell.tag} — reward-gate sweep ({len(rows)} bars, {pol}, τ={a.similarity}, "
        f"{cell.generator} 200-hub enumeration reused)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    print(f"[gate_curve] wrote {path}")


if __name__ == "__main__":
    main()
