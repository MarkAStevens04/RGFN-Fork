#!/usr/bin/env python
"""The greedy-oracle figure (Logs/060): how much does the static flow ranking leave on the table?

Three panels, left to right the way the argument runs:

A  Per-cell cost, three arms on one axis. The point the panel makes visually is that the flow walk
   sits far closer to the adaptive greedy than to the standing baseline.
B  The same gap expressed as the share of the achievable improvement flow recovers, against what it
   costs to get the rest: oracle calls. The greedy's residual edge is bought with 1-27x more scoring.
C  The quality side, because the cost gap alone is not the whole story: the greedy's libraries sit
   closer to the acceptance bar than the flow walk's, so part of its cost advantage is quality drift.

    python experiments/lsd_hubs/greedy_oracle/plot_greedy.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from validation.lsdflow.plot_style import ideal_marker  # noqa: E402

C_BASE = "#9aa4ad"
C_FLOW = "#0E5480"
C_GREEDY = "#C2551F"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default=str(HERE / "results"))
    a = ap.parse_args()
    root = Path(a.results)

    rows = list(csv.DictReader(open(root / "greedy_oracle_summary.csv")))
    audit = {}
    for r in csv.DictReader(open(root / "library_audit.csv")):
        audit[(r["cell"], r["arm"])] = r

    cells = [r["cell"] for r in rows]
    y = range(len(cells))
    fig, axes = plt.subplots(
        1, 3, figsize=(16.5, 6.2), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]}
    )

    # ---- A: cost per mode, three arms
    ax = axes[0]
    ax.barh(
        [i + 0.26 for i in y],
        [float(r["bc_rxn_per_mode"]) for r in rows],
        height=0.24,
        color=C_BASE,
        label="best-candidate (baseline)",
    )
    ax.barh(
        list(y),
        [float(r["flow_rxn_per_mode"]) for r in rows],
        height=0.24,
        color=C_FLOW,
        label="hub-batching (static flow rank)",
    )
    ax.barh(
        [i - 0.26 for i in y],
        [float(r["greedy_rxn_per_mode"]) for r in rows],
        height=0.24,
        color=C_GREEDY,
        label="cost-benefit greedy (oracle)",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(cells, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("reactions per mode")
    ax.set_title(f"A · library cost {ideal_marker('lower', axis='x')}", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    # ---- B: share of achievable gain recovered vs the oracle-call price of the rest
    ax = axes[1]
    xs = [float(r["oracle_overhead_x"]) for r in rows]
    ys = [100 * float(r["recovered_frac_cost"]) for r in rows]
    docking = [r["docking"] == "True" for r in rows]
    ax.scatter(
        [x for x, d in zip(xs, docking) if not d],
        [v for v, d in zip(ys, docking) if not d],
        s=70,
        color=C_FLOW,
        marker="o",
        label="surrogate reward",
        zorder=3,
    )
    ax.scatter(
        [x for x, d in zip(xs, docking) if d],
        [v for v, d in zip(ys, docking) if d],
        s=90,
        color=C_FLOW,
        marker="D",
        facecolors="none",
        linewidths=1.8,
        label="real GPU docking",
        zorder=3,
    )
    for x, v, c in zip(xs, ys, cells):
        ax.annotate(
            c, (x, v), fontsize=7, xytext=(4, 3), textcoords="offset points", color="#4a5560"
        )
    ax.axhline(100, color="#666", lw=0.9, ls="--")
    ax.annotate("greedy parity", (max(xs) * 0.97, 100.4), fontsize=8, ha="right", color="#666")
    ax.set_xscale("log")
    ax.set_xlabel("extra molecule scorings the greedy needs  (x, log)")
    ax.set_ylabel("% of the achievable cost gain recovered by flow")
    ax.set_title(
        f"B · what a static ranking gets, and what the rest costs {ideal_marker('higher')}",
        fontsize=11,
        loc="left",
    )
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    # ---- C: quality of the delivered library (distance above the acceptance bar)
    ax = axes[2]
    fl, gr, labs = [], [], []
    for r in rows:
        af = audit.get((r["cell"], "hub_batching"))
        ag = audit.get((r["cell"], "greedy_oracle"))
        if not af or not ag:
            continue
        bar = float(r["bar"])
        # Orientation comes from the run's own record, never inferred from the bar's sign.
        sign = 1.0 if r["higher_is_better"] == "True" else -1.0
        fl.append(sign * (float(af["median_reward"]) - bar))
        gr.append(sign * (float(ag["median_reward"]) - bar))
        labs.append(r["cell"])
    yy = range(len(labs))
    ax.barh([i + 0.19 for i in yy], fl, height=0.36, color=C_FLOW, label="hub-batching (flow)")
    ax.barh([i - 0.19 for i in yy], gr, height=0.36, color=C_GREEDY, label="cost-benefit greedy")
    ax.set_yticks(list(yy))
    ax.set_yticklabels(labs, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("median library reward, margin over the bar")
    ax.set_title(f"C · library quality {ideal_marker('higher', axis='x')}", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle(
        "A static ranking by recovered flow captures most of what an adaptive optimiser can find — "
        "and the optimiser's remaining edge is bought with scoring calls and library quality",
        fontsize=12.5,
        y=0.985,
    )
    fig.text(
        0.005,
        0.008,
        "14 cells (4 generators x 4 targets; diamonds = real GPU docking). Budget 300 modes, "
        "Tanimoto cutoff 0.5, Morgan r=3. Both arms share the enumeration, cost model, child "
        "policy and mode definition; only the choice of which hub to walk next differs.",
        fontsize=7.6,
        color="#4a5560",
    )
    fig.tight_layout(rect=[0, 0.028, 1, 0.955])
    for ext in ("png", "pdf"):
        fig.savefig(root / f"greedy_oracle.{ext}", dpi=180)
    print(f"-> {root}/greedy_oracle.png")


if __name__ == "__main__":
    main()
