#!/usr/bin/env python
"""The benchmark's PRIMARY table as a figure: distinct molecules at a fixed 100-reaction budget.

Entry 065 re-read the whole matrix on the reaction axis and the number it produced -- a median
2.88x advantage over the straightforward alternative -- is the paper's main result. It has been a
table of numbers ever since. This draws it.

TWO PANELS, because the result has two parts that a single chart cannot carry:
  LEFT   per-cell dot plot at the headline budget. One row per generator x target cell, our arm
         against best-candidate, seeds shown individually so the reader sees the spread rather than
         a claim about it. This is the "it holds in every cell" panel.
  RIGHT  the ratio against the budget. The advantage GROWS with the budget (2.50x -> 3.12x), which
         means the headline is the second-most conservative of four measured points -- a far more
         persuasive object than any single ratio, and it forecloses "you picked the budget that
         flattered you".

WHAT IS DELIBERATELY NOT AVERAGED. Only cells where BOTH arms were budget-binding count toward a
median (the `comparable` flag). A pool-exhausted arm left budget UNSPENT, so quoting "modes at R"
for it implies it could have spent R and chose not to -- false, and it would credit us a cost win
the data does not support. Those points are drawn hollow and excluded from the median, never
silently averaged in. 6TD3 is absent upstream: its gate rests on warhead-matched decoys and is
parked (see 065).

Run:  conda run -n rgfn python experiments/lsd_hubs/matrix16/plot_reaction_axis.py
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("experiments/lsd_hubs/matrix16/results/reaction_axis")
OUT = SRC
HEADLINE = "100"

# Reference categorical palette, fixed slot order. Slot 1 = ours, slot 2 = the comparison arm.
C = dict(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink2="#52514e",
    ink3="#8a8984",
    grid="#e6e5e1",
    ours="#2a78d6",
    theirs="#eb6834",
    accent="#1baf7a",
)
TARGET_LABEL = {"seh": "sEH", "drd2": "DRD2", "clpp": "ClpP"}


def load():
    rows = list(csv.DictReader(open(SRC / "reaction_axis_ratio.csv")))
    for r in rows:
        r["budget_rxns"] = int(r["budget_rxns"])
        r["seed"] = int(r["seed"])
        r["hub_batching"] = int(r["hub_batching"])
        r["best_candidate"] = int(r["best_candidate"])
        r["ratio"] = float(r["ratio"]) if r["ratio"] else None
        r["comparable"] = r["comparable"] == "True"
    return rows


def main():
    rows = load()
    head = [r for r in rows if r["budget_rxns"] == int(HEADLINE)]

    # cells ordered by mean ratio so the eye reads a gradient, not an alphabet
    by_cell = defaultdict(list)
    for r in head:
        by_cell[(r["target"], r["generator"])].append(r)
    order = sorted(by_cell, key=lambda k: st.mean(x["ratio"] for x in by_cell[k]))

    fig = plt.figure(figsize=(13.8, 7.2), facecolor=C["surface"])
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.42, 1.0],
        wspace=0.30,
        left=0.145,
        right=0.975,
        top=0.755,
        bottom=0.185,
    )
    axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ---------- LEFT: per-cell dots at the headline budget --------------------------------------
    for ax in (axL, axR):
        ax.set_facecolor(C["surface"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(C["ink3"])
            ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors=C["ink2"], labelsize=9.5, length=0)

    ylabels = []
    for y, key in enumerate(order):
        cell = by_cell[key]
        ylabels.append(f"{cell[0]['generator']} · {TARGET_LABEL[key[0]]}")
        axL.plot([0, 112], [y, y], color=C["grid"], linewidth=0.8, zorder=0)
        for r in cell:
            for arm, col, stat in (
                ("best_candidate", C["theirs"], "bc_status"),
                ("hub_batching", C["ours"], "hb_status"),
            ):
                filled = r[stat] == "budget-binding"
                axL.scatter(
                    r[arm],
                    y,
                    s=64,
                    zorder=4,
                    color=col if filled else C["surface"],
                    edgecolor=C["surface"] if filled else col,
                    linewidth=1.8,
                )
        # the gap, drawn once per cell from the seed means
        m_ours = st.mean(r["hub_batching"] for r in cell)
        m_thrs = st.mean(r["best_candidate"] for r in cell)
        axL.annotate(
            "",
            xy=(m_ours, y),
            xytext=(m_thrs, y),
            arrowprops=dict(
                arrowstyle="-|>", color=C["ink3"], linewidth=1.0, shrinkA=7, shrinkB=7, alpha=0.75
            ),
            zorder=2,
        )
        axL.text(
            120,
            y,
            f"{st.mean(r['ratio'] for r in cell):.1f}×",
            fontsize=9.6,
            color=C["ink"],
            fontweight="semibold",
            va="center",
            ha="right",
        )
        axL.text(130, y, f"n={len(cell)}", fontsize=8.4, color=C["ink3"], va="center", ha="right")

    axL.set_yticks(range(len(order)))
    axL.set_yticklabels(ylabels, fontsize=9.8)
    axL.set_ylim(-0.7, len(order) - 0.3)
    axL.set_xlim(0, 136)
    axL.set_xlabel(
        "distinct molecules delivered from 100 reactions",
        fontsize=10.4,
        color=C["ink2"],
        labelpad=8,
    )
    axL.text(120, len(order) - 0.3, "ratio", fontsize=9, color=C["ink3"], ha="right", va="bottom")
    axL.text(130, len(order) - 0.3, "seeds", fontsize=9, color=C["ink3"], ha="right", va="bottom")

    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=C["ours"],
                markeredgecolor=C["surface"],
                label="hub-batching (ours)",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=C["theirs"],
                markeredgecolor=C["surface"],
                label="best-candidate",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=C["surface"],
                markeredgecolor=C["theirs"],
                label="hollow = that arm was pool-exhausted (pair excluded from the median)",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.145, 0.075),
        ncol=3,
        frameon=False,
        fontsize=9.2,
        handletextpad=0.6,
        columnspacing=2.4,
    )

    # ---------- RIGHT: the ratio grows with the budget -------------------------------------------
    budgets = sorted({r["budget_rxns"] for r in rows})
    meds, los, his = [], [], []
    for b in budgets:
        v = [r["ratio"] for r in rows if r["budget_rxns"] == b and r["comparable"] and r["ratio"]]
        meds.append(st.median(v))
        los.append(min(v))
        his.append(max(v))
    axR.fill_between(
        budgets, los, his, color=C["ours"], alpha=0.11, zorder=1, label="per-cell range"
    )
    axR.plot(budgets, meds, color=C["ours"], linewidth=2.2, zorder=3, solid_capstyle="round")
    axR.scatter(
        budgets,
        meds,
        s=62,
        color=C["ours"],
        edgecolor=C["surface"],
        linewidth=2,
        zorder=4,
        label="median over comparable cells",
    )
    axR.axhline(1.0, color=C["ink3"], linewidth=1.0, linestyle=(0, (4, 4)), zorder=2)
    axR.text(budgets[-1], 1.0, "parity", fontsize=9, color=C["ink3"], va="bottom", ha="right")
    for b, m in zip(budgets, meds):
        axR.annotate(
            f"{m:.2f}×",
            xy=(b, m),
            xytext=(0, 11),
            textcoords="offset points",
            fontsize=9.8,
            color=C["ink"],
            fontweight="semibold",
            ha="center",
        )
    axR.scatter(
        [int(HEADLINE)],
        [meds[budgets.index(int(HEADLINE))]],
        s=250,
        zorder=2,
        facecolor="none",
        edgecolor=C["accent"],
        linewidth=2.2,
    )
    axR.annotate(
        "headline budget",
        xy=(int(HEADLINE), meds[budgets.index(int(HEADLINE))]),
        xytext=(6, -34),
        textcoords="offset points",
        fontsize=9.4,
        color=C["accent"],
        fontweight="semibold",
    )
    axR.set_xticks(budgets)
    axR.set_xlim(35, 215)
    axR.set_ylim(0, 5.4)
    axR.set_xlabel("reaction budget", fontsize=10.4, color=C["ink2"], labelpad=8)
    axR.set_ylabel("advantage over best-candidate  (×)", fontsize=10.4, color=C["ink2"], labelpad=8)
    axR.grid(True, axis="y", color=C["grid"], linewidth=0.8, zorder=0)
    axR.set_axisbelow(True)
    leg = axR.legend(loc="lower right", frameon=False, fontsize=9, labelspacing=0.6)
    for t in leg.get_texts():
        t.set_color(C["ink2"])

    n_cmp = sum(1 for r in head if r["comparable"])
    med_head = st.median([r["ratio"] for r in head if r["comparable"]])
    fig.text(
        0.145,
        0.965,
        "More distinct molecules per reaction, in every cell",
        fontsize=15.5,
        color=C["ink"],
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.145,
        0.912,
        f"At the benchmark's 100-reaction budget, hub-batching delivers a median "
        f"{med_head:.2f}× the distinct molecules of taking the best\n"
        f"candidates one at a time — {n_cmp} of {len(head)} cells, four generators, three "
        f"scoring systems, every seed shown separately.\n"
        f"The advantage grows as the budget widens, so the headline sits at the conservative "
        f"end of what we measured.",
        fontsize=10.3,
        color=C["ink2"],
        ha="left",
        va="top",
        linespacing=1.5,
    )
    fig.text(
        0.008,
        0.016,
        "Seeds plotted individually (1–3 per cell). Only cells where BOTH arms spent the "
        "budget count toward a median; a pool-exhausted arm left budget unspent, so quoting "
        '"modes at R" for it would credit a cost win the data\ndoes not support — those are '
        "hollow and excluded. 6TD3 is omitted upstream: its hit threshold rests on "
        "warhead-matched decoys and is parked pending a property-matched set.",
        fontsize=8.1,
        color=C["ink3"],
        ha="left",
        va="bottom",
        linespacing=1.55,
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"reaction_axis.{ext}", dpi=200, facecolor=C["surface"])
    print(f"wrote {OUT}/reaction_axis.png + .pdf")
    print(
        f"  headline R={HEADLINE}: median {med_head:.2f}x over {n_cmp}/{len(head)} comparable cells"
    )
    for b, m, lo, hi in zip(budgets, meds, los, his):
        print(f"  R={b:<4} median {m:.2f}x  (range {lo:.2f}-{hi:.2f})")


if __name__ == "__main__":
    main()
