#!/usr/bin/env python
"""Publication figures for the 48-run four-generator x four-system matrix (Logs/063).

Designed for a NeurIPS/ICLR *appendix* (the training comparison won't get much main-text space):
self-contained, complete, error bars = seed std. Two figures:
  (A) comparison_grid  -- the reference: reward / drug-likeness / size, all 4 systems, all 4 gens.
  (B) reward_vs_druglikeness -- the trade-off in one view (could double as a main-text teaser).

Color: Okabe-Ito CVD-safe categorical (published, pre-validated), fixed generator order. FragGFN is the
non-synthesizable FOIL -> gray + hatched so it reads as a control, not a competitor. Static PNG+PDF.

Run (rgfn env): source ~/bin/rgfn-smoke-env.sh && python experiments/fixed_reward/scale5k/make_figures.py
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
PER = f"{HERE}/analysis/per_cell.csv"
OUT = f"{HERE}/analysis"

GEN = ["rgfn", "fraggfn", "rxnflow", "scent"]  # fixed order, never cycled
GLAB = {"rgfn": "RGFN", "fraggfn": "FragGFN (foil)", "rxnflow": "RxnFlow", "scent": "SCENT"}
# Okabe-Ito: 3 synthesizable gens get distinct CVD-safe hues; the foil gets neutral gray + hatch.
GCOL = {"rgfn": "#0072B2", "rxnflow": "#E69F00", "scent": "#009E73", "fraggfn": "#9A9A9A"}
GHATCH = {"fraggfn": "////"}
SYS = ["6td3", "clpp", "seh", "drd2"]
SLAB = {
    "6td3": "6TD3\n(Tier2−Tier1 differential)",
    "clpp": "ClpP\n(−Vina, kcal/mol)",
    "seh": "sEH\n(surrogate proxy)",
    "drd2": "DRD2\n(TDC activity)",
}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)

df = pd.read_csv(PER)
g = df.groupby(["sys", "gen"])
MEAN, STD = g.mean(numeric_only=True), g.std(numeric_only=True)


def stat(sys_, gen, col):
    try:
        return float(MEAN.loc[(sys_, gen), col]), float(STD.loc[(sys_, gen), col])
    except KeyError:
        return float("nan"), float("nan")


def legend(fig):
    handles = [
        Patch(
            facecolor=GCOL[g_], edgecolor="black", lw=0.6, hatch=GHATCH.get(g_, ""), label=GLAB[g_]
        )
        for g_ in GEN
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        columnspacing=1.6,
        handlelength=1.4,
    )


# ---------- Figure A: the comprehensive small-multiples grid (appendix reference) ----------
ROWS = [
    ("reward_top10", "top-10 reward  (↑)", None),
    ("qed_med", "median QED  (↑)", (0, 0.82)),
    ("mw_med", "median MW (Da)", (350, 720)),
]
fig, axes = plt.subplots(len(ROWS), len(SYS), figsize=(9.2, 6.4))
x = np.arange(len(GEN))
for ri, (col, ylab, ylim) in enumerate(ROWS):
    for ci, sys_ in enumerate(SYS):
        ax = axes[ri][ci]
        means = [stat(sys_, gg, col)[0] for gg in GEN]
        errs = [stat(sys_, gg, col)[1] for gg in GEN]
        for xi, gg in enumerate(GEN):
            ax.bar(
                xi,
                means[xi],
                width=0.72,
                color=GCOL[gg],
                edgecolor="black",
                linewidth=0.6,
                hatch=GHATCH.get(gg, ""),
                yerr=errs[xi],
                capsize=2.6,
                error_kw=dict(lw=0.9, ecolor="#333333"),
            )
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(["RGFN", "FGFN", "RxnF", "SCENT"], rotation=0)
        if ri == 0:
            ax.set_title(SLAB[sys_], fontsize=9)
        if ci == 0:
            ax.set_ylabel(ylab)
        if ri < len(ROWS) - 1:
            ax.set_xticklabels([])
legend(fig)
fig.suptitle(
    "Four generators × four scoring systems  (mean ± std over 3 seeds)", y=1.055, fontsize=10.5
)
fig.tight_layout(rect=(0, 0, 1, 0.98))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig_comparison_grid.{ext}", bbox_inches="tight")
print(f"wrote {OUT}/fig_comparison_grid.png/.pdf")

# ---------- Figure B: reward vs drug-likeness trade-off (one view; 2D error crosses) ----------
fig2, ax2s = plt.subplots(1, len(SYS), figsize=(11.0, 3.1))
for ci, sys_ in enumerate(SYS):
    ax = ax2s[ci]
    for gg in GEN:
        rx, rxe = stat(sys_, gg, "reward_top10")
        qy, qye = stat(sys_, gg, "qed_med")
        ax.errorbar(
            rx,
            qy,
            xerr=rxe,
            yerr=qye,
            fmt="o",
            ms=9,
            color=GCOL[gg],
            markeredgecolor="black",
            markeredgewidth=0.6,
            lw=1.1,
            ecolor=GCOL[gg],
            capsize=2.6,
            zorder=3,
            alpha=0.95 if gg != "fraggfn" else 0.85,
        )
        # identity is carried by the color legend (labels collide where generators cluster / DRD2 saturates)
    ax.set_title(SLAB[sys_].replace("\n", "  "), fontsize=8.5)
    ax.set_xlabel("top-10 reward  (↑)")
    if ci == 0:
        ax.set_ylabel("median QED  (↑)")
    ax.set_ylim(0, 0.82)
legend(fig2)
fig2.suptitle(
    "Reward vs. drug-likeness trade-off  (points = mean, bars = ±std over 3 seeds)",
    y=1.10,
    fontsize=10.5,
)
fig2.tight_layout(rect=(0, 0, 1, 0.97))
for ext in ("png", "pdf"):
    fig2.savefig(f"{OUT}/fig_reward_vs_druglikeness.{ext}", bbox_inches="tight")
print(f"wrote {OUT}/fig_reward_vs_druglikeness.png/.pdf")
