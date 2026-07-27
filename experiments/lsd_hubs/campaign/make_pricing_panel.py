#!/usr/bin/env python
"""Consolidated pricing panel for the paper (SCENT sEH, tau=0.5, mode bar 7.0).

Three panels, one measure each (never a dual axis):
  A  reactions per mode, by pricing regime x selection strategy   (lower is better)
  B  modes actually priced -- coverage differs by regime          (higher is better)
  C  AiZynth route-success ladder from Logs/047-048               (higher is better)

EVERY number is transcribed from a result file, with the source named in DATA below -- no
values from memory. Palette = the validated default instance (dataviz references/palette.md):
categorical slot 1 blue #2a78d6 (hub-batching) + slot 2 orange #eb6834 (best-candidate), and a
one-hue ordinal blue ramp for panel C. Both were run through the six-checks validator
(categorical: CVD dE 24.7 protan, normal-vision 33.6, contrast 4.30/3.12 -- all PASS; ordinal:
monotone L, gaps 0.189/0.190, light end 2.06:1, hue spread 3deg -- all PASS). Panel C's light
step sits under 3:1, so the relief rule applies -> every bar carries a visible value label.

Static print figure: hover/dark-mode steps of the dataviz procedure do not apply; the emitted
CSV is the table view. Three deliberate deviations, checked against anti-patterns.md:
  - a value label on EVERY bar (the rule prefers selective labels + tooltip): print has no
    tooltip, exact values matter to a referee, and panel C's sub-3:1 light step obliges labels;
  - hatch on panel C's third bar: texture is "opt-in", but here it MEANS provisional, and is
    redundantly encoded (asterisk + footnote) so it is never texture-alone;
  - panel C uses an ordinal ramp rather than one flat hue: the three conditions are genuinely
    cumulative (each adds a lever), which is the ordered-category case the rule permits.
The dashed rule in panel B is a threshold marker, not a gridline (grids stay solid hairlines).

Run:  conda run -n rgfn python experiments/lsd_hubs/campaign/make_pricing_panel.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("experiments/lsd_hubs/campaign/results/paper_pricing_panel")

# ── palette (validated default instance) ──────────────────────────────────────
HUB, BEST = "#2a78d6", "#eb6834"  # categorical slots 1, 2
LADDER = ["#86b6ef", "#2a78d6", "#104281"]  # one-hue ordinal ramp (blue 250/450/650)
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"

# ── data: (regime label, hub reactions, hub modes, best reactions, best modes, source) ──
DATA = [
    (
        "Native\n(by construction)",
        367,
        300,
        929,
        300,
        "Logs/044 - results/scent_seh_strategy_summary/strategy_compute_summary.csv (committed)",
    ),
    (
        "MultiAiZ\n+ ZINC",
        301,
        163,
        536,
        182,
        "Logs/048 job 71531 - results/scent_seh_multiaiz_zinc/budget_efficiency.csv",
    ),
    (
        "MultiAiZ\n+ ZINC u SMALL",
        540,
        283,
        768,
        257,
        "Logs/048 job 71529 - results/scent_seh_multiaiz_zincsmall/budget_efficiency.csv",
    ),
    (
        "From-scratch\nAiZynth",
        502,
        183,
        564,
        154,
        "Logs/041 - results/scent_seh_sparrow_headline/budget_efficiency.csv",
    ),
]
LADDER_DATA = [
    ("ZINC\ncatalogue", 48.7, "2314/4749", "Logs/041 timed route cache", False),
    ("+ reaction-GFN\nblocks in stock", 73.8, "3506/4749", "Logs/047 job 71436", False),
    ("+ 10x search\nbudget", 92.2, "3228/3500", "Logs/048 job 71534 (partial - 74% routed)", True),
]


def style(ax, value_axis="x"):
    """Recessive grid/axes: grid on the value axis only, no top/right spines."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(axis=value_axis, color="#e8e7e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = [d[0] for d in DATA]
    hub_rpm = [d[1] / d[2] for d in DATA]
    best_rpm = [d[3] / d[4] for d in DATA]
    hub_modes = [d[2] for d in DATA]
    best_modes = [d[4] for d in DATA]

    fig = plt.figure(figsize=(13.2, 4.5), facecolor=SURFACE)
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.32, 1.0, 0.86],
        wspace=0.34,
        left=0.085,
        right=0.985,
        top=0.775,
        bottom=0.19,
    )

    y = range(len(labels))
    H = 0.30  # thin marks; the 0.04 gap between the pair is the 2px surface gap
    GAP = 0.02

    # ── Panel A: reactions per mode (horizontal grouped bars) ─────────────────
    axA = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    for i, (h, b) in enumerate(zip(hub_rpm, best_rpm)):
        axA.barh(i - H / 2 - GAP, h, height=H, color=HUB, zorder=3)
        axA.barh(i + H / 2 + GAP, b, height=H, color=BEST, zorder=3)
        axA.text(
            h + 0.06,
            i - H / 2 - GAP,
            f"{h:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
            fontweight="medium",
        )
        axA.text(
            b + 0.06,
            i + H / 2 + GAP,
            f"{b:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
            fontweight="medium",
        )
    axA.set_yticks(list(y))
    axA.set_yticklabels(labels, fontsize=8.5, color=INK)
    axA.set_xlim(0, max(best_rpm) * 1.16)
    axA.invert_yaxis()
    style(axA, "x")
    axA.set_xlabel("reactions per mode  ← lower is better", fontsize=8.5, color=INK2)
    axA.set_title(
        "A  Route cost by pricing regime",
        fontsize=10.5,
        color=INK,
        fontweight="semibold",
        loc="left",
        pad=8,
    )

    # ── Panel B: modes priced (coverage) ─────────────────────────────────────
    axB = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    for i, (hm, bm) in enumerate(zip(hub_modes, best_modes)):
        axB.barh(i - H / 2 - GAP, hm, height=H, color=HUB, zorder=3)
        axB.barh(i + H / 2 + GAP, bm, height=H, color=BEST, zorder=3)
        axB.text(
            hm + 6,
            i - H / 2 - GAP,
            str(hm),
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
            fontweight="medium",
        )
        axB.text(
            bm + 6,
            i + H / 2 + GAP,
            str(bm),
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
            fontweight="medium",
        )
    axB.set_yticks(list(y))
    axB.set_yticklabels([])
    axB.set_xlim(0, 300 * 1.22)
    axB.invert_yaxis()
    style(axB, "x")
    axB.axvline(300, color=INK3, linewidth=0.9, linestyle=(0, (3, 2)), zorder=2)
    axB.annotate(
        "300-mode budget",
        xy=(300, 0.015),
        xycoords=("data", "axes fraction"),
        fontsize=7.6,
        color=INK2,
        ha="right",
        va="bottom",
        xytext=(-4, 0),
        textcoords="offset points",
    )
    axB.set_xlabel("modes priced  → higher is better", fontsize=8.5, color=INK2)
    axB.set_title(
        "B  Coverage: modes a pricer can route",
        fontsize=10.5,
        color=INK,
        fontweight="semibold",
        loc="left",
        pad=8,
    )

    # ── Panel C: synthesizability ladder (single series → no legend) ──────────
    axC = fig.add_subplot(gs[0, 2], facecolor=SURFACE)
    for i, (lab, pct, frac, _src, partial) in enumerate(LADDER_DATA):
        axC.bar(
            i,
            pct,
            width=0.46,
            color=LADDER[i],
            zorder=3,
            hatch="//" if partial else None,
            edgecolor=SURFACE,
            linewidth=0.9,
        )
        axC.text(
            i,
            pct + 2.2,
            f"{pct:.1f}%" + ("*" if partial else ""),
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK,
            fontweight="semibold",
        )
    axC.set_xticks(range(len(LADDER_DATA)))
    axC.set_xticklabels([f"{d[0]}\n{d[2]}" for d in LADDER_DATA], fontsize=8.0, color=INK)
    axC.set_ylim(0, 108)
    style(axC, "y")
    axC.set_ylabel("AiZynth route-success  → higher is better", fontsize=8.5, color=INK2)
    axC.set_title(
        "C  Are the molecules makeable?",
        fontsize=10.5,
        color=INK,
        fontweight="semibold",
        loc="left",
        pad=8,
    )

    fig.text(
        0.085,
        0.945,
        "Hub-batching is cheaper per mode under every pricing regime — and the reaction-GFN's\n"
        "own routes stay cheaper than any post-hoc planner recovers",
        fontsize=11.4,
        color=INK,
        fontweight="semibold",
        ha="left",
        va="top",
        linespacing=1.45,
    )
    fig.text(
        0.085,
        0.845,
        "SCENT sEH · diversity cutoff τ=0.5 · mode bar reward>7.0 · panel C = the 4,749-molecule mode union",
        fontsize=8.6,
        color=INK2,
        ha="left",
        va="top",
    )
    fig.text(
        0.085,
        0.035,
        "* partial: 3,228/3,500 molecules routed before the 10 h wall (job 71534); finish run 71765 pending.",
        fontsize=7.4,
        color=INK2,
        ha="left",
        va="bottom",
    )

    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=HUB), plt.Rectangle((0, 0), 1, 1, color=BEST)],
        labels=["hub-batching (free-frag)", "best-candidate"],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.885),
        ncol=2,
        frameon=False,
        fontsize=8.8,
        labelcolor=INK2,
        handlelength=0.9,
        handleheight=0.9,
        columnspacing=1.4,
        borderpad=0.2,
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pricing_panel.{ext}", dpi=300, facecolor=SURFACE)
    print(f"wrote {OUT}/pricing_panel.png + .pdf")

    # table view (also the provenance record)
    with open(OUT / "pricing_panel.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["panel", "regime", "strategy", "reactions", "modes", "reactions_per_mode", "source"]
        )
        for lab, hr, hm, br, bm, src in DATA:
            flat = lab.replace("\n", " ")
            w.writerow(["A/B", flat, "hub-batching", hr, hm, f"{hr/hm:.3f}", src])
            w.writerow(["A/B", flat, "best-candidate", br, bm, f"{br/bm:.3f}", src])
        for lab, pct, frac, src, partial in LADDER_DATA:
            w.writerow(
                [
                    "C",
                    lab.replace("\n", " "),
                    "AiZynth route-success",
                    frac,
                    "",
                    f"{pct}%",
                    src + (" [PARTIAL]" if partial else ""),
                ]
            )
    print(f"wrote {OUT}/pricing_panel.csv")


if __name__ == "__main__":
    main()
