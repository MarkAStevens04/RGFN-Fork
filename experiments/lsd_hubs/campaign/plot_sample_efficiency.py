#!/usr/bin/env python
"""The sample-efficiency curve: does the competitor close the gap by sampling harder?

THE OBJECTION THIS ANSWERS. We deliver ~2.5x more distinct molecules than the SPARROW-batching
competitor at a fixed 100-reaction budget. The obvious reply is that we simply looked at more
candidates. This plots the competitor's distinct-molecule count against the size of its sampling
budget, so the reader can see where -- or whether -- the curve would ever reach us.

WHY A PREFIX IS A HONEST BUDGET SLICE. `records.csv` is a post-hoc sample from a FROZEN checkpoint,
not a training trace: mean reward is flat across the file (7.489 / 7.495 / 7.505 over the first,
middle and last 5k rows). The rows are therefore i.i.d. draws from one fixed policy, and the first k
of them are exactly "what if we had only sampled k times" -- no training-progress confound, and no
new oracle calls, because every molecule was already scored.

THE SECOND COMPETITOR ARM IS THE DECISIVE POINT, and it is why this figure is not just about
sampling. BC-Enum-SB is the same SPARROW batcher fed a pool built by ENUMERATING outward from hubs
rather than by sampling, so its pool cost is the enumeration: 131k-175k children, every one scored.
That puts it 1.8-2.4x ABOVE our own oracle spend on the x-axis while delivering under half as many
molecules -- the candidate-count objection answered by a point rather than by an extrapolation.

WHAT IS AND IS NOT A CONTROLLED SWEEP -- the reason the two arms are drawn differently. The
competitor points ARE one: a single knob (k) moves, everything else is fixed, so they are joined by
a line. Our own configurations vary child-policy AND prebuild-K together, so they are NOT a
controlled sample-efficiency sweep and are drawn as unjoined markers. Connecting them would assert a
curve we have not measured. This distinction was itself a correction: an earlier draft compared a
gate-5.0 run against a gate-7.0 one and read the difference as sampling effect.

Every point here is at reward gate 7.0, tau 0.5, R=100 -- the Logs/062 settings, NOT the newer
headline bar of 5.0, because this figure interrogates the numbers in that entry.

Run:  conda run -n rgfn python experiments/lsd_hubs/campaign/plot_sample_efficiency.py
"""
import csv
import glob
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PREFIX_ROOT = "/scratch/markymoo/rgfn_runs/lsdflow_sparrow/bc_sb_prefix"
OUT = Path("experiments/lsd_hubs/campaign/results/paper_sample_efficiency")

# Reference categorical palette, fixed slot order (slots 1, 2, 3). Light and dark are the same
# eight hues stepped for their surface, not two different palettes.
LIGHT = dict(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink2="#52514e",
    ink3="#8a8984",
    grid="#e6e5e1",
    s1="#2a78d6",
    s2="#eb6834",
    s3="#1baf7a",
    s4="#4a3aa7",
)
BUDGET_R = "100"

# Our own arm at gate 7.0, R=100. (calls, distinct, label) -- see module docstring on why these are
# markers rather than a line.
OURS = [
    (37774, 70, "prebuild-K 20"),
    (74606, 82, "no prebuild-K"),
    (92304, 80, "no prebuild-K, wider sample"),
]
OURS_HEADLINE = (74606, 82)

# BC-Enum-SB: (seed, oracle calls = children enumerated, distinct at R=100, solver converged).
# A capped solve is a LOWER bound on the competitor, so it flatters us -- marked hollow.
ENUM_SB = [
    (42, 131474, 24, False),
    (43, 170724, 33, True),
    (44, 175345, 32, True),
]


def load_prefix():
    """{seed: [(k_events, distinct), ...]} from the prefix sweep."""
    out = {}
    for p in sorted(glob.glob(f"{PREFIX_ROOT}/*/select_frontier.csv")):
        tag = os.path.basename(os.path.dirname(p))
        m = re.match(r"bcsb_prefix_seed(\d+)_k(\d+)", tag)
        if not m:
            continue
        seed, k = int(m.group(1)), int(m.group(2))
        for r in csv.DictReader(open(p)):
            if r["budget_rxns"] != BUDGET_R:
                continue
            out.setdefault(seed, []).append((k, int(r["n_modes_kept"])))
    for v in out.values():
        v.sort()
    return out


def main():
    data = load_prefix()
    if not data:
        raise SystemExit(f"no prefix results under {PREFIX_ROOT}")
    OUT.mkdir(parents=True, exist_ok=True)
    C = LIGHT

    fig, ax = plt.subplots(figsize=(10.2, 6.2), facecolor=C["surface"])
    ax.set_facecolor(C["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["ink3"])
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, axis="y", color=C["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # --- our arm: a reference band + unjoined markers -------------------------------------------
    ax.axhline(
        OURS_HEADLINE[1], color=C["s3"], linewidth=1.4, linestyle=(0, (5, 4)), zorder=2, alpha=0.85
    )
    ox = [p[0] for p in OURS]
    oy = [p[1] for p in OURS]
    ax.scatter(
        ox,
        oy,
        s=95,
        color=C["s3"],
        zorder=5,
        edgecolor=C["surface"],
        linewidth=2,
        marker="D",
        label="hub-batching (ours) — separate configurations",
    )
    ax.annotate(
        f"ours: {OURS_HEADLINE[1]} molecules",
        xy=OURS_HEADLINE,
        xytext=(0, 16),
        textcoords="offset points",
        fontsize=10.5,
        color=C["ink"],
        fontweight="semibold",
        ha="center",
    )

    # --- competitor arm 2: enumerated pool. Unjoined -- three seeds, not a swept knob.
    conv = [(c, d) for _, c, d, ok in ENUM_SB if ok]
    cap = [(c, d) for _, c, d, ok in ENUM_SB if not ok]
    ax.scatter(
        [c for c, _ in conv],
        [d for _, d in conv],
        s=88,
        color=C["s4"],
        zorder=5,
        marker="s",
        edgecolor=C["surface"],
        linewidth=2,
        label="BC-Enum-SB competitor — enumerated pool",
    )
    if cap:
        ax.scatter(
            [c for c, _ in cap],
            [d for _, d in cap],
            s=88,
            facecolor=C["surface"],
            zorder=5,
            marker="s",
            edgecolor=C["s4"],
            linewidth=2,
            label="BC-Enum-SB — solver hit its time limit (lower bound)",
        )
    ax.annotate(
        "outspends us on oracle calls,\ndelivers under half the molecules",
        xy=(172000, 32.5),
        xytext=(0, -46),
        textcoords="offset points",
        fontsize=9.8,
        color=C["ink2"],
        ha="center",
        linespacing=1.5,
        arrowprops=dict(arrowstyle="-", color=C["ink3"], linewidth=0.9, shrinkA=4, shrinkB=8),
    )

    # --- competitor: the controlled prefix sweep -------------------------------------------------
    for (seed, pts), col in zip(sorted(data.items()), (C["s1"], C["s2"])):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=col, linewidth=2, zorder=3, solid_capstyle="round")
        ax.scatter(
            xs,
            ys,
            s=58,
            color=col,
            zorder=4,
            edgecolor=C["surface"],
            linewidth=2,
            label=f"BC-SB competitor — seed {seed}",
        )
        ax.annotate(
            f"{ys[-1]}",
            xy=(xs[-1], ys[-1]),
            xytext=(9, -3),
            textcoords="offset points",
            fontsize=10,
            color=C["ink"],
            fontweight="semibold",
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xlim(1.6e3, 3.4e5)
    ax.set_ylim(0, 97)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}")
    )
    ax.set_xlabel(
        "oracle evaluations spent building the candidate pool  (log scale)",
        fontsize=10.5,
        color=C["ink2"],
        labelpad=9,
    )
    ax.set_ylabel(
        "distinct molecules delivered at 100 reactions", fontsize=10.5, color=C["ink2"], labelpad=9
    )
    ax.tick_params(colors=C["ink2"], labelsize=9.5, length=0)

    fig.text(
        0.085,
        0.965,
        "Sampling harder does not close the gap",
        fontsize=15,
        color=C["ink"],
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.085,
        0.918,
        "A 15× increase in sampling buys the competitor about a third more molecules —\n"
        "and its enumerated arm outspends us outright while still delivering under half.",
        fontsize=10.4,
        color=C["ink2"],
        ha="left",
        va="top",
    )

    leg = ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=9.8,
        labelspacing=0.7,
        handletextpad=0.8,
        borderpad=0.9,
    )
    for t in leg.get_texts():
        t.set_color(C["ink2"])

    fig.text(
        0.008,
        0.012,
        "All points: reward gate 7.0, τ=0.5, fixed 100-reaction budget (the Logs/062 settings). "
        "The competitor's points are a CONTROLLED\nsweep — one knob (how many molecules it "
        "sampled) moves, so they are joined. Ours vary child-policy and pre-filtering\ntogether, "
        "so they are drawn unjoined: connecting them would assert a curve we did not measure.",
        fontsize=8.1,
        color=C["ink3"],
        ha="left",
        va="bottom",
        linespacing=1.55,
    )

    fig.subplots_adjust(left=0.085, right=0.975, top=0.815, bottom=0.225)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"sample_efficiency.{ext}", dpi=200, facecolor=C["surface"])

    with open(OUT / "sample_efficiency.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "seed", "oracle_calls", "distinct_at_R100", "controlled_sweep"])
        for seed, pts in sorted(data.items()):
            for k, d in pts:
                w.writerow(["BC-SB", seed, k, d, True])
        for seed, c, d, ok in ENUM_SB:
            w.writerow(
                ["BC-Enum-SB" + ("" if ok else " (TimeLimit, lower bound)"), seed, c, d, False]
            )
        for k, d, lab in OURS:
            w.writerow([f"hub_batching ({lab})", 42, k, d, False])
    print(f"wrote {OUT}/sample_efficiency.png + .pdf + .csv")
    for seed, pts in sorted(data.items()):
        print(f"  seed {seed}: " + "  ".join(f"{k//1000}k->{d}" for k, d in pts))


if __name__ == "__main__":
    main()
