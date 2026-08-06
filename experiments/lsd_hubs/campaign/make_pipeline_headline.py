#!/usr/bin/env python
"""Headline figure (Logs/056): the competitor pipeline run as a chemist would run it.

PANEL A is a 2x2, and it is drawn that way ON PURPOSE. The earlier framing (entries 041/048) priced
BOTH pipelines by re-deriving every route from scratch, and under that pricing the non-reaction
baseline won. The reversal is not a different measurement of the same thing — it is the difference
between "re-derive the routes" and "each pipeline priced the way it is actually run". Showing only
the solid pair would be the flattering half of the story, so both regimes are plotted:

    hue     = pipeline (blue = LSD-Flow, orange = S3-GFN)     -- colour follows the ENTITY
    style   = pricing regime (dashed = from-scratch re-derivation, solid = as actually run)

Read three comparisons off the 100-mode guide line:
  * dashed vs dashed  -> from-scratch re-derivation: baseline AHEAD (269 vs 305). We lose.
  * solid  vs solid   -> each as actually run: 411 vs 131 = 3.13x in our favour (the headline).
  * our solid vs their STRONGEST configuration (235: MultiAiZ + diversity-aware greedy) -> 1.79x.
    The CONSERVATIVE number, and the one to quote if a reviewer objects that SPARROW-selects is not
    the baseline at its best. Plotted as a third orange series (dotted).

THE FORMERLY-MISSING CELL WAS RUN, and it did what we suspected: S3-GFN with MultiAiZ AND a
diversity-aware greedy selection reaches 100 modes in **235** reactions, beating both previously
tested baseline configurations (269 from-scratch+greedy, 411 MultiAiZ+SPARROW-selects). It is now
the strongest baseline and the one the conservative claim is measured against, which moves that
claim from 2.05x down to **1.79x**. Running it was the point: it could only ever hurt our number,
so leaving it untested would have left the weakest link unexamined.

PANEL B measures SPARROW's diversity blind spot rather than asserting it: cheap syntheses come from
shared intermediates, which come from structurally similar molecules, so its selections are LESS
diverse than the pool it draws from exactly when the budget binds.

Palette = the project's validated pair (blue #2a78d6 / orange #eb6834). `node` is unavailable on this
cluster, so the six-checks validator was not re-run here; this exact pair's recorded result is
CVD dE 24.7 protan / normal-vision 33.6 / contrast 4.30/3.12, all PASS. Only two hues are used --
the same-entity variants differ by line STYLE, never by a new hue.

Run:  conda run -n rgfn python experiments/lsd_hubs/campaign/make_pipeline_headline.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path("experiments/lsd_hubs/campaign/results")
OUT = RES / "paper_pipeline_headline"
OURS, THEIRS = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3"
TARGET_MODES = 100
POOL_MODE_RATE = 0.412  # S3-GFN top-500's own mode rate (pre-flight saturation, tau=0.5)

# SYMMETRY FIX. Three of the four curves are already SPARROW-priced (both from-scratch arms and the
# baseline's as-run arm all end in the SPARROW MILP). Our native curve is the exception: it is our
# own count-once estimate, because that is what budget_efficiency.csv records. Reading the headline
# off it would compare count-once against SPARROW -- two different cost models -- so the readout is
# overridden with the INDEPENDENT SPARROW audit of the same library at the same 100 modes
# (Logs/056: free-frag K=0, 125 count-once vs 131 SPARROW, agreeing to 4.8%). The curve is still
# drawn from count-once for its shape; only the quoted readout is the audited value, which is what
# makes the headline ratio a like-for-like SPARROW-vs-SPARROW comparison.
OURS_NATIVE_SPARROW = 131


def _budget_curve(path, strategy, cutoff="0.5"):
    """[(reactions, modes)] from a budget_efficiency.csv."""
    pts = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["strategy"] == strategy and r["cutoff"] == cutoff:
                pts.append((int(r["cum_reactions"]), int(r["cum_modes"])))
    return sorted(pts)


def _lerp_x_at_y(pts, y):
    """Interpolate the x (reactions) at which the curve reaches y modes."""
    lo = hi = None
    for x, yy in pts:
        if yy <= y:
            lo = (x, yy)
        if yy >= y and hi is None:
            hi = (x, yy)
    if lo and hi and hi[1] != lo[1]:
        f = (y - lo[1]) / (hi[1] - lo[1])
        return lo[0] + f * (hi[0] - lo[0])
    return (hi or lo)[0] if (hi or lo) else None


def load():
    ours_native = _budget_curve(RES / "scent_seh_freefrag/budget_efficiency.csv", "hub_batching")
    ours_scratch = _budget_curve(
        RES / "scent_seh_sparrow_headline/budget_efficiency.csv", "hub_batching"
    )
    # baseline, as actually run: MultiAiZ routes + SPARROW doing its own selection
    theirs_real = []
    rates = []
    with open(RES / "s3gfn_seh_select_N500/select_frontier.csv") as fh:
        for r in csv.DictReader(fh):
            theirs_real.append((int(float(r["used_rxns"])), int(float(r["n_modes"]))))
            rates.append((int(float(r["used_rxns"])), float(r["mode_rate"])))
    theirs_real.sort()
    rates.sort()
    # baseline at its STRONGEST: MultiAiZ routes + a diversity-aware greedy selection (SPARROW prices)
    theirs_greedy = []
    gp = RES / "s3gfn_seh_greedy_N500/greedy_frontier.csv"
    if gp.exists():
        with open(gp) as fh:
            for r in csv.DictReader(fh):
                theirs_greedy.append((int(float(r["used_rxns"])), int(float(r["n_modes"]))))
        theirs_greedy.sort()
    # baseline, from-scratch re-derivation (entry 048's T3.2 arm); curve rows are [modes, reactions]
    summ = json.load(open(RES / "s3gfn_seh/s3gfn_frontier_summary.json"))
    theirs_scratch = sorted(
        (int(rx), int(md)) for md, rx in summ["curves"]["0.50"] if rx is not None and md is not None
    )
    return ours_native, ours_scratch, theirs_real, theirs_scratch, theirs_greedy, rates


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ours_native, ours_scratch, theirs_real, theirs_scratch, theirs_greedy, rates = load()

    ours_native_countonce = _lerp_x_at_y(ours_native, TARGET_MODES)
    reads = {
        "ours_native": OURS_NATIVE_SPARROW,  # audited value, not the count-once curve (see above)
        "ours_scratch": _lerp_x_at_y(ours_scratch, TARGET_MODES),
        "theirs_real": _lerp_x_at_y(theirs_real, TARGET_MODES),
        "theirs_scratch": _lerp_x_at_y(theirs_scratch, TARGET_MODES),
        "theirs_greedy": _lerp_x_at_y(theirs_greedy, TARGET_MODES) if theirs_greedy else None,
    }
    print(
        f"  (our native count-once curve reads {ours_native_countonce:.0f}; "
        f"quoting the SPARROW audit {OURS_NATIVE_SPARROW} for like-for-like pricing)"
    )
    print(f"reactions to reach {TARGET_MODES} modes:")
    for k, v in reads.items():
        print(f"   {k:<15} {v:.0f}")
    headline = reads["theirs_real"] / reads["ours_native"]
    # conservative = vs the baseline's BEST result over every configuration we have tested
    best_baseline = min(v for k, v in reads.items() if k.startswith("theirs") and v)
    conservative = best_baseline / reads["ours_native"]
    print(f"   headline (solid vs solid)      {headline:.2f}x")
    print(f"   conservative (vs their best)   {conservative:.2f}x")

    fig = plt.figure(figsize=(12.6, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.24)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ---------------- Panel A: the 2x2 frontier ----------------
    axA.set_facecolor(SURFACE)
    series = [
        (ours_scratch, OURS, "--", "LSD-Flow — routes re-derived from scratch", 1.9),
        (theirs_scratch, THEIRS, "--", "S3-GFN — routes re-derived from scratch", 1.9),
        (ours_native, OURS, "-", "LSD-Flow — its own routes (as run)", 2.4),
        (theirs_real, THEIRS, "-", "S3-GFN → MultiAiZ → SPARROW selects (as run)", 2.4),
    ]
    if theirs_greedy:
        series.append(
            (
                theirs_greedy,
                THEIRS,
                ":",
                "S3-GFN → MultiAiZ → diversity-aware greedy (their best)",
                2.4,
            )
        )
    for pts, c, ls, lab, lw in series:
        axA.plot(
            [p[0] for p in pts], [p[1] for p in pts],
            ls, color=c, linewidth=lw, label=lab, zorder=3,
            alpha=0.55 if ls == "--" else 1.0,
            marker="o" if len(pts) < 20 else None, markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2,
        )  # fmt: skip

    axA.axhline(TARGET_MODES, color=INK3, linewidth=1.0, linestyle=":", zorder=2)
    axA.text(
        548, TARGET_MODES + 4, f"the deliverable: {TARGET_MODES} modes",
        fontsize=8.2, color=INK2, va="bottom", ha="right",
    )  # fmt: skip
    # selective direct labels: ONLY the four readouts that the comparisons are made from
    for key, c, dy in (
        ("ours_native", OURS, -13), ("ours_scratch", OURS, -13),
        ("theirs_real", THEIRS, -13), ("theirs_scratch", THEIRS, -13),
        ("theirs_greedy", THEIRS, -13),
    ):  # fmt: skip
        x = reads.get(key)
        if x is None:
            continue
        axA.plot([x], [TARGET_MODES], marker="o", markersize=9, color=c,
                 markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)  # fmt: skip
        axA.annotate(f"{x:.0f}", (x, TARGET_MODES), textcoords="offset points",
                     xytext=(0, dy), ha="center", fontsize=9.4, color=INK,
                     fontweight="semibold", zorder=6)  # fmt: skip

    axA.annotate(
        "", xy=(reads["ours_native"], TARGET_MODES + 26),
        xytext=(reads["theirs_real"], TARGET_MODES + 26),
        arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1.3),
    )  # fmt: skip
    axA.text(
        (reads["ours_native"] + reads["theirs_real"]) / 2, TARGET_MODES + 32,
        f"{headline:.2f}× fewer reactions", ha="center", fontsize=10.2,
        color=INK, fontweight="bold",
    )  # fmt: skip

    axA.set_xlim(0, 560)
    axA.set_ylim(0, 245)
    axA.set_xlabel("reactions (synthetic effort)  ← lower is better", fontsize=9, color=INK2)
    axA.set_ylabel("distinct molecule families delivered", fontsize=9, color=INK2)
    axA.legend(loc="lower right", frameon=False, fontsize=8.4, labelcolor=INK2, handlelength=2.2)
    axA.set_title(
        "A · Same deliverable, two pricing regimes", fontsize=10, color=INK,
        fontweight="semibold", loc="left", pad=8,
    )  # fmt: skip

    # ---------------- Panel B: the measured diversity blind spot ----------------
    axB.set_facecolor(SURFACE)
    axB.plot(
        [r[0] for r in rates], [r[1] for r in rates], "-o", color=THEIRS, linewidth=2.2,
        markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
        label="SPARROW's selection",
    )  # fmt: skip
    axB.axhline(POOL_MODE_RATE, color=INK3, linewidth=1.6, linestyle="--", zorder=2,
                label="the pool it draws from")  # fmt: skip
    worst = min(rates, key=lambda r: r[1])
    axB.annotate(
        f"{worst[1]:.0%} where the\nbudget binds", (worst[0], worst[1]),
        textcoords="offset points", xytext=(14, -6), fontsize=8.6, color=INK,
        fontweight="medium",
    )  # fmt: skip
    axB.text(1010, POOL_MODE_RATE + 0.012, f"{POOL_MODE_RATE:.0%}", fontsize=8.6,
             color=INK2, ha="right")  # fmt: skip
    axB.set_ylim(0, 0.52)
    axB.set_xlabel("reaction budget", fontsize=9, color=INK2)
    axB.set_ylabel("fraction of chosen molecules that are distinct", fontsize=9, color=INK2)
    axB.legend(loc="lower right", frameon=False, fontsize=8.4, labelcolor=INK2, handlelength=2.2)
    axB.set_title(
        "B · Selections are less diverse than the pool",
        fontsize=10, color=INK, fontweight="semibold", loc="left", pad=8,
    )  # fmt: skip

    for ax in (axA, axB):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK3)
            ax.spines[s].set_linewidth(0.8)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8.6, length=0)

    fig.text(
        0.008, 0.975,
        "Priced the way each pipeline is actually run, LSD-Flow delivers the same 100-family "
        "library for 3.1× fewer reactions —\nthe opposite of what the earlier from-scratch pricing "
        "showed, because that regime measured whose chemistry the planner recognised",
        fontsize=11.4, color=INK, fontweight="semibold", ha="left", va="top", linespacing=1.45,
    )  # fmt: skip
    fig.text(
        0.008, 0.020,
        f"SCENT sEH vs S3-GFN (500 candidates, 478 routed) · reward > 7.0 · τ = 0.5 · seed 42.\n"
        f"Both arms priced by SPARROW: our readout is the independent SPARROW audit of the same "
        f"library (our own count-once estimate reads {ours_native_countonce:.0f}, agreeing to 4.8%).\n"
        f"Conservative reading — our {reads['ours_native']:.0f} vs the baseline's STRONGEST tested "
        f"configuration ({best_baseline:.0f}: MultiAiZ + a diversity-aware selection) = {conservative:.2f}×.\n"
        "Asymmetries reported, not matched — pool 500 vs 26,069; surrogate calls ~500 vs 155,764; "
        "route planning 2.25 h vs 0.",
        fontsize=7.5, color=INK2, ha="left", va="bottom", linespacing=1.5,
    )  # fmt: skip
    fig.subplots_adjust(left=0.062, right=0.988, top=0.80, bottom=0.28)

    with open(OUT / "pipeline_headline.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "pricing_regime", f"reactions_for_{TARGET_MODES}_modes", "source"])
        w.writerow(["LSD-Flow", "native (as run)", f"{reads['ours_native']:.0f}",
                    "results/scent_seh_freefrag/budget_efficiency.csv"])  # fmt: skip
        w.writerow(["LSD-Flow", "from-scratch", f"{reads['ours_scratch']:.0f}",
                    "results/scent_seh_sparrow_headline/budget_efficiency.csv (Logs/041)"])  # fmt: skip
        w.writerow(["S3-GFN", "MultiAiZ + SPARROW selects (as run)", f"{reads['theirs_real']:.0f}",
                    "results/s3gfn_seh_select_N500/select_frontier.csv (Logs/056)"])  # fmt: skip
        w.writerow(["S3-GFN", "from-scratch", f"{reads['theirs_scratch']:.0f}",
                    "results/s3gfn_seh/s3gfn_frontier_summary.json (Logs/048)"])  # fmt: skip
        w.writerow(["ratio", "headline (solid vs solid)", f"{headline:.3f}", ""])
        if reads.get("theirs_greedy"):
            w.writerow(
                [
                    "S3-GFN",
                    "MultiAiZ + diversity-aware greedy (their BEST)",
                    f"{reads['theirs_greedy']:.0f}",
                    "results/s3gfn_seh_greedy_N500/greedy_frontier.csv (Logs/056)",
                ]
            )
        w.writerow(
            ["ratio", "conservative (vs their strongest tested config)", f"{conservative:.3f}", ""]
        )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pipeline_headline.{ext}", dpi=300, facecolor=SURFACE)
    print(f"wrote {OUT}/pipeline_headline.png + .pdf + .csv")


if __name__ == "__main__":
    main()
