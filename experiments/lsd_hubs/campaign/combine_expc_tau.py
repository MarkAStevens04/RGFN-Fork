#!/usr/bin/env python
"""Assemble the Exp C τ-sweep (Logs/048) into one reactions-per-mode vs diversity curve + figure.

Exp C prices SCENT's sEH library through MultiAiZ→SPARROW — the strongest post-hoc planner we give the
competition — one SLURM job per diversity cutoff τ (each τ selects a different library, so MultiAiZ
must re-plan it). This reads whatever τ points exist and emits `expc_tau_curve.{csv,png,pdf}`, so it
is safe to run while jobs are still queued (it reports what is missing).

Reads per τ: `results/scent_seh_multiaiz_<stock>[_t<NN>]/budget_efficiency.csv` (the endpoint row per
strategy = total reactions and modes for that library). reactions/mode = reactions / modes at that
endpoint — the same quantity Logs/041/048 report, so the curves overlay those anchors.

Anchors drawn as single POINTS at τ=0.5, never as horizontal lines: they are measured at that one
cutoff and drawing them across τ would assert a flatness we have not measured.
  * native (by construction, count-once)   hub 1.22   -- Logs/037/044
  * from-scratch AiZynth→SPARROW           hub 2.74   -- Logs/041

Palette = the validated default instance (blue #2a78d6 hub-batching / orange #eb6834 best-candidate;
six-checks validator: CVD ΔE 24.7 protan, normal-vision 33.6, contrast 4.30/3.12 — all PASS).

Usage:  python experiments/lsd_hubs/campaign/combine_expc_tau.py [--stock zincsmall] [--no-plot]
"""
import argparse
import csv
import re
from pathlib import Path

RESULTS = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results")
OUT = Path("experiments/lsd_hubs/campaign/results/expc_tau_curve")
HUB, BEST = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
ANCHORS = [  # (label, tau, rxn_per_mode, source)
    ("native (by construction)", 0.5, 1.22, "Logs/037/044"),
    ("from-scratch AiZynth", 0.5, 2.74, "Logs/041"),
]


def _endpoint(csv_path: Path):
    """{strategy: (reactions, modes)} from the LAST row per strategy of budget_efficiency.csv."""
    out = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["strategy"]] = (int(row["cum_reactions"]), int(row["cum_modes"]))
            except (KeyError, ValueError):
                continue
    return out


def collect(stock: str):
    """[(tau, strategy, reactions, modes, rxn_per_mode, dir)] over every τ dir found for this stock."""
    rows, missing_summary = [], []
    for d in sorted(RESULTS.glob(f"scent_seh_multiaiz_{stock}*")):
        if "smoke" in d.name:
            continue
        # launch_expc_tau_sweep.sh names dirs with the dot STRIPPED from "0.X" (τ=0.3 -> "_t03",
        # τ=0.35 -> "_t035"), so the encoded string is "0" + the decimal digits. Invert by putting
        # the dot back after the first char: "03"->0.3, "07"->0.7, "035"->0.35. The old
        # float(f"0.{m}") read "03" as 0.03 and clustered the whole sweep near zero.
        m = re.search(r"_t(\d+)$", d.name)
        s = m.group(1) if m else None
        tau = float(f"{s[0]}.{s[1:]}") if s else 0.5  # the un-suffixed dir is the τ=0.5 run
        be = d / "budget_efficiency.csv"
        if not be.exists():
            continue
        if not (d / "sweep_summary.json").exists():
            missing_summary.append(
                d.name
            )  # run was killed before the final write (pricing timing lost)
        for strat, (rx, md) in _endpoint(be).items():
            rows.append((tau, strat, rx, md, (rx / md if md else None), d.name))
    return sorted(rows), missing_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stock", default="zincsmall", choices=["zincsmall", "zinc"])
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows, missing = collect(a.stock)
    if not rows:
        raise SystemExit(f"[expc] no τ points yet for stock={a.stock} under {RESULTS}")

    taus = sorted({r[0] for r in rows})
    print(f"[expc] stock={a.stock} | τ points found: {taus}")
    expected = [0.3, 0.5, 0.7, 0.9]
    if [t for t in expected if t not in taus]:
        print(
            f"[expc] STILL MISSING τ={[t for t in expected if t not in taus]} "
            "(jobs queued — re-run this when they land)"
        )
    for name in missing:
        print(
            f"[expc] NOTE {name}: no sweep_summary.json — the run was cut off after writing its "
            "CSVs, so its MultiAiZ pricing TIME is unrecoverable (the rxn/mode numbers are fine)"
        )

    with open(OUT / "expc_tau_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["stock", "tau", "strategy", "reactions", "modes", "reactions_per_mode", "source_dir"]
        )
        for tau, strat, rx, md, rpm, dname in rows:
            w.writerow([a.stock, tau, strat, rx, md, f"{rpm:.3f}" if rpm else "", dname])
        for lab, tau, v, src in ANCHORS:
            w.writerow([f"anchor:{src}", tau, lab, "", "", f"{v:.3f}", src])
    print(f"[expc] wrote {OUT}/expc_tau_curve.csv")
    for tau, strat, rx, md, rpm, _ in rows:
        print(f"    τ={tau}  {strat:15s} {rx:5d} rxn / {md:4d} modes = {rpm:.2f} rxn/mode")

    if a.no_plot:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for strat, color, label in (
        ("hub_batching", HUB, "hub-batching (free-frag)"),
        ("best_candidate", BEST, "best-candidate"),
    ):
        pts = sorted((r[0], r[4]) for r in rows if r[1] == strat and r[4])
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "-o",
            color=color,
            linewidth=2,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=label,
            zorder=3,
        )
        for x, y in pts:
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8.5,
                color=INK,
                fontweight="medium",
            )
    for lab, tau, v, src in ANCHORS:
        ax.plot(
            [tau],
            [v],
            marker="D",
            markersize=7,
            color=INK3,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=4,
            linestyle="none",
        )
        ax.annotate(
            f"{lab} {v:.2f}",
            (tau, v),
            textcoords="offset points",
            xytext=(9, -3),
            ha="left",
            va="center",
            fontsize=8,
            color=INK2,
        )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_xlabel("diversity cutoff τ  (Tanimoto; lower = stricter)", fontsize=9, color=INK2)
    ax.set_ylabel("reactions per mode  ← lower is better", fontsize=9, color=INK2)
    ax.set_xlim(0.25, 0.95)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.set_ylim(bottom=0)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
        labelcolor=INK2,
        handlelength=1.4,
        borderpad=0.2,
    )
    n_tau = len({r[0] for r in rows})
    head = (
        "Hub-batching stays cheaper per mode across the diversity sweep,\n"
        "even priced by the strongest post-hoc planner we give the competition"
        if n_tau >= 3
        else "Reactions per mode under the strongest post-hoc planner\n"
        f"we give the competition — PARTIAL: {n_tau} of 4 diversity cutoffs"
    )
    fig.text(
        0.02,
        0.965,
        head,
        fontsize=10.6,
        color=INK,
        fontweight="semibold",
        ha="left",
        va="top",
        linespacing=1.4,
    )
    fig.text(
        0.02,
        0.845,
        f"SCENT sEH · MultiAiZ→SPARROW · stock = {a.stock} · "
        "grey diamonds = anchors measured at τ=0.5 only",
        fontsize=8.4,
        color=INK2,
        ha="left",
        va="top",
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.74, bottom=0.14)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"expc_tau_curve.{ext}", dpi=300, facecolor=SURFACE)
    print(f"[expc] wrote {OUT}/expc_tau_curve.png + .pdf")


if __name__ == "__main__":
    main()
