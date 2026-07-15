#!/usr/bin/env python
"""Overlay the hub-batching vs best-candidate sweeps across mode-reward thresholds (Logs/035).

The "mode" (hit) bar was 7.0 in Logs/029/033, but the sEH-proxy calibration (Logs/034) showed 7.0 is
far into the proxy-optimized tail — real inhibitors top out ~7.7 and the empirically meaningful
"potential hit" bar is ~5-6. This script re-reads the *committed* per-threshold sweep CSVs (produced by
``sweep_campaign.py`` at ``--reward-threshold 5/6/7``; pure CPU, cached enumeration) and overlays them so
the question "does relaxing the mis-calibrated threshold change the cost/Pareto conclusion?" is one plot.

Strategy = colour (hub-batching / best-candidate); threshold = linestyle (7 solid, 6 dashed, 5 dotted).
Reads only ``results/<tag>/{fixed_modes,pareto}.csv`` — no recompute. Two enum groups (50-hub, 200-hub),
two metrics each (cost to reach M* modes; modes at an R*-reaction budget) -> ``results/threshold_comparison/``.

    python experiments/lsd_hubs/campaign/compare_thresholds.py
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "threshold_comparison"

# (group label, {threshold: results-dir tag}); t7 = the original committed baseline (untagged).
GROUPS = {
    "50-hub": {7: "scent_seh", 6: "scent_seh_thr6", 5: "scent_seh_thr5"},
    "200-hub": {7: "scent_seh_1kx200", 6: "scent_seh_1kx200_thr6", 5: "scent_seh_1kx200_thr5"},
}
STRATS = ("hub_batching", "best_candidate")
COLOR = {"hub_batching": "#1f77b4", "best_candidate": "#d62728"}
STYLE = {7: "-", 6: "--", 5: ":"}
STRATS_LABEL = {"hub_batching": "hub-batching", "best_candidate": "best-candidate"}


def _read(tag: str, fname: str, value_col: str):
    """-> {(strategy, cutoff): float|None} from results/<tag>/<fname>."""
    path = HERE / "results" / tag / fname
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            v = r[value_col]
            out[(r["strategy"], float(r["cutoff"]))] = float(v) if v not in ("", None) else None
    return out


def _cutoffs(d):
    return sorted({c for (_, c) in d})


def _plot(fname, group, value_col, csv_name, ylabel, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tables = {t: _read(tag, csv_name, value_col) for t, tag in group.items()}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for strat in STRATS:
        for thr in (7, 6, 5):
            d = tables[thr]
            cs = _cutoffs(d)
            xs = [c for c in cs if d.get((strat, c)) is not None]
            ys = [d[(strat, c)] for c in xs]
            ax.plot(
                xs,
                ys,
                STYLE[thr],
                color=COLOR[strat],
                marker="o",
                ms=3.5,
                lw=1.6,
                alpha=0.9,
                label=f"{STRATS_LABEL[strat]} · thr {thr}",
            )
    ax.set_xlabel("diversity cutoff (Tanimoto similarity; lower = stricter)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)
    print(f"[compare] wrote {OUT / fname}")


def main():
    for glabel, group in GROUPS.items():
        gslug = glabel.replace("-", "")
        _plot(
            f"cost_{gslug}.png",
            group,
            "reactions_for_300modes",
            "fixed_modes.csv",
            "reactions to generate 300 modes",
            f"sEH {glabel}: cost to reach 300 modes vs threshold (hit bar 5/6/7)",
        )
        _plot(
            f"pareto_{gslug}.png",
            group,
            "modes_at_100rxn",
            "pareto.csv",
            "modes at 100-reaction budget",
            f"sEH {glabel}: Pareto (modes @ 100 rxns) vs threshold (hit bar 5/6/7)",
        )
    print(f"[compare] all overlays in {OUT}")


if __name__ == "__main__":
    main()
