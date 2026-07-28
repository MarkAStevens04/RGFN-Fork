#!/usr/bin/env python
"""Consolidated per-strategy compute + reward-call summary for LSD-Flow selection.

Puts the three strategies a chemist actually chooses between --- best-candidate,
free-frag (hub-batching K=0) and pre-select-K --- on the *same* two axes:

  1. total measured internal compute, broken down by pipeline component
     (setup / enumeration / reward-gen / flow-extract / mode-selection);
  2. reward-generator calls (== oracle calls in the AL loop).

This is a pure re-plot of the already-measured, committed numbers from
Logs/037 (reward calls) and Logs/039 (measured per-component wall-clock);
it runs no model and touches no cluster. Input is the committed
`scent_seh_1kx200_preselect/preselect.csv` (SCENT sEH, cutoff 0.5, bar 7.0,
300-mode library, surrogate proxy reward).

Usage (any env with pandas+matplotlib, e.g. rgfn/scent/aizynth):
    python strategy_compute_summary.py \
        --preselect results/scent_seh_1kx200_preselect/preselect.csv \
        --out-dir  results/scent_seh_strategy_summary
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from validation.lsdflow.plot_style import ideal_arrow  # noqa: E402

# Pipeline components, in the order they stack, with labels + colours.
# setup + hub_pick are folded together (hub_pick ~= 0.16 s, one-off).
COMPONENTS = [
    ("ct_setup_s", "setup / freeze", "#9aa0a6"),
    ("ct_enumeration_s", "enumeration (RDKit build children)", "#1f6feb"),
    ("ct_reward_gen_s", "reward-gen (score children)", "#f0883e"),
    ("ct_flow_extract_s", "flow-extract (P_F / P_B)", "#3fb950"),
    ("ct_mode_selection_s", "mode-selection (diversity filter)", "#a371f7"),
]


def load_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ct_setup_s"] = df["ct_setup_s"] + df["ct_hub_pick_s"]  # fold one-offs

    def pick(mask):
        sub = df[mask]
        return None if sub.empty else sub.iloc[0]

    picks = [
        ("best-candidate", pick(df["label"] == "best_candidate")),
        ("free-frag (K=0)", pick(df["label"] == "pre-select K=0")),
        ("pre-select K=50", pick(df["label"] == "pre-select K=50")),
        ("pre-select K=100", pick(df["label"] == "pre-select K=100")),
        ("pre-select K=200", pick(df["label"] == "pre-select K=200")),
    ]
    rows = [{"strategy": name, **r.to_dict()} for name, r in picks if r is not None]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preselect", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--title", default="LSD-Flow selection — SCENT sEH (surrogate reward, τ=0.5, 300 modes)"
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = load_rows(args.preselect)
    strategies = d["strategy"].tolist()
    x = range(len(strategies))

    fig, (axc, axr) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Panel A: stacked per-component compute (LINEAR y) -----------------
    # A stacked bar MUST be linear: on a log axis a segment's drawn length depends on where in the
    # stack it starts, so equal durations render at wildly different heights and the segments no
    # longer sum to the bar. (This panel was briefly log-scaled, which made it unreadable as a
    # stack -- and needed a 1e-3 floor hack just to stack off zero.) The ~3000x spread between
    # best-candidate and the hub strategies is not a plotting problem to hide: best-candidate doing
    # essentially no marginal compute IS the finding. It gets a value label so it stays legible.
    bottoms = [0.0] * len(strategies)
    for col, label, colour in COMPONENTS:
        vals = d[col].fillna(0.0).tolist()
        axc.bar(
            list(x),
            vals,
            bottom=bottoms,
            label=label,
            color=colour,
            width=0.62,
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    axc.set_ylim(0, max(bottoms) * 1.16)
    axc.set_ylabel("compute time (s)")
    axc.set_title("Where the time goes (measured, per component)")
    axc.set_xticks(list(x))
    axc.set_xticklabels(strategies, rotation=25, ha="right")
    pad = max(bottoms) * 0.012
    for xi, tot in zip(x, bottoms):
        axc.text(
            xi,
            tot + pad,
            f"{tot:,.0f} s" if tot >= 5 else f"{tot:.1f} s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axc.grid(axis="y", ls=":", alpha=0.4)
    ideal_arrow(axc, "down", label="ideal (less compute)", loc="upper right")

    # ---- Panel B: reward-gen (oracle) calls --------------------------------
    calls = d["reward_gen_calls"].fillna(0).astype(int).tolist()
    bars = axr.bar(list(x), calls, color="#f0883e", width=0.62, edgecolor="white", linewidth=0.4)
    axr.set_ylabel("reward-gen calls  (≈ oracle calls in the AL loop)")
    axr.set_title("How many reward calls")
    axr.set_xticks(list(x))
    axr.set_xticklabels(strategies, rotation=25, ha="right")
    for b, c in zip(bars, calls):
        axr.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(calls) * 0.01,
            f"{c:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axr.grid(axis="y", ls=":", alpha=0.4)
    axr.set_ylim(0, max(calls) * 1.15 if max(calls) else 1)
    ideal_arrow(axr, "down", label="ideal (fewer calls)", loc="upper right")

    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c, label=l) for _, l, c in COMPONENTS]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=5,
        fontsize=8,
        frameon=False,
    )

    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c, label=l) for _, l, c in COMPONENTS]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=5,
        fontsize=8,
        frameon=False,
    )

    fig.suptitle(args.title, fontsize=12, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "Surrogate proxy reward: reward-gen is only ~5% of compute and "
        "enumeration dominates (~79%). With a real docking oracle "
        "(~1 s/call) the reward-gen term dominates instead — the call "
        "count (right) is the reward-agnostic axis.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.88])
    out_png = args.out_dir / "strategy_compute_summary.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # ---- companion CSV + JSON ---------------------------------------------
    keep = [
        "strategy",
        "reactions",
        "reactions_per_mode",
        "hubs_used",
        "reward_gen_calls",
        "compute_total_s",
        "median_reward",
    ] + [c for c, _, _ in COMPONENTS]
    d[keep].to_csv(args.out_dir / "strategy_compute_summary.csv", index=False)
    summary = {
        r["strategy"]: {
            "reward_gen_calls": int(r["reward_gen_calls"]),
            "compute_total_s": round(float(r["compute_total_s"]), 1),
            "reactions": int(r["reactions"]),
            "reactions_per_mode": round(float(r["reactions_per_mode"]), 3),
            "enumeration_frac": round(float(r["ct_enumeration_s"]) / float(r["compute_total_s"]), 3)
            if float(r["compute_total_s"]) > 0
            else None,
        }
        for _, r in d.iterrows()
    }
    (args.out_dir / "strategy_compute_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_png}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
