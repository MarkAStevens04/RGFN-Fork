#!/usr/bin/env python
"""Where does each arm's hub set sit in the all-hub flow ranking? (Logs/053)

Answers *why* the three "good" arms land within 9% of each other. The candidate-reward ordering
never reads the flow estimate — it walks sampled molecules best-reward-first and takes their parent
hubs — yet if a trained GFlowNet samples trajectories in proportion to flow, those parents are
**already** high-flow hubs. If so, ordering by candidate reward is a *flow proxy*, and the agreement
between the arms is a positive statement about the flow field rather than evidence that hub ranking
does not matter.

This measures that directly: rank all hubs observed in ``records.csv`` by the same
``max_x F_hat(h;x)`` the production ranker uses, then report where each arm's 200 hubs fall in that
ranking. A uniform draw is the null (median rank ≈ N/2, 5% in the top 5%).

    python experiments/lsd_hubs/hub_order/hub_rank_overlap.py

Pure stdlib + matplotlib; login-node safe, a few seconds. Writes ``results/comparison/
hub_rank_distribution.{png,csv}``.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (arm, display label, colour) — matches compare_hub_order.py so the figures read as one set.
ARMS = [
    ("flow_top", "Highest flow first", "#2a9d8f"),
    ("incumbent", "Incumbent (top-1k cands → flow)", "#1f77b4"),
    ("cand_order", "Best-candidate order", "#e9a20c"),
    ("random", "Random order", "#9aa0a6"),
    ("flow_bottom", "Lowest flow first", "#b23a48"),
]


def all_hub_flow_ranking(records: Path):
    """{(hub_input, depth): rank} under max-child ``log F_hat``, best = rank 0 (the production
    estimate, extended to every observed hub)."""
    best = defaultdict(lambda: -1e18)
    with open(records) as fh:
        for r in csv.DictReader(fh):
            lf = (
                float(r["log_reward"])
                + float(r["log_pb_move"])
                - float(r["log_pf_move"])
                - float(r["log_pf_stop"])
            )
            if not math.isfinite(lf):
                continue
            key = (r.get("hub_stereo_key") or r["hub_key"], int(r["hub_depth"]))
            best[key] = max(best[key], lf)
    order = sorted(best, key=lambda k: best[k], reverse=True)
    return {k: i for i, k in enumerate(order)}, best


def _read_hubs(path: Path):
    with open(path) as fh:
        return [(r["smiles"], int(r["depth"])) for r in csv.DictReader(fh)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-dir", default="/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189")
    ap.add_argument("--out-root", default="/scratch/markymoo/rgfn_runs/lsdflow/hub_order")
    ap.add_argument(
        "--canonical", default="/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363"
    )
    ap.add_argument("--results", default=str(HERE / "results"))
    a = ap.parse_args()

    rank, _ = all_hub_flow_ranking(Path(a.analysis_dir) / "records.csv")
    n = len(rank)
    out = Path(a.results) / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    rows, series = [], []
    for name, label, color in ARMS:
        p = (
            Path(a.canonical) / "hubs.csv"
            if name == "incumbent"
            else Path(a.out_root) / "arms" / name / "hubs.csv"
        )
        if not p.exists():
            print(f"[rank] skip {name} (no hubs.csv at {p})")
            continue
        ranks = [rank[h] for h in _read_hubs(p) if h in rank]
        rows.append(
            {
                "arm": name,
                "n_hubs": len(ranks),
                "median_rank": int(st.median(ranks)),
                "median_percentile": round(100 * st.median(ranks) / n, 2),
                "pct_in_top_1000": round(100 * sum(1 for x in ranks if x < 1000) / len(ranks), 1),
                "pct_in_top_5pct": round(
                    100 * sum(1 for x in ranks if x < n * 0.05) / len(ranks), 1
                ),
                "pct_in_top_half": round(100 * sum(1 for x in ranks if x < n / 2) / len(ranks), 1),
            }
        )
        series.append((label, color, ranks))

    with open(out / "hub_rank_distribution.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"all-hub flow ranking: {n} hubs (uniform null: median {n // 2}, 5% in top 5%)\n")
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in rows[0]}
    print("  ".join(k.ljust(widths[k]) for k in rows[0]))
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in rows[0]))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[rank] plot skipped ({exc})")
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, (label, color, ranks) in enumerate(series):
        y = len(series) - 1 - i
        ax.scatter(
            [r + 1 for r in ranks],
            [y + (j % 7 - 3) * 0.035 for j in range(len(ranks))],
            s=9,
            color=color,
            alpha=0.55,
            edgecolors="none",
        )
        ax.scatter(
            [st.median(ranks) + 1], [y], marker="|", s=420, color=color, linewidths=2.4, zorder=5
        )
    ax.axvline(n / 2, color="#666", ls=":", lw=1.2)
    ax.text(
        n / 2 * 0.92,
        0.02,
        "uniform-draw median ",
        transform=ax.get_xaxis_transform(),
        fontsize=7.5,
        color="#666",
        ha="right",
        va="bottom",
    )
    ax.set_xscale("log")
    ax.set_xlim(1, n)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([lb for lb, _, _ in series][::-1], fontsize=8.5)
    ax.set_xlabel(f"position in the all-hub flow ranking (1 = highest flow, of {n:,}; log scale)")
    ax.set_title(
        "Where each ordering's 200 hubs sit in the flow ranking\n"
        "best-candidate order never reads flow, yet lands in the high-flow region",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(out / "hub_rank_distribution.png", dpi=140)
    print(f"\n[rank] wrote {out / 'hub_rank_distribution.png'} + .csv")


if __name__ == "__main__":
    main()
