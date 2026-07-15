#!/usr/bin/env python
"""Per-hub synthesis-batch-size distribution for hub-batching / pre-select-K (Logs/038).

A hub is one synthesis step: build the scaffold once, then run ``k`` diversifying reactions on it to
get ``k`` distinct kept products. That ``k`` (the accepted diverse hits charged to the hub — the
``k`` in the ``depth(h)+k`` cost model) is the hub's **batch size**. If batch sizes are wildly uneven,
one bench step runs 300 parallel reactions while another runs 2 — operationally lopsided even when the
reactions/mode average looks fine. This script measures that distribution and its imbalance, across
the pre-select-K sweep, so we can see whether stocking more universal blocks (larger K) makes the
batches more or less even.

It re-uses the exact same loaders / cost table / campaign as ``preselect_sweep.py`` (Logs/037), so the
K=0/…/200 points here line up with that Pareto. It runs the free-frag child policy with the top-K
pre-built stock, to the mode budget, then groups the accepted modes by ``source_hub`` (the field
``HubBatchingStrategy`` already stamps on every accepted mode) to recover each used hub's batch size.

Diagnostic only: no balancing, no new compute (pure-CPU re-selection over the cached enumeration).

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/batch_size_distribution.py \
        --analysis-dir /scratch/.../lsdflow/scent_seh_70189 \
        --enum-children /scratch/.../campaign_enum_seh_70363/enum_children.json \
        --snapshot /scratch/.../scent_seh/<ts>/additional_fragments/fragments_4000.json \
        --reward-threshold 7.0 --tag scent_seh_batch_dist
"""
import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from run_campaign import _load_candidates, _load_enumerated_hubs, build_strategy

from glue.samplers.lsdflow.campaign import RANK_METHODS, rank_fragments
from glue.samplers.lsdflow.child_select import FreeFragChildPolicy
from validation.lsdflow.metrics.cost.dynamic_amortization import (
    load_cost_table_from_snapshot,
)

HERE = Path(__file__).resolve().parent


def _gini(xs):
    """Gini coefficient of a list of positive counts (0 = perfectly even, →1 = one hub dominates)."""
    n = len(xs)
    if n == 0:
        return float("nan")
    s = sorted(xs)
    total = sum(s)
    if total == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(s))  # i is 0-based → rank (i+1)
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def _pct(xs, q):
    """Percentile via nearest-rank on the sorted list (xs already sortable, q in [0,100])."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]


def _stats_for(batch_sizes):
    """Distribution + imbalance stats for one K's per-hub batch sizes."""
    n = len(batch_sizes)
    total = sum(batch_sizes)
    mx, mn = (max(batch_sizes), min(batch_sizes)) if batch_sizes else (0, 0)
    ordered = sorted(batch_sizes, reverse=True)
    mean = total / n if n else float("nan")
    return {
        "n_hubs_used": n,
        "total_modes": total,
        "min": mn,
        "p25": _pct(batch_sizes, 25),
        "median": statistics.median(batch_sizes) if batch_sizes else float("nan"),
        "mean": round(mean, 3),
        "p75": _pct(batch_sizes, 75),
        "max": mx,
        "max_over_min": round(mx / mn, 2) if mn else None,
        "max_over_median": round(mx / statistics.median(batch_sizes), 2)
        if batch_sizes and statistics.median(batch_sizes)
        else None,
        "cv": round(statistics.pstdev(batch_sizes) / mean, 3) if n and mean else None,
        "gini": round(_gini(batch_sizes), 3),
        "top1_share": round(ordered[0] / total, 3) if total else None,
        "top5_share": round(sum(ordered[:5]) / total, 3) if total else None,
    }


def _plot_distribution(path, per_k, tag):
    """Two-panel diagnostic: (A) box+strip of per-hub batch sizes per K; (B) imbalance vs K."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[batch-dist] plot skipped ({exc})")
        return
    ks = [d["k"] for d in per_k]
    fig, (axb, axi) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Panel A — spread of per-hub batch sizes for each K (log-y; giant hubs annotated).
    data = [d["batch_sizes"] for d in per_k]
    positions = list(range(len(ks)))
    axb.boxplot(data, positions=positions, widths=0.55, showfliers=False, whis=(5, 95))
    for x, sizes in zip(positions, data):
        jitter = [x + (0.12 * ((i % 7) - 3) / 3.0) for i in range(len(sizes))]
        axb.scatter(jitter, sizes, s=9, alpha=0.35, color="#2a6f97", zorder=3)
        mx = max(sizes)
        axb.annotate(
            f"max {mx}",
            (x, mx),
            textcoords="offset points",
            xytext=(6, 2),
            fontsize=7,
            color="#b23a48",
        )
    axb.set_yscale("log")
    axb.set_xticks(positions)
    axb.set_xticklabels([f"K={k}" for k in ks])
    axb.set_ylabel("batch size per hub (accepted diverse hits, log scale)")
    axb.set_title("per-hub batch-size spread")

    # Panel B — imbalance metrics vs K (Gini + top-share on left, max/min ratio on right log axis).
    gini = [d["stats"]["gini"] for d in per_k]
    top1 = [d["stats"]["top1_share"] for d in per_k]
    top5 = [d["stats"]["top5_share"] for d in per_k]
    axi.plot(ks, gini, "-o", ms=5, color="#8a4fbf", label="Gini")
    axi.plot(ks, top1, "-s", ms=5, color="#2a9d8f", label="top-1 hub share")
    axi.plot(ks, top5, "-^", ms=5, color="#e9a20c", label="top-5 hub share")
    axi.set_xlabel("K (fragments pre-synthesized)")
    axi.set_ylabel("Gini / share  (0 = even, 1 = one hub dominates)")
    axi.set_ylim(0, 1.02)
    axr = axi.twinx()
    mm = [d["stats"]["max_over_min"] for d in per_k]
    axr.plot(ks, mm, "--D", ms=4, color="#b23a48", label="max/min ratio")
    axr.set_ylabel("max/min batch-size ratio", color="#b23a48")
    axr.set_yscale("log")
    axr.tick_params(axis="y", labelcolor="#b23a48")
    lines, labels = axi.get_legend_handles_labels()
    l2, lab2 = axr.get_legend_handles_labels()
    axi.legend(lines + l2, labels + lab2, fontsize=8, loc="center right")
    axi.set_title("batch-size imbalance vs K")

    fig.suptitle(f"SCENT {tag}: per-hub synthesis-batch-size distribution (pre-select-K)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[batch-dist] wrote {path}")


def _plot_ranksize(path, per_k, tag):
    """Rank-size (sorted batch sizes, descending) per K — the dominance curve in one view."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[batch-dist] rank-size plot skipped ({exc})")
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    cmap = ["#2a6f97", "#2a9d8f", "#e9a20c", "#8a4fbf", "#b23a48", "#555555"]
    for i, d in enumerate(per_k):
        ordered = sorted(d["batch_sizes"], reverse=True)
        ax.plot(
            range(1, len(ordered) + 1),
            ordered,
            "-o",
            ms=3,
            lw=1.3,
            color=cmap[i % len(cmap)],
            label=f"K={d['k']} ({len(ordered)} hubs)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("hub rank (largest batch first)")
    ax.set_ylabel("batch size (accepted diverse hits, log scale)")
    ax.set_title(f"SCENT {tag}: rank-size of hub batches")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[batch-dist] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--enum-children", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--reward-threshold", type=float, required=True)
    ap.add_argument("--similarity", type=float, default=0.5)
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--budget-modes", type=int, default=300)
    ap.add_argument("--k-list", default="0,16,50,100,200", help="comma-separated K values")
    ap.add_argument("--rank-by", default="build_score", choices=list(RANK_METHODS))
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    cands, comps = _load_candidates(Path(a.analysis_dir), a.higher_is_better)
    enum_hubs = _load_enumerated_hubs(Path(a.enum_children), comps)
    cost_table = load_cost_table_from_snapshot(json.load(open(a.snapshot)))
    raw_children = {h.hub_key: len(h.children) for h in enum_hubs}
    depth_of = {h.hub_key: h.depth for h in enum_hubs}
    ks = [int(x) for x in a.k_list.split(",")]
    ranked = rank_fragments(
        enum_hubs,
        cost_table,
        a.reward_threshold,
        method=a.rank_by,
        higher_is_better=a.higher_is_better,
    )
    common = dict(
        target=a.tag,
        reward_threshold=a.reward_threshold,
        similarity=a.similarity,
        higher_is_better=a.higher_is_better,
    )
    budget = ("modes", a.budget_modes)
    print(f"[batch-dist] {len(enum_hubs)} hubs; K={ks}; budget={budget}; rank-by={a.rank_by}")

    per_k = []
    per_hub_rows = []
    for k in ks:
        prebuilt = {f for f, _ in ranked[:k]} if k > 0 else None
        res = build_strategy(
            "hub_batching",
            enum_hubs,
            cost_table,
            comps,
            child_policy=FreeFragChildPolicy(),
            prebuilt_fragments=prebuilt,
            **common,
        ).run(budget=budget)
        # Batch size = accepted modes charged to each hub (source_hub stamped per mode).
        counts = Counter(p.source_hub for p in res.accepted if p.source_hub is not None)
        batch_sizes = list(counts.values())
        stats = _stats_for(batch_sizes)
        stats["total_reactions"] = res.total_reactions
        stats["reactions_per_mode"] = (
            round(res.total_reactions / res.total_modes, 3) if res.total_modes else None
        )
        stats["reward_gen_calls"] = res.total_reward_gen_calls
        stats["stop_reason"] = res.stop_reason
        per_k.append({"k": k, "batch_sizes": batch_sizes, "stats": stats})
        # rank hubs largest-batch-first for the per-hub CSV
        for rank, (hub_key, bsize) in enumerate(counts.most_common(), start=1):
            per_hub_rows.append(
                {
                    "k": k,
                    "rank": rank,
                    "hub_key": hub_key,
                    "depth": depth_of.get(hub_key),
                    "batch_size": bsize,
                    "raw_children": raw_children.get(hub_key),
                }
            )
        print(
            f"  K={k:>3}: {stats['n_hubs_used']:>3} hubs, {stats['total_modes']} modes | "
            f"batch min/med/max = {stats['min']}/{stats['median']}/{stats['max']} | "
            f"max/min={stats['max_over_min']} Gini={stats['gini']} top1={stats['top1_share']}"
        )

    out = HERE / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)

    # per-hub batch sizes (the raw distribution, largest-first per K)
    with open(out / "batch_per_hub.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["k", "rank", "hub_key", "depth", "batch_size", "raw_children"]
        )
        w.writeheader()
        w.writerows(per_hub_rows)

    # one stats row per K
    stat_keys = list(per_k[0]["stats"].keys())
    with open(out / "batch_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["k"] + stat_keys)
        w.writeheader()
        for d in per_k:
            w.writerow({"k": d["k"], **d["stats"]})

    (out / "batch_summary.json").write_text(
        json.dumps(
            {
                "tag": a.tag,
                "analysis_dir": a.analysis_dir,
                "enum_children": a.enum_children,
                "snapshot": a.snapshot,
                "reward_threshold": a.reward_threshold,
                "similarity": a.similarity,
                "budget_modes": a.budget_modes,
                "rank_by": a.rank_by,
                "child_policy": "free_frag",
                "per_k": [{"k": d["k"], "stats": d["stats"]} for d in per_k],
            },
            indent=2,
        )
    )
    _plot_distribution(out / "batch_distribution.png", per_k, a.tag)
    _plot_ranksize(out / "batch_ranksize.png", per_k, a.tag)
    print(
        f"[batch-dist] wrote batch_per_hub.csv + batch_stats.csv + batch_summary.json + 2 figures to {out}"
    )


if __name__ == "__main__":
    main()
