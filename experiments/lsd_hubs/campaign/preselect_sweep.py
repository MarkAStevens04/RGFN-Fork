#!/usr/bin/env python
"""Pre-select-K sweep for hub-batching (Logs/037): the reactions ↔ reward-gen-calls Pareto.

Pre-select-K = free_frag with the top-K fragments (by ``--rank-by``, default build-score) synthesized
up front. K=0 is plain free_frag. As K grows, each hub yields more free children, so we walk fewer
hubs → fewer reward-gen (≈ oracle) calls, at the price of a few upfront fragment builds. This sweeps K
and writes the curve + a Pareto plot, next to the naive-hub-batching (``reward``) and best-candidate
reference points. Pure CPU (re-selects over the cached enumeration; no GPU, no re-scoring).

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/preselect_sweep.py \
        --analysis-dir /scratch/.../lsdflow/scent_seh_70189 \
        --enum-children /scratch/.../campaign_enum_seh_70363/enum_children.json \
        --snapshot /scratch/.../scent_seh/<ts>/additional_fragments/fragments_4000.json \
        --reward-threshold 7.0 --tag scent_seh_1kx200_preselect
"""
import argparse
import csv
import json
from pathlib import Path

from run_campaign import (
    _load_candidates,
    _load_enumerated_hubs,
    build_strategy,
    load_enum_timings,
    load_hub_pick_s,
    run_timed,
)

from glue.samplers.lsdflow.campaign import RANK_METHODS, rank_fragments
from glue.samplers.lsdflow.child_select import FreeFragChildPolicy, RewardChildPolicy
from validation.lsdflow.metrics.cost.compute_time import account_strategy
from validation.lsdflow.metrics.cost.dynamic_amortization import (
    load_cost_table_from_snapshot,
)

# The per-hub measured components (+ derived) added to each pre-select-K row (Logs/039).
_CT_FIELDS = (
    "setup_s",
    "hub_pick_s",
    "enumeration_s",
    "reward_gen_s",
    "flow_extract_s",
    "mode_selection_s",
)

HERE = Path(__file__).resolve().parent


def _row(label, result, k=None, breakdown=None):
    m = result.total_modes or 1
    row = {
        "label": label,
        "k": k,
        "modes": result.total_modes,
        "reactions": result.total_reactions,
        "reactions_per_mode": round(result.total_reactions / m, 3),
        "hubs_used": result.distinct_hubs_used,
        "distinct_fragments": result.distinct_promoted_fragments,
        "reward_gen_calls": result.total_reward_gen_calls,
        "median_reward": round(result.median_reward, 3)
        if result.median_reward == result.median_reward
        else None,
        "best_reward": round(result.best_reward, 3)
        if result.best_reward == result.best_reward
        else None,
        "stop_reason": result.stop_reason,
    }
    # Measured compute time (Logs/039): the wall-clock companion to reward_gen_calls — pre-select-K's
    # fewer walked hubs cut real enumeration/reward-gen seconds, not just the call count.
    if breakdown is not None:
        for c in _CT_FIELDS:
            row[f"ct_{c}"] = round(getattr(breakdown, c), 4)
        row["compute_total_s"] = round(breakdown.total_s, 4)
    return row


def _run_row(label, strat, budget, enum_timings, hub_pick_s, k=None):
    """Run + time a strategy, and (when timings exist) attach its measured compute-time breakdown."""
    res, sel = run_timed(strat, budget)
    bd = (
        account_strategy(res, enum_timings, selection_s=sel, hub_pick_s=hub_pick_s)
        if enum_timings is not None
        else None
    )
    return _row(label, res, k=k, breakdown=bd)


def _plot(path: Path, rows, tag: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[preselect] plot skipped ({exc})")
        return
    pk = [r for r in rows if r["k"] is not None]
    # annotate only a sparse set of K on the Pareto so dense sweeps stay legible
    ann_ks = {pk[0]["k"], pk[-1]["k"]} | {
        r["k"] for i, r in enumerate(pk) if i % max(1, len(pk) // 8) == 0
    }
    fig, (axp, axk) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Panel A — the canonical reactions↔calls Pareto (pre-select curve + reference points).
    axp.plot(
        [r["reward_gen_calls"] for r in pk],
        [r["reactions_per_mode"] for r in pk],
        "-o",
        ms=4,
        lw=1.4,
        color="#2a6f97",
        label="pre-select-K",
    )
    for r in pk:
        if r["k"] in ann_ks:
            axp.annotate(
                f"K={r['k']}",
                (r["reward_gen_calls"], r["reactions_per_mode"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color="#2a6f97",
            )
    for r in rows:  # naive / best-candidate references
        if r["k"] is None:
            axp.scatter(
                [r["reward_gen_calls"]], [r["reactions_per_mode"]], marker="s", s=45, zorder=5
            )
            axp.annotate(
                r["label"],
                (r["reward_gen_calls"], r["reactions_per_mode"]),
                textcoords="offset points",
                xytext=(5, -10),
                fontsize=8,
            )
    axp.set_xlabel("reward-gen (≈ oracle) calls to the mode budget")
    axp.set_ylabel("reactions / mode")
    axp.set_title("reactions↔calls Pareto")
    axp.legend()

    # Panel B — total reactions vs K (the small-K dip below free-frag is visible here).
    ks = [r["k"] for r in pk]
    rx = [r["reactions"] for r in pk]
    axk.plot(ks, rx, "-o", ms=4, lw=1.4, color="#8a4fbf")
    k0 = next((r["reactions"] for r in pk if r["k"] == 0), None)
    if k0 is not None:
        axk.axhline(k0, ls="--", lw=1, color="0.55")
        axk.text(ks[-1], k0, " free-frag (K=0)", va="bottom", ha="right", fontsize=8, color="0.4")
        kmin = min(pk, key=lambda r: r["reactions"])
        axk.scatter([kmin["k"]], [kmin["reactions"]], marker="v", s=60, color="#8a4fbf", zorder=5)
        axk.annotate(
            f"min K={kmin['k']} ({kmin['reactions']})",
            (kmin["k"], kmin["reactions"]),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=8,
        )
    axk.set_xlabel("K (fragments pre-synthesized)")
    axk.set_ylabel("total reactions for the mode budget")
    axk.set_title("total reactions vs K")

    fig.suptitle(f"SCENT {tag}: pre-select-K")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[preselect] wrote {path}")


def _plot_compute(path: Path, rows, tag: str) -> None:
    """Measured compute vs K (Logs/039): the wall-clock companion to the reactions↔calls Pareto —
    pre-select-K walks fewer hubs → less real enumeration/reward-gen time."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[preselect] compute plot skipped ({exc})")
        return
    pk = [r for r in rows if r["k"] is not None and "compute_total_s" in r]
    if not pk:
        return
    fig, (axt, axc) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ks = [r["k"] for r in pk]
    # Panel A — total compute + its two dominant components vs K.
    axt.plot(ks, [r["compute_total_s"] for r in pk], "-o", ms=4, color="#2a6f97", label="total")
    axt.plot(
        ks, [r["ct_reward_gen_s"] for r in pk], "-s", ms=3, color="#b23a48", label="reward-gen"
    )
    axt.plot(
        ks, [r["ct_enumeration_s"] for r in pk], "-^", ms=3, color="#2a9d8f", label="enumeration"
    )
    axt.set_xlabel("K (fragments pre-synthesized)")
    axt.set_ylabel("measured compute time (s)")
    axt.set_title("compute vs K")
    axt.legend(fontsize=8)
    # Panel B — the measured-time Pareto: compute vs reactions/mode (+ naive / best-candidate refs).
    axc.plot(
        [r["compute_total_s"] for r in pk],
        [r["reactions_per_mode"] for r in pk],
        "-o",
        ms=4,
        color="#8a4fbf",
        label="pre-select-K",
    )
    for r in rows:  # reference points
        if r["k"] is None and "compute_total_s" in r:
            axc.scatter(
                [r["compute_total_s"]], [r["reactions_per_mode"]], marker="s", s=45, zorder=5
            )
            axc.annotate(
                r["label"],
                (r["compute_total_s"], r["reactions_per_mode"]),
                textcoords="offset points",
                xytext=(5, -10),
                fontsize=8,
            )
    axc.set_xlabel("measured compute time (s)")
    axc.set_ylabel("reactions / mode")
    axc.set_title("compute↔reactions Pareto")
    axc.legend(fontsize=8)
    fig.suptitle(f"SCENT {tag}: pre-select-K measured compute time")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[preselect] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--enum-children", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--reward-threshold", type=float, required=True)
    ap.add_argument("--similarity", type=float, default=0.5)
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--budget-modes", type=int, default=300)
    ap.add_argument("--k-list", default="0,5,10,20,50,100,200", help="comma-separated K values")
    ap.add_argument("--rank-by", default="build_score", choices=list(RANK_METHODS))
    ap.add_argument(
        "--enum-timings",
        default=None,
        help="measured per-hub compute timings (Logs/039); default = enum_timings.json beside "
        "--enum-children. Absent → compute-time columns/plot skipped.",
    )
    ap.add_argument(
        "--hub-pick-timing", default=None, help="pick_hubs_timing.json (default: beside)"
    )
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    cands, comps = _load_candidates(Path(a.analysis_dir), a.higher_is_better)
    enum_hubs = _load_enumerated_hubs(Path(a.enum_children), comps)
    cost_table = load_cost_table_from_snapshot(json.load(open(a.snapshot)))
    enum_timings = load_enum_timings(a.enum_children, a.enum_timings)
    hub_pick_s = load_hub_pick_s(a.enum_children, a.hub_pick_timing)
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
    print(
        f"[preselect] {len(enum_hubs)} hubs, {len(ranked)} rankable fragments; K={ks}, rank-by={a.rank_by}"
    )

    rows = []
    # reference: naive hub-batching (reward) + best-candidate
    rows.append(
        _run_row(
            "naive (reward)",
            build_strategy(
                "hub_batching",
                enum_hubs,
                cost_table,
                comps,
                child_policy=RewardChildPolicy(),
                **common,
            ),
            budget,
            enum_timings,
            hub_pick_s,
        )
    )
    rows.append(
        _run_row(
            "best_candidate",
            build_strategy("best_candidate", cands, cost_table, comps, **common),
            budget,
            enum_timings,
            hub_pick_s,
        )
    )
    # pre-select-K curve (K=0 == plain free_frag)
    for k in ks:
        prebuilt = {f for f, _ in ranked[:k]} if k > 0 else None
        rows.append(
            _run_row(
                f"pre-select K={k}",
                build_strategy(
                    "hub_batching",
                    enum_hubs,
                    cost_table,
                    comps,
                    child_policy=FreeFragChildPolicy(),
                    prebuilt_fragments=prebuilt,
                    **common,
                ),
                budget,
                enum_timings,
                hub_pick_s,
                k=k,
            )
        )
        r = rows[-1]
        ct = f" compute={r['compute_total_s']:.1f}s" if "compute_total_s" in r else ""
        print(
            f"  K={k:>3}: rx/mode={r['reactions_per_mode']:.2f} rxns={r['reactions']} "
            f"hubs={r['hubs_used']} calls={r['reward_gen_calls']} med={r['median_reward']}{ct}"
        )

    out = HERE / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "preselect.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "preselect_summary.json").write_text(
        json.dumps(
            {"tag": a.tag, "rank_by": a.rank_by, "budget_modes": a.budget_modes, "rows": rows},
            indent=2,
        )
    )
    _plot(out / "preselect_pareto.png", rows, a.tag)
    if enum_timings is not None:
        _plot_compute(out / "preselect_compute.png", rows, a.tag)
    print(
        f"[preselect] wrote preselect.csv + preselect_summary.json + preselect_pareto.png"
        + (" + preselect_compute.png" if enum_timings is not None else "")
        + f" to {out}"
    )


if __name__ == "__main__":
    main()
