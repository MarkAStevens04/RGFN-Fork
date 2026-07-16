#!/usr/bin/env python
"""Diversity/budget sweeps for hub-batching vs best-candidate (Logs/029).

Each *simulation* uses ONE predefined budget and yields ONE point — we do NOT read a whole curve off
a single run. The curves below are built by re-running the (cheap, CPU) budget-greedy selection many
times over the *cached* enumeration (``enum_children.json`` — rewards already computed, no re-scoring,
no GPU); only the diversity cutoff / budget changes between runs.

Three plots, each a hub-batching line and a best-candidate line:

  1. Pareto (fixed reaction budget R*): sweep the diversity cutoff -> how many modes.
  2. Fixed modes (fixed mode target M*):  sweep the diversity cutoff -> how many reactions.
  3. Budget vs efficiency (fixed cutoff*): sweep the reaction budget -> how many modes.

Implementation note: one run per cutoff to ``("modes", M*)`` yields all anchors of that cutoff via
its accepted prefix — "modes at <= R* reactions" (Plot 1), "reactions at M* modes" (Plot 2), and the
whole budget curve at the baseline cutoff (Plot 3). The accepted order is best-reward-first and
budget-independent, so this prefix read is identical to (and cheaper than) a separate run per budget
point, and it reads the "<= R*" boundary exactly (a run stopped at R* reactions overshoots by the
mode that crosses R*). Cutoff sweeps DO need one run per cutoff (the accepted set changes).

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/sweep_campaign.py \
        --analysis-dir /scratch/.../lsdflow/scent_seh_70189 \
        --enum-children /scratch/.../campaign_enum_seh_70295/enum_children.json \
        --snapshot /scratch/.../scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json \
        --reward-threshold 7.0 --tag scent_seh
"""
import argparse
import csv
import json
from pathlib import Path

from run_campaign import (  # same-dir helpers
    COMPUTE_COMPONENTS,
    _load_candidates,
    _load_enumerated_hubs,
    build_strategy,
    compute_time_section,
    load_enum_timings,
    load_hub_pick_s,
    run_timed,
)

from glue.samplers.lsdflow.campaign import RANK_METHODS, rank_fragments
from glue.samplers.lsdflow.child_select import make_child_policy
from validation.lsdflow.metrics.cost.dynamic_amortization import (
    load_cost_table_from_snapshot,
)

HERE = Path(__file__).resolve().parent
STRATS = ("hub_batching", "best_candidate")
STRAT_LABEL = {"hub_batching": "Hub Batching", "best_candidate": "Previous"}
POLICY_LABEL = {"reward": "naive", "free_frag": "free frag"}
XLABEL = "Diversity (Tanimoto similarity)"
# Reads left->right after the axis flip: loose (high-Tanimoto, 0.9) on the left, strict (0.3) on the right.
XSUB = "similar modes → diverse modes"


def _title(system, metric_phrase, policy_label, n_hubs, thr):
    """Human-readable title, e.g. 'SCENT sEH proxy - 100 rxn budget - free frag - 200-hub'."""
    parts = [f"SCENT {system}", metric_phrase, policy_label]
    if n_hubs:
        parts.append(f"{n_hubs}-hub")
    if thr != 7.0:  # 7.0 is the default hit bar; only annotate when it differs
        parts.append(f"hit bar {thr:g}")
    return " - ".join(parts)


def _cutoff_grid(lo: float, hi: float, step: float):
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def _modes_at_reactions(result, r_budget: int) -> int:
    return max((p.cum_modes for p in result.accepted if p.cum_reactions <= r_budget), default=0)


def _reactions_at_modes(result, m_budget: int):
    return next((p.cum_reactions for p in result.accepted if p.cum_modes >= m_budget), None)


def _run(
    strategy_name,
    pool,
    cost_table,
    comps,
    similarity,
    common,
    m_budget,
    child_policy=None,
    prebuilt_fragments=None,
):
    """Returns ``(CampaignResult, selection_seconds)`` — the live-measured run wall-clock feeds the
    compute-time accounting (Logs/039)."""
    strat = build_strategy(
        strategy_name,
        pool,
        cost_table,
        comps,
        similarity=similarity,
        child_policy=child_policy,
        prebuilt_fragments=prebuilt_fragments,
        **common,
    )
    return run_timed(strat, ("modes", m_budget))


def _plot(
    path, series, xlabel, ylabel, title, vline=None, invert_x=False, invert_y=False, xsub=None
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[sweep] plot skipped ({exc})")
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, xs, ys in series:
        ax.plot(xs, ys, marker="o", ms=4, lw=1.5, label=label)
    if vline is not None:  # mark the default diversity cutoff on cutoff-axis plots
        ax.axvline(vline, ls="--", lw=1, color="0.55", zorder=0)
        ax.text(
            vline,
            0.94,
            f"default {vline:g}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=8,
            color="0.4",
            rotation=90,
        )
    ax.set_xlabel(xlabel)
    if xsub:  # small grey second line under the x-axis label (direction hint)
        ax.text(
            0.5,
            -0.185,
            xsub,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="0.5",
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Flip axes so the "desired" corner is top-right (Pareto convention): more-diverse (stricter,
    # lower-Tanimoto) cutoffs on the right, and for the cost plot fewer reactions at the top.
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight" if xsub else None)
    print(f"[sweep] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--enum-children", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--reward-threshold", type=float, required=True)
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--budget-reactions", type=int, default=100, help="R* for Plot 1 (Pareto)")
    ap.add_argument("--budget-modes", type=int, default=300, help="M* for Plot 2 + sweep depth")
    ap.add_argument("--cutoff-min", type=float, default=0.30)
    ap.add_argument("--cutoff-max", type=float, default=0.90)
    ap.add_argument("--cutoff-step", type=float, default=0.05)
    ap.add_argument(
        "--baseline-cutoff",
        type=float,
        default=0.50,
        help="the default diversity cutoff — Plot 3's fixed cutoff + the marker on Plots 1/2",
    )
    ap.add_argument(
        "--child-policy",
        default="reward",
        choices=["reward", "free_frag"],
        help="within-hub child selection for the hub_batching line (Logs/037): reward = naive / free_frag",
    )
    ap.add_argument(
        "--prebuild-k", type=int, default=0, help="pre-select-K: pre-synthesize top-K fragments"
    )
    ap.add_argument("--rank-by", default="build_score", choices=list(RANK_METHODS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--system-label", default="sEH proxy", help="system name shown in plot titles")
    ap.add_argument(
        "--n-hubs",
        type=int,
        default=0,
        help="enumeration size shown in titles as '<N>-hub' (0 = omit)",
    )
    ap.add_argument(
        "--enum-timings",
        default=None,
        help="measured per-hub compute timings (Logs/039); default = enum_timings.json beside "
        "--enum-children. Absent → compute-time outputs skipped.",
    )
    ap.add_argument(
        "--hub-pick-timing", default=None, help="pick_hubs_timing.json (default: beside)"
    )
    a = ap.parse_args()
    policy_label = POLICY_LABEL[a.child_policy]

    adir = Path(a.analysis_dir)
    cands, comps = _load_candidates(adir, a.higher_is_better)
    enum_hubs = _load_enumerated_hubs(Path(a.enum_children), comps)
    snapshot = json.load(open(a.snapshot))
    cost_table = load_cost_table_from_snapshot(snapshot)
    child_policy = make_child_policy(a.child_policy)
    prebuilt = None
    if a.prebuild_k > 0:
        ranked = rank_fragments(
            enum_hubs,
            cost_table,
            a.reward_threshold,
            method=a.rank_by,
            higher_is_better=a.higher_is_better,
        )
        prebuilt = {f for f, _ in ranked[: a.prebuild_k]}
    pools = {"best_candidate": cands, "hub_batching": enum_hubs}
    common = dict(
        target=a.tag, reward_threshold=a.reward_threshold, higher_is_better=a.higher_is_better
    )
    cutoffs = _cutoff_grid(a.cutoff_min, a.cutoff_max, a.cutoff_step)
    print(
        f"[sweep] {len(cands)} candidates, {len(enum_hubs)} hubs, {len(cost_table.promoted_set)} "
        f"promoted; cutoffs={cutoffs}"
    )
    out = HERE / "results" / a.tag  # results/<target>/ — untagged names (the dir carries the tag)
    out.mkdir(parents=True, exist_ok=True)
    base = round(
        a.baseline_cutoff, 4
    )  # the default cutoff: Plot 3's fixed value + Plots 1/2 marker

    # One run per (strategy, cutoff) to the mode budget -> all three plots read off the prefixes.
    results = {}  # (strategy, cutoff) -> CampaignResult
    sels = {}  # (strategy, cutoff) -> live selection wall-clock (compute-time accounting, Logs/039)
    for cut in cutoffs:
        for s in STRATS:
            results[(s, cut)], sels[(s, cut)] = _run(
                s,
                pools[s],
                cost_table,
                comps,
                cut,
                common,
                a.budget_modes,
                child_policy,
                prebuilt_fragments=prebuilt,
            )
        hb, bc = results[("hub_batching", cut)], results[("best_candidate", cut)]
        print(
            f"  cutoff {cut:.2f}: hub modes@{a.budget_reactions}rxn={_modes_at_reactions(hb, a.budget_reactions)} "
            f"rxn@{a.budget_modes}modes={_reactions_at_modes(hb, a.budget_modes)} | "
            f"best modes@{a.budget_reactions}rxn={_modes_at_reactions(bc, a.budget_reactions)} "
            f"rxn@{a.budget_modes}modes={_reactions_at_modes(bc, a.budget_modes)}"
        )

    # ---- Plot 1: Pareto (modes at fixed reaction budget vs diversity cutoff) ----
    with open(out / "pareto.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "cutoff", f"modes_at_{a.budget_reactions}rxn"])
        for s in STRATS:
            for cut in cutoffs:
                w.writerow([s, cut, _modes_at_reactions(results[(s, cut)], a.budget_reactions)])
    _plot(
        out / "pareto.png",
        [
            (
                STRAT_LABEL[s],
                cutoffs,
                [_modes_at_reactions(results[(s, c)], a.budget_reactions) for c in cutoffs],
            )
            for s in STRATS
        ],
        XLABEL,
        "modes discovered",
        _title(
            a.system_label,
            f"{a.budget_reactions} rxn budget",
            policy_label,
            a.n_hubs,
            a.reward_threshold,
        ),
        vline=base,
        invert_x=True,  # stricter/more-diverse (low cutoff) -> right; more modes -> up; desired = top-right
        xsub=XSUB,
    )

    # ---- Plot 2: reactions to reach fixed mode target vs diversity cutoff ----
    with open(out / "fixed_modes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "cutoff", f"reactions_for_{a.budget_modes}modes"])
        for s in STRATS:
            for cut in cutoffs:
                w.writerow([s, cut, _reactions_at_modes(results[(s, cut)], a.budget_modes)])
    fm_series = []
    for s in STRATS:
        pts = [(c, _reactions_at_modes(results[(s, c)], a.budget_modes)) for c in cutoffs]
        pts = [(c, r) for c, r in pts if r is not None]  # drop cutoffs that can't reach M*
        fm_series.append((STRAT_LABEL[s], [c for c, _ in pts], [r for _, r in pts]))
    _plot(
        out / "fixed_modes.png",
        fm_series,
        XLABEL,
        f"Reactions required to synthesize {a.budget_modes} modes",
        _title(
            a.system_label,
            f"{a.budget_modes} candidate synthesis",
            policy_label,
            a.n_hubs,
            a.reward_threshold,
        ),
        vline=base,
        invert_x=True,  # stricter/more-diverse (low cutoff) -> right
        invert_y=True,  # fewer reactions (cheaper) -> up; desired = top-right
        xsub=XSUB,
    )

    # ---- Plot 3: budget vs efficiency (modes vs reaction budget) at the default cutoff ----
    if base not in cutoffs:  # ensure the baseline exists even if off-grid
        for s in STRATS:
            results[(s, base)], sels[(s, base)] = _run(
                s,
                pools[s],
                cost_table,
                comps,
                base,
                common,
                a.budget_modes,
                child_policy,
                prebuilt_fragments=prebuilt,
            )
    with open(out / "budget_efficiency.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "cutoff", "cum_reactions", "cum_modes"])
        for s in STRATS:
            for p in results[(s, base)].accepted:
                w.writerow([s, base, p.cum_reactions, p.cum_modes])
    _plot(
        out / "budget_efficiency.png",
        [
            (
                s,
                [p.cum_reactions for p in results[(s, base)].accepted],
                [p.cum_modes for p in results[(s, base)].accepted],
            )
            for s in STRATS
        ],
        "reaction budget (cumulative, true nested cost)",
        "modes obtained",
        f"SCENT {a.tag}: budget vs efficiency (cutoff {base})",
    )

    # ---- Compute-time vs diversity cutoff (Logs/039) — how much longer the computer works ----
    enum_timings = load_enum_timings(a.enum_children, a.enum_timings)
    hub_pick_s = load_hub_pick_s(a.enum_children, a.hub_pick_timing)
    compute_time = None
    if enum_timings is not None:
        sections = {
            cut: compute_time_section(
                results[("hub_batching", cut)],
                sels[("hub_batching", cut)],
                results[("best_candidate", cut)],
                sels[("best_candidate", cut)],
                enum_timings,
                hub_pick_s,
            )
            for cut in cutoffs
        }
        comp_keys = [c[0] for c in COMPUTE_COMPONENTS]
        with open(out / "compute_time_by_cutoff.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cutoff", "strategy", *comp_keys, "total_s", "extra_compute_s"])
            for cut in cutoffs:
                sec = sections[cut]
                extra = sec["head_to_head"]["extra_compute_s"]
                for s in STRATS:
                    bd = sec[s]
                    w.writerow([cut, s, *[bd.get(k, 0.0) for k in comp_keys], bd["total_s"], extra])
        # Plot: total measured compute for each strategy vs cutoff (the extra is the vertical gap).
        # Distinct name from run_campaign's per-component stacked bar (compute_time.png) — same tag dir.
        _plot(
            out / "compute_time_by_cutoff.png",
            [
                (
                    STRAT_LABEL[s],
                    cutoffs,
                    [sections[c][s]["total_s"] for c in cutoffs],
                )
                for s in STRATS
            ],
            XLABEL,
            "measured compute time (s)",
            _title(a.system_label, "compute time", policy_label, a.n_hubs, a.reward_threshold),
            vline=base,
            invert_x=True,  # stricter/more-diverse (low cutoff) -> right
            xsub=XSUB,
        )
        compute_time = {
            "enum_timings_meta": enum_timings.meta,
            "hub_pick_s": hub_pick_s,
            "baseline_cutoff": base,
            "at_baseline": sections.get(base),
            "hub_total_s_by_cutoff": [sections[c]["hub_batching"]["total_s"] for c in cutoffs],
            "best_total_s_by_cutoff": [sections[c]["best_candidate"]["total_s"] for c in cutoffs],
            "extra_compute_s_by_cutoff": [
                sections[c]["head_to_head"]["extra_compute_s"] for c in cutoffs
            ],
        }

    summary = {
        "tag": a.tag,
        "reward_threshold": a.reward_threshold,
        "budget_reactions": a.budget_reactions,
        "budget_modes": a.budget_modes,
        "baseline_cutoff": base,
        "child_policy": a.child_policy,
        "prebuild_k": a.prebuild_k,
        "rank_by": a.rank_by if a.prebuild_k > 0 else None,
        "cutoffs": cutoffs,
        "pareto_modes_at_R": {
            s: [_modes_at_reactions(results[(s, c)], a.budget_reactions) for c in cutoffs]
            for s in STRATS
        },
        "fixed_modes_reactions_at_M": {
            s: [_reactions_at_modes(results[(s, c)], a.budget_modes) for c in cutoffs]
            for s in STRATS
        },
        "budget_efficiency_endpoint": {
            s: {
                "reactions": results[(s, base)].total_reactions,
                "modes": results[(s, base)].total_modes,
            }
            for s in STRATS
        },
        "compute_time": compute_time,  # None if no measured enum_timings.json
    }
    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    if compute_time is not None and compute_time["at_baseline"]:
        h2h = compute_time["at_baseline"]["head_to_head"]
        print(
            f"[sweep] compute-time @cutoff {base}: hub {h2h['hub_total_s']:.1f}s vs best "
            f"{h2h['best_total_s']:.1f}s → +{h2h['extra_compute_s']:.1f}s ({h2h['ratio_hub_over_best']}×)"
        )
    print(
        f"\n[sweep] wrote sweep_summary.json + pareto/fixed_modes/budget_efficiency CSV+PNG to {out}"
    )


if __name__ == "__main__":
    main()
