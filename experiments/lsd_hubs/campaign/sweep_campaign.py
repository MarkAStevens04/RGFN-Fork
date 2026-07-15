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
    _load_candidates,
    _load_enumerated_hubs,
    build_strategy,
)

from glue.samplers.lsdflow.child_select import make_child_policy
from validation.lsdflow.metrics.cost.dynamic_amortization import (
    load_cost_table_from_snapshot,
    scaled_fragment_utilities,
)

HERE = Path(__file__).resolve().parent
STRATS = ("hub_batching", "best_candidate")


def _cutoff_grid(lo: float, hi: float, step: float):
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def _modes_at_reactions(result, r_budget: int) -> int:
    return max((p.cum_modes for p in result.accepted if p.cum_reactions <= r_budget), default=0)


def _reactions_at_modes(result, m_budget: int):
    return next((p.cum_reactions for p in result.accepted if p.cum_modes >= m_budget), None)


def _run(strategy_name, pool, cost_table, comps, similarity, common, m_budget, child_policy=None):
    return build_strategy(
        strategy_name,
        pool,
        cost_table,
        comps,
        similarity=similarity,
        child_policy=child_policy,
        **common,
    ).run(budget=("modes", m_budget))


def _plot(path, series, xlabel, ylabel, title, vline=None):
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
            0.98,
            f"default {vline:g}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=8,
            color="0.4",
            rotation=90,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
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
        choices=["reward", "free_frag", "smart_frag", "marginal", "smart_dyn", "ratio"],
        help="within-hub child selection for the hub_batching line (Logs/037)",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="penalty weight (smart_frag/marginal/smart_dyn/ratio)",
    )
    ap.add_argument("--utility-scale", default="logbeta", choices=["logbeta", "log", "raw"])
    ap.add_argument("--beta-train", type=float, default=8.0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    adir = Path(a.analysis_dir)
    cands, comps = _load_candidates(adir, a.higher_is_better)
    enum_hubs = _load_enumerated_hubs(Path(a.enum_children), comps)
    snapshot = json.load(open(a.snapshot))
    cost_table = load_cost_table_from_snapshot(snapshot)
    utilities = (
        scaled_fragment_utilities(snapshot, beta_train=a.beta_train, scale=a.utility_scale)
        if a.child_policy in ("smart_frag", "smart_dyn")
        else None
    )
    child_policy = make_child_policy(a.child_policy, beta=a.beta, utilities=utilities)
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
    for cut in cutoffs:
        for s in STRATS:
            results[(s, cut)] = _run(
                s, pools[s], cost_table, comps, cut, common, a.budget_modes, child_policy
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
                s,
                cutoffs,
                [_modes_at_reactions(results[(s, c)], a.budget_reactions) for c in cutoffs],
            )
            for s in STRATS
        ],
        "diversity cutoff (Tanimoto similarity; lower = stricter)",
        f"modes at {a.budget_reactions}-reaction budget",
        f"SCENT {a.tag}: Pareto (fixed {a.budget_reactions} reactions)",
        vline=base,
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
        fm_series.append((s, [c for c, _ in pts], [r for _, r in pts]))
    _plot(
        out / "fixed_modes.png",
        fm_series,
        "diversity cutoff (Tanimoto similarity; lower = stricter)",
        f"reactions to generate {a.budget_modes} modes",
        f"SCENT {a.tag}: cost to reach {a.budget_modes} modes",
        vline=base,
    )

    # ---- Plot 3: budget vs efficiency (modes vs reaction budget) at the default cutoff ----
    if base not in cutoffs:  # ensure the baseline exists even if off-grid
        for s in STRATS:
            results[(s, base)] = _run(
                s, pools[s], cost_table, comps, base, common, a.budget_modes, child_policy
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

    summary = {
        "tag": a.tag,
        "reward_threshold": a.reward_threshold,
        "budget_reactions": a.budget_reactions,
        "budget_modes": a.budget_modes,
        "baseline_cutoff": base,
        "child_policy": a.child_policy,
        "beta": a.beta if a.child_policy == "smart_frag" else None,
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
    }
    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\n[sweep] wrote sweep_summary.json + pareto/fixed_modes/budget_efficiency CSV+PNG to {out}"
    )


if __name__ == "__main__":
    main()
