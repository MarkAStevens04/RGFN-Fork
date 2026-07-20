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
from validation.lsdflow.eval import (
    CountOnceEvaluator,
    LibrarySet,
    canonical,
    load_price_table,
)
from validation.lsdflow.metrics.cost.compute_time import account_strategy
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


# ---- evaluator seam (T0.3) ------------------------------------------------------------------------
# The frontier no longer reads reactions straight off the strategy's CampaignResult; it prices ORDERED
# SNAPSHOTS of the selection with an INJECTED Evaluator (docs/LSD_FLOW_BENCHMARK_PLAN.md §0/T0.3), so
# the same driver serves count-once, from-scratch SPARROW, and MultiAiZ. --evaluator count_once
# surfaces the strategy's own DAG count-once estimate (CHECK 2); with the default every-mode schedule
# it reproduces the pre-seam curves bit-for-bit. sparrow/multiaiz (from-scratch pricing) are T1.4/T4.1.
_LIVE_EVALUATORS = ("count_once", "sparrow")
_PLANNED_EVALUATORS = {"multiaiz": "T4.1 (validation/lsdflow/eval/multiaiz.py)"}


def make_evaluator(name, a):
    """Injected Evaluator for the frontier. count_once = the strategy's own DAG estimate (CHECK 2);
    sparrow = from-scratch AiZynth->SPARROW MILP (HEADLINE) or native-route SPARROW (CHECK 1) via
    --route-source; multiaiz is planned (T4.1). Fails loud on unknown/unbuilt names."""
    if name == "count_once":
        return CountOnceEvaluator()
    if name == "sparrow":
        from validation.lsdflow.eval.route_recovery import RouteCache
        from validation.lsdflow.eval.sparrow import SparrowEvaluator

        # Route cache + MILP scratch live on $SCRATCH by default (many small per-snapshot files);
        # the cache is chemistry-specific, so its default name carries stock+expansion.
        scratch = Path(a.sparrow_work_dir) if a.sparrow_work_dir else _default_sparrow_dir(a.tag)
        cache_path = (
            Path(a.sparrow_cache)
            if a.sparrow_cache
            else scratch / f"routecache_{a.aizynth_stock}_{a.aizynth_expansion}.json"
        )
        price_table = (
            load_price_table(Path(a.price_library) / "fragments.csv") if a.price_library else None
        )
        return SparrowEvaluator(
            route_source=a.route_source,
            cache=RouteCache(cache_path),
            objective=a.milp_objective,
            aizynth_env=a.aizynth_env,
            sparrow_env=a.sparrow_env,
            aizynth_config=a.aizynth_config,
            stock=a.aizynth_stock,
            expansion=a.aizynth_expansion,
            filter_policy=(
                None if (a.aizynth_filter or "none").lower() == "none" else a.aizynth_filter
            ),
            time_limit=a.aizynth_time_limit,
            nproc=a.aizynth_nproc,
            max_seconds=a.milp_max_seconds,
            work_dir=scratch,
            price_table=price_table,
        )
    if name in _PLANNED_EVALUATORS:
        raise NotImplementedError(
            f"--evaluator {name} is not built yet — see {_PLANNED_EVALUATORS[name]}. "
            f"Available now: {sorted(_LIVE_EVALUATORS)}."
        )
    raise SystemExit(
        f"unknown --evaluator {name!r}; available: {sorted(_LIVE_EVALUATORS)} "
        f"(+ planned: {sorted(_PLANNED_EVALUATORS)})"
    )


def _default_sparrow_dir(tag):
    """Default SPARROW scratch (route cache + per-snapshot MILP files) — $SCRATCH if set, else /tmp;
    kept OUT of the repo (many small transient files)."""
    import os

    base = os.environ.get("SCRATCH", "/tmp")
    return Path(base) / "lsdflow_sparrow" / tag


def _snapshot_sizes(n, schedule, k, points):
    """Prefix sizes (ascending, 1..n, always including n) at which to price the ordering.

    ``all`` = every mode (bit-for-bit for count_once; the default). ``every_k`` = k, 2k, …, n.
    ``geometric`` = ~``points`` log-spaced sizes — for expensive per-snapshot pricers (SPARROW MILP)
    where one solve per mode is infeasible; the read-time slicers step over the coarse curve."""
    if n <= 0:
        return []
    if schedule == "all":
        return list(range(1, n + 1))
    if schedule == "every_k":
        step = max(1, int(k))
        sizes = list(range(step, n + 1, step))
        if not sizes or sizes[-1] != n:
            sizes.append(n)
        return sizes
    if schedule == "geometric":
        if n <= points:
            return list(range(1, n + 1))
        sizes = sorted({max(1, int(round(n ** (i / (points - 1))))) for i in range(points)})
        if sizes[-1] != n:
            sizes.append(n)
        return sizes
    raise SystemExit(f"unknown snapshot schedule {schedule!r} (all | every_k | geometric)")


def evaluate_ordering(
    result, evaluator, *, schedule="all", k=25, points=15, routes=None, provenance=None
):
    """Price snapshots of a strategy's ordering into a ``[(modes, reactions), …]`` curve.

    Each snapshot is the first-``size`` accepted modes as a :class:`LibrarySet`, carrying (a) the
    strategy's own count-once estimate at that prefix (``accepted[size-1].cum_reactions`` — CHECK 2)
    and (b) native routes when available (for the native-route/from-scratch SPARROW pricers).

    Curve point = ``(priced_modes, reactions)``. The mode axis is the evaluator's ``n_priced``, NOT
    the raw selection count: the from-scratch pricer EXCLUDES modes it can't route (plan T1.3), so a
    snapshot of 60 selected modes at 70% solve-rate is a 42-mode library priced over 42 modes — using
    the raw 60 would put the x and y on different sets. For count-once ``n_priced == n_modes`` (every
    mode has a DAG estimate), so this is a no-op there (curves stay bit-for-bit). ``reactions``/
    ``priced_modes`` may be ``None``/``0`` (unpriced) ⇒ dropped by the read-time slicers.
    """
    accepted = result.accepted
    curve = []
    for size in _snapshot_sizes(len(accepted), schedule, k, points):
        sub = accepted[:size]
        smis = [p.smiles for p in sub]
        lib = LibrarySet(
            smiles=smis,
            rewards={p.smiles: p.reward for p in sub},
            routes=({s: routes[s] for s in smis if s in routes} if routes else None),
            provenance=provenance or {},
            count_once_reactions=sub[-1].cum_reactions,
        )
        res = evaluator.score(lib)
        modes = res.n_priced if res.n_priced is not None else sub[-1].cum_modes
        curve.append((modes, res.total_reactions))
    return curve


# ---- T2.2 compute frontier ("where does the time go") --------------------------------------------
# Per-method end-to-end wall-clock to PRODUCE its library, differentiated by stage, for the SPARROW
# frontier (docs/LSD_FLOW_BENCHMARK_PLAN.md T2.2). Composes the count-once generation/selection
# compute (account_strategy, Logs/039) with the two SPARROW-evaluation stages:
#   * route_finding — from-scratch AiZynth search, AMORTIZED PER MOLECULE (Σ search_time over the
#     modes THIS method selected, from the timed RouteCache; shared molecules counted once by the
#     cache). 0 for native-route (routes are by-construction).
#   * sparrow_mip   — the MILP that prices the method's FINAL library (build + solve), measured by
#     re-scoring that library once (not summed over snapshots — snapshots are our curve-drawing
#     device, not the chemist's cost).
# Emits a reactions-per-compute-hour scalar with a lab_hours_per_reaction hook (=1.0; §8): later one
# config turn converts compute-hours saved into bench-days/$ saved.
_FRONTIER_COMPONENTS = [
    ("generation_s", "generation (sampling)", "#8a4fbf"),
    ("hub_pick_s", "hub pick", "#577590"),
    ("enumeration_s", "enumeration", "#2a9d8f"),
    ("reward_gen_s", "reward-gen", "#b23a48"),
    ("flow_extract_s", "flow-extract", "#e9a20c"),
    ("mode_selection_s", "mode-select", "#2a6f97"),
    ("route_finding_s", "route-finding (AiZynth)", "#bc5090"),
    ("sparrow_mip_s", "SPARROW MILP", "#ff764a"),
]


def _sample_generation_s(analysis_dir: Path) -> float:
    """Stage-1 sampling wall-clock (shared by both strategies) from sample_timings.json if present
    (the SCENT worker writes it beside records.csv); 0 if absent."""
    for p in (analysis_dir / "sample_timings.json", analysis_dir.parent / "sample_timings.json"):
        if p.exists():
            try:
                tot = json.load(open(p)).get("meta", {}).get("totals_s", {})
                return float(tot.get("sampling_s", 0.0))
            except Exception:  # noqa: BLE001
                return 0.0
    return 0.0


def sparrow_compute_frontier(
    results, sels, evaluator, enum_timings, hub_pick_s, cutoff, generation_s, lab_hours_per_reaction
):
    """Per-method compute breakdown + reactions-per-compute-hour at one cutoff (SPARROW evaluator).

    Re-scores each strategy's final library once for the MILP time and reads route-finding from the
    evaluator's timed RouteCache (amortized per molecule). Returns ``{strategy: {components..,
    total_s, reactions, reactions_per_compute_hour}}``."""
    cache = getattr(evaluator, "cache", None)
    strip = getattr(evaluator, "strip_stereo", True)
    out = {}
    for s in STRATS:
        res = results[(s, cutoff)]
        bd = account_strategy(
            res,
            enum_timings,
            selection_s=sels[(s, cutoff)],
            hub_pick_s=(hub_pick_s if s == "hub_batching" else 0.0),
        )
        comp = bd.components()  # count-once side (setup/hub_pick/enum/reward-gen/flow/mode-select)
        comp["generation_s"] = float(generation_s)
        # route-finding: Σ AiZynth search_time over the modes this method selected (per-molecule).
        modes = [p.smiles for p in res.accepted]
        canons = {c for c in (canonical(m, strip) for m in modes) if c}
        comp["route_finding_s"] = (
            float(cache.total_search_time(canons)) if cache is not None else 0.0
        )
        # SPARROW MILP for the FINAL library (one re-score; the chemist's price, not our snapshots').
        lib = LibrarySet(
            smiles=modes,
            rewards={p.smiles: p.reward for p in res.accepted},
            provenance={"strategy": s, "cutoff": cutoff, "stage": "compute_frontier"},
            count_once_reactions=res.total_reactions,
        )
        ev = evaluator.score(lib)
        comp["sparrow_mip_s"] = float(ev.timing_s.get("sparrow_build", 0.0)) + float(
            ev.timing_s.get("sparrow_mip", 0.0)
        )
        keys = [k for k, _, _ in _FRONTIER_COMPONENTS]
        total_s = sum(comp.get(k, 0.0) for k in keys)
        reactions = ev.total_reactions
        rpch = (
            (reactions / (total_s / 3600.0) * lab_hours_per_reaction)
            if (reactions and total_s > 0)
            else None
        )
        out[s] = {
            **{k: round(comp.get(k, 0.0), 4) for k in keys},
            "total_s": round(total_s, 4),
            "reactions": reactions,
            "n_priced_modes": ev.n_priced,
            "reactions_per_compute_hour": (round(rpch, 4) if rpch is not None else None),
        }
    return out


def plot_compute_frontier(path, frontier, tag, lab_hours_per_reaction):
    """Stacked horizontal bar of per-method compute by stage (which stage dominates each method)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[frontier] plot skipped ({exc})")
        return
    strategies = [("hub_batching", "Hub Batching"), ("best_candidate", "Best Candidate")]
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    for row, (skey, slabel) in enumerate(strategies):
        bd = frontier.get(skey, {})
        left = 0.0
        for comp, clabel, color in _FRONTIER_COMPONENTS:
            val = float(bd.get(comp, 0.0) or 0.0)
            if val <= 0:
                continue
            ax.barh(
                row,
                val,
                left=left,
                color=color,
                edgecolor="white",
                height=0.62,
                label=clabel if row == 0 else None,
            )
            left += val
        rpch = bd.get("reactions_per_compute_hour")
        ax.text(
            left,
            row,
            f"  {left:.0f}s" + (f" · {rpch:g} rxn/cpu-hr" if rpch else ""),
            va="center",
            ha="left",
            fontsize=8,
        )
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([s[1] for s in strategies])
    ax.set_xlabel("measured compute time (s) to produce the library")
    ax.set_title(
        f"{tag}: compute frontier by stage (lab_hours_per_reaction={lab_hours_per_reaction:g})",
        fontsize=10,
    )
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[frontier] wrote {path}")


def _modes_at_reactions(curve, r_budget: int) -> int:
    """Largest mode count reachable within the reaction budget (read-time, over the priced curve)."""
    return max((m for m, r in curve if r is not None and r <= r_budget), default=0)


def _reactions_at_modes(curve, m_budget: int):
    """Reactions to reach the mode target (read-time); ``None`` if never reached / unpriced."""
    return next((r for m, r in curve if m >= m_budget and r is not None), None)


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
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output root for results/<tag>/ (default: <script dir>/results). Set to a $SCRATCH path "
        "when running on a Balam compute node ($HOME is read-only there); sync back afterward.",
    )
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
    ap.add_argument(
        "--evaluator",
        default="count_once",
        help="how the frontier prices each ordered snapshot (T0.3): count_once = the strategy's own "
        "DAG estimate (CHECK 2, default, reproduces the pre-seam curves); sparrow/multiaiz = "
        "from-scratch route-recovery + MILP (T1.4/T4.1, planned).",
    )
    ap.add_argument(
        "--snapshot-schedule",
        default="all",
        choices=["all", "every_k", "geometric"],
        help="prefix sizes priced per ordering: all = every mode (bit-for-bit for count_once); "
        "every_k / geometric = coarse (for expensive per-snapshot pricers like SPARROW).",
    )
    ap.add_argument(
        "--snapshot-k", type=int, default=25, help="step for --snapshot-schedule every_k"
    )
    ap.add_argument(
        "--snapshot-points", type=int, default=15, help="#points for --snapshot-schedule geometric"
    )
    # ---- SPARROW evaluator config (only used when --evaluator sparrow) ----
    ap.add_argument(
        "--route-source",
        default="from_scratch",
        choices=["from_scratch", "native"],
        help="sparrow: from_scratch = AiZynth re-routes every molecule (HEADLINE); native = use the "
        "generator's own routes from --routes (CHECK 1).",
    )
    ap.add_argument(
        "--milp-objective",
        default="count",
        choices=["count", "count_cost"],
        help="sparrow MILP objective: count = minimize #reactions (default); count_cost = + "
        "starting-material $ (needs --price-library). [feasibility TODO — see memory]",
    )
    ap.add_argument("--aizynth-env", default="aizynth")
    ap.add_argument("--sparrow-env", default="sparrow")
    ap.add_argument("--aizynth-config", default="data/models/aizynthfinder/config.yml")
    ap.add_argument("--aizynth-stock", default="zinc")
    ap.add_argument("--aizynth-expansion", default="uspto")
    ap.add_argument("--aizynth-filter", default="uspto")
    ap.add_argument(
        "--aizynth-time-limit", type=int, default=60, help="per-molecule AiZynth seconds"
    )
    ap.add_argument("--aizynth-nproc", type=int, default=8)
    ap.add_argument("--milp-max-seconds", type=int, default=600, help="per-snapshot MILP seconds")
    ap.add_argument("--sparrow-cache", default=None, help="route cache JSON (default: on $SCRATCH)")
    ap.add_argument("--sparrow-work-dir", default=None, help="MILP scratch (default: on $SCRATCH)")
    ap.add_argument(
        "--price-library", default=None, help="glue library dir for building-block $ (count_cost)"
    )
    ap.add_argument(
        "--lab-hours-per-reaction",
        type=float,
        default=1.0,
        help="compute-frontier hook (T2.2/§8): multiplier turning reactions into lab-time; 1.0 for "
        "now — later one config turn converts compute-hours saved into bench-days/$ saved.",
    )
    a = ap.parse_args()
    policy_label = POLICY_LABEL[a.child_policy]
    evaluator = make_evaluator(a.evaluator, a)  # fail loud on unknown / not-yet-built evaluators
    # Expensive per-snapshot pricers (SPARROW MILP) can't afford one solve per mode; default them to a
    # coarse geometric schedule unless the user asked otherwise. count_once stays 'all' (bit-for-bit).
    if a.evaluator != "count_once" and a.snapshot_schedule == "all":
        a.snapshot_schedule = "geometric"
        print(
            f"[sweep] --evaluator {a.evaluator}: snapshot schedule -> geometric "
            f"({a.snapshot_points} pts) so the MILP isn't run once per mode "
            f"(override with --snapshot-schedule)"
        )

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
    # results/<target>/ — untagged names (the dir carries the tag). Default is the in-repo results
    # dir; --out-dir redirects it (e.g. to $SCRATCH when running on a Balam compute node, where $HOME
    # is read-only — then sync back to the repo from a login node).
    out = (Path(a.out_dir) if a.out_dir else HERE / "results") / a.tag
    out.mkdir(parents=True, exist_ok=True)
    base = round(
        a.baseline_cutoff, 4
    )  # the default cutoff: Plot 3's fixed value + Plots 1/2 marker

    # One run per (strategy, cutoff) to the mode budget -> all three plots read off the prefixes.
    # The strategy emits an ORDERING (CampaignResult.accepted); the injected evaluator prices its
    # snapshots into a (modes, reactions) curve, and all read-time budgets slice that curve (T0.3).
    results = {}  # (strategy, cutoff) -> CampaignResult
    sels = {}  # (strategy, cutoff) -> live selection wall-clock (compute-time accounting, Logs/039)
    curves = {}  # (strategy, cutoff) -> [(modes, reactions), ...] priced by the injected evaluator
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
            curves[(s, cut)] = evaluate_ordering(
                results[(s, cut)],
                evaluator,
                schedule=a.snapshot_schedule,
                k=a.snapshot_k,
                points=a.snapshot_points,
                provenance={
                    "strategy": s,
                    "cutoff": cut,
                    "target": a.tag,
                    "evaluator": evaluator.name,
                },
            )
        hb, bc = curves[("hub_batching", cut)], curves[("best_candidate", cut)]
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
                w.writerow([s, cut, _modes_at_reactions(curves[(s, cut)], a.budget_reactions)])
    _plot(
        out / "pareto.png",
        [
            (
                STRAT_LABEL[s],
                cutoffs,
                [_modes_at_reactions(curves[(s, c)], a.budget_reactions) for c in cutoffs],
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
                w.writerow([s, cut, _reactions_at_modes(curves[(s, cut)], a.budget_modes)])
    fm_series = []
    for s in STRATS:
        pts = [(c, _reactions_at_modes(curves[(s, c)], a.budget_modes)) for c in cutoffs]
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
            curves[(s, base)] = evaluate_ordering(
                results[(s, base)],
                evaluator,
                schedule=a.snapshot_schedule,
                k=a.snapshot_k,
                points=a.snapshot_points,
                provenance={
                    "strategy": s,
                    "cutoff": base,
                    "target": a.tag,
                    "evaluator": evaluator.name,
                },
            )
    # The one inherently-cumulative curve: the priced (reactions, modes) points at the default cutoff.
    be_curve = {s: [(m, r) for m, r in curves[(s, base)] if r is not None] for s in STRATS}
    with open(out / "budget_efficiency.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "cutoff", "cum_reactions", "cum_modes"])
        for s in STRATS:
            for m, r in be_curve[s]:
                w.writerow([s, base, r, m])
    _plot(
        out / "budget_efficiency.png",
        [
            (
                s,
                [r for _m, r in be_curve[s]],
                [m for m, _r in be_curve[s]],
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

    # ---- T2.2 compute frontier (SPARROW evaluators only; count-once has the Logs/039 view above) ----
    compute_frontier = None
    if a.evaluator != "count_once":
        gen_s = _sample_generation_s(adir)
        frontier = sparrow_compute_frontier(
            results,
            sels,
            evaluator,
            enum_timings,
            hub_pick_s,
            base,
            gen_s,
            a.lab_hours_per_reaction,
        )
        keys = [k for k, _, _ in _FRONTIER_COMPONENTS]
        with open(out / "compute_frontier.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "strategy",
                    *keys,
                    "total_s",
                    "reactions",
                    "n_priced_modes",
                    "reactions_per_compute_hour",
                ]
            )
            for s in STRATS:
                bd = frontier[s]
                w.writerow(
                    [
                        s,
                        *[bd.get(k, 0.0) for k in keys],
                        bd["total_s"],
                        bd["reactions"],
                        bd["n_priced_modes"],
                        bd["reactions_per_compute_hour"],
                    ]
                )
        plot_compute_frontier(
            out / "compute_frontier.png", frontier, a.tag, a.lab_hours_per_reaction
        )
        compute_frontier = {
            "baseline_cutoff": base,
            "lab_hours_per_reaction": a.lab_hours_per_reaction,
            "generation_s": gen_s,
            "by_strategy": frontier,
        }
        for s in STRATS:
            bd = frontier[s]
            print(
                f"[frontier] {s}: total {bd['total_s']:.0f}s | route-finding {bd['route_finding_s']:.0f}s "
                f"| MILP {bd['sparrow_mip_s']:.1f}s | {bd['reactions']} rxn over {bd['n_priced_modes']} modes "
                f"-> {bd['reactions_per_compute_hour']} rxn/cpu-hr"
            )

    summary = {
        "tag": a.tag,
        "reward_threshold": a.reward_threshold,
        "budget_reactions": a.budget_reactions,
        "budget_modes": a.budget_modes,
        "baseline_cutoff": base,
        "child_policy": a.child_policy,
        "prebuild_k": a.prebuild_k,
        "rank_by": a.rank_by if a.prebuild_k > 0 else None,
        "evaluator": a.evaluator,  # how snapshots were priced (T0.3): count_once | sparrow | multiaiz
        "snapshot_schedule": a.snapshot_schedule,
        "lab_hours_per_reaction": a.lab_hours_per_reaction,
        "cutoffs": cutoffs,
        "pareto_modes_at_R": {
            s: [_modes_at_reactions(curves[(s, c)], a.budget_reactions) for c in cutoffs]
            for s in STRATS
        },
        "fixed_modes_reactions_at_M": {
            s: [_reactions_at_modes(curves[(s, c)], a.budget_modes) for c in cutoffs]
            for s in STRATS
        },
        "budget_efficiency_endpoint": {
            s: {
                "reactions": (be_curve[s][-1][1] if be_curve[s] else None),
                "modes": (be_curve[s][-1][0] if be_curve[s] else 0),
            }
            for s in STRATS
        },
        "compute_time": compute_time,  # None if no measured enum_timings.json
        "compute_frontier": compute_frontier,  # T2.2; None for count_once
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
