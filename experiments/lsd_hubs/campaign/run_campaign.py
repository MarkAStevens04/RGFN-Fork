#!/usr/bin/env python
"""Hub-batching vs best-candidate campaign on a SCENT analysis (Logs/028).

Loads a completed SCENT hub-analysis dir (``records.csv`` = candidate pool, ``compositions.json`` =
per-molecule promoted fragments), a scent-env enumeration (``enum_children.json`` = each ranked
hub's children + the fragment added in the final reaction), and the recipe snapshot
(``fragments_<N>.json``), then runs both strategies to the full modes-vs-reactions curve and reads
off Case 1 (modes at a reaction budget) + Case 2 (reactions at a mode budget). Pure CPU.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/run_campaign.py \
        --analysis-dir /scratch/.../lsdflow/scent_seh_70189 \
        --enum-children /scratch/.../campaign_enum/enum_children.json \
        --snapshot /scratch/.../scent_seh/<ts>/additional_fragments/fragments_4000.json \
        --reward-threshold 7.0 --tag scent_seh
"""
import argparse
import csv
import json
import time
from pathlib import Path

from glue.samplers.lsdflow.campaign import (
    RANK_METHODS,
    BestCandidateStrategy,
    Candidate,
    EnumChild,
    EnumeratedHub,
    HubBatchingStrategy,
    rank_fragments,
)
from glue.samplers.lsdflow.child_select import make_child_policy
from validation.lsdflow.metrics.cost.compute_time import EnumTimings as _EnumTimings
from validation.lsdflow.metrics.cost.compute_time import account_strategy, head_to_head
from validation.lsdflow.metrics.cost.dynamic_amortization import (
    load_cost_table_from_snapshot,
)
from validation.lsdflow.plot_style import title_with_ideal

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------- compute-time helpers (Logs/039)
# Shared by all four campaign drivers: measure each strategy's live CPU selection wall-clock, load
# the worker's MEASURED per-hub enum timings, and attribute them over the strategy's actual walk.


def run_timed(strategy, budget):
    """Run a strategy and measure its live CPU wall-clock — the real Stage-4 mode-selection +
    book-keeping over the cached rewards. Returns ``(CampaignResult, selection_seconds)``."""
    t0 = time.perf_counter()
    res = strategy.run(budget=budget)
    return res, time.perf_counter() - t0


def load_enum_timings(enum_children_path, explicit=None):
    """Measured per-hub enum timings (Logs/039), defaulting to ``enum_timings.json`` beside
    ``enum_children.json``. Returns ``None`` if absent → the compute-time section is skipped
    (backward-compatible with pre-Logs/039 enumerations)."""
    p = Path(explicit) if explicit else Path(enum_children_path).parent / "enum_timings.json"
    return _EnumTimings.load(p) if p.exists() else None


def load_hub_pick_s(enum_children_path, explicit=None):
    """Stage-2 hub-pick wall-clock from ``pick_hubs_timing.json`` beside ``enum_children.json``
    (0.0 if absent). Charged to hub-batching only (best-candidate never picks hubs)."""
    p = Path(explicit) if explicit else Path(enum_children_path).parent / "pick_hubs_timing.json"
    if not p.exists():
        return 0.0
    try:
        return float(json.load(open(p)).get("hub_pick_s", 0.0))
    except Exception:  # noqa: BLE001
        return 0.0


def compute_time_section(hb, hb_sel, bc, bc_sel, enum_timings, hub_pick_s):
    """Differentiated compute-time breakdown for a hub-batching / best-candidate pair + the
    head-to-head. Returns ``None`` when no measured timings are available."""
    if enum_timings is None:
        return None
    hb_bd = account_strategy(hb, enum_timings, selection_s=hb_sel, hub_pick_s=hub_pick_s)
    bc_bd = account_strategy(bc, enum_timings, selection_s=bc_sel)
    return {
        "enum_timings_meta": enum_timings.meta,
        "hub_batching": hb_bd.to_dict(),
        "best_candidate": bc_bd.to_dict(),
        "head_to_head": head_to_head(hb_bd, bc_bd),
    }


def _load_candidates(analysis_dir: Path, higher_is_better: bool):
    """Candidate pool from ``records.csv`` + ``compositions.json``. Each candidate also carries its
    observed immediate-parent hub keys (all distinct ``hub_key`` it was seen under) so best-candidate
    can credit "accidental" hub-batching (Logs/033)."""
    comps = json.load(open(analysis_dir / "compositions.json"))
    best: dict = {}  # child_key -> reward (best)
    parents: dict = {}  # child_key -> set of observed parent hub keys
    with open(analysis_dir / "records.csv") as fh:
        for r in csv.DictReader(fh):
            c, reward = r["child_key"], float(r["reward"])
            if c not in best or (reward > best[c]) == higher_is_better:
                best[c] = reward
            hk = r.get("hub_key")
            if hk:
                parents.setdefault(c, set()).add(hk)
    cands = []
    for c, reward in best.items():
        comp = comps.get(c, {}) or {}
        cands.append(
            Candidate(
                smiles=c,
                reward=reward,
                num_reactions=int(comp.get("num_reactions", 1)),
                promoted=tuple(comp.get("promoted", ())),
                parents=tuple(sorted(parents.get(c, ()))),
            )
        )
    return cands, comps


def build_strategy(
    name: str,
    pool,
    cost_table,
    comps: dict,
    *,
    similarity: float,
    target: str,
    reward_threshold: float,
    higher_is_better: bool,
    assignment_policy=None,
    child_policy=None,
    prebuilt_fragments=None,
):
    """Construct either strategy on the ONE count-once cost model (Logs/033). Best-candidate gets the
    compositions (to cost shared parent hubs) + a swappable hub-assignment policy; hub-batching needs
    only its enumerated hubs + an optional within-hub ``child_policy`` (Logs/037: reward = naive /
    free_frag) and ``prebuilt_fragments`` (pre-select-K). Shared by ``run_campaign`` and
    ``sweep_campaign``."""
    common = dict(
        target=target,
        reward_threshold=reward_threshold,
        similarity=similarity,
        higher_is_better=higher_is_better,
    )
    if name == "hub_batching":
        return HubBatchingStrategy(
            pool,
            cost_table,
            child_policy=child_policy,
            prebuilt_fragments=prebuilt_fragments,
            **common,
        )
    return BestCandidateStrategy(
        pool,
        cost_table,
        hub_compositions=comps,
        assignment_policy=assignment_policy,
        **common,
    )


def _load_enumerated_hubs(enum_children_path: Path, comps: dict):
    data = json.load(open(enum_children_path))
    hubs = []
    for h in data.get("hubs", []):
        hub_comp = comps.get(h["hub_key"], {}) or {}
        hubs.append(
            EnumeratedHub(
                hub_key=h["hub_key"],
                hub_input=h.get(
                    "hub_input", h["hub_key"]
                ),  # unique id for compute-time join (Logs/039)
                depth=int(h["depth"]),
                promoted=tuple(hub_comp.get("promoted", ())),
                children=[
                    EnumChild(
                        smiles=c["smiles"],
                        reward=float(c["reward"]),
                        added_promoted=tuple(c.get("added_promoted", ())),
                    )
                    for c in h["children"]
                ],
                uncertainty=h.get("uncertainty"),  # U(h), carried for later (not used by selection)
                n_effective=int(h.get("n_effective", 0)),
            )
        )
    return hubs


def _readouts(result, budget_reactions: int, budget_modes: int) -> dict:
    """Case 1 (modes at a reaction budget) + Case 2 (reactions at a mode budget) off the curve."""
    modes_at_rxn = max(
        (p.cum_modes for p in result.accepted if p.cum_reactions <= budget_reactions), default=0
    )
    rxn_at_modes = next(
        (p.cum_reactions for p in result.accepted if p.cum_modes >= budget_modes), None
    )
    rgc = result.total_reward_gen_calls
    return {
        "strategy": result.strategy,
        "total_modes": result.total_modes,
        "total_reactions": result.total_reactions,
        "reactions_per_mode": round(result.total_reactions / result.total_modes, 3)
        if result.total_modes
        else None,
        "total_reward_gen_calls": rgc,
        "reward_gen_calls_per_mode": round(rgc / result.total_modes, 2)
        if result.total_modes
        else None,
        "distinct_promoted_fragments": result.distinct_promoted_fragments,
        "distinct_hubs_used": result.distinct_hubs_used,
        "n_scaffolds": result.n_scaffolds,
        "best_reward": round(result.best_reward, 3)
        if result.best_reward == result.best_reward
        else None,
        "median_reward": round(result.median_reward, 3)
        if result.median_reward == result.median_reward
        else None,
        f"case1_modes_at_{budget_reactions}rxn": modes_at_rxn,
        f"case2_reactions_at_{budget_modes}modes": rxn_at_modes,
        "stop_reason": result.stop_reason,
    }


def _write_curve(path: Path, result) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "strategy",
                "step",
                "cum_reactions",
                "cum_modes",
                "cum_reward_gen_calls",
                "reactions_added",
                "reward",
                "source_hub",
            ]
        )
        for p in result.accepted:
            w.writerow(
                [
                    result.strategy,
                    p.step,
                    p.cum_reactions,
                    p.cum_modes,
                    p.cum_reward_gen_calls,
                    p.reactions_added,
                    round(p.reward, 4),
                    p.source_hub or "",
                ]
            )


def _plot(path: Path, results, tag: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[campaign] plot skipped ({exc})")
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for res in results:
        xs = [p.cum_reactions for p in res.accepted]
        ys = [p.cum_modes for p in res.accepted]
        ax.plot(xs, ys, marker=".", ms=3, lw=1.5, label=res.strategy)
    ax.set_xlabel("cumulative reactions (true nested cost, count-once)")
    ax.set_ylabel("cumulative modes (diverse hits)")
    # Both axes are metrics here, so the marker names them (see plot_style: title-only, never drawn
    # inside the axes).
    ax.set_title(
        title_with_ideal(
            f"SCENT {tag}: hub-batching vs best-candidate",
            ("higher", "modes"),
            ("lower", "reactions"),
        )
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[campaign] wrote {path}")


# The differentiated compute-time components, in display order (matches ComputeTimeBreakdown).
COMPUTE_COMPONENTS = [
    ("setup_s", "setup (load+freeze)", "#8a4fbf"),
    ("hub_pick_s", "hub pick", "#577590"),
    ("enumeration_s", "enumeration", "#2a9d8f"),
    ("reward_gen_s", "reward-gen", "#b23a48"),
    ("flow_extract_s", "flow-extract", "#e9a20c"),
    # RGFN's enumeration is one opaque adapter call -> exact per-hub total, no observable split.
    # Grey so it reads as "measured but unsplit" rather than as a named pipeline stage.
    ("unattributed_s", "enum+reward+flow (unsplit)", "#9aa0a6"),
    ("mode_selection_s", "mode-select", "#2a6f97"),
]


def plot_compute_time(path: Path, section: dict, tag: str) -> None:
    """Stacked horizontal bar of the differentiated compute-time per strategy (Logs/039). Shows where
    the wall-clock goes and how much longer hub-batching runs than best-candidate."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[campaign] compute-time plot skipped ({exc})")
        return
    strategies = [("hub_batching", "Hub Batching"), ("best_candidate", "Best Candidate")]
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    for row, (skey, slabel) in enumerate(strategies):
        bd = section.get(skey, {})
        left = 0.0
        for comp, clabel, color in COMPUTE_COMPONENTS:
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
        ax.text(left, row, f" {left:.1f}s", va="center", ha="left", fontsize=9)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([s[1] for s in strategies])
    ax.set_xlabel("measured compute time (seconds)")
    h2h = section.get("head_to_head", {})
    extra = h2h.get("extra_compute_s")
    ratio = h2h.get("ratio_hub_over_best")
    sub = ""
    if extra is not None:
        sub = f"  (+{extra:.0f}s"
        sub += f", {ratio:g}× vs best-candidate)" if ratio else ")"
    ax.set_title(
        title_with_ideal(f"{tag}: measured compute time by component{sub}", "lower"), fontsize=10
    )
    ax.legend(fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[campaign] wrote {path}")


def write_compute_time_csv(path: Path, section: dict) -> None:
    """One row per strategy: each component (s) + total_s + the counts, for the paper table."""
    comps = [c[0] for c in COMPUTE_COMPONENTS]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", *comps, "total_s", "n_hubs_walked", "n_children_scored"])
        for skey in ("hub_batching", "best_candidate"):
            bd = section.get(skey, {})
            w.writerow(
                [skey]
                + [bd.get(c, 0.0) for c in comps]
                + [
                    bd.get("total_s", 0.0),
                    bd.get("n_hubs_walked", 0),
                    bd.get("n_children_scored", 0),
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument(
        "--enum-children", required=True, help="enum_children.json from the scent worker"
    )
    ap.add_argument(
        "--snapshot",
        default="",
        help="SCENT fragments_<N>.json (promoted-fragment recipes) for the nested cost model. "
        "Omit for generators with no dynamic library (RGFN/FragGFN/RxnFlow) -> the cost model "
        "falls back to min_num_reactions (no promoted fragments to nest).",
    )
    ap.add_argument("--reward-threshold", type=float, required=True)
    ap.add_argument(
        "--similarity", type=float, default=0.5
    )  # the campaign default cutoff (Logs/029)
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--budget-reactions", type=int, default=100, help="Case 1 reaction budget")
    ap.add_argument("--budget-modes", type=int, default=300, help="Case 2 mode budget")
    ap.add_argument(
        "--child-policy",
        default="reward",
        choices=["reward", "free_frag"],
        help="within-hub child selection for hub-batching (Logs/037); best-candidate is unaffected. "
        "reward = naive hub-batching (no fragment-cost awareness); free_frag = keep only "
        "already-available-fragment children (base + prebuilt stock).",
    )
    ap.add_argument(
        "--prebuild-k",
        type=int,
        default=0,
        help="pre-select-K (Logs/037): pre-synthesize the top-K fragments (by --rank-by), charge them "
        "upfront, then run the child policy (use with free_frag).",
    )
    ap.add_argument(
        "--rank-by",
        default="build_score",
        choices=list(RANK_METHODS),
        help="pre-select ranking: build_score = (reward-bar)*fanout/build_reactions (default); "
        "fanout / reward isolate one signal (ablations).",
    )
    ap.add_argument(
        "--enum-timings",
        default=None,
        help="measured per-hub compute timings (Logs/039); default = enum_timings.json beside "
        "--enum-children. Absent → compute-time section skipped.",
    )
    ap.add_argument(
        "--hub-pick-timing",
        default=None,
        help="pick_hubs_timing.json (Stage-2 hub-pick wall-clock); default = beside --enum-children.",
    )
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--out-dir",
        default="",
        help="results dir (default: <campaign>/results/<tag>); matrix16 routes cells to "
        "experiments/lsd_hubs/matrix16/results/<tag>",
    )
    a = ap.parse_args()

    adir = Path(a.analysis_dir)
    cands, comps = _load_candidates(adir, a.higher_is_better)
    enum_hubs = _load_enumerated_hubs(Path(a.enum_children), comps)
    snapshot = json.load(open(a.snapshot)) if a.snapshot else {}
    cost_table = load_cost_table_from_snapshot(snapshot)
    print(
        f"[campaign] {len(cands)} candidates, {len(enum_hubs)} enumerated hubs, "
        f"{len(cost_table.promoted_set)} promoted fragments (recipes={bool(cost_table.recipes)})"
    )

    child_policy = make_child_policy(a.child_policy)

    # Pre-select-K (Logs/037): synthesize the top-K fragments by --rank-by up front, charge once.
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
        upfront = cost_table.shared_build_cost(prebuilt)[0] if cost_table else 0
        print(
            f"[campaign] pre-select-K: {len(prebuilt)} fragments pre-synthesized "
            f"(top {a.rank_by}), {upfront} reactions charged upfront"
        )

    common = dict(
        target=a.tag,
        reward_threshold=a.reward_threshold,
        similarity=a.similarity,
        higher_is_better=a.higher_is_better,
    )
    # Run to the mode budget (greedy diversity is O(modes^2); Case 1's reaction point is reached
    # well before Case 2's mode budget, so this single curve covers both readouts).
    curve_budget = ("modes", a.budget_modes)
    # Time each run live (selection wall-clock) for the compute-time accounting (Logs/039).
    bc, bc_sel = run_timed(
        build_strategy("best_candidate", cands, cost_table, comps, **common), curve_budget
    )
    hb, hb_sel = run_timed(
        build_strategy(
            "hub_batching",
            enum_hubs,
            cost_table,
            comps,
            child_policy=child_policy,
            prebuilt_fragments=prebuilt,
            **common,
        ),
        curve_budget,
    )
    enum_timings = load_enum_timings(a.enum_children, a.enum_timings)
    hub_pick_s = load_hub_pick_s(a.enum_children, a.hub_pick_timing)
    ct_section = compute_time_section(hb, hb_sel, bc, bc_sel, enum_timings, hub_pick_s)
    if bc.total_reactions < a.budget_reactions or hb.total_reactions < a.budget_reactions:
        print(
            f"[campaign] WARNING a curve stopped below the Case-1 reaction budget "
            f"({a.budget_reactions}); raise --budget-modes for a valid Case-1 readout."
        )

    out = Path(a.out_dir) if a.out_dir else HERE / "results" / a.tag  # dir carries the tag
    out.mkdir(parents=True, exist_ok=True)
    _write_curve(out / "curve_best_candidate.csv", bc)
    _write_curve(out / "curve_hub_batching.csv", hb)
    summary = {
        "tag": a.tag,
        "reward_threshold": a.reward_threshold,
        "similarity": a.similarity,
        "budget_reactions": a.budget_reactions,
        "budget_modes": a.budget_modes,
        "child_policy": a.child_policy,
        "prebuild_k": a.prebuild_k,
        "rank_by": a.rank_by if a.prebuild_k > 0 else None,
        "best_candidate": _readouts(bc, a.budget_reactions, a.budget_modes),
        "hub_batching": _readouts(hb, a.budget_reactions, a.budget_modes),
        "compute_time": ct_section,  # None if no measured enum_timings.json found
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(out / "curve.png", [bc, hb], a.tag)
    if ct_section is not None:
        write_compute_time_csv(out / "compute_time.csv", ct_section)
        plot_compute_time(out / "compute_time.png", ct_section, a.tag)
        h2h = ct_section["head_to_head"]
        print(
            f"[campaign] compute-time: hub-batching {h2h['hub_total_s']:.1f}s vs best-candidate "
            f"{h2h['best_total_s']:.1f}s → +{h2h['extra_compute_s']:.1f}s extra "
            f"({h2h['ratio_hub_over_best']}× )"
        )
    else:
        print("[campaign] compute-time: no enum_timings.json found → section skipped")
    print(json.dumps(summary, indent=2))
    print(f"\n[campaign] wrote summary.json + curve_*.csv to {out}")


if __name__ == "__main__":
    main()
