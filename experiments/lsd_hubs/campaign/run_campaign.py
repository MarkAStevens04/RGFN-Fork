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
from pathlib import Path

from glue.samplers.lsdflow.campaign import (
    BestCandidateStrategy,
    Candidate,
    EnumChild,
    EnumeratedHub,
    HubBatchingStrategy,
)
from validation.lsdflow.metrics.cost.dynamic_amortization import (
    load_cost_table_from_snapshot,
)

HERE = Path(__file__).resolve().parent


def _load_candidates(analysis_dir: Path, higher_is_better: bool):
    comps = json.load(open(analysis_dir / "compositions.json"))
    best: dict = {}  # child_key -> reward (best)
    with open(analysis_dir / "records.csv") as fh:
        for r in csv.DictReader(fh):
            c, reward = r["child_key"], float(r["reward"])
            if c not in best or (reward > best[c]) == higher_is_better:
                best[c] = reward
    cands = []
    for c, reward in best.items():
        comp = comps.get(c, {}) or {}
        cands.append(
            Candidate(
                smiles=c,
                reward=reward,
                num_reactions=int(comp.get("num_reactions", 1)),
                promoted=tuple(comp.get("promoted", ())),
            )
        )
    return cands, comps


def _load_enumerated_hubs(enum_children_path: Path, comps: dict):
    data = json.load(open(enum_children_path))
    hubs = []
    for h in data.get("hubs", []):
        hub_comp = comps.get(h["hub_key"], {}) or {}
        hubs.append(
            EnumeratedHub(
                hub_key=h["hub_key"],
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
    ax.set_title(f"SCENT {tag}: hub-batching vs best-candidate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[campaign] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument(
        "--enum-children", required=True, help="enum_children.json from the scent worker"
    )
    ap.add_argument("--snapshot", required=True, help="fragments_<N>.json with smiles_to_route")
    ap.add_argument("--reward-threshold", type=float, required=True)
    ap.add_argument(
        "--similarity", type=float, default=0.5
    )  # the campaign default cutoff (Logs/029)
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--budget-reactions", type=int, default=100, help="Case 1 reaction budget")
    ap.add_argument("--budget-modes", type=int, default=300, help="Case 2 mode budget")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    adir = Path(a.analysis_dir)
    cands, comps = _load_candidates(adir, a.higher_is_better)
    enum_hubs = _load_enumerated_hubs(Path(a.enum_children), comps)
    cost_table = load_cost_table_from_snapshot(json.load(open(a.snapshot)))
    print(
        f"[campaign] {len(cands)} candidates, {len(enum_hubs)} enumerated hubs, "
        f"{len(cost_table.promoted_set)} promoted fragments (recipes={bool(cost_table.recipes)})"
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
    bc = BestCandidateStrategy(cands, cost_table, **common).run(budget=curve_budget)
    hb = HubBatchingStrategy(enum_hubs, cost_table, **common).run(budget=curve_budget)
    if bc.total_reactions < a.budget_reactions or hb.total_reactions < a.budget_reactions:
        print(
            f"[campaign] WARNING a curve stopped below the Case-1 reaction budget "
            f"({a.budget_reactions}); raise --budget-modes for a valid Case-1 readout."
        )

    out = HERE / "results" / a.tag  # results/<target>/ — untagged names (the dir carries the tag)
    out.mkdir(parents=True, exist_ok=True)
    _write_curve(out / "curve_best_candidate.csv", bc)
    _write_curve(out / "curve_hub_batching.csv", hb)
    summary = {
        "tag": a.tag,
        "reward_threshold": a.reward_threshold,
        "budget_reactions": a.budget_reactions,
        "budget_modes": a.budget_modes,
        "best_candidate": _readouts(bc, a.budget_reactions, a.budget_modes),
        "hub_batching": _readouts(hb, a.budget_reactions, a.budget_modes),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(out / "curve.png", [bc, hb], a.tag)
    print(json.dumps(summary, indent=2))
    print(f"\n[campaign] wrote summary.json + curve_*.csv to {out}")


if __name__ == "__main__":
    main()
