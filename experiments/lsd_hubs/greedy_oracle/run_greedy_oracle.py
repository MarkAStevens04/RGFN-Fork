#!/usr/bin/env python
"""How far below an adaptive optimiser does the flow ordering sit?

Runs three arms over ONE already-enumerated cell, changing only *which hub is walked next*:

  best_candidate   the standing baseline (no hubs at all)
  hub_batching     the shipped strategy — a STATIC ranking by recovered flow F_hat(h)
  greedy_oracle    cost-benefit greedy — at every step re-score every remaining hub by its true
                   marginal (new modes)/(marginal reactions) and take the argmax

Everything else is held fixed: same enumeration, same cost table, same child policy, same pre-select
stock, same mode definition, same budget. The loaders and `build_strategy` are imported from
`run_campaign.py` rather than re-implemented, so an arm here is bit-comparable to a standalone
`run_campaign.py` invocation on the same cell.

`--regress` additionally runs the greedy module in `order="static"` mode, which walks the flow
ranking using the greedy module's own probe/commit machinery. That must reproduce `hub_batching`
exactly; it is the test that validates the probe, the selector cloning and the pruning fast path.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/greedy_oracle/run_greedy_oracle.py \
        --analysis-dir /scratch/.../lsdflow/matrix16/scent_seh/sample \
        --enum-children /scratch/.../scent_seh/enum/enum_children.json \
        --snapshot /scratch/.../additional_fragments/fragments_4000.json \
        --reward-threshold 7.0 --child-policy free_frag --prebuild-k 20 --tag scent_seh
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "experiments" / "lsd_hubs" / "campaign"))

import run_campaign as rc  # noqa: E402  (the campaign driver, reused for its loaders)

from glue.samplers.lsdflow.campaign import RANK_METHODS, rank_fragments  # noqa: E402
from glue.samplers.lsdflow.child_select import make_child_policy  # noqa: E402
from glue.samplers.lsdflow.greedy_oracle import AdaptiveGreedyHubStrategy  # noqa: E402
from validation.lsdflow.metrics.cost.dynamic_amortization import (  # noqa: E402
    load_cost_table_from_snapshot,
)

ARM_LABELS = {
    "best_candidate": "best-candidate",
    "hub_batching": "hub-batching (flow, static)",
    "greedy_oracle": "cost-benefit greedy (oracle)",
    "greedy_oracle_static": "greedy module, static order (regression)",
}


def _curve_rows(tag, arm, result):
    for p in result.accepted:
        yield {
            "tag": tag,
            "arm": arm,
            "step": p.step,
            "cum_modes": p.cum_modes,
            "cum_reactions": p.cum_reactions,
            "reactions_added": p.reactions_added,
            "reward": p.reward,
            "source_hub": p.source_hub or "",
            "smiles": p.smiles,
        }


def _fingerprint(result):
    """A compact, order-sensitive signature of a run, for the regression comparison."""
    return {
        "total_modes": result.total_modes,
        "total_reactions": result.total_reactions,
        "distinct_hubs_used": result.distinct_hubs_used,
        "distinct_promoted_fragments": result.distinct_promoted_fragments,
        "total_reward_gen_calls": result.total_reward_gen_calls,
        "stop_reason": result.stop_reason,
        "sequence": [(p.smiles, p.cum_reactions) for p in result.accepted],
        "walked_hub_ids": list(result.walked_hub_ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--enum-children", required=True)
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--reward-threshold", type=float, required=True)
    ap.add_argument("--similarity", type=float, default=0.5)
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--budget-reactions", type=int, default=100)
    ap.add_argument("--budget-modes", type=int, default=300)
    ap.add_argument("--child-policy", default="reward", choices=["reward", "free_frag"])
    ap.add_argument("--prebuild-k", type=int, default=0)
    ap.add_argument("--rank-by", default="build_score", choices=list(RANK_METHODS))
    ap.add_argument(
        "--no-revisit",
        action="store_true",
        help="forbid the greedy from re-entering a hub it already built (stricter, weaker oracle)",
    )
    ap.add_argument(
        "--regress",
        action="store_true",
        help="also run the greedy module over the flow order and assert it reproduces hub_batching",
    )
    ap.add_argument("--max-hubs", type=int, default=0, help="truncate the hub pool (smoke tests)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()

    adir = Path(a.analysis_dir)
    cands, comps = rc._load_candidates(adir, a.higher_is_better)
    enum_hubs = rc._load_enumerated_hubs(Path(a.enum_children), comps)
    if a.max_hubs:
        enum_hubs = enum_hubs[: a.max_hubs]
    snapshot = json.load(open(a.snapshot)) if a.snapshot else {}
    cost_table = load_cost_table_from_snapshot(snapshot)
    print(
        f"[greedy] {a.tag}: {len(cands)} candidates, {len(enum_hubs)} hubs, "
        f"{sum(len(h.children) for h in enum_hubs)} children, "
        f"{len(cost_table.promoted_set)} promoted fragments (recipes={bool(cost_table.recipes)})",
        flush=True,
    )

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
        upfront = cost_table.shared_build_cost(prebuilt)[0] if cost_table else 0
        print(f"[greedy] pre-select-K: {len(prebuilt)} fragments, {upfront} reactions upfront")

    common = dict(
        target=a.tag,
        reward_threshold=a.reward_threshold,
        similarity=a.similarity,
        higher_is_better=a.higher_is_better,
    )
    budget = ("modes", a.budget_modes)

    results, timings = {}, {}

    for name, pool in (("best_candidate", cands), ("hub_batching", enum_hubs)):
        strat = rc.build_strategy(
            name,
            pool,
            cost_table,
            comps,
            child_policy=child_policy if name == "hub_batching" else None,
            prebuilt_fragments=prebuilt if name == "hub_batching" else None,
            **common,
        )
        res, secs = rc.run_timed(strat, budget)
        results[name], timings[name] = res, secs
        print(
            f"[greedy] {name:16s} {res.total_modes:4d} modes / {res.total_reactions:5d} rxn  ({secs:.1f}s)",
            flush=True,
        )

    greedy_arms = [("greedy_oracle", "greedy")]
    if a.regress:
        greedy_arms.append(("greedy_oracle_static", "static"))
    for name, order in greedy_arms:
        strat = AdaptiveGreedyHubStrategy(
            enum_hubs,
            cost_table,
            child_policy=child_policy,
            prebuilt_fragments=prebuilt,
            order=order,
            allow_revisit=not a.no_revisit,
            **common,
        )
        res, secs = rc.run_timed(strat, budget)
        results[name], timings[name] = res, secs
        print(
            f"[greedy] {name:16s} {res.total_modes:4d} modes / {res.total_reactions:5d} rxn  "
            f"({secs:.1f}s, {res.meta.get('n_probes')} probes)",
            flush=True,
        )

    # ---------------------------------------------------------------- regression
    regression = None
    if a.regress:
        want = _fingerprint(results["hub_batching"])
        got = _fingerprint(results["greedy_oracle_static"])
        # hub_batching charges every hub it iterates; the static greedy path does the same, so the
        # walked-hub lists must match too. reward_gen_calls therefore also match.
        diffs = {k: (want[k], got[k]) for k in want if want[k] != got[k] and k != "sequence"}
        seq_ok = want["sequence"] == got["sequence"]
        regression = {
            "identical": not diffs and seq_ok,
            "field_diffs": diffs,
            "sequence_identical": seq_ok,
        }
        if regression["identical"]:
            print("[greedy] REGRESSION PASS — static greedy reproduces hub_batching bit-for-bit")
        else:
            print(
                f"[greedy] REGRESSION FAIL — {diffs}, sequence_identical={seq_ok}", file=sys.stderr
            )

    # ---------------------------------------------------------------- readouts
    out = Path(a.out_dir) if a.out_dir else HERE / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)

    readouts = {}
    for name, res in results.items():
        r = rc._readouts(res, a.budget_reactions, a.budget_modes)
        r["selection_wall_s"] = round(timings[name], 3)
        r.update({k: v for k, v in res.meta.items() if k.startswith(("n_", "walk_", "pool_"))})
        readouts[name] = r

    hb, gr = readouts["hub_batching"], readouts["greedy_oracle"]
    key_rxn = f"case2_reactions_at_{a.budget_modes}modes"
    key_mod = f"case1_modes_at_{a.budget_reactions}rxn"
    gap = {
        # >1.0 means the greedy is cheaper, i.e. the flow ordering leaves that much on the table.
        "reactions_ratio_flow_over_greedy": (
            round(hb["reactions_per_mode"] / gr["reactions_per_mode"], 4)
            if hb.get("reactions_per_mode") and gr.get("reactions_per_mode")
            else None
        ),
        "flow_excess_pct": (
            round(100 * (hb["reactions_per_mode"] / gr["reactions_per_mode"] - 1), 2)
            if hb.get("reactions_per_mode") and gr.get("reactions_per_mode")
            else None
        ),
        "modes_at_rxn_budget": {"flow": hb.get(key_mod), "greedy": gr.get(key_mod)},
        "reactions_at_mode_budget": {"flow": hb.get(key_rxn), "greedy": gr.get(key_rxn)},
        "oracle_calls_flow_walk": hb.get("total_reward_gen_calls"),
        "oracle_calls_greedy_pool": gr.get("pool_reward_gen_calls"),
        "oracle_calls_greedy_walk": gr.get("walk_reward_gen_calls"),
        "oracle_call_overhead_x": (
            round(gr["pool_reward_gen_calls"] / hb["total_reward_gen_calls"], 3)
            if hb.get("total_reward_gen_calls")
            else None
        ),
        "selection_wall_s": {"flow": hb["selection_wall_s"], "greedy": gr["selection_wall_s"]},
    }

    summary = {
        "tag": a.tag,
        "reward_threshold": a.reward_threshold,
        "similarity": a.similarity,
        "higher_is_better": a.higher_is_better,
        "budget_reactions": a.budget_reactions,
        "budget_modes": a.budget_modes,
        "child_policy": a.child_policy,
        "prebuild_k": a.prebuild_k,
        "rank_by": a.rank_by,
        "n_hubs": len(enum_hubs),
        "allow_revisit": not a.no_revisit,
        "arms": readouts,
        "flow_vs_greedy": gap,
        "regression": regression,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    with open(out / "curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "tag",
                "arm",
                "step",
                "cum_modes",
                "cum_reactions",
                "reactions_added",
                "reward",
                "source_hub",
                "smiles",
            ],
        )
        w.writeheader()
        for name, res in results.items():
            w.writerows(_curve_rows(a.tag, name, res))

    print(json.dumps(gap, indent=2))
    print(f"[greedy] -> {out}")


if __name__ == "__main__":
    main()
