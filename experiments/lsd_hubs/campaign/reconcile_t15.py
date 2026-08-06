#!/usr/bin/env python
"""T1.5 — the unit-reconciliation GATE (docs/LSD_FLOW_BENCHMARK_PLAN.md; blocks Phase 2).

Before any cross-method plot is trusted, confirm SPARROW's reaction unit AGREES with our DAG
count-once on a *reaction-GFN* library. Both price the SAME native (by-construction) routes:

  * CHECK 2 — count-once (Logs/033): the campaign strategy's own DAG estimate (shallow assembly
    couplings + each promoted dynamic-library fragment built once).
  * CHECK 1 — native-route SPARROW: assemble each accepted mode's native route (best-candidate =
    the sampled molecule's route from routes.json; hub-batching = the hub's route + the final
    diversifying reaction from enum_children), **recipe-expand** every promoted fragment into its
    build steps (so SPARROW counts the same fragment-builds count-once charges), merge into one
    reaction network, and run SPARROW's MILP → distinct reactions (shared ones once).

If |sparrow − count_once| / count_once is within tolerance, the reaction unit is reconciled and the
frontier's cross-method numbers are comparable. If not, the unit definition is fixed HERE.

Uses a SCENT run that HAS routes.json + per-child enum `reaction` fields (the only complete one is
``recon_smoke_70526``); scale is irrelevant to a unit gate. Run in the rgfn env (needs sparrow via
the worker subprocess); pure post-hoc (no GPU).

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/reconcile_t15.py \
        --recon-dir /scratch/.../lsdflow/recon_smoke_70526 \
        --snapshot /scratch/.../fragments_4000.json --reward-threshold 7.0 --cutoff 0.5
"""
import argparse
import csv
import json
from pathlib import Path

from run_campaign import (
    RANK_METHODS,
    _load_candidates,
    _load_enumerated_hubs,
    build_strategy,
    rank_fragments,
)

from glue.samplers.lsdflow.child_select import make_child_policy
from validation.lsdflow.eval.base import LibrarySet
from validation.lsdflow.eval.network import expand_route_with_recipes
from validation.lsdflow.eval.sparrow import SparrowEvaluator
from validation.lsdflow.metrics.cost.dynamic_amortization import (
    load_cost_table_from_snapshot,
)

HERE = Path(__file__).resolve().parent


def _load_routes_json(path: Path) -> dict:
    """product SMILES (stripped key) -> shallow native route ({seed, num_reactions, steps})."""
    return json.load(open(path)) if Path(path).exists() else {}


def _hub_final_reaction_index(enum_children_path: Path):
    """child SMILES -> (hub_key, [final reaction step(s)]) from enum_children.json (the diversifying
    reaction the worker logged per child)."""
    data = json.load(open(enum_children_path))
    idx = {}
    for h in data.get("hubs", []):
        for c in h.get("children", []):
            idx[c["smiles"]] = (h["hub_key"], c.get("reaction") or [])
    return idx


def _native_route_for_mode(smiles, strategy, routes_json, hub_final_idx):
    """Assemble a mode's SHALLOW native route (pre-recipe-expansion), by STRATEGY.

    Strategy — not ``source_hub`` — decides the route source: a best-candidate ``CampaignPoint`` also
    carries a ``source_hub`` (its accidental-shared parent from ``MostSharedAssignment``), so branching
    on that would misroute best-candidate modes through the enum index. Instead:

    - **best_candidate**: the sampled molecule's own route from ``routes.json`` (keyed by the mode).
    - **hub_batching**: the hub's route (``routes.json``; empty if the hub is a bought base fragment)
      + the final diversifying reaction (``enum_children``'s per-child ``reaction``).

    Returns a route dict (``{product_smiles, num_reactions, steps}``) or ``None`` (no native route).
    """
    if strategy == "best_candidate":
        r = routes_json.get(smiles)
        return (
            {"product_smiles": smiles, "num_reactions": r["num_reactions"], "steps": r["steps"]}
            if r
            else None
        )
    # hub_batching: hub route + the child's final diversifying reaction
    hub_key, final = hub_final_idx.get(smiles, (None, []))
    hub_route = routes_json.get(hub_key) if hub_key else None
    steps = (list(hub_route["steps"]) if hub_route else []) + list(final)
    return (
        {"product_smiles": smiles, "num_reactions": len(steps), "steps": steps} if steps else None
    )


def _reconcile_one(
    strategy_name, result, routes_json, hub_final_idx, recipes, promoted, evaluator, tag
):
    """Price ONE strategy's accepted library both ways; return the reconciliation record."""
    modes = result.accepted
    smiles = [p.smiles for p in modes]
    # Native routes, recipe-expanded (fragment builds included, so SPARROW == count-once unit).
    routes = {}
    n_native = 0
    for p in modes:
        raw = _native_route_for_mode(p.smiles, strategy_name, routes_json, hub_final_idx)
        if raw is None:
            continue
        routes[p.smiles] = expand_route_with_recipes(raw, recipes, promoted)
        n_native += 1
    lib = LibrarySet(
        smiles=smiles,
        rewards={p.smiles: p.reward for p in modes},
        routes=routes,
        provenance={"strategy": strategy_name, "tag": tag},
        count_once_reactions=result.total_reactions,  # CHECK 2 (the campaign's DAG estimate)
    )
    ev_res = evaluator.score(lib)  # CHECK 1 (native-route SPARROW over expanded routes)
    count_once = result.total_reactions
    sparrow = ev_res.total_reactions
    rel = (abs(sparrow - count_once) / count_once) if (sparrow is not None and count_once) else None
    return {
        "strategy": strategy_name,
        "n_modes": len(modes),
        "n_native_routes": n_native,
        "native_route_coverage": round(n_native / len(modes), 4) if modes else None,
        "count_once_reactions": count_once,
        "sparrow_reactions": sparrow,
        "sparrow_priced_modes": ev_res.n_priced,
        "abs_diff": (abs(sparrow - count_once) if sparrow is not None else None),
        "rel_diff": (round(rel, 4) if rel is not None else None),
        "sparrow_milp_status": ev_res.provenance.get("milp_status"),
        "sparrow_n_reaction_nodes": ev_res.per_tool.get("n_reaction_nodes"),
        "sparrow_n_shared_compounds": ev_res.per_tool.get("n_shared_compounds"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recon-dir", required=True, help="SCENT run WITH routes.json (recon_smoke_*)")
    ap.add_argument(
        "--snapshot", required=True, help="fragments_<N>.json (smiles_to_route recipes)"
    )
    ap.add_argument("--reward-threshold", type=float, required=True)
    ap.add_argument("--cutoff", type=float, default=0.5, help="diversity cutoff for the library")
    ap.add_argument("--budget-modes", type=int, default=100)
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--tolerance", type=float, default=0.10, help="pass gate if rel_diff <= this")
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.95,
        help="a strategy counts toward the gate only if its native-route coverage >= this; count-once "
        "and SPARROW must price the SAME modes, so a partially-routed strategy (e.g. best-candidate on "
        "a sparse smoke run whose routes.json misses modes) is reported but not gated (its count_once "
        "and sparrow are over different sets, so the rel_diff is a coverage artifact, not a unit gap).",
    )
    ap.add_argument("--tag", default="recon_smoke")
    ap.add_argument("--sparrow-env", default="sparrow")
    # ACQUISITION POLICY — must match the configuration whose number the paper quotes.
    # These default to run_campaign's own defaults (naive `reward`, no pre-select), which is what
    # every audit before 2026-08-04 silently used: reconcile_t15 never passed them, so entry 049's
    # 3.1% gap was measured on the NAIVE library (100 modes / 288 count-once = 2.88 rxn/mode,
    # matching matrix16's scent_seh_naive 2.78) — NOT on the free-frag+K20 configuration the
    # headline reports (1.223-1.303 rxn/mode). The gap is library-dependent (on DRD2, hub-batching
    # scored 0.00% vs best-candidate's 4.95% on the same network), so it cannot be assumed to
    # transfer between policies. Pass --child-policy free_frag --prebuild-k 20 to audit the
    # configuration the headline actually uses.
    ap.add_argument(
        "--child-policy",
        default="reward",
        choices=["reward", "free_frag"],
        help="within-hub child selection (Logs/037): reward = naive; free_frag = fragment-aware. "
        "Default `reward` reproduces every pre-2026-08-04 audit.",
    )
    ap.add_argument(
        "--prebuild-k",
        type=int,
        default=0,
        help="pre-select-K: pre-synthesize the top-K fragments (by --rank-by) and charge them "
        "upfront. Use with --child-policy free_frag (the headline uses K=20).",
    )
    ap.add_argument(
        "--rank-by",
        default="build_score",
        choices=list(RANK_METHODS),
        help="pre-select ranking (must match run_campaign's, or the two price different libraries)",
    )
    ap.add_argument(
        "--min-recipe-coverage",
        type=float,
        default=0.95,
        help="fraction of the snapshot's promoted fragments that must carry a `smiles_to_route` "
        "recipe for this cell to be auditable at all (see the guard below). Set 0 to force.",
    )
    a = ap.parse_args()

    recon = Path(a.recon_dir)
    sample_dir = recon / "sample"
    enum_children = recon / "enum" / "enum_children.json"
    snapshot = json.load(open(a.snapshot))
    recipes = snapshot.get("smiles_to_route") or {}
    promoted = set(snapshot.get("chosen_smiles", []))
    cost_table = load_cost_table_from_snapshot(snapshot)

    # GUARD (entry 049) — this cell must be AUDITABLE before we report a unit gap.
    # count-once charges every promoted fragment's BUILD (campaign.py `_charge_promoted`), but SPARROW
    # can only charge it when the snapshot carries that fragment's recipe (`expand_route_with_recipes`
    # needs `smiles_to_route`). A snapshot without recipes makes the two sides price DIFFERENT
    # assumptions — SPARROW buys what count-once builds — and the run reports a huge bogus "unit gap"
    # that looks like a cost-model defect. Real case: scent_drd2_5k/seed42 has 0/1600 recipes and
    # produced rel_diff 62.7%, while sEH's recipe-logging re-run (job 70180) has 1600/1600 and
    # reconciles at 3.1%. Fail fast with the actual remedy instead of emitting the number.
    recipe_cov = (sum(1 for s in promoted if s in recipes) / len(promoted)) if promoted else 1.0
    if promoted and recipe_cov < a.min_recipe_coverage:
        raise SystemExit(
            f"[t15] ABORT: snapshot has recipes for only {recipe_cov:.1%} of its {len(promoted)} "
            f"promoted fragments ({len(recipes)} `smiles_to_route` entries) — below "
            f"--min-recipe-coverage {a.min_recipe_coverage:.0%}.\n"
            f"       {Path(a.snapshot)}\n"
            "       This cell CANNOT be audited: SPARROW would treat promoted fragments as bought "
            "while count-once builds them, so the resulting rel_diff is an assumptions mismatch, not "
            "a reaction-unit gap. Fix: re-run this generator with recipe logging enabled (the sEH "
            "precedent is the 2026-07-10 re-run, job 70180) and point --snapshot at the new "
            "fragments_<N>.json. Override with --min-recipe-coverage 0 only to reproduce entry 049."
        )
    print(
        f"[t15] recipe coverage {recipe_cov:.1%} of {len(promoted)} promoted fragments — auditable"
    )

    cands, comps = _load_candidates(sample_dir, a.higher_is_better)
    enum_hubs = _load_enumerated_hubs(enum_children, comps)
    routes_json = _load_routes_json(sample_dir / "routes.json")
    hub_final_idx = _hub_final_reaction_index(enum_children)
    print(
        f"[t15] {len(cands)} candidates, {len(enum_hubs)} hubs, {len(routes_json)} native routes, "
        f"{len(promoted)} promoted frags, cutoff {a.cutoff}"
    )

    # Acquisition policy, built exactly as run_campaign does so the audited library is bit-identical
    # to the campaign's. hub_batching consumes both; best_candidate ignores them (it has no hubs).
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
        print(
            f"[t15] pre-select-K: {len(prebuilt)} fragments pre-synthesized (top {a.rank_by}), "
            f"{upfront} reactions charged upfront"
        )
    print(f"[t15] acquisition: child_policy={a.child_policy} prebuild_k={a.prebuild_k}")

    common = dict(
        target=a.tag, reward_threshold=a.reward_threshold, higher_is_better=a.higher_is_better
    )
    work = Path("/tmp") / f"t15_sparrow_{a.tag}"
    evaluator = SparrowEvaluator(
        route_source="native", cache=None, sparrow_env=a.sparrow_env, work_dir=work
    )

    out = HERE / "results" / f"{a.tag}_reconcile"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name, pool in (("best_candidate", cands), ("hub_batching", enum_hubs)):
        result = build_strategy(
            name,
            pool,
            cost_table,
            comps,
            similarity=a.cutoff,
            child_policy=child_policy,
            prebuilt_fragments=prebuilt,
            **common,
        ).run(budget=("modes", a.budget_modes))
        rec = _reconcile_one(
            name, result, routes_json, hub_final_idx, recipes, promoted, evaluator, a.tag
        )
        records.append(rec)
        print(
            f"[t15] {name}: modes={rec['n_modes']} native_cov={rec['native_route_coverage']} | "
            f"count_once={rec['count_once_reactions']} sparrow={rec['sparrow_reactions']} "
            f"rel_diff={rec['rel_diff']} (status {rec['sparrow_milp_status']})"
        )

    # Gate verdict: reconciled iff every FULLY-ROUTED strategy is within tolerance (same-set only).
    # A strategy with coverage < min-coverage prices count-once and SPARROW over DIFFERENT mode sets,
    # so its rel_diff is a coverage artifact — report it, don't gate on it.
    gated = [
        r
        for r in records
        if r["rel_diff"] is not None and (r["native_route_coverage"] or 0) >= a.min_coverage
    ]
    ungated = [r for r in records if r not in gated]
    worst = max((r["rel_diff"] for r in gated), default=None)
    passed = bool(gated) and all(r["rel_diff"] <= a.tolerance for r in gated)
    for r in ungated:
        print(
            f"[t15] NOTE {r['strategy']} excluded from gate: native-route coverage "
            f"{r['native_route_coverage']} < {a.min_coverage} (count_once/sparrow are different sets "
            f"→ rel_diff {r['rel_diff']} is a coverage artifact, not a unit gap)"
        )
    summary = {
        "tag": a.tag,
        "recon_dir": str(recon),
        "cutoff": a.cutoff,
        "reward_threshold": a.reward_threshold,
        "tolerance": a.tolerance,
        "min_coverage": a.min_coverage,
        "records": records,
        "gated_strategies": [r["strategy"] for r in gated],
        "ungated_strategies": [r["strategy"] for r in ungated],
        "worst_rel_diff": worst,
        "gate_passed": passed,
    }
    (out / "reconcile_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "reconcile.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    verdict = "PASSED ✔" if passed else "FAILED ✗ — fix the reaction-unit definition before Phase 2"
    print(f"\n[t15] GATE {verdict}  (worst rel_diff={worst}, tolerance {a.tolerance}) -> {out}")


if __name__ == "__main__":
    main()
