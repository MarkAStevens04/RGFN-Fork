#!/usr/bin/env python
"""SPARROW per-env worker — batch route-selection MILP (LSD-Flow benchmark T1.4).

SPARROW (Fromer & Coley) lives in its own ``sparrow`` conda env (python 3.12 + its own stack;
``external/setup_sparrow.sh``) and cannot co-import with our ``rgfn``/``glue``. This worker runs
standalone *inside the sparrow env*; the validation-side :class:`~validation.lsdflow.eval.sparrow.
SparrowEvaluator` shells to it (the ``scripts/score_batch.py`` cross-env pattern) and exchanges files.

Input: a merged reaction network (``--tree`` = ``tree.json`` from
``validation/lsdflow/eval/network.py``, ``Compound Nodes`` + ``Reaction Nodes``) and the library's
targets+rewards (``--targets`` = ``SMILES,Reward`` CSV). Output (``--out`` JSON): the MILP's chosen
route set, headlined by ``total_reactions`` = the number of distinct reactions to synthesize the
whole library, **shared intermediates counted once** (each reaction is one binary variable). That is
the from-scratch reactions-per-mode price every method is compared on.

Solver = **PuLP/CBC** (open-source; no Gurobi license). Objective (``--objective``):
  * ``count`` (default): minimize the number of reactions (``weights=[0,0,1,0,0]``,
    ``constrain_all_targets`` so all library modes are synthesized). The clean reactions-per-mode
    unit, matching DAG count-once.
  * ``count_cost``: also weight buying starting materials (SPARROW's native $ term).
  * ``reward``: maximize total reward of the SELECTED subset (``weights=[1,0,0,0,0]``) — see
    "TWO MODES" below.
  * ``feasibility`` (TODO, memory sparrow-milp-objective-revisit-feasibility): weight reactions by
    AiZynth confidence (penalty=1/score); needs per-reaction scores in the tree — not emitted yet.

TWO MODES — PRICING vs SELECTION. These answer different questions and must not be conflated.

* **PRICING** (default; ``--objective count``, no ``--select``). We hand SPARROW a library that has
  ALREADY been chosen and ask "what is the cheapest way to make all of it?" ``constrain_all_targets``
  forces every target into the solution. This is the T1.5/T4.4 cost-model audit: SPARROW is used as
  an independent cost model, not as a decision maker.

* **SELECTION** (``--select``, ``--objective reward``, plus a budget). SPARROW does the job it was
  designed for: choose WHICH targets to make. This is the competitor pipeline a chemist would
  actually run — generate candidates, retro-plan them all, then let SPARROW pick the subset worth
  synthesizing. ``constrain_all_targets`` is off, so the MILP selects.

  Sign convention (verified against ``LinearSelector.set_objective``): the problem is a
  MINIMIZATION of ``-w[0]*Σ(reward × selected) + w[1]*Σ(SM cost) + w[2]*Σ(reaction penalty)``.
  So ``weights=[1,0,0,0,0]`` with a hard ``--max-rxns R`` budget maximizes total selected reward
  subject to "at most R reactions". Using a hard constraint rather than a reaction PENALTY is
  deliberate: a penalty would require choosing an exchange rate between reward units and reactions,
  an arbitrary knob that changes the answer; ``max_rxns`` needs no such calibration and yields
  exactly the "best library makeable in R reactions" the frontier plots against.

  NOTE the expected behaviour: reward is positive for every target and nothing penalizes selecting
  more, so SPARROW packs in as many targets as the budget allows and favours cheap, shared-route
  (i.e. structurally similar) ones. It has NO diversity term unless ``clusters``/``N_per_cluster``
  are supplied. That is a real property of the baseline, not a defect to hide — the caller measures
  how many of the selected targets are distinct modes and reports it.

Intermediates/targets are non-buyable (``coster=None`` keeps the network's inline buyability, and
``dont_buy_targets`` forbids buying a target), so every mode must be *synthesized*, not purchased.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# --- objective -> SPARROW LinearSelector weights [reward, start_cost, reaction, diversity, class] ---
_OBJECTIVES = {
    "count": [0, 0, 1, 0, 0],  # minimize #reactions (each penalty 1); reward/cost off
    "count_cost": [0, 1, 1, 0, 0],  # + starting-material $ (SPARROW native cost term)
    "reward": [
        1,
        0,
        0,
        0,
        0,
    ],  # SELECTION: maximize selected reward s.t. --max-rxns (see docstring)
    # "feasibility": weight by AiZynth confidence (penalty=1/score) -- TODO (see module docstring)
}
# Objectives whose reward term is zero can only be used to PRICE a fixed target set: with no reward
# bonus the MILP has no reason to select anything, so in --select mode it would return the empty
# library (0 reactions, 0 targets) as trivially optimal. Guarded in main().
_SELECTION_OBJECTIVES = {name for name, w in _OBJECTIVES.items() if w[0] > 0}


def _read_targets(path):
    import csv

    target_dict = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            target_dict[row["SMILES"]] = float(row["Reward"])
    return target_dict


def main():
    ap = argparse.ArgumentParser(description="SPARROW batch route-selection MILP worker.")
    ap.add_argument("--tree", required=True, help="merged reaction network (network.py tree.json)")
    ap.add_argument("--targets", required=True, help="targets CSV (SMILES,Reward)")
    ap.add_argument("--out", required=True, help="output JSON (MILP result)")
    ap.add_argument("--objective", default="count", choices=sorted(_OBJECTIVES))
    ap.add_argument("--work-dir", default=None, help="SPARROW output_dir (default: beside --out)")
    ap.add_argument(
        "--max-seconds", type=int, default=600, help="MILP solve time limit (per snapshot)"
    )
    ap.add_argument(
        "--select",
        action="store_true",
        help="SELECTION mode: let the MILP choose WHICH targets to make (constrain_all_targets off). "
        "Requires a reward-bearing --objective and at least one budget (--max-rxns/--max-targets). "
        "Default (off) is PRICING mode: every target must be synthesized.",
    )
    ap.add_argument(
        "--max-rxns",
        type=int,
        default=None,
        help="hard cap on non-dummy reactions (a real MILP constraint in LinearSelector). The "
        "reaction-budget axis the frontier is plotted against.",
    )
    ap.add_argument(
        "--max-targets", type=int, default=None, help="hard cap on number of selected targets"
    )
    ap.add_argument(
        "--clusters",
        default="",
        help="JSON {cluster_name: [SMILES, ...]} partitioning the pool. SPARROW treats clusters as "
        "an arbitrary caller-supplied partition ([fromer2025diversity]), so we pass OUR tau-modes "
        "and the diversity it optimizes is the same quantity the benchmark reports.",
    )
    ap.add_argument(
        "--lambda-div",
        type=float,
        default=0.0,
        help="SPARROW'S OWN diversity mechanism ([fromer2025diversity] sec 2.2, objective form): "
        "weight on the number of clusters represented, added to the objective so diversity is "
        "TRADED OFF against reward rather than required. This is the native path and the one we "
        "use; sweep it to trace their Pareto front. Needs --clusters.",
    )
    ap.add_argument(
        "--gap-rel",
        type=float,
        default=None,
        help="relative MIP gap for CBC. SPARROW hardcodes gapRel=1e-7 -- proving optimality to one "
        "part in ten million, far tighter than any effect this benchmark can resolve, and the reason "
        "diversity-weighted solves run for hours without converging. Setting e.g. 1e-3 accepts a "
        "solution provably within 0.1%% of optimal, which is well inside our noise. Applied by "
        "overriding the solver at call time; external/sparrow is left untouched.",
    )
    ap.add_argument(
        "--min-clusters",
        type=int,
        default=None,
        help="require the selection to represent at least this many clusters — the CONSTRAINT form "
        "of [fromer2025diversity] sec 2.2. Needs --clusters.",
    )
    a = ap.parse_args()

    if a.select:
        if a.objective not in _SELECTION_OBJECTIVES:
            ap.error(
                f"--select needs a reward-bearing objective (one of "
                f"{sorted(_SELECTION_OBJECTIVES)}); '{a.objective}' has reward weight 0, so the "
                "empty library would be optimal and the run would report 0 reactions / 0 targets."
            )
        if a.max_rxns is None and a.max_targets is None:
            ap.error(
                "--select needs a budget (--max-rxns and/or --max-targets): with a positive reward "
                "per target and no penalty, selecting EVERY target is optimal and the result is "
                "degenerate."
            )
    elif a.max_rxns is not None or a.max_targets is not None:
        ap.error("--max-rxns/--max-targets only apply to --select (pricing must make every target)")
    if a.min_clusters is not None and not a.clusters:
        ap.error("--min-clusters needs --clusters (the partition it counts representation over)")
    if a.lambda_div > 0 and not a.clusters:
        ap.error("--lambda-div needs --clusters (the partition whose representation it rewards)")
    if a.lambda_div > 0 and a.min_clusters is not None:
        ap.error(
            "--lambda-div and --min-clusters are two DIFFERENT mechanisms for the same goal "
            "(SPARROW's soft objective vs a hard rule we add on top). Combining them makes the "
            "result attributable to neither. Pick one."
        )

    from sparrow.route_graph import RouteGraph
    from sparrow.selector.linear import LinearSelector

    t0 = time.perf_counter()
    target_dict = _read_targets(a.targets)
    graph = RouteGraph(node_filename=a.tree)
    build_s = time.perf_counter() - t0

    work_dir = Path(a.work_dir) if a.work_dir else Path(a.out).parent / "sparrow_run"
    work_dir.mkdir(parents=True, exist_ok=True)

    weights = list(_OBJECTIVES[a.objective])
    weights[3] = a.lambda_div  # slot 3 is SPARROW's diversity term; >0 makes it build the d_i vars

    sel = LinearSelector(
        route_graph=graph,
        target_dict=target_dict,
        weights=weights,
        # PRICING forces every target in; SELECTION lets the MILP choose (see module docstring).
        constrain_all_targets=not a.select,
        dont_buy_targets=True,  # ... and cannot be trivially "bought"
        coster=None,  # keep the network's inline buyable flags (intermediates NOT buyable)
        solver="pulp",  # CBC, no Gurobi license
        output_dir=str(work_dir),
        max_seconds=a.max_seconds,
        max_rxns=a.max_rxns,  # None in pricing mode -> constraint not added
        max_targets=a.max_targets,
        # clusters are CLEANED by Selector.clean_clusters (SMILES -> graph ids, canonicalizing on
        # the way), so we hand it SMILES and let it map. N_per_cluster stays 0 on purpose: that
        # switch is SPARROW's OTHER cluster mechanism ("take >= N from EVERY cluster", Briem-style),
        # which with tau-modes clamps to 1 and would force one pick per mode -- not the deliverable
        # we measure. The constraint we want is "represent at least K clusters", added below.
        clusters=json.loads(Path(a.clusters).read_text()) if a.clusters else None,
    )

    result = {
        "objective": a.objective,
        "weights": weights,
        "lambda_div": a.lambda_div,
        "gap_rel": a.gap_rel,
        "mode": "selection" if a.select else "pricing",
        "max_rxns": a.max_rxns,
        "max_targets": a.max_targets,
        "n_targets_requested": len(target_dict),
        "n_compound_nodes": len(graph.compound_nodes),
        "n_reaction_nodes": len(graph.reaction_nodes),
        "build_s": round(build_s, 3),
    }

    def _add_min_cluster_constraint(selector, k):
        """Require the selection to represent >= k clusters ([fromer2025diversity] sec 2.2, the
        CONSTRAINT variant). Built here rather than upstream because SPARROW ships the two adjacent
        mechanisms but not this one: `add_diversity_objective` creates the same "cluster represented"
        binaries but spends them as a weighted OBJECTIVE term (soft, traces a Pareto front), and
        `set_cluster_constraints` enforces ">= N from EVERY cluster" (a different, much stronger
        requirement). We need the hard "deliver K distinct families" form so the arm is read at the
        same fixed deliverable as every other arm in the benchmark.

        d[i] = 1 iff cluster i has any selected member. `d[i] <= sum(members)` alone would let the
        solver set every d to 0 for free, so it is the `sum(d) >= k` constraint that does the work:
        together they force at least k clusters to contain a selection. Upper-linking only is
        correct here precisely BECAUSE d is pushed up by the constraint rather than by an objective.
        """
        from pulp import LpVariable, lpSum

        groups = list(selector.clusters.values())
        if k > len(groups):
            raise SystemExit(
                f"[sparrow_worker] --min-clusters {k} exceeds the {len(groups)} clusters present; "
                "the problem would be infeasible. Lower it or widen the pool."
            )
        d = LpVariable.dicts("mincluster", indices=range(len(groups)), cat="Binary")
        for i, ids in enumerate(groups):
            selector.problem += d[i] <= lpSum(selector.m[cid] for cid in ids)
        selector.problem += lpSum(d.values()) >= k
        return len(groups)

    try:
        s0 = time.perf_counter()
        sel.define_variables()
        sel.set_objective()
        sel.set_constraints()
        if a.min_clusters is not None:
            n_groups = _add_min_cluster_constraint(sel, a.min_clusters)
            result["n_clusters"] = n_groups
            result["min_clusters"] = a.min_clusters
            print(
                f"[sparrow_worker] diversity CONSTRAINT: represent >= {a.min_clusters} "
                f"of {n_groups} clusters",
                flush=True,
            )
        if a.gap_rel is not None:
            # Override CBC's gap WITHOUT editing the upstream clone: re-bind problem.solve to a
            # solver carrying our gap. Upstream's optimize() calls self.problem.solve(...) with its
            # own hardcoded PULP_CBC_CMD, so intercepting at the problem object is the least
            # invasive seam that still takes effect.
            from pulp import PULP_CBC_CMD

            _solver = PULP_CBC_CMD(
                gapRel=a.gap_rel, gapAbs=1e-9, msg=False, timeLimit=a.max_seconds
            )
            _orig_solve = sel.problem.solve
            sel.problem.solve = lambda *_args, **_kw: _orig_solve(_solver)
            print(
                f"[sparrow_worker] CBC gapRel overridden to {a.gap_rel} (upstream default 1e-7)",
                flush=True,
            )
        sel.optimize()  # raises RuntimeError if infeasible
        solve_s = time.perf_counter() - s0
        import pulp

        # TIME-LIMIT DETECTION. CBC reports status "Optimal" for the best solution it happened to
        # hold when a time limit stops it, and PuLP's sol_status does not distinguish either
        # (verified on pulp 3.3.2: a 2 s-limited solve still reports "Optimal Solution Found").
        # Taking that at face value publishes a truncated search as a proven optimum -- exactly the
        # failure that made a budget-1200 solve return LOWER reward (4893) than a budget-800 solve
        # (5760) while both claimed Optimal, which is impossible for true optima since the tighter
        # problem's solution is feasible in the looser one.
        #
        # We therefore judge by wall-clock against the cap. A solve that ran to ~the limit is
        # reported as TimeLimit and its objective treated as a LOWER BOUND, never as optimal. The
        # threshold errs toward caution: mislabelling a genuine optimum as unproven costs us a
        # footnote, the reverse costs us a wrong claim.
        time_capped = a.max_seconds is not None and solve_s >= 0.95 * a.max_seconds
        raw_status = pulp.LpStatus[sel.problem.status]

        mol_ids, rxn_ids, _ = sel.extract_selected_ids()
        non_dummy = [r for r in rxn_ids if sel.graph.node_from_id(r).dummy == 0]
        dummy = [r for r in rxn_ids if sel.graph.node_from_id(r).dummy != 0]
        sel_targets = set(mol_ids) & set(sel.targets)
        # Emit the SMILES, not just the count: in selection mode the caller must know WHICH targets
        # the MILP chose in order to count how many are distinct modes (SPARROW optimizes reward and
        # cost, never diversity, so the mode count is a property of the output that only the caller
        # can measure). Ids are graph-internal, hence the resolve.
        selected_smiles = []
        for tid in sel_targets:
            try:
                selected_smiles.append(sel.graph.smiles_from_id(tid))
            except Exception:  # id not resolvable -> keep the raw id so nothing is silently dropped
                selected_smiles.append(str(tid))
        result.update(
            {
                "milp_status": "TimeLimit" if time_capped else raw_status,
                "milp_status_raw": raw_status,
                "time_capped": time_capped,
                "max_seconds": a.max_seconds,
                "total_reactions": len(non_dummy),  # HEADLINE: distinct reactions, shared once
                "n_targets_selected": len(sel_targets),
                "selected_target_smiles": sorted(selected_smiles),
                "selected_reward_total": round(
                    sum(target_dict.get(s, 0.0) for s in selected_smiles), 4
                ),
                "n_starting_materials": len(dummy),
                "n_variables": sel.get_num_variables(),
                "n_constraints": sel.get_num_constraints(),
                "solve_s": round(solve_s, 3),
            }
        )
    except Exception as exc:  # infeasible or solver failure -> report, don't crash the frontier
        result.update(
            {
                "milp_status": "Error",
                "total_reactions": None,
                "n_targets_selected": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    if result.get("time_capped"):
        print(
            f"[sparrow_worker] WARNING: solve hit the {a.max_seconds}s limit "
            f"({result.get('solve_s')}s). CBC still says '{result.get('milp_status_raw')}', but the "
            "search was truncated -- treat total_reactions as a LOWER BOUND on what SPARROW could "
            "achieve, not as a proven optimum.",
            flush=True,
        )
    print(
        f"[sparrow_worker] objective={a.objective} status={result.get('milp_status')} "
        f"total_reactions={result.get('total_reactions')} "
        f"targets={result.get('n_targets_selected')}/{result['n_targets_requested']} "
        f"-> {a.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
