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
  * ``feasibility`` (TODO, memory sparrow-milp-objective-revisit-feasibility): weight reactions by
    AiZynth confidence (penalty=1/score); needs per-reaction scores in the tree — not emitted yet.

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
    # "feasibility": weight by AiZynth confidence (penalty=1/score) -- TODO (see module docstring)
}


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
    a = ap.parse_args()

    from sparrow.route_graph import RouteGraph
    from sparrow.selector.linear import LinearSelector

    t0 = time.perf_counter()
    target_dict = _read_targets(a.targets)
    graph = RouteGraph(node_filename=a.tree)
    build_s = time.perf_counter() - t0

    work_dir = Path(a.work_dir) if a.work_dir else Path(a.out).parent / "sparrow_run"
    work_dir.mkdir(parents=True, exist_ok=True)

    sel = LinearSelector(
        route_graph=graph,
        target_dict=target_dict,
        weights=_OBJECTIVES[a.objective],
        constrain_all_targets=True,  # every selected mode must be synthesized
        dont_buy_targets=True,  # ... and cannot be trivially "bought"
        coster=None,  # keep the network's inline buyable flags (intermediates NOT buyable)
        solver="pulp",  # CBC, no Gurobi license
        output_dir=str(work_dir),
        max_seconds=a.max_seconds,
    )

    result = {
        "objective": a.objective,
        "weights": _OBJECTIVES[a.objective],
        "n_targets_requested": len(target_dict),
        "n_compound_nodes": len(graph.compound_nodes),
        "n_reaction_nodes": len(graph.reaction_nodes),
        "build_s": round(build_s, 3),
    }
    try:
        s0 = time.perf_counter()
        sel.define_variables()
        sel.set_objective()
        sel.set_constraints()
        sel.optimize()  # raises RuntimeError if infeasible
        solve_s = time.perf_counter() - s0
        import pulp

        mol_ids, rxn_ids, _ = sel.extract_selected_ids()
        non_dummy = [r for r in rxn_ids if sel.graph.node_from_id(r).dummy == 0]
        dummy = [r for r in rxn_ids if sel.graph.node_from_id(r).dummy != 0]
        sel_targets = set(mol_ids) & set(sel.targets)
        result.update(
            {
                "milp_status": pulp.LpStatus[sel.problem.status],
                "total_reactions": len(non_dummy),  # HEADLINE: distinct reactions, shared once
                "n_targets_selected": len(sel_targets),
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
