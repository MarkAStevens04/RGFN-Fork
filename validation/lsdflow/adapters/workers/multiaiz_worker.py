#!/usr/bin/env python
"""MultiAiZ route-discovery worker for the LSD-Flow benchmark (T4.1) — runs in the ``aizynth`` env.

Given ONE acquisition function's accepted-mode pool (SMILES), run MultiAiZ (`[ianez2026multiaiz]`,
``MolecularAI/multiaiz``) — AiZynthFinder over the whole SET for N cycles, appending discovered
intermediates to the stock each cycle so targets converge on shared intermediates — then emit, per
target, its candidate synthesis routes **flattened and stitched to real ZINC stock**, in the flat
``{product, reactants}`` step schema ``validation/lsdflow/eval/network.py::build_network`` consumes.

WHY STITCH TO ZINC (the crux — see Logs/043): MultiAiZ appends discovered intermediates to the stock,
so a later-cycle route can terminate at a discovered intermediate X as an ``in_stock`` LEAF. Fed to
SPARROW as-is, X is a FREE buyable and the convergent saving reads as zero. We instead expand every
such leaf back into its build subtree (harvested from the cycle where X was built to ZINC), so X
becomes a reaction PRODUCT; ``build_network`` then marks X non-buyable and the SPARROW MILP builds it
ONCE and amortizes it across the targets that share it. Real ZINC blocks (never a product) stay buyable.

We emit ALL candidate routes per target and let SPARROW's MILP pick the max-sharing combination — no
per-target selection heuristic (that IS what SPARROW is for; ``build_network`` tolerates duplicate
targets, unioning their reactions).

PER-POOL ISOLATION: invoked on ONE pool at a time, so the shared intermediates it finds exist only
within that pool — hub-batching's convergent routes never leak to best-candidate / S3-GFN.

Reached by subprocess from ``validation/lsdflow/eval/multiaiz.py``; never imported into ``rgfn``.
"""

import argparse
import glob
import gzip
import json
import sys
from pathlib import Path


def _canon(smi, strip_stereo=True):
    from rdkit import Chem

    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(m, isomericSmiles=not strip_stereo) if m is not None else None


def _harvest_build_subtrees(trees_by_target, strip_stereo):
    """``canon(smiles) -> a nested AiZynth mol-node subtree that BUILDS it`` (has a reaction child),
    over all routes/cycles (first-found wins). Used to stitch discovered-intermediate leaves back into
    their builds. Base ZINC blocks never appear as a product, so they never enter this map."""
    build = {}

    def walk(node):
        if not isinstance(node, dict) or node.get("type") != "mol":
            return
        children = node.get("children") or []
        if children:  # produced by a reaction -> buildable
            c = _canon(node.get("smiles"), strip_stereo)
            if c and c not in build:
                build[c] = node
            for rxn in children:
                for m in rxn.get("children") or []:
                    walk(m)

    for routes in trees_by_target.values():
        for r in routes:
            walk(r)
    return build


def _flatten_stitch(mol_node, build_map, strip_stereo, steps, seen):
    """Flatten a nested AiZynth mol-node into flat ``{product, reactants}`` steps, stitching any
    discovered-intermediate leaf (present in ``build_map``) into its build subtree. Base leaves stop.
    """
    if not isinstance(mol_node, dict):
        return
    smi = mol_node.get("smiles")
    children = mol_node.get("children") or []
    if not children:
        # leaf: stitch iff it is a discovered intermediate (buildable elsewhere); else it's a real
        # ZINC starting material -> stop (build_network will mark it buyable).
        c = _canon(smi, strip_stereo)
        if c and c in build_map and c not in seen:
            seen.add(c)
            _flatten_stitch(build_map[c], build_map, strip_stereo, steps, seen)
        return
    rxn = children[0]  # in a ROUTE (not a search tree) a mol has exactly one reaction child
    reactant_mols = rxn.get("children") or []
    steps.append({"product": smi, "reactants": [m.get("smiles") for m in reactant_mols]})
    for m in reactant_mols:
        _flatten_stitch(m, build_map, strip_stereo, steps, seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-smi", required=True, help="pool SMILES, one per line")
    ap.add_argument(
        "--out", required=True, help="output JSON: {target_canon: [ {steps:[...]}, ... ]}"
    )
    ap.add_argument("--config", required=True, help="AiZynthFinder config.yml")
    ap.add_argument("--stock", default="zinc")
    ap.add_argument("--expansion", default="uspto")
    ap.add_argument("--filter", default="uspto")
    ap.add_argument("--n-iters", type=int, default=5)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--strip-stereo", default="true")
    ap.add_argument(
        "--max-routes-per-target",
        type=int,
        default=0,
        help="cap candidate routes emitted per target (0 = all); bounds the SPARROW network size.",
    )
    a = ap.parse_args()
    strip = a.strip_stereo.lower() != "false"

    from aizynthfinder.aizynthfinder import AiZynthFinder
    from aizynthfinder.context.scoring import AverageTemplateOccurrenceScorer
    from aizynthfinder.reactiontree import ReactionTree
    from multiaiz.multiaiz import MultiAiZ

    # MultiAiZ (pinned aizynthfinder ^4.4.0) calls ReactionTree.get_subtree(mol) in
    # post_processing.select_best_route, but aizynthfinder 4.4.1 (our env) dropped it — it kept
    # subtrees() (all non-leaf subtrees, each with .root). Shim get_subtree(mol) = the subtree rooted
    # at mol, recovered from subtrees(). Version-tolerant (only added if missing); avoids editing the
    # upstream clone or changing the aizynth env's version (which the SPARROW route recovery shares).
    if not hasattr(ReactionTree, "get_subtree"):

        def _get_subtree(self, mol):
            target = getattr(mol, "smiles", None)
            for st in self.subtrees():
                if st.root is mol or getattr(st.root, "smiles", None) == target:
                    return st
            return None

        ReactionTree.get_subtree = _get_subtree

    targets = [ln.strip() for ln in open(a.pool_smi) if ln.strip()]
    finder = AiZynthFinder(configfile=a.config)
    finder.stock.select(a.stock)
    finder.expansion_policy.select(a.expansion)
    if a.filter and a.filter.lower() != "none":
        finder.filter_policy.select(a.filter)

    # MultiAiZ ranks intermediates for stock-promotion by an "intermediate score" whose UTILITY term
    # is a "reaction class-rank score" (a reaction-feasibility proxy) needing AstraZeneca-internal
    # reaction-class data we don't have (Logs/043). Substitute AiZynthFinder's DATA-FREE
    # AverageTemplateOccurrenceScorer under that exact name: well-precedented (common-template)
    # reactions score higher = a data-free feasibility proxy matching the paper's stated intent. Without
    # this, MultiAiZ crashes mid-cycle (KeyError) on the first real target's intermediates.
    # User-approved deviation (2026-07-22).
    class _TemplateOccUtility(AverageTemplateOccurrenceScorer):
        scorer_name = "reaction class-rank score"

    finder.scorers.load(_TemplateOccUtility(finder.config))

    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    m = MultiAiZ(finder, targets, str(work))
    try:
        m.run(n_iters=a.n_iters)
    except Exception as exc:  # noqa: BLE001
        # MultiAiZ's final intermediate-SCORING post-processing needs a scorer our config doesn't
        # load; it crashes AFTER the per-cycle route trees are written. We only need the trees.
        print(
            f"[multiaiz_worker] post-processing skipped ({type(exc).__name__}: {exc}); "
            "per-cycle trees already written",
            file=sys.stderr,
            flush=True,
        )

    # Gather per-target route trees across all cycles (iter_1 = base, later = convergent, deduped).
    trees_by_target = {}
    for tf in sorted(glob.glob(str(work / "trees" / "*.json.gz"))):
        with gzip.open(tf) as fh:
            data = json.load(fh)
        rows = data.get("data", []) if isinstance(data, dict) else data
        for row in rows:
            tc = _canon(row.get("target"), strip)
            if not tc:
                continue
            for rt in row.get("trees") or []:
                trees_by_target.setdefault(tc, []).append(rt)

    build_map = _harvest_build_subtrees(trees_by_target, strip)

    out = {}
    n_candidate = 0
    for tc, routes in trees_by_target.items():
        flat_routes = []
        for rt in routes:
            steps = []
            _flatten_stitch(rt, build_map, strip, steps, set())
            if steps:
                flat_routes.append(
                    {"product_smiles": tc, "num_reactions": len(steps), "steps": steps}
                )
        if a.max_routes_per_target and len(flat_routes) > a.max_routes_per_target:
            flat_routes.sort(key=lambda r: r["num_reactions"])  # keep the shortest-built candidates
            flat_routes = flat_routes[: a.max_routes_per_target]
        if flat_routes:
            out[tc] = flat_routes
            n_candidate += len(flat_routes)

    json.dump(out, open(a.out, "w"))
    print(
        f"[multiaiz_worker] {len(targets)} targets -> {len(out)} routed; "
        f"{n_candidate} candidate routes (n_iters={a.n_iters}) -> {a.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
