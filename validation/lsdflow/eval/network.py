"""Merge per-molecule synthesis routes into ONE SPARROW reaction network (T1.2).

The from-scratch / native-route SPARROW evaluators price a *whole library at once*: the win of a
reaction-GFN's hub-batching is that its molecules **share intermediates**, so a batch route planner
can build each shared piece once. This module turns a set of per-molecule routes into the single
merged reaction network SPARROW's MILP consumes — **deduplicating nodes by canonical SMILES**, so a
shared intermediate becomes exactly one node (built once by the MILP) and molecules with no shared
structure stay separate (no spurious merges).

Input route schema — the ``glue/active_learning/route.py`` / ``routes.jsonl`` dict::

    {product_smiles, num_reactions, building_block: {smiles, idx},
     steps: [{step, reaction_idx, reaction_smarts, reactant, fragments: [...], product}]}

It also tolerates the SCENT worker's ``routes.json`` step shape (``{reaction, reactants, input,
product}``) so native-route SCENT libraries (CHECK 1) feed the same path.

Output — SPARROW's ``tree.json`` schema (``RouteGraph(node_filename=...)``)::

    {"Compound Nodes": [{smiles, buyable, cost_per_g?}, ...],
     "Reaction Nodes": [{smiles: "r1.r2>>product", score}, ...]}

plus a ``target_dict`` {canonical_target_smiles: reward} for SPARROW's ``targets.csv``.

Env-light on purpose: imports only RDKit + stdlib (no ``glue``/``rgfn``/``sparrow``), so it runs in
the ``rgfn`` env (building the network) and inside the ``sparrow`` worker env alike.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# --- canonicalization (inline RDKit; mirrors validation/harness/cost.py so keys agree) --------------
# strip_stereo: the LSD-Flow campaign's node identity is the STEREO-STRIPPED cross-model key
# (MolToSmiles(mol, isomericSmiles=False); proposal §6) — count-once, modes, and diversity all use
# it. So the SPARROW network must strip too, or a mode's stripped identity won't match its route
# step products (which keep stereo) and SPARROW drops it. Default True for benchmark consistency;
# pass False to keep stereo (e.g. a stereo-aware within-reaction-model study).
def canonical(smiles: Optional[str], strip_stereo: bool = True) -> Optional[str]:
    """RDKit canonical SMILES, or ``None`` if unparseable/empty. ``strip_stereo`` drops stereochem
    (the campaign's cross-model key convention)."""
    if not smiles:
        return None
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=not strip_stereo)


def canonical_reaction(
    reactants: Sequence[str], product: str, strip_stereo: bool = True
) -> Optional[str]:
    """Order-invariant reaction key ``"r1.r2>>product"`` with every SMILES canonicalized.

    Reactants are canonicalized then **sorted** (RGFN/SCENT anchor the first reactant, so the same
    template is written in either reactant order — sorting makes the merge order-invariant, exactly
    as ``validation/harness/cost.py::_canonical_reaction`` does). ``None`` if the product or any
    reactant is unparseable (the step is dropped and counted, never silently guessed)."""
    cp = canonical(product, strip_stereo)
    if cp is None:
        return None
    cr = [canonical(r, strip_stereo) for r in reactants]
    if any(r is None for r in cr):
        return None
    return f"{'.'.join(sorted(cr))}>>{cp}"


def _step_reactants_product(step: dict) -> Tuple[List[str], Optional[str]]:
    """Extract (reactants, product) from one route step, tolerating both schemas.

    route.py / routes.jsonl: ``reactant`` (the input molecule) + ``fragments`` (attached blocks).
    SCENT routes.json:       ``input`` (or ``reactants``) + ``reactants``.
    """
    product = step.get("product")
    if "reactant" in step:  # route.py schema
        reactants = [step["reactant"], *(step.get("fragments") or [])]
    elif "input" in step:  # SCENT routes.json schema (input molecule + attached reactants)
        reactants = [step["input"], *(step.get("reactants") or [])]
    else:
        reactants = list(step.get("reactants") or [])
    return [r for r in reactants if r], product


def expand_route_with_recipes(
    route: dict,
    recipes: Dict[str, dict],
    promoted: set,
    _seen: Optional[set] = None,
) -> dict:
    """Recipe-expand a *shallow* native route into a FULLY-NESTED one for fair SPARROW pricing (T1.5).

    A reaction-GFN's native route (``routes.json`` / a hub's route + final reaction) is *shallow*: a
    promoted dynamic-library fragment is attached in ONE step, not synthesized. But DAG count-once
    charges each promoted fragment's own nested build (once). So to reconcile native-route SPARROW
    against count-once, the promoted fragments a route consumes must be expanded into their build
    steps (``smiles_to_route`` recipes), recursively — then SPARROW counts the same fragment-build
    reactions count-once does (both deduped: SPARROW by canonical reaction, count-once by fragment).

    Returns a route with the SAME shape whose ``steps`` are ``[recipe steps of every promoted
    fragment consumed (closure, each once), ..., original steps]`` — i.e. every promoted fragment is
    now a reaction *product* (built) rather than a bought leaf. ``_seen`` guards against cycles /
    double-expansion within one call. Non-promoted (base) reactants are untouched (stay buyable).
    """
    seen = _seen if _seen is not None else set()
    expanded_steps: List[dict] = []
    for step in route.get("steps", []) or []:
        reactants, _product = _step_reactants_product(step)
        for r in reactants:
            if r in promoted and r not in seen:
                seen.add(r)
                sub = recipes.get(r)
                if sub:  # prepend the fragment's (recursively expanded) build steps
                    expanded_steps.extend(
                        expand_route_with_recipes(sub, recipes, promoted, seen)["steps"]
                    )
        expanded_steps.append(step)
    return {
        "product_smiles": route.get("product_smiles") or route.get("product"),
        "num_reactions": len(expanded_steps),
        "steps": expanded_steps,
    }


@dataclass
class NetworkStats:
    """What the merge produced — a fairness/debug record carried alongside the network."""

    n_targets: int = 0
    n_compound_nodes: int = 0
    n_reaction_nodes: int = 0
    n_buyable: int = 0
    n_intermediates: int = 0  # compounds that are produced AND consumed (shared or not)
    n_shared_compounds: int = 0  # compound nodes reachable in >1 target's route (the merge payoff)
    n_dropped_steps: int = 0  # steps with an unparseable SMILES (excluded, reported)
    n_routeless_targets: int = 0  # entries with no usable route (excluded from the network)


@dataclass
class ReactionNetwork:
    """The merged network + everything SPARROW needs, plus provenance."""

    graph: Dict[str, list]  # {"Compound Nodes": [...], "Reaction Nodes": [...]}
    target_dict: Dict[str, float]  # {canonical_target_smiles: reward}
    stats: NetworkStats = field(default_factory=NetworkStats)

    def write(self, out_dir: Path) -> Tuple[Path, Path]:
        """Write ``tree.json`` + ``targets.csv`` (SPARROW's two inputs); returns their paths."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tree = out_dir / "tree.json"
        targets = out_dir / "targets.csv"
        tree.write_text(json.dumps(self.graph, indent=2))
        with open(targets, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["SMILES", "Reward"])
            for smi, reward in self.target_dict.items():
                w.writerow([smi, reward])
        (out_dir / "network_stats.json").write_text(json.dumps(asdict(self.stats), indent=2))
        return tree, targets


def load_price_table(fragments_csv: Path, strip_stereo: bool = True) -> Dict[str, float]:
    """{canonical_smiles: cost} from a glue chemistry library ``fragments.csv`` (smiles,cost). Keys
    canonicalized (``strip_stereo`` matching the network) so they match compound keys. Missing/blank
    costs skipped."""
    table: Dict[str, float] = {}
    with open(fragments_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            cost = row.get("cost", "")
            if cost in ("", None):
                continue
            c = canonical(row.get("smiles", "").strip(), strip_stereo)
            if c is not None:
                table[c] = float(cost)
    return table


def build_network(
    entries: Sequence[dict],
    *,
    price_table: Optional[Dict[str, float]] = None,
    default_cost: float = 1.0,
    reaction_score: float = 1.0,
    strip_stereo: bool = True,
) -> ReactionNetwork:
    """Merge per-molecule routes into one SPARROW reaction network.

    ``entries``: ``[{"smiles": <target>, "reward": <float>, "route": <route dict>}, ...]``. Each
    route's steps become reaction nodes ``"r1.r2>>product"`` (canonical, order-invariant); every
    molecule (reactants, fragments, products, target) becomes a compound node, **deduplicated by
    canonical SMILES** — so a shared intermediate is a single node the MILP builds once. A compound
    that is never any reaction's product is a **starting material** (``buyable=True``), priced from
    ``price_table`` (else ``default_cost``). ``reaction_score`` sets each reaction's SPARROW score
    (1.0 ⇒ penalty 1.0 ⇒ objective can count reactions uniformly).

    Returns a :class:`ReactionNetwork` (graph + target_dict + stats). Unparseable steps / route-less
    targets are dropped and counted in ``stats`` (a reported fairness stat, never a silent guess).
    """
    price_table = price_table or {}
    reaction_nodes: Dict[str, dict] = {}  # canonical rxn smiles -> node
    products: set = set()  # canonical SMILES ever produced by a reaction
    compounds: set = set()  # every canonical compound SMILES seen
    # For "shared" accounting: which targets' routes touch each compound.
    compound_targets: Dict[str, set] = {}
    target_dict: Dict[str, float] = {}
    stats = NetworkStats()

    for entry in entries:
        target = canonical(entry.get("smiles"), strip_stereo)
        route = entry.get("route") or {}
        steps = route.get("steps") or []
        if target is None or not steps:
            stats.n_routeless_targets += 1
            continue
        target_dict[target] = float(entry.get("reward", 0.0))
        touched: set = set()
        for step in steps:
            reactants, product = _step_reactants_product(step)
            rxn_key = canonical_reaction(reactants, product, strip_stereo) if product else None
            if rxn_key is None:
                stats.n_dropped_steps += 1
                continue
            reactant_part, product_part = rxn_key.split(">>")
            r_smis = reactant_part.split(".") if reactant_part else []
            for c in [*r_smis, product_part]:
                compounds.add(c)
                touched.add(c)
            products.add(product_part)
            if rxn_key not in reaction_nodes:
                reaction_nodes[rxn_key] = {"smiles": rxn_key, "score": reaction_score}
        for c in touched:
            compound_targets.setdefault(c, set()).add(target)

    # Compound nodes: buyable == never produced by any reaction (a starting material / leaf).
    compound_nodes = []
    for smi in sorted(compounds):
        buyable = smi not in products
        node: Dict[str, object] = {"smiles": smi, "buyable": buyable}
        if buyable:
            node["cost_per_g"] = float(price_table.get(smi, default_cost))
            stats.n_buyable += 1
        else:
            stats.n_intermediates += 1  # produced (target or true intermediate)
        compound_nodes.append(node)

    stats.n_targets = len(target_dict)
    stats.n_compound_nodes = len(compound_nodes)
    stats.n_reaction_nodes = len(reaction_nodes)
    stats.n_shared_compounds = sum(1 for t in compound_targets.values() if len(t) > 1)
    # Targets are products, not starting materials; n_intermediates counts targets too, so subtract.
    stats.n_intermediates = max(0, stats.n_intermediates - stats.n_targets)

    graph = {
        "Compound Nodes": compound_nodes,
        "Reaction Nodes": list(reaction_nodes.values()),
    }
    return ReactionNetwork(graph=graph, target_dict=target_dict, stats=stats)
