"""Nested dynamic-fragment amortization for the SCENT cost metric (Logs/028; proposal §11).

RGFN's reactions-per-mode (Logs/025/026) treats every building block as a free, purchasable base
fragment: cost = the reactions that *assemble* a molecule (``num_reactions``). SCENT breaks that
assumption — it promotes high-reward intermediates into its library and attaches them in one step,
so a molecule's ``num_reactions`` **hides** the reactions that built each promoted fragment. This
module adds those hidden costs back, honestly:

  * each **distinct** promoted fragment used anywhere in a costed library is built + charged
    **exactly once** (you'd synthesize a batch of it once and draw from it — in *both* the
    hub-amortized and independent plans), so it never inflates per-molecule cost and cancels out of
    the hub-vs-independent *saving*;
  * **nesting** is expanded: a promoted fragment built from another promoted fragment pulls the
    inner one into the charged set (closure under the recipes), each still charged once;
  * cost is reported in **reactions** (PRIMARY, RGFN-comparable — a promoted fragment's unit is the
    number of reaction steps in its logged route) and SCENT's **$-cost** (SECONDARY, from its cost
    tables — marginal per fragment so shared sub-fragments aren't double-counted).

Inputs come from the recipe re-run (``fragments_<N>.json``: ``chosen_smiles`` +
``smiles_to_route`` + ``smiles_to_min_num_reactions`` + ``chosen_smiles_costs``) and the worker's
per-molecule composition (``compositions.json``: which promoted fragments each hub/child uses). No
recipes (a recipe-less checkpoint) → falls back to ``min_num_reactions`` as the unit (no nesting),
which is the honest approximation entry `027` shipped before the recipe re-runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class FragmentCostTable:
    """Per-promoted-fragment unit build cost, with nesting closure.

    ``recipes``: ``{smiles: {"num_reactions": k, "steps": [{"reactants": [smiles...], ...}],
    "seed": smiles}}`` (the logged synthesis routes). ``min_num_reactions`` / ``dollar_costs``:
    ``{smiles: value}`` fallbacks / the $ secondary. ``promoted_set``: the set of promoted-fragment
    SMILES (``chosen_smiles``) — membership test for "is this reactant a promoted fragment".
    """

    promoted_set: Set[str]
    recipes: Optional[Dict[str, dict]] = None
    min_num_reactions: Optional[Dict[str, int]] = None
    dollar_costs: Optional[Dict[str, float]] = None

    def _route(self, f: str) -> Optional[dict]:
        return (self.recipes or {}).get(f)

    def promoted_reactants_of(self, f: str) -> Set[str]:
        """Promoted fragments directly consumed in ``f``'s route (its seed + step reactants)."""
        route = self._route(f)
        if route is None:
            return set()
        used = set()
        seed = route.get("seed")
        if seed in self.promoted_set and seed != f:
            used.add(seed)
        for step in route.get("steps", []):
            for r in step.get("reactants", []):
                if r in self.promoted_set and r != f:
                    used.add(r)
        return used

    def closure(self, fragments: Iterable[str]) -> Set[str]:
        """All distinct promoted fragments that must be built to make ``fragments`` — the input
        promoted fragments plus, recursively, every promoted fragment their routes consume."""
        seen: Set[str] = set()
        stack = [f for f in fragments if f in self.promoted_set]
        while stack:
            f = stack.pop()
            if f in seen:
                continue
            seen.add(f)
            for g in self.promoted_reactants_of(f):
                if g not in seen:
                    stack.append(g)
        return seen

    def unit_reactions(self, f: str) -> int:
        """``f``'s OWN reaction steps (its route length); promoted sub-fragments are charged
        separately via the closure. Falls back to ``min_num_reactions`` when no route is logged."""
        route = self._route(f)
        if route is not None:
            return int(route.get("num_reactions", len(route.get("steps", []))))
        if self.min_num_reactions is not None and f in self.min_num_reactions:
            return int(self.min_num_reactions[f])
        return 0

    def unit_dollars(self, f: str) -> float:
        """``f``'s MARGINAL $ — its total path cost minus the full cost of the promoted fragments
        its route consumes (those are charged once via the closure). Approximate (SCENT's cost
        tables); reactions is the exact primary. 0 when no $ table."""
        if not self.dollar_costs or f not in self.dollar_costs:
            return 0.0
        total = float(self.dollar_costs[f])
        sub = sum(float(self.dollar_costs.get(g, 0.0)) for g in self.promoted_reactants_of(f))
        return max(0.0, total - sub)

    def shared_build_cost(self, fragments: Iterable[str]) -> Tuple[int, float]:
        """(reactions, $) to build every distinct promoted fragment needed by ``fragments``, each
        charged exactly once (closure)."""
        clo = self.closure(fragments)
        reactions = sum(self.unit_reactions(f) for f in clo)
        dollars = sum(self.unit_dollars(f) for f in clo)
        return reactions, dollars


def load_cost_table_from_snapshot(
    snapshot: dict, promoted: Optional[List[str]] = None
) -> FragmentCostTable:
    """Build a :class:`FragmentCostTable` from a ``fragments_<N>.json`` dict (recipe re-run)."""
    chosen = list(promoted if promoted is not None else snapshot.get("chosen_smiles", []))
    costs = snapshot.get("chosen_smiles_costs", [])
    dollar = {s: costs[i] for i, s in enumerate(chosen) if i < len(costs)}
    return FragmentCostTable(
        promoted_set=set(chosen),
        recipes=snapshot.get("smiles_to_route") or None,
        min_num_reactions=snapshot.get("smiles_to_min_num_reactions") or None,
        dollar_costs=dollar or None,
    )
