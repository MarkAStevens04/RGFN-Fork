"""Turn SynFormer's synthesis stacks into the project's route schema.

SynFormer is the one entrant besides our own generators that produces molecules **with a route**, so
this file is what makes it comparable: it converts SynFormer's internal synthesis representation into
the same ``steps`` schema ``glue/active_learning/route.py`` defines and
``validation/lsdflow/eval/network.py`` consumes, letting the identical SPARROW pricing run over both
sides.

WHY WE PATCH RATHER THAN PARSE. The obvious source is the ``synthesis`` column of the sampler's
result frame, which is ``Stack.get_action_string()`` — a postfix string of building-block SMILES and
``R<idx>`` reaction tokens. It is not enough. It records the leaves and the reactions but **not the
intermediate products**, and a reaction network is exactly a graph of intermediates. Recovering them
would mean re-running each template with RDKit and disambiguating among multiple products — a
reimplementation of the model's own bookkeeping, with a fresh opportunity to be subtly wrong.

The intermediates *are* known in-process: ``Stack.get_tree()`` returns a node per molecule, each
carrying its ``mol`` and the ``rxn`` that produced it. They are simply not serialized. So we patch
``StatePool.get_dataframe`` to add one extra column carrying the full tree, and read it back out.
The workers are ``mp.Process`` under Linux's default **fork** start method, so a patch applied in the
parent before sampling begins is inherited by every worker (their CUDA init happens after the fork,
which is what makes fork safe here).

CHILD ORDER IS REVERSED, AND IT MATTERS. ``Stack.get_tree`` builds each node with
``for _ in range(n_react): item.children.append(stack.pop())`` — popping a stack, so ``children[0]``
is the LAST reactant pushed. We reverse it to recover reactant order before emitting
``reactant``/``fragments``, which is the difference between a route that reads correctly and one
whose reactants are silently transposed.

LEAVES ARE PURCHASABLE. Every leaf is an Enamine building block from SynFormer's catalogue, so unlike
our own native routes these need **no recipe expansion** — nothing has to be built before it can be
used. That is why the frontier prices them with ``--route-source external`` rather than ``native``.
"""

from __future__ import annotations

__all__ = ["tree_to_steps", "patch_get_dataframe", "ROUTE_COLUMN"]

import json
from typing import Dict, List

#: Extra column the patch adds to the sampler's result frame.
ROUTE_COLUMN = "route_steps_json"


def tree_to_steps(node) -> List[Dict]:
    """Post-order walk of a SynFormer ``Stack`` tree into our ``steps`` list.

    Each emitted step is ``{step, reaction_idx, reaction_smarts, reactant, fragments, product}``,
    ordered so that every step's inputs are either purchasable leaves or products of earlier steps —
    the ordering ``network.py`` assumes when it stitches a route into the merged reaction graph.
    """
    steps: List[Dict] = []

    def visit(n) -> str:
        smiles = n.mol.smiles
        if n.rxn is None:  # leaf: a purchasable building block, not a reaction
            return smiles
        # children are stack-popped, so reverse to recover the template's reactant order
        child_smiles = [visit(c) for c in reversed(n.children)]
        steps.append(
            {
                "step": len(steps) + 1,
                # token is (num_reactants, idx); idx indexes SynFormer's reaction list
                "reaction_idx": int(n.token[1]) if n.token is not None else None,
                "reaction_smarts": n.rxn.smarts,
                "reactant": child_smiles[0] if child_smiles else None,
                "fragments": child_smiles[1:],
                "product": smiles,
            }
        )
        return smiles

    visit(node)
    return steps


def patch_get_dataframe() -> None:
    """Make ``StatePool.get_dataframe`` also emit :data:`ROUTE_COLUMN`.

    Call once in the parent process **before** any sampling worker is forked. Idempotent.
    """
    from synformer.sampler.analog.state_pool import StatePool

    if getattr(StatePool.get_dataframe, "_glue_patched", False):
        return

    original = StatePool.get_dataframe

    def get_dataframe(self, *args, **kwargs):
        df = original(self, *args, **kwargs)
        if len(df) == 0:
            return df
        # Re-walk the products in the same order `original` built its rows, so the column aligns.
        # `original` sorts rows by score descending, so build a smiles -> route map instead of
        # relying on positional order — products are unique by SMILES within a StatePool.
        by_smiles: Dict[str, str] = {}
        for product in self.get_products():
            try:
                by_smiles[product.molecule.smiles] = json.dumps(
                    tree_to_steps(product.stack.get_tree())
                )
            except Exception:
                # A route we cannot serialize must be visibly absent, never silently empty: an
                # empty step list would price as a FREE molecule and flatter this baseline.
                by_smiles[product.molecule.smiles] = ""
        df[ROUTE_COLUMN] = [by_smiles.get(s, "") for s in df["smiles"]]
        return df

    get_dataframe._glue_patched = True  # type: ignore[attr-defined]
    StatePool.get_dataframe = get_dataframe  # type: ignore[method-assign]
