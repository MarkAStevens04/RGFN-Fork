"""Within-hub child-selection policies for hub-batching (Logs/037).

A hub exposes many one-reaction children; the campaign feeds them to a :class:`ModeSelector`
(reward gate + Tanimoto dedup) **in some order, possibly filtered**, and keeps the accepted ones as
modes. *Which* children we consider, and *in what order*, is this knob — the fragment-aware analogue
of ``mode_select.py`` (which decides "is this a new mode"; this decides "which children even get
offered, best-first").

The lean, current set (Logs/037) — two swappable within-hub policies:

- :class:`RewardChildPolicy` — **naive hub-batching**: order by reward, keep everything, and don't
  consider the cost of the fragment a child attaches at all. This is the original approach (Logs/029),
  **byte-identical to the historical inline sort**, kept as the no-cost-awareness baseline/control.
- :class:`FreeFragChildPolicy` — keep only children that require **no new synthesis** beyond building
  the hub: every promoted fragment the final reaction attaches is already available (base stock,
  already built this campaign, or built as part of *this* hub's scaffold). Every kept child then costs
  exactly **one** marginal reaction (the coupling). On its own it is the low-reaction extreme; with a
  **pre-built fragment stock** (``HubBatchingStrategy(prebuilt_fragments=...)``) it becomes the
  **pre-select-K** strategy — pre-synthesize K high-value fragments (ranked by build-score, see
  :func:`~glue.samplers.lsdflow.campaign.rank_fragments`), charge them once up front, then free-fill.
  K trades a few upfront fragment builds for fewer hubs walked (fewer reward-gen/oracle calls).

Both operate on the same duck-typed child (``.smiles``, ``.reward``, ``.added_promoted`` = the
promoted dynamic-library fragments attached in the final reaction, empty when only base building
blocks are used), and rank children ONCE via ``order`` against a fixed ``available`` set (the set of
promoted fragments already built + this hub's own scaffold fragments; consulted only by free-frag).

To add an ablation policy, subclass :class:`ChildSelectionPolicy`, implement ``order``, and register
it in ``_POLICIES`` — nothing else in the loop needs to change.

**Placement / AL reuse.** Pure ``glue/`` (RDKit-free here; fingerprints live in ``mode_select``), no
campaign/validation coupling — the policies take a duck-typed ``cost_table`` (anything exposing
``closure`` / ``unit_reactions``, e.g. the validation ``FragmentCostTable``), so the same objects
drive the budget campaign *and* a future ``LSDFlowAcquisition`` in the AL loop. RGFN (no dynamic
library → every child's ``added_promoted`` is empty) degrades gracefully: free-frag keeps everything
(collapses to reward order — nothing to amortize).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence


def _reward_key(reward: float) -> float:
    """Sort key that demotes NaN to the tail; NaN → -inf so ``reverse=higher_is_better`` places it
    last for higher-is-better."""
    return reward if reward == reward else float("-inf")


def _needs_new_build(added_promoted: Sequence[str], available: set, cost_table) -> bool:
    """True iff attaching ``added_promoted`` would build a promoted fragment not yet available.

    A fragment is "available" if it is base stock (never appears in ``added_promoted`` — that field
    lists only promoted fragments) or already in ``available``. Nesting counts: if a promoted
    fragment's route consumes another promoted fragment, that inner one must be available too, so we
    test the whole closure. No cost table (RGFN / no dynamic library) → nothing is promoted → free.
    """
    if cost_table is None or not added_promoted:
        return False
    closure = cost_table.closure(added_promoted)
    return any(f not in available for f in closure)


class ChildSelectionPolicy(ABC):
    """Order (and optionally filter) a hub's children before the :class:`ModeSelector` sees them.

    ``order`` returns the children to offer, best-first. ``available`` is the set of promoted
    fragments already built *plus this hub's own scaffold fragments* (the caller folds those in,
    since building the hub makes them free for its children); it is only consulted by free-frag.
    """

    name: str = "reward"

    @abstractmethod
    def order(
        self,
        children: Sequence,
        *,
        higher_is_better: bool = True,
        cost_table=None,
        available: Optional[set] = None,
    ) -> List:
        ...


class RewardChildPolicy(ChildSelectionPolicy):
    """Reward-first, keep everything — the historical hub-batching behaviour (Logs/029/033/035)."""

    name = "reward"

    def order(self, children, *, higher_is_better=True, cost_table=None, available=None):
        return sorted(children, key=lambda c: _reward_key(c.reward), reverse=higher_is_better)


class FreeFragChildPolicy(ChildSelectionPolicy):
    """Keep only children whose final reaction attaches an already-available fragment (base stock,
    already built, or part of this hub) → each kept child costs exactly one marginal reaction. Order
    the survivors by reward. A hub whose every diverse child needs a fresh fragment yields 0 modes
    (and is never built). With a pre-built fragment stock (``prebuilt_fragments`` on the strategy) its
    "available" set starts non-empty → the stock fragments' children are free too (pre-select-K)."""

    name = "free_frag"

    def order(self, children, *, higher_is_better=True, cost_table=None, available=None):
        avail = available or set()
        free = [
            c
            for c in children
            if not _needs_new_build(getattr(c, "added_promoted", ()), avail, cost_table)
        ]
        return sorted(free, key=lambda c: _reward_key(c.reward), reverse=higher_is_better)


_POLICIES = {
    RewardChildPolicy.name: RewardChildPolicy,
    FreeFragChildPolicy.name: FreeFragChildPolicy,
}


def available_child_policies() -> List[str]:
    return list(_POLICIES)


def make_child_policy(name: str, **kwargs) -> ChildSelectionPolicy:
    """Build a policy by name (``reward`` / ``free_frag``; both ignore kwargs). Unknown name →
    ``KeyError`` with the available set (fail fast in the driver)."""
    try:
        cls = _POLICIES[name]
    except KeyError:
        raise KeyError(f"unknown child policy {name!r}; available: {available_child_policies()}")
    return cls()
