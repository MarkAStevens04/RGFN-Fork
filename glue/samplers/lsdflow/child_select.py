"""Within-hub child-selection policies for hub-batching (Logs/037).

A hub exposes many one-reaction children; the campaign feeds them to a :class:`ModeSelector`
(reward gate + Tanimoto dedup) **in some order, possibly filtered**, and keeps the accepted ones as
modes. *Which* children we consider, and *in what order*, is this knob — the fragment-aware analogue
of ``mode_select.py`` (which decides "is this a new mode"; this decides "which children even get
offered, best-first").

The lean, current set (three policies + one strategy-level control, Logs/037):

- :class:`RewardChildPolicy` — the default; order by reward, keep everything. **Byte-identical to
  the historical inline sort** in :class:`~glue.samplers.lsdflow.campaign.HubBatchingStrategy`.
- :class:`FreeFragChildPolicy` — keep only children that require **no new synthesis** beyond
  building the hub: every promoted fragment the final reaction attaches is already available (base
  stock, already built this campaign, or built as part of *this* hub's scaffold). Every kept child
  then costs exactly **one** marginal reaction (the coupling). Used on its own it is the low-reaction
  extreme; with a **pre-built fragment stock** (``HubBatchingStrategy(prebuilt_fragments=...)``) it
  becomes the "pre-select-K" strategy — pre-synthesize K high-value fragments, then free-fill.
- :class:`FanoutMarginalPolicy` — the dynamic amortization policy: score each child against the
  *running* built set by ``reward − β · Σ_g cost(g)/fanout(g)`` over the fragments it would build
  **now** (its not-yet-built closure). ``cost(g)`` = ``g``'s reaction build cost; ``fanout(g)`` =
  how many hubs reuse it (from :func:`~glue.samplers.lsdflow.campaign.fragment_fanout`). A
  widely-reused fragment is cheap to justify building once, then free for every downstream hub; a
  one-off is crushed. Paired with the strategy's ``value_threshold`` move-on rule, this builds a few
  high-fan-out fragments early and reuses them across many hubs. β = 0 → reward order; β → ∞ →
  free-frag.

All operate on the same duck-typed child (``.smiles``, ``.reward``, ``.added_promoted`` = the
promoted dynamic-library fragments attached in the final reaction, empty when only base building
blocks are used).

**Static vs dynamic.** Static policies (reward, free_frag) rank children ONCE via ``order`` against
a fixed ``available`` set. Dynamic policies (:class:`DynamicChildPolicy`) instead ``score`` each
child against the *running* built set and are re-ranked by the strategy as fragments get built.

**Placement / AL reuse.** Pure ``glue/`` (RDKit-free here; fingerprints live in ``mode_select``), no
campaign/validation coupling — the policies take a duck-typed ``cost_table`` (anything exposing
``closure`` / ``unit_reactions``, e.g. the validation ``FragmentCostTable``) and a plain ``fanout``
dict, so the same objects drive the budget campaign *and* a future ``LSDFlowAcquisition`` in the AL
loop. RGFN (no dynamic library → every child's ``added_promoted`` is empty) degrades gracefully:
free-frag keeps everything, fanout's penalty is 0 — both collapse to reward order (nothing to
amortize).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence


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


def _new_fragments(added_promoted: Sequence[str], built: set, cost_table):
    """The promoted fragments a child would have to build *now* — its closure minus what's already
    built (``built``). Empty if no cost table / no promoted fragments. This is the set the count-once
    cost model actually charges (mirrors ``campaign._charge_promoted`` without mutating)."""
    if cost_table is None or not added_promoted:
        return ()
    return tuple(f for f in cost_table.closure(added_promoted) if f not in built)


class ChildSelectionPolicy(ABC):
    """Order (and optionally filter) a hub's children before the :class:`ModeSelector` sees them.

    ``order`` returns the children to offer, best-first. ``available`` is the set of promoted
    fragments already built *plus this hub's own scaffold fragments* (the caller folds those in,
    since building the hub makes them free for its children); it is only consulted by free-frag.
    """

    name: str = "reward"
    # Static policies rank children ONCE via ``order`` against a fixed ``available`` set. Dynamic
    # policies (:class:`DynamicChildPolicy`) instead ``score`` each child against the *running* built
    # set and are re-ranked by the strategy as fragments get built — set ``is_dynamic = True``.
    is_dynamic: bool = False

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


# ----------------------------------------------------------------- dynamic (amortization-aware) family
class DynamicChildPolicy(ChildSelectionPolicy):
    """Score each child against the **running built set** (higher = better), so a fragment costs only
    the *first* time it is built, then is free for every downstream child (Logs/037).

    This is what makes "build a few good fragments early, then reuse them" emerge: a fragment gets
    built only when a child's value justifies its one-time cost, after which the strategy re-ranks the
    remaining children — the fragment's siblings now score higher (their marginal cost dropped to 0).
    :class:`~glue.samplers.lsdflow.campaign.HubBatchingStrategy` runs the greedy re-ranking loop when
    ``is_dynamic`` (and advances to the next hub once the best remaining child falls below its
    ``value_threshold``). ``order`` is only a static fallback (score vs a frozen ``available``).
    """

    is_dynamic = True

    def score(self, child, built: set, cost_table, *, higher_is_better: bool = True) -> float:
        raise NotImplementedError

    def order(self, children, *, higher_is_better=True, cost_table=None, available=None):
        built = available or set()
        return sorted(
            children,
            key=lambda c: self.score(c, built, cost_table, higher_is_better=higher_is_better),
            reverse=True,
        )


def _oriented_reward(reward: float, higher_is_better: bool) -> float:
    if reward != reward:  # NaN → worst
        return float("-inf")
    return reward if higher_is_better else -reward


class FanoutMarginalPolicy(DynamicChildPolicy):
    """``score = reward − β · Σ_g cost(g)/fanout(g)`` over the fragments the child would build **now**
    (its not-yet-built closure). ``cost(g)`` = ``g``'s reaction build cost; ``fanout(g)`` = how many
    hubs reuse ``g`` (:func:`~glue.samplers.lsdflow.campaign.fragment_fanout`).

    So the one-time cost of a fragment is discounted by how widely it will be reused: a high-fan-out
    block (used across dozens of hubs) is nearly free to justify and, once built, is free for every
    downstream hub; a one-off carries its full cost and is left behind. ``fanout`` is measured on the
    reward-gated (hit) children — only those can become modes. β = 0 → reward order; large β →
    free-frag. A base-only child (no promoted fragment) always has penalty 0.
    """

    name = "fanout"

    def __init__(
        self,
        beta: float,
        fanout: Optional[Dict[str, float]] = None,
        *,
        fanout_floor: float = 1.0,
    ):
        self.beta = float(beta)
        self.fanout = fanout or {}
        self.fanout_floor = float(fanout_floor)

    def score(self, child, built, cost_table, *, higher_is_better=True):
        r = _oriented_reward(child.reward, higher_is_better)
        pen = 0.0
        for g in _new_fragments(getattr(child, "added_promoted", ()), built, cost_table):
            pen += float(cost_table.unit_reactions(g)) / max(
                self.fanout.get(g, 0.0), self.fanout_floor
            )
        return r - self.beta * pen


_POLICIES = {
    RewardChildPolicy.name: RewardChildPolicy,
    FreeFragChildPolicy.name: FreeFragChildPolicy,
    FanoutMarginalPolicy.name: FanoutMarginalPolicy,
}


def available_child_policies() -> List[str]:
    return list(_POLICIES)


def make_child_policy(name: str, **kwargs) -> ChildSelectionPolicy:
    """Build a policy by name. ``fanout`` accepts ``beta`` + ``fanout`` (the per-fragment reuse
    counts); ``reward``/``free_frag`` ignore kwargs. Unknown name → ``KeyError`` with the available
    set (fail fast in the driver)."""
    try:
        cls = _POLICIES[name]
    except KeyError:
        raise KeyError(f"unknown child policy {name!r}; available: {available_child_policies()}")
    if cls is FanoutMarginalPolicy:
        return cls(beta=kwargs.get("beta", 0.0), fanout=kwargs.get("fanout"))
    return cls()
