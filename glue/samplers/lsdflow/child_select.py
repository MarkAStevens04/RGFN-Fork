"""Within-hub child-selection policies for hub-batching (Logs/036).

A hub exposes many one-reaction children; the campaign feeds them to a :class:`ModeSelector`
(reward gate + Tanimoto dedup) **in some order, possibly filtered**, and keeps the accepted ones
as modes. *Which* children we consider, and *in what order*, is this knob — the fragment-aware
analogue of ``mode_select.py`` (which decides "is this a new mode"; this decides "which children
even get offered, best-first").

Three policies, all operating on the same duck-typed child (``.smiles``, ``.reward``,
``.added_promoted`` = the promoted dynamic-library fragments attached in the final reaction, empty
when the final reaction uses only base building blocks):

- :class:`RewardChildPolicy` — the default; order by reward, keep everything. **Byte-identical to
  the pre-Logs/036 inline sort in** :class:`~glue.samplers.lsdflow.campaign.HubBatchingStrategy`.
- :class:`FreeFragChildPolicy` — keep only children that require **no new synthesis** beyond
  building the hub: every promoted fragment the final reaction attaches is already available
  (base stock, already built this campaign, or built as part of *this* hub's scaffold). Every kept
  child then costs exactly **one** marginal reaction (the coupling). Order the survivors by reward.
- :class:`SmartFragChildPolicy` — soft version: order by ``reward − β · Σ_f cost(f)/utility(f)``
  over the attached promoted fragments ``f``. ``cost(f)`` is ``f``'s nested reaction build cost;
  ``utility(f)`` is a per-fragment goodness on the reward scale (SCENT's ``smiles_to_mean_reward``,
  Eq. 13, pre-scaled by the caller — see the note below). ``β = 0`` recovers reward-only order;
  larger ``β`` tilts toward cheap, high-utility fragments. Nothing is filtered — cheap decorations
  simply rank ahead of expensive ones, and the reward gate / diversity dedup still run downstream.

**Placement / AL reuse.** Pure ``glue/`` (RDKit-free here; fingerprints live in ``mode_select``),
no campaign/validation coupling — the policies take a duck-typed ``cost_table`` (anything exposing
``closure`` / ``shared_build_cost``, e.g. the validation ``FragmentCostTable``) and a plain
``utilities`` dict, so the same objects drive the budget campaign *and* a future
``LSDFlowAcquisition`` in the active-learning loop. RGFN (no dynamic library → every child's
``added_promoted`` is empty) degrades gracefully: free-frag keeps everything, smart-frag's penalty
is 0 — both collapse to reward order, which is correct (nothing to amortize).

**Utility scale (important).** SCENT stores ``smiles_to_mean_reward`` as the mean of the *shaped*
reward ``R(x) = exp(β_train · proxy)`` — astronomically large (~1e27 for β_train=8). Divided into a
reaction count that is ~0, which would make smart-frag ≡ reward order. The caller must therefore
pass ``utilities`` **already on the reward/proxy scale** (e.g. ``log(mean_reward)/β_train``); this
module does not rescale — it just divides cost by whatever "goodness" it is handed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence


def _reward_key(reward: float) -> float:
    """Sort key that demotes NaN to the tail regardless of ``reverse`` direction is handled by the
    caller; here NaN → -inf so ``reverse=higher_is_better`` places it last for higher-is-better."""
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


def marginal_new_reactions(added_promoted: Sequence[str], built: set, cost_table) -> int:
    """Reactions a child *actually* adds given ``built``: the build cost of only its not-yet-built
    promoted fragments (0 for base-only or fully-reused children). The reaction the child's final
    coupling costs (+1) is common to every child, so it's left out of the penalty."""
    if cost_table is None:
        return 0
    return sum(
        int(cost_table.unit_reactions(f)) for f in _new_fragments(added_promoted, built, cost_table)
    )


class ChildSelectionPolicy(ABC):
    """Order (and optionally filter) a hub's children before the :class:`ModeSelector` sees them.

    ``order`` returns the children to offer, best-first. ``available`` is the set of promoted
    fragments already built *plus this hub's own scaffold fragments* (the caller folds those in,
    since building the hub makes them free for its children); it is only consulted by free-frag.
    """

    name: str = "reward"
    # Static policies (below) rank children ONCE via ``order`` against a fixed ``available`` set.
    # Dynamic policies (:class:`DynamicChildPolicy`) instead score each child against the *running*
    # built set and are re-ranked by the strategy as fragments get built — set ``is_dynamic = True``.
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
    (and is never built) — the cost-vs-coverage trade this policy exists to measure."""

    name = "free_frag"

    def order(self, children, *, higher_is_better=True, cost_table=None, available=None):
        avail = available or set()
        free = [
            c
            for c in children
            if not _needs_new_build(getattr(c, "added_promoted", ()), avail, cost_table)
        ]
        return sorted(free, key=lambda c: _reward_key(c.reward), reverse=higher_is_better)


class SmartFragChildPolicy(ChildSelectionPolicy):
    """Order by ``reward − β · Σ_f cost(f)/utility(f)`` over the attached promoted fragments.

    ``cost(f)`` = ``f``'s nested reaction build cost (``cost_table.shared_build_cost([f])[0]``);
    ``utilities[f]`` = ``f``'s goodness on the reward scale (see module note). A child with no
    promoted fragment (base-only decoration) has penalty 0 and keeps its raw reward. ``β = 0``
    reproduces reward order; large ``β`` prefers cheap / high-utility fragments. Nothing is
    filtered; the reward gate + diversity dedup downstream are unchanged.
    """

    name = "smart_frag"

    def __init__(
        self,
        beta: float,
        utilities: Optional[Dict[str, float]] = None,
        *,
        utility_floor: float = 1e-6,
    ):
        self.beta = float(beta)
        self.utilities = utilities or {}
        self.utility_floor = float(utility_floor)

    def _penalty(self, child, cost_table) -> float:
        added = getattr(child, "added_promoted", ())
        if not added or cost_table is None:
            return 0.0
        total = 0.0
        for f in added:
            cost = float(cost_table.shared_build_cost([f])[0])
            util = max(self.utilities.get(f, 0.0), self.utility_floor)
            total += cost / util
        return total

    def order(self, children, *, higher_is_better=True, cost_table=None, available=None):
        sign = 1.0 if higher_is_better else -1.0

        def score(c) -> float:
            r = c.reward if c.reward == c.reward else float("-inf")
            return sign * r - self.beta * self._penalty(c, cost_table)

        # Higher score is always better after the orientation flip; NaN reward → -inf → last.
        return sorted(children, key=score, reverse=True)


# ----------------------------------------------------------------- dynamic (amortization-aware) family
class DynamicChildPolicy(ChildSelectionPolicy):
    """Score each child against the **running built set** (higher = better), so a fragment costs only
    the *first* time it is built, then is free for every downstream child (Logs/037 rework).

    This is what makes "build a few good fragments early, then reuse them" emerge: a fragment gets
    built only when some child's reward justifies its one-time cost, after which the strategy re-ranks
    the remaining children — the fragment's siblings now score higher (their marginal cost dropped to
    0). The strategy (:class:`~glue.samplers.lsdflow.campaign.HubBatchingStrategy`) runs a greedy
    re-ranking loop when ``is_dynamic``; ``order`` is only a static fallback (score vs a frozen
    ``available``). β → ∞ recovers free-frag; β = 0 recovers reward order.
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


class MarginalReactionPolicy(DynamicChildPolicy):
    """(A) ``score = reward − β · (new reactions the child adds given what's built)``. The penalty is
    exactly the count-once marginal reaction cost, so greedy selection directly minimises
    reactions/mode. A child reusing base/already-built fragments has penalty 0; one needing a fresh
    fragment pays β·(its build cost) — justified only by enough reward, then free for its siblings.
    """

    name = "marginal"

    def __init__(self, beta: float):
        self.beta = float(beta)

    def score(self, child, built, cost_table, *, higher_is_better=True):
        r = _oriented_reward(child.reward, higher_is_better)
        m = marginal_new_reactions(getattr(child, "added_promoted", ()), built, cost_table)
        return r - self.beta * m


class SmartFragDynamicPolicy(DynamicChildPolicy):
    """The original smart-frag intent, fixed to be dynamic: ``score = reward − β · Σ_g cost(g)/util(g)``
    over the fragments the child would build **now** (its not-yet-built closure). ``cost(g)`` = ``g``'s
    own reaction step count; ``util(g)`` = its SCENT utility on the reward scale. So a high-*potential*
    (high-utility) fragment is cheap to justify building once; afterwards it is free. β = 0 → reward
    order."""

    name = "smart_dyn"

    def __init__(
        self,
        beta: float,
        utilities: Optional[Dict[str, float]] = None,
        *,
        utility_floor: float = 1e-6,
    ):
        self.beta = float(beta)
        self.utilities = utilities or {}
        self.utility_floor = float(utility_floor)

    def score(self, child, built, cost_table, *, higher_is_better=True):
        r = _oriented_reward(child.reward, higher_is_better)
        pen = 0.0
        for g in _new_fragments(getattr(child, "added_promoted", ()), built, cost_table):
            pen += float(cost_table.unit_reactions(g)) / max(
                self.utilities.get(g, 0.0), self.utility_floor
            )
        return r - self.beta * pen


class RewardPerReactionPolicy(DynamicChildPolicy):
    """(C) ``score = reward / (1 + β · new reactions)`` — reward earned per reaction spent. A free child
    keeps its full reward; a child needing a fresh k-reaction fragment is divided by ``1 + βk`` unless
    already built. Assumes higher-is-better with positive reward (true for the sEH/proxy scale)."""

    name = "ratio"

    def __init__(self, beta: float = 1.0):
        self.beta = float(beta)

    def score(self, child, built, cost_table, *, higher_is_better=True):
        r = _oriented_reward(child.reward, higher_is_better)
        m = marginal_new_reactions(getattr(child, "added_promoted", ()), built, cost_table)
        return r / (1.0 + self.beta * m)


_POLICIES = {
    RewardChildPolicy.name: RewardChildPolicy,
    FreeFragChildPolicy.name: FreeFragChildPolicy,
    SmartFragChildPolicy.name: SmartFragChildPolicy,
    MarginalReactionPolicy.name: MarginalReactionPolicy,
    SmartFragDynamicPolicy.name: SmartFragDynamicPolicy,
    RewardPerReactionPolicy.name: RewardPerReactionPolicy,
}


def available_child_policies() -> List[str]:
    return list(_POLICIES)


def make_child_policy(name: str, **kwargs) -> ChildSelectionPolicy:
    """Build a policy by name. ``smart_frag``/``smart_dyn`` accept ``beta`` + ``utilities``;
    ``marginal``/``ratio`` accept ``beta``; ``reward``/``free_frag`` ignore kwargs. Unknown name →
    ``KeyError`` with the available set (fail fast in the driver)."""
    try:
        cls = _POLICIES[name]
    except KeyError:
        raise KeyError(f"unknown child policy {name!r}; available: {available_child_policies()}")
    if cls in (SmartFragChildPolicy, SmartFragDynamicPolicy):
        return cls(beta=kwargs.get("beta", 0.0), utilities=kwargs.get("utilities"))
    if cls in (MarginalReactionPolicy, RewardPerReactionPolicy):
        return cls(beta=kwargs.get("beta", 0.0))
    return cls()
