"""Budget-greedy synthesis-campaign simulator: hub-batching vs best-candidate (Logs/028, Logs/033).

Two **independent, swappable** strategies for building a diverse library of hits (modes) under a
budget, compared on reactions/mode (+ other axes). They are separate classes that return the **same**
:class:`CampaignResult`, so the future AL loop calls either uniformly
(``batch = [p.smiles for p in strategy.run(budget).accepted]``).

- :class:`BestCandidateStrategy` — rank the generator's sampled candidates by reward, greedily keep
  modes. Mirrors RGFN's paper "top-k". Reward evaluations are reused from sampling → 0 new.
- :class:`HubBatchingStrategy` — walk pre-ranked, pre-enumerated hubs; build each scaffold once,
  diversify into modes. Reward evaluations = the enumerated children scored (the second cost axis).

**Cost accounting — ONE count-once model for both strategies (Logs/033).** A library's reaction
cost decomposes into two count-once parts:

  1. **Assembly couplings** — the reaction steps that *join* pre-built fragments. Each molecule's
     shallow assembly depth is ``couplings = num_reactions − Σ(nested build cost of every promoted
     fragment it attaches)`` (SCENT's ``num_reactions`` is FULLY NESTED — it already includes
     building each dynamic-library fragment; see ``reaction_env.py``). A **shared hub** (a scaffold
     several modes descend from) has its couplings charged **once**, then each mode pays only its
     marginal couplings (≈1 for an immediate-parent hub).
  2. **Promoted-fragment builds** — each distinct dynamic-library fragment is built + charged
     **exactly once** across the whole campaign (closure under recipes; reusable stock).

Both strategies amortize both parts identically; the ONLY difference is the SELECTION. Hub-batching
shares hubs **by design** (enumerated children of one scaffold). Best-candidate shares hubs
**accidentally** — when its top-reward picks happen to descend from a common parent hub — via a
swappable :class:`HubAssignmentPolicy` (default :class:`MostSharedAssignment`: assign each mode to
the parent hub most reused among the accepted modes → a greedy upper bound on realistic savings).
Hub sharing here is **immediate-parent only** (``records.csv`` ``hub_key``); deeper-backbone sharing
awaits the sample re-run's per-molecule ``routes.json``.

Both budget dimensions read off the same run: Case 1 ``("reactions", N)`` → stop at N reactions,
report modes; Case 2 ``("modes", N)`` → stop at N modes, report reactions. ``budget=None`` runs the
whole pool (the full modes-vs-reactions curve). NO active-learning loop here — that consumes this.
"""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from glue.samplers.lsdflow.child_select import ChildSelectionPolicy, RewardChildPolicy
from glue.samplers.lsdflow.mode_select import DiverseThresholdModeSelector, ModeSelector

Budget = Optional[Tuple[str, float]]  # ("reactions", N) | ("modes", N) | None (full curve)


# ----------------------------------------------------------------- inputs (per strategy)
@dataclass(frozen=True)
class Candidate:
    """A sampled terminal molecule (best-candidate input; also the hub-discovery pool upstream)."""

    smiles: str
    reward: float
    num_reactions: int  # SCENT's FULLY-NESTED reaction count (includes promoted-fragment builds)
    promoted: Tuple[str, ...] = ()  # promoted fragments directly attached in its synthesis
    parents: Tuple[str, ...] = ()  # observed immediate-parent hub keys (records.csv hub_key)


@dataclass(frozen=True)
class EnumChild:
    """One enumerated one-reaction child of a hub (hub-batching input)."""

    smiles: str
    reward: float
    added_promoted: Tuple[str, ...] = ()  # promoted fragments attached in the final reaction


@dataclass
class EnumeratedHub:
    """A pre-ranked, pre-enumerated hub (hub-batching input). ``children`` already scored.

    ``depth`` is the hub's FULLY-NESTED ``num_reactions`` (SCENT); its shallow assembly couplings
    (what we actually charge, once) are recovered via :func:`shallow_couplings` with ``promoted``.
    """

    hub_key: str
    depth: int
    promoted: Tuple[str, ...]  # promoted fragments in the hub scaffold itself
    children: List[EnumChild]
    # U(h) = Var_i[log F_hat(h;x_i)] over the enumerated children (§2 flow-matching residual).
    # Carried for later use (an epistemic-uncertainty / acquisition signal in the AL phase);
    # NOT used by the campaign selection yet — just kept easy to extract.
    uncertainty: Optional[float] = None
    n_effective: int = 0


# ----------------------------------------------------------------- output (shared)
@dataclass(frozen=True)
class CampaignPoint:
    """One accepted mode. The ordered list of these IS the modes-vs-reactions curve."""

    step: int  # nth mode accepted (== cum_modes)
    smiles: str
    reward: float
    reactions_added: int  # NEW reactions this mode cost (incremental)
    cum_reactions: int
    cum_modes: int
    reward_gen_calls_added: int
    cum_reward_gen_calls: int
    source_hub: Optional[str] = None  # the hub it was charged against (shared scaffold), else None


@dataclass
class CampaignResult:
    """Identical schema for both strategies (the interop contract; AL loop reads ``accepted``)."""

    strategy: str
    target: str
    accepted: List[CampaignPoint] = field(default_factory=list)
    total_reactions: int = 0
    total_modes: int = 0
    total_reward_gen_calls: int = 0
    distinct_promoted_fragments: int = 0
    distinct_hubs_used: int = (
        0  # shared hubs built (accidental for best-candidate, by design for HB)
    )
    n_scaffolds: int = 0
    best_reward: float = float("nan")
    median_reward: float = float("nan")
    stop_reason: str = "pool_exhausted"  # "reactions" | "modes" | "pool_exhausted"
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------- cost helpers (shared)
def shallow_couplings(num_reactions: int, promoted: Sequence[str], cost_table) -> int:
    """A molecule's **assembly couplings** — the reaction steps that join pre-built fragments.

    SCENT's ``num_reactions`` is fully nested (each attached promoted fragment contributes its OWN
    nested build cost, +1 per coupling; see ``reaction_env.py``). Subtracting the nested build cost
    of every directly-attached promoted fragment leaves the shallow assembly depth — the couplings
    we charge (the fragment builds are charged separately, once each). No cost table (RGFN / no
    dynamic library) → ``num_reactions`` (nothing to subtract). Clamped ≥ 0.
    """
    if cost_table is None or not promoted:
        return int(num_reactions)
    sub = 0
    for f in promoted:
        sub += int(cost_table.shared_build_cost([f])[0])  # f's full nested build cost
    return max(0, int(num_reactions) - sub)


def _charge_promoted(promoted: Sequence[str], built: Set[str], cost_table) -> int:
    """Reactions to build the *new* promoted fragments in ``promoted`` (closed under recipes),
    charging each exactly once; marks them built. 0 if no cost table or nothing new."""
    if cost_table is None or not promoted:
        return 0
    closure = cost_table.closure(promoted)
    added = 0
    for f in closure:
        if f not in built:
            built.add(f)
            added += int(cost_table.unit_reactions(f))
    return added


def _budget_hit(budget: Budget, cum_reactions: int, cum_modes: int) -> bool:
    if budget is None:
        return False
    kind, limit = budget
    return (kind == "reactions" and cum_reactions >= limit) or (
        kind == "modes" and cum_modes >= limit
    )


def _finalize(
    result: CampaignResult, built_promoted: Set[str], built_hubs: Set[str]
) -> CampaignResult:
    rewards = [p.reward for p in result.accepted if p.reward == p.reward]
    result.total_modes = len(result.accepted)
    result.total_reactions = result.accepted[-1].cum_reactions if result.accepted else 0
    result.total_reward_gen_calls = (
        result.accepted[-1].cum_reward_gen_calls if result.accepted else 0
    )
    result.distinct_promoted_fragments = len(built_promoted)
    result.distinct_hubs_used = len(built_hubs)
    result.best_reward = max(rewards) if rewards else float("nan")
    result.median_reward = statistics.median(rewards) if rewards else float("nan")
    result.n_scaffolds = _count_scaffolds([p.smiles for p in result.accepted])
    return result


def _count_scaffolds(smiles: Sequence[str]) -> int:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return len(set(smiles))
    scaffolds = set()
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is None:
            continue
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except Exception:
            continue
    return len(scaffolds)


# ----------------------------------------------------------------- hub-assignment policy (best-candidate)
class HubAssignmentPolicy(ABC):
    """Maps each accepted best-candidate terminal → the parent hub it is charged against (or None).

    Swappable so the "accidental hub-batching" credit can use a different rule (most-shared, cheapest,
    canonical, none...) without touching the cost accounting. ``hub_cost(h)`` returns hub ``h``'s
    assembly couplings (a tie-break signal); ``parents[t]`` is terminal ``t``'s observed parent hubs.
    """

    @abstractmethod
    def assign(
        self,
        accepted: Sequence[str],
        parents: Dict[str, Sequence[str]],
        hub_cost: Callable[[str], int],
    ) -> Dict[str, Optional[str]]:
        ...


class MostSharedAssignment(HubAssignmentPolicy):
    """Assign each terminal to the parent hub most **reused** among the accepted modes (greedy upper
    bound on realistic accidental savings). Frequency = #accepted terminals that could use the hub;
    ties → cheaper hub, then lexicographic (deterministic)."""

    def assign(self, accepted, parents, hub_cost):
        freq: Counter = Counter()
        for t in accepted:
            for h in dict.fromkeys(parents.get(t, ())):  # dedup a terminal's own repeated parents
                freq[h] += 1
        out: Dict[str, Optional[str]] = {}
        for t in accepted:
            cands = list(dict.fromkeys(parents.get(t, ())))
            out[t] = max(cands, key=lambda h: (freq[h], -hub_cost(h), h)) if cands else None
        return out


class NoHubSharing(HubAssignmentPolicy):
    """Charge every terminal its own full assembly (no accidental batching) — the pre-Logs/033
    best-candidate behaviour, but with the fragment double-count fixed."""

    def assign(self, accepted, parents, hub_cost):
        return {t: None for t in accepted}


# ----------------------------------------------------------------- strategies
class CampaignStrategy(ABC):
    """Swappable: ``run(budget) -> CampaignResult``. Constructors differ (disjoint inputs)."""

    @abstractmethod
    def run(self, budget: Budget = None) -> CampaignResult:
        ...


class BestCandidateStrategy(CampaignStrategy):
    """Top-reward modes from the sampled pool. Cost = assembly couplings (shared hubs charged once
    via ``assignment_policy``) + distinct promoted-fragment builds (once). ``hub_compositions`` maps
    a parent hub_key → its ``{num_reactions, promoted}`` (from ``compositions.json``) so shared hubs
    can be costed; omit it (or pass :class:`NoHubSharing`) to charge each mode independently."""

    def __init__(
        self,
        candidates: Sequence[Candidate],
        cost_table=None,
        *,
        target: str = "unknown",
        reward_threshold: Optional[float] = None,
        similarity: float = 0.5,  # campaign/AL default cutoff (Logs/029); benchmark mode-def stays 0.7
        higher_is_better: bool = True,
        mode_selector_factory: Optional[Callable[[], ModeSelector]] = None,
        hub_compositions: Optional[Dict[str, dict]] = None,
        assignment_policy: Optional[HubAssignmentPolicy] = None,
        max_modes: int = 5000,  # safety cap for Pass-1 selection when no mode budget is given
    ):
        self.candidates = list(candidates)
        self.cost_table = cost_table
        self.target = target
        self.higher_is_better = higher_is_better
        self._make_selector = mode_selector_factory or (
            lambda: DiverseThresholdModeSelector(reward_threshold, similarity, higher_is_better)
        )
        self.hub_compositions = hub_compositions or {}
        self.assignment_policy = assignment_policy or (
            MostSharedAssignment() if self.hub_compositions else NoHubSharing()
        )
        self.max_modes = max_modes
        self._hub_coup_cache: Dict[str, int] = {}

    def _hub_couplings(self, h: str) -> int:
        c = self._hub_coup_cache.get(h)
        if c is None:
            comp = self.hub_compositions.get(h)
            c = (
                shallow_couplings(comp["num_reactions"], comp.get("promoted", ()), self.cost_table)
                if comp
                else 0
            )
            self._hub_coup_cache[h] = c
        return c

    def run(self, budget: Budget = None) -> CampaignResult:
        selector = self._make_selector()
        # Pass 1: reward-first diversity selection -> the accepted set (independent of cost).
        mode_cap = int(budget[1]) if (budget and budget[0] == "modes") else self.max_modes
        order = sorted(
            self.candidates,
            key=lambda c: (c.reward if c.reward == c.reward else float("-inf")),
            reverse=self.higher_is_better,
        )
        accepted: List[Candidate] = []
        for cand in order:
            if not selector.accept(cand.smiles, cand.reward):
                continue
            accepted.append(cand)
            if mode_cap is not None and len(accepted) >= mode_cap:
                break

        # Pass 2: assign each accepted mode to a shared parent hub (or None) — the amortization plan.
        # A parent hub is only a VALID prefix if it is genuinely cheaper than the mode's own min-cost
        # route (couplings(h) < couplings(t)); a mode's recorded parents come from different
        # trajectories, so a popular-but-longer-route parent would otherwise inflate its marginal.
        t_coup = {
            c.smiles: shallow_couplings(c.num_reactions, c.promoted, self.cost_table)
            for c in accepted
        }
        valid_parents = {
            c.smiles: tuple(
                h
                for h in c.parents
                if h in self.hub_compositions and self._hub_couplings(h) < t_coup[c.smiles]
            )
            for c in accepted
        }
        assignment = self.assignment_policy.assign(
            [c.smiles for c in accepted], valid_parents, self._hub_couplings
        )

        # Pass 3: walk the accepted order, charging count-once cost; hubs shared across modes.
        # reactions = Σ(distinct shared-hub couplings, once) + Σ(marginal couplings per mode)
        #           + Σ(distinct promoted-fragment builds, once).  Fragments are charged via each
        # mode's FULL promoted set (count-once), so a shared hub's fragments are never double-charged.
        built_promoted: Set[str] = set()
        built_hubs: Set[str] = set()
        result = CampaignResult(strategy="best_candidate", target=self.target)
        cum_rx = 0
        for cand in accepted:
            hub = assignment.get(cand.smiles)
            rx = 0
            h_coup = 0
            if hub is not None:
                h_coup = self._hub_couplings(hub)
                if hub not in built_hubs:  # build the shared scaffold prefix once
                    built_hubs.add(hub)
                    rx += h_coup
            rx += t_coup[cand.smiles] - h_coup  # marginal assembly (≥1 by the validity filter)
            rx += _charge_promoted(cand.promoted, built_promoted, self.cost_table)
            cum_rx += rx
            step = len(result.accepted) + 1
            result.accepted.append(
                CampaignPoint(
                    step=step,
                    smiles=cand.smiles,
                    reward=cand.reward,
                    reactions_added=rx,
                    cum_reactions=cum_rx,
                    cum_modes=step,
                    reward_gen_calls_added=0,
                    cum_reward_gen_calls=0,
                    source_hub=hub,
                )
            )
            if _budget_hit(budget, cum_rx, step):
                result.stop_reason = budget[0]
                break
        return _finalize(result, built_promoted, built_hubs)


class HubBatchingStrategy(CampaignStrategy):
    """Walk pre-ranked, pre-enumerated hubs; build each scaffold once (assembly couplings charged on
    first accepted child), diversify into modes. Needs ONLY the enumerated hubs.

    ``child_policy`` (Logs/036) decides which of a hub's children get offered to the mode selector
    and in what order — default :class:`RewardChildPolicy` (reward-first, keep all) is byte-identical
    to the historical behaviour; swap in ``FreeFragChildPolicy`` / ``SmartFragChildPolicy`` to make
    selection fragment-aware. It is orthogonal to the cost accounting, which is unchanged.
    """

    def __init__(
        self,
        enumerated_hubs: Sequence[EnumeratedHub],
        cost_table=None,
        *,
        target: str = "unknown",
        reward_threshold: Optional[float] = None,
        similarity: float = 0.5,  # campaign/AL default cutoff (Logs/029); benchmark mode-def stays 0.7
        higher_is_better: bool = True,
        mode_selector_factory: Optional[Callable[[], ModeSelector]] = None,
        child_policy: Optional[ChildSelectionPolicy] = None,
    ):
        self.hubs = list(enumerated_hubs)  # already in rank order (hub strategy applied upstream)
        self.cost_table = cost_table
        self.target = target
        self.higher_is_better = higher_is_better
        self._make_selector = mode_selector_factory or (
            lambda: DiverseThresholdModeSelector(reward_threshold, similarity, higher_is_better)
        )
        self.child_policy = child_policy or RewardChildPolicy()

    def run(self, budget: Budget = None) -> CampaignResult:
        selector = self._make_selector()
        built_promoted: Set[str] = set()
        built_hubs: Set[str] = set()
        result = CampaignResult(strategy="hub_batching", target=self.target)
        cum_rx = 0
        cum_calls = 0
        dynamic = getattr(self.child_policy, "is_dynamic", False)

        def _accept(hub, child, hub_coup) -> Tuple[bool, bool]:
            """Charge count-once cost for an accepted child, record the mode. Returns
            ``(budget_stopped, new_fragment_built)``; the latter tells the dynamic loop to re-rank.
            """
            nonlocal cum_rx
            rx = 0
            new_frag = False
            if hub.hub_key not in built_hubs:  # build the shared scaffold once, lazily
                built_hubs.add(hub.hub_key)
                before = len(built_promoted)
                rx += hub_coup + _charge_promoted(hub.promoted, built_promoted, self.cost_table)
                new_frag = new_frag or len(built_promoted) > before
            rx += 1  # the final diversifying reaction (one coupling)
            before = len(built_promoted)
            rx += _charge_promoted(child.added_promoted, built_promoted, self.cost_table)
            new_frag = new_frag or len(built_promoted) > before
            cum_rx += rx
            step = len(result.accepted) + 1
            result.accepted.append(
                CampaignPoint(
                    step=step,
                    smiles=child.smiles,
                    reward=child.reward,
                    reactions_added=rx,
                    cum_reactions=cum_rx,
                    cum_modes=step,
                    reward_gen_calls_added=0,
                    cum_reward_gen_calls=cum_calls,
                    source_hub=hub.hub_key,
                )
            )
            stopped = _budget_hit(budget, cum_rx, step)
            if stopped:
                result.stop_reason = budget[0]
            return stopped, new_frag

        stopped = False
        for hub in self.hubs:
            if stopped:
                break
            # Enumerating + scoring this hub's children is the reward-gen (oracle) cost.
            cum_calls += len(hub.children)
            hub_coup = shallow_couplings(hub.depth, hub.promoted, self.cost_table)
            # This hub's own scaffold fragments are built with the hub (charged once), so they are
            # free to its children — fold them into the "available" view for scoring/filtering.
            hub_closure = (
                self.cost_table.closure(hub.promoted)
                if (self.cost_table is not None and hub.promoted)
                else set()
            )
            if dynamic:
                stopped = self._process_hub_dynamic(
                    hub, hub_coup, hub_closure, selector, built_promoted, _accept
                )
            else:
                available = set(built_promoted) | hub_closure
                children = self.child_policy.order(
                    hub.children,
                    higher_is_better=self.higher_is_better,
                    cost_table=self.cost_table,
                    available=available,
                )
                for child in children:
                    if not selector.accept(child.smiles, child.reward):
                        continue
                    stopped, _ = _accept(hub, child, hub_coup)
                    if stopped:
                        break
        # backfill reward_gen_calls_added (per-point delta) for readability
        prev = 0
        for p in result.accepted:
            object.__setattr__(p, "reward_gen_calls_added", p.cum_reward_gen_calls - prev)
            prev = p.cum_reward_gen_calls
        return _finalize(result, built_promoted, built_hubs)

    def _process_hub_dynamic(
        self, hub, hub_coup, hub_closure, selector, built_promoted, accept
    ) -> bool:
        """Greedy amortization-aware selection (Logs/037): score children against the *running* built
        set, offer best-first, and re-rank the not-yet-offered tail whenever a new fragment is built
        (its siblings' marginal cost just dropped). Returns True if the budget stopped the campaign.

        Scores treat this hub's own fragments (``hub_closure``) as free — they are built with the hub,
        charged once by ``accept`` — so ``built_promoted | hub_closure`` is the effective built set.
        """

        def _key(child):
            return self.child_policy.score(
                child,
                built_promoted | hub_closure,
                self.cost_table,
                higher_is_better=self.higher_is_better,
            )

        remaining = sorted(hub.children, key=_key, reverse=True)
        i = 0
        while i < len(remaining):
            child = remaining[i]
            i += 1
            if not selector.accept(child.smiles, child.reward):
                continue
            stopped, new_frag = accept(hub, child, hub_coup)
            if stopped:
                return True
            if new_frag and i < len(remaining):  # a fragment just went free → re-rank the tail
                remaining[i:] = sorted(remaining[i:], key=_key, reverse=True)
        return False
