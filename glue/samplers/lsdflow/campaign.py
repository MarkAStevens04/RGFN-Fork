"""Budget-greedy synthesis-campaign simulator: hub-batching vs best-candidate (Logs/028).

Two **independent, swappable** strategies for building a diverse library of hits (modes) under a
budget, compared on reactions/mode (+ other axes). They are separate classes with **disjoint
required inputs** — a best-candidate campaign never touches hub/enumeration data — but return the
**same** :class:`CampaignResult`, so the future AL loop calls either uniformly
(``batch = [p.smiles for p in strategy.run(budget).accepted]``).

- :class:`BestCandidateStrategy` — rank the generator's sampled candidates by reward, greedily keep
  modes; each pays its **own** full nested synthesis (base assembly + its promoted fragments). No
  hub knowledge; mirrors RGFN's paper "top-k". Reward evaluations are reused from sampling → 0 new.
- :class:`HubBatchingStrategy` — walk pre-ranked, pre-enumerated hubs; build each hub scaffold once
  (charged on first accepted child), diversify into modes (+1 reaction/child). Reward evaluations =
  the enumerated children scored (the second cost axis).

**Cost accounting (the "count once" running built-set).** Reactions = base assembly (a promoted-
fragment attach = 1 step) **plus** the build cost of every *new* promoted fragment introduced (its
logged route, expanded through nested sub-fragments via the recipe closure), each charged exactly
once across the campaign (SCENT's dynamic-library fragments are reusable stock). Hub scaffolds are
shared (built once) only in hub-batching — the one structural advantage it has over best-candidate.

Both budget dimensions read off the same run: Case 1 ``("reactions", N)`` → stop at N reactions,
report modes; Case 2 ``("modes", N)`` → stop at N modes, report reactions. ``budget=None`` runs the
whole pool (the full modes-vs-reactions curve). NO active-learning loop here — that consumes this.
"""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set, Tuple

from glue.samplers.lsdflow.mode_select import DiverseThresholdModeSelector, ModeSelector

Budget = Optional[Tuple[str, float]]  # ("reactions", N) | ("modes", N) | None (full curve)


# ----------------------------------------------------------------- inputs (per strategy)
@dataclass(frozen=True)
class Candidate:
    """A sampled terminal molecule (best-candidate input; also the hub-discovery pool upstream)."""

    smiles: str
    reward: float
    num_reactions: int
    promoted: Tuple[str, ...] = ()  # promoted fragments in its synthesis


@dataclass(frozen=True)
class EnumChild:
    """One enumerated one-reaction child of a hub (hub-batching input)."""

    smiles: str
    reward: float
    added_promoted: Tuple[str, ...] = ()  # promoted fragments attached in the final reaction


@dataclass
class EnumeratedHub:
    """A pre-ranked, pre-enumerated hub (hub-batching input). ``children`` already scored."""

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
    source_hub: Optional[str] = None  # the hub it came from (None for best-candidate)


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
    distinct_hubs_used: int = 0  # 0 for best-candidate
    n_scaffolds: int = 0
    best_reward: float = float("nan")
    median_reward: float = float("nan")
    stop_reason: str = "pool_exhausted"  # "reactions" | "modes" | "pool_exhausted"
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------- cost helper (shared)
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


# ----------------------------------------------------------------- strategies
class CampaignStrategy(ABC):
    """Swappable: ``run(budget) -> CampaignResult``. Constructors differ (disjoint inputs)."""

    @abstractmethod
    def run(self, budget: Budget = None) -> CampaignResult:
        ...


class BestCandidateStrategy(CampaignStrategy):
    """Top-reward modes from the sampled pool; each built independently (promoted fragments shared
    as reusable stock). Needs ONLY the candidate pool — never any hub/enumeration data."""

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
    ):
        self.candidates = list(candidates)
        self.cost_table = cost_table
        self.target = target
        self.higher_is_better = higher_is_better
        self._make_selector = mode_selector_factory or (
            lambda: DiverseThresholdModeSelector(reward_threshold, similarity, higher_is_better)
        )

    def run(self, budget: Budget = None) -> CampaignResult:
        selector = self._make_selector()
        built_promoted: Set[str] = set()
        result = CampaignResult(strategy="best_candidate", target=self.target)
        cum_rx = 0
        order = sorted(
            self.candidates,
            key=lambda c: (c.reward if c.reward == c.reward else float("-inf")),
            reverse=self.higher_is_better,
        )
        for cand in order:
            if not selector.accept(cand.smiles, cand.reward):
                continue
            rx = int(cand.num_reactions) + _charge_promoted(
                cand.promoted, built_promoted, self.cost_table
            )
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
                    source_hub=None,
                )
            )
            if _budget_hit(budget, cum_rx, step):
                result.stop_reason = budget[0]
                break
        return _finalize(result, built_promoted, set())


class HubBatchingStrategy(CampaignStrategy):
    """Walk pre-ranked, pre-enumerated hubs; build each scaffold once (charged on first accepted
    child), diversify into modes. Needs ONLY the enumerated hubs — never the raw candidate pool."""

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
    ):
        self.hubs = list(enumerated_hubs)  # already in rank order (hub strategy applied upstream)
        self.cost_table = cost_table
        self.target = target
        self.higher_is_better = higher_is_better
        self._make_selector = mode_selector_factory or (
            lambda: DiverseThresholdModeSelector(reward_threshold, similarity, higher_is_better)
        )

    def run(self, budget: Budget = None) -> CampaignResult:
        selector = self._make_selector()
        built_promoted: Set[str] = set()
        built_hubs: Set[str] = set()
        result = CampaignResult(strategy="hub_batching", target=self.target)
        cum_rx = 0
        cum_calls = 0
        stopped = False
        for hub in self.hubs:
            if stopped:
                break
            # Enumerating + scoring this hub's children is the reward-gen (oracle) cost.
            cum_calls += len(hub.children)
            children = sorted(
                hub.children,
                key=lambda c: (c.reward if c.reward == c.reward else float("-inf")),
                reverse=self.higher_is_better,
            )
            for child in children:
                if not selector.accept(child.smiles, child.reward):
                    continue
                rx = 0
                if hub.hub_key not in built_hubs:  # build the shared scaffold once, lazily
                    built_hubs.add(hub.hub_key)
                    rx += int(hub.depth) + _charge_promoted(
                        hub.promoted, built_promoted, self.cost_table
                    )
                rx += 1  # the final diversifying reaction
                rx += _charge_promoted(child.added_promoted, built_promoted, self.cost_table)
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
                if _budget_hit(budget, cum_rx, step):
                    result.stop_reason = budget[0]
                    stopped = True
                    break
        # backfill reward_gen_calls_added (per-point delta) for readability
        prev = 0
        for p in result.accepted:
            object.__setattr__(p, "reward_gen_calls_added", p.cum_reward_gen_calls - prev)
            prev = p.cum_reward_gen_calls
        return _finalize(result, built_promoted, built_hubs)
