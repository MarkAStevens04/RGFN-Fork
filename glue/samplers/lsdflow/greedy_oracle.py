"""Adaptive cost-benefit greedy over the enumerated hub pool — the *selection* oracle arm.

Why this exists
---------------
:class:`~glue.samplers.lsdflow.campaign.HubBatchingStrategy` walks a **static** hub ranking: hubs are
scored once by the recovered flow ``F_hat(h)``, sorted, and walked in that order (``self.hubs =
list(enumerated_hubs)  # already in rank order``). The order is never revisited, so a hub whose every
child duplicates something an earlier hub already contributed still gets its scaffold built and its
children scored.

That leaves one question unanswered: **how far below the achievable ceiling is the flow ordering?**
The hub-ordering ablation (Logs/053) establishes a *floor* — flow beats a random ordering by 1.53x and
reversing it breaks the method outright — but nothing measures the gap upward. This module supplies
the reference point: the classic **cost-benefit greedy**, which at every step re-scores every
remaining hub by its *true marginal* value given the library built so far

    score(h) = (new modes h would add) / (marginal reactions h would cost)

and commits to the argmax. It is the same shape of instrument as the SPARROW MILP audit of the cost
model (Logs/049 / Logs/056): an independent, stronger procedure run on identical inputs, so the
shipped heuristic's gap to it is *measured* rather than argued.

What it is NOT
--------------
**This is not the optimum and must never be reported as one.** Two reasons, both load-bearing:

1. The greedy is an *approximation*. Budgeted maximum coverage admits ``(1 - 1/sqrt(e))`` for the
   cost-benefit greedy alone, and ``(1 - 1/e)`` only with Khuller-Moss-Naor's partial enumeration of
   starting triples, which we do not run.
2. **The objective is not exactly submodular.** A "mode" is decided by a *greedy sphere-exclusion*
   filter (:class:`~glue.samplers.lsdflow.mode_select.DiverseThresholdModeSelector`), so which
   molecules count as distinct depends on the order they were offered in. Coverage over an
   order-dependent set system is not a matroid/coverage structure, so the textbook guarantee does not
   transfer. We therefore make **no formal claim** and use the greedy purely as a measured reference.

It is also *not free*: the greedy must have every hub in the pool enumerated **and scored** before it
can choose its first hub, whereas the flow walk reads the trained model once and enumerates only what
it walks. That asymmetry is reported (``walk_reward_gen_calls`` vs ``pool_reward_gen_calls``) rather
than hidden — it is the point of the comparison, not a caveat to it.

Bit-comparability
-----------------
Every cost decision is delegated to the *same* module-level helpers the shipped strategy uses
(:func:`~glue.samplers.lsdflow.campaign.shallow_couplings`, ``_charge_promoted``, ``_budget_hit``,
``_finalize``), and acceptance is delegated to a **clone of the live selector object**, so the two
arms cannot drift on accounting or on the mode definition. ``order="static"`` runs this module's own
walk machinery over the ranking as given, which must reproduce ``HubBatchingStrategy`` exactly — that
equivalence is the regression test (``run_greedy_oracle.py --regress``) and it is what validates the
fast paths below.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from glue.samplers.lsdflow.campaign import (
    Budget,
    CampaignPoint,
    CampaignResult,
    EnumeratedHub,
    _budget_hit,
    _charge_promoted,
    _finalize,
    shallow_couplings,
)
from glue.samplers.lsdflow.child_select import ChildSelectionPolicy, RewardChildPolicy
from glue.samplers.lsdflow.mode_select import (
    DiverseThresholdModeSelector,
    ModeSelector,
    RewardOnlyModeSelector,
    ecfp,
    passes_reward_gate,
)

try:  # RDKit is present in-env; guarded to match mode_select.py
    from rdkit import DataStructs
except Exception:  # pragma: no cover
    DataStructs = None


# Mutable state each known selector carries, so a probe can be given an independent copy without
# deep-copying RDKit fingerprint objects (which would dominate the runtime). Fingerprints are only
# ever appended, never mutated, so a shallow container copy is exact. A selector class absent from
# this map is refused rather than silently cloned wrong.
_SELECTOR_STATE: Dict[type, Tuple[Tuple[str, Callable], ...]] = {
    DiverseThresholdModeSelector: (("_fps", list),),
    RewardOnlyModeSelector: (("_seen", set),),
}


def clone_selector(selector: ModeSelector) -> ModeSelector:
    """An independent selector carrying the same accepted-state, for non-committing probes.

    Uses the selector's own class and its own ``accept`` code path, so a probe's accept/reject
    decisions are identical to the committed walk's by construction rather than by reimplementation.
    """
    spec = _SELECTOR_STATE.get(type(selector))
    if spec is None:
        raise TypeError(
            f"greedy_oracle cannot clone selector {type(selector).__name__}: its mutable state is "
            f"not declared in _SELECTOR_STATE. Add it there (and confirm nothing else mutates) "
            f"before using this selector with the greedy arm."
        )
    probe = copy.copy(selector)
    for attr, ctor in spec:
        setattr(probe, attr, ctor(getattr(selector, attr)))
    return probe


def permanently_blocked(selector: ModeSelector, children: Sequence) -> List:
    """Children the **committed** library already rejects, and therefore rejects forever.

    The accepted set is append-only, so a child rejected by the current library can never become
    acceptable later; dropping it is exact, not heuristic. This is a fast path (one bulk similarity
    call per child, no cloning) for the selector we ship. For any other selector it returns the input
    unchanged — pruning is only an optimisation, so declining to prune is always safe.

    The fast path duplicates :meth:`DiverseThresholdModeSelector.accept`'s rejection conditions; the
    ``--regress`` equivalence against ``HubBatchingStrategy`` is what holds the two in agreement.
    """
    if not isinstance(selector, DiverseThresholdModeSelector) or DataStructs is None:
        return list(children)
    fps = selector._fps
    keep = []
    for c in children:
        fp = ecfp(c.smiles)
        if fp is None:
            continue  # unparseable: rejected by every selector, forever
        if fps and max(DataStructs.BulkTanimotoSimilarity(fp, fps)) > selector.similarity:
            continue
        keep.append(c)
    return keep


@dataclass
class _Probe:
    """What committing to a hub would buy, evaluated against the current library state."""

    n_modes: int
    n_reactions: int
    children: List = field(default_factory=list)  # accepted children, in acceptance order

    @property
    def score(self) -> float:
        """Modes per reaction — the cost-benefit ratio. Zero-gain hubs score 0 and are never taken."""
        if self.n_modes <= 0:
            return 0.0
        return self.n_modes / max(self.n_reactions, 1)


class AdaptiveGreedyHubStrategy:
    """Cost-benefit greedy hub selection over an already-enumerated pool.

    Parameters mirror :class:`~glue.samplers.lsdflow.campaign.HubBatchingStrategy` exactly, plus:

    ``order``
        ``"greedy"`` (default) re-scores every remaining hub at every step and takes the argmax
        marginal modes-per-reaction. ``"static"`` walks ``enumerated_hubs`` in the order given, using
        this module's own machinery — the regression path that must reproduce the shipped strategy.

    ``allow_revisit``
        Whether a hub already entered may be entered again later. Default ``True``: after other hubs
        build promoted fragments, a previously exhausted hub can offer newly-free children, and the
        oracle arm should be allowed to exploit that. Re-entry costs no reward-gen calls (its children
        are already scored) and no scaffold rebuild (``built_hubs`` suppresses it), so it can only
        strengthen the reference point.
    """

    def __init__(
        self,
        enumerated_hubs: Sequence[EnumeratedHub],
        cost_table=None,
        *,
        target: str = "unknown",
        reward_threshold: Optional[float] = None,
        similarity: float = 0.5,
        higher_is_better: bool = True,
        mode_selector_factory: Optional[Callable[[], ModeSelector]] = None,
        child_policy: Optional[ChildSelectionPolicy] = None,
        prebuilt_fragments: Optional[Set[str]] = None,
        order: str = "greedy",
        allow_revisit: bool = True,
    ):
        if order not in ("greedy", "static"):
            raise ValueError(f"order must be 'greedy' or 'static', got {order!r}")
        self.hubs = list(enumerated_hubs)
        self.cost_table = cost_table
        self.target = target
        self.reward_threshold = reward_threshold
        self.higher_is_better = higher_is_better
        self._make_selector = mode_selector_factory or (
            lambda: DiverseThresholdModeSelector(reward_threshold, similarity, higher_is_better)
        )
        self.child_policy = child_policy or RewardChildPolicy()
        self.prebuilt_fragments = set(prebuilt_fragments or ())
        self.order = order
        self.allow_revisit = allow_revisit

        # Pre-gate once. A child failing the reward gate can never be accepted, and dropping it here
        # changes nothing downstream (the selector would reject it, and a rejected child leaves no
        # trace in the selector's state) while removing the great majority of children before any
        # fingerprint work is done.
        self._live: List[List] = [
            [
                c
                for c in h.children
                if passes_reward_gate(c.reward, reward_threshold, higher_is_better)
            ]
            for h in self.hubs
        ]
        self._pool_calls = sum(len(h.children) for h in self.hubs)

    # ------------------------------------------------------------------ probing
    def _available(self, hub: EnumeratedHub, built_promoted: Set[str]) -> Set[str]:
        """Fragments free to this hub's children: everything already built, plus the hub's own
        scaffold fragments (building the hub builds them). Mirrors the shipped walk exactly."""
        available = set(built_promoted)
        if self.cost_table is not None and hub.promoted:
            available |= self.cost_table.closure(hub.promoted)
        return available

    def _probe(self, idx: int, built_promoted: Set[str], built_hubs: Set[str], selector) -> _Probe:
        """What hub ``idx`` would add right now, committing nothing.

        Prunes children the committed library already blocks (exact — see
        :func:`permanently_blocked`), then replays the shipped inner loop against a selector clone.
        """
        hub = self.hubs[idx]
        self._live[idx] = permanently_blocked(selector, self._live[idx])
        if not self._live[idx]:
            return _Probe(0, 0)
        children = self.child_policy.order(
            self._live[idx],
            higher_is_better=self.higher_is_better,
            cost_table=self.cost_table,
            available=self._available(hub, built_promoted),
        )
        if not children:
            return _Probe(0, 0)

        probe_sel = clone_selector(selector)
        local_built = set(built_promoted)
        hub_built = hub.hub_key in built_hubs
        rx = 0
        taken: List = []
        for child in children:
            if not probe_sel.accept(child.smiles, child.reward):
                continue
            if not hub_built:
                hub_built = True
                rx += shallow_couplings(hub.depth, hub.promoted, self.cost_table)
                rx += _charge_promoted(hub.promoted, local_built, self.cost_table)
            rx += 1  # the diversifying reaction
            rx += _charge_promoted(child.added_promoted, local_built, self.cost_table)
            taken.append(child)
        return _Probe(len(taken), rx, taken)

    def _pick_greedy(self, remaining, built_promoted, built_hubs, selector):
        """Argmax marginal modes-per-reaction over ``remaining``. Returns ``(idx, probe, n_probes)``.

        Tie-break is deterministic: higher ratio, then more modes, then earlier in the pool's own
        (flow-ranked) order — so a tie can never be broken in the greedy's favour by accident.
        """
        best_idx: Optional[int] = None
        best_key = None
        best_probe: Optional[_Probe] = None
        n = 0
        for idx in remaining:
            probe = self._probe(idx, built_promoted, built_hubs, selector)
            n += 1
            if probe.n_modes == 0:
                continue
            key = (probe.score, probe.n_modes, -idx)
            if best_key is None or key > best_key:
                best_idx, best_key, best_probe = idx, key, probe
        return best_idx, best_probe, n

    # ------------------------------------------------------------------ the walk
    def run(self, budget: Budget = None) -> CampaignResult:
        selector = self._make_selector()
        built_promoted: Set[str] = set()
        built_hubs: Set[str] = set()
        scored_hubs: Set[int] = set()  # hubs whose children were scored (the oracle cost)
        result = CampaignResult(
            strategy="greedy_oracle" if self.order == "greedy" else "greedy_oracle_static",
            target=self.target,
        )
        cum_rx = 0
        cum_calls = 0
        n_probes = 0

        if self.prebuilt_fragments:
            cum_rx += _charge_promoted(self.prebuilt_fragments, built_promoted, self.cost_table)

        remaining = list(range(len(self.hubs)))
        stopped = False

        while remaining and not stopped:
            if self.order == "static":
                idx = remaining.pop(0)
                probe = self._probe(idx, built_promoted, built_hubs, selector)
                n_probes += 1
            else:
                idx, probe, n = self._pick_greedy(remaining, built_promoted, built_hubs, selector)
                n_probes += n
                if idx is None:
                    break  # nothing left can add a mode
                if not self.allow_revisit:
                    remaining.remove(idx)

            hub = self.hubs[idx]
            # Scoring a hub's children is the oracle cost, charged the first time it is entered.
            # The static walk enters every hub in order (matching the shipped strategy, which charges
            # a hub's children even when it yields nothing); the greedy only enters what it takes.
            if probe.n_modes or self.order == "static":
                if idx not in scored_hubs:
                    scored_hubs.add(idx)
                    cum_calls += len(hub.children)
                    result.walked_hub_ids.append(hub.hub_input or hub.hub_key)

            hub_coup = shallow_couplings(hub.depth, hub.promoted, self.cost_table)
            committed = 0
            for child in probe.children:
                if not selector.accept(child.smiles, child.reward):
                    continue  # unreachable (the probe used an identical clone); kept as a guard
                rx = 0
                if hub.hub_key not in built_hubs:
                    built_hubs.add(hub.hub_key)
                    rx += hub_coup + _charge_promoted(hub.promoted, built_promoted, self.cost_table)
                rx += 1
                rx += _charge_promoted(child.added_promoted, built_promoted, self.cost_table)
                cum_rx += rx
                committed += 1
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
            # Progress guarantee: a hub that committed nothing is dropped, so the loop cannot spin on
            # it even if a probe and its commit ever disagreed.
            if self.order == "greedy" and committed == 0 and idx in remaining:
                remaining.remove(idx)

        prev = 0
        for p in result.accepted:
            object.__setattr__(p, "reward_gen_calls_added", p.cum_reward_gen_calls - prev)
            prev = p.cum_reward_gen_calls

        out = _finalize(result, built_promoted, built_hubs, self.higher_is_better)
        out.meta.update(
            {
                "order": self.order,
                "allow_revisit": self.allow_revisit,
                "n_probes": n_probes,
                "n_hubs_scored": len(scored_hubs),
                # The two honest oracle-cost readings. ``walk`` is directly comparable to
                # hub-batching's ``total_reward_gen_calls`` (children of hubs actually entered);
                # ``pool`` is what the greedy genuinely requires, since it cannot rank a hub it has
                # not scored.
                "walk_reward_gen_calls": cum_calls,
                "pool_reward_gen_calls": self._pool_calls,
            }
        )
        if self.order == "greedy":
            out.total_reward_gen_calls = self._pool_calls
        return out
