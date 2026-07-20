"""``ucb`` — the active-learning hub-acquisition score (proposal §2 phase-2).

Phase 1 ranks hubs by consistency (``lowest_uncertainty``) or reward/flow. The **active-learning**
phase wants the opposite pull on the uncertainty term: a hub where the flow field is internally
*inconsistent* about downstream reward (high ``U(h)``) is where labelling is most informative. So the
acquisition score is the classic explore/exploit blend

    score(h) = z(reward(h)) + λ · z(U(h))          # higher = acquire first

- **reward(h)** — the *exploitation* term (default: the hub's best-child reward, oriented so higher =
  better regardless of the oracle's sign convention).
- **U(h)** — the *exploration* term: the §2 flow-matching residual
  :func:`glue.metrics.uncertainty.flow_matching_uncertainty` (variance of the per-child ``log F_hat``).
- **z(·)** — the two terms live on different scales (proxy reward ~ single digits; ``U(h)`` = variance
  of log-flow), so each is **z-scored across the round's eligible hubs** before blending. That makes
  ``λ`` a clean, scale-free relative weight (``λ = 1`` ⇒ equal pull) rather than a target-specific fudge
  factor.

**Everything is swappable** (the researcher's explicit requirement): ``reward_fn`` and
``uncertainty_fn`` are chosen by name from small registries here (or passed as callables), ``λ`` is a
config knob, and the whole strategy is a drop-in :class:`HubSelectionStrategy` — add a new blend by
registering a function or subclassing, nothing else changes.

Unlike the other strategies, the score is **not** per-hub (z-scoring needs the whole eligible set), so
this overrides :meth:`rank`; :meth:`score` returns the raw exploitation term as a sensible per-hub
fallback.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import gin

from glue.metrics import uncertainty
from glue.samplers.lsdflow.hub.base import HubSelectionStrategy

_NEG_INF = float("-inf")


# --------------------------------------------------------------------- swappable term registries
def _best_child_reward(hub, dag) -> float:
    """Exploitation term: the hub's single best-reward child, oriented higher = better."""
    rewards = [c.reward for c in hub.children if c.reward == c.reward]
    if not rewards:
        return float("nan")
    return max(rewards) if dag.higher_is_better else -min(rewards)


def _median_flow_reward(hub, dag) -> float:
    """Exploitation term variant: the hub's consensus (median) log-flow. Higher = more flow."""
    return hub.log_flow_consensus()


def _terminating_flow_reward(hub, dag) -> float:
    """Exploitation term variant: total flow terminating one reaction downstream (§ logsumexp)."""
    return hub.log_flow_terminating()


#: name -> exploitation term. Register a new one here (or pass a callable to the strategy).
REWARD_FNS: Dict[str, Callable[[object, object], float]] = {
    "best_child": _best_child_reward,
    "median_flow": _median_flow_reward,
    "terminating_flow": _terminating_flow_reward,
}


def _flow_variance_uncertainty(hub, dag) -> float:
    """Exploration term: ``U(h)`` = variance of the per-child ``log F_hat`` (§2)."""
    return uncertainty.flow_matching_uncertainty(hub.child_log_flows())


#: name -> exploration term. Register a new ``U(h)`` here (or pass a callable to the strategy).
UNCERTAINTY_FNS: Dict[str, Callable[[object, object], float]] = {
    "flow_variance": _flow_variance_uncertainty,
}


def _resolve(spec, registry, kind: str) -> Callable[[object, object], float]:
    if callable(spec):
        return spec
    try:
        return registry[spec]
    except KeyError:
        raise KeyError(
            f"Unknown {kind} {spec!r}. Available: {sorted(registry)} (or pass a callable)."
        ) from None


def _zscore(values: List[float]) -> List[float]:
    """Population z-score, NaN-safe. NaN entries stay NaN; a zero-variance finite set → all 0.0.

    Standardising over the *finite* entries only means a term with no spread (every eligible hub has
    the same reward, or ``U(h)`` is degenerate) contributes nothing to the blend — exactly the "then
    the other term decides" behaviour we want, with ``λ`` still meaningful."""
    finite = [v for v in values if v == v]
    n = len(finite)
    if n == 0:
        return [float("nan")] * len(values)
    mean = sum(finite) / n
    var = sum((v - mean) ** 2 for v in finite) / n
    std = var**0.5
    if std <= 0:
        return [0.0 if v == v else float("nan") for v in values]
    return [((v - mean) / std) if v == v else float("nan") for v in values]


@gin.configurable()
class UcbHubStrategy(HubSelectionStrategy):
    """Rank hubs by ``z(reward(h)) + λ · z(U(h))`` — the AL explore/exploit acquisition (§2)."""

    name = "ucb"

    def __init__(
        self,
        min_children: int = 2,
        min_effective_n: int = 2,
        lam: float = 1.0,
        reward_fn="best_child",
        uncertainty_fn="flow_variance",
    ):
        """
        Args:
            min_children/min_effective_n: gate hubs with too few children / finite flow estimates —
                ``U(h)`` is undefined below 2, so both default to 2.
            lam: λ, the exploration weight on the z-scored ``U(h)`` term (0 ⇒ pure exploitation,
                large ⇒ pure exploration). Config/sweep knob.
            reward_fn: exploitation term — a key in :data:`REWARD_FNS` or a ``(hub, dag) -> float``
                callable (higher = better).
            uncertainty_fn: exploration term — a key in :data:`UNCERTAINTY_FNS` or a callable
                (higher = more uncertain / informative).
        """
        super().__init__(min_children=min_children, min_effective_n=min_effective_n)
        self.lam = lam
        self._reward_fn = _resolve(reward_fn, REWARD_FNS, "reward_fn")
        self._uncertainty_fn = _resolve(uncertainty_fn, UNCERTAINTY_FNS, "uncertainty_fn")

    def score(self, hub, dag) -> float:
        """Per-hub fallback = the raw exploitation term (the real ranking is z-scored in :meth:`rank`)."""
        return self._reward_fn(hub, dag)

    def rank(self, dag) -> List:
        """Eligible hubs, best-first by ``z(reward) + λ·z(U)`` (z-scored across this set)."""
        eligible = [h for h in dag.hubs_iter() if self.is_eligible(h, dag)]
        if not eligible:
            return []
        z_reward = _zscore([self._reward_fn(h, dag) for h in eligible])
        z_unc = _zscore([self._uncertainty_fn(h, dag) for h in eligible])
        scored = []
        for hub, zr, zu in zip(eligible, z_reward, z_unc):
            # A NaN in either term drops that term's contribution (treat as 0) rather than voiding the
            # hub — a hub eligible by the gates always deserves a finite score.
            s = (0.0 if zr != zr else zr) + self.lam * (0.0 if zu != zu else zu)
            scored.append((hub, s))
        scored.sort(key=lambda hs: hs[1], reverse=True)
        return [h for h, _ in scored]

    def score_breakdown(self, dag) -> List[dict]:
        """Provenance: per eligible hub, its raw + z-scored reward / U(h) and the blended score.

        The active-learning loop logs this each round so the acquisition's explore/exploit choice is
        fully inspectable after the fact (which hubs it picked and *why*)."""
        eligible = [h for h in dag.hubs_iter() if self.is_eligible(h, dag)]
        raw_r = [self._reward_fn(h, dag) for h in eligible]
        raw_u = [self._uncertainty_fn(h, dag) for h in eligible]
        z_r, z_u = _zscore(raw_r), _zscore(raw_u)
        rows = []
        for hub, rr, ru, zr, zu in zip(eligible, raw_r, raw_u, z_r, z_u):
            s = (0.0 if zr != zr else zr) + self.lam * (0.0 if zu != zu else zu)
            rows.append(
                {
                    "hub_key": hub.key,
                    "depth": hub.depth,
                    "n_children": hub.n_children,
                    "reward": rr,
                    "uncertainty": ru,
                    "z_reward": zr,
                    "z_uncertainty": zu,
                    "score": s,
                }
            )
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows
