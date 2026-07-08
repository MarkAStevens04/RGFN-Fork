"""LSD-Flow flow recovery — the ``docs/LSD_FLOW_PROPOSAL.md`` §2 math, log-space.

For a hub ``h`` and a terminal product ``x`` reachable from ``h`` by **one reaction
then stop**, the trajectory-balance flow the model implies is

    F_hat(h ; x) = R(x) * P_B(h | x) / ( P_F(x | h) * P_F(stop | x) )                (§2)

Everything here is computed in **log space** (§2: estimates are heavy-tailed; raw-space
variance is dominated by a single outlier), from the per-step ``logP_F`` / ``logP_B`` that
``objective.assign_log_probs`` produces plus the shaped ``log R(x)`` from ``Reward``:

    log F_hat(h ; x) = logR(x) + logP_B(h|x) - logP_F(x|h) - logP_F(stop|x)

This module is **pure** — it takes already-extracted scalar log-quantities and knows
nothing about RGFN internals, ``networkx``, or RDKit. The rgfn-native extraction of those
scalars from a ``Trajectories`` object lives in ``glue.samplers.lsdflow.rgfn_extract``; the
cross-env extraction lives in ``validation/lsdflow``. Both feed the numbers here.

The two consuming axes (proposal §3):
  * the production AL path (``glue.samplers.lsdflow``) — hub selection strategies call the
    aggregators below over each hub's sampled terminal children;
  * the validation analysis world (``validation/lsdflow``) — the DAG stores per-child
    ``log_flow`` from :func:`log_flow_estimate` and the severe-test suite reuses the
    aggregators for consistency.
"""

from __future__ import annotations

import math
from typing import Sequence

_NEG_INF = float("-inf")


def log_flow_estimate(
    log_reward: float,
    log_pb_move: float,
    log_pf_move: float,
    log_pf_stop: float,
) -> float:
    """One terminal child's estimate of ``log F(h)`` (proposal §2).

    Args:
        log_reward: ``log R(x)`` — the **shaped** log reward at the terminal product
            (the same ``log_reward`` the objective trains against, from
            ``RewardOutput.log_reward``). Reward is the only source term (§2).
        log_pb_move: ``log P_B(h | x)`` under the model's **own trained** backward
            policy (§5). For RGFN this is the learned C-disconnection log-prob (the only
            non-deterministic backward step); undoing the template/reactant/stop micro-steps
            contributes 0.
        log_pf_move: ``log P_F(x | h)`` — the forward log-prob of the diversifying
            reaction, i.e. the sum of the A(template)+B(reactant(s))+C(commit) micro-step
            forward log-probs composing the one molecule->molecule move (§4b).
        log_pf_stop: ``log P_F(stop | x)`` — the terminating factor (§2: **do not drop
            this**; it equals 0 in log-space only for true leaves and omitting it biases
            ``F_hat`` upward for products eager to react further). For RGFN this is the
            forward log-prob of the final ``StateA -> Terminal`` stop micro-step.

    Returns:
        ``log F_hat(h ; x)``.
    """
    return log_reward + log_pb_move - log_pf_move - log_pf_stop


def logsumexp(values: Sequence[float]) -> float:
    """Numerically-stable ``log(sum_i exp(v_i))``; ``-inf`` for an empty input."""
    finite = [v for v in values if v != _NEG_INF]
    if not finite:
        return _NEG_INF
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))


def consensus_log_flow(child_log_flows: Sequence[float]) -> float:
    """The hub's single ``log F(h)`` estimate: the **median** of its per-child estimates.

    Under a converged GFN every terminal child ``x_i`` independently estimates the same
    ``F(h)`` and they agree (§2). The median (not the mean) is the consensus statistic —
    robust to the heavy tail that motivates log-space in the first place. Ranking hubs by
    this is the ``highest_flow`` strategy.
    """
    xs = sorted(v for v in child_log_flows if v == v)  # drop NaN
    n = len(xs)
    if n == 0:
        return _NEG_INF
    mid = n // 2
    return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def total_terminating_log_flow(child_log_flows: Sequence[float]) -> float:
    """Total flow that **terminates one reaction downstream** of the hub: ``logsumexp`` of
    the per-child terminating estimates.

    This is the quantity the ``highest_terminating_flow`` strategy ranks by, and the one
    the proposal (§8) singles out as rewarding *diversification parents* — hubs from which
    many distinct high-reward products terminate in a single step — rather than good
    general building blocks. A hub with many good terminating children scores higher than a
    hub with one, which is exactly the late-stage-diversification signal.
    """
    return logsumexp(child_log_flows)


def log_visitation_estimate(
    visit_count: int,
    total_trajectories: int,
    log_z: float,
) -> float:
    """Reward-free MC estimate of ``log F(h)`` from empirical visitation (proposal §2).

    A trained GFN visits states in proportion to their flow, so the empirical fraction of
    trajectories passing through ``h`` is a reward-free estimate of ``F(h)/Z``; adding
    ``logZ`` (the trained ``objective.logZ``) puts it on the same absolute scale as
    :func:`log_flow_estimate`. Disagreement between the two — after the ``logZ`` shift — is
    the diagnostic that directly tests whether a cost-guided backward policy (SCENT) has
    broken trajectory balance (§2, §8).

    Returns ``-inf`` for an unvisited hub.
    """
    if visit_count <= 0 or total_trajectories <= 0:
        return _NEG_INF
    return math.log(visit_count / total_trajectories) + log_z
