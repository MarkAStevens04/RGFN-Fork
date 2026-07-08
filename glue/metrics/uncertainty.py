"""LSD-Flow uncertainty signal ``U(h)`` — the ``docs/LSD_FLOW_PROPOSAL.md`` §2 residual.

Each terminal child ``x_i`` of a hub ``h`` independently estimates the same ``F(h)`` via
:func:`glue.metrics.lsdflow_flow.log_flow_estimate`. Under a converged GFN they agree, so
their spread is the **local flow-matching residual** — an epistemic-uncertainty signal
obtained with no ensemble, no dropout, and no extra model:

    U(h) = Var_i [ log F_hat(h ; x_i) ]        # population variance, log space (§2)

Log space is mandatory (§2): the estimates are heavy-tailed and a raw-space variance is
dominated by a single outlier.

Three roles across the two project phases (§2):
  * **hub-selection criterion** (phase 1, fixed reward): rank/gate hubs by consistency
    (``lowest_uncertainty`` strategy);
  * **diagnostic** (phase 1): does ``U(h)`` track disagreement between flow estimators?
  * **acquisition signal** (phase 2, active learning): high-``U`` hubs are where the model
    is internally inconsistent about downstream reward, hence most informative to query.
    This strong claim needs the AL phase — do not over-promise it in phase 1.

Pure module: takes a sequence of per-child log-flow estimates, returns floats.
"""

from __future__ import annotations

from typing import Sequence


def flow_matching_uncertainty(child_log_flows: Sequence[float]) -> float:
    """``U(h)`` = population variance of the per-child ``log F_hat`` estimates (§2).

    Returns ``nan`` for a hub with fewer than two children (variance is undefined — such a
    hub carries no consistency signal and should be gated out, not ranked). Population
    (not sample) variance so a hub with two identical estimates reads exactly 0.
    """
    xs = [v for v in child_log_flows if v == v]  # drop NaN/inf-free handled by caller
    n = len(xs)
    if n < 2:
        return float("nan")
    mean = sum(xs) / n
    return sum((x - mean) ** 2 for x in xs) / n


def effective_sample_count(child_log_flows: Sequence[float]) -> int:
    """Number of finite per-child estimates backing a hub's ``U(h)`` / flow value.

    The proposal's severe-test #5 (noise floor) logs this per hub: a ``U(h)`` computed
    from two children is far noisier than one from twenty, and rankings that flip when
    ``N`` grows are "estimates too noisy to act on". Callers gate on a minimum count.
    """
    return sum(1 for v in child_log_flows if v == v)
