"""``highest_visitation`` — the reward-free hub strategy (proposal §2, §3a).

A trained GFN visits states in proportion to their flow, so a hub's empirical trajectory
visit count is a **reward-free** estimate of ``F(h)``. Ranking by it needs no reward term at
all — the cross-check against the reward-based ``highest_flow`` is itself a diagnostic (their
disagreement, after the ``logZ`` shift, tests whether the backward policy broke trajectory
balance — §2, §8).
"""

from __future__ import annotations

import gin

from glue.samplers.lsdflow.hub.base import HubSelectionStrategy


@gin.configurable()
class HighestVisitationStrategy(HubSelectionStrategy):
    name = "highest_visitation"

    def score(self, hub, dag) -> float:
        return float(hub.visit_count)
