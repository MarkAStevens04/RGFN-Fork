"""``reactions_per_mode`` — the PRIMARY cost metric (``docs/LSD_FLOW_PROPOSAL.md`` §11).

``total_reactions_to_build_batch / n_modes`` (lower is better). It encodes the late-stage-
diversification value proposition and reads to a chemist:

  * **hub batch:** ``total_reactions = depth(h) + k`` — the parent hub is built once
    (``depth(h)`` reactions) and amortized across ``k`` final diversifying reactions.
  * **independent top-k:** ``sum_j depth(x_j)`` — every product built from scratch.

Since each terminal child is exactly one reaction past its hub, ``depth(x_j) = depth(h) + 1``,
so the hub batch saves ``(k - 1) * depth(h)`` reactions. The real saving is reaction *time*,
not reactant cost (§11).
"""

from __future__ import annotations

from typing import Sequence

from validation.lsdflow.metrics.cost.base import CostModel


class ReactionsPerModeCost(CostModel):
    name = "reactions_per_mode"

    def batch_reactions(self, hub_depth: int, k: int) -> float:
        return float(hub_depth + k)

    def independent_reactions(self, child_depths: Sequence[int]) -> float:
        return float(sum(child_depths))
