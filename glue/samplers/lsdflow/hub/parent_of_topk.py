"""``parent_of_topk`` — the control strategy (proposal §3a, §12).

Rank hubs by the **single best-reward child** they produced, ignoring flow and diversity
entirely. This is deliberately the naive "just take the parents of the top molecules" baseline
the flow-based strategies must beat: if ``highest_terminating_flow`` doesn't produce more
diverse / cheaper batches than this control, the flow signal is buying nothing.
"""

from __future__ import annotations

import gin

from glue.samplers.lsdflow.hub.base import HubSelectionStrategy


@gin.configurable()
class ParentOfTopKStrategy(HubSelectionStrategy):
    name = "parent_of_topk"

    def score(self, hub, dag) -> float:
        rewards = [c.reward for c in hub.children if c.reward == c.reward]
        if not rewards:
            return float("nan")
        # Orient so higher score = better regardless of the reward's sign convention.
        return max(rewards) if dag.higher_is_better else -min(rewards)
