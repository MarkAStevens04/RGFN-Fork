"""``highest_terminating_flow`` — rank hubs by total flow that terminates one reaction
downstream (proposal §3a, §8).

``logsumexp`` of the per-child ``log F_hat`` estimates: a hub with many distinct high-reward
products that each terminate in a single step scores higher than one with a single good
child. This is the strategy the proposal (§8) singles out as rewarding *diversification
parents* over good general building blocks — the core late-stage-diversification signal, and
the variant most likely to find hubs SCENT's reward-averaging utility misses.
"""

from __future__ import annotations

import gin

from glue.metrics import lsdflow_flow
from glue.samplers.lsdflow.hub.base import HubSelectionStrategy


@gin.configurable()
class HighestTerminatingFlowStrategy(HubSelectionStrategy):
    name = "highest_terminating_flow"

    def score(self, hub, dag) -> float:
        return lsdflow_flow.total_terminating_log_flow(hub.child_log_flows())
