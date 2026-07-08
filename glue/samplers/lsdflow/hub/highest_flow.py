"""``highest_flow`` — rank hubs by their consensus ``log F(h)`` estimate (proposal §3a).

The median of a hub's per-child ``log F_hat`` estimates (robust to the heavy tail — §2). This
favours hubs sitting on high absolute flow; contrast with ``highest_terminating_flow``, which
favours hubs with *many* terminating children regardless of each one's individual flow.
"""

from __future__ import annotations

import gin

from glue.metrics import lsdflow_flow
from glue.samplers.lsdflow.hub.base import HubSelectionStrategy


@gin.configurable()
class HighestFlowStrategy(HubSelectionStrategy):
    name = "highest_flow"

    def score(self, hub, dag) -> float:
        return lsdflow_flow.consensus_log_flow(hub.child_log_flows())
