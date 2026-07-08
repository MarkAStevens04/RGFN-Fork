"""``lowest_uncertainty`` — rank/gate hubs by flow-matching consistency (proposal §2, §3a).

``U(h)`` = variance of the per-child ``log F_hat`` estimates (§2). A low ``U(h)`` means the
hub's terminal children agree on ``F(h)`` — the model is internally consistent there, so its
flow estimate is trustworthy. This strategy prefers those hubs (score = ``-U(h)``) and, via
``min_children=2``, gates out hubs that carry no consistency signal at all.

NOTE (§2): the *inverse* reading — high ``U(h)`` = most informative to query — is the phase-2
active-learning acquisition claim and must not be leaned on in phase 1.
"""

from __future__ import annotations

import gin

from glue.metrics import uncertainty
from glue.samplers.lsdflow.hub.base import HubSelectionStrategy


@gin.configurable()
class LowestUncertaintyStrategy(HubSelectionStrategy):
    name = "lowest_uncertainty"

    def __init__(self, min_children: int = 2, min_effective_n: int = 2):
        # U(h) is undefined for <2 children — gate them out by default.
        super().__init__(min_children=min_children, min_effective_n=min_effective_n)

    def score(self, hub, dag) -> float:
        u = uncertainty.flow_matching_uncertainty(hub.child_log_flows())
        return -u if u == u else float("nan")
