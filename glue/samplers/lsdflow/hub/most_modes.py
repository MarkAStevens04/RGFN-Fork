"""``most_modes`` — rank hubs by how many distinct products branch from them (proposal §3a).

The purest diversity criterion: a hub's score is the number of **distinct** terminal children
observed from it. By default "distinct" means distinct canonical keys; pass a ``mode_counter``
(keys -> cluster count, e.g. Butina/ECFP from ``validation/lsdflow/metrics/diversity``) to
score by chemical *modes* rather than raw structural distinctness — kept injectable so this
production strategy stays free of the RDKit clustering that lives on the validation axis.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import gin

from glue.samplers.lsdflow.hub.base import HubSelectionStrategy


@gin.configurable()
class MostModesStrategy(HubSelectionStrategy):
    name = "most_modes"

    def __init__(
        self,
        min_children: int = 1,
        min_effective_n: int = 1,
        mode_counter: Optional[Callable[[Sequence[str]], int]] = None,
    ):
        super().__init__(min_children=min_children, min_effective_n=min_effective_n)
        self.mode_counter = mode_counter

    def score(self, hub, dag) -> float:
        keys = [c.key for c in hub.children]
        if self.mode_counter is not None:
            return float(self.mode_counter(keys))
        return float(len(set(keys)))
