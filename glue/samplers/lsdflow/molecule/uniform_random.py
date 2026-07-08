"""``uniform_random`` — equal-weight all of a hub's children (proposal §3a).

The diversity-first molecule strategy: sample ``k`` children uniformly without replacement,
ignoring reward and flow. Paired with a flow-based hub strategy it tests whether the *hub*
already concentrates diversity even when the per-hub pick does no further selection.
"""

from __future__ import annotations

import random
from typing import List, Optional

import gin

from glue.samplers.lsdflow.molecule.base import MoleculeSelectionStrategy


@gin.configurable()
class UniformRandomStrategy(MoleculeSelectionStrategy):
    name = "uniform_random"

    def select(
        self,
        hub,
        k: int,
        *,
        higher_is_better: bool = True,
        rng: Optional[random.Random] = None,
    ) -> List:
        rng = rng or random.Random()
        children = list(hub.children)
        if k <= 0 or not children:
            return []
        if len(children) <= k:
            return children
        return rng.sample(children, k)
