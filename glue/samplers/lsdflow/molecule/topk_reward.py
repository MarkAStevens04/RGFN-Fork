"""``topk_reward`` — take a hub's ``k`` highest-reward children (proposal §3a).

The reward-greedy molecule strategy. Orientation follows the reward's sign convention
(``higher_is_better``), so it works unchanged for higher-is-better surrogates (sEH) and
lower-is-better docking differentials.
"""

from __future__ import annotations

import random
from typing import List, Optional

import gin

from glue.samplers.lsdflow.molecule.base import MoleculeSelectionStrategy


@gin.configurable()
class TopKRewardStrategy(MoleculeSelectionStrategy):
    name = "topk_reward"

    def select(
        self,
        hub,
        k: int,
        *,
        higher_is_better: bool = True,
        rng: Optional[random.Random] = None,
    ) -> List:
        children = [c for c in hub.children if c.reward == c.reward]
        children.sort(key=lambda c: c.reward, reverse=higher_is_better)
        return children[: max(0, k)]
