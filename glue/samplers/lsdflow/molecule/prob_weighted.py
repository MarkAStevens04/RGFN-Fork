"""``prob_weighted`` — sample a hub's children by flow (trajectory balance) (proposal §3a).

Sample ``k`` children **without replacement** with probability proportional to their
recovered terminating flow ``exp(log F_hat)``. Under trajectory balance a terminal state's
flow is proportional to its reward, so this draws the batch the way the trained policy itself
would weight these products — between reward-greedy ``topk_reward`` and diversity-first
``uniform_random``. Weighted-without-replacement uses the Efraimidis–Spirakis reservoir key
``log(u)/w`` so it is exact and order-stable under a seeded ``rng``.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

import gin

from glue.samplers.lsdflow.molecule.base import MoleculeSelectionStrategy


@gin.configurable()
class ProbWeightedStrategy(MoleculeSelectionStrategy):
    name = "prob_weighted"

    def select(
        self,
        hub,
        k: int,
        *,
        higher_is_better: bool = True,
        rng: Optional[random.Random] = None,
    ) -> List:
        rng = rng or random.Random()
        children = [c for c in hub.children if c.log_flow == c.log_flow]  # drop NaN
        if k <= 0 or not children:
            return []
        if len(children) <= k:
            return list(children)
        # Softmax weights over log-flow (shift by max for numerical stability).
        m = max(c.log_flow for c in children)
        keyed = []
        for c in children:
            w = math.exp(c.log_flow - m)
            if w <= 0.0:
                continue
            u = rng.random()
            # log(u)/w : larger key = more likely drawn; exact weighted sampling w/o replacement.
            key = math.log(u) / w if u > 0.0 else float("-inf")
            keyed.append((key, c))
        keyed.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in keyed[:k]]
