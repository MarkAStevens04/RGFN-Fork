"""Name -> molecule-selection-strategy registry (proposal §1)."""

from __future__ import annotations

from typing import Dict, List, Type

from glue.samplers.lsdflow.molecule.base import MoleculeSelectionStrategy
from glue.samplers.lsdflow.molecule.prob_weighted import ProbWeightedStrategy
from glue.samplers.lsdflow.molecule.topk_reward import TopKRewardStrategy
from glue.samplers.lsdflow.molecule.uniform_random import UniformRandomStrategy

_STRATEGIES: Dict[str, Type[MoleculeSelectionStrategy]] = {
    cls.name: cls for cls in (TopKRewardStrategy, ProbWeightedStrategy, UniformRandomStrategy)
}


def available() -> List[str]:
    return sorted(_STRATEGIES)


def register(cls: Type[MoleculeSelectionStrategy]) -> Type[MoleculeSelectionStrategy]:
    _STRATEGIES[cls.name] = cls
    return cls


def get_molecule_strategy(name: str, **kwargs) -> MoleculeSelectionStrategy:
    try:
        cls = _STRATEGIES[name]
    except KeyError:
        raise KeyError(f"Unknown molecule strategy {name!r}. Available: {available()}") from None
    return cls(**kwargs)
