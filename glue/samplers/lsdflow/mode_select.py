"""Incremental mode-acceptance strategies for the LSD-Flow campaign (Logs/028).

A "mode" (RGFN / SCENT / ``[bengio2021gflownet]`` definition) is a molecule that is both (a) above
a reward/binding threshold and (b) Tanimoto-dissimilar from every mode already accepted. The
campaign feeds candidates **best-reward-first** and asks the selector, one at a time, whether each
is a *new* mode — so acceptance is stateful (it remembers the fingerprints accepted so far). This
is the pluggable "which molecules within a hub / from the pool do we keep" knob; swap the selector
to try alternatives (e.g. reward-only, scaffold-based) without touching the campaign.

Lives in ``glue/`` (RDKit-guarded) so the future AL loop imports it directly, not the validation
axis. Matches ``validation.lsdflow.metrics.diversity`` and ``glue.metrics.dataset_metrics`` (ECFP
Morgan r=3, 2048 bits, similarity 0.7).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - RDKit present in-env
    Chem = None

_FP_RADIUS = 3
_FP_BITS = 2048


@lru_cache(maxsize=None)
def ecfp(smiles: str):
    """ECFP r=3/2048 bit-vector, memoised by SMILES. Caching matters for the campaign sweeps, which
    re-select over the *same* molecule pool at many diversity cutoffs — each fingerprint is built
    once, not once per cutoff. Harmless for the (later) AL loop, which also re-sees molecules."""
    if Chem is None or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=_FP_RADIUS, nBits=_FP_BITS, useFeatures=False, useChirality=False
    )


class ModeSelector(ABC):
    """Stateful, one-at-a-time acceptance: ``accept(smiles, reward) -> bool`` (updates state)."""

    @abstractmethod
    def accept(self, smiles: str, reward: float) -> bool:
        ...


class DiverseThresholdModeSelector(ModeSelector):
    """Accept iff reward clears the threshold AND the molecule is Tanimoto-dissimilar
    (``<= similarity``, ECFP r=3/2048) from every already-accepted mode. Best-reward-first feeding
    (the campaign's job) makes each accepted molecule its cluster's best binder."""

    def __init__(
        self,
        reward_threshold: float | None,
        similarity: float = 0.5,  # campaign/AL default (Logs/029); NOTE the paper-comparable mode
        # count in glue/metrics/dataset_metrics.py + validation/lsdflow/metrics/diversity.py stays 0.7
        higher_is_better: bool = True,
    ):
        self.reward_threshold = reward_threshold
        self.similarity = similarity
        self.higher_is_better = higher_is_better
        self._fps: list = []

    def _passes_gate(self, reward: float) -> bool:
        if self.reward_threshold is None:
            return True
        if reward != reward:  # NaN
            return False
        return (
            reward >= self.reward_threshold
            if self.higher_is_better
            else reward <= self.reward_threshold
        )

    def accept(self, smiles: str, reward: float) -> bool:
        if not self._passes_gate(reward):
            return False
        fp = ecfp(smiles)
        if fp is None:
            return False
        if any(DataStructs.TanimotoSimilarity(fp, a) > self.similarity for a in self._fps):
            return False
        self._fps.append(fp)
        return True
