"""Incremental mode-acceptance strategies for the LSD-Flow campaign (Logs/028).

A "mode" (RGFN / SCENT / ``[bengio2021gflownet]`` definition) is a molecule that is both (a) above
a reward/binding threshold and (b) Tanimoto-dissimilar from every mode already accepted. The
campaign feeds candidates **best-reward-first** and asks the selector, one at a time, whether each
is a *new* mode — so acceptance is stateful (it remembers the fingerprints accepted so far). This
is the pluggable "which molecules within a hub / from the pool do we keep" knob; swap the selector
to try alternatives (e.g. reward-only, scaffold-based) without touching the campaign.

Two selectors, which together are the **filter ablation** (Logs/054, via the strategies'
``mode_selector_factory`` seam):

- :class:`DiverseThresholdModeSelector` — both halves of the mode definition (reward gate + Tanimoto
  dedup). Setting ``reward_threshold=None`` ablates the reward half.
- :class:`RewardOnlyModeSelector` — ablates the **diversity** half (exact duplicates still dropped).

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


def passes_reward_gate(
    reward: float, reward_threshold: float | None, higher_is_better: bool = True
) -> bool:
    """The reward half of the mode definition, shared by every selector here.

    ``None`` threshold admits everything (the reward-filter ablation); a NaN reward never passes a
    set gate. Orientation-aware, so lower-is-better docking differentials work unchanged.
    """
    if reward_threshold is None:
        return True
    if reward != reward:  # NaN
        return False
    return reward >= reward_threshold if higher_is_better else reward <= reward_threshold


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
        return passes_reward_gate(reward, self.reward_threshold, self.higher_is_better)

    def accept(self, smiles: str, reward: float) -> bool:
        if not self._passes_gate(reward):
            return False
        fp = ecfp(smiles)
        if fp is None:
            return False
        # One BulkTanimotoSimilarity call instead of a Python loop over the accepted modes. Same
        # metric, same values, so the accept/reject decision (and hence every campaign number) is
        # unchanged — but the inner loop runs in C++. This test is the dominant cost of a
        # best-candidate run, which walks the whole pool against a growing accepted set.
        if self._fps and max(DataStructs.BulkTanimotoSimilarity(fp, self._fps)) > self.similarity:
            return False
        self._fps.append(fp)
        return True


class RewardOnlyModeSelector(ModeSelector):
    """The **diversity-filter ablation** (Logs/054): keep every molecule clearing the reward gate,
    with no similarity test at all — only *exact-duplicate* suppression.

    Why duplicates are still dropped. The knob under test is the **similarity** criterion, not
    deduplication: no chemist synthesises the identical molecule twice, and a library that counted
    the same SMILES repeatedly would be a strawman rather than a measurement of what the Tanimoto
    filter buys. :attr:`n_duplicates_suppressed` records how often that happened, so the size of the
    effect we removed is reported rather than hidden (a hub-batching walk can re-derive one child
    from several hubs).

    Parity with :class:`DiverseThresholdModeSelector` is deliberate on the two axes that are *not*
    under test, so any difference in the delivered library is attributable to the similarity test
    alone: the same reward gate (:func:`passes_reward_gate`, NaN-safe, orientation-aware) and the
    same validity requirement (an unparseable SMILES is rejected — there, because no fingerprint can
    be built; here, because no molecule can). Note that with the similarity test gone, molecules that
    share a fingerprint but not a SMILES (stereoisomers — the ECFP recipe sets
    ``useChirality=False``) become separate library members where the diverse selector would have
    collapsed them; that is part of what the filter was doing.
    """

    def __init__(
        self,
        reward_threshold: float | None,
        higher_is_better: bool = True,
    ):
        self.reward_threshold = reward_threshold
        self.higher_is_better = higher_is_better
        self.n_duplicates_suppressed = 0
        self._seen: set = set()

    def accept(self, smiles: str, reward: float) -> bool:
        if not passes_reward_gate(reward, self.reward_threshold, self.higher_is_better):
            return False
        if Chem is not None:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                return False
            key = Chem.MolToSmiles(mol)  # canonical, so two spellings of one molecule collide
        else:  # pragma: no cover - RDKit present in-env
            if not smiles:
                return False
            key = smiles
        if key in self._seen:
            self.n_duplicates_suppressed += 1
            return False
        self._seen.add(key)
        return True
