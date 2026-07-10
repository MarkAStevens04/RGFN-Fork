"""Diversity + mode metrics for LSD-Flow batches (``docs/LSD_FLOW_PROPOSAL.md`` §7 #1, §11).

**Mode (paper-comparable).** Matches the RGFN / ``[bengio2021gflownet]`` mode definition exactly
as implemented by upstream ``rgfn.trainer.metrics.TanimotoSimilarityModes._extract_modes`` (and
the project's ``glue.metrics.dataset_metrics.count_modes``): a molecule is a new mode iff it is

  (a) **above a reward/binding threshold** (``proxy_term_threshold`` upstream — the "hit" bar), and
  (b) **Tanimoto-dissimilar from every mode already accepted** — greedy sphere-exclusion on ECFP
      (Morgan **radius 3**, 2048 bits, no features/chirality) at a fixed similarity threshold
      (default **0.7**).

Candidates are processed **best-reward-first**, so each mode's representative is its
highest-reward member — the molecule you would actually synthesize. This makes "modes" here
directly comparable to the generative-model mode counts RGFN/SCENT report.

Secondary: unique Bemis-Murcko scaffolds. Lives on the validation axis (needs RDKit); the
production ``most_modes`` strategy takes an injected ``mode_counter`` so it can use this without
importing it.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover - RDKit present in-env
    Chem = None

# ECFP recipe identical to rgfn TanimotoSimilarityModes + glue.metrics.dataset_metrics (r=3, 2048).
_FP_RADIUS = 3
_FP_BITS = 2048
_MODE_SIMILARITY_THRESHOLD = 0.7


def _ecfp(smiles: str):
    if Chem is None or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=_FP_RADIUS, nBits=_FP_BITS, useFeatures=False, useChirality=False
    )


def _passes_gate(reward, reward_threshold, higher_is_better) -> bool:
    """The reward/binding gate: ``>= threshold`` (higher-is-better) or ``<= threshold``. A
    missing threshold admits everything; NaN reward never passes a set gate."""
    if reward_threshold is None:
        return True
    if reward != reward:  # NaN
        return False
    return reward >= reward_threshold if higher_is_better else reward <= reward_threshold


def mode_representatives(
    keys: Sequence[str],
    rewards: Optional[Sequence[float]] = None,
    *,
    higher_is_better: bool = True,
    reward_threshold: Optional[float] = None,
    similarity_threshold: float = _MODE_SIMILARITY_THRESHOLD,
    max_modes: Optional[int] = None,
) -> List[int]:
    """Indices (into ``keys``) of the mode representatives — the paper's greedy definition.

    Reproduces ``TanimotoSimilarityModes._extract_modes``: (1) drop keys failing the reward gate;
    (2) sort survivors best-reward-first (when ``rewards`` is given); (3) greedily accept a
    molecule as a new mode iff its Tanimoto similarity to every accepted mode is
    ``<= similarity_threshold``. The accepted molecule is that mode's representative (its
    highest-reward member, by the best-first order). Without ``rewards`` it degrades to
    structure-only modes in input order (no gate, no sort).
    """
    idxs = [i for i, k in enumerate(keys) if k]
    if rewards is not None:
        idxs = [i for i in idxs if _passes_gate(rewards[i], reward_threshold, higher_is_better)]
        idxs.sort(
            key=lambda i: (rewards[i] if rewards[i] == rewards[i] else float("-inf")),
            reverse=higher_is_better,
        )
    reps: List[int] = []
    rep_fps: List[object] = []
    for i in idxs:
        fp = _ecfp(keys[i])
        if fp is None:
            continue
        if all(DataStructs.TanimotoSimilarity(fp, m) <= similarity_threshold for m in rep_fps):
            reps.append(i)
            rep_fps.append(fp)
            if max_modes and len(reps) >= max_modes:
                break
    return reps


def count_modes(
    keys: Sequence[str],
    rewards: Optional[Sequence[float]] = None,
    *,
    higher_is_better: bool = True,
    reward_threshold: Optional[float] = None,
    similarity_threshold: float = _MODE_SIMILARITY_THRESHOLD,
) -> int:
    """Number of paper-comparable modes (see :func:`mode_representatives`). RDKit-unavailable
    fallback: distinct-SMILES count."""
    if Chem is None:
        return len({k for k in keys if k})
    return len(
        mode_representatives(
            keys,
            rewards,
            higher_is_better=higher_is_better,
            reward_threshold=reward_threshold,
            similarity_threshold=similarity_threshold,
        )
    )


def unique_scaffolds(smiles: Sequence[str]) -> int:
    """Number of distinct Bemis-Murcko scaffolds (secondary diversity metric, §11)."""
    valid = [s for s in smiles if s]
    if not valid or Chem is None:
        return len(set(valid))
    scaffolds = set()
    for smi in valid:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except Exception:
            continue
    return len(scaffolds)


def mode_counter(similarity_threshold: float = _MODE_SIMILARITY_THRESHOLD):
    """A ``keys -> n_modes`` closure for the production ``most_modes`` strategy. Structure-only
    (no reward gate): that strategy ranks hubs by the raw structural diversity of their children."""

    def _count(keys: Sequence[str]) -> int:
        return count_modes(keys, similarity_threshold=similarity_threshold)

    return _count
