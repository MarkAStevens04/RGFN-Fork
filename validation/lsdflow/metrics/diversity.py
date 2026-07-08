"""Diversity metrics for LSD-Flow batches (``docs/LSD_FLOW_PROPOSAL.md`` §7 #1, §11).

**Mode (primary, §11):** a Butina / sphere-exclusion cluster on ECFP4 (Morgan radius 2, 2048
bits) at a fixed Tanimoto cutoff (default 0.65, a config knob). "One mode = one representative
you'd actually synthesize." Secondary: unique Bemis-Murcko scaffolds. These back severe-test
#1 (diversity, not redundancy) and the reactions-per-mode cost denominator.

Lives on the validation axis because it needs RDKit clustering; the production ``most_modes``
strategy takes an injected ``mode_counter`` so it can use this without importing it.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.ML.Cluster import Butina
except Exception:  # pragma: no cover - RDKit present in-env
    Chem = None


def _fingerprints(smiles: Sequence[str]) -> List[Optional[object]]:
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if Chem is not None else None
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None)
    return fps


def count_modes(smiles: Sequence[str], cutoff: float = 0.65) -> int:
    """Number of Butina clusters (modes) among ``smiles`` at Tanimoto ``cutoff`` (§11).

    Butina clusters on distance ``1 - Tanimoto`` with threshold ``1 - cutoff``. Empty input
    -> 0; RDKit unavailable -> the count of distinct SMILES (degraded fallback).
    """
    valid = [s for s in smiles if s]
    if not valid:
        return 0
    if Chem is None:
        return len(set(valid))
    fps = [f for f in _fingerprints(valid) if f is not None]
    n = len(fps)
    if n == 0:
        return 0
    if n == 1:
        return 1
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, 1.0 - cutoff, isDistData=True)
    return len(clusters)


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


def mode_counter(cutoff: float = 0.65):
    """A ``keys -> n_modes`` closure for injection into the production ``most_modes`` strategy."""

    def _count(keys: Sequence[str]) -> int:
        return count_modes(keys, cutoff=cutoff)

    return _count
