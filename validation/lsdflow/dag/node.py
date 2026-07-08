"""Canonical node identity for the cross-model DAG (``docs/LSD_FLOW_PROPOSAL.md`` §6).

One shared node key across all four models so aggregation is model-agnostic:

  * **primary key** = RDKit canonical SMILES with **stereo stripped**
    (``MolToSmiles(mol, isomericSmiles=False)``). FragGFN strips stereo while the reaction
    models keep it, so a stereo-aware key would block same-skeleton matches across models (§6).
  * **secondary key** = the stereo-aware canonical SMILES, kept for within-reaction-model work.

The RGFN adapter derives these from ``Molecule`` objects in
``glue.samplers.lsdflow.rgfn_extract``; the helpers here canonicalize raw SMILES strings for
the cross-env workers and any post-hoc analysis that starts from SMILES.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - RDKit present in-env
    Chem = None


def canonical_key(smiles: str, strip_stereo: bool = True) -> str:
    """Cross-model canonical key for a SMILES (stereo stripped by default, §6)."""
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, isomericSmiles=not strip_stereo)


@dataclass
class NodeStats:
    """Per-molecule aggregate over a run's trajectories — the DAG's node payload (§6)."""

    key: str  # cross-model canonical key
    stereo_key: str  # stereo-aware secondary key
    visit_count: int = 0  # trajectories passing through (reward-free estimator)
    depth: int = 0  # min #reactions to build (cheapest observed route)
    n_as_hub: int = 0  # times seen as a pre-terminal hub
    n_as_terminal: int = 0  # times seen as a terminal product
    best_reward: float = float("nan")
