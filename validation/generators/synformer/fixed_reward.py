"""Frozen reward generators for the SynFormer entrant.

Per the repo convention this is a **per-adapter copy** (mirrors ``validation/generators/{rxnflow,
fraggfn,s3gfn,reinvent,saturn}/fixed_reward.py``) so the ``synformer`` env stays self-contained. The
classes are byte-for-byte the REINVENT and Saturn adapters' — verified by AST comparison, because the
benchmark's central fairness claim is that every entrant optimizes the *same* frozen reward, and
identical code is the cheapest way to keep that true. It is also checked numerically: the sEH MPNN
returns 0.0273 for ethanol in the ``synformer``, ``saturn``, ``reinvent4`` and ``s3gfn`` envs alike.

- :class:`SEHFrozenReward` — the Bengio-2021 sEH MPNN.
- :class:`DRD2FrozenReward` — the cached TDC DRD2 activity oracle (``oracle/drd2_current.pkl``),
  loaded from the pickle rather than through ``tdc.Oracle`` on purpose: TDC self-downloads into
  ``./oracle`` on first use, which fails on a compute node. The ``synformer`` env has no ``pytdc``
  at all for that reason.

:meth:`predict` returns the RAW oracle value — the scale the per-target mode gate uses (sEH ``> 7.0``,
DRD2 ``> 0.5``) — and is the only method the GA driver calls. ``reward()`` exists for interface parity
with the GFlowNet entrants and is unused here.
"""

from typing import List, Optional

import numpy as np
import torch
from gflownet.models import bengio2021flow
from rdkit import Chem


class SEHFrozenReward:
    """Frozen sEH MPNN reward generator. ``predict`` returns the RAW proxy value (mode-gate scale)."""

    higher_is_better = True

    def __init__(self, device: str = "cpu", clip: float = 10.0, batch_size: int = 128):
        self.device = device
        self.clip = float(clip)
        self.batch_size = int(batch_size)
        self.model = bengio2021flow.load_original_model()
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, smiles: List[str]) -> List[float]:
        """Raw sEH proxy value per SMILES (higher = better); ``nan`` for invalid molecules."""
        out: List[float] = []
        for start in range(0, len(smiles), self.batch_size):
            chunk = smiles[start : start + self.batch_size]
            graphs, valid = [], []
            for s in chunk:
                mol = Chem.MolFromSmiles(s) if s else None
                g = None
                if mol is not None:
                    try:
                        g = bengio2021flow.mol2graph(mol)
                    except Exception:
                        # Atom/feature outside the sEH featurizer's set. An unconstrained SMILES
                        # generator can emit these; treat as invalid, exactly as the S3-GFN adapter
                        # and native mol2seh do. Silently scoring them 0 would instead teach the
                        # policy that exotic elements are merely bad rather than unscoreable.
                        g = None
                valid.append(g is not None)
                if g is not None:
                    graphs.append(g)
            preds: List[float] = []
            if graphs:
                batch = bengio2021flow.mols2batch(graphs).to(self.device)
                preds = self.model(batch).view(-1).cpu().numpy().tolist()
            it = iter(preds)
            for ok in valid:
                out.append(float(next(it)) if ok else float("nan"))
        return out

    def reward(self, smiles: List[str]) -> List[float]:
        """Interface parity with the GFlowNet entrants; UNUSED on the REINVENT path (see module doc)."""
        return [
            float(np.exp(-self.clip))
            if v != v
            else float(np.exp(np.clip(v, -self.clip, self.clip)))
            for v in self.predict(smiles)
        ]

    def fit(self, *args, **kwargs) -> dict:  # interface parity; never called (fixed reward)
        return {}

    def set_device(self, device: str) -> None:
        self.device = device
        self.model.to(device)


class DRD2FrozenReward:
    """Frozen DRD2 activity oracle. Reproduces ``tdc.Oracle("DRD2")`` bit-for-bit (cached sklearn
    model + TDC's count-Morgan/FCFP6 featurization), matching the reaction-GFNs' ``DRD2Proxy``.

    Loaded from the cached pickle rather than through ``tdc.Oracle`` on purpose: TDC self-downloads
    into ``./oracle`` on first use, which fails on a compute node ($HOME read-only, no internet)."""

    higher_is_better = True

    def __init__(self, model_path: str, clip: float = 10.0, **_ignored):
        import pickle

        with open(model_path, "rb") as fh:
            self.model = pickle.load(fh)  # nosec - trusted local TDC oracle
        self.clip = float(clip)

    @staticmethod
    def _fp(mol):
        from rdkit.Chem import AllChem

        f = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        nfp = np.zeros((1, 2048), np.int32)
        for idx, v in f.GetNonzeroElements().items():
            nfp[0, idx % 2048] += int(v)
        return nfp

    def predict(self, smiles: List[str]) -> List[float]:
        """DRD2 activity probability in [0, 1] per SMILES; ``nan`` for invalid molecules."""
        mols = [Chem.MolFromSmiles(s) if s else None for s in smiles]
        valid_idx = [i for i, m in enumerate(mols) if m is not None]
        out = [float("nan")] * len(smiles)
        if valid_idx:
            X = np.concatenate([self._fp(mols[i]) for i in valid_idx], axis=0)
            probs = self.model.predict_proba(X)[:, 1]
            for j, i in enumerate(valid_idx):
                out[i] = float(probs[j])
        return out

    def reward(self, smiles: List[str]) -> List[float]:
        """Interface parity with the GFlowNet entrants; UNUSED on the REINVENT path (see module doc)."""
        return [
            float(np.exp(-self.clip))
            if v != v
            else float(np.exp(np.clip(v, -self.clip, self.clip)))
            for v in self.predict(smiles)
        ]

    def fit(self, *args, **kwargs) -> dict:
        return {}

    def set_device(self, *args, **kwargs) -> None:
        pass


def build_provider(
    reward_type: str,
    device: str = "cpu",
    model_path: Optional[str] = None,
    clip: float = 10.0,
    batch_size: int = 128,
):
    """One place that maps a config ``reward.type`` onto a provider, shared by the run driver and
    the scoring component so the two can never disagree about what 'seh_proxy' means."""
    if reward_type == "drd2":
        return DRD2FrozenReward(model_path=model_path or "oracle/drd2_current.pkl", clip=clip)
    if reward_type == "seh_proxy":
        return SEHFrozenReward(device=device, clip=clip, batch_size=batch_size)
    raise SystemExit(
        f"unknown reward type {reward_type!r} for the REINVENT entrant; "
        "expected 'seh_proxy' or 'drd2' (the two surrogate targets in scope). A docking target "
        "would follow the sibling adapters' DockingBridgeReward (score_batch.py bridge) pattern."
    )
