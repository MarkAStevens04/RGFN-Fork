"""Saturn ``OracleComponent`` exposing the benchmark's FROZEN surrogate rewards.

This is the seam that makes Saturn a fair entrant: it optimizes the *same* frozen reward as every
other generator (the Bengio-2021 sEH MPNN, or the cached TDC DRD2 oracle), not a Saturn-native
approximation of it.

HOW IT GETS IN — and why it is not a plugin. Unlike REINVENT, Saturn has **no plugin discovery**:
``oracles/utils.py`` imports every oracle module eagerly and ``construct_oracle_component`` is a
hard-coded ``if name == ...`` chain. Adding a component upstream-style would mean editing the clone.
Instead ``run_saturn_fixed.py`` monkey-patches ``oracles.oracle.construct_oracle_component`` to
return this class for ``name == "glue_surrogate"`` and to defer to the original for anything else —
the same technique ``validation/generators/s3gfn/run_s3gfn_fixed.py`` uses to inject its reward in
place of S3-GFN's module-global ``get_scores``. The clone stays pristine.

Configured through the component's ``specific_parameters``::

    {"name": "glue_surrogate",
     "weight": 1,
     "specific_parameters": {"reward_type": "seh_proxy", "model_path": "", "device": "cpu"},
     "reward_shaping_function_parameters": {
         "transformation_function": "sigmoid",
         "parameters": {"low": 4.0, "high": 9.0, "k": 0.5}}}

RAW IN, SHAPED OUT. ``__call__`` returns the RAW oracle value; Saturn's ``calculate_reward`` then
applies the configured shaping to get a ``[0, 1]`` reward. Keeping the shaping in the config mirrors
what the REINVENT adapter does with ``transform``, so the one knob that changes what the generator
chases is visible in the same place for both entrants.

FAILURE IS 0.0 HERE, NOT NaN — the opposite of the REINVENT adapter, on purpose. That is each
framework's own documented convention: REINVENT's ``ComponentResults`` states failures *must* be NaN
and masks them, whereas Saturn's ``OracleComponent.calculate_reward`` says "Errors are assigned a
reward of 0.0" and its shaping functions have no NaN path. For both of our targets 0.0 is also the
correct floor on the merits (sEH and DRD2 are both higher-is-better with a non-negative raw scale),
so this loses no information. In practice it is nearly unreachable: Saturn hands the component
already-parsed RDKit ``Mol`` objects, so the only way to get here is an atom outside the sEH
featurizer's set.

DEVICE DEFAULTS TO CPU. Saturn's Mamba agent owns the GPU; the sEH MPNN scores a batch on CPU in
tens of milliseconds, far below Saturn's per-step cost, so there is nothing to gain from contending
for VRAM.
"""

from __future__ import annotations

__all__ = ["GlueSurrogateOracle", "GLUE_SURROGATE_NAME"]

import sys
from pathlib import Path
from typing import List

import numpy as np
from rdkit import Chem

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oracles.dataclass import OracleComponentParameters  # noqa: E402
from oracles.oracle_component import (  # noqa: E402  (Saturn clone, on PYTHONPATH)
    OracleComponent,
)

from validation.generators.saturn.fixed_reward import build_provider  # noqa: E402

#: The ``name`` a component must carry in the run config to be routed to this class.
GLUE_SURROGATE_NAME = "glue_surrogate"


class GlueSurrogateOracle(OracleComponent):
    """Saturn oracle component backed by a frozen benchmark reward provider."""

    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        sp = dict(parameters.specific_parameters or {})
        reward_type = sp.get("reward_type", "seh_proxy")
        model_path = (sp.get("model_path") or "").strip()
        if model_path and not Path(model_path).is_absolute():
            model_path = str(_REPO_ROOT / model_path)
        # Docking is reached across the env boundary (this env has no docking stack); for the
        # surrogates these extra arguments are ignored by build_provider.
        self.provider = build_provider(
            reward_type=reward_type,
            device=sp.get("device", "cpu"),
            model_path=(sp.get("model_path") or "").strip() or None,
            oracle=(sp.get("oracle") or "").strip() or None,
            repo_root=str(_REPO_ROOT),
            norm=float(sp.get("norm", 1.0)),
            oracle_args=dict(sp.get("oracle_args") or {}),
            workdir=(sp.get("workdir") or "").strip() or None,
        )
        self.reward_type = reward_type

    def __call__(self, mols: "np.ndarray") -> "np.ndarray":
        """RAW oracle value per molecule (higher = better); 0.0 for anything unscoreable."""
        smiles: List[str] = [Chem.MolToSmiles(m) if m is not None else "" for m in mols]
        raw = self.provider.predict(smiles)
        return np.array([0.0 if v != v else float(v) for v in raw], dtype=float)
