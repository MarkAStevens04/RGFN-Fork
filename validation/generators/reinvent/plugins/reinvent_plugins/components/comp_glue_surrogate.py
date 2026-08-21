"""REINVENT 4 scoring component exposing the benchmark's FROZEN surrogate rewards.

This is the seam that makes REINVENT a fair entrant in the LSD-Flow library-efficiency benchmark:
it optimizes the *same* frozen reward as every other generator (the Bengio-2021 sEH MPNN, or the
cached TDC DRD2 oracle), not a REINVENT-native approximation of it.

    [[stage.scoring.component]]
    [stage.scoring.component.GlueSurrogate]

    [[stage.scoring.component.GlueSurrogate.endpoint]]
    name = "sEH proxy"
    weight = 1.0
    params.reward_type = "seh_proxy"        # or "drd2"
    params.model_path = ""                  # required for drd2: oracle/drd2_current.pkl
    params.device = "cpu"                   # see the device note below

    transform.type = "sigmoid"              # maps the RAW value into [0, 1] for DAP
    transform.low = 4.0
    transform.high = 9.0
    transform.k = 0.5

WHY THIS FILE IS NOT INSIDE THE CLONE. ``reinvent/scoring/importer.py`` requires
``reinvent_plugins.components`` to be a NAMESPACE package — it raises outright if the package has a
``__file__`` — and then discovers components by walking ``sys.path``. So any directory on
``PYTHONPATH`` that contains ``reinvent_plugins/components/comp_*.py`` contributes components. That
is the upstream-supported extension point, so the component can live in our repo under version
control and the clone stays pristine. **Do not add ``__init__.py`` anywhere under ``plugins/``** —
that converts these into regular packages and REINVENT refuses to start.

RAW IN, TRANSFORM OUT. ``__call__`` returns the RAW oracle value, which is the scale the benchmark's
per-target mode gate is defined on (sEH ``> 7.0``, DRD2 ``> 0.5``). REINVENT's declarative
``transform`` block maps it into ``[0, 1]`` for the DAP objective. Keeping the shaping in the TOML
rather than in this file means the one knob that changes what REINVENT chases is visible in the run
config a reader (or reviewer) opens first.

NaN, NOT ZERO, FOR FAILURE. ``ComponentResults`` documents this explicitly: an unscoreable molecule
must be NaN. Scoring it 0 would instead teach the policy that such molecules are merely *bad*, which
is a different and wrong signal.

DEVICE DEFAULTS TO CPU ON PURPOSE. REINVENT's ``Scorer`` may evaluate components in worker processes,
and the RNN already owns ``cuda:0``; putting a second torch model on the same device invites forked
CUDA contexts and VRAM contention for no gain — the sEH MPNN scores a 128-SMILES batch on CPU in
tens of milliseconds, far below REINVENT's per-step cost.
"""

from __future__ import annotations

__all__ = ["GlueSurrogate"]

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass

from .add_tag import add_tag
from .component_results import ComponentResults

logger = logging.getLogger("reinvent")

# The providers live in our repo, beside this plugin dir. Resolve the repo root from THIS file
# rather than from the caller's cwd, because REINVENT is normally launched from the clone:
#   components/ -> reinvent_plugins/ -> plugins/ -> reinvent/ -> generators/ -> validation/ -> repo
#        [0]            [1]              [2]         [3]           [4]            [5]        [6]
_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.generators.reinvent.fixed_reward import build_provider  # noqa: E402


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters are always lists because a component can carry several endpoints; we use [0]."""

    reward_type: List[str]
    model_path: List[str]
    device: List[str]
    # Docking-only; optional so the surrogate configs need not carry them. REINVENT builds this
    # dataclass from the TOML's `params.*` keys, and a missing key must not be an error.
    # NOTE this is a PYDANTIC dataclass (`from pydantic.dataclasses import dataclass`), not the
    # stdlib one, so `dataclasses.field` is not in scope -- pydantic handles mutable defaults itself.
    oracle: List[str] = ("",)
    workdir: List[str] = ("",)
    norm: List[float] = (1.0,)


@add_tag("__component")
class GlueSurrogate:
    def __init__(self, params: Parameters):
        reward_type = params.reward_type[0]
        model_path = (params.model_path[0] or "").strip()
        device = (params.device[0] or "cpu").strip()

        if model_path and not Path(model_path).is_absolute():
            model_path = str(_REPO_ROOT / model_path)

        # Docking is reached across the env boundary (this env has no docking stack); for the
        # surrogates these extra arguments are ignored by build_provider.
        self.provider = build_provider(
            reward_type=reward_type,
            device=device,
            model_path=model_path or None,
            oracle=(params.oracle[0] or "").strip() or None,
            repo_root=str(_REPO_ROOT),
            norm=float(params.norm[0]),
            workdir=(params.workdir[0] or "").strip() or None,
        )
        self.reward_type = reward_type
        logger.info(f"GlueSurrogate: frozen {reward_type} reward on {device} (raw value)")

    def __call__(self, smilies: List[str]) -> ComponentResults:
        scores = self.provider.predict(list(smilies))
        return ComponentResults([np.array(scores, dtype=float)])
