"""Name -> adapter registry (``docs/LSD_FLOW_PROPOSAL.md`` §3b, §4b, §10).

RGFN is native to this (``rgfn``) env and runs **in-process** (:class:`RGFNAdapter`). The other
three targets live in mutually-incompatible envs and are reached over a subprocess/RPC bridge
(the ``scripts/score_batch.py`` shape) by a per-env worker; those are declared here with their
env + build status so an un-wired target fails with a clear message rather than a vague import
error. Build order (§10): RGFN now (step 1-2); SCENT once the ModuleList-patched retrain lands
(step 3, §9); FragGFN control + RxnFlow robustness last (step 4).
"""

from __future__ import annotations

from typing import Dict, List

from validation.lsdflow.adapters.base import GFNAdapter

# Per-target descriptor: conda env + how it is reached + current build status (§4b table).
ADAPTERS: Dict[str, Dict[str, str]] = {
    "rgfn": {"env": "rgfn", "kind": "in_process", "status": "live (anchor, §10 step 1)"},
    "scent": {
        "env": "scent",
        "kind": "subprocess",
        "status": "phase 3 — needs patched retrain (§9)",
    },
    "fraggfn": {"env": "fraggfn", "kind": "subprocess", "status": "phase 4 — control (§4b)"},
    "rxnflow": {"env": "rxnflow", "kind": "subprocess", "status": "phase 4 — robustness (§4b)"},
}


def available() -> List[str]:
    return sorted(ADAPTERS)


def get_adapter(name: str, **kwargs) -> GFNAdapter:
    """Instantiate the adapter for ``name``. Only the in-process RGFN anchor is wired in v1."""
    if name not in ADAPTERS:
        raise KeyError(f"Unknown adapter {name!r}. Known: {available()}")
    descriptor = ADAPTERS[name]
    if descriptor["kind"] == "in_process" and name == "rgfn":
        from validation.lsdflow.adapters.rgfn_adapter import RGFNAdapter

        return RGFNAdapter(**kwargs)
    raise NotImplementedError(
        f"Adapter {name!r} ({descriptor['status']}) is not wired yet. It runs in the "
        f"'{descriptor['env']}' env over the subprocess bridge; build its worker under "
        f"validation/lsdflow/adapters/workers/ per §4b when its phase begins."
    )
