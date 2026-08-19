"""Stub the heavy third-party deps of Saturn oracles we never call, so its registry can import.

WHY THIS IS NEEDED. ``oracles/utils.py`` imports **every** oracle module eagerly — that is how
``construct_oracle_component``'s if/elif dispatch table gets its classes — and ``oracles.oracle``
imports it in turn. So merely constructing Saturn's ``Oracle`` drags in the GEAM docking oracle,
which does ``from openbabel import pybel`` at module level. We use exactly one oracle component (our
own frozen sEH / DRD2 reward); installing openbabel to satisfy an import we never execute would add a
large conda dependency whose GLIBCXX_3.4.29 failure Saturn's own README documents a workaround for.

This mirrors ``validation/generators/s3gfn/run_s3gfn_fixed.py::_stub_unidock_vina``, which stubs
``rxnflow.tasks.unidock_vina`` for precisely the same reason.

**The stubs raise on use, they do not no-op.** A stub that silently returned would let a
mis-specified config quietly run a fake docking oracle and report the result as a number. If any of
these is ever actually needed, the traceback says so and names the fix.
"""

import sys
import types
from typing import List

# Modules that only the oracle components we never construct depend on. Keep this list minimal and
# justified — every entry is a dependency deliberately left uninstalled, not a workaround for a
# broken env.
_STUBBED = {
    # openbabel: `oracles/docking/geam_oracle.py` (GEAM's docking oracle, a different benchmark's
    # setup). Saturn's setup.sh installs it from conda-forge; we skip it.
    "openbabel": ["pybel"],
}


def _raising_module(name: str, attr: str) -> types.ModuleType:
    mod = types.ModuleType(name)

    def _explode(*_a, **_k):
        raise RuntimeError(
            f"{name}.{attr} is STUBBED by validation/generators/saturn/_stubs.py — this adapter "
            f"never calls the Saturn oracle that needs it. If you genuinely need it, install the "
            f"real dependency ({name.split('.')[0]}) into the saturn env and remove it from _STUBBED."
        )

    setattr(mod, attr, _explode)
    return mod


def stub_unused_oracle_deps() -> List[str]:
    """Install the stubs into ``sys.modules``. Call BEFORE importing anything under ``oracles``.

    Returns the names actually stubbed — a real install always wins, so this is a no-op for any
    dependency that is genuinely present."""
    stubbed = []
    for pkg, submods in _STUBBED.items():
        try:  # a real installation always takes precedence over the stub
            __import__(pkg)
            continue
        except ImportError:
            pass
        parent = types.ModuleType(pkg)
        for sub in submods:
            child = _raising_module(f"{pkg}.{sub}", sub)
            setattr(parent, sub, child)
            sys.modules[f"{pkg}.{sub}"] = child
        sys.modules[pkg] = parent
        stubbed.append(pkg)
    return stubbed
