"""Route recovery for route-less molecules — the AiZynth cross-env bridge (T1.3).

The from-scratch SPARROW headline re-routes EVERY molecule independently, and route-less generators
(S3-GFN, FragGFN) have no native route at all. Both need a recovered synthesis route before the
SPARROW MILP can price the library. AiZynth does that, but it pins its own stack and lives in the
``aizynth`` conda env — so this module is the validation-side orchestrator (runs in the ``rgfn`` env
or any RDKit env) that shells out to the AiZynth route-recovery entrypoint we added to
``validation/harness/synthesizability.py`` (``--recover-routes``), mirroring the
``scripts/score_batch.py`` cross-env pattern.

**Caching is the point.** The frontier prices many snapshots (prefixes) of many orderings across
cutoffs, strategies, and seeds; the same molecule recurs constantly. AiZynth is by far the slowest
stage, so every molecule is routed **exactly once** and memoized to a persistent JSON cache keyed by
canonical SMILES. A run recovers only the molecules it hasn't seen.

The cache is **chemistry-specific**: a standard-USPTO/ZINC recovery and a glue-constrained recovery
(T1.5) produce different routes, so use a different ``cache_path`` per chemistry (the caller owns
this; the default encodes ``stock``+``expansion``).

Env-light: stdlib + RDKit (for canonicalization, via :func:`network.canonical`); the heavy AiZynth
work happens in the ``aizynth`` env behind ``conda run``.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

from .network import canonical

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "data/models/aizynthfinder/config.yml"
SYNTH_SCRIPT = "validation/harness/synthesizability.py"


def env_python(env_name: str):
    """Command prefix to run python in conda env ``env_name``. Prefers the env's python DIRECTLY
    (``.../envs/<env>/bin/python``) over ``conda run -n <env> python`` — the latter re-resolves the
    env on every call (~2.5-3s), which dominates when the evaluator spawns hundreds of MILP/AiZynth
    subprocesses; direct invocation is ~0.9s. Falls back to ``conda run`` if the path isn't found.
    Both benchmark envs (sparrow, aizynth) are pure-python/CPU, so they need no ``conda run`` env-var
    setup (verified). Returns a list to prepend to the python args."""
    import sys

    cur = Path(sys.executable)
    candidates = []
    for up in (2, 1):  # current interp is <base>/envs/<cur>/bin/python or <base>/bin/python
        if len(cur.parents) > up:
            candidates.append(cur.parents[up] / "envs" / env_name / "bin" / "python")
    candidates.append(Path.home() / "miniconda3" / "envs" / env_name / "bin" / "python")
    for c in candidates:
        if c.exists():
            return [str(c)]
    return ["conda", "run", "-n", env_name, "python"]


@dataclass
class RouteCache:
    """Persistent ``{canonical_smiles: {"solved": 0|1, "route": route|None}}`` memo (JSON).

    ``route`` is the ``network.py``-consumable dict (``{product_smiles, num_reactions, steps:[...]}``)
    or ``None`` when AiZynth could not solve the molecule (an *answered* miss — cached so we never
    re-attempt it). ``has(c)`` distinguishes "never attempted" from "attempted, unsolved"."""

    path: Path
    _data: Dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        self.path = Path(self.path)
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def has(self, canon: str) -> bool:
        return canon in self._data

    def get(self, canon: str) -> Optional[dict]:
        """The recovered route dict, or ``None`` (unsolved or never attempted)."""
        rec = self._data.get(canon)
        return rec.get("route") if rec else None

    def put(self, canon: str, solved: bool, route: Optional[dict]) -> None:
        self._data[canon] = {"solved": int(bool(solved)), "route": route}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data))
        tmp.replace(self.path)  # atomic

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class RecoveryResult:
    """Routes for the requested molecules + fairness/debug stats."""

    routes: Dict[str, Optional[dict]]  # canonical SMILES -> route dict or None (unsolved)
    n_requested: int = 0
    n_solved: int = 0
    n_unsolved: int = 0
    n_from_cache: int = 0
    n_newly_routed: int = 0
    timing_s: float = 0.0

    @property
    def solve_rate(self) -> float:
        return (self.n_solved / self.n_requested) if self.n_requested else 0.0


def recover_routes(
    smiles: Sequence[str],
    cache: RouteCache,
    *,
    aizynth_env: str = "aizynth",
    config: str = DEFAULT_CONFIG,
    stock: str = "zinc",
    expansion: str = "uspto",
    filter_policy: Optional[str] = "uspto",
    time_limit: Optional[int] = 60,
    iteration_limit: Optional[int] = None,
    nproc: int = 8,
    work_dir: Optional[Path] = None,
    repo_root: Path = REPO_ROOT,
) -> RecoveryResult:
    """Recover a route per molecule, using ``cache`` and routing only cache-misses via AiZynth.

    Returns a :class:`RecoveryResult` mapping canonical SMILES -> route dict (or ``None`` if AiZynth
    could not solve it), plus solve-rate / cache-hit stats. Unparseable SMILES are dropped. The heavy
    AiZynth call runs in ``aizynth_env`` via ``conda run`` (cross-env bridge); its per-molecule knobs
    (``config``/``stock``/``expansion``/``filter``/limits/``nproc``) mirror the synthesizability
    metric. ``work_dir`` holds the transient .smi / .jsonl exchange files (default: beside the cache).
    """
    t0 = time.perf_counter()
    canon = []
    seen = set()
    for s in smiles:
        c = canonical(s)
        if c is not None and c not in seen:
            seen.add(c)
            canon.append(c)

    misses = [c for c in canon if not cache.has(c)]
    n_newly = 0
    if misses:
        work = Path(work_dir) if work_dir else cache.path.parent
        work.mkdir(parents=True, exist_ok=True)
        smi_in = work / "recover_in.smi"
        jsonl_out = work / "recover_out.jsonl"
        smi_in.write_text("\n".join(misses) + "\n")
        cmd = [
            *env_python(aizynth_env),
            str(Path(repo_root) / SYNTH_SCRIPT),
            "--recover-routes",
            str(smi_in),
            "--routes-out",
            str(jsonl_out),
            "--config",
            config,
            "--stock",
            stock,
            "--expansion",
            expansion,
            "--filter",
            filter_policy or "none",
            "--nproc",
            str(nproc),
        ]
        if time_limit is not None:
            cmd += ["--time-limit", str(time_limit)]
        if iteration_limit is not None:
            cmd += ["--iteration-limit", str(iteration_limit)]
        print(
            f"[route_recovery] routing {len(misses)} new molecules via {aizynth_env} (cache has "
            f"{len(cache)})",
            flush=True,
        )
        subprocess.run(cmd, check=True, cwd=str(repo_root))
        for line in jsonl_out.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            c = canonical(rec.get("smiles"))
            if c is None:
                continue
            cache.put(c, bool(rec.get("solved")), rec.get("route"))
            n_newly += 1
        cache.save()

    routes = {c: cache.get(c) for c in canon}
    n_solved = sum(1 for r in routes.values() if r is not None)
    return RecoveryResult(
        routes=routes,
        n_requested=len(canon),
        n_solved=n_solved,
        n_unsolved=len(canon) - n_solved,
        n_from_cache=len(canon) - len(misses),
        n_newly_routed=n_newly,
        timing_s=time.perf_counter() - t0,
    )
