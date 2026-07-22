"""MultiAiZ evaluator — the second, smarter competitor route-planner (T4.1).

Where :class:`~validation.lsdflow.eval.sparrow.SparrowEvaluator` (from_scratch) routes every molecule
**independently** via AiZynth, this prices a library with **MultiAiZ** (`[ianez2026multiaiz]`): it runs
AiZynthFinder over the whole pool for ``n_iters`` cycles, appending discovered intermediates to the
stock each cycle so the pool's targets **converge on shared intermediates**. The convergent routes
(flattened + stitched down to real ZINC stock by the worker, so a shared intermediate is a reaction
PRODUCT built ONCE — see Logs/043) then feed the **same** SPARROW MILP the from-scratch headline uses.

PER-POOL ISOLATION (the experimental point): each ``score(library)`` call runs MultiAiZ on THAT
library's molecules alone — sharing discovered within hub-batching's pool never leaks to
best-candidate's or S3-GFN's. So this measures whether an acquisition function *selects molecules that
are mutually shareable*, priced by the strongest available planner.

Runs in the ``rgfn`` env; MultiAiZ crosses to the ``aizynth`` env by subprocess
(``adapters/workers/multiaiz_worker.py``) and the MILP to the ``sparrow`` env (``sparrow_worker.py``),
exactly like SparrowEvaluator. Per :class:`~validation.lsdflow.eval.base.Evaluator`.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .base import EvaluationResult, LibrarySet
from .network import build_network, canonical
from .route_recovery import DEFAULT_CONFIG, REPO_ROOT, env_python

MULTIAIZ_WORKER = "validation/lsdflow/adapters/workers/multiaiz_worker.py"
SPARROW_WORKER = "validation/lsdflow/adapters/workers/sparrow_worker.py"


@dataclass
class MultiAiZEvaluator:
    """Price an ordered :class:`LibrarySet` with per-pool MultiAiZ route discovery → SPARROW MILP.

    ``n_iters`` = MultiAiZ cycles (paper: 5). AiZynth knobs mirror the from-scratch SparrowEvaluator
    (same USPTO/ZINC chemistry, so the two competitor curves overlay). ``max_routes_per_target`` caps
    candidate routes fed to SPARROW (0 = all; SPARROW's MILP picks the max-sharing set)."""

    n_iters: int = 5
    objective: str = "count"
    aizynth_env: str = "aizynth"  # MultiAiZ lives here (Logs/043)
    sparrow_env: str = "sparrow"
    aizynth_config: str = DEFAULT_CONFIG
    stock: str = "zinc"
    expansion: str = "uspto"
    filter_policy: Optional[str] = "uspto"
    max_routes_per_target: int = 0
    max_seconds: int = 600  # per-snapshot MILP seconds
    work_dir: Path = field(default_factory=lambda: REPO_ROOT / "scratch_multiaiz_eval")
    repo_root: Path = REPO_ROOT
    price_table: Optional[dict] = None
    strip_stereo: bool = True  # match the campaign's stereo-stripped cross-model key (proposal §6)
    _call: int = 0

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        self.repo_root = Path(self.repo_root)

    @property
    def name(self) -> str:
        return "multiaiz"

    def _discover_routes(self, library: LibrarySet, snap_dir: Path):
        """Run MultiAiZ on this pool (isolated) → ``{target_canon: [route, ...]}`` + discovery seconds."""
        snap_dir.mkdir(parents=True, exist_ok=True)
        smi_path = snap_dir / "pool.smi"
        # dedup canonical pool SMILES (MultiAiZ is set-based; identical targets add nothing)
        seen, lines = set(), []
        for s in library.smiles:
            c = canonical(s, self.strip_stereo)
            if c and c not in seen:
                seen.add(c)
                lines.append(s)  # feed the original SMILES; the worker canonicalizes keys
        smi_path.write_text("\n".join(lines) + "\n")
        out_path = snap_dir / "multiaiz_routes.json"
        cmd = [
            *env_python(self.aizynth_env),
            str(Path(self.repo_root) / MULTIAIZ_WORKER),
            "--pool-smi", str(smi_path),
            "--out", str(out_path),
            "--config", str(self.aizynth_config),
            "--stock", self.stock,
            "--expansion", self.expansion,
            "--filter", self.filter_policy or "none",
            "--n-iters", str(self.n_iters),
            "--work-dir", str(snap_dir / "multiaiz_run"),
            "--strip-stereo", "true" if self.strip_stereo else "false",
            "--max-routes-per-target", str(self.max_routes_per_target),
        ]  # fmt: skip
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=str(self.repo_root))
        discovery_s = time.time() - t0
        routes_by_canon = json.loads(out_path.read_text()) if out_path.exists() else {}
        return routes_by_canon, discovery_s

    def score(self, library: LibrarySet) -> EvaluationResult:
        self._call += 1
        snap_dir = Path(self.work_dir) / f"snap_{self._call:05d}"
        routes_by_canon, discovery_s = self._discover_routes(library, snap_dir)

        # One entry per (target, candidate route); duplicate targets are fine — build_network unions
        # their reactions and SPARROW's MILP picks the min-reaction (max-sharing) set. n_priced counts
        # DISTINCT pool modes that got at least one route.
        entries = []
        priced_modes = set()
        for s in library.smiles:
            c = canonical(s, self.strip_stereo)
            routes = routes_by_canon.get(c) if c is not None else None
            if not routes:
                continue
            priced_modes.add(c)
            for route in routes:
                entries.append({"smiles": s, "reward": library.rewards.get(s, 0.0), "route": route})

        n_priced = len(priced_modes)
        if not entries:  # nothing routable -> unpriced (reported, never a silent 0)
            return EvaluationResult(
                total_reactions=None,
                per_tool={},
                timing_s={"route_discovery": round(discovery_s, 3)},
                n_priced=0,
                n_unsolved=library.n_modes,
                provenance={
                    "evaluator": self.name,
                    "reason": "no routable molecules",
                    "n_iters": self.n_iters,
                },
            )

        net = build_network(entries, price_table=self.price_table, strip_stereo=self.strip_stereo)
        tree, targets = net.write(snap_dir)
        out = snap_dir / "milp.json"
        cmd = [
            *env_python(self.sparrow_env),
            str(Path(self.repo_root) / SPARROW_WORKER),
            "--tree", str(tree),
            "--targets", str(targets),
            "--out", str(out),
            "--objective", self.objective,
            "--max-seconds", str(self.max_seconds),
        ]  # fmt: skip
        subprocess.run(cmd, check=True, cwd=str(self.repo_root))
        res = json.loads(out.read_text())

        total = res.get("total_reactions")
        return EvaluationResult(
            total_reactions=total,
            per_tool={
                "sparrow_mip": total if total is not None else 0,
                "n_reaction_nodes": net.stats.n_reaction_nodes,
                "n_shared_compounds": net.stats.n_shared_compounds,  # MultiAiZ's convergence payoff
            },
            timing_s={
                "route_discovery": round(discovery_s, 3),  # MultiAiZ (n_iters × |pool| AiZynth)
                "sparrow_build": res.get("build_s", 0.0),
                "sparrow_mip": res.get("solve_s", 0.0),
            },
            n_priced=n_priced,
            n_unsolved=library.n_modes - n_priced,
            provenance={
                "evaluator": self.name,
                "n_iters": self.n_iters,
                "objective": self.objective,
                "milp_status": res.get("milp_status"),
                "solve_rate": round(n_priced / library.n_modes, 4) if library.n_modes else None,
                "n_candidate_routes": len(entries),
                "n_compound_nodes": net.stats.n_compound_nodes,
                "n_shared_compounds": net.stats.n_shared_compounds,
            },
        )
