"""SPARROW evaluator — the neutral, from-scratch library pricer (T1.4).

The cross-method HEADLINE: hand it the ordered library a strategy chose (SMILES), independently
recover a route for every molecule (AiZynth, standard USPTO/ZINC — T1.3), **merge** the routes into
one reaction network (T1.2), and let SPARROW's batch MILP pick the minimal reaction set that makes
the whole library with **shared intermediates built once** (T1.1 worker). The returned
``total_reactions`` is the from-scratch reactions-per-mode price applied *identically* to every
method — including route-less generators (S3-GFN, FragGFN) that have no DAG for count-once.

Two route sources (the benchmark's two SPARROW readings):
  * ``from_scratch`` (HEADLINE): re-route **every** molecule via AiZynth, ignoring native routes —
    the fair, uniform unit; hub-batching's shared scaffolds must survive as shared substructure that
    AiZynth+SPARROW recover on their own.
  * ``native`` (CHECK 1): use the reaction-GFN's by-construction routes (``LibrarySet.routes``);
    SPARROW prices those. Compared against DAG count-once (CHECK 2) at the T1.5 reconciliation gate.

Runs in the ``rgfn`` env; the two heavy stages cross conda boundaries by subprocess (AiZynth in the
``aizynth`` env via :mod:`~validation.lsdflow.eval.route_recovery`; the MILP in the ``sparrow`` env
via ``adapters/workers/sparrow_worker.py``). Per :class:`~validation.lsdflow.eval.base.Evaluator`.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .base import EvaluationResult, LibrarySet
from .network import build_network, canonical
from .route_recovery import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    RouteCache,
    env_python,
    recover_routes,
)

SPARROW_WORKER = "validation/lsdflow/adapters/workers/sparrow_worker.py"


@dataclass
class SparrowEvaluator:
    """Price an ordered :class:`LibrarySet` from scratch (or on native routes) via AiZynth→SPARROW.

    ``route_source`` = ``from_scratch`` (HEADLINE; AiZynth re-routes all) or ``native`` (CHECK 1;
    uses ``LibrarySet.routes``). ``cache`` (a :class:`RouteCache`) is required for ``from_scratch``
    so each molecule is routed once across the whole frontier. ``objective`` picks the MILP objective
    (``count`` default). AiZynth knobs mirror the synthesizability metric; ``price_table`` (optional)
    attaches building-block $ for the ``count_cost`` objective.
    """

    route_source: str = "from_scratch"  # "from_scratch" | "native"
    cache: Optional[RouteCache] = None
    objective: str = "count"
    sparrow_env: str = "sparrow"
    aizynth_env: str = "aizynth"
    aizynth_config: str = DEFAULT_CONFIG
    stock: str = "zinc"
    expansion: str = "uspto"
    filter_policy: Optional[str] = "uspto"
    time_limit: Optional[int] = 60  # per-molecule AiZynth search seconds
    nproc: int = 8
    max_seconds: int = 600  # per-snapshot MILP seconds
    work_dir: Path = field(default_factory=lambda: REPO_ROOT / "scratch_sparrow_eval")
    repo_root: Path = REPO_ROOT
    price_table: Optional[dict] = None
    strip_stereo: bool = True  # match the campaign's stereo-stripped cross-model key (proposal §6)
    _call: int = 0

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        self.repo_root = Path(self.repo_root)

    @property
    def name(self) -> str:
        return "sparrow" if self.route_source == "from_scratch" else "sparrow_native"

    def _gather_routes(self, library: LibrarySet):
        """Return ``(routes_by_canon, n_unsolved, route_recovery_seconds)`` for the library."""
        if self.route_source == "native":
            native = library.routes or {}
            routes = {}
            for s in library.smiles:
                c = canonical(s, self.strip_stereo)
                if c is None:
                    continue
                routes[c] = native.get(s) or native.get(c)
            n_unsolved = sum(1 for r in routes.values() if r is None)
            return routes, n_unsolved, 0.0
        # from_scratch: AiZynth re-routes everything (cached across the frontier).
        if self.cache is None:
            raise ValueError("SparrowEvaluator(route_source='from_scratch') requires a RouteCache")
        rr = recover_routes(
            library.smiles,
            self.cache,
            aizynth_env=self.aizynth_env,
            config=self.aizynth_config,
            stock=self.stock,
            expansion=self.expansion,
            filter_policy=self.filter_policy,
            time_limit=self.time_limit,
            nproc=self.nproc,
            work_dir=self.work_dir / "route_recovery",
            repo_root=self.repo_root,
        )
        return rr.routes, rr.n_unsolved, rr.timing_s

    def score(self, library: LibrarySet) -> EvaluationResult:
        routes_by_canon, n_unsolved, route_s = self._gather_routes(library)

        entries = []
        for s in library.smiles:
            c = canonical(s, self.strip_stereo)
            route = routes_by_canon.get(c) if c is not None else None
            if route is not None:
                entries.append({"smiles": s, "reward": library.rewards.get(s, 0.0), "route": route})

        if not entries:  # nothing routable -> unpriced (reported, never a silent 0)
            return EvaluationResult(
                total_reactions=None,
                per_tool={},
                timing_s={"route_recovery": round(route_s, 3)},
                n_priced=0,
                n_unsolved=library.n_modes,
                provenance={
                    "evaluator": self.name,
                    "route_source": self.route_source,
                    "reason": "no routable molecules",
                },
            )

        net = build_network(entries, price_table=self.price_table, strip_stereo=self.strip_stereo)

        call = self._call
        self._call += 1
        snap_dir = Path(self.work_dir) / f"snap_{call:05d}"
        tree, targets = net.write(snap_dir)
        out = snap_dir / "milp.json"
        cmd = [
            *env_python(self.sparrow_env),
            str(Path(self.repo_root) / SPARROW_WORKER),
            "--tree",
            str(tree),
            "--targets",
            str(targets),
            "--out",
            str(out),
            "--objective",
            self.objective,
            "--max-seconds",
            str(self.max_seconds),
        ]
        subprocess.run(cmd, check=True, cwd=str(self.repo_root))
        res = json.loads(out.read_text())

        total = res.get("total_reactions")
        n_selected = res.get("n_targets_selected", len(entries))
        return EvaluationResult(
            total_reactions=total,
            per_tool={
                "sparrow_mip": total if total is not None else 0,  # reactions are the MILP's output
                "n_reaction_nodes": net.stats.n_reaction_nodes,  # the merged network size
                "n_shared_compounds": net.stats.n_shared_compounds,  # the merge payoff
            },
            timing_s={
                "route_recovery": round(route_s, 3),
                "sparrow_build": res.get("build_s", 0.0),
                "sparrow_mip": res.get("solve_s", 0.0),
            },
            n_priced=n_selected,
            n_unsolved=n_unsolved,
            provenance={
                "evaluator": self.name,
                "route_source": self.route_source,
                "objective": self.objective,
                "milp_status": res.get("milp_status"),
                "solve_rate": round(len(entries) / library.n_modes, 4) if library.n_modes else None,
                "n_compound_nodes": net.stats.n_compound_nodes,
                "n_routeless_dropped": net.stats.n_routeless_targets,
            },
        )
