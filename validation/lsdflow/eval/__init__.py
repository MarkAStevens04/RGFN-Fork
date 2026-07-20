"""LSD-Flow library-efficiency **evaluators** (``docs/LSD_FLOW_BENCHMARK_PLAN.md`` Phase 0–1).

The neutral, cross-method pricing seam: a selection strategy emits an ordered :class:`LibrarySet`;
an :class:`Evaluator` independently prices it into an :class:`EvaluationResult` (total reactions +
per-tool/timing attribution). One interface, swapped by the frontier driver via
``--evaluator {count_once,sparrow,multiaiz}``:

- :class:`CountOnceEvaluator` — CHECK 2, the strategy's own DAG count-once estimate (Logs/033).
- ``SparrowEvaluator`` — headline (from-scratch) + CHECK 1 (native-route); T1.4, planned.
- ``MultiAizEvaluator`` — smarter route discovery feeding the same SPARROW MILP; T4.1, planned.

Env-light on purpose (no ``glue``/``rgfn``/heavy-tool imports at package import); the SPARROW/AiZynth
workers run in their own conda envs, reached by subprocess.
"""

from .base import (  # noqa: F401
    EvaluationResult,
    Evaluator,
    LibrarySet,
    NoOpEvaluator,
    Route,
)
from .count_once import CountOnceEvaluator  # noqa: F401
from .network import (  # noqa: F401
    NetworkStats,
    ReactionNetwork,
    build_network,
    canonical,
    canonical_reaction,
    expand_route_with_recipes,
    load_price_table,
)

__all__ = [
    "EvaluationResult",
    "Evaluator",
    "LibrarySet",
    "NoOpEvaluator",
    "Route",
    "CountOnceEvaluator",
    "ReactionNetwork",
    "NetworkStats",
    "build_network",
    "canonical",
    "canonical_reaction",
    "expand_route_with_recipes",
    "load_price_table",
]
