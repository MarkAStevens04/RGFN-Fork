"""The Evaluator seam for the LSD-Flow library-efficiency benchmark
(``docs/LSD_FLOW_BENCHMARK_PLAN.md`` §0 / T0.1).

The benchmark is three **orthogonal, pluggable stages** joined by two contracts::

    Generator ──pool──▶ Selection strategy ──ordered library──▶ Evaluator ──▶ (reactions, timing)

This module defines the **second contract** — the ordered library (:class:`LibrarySet`) a selection
strategy emits, and the :class:`Evaluator` that prices it (:class:`EvaluationResult`). Selection and
evaluation are decoupled *on purpose*: a strategy *tries* to be cheap using its own logic; the
evaluator *independently* prices whatever set the strategy actually chose. The evaluator scores the
set the strategy selected — never assume it is the reward-optimal or reaction-minimal subset.

**Three cost readings, one seam (the decision recorded in the benchmark design).**
- **HEADLINE — from-scratch SPARROW** (:mod:`validation.lsdflow.eval.sparrow`, T1.4): re-route *every*
  molecule independently (route recovery → batch MILP), ignoring native routes. One uniform unit that
  applies to **every** method, including route-less generators (S3-GFN, FragGFN) that have no DAG.
- **CHECK 1 — native-route SPARROW**: reaction-GFNs (SCENT/RGFN/RxnFlow) keep their by-construction
  routes (:attr:`LibrarySet.routes`); SPARROW prices those. Tests whether the flow's shared structure
  survives independent pricing.
- **CHECK 2 — DAG count-once** (:mod:`validation.lsdflow.eval.count_once`, T0.2): the strategy's OWN
  recoverable estimate (:attr:`LibrarySet.count_once_reactions`), the current campaign metric
  (Logs/033). Only defined over the reaction DAG; ``None`` for route-less libraries.

T1.5 (the reconciliation gate) asks whether CHECK 1 and CHECK 2 agree on a reaction-GFN library
*before* any cross-method plot is trusted. All three are the **same** ``Evaluator`` interface so the
frontier driver (T0.3) swaps them with ``--evaluator {count_once,sparrow,multiaiz}`` and reuses one
read-time slice logic.

Nothing here imports ``glue``/``rgfn`` or any heavy tool — it is a pure data/typing contract, so it
loads in any env (the SPARROW/AiZynth workers cross the conda boundary by subprocess, T1.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

# A "route" is the dict produced by ``glue/active_learning/route.py::extract_route`` (there is no
# ``Route`` class — the pipeline uses plain dicts / ``routes.jsonl`` records):
#   {product_smiles, num_reactions, building_block: {smiles, idx},
#    steps: [{step, reaction_idx, reaction_smarts, reactant, fragments: [...], product}]}
Route = Dict[str, object]


@dataclass
class LibrarySet:
    """An **ordered** library of selected modes handed to an :class:`Evaluator` (the plan's §0 middle
    arrow). The ordering is the selection order the strategy emitted (best-reward-first for
    best-candidate; hub-walk order for hub-batching); an evaluator prices *prefixes* of it, so the
    frontier reads a whole (modes, reactions) curve off one selection.

    Fields:
      * ``smiles`` — canonical SMILES, in selection order. ``len(smiles)`` **is** the mode count
        (the strategy only ever appends accepted modes), so a size-``k`` snapshot == ``k`` modes.
      * ``rewards`` — reward per SMILES (the generator's/oracle's value).
      * ``routes`` — each molecule's **native** synthesis route (the ``route.py`` dict schema) when the
        generator produced one (reaction-GFNs). ``None`` or a missing key ⇒ route-less (S3-GFN,
        FragGFN); the from-scratch evaluator route-recovers those (T1.3). Used by CHECK 1.
      * ``provenance`` — free-form run card: generator, strategy, τ (diversity cutoff), seed, target,
        reward gate, child_policy / prebuild_k, chemistry, … (§6). Never load-bearing for scoring;
        it is carried into ``EvaluationResult`` provenance for the results tables.
      * ``count_once_reactions`` — the strategy's OWN recoverable DAG count-once estimate for *this
        exact ordered set* (CHECK 2, Logs/033), attached by the driver from the ``CampaignResult``.
        ``None`` when the strategy computed no DAG estimate (route-less generators). This is the
        "old strategy produces its estimate in a recoverable way" hook: count-once is only
        well-defined over the reaction DAG (SCENT's fully-nested ``num_reactions`` + promoted-fragment
        closure), which ``routes`` does not carry — so we surface the strategy's number rather than
        re-deriving it. See :class:`~validation.lsdflow.eval.count_once.CountOnceEvaluator`.
    """

    smiles: List[str]
    rewards: Dict[str, float]
    routes: Optional[Dict[str, Route]] = None
    provenance: Dict[str, object] = field(default_factory=dict)
    count_once_reactions: Optional[int] = None

    @property
    def n_modes(self) -> int:
        """Library size == mode count (the strategy emits only accepted modes)."""
        return len(self.smiles)

    def has_routes(self) -> bool:
        """True iff every molecule carries a native route (a fully-routed reaction-GFN library)."""
        if not self.routes:
            return False
        return all(s in self.routes for s in self.smiles)

    def prefix(self, k: int) -> "LibrarySet":
        """The first ``k`` selected modes as a snapshot :class:`LibrarySet`.

        Slices ``smiles``/``rewards``/``routes`` but **drops** ``count_once_reactions`` (it is a
        whole-set scalar, not sliceable): the frontier driver attaches the correct per-prefix
        count-once value (``accepted[k-1].cum_reactions``) when it builds each snapshot.
        """
        keep = self.smiles[:k]
        keep_set = set(keep)
        return LibrarySet(
            smiles=list(keep),
            rewards={s: self.rewards[s] for s in keep if s in self.rewards},
            routes=(
                {s: self.routes[s] for s in keep if s in self.routes}
                if self.routes is not None
                else None
            ),
            provenance=dict(self.provenance),
            count_once_reactions=None,
        )


@dataclass
class EvaluationResult:
    """What an :class:`Evaluator` returns for one :class:`LibrarySet`.

    ``per_tool`` and ``timing_s`` are **always populated** (a hard requirement, not a nicety —
    debuggability, §T0-guardrails): they attribute the reaction count and wall-clock across the
    pipeline stages that produced them (e.g. ``aizynth_routing`` vs ``sparrow_mip``), which feeds the
    compute frontier (T2.2).

      * ``total_reactions`` — the headline number for this library (``None`` when undefined, e.g.
        count-once on a route-less library).
      * ``per_tool`` — reaction count attributed per pipeline tool/stage. Sums to
        ``total_reactions`` where meaningful; used for debugging + the honest end-to-end accounting.
      * ``timing_s`` — wall-clock per stage.
      * ``per_molecule`` — optional per-SMILES reaction attribution.
      * ``n_priced`` / ``n_unsolved`` — how many modes were actually priced vs dropped because no
        route was found (SPARROW/AiZynth fairness stat, T1.3). Default: all modes priced.
      * ``provenance`` — echoed run card (evaluator name, chemistry, solver, …).
    """

    total_reactions: Optional[int]
    per_tool: Dict[str, float] = field(default_factory=dict)
    timing_s: Dict[str, float] = field(default_factory=dict)
    per_molecule: Optional[Dict[str, float]] = None
    n_priced: Optional[int] = None
    n_unsolved: int = 0
    provenance: Dict[str, object] = field(default_factory=dict)

    def reactions_per_mode(self, n_modes: int) -> Optional[float]:
        """``total_reactions / n_modes`` — the benchmark's primary axis; ``None`` if unpriced or
        no modes. ``n_modes`` is the caller's (usually ``LibrarySet.n_modes``)."""
        if self.total_reactions is None or not n_modes:
            return None
        return self.total_reactions / n_modes


@runtime_checkable
class Evaluator(Protocol):
    """Prices an ordered :class:`LibrarySet` and returns an :class:`EvaluationResult`.

    Implementations: :class:`~validation.lsdflow.eval.count_once.CountOnceEvaluator` (CHECK 2),
    ``SparrowEvaluator`` (headline + CHECK 1, T1.4), ``MultiAizEvaluator`` (T4.1). Each carries a
    ``name`` matching the ``--evaluator`` CLI value so the driver can select and label it.
    """

    name: str

    def score(self, library: LibrarySet) -> EvaluationResult:
        ...


class NoOpEvaluator:
    """A well-formed stub (T0.1 acceptance): returns the library's carried count-once value (or
    ``None``) with empty attributions. Useful for wiring tests before a real pricer exists."""

    name = "noop"

    def score(self, library: LibrarySet) -> EvaluationResult:
        return EvaluationResult(
            total_reactions=library.count_once_reactions,
            per_tool={},
            timing_s={},
            per_molecule=None,
            n_priced=library.n_modes,
            provenance={"evaluator": self.name},
        )
