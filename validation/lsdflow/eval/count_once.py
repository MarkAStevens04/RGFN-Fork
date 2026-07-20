"""CHECK 2 — the DAG count-once evaluator (``docs/LSD_FLOW_BENCHMARK_PLAN.md`` T0.2).

Count-once (Logs/033) is the current campaign metric: a library's reaction cost = each molecule's
**assembly couplings** (``num_reactions − Σ nested build of every attached promoted fragment``) **+**
each distinct promoted dynamic-library fragment built **once** (closure under recipes). It is the
honest "a chemist synthesizes each shared piece once" accounting — the whole reason it exists is
SCENT's *dynamic library*, whose fragments are promoted, built once, and reused (SCENT's
``num_reactions`` is fully nested, Logs/033).

**This evaluator is a thin adapter, NOT an independent re-pricer.** Count-once is only well-defined
over the reaction DAG (the fully-nested ``num_reactions`` + promoted-fragment closure), which
``routes.jsonl`` does not carry — so re-deriving it from routes alone would silently drop SCENT's
nesting and fail to reproduce the committed numbers. Instead the campaign strategies
(``glue/samplers/lsdflow/campaign.py``, the single source of truth) compute it, and the frontier
driver attaches the value for each ordered snapshot as ``LibrarySet.count_once_reactions``; this
evaluator surfaces it through the common :class:`Evaluator` seam so count-once plots on the same
frontier as the from-scratch SPARROW pricer and T1.5 can compare the two on a reaction-GFN library.

Route-less libraries (S3-GFN, FragGFN — no DAG) carry ``count_once_reactions=None`` ⇒
``total_reactions=None``: count-once cannot score them, which is exactly why the benchmark needs the
route-agnostic SPARROW evaluator for cross-method comparison.

Kept ``glue``/``rgfn``-free (reads only the value the driver attached), so it loads in any env.
"""

from __future__ import annotations

from .base import EvaluationResult, LibrarySet


class CountOnceEvaluator:
    """Surface the strategy's recoverable DAG count-once estimate for a snapshot (CHECK 2).

    ``score`` returns the ``LibrarySet.count_once_reactions`` the driver attached (from the
    ``CampaignResult`` the campaign strategy produced), so ``--evaluator count_once`` reproduces the
    committed campaign numbers bit-for-bit when the frontier's snapshot schedule is every-mode (the
    default). ``per_tool``/``timing_s`` carry the count-once attribution; the real compute cost of
    *producing* the estimate (enumeration/reward-gen/flow-extract) is tracked separately by the
    campaign's measured compute-time accounting (Logs/039), not here.
    """

    name = "count_once"

    def score(self, library: LibrarySet) -> EvaluationResult:
        r = library.count_once_reactions
        return EvaluationResult(
            total_reactions=r,
            per_tool=({"count_once": r} if r is not None else {}),
            timing_s={},  # negligible read; measured compute-time lives in the campaign accounting
            per_molecule=None,
            n_priced=(library.n_modes if r is not None else 0),
            n_unsolved=(0 if r is not None else library.n_modes),
            provenance={"evaluator": self.name},
        )
