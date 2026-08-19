"""SynFormer entrant adapter (`[gao2025synformer]`) — the REACTION-AWARE, non-GFN baseline.

The other external entrants (S3-GFN, REINVENT 4, Saturn) are route-LESS: they emit molecule strings
and a planner recovers routes afterwards. SynFormer generates molecules AS SYNTHETIC PATHWAYS, so its
molecules carry a route by construction (``has_route=1``) exactly as ours do. That makes this the one
cell that separates the two things the headline otherwise conflates — is the advantage the **flow
field**, or merely **reaction-grounding**? No ablation on our own generators can answer it, and
``docs/LSD_FLOW_BENCHMARK_PLAN.md`` §7 names the gap explicitly.

Consequences: it skips MultiAiZ entirely (~2.25 h/pool the route-less entrants pay) and prices
through ``sparrow_select_frontier.py --route-source external`` — no recipe expansion either, because
every leaf is a purchasable catalogue block rather than something we must build.

- ``fixed_reward.py`` — the frozen reward providers, byte-for-byte the REINVENT/Saturn copies
  (AST-verified), so all entrants provably optimize the same function.
- ``route_convert.py`` — turns SynFormer's synthesis stacks into our ``steps`` schema. Patches
  ``StatePool.get_dataframe`` to carry the full tree, because the ``synthesis`` column alone records
  leaves and reactions but **not intermediates**, and a reaction network is a graph of intermediates.
- ``run_synformer_fixed.py`` — upstream's GraphGA-SF loop (Graph GA proposes, SynFormer projects,
  oracle scores) against our frozen reward, emitting a candidate dataset with routes.

Runs in the ``synformer`` prefix env; the pool crosses to ``rgfn`` by subprocess
(``scripts/ingest_candidates.py --routes``) — the same two-env pattern as every other baseline.
"""
