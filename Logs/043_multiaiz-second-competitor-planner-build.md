# MultiAiZ — the second competitor route-planner (T4.1): build + convergent-route pricing design
**Date:** 2026-07-22, ~1pm

## Question

Can we give the non-reaction competitor the *strongest* post-hoc route planner — one that actively
discovers shared intermediates across a set of molecules — and does the reaction-GFN + hub-batching
still win on reactions-per-mode when the competitor is handed it?

## Context & Summary

The LSD-Flow headline (docs/LSD_FLOW_BENCHMARK_PLAN.md) claims a reaction-GFN with hub-batching beats
any other way of assembling a *library*, even when the competitors are given state-of-the-art batch
retrosynthesis planners — the plan names two: SPARROW (already built, T1.4) and **MultiAiZ** (T4.1,
this entry). Plain AiZynth routes each molecule independently, so it can miss sharing that a smarter
planner would find. MultiAiZ (`[ianez2026multiaiz]`, `MolecularAI/multiaiz`) is exactly that smarter
planner: it runs AiZynthFinder over a *set* of targets for N cycles, appending each cycle's discovered
intermediates to the stock so later targets converge on shared intermediates. This entry stands it up
as `--evaluator multiaiz` and settles the one design point that decides whether its numbers mean
anything: how to price its convergent routes so a shared intermediate is **built once**, not free.

The key experimental-design choice (agreed with the PI): each acquisition function's accepted-mode
pool is priced **in isolation** — MultiAiZ runs on hub-batching's molecules alone, and the shared
intermediates it finds there are *not* made available when pricing best-candidate's or S3-GFN's pools.
That is the whole point: we are measuring whether an acquisition function *selects molecules that are
mutually shareable*, so each pool must stand or fall on its own sharing.

## Answer

*(Build + design entry — the headline numbers come once the nodes are back and the coarse-cutoff run
finishes; this records what we built and why so the run is interpretable.)* MultiAiZ integrates
cleanly: it reuses our existing AiZynth env + USPTO/ZINC models, emits standard AiZynthFinder route
trees, and those feed the *same* SPARROW MILP the from-scratch headline uses. The one thing that had
to be gotten right — and the reason this is a design entry, not just a wiring one — is that MultiAiZ
makes routes terminate at discovered intermediates as if purchasable; we reconstruct each route down
to real ZINC stock so the shared intermediate is a reaction product SPARROW builds once and amortizes.

## Relevance to our Publication

This is the second half of the headline's fairness argument. A reviewer will ask "did you give the
baseline the *best* planner, or just vanilla AiZynth?" MultiAiZ is the answer — a published,
AstraZeneca multi-target planner whose entire purpose is finding shared intermediates. If the
reaction-GFN still wins reactions-per-mode when its competitor gets MultiAiZ→SPARROW, the
"reaction-MDP-is-necessary-for-library-economics" claim is airtight. It also answers the plan's own
sub-question: does smarter *discovery* lower reactions, or does SPARROW's MILP already extract the
sharing from plain AiZynth routes?

## Next Experiments

**Refining for publication**
- Run MultiAiZ→SPARROW per pool at coarse cutoffs (τ = 0.3/0.5/0.7) first; expand to the full τ sweep
  once preliminary numbers look right (MultiAiZ is `n_iters × |pool|` AiZynth searches, so cost scales
  with sweep density).
- Report MultiAiZ→SPARROW vs plain AiZynth→SPARROW side by side (the "does smarter discovery help"
  number) alongside the reaction-GFN's hub-batching curve.

**Next steps in project**
- T4.2 (RGFN + RxnFlow hub-batching curves), T4.3 (non-sEH targets), T4.4 (leave-on-the-table
  ablation), T4.5 (replicates).

---

# Re-creation

### Relevant Files

Root: repository root unless noted.

**Scripts**
- `./external/setup_multiaiz.sh` — installs MultiAiZ **into the existing `aizynth` env** (see decision
  below), reusing our USPTO/ZINC models.
- `./validation/lsdflow/adapters/workers/multiaiz_worker.py` — runs in `aizynth` env: one pool of
  SMILES → `MultiAiZ(finder, targets, out).run(n_iters=5)` (post-processing guarded) → per target,
  its candidate route trees **flattened + stitched to ZINC** → JSON `{target: [route, ...]}`.
- `./validation/lsdflow/eval/multiaiz.py` — `MultiAiZEvaluator` (`--evaluator multiaiz`): per pool,
  bridge to the worker → `build_network` (merge, dedup by canonical SMILES) → the same
  `sparrow_worker` MILP → `EvaluationResult`. Reuses SPARROW's pricing stage wholesale.
- `./validation/lsdflow/eval/network.py` — unchanged; consumes the worker's flat `{product,reactants}`
  steps and marks a compound buyable iff it is never a product (→ shared intermediates built once).

**Models/Data** — `data/models/aizynthfinder/{uspto_*.onnx, zinc_stock.hdf5, config.yml}` (reused).
**Upstream** — `external/multiaiz/` (git-ignored clone, `MolecularAI/multiaiz`).

### Relevant Versions

Uncommitted (branch `Hub-Analysis`). To commit: `external/setup_multiaiz.sh`, the worker + evaluator,
the `sweep_campaign.py` wiring (multiaiz off the planned list), the `[ianez2026multiaiz]` bib/README
entries, this log. `[TODO — add commit hash]`

### Relevant Resources

**Sources** — `[ianez2026multiaiz]` MultiAiZ (AILSCI 2026; DOI 10.1016/j.ailsci.2026.100175);
`[fromer2023sparrow]` SPARROW; `[genheden2020aizynth]` AiZynthFinder.
**Packages** — `multiaiz` (installed `--no-deps` into `aizynth` env) + aizynthfinder 4.4.1; `sparrow`
env (PuLP/CBC) for the MILP; `rgfn` env runs the evaluator (imports `glue`).

### Method / key decisions

1. **Env: reuse `aizynth`, not a fresh env.** A separate `multiaiz` env (repo `env-dev.yml` + poetry)
   failed: poetry tried to source-build the legacy `cgrtools 4.1.35` (Cython PEP-517 error). Our
   `aizynth` env already has **aizynthfinder 4.4.1** (satisfies MultiAiZ's `^4.4.0`, needs no cgrtools)
   and every API MultiAiZ imports. So MultiAiZ installs `--no-deps` into `aizynth` (+ `adjustText`).
   MultiAiZ *is* an AiZynth extension sharing its exact stack, so this is its natural home.
2. **n_iters = 5** — the paper's value ("over 5000 molecules ... in ~15 h for five cycles"); also the
   repo default. ~10.8 s/target for a full 5-cycle run → ~50 min for a 300-mol pool.
3. **Post-processing guarded.** MultiAiZ's final *intermediate-scoring* step needs a "reaction
   class-rank score" scorer our config doesn't load, and crashes — but it runs AFTER the per-cycle
   route trees are written, and we only need the routes, so the worker catches it and reads the trees.
4. **Convergent-route pricing (THE design point).** MultiAiZ appends discovered intermediates to the
   stock, so a route can end at a discovered intermediate X as an `in_stock` LEAF. Priced as-is, X
   would be a **free** buyable and the convergent saving would read as zero. Fix: harvest every
   molecule's build-subtree from the cycles' trees, then **flatten each target's route to flat
   `{product,reactants}` steps, expanding any discovered-intermediate leaf back into its build subtree
   (down to real ZINC stock)**. `build_network` then sees X as a reaction product → non-buyable → the
   SPARROW MILP **builds X once and amortizes it** across the pool's targets that share it. Base ZINC
   blocks (never a product) stay buyable. We feed **all** candidate routes per target and let SPARROW's
   MILP pick the max-sharing combination (no per-target selection heuristic — that's what SPARROW is
   for; `build_network` tolerates duplicate targets, unioning their reactions).
5. **Per-pool isolation:** the evaluator runs MultiAiZ separately per library, so sharing found in one
   acquisition function's pool never leaks to another's.

### Results

**Pipeline built + validated end-to-end** (login node; per-pool coarse-cutoff *headline numbers*
pending the compute nodes' return). Two-target smoke (benzamide pair sharing aniline, `n_iters=2`)
through the full `--evaluator multiaiz` path — worker (MultiAiZ in `aizynth` env, post-processing
guarded) → flatten+stitch to ZINC → `build_network` merge → SPARROW MILP:

| metric | value |
|---|---|
| total_reactions | 2 (each target its own amide coupling; shared aniline bought once) |
| MILP status | Optimal |
| n_priced / n_unsolved | 2 / 0 (solve_rate 1.0) |
| n_candidate_routes → n_reaction_nodes | 12 → 12 (SPARROW picked 2) |
| n_shared_compounds | 3 |
| route_discovery / MILP time | 18.9 s / 0.06 s |

Worker on the same pair: 2 targets routed, 12 candidate routes, flat `{product, reactants}` steps,
both sharing `Nc1ccccc1`. Compile-clean across all files; `--evaluator multiaiz` off the planned list
and live in `sweep_campaign.py` + `s3gfn_frontier.py`. Full per-pool reactions-per-mode numbers + the
MultiAiZ-vs-AiZynth comparison (does smarter discovery help) follow once nodes are back.
