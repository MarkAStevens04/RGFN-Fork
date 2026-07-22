# SCENT sEH — T1.5 cost-model reconciliation gate (count-once vs native-route SPARROW), full coverage
**Date:** 2026-07-20, ~8:30pm

## Question

Do our two independent ways of counting a library's reactions — the fast internal "count-once"
estimate we've been reporting, and an external route planner's optimizer run on the molecules' actual
by-construction routes — agree closely enough that the reactions-per-mode comparisons can be trusted?

## Context & Summary

The whole benchmark ranks strategies by reactions per mode. We compute that number two ways: a cheap
internal **count-once** DAG estimate (Logs/033) that the frontier uses for speed, and — independently
— **SPARROW**, a published set-cover optimizer that, given each molecule's route, finds the minimum
number of distinct reactions to make the whole set. Before *any* comparison gets plotted, the plan's
reconciliation gate (T1.5) demands these two agree on the reaction-GFN's own library; otherwise a
reported "win" could be an artifact of how we count rather than a real chemistry saving. An earlier
smoke passed the gate but only at **59%** route coverage (the small run's `routes.json` was missing
routes for many modes, so the two counts were over different molecule sets). This entry runs the gate
at **full coverage**, using the full-scale SCENT re-run that emits routes for every molecule plus the
rebuilt hub enumeration. (That enumeration had to be repaired first: the enumerator crashed on an
early-terminate state — `ReactionStateEarlyTerminal` — and silently produced no output; a one-line
guard fixed it, and the 64-hub enumeration then completed cleanly.)

## Answer

The two counts agree to within ~5% at full coverage — the cost model is trustworthy. For both
selection strategies, count-once and native-route SPARROW land within 3–5% of each other on the same
100 modes, with **full (1.00) native-route coverage** this time. SPARROW consistently finds *slightly*
fewer reactions than count-once (it's a global optimizer, so it catches a bit more sharing), but the
gap is small and well inside the 10% tolerance. As a bonus this directly previews the T4.4 "what does
LSD-Flow leave on the table" question at this operating point: SPARROW shaves only ~3% off
hub-batching's own reaction count, so the flow-based selection is already near what an optimal planner
would do on the same molecules.

## Relevance to our Publication

A reviewer's sharpest objection to the headline is "you counted the reactions yourself — is the
advantage real or a bookkeeping choice?" This gate answers it: an independent, published optimizer,
run on the molecules' real routes, reproduces our count within 5%. It is the green light to plot the
comparisons at all, and it doubles as evidence that LSD-Flow's cheap flow-based selection isn't
leaving much on the table versus optimal set-cover.

## Next Experiments

**Refining for publication**
- **T4.4 full ablation:** formalize the count-once-vs-SPARROW gap across the whole τ sweep (not just
  cutoff 0.5) into the single "left on the table" number for the paper.

**Next steps in project**
- **T3.2:** S3-GFN onto the frontier — the route recovery for its pool is queued (job 71113, waiting
  on the maintenance window); then the S3-GFN best-candidate curve drops onto the headline axes.
- **T4.1:** MultiAiZ as a second, smarter competitor planner (does iterative shared-intermediate
  discovery beat plain AiZynth→SPARROW, or does the MILP already capture the sharing?).

---

# Re-creation

### Relevant Files

Root: repository root unless noted. `/…` = absolute (on `$SCRATCH`).

**Scripts**
- `./experiments/lsd_hubs/campaign/reconcile_t15.py` — the gate. Loads the reaction-GFN's candidate
  pool + hub enumeration + native `routes.json`, builds each strategy's library at a cutoff, prices it
  BOTH ways (count-once and native-route SPARROW over the by-construction routes), reports `rel_diff`
  per strategy, and passes if the worst `rel_diff` ≤ tolerance. `--min-coverage` excludes any strategy
  whose native routes don't cover its modes (so a coverage gap isn't misread as a unit gap).
- `./validation/lsdflow/adapters/workers/scent_worker.py` — the enumerator. **Fix this session:** the
  DFS now skips terminal / early-terminal states (guard `isinstance(state, (RSA, RSB, RSC))`) instead
  of calling `get_forward_action_spaces` on a `ReactionStateEarlyTerminal` (KeyError → the 200-hub
  enum crashed with no output in job 70974). The re-run (71010) then wrote all 64 hubs.
- `./experiments/lsd_hubs/campaign/submit_scent_seh_native.sh` — now RESUMABLE (skips sample/pick if
  present; re-runs just the enum) with a hard guard that `enum_children.json` was written.
- `./validation/lsdflow/eval/{sparrow.py,network.py}` + `.../workers/sparrow_worker.py` — native-route
  SparrowEvaluator + route→tree builder + CBC MILP (sparrow env).

**Datasets** (`/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_native_70974/`)
- `sample/{routes.json,records.csv,compositions.json}` — 30k-trajectory sample WITH native routes
  (26,069 candidates; 55,095 native routes).
- `enum/enum_children.json` — the rebuilt 64-hub enumeration (169,230 children; job 71010).
- snapshot: `…/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json`
  (`smiles_to_route` recipes + 1,600 promoted fragments).

**Results** — `./experiments/lsd_hubs/campaign/results/scent_seh_recon_full_reconcile/`.

### Relevant Versions

Uncommitted (branch `Hub-Analysis`). To commit: the `scent_worker.py` enum fix, the resumable
`submit_scent_seh_native.sh`, the reconcile result dir, and this log. `[TODO — add commit hash]`

### Relevant Resources

**Sources** — SPARROW (route-aware library-selection MILP). **Packages** — `sparrow` env (PuLP/CBC);
`rgfn` env runs `reconcile_t15.py` (imports `glue`→dgl).

### Method

1. Repair + rebuild the enum: fix `scent_worker.py`, re-run the enumerate step (job 71010, reusing the
   existing sample/hubs) → `enum_children.json`, 64 hubs, 0 KeyError.
2. Gate: `python experiments/lsd_hubs/campaign/reconcile_t15.py --recon-dir …/scent_seh_native_70974
   --snapshot …/fragments_4000.json --reward-threshold 7.0 --cutoff 0.5 --budget-modes 100
   --tag scent_seh_recon_full` (rgfn env + CUDA libs; SPARROW MILP is CPU, ~0.1 s/solve).

### Results

| strategy | modes | native-route coverage | count-once | native-route SPARROW | rel_diff | MILP status |
|---|---|---|---|---|---|---|
| best-candidate | 100 | 1.00 | 314 | 299 | 4.8% | Optimal |
| hub-batching | 100 | 1.00 | 288 | 279 | 3.1% | Optimal |

**GATE PASSED** — worst `rel_diff` = 4.8% ≤ 10% tolerance. SPARROW finds slightly more sharing than
count-once for both strategies (299<314, 279<288); the gap is within tolerance and consistent, not a
unit error. Full (1.00) native-route coverage for both (vs 59% at the earlier smoke scale).
