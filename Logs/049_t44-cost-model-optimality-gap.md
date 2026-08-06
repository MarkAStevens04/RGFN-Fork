# T4.4 — how much does our cost estimate leave on the table? (and what DRD2 needs to be auditable)

**Date:** 2026-07-27, ~2pm

## Question

When our fast cost estimate says a library needs N reactions, how close is that to the fewest
reactions anyone could possibly get away with, given the same molecules and the same recipes?

## Context & Summary

Our headline number is *reactions per distinct molecule*, and we compute it with a fast in-house
"count-once" rule: walk every selected molecule's recipe and charge each distinct reaction once, so a
step shared by two molecules is paid for once. That estimate is what every curve in the benchmark is
plotted from, which invites an obvious challenge — *is the estimate flattering itself?* The honest
check is to hand the very same molecules and recipes to an independent global optimizer (SPARROW's
integer program, which provably finds the cheapest possible combination) and see how much cheaper it
can make the same library. Entry [042] built this machinery as a pass/fail gate before any curve was
allowed to be plotted; this entry turns it into the milestone's *number*, and tries to extend it from
sEH to a second target (DRD2) now that those cells are merged in.

## Answer

**Our estimate is within 3.1% of provably optimal, and hub-batching sits closer to the optimum than
best-candidate does (3.1% vs 4.8%).** On the sEH library, at full recipe coverage and with the
optimizer reporting a proven optimum, hub-batching's 288 estimated reactions against an optimal 279
means the flow-based selection leaves only 9 reactions on the table out of 288 — so essentially none
of hub-batching's advantage is an artifact of our own accounting, and the selection is *nearer* to
what a global optimizer would achieve than the baseline is. **The attempt to repeat this on DRD2 was
correctly refused by the gate, because that cell is missing an input the audit requires, not because
either cost model is wrong.** Our estimate charges for building each of SCENT's promoted
library fragments; the optimizer can only charge for them if it is given their recipes — and the DRD2
snapshot contains **no recipes at all** (0 of 1600 promoted fragments, versus 1600 of 1600 for sEH,
whose snapshot came from a dedicated recipe-logging re-run). With no recipes the optimizer silently
treats every promoted fragment as something you simply buy, so the two sides were pricing different
assumptions rather than disagreeing about arithmetic. DRD2 can be brought in **without retraining**: a
recipe-carrying, fully-trained DRD2 checkpoint from the same job-70180 re-run already exists, so what
is missing is only a sample+enumerate pass against it (see Next Experiments). **The DRD2 count-once
results already in the benchmark are unaffected**, because count-once needs only the per-fragment
reaction counts, which that snapshot does have.

## Update (2026-08-04) — DRD2 audited, gate passed on the second target

The predicted sample+enumerate pass ran (job **72145**, native run
`scent_drd2_native_72145`) against the recipe-carrying `scent_drd2/2026-07-10_17-28-06` checkpoint —
**no retrain**, exactly as scoped below. Recipe coverage came back **100.0% (1600/1600)**, so the
guard passed and the audit is valid. Result, on 100 modes at full (1.00) native-route coverage, MILP
status **Optimal** on both strategies:

| target | strategy | count-once | SPARROW optimum | gap |
|---|---|---|---|---|
| sEH | hub-batching | 288 | 279 | 3.12% |
| sEH | best-candidate | 314 | 299 | 4.78% |
| **DRD2** | **hub-batching** | **117** | **117** | **0.00%** |
| **DRD2** | best-candidate | 364 | 346 | 4.95% |

**The finding generalizes and, on DRD2, sharpens: hub-batching hits the global optimum exactly (117 =
117, 0.00% gap), while best-candidate leaves 4.95% on the table.** So on both independent targets
hub-batching is nearer the achievable floor than the baseline, and the earlier worry that the 3.1% was
a single-target artifact is retired. T4.4 can now be quoted as a range: **hub-batching 0.0–3.1% of
optimal, best-candidate 4.8–5.0%.**

Two mechanical notes for reproduction. (1) The first attempt (job **71800**) crashed in 15 s: the TDC
DRD2 oracle self-downloads to a relative `./oracle` dir, and Balam compute nodes have a read-only
`$HOME` (and no internet), so `os.mkdir('./oracle')` raised `PermissionError`. sEH never hits this
(frozen proxy checkpoint, no TDC download). Fix = pre-stage `external/scent/oracle/drd2_current.pkl` on
the login node so the worker takes TDC's "found local copy" branch; it is a gitignored cache like the
sEH proxy. (2) This is a *different same-scale run* than matrix16's `scent_drd2` cell (native resample
off the 2026-07-10 checkpoint, 100-mode audit), the same caveat that already applies to the sEH 3.1%.
`results/scent_drd2_native_reconcile/`.

## Relevance to our Publication

T4.4's acceptance criterion was "a single number for the gap", and **3.1%** is it, with the strongest
possible qualifier: the optimizer proved optimality, so this is a bound rather than an estimate. This
pre-empts the sharpest methodological objection available to a reviewer — that a paper reporting a
cost advantage under its own cost model has graded its own homework. It also converts a potential
weakness into a second finding: hub-batching is not merely cheaper than best-candidate, it is closer
to the achievable floor, which is what one would expect if the flow field is genuinely informative
about shared structure. The scope is honestly sEH-only, and the reason is now a known, fixable data
gap rather than an open question about the cost model.

## Next Experiments

**Refining for publication**
- **Audit DRD2 off the recipe-carrying checkpoint that ALREADY EXISTS — a sample+enumerate pass, no
  retraining.** A snapshot audit across every SCENT run settles the scope (table below):
  `scent_drd2/2026-07-10_17-28-06` (the same job-70180 recipe re-run that made sEH auditable) carries
  **1600/1600** recipes and is trained to **epoch 4999**, i.e. the same 5,000-iteration scale as
  matrix16's cell. Recipes are captured from observed trajectories at run time
  (`recipe_logging.capture_routes`), so they cannot be reconstructed post-hoc for the recipe-less
  `scent_drd2_5k/seed42` snapshot — but they need not be. Run `submit_scent_seh_native.sh`'s recipe
  with `CKPT`/`CFG`/`REWARD` pointed at the 2026-07-10 DRD2 checkpoint to produce a
  `scent_drd2_native_*` run, then reconcile that. Enumeration is the long pole.
- **Do NOT mix a snapshot with a different checkpoint's sample/enum.** The snapshot defines the frozen
  library the model sampled under; pairing the 2026-07-10 recipes with the 5k/seed42 enumeration would
  silently mis-cost molecules against a promoted-fragment set they were never built from.
- **Add a loud guard to the reconciler.** A snapshot with zero recipes should abort with "this cell
  cannot be audited" instead of running and reporting a 62.7% unit gap that looks like a cost-model
  defect. The gate caught it, but only after a full run and a misleading-looking number.
- Once DRD2 passes, quote the gap as a range across targets rather than a single sEH number.
- **Separate observation worth following up:** 97 of the 100 accepted DRD2 modes came from a *single*
  depth-0 hub (4-fluorobenzoic acid — a bought base fragment), each one reaction away from it. That is
  a statement about how loose the DRD2 gate (reward > 0.5) is, not about hub-batching, and it deserves
  a look alongside the gate-sweep work.

**Next steps in project**
- Keep count-once as the plotting model (it is the one validated here) and treat native-route SPARROW
  as the audit, exactly as [042] framed it.
- The remaining generality work (the two RGFN cells, jobs `71766`/`71767`) is queued and needs no
  action; the FragGFN cap-6 re-run remains the other open cell.

# Re-creation

### Relevant Files

**Scripts**
- `./experiments/lsd_hubs/campaign/reconcile_t15.py` — the reconciler: selects a library at a given
  reward gate + diversity cutoff, prices it with count-once, re-prices the *same* molecules and
  recipes through the SPARROW MILP (`sparrow` env, by subprocess), and reports `rel_diff` against a
  tolerance. Unchanged for this entry; only invoked on a new cell.
- `./glue/samplers/lsdflow/campaign.py` (`HubBatchingStrategy._accept`, ~L449) — the count-once unit:
  per accepted child, `1` final coupling + each *newly built* promoted fragment's
  `cost_table.unit_reactions`, plus the shared hub's own build charged once. This is the side that
  charges fragment builds.
- `./validation/lsdflow/eval/network.py` (`expand_route_with_recipes`) — the matching side: expands a
  promoted fragment into its build steps for SPARROW **using the snapshot's `smiles_to_route`**. With
  no recipes, nothing expands and the fragment enters the network as purchasable.

**Inputs**
- sEH: `/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_native_70974/` + snapshot
  `.../fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json`
  (**carries `smiles_to_route`** — this is why the sEH audit is possible at all).
- DRD2: `/scratch/markymoo/rgfn_runs/lsdflow/matrix16/scent_drd2/` (merged matrix16 cell; artifact
  shape verified, per-child `reaction` present) + snapshot
  `.../fixed_reward/scent_drd2_5k/seed42/additional_fragments/fragments_4000.json` — matched to the
  checkpoint named in `matrix16/manifest.csv`, but **has no `smiles_to_route` key**.

**Results (committed)**
- `./experiments/lsd_hubs/campaign/results/scent_seh_recon_full_reconcile/` — the sEH gate pass
  (entry [042]); the source of the 3.1% headline.
- `./experiments/lsd_hubs/campaign/results/scent_drd2_recon_reconcile/` — the DRD2 refusal.

### Relevant Versions

Branch `Hub-Analysis`, post-merge tree **`b7b72dd`** (the merge of `matrix16-lsdflow`, which supplied
the DRD2 cell). The DRD2 reconcile outputs + this log are uncommitted at time of writing.
`[TODO — add commit hash after pushing]`

### Relevant Resources

**Sources** — entry [042] (the T1.5 gate that built this machinery, and the sEH numbers reused here);
entry [048] (the matrix16 merge that supplied the DRD2 cell); entry [024]/recipe-logging (job 70180 —
the re-run that made sEH's snapshot auditable). SPARROW (MILP route selection) as the referee.

**Packages** — SPARROW + PuLP/CBC in the `sparrow` env (the MILP); `rgfn` env for the selection side
(imports `glue`→dgl, so a login-node run needs `source ~/bin/rgfn-smoke-env.sh` for the CUDA libs, or
it dies in `dgl.graphbolt`).

### Method

Login node, CPU only; the MILP solves in ~0.02 s.

1. **Read the sEH number from the committed [042] artifact** rather than re-running it
   (`scent_seh_recon_full_reconcile/reconcile.csv`).
2. **Verify the DRD2 cell is reconcilable** — same `{"hubs":[{children:[{reaction,reward,smiles,…}]}]}`
   schema as the sEH native run, `sample/routes.json` present (65 MB), snapshot matched to the cell's
   checkpoint via `manifest.csv`.
3. **Run the reconciler on DRD2** at that target's gate (reward > 0.5 per `matrix16/targets.py`),
   cutoff 0.5, 100 modes — matching the sEH invocation:
   ```bash
   source ~/bin/rgfn-smoke-env.sh && python experiments/lsd_hubs/campaign/reconcile_t15.py \
       --recon-dir /scratch/markymoo/rgfn_runs/lsdflow/matrix16/scent_drd2 \
       --snapshot .../scent_drd2_5k/seed42/additional_fragments/fragments_4000.json \
       --reward-threshold 0.5 --cutoff 0.5 --budget-modes 100 --tag scent_drd2_recon
   ```
4. **Diagnose the refusal** — three passes, because the first two hypotheses were wrong and are
   recorded here so they are not re-tried:
   - *(rejected)* "hub prefixes are missing from the priced routes": **190/200** DRD2 hubs **are** in
     `routes.json`; only 10 miss, and 7 of those are depth-0 base fragments that correctly have no
     route. Too few to explain 284 → 106.
   - *(rejected)* "count-once over-charges": it charges exactly what its model says (1 coupling +
     fragment builds); the fragment builds are real work under that model.
   - *(confirmed)* **recipe coverage**: rebuilt the same library and counted the globally deduplicated
     steps actually handed to SPARROW, then checked both snapshots for `smiles_to_route`.

### Results

**The T4.4 number (sEH, 100 modes, native-route coverage 1.00, MILP status Optimal).**

| strategy | count-once | SPARROW optimum | left on the table | rel_diff |
|---|---|---|---|---|
| **hub-batching** | 288 | **279** | 9 reactions | **3.12%** |
| best-candidate | 314 | 299 | 15 reactions | 4.78% |

**The DRD2 refusal, and the asymmetry behind it.**

| strategy | count-once | SPARROW | rel_diff | gate |
|---|---|---|---|---|
| best-candidate | 369 | 336 | 8.94% | passes |
| **hub-batching** | 284 | **106** | **62.68%** | **FAIL** |

Rebuilding the same libraries and counting the steps that reach SPARROW shows where the two sides part
company — and that sEH expands fragment builds while DRD2 does not:

| cell / strategy | count-once | steps handed to SPARROW | distinct | source hubs | modes with no prefix |
|---|---|---|---|---|---|
| DRD2 hub-batching | 284 | **106** | 106 | **4** | 97 |
| DRD2 best-candidate | 369 | 351 | 344 | — | 0 |
| sEH hub-batching | 288 | **505** | 286 | 9 | 0 |
| sEH best-candidate | 314 | 411 | 307 | — | 0 |

sEH's 505 steps collapsing to 286 distinct is recipe expansion doing its job; DRD2's 106 (≈ 100 modes
× one final reaction) is recipe expansion doing nothing. The cause is in the snapshots:

| snapshot | promoted fragments | `smiles_to_route` recipes | coverage |
|---|---|---|---|
| sEH `2026-07-10_17-28-06` (recipe re-run, job 70180) | 1600 | 1600 | **100%** |
| DRD2 `scent_drd2_5k/seed42` | 1600 | **key absent** | **0%** |

With no recipes, `expand_route_with_recipes` cannot turn a promoted fragment into build steps, so
SPARROW buys what count-once builds. Compounding the visibility of it, hub-batching's 100 DRD2 modes
come from just **4** hubs and **97 from one depth-0 bought fragment** (4-fluorobenzoic acid), so the
un-expanded library is almost pure final-couplings — hence the dramatic 284 → 106.

**Which cells can be audited at all** (highest-N snapshot per SCENT run; "auditable" = recipes cover
≥95% of that snapshot's promoted fragments):

| run | promoted | recipes | auditable |
|---|---|---|---|
| `scent_seh/2026-07-10_17-28-06` (recipe re-run, job 70180) | 1600 | 1600 | **yes** |
| `scent_drd2/2026-07-10_17-28-06` (same re-run) | 1600 | 1600 | **yes** |
| `scent_seh_5k/seed42`, `scent_drd2_5k/seed42` (matrix16's cells) | 1600 | 0 | no |
| `scent_6td3_5k/seed{42,43,44}` | 800–1200 | 0 | no |
| `scent_clpp_5k/seed{42,43,44}` | 1200–1600 | 0 | no |
| `scent_{seh,drd2}` pre-07-10 runs (07-01/07-02/07-07) | 1600 | 0 | no |

Two consequences worth carrying: (1) **both surrogate targets are auditable** off the 2026-07-10
re-run, while **both docking targets (6TD3/ClpP) have no recipes on any seed** — auditing those would
need a training re-run with logging enabled, though they are the deliberately deferred cells anyway;
(2) matrix16's own `scent_seh` cell (the committed 1.303 rxn/mode) is **also** recipe-less, so the
3.1% is measured on a *different, same-scale* SCENT sEH run. T4.4 therefore validates the **cost
model**, which is what it is for — not that particular cell's number.

**Bottom line:** sEH's **3.1%** is the T4.4 number. DRD2 can be audited without retraining, via a
sample+enumerate pass on the 2026-07-10 checkpoint; until that lands, no DRD2 native-route SPARROW
figure should be quoted. The DRD2 count-once numbers carried in from the merge remain valid.
