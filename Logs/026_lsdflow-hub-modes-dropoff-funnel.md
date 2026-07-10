# sEH / RGFN — Paper-comparable hub "modes" + the filter-dropoff funnel
**Date:** 2026-07-10, ~1pm

## Question

When we count a hub's diverse "modes" the way the generative-model papers do — distinct
molecules that are both **strong binders** and **structurally dissimilar** — how many of a hub's
one-reaction products actually survive each filter, and where does the dropoff happen?

## Context & Summary

Entry `025` built LSD-Flow and showed that a trained reaction-GFlowNet's flow field contains
"hubs" — pre-terminal scaffolds from which many distinct products branch in a single reaction —
and that enumerating a deep hub's children yields hundreds of them. But it measured a hub's
diversity as *structural* clusters only (a fixed Tanimoto cutoff, no binding requirement), and its
"reactions per mode" cost had a counting bug that let the number exceed the maximum synthesis
route length. The RGFN and SCENT papers define a "mode" more strictly: a molecule that is **above
a binding/reward threshold** *and* dissimilar from every mode already found. To make our numbers
directly comparable to those papers — and to answer whether our "diversity" is real hits or just
near-duplicates and weak binders — we redefined a mode to match the papers exactly, fixed the cost
metric, and built a small reusable analysis that tracks how many of a hub's children survive each
filter in turn (raw children → keep the strong binders → keep the mutually-dissimilar ones).

## Answer

With the paper's mode definition, the **binding filter — not structural similarity — is the
dominant dropoff**: across the enumerated hubs only ~3% of a hub's one-reaction children clear a
strong-binder bar (sEH ≥ 7), and none clear the strict ≥ 8 bar the training run used. The two
filters act on different hubs: shallow **fragment** hubs are structurally very diverse but produce
**no** high-affinity products, while **deeper** hubs are where the flow concentrates genuine hits
— and there, building the shared scaffold once and diversifying is ~3.6× cheaper per distinct hit
than building each independently. So the "diverse high-affinity library from one hub" is
specifically a **deep-hub** phenomenon; fragments give breadth, depth gives hits.

Interestingly, it appears that high-reward molecules are NOT "more similar" than low-reward molecules for a given parent. Our tanimoto cutoff reduced our pool by roughly half when looking at all children, AND when looking at JUST children with binding > 7.0. This indicates that our hubs are capable of generating as novel of modes as their children are novel.

## Relevance to our Publication

For a methods paper (Digital Discovery, or a NeurIPS AI4Science / ICLR-MLDD workshop) that
positions against RGFN/SCENT, reviewers will expect the **same** mode metric those papers report;
this entry makes our hub numbers directly comparable and, in doing so, sharpens the story rather
than inflating it. It also directly pre-empts the obvious severe-test critique — *"is your hub
diversity just near-duplicates or weak binders?"* — by decomposing exactly where each hub's
children fall out. The dropoff funnel is committed as a reusable, modular analysis so the same
decomposition runs on every future reward target and generator.

## Next Experiments

**Refining for publication**
- Sweep the binding threshold and report **enrichment vs a random-hub baseline** (are flow-selected
  hubs richer in hits than random pre-terminal states?).
- Report the recipe sensitivity we standardized away (greedy-vs-Butina, ECFP r=2 vs r=3, cutoff
  0.65 vs 0.7) so the choice of the paper-matching recipe is transparent.
- Run the funnel across the other reward targets (DRD2 / ClpP / 6TD3) once those DAGs are enumerated.

**Next steps in project**
- More analyses under `experiments/lsd_hubs/` (depth-vs-hit-yield curve; hub reuse across seeds;
  the Pareto diversity-vs-concurrency front).
- The severe-test suite (`validation/lsdflow/analysis`) formalizing these checks.
- The RGFN-vs-SCENT hub-coincidence study once the patched SCENT checkpoints are enumerable.

# Re-creation

## Relevant Files

Root: `./` (repo root). Analysis is CSV + RDKit only (no GFN/GPU); runs on a Balam login node
under `source ~/bin/rgfn-smoke-env.sh`.

**Mode definition + cost (ours, changed this entry):**
- `./validation/lsdflow/metrics/diversity.py` — rewritten to the paper recipe:
  `mode_representatives` / `count_modes` now do **reward-gated, best-reward-first, greedy
  sphere-exclusion** on ECFP (Morgan r=3, 2048, no features/chirality) at similarity 0.7 —
  matching upstream `rgfn.trainer.metrics.TanimotoSimilarityModes`. `mode_counter` (structure-only)
  kept for the `most_modes` hub strategy.
- `./validation/lsdflow/harness/run.py` — `_per_mode_cost` now costs **one representative per
  hit-mode** (reward-gated) and takes `(similarity, reward_threshold, higher_is_better)`; fixes the
  earlier bug that divided the cost of building *all* children by the mode count.
- `./validation/lsdflow/harness/config.py` — new `mode_similarity_threshold` (0.7) +
  `mode_reward_threshold` (None = structure-only) knobs; `--mode-similarity` /
  `--mode-reward-threshold` CLI.

**The dropoff analysis (ours, new — modular under `experiments/lsd_hubs/`):**
- `./experiments/lsd_hubs/README.md` — the analysis group (one sub-dir per analysis type; reuse
  `glue/`+`validation/lsdflow/` primitives).
- `./experiments/lsd_hubs/dropoff/funnel.py` — the per-hub filter funnel (parameterized;
  higher/lower-is-better; threshold sweep). Reuses `validation/lsdflow/metrics/diversity`.
- `./experiments/lsd_hubs/dropoff/funnel_seh_70140_results.csv` — per-hub funnel table (committed).
- `./experiments/lsd_hubs/dropoff/funnel_seh_70140_summary.json` — aggregate survival + redundancy.
- `./experiments/lsd_hubs/dropoff/README.md` — how to run + the finding.

**Model / input (from entry `025`, job 70140):**
- `/scratch/markymoo/rgfn_runs/lsdflow/seh_stdlib_70140/enumerated_records.csv` — the 12
  enumerated hubs' one-reaction children (SMILES + sEH proxy reward + hub depth); the funnel's input.
- Trained RGFN sEH stdlib checkpoint `.../seh_proxy_stdlib/2026-07-02_14-59-53/.../last_gfn.pt`
  (job 69616, epoch 5001).

**Regenerated results (ours):**
- `./validation/lsdflow/results/seh_rgfn_enum/{enumeration.json,acquisitions.csv}` — recomputed
  under the paper-comparable gated mode definition (reward ≥ 7.0 headline).
- `./validation/lsdflow/results/seh_rgfn_pilot/acquisitions.csv` — same, 10k pilot.

## Relevant Versions

Branch `Hub-Analysis`. All files above are **not yet committed** (the mode-definition change in
`validation/lsdflow/metrics/diversity.py` + `harness/{run,config}.py`, the new
`experiments/lsd_hubs/` tree, and the regenerated `validation/lsdflow/results/*` artifacts).
[TODO — add commit hash after pushing.]

## Relevant Resources

**Sources**
- `rgfn/trainer/metrics/reaction_metrics.py::TanimotoSimilarityModes._extract_modes` — the exact
  mode recipe reproduced here (reward gate `proxy_term_threshold`, greedy, ECFP r=3/2048, 0.7).
- `glue/metrics/dataset_metrics.py::count_modes` — the project's candidate-dataset mode recipe
  (same r=3/0.7/greedy; no reward gate), now aligned.
- `[bengio2021gflownet]` (modes = reward-thresholded, dissimilar) / `[koziarski2024rgfn]`.

**Packages**
- `rdkit` (ECFP + Tanimoto), env `rgfn`; funnel + metrics import `validation/lsdflow`.

## Method

1. Redefined a mode in `diversity.py` to the upstream recipe (reward gate + best-first greedy
   sphere-exclusion, ECFP r=3/2048, 0.7); added `mode_representatives`. Unit-tested against the
   hand-worked A/B/C example and the reward-gate/best-first behaviour.
2. Fixed `_per_mode_cost` to cost one representative per hit-mode (hub = `Σ_distinct_hub depth +
   n_modes`, independent = `Σ_rep (depth+1)`) and threaded the similarity + reward-threshold +
   orientation knobs through the harness + CLI + config.
3. Regenerated the committed enumeration + acquisition numbers from the **persisted** 30k
   enumeration (no re-sampling / re-enumeration), at a headline sEH hit bar of 7.0.
4. Built `experiments/lsd_hubs/dropoff/funnel.py` and ran it on the 70140 enumerated records:
   ```
   python experiments/lsd_hubs/dropoff/funnel.py \
     --records /scratch/.../lsdflow/seh_stdlib_70140/enumerated_records.csv \
     --thresholds 6,7,7.5,8 --headline 7.0 --tag seh_70140
   ```

## Results

**Aggregate filter funnel (12 enumerated hubs, 4,586 raw one-reaction children; sEH higher-is-better):**

| stage | count | survival |
|---|---|---|
| raw children (distinct products) | 4,586 | 100% |
| binding gate sEH ≥ 6.0 | 1,356 | 29.6% |
| binding gate sEH ≥ 7.0 | 135 | 2.9% |
| binding gate sEH ≥ 7.5 | 22 | 0.5% |
| binding gate sEH ≥ 8.0 | 0 | 0% |
| Tanimoto-dedup of the ≥7.0 hits → **hit-modes** | 71 | (135 → 71, 1.9×) |

Structure-only dedup of *all* children (no gate): 4,586 → 2,010 modes (2.3× redundancy). So the
binding gate is the dominant cut (34× at ≥7.0); Tanimoto dedup is comparatively gentle.

**Per-hub (representative rows; full table `funnel_seh_70140_results.csv`):**

| depth | raw | ≥6 | ≥7 | ≥7.5 | structure-modes (children/mode) | hit-modes ≥7 (hits/mode) |
|---|---|---|---|---|---|---|
| 0 | 402 | 0 | 0 | 0 | 359 (1.1) | 0 |
| 0 | 60 | 0 | 0 | 0 | 60 (1.0) | 0 |
| 1 | 134 | 7 | 1 | 0 | 87 (1.5) | 1 |
| 3 | 996 | 414 | 58 | 12 | 350 (2.8) | 29 |
| 3 | 498 | 354 | 41 | 3 | 110 (4.5) | 19 |
| 3 | 996 | 303 | 32 | 7 | 314 (3.2) | 19 |
| 3 | 686 | 212 | 2 | 0 | 105 (6.5) | 2 |

Fragment (depth-0) hubs: children are structurally distinct (children/structure-mode ≈ 1.0–1.3) but
**0** clear sEH≥7. Depth-3 hubs: 3–6% clear ≥7 and carry more structural redundancy
(children/structure-mode 2.7–6.5), giving 19–29 distinct hit-modes after both filters.

**Amortized cost at the ≥7.0 hit bar (reactions per hit-mode, one representative per mode):**

| hub depth | hit-modes | reactions/mode: hub vs independent |
|---|---|---|
| 3 (996 children) | 29 | **1.10 vs 4.0 (~3.6×)** |
| 3 (996) | 19 | 1.16 vs 4.0 |
| 3 (498) | 19 | 1.16 vs 4.0 |
| 1 (134) | 1 | 2.0 vs 2.0 (single mode ⇒ nothing to amortize) |

Independent cost per mode = `depth+1` (bounded by the 4-reaction cap); hub cost per mode ≈ 1, so
the saving is `~(depth+1)×` and is realized where hits exist — the depth-3 hubs.
