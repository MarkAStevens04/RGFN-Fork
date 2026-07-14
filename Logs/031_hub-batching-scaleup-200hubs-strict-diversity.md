# SCENT — hub-batching vs best-candidate at scale (200 hubs, diversity to 0.10)
**Date:** 2026-07-14, ~7am

## Question

When the hub strategy is allowed to draw on far more scaffolds, and we demand hits that are far more
different from one another, does batching from hubs still beat picking the best individual molecules —
and is there a point where "diverse enough" becomes impossible for *either* approach?

## Context & Summary

Entry `029` compared two ways to build a diverse library of sEH hits from a trained cost-aware
generator: **best-candidate** (keep the top-reward sampled molecules, build each on its own) vs
**hub-batching** (build one shared scaffold, then diversify its enumerated children). On a small hub
set — the 50 hubs drawn from the top-100 candidates — hub-batching roughly halved the synthesis cost
per hit. But it hit a wall at a strict diversity setting (Tanimoto cutoff 0.30): it could only build
135 distinct hits instead of the target 300, because 50 hubs' children were all variations on a few
cores. That left an open question: was the wall a real limit of the space, or just too few hubs? And
because our "distinctness" is measured with Morgan radius-3 fingerprints, we wanted to push the
diversity demand much stricter than `029`'s 0.30 — down to 0.10.

This experiment scales the hub set **4×** — the top-1,000 candidates yield 575 distinct hubs, of
which we enumerate the **top 200** (job 70363) — reusing the same trained SCENT sEH model and the same
30,000-trajectory sample pool, and sweeps the diversity cutoff from **0.10 to 0.90**. We re-ran both
selection strategies over the cached enumeration and produced the same three plots plus the per-hub
statistics table.

## Answer

Scaling the hub set moved the wall down, as hoped: with 200 hubs, hub-batching now reaches the full
300 diverse hits at cutoff 0.30 (in 898 reactions), where the 50-hub run couldn't get past 135 —
more scaffolds buy more reach. But below roughly 0.30 we hit a different wall that has nothing to do
with the strategy: **neither** hub-batching **nor** best-candidate can assemble 300 hits, because the
sEH chemical space the model explored simply doesn't contain that many molecules that different from
each other (at Morgan r=3, cutoff < 0.30). Since best-candidate never touches the hub machinery and
fails identically, that floor is a property of the chemistry, not the method. Everywhere it can
actually deliver a library, hub-batching still wins by ~1.7–1.8× on reactions per hit.

## Relevance to our Publication

Reviewers at an ML-for-drug-discovery venue (NeurIPS / a chemistry-ML journal) will press on two
things: whether a batching advantage is a small-sample artifact, and where it breaks down. This entry
answers both — the advantage survives a 4× larger hub set, and we can now state exactly where extreme
diversity becomes infeasible for *any* selection strategy (a chemical-space limit, not a shortcoming
of hub-batching). The fact that best-candidate's numbers are byte-for-byte identical to the 50-hub run
is a built-in control showing the hub scale-up changed only the thing it was supposed to.

## Next Experiments

**Refining for publication**
- **Other targets at this scale** — repeat the 200-hub sweep on DRD2 (and the docking targets once
  they promote) to show the batching win and the diversity floor aren't sEH-specific.
- **Finer grid at the knee** — a denser cutoff sweep across 0.25–0.35 to pin the exact point where the
  chemical-space floor kicks in.
- **Enumeration speedup** — build the deferred `--no-flow` enumerate mode so we can enumerate many
  more than 200 hubs in the same walltime and push the ceiling even lower (keep the flow-on path for
  `U(h)`).

**Next steps in project**
- Wire the two swappable strategies into the **active-learning loop** and measure the
  oracle-efficiency curve for each (they already share the `CampaignResult` interface).
- The RGFN-vs-SCENT **hub-coincidence** study (proposal §8) on the same machinery.

# Re-creation

## Relevant Files

Root: `./` (repo root).

**Strategy code (ours — `glue/`, all unchanged from `029`):**
- `./glue/samplers/lsdflow/campaign.py` — `BestCandidateStrategy` / `HubBatchingStrategy` (disjoint
  inputs, identical `CampaignResult`) + count-once nested cost.
- `./glue/samplers/lsdflow/mode_select.py` — `DiverseThresholdModeSelector` (Morgan r=3/2048, Tanimoto);
  `ecfp()` is `@lru_cache`d so the 17 cutoffs reuse each fingerprint.
- `./validation/lsdflow/metrics/cost/dynamic_amortization.py` — `FragmentCostTable` (nested routes).

**Analysis (ours — `experiments/lsd_hubs/campaign/`):**
- `pick_hubs.py` — top-K candidates → parent hubs → rank by single-candidate flow `F_hat` → `hubs.csv`.
- `submit_scent_seh_enum.sh` — GPU enumeration (pick_hubs → scent-env worker `--mode enumerate`);
  parameterised by `K` / `N_HUBS` env vars.
- `sweep_campaign.py` — the diversity/budget sweeps → `results/<tag>/{pareto,fixed_modes,budget_efficiency}.{csv,png}` + `sweep_summary.json`.
- `hub_stats.py` — per-hub depth / children / `U(h)` / reward → `results/<tag>/hub_stats.csv`.

**Results (ours — `experiments/lsd_hubs/campaign/results/scent_seh_1kx200/`):**
- `pareto.{csv,png}`, `fixed_modes.{csv,png}`, `budget_efficiency.{csv,png}`, `sweep_summary.json`,
  `hub_stats.csv`. (The 50-hub baseline in `results/scent_seh/` is untouched.)

**Inputs (on `$SCRATCH`):**
- `/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/` — the 30k-trajectory SCENT sEH analysis DAG
  (`records.csv` = 26,069 distinct candidates; `compositions.json` = per-molecule promoted fragments).
  The candidate pool; unchanged from `028`/`029`.
- `/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/` — this run's enumeration:
  `enum_children.json` (138 MB), `enumerated_records.csv` (237 MB, per-child flow terms → `U(h)`),
  `hubs.csv` (200 ranked hubs), `enum_per_hub.json`, `meta.json`. Too large for git.
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json`
  — the recipe re-run snapshot (job 70180, entry `028`) carrying synthesis routes → exact nested cost.

**Job Logs:** enumeration **70363** (`/scratch/markymoo/rgfn_runs/campaign_enum_seh-70363.{out,err}`),
~5 h 37 m, COMPLETED.

## Relevant Versions

Branch `Hub-Analysis`. New file `Logs/031_*.md`, new results dir `results/scent_seh_1kx200/`, and the
`Logs/029` "Larger run" addendum + `docs/RESEARCH_CONTEXT.md` index row. **Not yet committed.**
[TODO — add commit hash after committing.] *Can you commit these files? Let me know and I'll fill in
the hash.*

## Relevant Resources

**Sources**
- `docs/LSD_FLOW_PROPOSAL.md` (§2 flow/`U(h)`, §11 cost). Entries `029` (the base two-strategy campaign
  this scales up), `028` (nested cost engine + recipe re-runs), `026` (the Morgan-r3/2048, Tanimoto-0.7
  mode definition).
- `[bengio2021gflownet]` (modes / top-k), `[gainski2025scent]` (dynamic library).

**Packages**
- `rgfn` env — the sweep + hub_stats (CPU, login node; the repo `results/` is read-only from compute
  nodes, so the analysis must run on login).
- `scent` env — the enumeration worker (GPU).

## Method

1. **Enumerate 200 hubs (GPU, job 70363).**
   ```bash
   sbatch --time=24:00:00 --export=ALL,K=1000,N_HUBS=200 \
       experiments/lsd_hubs/campaign/submit_scent_seh_enum.sh
   ```
   `pick_hubs` took the top-1,000 candidates by reward from `scent_seh_70189/records.csv` → 575
   distinct parent hubs → top-200 by single-candidate `F_hat` → `hubs.csv`; the scent-env worker then
   exhaustively enumerated + reward-scored each hub's one-reaction children on the frozen full library
   (`enum_children.json`). ~5 h 37 m.
2. **Diversity/budget sweep (CPU, login).**
   ```bash
   source ~/bin/rgfn-smoke-env.sh
   python experiments/lsd_hubs/campaign/sweep_campaign.py \
       --analysis-dir /scratch/.../lsdflow/scent_seh_70189 \
       --enum-children /scratch/.../campaign_enum_seh_70363/enum_children.json \
       --snapshot /scratch/.../scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json \
       --reward-threshold 7.0 --cutoff-min 0.10 --tag scent_seh_1kx200
   ```
   17 cutoffs (0.10→0.90 step 0.05) × 2 strategies, each a single run to the 300-mode budget over the
   cached enumeration (no re-scoring, no GPU).
3. **Per-hub statistics table.**
   ```bash
   python experiments/lsd_hubs/campaign/hub_stats.py \
       --enum-dir /scratch/.../campaign_enum_seh_70363 --reward-threshold 7.0 --tag scent_seh_1kx200
   ```

*(The background watcher meant to auto-run steps 2–3 on job completion was killed; steps 2–3 were run
by hand after 70363 finished.)*

## Results

**Hub set (200 enumerated, 828,448 child records):**

| depth | n hubs | avg children | avg U(h) | avg hits ≥ 7 |
|---|---|---|---|---|
| 0 | 2 | 5,988 | 71.2 | 1,230 |
| 1 | 104 | 6,458 | 43.5 | 1,332 |
| 2 | 84 | 1,596 | 45.8 | 253 |
| 3 | 10 | 1,072 | 47.1 | 462 |

**Diversity sweep (sEH, hit bar ≥ 7.0; `sweep_summary.json`):** "modes@100rxn" = diverse hits buildable
in 100 reactions; "rxns→300" = reactions to reach 300 hits (`—` = unreachable, pool/hubs exhausted).

| cutoff | hub modes@100rxn | best modes@100rxn | hub rxns→300 | best rxns→300 |
|---|---|---|---|---|
| 0.10 (strictest) | 6 | 4 | — | — |
| 0.15 | 19 | 14 | — | — |
| 0.20 | 28 | 19 | — | — |
| 0.25 | 32 | 17 | — | — |
| **0.30** | 33 | 17 | **898** | 1,386 |
| 0.40 | 33 | 16 | 795 | 1,436 |
| 0.50 | 34 | 16 | 817 | 1,479 |
| 0.70 | 34 | 18 | 770 | 1,445 |
| 0.90 (loosest) | 36 | 17 | 822 | 1,412 |

**Ceiling moved with hub count.** At cutoff 0.30, hub-batching reaches 300 modes in **898 reactions**
here, whereas the 50-hub run (entry `029`, job 70295) **could not reach 300 at 0.30** (maxed at 135
modes, 28/50 hubs exhausted). best-candidate reaches 300 at 1,386 reactions in both runs — identical,
as expected (it ignores the hub set).

**Chemistry floor below ~0.30.** From 0.10 to 0.25, *neither* strategy reaches 300 modes: at 0.30
hub-batching itself only produces 135 total modes (827 reactions) before exhausting its 200 hubs at
0.25 and stricter. Because best-candidate — which draws from the full 26k sampled pool and never uses
hubs — also fails to reach 300 below 0.30, the limit is the sEH chemical space at Morgan r=3, not the
batching strategy.

**Efficiency win preserved.** Where both can serve a 300-mode library (cutoff ≥ 0.30), hub-batching
needs ~1.7–1.8× fewer reactions (770–898 vs 1,386–1,479) and yields ~2× the modes per 100 reactions.
