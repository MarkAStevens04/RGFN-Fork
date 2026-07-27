# sEH — chemistry homogenization: can a SMILES generator or a SOTA planner match the reaction-GFN's route economy?

**Date:** 2026-07-27, ~11am

## Question

If we give a from-scratch SMILES generator — or a state-of-the-art route planner — the reaction-GFN's *own* building blocks, can either one reproduce the reaction-GFN's low number of reactions per distinct molecule?

## Context & Summary

Entry [047] showed that the reaction-GFN's molecules only *looked* hard to make because the standard catalogue (ZINC) doesn't stock its reactive building blocks — adding those blocks lifted route-success 48.7% → 73.8%. Two objections to the "reactions-per-mode" story still stood: (1) maybe the reaction-GFN's molecules are simply less synthesizable, and (2) maybe the from-scratch route-cost penalty ([041]: reaction-GFN priced at 2.74 reactions/mode vs its native 1.22 from entry [037]) is a real cost, not an artifact. These experiments settle both by **homogenizing the chemistry** — crucially, **fragments only: we hand over the reaction-GFN's 418 building blocks but never its reactions** (each method keeps its own chemistry, mirroring how [047] left the planner's reaction rules untouched). **Exp A** trains the marquee SMILES baseline (S3-GFN) with *only* those 418 blocks as its synthesizability library; **Exp B** gives it its best library *plus* those blocks ("best chance"); **Exp C** re-prices the reaction-GFN's actual molecules through a SOTA convergent planner (MultiAiZ→SPARROW) with the blocks in stock, to see how close a smart planner gets to the native 1.22. Alongside, we completed the MVP milestone (T3.2, the S3-GFN frontier) and pushed the [047] synthesizability ceiling to a generous search budget.

## Answer

**Exp C is the solid result; Exp A turned out to be inconclusive and must not be cited as a
synthesizability claim** (a follow-up falsification test, below, changed its interpretation — see
"The Exp A correction"). A convergent planner (MultiAiZ) does far better than plain retrosynthesis —
it pulls the reaction-GFN's from-scratch cost from 2.74 down to ~1.85 reactions/mode — but still
cannot match the reaction-GFN's own by-construction routes (1.22), and hub-batching stays ~1.5×
cheaper than best-candidate under *every* pricing regime. So the from-scratch penalty was mostly a
weak planner plus the stock mismatch, not the molecules; and the reaction-GFN's shared-route
structure is doing work that no post-hoc method fully recovers. The synthesizability ceiling (blocks
in stock + generous search) reaches ~92%, confirming the molecules are overwhelmingly makeable from
the reaction-GFN's own blocks. What Exp A/B *did* establish is methodological: **a fragments-only
transfer cannot homogenize a template-based synthesizability signal** — S3-GFN's retro-solver judges
molecules with its own curated template set, and that set cannot reverse the reaction-GFN's
chemistry no matter how many blocks it is given, so "give it the same blocks" is not a well-posed
fairness test for this baseline (it is for AiZynth's broad USPTO templates, which is why [047] worked).

## Relevance to our Publication

This is the direct rebuttal panel for the two objections a NeurIPS reviewer will raise about the reactions-per-mode headline. "The reaction-GFN just makes less-synthesizable molecules" is answered by the ~92% ceiling together with [047] (the molecules are overwhelmingly makeable from the generator's own blocks; the shortfall is catalogue + search budget). "The from-scratch route-cost penalty is a real synthesizability cost" is answered by Exp C (a SOTA planner recovers most of it, and native routes still win) — which is why we report **native-route pricing as primary and from-scratch as the deliberately-unfair floor**. Exp A/B contribute a narrower but useful methodological point (fragments-only transfer can't homogenize a template-based synthesizability signal) and **must not be cited as evidence about molecule synthesizability**. It also explains the otherwise-awkward T3.2 result (S3-GFN looks competitive at from-scratch pricing *because* it emits ZINC-native molecules while the reaction-GFN is penalized by the stock mismatch) — turning a confound into evidence for the pricing choice.

## Next Experiments

**Refining for publication**
- **Re-pose the Exp A question properly.** Since a fragments-only transfer can't homogenize S3-GFN's
  template-based signal, the fair version needs *either* the reaction-GFN's 112 templates transferred
  alongside its blocks (requires retraining S3-GFN's retro env — the cost we were avoiding) *or* a
  general-template retro-solver (AiZynth/USPTO, as in [047]) as S3-GFN's synthesizability signal.
  Decide which before re-running; the current Exp A should not be reported as a result.
- Full τ-sweep of Exp C (cutoffs 0.3/0.7/0.9, both stocks) for the complete reactions/mode-vs-diversity curve in homogenized chemistry (the cutoff-0.5 point is in hand).
- Repeat the [047] stock test on the other reaction-GFN generators (RGFN/RxnFlow) once the campaign regenerates their sEH pools, to show the confound is not SCENT-specific.
- Right-size wall-clocks: the MultiAiZ and high-budget-ceiling runs need ~20 h and ~13 h respectively (they timed out at 14 h / 10 h; results were recovered because both write incrementally).

**Next steps in project**
- Fold native-route pricing in as the primary frontier axis; keep from-scratch and MultiAiZ as the two competitor references, with this entry quantifying the gap.
- Proceed to target generality (T4.3) on DRD2/6TD3/ClpP as those pools land.

# Re-creation

### Relevant Files

Root: `./experiments/` unless noted. All committed at `3888c35` ("Experiments A/B/C running") except this log + the ceiling-finish tweak.

**Scripts / configs**
- `lsd_hubs/campaign/build_s3gfn_homogenized_envs.py` — builds the two S3-GFN retro envs from `data/libraries/glue_standard_v1/fragments.csv`: `small_hb105` (389 stereo-stripped SMALL blocks) and `zincfrag_small_hb105` (ZINCFrag ∪ SMALL); both copy `hb105`'s `template.txt` verbatim (**no reaction transfer**).
- `../validation/configs/s3gfn_seh_small.yaml`, `s3gfn_seh_zincfrag_small.yaml` — Exp A / Exp B configs, identical to `s3gfn_seh_fixed.yaml` except `retro_env` + run name (only the stock varies).
- `lsd_hubs/campaign/submit_multiaiz_headline.sh` — Exp C driver; gained `AICONFIG`/`STOCK` overrides to point MultiAiZ→SPARROW at the merged stock.
- `lsd_hubs/campaign/submit_s3gfn_frontier.sh` + `s3gfn_frontier.py` — T3.2 (S3-GFN on the frontier, from-scratch AiZynth→SPARROW).
- `oracle_validation/aizynth_failure_modes/{aiz_fullpool.py,submit_aiz_ceiling.sh,build_block_stock.py}` — the [047] ceiling (env-parametrized budget `AIZ_IT/TL/MT`; inputs overridable for the finish run).
- `oracle_validation/s3gfn_retro_signal/{probe_retro_env_coverage.py,probe_depth_and_steps.py,probe_small_block_usage.py}` + `README.md` — the Exp A falsification probes (read-only, login node) that corrected Exp A's interpretation: they call S3-GFN's own `SynthesizabilityEvaluator` on SMALL blocks / SCENT molecules / Exp B molecules across both retro envs and `max_steps` settings.

**Figures (committed)**
- `lsd_hubs/campaign/make_pricing_panel.py` + `results/paper_pricing_panel/pricing_panel.{png,pdf,csv}`
  — the consolidated paper panel: reactions/mode by pricing regime x strategy, modes-priced coverage,
  and the [047] route-success ladder. Every value transcribed from a result file (sources in the CSV).
- `oracle_validation/aizynth_failure_modes/residual_taxonomy.py` + `results/residual_taxonomy.{png,pdf,csv}`
  — what the stubborn residual leaves actually are (see [047]).

**Data (scratch)** — root `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/`
- `zinc_plus_small_stock.hdf5` (+ `config_zincsmall.yml`) — ZINC ∪ SMALL merged stock for Exp C (single `zincsmall` key; only **77 keys new** to ZINC = the reactive-handle blocks). `config_rgfnlib_flat.yml` — the union stock for the ceiling.
- `external/s3gfn/data/envs/{small_hb105,zincfrag_small_hb105}/` — the two homogenized retro envs.

**Results (scratch)**
- `results/scent_seh_multiaiz_{zincsmall,zinc}/{pareto,budget_efficiency}.csv` — Exp C.
- `results/s3gfn_seh_frontier/` — T3.2.
- `experiments/fixed_reward/s3gfn_seh_zincfrag_small/fixed_reward/candidates/candidates.csv` — Exp B pool (Exp A emitted none).
- `scent_rawpool/aiz_fullpool_ceiling.jsonl` (+ `_finish`) — the [047] ceiling.

**Job Logs** — `/scratch/markymoo/rgfn_runs/{s3gfn_seh,scent_multiaiz,aiz_ceiling,s3gfn_frontier}-{71522,71523,71529,71531,71534,71536}.{out,err}`.

### Relevant Versions

Experiment infra committed at **`3888c35`** ("Experiments A/B/C running"); this log + the `submit_aiz_ceiling.sh` input-override tweak committed at **`b27c6f5`** ("LSD-Benchmark ZINCuSMALL") on branch `Hub-Analysis`. Scratch stock/pool/result artifacts are large and regenerated from the committed builders.

### Relevant Resources

**Sources** — entry [047] (stock-mismatch finding), [037] (native 1.22 rxn/mode), [041] (from-scratch SPARROW 2.74), [040] (S3-GFN bring-up + `zincfrag_hb105`), [043] (MultiAiZ build). S3-GFN `[kim2026s3gfn]`; MultiAiZ `[ianez2026multiaiz]`; SCENT `[gainski2025scent]`.

**Packages** — S3-GFN in the `s3gfn` env (Exp A/B training); MultiAiZ in `aizynth` + SPARROW MILP in `sparrow` (Exp C, crossed by subprocess); AiZynthFinder 4.4.1 in `aizynth` (ceiling + T3.2 route cache).

### Method

All on Balam. Exp A/B GPU (s3gfn env), 5000 steps, seed 42, same frozen sEH proxy as the reaction-GFN entrants (only `retro_env` differs). Exp C + ceiling + T3.2 CPU.

1. **Build homogenized envs** — `build_s3gfn_homogenized_envs.py` → `small_hb105` (389 blocks), `zincfrag_small_hb105` (179,011).
2. **Exp A/B** — `submit_s3gfn_seh.sh` with `CFG`/`RETRO_ENV` overrides (jobs 71522, 71523).
3. **Exp C** — merged stock via `build_block_stock`-style script → `zinc_plus_small_stock.hdf5`; `submit_multiaiz_headline.sh` at cutoff 0.5, 300 modes, n_iters=5, both stocks (71529 zincsmall, 71531 zinc). Smoke 71524 validated the merged stock end-to-end.
4. **T3.2** — `submit_s3gfn_frontier.sh` on the S3-GFN pool (job 71007) + its route cache (71536).
5. **[047] ceiling** — `submit_aiz_ceiling.sh` at `AIZ_IT=1000/TL=300/MT=9` over the 4,749 union (71534, timed out at 74%); finish run on the remaining 1,249 + control (71765).

### Results

**Exp A/B — same generator, same reward, only the stock library differs.**

| Exp | Retro stock | synth_ratio | Candidates emitted |
|---|---|---|---|
| A | 418 SMALL blocks only | **0.0** (loss = nan; positive buffer stalled at 15 items) | **0** |
| B | ZINCFrag ∪ SMALL (179,011) | ~0.81–0.90 | 2,000 |

**The Exp A correction (falsification test, login node, `s3gfn` env).** Exp A's 0.0 admits two
explanations: (H1) GP-MolFormer's molecules genuinely aren't reachable from 389 small blocks, or (H2)
the `hb105` templates cannot reverse that chemistry, so *nothing* could score. We probed S3-GFN's
actual `SynthesizabilityEvaluator` (the same object the training loop calls, `max_steps=3` as
configured):

| Probe | `small_hb105` | `zincfrag_hb105` | Reading |
|---|---|---|---|
| Raw SMALL blocks (in-stock sanity) | **20/20 = 100%** | 9/20 = 45% | env loads + indexes correctly — **not a plumbing bug** |
| S3-GFN's own molecules (Exp B pool) | 0/30 | **28/30 = 93%** | templates work; a 389-block stock is too small for its molecules |
| SCENT molecules by hub_depth 0/1/2/3 | 7% / 0 / 0 / 0 | 7% / 0 / 0 / 0 | **`hb105` cannot reverse SCENT chemistry even with 178k blocks** |
| `max_steps` 3 → 6 | identical | identical | step budget is **not** the cause |

So H2 dominates: Exp A cannot distinguish "blocks insufficient" from "templates incompatible", and is
**inconclusive as a claim about molecule synthesizability**. The 418-block library is inseparable
from the 112 reaction templates it was designed with; stripping the templates removes what makes
those fragments productive. (Contrast [047]: AiZynth's broad USPTO templates *did* productively
terminate at the same blocks, +25 pts — narrow curated template sets are the special case.)

Exp B, however, is **not** inert — the SMALL blocks are genuinely reachable there: of 59 solved Exp B
molecules, **14 (24%) have at least one route using a SMALL block, and 6.5% of all block-uses
(3,666/56,098 across the exhaustive retro trees) are SMALL blocks**. Exp B remains a valid
"best chance" arm; only Exp A's interpretation changed.

**Exp C — reactions per mode (lower = better), SCENT sEH, cutoff 0.5.**

| Pricing | hub-batching | best-candidate |
|---|---|---|
| native by-construction (entry [037]) | **1.22** | — |
| MultiAiZ + ZINC | **1.85** (301 rxn / 163 modes) | 2.95 (536 / 182) |
| MultiAiZ + ZINC∪SMALL | **1.91** (540 / 283) | 2.99 (768 / 257) |
| plain AiZynth from-scratch (entry [041]) | 2.74 (502 / 183) | 3.66 (564 / 154) |

MultiAiZ recovers ~60% of the from-scratch→native gap; the reaction-GFN's blocks lift routable modes 163→283 (+73%) at ~constant reactions/mode (coverage, not efficiency — the [047] effect); hub-batching beats best-candidate under every pricing.

**T3.2 — S3-GFN on the frontier (from-scratch AiZynth→SPARROW).** S3-GFN pool is **88% route-solvable** vs SCENT hub 48.7% (S3-GFN emits ZINC-native molecules; SCENT is penalized by the stock mismatch), so S3-GFN reaches ~35 modes at a 100-reaction budget vs SCENT hub's ~24 — the [047] confound, resolved by native-route pricing.

**[047] ceiling — zinc+blocks @ iter=1000/time=300/depth=9.** Partial (3,500 of 4,749 in 10 h): **92.2%** solvable — ladder 48.7% (zinc) → 73.8% (zinc+blocks) → ~92% (generous search). Finish run (job 71765) completes the remaining 1,249 + zinc control. `[TODO — replace with final full-pool % when 71765 lands]`
