# SCENT hub-batching — diversity-pairs gallery (closest still-distinct pair per cutoff)
**Date:** 2026-07-14, ~1pm

## Question

For the molecules our large SCENT hub-batching run generated, what do the two *most similar*
molecules that a given diversity cutoff still treats as different actually look like — and how does
that closest pair change as we tighten or loosen the cutoff?

## Context & Summary

Our hub-batching campaign (`029`, and the 200-hub scale-up `031`) builds a diverse library of sEH
hits and reports it in numbers: how many distinct molecules ("modes") fit in a reaction budget, how
many reactions to reach 300 of them. "Distinct" is defined by a **diversity cutoff** — two molecules
count as the same if their Morgan-fingerprint Tanimoto similarity is above the cutoff, different if
below — and the campaign sweeps that cutoff from 0.30 (demanding very different molecules) to 0.90
(allowing near-copies). But those runs never *show* the molecules, so it's hard to feel what a cutoff
of, say, 0.5 versus 0.8 actually buys you in real chemical structure.

This experiment turns each cutoff into a picture. For every cutoff in the sweep (0.30, 0.35, … 0.90)
we rebuild the hub-batching library exactly as the campaign does, then pull out the single **closest
pair** of accepted molecules — the two that are as similar as possible while still being counted as
distinct — and draw them side by side, with their shared core highlighted. That "tightest still-
distinct pair" is the honest worst case for diversity at that cutoff: nothing in the library is more
alike than these two.

## Answer

The closest still-distinct pair sits essentially *on* the cutoff — exactly on it from 0.30 through
0.80, and just under it (0.84, 0.88) at 0.85 and 0.90 — so the cutoff behaves as a faithful,
interpretable dial rather than a loose bound. Visually the progression is exactly what a chemist
would hope: at a loose cutoff (0.90) the two "distinct" molecules differ only by a single ring-size
change (an azetidine grown to a pyrrolidine); at 0.75 they differ by a regiochemistry swap in one
ring (oxazole vs. isoxazole); and at the strict end (0.30) they share only a small fused-ring core
and are otherwise clearly different scaffolds. In short, the Morgan-r3/2048 cutoff we've been quoting
in the campaign really does map onto an intuitive "how different do these look" scale.

## Relevance to our Publication

The campaign figures for a chemistry-ML venue (NeurIPS / a drug-discovery ML journal) live or die on
one skeptical question: *when you say your library is "diverse at cutoff c," how different are those
molecules really?* Reviewers and bench chemists distrust a similarity threshold they can't picture.
This gallery answers it directly — one representative worst-case pair per cutoff — and shows the
threshold is a clean structural knob, making the diversity axis of `029`/`031` legible instead of
abstract. It's a supporting/intuition figure, not a new claim.

## Next Experiments

**Refining for publication**
- **Same gallery on the other targets** (DRD2, and the docking systems once they promote) to show the
  cutoff↔structure mapping isn't sEH-specific.
- **best-candidate counterpart** — draw the same closest-pair gallery for the best-candidate library
  so the two selection strategies can be compared visually at matched cutoffs, not just on cost.
- **A "median" pair alongside the tightest** — the closest pair is the worst case; adding a typical
  pair would show the library isn't all near-boundary.

**Next steps in project**
- Fold the closest-pair readout into the active-learning loop's per-round diagnostics (it reuses the
  same `CampaignResult`), so a run can be watched for diversity collapse as it learns.
- The RGFN-vs-SCENT hub-coincidence study (proposal §8) on the same machinery.

# Re-creation

## Relevant Files

Root: `./` (repo root).

**Strategy code (ours — `glue/`, unchanged from `029`/`031`):**
- `./glue/samplers/lsdflow/campaign.py` — `HubBatchingStrategy` (walks the ranked, enumerated hubs;
  feeds children best-reward-first to the selector). Reused as-is.
- `./glue/samplers/lsdflow/mode_select.py` — `DiverseThresholdModeSelector` (Morgan r=3/2048 Tanimoto,
  rejects a candidate only if its similarity to an accepted mode is **strictly** above the cutoff) +
  the `@lru_cache`d `ecfp()`. The strict-`>` rule is why the tightest pair can land exactly on the
  cutoff.

**Analysis (ours — `experiments/lsd_hubs/campaign/`, new):**
- `diversity_pairs.py` — for each cutoff, rebuild the hub-batching library (reward gate + budget),
  find the max-Tanimoto accepted pair, and render an MCS-aligned side-by-side gallery + a CSV of the
  pairs. Reuses `run_campaign._load_enumerated_hubs`. Loads **only** `enum_children.json` — acceptance
  depends solely on reward + Tanimoto, so the cost table / recipe snapshot (which only fill reaction
  accounting, never *which* molecules are modes) are deliberately not loaded.
- `route_trees.py` (addendum) — reads the `pairs.csv` above + `enum_children.json` (each molecule's
  hub + the promoted fragment `added_promoted` attached in its final reaction) + the recipe snapshot
  (`smiles_to_route`), and draws each pair's shared building block/intermediate forking into the two
  products (a schematic). Writes `diversity_pairs/route_trees.{png,csv}`.
- `synthesis_routes.py` (addendum) — the chemist-actionable protocol: same inputs, but names each
  reaction from its RGFN template, tags reactants buy/make (route-aware), and emits per-molecule
  step-by-step instructions + shopping lists (`synthesis_protocol.md`), a machine-readable step table
  (`synthesis_steps.csv`), and one reaction-scheme PNG per pair (`schemes/scheme_cut*.png`).

**Results (ours — `experiments/lsd_hubs/campaign/results/scent_seh_1kx200/diversity_pairs/`):**
- `gallery.png` — 13 rows (cutoff 0.30 → 0.90), each the closest still-distinct pair drawn side by
  side (shared MCS highlighted), labelled with the cutoff, the pair's Tanimoto, and each molecule's
  sEH reward.
- `pairs.csv` — one row per cutoff: `cutoff, tanimoto, n_modes, smiles_a/reward_a/hub_a,
  smiles_b/reward_b/hub_b`.
- `route_trees.png` — 13 shared-structure schematics (addendum); `route_trees.csv` — per-pair route
  decomposition (`same_hub, hub_a/b, frag_a/b, n_shared_steps, shared_drawn, shared_all`).
- `synthesis_protocol.md` — step-by-step protocol + shopping list for all 26 molecules; `synthesis_steps.csv`
  — machine-readable steps (`cutoff, molecule, step, reaction, input, reactants, product`); `schemes/scheme_cut*.png`
  — 13 reaction-scheme figures (one per pair: each step drawn as input + reactant → product, buy/make tagged).

**Inputs (on `$SCRATCH`, unchanged from `031`):**
- `/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json` — the 200-hub
  enumeration (828,448 child records + rewards) from job 70363. The main input.
- `/scratch/.../scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json` — the recipe
  snapshot (job 70180, entry `028`): `smiles_to_route` = each promoted fragment's logged synthesis
  route. Used **only** by the route-tree addendum, not by the diversity gallery.

**Job Logs:** none — pure CPU login-node analysis (~1 min), no SLURM job.

## Relevant Versions

Branch `Hub-Analysis`. New file `Logs/032_*.md`, new script
`experiments/lsd_hubs/campaign/diversity_pairs.py`, new results dir
`results/scent_seh_1kx200/diversity_pairs/`, the campaign `README.md` "diversity gallery" note, and
the `docs/RESEARCH_CONTEXT.md` index row. **Not yet committed.**
[TODO — add commit hash after committing.] *Can you commit these files? Let me know and I'll fill in
the hash.*

## Relevant Resources

**Sources**
- Entries `031` (the 200-hub enumeration this reads), `029` (the two-strategy campaign + the diversity
  sweep framing), `026` (the Morgan-r3/2048, Tanimoto mode definition). `docs/LSD_FLOW_PROPOSAL.md` §2.
- `[bengio2021gflownet]` (modes / diverse top-k), `[gainski2025scent]` (dynamic library the hubs draw on).

**Packages**
- `rgfn` env (RDKit for fingerprints + MCS + 2D depiction, matplotlib for the montage), CPU login node
  (`source ~/bin/rgfn-smoke-env.sh`). Used by `experiments/lsd_hubs/campaign/diversity_pairs.py`.

## Method

1. **Build the gallery (CPU, login).**
   ```bash
   source ~/bin/rgfn-smoke-env.sh
   python experiments/lsd_hubs/campaign/diversity_pairs.py \
       --enum-children /scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json \
       --reward-threshold 7.0 --tag scent_seh_1kx200
   ```
   For each cutoff in 0.30→0.90 (step 0.05): run `HubBatchingStrategy` (reward ≥ 7.0, 300-mode budget)
   over the cached enumeration to get the accepted modes, compute all pairwise Morgan-r3/2048 Tanimoto
   similarities among them, take the maximum-similarity pair, align the two molecules over their
   maximum common substructure (RDKit `rdFMCS`), and draw them. Writes `pairs.csv` + `gallery.png`.

## Results

**Closest still-distinct pair vs. cutoff** (hub-batching, sEH, reward ≥ 7.0, 300-mode library;
`pairs.csv`). "max Tanimoto" = the highest pairwise similarity among the 300 accepted modes — the two
most similar molecules the cutoff still counts as distinct. Every library reaches the full 300 modes.

| cutoff | max Tanimoto | how the closest pair differs (structural read from the SMILES) |
|---|---|---|
| 0.30 | 0.300 | share only a benzo-fused azole core; different scaffolds otherwise |
| 0.35 | 0.350 | shared imidazole/cyclopropyl arm; different amide linker + second ring |
| 0.40 | 0.400 | shared indole-carboxamide core; different pendant heterocycle |
| 0.45 | 0.450 | shared indole-ethylamine arm; different capping ring system |
| 0.50 | 0.500 | shared tetrazole/triazole arm; azetidine-amide vs. its homolog |
| 0.55 | 0.550 | shared indole-ethyl-amide; different central ring + linker |
| 0.60 | 0.600 | shared benzofused-oxazole + cyclopropyl; different substitution |
| 0.65 | 0.650 | one aromatic ring inserted/removed between two shared motifs |
| 0.70 | 0.700 | isoxazole ↔ oxazole regiochemistry swap in one ring |
| 0.75 | 0.750 | oxazole ↔ isoxazole regiochemistry swap in one ring |
| 0.80 | 0.800 | oxazole ↔ isoxazole swap; rest identical |
| 0.85 | 0.844 | azetidine ring grown to a piperidine; rest identical |
| 0.90 | 0.884 | azetidine grown to a pyrrolidine (one CH₂); otherwise identical |

**The tightest pair tracks the cutoff.** From 0.30 to 0.80 the closest accepted pair sits *exactly*
at the cutoff (0.300, 0.350, … 0.800): the hub-children neighborhoods are dense enough that some pair
realizes the exact rational Tanimoto value equal to the threshold, and the selector — which rejects
only pairs *strictly* above the cutoff — keeps it. At 0.85 and 0.90 the closest pair falls a little
short of the cutoff (0.844, 0.884) because the 300-mode library fills before any admitted pair reaches
exactly that similarity. So the diversity cutoff is a tight, faithful bound on the library's internal
similarity, not a loose one.

**Where the pairs come from.** For all but two cutoffs both molecules of the closest pair trace back
to the same depth-0 hub (`ClCC1CC1`) — unsurprising, since a hub's enumerated children are decorations
of one scaffold and are therefore each other's nearest neighbors (at 0.35 and 0.45 one member comes
from a depth-1 hub). This is the same single-hub concentration `029`/`031` reported on the cost side,
seen here on the structural side.

## Addendum — synthesis routes + step-by-step protocols (same day)

*Follow-up questions: can we see the synthesis route for each molecule as a tree with shared steps
then a diverging step — and then write down **exactly** what reactions we run, what to buy, and what
the intermediates look like?*

**Where the route data comes from (nothing inferred).** Every molecule the campaign keeps is a
one-reaction **diversification of a hub**: the enumeration records its `hub` + the promoted
dynamic-library fragment attached in the final reaction (`added_promoted`). Each promoted fragment
carries a **logged synthesis route** (`smiles_to_route`: ordered `steps`, each with the RGFN reaction
**template**, `input`, `reactants`, `product`), expanded recursively through nested promoted
sub-fragments — the same route data the cost model in `028`/`029` charges. Two views were built:

- **`route_trees.py`** (schematic → `route_trees.png/csv`): draws the shared building block or
  intermediate, forking into the two products. A quick visual of where the pair overlaps.
- **`synthesis_routes.py`** (exact protocol → `synthesis_protocol.md`, `synthesis_steps.csv`, one
  reaction-scheme PNG per pair in `schemes/`): the chemist-actionable version. Each reaction is
  **named from its template** (amide coupling, Suzuki, Buchwald–Hartwig amination, reductive
  amination, benzoxazole/benzimidazole formation, N-alkylation, Sonogashira, SNAr, tetrazole
  formation); each reactant is tagged **buy** (a purchased building block — a leaf of the route) or
  **make** (an intermediate produced by an earlier step). The one step without a logged template is
  the final hub attach; it is named from the structural change (alkyl-halide hub → N-alkylation,
  nitrile → tetrazole, carboxylic-acid hub → amide coupling) and flagged as such. **This is the
  SCENT dynamic-library assembly route we logged, not a claim of a shortest retrosynthesis.**

**Result.** Every molecule assembles in **3 logged reactions + the hub attach**, from 4 purchased
building blocks. **11 of 13** pairs come from the **same hub**; how much they truly share splits three
ways (a "shared reaction step" means a step producing an *identical* intermediate reused by both — a
tighter, more honest count than "shared route pieces," which also counts jointly-purchased building
blocks):

| pairs | what is shared | example |
|---|---|---|
| **0.60, 0.85, 0.90** (3) | a **fully-built intermediate** — diverge only at the final decoration | 0.90: build `Nc1cc(NCc2c[nH]c3ccccc23)ccc1O` once, then + azetidine- vs. pyrrolidine-carboxylic acid |
| 0.30, 0.40, 0.50, 0.55, 0.65, 0.70, 0.75, 0.80 (8) | **same hub, same reaction sequence, overlapping purchases**, but they diverge at the **first** reaction (no reusable intermediate) | 0.75: both are amide(indolylmethylamine + a benzoxazole-acid) then N-alkylation, but the acid is the oxazole vs. isoxazole regioisomer |
| **0.35, 0.45** (2) | **different hubs — built independently** | — |

The cleanest illustration is **0.90**: (1) Buchwald–Hartwig amination of 4-bromo-2-aminophenol with
indol-3-ylmethanamine → the shared intermediate; (2) benzoxazole formation with azetidine-3- (A) or
pyrrolidine-2- (B) carboxylic acid; (3) N-alkylation of the indole with the hub `ClCC1CC1`
(cyclopropylmethyl chloride). Steps 1 and the reaction *types* are identical; only the acid in step 2
differs, giving the one-CH₂ ring difference the gallery shows. Full per-molecule instructions +
shopping lists for all 13 pairs are in `synthesis_protocol.md`; per-pair schemes in `schemes/`.
