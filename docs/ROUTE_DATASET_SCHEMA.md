# LSD-Flow route dataset — storage schema

**Status:** specification. Written before the exporter so the format is a decision, not a by-product
of whatever the first script happened to emit.

**Audience:** a synthetic chemist deciding whether to make these compounds, and a future maintainer
regenerating or extending the dataset. Both are served by the same tree; neither should need to read
our pipeline code to use it.

---

## 1. What this dataset is

For each published library we ship **every molecule's full synthesis route** and **the batch it
belongs to**. A "batch" is a set of molecules that share a synthetic intermediate, so they are made
together off one common prefix — the thing a chemist would run as one plate.

The libraries are produced under a **fixed reaction budget** (default 100 reactions): the question is
*"I can run 100 reactions — how many genuinely different molecules do I get, and how do I make them?"*

Two selection strategies are shipped for every cell, on **identical chemistry** (same enumeration,
same hit threshold, same budget — only the chooser differs), so the strategies can be compared
directly rather than across incomparable runs:

| strategy | what it does |
|---|---|
| `hub_batching` | picks molecules that reuse already-paid-for intermediates, subject to a diversity constraint |
| `best_candidate` | takes the highest-scoring molecules one at a time, subject to the same diversity constraint |

## 2. What a "route" here IS and IS NOT

**IS:** the *logged assembly route* — the exact sequence of reactions the generative model used to
construct the molecule, recorded as it happened. Every step is grounded in a reaction template that
fired, not inferred after the fact.

**IS NOT:** a claim of optimal retrosynthesis. A chemist may well know a shorter or more robust route.
Reaction conditions, solvents, temperatures, yields and stereochemical outcomes are **not** included —
this is a connectivity-level plan, not a procedure. `docs/` and the paper carry the caveats;
`README.md` in each cell repeats them so a file opened in isolation is not misread.

**Leaves are purchasable.** Every route bottoms out in catalogue building blocks. Where an
intermediate is itself something the model built (a "promoted fragment"), its own sub-route is
expanded inline, so no step ever says *buy X* for an X that cannot be bought.

## 3. Directory layout

One directory per **cell** = (generator, target, seed), one subdirectory per **strategy**:

    lsdflow-routes/
      README.md                     what the dataset is, how to cite, the caveats above
      SCHEMA.md                     this file
      scent_seh_seed43/
        README.md                   this cell: model, target, hit threshold, budget, provenance
        hub_batching/
          molecules.csv             one row per molecule in the library
          batches.csv               one row per batch
          routes.json               {smiles: AiZynth-style route tree}   <- interoperable view
          steps.csv                 one row per reaction step            <- flat view
          batch_01/
            buy_list.csv            what to order for this batch
            protocol.md             ordered, human-readable steps
            scheme.png              rendered reaction scheme
          batch_02/ ...
        best_candidate/ ...
      scent_seh_seed44/ ...
      scent_drd2_seed43/ ...
      scent_drd2_seed44/ ...

Cell directories are named `<generator>_<target>_seed<N>` and are self-contained: nothing in one cell
refers to another.

## 4. File formats

### 4.1 `molecules.csv` — the library

| column | type | meaning |
|---|---|---|
| `mol_id` | str | stable id within the cell, `M0001`… (assignment order = selection order) |
| `smiles` | str | canonical SMILES, **stereo-aware** — the structure to make (§6) |
| `smiles_flat` | str | stereo-stripped form — the key selection and the diversity metric used |
| `has_unassigned_stereo` | bool | true iff a centre is genuinely undefined; only these warrant a squiggle |
| `batch_id` | str | which batch makes it, `B01`… |
| `reward` | float | the oracle score at selection time |
| `reward_name` | str | which oracle (`seh`, `drd2`, …) |
| `selection_step` | int | nth molecule accepted by the strategy (1-based) |
| `cum_reactions` | int | total reactions spent by the library after this molecule |
| `reactions_added` | int | NEW reactions this molecule cost — the marginal price |
| `n_steps` | int | length of its own full route |

`reactions_added` is the number that shows batching working: within a batch it is typically 1, because
the prefix is already built. It is **not** `n_steps`, and the two must not be confused — `n_steps` is
what it takes to make the molecule from scratch, `reactions_added` is what it cost *given everything
already made*.

### 4.2 `batches.csv` — the plates

| column | type | meaning |
|---|---|---|
| `batch_id` | str | `B01`… |
| `shared_intermediate` | str | the SMILES every member is built from (the hub) |
| `n_molecules` | int | members |
| `prefix_reactions` | int | reactions to build the shared intermediate once |
| `total_reactions` | int | `prefix_reactions` + the per-member diversifying steps |
| `reactions_per_molecule` | float | `total_reactions / n_molecules` — the batching efficiency |

`best_candidate` libraries also get this file. Most of their batches are singletons; that IS the
result, and the format states it rather than hiding it by omitting the file.

### 4.3 `routes.json` — interoperable route trees

Keyed by molecule SMILES. Each value is a nested tree in **AiZynthFinder's route format**
(`genheden2020aizynth`), so existing retrosynthesis tooling reads it unmodified:

```json
{"<target_smiles>": {
  "type": "mol",
  "smiles": "<target>",
  "in_stock": false,
  "children": [{
    "type": "reaction",
    "metadata": {"template_id": "t183", "template": "R'(...)", "named_reaction": "Benzimidazole formation"},
    "children": [
      {"type": "mol", "smiles": "<reactant A>", "in_stock": true,  "children": []},
      {"type": "mol", "smiles": "<intermediate>", "in_stock": false, "children": [ ... ]}
    ]}]}}
```

Conventions, and why:
* `in_stock: true` marks a **purchasable** leaf. A tree is well-formed here iff every leaf is
  `in_stock` — the same "solved" test AiZynth uses.
* `metadata.template` is the reaction template that actually fired. AiZynth's schema has no
  first-class slot for it, so it lives in `metadata` where a foreign reader can ignore it safely.
  **Routes recovered by AiZynth rather than logged by the model will have no `template`** — the field
  is absent, not empty, so the two provenances stay distinguishable.
* `named_reaction` is a human label recognised from the template's transformation. It is a
  convenience for reading, never an identifier — join on `template_id`.

### 4.4 `steps.csv` — the flat view

The same routes, one row per step, for tabulating and diffing without walking trees:

| column | type | meaning |
|---|---|---|
| `mol_id` | str | which molecule's route |
| `step` | int | 1-based position in that route |
| `reaction` | str | template id (`""` if the route was recovered, not logged) |
| `named_reaction` | str | human label |
| `reactants` | str | `.`-joined SMILES, matching reaction-SMILES convention |
| `product` | str | this step's product |
| `is_shared` | bool | true if this step is part of the batch prefix, i.e. run ONCE for the whole batch |

`is_shared` is what makes the batch legible: filter to `is_shared=false` and you have exactly the
per-molecule diversifying work.

**`routes.json` and `steps.csv` are two views of one object.** The exporter derives both from the
same in-memory route and asserts they agree on step count and product identity before writing; a
mismatch is a build failure, not a warning.

### 4.5 `batch_NN/buy_list.csv` and `protocol.md`

`buy_list.csv`: `smiles, role, n_molecules_needing_it` where `role ∈ {shared, diverging}` — shared
building blocks are bought once for the plate; diverging ones are per-molecule.

`protocol.md`: prose, ordered, with the shared prefix stated once at the top and each molecule's
divergence listed under it. Generated from the same source, never hand-written.

## 5. Cell `README.md` — required provenance

Every cell records enough to regenerate it and to know what it is worth:

* generator, target, oracle name, **hit threshold and its direction** (higher- or lower-is-better)
* reaction budget, diversity cutoff τ, and the similarity metric it is computed on
* the model checkpoint, the enumeration artifact, and the fragment snapshot — **by absolute path and
  md5** (see §7)
* how many molecules and batches, and reactions actually spent
* known limitations for that cell

## 6. Stereochemistry and canonicalisation

**Stereocentres in this dataset are BOUGHT, not MADE — so the stereo-aware structure is what we
ship, and it costs a chemist nothing to obtain.** Measured on 10,248 route steps across 4,000
routes:

| | |
|---|---|
| steps where a stereocentre is PRESERVED from an input | **10,248 (100.00%)** |
| steps where a step CREATES a stereocentre | **0 (0.00%)** |

That is a property of the reaction set, not luck. Amide coupling, Buchwald-Hartwig, SNAr, Suzuki,
Sonogashira, benzimidazole/benzoxazole formation, N-alkylation and tetrazole formation all form bonds
at **sp2 carbon or nitrogen**; none generates a new sp3 stereocentre. The chirality enters through the
catalogue instead — of 418 purchasable building blocks, **67 are sold as single enantiomers with
assigned centres** (chiral-pool staples: lysine ``NCCCC[C@H](N)C(=O)O``, aspartate, (R)-nipecotic
acid, a tetrahydroisoquinoline-3-carboxylic acid), 347 have no centre, and 4 have an unassigned one.

**Consequence, and the reason this section exists.** It would be wrong to present these libraries as
racemates needing separation. The usual chemist's intuition -- *a new stereocentre comes out racemic
and separating enantiomers is painful* -- is correct in general and does not apply here, because no
step creates a centre. Drawing a squiggle bond would invent a separation problem the route does not
have, and would discard information that is free: **44% of sampled molecules carry defined
stereochemistry, fully determined by which enantiomer of a block you order.**

So:
* ``smiles`` is **stereo-aware** — the structure you would actually make.
* ``smiles_flat`` is the stereo-stripped form, carried because it is the key the diversity metric
  and the selection ran on.
* ``buy_list.csv`` names the **stereo-defined catalogue entry** (``O=C(O)[C@@H]1CCCNC1``, not
  ``O=C(O)C1CCCNC1``) — ordering the flat form is ordering the wrong thing.
* The 4 blocks with genuinely unassigned centres are flagged **per molecule**
  (``has_unassigned_stereo``), not by globally de-specifying every structure. Those are the only
  places a squiggle is the honest depiction.

**The real limitation, stated precisely.** Selection and mode-counting ran on stereo-stripped
structures, so **two enantiomers counted as ONE molecule during selection**. A library can therefore
contain a near-duplicate enantiomeric pair that the diversity metric merged. That is a narrower claim
than "the stereochemistry is unknown" — it is known; it just did not participate in the choosing.

Molecule identity across files is by ``smiles`` (stereo-aware); ``mol_id`` is a convenience for
joining within a cell and is **not** stable across regenerations.

## 7. Reproducibility, and one hazard worth naming

Each cell's `README.md` records the md5 of the enumeration it was built from. This is not ceremony:
these artifacts live in a shared scratch tree and have been rewritten mid-experiment before, silently
turning a same-chemistry comparison into a cross-chemistry one. **A dataset that cannot prove which
enumeration it came from cannot support the comparison it is published to make.**

Datasets are therefore built from a **frozen snapshot**, never a live path.

## 8. What is deliberately excluded

* **Reaction conditions, yields, stereochemistry** — not measured; including guesses would be worse
  than omitting them.
* **Cells whose promoted fragments cannot be expanded.** Some training runs did not log fragment
  recipes, so their routes would contain *buy X* for an unbuyable X. Those cells are excluded
  entirely rather than shipped with broken leaves; `check_route_readiness.py` is the gate.
* **Competitor (SPARROW) selections**, for now. On the corrected enumerations those solves are
  time-limited lower bounds, and publishing a bounded selection invites reading it as an optimal one.

## 9. Precedent

* **Route format** — AiZynthFinder (`genheden2020aizynth`); its tree JSON is the de facto interchange
  format for retrosynthetic routes and what the PaRoutes benchmark distributes.
* **Batching** — SPARROW (`fromer2024sparrow`), *"Optimal Compound Downselection to Promote Diversity
  and Parallel Chemistry"*, is the nearest precedent for treating shared intermediates as the unit of
  experimental work. We know of **no standard format that encodes a cross-molecule batch layer**;
  §4.2 and the `is_shared` flag are ours, and are kept deliberately thin so the per-molecule routes
  remain readable by tools that ignore them.
