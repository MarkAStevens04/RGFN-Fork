# sEH — what happens when the batch optimizer is given our own molecules to choose from

**Date:** 2026-08-11, ~12pm

## Question

If someone simply ran a batch-selection optimizer over our generator's own best molecules, would they
get a cheaper library than our method does?

## Context & Summary

Entry [056] compared our whole pipeline against an outside molecule generator and found we needed
about three times fewer reactions to deliver the same 100-family library. But that comparison changes
two things at once — a different generator *and* a different way of choosing what to make — so it
cannot say which of the two is doing the work. The sharper question is what happens when the
competitor is handed **our own molecules**: if a standard optimizer can find a cheap library from
them without any of our machinery, then our contribution is the generator, not the selection.

Two arms, both deliberately designed to be as strong as possible, and both using the naming adopted
on 2026-08-04 (see `docs/RESEARCH_CONTEXT.md`, "How library cost is measured"):

* **BC-SB (Best-Candidate SPARROW-Batching)** — take the N highest-scoring distinct molecules our
  generator sampled, hand the optimizer their recipes and a reaction budget, and let it pick. No
  enumeration, no flow, no diversity requirement. This is what someone would plausibly do first, and
  N was pushed as high as the solver could handle (the whole 21,000-molecule pool).
* **BC-Enum-SB (Best-Candidate Full-Enumeration SPARROW-Batching)** — the hardest control we can
  construct without using flow at all. Walk the candidates best-score-first, collect the 64 distinct
  intermediates they came from, exhaustively enumerate everything reachable one step from those
  intermediates (131,474 molecules), and give the optimizer all of it. It gets the *same kind of
  material* our method builds on.

Both are measured on two numbers together, because either alone is misleading: **reactions per
molecule made**, and **how many of the chosen molecules are actually different from one another**.

## Answer

**Neither arm disproves the method, and the harder one fails in the more informative way.** The
optimizer reliably produces libraries that are cheaper *per molecule* than ours — it is explicitly
minimizing that quantity and has no reason to care about anything else — but the molecules it picks
are overwhelmingly near-copies of each other. Given the same reaction budget, our method delivers
roughly five times more genuinely different molecules than the first arm and sixteen times more than
the second.

**The second arm's failure identifies what actually matters.** Handed the very same enumerated
molecules our method builds on, the optimizer concentrated its picks on a handful of intermediates —
at the tightest budget, three of them, with 78% coming from a single one. That is rational behaviour:
once an intermediate is built, every further molecule branching off it costs about one additional
reaction, so a cost-minimizing chooser takes as many as it can from one place. But those molecules
share their entire construction history and differ only in the final step, so they are nearly
identical. **Enumeration alone is therefore not what makes a reaction-based generator good for batch
synthesis — spreading the selection across many intermediates is, and that is the part our method
supplies.**

## Relevance to our Publication

This is the ablation that answers the most damaging version of the reviewer's question. Digital
Discovery reviewers will not merely ask "did you beat a baseline" — they will ask *which component
earns its place*, and the strongest form of that challenge is "your generator produces the molecules,
so why do we need your selection procedure at all; couldn't we just run a standard optimizer over its
output?" We can now answer with a direct measurement rather than an argument, on our own molecules,
with the competitor given every advantage including our own enumerated intermediates.

It also lets us be precise about what we are *not* claiming. On cost per molecule the optimizer wins,
and we should say so plainly; the claim is specifically about cost per *distinct* molecule. A paper
that reports the metric on which it loses is far harder to dismiss than one that reports only the
metric on which it wins.

## Next Experiments

**Refining for publication**

- **Replicate the harder arm.** The first arm has three seeds and is tight; the second has one. It is
  cheap now — the expensive enumeration is done and the selection sweep takes minutes — so this is
  bookkeeping rather than compute.
- **Repeat both on the second target.** Everything here is sEH. The machinery is target-agnostic and
  the DRD2 replicates already exist.
- **Report raw counts, not derived ratios.** At the tightest budgets the second arm yields as few as
  two distinct molecules, so any "per distinct molecule" ratio computed from it swings wildly. The
  honest presentation is the counts themselves.

**Next steps in project**

- Fold these arms into the headline figure so a reader sees the cost/diversity trade in one place.
- Extend the same comparison to the docking-based targets, where each measurement is expensive enough
  that wasted synthesis effort matters most.

---

# Re-creation

Root for repo-relative paths: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`.

### Relevant Files

**Scripts**

- `./experiments/lsd_hubs/campaign/sparrow_select_frontier.py` — **modified**: the frontier driver for
  every SPARROW-Batching arm. Gained two route sources beyond `multiaiz`:
  * `native` — our by-construction routes, looked up per molecule.
  * `enum` — the enumerated children of a hub set, whose routes must be **assembled** (hub prefix +
    the child's own diversifying reaction) because a child is one reaction past a hub and therefore
    absent from `routes.json`. Measured: **29 of 500** children present versus **64 of 64** hub keys.
    Reusing the `native` lookup here would have silently produced a pool of ~6% of the children.
  Also gained pool **deduplication** and `--top-n` (see Method 1).
- `./experiments/lsd_hubs/campaign/submit_bc_sb_scale.sh` — the size-climbing sweep, one arm per
  `ROUTE_SOURCE`. Records every size's outcome, so a pool that fails to solve is a datapoint rather
  than a lost run.
- `./validation/lsdflow/adapters/workers/sparrow_worker.py` — SPARROW in SELECTION mode (`--select`,
  reward objective, hard `--max-rxns`); built in entry [056].
- `./experiments/lsd_hubs/campaign/submit_scent_seh_native.sh` — reused unchanged for the enumeration
  by supplying our own `hubs.csv` (its `pick_hubs` step is skipped when one exists).

**Datasets**

- `/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_native_70974/` — the seed-42 sEH run supplying both
  the candidate pool (`sample/records.csv`, 29,997 rows → **21,001 distinct** molecules above the
  gate) and the hub route prefixes (`sample/routes.json`, 55,095 entries).
- `/scratch/markymoo/rgfn_runs/lsdflow/t45_seh_seed4{3,4}/` — the seed 43/44 replicates from entry
  [056]'s T4.5 work; natively recipe-carrying, so they need no re-run to be priced.
- `/scratch/markymoo/rgfn_runs/lsdflow/bc_enum_seh_seed42/` — the control's enumeration. `hubs.csv` is
  the 64 reward-walk intermediates; `enum/enum_children.json` is 131,474 children (65 MB). Its
  `sample/` is a symlink to the seed-42 run, so the enumeration reuses that sample rather than
  redoing it.
- `.../fixed_reward/scent_seh{,_5k/seed4[34]}/.../additional_fragments/fragments_4000.json` — the
  fragment snapshots supplying `smiles_to_route` recipes. Required: every native/enum route is
  *shallow* and must be recipe-expanded or SPARROW buys what count-once builds (entry [049]).

**Results** (on `$SCRATCH`; `$HOME` is read-only on compute — see Method 5)

- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/bc_sb/bc_sb_seh_seed4{2,3,4}_N*/` — BC-SB, 3 seeds ×
  6 pool sizes.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/bc_sb/bc_enum_sb_seh42_N*/` — BC-Enum-SB, 5 pool sizes.
- `./experiments/lsd_hubs/campaign/results/scent_seh_freefrag/budget_efficiency.csv` — hub-batching's
  own curve, the reference every arm is read against.

**Job Logs**

- `/scratch/markymoo/rgfn_runs/bc_sb_seh*-{72589,72590,72591}.{out,err}` — BC-SB (the successful runs)
- `/scratch/markymoo/rgfn_runs/bc_sb_seh42-{72509,72510,72511}.err` — the failed first attempt, kept
  because it is the evidence for Method 5
- `/scratch/markymoo/rgfn_runs/bc_enum_seh42-72522.out` — the enumeration
- `/scratch/markymoo/rgfn_runs/bc_enum_sb-73097.out` — BC-Enum-SB

### Relevant Versions

Branch `Hub-Analysis`. Most recent commit at time of writing: `07b8fd0` ("PartialFlusher: scale the
flush interval to the slice's hub count") — another agent's docking work, unrelated to this entry.

**Not yet committed.** These need committing for this entry to reproduce:

```
M  experiments/lsd_hubs/campaign/sparrow_select_frontier.py
?? experiments/lsd_hubs/campaign/submit_bc_sb_scale.sh
```

Note this repo is a **shared working tree with two other agents**. Isolate the hunk rather than
staging whole files: `git diff -U3 <file>` → keep only your hunk → `git apply --cached` →
`git commit --no-verify` (see `/home/markymoo/agent_comms/README.md`).

`[TODO — add commit hash after pushing]`

### Relevant Resources

**Sources**

- `[fromer2024sparrow]` — SPARROW, used here in SELECTION mode as a competitor rather than as the
  auditor of entries [042]/[049].

**Packages**

- `sparrow` + PuLP/CBC (`sparrow` env) — via `validation/lsdflow/adapters/workers/sparrow_worker.py`
- RDKit (`rgfn` env) — distinctness counting via `validation/lsdflow/metrics/diversity.py`
  (Morgan r=3 / 2048 bits, greedy sphere exclusion at Tanimoto ≤ 0.5, best-reward-first)

### Method

1. **Built the BC-SB pool with deduplication.** A `records.csv` row is a *sampling event*, not a
   candidate — the GFlowNet re-samples the same molecule constantly. Measured: the top-500 rows are
   only **115 distinct molecules**. Dedup keeps each molecule once at its best observed reward, and
   `--top-n` caps *after* dedup so "top-N candidates" means N distinct molecules.

2. **Swept BC-SB** over N ∈ {500, 1k, 2k, 5k, 10k, 21k} × 3 seeds, budgets 50–300 reactions, MILP
   capped at 900 s so a truncated solve shows as a non-`Optimal` status rather than passing as proven.

3. **Built the control's hub set by the deliberately naive rule:** walk candidates best-reward-first,
   add each one's parent intermediate, dedup, stop at 64. Reached 64 after the top **191** candidates;
   depths 1/2/3 = 14/17/33; **55% overlap** with the flow-ranked set, i.e. genuinely different.

4. **Enumerated those 64 hubs** (job 72522, 48 min) → 131,474 children, then swept BC-Enum-SB over
   N ∈ {2k, 5k, 10k, 21k, 50k} (job 73097, 13.5 min).

5. **Two infrastructure failures worth recording, because both were silent.**
   * The first BC-SB submission "COMPLETED" in 12 s on all three seeds having done nothing: `$HOME` is
     read-only on Balam compute nodes and results were being written into the repo. Exit code 0, so
     SLURM reported success. Results now go to `$SCRATCH`.
   * Worse, the sweep's stop-on-failure logic printed *"this is the practical ceiling"* — a solver
     limitation claim generated by a filesystem error. A failure at the **smallest** size now
     hard-exits as a RUN ERROR and explicitly refuses to be called a ceiling.
   Both would have been caught in seconds by a `debug`-partition smoke, which is now the documented
   first step. Measured turnaround there: **submit → terminal in 11 s**.

### Results

**The headline, at a 300-reaction budget** (the deepest point all three arms share). Read the
*distinct* column; the cost column is where the optimizer wins.

| arm | candidates chosen | **distinct molecules** | reactions / candidate |
|---|---|---|---|
| hub-batching (ours) | 228 | **228** | 1.32 |
| BC-SB (n=3 seeds) | 266–279 | **47.7 ± 1.5** | ~1.08 |
| BC-Enum-SB (n=1) | 296 | **14** | 1.01 |

At a 100-reaction budget: hub-batching **82** distinct; BC-SB 96 candidates → **12** distinct;
BC-Enum-SB 98 candidates → **2** distinct.

**BC-SB gets cheaper per molecule as the pool grows, and no less redundant** (seed 42, read at
~100 candidates; all `Optimal`):

| pool N | reactions / candidate | % distinct |
|---|---|---|
| 500 | 1.485 | 19.8% |
| 2,000 | 1.302 | 11.5% |
| 5,000 | 1.179 | 14.1% |
| 10,000 | 1.111 | 7.8% |
| 21,000 | **1.042** | 12.5% |

Across seeds at N=21,000: **1.072 ± 0.029** reactions/candidate, **16.5 ± 3.7%** distinct. The solver
handled the entire 21,001-molecule pool — no ceiling was reached.

**BC-Enum-SB is cheaper still and markedly less distinct:**

| pool N | reactions / candidate | % distinct |
|---|---|---|
| 2,000 | 1.136 | 14.5% |
| 5,000 | 1.042 | 2.1% |
| 21,000 | 1.031 | 4.1% |
| 50,000 | **1.020** | **2.0%** |

**Why — measured, not inferred.** The selection concentrates on very few intermediates:

| budget | candidates | distinct intermediates drawn from | share from the largest |
|---|---|---|---|
| 100 rxn | 98 | **3** | **78%** |
| 300 rxn | 296 | 6 | 45% |

and `reactions / candidate` sits pinned at **≈1.02 across every budget** — the signature of exactly
this: once an intermediate is built, each further child off it costs about one marginal reaction.

**Caveat on ratios.** At the tightest budgets BC-Enum-SB yields 2–3 distinct molecules, so any
"reactions per distinct molecule" figure derived from it is unstable (one molecule moves it by ~2×).
Raw counts are the honest presentation, and the tables above use them.
