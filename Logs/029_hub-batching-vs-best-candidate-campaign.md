# SCENT — hub-batching vs best-candidate: a budget campaign (reactions per mode)
**Date:** 2026-07-11, ~1pm

## Question

To build a diverse library of hits from a trained cost-aware generator, is it cheaper — in total
synthesis reactions per distinct hit — to **batch from hubs** (build a shared scaffold once, then
diversify it) than to just **pick the best individual candidates**, and how do the two trade off as
you spend more reactions?

## Context & Summary

Earlier entries (`026`, `028`) costed a *fixed* set of molecules two ways — "build these exact N
molecules independently" vs "…recognising their shared hub." That framing had a flaw: the two plans
build the *same* molecules, so they need the *same* reusable intermediates, and the intermediate
cost cancels out — hiding the very effect we care about (see `028`'s [OUTDATED] note). The right
comparison is between two **selection strategies run from scratch under a budget**, which produce
*different* libraries with *different* intermediate breadth. That is this entry.

We compare, on the same trained SCENT model:
- **best-candidate** — take the top-reward molecules the generator sampled, keep the diverse hits
  (modes); each is synthesised on its own (SCENT's promoted "dynamic-library" fragments are shared
  reusable stock, built once). This mirrors how the RGFN paper reports its top-k.
- **hub-batching** — rank hubs (parents of the top candidates, by flow), and for each, build the
  shared scaffold once and diversify its enumerated children into modes.

Both are scored on the *same* honest cost — total reactions, counting each distinct intermediate
once and unwinding nested fragments-built-from-fragments (the `FragmentCostTable` from `028`) — and
compared two ways: **Case 1**, how many modes fit in a fixed reaction budget; **Case 2**, how many
reactions to reach a fixed number of modes (≈ oracle calls). The hypothesis is that hub-batching
*concentrates* intermediate synthesis (few hubs → few distinct intermediates shared across many
hits), while best-candidate spreads across a broad, expensive set of intermediates.

## Answer

**Hub-batching roughly halves the synthesis cost per hit.** To build 300 diverse sEH hits (modes) at
our default diversity cutoff (Tanimoto **0.5** — a stricter, more meaningful "distinct" than the
0.7 the paper uses), hub-batching needs **820 reactions (2.73 / mode)** vs best-candidate's **1,479
(4.93 / mode)**; at a fixed 100-reaction budget it yields **33 modes vs 16**. Both libraries are
fully diverse (300 distinct Bemis-Murcko scaffolds) at essentially the same quality (best sEH 8.40
both). The catch is where the win comes from — and it is *not* the intermediate concentration we
guessed: hub-batching's 300 modes come from just **two flow-ranked hub scaffolds**, each built once
and diversified by cheap one-reaction steps, whereas best-candidate rebuilds each hit's full
~5-reaction assembly. Hub batching actually touches *more* distinct promoted intermediates (328 vs
205) — the saving is pure **scaffold amortization**, not fewer intermediates. And it isn't free:
hub-batching spent **18,856 reward-generator evaluations (63 / mode)** enumerating + scoring those
hubs' neighborhoods to find the hits, while best-candidate reused the generator's sample rewards (0
extra). That trade — cheap synthesis bought with expensive enumeration scoring — is exactly the axis
the active-learning loop will have to balance. *(At the looser 0.7 cutoff the win is even larger — 706
vs 1,445, all from one hub — but 0.7 counts near-identical molecules as distinct; see the sweep.)*

## Relevance to our Publication

This is the headline "why batch from a hub" result, framed the way a reviewer (and a chemist) would
actually pose it: *given a synthesis budget, which selection strategy yields more diverse hits?* It
generalises the RGFN-style top-k reporting to a cost-aware, hub-aware setting, and the two strategies
share an interface (`CampaignResult`) so the same comparison drops straight into the active-learning
loop later (Objective 1's oracle-efficiency curve).

## Next Experiments

**Refining for publication**
- **Hub-batching curve** (enumeration job 70295) → the head-to-head modes-vs-reactions curve + Case
  1/2 numbers on sEH; then DRD2.
- **Hub / mode-selection strategy sweep** — the hub ranker and within-hub mode acceptor are pluggable;
  compare flow-ranked vs reward-ranked hubs, diverse+threshold vs other acceptors.
- **Enumeration speedup (measured — TODO, come back to it).** Full-flow enumeration is **~3m39s/hub**
  (one depth-1 hub, 9,428 children, cap 12k; ~50 hubs ≈ 2–4 h). Most of that is `assign_log_probs`
  (forward **and** backward policy passes) + backward-action-space construction — computed only to
  recover the flow field (`F_hat`/`U(h)`), which the **campaign discards**. A `--no-flow` enumerate
  mode (forward DFS → child + proxy reward + added fragment; skip both) would be much faster **and
  more complete** (it keeps children the P_B-invertibility check currently drops), letting us
  enumerate many more hubs in the same walltime. **Caveat:** `--no-flow` drops the per-child flow
  terms, so keep the flow-on path available whenever `U(h)` on enumerated neighborhoods is wanted
  (see below). Not built yet — deferred.
- **`U(h)` is tracked, not yet used as a signal.** The flow-matching-residual uncertainty
  `U(h) = Var_i[log F_hat(h;x_i)]` (proposal §2, an epistemic-uncertainty / AL acquisition signal
  for later) stays easy to extract: sampled hubs carry it in `hub_summary.csv` / `report.json`
  (`hub.uncertainty()`), and enumerated hubs now carry it per hub in `enum_children.json` →
  `EnumeratedHub.uncertainty` (+ `n_effective`), computed from the flow terms. The campaign does not
  consume it yet — the infrastructure just keeps it one field away.

**Next steps in project**
- Wire the two strategies into the **active-learning loop** (they're already swappable) and measure
  the oracle-efficiency curve for each (`[bengio2021gflownet]` Fig. 7 analogue).
- The RGFN-vs-SCENT hub-coincidence study (proposal §8) using the same campaign machinery.

# Re-creation

## Relevant Files

Root: `./` (repo root).

**AL-ready strategy code (ours, new — `glue/`):**
- `./glue/samplers/lsdflow/campaign.py` — `BestCandidateStrategy` / `HubBatchingStrategy` (independent
  constructors, **disjoint inputs** — best-candidate never touches hub/enumeration data — identical
  `CampaignResult` output so the AL loop swaps them freely) + the count-once running-built-set cost.
- `./glue/samplers/lsdflow/mode_select.py` — pluggable `DiverseThresholdModeSelector` (reward-gated +
  Tanimoto-dedup, ECFP r=3/0.7; the paper mode definition from `026`, still current).
- `./validation/lsdflow/metrics/cost/dynamic_amortization.py` — `FragmentCostTable` (nested
  per-fragment cost from the logged routes; from `028`, still current).

**Analysis (ours, new — `experiments/lsd_hubs/campaign/`):**
- `pick_hubs.py` — top-K candidates → parent hubs → rank by single-candidate flow → `hubs.csv`.
- `submit_scent_seh_enum.sh` — GPU pre-enumeration (pick_hubs → scent-env worker `--mode enumerate`
  → `enum_children.json`); uses the recipe re-run checkpoint (`028`, job 70180).
- `run_campaign.py` — single head-to-head point (both strategies at one cutoff/budget); writes
  `results/<tag>/summary.json` + the per-run cumulative `curve_*.csv`/`curve.png`. Also holds the
  shared loaders (`_load_candidates` / `_load_enumerated_hubs`) that `sweep_campaign.py` imports.
- `sweep_campaign.py` — the diversity/budget sweeps: re-runs the greedy selection over the cached
  enumeration at 13 cutoffs, writes `results/<tag>/{pareto,fixed_modes,budget_efficiency}.{csv,png}`
  + `sweep_summary.json`. Pure CPU (~44 s for sEH); no GPU / no re-scoring.
- `hub_stats.py` — per-hub depth / children / `U(h)` / reward table from the enumeration →
  `results/<tag>/hub_stats.csv`.
- `README.md` — the pipeline + status. **All generated artifacts live under `results/<tag>/`**
  (e.g. `results/scent_seh/`); the dir carries the target, so filenames are untagged.

**Worker (ours, extended):** `./validation/lsdflow/adapters/workers/scent_worker.py` — `--mode
enumerate` now emits `enum_children.json` (each child + the promoted fragment added in its final
reaction), for the campaign's cost/composition.

**Inputs (on $SCRATCH):** SCENT sEH analysis DAG `lsdflow/scent_seh_70189/` (`records.csv` +
`compositions.json`); recipe re-run `fixed_reward/scent_seh/2026-07-10_17-28-06/` (checkpoint +
`fragments_4000.json` with routes).

**Job Logs:** enumeration **70295** (`/scratch/markymoo/rgfn_runs/campaign_enum_seh-70295.{out,err}`).

## Relevant Versions

Branch `Hub-Analysis`. New campaign + analysis files (above) + the `026`/`027`/`028` [OUTDATED]/scope
notes. **Not yet committed.** [TODO — add commit hash after committing.]

## Relevant Resources

**Sources**
- `docs/LSD_FLOW_PROPOSAL.md` §11 (cost metrics), §4a (AL-facing selection). Entries `028` (nested
  cost engine + recipe re-runs, its fixed-set cost comparison superseded here), `027` (SCENT
  cross-env adapter + freeze + enumeration + recipe logging — the still-current infrastructure this
  builds on), `026` (the mode definition, still used).
- `[bengio2021gflownet]` (top-k / modes; Fig. 7 oracle efficiency), `[gainski2025scent]` (dynamic library).

**Packages**
- `rgfn` env (campaign + analysis, CPU); `scent` env (the enumeration worker).

## Method

1. Built the two swappable strategies + pluggable mode acceptor; unit-tested (disjoint inputs,
   identical output, count-once nested cost with/without hub-scaffold sharing).
2. `pick_hubs.py` on the sEH DAG: 26,069 candidates → top-100 → 64 distinct hubs → 50 ranked by
   single-candidate flow.
3. Validated best-candidate on the real sEH DAG (below). Queued the hub enumeration (70295); the
   head-to-head is produced by `run_campaign.py` once it lands.

## Results

**Head-to-head (sEH, hit bar ≥ 7.0, default diversity cutoff 0.5; enumeration job 70295 = 50 hubs;
`run_campaign.py`):**

| metric | best-candidate | hub-batching |
|---|---|---|
| reactions to generate 300 modes | 1,479 rxns (4.93 rxns/mode) | **820 rxns (2.73 rxns/mode)** |
| Case 1: modes at 100 reactions | 16 | **33** |
| reward-gen calls (/ mode) | 0 (0) | 18,856 (62.9) |
| distinct promoted intermediates¹ | 205 | 328 |
| distinct hubs used | 0 | **2** |
| Bemis-Murcko scaffolds (of 300) | 300 | 300 |
| best / median sEH | 8.40 / 8.10 | 8.40 / 7.48 |

¹ intermediate counts (205 / 328) are for the **fixed-mode** (300-mode) budget — the run the
whole table reports; Case-1 (100-rxn) numbers are read off the same curve. (Numbers are at the
**default cutoff 0.5**; the looser 0.7 gives 706 vs 1,445 from a single hub — see the sweep.)

**Why hub-batching costs 2.73 rxns/mode, not ~1, from two shallow hubs.** A shallow hub +
one final reaction per child *looks* like it should approach 1 rxn/mode once the scaffold is
amortized. It doesn't, because in SCENT the fragment attached in that final reaction is usually a
*promoted dynamic-library fragment*, which is **not** free stock — it has its own (often nested)
synthesis route that we charge once. Decomposing the 820 reactions over the 300 modes:
scaffold build once = 4 rxns (step 1); then per additional mode, **242 cost 3** (final rxn + a
fresh 2-reaction nested intermediate), **30 cost 2** (final rxn + a fresh 1-reaction intermediate),
and only **26 cost 1** (the intermediate was already built for an earlier mode — the pure
amortization case). So the marginal cost is dominated by *building a distinct promoted intermediate
per hit*, exactly as hypothesised — the win over best-candidate is that the shared scaffolds are
built once, not that intermediates disappear. Tightening the cutoff from 0.7 → 0.5 costs a bit more
(706 → 820 rxns, one hub → two) because stricter diversity forces a second scaffold and more fresh
intermediates — but the ~1.8× advantage over best-candidate holds.

**Hub statistics.** Preliminary set = the parent hubs of the top-100 sEH candidates: **64 distinct
hubs** (depth-1 = 22, depth-2 = 29, depth-3 = 13); average single-candidate flow `F_hat` falls with
depth (d1 = 68.8, d2 = 67.4, d3 = 64.8) — shallower hubs sit on more paths. We enumerated the
**top 50 by flow** (d1 = 21, d2 = 25, d3 = 4), 148,153 child records total. Per-depth over those 50
(`results/scent_seh/hub_stats.csv`):

| depth | n hubs | avg children | avg U(h) | avg hits ≥ 7 | avg max reward |
|---|---|---|---|---|---|
| 1 | 21 | 5,431 | 47.6 | 1,512 | 8.34 |
| 2 | 25 | 1,200 | 42.7 | 231 | 8.31 |
| 3 | 4 | 1,023 | 39.0 | 362 | 8.30 |

Depth-1 hubs have both far more children *and* far more hits (avg 1,512 vs 231) — which is why a
single depth-1 hub can supply all 300 modes; the max reward is depth-independent (~8.3), so the
diversification neighborhood scales with breadth, not quality. `U(h)` also falls with depth (47.6 →
39.0): the broad depth-1 neighborhoods span a wider range of terminal flow. The rank-1 hub actually
used — `O=C(O)c1ccc2nc(C3CNC3)oc2c1` — has 9,428 children, **2,665 of them hits (28.3%)**, max sEH
8.38, median 6.52, and `U(h) = 36.2` (a *lower*-uncertainty depth-1 hub — reassuring, though the
campaign doesn't select on U(h) yet). Full per-hub table committed as
`experiments/lsd_hubs/campaign/results/scent_seh/hub_stats.csv`.

**Enumeration size / persistence.** The full enumeration is saved on `$SCRATCH`
(`lsdflow/campaign_enum_seh_70295/`): `enum_children.json` = **24 MB** (every hub's children +
reward + fragment added), `enumerated_records.csv` = **42 MB** (per-child flow terms → recomputable
`U(h)`). Too large for git; only the small `summary`/`curve`/`hub_stats` artifacts are committed.

At the **default cutoff 0.5** the campaign draws on **two** flow-ranked hubs (the rank-1
`O=C(O)c1ccc2nc(C3CNC3)oc2c1` + a second scaffold), because the stricter cutoff exhausts one hub's
diverse-enough children before 300 modes. (At the looser 0.7 a single hub supplies all 300.) Either
way the win is **scaffold amortization** — a couple of scaffolds built once + cheap one-reaction
diversifications — not intermediate concentration (hub-batching uses *more* distinct intermediates,
328 vs 205 at 0.5). Artifacts committed to `experiments/lsd_hubs/campaign/results/scent_seh/`:
`summary.json`, `curve_{best_candidate,hub_batching}.csv`, `curve.png` (all at the 0.5 default).

**`U(h)` note.** Job 70295 ran the worker version *before* the `U(h)`-in-`enum_children.json` change,
so that field is absent there — but `U(h)` is fully recoverable from the persisted flow terms in
`enumerated_records.csv` (top hub `U(h) = Var_i[log F_hat] = 36.2` over its 9,428 children), and the
current worker emits it per hub automatically on the next enumeration. Not used as a signal yet.

### Diversity / budget sweeps (`sweep_campaign.py`)

The single head-to-head above fixes one diversity cutoff (the default 0.5) and one budget. To see the
*tradeoff* we sweep. **Framing correction:** each simulation uses ONE predefined budget and yields ONE point;
curves are built by re-running the (cheap, CPU) budget-greedy selection over the **cached**
enumeration (rewards already computed — no re-scoring, no GPU), changing only the diversity cutoff or
budget. One run per cutoff to `("modes", 300)` yields all anchors of that cutoff off its accepted
prefix (best-reward-first order is budget-independent, so this reads "modes at ≤ R* reactions"
exactly — a run *stopped* at R* reactions overshoots by the mode that crosses R*). All 26 runs =
**44 s** on the login CPU. Three plots, each hub-batching vs best-candidate:

1. **Pareto — modes at a fixed 100-reaction budget vs cutoff** (`results/scent_seh/pareto.*`).
2. **Cost — reactions to reach 300 modes vs cutoff** (`results/scent_seh/fixed_modes.*`).
3. **Budget vs efficiency — modes vs reaction budget at the default cutoff 0.5**
   (`results/scent_seh/budget_efficiency.*`; the one inherently-cumulative curve, read off a single
   run's prefix chain). Pareto/cost plots also mark the 0.5 default with a dashed line.

Sweeping similarity **0.30 → 0.90 (step 0.05)**, hit bar fixed at 7.0:

| cutoff | hub modes@100rxn | best modes@100rxn | hub rxns→300 | best rxns→300 | hubs walked (hub) |
|---|---|---|---|---|---|
| 0.30 (strictest) | 32 | 17 | **can't** (135 max, 28 hubs) | 1,386 | 28 |
| 0.50 | 33 | 16 | 820 | 1,479 | — |
| 0.70 | 35 | 18 | 706 | 1,445 | 1 |
| 0.90 (loosest) | 40 | 17 | **644** | 1,412 | 1 |

**Findings.** (a) **Hub-batching dominates efficiency across the whole diversity range it can serve**
— ~2× more modes per 100 reactions (32–40 vs 16–18) and ~2× fewer reactions for 300 modes (644–824
vs 1,386–1,479). best-candidate is essentially flat in the cutoff (its broad sampled pool is diverse
at any threshold); hub-batching's advantage *grows* as diversity loosens (one hub serves everything →
maximal amortization) and *erodes* as it tightens. (b) **As the cutoff tightens, hub-batching is
forced to walk more hubs** (1 hub at 0.70 → 26 at 0.35 → 28 at 0.30): each hub's neighborhood
saturates under strict diversity, so it must recruit new scaffolds — eroding the amortization win.
(c) **The scaffold-concentration ceiling.** At the strictest cutoff (0.30) hub-batching **cannot
reach 300 modes at all — it maxes out at 135** (28 of 50 hubs exhausted), because the enumerated
hubs' children are all decorations of a few cores and can't supply 300 mutually-Tanimoto-<0.3 hits;
best-candidate reaches 300 from the full 26k sampled pool. This is the genuine cost of trading
scaffold diversity for synthesis efficiency — and it is *partly* an artifact of enumerating only 50
hubs (more hubs raise the strict-cutoff ceiling → ties directly into the deferred `--no-flow`
enumeration speedup). Artifacts (committed, under `results/scent_seh/`): `sweep_summary.json`,
`{pareto,fixed_modes,budget_efficiency}.{csv,png}`. The old single-run `curve.*` is superseded by
`budget_efficiency.*` (identical at the default cutoff 0.5).

### Larger run → promoted to entry `031`

The 200-hub / cutoff-0.10 scale-up of this campaign is written up as its own experiment, **`031`**
(`results/scent_seh_1kx200/`, job 70363). Headline: scaling to 200 hubs moves the strict-cutoff
ceiling *down* — hub-batching reaches 300 modes at cutoff 0.30 (898 reactions), where the 50-hub run
here capped at 135 — but below ~0.30 *neither* strategy reaches 300 (a chemical-space floor, since
best-candidate also fails). See `031` for the full sweep table and per-hub stats.
