# SCENT / sEH — does sorting hubs by flow actually buy anything?

**Date:** 2026-07-29, ~3pm (enumeration ran overnight; analysis completed 2026-07-30 ~00:45)

## Question

When we build a diverse library by picking a few molecular scaffolds and decorating each one many
ways, does it matter *which* scaffolds we pick — or would picking them at random, or worst-first,
work just as well?

## Context & Summary

Our central claim is that a trained generative model's internal "flow" field already knows which
intermediates are worth building once and diversifying many times. We act on that claim by ranking
candidate scaffolds ("hubs") by their estimated flow and working down the list. Every cost result we
have — entries `029`, `031`, `033`, `035`, `050`, `052` — uses that one ranking, and none of them
tests it. A reviewer's obvious question is whether the ranking is doing any work at all: if you get
the same library from a random ordering, the flow field isn't the source of the advantage, and the
paper's whole framing is decoration on top of "enumerate some scaffolds and pick the good children."

Two things make this worth testing carefully rather than assuming. First, the shipped ranking is not
a pure flow sort: it keeps only the parent scaffolds of the top-1000 highest-scoring molecules and
*then* sorts those by flow, so a reward filter has already done most of the selecting. Second, the
flow estimate is itself dominated by the molecule's score (the reward enters multiplied by 8, while
the policy terms span about one score-unit), so flow order and score order are correlated but not
identical — Spearman 0.76 across all 20,874 scaffolds we observed.

So we run six versions of the same campaign, changing **only** which 200 scaffolds hub-batching may
use and in what sequence: the shipped ranking; highest-flow-first over *all* scaffolds (no reward
pre-filter); lowest-flow-first; a random order; and two "best-candidate" controls that order
scaffolds by how good their single best molecule was — one over all scaffolds, one restricted to the
shipped ranking's exact 200 scaffolds so that *ordering* is isolated from *selection*. Everything
else is held fixed: same trained model, same sampled pool, same cost model, same acquisition
settings, same 200-scaffold budget. We then ask each arm the same question — how many reactions does
it take to build a 300-molecule diverse library — across the full range of what counts as "diverse."

## Answer

**The flow field is doing real work, but the honest statement is about *neighbourhood*, not rank
order.** Reversing the sort breaks the method outright — lowest-flow-first cannot build the
300-molecule library at all, running out of all 200 scaffolds at 279, while burning more compute than
any other arm. A random order does complete, but costs 1.53× the reactions of flow order. Yet the
three arms that all draw on *good* scaffolds — flow over everything, flow over the reward-filtered
pool, and ordering by best-molecule score — land within 9% of each other. So flow reliably identifies
which region of the scaffold space is worth building in; it is not finely calibrated within that
region.

**The sharpest result is on the two cost axes together.** Reactions (bench cost) and compute
(enumeration cost) trade against each other, and *every* point on the resulting frontier is a
flow-informed or reward-informed ordering: flow-first is cheapest in reactions (357) but most
expensive in compute (5,648 s), while ordering by best-molecule score costs 9% more reactions (389)
for **2.3× less compute** (2,481 s). Both no-signal controls are strictly dominated — random is beaten
on *both* axes simultaneously by the candidate ordering, and reverse-flow is beaten by everything. The
mechanism is visible in what each ordering selects: flow prefers shallow scaffolds (11 depth-0 /
150 depth-1), which have thousands of children each — more reaction savings per scaffold, but a much
bigger enumeration bill.

**Why the best-candidate ordering does so well has a concrete explanation, and it favours us.** A
trained GFlowNet samples trajectories roughly in proportion to flow, so the molecules it produces
have *already* been routed through high-flow intermediates. Taking the parents of the best sampled
molecules is therefore an indirect way of selecting high-flow hubs. Measured directly: the
candidate-ordered arm never reads the flow estimate, yet its 200 hubs sit at median rank **546 of
20,874** — the top 2.6% — with 63% inside the top 1,000. The random arm sits at median 11,043,
matching the uniform-draw null of 10,437. So the three "good" arms agree *because they are all
selecting from the high-flow region by different routes*, which is a positive statement about the
flow field rather than evidence that hub choice is unimportant.

**Addendum — the flow field moves a lot during training, but that does not touch these results.**
Prompted by the question "could hubs be high-flow early in training and low-flow by the end?",
`flow_drift_over_training.py` reads the training log's per-molecule synthesis paths and locates each
era's hubs in the final model's flow ranking. Only **7.3%** of the hubs visited in the first 250
iterations still appear in the post-training sample, rising monotonically to **24.2%** for the last
250; among the survivors, median final flow-rank improves from 6,192 to 3,438. Both curves step
sharply at iteration ~1000 — exactly `DynamicLibrary.every_n_iterations`, the first fragment
promotion. **This is not a confound for the ablation**: our candidate pool and every flow term come
from sampling the *final* checkpoint, so no training-time signal enters. It matters for anything that
consumes training-time signal — most concretely SCENT's dynamic library, whose promoted fragments are
selected by a training-time utility and then frozen into all downstream analysis.

**Two caveats we should state ourselves.** The advantage of every arm collapses to ~1.04× by a
diversity cutoff of 0.9, where "distinct" is loose enough that almost any scaffold's children qualify
and scaffold quality stops binding. And the whole comparison is one model on one target with one
random draw, so it establishes the effect exists, not its size across models.

## Relevance to our Publication

`docs/paper_planning/lsd-flow-publication-strategy.md` lists "hub definition arbitrariness" and the
baseline suite ("random hub; parent-of-top-N") as free criticisms we currently have no answer to,
and §4 anticipates the sharper attack that a post-hoc extraction is a heuristic rather than a
principled method. This entry is the direct answer: it is the ablation that separates "the flow field
identifies batchable neighbourhoods" from "any 200 scaffolds would do." The best-candidate-order arms
matter most — if ordering by a scaffold's best molecule does *worse* than ordering by flow, then hub
value is not readable off the reward and the flow field is carrying structural information the reward
ranking is not, which is exactly the claim §1.3(c) wants to make ("sort hubs by flow" = "sort hubs by
the reward mass they capture").

## Next Experiments

**Refining for publication**

- Repeat on a second generator/target once the arms are settled — entry `050` already has sampled +
  enumerated pools for RxnFlow and DRD2, so the same drivers answer whether the ordering effect is
  general or SCENT-specific.
- Extend to the incoming SCENT sEH seeds 43/44 (jobs 71732/71740) for a three-model version of the
  same ablation, which is a stronger generality statement than repeating one model with more random
  draws.
- `flow_bottom` is pool-limited across cutoffs 0.30-0.50, so its cost there is a lower bound. A
  larger scaffold budget for that arm alone would separate "this ordering is bad" from "this
  ordering needed more scaffolds" — worth doing before the number appears in a paper.
- **Follow up the pre-select-K pool coupling (above).** Three threads, all cheap and CPU-only:
  (a) *Fairness* — pre-select currently reads fan-out from hubs whose enumeration is not charged to
  the strategy. Compare against an honest online variant that ranks fragments using only the hubs
  walked so far, and against charging the full pool; if the gap is real, the online variant is the
  defensible one to report. (b) *The saturation is an opportunity, not just a caveat* — if the top-K
  fragments are already identifiable from ~100 hubs, a cheap "scout" enumeration could choose the
  stock and a much smaller walk could then exploit it, cutting the enumeration bill that is
  hub-batching's main cost (Logs/037's reactions↔calls dial, from the other end). (c) *Sensitivity* —
  how does the coupling scale with K and with pool size? Measured here only at K=20 on one arm.

- **Save periodic checkpoints in future training runs.** The drift measurement above is indirect —
  it infers movement from which hubs survive into the final sample, because only `last_gfn.pt` and
  `best_gfn.pt` exist. With checkpoints every ~1000 iterations the direct question becomes
  answerable: recompute `F(h)` for a fixed hub set under each checkpoint and watch the ranking move.
  Cheap to enable, and it would also let us ask whether the dynamic library's promotion decisions
  still look right under the final flow field (the §8 hub-coincidence study).
- **Test the sampling-concentration hypothesis properly.** The rank measurement above is consistent
  with "sampling routes molecules through high-flow hubs, so best-candidate order is a flow proxy",
  but it is correlational: high-flow hubs also tend to carry high-reward children, so the two
  explanations are not separated by this data. A clean test would compare the parent-hub flow
  distribution of *sampled* molecules against that of *uniformly enumerated* molecules of matched
  reward — if sampling is doing the concentrating, only the former should be flow-skewed.

**Next steps in project**

- Feed the winning ordering into the active-learning acquisition function, where hub choice is made
  once per round rather than once per campaign.

# Re-creation

## Relevant Files

Root: `./` = repo root; `/scratch/markymoo/rgfn_runs/` for run artifacts.

**Scripts**
- `./experiments/lsd_hubs/campaign/pick_hubs.py` — the hub ranker. Extended here with `--pool`
  (`topk_candidates` legacy | `all`), `--order` (`flow_desc` | `flow_asc` | `random` |
  `candidate_reward`), `--seed`, and `--restrict-to` (apply an order to a fixed hub set). Defaults
  reproduce the pre-existing recipe byte-for-byte — verified by diffing against the canonical
  `campaign_enum_seh_70363/hubs.csv`.
- `./experiments/lsd_hubs/hub_order/plan_arms.py` — builds all six arms' hub sets, subtracts hubs
  already enumerated anywhere, and slices the remainder into debug-sized GPU jobs using per-depth
  times measured from previous runs. Re-runnable: folds finished slices back into the cache and
  refits its cost model.
- `./experiments/lsd_hubs/hub_order/submit_slice.sh` — one `debug` job = one hub slice.
- `./experiments/lsd_hubs/hub_order/chain.sh` — sequential submit/wait driver (the `debug` QoS is
  `MaxJobsPU=1` **and** `MaxSubmitJobsPU=1`, so `--dependency` chains are rejected at submit time).
- `./experiments/lsd_hubs/hub_order/merge_enum.py` — assembles each arm's 200 hubs from the shared
  cache **in that arm's own walk order** (the order is the strategy).
- `./experiments/lsd_hubs/hub_order/run_arms.sh` — per-arm `run_campaign.py` + `sweep_campaign.py`.
- `./experiments/lsd_hubs/hub_order/compare_hub_order.py` — the overlay figures + operating-point
  table; asserts best-candidate is identical across arms.
- `./experiments/lsd_hubs/hub_order/hub_rank_overlap.py` — where each arm's 200 hubs fall in the
  all-hub flow ranking; the measurement behind the sampling-concentration explanation.
- `./experiments/lsd_hubs/hub_order/flow_drift_over_training.py` — how far the flow field moves
  during training, from the training log's per-molecule paths (`paths.csv`). Note `iteration` there
  counts **molecules** (64 per training step), not steps.

**Models**
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/train/checkpoints/last_gfn.pt`
  — the SCENT sEH anchor. 5,000 iterations (`paths.csv` = 320,001 rows); `guidance_models.pt`
  sidecar present so P_B is the model's own trained backward policy (entry `024`); **the only sEH
  run whose fragment snapshot carries `smiles_to_route`**, which is what makes the exact nested cost
  model available. Rejected alternatives: `2026-07-01` (crashed, no checkpoint), `2026-07-02`
  (`BACKWARD_POLICY_NOT_SAVED.txt` — P_B irrecoverable), `2026-07-07` (complete but no recipes),
  `scent_seh_5k/seed42` (complete but no recipes; the matrix16 cell).
- `.../scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json` — the frozen dynamic
  library: 418 base + 1,600 promoted fragments, all 1,600 carrying synthesis routes.

**Datasets**
- `/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/` — the sampled DAG all arms select from.
  30,000 trajectories → 29,997 terminal transitions, 20,874 distinct hubs, 26,069 unique candidates,
  logZ 74.3259.
- `/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/` — the incumbent enumeration
  (200 hubs / 828,448 children, 5:36:38). Serves as both the `incumbent` arm and the shared hub
  cache; supplies 91 of `flow_top`'s hubs, 109 of `cand_order`'s and all 200 of
  `cand_order_fixedset`'s.
- `/scratch/markymoo/rgfn_runs/lsdflow/hub_order/` — this experiment's scratch root: `arms/`
  (per-arm hub sets + slices), `enum/` (per-slice worker output), `merged/` (per-arm assembled
  enumeration), `manifest.json`, `chain.log`.

**Results**
- `./experiments/lsd_hubs/hub_order/results/hubord_<arm>/` — per-arm `summary.json` (operating
  point) + `sweep_summary.json` (the cutoff sweep) + figures.
- `./experiments/lsd_hubs/hub_order/results/comparison/` — `cost_vs_cutoff.png` (the headline:
  reactions to 300 modes vs diversity cutoff, per arm), `cost_pareto.png` (the two cost axes and the
  dominance structure), `hub_rank_distribution.png` (+ `.csv`), `compute_vs_cutoff.png`,
  `summary.csv`.

**Job Logs**
- `/scratch/markymoo/rgfn_runs/hubord_enum-<jobid>.{out,err}` — one pair per slice.

## Relevant Versions

Branch `Hub-Analysis`, worktree branch `worktree-hub-order-ablation` off `7d99b27`.
Infrastructure `1a6dcb7`, index row `21247f1`, per-arm results `e04f85a` / `6de1deb` / `b846552` + the final commit carrying this write-up.
**Not pushed** — no git credentials / `gh` in the run environment; the branch is local to the worktree at `.claude/worktrees/hub-order-ablation`.

## Relevant Resources

**Sources**
- `docs/LSD_FLOW_PROPOSAL.md` §2 (the flow recovery), §12 (baseline suite: "random hub;
  parent-of-top-N") — this entry implements the missing baselines.
- `docs/paper_planning/lsd-flow-publication-strategy.md` §1.3(c), §4, §5 (hub-definition
  arbitrariness; the post-hoc-heuristic criticism).
- `[malkin2022trajectorybalance]` — detailed balance, the identity `F_hat` rearranges.

**Packages**
- SCENT (`external/scent`, `scent` conda env) — `validation/lsdflow/adapters/workers/scent_worker.py`
- RDKit — enumeration + diversity (`validation/lsdflow/metrics/diversity.py`)

## Method

1. **Verify the substrate.** Audited all five SCENT sEH training runs for iteration count, checkpoint
   presence, P_B recoverability and recipe logging; confirmed the sample job (70189) and enumeration
   job (70363) both name the `2026-07-10_17-28-06` checkpoint and report the same logZ.
2. **Extend the ranker.** Added the pool/order/seed/restrict options to `pick_hubs.py`; confirmed the
   default invocation still reproduces `campaign_enum_seh_70363/hubs.csv` exactly.
3. **Plan the arms.** `plan_arms.py --out-root $SCRATCH/rgfn_runs/lsdflow/hub_order`.
4. **Enumerate the new hubs.** `chain.sh`, one `debug` job at a time.
5. **Assemble + analyse.** `merge_enum.py`, then `run_arms.sh` (per-arm campaign + cutoff sweep),
   then `compare_hub_order.py`.
6. **Explain the agreement between arms.** `hub_rank_overlap.py` — rank every observed hub by the
   same `max_x F_hat(h;x)` the production ranker uses, then locate each arm's 200 hubs in it.

## Results

**Operating point (τ=7.0, cutoff 0.5, budget 300 modes, `free_frag` + `prebuild_k=20`).** n=1 model,
1 target, 1 random draw. "Compute" is measured hub-batching wall-clock attributed over the hubs each
arm actually walked (Logs/039), not modelled.

| arm | rxn/mode | modes | reactions | hubs walked | reward-gen calls | compute (s) | status |
|---|---|---|---|---|---|---|---|
| `flow_top` | **1.190** | 300 | 357 | 41 | 244,685 | 5,648 | Pareto frontier |
| `incumbent` | 1.217 | 300 | 365 | 43 | 241,158 | 5,685 | dominated by `flow_top` |
| `cand_order_fixedset` | 1.263 | 300 | 379 | 41 | 145,490 | 3,365 | Pareto frontier |
| `cand_order` | 1.297 | 300 | 389 | 41 | 109,897 | **2,481** | Pareto frontier |
| `random` | 1.820 | 300 | 546 | 94 | 184,159 | 4,332 | **dominated on both axes** |
| `flow_bottom` | 2.050 | **279** | 572 | 108 | 283,363 | 6,580 | **pool-exhausted** |
| best-candidate (ref) | 3.097 | 300 | 929 | 254 (accidental) | 0 | — | reference |

**Reactions to reach 300 modes, across the diversity cutoff.** `pool-lim` = the arm exhausted its 200
hubs before reaching 300 modes; those cells are excluded from ratios, never counted as wins.

| cutoff | flow_top | incumbent | cand_order_fixedset | cand_order | random | flow_bottom | best-cand |
|---|---|---|---|---|---|---|---|
| 0.30 | pool-lim | pool-lim | pool-lim | pool-lim | pool-lim | pool-lim | 1123 |
| 0.35 | pool-lim | pool-lim | pool-lim | pool-lim | pool-lim | pool-lim | 1080 |
| 0.40 | **466** | 484 | 477 | 521 | pool-lim | pool-lim | 1018 |
| 0.45 | **396** | 396 | 417 | 440 | 625 | pool-lim | 955 |
| 0.50 | **357** | 365 | 379 | 389 | 546 | pool-lim | 929 |
| 0.55 | **345** | 348 | 356 | 362 | 466 | 486 | 866 |
| 0.60 | 337 | 341 | 343 | **341** | 433 | 437 | 835 |
| 0.65 | **331** | 336 | 334 | 334 | 393 | 405 | 823 |
| 0.70 | 329 | 330 | 331 | **330** | 363 | 388 | 793 |
| 0.75 | **327** | 327 | 328 | 328 | 356 | 364 | 751 |
| 0.80 | 327 | **325** | 327 | 327 | 345 | 343 | 722 |
| 0.90 | 326 | **325** | 327 | 327 | 343 | 340 | 728 |

Ratio to `flow_top` where both complete: `random` 1.58× (0.45) → 1.53× (0.50) → 1.19× (0.65) → 1.05×
(0.90); `flow_bottom` 1.41× (0.55) → 1.22× (0.65) → 1.04× (0.90).

**Pool-limitation is itself monotone in flow quality** — the cutoff below which an arm can no longer
build the library: flow arms and `cand_order` < 0.40, `random` < 0.45, `flow_bottom` < 0.55. The flow
arms' 0.30/0.35 failure reproduces Logs/052's strict-corner finding on the same substrate.

**What each ordering selects (the mechanism behind the compute axis).**

| arm | depth mix (0/1/2/3) | children enumerated |
|---|---|---|
| `flow_top` | 11/150/37/2 | 1,077,049 |
| `incumbent` / `cand_order_fixedset` | 2/104/84/10 | 828,448 |
| `cand_order` | 0/59/89/52 | 535,476 |
| `random` | 2/19/51/128 | 350,764 |
| `flow_bottom` | 7/5/34/154 | 283,363 |

Flow prefers shallow scaffolds (~7,100 children each at depth 1 vs ~700 at depth 3): more reaction
savings per scaffold, a much larger enumeration bill. That is the trade the frontier expresses — and
note it runs *opposite* to naive intuition, since the arm that enumerates the FEWEST children
(`flow_bottom`, 283k) is also the one that fails.

**Where each arm's hubs sit in the all-hub flow ranking** (`hub_rank_overlap.py`; 20,874 hubs,
uniform-draw null = median 10,437, 5% in the top 5%). This is the evidence for the
sampling-concentration explanation above — and it is correlational, see Next Experiments.

| arm | median flow-rank | percentile | in top 1,000 | in top half |
|---|---|---|---|---|
| `flow_top` | 99 | 0.48% | 100% | 100% |
| `incumbent` | 215 | 1.03% | 100% | 100% |
| `cand_order` | **546** | **2.62%** | **63%** | 97.5% |
| `random` | 11,043 | 52.9% | 2.5% | 46% |
| `flow_bottom` | 20,773 | 99.5% | 0% | 0% |

**Flow drift over training** (`flow_drift_over_training.py`, 20 bins of 250 iterations). Survivorship
is confounded by the finite post-training sample; median rank conditions on presence and is not. They
agree.

| training iters | distinct hubs | % still in final sample | median final rank | % in top 5% |
|---|---|---|---|---|
| 0–250 | 13,141 | 7.3% | 6,192 | 11.8% |
| 1000–1250 | 12,507 | 15.6% | 4,822 | 16.8% |
| 2500–2750 | 12,418 | 19.9% | 4,120 | 19.5% |
| 4750–5000 | 11,941 | 24.2% | 3,438 | 22.1% |

**Caveat found while sizing the deeper `flow_bottom` arm — pre-select-K couples the result to the
whole enumeration pool.** The budget is 300 *modes*, and at cutoff 0.5 the completing arms stop
having consumed only 17–52% of their 200-hub pool (flow_top 22%, cand_order 20%, random 52%; only
`flow_bottom` reaches 100%). So enlarging the pool should be a no-op for them — *except* that
`rank_fragments` picks the pre-select-K fragments by scanning **every enumerated hub, including hubs
the walk never reaches**. Measured on `flow_top` by truncating its pool and re-running:

| pool | reactions to 300 modes | hubs walked | reward-gen calls |
|---|---|---|---|
| 200 hubs | 357 | 43 | 244,685 |
| 100 hubs | 352 | 39 | 231,948 |
| 60 hubs | 352 | 39 | 231,948 |

A 1.4% effect that **saturates below 100 hubs**. Two consequences. (1) It is small enough that the
600-hub `flow_bottom` arm remains a like-for-like comparison against the 200-hub arms, and the
`--prebuild-k 0` control removes the channel entirely. (2) There is an accounting inconsistency
worth naming: pre-select benefits from fan-out information across the *whole* pool, while the
compute-time axis charges only the hubs actually walked. Nobody would notice this from the headline
numbers — it surfaced only because the pool size changed.

**Enumeration cost.** 16 `debug` slices, 15:17 → 00:31 (~9.2 h wall clock) for 598 new hubs; 402 of
the 1,000 arm-hub slots were served from cache. The self-refitting slice planner had zero wall-clock
failures; measured slice times ran 33–50 min against a 2 h limit.

**Arm sizing (from `plan_arms.py`, before enumeration):**

| arm | pool | order | hubs cached | new | depth mix (0/1/2/3) |
|---|---|---|---|---|---|
| `incumbent` | top-1000 candidates | flow ↓ | 200 | 0 | 2/104/84/10 |
| `flow_top` | all 20,874 | flow ↓ | 91 | 109 | 11/150/37/2 |
| `flow_bottom` | all | flow ↑ | 0 | 200 | 7/5/34/154 |
| `random` | all | uniform (seed 0) | 3 | 197 | 2/19/51/128 |
| `cand_order_fixedset` | incumbent's 200 | best-candidate reward | 200 | 0 | 2/104/84/10 |
| `cand_order` | all | best-candidate reward | 110 | 90 | 0/59/89/52 |

598 of the 1,000 arm-hub slots are new work; the rest is shared cache.

best-candidate reproduces entry `033`'s 929 reactions / 3.097 exactly and is byte-identical across
all six arms (it never reads hub data) — the built-in consistency check, asserted by
`compare_hub_order.py` on every run.
