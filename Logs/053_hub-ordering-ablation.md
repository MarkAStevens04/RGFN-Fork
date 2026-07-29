# SCENT / sEH — does sorting hubs by flow actually buy anything?

**Date:** 2026-07-29, ~3pm

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

[TODO — fill in at END. Expected shape: whether flow order beats random/reverse, whether the reward
pre-filter is load-bearing, and whether the best-candidate ordering matches flow (the key control
for "a great molecule implies a great hub").]

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
- If an arm turns out pool-limited (runs out of its 200 scaffolds before reaching 300 molecules),
  re-run that arm with a larger scaffold budget to separate "this ordering is bad" from "this
  ordering needed more scaffolds."

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
- `./experiments/lsd_hubs/hub_order/results/comparison/` — `cost_vs_cutoff.png`,
  `compute_vs_cutoff.png`, `summary.csv`.

**Job Logs**
- `/scratch/markymoo/rgfn_runs/hubord_enum-<jobid>.{out,err}` — one pair per slice.

## Relevant Versions

Branch `Hub-Analysis`, worktree branch `worktree-hub-order-ablation` off `7d99b27`.
[TODO — add commit hash after pushing]

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

## Results

[TODO — fill in at END]

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

**Operating point (τ=7.0, cutoff 0.5, 300 modes, `free_frag` + `prebuild_k=20`)** — the two arms that
needed no new enumeration, run first as an end-to-end validation:

| arm | rxn/mode | modes | reactions | hubs walked | reward-gen calls | measured compute (s) |
|---|---|---|---|---|---|---|
| `incumbent` | 1.217 | 300 | 365 | 43 | 241,158 | 5,685 |
| `cand_order_fixedset` | 1.263 | 300 | 379 | 41 | 145,490 | 3,365 |
| best-candidate (ref) | 3.097 | 300 | 929 | 254 | 0 | — |

best-candidate reproduces entry `033`'s 929 reactions / 3.097 exactly, and is identical across arms
(it never reads hub data) — the built-in consistency check.
