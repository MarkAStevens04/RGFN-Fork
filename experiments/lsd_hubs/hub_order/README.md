# `hub_order/` — does sorting hubs by flow actually buy anything? (Logs/053)

Hub-batching walks a ranked hub list top-to-bottom, so **the order of `hubs.csv` *is* the strategy**.
Every LSD-Flow result so far uses one ordering (highest flow first, over a reward-pre-filtered pool).
This ablation swaps that ordering out for controls that ignore the flow signal, holding *everything
else* fixed — same trained model, same sampled candidate pool, same cost model, same acquisition
config, same 200-hub budget — so any difference is attributable to hub selection alone.

## The arms

The incumbent recipe is a **two-stage filter**, not a pure flow sort: `pick_hubs.py` keeps the
parents of the top-1000 candidates by reward (575 hubs) and *then* ranks those by flow. The controls
below relax each stage in turn. All draw on the same 20,874 hubs observed across 30,000 sampled
trajectories; all take 200 hubs.

| arm | pool | order | asks |
|---|---|---|---|
| `incumbent` | parents of top-1000 candidates | flow ↓ | the published strategy (Logs/029/031/052) |
| `flow_top` | **all 20,874 hubs** | flow ↓ | does the reward pre-filter matter, or can flow select on its own? |
| `flow_bottom` | all hubs | flow ↑ | the reverse control — is flow order *directional*? |
| `random` | all hubs | uniform (seed 0) | the no-signal floor |
| `cand_order_fixedset` | **incumbent's exact 200** | best-candidate reward | ordering isolated from selection (free — no new enumeration) |
| `cand_order` | all hubs | best-candidate reward | "a great molecule must sit on a great hub" — does it? |

`cand_order*` are the sharpest controls for the paper: they say a hub's value is *not* read off its
best child's reward, so the flow field is carrying structural information the reward ranking is not.

**What flow is here.** `F_hat(h;x) = logR + logP_B − logP_F(move) − logP_F(stop)`, aggregated per hub
as the **max** over its observed children. Worth stating plainly: `log R = 8·reward` (β=8) dominates,
and the policy term spans only ~9 nats p5–p95 ≈ 1.1 reward units, so flow order and reward order are
correlated (Spearman 0.76 over all hubs) but far from identical — which is exactly what `cand_order`
isolates.

## Pipeline

```bash
OUT_ROOT=$SCRATCH/rgfn_runs/lsdflow/hub_order

# 1. build every arm's hub set + slice the un-enumerated remainder into debug-sized jobs
python experiments/lsd_hubs/hub_order/plan_arms.py --out-root $OUT_ROOT

# 2. enumerate the new hubs, one debug job at a time (see "Why a driver" below). ~12h.
nohup bash experiments/lsd_hubs/hub_order/chain.sh > /dev/null 2>&1 &
tail -f $OUT_ROOT/chain.log

# 3. assemble each arm's 200 hubs in ITS OWN walk order (chain.sh does this at the end too)
python experiments/lsd_hubs/hub_order/merge_enum.py --out-root $OUT_ROOT

# 4. campaign + diversity sweep per arm, then the overlay  (pure CPU, ~5 min/arm)
source ~/bin/rgfn-smoke-env.sh
bash experiments/lsd_hubs/hub_order/run_arms.sh
```

**Why a driver and not `--dependency`.** The `debug` QoS is `MaxJobsPU=1` **and**
`MaxSubmitJobsPU=1`: a second job cannot even be *queued*, so a dependency chain is rejected at
submit time. `chain.sh` submits one slice, waits for it to leave the queue, then submits the next.

**Why slices.** The worker writes its outputs only after the whole hub loop, so a slice that
overruns the 2 h wall clock loses everything. `plan_arms.py` packs slices to ~half the limit using
per-depth times measured from previous runs, and `chain.sh` re-plans before every submission so the
cost model refits on real data as it goes (the reverse/random arms are dominated by depth-3 hubs the
incumbent barely sampled). A slice that dies halves the budget and re-plans rather than retrying the
same job.

**Why enumeration is shared.** It is deterministic given (checkpoint, frozen library,
`--enum-max-children`), so each hub is enumerated once and reused by every arm that selects it —
`flow_top` inherits 91 hubs from the incumbent, `cand_order` 109, `cand_order_fixedset` all 200.
Only 598 of the 1,000 arm-hub slots are new work.

## Substrate (all arms, verified end to end)

| stage | artifact | check |
|---|---|---|
| train | `scent_seh/2026-07-10_17-28-06` | 5,000 iters (`paths.csv` 320,001); `guidance_models.pt` present (P_B trained, entry 024); **the only sEH run with `smiles_to_route`** → exact nested cost |
| sample | `lsdflow/scent_seh_70189` | 30,000 trajectories → 29,997 records, 20,874 hubs, logZ 74.3259, guidance loaded `unmatched=[]` |
| enumerate | `lsdflow/campaign_enum_seh_70363` | 200 hubs / 828,448 children, same checkpoint + logZ, `frozen=true` (+1600 frags) |
| cost | `additional_fragments/fragments_4000.json` | 1600 promoted fragments with routes |

Same triple as the Logs/052 τ×similarity surface, so these numbers sit alongside it directly.

## A subtlety: pre-select-K reads the whole pool

The budget is **300 modes**, not hubs — the 200 is an enumeration pool, and at cutoff 0.5 the
completing arms stop having used only 17–52% of it. So pool size *should* be a no-op for them, which
is what makes the 600-hub `flow_bottom` arm a fair comparison rather than a privileged one.

It is not quite a no-op, because `rank_fragments` selects the pre-select-K stock by scanning **every
enumerated hub, including ones the walk never reaches**. Measured on `flow_top`: 357 reactions with a
200-hub pool vs 352 with 100 (1.4%), saturating below 100 hubs. Consequences: run the
`--prebuild-k 0` control when the pool sizes differ (it removes the channel), and note the accounting
mismatch — pre-select uses fan-out from hubs the compute-time axis never charges. See Logs/053's
Next Experiments; the saturation may also be exploitable (a cheap "scout" enumeration to pick stock).

## Operating point

τ (hit bar) 7.0, diversity cutoff 0.5, **budget = 300 modes** (Case 2: reactions needed to get
there), `--child-policy free_frag --prebuild-k 20 --rank-by build_score` — the Logs/052 hero config.
Pre-select fragments are ranked over **each arm's own hubs**, so a weak arm gets weak stock and still
pays the 20 upfront reactions; that is the honest treatment, not a handicap.

The headline sweep walks the diversity cutoff 0.30→0.90 at that mode budget. Arms that exhaust their
200 hubs before reaching 300 modes are reported as **pool-limited** (a gap in the line, listed in the
table) — running out of library is a real result about the ordering, but it is not the same
measurement as completing the budget, and it is never plotted as a win.

`best_candidate` is recomputed per arm and must be identical everywhere (it never reads hub data);
`compare_hub_order.py` asserts that and flags any drift.

## Result (SCENT/sEH, τ=7, cutoff 0.5, 300 modes — Logs/053)

| arm | rxn/mode | reactions | compute (s) | modes | |
|---|---|---|---|---|---|
| `flow_top` | **1.190** | 357 | 5,648 | 300 | Pareto frontier |
| `incumbent` | 1.217 | 365 | 5,685 | 300 | dominated by `flow_top` |
| `cand_order_fixedset` | 1.263 | 379 | 3,365 | 300 | Pareto frontier |
| `cand_order` | 1.297 | 389 | **2,481** | 300 | Pareto frontier |
| `random` | 1.820 | 546 | 4,332 | 300 | **dominated on both axes** |
| `flow_bottom` | 2.050 | 572 | 6,580 | **279** ✗ | **pool-exhausted** |
| best-candidate | 3.097 | 929 | — | 300 | reference |

**Flow picks the right neighbourhood, not the right rank.** Reversing the sort breaks the method
(can't build the library at all); randomising costs 1.53×; but the three good-hub arms sit within 9%
of each other. On the two cost axes together, every frontier point is a flow- or reward-informed
ordering and both no-signal controls are strictly dominated — that dominance is the strongest form
of the claim, since it needs no weighting between bench and compute cost.

**Why best-candidate order does so well** (`hub_rank_overlap.py`): it never reads flow, yet its 200
hubs sit at median rank **546 of 20,874** (top 2.6%, 63% inside the top 1,000) versus 11,043 for the
random arm. Sampling concentrates trajectories on high-flow hubs, so "parents of the best molecules"
is a flow proxy. Correlational, not causal — see Logs/053's Next Experiments for the clean test.

All arms converge to ~1.04× by cutoff 0.90, where loose distinctness stops making hub quality bind.
Caveat: n=1 model, 1 target, 1 random draw.

**The flow field moves during training** (`flow_drift_over_training.py`): only 7.3% of the hubs
visited in the first 250 training iterations still appear in the post-training sample, rising to
24.2% for the last 250, with median final flow-rank improving 6,192 → 3,438 and a step at iteration
~1000 (the first dynamic-library promotion). This is **not** a confound here — our pool and flow
terms all come from the final checkpoint — but it means any signal taken from *training-time*
behaviour is stale, which is exactly how SCENT's dynamic library selects fragments.

## Layout

```
hub_order/
  plan_arms.py   merge_enum.py   compare_hub_order.py     # python: plan / assemble / compare
  hub_rank_overlap.py                                     # where each arm sits in the flow ranking
  flow_drift_over_training.py                             # how far the flow field moves in training
  chain.sh       submit_slice.sh   run_arms.sh            # slurm driver / one slice / cpu analysis
  results/hubord_<arm>/    summary.json sweep_summary.json *.png
  results/comparison/      cost_vs_cutoff.png     <- headline: reactions vs diversity cutoff
                           cost_pareto.png        <- the two cost axes + dominance
                           hub_rank_distribution.png/.csv
                           flow_drift.png/.csv
                           compute_vs_cutoff.png  summary.csv
```

Heavy artifacts stay on `$SCRATCH` (`lsdflow/hub_order/`): `arms/` (hub sets + slice files),
`enum/` (per-slice worker output), `merged/` (per-arm assembled enumeration), `chain.log`.

## Re-running

The enumeration is done and cached, so **everything above re-derives on CPU in ~5 min/arm** — you
only need `chain.sh` again if you add an arm whose hubs aren't in `$SCRATCH/.../hub_order/enum/`.
To add one: append it to `ARMS` in `plan_arms.py`, re-run `plan_arms.py` (it will report how much new
GPU work it needs — possibly none), then `merge_enum.py` → `run_arms.sh`.
