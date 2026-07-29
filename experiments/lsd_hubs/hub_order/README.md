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

## Layout

```
hub_order/
  plan_arms.py   merge_enum.py   compare_hub_order.py     # python: plan / assemble / compare
  chain.sh       submit_slice.sh   run_arms.sh            # slurm driver / one slice / cpu analysis
  results/hubord_<arm>/         summary.json sweep_summary.json *.png
  results/comparison/           cost_vs_cutoff.png compute_vs_cutoff.png summary.csv
```

Heavy artifacts stay on `$SCRATCH` (`lsdflow/hub_order/`): `arms/` (hub sets + slice files),
`enum/` (per-slice worker output), `merged/` (per-arm assembled enumeration), `chain.log`.
