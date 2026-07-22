# SCENT sEH — from-scratch SPARROW headline + compute frontier (the neutral referee)
**Date:** 2026-07-20, ~4:30pm

## Question

When a neutral from-scratch route planner (AiZynth → SPARROW) prices the reaction-GFN's library —
ignoring the routes it built molecules with — does hub-batching's reactions-per-mode advantage over
plain best-candidate survive, and where does the compute actually go?

## Context & Summary

The benchmark's headline must be *fair*: every library — reaction-GFN or not — is priced by the SAME
independent referee (recover a route for each molecule from scratch with AiZynth over standard
USPTO/ZINC, then let SPARROW's MILP find the minimum shared-reaction set). No method gets credit for
routes it happened to get for free. Entry 039 measured the reaction-GFN's *internal* compute
(enumeration, reward-gen, flow-extraction); this entry does the other half — it prices SCENT's sEH
library through the neutral referee and, using the timed route cache (job 70976), adds the real
per-molecule AiZynth **route-finding** time to the compute breakdown. We compare the two selection
strategies a chemist could use on the same pool: **hub-batching** (LSD-Flow; here the free-frag +
pre-select-K variant) vs **best-candidate** (sort by reward, take the best that passes the diversity
filter).

## Answer

Hub-batching keeps its edge even when SCENT's own molecules are re-routed from scratch: **2.74
reactions per mode vs best-candidate's 3.66**, with *fewer* total reactions (502 vs 564) and *more*
routable modes (183 vs 154). And the compute story is stark — **finding the routes is the whole cost**:
AiZynth route search takes ~12,000–13,000 seconds, while the reaction-GFN's own enumeration is ~2,000 s
and the SPARROW MILP that actually assembles the shared-reaction library is **0.06 seconds**. So for a
lab pricing a library with a from-scratch planner, the bottleneck is retrosynthesis search, not the
set-cover optimization — and a reaction-grounded generator that already knows its routes sidesteps
almost all of it.

## Relevance to our Publication

This is the backbone of the headline figure. The whole "reaction MDP is necessary for library
economics" argument only lands if the reaction-GFN's advantage holds under a *neutral* pricer (a
reviewer's first objection is "you scored yourself with your own routes"). It does. The compute
breakdown also motivates the paper's efficiency framing: the reaction-GFN gets for free (routes by
construction) exactly the step that dominates a from-scratch planner's compute.

## Next Experiments

**Refining for publication**
- **Route coverage caveat:** from-scratch AiZynth solves only **48.7%** of the accepted-mode pool, so
  neither strategy assembles 300 *routable* modes → the fixed-mode ("reactions to reach 300 modes")
  plot is empty on the from-scratch axis. Either report the fixed-*reaction*-budget plot (populated) as
  the headline and lower the mode target, or raise AiZynth's per-molecule time/expansion to lift
  coverage. Decide before finalizing the figure.
- **Acquisition-function validation (deferred, on purpose):** this run's hub-batching line is the
  free-frag + pre-select-K variant. Going forward, **free-frag is the single "hero" acquisition** for
  smokes; a dedicated later experiment will compare the variants (naive hub-batching / free-frag /
  pre-select-K) head-to-head so the headline cites one, justified, hero.

**Next steps in project**
- **T3.2:** put the S3-GFN pool on these exact axes (best-candidate only; from-scratch SPARROW) — the
  "MDP-necessary" figure. Route recovery for the S3-GFN pool is prepped (`submit_s3gfn_routes.sh`).
- **Native-route cross-check (CHECK 1) + T1.5 gate:** the fixed SCENT enum (job 71010) will supply the
  by-construction routes; then native-route SPARROW should reproduce count-once within tolerance.

---

# Re-creation

### Relevant Files

Root: repository root unless noted. `/…` = absolute (on `$SCRATCH`).

**Scripts**
- `./experiments/lsd_hubs/campaign/sweep_campaign.py` — the frontier driver. `--evaluator sparrow
  --route-source from_scratch` prices each ordered snapshot via AiZynth→SPARROW; the T2.2 compute
  frontier (`sparrow_compute_frontier`) reads per-molecule route-finding time from the timed
  `RouteCache`. Added `--out-dir` this run so it can write to `$SCRATCH` from a Balam compute node
  (`$HOME` is read-only there).
- `./experiments/lsd_hubs/campaign/submit_sparrow_headline_timed.sh` — the Balam job (71009) that ran it.
- `./validation/lsdflow/eval/{sparrow.py,route_recovery.py,network.py}` — SparrowEvaluator + timed
  RouteCache + route→SPARROW-tree network builder.
- `./validation/lsdflow/adapters/workers/sparrow_worker.py` — the CBC MILP (runs in the `sparrow` env).

**Datasets** (`/scratch/markymoo/rgfn_runs/…`)
- `lsdflow_sparrow/scent_seh/routecache_zinc_uspto_timed.json` — from-scratch AiZynth routes for the
  4,749 accepted-mode molecules, **2,314 solved (48.7%)**, each with `search_time` (job 70976).
- `lsdflow/campaign_enum_seh_70363/enum_children.json` — the 200-hub enumeration (hub-batching pool).
- `lsdflow/campaign_enum_seh_70363/enum_timings.json` — merged per-hub enum timings (6 slices; Logs/039).
- `lsdflow/scent_seh_70189/` — the 30k-trajectory sample (best-candidate pool).

**Results** — `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/scent_seh_sparrow_headline/`
(sync to `experiments/lsd_hubs/campaign/results/` from a login node): `compute_frontier.{csv,png}`,
`pareto.{csv,png}`, `budget_efficiency.{csv,png}`, `fixed_modes.{csv,png}`, `sweep_summary.json`.

### Relevant Versions

Uncommitted (branch `Hub-Analysis`). Files to commit: the `sweep_campaign.py --out-dir` change +
`submit_sparrow_headline_timed.sh` + this log. Results live on `$SCRATCH` (sync back before commit).
`[TODO — add commit hash after pushing]`

### Relevant Resources

**Sources** — SPARROW (route-aware library MILP); AiZynthFinder (retrosynthesis, USPTO/ZINC).
**Packages** — `sparrow` env (PuLP/CBC) via `sparrow_worker.py`; `aizynth` env for route recovery;
`rgfn` env runs the driver (imports `glue`→dgl).

### Method

1. Route recovery (job 70957/70976): AiZynth over the accepted-mode union → `routecache_zinc_uspto*.json`
   (timed variant records `search_time`). 48.7% solved.
2. Frontier (job 71009): `sweep_campaign.py --evaluator sparrow --route-source from_scratch
   --snapshot-schedule geometric --child-policy free_frag --prebuild-k 100 --n-hubs 200
   --sparrow-cache …_timed.json --reward-threshold 7.0 --out-dir $SCRATCH/…` over cutoffs 0.30–0.90.
   CPU-only (cached routes → no AiZynth calls; SPARROW CBC per snapshot). Elapsed 7m37s.

### Results

**Compute frontier at the baseline cutoff (τ=0.5)** — seconds per stage, from `compute_frontier.csv`:

| strategy | route-finding (AiZynth) | enumeration | reward-gen | flow-extract | SPARROW MILP | total | reactions | priced modes | rxn/mode |
|---|---|---|---|---|---|---|---|---|---|
| hub-batching (free-frag+K100) | 12,153 | 1,578 | 102 | 296 | 0.06 | 14,131 | 502 | 183 | **2.74** |
| best-candidate | 13,395 | 0 | 0 | 0 | 0.07 | 13,395 | 564 | 154 | **3.66** |

**Fixed-reaction-budget (modes at R=100 reactions), cutoffs 0.30→0.90:**

| | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|
| hub-batching | 23 | 21 | 27 | 26 | 23 | 24 | 24 |
| best-candidate | 16 | 18 | 26 | 27 | 23 | 22 | 22 |

**Fixed-mode target (reactions to 300 modes):** empty (None) for both — only 48.7% of molecules route
from scratch, so 300 *routable* modes are unreachable from this pool. Use the fixed-reaction-budget
plot as the from-scratch headline, or raise AiZynth coverage.
