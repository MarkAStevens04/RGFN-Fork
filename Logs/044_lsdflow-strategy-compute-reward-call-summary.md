# LSD-Flow selection — consolidated compute & reward-call accounting (best-candidate / free-frag / pre-select-K)
**Date:** 2026-07-22, ~afternoon

## Question

For the three ways we assemble a diverse hit library — keep the best molecules the
generator already sampled, or build one shared scaffold and diversify it (with, or without,
a handful of pre-stocked building blocks) — how much computer time does each take, how many
molecule-scoring calls does each make, and which step actually eats the time?

## Context & Summary

The LSD-Flow campaign compares three selection strategies on the same candidate pool:
**best-candidate** (sort by reward, keep the top molecules that pass the diversity filter —
the baseline a chemist would use today), **free-frag** (our "hero" hub-batching variant:
build a shared scaffold once, then keep only children whose last reaction attaches an
already-available fragment), and **pre-select-K** (free-frag after first paying to stock the
top-K most reusable building blocks — K is a dial). We have already measured the two axes a
reviewer cares about, but in separate places: entry `037` reported the **scoring-call** count
per strategy (0 for best-candidate, 315,539 for free-frag, dropping to ~53k as K rises), and
entry `039` reported the **measured, per-component wall-clock** (best-candidate ~1 s;
free-frag ~7,472 s; K=200 ~1,219 s) and found enumeration — the RDKit work of *building* each
scaffold's children — dominates. This entry doesn't run anything new: it consolidates those
committed numbers into a single figure that puts all three strategies on both axes at once,
plus a small reusable driver, so the "where does the time go" answer is one picture.

## Answer

Best-candidate is essentially free (~1 s, **zero** scoring calls) because it only re-uses
scores the generator already produced and pays for nothing but the diversity filter. The
hub-batching family pays real compute, and — for our fast surrogate sEH reward — that compute
is dominated by **enumeration (~79%)**, i.e. *building* the scaffold's children, not scoring
them (reward-gen is only ~5%). Pre-select-K is the knob that buys compute back: stocking a
few reusable blocks lets each hub yield more free children, so we walk fewer hubs and both the
scoring-call count and the wall-clock fall in step (free-frag → K=200 is ~6× less compute and
~6× fewer calls). The load-bearing caveat, already flagged in `039`: this component split is
reward-dependent — with a real docking oracle (~1 s/call) the reward-gen term would dominate
instead, so the **scoring-call count is the reward-agnostic axis** to report in the AL loop.

## Relevance to our Publication

This is the cost side of the LSD-Flow "build once, diversify late" story (targeted at
*Digital Discovery* / *J. Cheminformatics*). A reviewer's first question about any efficiency
claim is "what does it cost you in compute versus just keeping your best samples?" — and for
the active-learning framing, "how many oracle calls?" This single consolidated figure answers
both for all three strategies at the benchmark's baseline operating point, and makes explicit
that the reward-gen call count (not the surrogate-run wall-clock) is the number that transfers
to an expensive oracle.

## Next Experiments

**Refining for publication**
- **Docking-target timing:** re-run the same per-component instrumentation against a real
  docking oracle (6TD3 / ClpP) so the figure's twin — where reward-gen dominates — is shown
  side by side; the instrumentation already supports it (no code change, per `039`).
- **Cutoff annotation:** the consolidated figure fixes the diversity cutoff at τ=0.5; a small
  inset or companion using the already-committed `compute_time_by_cutoff.csv` would show how
  strict diversity (τ=0.30 → ~13,485 s) explodes hub-batching's compute.

**Next steps in project**
- Fold this consolidated figure into the LSD-Flow benchmark's headline cost panel alongside
  the from-scratch referee view (entry `041`: external AiZynth route-finding dominates at
  ~12–13k s, SPARROW MILP is 0.06 s) so the paper shows both the *internal* generator compute
  and the *external* pricing compute.

---

# Re-creation

## Relevant Files

Root: `./` (repo root).

**Scripts (ours — new this entry):**
- `./experiments/lsd_hubs/campaign/strategy_compute_summary.py` — reads the committed
  `preselect.csv` and emits the consolidated two-panel figure (per-component compute, log-y,
  stacked + reward-gen calls) plus a `strategy_compute_summary.{csv,json}` roll-up. Pure
  re-plot of already-measured numbers; runs no model and touches no cluster.

**Datasets (input — committed):**
- `./experiments/lsd_hubs/campaign/results/scent_seh_1kx200_preselect/preselect.csv` — the
  measured source table (SCENT sEH, cutoff 0.5, reward bar 7.0, 300-mode target): per-strategy
  reactions, `reward_gen_calls`, and the `ct_*` per-component wall-clock columns produced by
  the timed enumeration re-run in entry `039`.

**Results (ours — new this entry):**
- `./experiments/lsd_hubs/campaign/results/scent_seh_strategy_summary/strategy_compute_summary.png`
  — the consolidated figure.
- `.../strategy_compute_summary.{csv,json}` — the five-row roll-up (best-candidate, free-frag
  K=0, pre-select K=50/100/200) with total compute, enumeration fraction, and call count.

**Prior source material (numbers consolidated here, not re-derived):**
- Entry `037` — the per-strategy reward-gen-call Pareto and pre-select-K sweep.
- Entry `039` — the measured per-component compute breakdown (timed enumeration jobs 70696–70702).
- Entry `041` — the external-referee compute view (from-scratch AiZynth→SPARROW).

## Relevant Versions

Branch `Hub-Analysis`. New/edited files **not yet committed**: the driver
`experiments/lsd_hubs/campaign/strategy_compute_summary.py`, the result dir
`experiments/lsd_hubs/campaign/results/scent_seh_strategy_summary/`, and this log.
`[TODO — add commit hash after pushing.]`

## Relevant Resources

**Sources** — entries `037` (reward-call axis), `039` (measured compute axis), `041`
(external-referee compute). `[gainski2025scent]` (SCENT / Dynamic Library),
`[bengio2021gflownet]` (modes / oracle efficiency).

**Packages** — pandas + matplotlib (any of the `rgfn` / `scent` / `aizynth` conda envs has
both; the base env does not). No GPU, no dgl import needed.

## Method

1. Confirmed the source numbers are committed and self-consistent: `preselect.csv` (all
   strategies at one operating point) cross-checks against `batch_stats.csv` and the entry-`039`
   `compute_time.csv` (naive hub-batching / best-candidate) byte-for-byte on the shared rows.
2. Wrote `strategy_compute_summary.py` to select the five reported strategies from
   `preselect.csv`, fold `setup + hub_pick` into one one-off "setup" component, and render:
   ```
   python experiments/lsd_hubs/campaign/strategy_compute_summary.py \
     --preselect experiments/lsd_hubs/campaign/results/scent_seh_1kx200_preselect/preselect.csv \
     --out-dir   experiments/lsd_hubs/campaign/results/scent_seh_strategy_summary
   ```

## Results

**SCENT sEH, diversity cutoff τ=0.5, reward bar 7.0, 300-mode library, surrogate proxy reward.**
Compute is measured (entry `039`); calls are exact (entry `037`).

| strategy | reward-gen calls | total compute (s) | enumeration | reward-gen | flow-extract | setup | mode-select | reactions/mode |
|---|---|---|---|---|---|---|---|---|
| best-candidate | 0 | **1.1** | 0 | 0 | 0 | 0 | 1.1 (100%) | 3.10 |
| free-frag (K=0) | 315,539 | **7,472** | 5,941 (79%) | 366 (5%) | 1,136 (15%) | 26.8 | 1.7 | 1.22 |
| pre-select K=50 | 158,660 | **3,732** | 2,951 (79%) | 186 (5%) | 567 (15%) | 26.8 | 0.9 | 1.26 |
| pre-select K=100 | 85,165 | **2,004** | 1,579 (79%) | 102 (5%) | 296 (15%) | 26.8 | 1.0 | 1.41 |
| pre-select K=200 | 52,865 | **1,219** | 945 (78%) | 63 (5%) | 183 (15%) | 26.8 | 1.0 | 1.85 |
| *naive hub-batching (ref, entry 039)* | 7,086 | 170 | 109 (64%) | 9 (5%) | 23 (14%) | 26.8 | 1.5 | 2.72 |

**Reading:**
- **Best-candidate is ~free** — no scoring calls, ~1 s (the O(n²) diversity filter is its only
  marginal cost). It "spends" bench reactions (3.10/mode) instead of compute.
- **Hub-batching's compute is enumeration, not scoring** — building the children is ~12× the
  cost of scoring them under the surrogate reward (reward-gen is a flat ~5% across the family).
- **Pre-select-K tracks both axes together** — K=0→200 cuts calls 6.0× (315,539→52,865) and
  compute 6.1× (7,472→1,219 s), at the cost of more bench reactions per mode (1.22→1.85).
- **Reward-agnostic caveat** — the ~5% reward-gen share is a property of the millisecond sEH
  surrogate; for a ~1 s/call docking oracle the reward-gen term (≈ calls × per-call cost) would
  dominate, which is why the call-count panel is the number that transfers to the AL loop.

**External-referee cross-reference (entry `041`, from-scratch AiZynth→SPARROW pricing, τ=0.5):**
the *internal* generator compute above is separate from the *external* pricing compute, where
AiZynth route search dominates (~12,153 s for hub-batching, ~13,395 s for best-candidate) and
the SPARROW set-cover MILP is 0.06 s.
