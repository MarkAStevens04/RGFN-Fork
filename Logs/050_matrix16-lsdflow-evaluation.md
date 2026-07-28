# Four generators × four targets — LSD-Flow hub-batching across the publication matrix, and a severe test of the flow field it rests on

**Date:** 2026-07-27, ~2pm

## Question

Does building a shared scaffold once and diversifying it ("hub-batching") actually save synthesis
effort for *every* molecule generator and target we benchmark — and are the internal signals we use to
pick those scaffolds measuring what we think they are?

## Context & Summary

Everything we know about hub-batching so far comes from **one** model on **one** target. Entries
`029`–`039` built the comparison — hub-batching versus the naive "just synthesize your best molecules"
strategy — and ran it on SCENT for the sEH target, where building shared scaffolds turned out to cost
about 1.1× fewer bench reactions per useful molecule under a fair accounting (entry `033`), improving to
roughly 2.5× cheaper once we let it prefer scaffolds whose building blocks were already in stock
(entry `037`). Entry `030` then trained the full **4 generators × 4 targets** matrix at publication
scale, which finally makes a real question askable: is that saving a property of hub-batching, or was it
a quirk of one model and one target?

This entry runs the LSD-Flow pipeline across that matrix. For each trained model we sample 30,000
molecule-building trajectories, read off the "hub" scaffolds the model visits most productively,
exhaustively enumerate **every** molecule reachable from those scaffolds in one more step, and then let
the two strategies compete to assemble a diverse, high-scoring library. Because the earlier work leaned
on two internal quantities — a per-scaffold flow estimate `F(h)` and an uncertainty `U(h)` derived from
it — we also used the exhaustive enumeration to do something we had never been able to do before:
**check those estimates against a value reconstructed from the model's own conservation laws**, rather
than only against each other.

## Answer

**Hub-batching wins on every reaction-grounded cell we completed, and the margin is robust** — 1.2× to
3.2× fewer reactions per useful molecule across two generators and two targets, and the advantage grows
as the diversity/quality bar loosens while the naive strategy barely moves. The pipeline itself passed a
strong internal audit: summing the model's own forward probabilities over our enumerated molecules
recovers 0.999 of the available probability, which independently confirms both that the enumeration is
exhaustive and that we are reading the model's policy correctly.

**But two of our internal signals do not mean what we assumed.** The uncertainty `U(h)` is dominated by
how sharply the model concentrates its choices, not by any genuine uncertainty about molecule quality —
and the cheap version we would compute during an active-learning run is 40–55× smaller than the true
value and undefined for most scaffolds, which undercuts the uncertainty-driven scaffold-selection plan
before it is built. Separately, **both** trained models we tested **violate their own flow-conservation law at these
scaffolds by a wide margin**, terminating there far more often than their reward would justify — and the
violation is *larger* in the model whose backward policy is properly trained, so it is a property of the
trained models rather than of one model's approximations.
Encouragingly, the estimate we actually ship for picking scaffolds is nearly unbiased despite all this —
though it is too noisy to *rank* scaffolds reliably, which explains an earlier puzzling near-tie
(entry `025`) between flow-based selection and a trivial control.

## Relevance to our Publication

The headline claim we intend to make — that a reaction-grounded generator can assemble a diverse library
far more cheaply than assembling its best molecules independently — has until now rested on a single
model and a single target. Reviewers at **Digital Discovery** or **JCIM** would immediately ask whether
the effect generalizes; this entry answers with two generators × two targets, plus a threshold sweep
showing the result is not an artifact of where we set the quality bar. The severe tests matter for a
different reason: they are the kind of self-criticism that makes a methods paper credible, and one of
them (the `U(h)` finding) redirects the planned active-learning section away from a signal that would not
have worked. It also lets us state the flow-recovery validation as a positive result rather than an
assumption.

## Next Experiments

**Refining for publication**
- Finish the two RGFN cells (jobs `71766`/`71767`) so the reaction-grounded comparison covers all three
  synthesizable generators rather than two.
- Re-run the remaining FragGFN cell against the corrected fragment cap. **DRD2 is done** (see the Update
  below: 3.99× at cap-6, versus 2.01× at the superseded cap-9 that entry `046` showed collapses DRD2).
  The sEH cap-6 model does not exist yet, so that cell is correctly marked not-ready until its training
  lands.
- Complete the threshold sweep for the one cell that is missing intermediate bars.
- Additional random seeds, so the per-cell margins carry error bars.

**Next steps in project**
- Settle what the scaffold-uncertainty signal should actually weight before building the
  uncertainty-driven acquisition step — the current definition is measuring the wrong thing.
- Explain the termination-behaviour violation now that it is confirmed on two generators — it is the
  largest single inconsistency we have measured in these trained models, and it sits in the one channel
  our scaffold estimate ignores.
- Reconstruct the scaffold flow from the *other* end of the model's balance equation (anchored on the
  model's learned total-flow scalar rather than on measured molecule quality). The two reconstructions
  must agree if the model is internally consistent, so their disagreement is a direct measure of training
  quality. One caveat found here: one model's learned total-flow scalar is smaller than the reward of
  individual molecules it generates, so that anchor is unreliable for that model.
- Extend to the two docking targets, where scoring each enumerated molecule requires the GPU docking
  oracle rather than a fast surrogate.

---

# Re-creation

## Relevant Files

Root: `./` (repo root `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`).

**Scripts — orchestration (`experiments/lsd_hubs/matrix16/`):**
- `manifest.csv` — the static per-cell spec (generator, target, seed, gin/yaml config, checkpoint,
  SCENT guidance sidecar) for all 16 cells. Human-editable single source of truth.
- `targets.py` — per-target science: the reward gate's value **and direction** (sEH `>7.0`, DRD2 `>0.5`,
  6TD3 `<−2.0`, ClpP `<−8.0`) plus each target's enumeration cost class (`surrogate` vs `docking`).
  Exists because the campaign drivers previously hard-coded the sEH-only `7.0`.
- `manifest.py` — the loader that joins the two above and adds **live** filesystem status (checkpoint
  present? sidecar present? how many candidates has training emitted?). `--emit` gives a
  shell-sourceable view for the submit script; `--list` drives the launcher.
- `submit_cell.sh` — one SLURM cell: `sample → pick_hubs → enumerate`, dispatching to the right conda
  env per generator. Knobs `N_TRAJ`/`N_HUBS`/`TOPK`/`ENUM_MAX`/`STAGE`/`TIME`.
- `launch_surrogates.sh` — fans `submit_cell.sh` across all *runnable* surrogate cells; docking cells are
  excluded automatically.
- `run_cell_campaign.sh` — the CPU count-once readout for one cell. Defaults the within-hub child policy
  **per generator** (SCENT → `free_frag` + pre-select-K=20; baselines → naive `reward`) so re-running
  reproduces committed numbers instead of overwriting them with another policy's.
- `gate_sweep.sh` — re-scores **both** strategies at each target's threshold variants over the cached
  enumeration (pure CPU, no re-enumeration).
- `link_worktree_data.sh` — symlinks the gitignored `data/` + `external/` payloads that `git worktree`
  omits. Needed only when working from a worktree; unnecessary in the main checkout.
- `MERGE_NOTES.md` — the branch-reconciliation handoff (kept because it also documents run state,
  provenance caveats, and open threads).

**Scripts — the severe tests (`experiments/lsd_hubs/matrix16/`):**
- `hub_rank_sensitivity.py` — reproduces `pick_hubs`' ranking and re-runs it with the backward-policy term
  ablated, and against a pure reward ranking; reports top-200 overlap and rank correlation. Answers
  whether RxnFlow's *heuristic* backward policy is driving scaffold choice.
- `flow_consistency.py` — reconstructs `F_true(h)` from flow conservation over all enumerated children,
  reports the normalization residual `Σ_x P_F(x|h)`, decomposes `U(h)` into per-term variances, and
  compares sampled vs enumerated `U(h)`.
- `hub_flow_estimators.py` — scores four candidate `F(h)` estimators against `F_true` on **bias and rank
  correlation**, plus the dimensionless flow-conservation test.

**Scripts — per-env workers (`validation/lsdflow/adapters/workers/`):**
- `_artifacts.py` — shared stdlib writer so all four generators emit a byte-compatible artifact set
  (`records.csv`, `enum_children.json`, `U(h)`), and `compositions_from_records` which charges each
  molecule its flat build depth for generators with no promoted-fragment library.
- `rgfn_worker.py` — RGFN (in-process). Enumerate loops one hub at a time with progress logging and a
  10-hub partial flush, because RGFN's enumeration is slow enough to hit walltime.
- `fraggfn_worker.py` — FragGFN (own env). Persists `hub_graphs.pkl` during sampling because
  reconstructing a hub graph from SMILES mis-decomposes ~6% of hubs.
- `rxnflow_worker.py` — RxnFlow (own env). Handles both terminal kinds (explicit stop vs reaction-cap
  truncation) and adds `--mode probe_hubs`, which measures a hub's own reward and stop-probability.
- `scent_worker.py` — pre-existing; one added guard so the enumeration DFS skips dead-end states.

**Models (all `$SCRATCH/rgfn_runs/experiments/fixed_reward/<cell>/`):** the entry-`030` matrix
checkpoints, 5,000 iterations each on the shared `glue_standard_v1` library. SCENT cells additionally
require the `guidance_models.pt` sidecar (entry `024`) so its trained backward policy is recoverable.
FragGFN cells here are the **superseded 9-fragment** runs, preserved at `..._5k/seed42_maxnodes9/`.

**Results (committed, `experiments/lsd_hubs/matrix16/results/`):** per-cell `summary.json` + `curve.png`
(+ `compute_time.png` for SCENT); `<cell>_naive/` preserves SCENT's naive-policy control;
`fraggfn_cap9_*/` preserves the superseded-cap FragGFN numbers; `<cell>_thr<gate>/` the sweep points;
`gate_sweep/summary.csv` the compiled cross-cell table; `gate_sweep/{hub_rank_sensitivity,
flow_consistency,hub_flow_estimators}.json` the severe-test outputs.

**Heavy artifacts (not in git):** `$SCRATCH/rgfn_runs/lsdflow/matrix16/<cell>/{sample,enum}/` —
`records.csv`, `compositions.json`, `hub_graphs.pkl`, `hubs.csv`, `enum_children.json` (83 MB for
rxnflow_seh), `enumerated_records.csv`, `hub_terminal.json`.

**Job Logs:** `/scratch/markymoo/rgfn_runs/m16_*-<jobid>.{out,err}`.

## Relevant Versions

Branch `Hub-Analysis`. The work was built on branch `matrix16-lsdflow` (11 commits, `49df406` →
`7914378`), merged as **`b7b72dd`**, then corrected post-merge by **`928dc48`**.

Key commits: `49df406` the build; `5cbcc23`/`0e17acc`/`13b6888` the SLURM-environment fixes;
`557296e`/`f3d7d97` the two enumeration crashes; `01619c3` RGFN enumerate observability;
`09c8031`/`34d9177`/`a7c457e`/`16b421e` the severe tests; `cc30789` merge-readiness;
`928dc48` the FragGFN cap-9/cap-6 config-weight fix.

## Relevant Resources

**Sources**
- `[malkin2022trajectorybalance]` — Trajectory Balance (NeurIPS 2022). Eq. 6 defines the forward/backward
  policies in terms of edge and state flows; Eq. 7 is the detailed-balance constraint. Both are the basis
  of the `F(h)` reconstruction used here. `Logs/references/pdfs/malkin2022trajectorybalance.pdf`
  (arXiv:2106.04399 is the companion `[bengio2021gflownet]`).
- `[bengio2021gflownet]` — the flow-matching conservation law (inflow = outflow).
- `[gainski2025scent]`, `[seo2024rxnflow]`, `[koziarski2024rgfn]` — the generators.
- Prior entries: `025` (hub extraction + the flow-vs-control near-tie), `029`/`033` (the fair count-once
  cost model), `034` (the sEH proxy's usable thresholds ~5–6), `035` (threshold robustness),
  `037` (free-frag / pre-select-K), `030` (the 4×4 matrix), `046` (FragGFN fragment cap 9→6).

**Packages**
- torch, RDKit, `rxnflow`, Recursion `gflownet`, SCENT's `rgfn` fork — each in its own conda env
  (`rgfn`, `scent`, `fraggfn`, `rxnflow`), crossed only by subprocess.
- The shared per-generator artifact contract lives in
  `validation/lsdflow/adapters/workers/_artifacts.py`.

## Method

1. **Build the run group.** `manifest.csv` + `targets.py` + `manifest.py` resolve all 16 cells;
   `python experiments/lsd_hubs/matrix16/manifest.py` prints live readiness.
2. **Per-cell GPU pipeline**, 30,000 trajectories and the top 200 hubs:
   ```bash
   N_TRAJ=30000 N_HUBS=200 TIME=1-00:00:00 bash experiments/lsd_hubs/matrix16/launch_surrogates.sh
   ```
   Each cell runs `<gen>_worker.py --mode sample` → `campaign/pick_hubs.py` → `--mode enumerate`.
3. **Per-cell CPU readout:** `bash experiments/lsd_hubs/matrix16/run_cell_campaign.sh <gen> <target>`.
4. **Threshold sweep** (both strategies, cached enumeration):
   `bash experiments/lsd_hubs/matrix16/gate_sweep.sh <cell> [...]`.
5. **Severe tests** (pure CPU, login node):
   ```bash
   python experiments/lsd_hubs/matrix16/hub_rank_sensitivity.py rxnflow_seh scent_seh
   python experiments/lsd_hubs/matrix16/flow_consistency.py    rxnflow_seh scent_seh
   python experiments/lsd_hubs/matrix16/hub_flow_estimators.py rxnflow_seh scent_seh
   ```
6. **Hub-level probe** (for the conservation test), in the generator's env:
   `rxnflow_worker.py --mode probe_hubs --hubs-file <enum>/hubs.csv --out-dir <enum>` — measures each
   hub's own reward and `P_F(stop|h)`.

## Results

### Cells completed — 6 of 8 surrogate cells, 30k trajectories / 200 hubs each

Count-once reactions per mode, at each target's default gate. "Modes" = molecules clearing the reward
gate that are Tanimoto-<0.5 (Morgan r=3) from every already-selected member.

| cell | best-candidate | hub-batching | edge | child policy |
|---|---|---|---|---|
| scent_seh (>7.0) | 3.383 | **1.303** | 2.60× | free_frag + pre-select-K=20 |
| scent_drd2 (>0.5) | 3.673 | **1.160** | 3.17× | free_frag + pre-select-K=20 |
| rxnflow_seh (>7.0) | 3.000 | 2.446 | 1.23× | naive `reward` |
| rxnflow_drd2 (>0.5) | 2.935 | 1.833 | 1.60× | naive `reward` |
| fraggfn_cap9_seh (>7.0) | 8.000 | 2.307 | 3.47× | naive `reward` — ⚠️ deprecated cap-9 |
| fraggfn_cap9_drd2 (>0.5) | 8.000 | 3.984 | 2.01× | naive `reward` — ⚠️ deprecated cap-9 |
| **fraggfn_drd2 (>0.5)** | **4.680** | **1.173** | **3.99×** | naive `reward` — **corrected cap-6** |

Apples-to-apples (naive policy for every generator): SCENT 1.22× (sEH) / 1.38× (DRD2) — preserved in
`results/scent_*_naive/` — versus RxnFlow 1.23× / 1.60×. The larger SCENT numbers in the table come from
`free_frag` + pre-select-K, which requires promoted fragments and therefore only applies to SCENT (it
pre-builds 0 fragments for the others).

**Two caveats on this table.** (i) FragGFN is a **control**, not a peer: its count-once "reactions" are
fragment attachments and its molecules carry no synthesis route, so its apparent edge is a cost-model
artifact. (ii) Those FragGFN rows additionally use the **superseded 9-fragment cap** that entry `046`
showed collapses DRD2; treat them as provisional.

Enumeration scale (children of the 200 hubs): rxnflow_seh 232,213; rxnflow_drd2 162,042; scent_seh
438,984; scent_drd2 344,469; fraggfn_cap9_seh 428,400; fraggfn_cap9_drd2 280,875. For rxnflow_seh the
per-hub counts are **even, not front-loaded**: mean 1,161, median 1,104, range 444–1,905, Gini 0.187;
195/200 hubs are at depth 2.

### Threshold sweep — 16 (cell × gate) points

Hub-batching reactions per mode (edge over best-candidate in parentheses):

| cell | strict | mid | loose |
|---|---|---|---|
| scent_seh | 7.0 → 1.303 (2.60×) | 6.0 → 1.133 (2.99×) | 5.0 → 1.093 (3.10×) |
| scent_drd2 | 0.9 → 1.297 (2.83×) | 0.7 → 1.180 (3.11×) | 0.5 → 1.160 (3.17×) |
| rxnflow_seh | 7.0 → 2.446 (1.23×) | 6.0 → 1.437 (2.07×) | 5.0 → 1.283 (2.32×) |
| rxnflow_drd2 | 0.9 → 2.516 (1.18×) | 0.7 → 2.080 (1.42×) | 0.5 → 1.833 (1.60×) |
| fraggfn_cap9_seh | 7.0 → 2.307 (3.47×) | 6.0 → 1.933 (4.14×) | 5.0 → 1.840 (4.35×) |

Best-candidate is **nearly gate-invariant** (rxnflow_seh 2.97–3.00 across all three bars; scent_seh
3.383 at every bar), independently reproducing the built-in control of entry `035`: its top picks are all
high-reward, so the bar never binds. `fraggfn_cap9_drd2` has only its base gate (0.5).

Why rxnflow_seh looks weak at the strict bar: only **1.2%** of its 232,213 enumerated children clear
sEH 7.0 (2,674 hits; median child reward 5.30, max 7.96; 30 hubs have zero hits), so 70% of the hubs it
uses yield exactly one mode and 134 of 186 modes pay for a fresh hub. At the entry-`034`-calibrated bars
it amortizes normally (4.6 and 7.0 modes per hub at gates 6 and 5).

### Severe test 1 — is scaffold choice driven by the heuristic backward policy?

| | rxnflow_seh (heuristic P_B) | scent_seh (trained P_B) — control |
|---|---|---|
| candidates with >1 observed parent hub | **8 / 1000** | 369 / 1000 |
| top-200 overlap, P_B term ablated | **175/200 (87.5%)**, ρ 0.895 | 141/200 (70.5%), ρ 0.747 |
| top-200 overlap vs pure reward ranking | 147/200 (73.5%), ρ 0.739 | 66/200 (33.0%), ρ 0.248 |

RxnFlow's scaffold choice is effectively independent of its backward policy: its trees are shallow, so
99.2% of top candidates were seen via a single parent and the term can reorder but not reassign. The
SCENT control confirms the test has power — with a trained backward policy and 37% multi-parent
candidates, ablating it moves 30% of the set.

### How the hub-flow estimator works — and why one hub yields many answers

Worth stating plainly, because it is the crux of every severe test below and it is easy to assume the
difficulty is *computing* `F(h)`. It is not. Detailed balance (`[malkin2022trajectorybalance]` Eq. 7)
rearranges directly to

```
F(h) = F(x) · P_B(h|x) / P_F(x|h)        with   F(x) = R(x) / P_F(stop|x)
```

and we hold every term: `R(x)` is a direct proxy call, `P_F(x|h)` is the trained forward policy scored
exactly (`sampling_ratio=1.0`), `P_B(h|x)` is the model's backward policy, `P_F(stop|x)` the terminating
factor. **This is the estimator the pipeline ships.** Nothing blocks it, and for RxnFlow the common
simplification `F(x) = R(x)` is not an approximation but *exact*: on a representative hub, **1799/1799**
children sit at the reaction cap, so `P_F(stop|x) = 1`. (We keep the `/P_F(stop|x)` term only for
generality — non-cap children and other generators.)

The difficulty is that **the formula is per-child, and the children disagree.** Applying it to each of one
hub's 1,799 enumerated children — all estimating the same `F(h)` — gives:

| | log F(h) from that child |
|---|---|
| min | 48.5 |
| median | 68.2 |
| max | 113.5 |
| **spread** | **65 nats** (a factor of e⁶⁵) |

Under a converged GFlowNet all 1,799 would be identical; that is what detailed balance asserts. So the
question is never "can we compute `F(h)`" but **"which of the 1,799 is it?"** — and that disagreement is
exactly what `U(h)` measures.

**Where the spread comes from.** Across a hub's children `log R(x)` moves only ~8 nats while
`log P_F(x|h)` moves ~20 (and far more across the full set). Because the estimator *divides* by `P_F`,
the policy's uneven allocation of probability — not the rewards — dominates the answer. Hence the
variance decomposition (`Var[log P_F]` 214 vs `Var[log F(x)]` 23 vs `Var[log P_B]` 0.36 for rxnflow_seh).

**Conservation picks the weighting, we do not.** Summing the same identity over all children,
`F(h)·Σ_x P_F(x|h) = Σ_x F(x)·P_B(h|x)`, so `F(h)` is the **P_F-weighted** mean of the per-child
estimates. That is why the *unweighted* mean over all children is biased (+6.35 nats) while the
single-sampled-child value we ship is nearly unbiased (+0.43): sampling visits high-`P_F` children, which
are the ones the weighting favours anyway.

**One RxnFlow-specific caveat.** Its `P_B` is an untrained retro heuristic, not a policy fitted to
satisfy detailed balance against `P_F` — 432/1,799 children have `P_B = 1` exactly and it takes only ~6
discrete values. We apply the formula faithfully, but *the model does not honour the equation the formula
is derived from*. That is a property of RxnFlow (and why its hub ranking is P_B-insensitive: 87.5%
identical hubs with the term deleted), not a defect in the extraction.

**The Z-anchored form is a second opinion, not a replacement.** `F(h) = Z·∏(P_F/P_B)` from the source
shares *no terms* with the above, so the two disagreeing measures trajectory-balance violation directly.
The R-anchored estimator above remains the one we ship.

### Severe test 2 — reconstructing `F(h)` from conservation, and scoring our estimators

Reconstruction: `F_true(h) = Σ_x F(x)·P_B(h|x) / Σ_x P_F(x|h)` with `F(x) = R(x)/P_F(stop|x)`
(`[malkin2022trajectorybalance]` Eqs. 6–7). `log F_true` median 58.71 (rxnflow_seh) / 68.10 (scent_seh),
in β-shaped reward units (β=8).

**Normalization (the validation).** `Σ_x P_F(x|h)` median **0.999** (rxnflow_seh) / **1.000**
(scent_seh); **0/200** hubs exceed 1 and **0/200** fall below 0.01. SCENT's 5th percentile of 0.402 is
genuine stop-probability mass at hubs the policy likes to terminate at.

**Estimators vs `F_true`** — median bias in nats, and Spearman rank correlation across hubs:

| estimator | rxnflow_seh | scent_seh |
|---|---|---|
| `pick_hubs` (production: max over sampled children) | **+0.43** / 0.665 | **+0.28** / 0.608 |
| mean over sampled children | −0.19 / 0.646 | −0.77 / 0.501 |
| mean over all enumerated children (unweighted) | +6.35 / 0.438 | −3.74 / 0.238 |
| P_F-weighted mean over all children | −1.69 / **0.882** | −3.24 / 0.075 |

The shipped estimator is nearly unbiased (~0.4 nats ≈ 0.05 proxy units), but its own error spread
(p5–p95 ≈ 4.8 nats) is as large as the entire across-hub spread of `log F_true` (4.9–6.5 nats) — hence
ρ ≈ 0.6, and hence entry `025`'s near-tie between flow ranking and the `parent_of_topk` control.

### Severe test 3 — what `U(h)` actually measures

Variance decomposition of `log F̂ = log F(x) + log P_B − log P_F` (medians over 200 hubs):

| cell | U(h) | Var[log P_F] | Var[log F(x)] | Var[log P_B] |
|---|---|---|---|---|
| rxnflow_seh | 156.5 | **213.6** | 23.4 | 0.36 |
| scent_seh | 41.0 | 18.5 | 34.4 | 15.7 |

For RxnFlow, `U(h)` is dominated by forward-policy peakedness, not uncertainty about molecule quality;
SCENT is balanced, and its *trained* backward policy contributes real signal (15.7) versus RxnFlow's
heuristic (0.36).

**Sampled vs enumerated `U(h)`:** median **2.96 vs 162.4** (rxnflow_seh; sampled is lower for 82/86 hubs;
only 86/200 hubs have ≥2 sampled children) and **1.04 vs 40.5** (scent_seh; 172/175; 175/200 hubs).
Sampling only visits high-probability children, so it measures a different quantity.

**Proposed replacement.** The P_F-weighted variance is the statistic consistent with `F_true`: median
**5.15** (rxnflow_seh) / **12.8** (scent_seh), with the consensus bias shrinking to −1.7 / −3.2 nats.
Kish effective sample size shows a hub's flow is carried by only **~18–23 effective children out of
1,161–2,195 enumerated** — i.e. hub-batching harvests children the policy itself would rarely sample.

### Severe test 4 — flow conservation at the hub

Eliminating `F(h)` from `F(h) = R(h) + N(h)` and `R(h) = F(h)·P_F(stop|h)` gives a dimensionless
identity, `R(h)/(R(h)+N(h)) = P_F(stop|h)`, with both sides measured independently (`probe_hubs` supplies
the hub's own reward and stop-probability).

| quantity | rxnflow_seh (heuristic P_B), 200/200 hubs | scent_seh (trained P_B), 198/200 hubs |
|---|---|---|
| log `P_F(stop\|h)`: implied − measured | median **−6.05** (p5 −9.45, p95 −1.65) | median **−14.03** (p5 −29.52, p95 −2.21) |
| log F via `(R+N)` vs via `(N/S)` | median −0.001 (p5 −0.003) | median −0.000 (p5 −0.912) |

Both models terminate at hubs far more often than their own reward justifies — RxnFlow by ~430×
(≈0.75 sEH units at β=8), SCENT by ~1.2×10⁶. **The violation is therefore not an artifact of RxnFlow's
untrained retro backward policy**: it is *larger* in the model whose backward policy is trained and
recovered exactly (entry `024`). The second row also shows the test gains power on SCENT — its 5th
percentile departs from 0 (−0.912) because SCENT has genuine stop-mass at some hubs, whereas on RxnFlow
both forms reduce to `N` (`R/N ≈ 1e-6`, `S ≈ 0.999`) and agree tautologically.

Note the violation lives in the **stop channel**, which the shipped estimator never touches for
cap-truncated children — which is why that estimator can be unbiased while conservation fails.

*(One untested reading, recorded as a hypothesis only: SCENT is explicitly cost-aware — cost-guided
backward policy plus an exploitation penalty — so a preference for terminating early is what its training
objective rewards, and it is the more cost-tilted of the two models. We have not tested this.)*

**Learned total-flow scalars.** rxnflow_seh `log Z` = 53.07, which is *below* both its own maximum
single-molecule reward (63.69) and one hub's child-flow sum (58.71) — internally inconsistent, so it is
unusable as a reconstruction anchor. scent_seh 74.65 (> 68.10) and fraggfn 97.17 are consistent.

### Engineering: five environment/scale bugs, all fixed

Found only at real scale on compute nodes; none were visible in login-node smoke tests.

| # | bug | fix (commit) |
|---|---|---|
| 1 | bare SLURM shell has no `python` before conda activation | bootstrap `conda activate base` before the manifest emit (`5cbcc23`) |
| 2 | SLURM copies the batch script to `/var/spool`, breaking `BASH_SOURCE` repo-root | use `$SLURM_SUBMIT_DIR` (`0e17acc`) |
| 3 | `TOPK` caps the distinct-hub count, so `N_HUBS=200` was unreachable (only 100 hubs) | default `TOPK=1000` (`13b6888`) |
| 4 | `RxnAction.block` is an *asserting* property; UniRxn actions have no block | read it only for BiRxn (`557296e`) |
| 5 | SCENT's env raises `KeyError` for early-terminal states in the enumeration DFS | skip non-{A,B,C} states (`f3d7d97`) |

Jobs: the successful 200-hub run is `71344`–`71350`; enumerate-only re-runs `71417`/`71421`/`71422`/
`71423`/`71537`. RGFN cells `71766`/`71767` and the corrected-cap FragGFN DRD2 cell `71795` are queued.
Earlier batches `71115`–`71121`, `71262`–`71268`, `71270`–`71276` failed on bugs 1–3 and are superseded.

### Update (same day) — the corrected cap-6 FragGFN DRD2 cell, and the first full-scale TB residual

The cap-6 re-run completed (job **71856**, `debug` partition, 24 min: 30,000 trajectories → 200 hubs →
198,765 enumerated children). It replaces the deprecated cap-9 row and changes that cell materially:

| FragGFN DRD2 | best-candidate | hub-batching | edge | modes reached |
|---|---|---|---|---|
| cap-9 (deprecated, entry `046`) | 8.000 | 3.984 | 2.01× | **19** / 183 |
| **cap-6 (corrected)** | 4.680 | **1.173** | **3.99×** | **300 / 300** |

At cap-9 the pool was degraded enough that best-candidate could assemble only **19** diverse hits; at the
paper-faithful cap-6 both strategies reach the full 300-mode target. This is entry `046`'s finding
reproduced at the library-assembly level. The absolute reactions/mode also drops because a cap-6 molecule
carries fewer fragment attachments — so the FragGFN caveat above still stands: these are attachments, not
synthesis steps, and this cell remains a **control**.

**First full-scale trajectory-balance residual.** Because the Z-anchored prefix capture landed before this
job ran, it emitted `prefix_terms.csv` (30,000 rows) at no extra compute. Scoring the two independent
reconstructions against each other (`tb_residual.py`):

| fraggfn_drd2, n=30,000, log Z = 63.925 | median | p5 | p95 |
|---|---|---|---|
| log F_prefix (Z-anchored) | 49.57 | 43.45 | 55.99 |
| log F_suffix (R-anchored, what we ship) | 50.30 | 42.36 | 57.41 |
| **TB residual (prefix − suffix)** | **−0.86** | −4.11 | +4.52 |

The two anchors nearly agree in the median (−0.86 nats) with real per-trajectory scatter (sd 3.85) — so
the disagreement is a *local* P_F/P_B imbalance rather than a global `log Z` scale error. That is a
markedly healthier picture than the n=40 smokes suggested (RxnFlow −3.62, SCENT +1.65), and it is the
first measurement of this quantity at full scale. Caveats: FragGFN's prefix ends at the last-AddNode
*skeleton* state (which is also where its suffix estimator is anchored, so the comparison is internally
consistent), and the 3 depth-0 trajectories are a degenerate outlier (empty prefix ⇒ log F_prefix = log Z
exactly, median residual +49.56).

### The TB residual on SCENT — the cell where both anchors are sound

FragGFN's residual is reassuring but it is the *control* model: uniform `P_B`, and a prefix that ends at a
skeleton state. SCENT is the cell that actually tests the reconstruction — a genuinely **trained**,
cost-guided backward policy recovered exactly from the `guidance_models.pt` sidecar (entry `024`), and a
learned `log Z` = 74.653 that sits *above* its own maximum single-molecule log-reward (68.10), so the
Z-anchor is self-consistent (unlike RxnFlow, whose 53.07 < 63.69 makes its prefix unusable). Job **71859**
re-sampled the cell with the prefix capture on the `debug` partition — 30,000 trajectories → 29,997 records
in 27 min (1,303 s sampling + 293 s flow extraction), writing `prefix_terms.csv` alongside the usual
artifacts at no extra model cost.

| scent_seh, n=29,997, log Z = 74.653 | median | p5 | p95 |
|---|---|---|---|
| log F_prefix (Z-anchored) | 64.04 | 57.22 | 69.32 |
| log F_suffix (R-anchored, what we ship) | 62.31 | 52.64 | 67.89 |
| **TB residual (prefix − suffix)** | **+1.42** | −2.84 | +9.78 |

Three things this establishes.

**1. The shipped estimator is not systematically broken.** Two reconstructions that share *no terms* — one
walking down from the learned partition function, one walking up from the reward — land 1.42 nats apart in
the median. Put next to the ~65-nat spread *across children of a single hub* (section above), the choice of
anchor contributes roughly **1/45th** of the disagreement we already tolerate. The estimator's problem is
child variance, not anchoring. This is the direct answer to "is our heuristic `F(h)` the right number":
it is the right *quantity*; the open question is only which child to trust.

**2. The imbalance is local and accumulates per step — now visible cleanly.** The residual by hub depth:

| hub depth | n | median residual |
|---|---|---|
| 0 (degenerate, empty prefix) | 185 | +3.38 |
| 1 | 4,834 | +0.70 |
| 2 | 7,770 | +1.25 |
| 3 | 17,208 | +1.76 |

Depths 1→3 drift almost perfectly linearly: **+0.55, +0.51 nats per additional step**. A global `log Z`
scale error would show up as a *constant* offset at every depth; a per-step `P_F`/`P_B` imbalance shows up
as exactly this ramp. So the sd of 4.50 nats is not noise around a good model — it is ≈0.53 nats of
one-directional bias compounding once per reaction step. That also explains the sign flip against FragGFN
(−0.86): with uniform `P_B` there is no learned backward policy to drift, whereas SCENT's cost-guided `P_B`
is trained against a different objective than plain TB and accumulates a consistent tilt.

**3. It corroborates severe test 4 independently.** The flow-conservation violation measured at the hub
(−14.03 nats on this same SCENT cell, trained `P_B`) and this per-step ramp are two views of one defect:
the model's forward and backward policies do not balance locally. Two methods sharing no machinery — one
enumerative and conservation-based, one per-trajectory and anchor-based — agree that the failure is local
rather than global. Neither is a bug in our extraction; both are properties of the trained model.

Depth 0 (n=185) remains degenerate by construction — an empty prefix makes log F_prefix = log Z exactly —
but note it is +3.38 here versus +49.56 for FragGFN, i.e. SCENT's depth-0 hubs really do carry a large
share of total flow, as a single-fragment hub should.

`tb_residual.py`'s writer was changed to **upsert by cell** rather than overwrite
`results/gate_sweep/tb_residual.json`. Cells land one at a time (each needs its own prefix-capable
re-sample) and two agents share that file, so a plain overwrite silently dropped every cell not named in
the current invocation — it discarded the `fraggfn_drd2` row when the SCENT row was written. The file now
holds both.
