# SCENT / sEH — are high-scoring molecules more similar to each other than low-scoring ones?

**Date:** 2026-07-28, ~4pm

## Question

When we keep only the molecules a generator scores above some quality bar, do the survivors become
more similar to one another as we raise that bar — or is how similar the molecules are unrelated to
how good they score?

## Context & Summary

Our headline number is **reactions per useful molecule** ("reactions/mode"): how much bench work it
takes to assemble a library of distinct, high-scoring molecules. Both halves of that ratio depend on a
quality bar — we only count a molecule as useful if it scores above a threshold — and our publication
notes flag exactly one risk with that design: if raising the bar *itself* makes the surviving molecules
look more alike, then the "distinct molecules" count shrinks for reasons that have nothing to do with
which selection strategy we used. Every threshold-conditioned comparison we report (entries `035`,
`050`, and the τ-sweeps) inherits that ambiguity until we measure it directly.

This entry measures it. We take the SCENT model trained on the sEH target — the model that anchors the
whole cost campaign (entries `029`–`039`, `041`, `047`, `048`) — and look at two molecule sets: the
**30,000 molecules it generated** when we sampled it, and the **605,000 molecules** we got by
exhaustively enumerating one more reaction step from its 200 best shared scaffolds. For each of 44
quality bars from 1.0 to 8.0 we keep only the molecules scoring at least that high and ask four
questions about the survivors: how chemically similar are they to each other on average, how many
genuinely distinct molecules ("modes") they contain per molecule kept, how many distinct families a
**reward-blind** clustering finds in them, and how many distinct core skeletons. Every bar is measured
on the same number of molecules, so the comparison is not confounded by set size.

The reward-blind count matters more than it sounds. Our usual way of counting distinct molecules walks
the set from the best-scoring molecule downwards, which is exactly the right thing when you want to
report *which* molecules to make — but it puts the score inside the counting procedure, and the score
is the variable under test here. So we also count with a clustering method that never looks at the
score (Butina), and check the two agree.

## Answer

**Similarity is not independent of score — but the coupling is negligible exactly where we work, and
only becomes large past a bar no real sEH inhibitor reaches.** Raising the quality bar makes the
survivors more alike, perfectly monotonically, in both molecule sets. Below a bar of ~7 the effect is
tiny: moving from 5 to 6 costs 0.2% (generated) / 0.7% (enumerated) of the distinct-molecule count, and
5 to 7 costs 2.3% / 4.2%. Above ~7 it turns steep, and by 8.0 the sets have lost roughly 42–44% of
their distinct molecules per molecule kept — the survivors there are heavily overlapping families.

Our headline comparisons all sit at bars of 5, 6, or 7, i.e. inside the flat regime. The **fine gate
curves do not** — the committed `gate_curve` sweep runs 4.0 → 8.0 in half-steps — so their top two
points (7.5 and 8.0) are measured on sets whose own diversity has materially changed, and should be read
as a different regime rather than as the same curve continued.

**The reward-blind count agrees, so this is not an artifact of how we count.** Clustering without ever
looking at the score gives the same monotone story and a marginally *steeper* collapse (−47.2% vs
−44.2% from bar 1 to 8), so the coupling is a property of the molecules, not of our reward-ordered
counting procedure. That also means our usual count is very slightly generous in the collapse regime,
which is worth knowing but changes nothing at the bars we use.

Two secondary findings matter as much as the headline. First, the **scaffold cross-check disagrees in
sign** for the enumerated library: its core-skeleton diversity per molecule *rises* as the bar rises
(0.968 → 0.998), so the mild fingerprint-level trend below 7 is not a robust claim that high-scoring
chemistry is more redundant. The descriptors agree only about the steep collapse in the extreme tail.
Second, the enumerated library is **more** internally diverse than the model's own generated sample at
every bar below ~7.6 — measured on equal numbers of molecules — which is the opposite of the intuition
that molecules built one reaction off 200 shared scaffolds must be redundant.

## Relevance to our Publication

This supplies the missing *prerequisite* for the quality-bar axis our publication notes propose
(`docs/paper_planning/lsd-flow-publication-strategy.md` §2.5): conditioning reactions/mode on a bar is
only a clean control if the bar doesn't itself change the pools' diversity. It answers the attack a
sharp reviewer at Digital Discovery (or an ML4Science workshop) would reach for first — our headline
metric counts distinct molecules above a quality bar, so "you improved the ratio by moving the bar" is
the obvious line — with a number instead of an argument: across the bars we report at, the
distinct-molecule count of the pools themselves moves by under 5%, so the hub-batching-vs-naive
differences (1.2–3.2× in entry `050`) are not riding a diversity artifact. It also gives us an **honest
operating range to state in the text**: at or below ~7 the comparison is like-with-like, and we say so
rather than leaving the sweep truncated where it happens to look best (§3 asks us to report the
crossover instead of stopping short of it). §2.3's request for a second, non-fingerprint descriptor is
satisfied by the Murcko cross-check, and its disagreement in sign is reported rather than buried.

It does **not** cover §2.4's other half — the numerator guard (reward and synthetic-depth distributions
alongside the ratio, showing reaction savings aren't bought with molecular quality). That remains
unwritten.

## Next Experiments

**Refining for publication**

- **Repeat on a second model and target.** Everything here is one checkpoint on one reward. The
  matrix16 cells (entry `050`) already hold sampled + enumerated pools for RxnFlow and a second SCENT
  seed, so the same driver answers "is the flat-then-collapse shape a property of the reward landscape
  or of this one run?" cheaply.
- **Split the pairs into hub-mates vs strangers.** The enumerated library's molecules are children of
  200 shared scaffolds, so a rising bar also re-weights *which* scaffolds survive. Separating pairs
  that share a parent from those that don't would say whether the tail collapse is scaffold-mates
  collapsing or the whole pool tightening — the driver does not do this yet.
- **Sweep the two knobs together — started.** Distinct-molecule counts here use one similarity cutoff
  (0.7). Entry `052` crosses the quality bar with the similarity cutoff on the *cost* side, making the
  two robustness analyses one surface instead of two independent lines (how §2.3/§2.5 want them
  presented). What is still missing is the same crossing on the *pool* side measured here — i.e. this
  sweep repeated at several similarity thresholds, not just 0.7.

**Next steps in project**

- **State the operating range wherever a quality bar appears.** The gate curves (`gate_curve.py`,
  entry `035`) already sweep to 8.0, so they should carry the ~7 boundary explicitly — a reader needs to
  know which points compare like with like and which are measured on a materially more redundant set.
- Feed the same measurement into the docking-reward targets (6TD3 / ClpP) before we condition any
  docking-based cost claim on a threshold — those bars were calibrated separately (entry `045`) and
  have no equivalent check yet.

# Re-creation

## Relevant Files

Repo root `./`; scratch paths absolute.

**Scripts**

- `./experiments/lsd_hubs/reward_diversity/reward_diversity_sweep.py` — the whole analysis: loads one
  or both record pools, sweeps the reward cutoffs at fixed subsample size, writes CSV + JSON + the
  four-panel figure. Pure CPU; no model, GPU, or oracle call.
- `./experiments/lsd_hubs/reward_diversity/README.md` — what the sub-dir measures, the fixed-N design
  constraint, and how to re-run it.
- `./validation/lsdflow/metrics/diversity.py` — the canonical diversity/mode metrics. Extended here
  with `mean_pairwise_similarity`, `count_butina_clusters` (the reward-blind cluster count), public
  `ecfp` / `murcko_scaffold` (so a caller can cache them), an optional `fps=` argument on the mode
  functions, and a `BulkTanimotoSimilarity` fast path in the sphere-exclusion greedy. Output-identical
  to the previous implementation (checked below), 2.7–5.3× faster — the sweep re-scores 44 nested
  subsets per pool, so caching is what makes it a minutes-long login-node job.
- `./glue/samplers/lsdflow/mode_select.py` — the **production** mode selector the campaign runs on.
  Same `BulkTanimotoSimilarity` substitution inside `DiverseThresholdModeSelector.accept`, 3.5–6.1×
  faster on a best-candidate run, verified to leave every accepted point byte-identical. Made for the
  entry-`052` surface (200+ campaign runs), recorded here because the verification lives here.

**Datasets** (both from the SCENT × sEH run that anchors the cost campaign, checkpoint
`/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/train/checkpoints/last_gfn.pt`)

- `/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/records.csv` — the sampling stage's 30,000
  trajectories (29,997 valid terminals → 26,069 unique molecules). This is **what the trained policy
  generates**, i.e. the candidate pool best-candidate selects from.
- `/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enumerated_records.csv` — exhaustive
  one-reaction children of the 200 ranked hubs (828,448 records → 604,840 unique molecules). This is
  **the library hub-batching selects from**, and the same enumeration Logs/031–039/041/044 cost.

**Results**

- `./experiments/lsd_hubs/reward_diversity/results/scent_seh/reward_diversity.csv` — one row per
  (set, cutoff): the three readouts with replicate SD, pool size above the cutoff, molecules scored.
- `./experiments/lsd_hubs/reward_diversity/results/scent_seh/reward_diversity.json` — run config,
  per-replicate detail, and the summary (Spearman ρ vs τ; readouts at the campaign bars 5/6/7 and the
  sweep extremes).
- `./experiments/lsd_hubs/reward_diversity/results/scent_seh/reward_diversity.{png,pdf}` — the figure.

## Relevant Versions

Branch `Hub-Analysis`, last commit `1e82730` ("sweep_campaign: ideal-direction arrows"). The files this
entry adds/changes are **not yet committed**:

- new: `experiments/lsd_hubs/reward_diversity/` (driver, README, `results/scent_seh/*`)
- new: `Logs/051_reward-cutoff-vs-intrinsic-diversity.md`
- modified: `validation/lsdflow/metrics/diversity.py`, `validation/lsdflow/metrics/__init__.py`
- index row added to `docs/RESEARCH_CONTEXT.md`

`[TODO — add commit hash after pushing]`

## Relevant Resources

**Sources**

- `docs/paper_planning/lsd-flow-publication-strategy.md` §2.3–2.5 — the reason this experiment exists:
  the mode denominator must be shown not to move with the reward gate, and §2.3 asks for a second
  descriptor so the result isn't fingerprint-specific.
- `docs/LSD_FLOW_PROPOSAL.md` §11 — pins the mode definition (greedy sphere-exclusion) and the
  reactions-per-mode cost metric this diagnostic protects.
- Logs/034 — the sEH-proxy calibration that made the choice of reward bar a live question
  (7.0 is deep in the proxy-optimised tail; the empirically meaningful bar is ≈5–6).
- Logs/035 — the earlier robustness check at bars 5/6/7, which compared *strategies* across bars but
  never measured what the bar does to the pools themselves.
- `[bengio2021gflownet]` / `[koziarski2024rgfn]` — origin of the mode metric (Tanimoto sphere
  exclusion on ECFP, best-reward-first), reproduced by `validation/lsdflow/metrics/diversity.py`.

**Packages**

- RDKit — fingerprints (`AllChem.GetMorganFingerprintAsBitVect`, Morgan r=3/2048, no
  features/chirality), `DataStructs.BulkTanimotoSimilarity`, `MurckoScaffold` — all via
  `validation/lsdflow/metrics/diversity.py`.
- matplotlib — the figure, in `reward_diversity_sweep.py::plot`.
- No torch/dgl/gin: the driver reads persisted CSVs only.

## Method

1. **Verified the metric-module speedup changes nothing.** Re-implemented the pre-edit per-pair greedy
   inline and compared against the new `BulkTanimotoSimilarity` path on 2,500 sampled molecules plus
   two deliberately broken keys (empty string, non-SMILES), across similarity thresholds 0.5/0.7/0.9 ×
   {no gate, gate 7.0}, and on the no-reward and `max_modes` paths. Every mode-index list identical;
   cached-fingerprint results identical to uncached; `unique_scaffolds` identical cached vs uncached.

2. **Built the pools.** Loaded both record CSVs, deduplicating by molecule and keeping each molecule's
   best reward — the same pool construction `campaign/run_campaign.py::_load_candidates` uses, so the
   sweep measures the library the campaign costs.

3. **Swept 44 reward cutoffs** — `1, 2, 3` then `4.0 → 8.0` in steps of `0.1` — on each pool:

   ```bash
   source ~/bin/rgfn-smoke-env.sh
   python experiments/lsd_hubs/reward_diversity/reward_diversity_sweep.py \
       --sample-records /scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/records.csv \
       --enum-records   /scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enumerated_records.csv \
       --cutoffs "1,2,3,4.0:8.0:0.1" --subsample 2500 --replicates 3 --seed 42 --tag scent_seh
   ```

   At each cutoff, keep the molecules with `reward >= τ` and score **exactly 2,500 of them**, drawn
   uniformly at random, three independent times. 2,500 is the largest N that fits inside the smallest
   gated set on both pools (τ=8.0 leaves 2,821 sampled / 5,195 enumerated molecules), so no cutoff
   falls back to a smaller sample and the mode/scaffold counts — which saturate with set size — stay
   comparable across the whole sweep. Draws come from three random permutations of the pool (first N
   qualifying molecules in each), which is a uniform random size-N subset of the gated set; since the
   gated sets are nested in τ, adjacent cutoffs share most of their molecules, making the curve paired
   along τ rather than independently jittered.

4. **Readouts per subsample:** mean pairwise Tanimoto over all 3,123,750 pairs; the canonical greedy
   mode count at Tanimoto 0.7 (best-reward-first ordering, gate already applied by the draw) divided
   by 2,500; the **Butina cluster count** at distance 0.3 (= similarity 0.7), which picks cluster
   centres by neighbourhood density and never reads the reward, divided by 2,500; and unique
   Bemis-Murcko scaffolds divided by 2,500. Reported as the mean over the three replicates with their
   SD. Butina needs the full lower-triangular distance list up front (3.1M distances at N=2,500,
   ~0.5 s, ~200 MB) — that O(N²) memory, not the mode count, is what pins the subsample size.

5. **Two output-identical speedups** were needed to make the sweeps affordable, both replacing a
   per-pair Python loop over `TanimotoSimilarity` with one `BulkTanimotoSimilarity` call:
   `validation/lsdflow/metrics/diversity.py` (2.7–5.3× on the mode count) and — because the τ×cutoff
   cost surface in Logs/052 calls it 200+ times — `glue/samplers/lsdflow/mode_select.py`, the
   production selector every campaign number depends on (3.5–6.1× on a best-candidate run). Both were
   verified to produce identical output before use; see Results.

## Results

Whole sweep: **381 s** on the Balam login node with all four readouts (sample pool 180.2 s, enumerated
201.1 s; pool loads 0.1 s / 2.4 s — Butina roughly doubles the 189 s three-readout run). 44 cutoffs ×
3 replicates × 2,500 molecules = 330,000 scoring slots served by 13,861 (sample) + 33,509 (enumerated)
distinct fingerprints, which is what the cache buys.

`n_scored` = 2,500 at every row; `mean pairwise` = mean Tanimoto over all 3,123,750 pairs (Morgan
r=3/2048); `modes/mol` = greedy sphere-exclusion count at Tanimoto 0.7 ÷ 2,500; `scaffolds/mol` =
unique Bemis-Murcko scaffolds ÷ 2,500. Values are means over 3 draws; SD of `modes/mol` ≤ 0.013
everywhere (largest at τ=7.7, sample).

**Generated sample** — `scent_seh_70189/records.csv`, 26,069 unique molecules, reward 0.43–8.40:

| τ | molecules ≥ τ | mean pairwise | modes/mol | butina/mol | scaffolds/mol |
|---|---|---|---|---|---|
| 1.0 | 26,068 | 0.1686 | 0.9451 | 0.9404 | 0.9972 |
| 5.0 | 25,878 | 0.1690 | 0.9444 | 0.9395 | 0.9972 |
| 6.0 | 25,372 | 0.1701 | 0.9427 | 0.9377 | 0.9972 |
| 7.0 | 21,001 | 0.1777 | 0.9229 | 0.9153 | 0.9964 |
| 7.5 | 13,601 | 0.1906 | 0.8619 | 0.8439 | 0.9949 |
| 8.0 | 2,821 | 0.2251 | 0.5271 | 0.4964 | 0.9941 |

**Enumerated library** — `campaign_enum_seh_70363/enumerated_records.csv`, 604,840 unique molecules,
reward 0.00–8.40:

| τ | molecules ≥ τ | mean pairwise | modes/mol | butina/mol | scaffolds/mol |
|---|---|---|---|---|---|
| 1.0 | 604,626 | 0.1459 | 0.9797 | 0.9783 | 0.9677 |
| 5.0 | 511,677 | 0.1500 | 0.9777 | 0.9767 | 0.9849 |
| 6.0 | 362,820 | 0.1550 | 0.9711 | 0.9697 | 0.9888 |
| 7.0 | 140,934 | 0.1663 | 0.9384 | 0.9320 | 0.9905 |
| 7.5 | 43,609 | 0.1828 | 0.8649 | 0.8463 | 0.9924 |
| 8.0 | 5,195 | 0.2091 | 0.5723 | 0.5295 | 0.9981 |

**Rank correlation with τ over the 44 cutoffs** (Spearman ρ):

| readout | generated sample | enumerated library |
|---|---|---|
| mean pairwise Tanimoto | **+1.000** | **+1.000** |
| modes per molecule (reward-ordered) | −0.996 | −0.995 |
| Butina clusters per molecule (reward-blind) | **−0.998** | **−0.991** |
| Murcko scaffolds per molecule | −0.844 | **+0.961** |

**Where the flat regime ends.** First 0.1-wide step of τ that costs more than 0.5 pp of `modes/mol`:
**τ = 7.1** (sample) and **τ = 7.0** (enumerated). Relative to τ=1, `modes/mol` has fallen 2.3% / 4.2%
at τ=7, 8.8% / 11.7% at τ=7.5, and 44.2% / 41.6% at τ=8.0; mean pairwise similarity has risen 5.4% /
14.0%, 13.1% / 25.3%, and 33.5% / 43.3% at those same points.

**Cross-set comparison at matched N** (2,500 molecules scored in both): the enumerated library is less
self-similar than the generated sample at every cutoff up to ~7.6 — 0.1459 vs 0.1686 at τ=1, 0.1663 vs
0.1777 at τ=7 — and the two curves converge only in the far tail (0.2091 vs 0.2251 at τ=8.0).

**Two caveats on the numbers themselves.**

1. At τ=8.0 the sample pool holds 2,821 molecules, so the 2,500 scored are 89% of the gated set (48%
   for the enumerated pool). The three replicates there overlap almost completely, so that point's SD
   is not an independent-resample band — it is close to a measurement of the whole set.
2. Context for the collapse regime, from entry `034`: the sEH proxy puts real ChEMBL sEH inhibitors at
   a median around 7.7 and **0 of 2,315 actives reach 8.0**. The bars where diversity collapses are
   therefore inside the proxy-optimised tail, not the range where the score tracks measured activity —
   which is also the range (≈5–6) the calibration recommends and where this sweep finds the coupling
   negligible.

**Reward-ordered vs reward-blind counting.** Butina tracks the mode count closely and is uniformly a
little stricter, with the gap widening as the sets get redundant — generated sample: 0.6% fewer
families at τ=1 (0.9404 vs 0.9451), 0.8% at τ=7, **5.8% at τ=8** (0.4964 vs 0.5271). Relative collapse
from τ=1 to τ=8 is −47.2% (Butina) vs −44.2% (modes) on the sample and −45.9% vs −41.6% on the
enumerated library. So the reward-ordered count is marginally optimistic in the tail, and the
coupling itself is not an artifact of feeding the greedy selector in reward order.

**Metric-module equivalence check** (step 1 of the method): mode-index lists identical between the
pre-edit per-pair greedy and the new `BulkTanimotoSimilarity` path for every combination tested
(similarity 0.5/0.7/0.9 × gate {none, 7.0}, plus the no-reward and `max_modes` paths); cached-fingerprint
results identical to uncached; `unique_scaffolds` identical cached vs uncached. Speedup on 2,500
molecules: 5.11 s → 1.08 s at similarity 0.7.
