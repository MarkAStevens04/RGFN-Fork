# 6TD3 + DRD2 — do our two unvalidated scoring systems actually recognise real drugs?

**Date:** 2026-08-20, ~1pm

## Question

For the two protein systems whose hit thresholds were never properly checked, can the scoring system
tell real, experimentally confirmed binders from realistic look-alikes that happen to have the same
size and greasiness?

## Context & Summary

**Context** — Four of our scoring systems decide which generated molecules count as "hits", and two of
them had never been checked against a realistic negative set. Entry `045` did this properly for ClpP
and moved its bar from a value that let essentially everything through to one that separates real
binders from look-alikes. The other two were outstanding for different reasons. **6TD3** was tested in
entries `002`/`005`/`007`, but every one of those tests used decoys that all carry the same
kinase-binding fragment as the real glues — an intentionally hard control, but one that leaves open
whether the score is really just reading molecular size and greasiness. **DRD2** had no check at all;
entry `066` showed only that the scorer says "no" to unrelated chemistry, never that it says "yes"
only to real dopamine drugs.

**Summary** — For each system we build a negative set the standard way: molecules matched to the real
binders on six physical properties (size, greasiness, hydrogen-bond donors and acceptors, flexibility,
charge) but structurally unrelated to all of them. We then ask how well the score separates real
binders from those look-alikes, and what the separation is at the threshold the benchmark currently
uses. For DRD2 there is an extra wrinkle worth being careful about: unlike the docking systems, its
scorer is a *trained model*, so we also had to establish that we were not simply measuring how well it
remembers its own training data.

## Answer

**The two systems come out in opposite directions, and both answers are useful.**

**DRD2 passes cleanly, and its threshold is justified.** On real dopamine binders the model had almost
certainly never seen, it separates binders from matched look-alikes with an accuracy of 0.949 — and it
scores molecules from its own training era barely better (0.961). That near-identical pair is the
important number: it means the scorer is genuinely recognising dopamine-active chemistry rather than
recalling memorised examples, which is the thing that could not be assumed. At the threshold the
benchmark uses, a real binder is **18 times** more likely to pass than a matched look-alike.

**6TD3 fails, and its threshold cannot be defended.** Against the old warhead-sharing decoys the score
separates binders beautifully (0.946) and only 1% of decoys pass the gate. Against look-alikes matched
on physical properties, that separation falls to 0.688 and **31%** of them pass — so instead of being
83 times more likely to pass, a real glue is only **2.1 times** more likely. No choice of threshold
rescues this: the best available anywhere on the scale is about the same 2×. The concern that prompted
this experiment was correct.

## Relevance to our Publication

These are the two entries a reviewer needs before believing any "hit" count, and they land differently
enough that reporting both is what makes the pair credible. DRD2 becomes quotable: we can state the
threshold, the enrichment behind it, and — unusually — evidence that the number is not an artifact of
the model having memorised its benchmark. That last point is the sort of thing reviewers ask about
trained scoring models and almost nobody answers.

6TD3 is the harder message and the more important one. Any molecule count on that system rests on a
threshold that barely discriminates once the negatives are realistic, so the honest move is to keep it
out of the headline rather than defend it. That decision was already taken provisionally when the
system was parked; this entry supplies the number that justifies it, and converts "we are not sure
about that gate" into a measured limitation. A benchmark that reports which of its own scoring systems
does *not* hold up is considerably harder to dismiss than one that reports four successes.

## Next Experiments

**Refining for publication**

- **Decide what 6TD3 is for.** The scoring still ranks real glues above look-alikes, just weakly, so it
  may remain usable as a *ranking* signal while being unfit as a *pass/fail* threshold. Those are
  different claims and the paper should make only the one the data supports.
- **Close the remaining leak in the DRD2 holdout.** The scorer's training set was assembled from two
  public databases; our holdout can only exclude molecules by their date in one of them. Excluding by
  structure against the actual training set would remove the last loophole.
- **Report the two-tier and single-tier scores separately for 6TD3.** The combined score is the better
  discriminator against warhead-sharing decoys but the *worse* one against property-matched ones,
  which reverses an earlier finding and needs stating plainly rather than averaging over.

**Next steps in project**

- Give the same treatment to the remaining scoring system so all four are on the same footing.
- Re-read any published 6TD3 molecule count with the weakened threshold in view.

---

# Re-creation

Root for repo-relative paths: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`.

### Relevant Files

**Scripts**

- `./experiments/oracle_validation/docking_clpp/make_matched_decoys.py` — **reused unchanged**, for
  both systems. The DUD-E-style matcher behind entry `045`: matches on MW / logP / HBD / HBA /
  rotatable bonds / formal charge and requires ECFP4 Tanimoto **< 0.35 to every active**, so a decoy
  cannot be a near-analogue. Draws from the cached target-agnostic background pool, so no network.
- `./experiments/oracle_validation/docking_6td3/prep_actives_6td3.py` — **new**. Normalises the
  actives to the matcher's `smiles` column, taking them **from `known_results.csv` rather than the
  curated 175-row file**: the ROC's positives must be the molecules that actually have docking scores
  (160 of 175 docked `ok`), or the property match is balanced against molecules that never scored.
- `./experiments/oracle_validation/docking_6td3/dock_cluster.py` — **modified**: `KNOWN_CSV` and
  `DECOY_SMI` are now env-overridable, so the same pipeline can score a different negative set under
  identical conditions. No behavioural change when unset.
- `./experiments/oracle_validation/docking_6td3/submit_dock_6td3_matched.sh` — **new**. Re-docks
  actives AND the property-matched decoys **in one job**, deliberately: the actives already had scores
  from June, but scoring both in a single run removes any drift between two docking campaigns, and
  reproduces the published 0.946 as a side-effect check (it did).
- `./experiments/oracle_validation/docking_6td3/benchmark_6td3_gate.py` — **new**. Reports both
  negative sets side by side at two operating points: Youden's J, and **max enrichment** — the latter
  because a mode gate is not a diagnosis. Admitting decoys inflates every arm's mode count, so
  specificity is worth more than sensitivity, which is why the incumbent −2.0 could be a good gate
  without being Youden-optimal.
- `./experiments/oracle_validation/drd2_proxy/fetch_drd2_actives.py` — **new**. Fetches DRD2 actives
  and builds the temporal holdout. Two details are load-bearing: the holdout keys on the **earliest**
  document year across *all* of a molecule's activity rows (keying on its best measurement would let a
  molecule first reported in 2010 and re-measured in 2020 pass as unseen), and the fetch is done
  **once** with the recent subset derived locally — a second server-side year-filtered fetch would
  double the network exposure and the slow standardisation for no new information. Retries with
  backoff after a bare request lost an 8-minute fetch to a read timeout.
- `./experiments/oracle_validation/drd2_proxy/benchmark_drd2_proxy.py` — **new**. Scores both active
  sets and the decoys with the frozen oracle, using `DRD2FrozenReward._fp`'s exact featurization
  (count Morgan, r=3, `useFeatures=True`, folded to 2048) so these are the numbers the benchmark gates
  on. Emits the in-domain-minus-held-out gap as its own field.

**Models**

- `./external/scent/oracle/drd2_current.pkl` — the frozen DRD2 SVM (35,417,622 B). Almost certainly
  the Olivecrona model: `MarcusOlivecrona/REINVENT`'s `data/clf.pkl` is 35,417,609 B, a 13-byte
  difference consistent with a re-pickle.
- `./experiments/oracle_validation/docking_6td3/6TD3_tier{1,2}.pdbqt` — Tier 1 = CDK12 alone
  (isolates kinase-pocket binding), Tier 2 = the CDK12–DDB1 ternary complex. Their difference is the
  gated quantity.

**Datasets**

- `./data/validation-molecules/DDB1_CDK12_actives_docked.csv` — **new**, 160 cleaned actives that
  carry docking scores.
- `./data/validation-molecules/DDB1_CDK12_decoys_matched.csv` + `.../docking_6td3/decoys_matched.smiles`
  — **new**, 160 property-matched decoys. Match: actives MW 436 / logP 3.95, decoys MW 440 / logP 3.57
  (the old warhead-matched decoys sat at MW 343 — a 93 Da gap, which is what entry `007` flagged).
- `./data/validation-molecules/DRD2_actives_chembl_{all,heldout}.csv` — **new**, 7,120 all-years and
  2,287 held-out actives (pChEMBL ≥ 6.0, Ki/Kd/IC50, target CHEMBL217).
- `./data/validation-molecules/DRD2_decoys_matched.csv` — **new**, 2,265 property-matched decoys
  (actives MW 445 / logP 4.05, decoys MW 444 / logP 4.12).
- `./experiments/oracle_validation/drd2_proxy/drd2_activities_raw.jsonl` — the 11,896 raw activity
  rows, cached so the analysis is re-runnable without the network.

**Results**

- `./experiments/oracle_validation/docking_6td3/summary_gate_recalibration.json`
- `./experiments/oracle_validation/drd2_proxy/summary_drd2_proxy.json`

**Job Logs**

- `/scratch/markymoo/rgfn_runs/dock6td3m-74500.out` — the full docking run (14.6 min, 4× A100,
  160/160 actives and 160/160 decoys `ok`). `74499` was the 6-molecule smoke.

### Relevant Versions

Not yet committed. [TODO — add commit hash after pushing]

### Relevant Resources

**Sources**

- Olivecrona et al., *Molecular De-Novo Design through Deep Reinforcement Learning*, J Cheminform
  9:48 (2017). DOI 10.1186/s13321-017-0235-x. The DRD2 scorer's origin, and the source of the
  training-data provenance that makes the holdout meaningful: **an SVM (Gaussian kernel, C = 2⁷,
  γ = 2⁻⁶) trained on ExCAPE-DB — 7,218 actives at pIC50 > 5 plus 100,000 sampled inactives**, split
  by Butina clustering (ECFP6, 0.4) into 4,526 / 1,287 / 1,405 train / validation / test actives. The
  paper states **no explicit data cutoff**.
- Sun et al., *ExCAPE-DB*, J Cheminform 9:17 (2017), dataset posted 2017-03-07 — built from
  **PubChem and ChEMBL**. This bounds the training data above (nothing after early 2017 entered the
  model, so a 2017 holdout is safe) and identifies the one residual leak: a molecule could have
  entered training via **PubChem** while its ChEMBL record carries a later year, which a ChEMBL-date
  filter cannot see.
- Entry `045` — the ClpP calibration this follows.
- Entries `002` / `005` / `007` — the 6TD3 validation series; `007` is the MW control whose logic this
  entry extends to the remaining property axes.
- Entry `066` — the negative-only half of the DRD2 check.

**Packages**

- scikit-learn 1.8.0 (loads the pickled SVC; emits a benign `InconsistentVersionWarning` — the oracle
  is verified identical to 12 d.p. across 1.2.2 / 1.7.2 / 1.8.0), RDKit, gnina (via
  `/scratch/markymoo/gnina/run_gnina.sh`), `sklearn.metrics` for ROC/AP.

### Method

1. **6TD3 actives** — take the 160 `status == ok` rows of `known_results.csv`, clean each (largest
   organic fragment, uncharged, canonical) exactly as the ClpP/sEH actives were.
2. **6TD3 decoys** — run the shared matcher against those 160, 1:1, from the cached background pool.
3. **Dock both** in one job (`EXH=16`, 9 modes, 4× A100, 16 shards), producing Tier-2 and Tier-1 Vina
   plus the differential for each molecule.
4. **Recalibrate** with `benchmark_6td3_gate.py` against *both* negative sets.
5. **DRD2 actives** — fetch all CHEMBL217 Ki/Kd/IC50 activities at pChEMBL ≥ 6.0 (11,896 rows →
   7,120 distinct molecules); the held-out set is those whose earliest document year is ≥ 2017
   (2,287 molecules; by year 2017:338, 2018:174, 2019:465, 2020:449, 2021:330, 2022:194, 2023:298,
   2024:33, 2025:6).
6. **DRD2 decoys** — the same matcher against the 2,287 held-out actives (2,265 selected).
7. **Score** all three sets with the frozen oracle and report AUROC / AP / operating points.

```
conda run -n rgfn python experiments/oracle_validation/docking_6td3/prep_actives_6td3.py
conda run -n rgfn python experiments/oracle_validation/docking_clpp/make_matched_decoys.py \
    --actives data/validation-molecules/DDB1_CDK12_actives_docked.csv \
    --out data/validation-molecules/DDB1_CDK12_decoys_matched.csv --per-active 2
sbatch -p debug_full_node --time=01:00:00 \
    experiments/oracle_validation/docking_6td3/submit_dock_6td3_matched.sh          # job 74500
conda run -n rgfn python experiments/oracle_validation/docking_6td3/benchmark_6td3_gate.py \
    --property-decoys  /scratch/markymoo/rgfn_runs/dock_6td3_matched_74500/decoy_cdk_results.csv \
    --property-actives /scratch/markymoo/rgfn_runs/dock_6td3_matched_74500/known_results.csv
conda run -n rgfn python experiments/oracle_validation/drd2_proxy/fetch_drd2_actives.py
conda run -n rgfn python experiments/oracle_validation/docking_clpp/make_matched_decoys.py \
    --actives data/validation-molecules/DRD2_actives_chembl_heldout.csv \
    --out data/validation-molecules/DRD2_decoys_matched.csv --per-active 1
conda run -n rgfn python experiments/oracle_validation/drd2_proxy/benchmark_drd2_proxy.py
```

### Results

**1 — 6TD3: the gate collapses against property-matched decoys.** 160 actives, 160 decoys per set;
the gated quantity is the differential `ddb1_dvina` (Vina Tier2 − Tier1), lower is better.

| negative set | median MW (act / dec) | AUROC | AP | at gate −2.0: TPR / FPR / **enrichment** |
|---|---|---|---|---|
| warhead-matched (incumbent) | 436 / 343 | **0.946** | 0.920 | 67% / 1% / **82.9×** |
| **property-matched (new)** | 436 / 440 | **0.688** | 0.587 | 67% / **31%** / **2.1×** |

The actives' scores are unchanged between rows — only the negatives differ. Decoy pass-rate at the
incumbent gate rises **1% → 31%**, and enrichment falls **83× → 2.1×**.

**2 — no threshold rescues it.** Scanning every cutoff on the property-matched set, the maximum
achievable enrichment is **2.1× at −2.02** — i.e. the incumbent −2.0 already *is* the
enrichment-optimal point, and the optimum itself is only 2×. Youden's J picks −1.66 (TPR 83% / FPR
44%), which is worse. The medians show why: actives −2.20 against decoys −1.52.

**3 — the differential's advantage was specific to warhead-sharing negatives, and REVERSES.** This
contradicts the reading in entries `005`/`007` and needs stating rather than averaging away:

| quantity | AUROC vs warhead-matched | AUROC vs property-matched |
|---|---|---|
| differential (T2 − T1) | **0.946** | 0.688 |
| absolute Tier-2 | 0.890 | **0.744** |
| differential CNN affinity | 0.850 | 0.604 |

Mechanistically consistent: the differential cancels the kinase-pocket contribution, which is exactly
right when every decoy shares that warhead and exactly wasteful when none of them do. A plausible
reading of the 31% is that a molecule of the right size can gain from DDB1's extra surface without
binding the kinase pocket at all — the warhead-matched set controlled for the warhead but not for
that.

**4 — DRD2: validated, and the threshold is justified.** 2,287 held-out actives / 7,120 all-years
actives against 2,265 property-matched decoys (median score 0.0082; 3.6% above 0.5).

| active set | AUROC | AP | median score (act / dec) | at bar 0.5: TPR / FPR / **enrichment** |
|---|---|---|---|---|
| **held-out (earliest evidence ≥ 2017)** | **0.949** | 0.944 | 0.798 / 0.0082 | 65% / 4% / **18.0×** |
| all-years (in-domain) | 0.961 | 0.985 | 0.902 / 0.0082 | 74% / 4% / **20.4×** |

At the stricter declared variants the held-out enrichment rises to **27.0× (bar 0.7)** and **35.0×
(bar 0.9)**, at 56% and 40% sensitivity.

**5 — the memorisation gap is negligible, which is the point of the holdout.**

| | in-domain | held-out | gap |
|---|---|---|---|
| AUROC | 0.961 | 0.949 | **+0.012** |
| median score on actives | 0.902 | 0.798 | +0.104 |

An AUROC gap of 0.012 means the in-domain figure is **not** substantially memorisation: the model
generalises to dopamine binders reported years after its training data was assembled. The larger shift
in median score (0.90 → 0.80) shows it is *more confident* on molecules it has seen, which is expected
and does not affect ranking.

**6 — what is NOT established.** The DRD2 decoys are **presumed** inactive, not measured inactive
(standard DUD-E practice), so the 4% false-positive rate should not be over-read. And the holdout can
only exclude by ChEMBL date, while the scorer's training set came from ExCAPE-DB, which also draws on
**PubChem** — a molecule could have entered training via PubChem while its ChEMBL record is dated
later. Excluding by structure against the ExCAPE DRD2 active set would close that; the 0.012 gap makes
it unlikely to matter much, but it is not zero.

---

## Update 2026-08-20 — the entry-006 six-way ranking, re-read on the harder negatives

Entry `006`'s 2×3 grid (two scoring functions × three tiers) is what selected Vina ΔT2−T1 as the
oracle signal, and every one of its six panels was measured against the warhead-matched decoys. Re-run
unchanged against the property-matched set — same six panels, same statistics, same actives, only the
negatives differ. `experiments/ablations/sixway/{plot_violins,discrimination_stats,confusion_matrices}.py`
now take `KNOWN_CSV` / `DECOY_CSV` / `OUT_SUFFIX` / `SET_LABEL` from the environment so both sets sit
side by side rather than one overwriting the other.

**7 — the ranking does not merely shift, it INVERTS.**

| rank | warhead-matched (entry `006`) | AUROC | property-matched (new) | AUROC |
|---|---|---|---|---|
| 1 | **Vina ΔT2−T1** | **0.946** | **CNN Tier 2** | **0.804** |
| 2 | CNN Tier 2 | 0.907 | **CNN Tier 1** | 0.782 |
| 3 | Vina Tier 2 | 0.890 | Vina Tier 2 | 0.744 |
| 4 | CNN Tier 1 | 0.863 | **Vina ΔT2−T1** | **0.688** |
| 5 | CNN ΔT2−T1 | 0.850 | Vina Tier 1 | 0.619 |
| 6 | Vina Tier 1 | 0.691 | CNN ΔT2−T1 | 0.604 |

The chosen oracle signal falls from **1st to 4th of 6**. Both differentials sink; both absolute Tier-2
scores rise relative to them; and **gnina's CNN now beats Vina on both tiers**, which is a second
justification for keeping the CNN beyond the pose-selection result of entry `008`.

**8 — CNN Tier 1 is second, and that is the interpretive key.** Tier 1 uses the kinase pocket ALONE,
with no DDB1 at all. Entry `006` concluded "the discriminating information lives in what DDB1 adds,
not in the warhead pocket" — that conclusion was an artifact of its negative set. The two decoy sets
pose genuinely different questions:

* **warhead-matched** — every decoy binds the ATP pocket, so "is it a CDK12 binder?" carries no signal
  (Tier 1 = 0.691) and only the DDB1 term separates. The differential wins by design.
* **property-matched** — no decoy binds the pocket, so "is it a CDK12 binder?" is highly informative
  (Tier 1 = 0.782) while the differential, which explicitly CANCELS that term, throws the signal away.

So the differential is a **glue-specificity** signal conditional on the molecule binding the pocket.
It was never an **absolute hit** filter, and the gate has been used as one.

**9 — a composite gate helps but does not rescue.** If the missing ingredient is a pocket-binding
requirement, adding one should recover enrichment. It does, partially:

| gate on the property-matched set | TPR | FPR | enrichment |
|---|---|---|---|
| `dvina ≤ −2.0` (incumbent) | 67% | 31% | 2.1× |
| `vina_t2 ≤ −9.6` alone | 65% | 26% | 2.5× |
| `vina_t2 ≤ −9.6` **AND** `dvina ≤ −2.0` | 57% | 16% | **3.6×** |
| `vina_t2 ≤ −10.0` **AND** `dvina ≤ −2.0` | 52% | 13% | **4.0×** |

2.1× → 4.0×, at the cost of sensitivity — better, still not a strong gate. The mechanism check
confirms the two terms are orthogonal rather than redundant: the SAME composite against the
warhead-matched decoys drives FPR to **0%** (every one of those decoys binds the pocket, so the added
term is free there). A pocket term is real information the differential discards; it is just not
enough on its own.

**10 — what the false positives actually look like.** `decoy_neighbours.{png,pdf,csv}` shows the four
decoys with the strongest differential beside the real glue each most resembles. They score
**−4.52, −4.23, −4.20, −3.72** against nearest-glue values of −2.41, **+0.68**, −1.83, −1.16 — i.e.
structurally unrelated molecules (Tanimoto 0.16–0.20) beating real glues outright, one of them beating
a glue that fails the gate entirely.

One correction worth recording, caught by looking at the picture rather than the number: the matcher
enforces global dissimilarity (ECFP4 < 0.35), **not absence of a fragment**, so it does not forbid a
decoy from containing a purine — **5 of 160 (3%) do**, and one is the top-left pair. But purine
content is not the cause: **31% of non-purine decoys clear the gate against 40% of the five that have
one**. The failure is not a contaminated decoy set.
