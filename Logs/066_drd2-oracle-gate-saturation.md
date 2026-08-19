# DRD2 — is our hit bar too easy, or has the generator simply solved the oracle?

**Date:** 2026-08-19, ~9am

## Question

Almost every molecule our generators make for the dopamine-receptor target counts as a "hit" — is the
bar set too low, or are the molecules genuinely good?

## Context & Summary

**Context** — Three of our four scoring systems have an entry establishing where their hit threshold
should sit: entry `034` for the sEH proxy, `045` for ClpP (which moved the bar from a value that
passed essentially everything to one that separates real binders from lookalikes), and `002`/`005`
for the CDK12 system. The dopamine target, DRD2, has no such entry. Its bar of 0.5 arrived in entry
`028` as a *bug fix* — the code had been applying the sEH bar of 7.0 to a score that only ever runs
between 0 and 1, so nothing passed at all — and 0.5 is simply the midpoint of that range. When we
queued the dopamine cells we noticed that **97.5% of the molecules our generator produces clear the
bar**, and even the strictest variant we offer (0.9) passes 83%. That is exactly the pattern entry
`045` diagnosed as a broken threshold on a different target, so it needed checking before any
dopamine number goes in the paper.

**Summary** — We take the scoring model itself and ask it about molecules that are certainly *not*
dopamine-receptor drugs: the 160 known glue compounds and 248 decoys from the CDK12 work (a
completely unrelated protein), and 3,000 purchasable building blocks. If the bar is too low, these
should score highly too. If the bar is sound, they should not.

## Answer

**The bar is not too low — the generator has saturated the scorer.** Not one of the 3,408 unrelated
drug-like molecules we tested scores above 0.5; their typical score is under 0.01, while our
generated molecules sit at 0.98. The separation is essentially total, so 0.5 is a defensible
definition of a hit and the earlier concern was misdirected. The right word for what we observed is
not "soft" but "saturated", and that is a statement about the generator, not the threshold.

What this does **not** establish is the other half of the question. Every molecule we tested is a
true negative, so we have shown the scorer says "no" to the wrong chemistry — we have *not* shown it
says "yes" only to genuine dopamine-receptor ligands, because no known active appeared in the
comparison. That distinction matters more here than it usually would, because this particular scorer
is a well-known target for gaming, and a generator that scores 0.98 on almost everything it makes is
what being gamed would look like. So the threshold is sound, and whether the scorer can be fooled is
open and untested.

## Relevance to our Publication

The practical consequence is a labelling one, and it is easy to get wrong. Because the bar filters
almost nothing for this target, the dopamine cells are effectively a **diversity-only** comparison —
whatever advantage we show there comes entirely from picking varied molecules, not from picking
high-scoring ones. On the sEH target the bar removes a meaningful fraction of candidates, so the two
targets are not measuring quite the same thing and must not be averaged into a single number without
saying so. Separately, a reviewer who knows this scorer will ask whether it has been gamed; we can
now say we checked its behaviour on unrelated chemistry and found no false positives, while being
straightforward that the complementary check is still outstanding. Being first to name the limitation
is considerably better than being asked about it.

None of this disturbs the comparison itself. Every method in the benchmark — ours and both
competitors — is scored by the identical model, so even if that model can be fooled, it is fooled
the same way for everyone, and a comparison of *batching strategies at fixed reward* survives intact.

## Next Experiments

**Refining for publication**

- **Do the missing half of the calibration.** Score known dopamine-receptor actives against
  property-matched inactives and report the separation the way entry `045` did for ClpP. That is the
  measurement that would let us claim the scorer is trustworthy rather than merely not obviously
  broken, and it is the only one of our four targets still missing it.
- **Check the size confound.** Our molecules are markedly larger than the negatives we tested here
  (median 515 vs 343–436). Entry `007` found on a different target that a size gap of that kind could
  account for most of an apparent signal, so it should be ruled out rather than assumed absent.
- **Say in the paper which targets have a binding bar and which do not.** One sentence, but it
  changes how the dopamine column should be read.

**Next steps in project**

- Report the dopamine competitor cells (now queued) as a diversity-only comparison, kept separate
  from the sEH numbers rather than pooled with them.

---

# Re-creation

Root for repo-relative paths: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`.

### Relevant Files

**Models**

- `./external/scent/oracle/drd2_current.pkl` — the frozen TDC DRD2 activity model (an sklearn SVC),
  35 MB. This is the single scorer shared by RGFN's `DRD2Proxy` and every baseline entrant, which is
  why a flaw in it would be a flaw in all arms equally rather than a confound between them.

**Datasets** (all used as TRUE NEGATIVES — none is a dopamine ligand)

- `./experiments/oracle_validation/docking_6td3/known_results.csv` — 160 known CDK12–DDB1 glue
  compounds. Real, drug-like, and active against a completely unrelated protein: the sharpest
  available test of whether the scorer rewards generic drug-likeness.
- `./experiments/oracle_validation/docking_6td3/decoy_cdk_results.csv` — 248 purine-bearing decoys
  from the same entry.
- `./data/models/rxnflow_env/building_block.smi` — purchasable building blocks; 3,000 sampled with
  `random.Random(0)`. Smaller and simpler than the others, giving a low-complexity reference point.

**Inputs for the positive side**

- `/scratch/markymoo/rgfn_runs/lsdflow/t45_drd2_seed43/sample/records.csv` — 29,999 sampling events,
  29,822 distinct molecules, from the seed-43 dopamine run.

### Relevant Versions

`0296459` — the state this entry reads. No repo files were modified; this entry is a measurement
against existing artifacts. The scoring snippet was run inline and is reproduced in full under
Method, since it is short enough that a standalone script would add a file without adding clarity.

### Relevant Resources

**Sources**

- Entry `028` — where the 0.5 bar entered, as a fix for the sEH bar leaking onto a 0–1 scale.
- Entry `045` — the model for what a proper threshold calibration looks like (ClpP, AUROC 0.895,
  Youden-optimal cutoff against property-matched decoys).
- Entry `034` — the equivalent for the sEH proxy, and the entry that established this project treats
  "the bar is in the optimised tail" as a real risk rather than a hypothetical.
- Entry `007` — the molecular-weight control on a different target; the reason the size gap noted
  above is flagged rather than dismissed.
- `./validation/generators/fraggfn/fixed_reward.py` (`DRD2FrozenReward`) — the featurization
  reproduced below, verified bit-for-bit against `tdc.Oracle("DRD2")`.

**Packages**

- scikit-learn 1.8.0 — loads the pickled SVC. Emits `InconsistentVersionWarning` (pickled under
  0.23.0); this is known-benign, as the DRD2 oracle has been verified identical to 12 decimal places
  across sklearn 1.2.2 / 1.7.2 / 1.8.0.
- RDKit — count Morgan fingerprints, `radius=3`, `useFeatures=True` (FCFP6), folded to 2048.

### Method

1. Load `drd2_current.pkl` and apply TDC's exact featurization — count Morgan fingerprint,
   radius 3, `useFeatures=True`, folded modulo 2048 — matching `DRD2FrozenReward._fp` so the scores
   are the same ones the benchmark uses.
2. Score the three negative sets and, for contrast, the distinct molecules from the seed-43 dopamine
   run (best score per molecule, matching how the pool is deduplicated for selection).
3. Report mean, median, and the fraction above 0.5 and 0.9, alongside median molecular weight so the
   size gap is visible rather than hidden.

```
conda run -n rgfn python -c "<the snippet in Method step 1-3; see git history of this entry>"
```

### Results

**1 — the scorer does not reward generic drug-likeness.** `p` is the predicted dopamine-activity
probability; the bar is 0.5.

| set | n | median MW | mean p | median p | frac > 0.5 | frac > 0.9 |
|---|---|---|---|---|---|---|
| CDK12–DDB1 known glues | 160 | 436 | 0.011 | 0.007 | **0.0%** | 0.0% |
| CDK12 decoys | 248 | 343 | 0.004 | 0.002 | **0.0%** | 0.0% |
| ZINC building blocks | 3,000 | 184 | 0.008 | 0.003 | **0.0%** | 0.0% |
| **our generated (seed 43)** | 29,822 | 515 | 0.933 | 0.980 | **97.5%** | 83.0% |

**0 of 3,408** true negatives cross the bar. The bar is therefore not "too low" in chemical space;
it is non-binding *within our pool* only because the generator has saturated the scorer.

**2 — the gate is non-binding at every variant we offer.** Distinct molecules passing, seed 43:

| bar | distinct passing | fraction |
|---|---|---|
| 0.5 (headline) | 29,078 | 97.5% |
| 0.7 | 28,213 | 94.6% |
| 0.9 | 24,762 | 83.0% |

Even the strictest declared variant leaves five sixths of the pool, so no available setting turns
this into a binding reward filter. The dopamine cells measure diversity, not hit-rate.

**3 — what is NOT shown.** Every molecule in the table above is a true negative. The complementary
measurement — known dopamine actives against property-matched inactives, i.e. the ClpP protocol of
entry `045` — has not been run, so no claim is made here about the scorer's precision, only about
its specificity against unrelated chemistry. The 79–171 Da median size gap between our molecules and
these negatives is likewise unexamined and is a live alternative explanation for part of the
separation (cf. entry `007`).
