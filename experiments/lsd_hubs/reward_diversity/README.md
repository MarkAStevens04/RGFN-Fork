# `reward_diversity/` — is intrinsic diversity coupled to the reward cutoff?

**One question:** when we keep only the molecules scoring `>= τ`, do the survivors get more similar to
each other as τ rises, or is their similarity independent of reward?

**Why it belongs here.** Every cost number the campaign reports is conditioned on a reward gate —
reactions/**mode**, where a molecule counts only above a "hit" bar (7.0 in Logs/029/033; 5/6 after the
sEH-proxy calibration of Logs/034; swept in Logs/035, `campaign/compare_thresholds.py`, and
`matrix16/gate_curve.py`). `docs/paper_planning/lsd-flow-publication-strategy.md` §2.4–2.5 names the
resulting exposure exactly: the mode denominator is a fair yardstick only if raising the bar does not
*itself* change how alike the survivors are. If it does, every τ-conditioned comparison moves partly
for reasons unrelated to the selection strategy. This sub-dir measures that coupling directly on the
molecules the campaign actually costs.

It is a **diagnostic of the pools**, not a strategy comparison: no hub-batching, no best-candidate, no
cost model. Both selection strategies draw from the sets measured here, so whatever coupling exists
applies to both.

## What it measures

For each cutoff τ, keep the molecules with `reward >= τ` and report three readouts on the survivors:

| readout | what it says | direction |
|---|---|---|
| **mean pairwise Tanimoto** | how alike an average *pair* is (Morgan r=3/2048 — the same fingerprint the mode definition uses) | lower = more diverse |
| **modes per molecule kept** | canonical greedy sphere-exclusion count (Tanimoto 0.7) ÷ molecules scored — literally the reactions/mode denominator, restricted to the τ-gated pool | higher = more diverse |
| **Butina clusters per molecule kept** | the same "how many families?" question with reward taken **out** of the procedure | higher = more diverse |
| **Murcko scaffolds per molecule kept** | fingerprint-independent cross-check (§2.3 asks for a second descriptor so the finding isn't an ECFP artifact) | higher = more diverse |

**Why Butina sits next to modes.** Our mode count is greedy sphere exclusion fed *best-reward-first* —
deliberately, because that makes each mode's representative the molecule you would synthesise and keeps
our counts comparable to the RGFN/SCENT tables. But greedy exclusion is **order-dependent**, and here
the order is the reward, which is the very variable under test. Butina picks cluster centres by
neighbourhood density and never sees the reward, so it answers the same question without that
circularity. Cost: it needs the full lower-triangular distance list up front, so memory is O(N²) —
0.5 s and ~200 MB at N=2,500, but impractical above ~10k. That, not the mode count, is what pins
`--subsample` where it is.

**The design constraint that makes it valid:** mode and scaffold counts saturate with set size, so
scoring a 600k-molecule pool at τ=4 against a 5k-molecule pool at τ=8 would measure set size, not
reward. Every cutoff is therefore scored on the **same number of molecules** (`--subsample`, default
2500 — the largest N that fits inside the smallest gated set on both pools), drawn uniformly at random,
with `--replicates` independent draws to expose the sampling spread. Draws come from random
*permutations* of the pool (first N qualifying molecules), which is a uniform random size-N subset of
the gated set; because the gated sets are nested in τ, adjacent cutoffs share most molecules, so the
curve is paired along τ rather than independently jittered.

Mean pairwise similarity is a *pair* average, hence unbiased under this subsampling — but it is also
insensitive by construction (one added tight cluster barely moves it). That is why the clustering
readouts sit beside it, and why it is never quoted alone.

## Related, but a different question

This sub-dir is a **pool diagnostic**: no strategies, no cost model. The strategy-side companion is
`campaign/tau_similarity_surface.py` (Logs/052), which sweeps the same reward bar *crossed with* the
diversity cutoff and reports what hub-batching vs best-candidate actually **cost** on each cell. Read
this one first: it says whether the knob moves the pools, which is the precondition for reading the
cost surface as a strategy comparison.

## Files

| file | role |
|---|---|
| `reward_diversity_sweep.py` | the whole analysis: load pools → sweep cutoffs → CSV + JSON + 4-panel figure. Pure CPU, no model/GPU/oracle; re-reads persisted records only. |
| `results/<tag>/reward_diversity.csv` | one row per (set, cutoff): the three readouts (mean + replicate SD), pool size above the cutoff, molecules scored. |
| `results/<tag>/reward_diversity.json` | run config (records used, subsample, replicates, seed, fingerprint) + per-replicate detail + a summary (Spearman ρ of each readout against τ; readouts at the campaign's own bars 5/6/7 and at the extremes). |
| `results/<tag>/reward_diversity.{png,pdf}` | the four panels: three readouts + the fraction of the set surviving each cutoff (so a reader sees where the tail thins). |

## Running it

```bash
source ~/bin/rgfn-smoke-env.sh
python experiments/lsd_hubs/reward_diversity/reward_diversity_sweep.py \
    --sample-records /scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/records.csv \
    --enum-records   /scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enumerated_records.csv \
    --cutoffs "1,2,3,4.0:8.0:0.1" --subsample 2500 --replicates 3 --seed 42 --tag scent_seh
```

A login-node job of a few minutes. `--cutoffs` takes comma-separated values and/or inclusive
`lo:hi:step` ranges. Either `--sample-records` or `--enum-records` alone is fine.

## The `scent_seh` inputs (Logs/051)

Both pools come from the SCENT × sEH run that anchors the cost campaign (checkpoint
`fixed_reward/scent_seh/2026-07-10_17-28-06`):

- **`sample`** — `lsdflow/scent_seh_70189/records.csv`: the 30,000 sampled trajectories
  (29,997 valid terminals → **26,069 unique** molecules). What the trained policy actually generates.
- **`enumerated`** — `lsdflow/campaign_enum_seh_70363/enumerated_records.csv`: exhaustive one-reaction
  children of the 200 ranked hubs (828,448 records → **604,840 unique**). The library the campaign
  selects from.

Pools are deduplicated per molecule keeping its best reward — the same construction
`campaign/run_campaign.py::_load_candidates` uses, so this measures the library the campaign costs.

**Read the enumerated curve with its structure in mind.** Those molecules are, by construction,
one-reaction children of 200 shared hubs, so hub-mates share substructure and τ also re-weights *which
hubs* survive. The enumerated numbers therefore describe the pool the strategies pick from — not an
independent sample of chemical space. Attributing the coupling to individual hubs would need a
within-hub/cross-hub pair split, which this driver does not do.
