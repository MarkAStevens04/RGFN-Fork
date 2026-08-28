# FragGFN — moving a generator between two benchmarks that were never normalized the same way

**Date:** 2026-08-28, ~5pm

## Question

When a generator is reclassified from one half of our benchmark to the other, what has to change for
its numbers to mean the same thing as its new neighbours'?

## Context & Summary

**Context.** The project runs two comparisons. One is the **competitor block** — five external
programs (REINVENT, Saturn, S3-GFN, TANGO, SynFormer) measured against ours on how many distinct
good molecules a fixed budget of 100 chemical reactions buys. The other is the **16-cell matrix**,
comparing reaction-aware generators (RGFN, SCENT, RxnFlow) where hub-batching applies. FragGFN sat in
the matrix, but it assembles molecules from a fragment library with no reaction model at all — which
makes it a sibling of S3-GFN, not of RGFN. Entry [046] had already found and fixed a collapse in its
DRD2 cell, and entry [030]'s campaign had run it at matrix scale.

**Summary.** Moving it required a full retrain, because the two halves are normalized on *different
quantities*: the competitor block on how many times the scoring function is called (~10,000, taken
from each program's own published default), the matrix on how many training steps are run (5,000).
Those are not interchangeable — a training step scores a whole batch of molecules. This entry covers
the migration, the four things that had to change, and two mistakes made while measuring them.

## Answer

FragGFN had been training on **thirty-two times** the budget every other competitor is held to, which
is the same kind of error a previous audit found in REINVENT and which there demonstrably distorted
the results. Normalizing it also required giving it the per-molecule logging every other competitor
has, making its fragment cap consistent across seeds, and moving where its runs are stored. All nine
cells now train on exactly the same number of scoring calls as REINVENT and S3-GFN, and six of the
nine already contain more than enough diverse molecules for the next stage without any further work.

The wider point is that the mismatch was invisible until a generator had to satisfy both conventions
at once. The other three generators still sit on the matrix convention, and whether that is a real
unfairness has not been measured.

## Relevance to our Publication

Reviewers will read the competitor table as "everyone got the same budget", and for the first time
that is now true of FragGFN. It also strengthens a claim we can make positively: our two non-reaction
GFlowNets, FragGFN and S3-GFN, can now be read directly against each other, because they differ in
architecture rather than in how much they were allowed to spend. The disclosed deviation on the
fragment cap follows the doctrine already used for TANGO — publishing a baseline at a setting we have
measured to be broken would read as sandbagging, so we take the working setting and say so.

## Next Experiments

**Refining for publication.** The three DRD2 cells hold 376-435 diverse molecules against a target of
500, so they are the only FragGFN cells needing further sampling; the reward profile of what that
extra sampling turns up should be reported beside the count, since molecules found late are not
equivalent to molecules found early.

**Next steps in project.** Measure what the three remaining matrix generators actually spend on
scoring calls. If it is comparable to FragGFN's old 320,000 against the competitors' 10,000, the
asymmetry runs in our own favour and is the first thing a careful reader would check.

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**

- `./validation/generators/fraggfn/run_fraggfn_fixed.py` — the FragGFN adapter. Gained a
  `_TracedReward` wrapper (records every reward evaluation for all three reward types, taking
  `raw_scores()` where available), `timing.json` output, and an explicit
  `gcfg.algo.num_from_policy` so the budget cannot drift with an upstream default change.
- `./experiments/fixed_reward/scale5k/submit_baseline.sh` — repointed at the normalized configs and
  moved `fraggfn` into the external-entrant run-dir naming.
- `./experiments/lsd_hubs/campaign/upsample_to_modes.py` — Stage 2. `fraggfn` added to its generator
  list; `synformer` deliberately absent.

**Datasets**

- `./external/gflownet/src/gflownet/tasks/seh_frag.py` — the authors' own task defaults, and the
  source of truth for both `num_from_policy = 64` and `max_nodes = 9`.
- `./validation/configs/fraggfn_{seh,drd2,clpp_docking}_fixed_norm.yaml` — the normalized configs.

**Results**

- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/fraggfn_{seh,drd2,clpp}/seed{42,43,44}/` —
  the nine retrained cells, each with `trace.csv`, `timing.json` and 2,000 candidates.

**Job Logs**

Root: `/scratch/markymoo/rgfn_runs`

- `fg_surrogate-75136.out` — the six surrogate cells, one job.
- `fg_clpp_{42,43,44}-7513{7,8,9}.out` — the three docking cells.
- `fg_resume-75151.out` — the Stage-2 readiness test.
- `fg_smoke2-75134.out` — the first full normalized cell, used for sizing.

## Relevant Versions

```
7b8b3eb  FragGFN is Stage-2 ready: it has a sampler, so it can be upsampled; SynFormer cannot
a327c3a  FragGFN joins the competitor block, at 1/32nd of the budget it was training on
```

## Relevant Resources

**Sources**

- Bengio et al., *Flow Network based Generative Models for Non-Iterative Diverse Candidate
  Generation* — the fragment-based GFlowNet this entrant implements. Upstream code:
  https://github.com/recursionpharma/gflownet
- Entry [046] — the fragment-cap measurement this entry's deviation rests on.
- Entry [030] — the matrix campaign FragGFN is leaving.

**Packages**

- `gflownet` (clone in `external/gflownet`), used by
  `validation/generators/fraggfn/run_fraggfn_fixed.py`. `algo/config.py` supplies the
  `num_from_policy` default; `tasks/seh_frag.py` supplies the authors' task settings.

## Method

1. Read the authors' own settings out of the upstream clone rather than from our config comments:
   `grep -nE "num_from_policy|max_nodes" external/gflownet/src/gflownet/tasks/seh_frag.py`
2. Wrote three normalized configs at 157 steps x 64 = 10,048 scoring calls, `n_samples` 2000,
   `max_nodes` 6, with `num_from_policy` pinned.
3. Added the reward-tracing wrapper and timing output to the adapter; confirmed on a 12-step run that
   the trace held exactly 12 x 64 = 768 rows.
4. Ran one full cell to measure cost before committing to nine:
   `CELLS="fraggfn:seh:42" OUT_ROOT=<smoke dir> sbatch -p compute experiments/fixed_reward/scale5k/submit_baseline_chain.sh`
5. Queued the retrains — six surrogate cells chained in one job, each docking cell in its own:
   `CELLS="fraggfn:seh:42 ... fraggfn:drd2:44" sbatch -t 04:00:00 ...`
   `CELLS="fraggfn:clpp:42" sbatch -t 1-12:00:00 ...`
6. Verified Stage-2 readiness on an isolated copy of a finished cell, re-invoking the runner with a
   larger `--n-samples` and checking that the history survived, training was skipped, and the harvest
   read across the rotation.

## Results

**Budget.** All nine cells recorded exactly **10,048** scoring calls (157 x 64), matching REINVENT
(157 x 64) and S3-GFN (157 steps). The previous configs ran 5,000 steps = **320,000**, i.e. 32x.
Repeated molecules per cell ran 435-677, a real property of the generator and the sharpest signal
that further sampling is returning nothing new.

**Cost, measured rather than estimated.**

| cell type | wall-clock | note |
|---|---|---|
| surrogate (sEH, DRD2) | 3 min 46 s | six cells finished in ~23 min in one job |
| docking (ClpP) | 2 h 07 | 11,370 molecules, 167 batched requests, **0.65 s/molecule** |

**Diverse molecules already available**, from the training history alone at the 5%-false-positive
gates, before any further sampling:

| target | seed 42 | seed 43 | seed 44 | needs more? |
|---|---|---|---|---|
| sEH | 982 | 1283 | 1283 | no |
| ClpP | 668 | 616 | 572 | no |
| DRD2 | 376 | 426 | 435 | yes |

Six of nine already exceed the 500 target at zero additional cost.

**Stage-2 readiness**, verified on an isolated copy (job 75151): the previous history was preserved
rather than overwritten; training was skipped (`already trained 157 >= 157`); 3,000 candidates were
produced where the cell had 2,000; and the harvest read 9,481 distinct molecules across both rounds.

**Two measurement mistakes made and corrected.** The overage was first reported as 64x, from reading
the frozen proxy's inference batch (128) instead of the training batch (64) — the real figure is 32x,
and the training batch was already the authors' value. The docking cells were first sized at ~17 h by
borrowing Saturn's 5.1 s/molecule; FragGFN's molecules are much smaller and dock at 0.65 s/molecule,
so the true cost is 2 h. Both are now pinned in config rather than inferred.

**Not resolved.** RGFN, SCENT and RxnFlow remain on the matrix's 5,000-step convention. Whether that
amounts to hundreds of thousands of scoring calls against the competitors' ~10,000 has not been
measured; it is inferred from configuration files only.
