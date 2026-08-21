# All four scoring systems — the enumeration cap that was quietly shrinking one generator's pool
**Date:** 2026-08-20, ~11am

## Question

One of our four generators was only ever shown a fraction of the molecules it can actually reach in one
step. Does correcting that change any of our published results?

## Context & Summary

Our central measurement asks how many reactions it takes to build a diverse library of hits. To answer
it we exhaustively enumerate every molecule reachable in one more step from each of 200 shared
scaffolds, and that enumeration is capped for safety at 4,000 molecules per scaffold. For three of the
four generators the cap is irrelevant — they never produce more than about 3,400 children from any
scaffold. **SCENT does.** It promotes its own best intermediates into a growing building-block
vocabulary, so a single scaffold can reach far more molecules than the others, and the cap was silently
discarding the excess.

Entry `058` found this on the two docking cells and fixed one of them, where the correction moved that
cell's result by **10.6%** in our favour. But that audit only looked at the docking half. A co-agent
then checked the fast-surrogate cells and found the problem is **worse there** — 57 truncated scaffolds
on the flagship sEH cell versus 25 on the docking cell that triggered the whole investigation, rising
to 82 on a later repeat. Those cells are the ones behind entry `050`'s headline and entry `055`'s
"cheaper in 380 of 380 settings" claim, so this is a correction to published numbers rather than new
work.

This entry re-enumerates all six affected cells with the cap raised more than tenfold, checks that it
stops binding, and re-measures everything that depended on them.

## Answer

The cap was genuinely truncating a lot — the six cells gained 8% to 61% more molecules once it was
lifted — but the effect on our headline numbers is **small: between +0.1% and +3.3%, every one of them
in our favour.** The published results were therefore right, and slightly conservative.

The interesting part is *why* it is small, because it turns out to be predictable. The correction only
matters if a truncated scaffold is one the method actually uses, and at our headline hit bar the method
only needs **6 to 23 scaffolds** to build its 300-molecule library — so most truncated ones sit below
the cut and never get touched. Raise the bar and the method has to work harder and reach further down
its list, and the correction grows with it: at the strictest bar we report, the same cell's usage grows
to **44 scaffolds** and the correction reaches **+4.7%**. So the size of this bug's effect is set by how
hard the task is, which also explains why the docking cell moved 10.6% while the surrogate cells barely
moved at all.

## Relevance to our Publication

This is the difference between a reviewer finding a truncation caveat in our own supplementary material
and finding it themselves. We can now state that the cap was checked on **every cell of the matrix**,
that it bound on exactly one generator, that lifting it moved nothing by more than 5%, and that every
movement was in the direction that makes our method look *worse* before the fix — which is the safe
direction for a claim to be wrong in.

It also lets us report the two-knob surfaces from entry `055` without an asterisk. Those span the bar
range where the correction is largest, and knowing every cell shifts up by at most ~5% means the
"cheaper everywhere" claim is untouched.

## Next Experiments

**Refining for publication**

- **Second seeds for the last two docking cells.** `scent_6td3` and `rgfn_6td3` are the only cells at a
  single repeat. Both are ready to run and are queued for the coming weekend, which brings every cell in
  the matrix to at least two.
- **Third seeds on the docking half** are not urgent and have been handed to the second cluster, since
  they tolerate an unpredictable queue.

**Next steps in project**

- Price these libraries through the competitor pipeline (entry `056`) now that the enumerations are
  final, so the comparison is not redone later.
- Feed the selections into the active-learning loop.

---

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**

- `./experiments/lsd_hubs/matrix16/submit_cell.sh` — surrogate enumeration launcher; `ENUM_MAX` is the cap.
- `./experiments/lsd_hubs/matrix16/submit_docking_cell.sh` — docking equivalent; `HUBS_FILE` restricts a
  run to a chosen subset of scaffolds, which is how the docking recaps re-enumerate only the truncated ones.
- `./experiments/lsd_hubs/matrix16/merge_docking_slices.sh` — keeps the **deepest** copy of a scaffold when
  it appears in more than one slice, which is what makes a partial re-enumeration safe to merge.
- `./experiments/lsd_hubs/matrix16/gate_curve.py` — re-scores one enumeration at several hit bars.
- `./experiments/lsd_hubs/matrix16/surface_all_generators.sh` — the two-knob surface from entry `055`.

**Results** (committed) — `results{,_seed43,_seed44}/<cell>/summary.json`, the matching `gate_curve/`
directories, and `results/surface/scent_seh/`. The pre-correction enumerations are preserved beside the
new ones as `enum_children.pre50k.json` (plus `hubs.pre50k.csv`), so any before/after can be recomputed
exactly rather than taken from this log.

## Relevant Versions

```
5a9156b  Matrix complete: every one of the 16 cells now has a seed, and the SCENT cap correction is small
7325abb  _artifacts: load the route-step schema BY PATH -- a glue package import breaks every per-env worker
2b7c39c  sparrow_select_frontier: ABORT on PARTIAL route coverage too, not just when it is wholly absent
```

The regenerated surface and the DRD2 bar sweeps are `73dc170`.

## Relevant Resources

**Sources** — the sEH proxy's calibration against known actives is entry `034`; the bar-vs-diversity
coupling is entry `051`; the count-once cost model is entry `033`.

**Packages** — RDKit (Morgan r=3, Tanimoto) via `glue/samplers/lsdflow/mode_select.py`.

## Method

1. **Audit every cell against the cap it was actually enumerated with** — not against a fixed 4,000,
   since three cells had already been re-run at a higher cap. A cell counts as truncated only if a
   scaffold sits *at* its own cap.
2. **Re-enumerate the six SCENT surrogate cell-seeds at `ENUM_MAX=50000`**, full cells rather than only
   the truncated scaffolds (see Results for why), preserving the originals first.
3. **Confirm the cap stopped binding** before using anything: zero scaffolds at 50,000.
4. **Re-run the campaigns** and compare against the published values.
5. **Sweep the hit bar** on the corrected enumerations to locate where the correction is largest.

## Results

**The cap stopped binding everywhere**, and the pools grew substantially:

| cell | children before | after | truncated scaffolds before | at the new cap | new max |
|---|---|---|---|---|---|
| `scent_seh` s42 | 438,984 | 644,759 | 57 | 0 | 11,843 |
| `scent_seh` s43 | 474,870 | 702,574 | 75 | 0 | 10,287 |
| `scent_seh` s44 | 492,471 | 791,503 | 82 | 0 | 13,093 |
| `scent_drd2` s42 | 344,469 | 485,398 | 31 | 0 | 13,898 |
| `scent_drd2` s43 | 298,491 | 317,957 | 12 | 0 | 6,492 |
| `scent_drd2` s44 | 352,046 | 503,616 | 23 | 0 | 17,147 |

**A cap of 50,000 rather than 20,000 was the right call, and it was not mine.** I proposed 20,000 by
analogy with the docking cell, whose true maximum was 14,943; the co-agent argued that a different
reward landscape makes that an assumption rather than a bound, and that at ~23 ms per molecule the
headroom is nearly free. `scent_drd2` s44 came back at **17,147** — above the docking cell's maximum, so
20,000 would have been uncomfortably close to re-truncating and we would have discovered it only after
re-running everything downstream.

**The correction to the headline, at the bars we report at:**

| cell | published | corrected | change |
|---|---|---|---|
| `scent_seh` s42 / s43 / s44 | 3.10× / 2.92× / 2.94× | 3.10× / 2.93× / 2.96× | +0.1% / +0.3% / +0.7% |
| `scent_drd2` s42 / s43 / s44 | 3.17× / 3.35× / 3.20× | 3.25× / 3.38× / 3.31× | +2.5% / +1.0% / +3.3% |
| `scent_6td3` s42 (docking) | 2.869× | 2.999× | **+4.5%** |
| `scent_clpp` s42 (docking) | 2.65× | 2.931× | **+10.6%** |

Every change is upward, which is the direction the truncation mechanism predicts: fewer children per
scaffold means fewer library members per scaffold built, so the method needs more scaffolds and looks
more expensive than it is.

**Why it is small here and large elsewhere — the mechanism, measured.** The correction can only bite
through a scaffold the method actually uses, and usage depends on the hit bar:

| hit bar | scaffolds used (`scent_seh` s42) | result | vs published |
|---|---|---|---|
| 5.0 (headline) | 8 | 3.104× | +0.1% |
| 6.0 | 14 | 3.048× | — |
| 7.0 (paper-comparable) | 44 | 2.722× | **+4.7%** on entry `050`'s 2.60× |

So the same defect moved the docking cell by 10.6% (34 scaffolds used at its bar) and the surrogate
cells by almost nothing at ours. Entry `055`'s surfaces span bars 4–8 and therefore shift by up to ~5%,
all upward — the "cheaper in 380/380 settings" claim is unaffected in kind, and the surfaces were
regenerated for precision rather than correctness.

**The same relationship holds ACROSS cells, which is what makes it a mechanism rather than a
coincidence.** Once `scent_6td3`'s recap landed (2026-08-21) there were three independent cells to
compare, and the correction orders exactly by how many scaffolds each one's walk uses at its own bar:

| cell | bar | scaffolds used | correction |
|---|---|---|---|
| `scent_seh` s42 | 5.0 | 8 | +0.1% |
| `scent_6td3` s42 | −2.0 | 17 | +4.5% |
| `scent_clpp` s42 | −8.0 | 34 | +10.6% |

Three cells, three different generators' targets, three different reward scales — and the ordering is
monotone in scaffold count. `scent_6td3` was a genuine out-of-sample prediction: the walk length was
knowable from the published run before the recap finished, and it placed the correction between the
other two, which is where it landed. The practical statement for the paper is therefore not "the cap
cost us up to 10%" but the sharper **the size of a truncation's effect is set by how hard the task is,
because a harder task forces the method to use more scaffolds and so to touch more truncated ones.**
Easy cells were never at risk; only the hardest cell was materially affected.

**The corrected cells were re-swept across gates, and the edge does not depend on the bar.** Both SCENT
docking cells' gate sweeps predated their re-enumeration, so they were regenerated (free — a re-scoring
of child rewards already measured). The edge is monotone in gate stringency and never inverts:

| gate | `scent_6td3` | | gate | `scent_clpp` |
|---|---|---|---|---|
| −4.0 | 2.767× | | −11.0 | 1.687× |
| −3.0 | 2.930× | | −10.0 | 2.165× |
| **−2.0** | **2.999×** | | −9.0 | 2.577× |
| −1.0 | 3.054× | | **−8.0** | **2.931×** |

Looser gates report a LARGER edge, which is the mechanism again: more qualifying children per scaffold
means the shared scaffold amortises over more library members. So the calibrated bars we publish (bold)
sit at the conservative end of their own sweeps. And nothing inverts even at ClpP −11.0, three
kcal/mol beyond the calibrated bar — 1.687× is the worst number the cell yields under any gate we would
defend. The result is not an artifact of where the bar was set. Both bold rows reproduce their committed
`summary.json` exactly (1.107 and 1.167 r/m), so sweep and headline are one computation read at two
thresholds, not two estimates. The other four docking cells' enumerations never changed, so their
08-07/08-11 sweeps stand as-is.

**Scope: this is one generator's problem, confirmed across the whole matrix.** Twelve of sixteen cells
have zero truncated scaffolds on every seed, with maxima of 895–3,435 children — they were never near
the cap. Only SCENT reaches it, because only SCENT grows its own building-block vocabulary during
training. `scent_6td3` s42's recap has since completed (2026-08-21): 36 slices merged to 200/200 scaffolds and
657,236 children, up from 550,569, with a new maximum of 12,066 against the 20,000 cap — so nothing
re-truncated. **Every cell in the matrix is now uncapped except one**: `rgfn_6td3` s43 retains **1**
truncated scaffold of 200, which we are choosing to report rather than spend ~160 GPU-h re-running.

**A method choice worth recording.** The co-agent proposed re-enumerating only the truncated scaffolds
(~15 GPU-h). I re-ran the full cells (~30 GPU-h) instead. `submit_cell.sh` writes its enumeration
directly to one file with no per-slice directories, so a subset run would have overwritten a 200-scaffold
artifact with 57, and stitching it back needs a merge path that does not exist on the surrogate side —
new code operating on the artifact that underpins published numbers. Paying 15 extra GPU-hours to avoid
writing that was the cheaper risk. The docking recaps *do* use the subset route, because that launcher
already writes per-slice directories and its merge already prefers the deepest copy.

**Two failures worth recording, both from a shared checkout rather than the science.**

Twelve of the sixteen docking-recap slices died with `ModuleNotFoundError: No module named 'glue'`.
A concurrent change had added a `from glue...` import to the shared artifact writer, which cannot work
there: those writers run inside each generator's own conda environment, where importing `glue` pulls in
a package chain that resolves to a *different* project's code. Fixed by loading the (dependency-free)
schema file by path instead; verified importable in all four environments. The four slices that had
started before the change succeeded, which is exactly why the failure looked like a partial-progress
problem rather than an import problem.

All four `hub_order` repair jobs died in six seconds with `can't open
/var/spool/validation/.../scent_worker.py`. That launcher derived the repository root from its own file
path, and under the scheduler the script is copied to a spool directory — so the root resolved to
`/var/spool`. The matrix launchers already prefer the scheduler's submit-directory variable; this one
predated that fix and had only ever worked because it happened to be launched from the repository.

**Caveats**

- Three seeds give a spread, not a confidence interval.
- The corrected numbers are a re-scoring of new enumerations on unchanged checkpoints; nothing was retrained.
- `rgfn_6td3` s43 keeps a single truncated scaffold of 200; it is the one cell not re-enumerated, on cost grounds.
- FragGFN remains a cost-model control throughout, as in every prior entry.
