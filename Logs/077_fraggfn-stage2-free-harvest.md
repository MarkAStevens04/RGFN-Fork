# FragGFN — Stage 2: how much of the 500-molecule pool was already paid for?

**Date:** 2026-08-28, ~6pm

## Question

When we need 500 distinct, structurally-varied high-scoring molecules out of a trained generator,
how many do we already have from training, and how many *extra* scoring calls do we have to spend to
get the rest?

## Context & Summary

The benchmark compares our hub-batching selection against SPARROW on a fixed budget of 100
reactions. SPARROW is a *selector*: it only looks good when it has a large, varied pool to choose
from, so before the comparison is fair every entrant has to be brought up to the same pool size —
500 "modes" (molecules that clear the target's hit gate **and** are structurally unlike each other,
Tanimoto τ = 0.5). That is Stage 2 of the four-stage pipeline (Train → **Upsample & Filter** →
Retrosynthesis → Selection).

Stage 2 costs scoring calls that the competitor's own published protocol never had to pay, and the
project's position is that this is an **honest limitation to report, not a cost to match** — the end
deliverable is still modes-at-100-reactions. So the number that matters is: how much did each cell
actually cost? The key lever is that `trace.csv` records every molecule the oracle scored *during
training*, all of it already paid for; entry 075 showed pool size, not pipeline quality, had been
driving earlier S3-GFN comparisons, which is why this stage exists at all.

FragGFN joined the competitor block late (it is a fragment-based GFlowNet, in the same non-reaction
family as S3-GFN) and had just been re-trained at a normalized budget of 10,048 scoring calls. This
entry runs Stage 2 on all nine of its cells — three targets (sEH, DRD2, ClpP) × three seeds — and
measures the free-versus-paid split.

## Answer

**Six of nine FragGFN cells reached the full 500-mode pool from their training history alone, at zero
additional scoring calls** (five confirmed by a completed Stage-2 run, the sixth measured from its
finished training trace with its job still queued). Only DRD2 needed to sample, and it needed one short round per seed —
about 2,000 extra calls, a ~20% surcharge on the 10,048-call training budget, in 1.3–1.4 minutes.
Harvesting the training trace, rather than throwing it away and drawing a fresh sample, is what
makes Stage 2 nearly free for this generator; on ClpP, where every scoring call is a GPU docking,
that is the difference between zero and roughly 28 GPU-hours per cell.

The split is a property of the target, not of the generator's competence: sEH and ClpP training
histories are rich in gate-clearing chemistry (990–1,303 and 572–673 molecules above the bar), while
DRD2's is thin (376–435) and needed topping up.

## Relevance to our Publication

The ICLR submission has to state plainly what the SPARROW comparison cost us, because a reviewer
will ask whether we bought our pool advantage with extra oracle calls the baseline never spent. This
entry gives that number for FragGFN — 5,992 calls across nine cells, all of it on one target — and
shows it is small relative to training. It also closes the last gap in the competitor block: with
FragGFN wired in, all six entrants pass through one standardized Stage 2 with one gate source, one
τ, and one mode target, which is what makes the cross-generator table comparable at all.

## Next Experiments

**Refining for publication.** Report the Stage-2 surcharge for every generator in one table, not just
FragGFN, so the limitation is visible per-cell rather than as an aggregate. The per-cell
`upsample_log.json` already carries `free_from_training`, `rounds`, and `total_sampling_seconds`, so
this is a reporting pass, not new compute. Also worth stating: modes drawn from the training history
come from earlier, weaker policies, so the reward profile of the selected set shifts slightly
relative to a final-policy-only pool — modes are chosen best-reward-first, which bounds the risk, but
the shift belongs in the violin plot.

**Next steps in project.** Push these nine pools through Stage 3 (retrosynthesis) and Stage 4
(selection) to get FragGFN's modes-at-100-reactions number. FragGFN has no native reaction routes, so
it takes the same route-less path as S3-GFN; the existing route script already handles that.

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork` (worktree `.claude/worktrees/fraggfn-stage2`,
branch `worktree-fraggfn-stage2`).

**Scripts**
- `./experiments/lsd_hubs/campaign/upsample_to_modes.py` — Stage 2 itself: harvests the training
  trace, counts modes, and samples further rounds only if the target is not already met.
- `./experiments/lsd_hubs/campaign/submit_stage2_upsample.sh` — the SLURM launcher; maps a
  `GENERATOR:TARGET:SEED` cell to its conda env, runner, and config, and (new here) starts a
  persistent docking server for docking targets.
- `./validation/generators/fraggfn/run_fraggfn_fixed.py` — FragGFN's runner. Its resume guard,
  `remaining = n_train_steps - loop._it`, is what makes the config choice load-bearing.
- `./experiments/lsd_hubs/matrix16/targets.py` — the single source of truth for each target's hit
  gate and score direction.

**Models / inputs**
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/fraggfn_{seh,drd2,clpp}/seed{42,43,44}/` —
  the nine trained cells: `checkpoints/last_gfn.pt`, the `trace.csv` training history, and the
  budget-faithful 2,000-molecule `fixed_reward/candidates/candidates.csv`.
- `./validation/configs/fraggfn_{seh,drd2}_fixed_norm.yaml`,
  `./validation/configs/fraggfn_clpp_docking_fixed_norm.yaml` — the normalized-budget configs the
  cells were actually trained on (157 steps × 64 = 10,048 scoring calls).

**Results**
- `/scratch/markymoo/rgfn_runs/stage2/fraggfn_{target}_seed{seed}/` — per cell:
  `stage2_candidates.csv` (the Stage-3 input), `modes.smi` (the 500 representatives), and
  `upsample_log.json` (the cost accounting).

**Job Logs**
- `/scratch/markymoo/rgfn_runs/stage2_fg_free-75154.out` — sEH ×3 + ClpP seed 42.
- `/scratch/markymoo/rgfn_runs/stage2_fg_drd2-75155.out` — DRD2 ×3.
- `/scratch/markymoo/rgfn_runs/stage2_fg_clpp43-75158.out`,
  `/scratch/markymoo/rgfn_runs/stage2_fg_clpp44-75159.out` — the two ClpP cells whose Stage 1
  finished last.

## Relevant Versions

```
a696e92 Name fraggfn in the route script's usage line; it was already agnostic, the help text was not
fe3762e Stage 2 can dock: a ClpP cell that needs sampling no longer falls back to a subprocess per step
cc54a06 The Stage-2 launcher can reach FragGFN, and cannot reach its stale 5,000-step config
7b8b3eb FragGFN is Stage-2 ready: it has a sampler, so it can be upsampled; SynFormer cannot
a327c3a FragGFN joins the competitor block, at 1/32nd of the budget it was training on
```

Branch `worktree-fraggfn-stage2`, not yet merged to `Hub-Analysis`.

**One version caveat that changes the numbers.** The 5%-FPR hit gates (sEH 5.68, DRD2 0.345, ClpP
−9.1) existed only as an **uncommitted** edit to `experiments/lsd_hubs/matrix16/targets.py` in the
shared checkout while this ran; `Hub-Analysis` HEAD still carried the old 5.0 / 0.5 / −8.0. The
launcher `cd`s to the shared checkout, so the jobs used the correct gates — but a first pass of this
analysis, run inside the worktree, silently used the old ones and reported 8 of 9 cells free instead
of 6. Any re-run must confirm which `targets.py` was on the import path before quoting a mode count.

## Relevant Resources

**Sources**
- `[bengio2021gflownet]` — GFlowNets sample proportional to reward, which is why a trained model's
  history contains many distinct high-reward molecules rather than one optimum.
- `docs/LSD_FLOW_BENCHMARK_PLAN.md` §0 — the fixed-reaction-budget readout Stage 2 feeds.
- Entry 075 — pool size, not pipeline quality, drove earlier S3-GFN comparisons.

**Packages**
- RDKit (Morgan r=3, 2048-bit fingerprints) via
  `validation/lsdflow/metrics/diversity.py::mode_representatives`, the single mode implementation all
  campaign scripts import.
- QuickVina2-GPU + gnina, via `glue/oracles/docking_server.py`, for the ClpP docking oracle.

## Method

1. Wired `fraggfn` into `submit_stage2_upsample.sh` (env, runner, and an explicit per-target config
   map). The config map is not cosmetic: the shared `${GEN}_${TGT}_fixed.yaml` convention resolves to
   `fraggfn_seh_fixed.yaml`, a file that **exists and is the old 5,000-step build**, so the resume
   guard would have read `157 < 5000` and silently re-trained 4,843 steps inside a sampling stage.
   Resolution was simulated for all three targets plus an S3-GFN control before anything was
   launched.
2. Measured the free harvest read-only, ahead of any job: loaded each cell's `trace.csv` through the
   campaign's own `load_trace` / `count_modes` / `gate_for` and counted modes at τ = 0.5.
3. Ran Stage 2 on the seven cells whose Stage 1 had completed, as two jobs —
   `CELLS="fraggfn:seh:42 fraggfn:seh:43 fraggfn:seh:44 fraggfn:clpp:42"` (job 75154) and
   `CELLS="fraggfn:drd2:42 fraggfn:drd2:43 fraggfn:drd2:44"` (job 75155). The two ClpP cells still
   training were deliberately held back: Stage 2 rotates `trace.csv` when it samples, and doing that
   under a live writer is how `s3gfn_seh/seed43`'s history was lost.
4. Added a persistent docking server to the Stage-2 launcher (OpenCL health gate once per job in
   preflight; server started and torn down per cell), then ran the last two ClpP cells as jobs 75158
   and 75159, the latter with `--dependency=afterany` on its Stage-1 job.

## Results

Gate = the 5%-FPR standard, τ = 0.5, target 500 modes, cap 50,000 distinct (20,000 for ClpP).
"Free" = distinct molecules recovered from the training history. "Extra" = additional distinct
molecules scored by Stage-2 sampling, i.e. the surcharge.

| cell | gate | scored in training | above gate | modes from history | extra calls | stop reason |
|---|---|---|---|---|---|---|
| fraggfn/seh/42 | 5.68 | 9,481 | 990 | **500** | 0 | target-reached |
| fraggfn/seh/43 | 5.68 | 9,613 | 1,301 | **500** | 0 | target-reached |
| fraggfn/seh/44 | 5.68 | 9,486 | 1,303 | **500** | 0 | target-reached |
| fraggfn/drd2/42 | 0.345 | 9,579 | 376 | 376 | 1,998 | target-reached |
| fraggfn/drd2/43 | 0.345 | 9,572 | 427 | 426 | 1,996 | target-reached |
| fraggfn/drd2/44 | 0.345 | 9,512 | 435 | 435 | 1,998 | target-reached |
| fraggfn/clpp/42 | −9.1 | 9,364 | 673 | **500** | 0 | target-reached |
| fraggfn/clpp/43 | −9.1 | 9,369 | 618 | **500** | 0 | target-reached |
| fraggfn/clpp/44 | −9.1 | 9,249 | 572 | **500** \* | 0 \* | *Stage 2 queued (75159)* |

\* `clpp/44`'s free-mode count is measured — its training trace is complete (10,049 rows) and was
read through the same `load_trace`/`count_modes` path as every other row — but its Stage-2 job was
still held on a `--dependency` behind Stage 1 when this was written, so no `upsample_log.json` exists
for it yet. Treat that row as a measurement of the input, not a completed run.

**Eight of nine cells are confirmed at 500/500 modes, none pool-limited; the ninth is predicted free
from its completed trace.** Total Stage-2 surcharge across the eight: **5,992
scoring calls**, entirely on DRD2 (~2,000 per seed, ~20% of that cell's 10,048-call training budget),
in one sampling round of 1.3–1.4 minutes each. The four sEH and ClpP jobs completed in about two
minutes of wall-clock combined, since no sampling ran.

On DRD2 the number of gate-clearing molecules and the number of modes are nearly identical
(376→376, 427→426, 435→435): almost every molecule above the DRD2 gate is already structurally
unlike the others, so the diversity filter removes essentially nothing and the shortfall is a
shortage of gate-clearing chemistry, not redundancy. sEH is the opposite — 990–1,303 above the gate
collapse to 500 modes only because the cap stops the count.

**One prediction made here was wrong, and the correction matters for how partial traces are read.**
Midway through, `clpp/44` was still training and its partial trace showed a 4.25% gate-clearing rate
against its siblings' 7.2% and 6.6%, which projected to roughly 400 modes — i.e. the first cell that
would need docking-based sampling. Once its trace completed the rate had caught up to 6.2% (572 of
9,249) and it reached 500 modes free. **A gate-clearing rate measured mid-training is not an
estimate of the final rate**, because the policy is still improving; the projection should not have
been made from it.

Consequently the new docking-server path was exercised only in its start-and-teardown form (job
75158 logged `OpenCL health: OK`, `[server] docking server (oracle=docking_clpp)`, and
`[dock-server] ... listening`), never with a real docking round. **It remains unexercised under
load.**
