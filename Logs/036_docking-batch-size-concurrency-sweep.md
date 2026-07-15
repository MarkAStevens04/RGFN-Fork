# QuickVina2-GPU — docking batch-size sweep & 1-vs-2 concurrent-process comparison
**Date:** 2026-07-14, ~6pm

## Question

For our GPU docking, how many molecules should we hand to QuickVina2-GPU per
invocation, and would running two docking processes at once on the same A100 get
us more molecules per second — or is one process already using the whole card?

## Context & Summary

**Context.** Every live-docking cell of the 16-cell publication run (entry
`030`) scores molecules with **QuickVina2-GPU-2.1** on a single 40 GB A100, and
docking is the wall-clock bottleneck (the persistent docking server runs ~95%
busy). Two throughput knobs have never been measured on our hardware: (1) the
**batch size** — how many SMILES go into one QuickVina2-GPU process. The
single-target oracle (ClpP/sEH) currently chunks the per-step batch into groups
of **25**, so a ~200-molecule step spawns ~8 separate docking processes, each
re-paying the fixed OpenCL-context + receptor-grid setup; the 6TD3 differential
oracle instead docks the whole call in one process. (2) whether a single
QuickVina2-GPU process (thread=8000 search lanes) actually saturates the A100, or
whether **two concurrent processes** would raise aggregate throughput.

**Summary.** On one dedicated A100 we (A) re-dock a fixed pool of molecules at
batch sizes 25/50/100/200 for both live systems (ClpP single-target, 6TD3
differential) and record seconds/molecule and GPU utilisation, and (B) dock a
fixed ligand set through one QuickVina2-GPU process versus two concurrent
processes on the same card, comparing aggregate throughput with `nvidia-smi`
utilisation sampled throughout.

## Answer

Batch size is a large, essentially **free** throughput win for the single-target
oracle: docking a 200-molecule step as **one** QuickVina2-GPU process instead of
eight 25-molecule processes cut ClpP docking from **1.31 → 0.40 s/mol (3.3×
faster)**, and sEH rides the same code path. The gain costs no extra GPU memory —
a QuickVina2-GPU process holds ~20 GB regardless of batch size, because ligands
are docked sequentially inside it, so a bigger batch simply amortises the
per-process OpenCL-context + receptor-grid setup over more molecules. The **6TD3**
differential oracle already docks the whole step in one process (its `score_detailed`
ignores `docking_batch_size`), so it already sits at the fast end (**0.63 s/mol**)
and needs no change. Running **two** QuickVina2-GPU processes on one A100 gave **no
throughput gain (1.01× ClpP, 1.07× 6TD3)** and used **~39.9 GB of the 40 GB card**,
so a single process at `thread=8000` already exhausts the card's useful capacity —
concurrency is not worth it (and would OOM with any larger ligand or box).

## Relevance to our Publication

A NeurIPS/methods reviewer (and our own compute budget) cares that the docking
oracle — the run's bottleneck — is run efficiently. This entry either justifies
the current settings or gives a measured, one-line config change that buys back
wall-clock across the whole 4×4 matrix, and documents that the batch/concurrency
choice was made on data rather than left at an inherited default.

## Next Experiments

**Refining for publication**
- Fold `docking_batch_size = 200` (i.e. ≥ the per-step sample count) into the
  single-target oracle defaults (`DockingSEHOracle`/`DockingClpPOracle`) and/or
  the docking-server launch, for the **next** run — not mid-campaign. Scores are
  batch-invariant (n_ok was 199/200 at every batch size — one molecule fails prep
  regardless), so this only changes wall-clock, not results. Do NOT change it on
  the live 16-cell run (entry `030`): a checkpoint-resume would re-read the new
  default and split the run across two settings.
- GPU utilisation stays **< 50 %** even at batch 200 / single process, yet a
  second concurrent process adds no throughput. The remaining ceiling is therefore
  per-ligand host/dispatch overhead (OpenCL launch + pose I/O), not raw GPU
  compute. A future lever would be pipelining Meeko prep against docking or cutting
  per-ligand I/O — **not** a bigger batch or naive concurrency.

**Next steps in project**
- Let the 16-cell run (entry `030`) finish on its current settings; apply the
  batch fix to any future docking cells / active-learning runs.

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**
- `./experiments/fixed_reward/docking_benchmark/bench_batch_concurrency.py` —
  the benchmark: Part A batch-size sweep (end-to-end oracle) + Part B 1-vs-2
  concurrent raw-QuickVina2-GPU comparison, with GPU-utilisation sampling.
- `./experiments/fixed_reward/docking_benchmark/submit_batch_concurrency.sh` —
  Balam SLURM submit (1 GPU, `compute`, `--exclude=balam008`, OpenCL health gate).

**Datasets**
- `./experiments/active_learning/6td3/seed_6td3.csv` — SMILES pool (408 mols) used
  as the docking input for both parts.
- `./data/targets/ClpP.pdbqt` — ClpP receptor (single-target oracle).
- `./experiments/oracle_validation/docking_6td3/6TD3_tier{1,2}.pdbqt`,
  `crystal_RC8.pdb` — 6TD3 differential receptors + autobox crystal ligand.

**Results**
- `./experiments/fixed_reward/docking_benchmark/results/70623/benchmark_results.json`
  — per-config s/mol, mol/s, n_ok, and sampled GPU utilisation/memory (copied back
  from `/scratch/markymoo/rgfn_runs/docking_benchmark/70623/`).
- `./experiments/fixed_reward/docking_benchmark/results/70623/console.out` — full run log.

**Job Logs**
- `/scratch/markymoo/rgfn_runs/fr_batch_bench-70623.{out,err}` — SLURM job 70623
  (partition `debug`, node balam001, 34 min, ExitCode 0:0).

## Relevant Versions

NOT yet committed (branch Hub-Analysis). To commit:
`experiments/fixed_reward/docking_benchmark/{bench_batch_concurrency.py,submit_batch_concurrency.sh,results/70623/}`
and this log. [TODO — add commit hash after pushing.]

## Relevant Resources

**Sources**
- Tang et al., *Vina-GPU 2.1* (bioRxiv 2023.11.04.565429) — QuickVina2-GPU-2.1;
  recommended `thread` ≈ 5000, cap < 10000.
- DeltaGroupNJUPT/Vina-GPU-2.1 README — `thread`/`ligand_directory` parameters.

**Packages**
- QuickVina2-GPU-2.1 (`$SCRATCH/vina_gpu/Vina-GPU-2.1`, binary
  `QuickVina2-GPU-2-1`) — the GPU docking engine under test.
- gnina (`$GNINA`) — CNN pose-selection / `--score_only` for the 6TD3 differential.

## Method

1. `sbatch experiments/fixed_reward/docking_benchmark/submit_batch_concurrency.sh`
   — 1 GPU, partition `debug` (`--exclude=balam008`), OpenCL health gate, `rgfn` env.
   Ran job 70623 on balam001 (A100-PCIE-40 GB).
2. The job ran `bench_batch_concurrency.py --targets clpp 6td3 --n 200
   --batch-sizes 25 50 100 200 --concurrency-n 96 --repeats 2`.
   * **Part A**: for each target, build the live oracle once, warm up, then re-dock
     the same 200-molecule pool (`seed_6td3.csv`) at each batch size. ClpP = mutate
     `DockingClpPOracle.docking_batch_size` (its `score()` chunks by it); 6TD3 =
     feed `Docking6TD3GpuOracle.score_detailed` calls of size `b` (one QV2 process
     each). GPU util sampled with `nvidia-smi` every 0.25 s during each config.
   * **Part B**: Meeko-prepare 96 ligands once, then dock them at the raw
     QuickVina2-GPU level (reusing the oracle's receptor/box/`thread=8000`/
     `num_modes`) as (a) 1 process over all 96 and (b) 2 processes of 48 launched
     concurrently on `CUDA_VISIBLE_DEVICES=0`; compare aggregate lig/s.
3. Copied `benchmark_results.json` + console log into
   `experiments/fixed_reward/docking_benchmark/results/70623/`.

## Results

**Part A — throughput vs QuickVina2-GPU batch size** (200-molecule pool; each row
docks the SAME molecules, only the process count changes; VRAM ~20 GB at every
batch — ligands dock sequentially within a process so batch size is memory-free;
`n_ok` = 199/200 at every batch, confirming scores are batch-invariant).

| batch | QV2 procs | ClpP s/mol | ClpP mol/s | ClpP GPU util (mean) | 6TD3 s/mol | 6TD3 mol/s |
|------:|----------:|-----------:|-----------:|---------------------:|-----------:|-----------:|
| 25    | 8         | 1.312      | 0.76       | 16 %                 | 1.750      | 0.57       |
| 50    | 4         | 0.779      | 1.28       | 26 %                 | 1.107      | 0.90       |
| 100   | 2         | 0.534      | 1.87       | 37 %                 | 0.785      | 1.27       |
| 200   | 1         | **0.398**  | **2.51**   | 49 %                 | **0.632**  | **1.58**   |

ClpP 25→200 = **3.3× faster** per molecule; 6TD3 25→200 = **2.77×**. In production
the single-target oracle chunks at **25** (leaving the 3.3× on the table); the 6TD3
oracle already docks the whole per-step call in one process (≈ the batch-200 row).

**Part B — 1 vs 2 concurrent QuickVina2-GPU processes on one A100** (`thread=8000`;
best of 2 repeats; all runs produced complete pose output).

| target | 1 process (lig/s) | 2 concurrent (lig/s) | speed-up | dual VRAM | verdict |
|--------|------------------:|---------------------:|---------:|----------:|---------|
| ClpP (box 17³, modes 1)          | 1.86 | 1.87 | **1.01×** | ~39.9 GB | saturated, no gain |
| 6TD3 (autobox ~19×15×22, modes 9)| 1.65 | 1.76 | **1.07×** | ~39.9 GB | saturated, no gain |

A single QuickVina2-GPU process (~20 GB) already exhausts the card's useful docking
throughput; a second process only pushes VRAM to ~39.9 GB of 40 GB for ~no extra
molecules/sec (and any larger ligand/box would OOM).
