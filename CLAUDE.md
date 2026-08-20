# CLAUDE.md — repository guide for AI agents and contributors

Do not include a Co-Authored-By line in commit messages.

This is a research fork of **RGFN** (Reaction-GFlowNet) for generating novel
**molecular glue degraders**. We extend upstream RGFN with new oracles, reward
shaping, batch-selection strategies, benchmarks, and dataset tooling.

**Read this file and `docs/ARCHITECTURE.md` before making structural changes.**
For the science/goals, read `docs/RESEARCH_CONTEXT.md`.

---

## The one rule that governs everything: old vs. new

| Path | Owner | Rule |
|---|---|---|
| `rgfn/` | **Upstream** | Pristine RGFN. **Do not edit.** Extend it from `glue/`. |
| `configs/` (except `configs/glue/`) | **Upstream** | Treat as pristine. New configs go in `configs/glue/`. |
| `gin_config/`, `train.py`, `grid_search.py`, `tests/` (upstream parts), `data/chemistry.xlsx`, `data/targets/`, `external/setup_{gnina,gneprop,reinvent,shared}.sh` | **Upstream** | Leave as-is. |
| `glue/` (**production pipeline**) | **Ours** | All new pipeline Python: oracles, rewards, samplers, proxies, datasets, the active-learning loop. The thing we ship. |
| `configs/glue/` | **Ours** | All new gin configs (overlay upstream via `include`). |
| `scripts/` | **Ours** | **Generic** launch layer only: pipeline-wide entry points (`train.py`/`infer.py` wrappers, `active_learning.py`) + generic `submit.sh`. Experiment-specific scripts go in `experiments/<group>/<run>/`, not here. |
| `validation/` (**validation**) | **Ours** | The whole comparative-evaluation world: baseline generators (SynFlowNet, FragGFN, VAE-BO, RGFN adapter), validation-only oracles (Boltz-2), benchmark suites (PMO + our own), the harness, and committed results. |
| `data/` | **Ours** (+ upstream files) | The single **inputs** dir. Ours: `data/models/` (structures/checkpoints), `data/validation-molecules/` (curated known-glue sets), `data/synthetic/` (generated, git-ignored). Upstream (don't move): `data/chemistry.xlsx`, `data/targets/`. |
| `experiments/` | **Ours** | One self-contained dir **per run/experiment**, grouped by type: `active_learning/`, `oracle_validation/`, `ablations/`. Holds that run's code + seeds + small results + README; timestamped run outputs land alongside (git-ignored). Reusable science graduates into `glue/`. See `experiments/README.md`. |
| `Logs/`, `docs/` | **Ours** | Experiment logs + project documentation. |
| `Logs/references/` | **Ours** | Canonical bibliography for papers we build on. Cite by key (`[koziarski2024rgfn]`); annotated index in its `README.md`. PDFs in `pdfs/` are git-ignored. |

We keep `rgfn/` and `configs/` mergeable with upstream. The only deliberate edits
to upstream files are three small operational overrides documented in
**`docs/PATCHES.md`** — read that before "fixing" anything odd in those files.

### The second axis: production pipeline vs. validation

Beyond upstream-vs-ours, our own code splits along a second axis — the **training
pipeline** (`glue/`) vs. the **validation/benchmarking layer** (`validation/`) —
kept apart by one rule:

> **The dependency arrow points one way.** `validation/` may import from `glue/`
> and `rgfn/`. The production pipeline — `glue/`, `scripts/train.py`,
> `configs/glue/` — must **never** import from `validation/`.

This is what keeps slow validation-only oracles (Boltz-2, co-folding, MD) out of
the in-loop reward by construction, and keeps the shipped pipeline understandable
without dragging in every baseline. Everything comparative lives under the single
`validation/` umbrella (generators, oracles, `suites/`, `harness/`, `results/`);
baseline generators are **thin adapters** in `validation/generators/`, their heavy
upstream code installed via `external/setup_*.sh`, not vendored. See
`validation/README.md` and `docs/ARCHITECTURE.md` for the full picture.

---

## How to extend the system (don't edit `rgfn/`)

0. **Understand before you change.** Ground the work in the science *first*: read
   `docs/RESEARCH_CONTEXT.md` (goals, the validated oracle/metric, terminology),
   the relevant experiment logs in `Logs/`, and the source papers in
   `Logs/references/`. Base each consequential choice — architecture, metric, sign
   convention — on the file that *recorded* the decision (paper section, upstream
   `rgfn/` code, or our own log/analysis script), **never** on what merely seems
   plausible. Confirm the change is consistent with prior results and the
   project's direction, and **ask the user to clarify** anything ambiguous before
   building. A design that looks reasonable but silently diverges from the
   publications or our findings is the costliest mistake here — and it passes
   compile/import/smoke tests, so only this step catches it.
1. **New oracle** → implement the science in `glue/oracles/`.
2. **Expose it to training** → write a `@gin.configurable` adapter in
   `glue/proxies/` subclassing upstream `CachedProxyBase`/`ProxyBase`. See
   `glue/proxies/example_glue_proxy.py` for the working template.
3. **New reward / sampler** → `glue/rewards/` or `glue/samplers/`, subclassing the
   upstream base classes in `rgfn/api/`.
4. **Register it** → make sure the module is imported by `glue/registry.py`
   (directly or via its subpackage `__init__`). Gin finds classes by name only
   after they are imported.
5. **Configure it** → add a gin config in `configs/glue/` that `include`s the
   upstream base and references your class as `@YourClass`.
6. **Run it** → `python scripts/train.py --cfg configs/glue/<cfg>.gin`. The
   wrapper imports `glue` first so gin can resolve your component; the root
   `train.py` (upstream) only knows about `rgfn`.

If gin says *"No configurable matching @X"*, the defining module wasn't imported
on the startup path — fix `glue/registry.py`, not the config.

---

## Project goals (what we're building toward)

- New **oracles** (ternary docking / neosubstrate differential; possibly MD).
- New **rewards** (differential reward isolating glue cooperativity).
- Possibly expanded **batch selection**.
- **Benchmarks** comparing RGFN vs. baselines and across protein systems.
- **Models & datasets** as input; **synthetic datasets** as output.

---

## ⛔ THE BENCHMARK'S PRIMARY READOUT IS A FIXED **REACTION** BUDGET (decided 2026-08-17)

**Default to a fixed reaction budget of 100 reactions. Do NOT default to a fixed mode target.**

> **"I have 100 reactions. How many distinct high-reward molecules do I get?"**
> — not "I need 100 modes, what do they cost?"

Both stopping conditions are read off the *same* ordering at read time
(`docs/LSD_FLOW_BENCHMARK_PLAN.md` §0), so this is a reporting convention, not a different experiment.
`sweep_campaign.py` already emits both: `pareto.csv` (`pareto_modes_at_R`, **use this**) and
`fixed_modes.csv` (`fixed_modes_reactions_at_M`, secondary).

**Why reactions, not modes:**
1. It is what the benchmark plan designated as the headline all along (`--rxn-budget 100`); the drift to
   a 100-*mode* readout began with Logs/056 and was never a decision.
2. **A reaction budget turns an EXCLUSION into a FLAGGED DATAPOINT** — it does *not* remove the
   failure mode, and must not be described as doing so. A mode target excludes any pool that cannot
   reach the target ("pool-limited"): Saturn on all 6 cells (18–45 modes at N=500), two sEH cells at
   bar 7.0. A reaction budget still yields a number for those cells, but the number is only
   *comparable* if the budget was actually spent. **Always report `used_rxns` beside the mode count**,
   and classify every cell:
   - **budget-binding** — qualifying candidates remain, but the cheapest next mode does not fit in the
     remaining budget. This is the only like-for-like case. (You can rarely land exactly on R, so
     "exhausted" can never mean `used == R`; it means *the next step would exceed R*.)
   - **pool-exhausted** — nothing qualifying is left to add. Note the SB arm does not merely leave
     the budget unspent here: its reported `used_rxns` **inflates**, climbing (non-monotonically) with
     the budget while the true cost of the identical selected set stays flat — measured 2026-08-20 on
     Saturn's pruned sEH cell, 65 molecules priced at 247 reactions while `used_rxns` read 300→387
     across R=300…1000, i.e. up to 140 reactions of slack. So **outside the budget-binding regime
     quote `cost_kept_rxns`, not `used_rxns`**; the gap between them is itself the exhaustion
     detector. Reporting
     such a cell as "modes at R reactions" implies it could have spent R and chose not to, which is
     false. Flag it (hatch it, as the two-knob surfaces already do) and never count it as a win or a
     loss on cost. Report `n_modes_available` too: it separates *collapse* (Saturn — only 18 modes
     exist in the pool) from mere *redundancy* (REINVENT — 171 modes among 459 routed molecules).
   - **solver-truncated (SB arm only)** — `used_rxns < budget` because CBC ran out of time, not
     because the pool or the budget bound. This is COMPUTE-limited and is a completely different
     claim; conflating it with pool-exhausted would read a solver failure as a property of the
     generator. Check `time_capped` / `milp_status` before interpreting any short SB row: Logs/062
     found **25 of 36** points still unconverged at a 2 h cap, and CBC reports `Optimal` for whatever
     it happens to hold when the limit stops it, so status alone is not enough — the frontier detects
     this by wall-clock. A truncated row is a LOWER BOUND on the competitor, i.e. it flatters us.

   This is the same discipline the project already applies on the mode axis — the handoff's §5.2b
   only claims like-for-like where "**all twelve arms reached the full 300-mode budget**" and counts
   35 of 42 gate points as strictly comparable. Carry it across, do not drop it.
   The data is already there: `sparrow_select_frontier.py` records `used_rxns` against `budget_rxns`
   and prints `N modes available from M routed molecules`; the greedy CSV carries `used_rxns`.
3. It is the constraint a chemist actually has: a budget, not a shopping list.
4. **The competitor's MILP converges at 100 where it does not at 200–300.** Measured on S3-GFN at a
   1800 s cap: R=50/100/400/1000 solved in 6–57 s and returned `Optimal`, while **R=200 and R=300 both
   hit the 1800 s wall**. So 100 is the budget most likely to yield a certified optimum rather than a
   lower bound — but **tractability is POOL-dependent, not budget-dependent, so this is not a
   guarantee**. REINVENT's sEH network needs 5,597 intermediates against S3-GFN's 2,943 for the same
   ~455 targets, and its R=50 ran >30 min of CBC without returning. Read `time_capped` on every SB row
   before calling it optimal; a capped row is a lower bound on the competitor, i.e. it flatters us.
5. ~100 reactions is about one plate — a real bench anchor.

**Numbers already published at the 100-MODE readout are not wrong, they are the secondary readout.**
Re-read them on the reaction axis before quoting (131 / 264 / 434 reactions-for-100-modes become
modes-at-100-reactions). Do not mix the two axes in one table or figure — a co-agent already lost a
result that way (`n_modes` vs `n_modes_kept`, see the shared-tree drift note in Logs/062 §Method 4).

**One trap when re-slicing an existing solve to a smaller budget.** Trimming the most expensive modes
off a solved selection is *exact* for greedy and for our own hub-batching, because greedy mode
selection is prefix-stable — verified 2026-08-17 on REINVENT sEH seed 42: the first 100 modes are
identical whether drawn from the top-500 pool or all 1,860 candidates. It is **not** exact for SPARROW,
whose selection is jointly optimised over shared intermediates; trimming gives a feasible but
suboptimal set, i.e. a *lower bound on the competitor*, which flatters us. Re-solve SB, trim the rest.

## Working notes for agents

- **Compute:** Heavy stack (torch-geometric, openbabel, meeko, gnina) + GPU
  docking runs on **Balam** (SciNet). Balam may be down; when it is, work on a
  Mac laptop and **validate what you can locally** (imports, `py_compile`, gin
  config-include integrity, `bash -n`) — full train/dock validation happens on
  Balam. State clearly in your summary what you did vs. couldn't verify.
- **SLURM walltime limits (authoritative — `sinfo -o "%P %l"`, NOT the public SciNet
  page, which still says 24 h):** `compute` and `compute_full` **3 days**; `debug`
  **2 hours**; `debug_full_node` **1 hour**. Ask for what the work needs, not the
  max — short requests backfill sooner.
- **ANY job that docks must `source ~/bin/rgfn-smoke-env.sh` — batch jobs included,
  not just login smokes. Never hand-roll `LD_LIBRARY_PATH`.** QuickVina2-GPU links
  against `libboost_{program_options,system,filesystem}.so.1.83.0`, which live in
  `$SCRATCH/vina_gpu/boost/lib` and **not** in the conda env. Omit that path and
  `ldd` shows 3 unresolved libs, the docker cannot start **on any node**, and the
  failure surfaces as `Docking attempt #N failed on GPU 0` + all-`nan` — which reads
  exactly like a degraded GPU and has already been misdiagnosed as one (job 73370:
  balam006 blamed, excluded, and an auto-resubmit feature written, all for a missing
  `-L` path). `submit_docking_cell.sh` sources the helper; copy that, don't reinvent
  it. Diagnostic: `ldd $(find $SCRATCH/vina_gpu -name 'QuickVina2-GPU*' -perm -u+x | head -1)`
  — 3 "not found" means the environment, 0 means look at the hardware.
- **Login-node smoke tests (Balam *or* Trillium):** before any interactive smoke
  test that imports `glue`/`rgfn` (pulls in dgl) or runs the GPU docking oracle,
  prefix the command with `source ~/bin/rgfn-smoke-env.sh &&`. That one helper
  activates the `rgfn` env and sets the `LD_LIBRARY_PATH` (torch-bundled CUDA libs
  for dgl + QuickVina2-GPU boost libs) + `GNINA` — and works **unchanged on both
  login nodes** (Balam is SciNet-legacy; **Trillium** is an Alliance cluster where
  `module load cuda/11.8.0` does not exist). Balam and Trillium **share
  `/scratch` + `/home`**, so the conda envs and `$SCRATCH/vina_gpu`/`gnina` builds
  are identical from either. **Jobs still submit to Balam compute only** — the
  `submit_*.sh` headers are Balam-specific (SLURM account/partition/`--exclude`)
  and the helper does not touch them.
- **Document as you go:** record structural changes and anything left unverified
  in `docs/REFACTOR_LOG.md` so the next agent can continue or repair the work.
- **Experiment logs:** use the `experiment-log` skill for any real computational
  experiment.
