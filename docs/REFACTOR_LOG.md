# Refactor log — 2026-06-24 repo restructure

A running record of the restructure that introduced the `glue/` package and the
upstream/ours boundary. **If something is broken, start here**: this documents
what changed, why, what was verified, and what could not be verified (Balam was
down; work was done on a Mac laptop, so no GPU/docking/training runs were possible).

Owner: Mark + Claude. Goal context: `docs/RESEARCH_CONTEXT.md`. Layout: `docs/ARCHITECTURE.md`.

---

## Why

The repo had fused upstream RGFN with our additions at the top level with no clear
boundary. We separated them: `rgfn/` + `configs/` stay pristine/mergeable; all new
work lives in `glue/` and clearly-owned top-level dirs. See `CLAUDE.md`.

Key finding during planning: the upstream package was *already* ~pristine. The
proxies/docking code/configs that looked like ours are actually upstream (2024
commits). Our only edits to upstream files were **three operational overrides**,
now documented in `docs/PATCHES.md`. So this was mostly **additive scaffolding +
moving our own files**, not extracting code out of `rgfn/`.

---

## What changed

### Added (new, zero risk to upstream)
- `glue/` package: `oracles/`, `rewards/`, `samplers/`, `proxies/`, `datasets/`,
  plus `registry.py` (import hub for gin discovery) and `__init__.py`.
  - `glue/proxies/example_glue_proxy.py` — working template adapter (returns QED)
    showing the `CachedProxyBase` + `@gin.configurable` pattern.
- `configs/glue/README.md` — overlay pattern for new configs.
- `scripts/train.py`, `scripts/infer.py` — wrappers that `import glue` (so gin
  sees our components) then delegate to the root entry points via `runpy`.
- `scripts/README.md`, `scripts/hpc/` (holds `submit.sh`).
- `benchmarks/README.md`, `models/README.md` + `.gitkeep`,
  `data/synthetic/README.md` + `.gitkeep`.
- `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/PATCHES.md`, this file.

### Moved
- `pre-processing/`  →  `research/preprocessing/` (whole tree via `git mv`, so
  internal `__file__`-relative and `../test-data` paths are preserved).
- `submit.sh`  →  `scripts/hpc/submit.sh` (and its train line now calls
  `scripts/train.py`).

### Edited (path fixups caused by the moves)
- `research/preprocessing/docking_gnina/submit_dock_crbn.sh` — `$REPO/pre-processing/...`
  → `$REPO/research/preprocessing/...` (2 refs).
- `research/preprocessing/docking_6td3/submit_dock_6td3.sh` — same (2 refs).
- `research/preprocessing/docking_gnina/analysis/plot_discrimination_curves.py` —
  `ROOT` now `parents[4]`; `PRE = ROOT / "research" / "preprocessing"`.
- `docs/RESEARCH_CONTEXT.md` — "Where things live" updated + restructure note.
- `Logs/000_TEMPLATE.md` — `pre-processing/` → `research/preprocessing/` prefixes.
- `scripts/hpc/submit.sh` — train line → `python scripts/train.py ...`.

### Upstream files (deliberately left edited — see docs/PATCHES.md)
- `rgfn/trainer/logger/wandb_logger.py`, `configs/loggers/wandb.gin`,
  `configs/rgfn_seh_docking.gin`. Not reverted; documented as patches.

### Not changed
- `rgfn/` (apart from the documented wandb one-liner), `gin_config/`, upstream
  `configs/`, `train.py`, `grid_search.py`, upstream `data/` inputs, `tests/`.
- Historical experiment logs `Logs/001`–`005` keep their original
  `pre-processing/` paths as point-in-time records (noted in RESEARCH_CONTEXT).

---

## Validation

Environment: macOS laptop, **Balam down**. Could NOT run training, docking, or
anything needing the GPU/heavy deps. Verified what is checkable statically.

### Verified (static, on the laptop — 2026-06-24)
- **`py_compile`** of every `.py` under `glue/`, `scripts/`, `research/` → all compile.
- **No dangling `pre-processing` references** in any `*.py` / `*.sh` / `*.gin` /
  `*.toml` (only historical `Logs/001`–`005` still mention the old path, by design).
- **`bash -n`** on `scripts/hpc/submit.sh` and both moved `submit_dock_*.sh` → OK.
- **gin `include` integrity**: every `include '...'` in `configs/**/*.gin` resolves
  to an existing file.
- **glue import chain**: `glue/__init__` → `registry` → all 5 subpackages →
  `example_glue_proxy` all exist and reference real upstream symbols
  (`CachedProxyBase`, `ReactionStateEarlyTerminal`).
- **Upstream untouched this session**: `git status` shows no modifications to
  `rgfn/` or upstream `configs/` (only `configs/glue/` added). The 3 documented
  patches in `docs/PATCHES.md` are prior commits, not new edits.

### Could NOT verify locally (heavy deps not installed → do on Balam)
- `import glue` fails with `ModuleNotFoundError: No module named 'gin'` — this is
  an **environment** issue (gin-config/torch-geometric/etc. aren't installed on the
  laptop), **not a code issue**. Re-run `python -c "import glue"` in the `rgfn`
  conda env to confirm registration works.

### NOT verified (do on Balam when it's back)
- `python scripts/train.py --cfg configs/glue/<cfg>.gin` actually trains.
- `import glue` succeeds in the real env (depends on torch-geometric etc., which
  may not be installed on the laptop — a local import failure here is likely an
  environment issue, not a code issue; confirm on Balam).
- The moved docking scripts (`research/preprocessing/...`) run end-to-end with
  gnina on a full node.
- The example proxy produces a sensible reward in a real run.

---

## How to continue / repair

- **gin "No configurable matching @X":** the module defining `X` isn't imported on
  startup. Ensure it's imported by `glue/registry.py` (or its subpackage
  `__init__`), and that you launched via `scripts/train.py` (not root `train.py`).
- **A moved docking script can't find a file:** check it's reading relative to its
  own dir (`HERE`) or `../test-data`; repo-root-relative refs were updated to
  `research/preprocessing/` — grep for any stragglers: `grep -rn pre-processing .`
- **An upstream patch vanished after a merge:** re-apply from `docs/PATCHES.md`.
- **Adding new components:** follow the steps in `CLAUDE.md` ("How to extend").

---

# 2026-06-24 — Active-learning loop (branch `active-learning-loop`)

First implementation of the multi-round active-learning loop from
`[bengio2021gflownet]` Alg. 1 (transcribed in `docs/RESEARCH_CONTEXT.md`), wired
to our 6TD3 docking oracle and kept oracle-agnostic for future oracles (MD, etc.).

## Grounding decision (important — re-checked against the papers)

An earlier draft proposed a hand-rolled **MLP** proxy. That was wrong: the proxy
in `[bengio2021gflownet]` A.4 is an **MPNN over the RDKit atom graph** (NNConv +
GRU ×12 → Set2Set → 3-layer MLP, dim 64, LeakyReLU). RGFN already ships *exactly*
that network as `MPNNet` (the class behind `SehMoleculeProxy`, which loads
Bengio's pretrained sEH weights). Key point that resolves the "RGFN isn't
atom-by-atom" worry: RGFN's **reaction/fragment graph drives only the policy/flow
predictor**; the **proxy scores a finished molecule from its atom graph**
(`mol2graph(SMILES)`), identically in both `[bengio2021gflownet]` and RGFN's own
sEH proxy. So we **reuse `MPNNet` + `mol2graph`/`mols2batch` verbatim** rather
than reinventing an architecture.

**We reuse the architecture, NOT the weights.** `LearnedGlueProxy` never calls
`seh_proxy.load_original_model` / `SEHProxyWrapper` — it constructs a fresh
random-init `MPNNet` and trains it from scratch on `D_0` (and refits each round),
exactly as `[bengio2021gflownet]` does ("initializes an MPNN proxy"). Loading the
pretrained sEH weights would be wrong anyway: that proxy predicts sEH affinity,
not our DDB1 differential. (Verified at runtime: our init weights ≠ the pretrained
weights, and differ across seeds.)

## Added (all under `glue/`, zero edits to `rgfn/`)

- `glue/oracles/base.py` — `GlueOracle` ABC (`score(smiles) -> list[float]`): the
  modular seam so the loop is oracle-agnostic.
- `glue/oracles/mock_oracle.py` — `MockGlueOracle`: cheap CPU oracle (QED × MW
  band) so the whole loop runs on a laptop without gnina/GPU. Test fixture, not
  science.
- `glue/oracles/docking_6td3_oracle.py` — `Docking6TD3Oracle`: real two-tier
  gnina docking returning the DDB1 neosubstrate differential (`ddb1_dcnnaff`).
  **Mirrors `research/preprocessing/docking_6td3/dock_cluster.py`** (identical
  gnina flags, best-pose-by-CNNscore, Tier-1 `--score_only`, same differential);
  differs only in orchestration (one in-process batch vs sharded multi-GPU job).
- `glue/proxies/learned_proxy.py` — `LearnedGlueProxy`: trainable wrapper around
  RGFN's `MPNNet` with a `.fit()` (the in-loop reward `M`).
- `glue/datasets/oracle_labeled.py` — `OracleLabeledDataset`: the accumulating
  `D_i = D̂_i ∪ D_{i-1}` store (canonical-SMILES keyed, seed-CSV load, Top-K).
- `glue/active_learning/{__init__,loop}.py` — `ActiveLearningLoop`: the outer
  Alg. 1 orchestrator. New subpackage (registered in `glue/registry.py`).
- `scripts/active_learning.py` — entry point (imports `glue` **and** `Trainer`,
  parses an `configs/glue/` config, builds the loop, runs it).
- `configs/glue/active_learning_mock.gin` (local smoke) and
  `configs/glue/active_learning_6td3.gin` (Balam real run).
- `experiments/active_learning_6td3/` — worked example: `README.md`,
  `build_seeds.py`, `seed_6td3.csv` (408 real validated-docking labels = 160
  known + 248 decoys), `seed_mock.csv` (14 mols for the smoke test).
- Registrations: `glue/registry.py` (+`active_learning`), and the `oracles` /
  `proxies` / `datasets` subpackage `__init__.py` exports.

## How the loop works (and the one invariant)

Each round: **fit `M` on `D_{i-1}`** → `clear_cache()` → **`Trainer.train()`**
against `M(x)^β` → **sample query batch** from the trained forward policy →
**score with `O`** → **`D_i = D̂_i ∪ D_{i-1}`**. The trainer's reward and the loop
share **one** `LearnedGlueProxy` singleton, so refitting updates the reward RGFN
trains on. Invariant preserved: oracle labels enter training **only** by
retraining `M`, never as a direct RGFN reward. Critically, the in-loop reward is
the *cheap MPNN proxy*, so GFN training uses the `reaction` env (not
`reaction_docking`) — **docking never runs in the inner loop**, only on the
per-round query batch.

## Correction (2026-06-25): oracle uses the VINA differential, lower-is-better

The first draft of `Docking6TD3Oracle` computed the **CNNaffinity** differential
(`ddb1_dcnnaff`) with `higher_is_better=True`. That was wrong on both counts.
Checking log 002 + `compare_systems.py` + the result CSVs: the validated +78pt
discrimination is on **`ddb1_dvina` = `Vina(Tier2) − Vina(Tier1)`**, where *more
negative = better glue* (known median −2.20 vs decoy −0.60; 85.6% vs 7.3% below
−1.5). The CNNaffinity differential does **not** discriminate (known +0.23 vs
decoy +0.04, decoy range up to +1.80). CNNscore is still used — but only to pick
the pose (`max cnnsc`), matching `dock_cluster.py`. Fixed (user chose: store raw
`ddb1_dvina`, lower-is-better):
- `docking_6td3_oracle.py` now returns `vina_t2 − vina_t1`; `higher_is_better=False`.
- `LearnedGlueProxy.higher_is_better` is now a constructor arg (set `False` in
  `active_learning_6td3.gin`); dropped the positive output clip (the base config's
  exponential boosting makes the reward positive for any prediction) in favour of
  a symmetric `±clip` sanity bound; invalid molecules get a sign-aware worst-case.
- `ActiveLearningLoop.run()` now asserts `proxy.higher_is_better ==
  oracle.higher_is_better` to catch this class of bug.
- `seed_6td3.csv` / `build_seeds.py` rebuilt from the `ddb1_dvina` column.
- **Regression guard added:** `glue/tests/test_oracle_discrimination.py` — the
  "science validated" counterpart to the wiring smoke test. It checks the metric
  feeding the loop actually separates known glues from decoys (reproduces Log
  002: known 85.6% vs decoy 7.3% strong-bonus, +78pts), and a contrast check
  shows the Vina differential discriminates (+78pts) while CNNaffinity does not
  (+0pts). Verified non-vacuous: it FAILS if the seed is built from
  `ddb1_dcnnaff`. Runs on a laptop from committed data (no gnina); also runnable
  standalone (`python glue/tests/test_oracle_discrimination.py`).

**Process lesson (root cause of both this and the MLP/MPNN error):** each
consequential scientific choice was filled by inference instead of reading the
file that recorded the decision. The fix is to read all three where they exist —
the paper section, the upstream `rgfn/` code, AND our own `Logs/` experiment log
+ analysis script (`compare_systems.py`) — and to treat "smoke test passed" as
"wiring works", never "science is correct". The discrimination test above
operationalises the latter.

## Update (2026-06-25, cont.): Logs 006 + 007 confirm the Vina ΔT2−T1 choice

After the metric correction above, experiment logs **006** (six-way signal
ablation) and **007** (molecular-weight control) were written and independently
confirm `ddb1_dvina` (Vina Tier2−Tier1, lower-is-better) is the right oracle
signal — so **no reward-signal code change was needed**; the implementation was
already aligned. What changed is documentation + the guard:
- 006: ranked all six candidate signals on the entry-002 poses; **Vina ΔT2−T1
  wins, AUROC 0.946** (Cohen's d 2.38); its Youden cut −1.58 matches our −1.5.
  CNNaffinity differential is 0.850; absolute Vina Tier 1 is 0.69.
- 007: after matching glues/decoys on molecular weight, Vina ΔT2−T1 stays top
  (0.946 → 0.866) while absolute Vina Tier 1 falls below chance (0.38) — the
  differential isn't a size artifact.
- Citations to 006/007 added to: `docking_6td3_oracle.py` docstring,
  `test_oracle_discrimination.py`, the example README, and the gin config.
- Fixed two stale comments that still said `ddb1_dcnnaff` (config + build_seeds).
- Strengthened the discrimination test with an **AUROC check** (asserts > 0.90;
  reproduces 006's 0.946) and an AUROC-based contrast (Vina 0.946 vs CNN 0.850).
  Caught a sign slip while doing so: CNNaffinity is higher-is-better (pK), Vina is
  lower-is-better (energy) — using one orientation for both flips AUROC to
  1−AUROC. Now oriented per metric; both reproduce 006 exactly.

## Deliberate divergences from the publications (validate/revisit on Balam)

1. **Proxy target** — predicts our docking neosubstrate *differential*
   (`ddb1_dvina`, lower-is-better), not AutoDock sEH affinity. The project's
   novel oracle. Signal choice justified by Logs 002/006/007.
2. **Trainable proxy** — `SehMoleculeProxy` is inference-only with frozen
   weights; we add `.fit()`. This is what Alg. 1 requires; the shipped proxy just
   doesn't expose it.
3. **Random init each round** — we rebuild `MPNNet` weights every `fit()` rather
   than annealing from the previous round (avoids compounding drift; revisit).
4. **Label scaling** — we standardise labels at `fit` time and clip predictions
   to `[min_value, max_value]` at inference (mirrors the paper's renormalisation
   to a positive reward and `SehMoleculeProxy`'s `clip(1e-4, 100)`).
5. **Scale** — paper uses `|D_0|=2000` and 200 mols/round; our `D_0` is 408
   (all the validated labels we have). `query_batch_size=200` matches the paper.
6. **Validation split** — paper uses a fixed 3000-mol val set for early stopping;
   we hold out a fraction (`val_fraction`) — matters for small `D`.
7. **Replay buffer reset** — we clear it each round (its priorities go stale once
   `M` is refit). Best-effort, guarded against upstream layout changes.

## Verified locally (Mac, `rgfn` conda env — has torch/rdkit/gin/torch_geometric)

- `py_compile` of all new modules; `import glue` registers every new component.
- Both AL configs parse (after the `Trainer`-import fix below).
- `LearnedGlueProxy` fits on toy data, predicts positive rewards, handles invalid
  SMILES and early-terminal states.
- **Full mock loop end-to-end** via `scripts/active_learning.py` (2 rounds, CPU,
  `WANDB_MODE=offline`): seed `|D_0|=13` → fit → `Trainer.train()` → sample 16 →
  score → accumulate → `|D|=29` → round 2 → `|D|=45` → `top_k.csv` written. EXIT 0.
  RGFN-generated (reaction-assembled) molecules out-scored seeds under the mock
  oracle and rose to the top of the Top-K. (Smoke artifacts deleted, not committed.)
- `Docking6TD3Oracle` imports without gnina and raises a clear `FileNotFoundError`
  when receptors are absent (laptop) — Balam-only by design.

### Two bugs found and fixed during local validation
- **`scripts/active_learning.py` didn't register `Trainer`** → gin "No
  configurable matching 'Trainer'". Upstream `train.py` imports it explicitly;
  added `from rgfn.trainer.trainer import Trainer`.
- **`LearnedGlueProxy._compute_proxy_output` assumed `.molecule`** → crashed on
  `ReactionStateEarlyTerminal` after the loop's post-`fit` `clear_cache()` wiped
  the pre-seeded entry. Now guards early-terminal states → `min_value`.

## NOT verified — needs Balam (gnina + GPU + prepared receptors)

- `Docking6TD3Oracle.score()` against real `6TD3_tier{1,2}.pdbqt` + `crystal_RC8.pdb`
  (git-ignored, absent on laptop). **Reconcile/cross-validate it against
  `dock_cluster.py`** — the two duplicate the docking logic; `dock_cluster.py` is
  the source of truth. Consider unifying them into one shared `dock_batch_6td3()`.
- `scripts/active_learning.py --cfg configs/glue/active_learning_6td3.gin` full run.
- Hyperparameters: `β`, per-round `Trainer.n_iterations`, proxy `max_epochs` /
  `patience`, `query_batch_size` — all first-guess defaults, untuned.
- Reward-scale sanity: with `β=8` and standardised proxy outputs, confirm the
  exponential reward boosting doesn't explode/vanish on the real differential.
- The headline plots reviewers will want: Top-K-vs-oracle-calls curve and a
  random-acquisition baseline (`[bengio2021gflownet]` Fig. 7 analog).

---

# 2026-06-25 (cont.): documentation consolidation

Reduced duplication across the project docs (goal: one home per fact, fewer
chances for versions to drift):
- **Moved `Logs/RESEARCH_CONTEXT.md` → `docs/RESEARCH_CONTEXT.md`** (`git mv`);
  updated all ~14 referencing files (code docstrings, READMEs, CLAUDE.md, configs).
- **Slimmed `Logs/README.md`** to a short intro + pointers. Its duplicated
  "The systems" and "Common methodology" sections were deleted (already covered by
  RESEARCH_CONTEXT's systems table + terminology; 5HXB anchoring specifics remain
  in log 001). Its experiment **Index** moved into `RESEARCH_CONTEXT.md` (links now
  `../Logs/`). Its "Where results live" + "Datasets" moved into `ARCHITECTURE.md`.
- **`ARCHITECTURE.md`** is now the single source for repo layout + data/result
  locations (new "Data, datasets & results" section); RESEARCH_CONTEXT's
  "Where things live" trimmed to a pointer to it.
- Net: systems/terminology live once (RESEARCH_CONTEXT); locations live once
  (ARCHITECTURE); the experiment index lives once (RESEARCH_CONTEXT).
- **Path-corrected the historical logs** (001–005): their `pre-processing/` path
  references are now `research/preprocessing/` (real, clickable paths), so the
  "read these as research/preprocessing/" translation note was **removed** from
  both `RESEARCH_CONTEXT.md` and `CLAUDE.md`. Only verbatim *commit messages* in
  the logs' "Relevant Versions" blocks still say "pre-processing" — left intact,
  since those quote real historical commits.

---

# 2026-06-26: validation/benchmarking scaffold (second axis)

Introduced a **second organizing axis** for our own code — **production pipeline
vs. validation** — on top of the existing upstream/ours boundary. Goal: get the
structure ready for the comparative studies (multiple generators, PMO, our own
benchmarks, VAE-BO, Boltz-2) **before** Balam is back, keeping the validation
layer as separate as possible from the training pipeline. **Scaffolding only — no
implementation code imported yet** (per the user's request).

### Decisions (chosen by the user)
- **One umbrella folder, not two.** Initially scaffolded as two top-level dirs
  (`validation/` for entrants/scorers + `benchmarks/` for suites/harness/results),
  but the split was thin and tightly coupled (the harness only exists to run the
  generators) and produced two `configs/` dirs. Consolidated into a single
  `validation/` holding everything; the "benchmark" idea survives as
  `validation/suites/`. (`benchmarks/` was created and then removed in the same
  session — it was never committed.)
- **Validation oracles separate from in-loop oracles.** Boltz-2 / co-folding /
  high-fidelity checks go in `validation/oracles/` (never in-loop). In-loop
  oracles stay in `glue/oracles/`. Short-MD as a *complementary in-loop* oracle
  (CRBN ceiling, Objective 3) would still go in `glue/`, not here.
- **Baselines = thin adapters + external installs:** each baseline is a thin
  adapter in `validation/generators/<name>/`; heavy upstream code installs via
  `external/setup_<name>.sh` (not vendored).

### The rule that enforces the split
> `validation/` may import from `glue/` and `rgfn/`; the production pipeline
> (`glue/`, `scripts/train.py`, `configs/glue/`) must **never** import from
> `validation/`. The dependency arrow points one way.

### Created (directories + READMEs / `.gitkeep` only)
- `validation/` + `README.md` (boundary rule, full single-folder layout,
  what-goes-where guide, and the "why reviewers care" rationale).
  - `validation/generators/{rgfn,synflownet,fraggfn,vae_bo}/` + `generators/README.md`
    (entrant table, planned common `Generator` interface).
  - `validation/oracles/boltz2/` + `oracles/README.md`
    (in-loop vs. validation-oracle contrast table).
  - `validation/suites/{pmo,glue_suite}/` + `suites/README.md`.
  - `validation/harness/README.md`, `validation/configs/README.md`,
    `validation/results/` (`.gitkeep`).
- `external/setup_{synflownet,fraggfn,vae_bo}.sh` — **placeholder stubs** (valid
  bash, `exit 1` with a TODO; mirror `setup_reinvent.sh` when implemented).

### Docs updated
- `CLAUDE.md`: ownership table now tags `glue/` as *production pipeline* and adds
  a single `validation/` row as *validation*; new "second axis" subsection with
  the dependency rule.
- `docs/ARCHITECTURE.md`: layout tree expanded with the `validation/` subtree;
  new "Two axes" section + "Component flow (validation)" diagram.

### Verified
- `bash -n` on all three new `external/setup_*.sh` stubs (pass).
- Directory tree + placeholder files created as listed; `benchmarks/` removed.

### NOT done / deferred (by design — structure only)
- No `Generator` base class, adapters, harness, suite, or oracle code written.
- `__init__.py` files intentionally omitted until real code lands (each Python
  subtree gets one then, so the harness can import it).
- Config format for `validation/configs/` (gin vs. YAML) not decided yet.
- The setup stubs do not actually install anything.

---

## 2026-06-26 — sEH GPU-docking oracle for the active-learning loop

Added a fast docking oracle so the multi-round AL loop can actually finish (the
CPU-bound 6TD3 gnina oracle was killed by the login-node CPU cap; exp `009`/`010`).

### Created (ours)
- `glue/oracles/docking_seh_oracle.py` — `DockingSEHOracle(GlueOracle)`. Real sEH
  docking via QuickVina2-GPU-2.1. **Composes** upstream `DockingMoleculeProxy` and
  calls its `dock_batch_qv2gpu` directly on SMILES — the docking is byte-for-byte
  the upstream path (no duplication, unlike the 6TD3 oracle). `higher_is_better=
  False` (Vina energy); returns `nan` on failure per the `GlueOracle` contract;
  the heavy proxy is built lazily so the module imports on a laptop.
- `configs/glue/active_learning_seh.gin` — AL loop wiring (mirrors
  `active_learning_6td3.gin`): `LearnedGlueProxy` (MPNN) as `M`, `DockingSEHOracle`
  as `O`, `reaction` env (docking stays out of the inner loop).
- `research/active_learning_seh/` — sEH-specific run tooling (sibling of
  `research/preprocessing/`; bundles scripts + submit + README like the
  `pose_selection_ablation/` precedent): `make_seh_seed.py` (builds `D_0` by
  sampling the untrained RGFN policy and docking → `seed_seh.csv`; reports
  s/mol), `validate_seh_oracle.py` (quick live smoke), `submit_al_seh.sh` (Balam
  compute-node run; cuda module + boost libs; regenerates a 300-mol seed on
  `$SCRATCH`), `README.md`. The generic `scripts/active_learning.py` entry point
  stays in `scripts/` (shared with 6TD3).
- `experiments/active_learning_seh/seed_seh.csv` — committed starter seed.

### Edited (ours)
- `glue/oracles/__init__.py` — export/register `DockingSEHOracle`.
- `docs/RESEARCH_CONTEXT.md` — index row 010 + Objective 1 note.

### Verified (on balam-login01 A100, `rgfn` env)
- `py_compile`; import via `glue.registry`; gin parse of the config + class
  resolves by name.
- **Live GPU dock**: aspirin −6.30 (matches the recorded sEH validation),
  ibuprofen −6.90, caffeine −6.50; invalid SMILES → `nan`; ~3.7 s/mol.
- **Seed gen**: 36/40 RGFN-sampled molecules docked; dataset loads |D_0|=36;
  proxy/oracle sign check passes (both `higher_is_better=False`).
- `bash -n` on `submit_al_seh.sh`.

### NOT done / deferred
- The **full multi-round loop** has not been run yet (needs a compute node;
  `submit_al_seh.sh` is ready). No top-k-vs-oracle-calls curve / random baseline.
- Oracle outputs not yet cross-checked against an actual
  `configs/rgfn_seh_docking.gin` run on the same molecules (numbers should match).

---

## 2026-06-26 — per-phase timing for the active-learning loop

Experiment 009 could not say where the loop spent its time (only eyeballed
per-step rates; the run died at the oracle step with no number for it). Added
intrinsic, always-on phase timing.

### Added (ours)
- `glue/active_learning/timing.py` — `PhaseTimer`: a context-manager that times
  the four per-round phases (`fit_proxy`, `train_gfn`, `sample_batch`,
  `oracle_score`). On each phase exit it prints (`[AL]`), logs to the trainer
  logger under the `timing` prefix (→ wandb), and **appends** to
  `<run_dir>/active_learning/phase_timings.csv`. The append-before-next-phase
  ordering means a mid-run crash (the SIGXCPU that ended exp 009) still leaves a
  record of every phase that finished. `report_total()` ranks phases by share of
  total wall-clock — the "where to save time" answer.

### Edited (ours)
- `glue/active_learning/loop.py` — wrapped the four phases in `timer.phase(...)`,
  added `report_round`/`report_total`. No behavioural change to the algorithm;
  recorded via `finally`, so timing is robust to a raising phase.

### Verified (laptop, no heavy deps)
- `py_compile` on both files. Standalone smoke test of `PhaseTimer`: `_fmt`
  formatting, CSV append, crash-phase recorded via `finally`, share-ranked
  summary.

### NOT done / deferred
- Not yet run inside a real loop on Balam (needs the `rgfn` env / GPU). The four
  phase labels and their wiring are unverified against a live `Trainer.train()`.
- Overhead is negligible (one `perf_counter` + a CSV append per phase, 4/round),
  but unconfirmed under the real loop.

---

## 2026-06-26 — directory reorg: experiments/ (per-run), data/ (inputs), flat scripts/

Reorganized the top-level layout for clarity (one home per concept). No `glue/`
or `rgfn/` logic changed; this is moves + path-rewiring + docs.

### Moved / merged
- **`research/` dissolved into `experiments/`**, now **grouped by type, one dir
  per run**: `experiments/{active_learning,oracle_validation,ablations}/<run>/`.
  - `research/active_learning_{seh,6td3}/` → `experiments/active_learning/{seh,6td3}/`
    (merged with the matching `experiments/active_learning_*` seeds/outputs).
  - `research/preprocessing/{docking_6td3, docking_gnina→docking_crbn, clean*.py,
    compare_systems.py}` → `experiments/oracle_validation/…`.
  - `research/preprocessing/{pose_selection_ablation→pose_selection,
    full_comparison→sixway, full_comparison_mw→mw}` → `experiments/ablations/…`.
  - Old flat AL output dirs (`active_learning_mock`, `al_probe→probe`,
    `active_learning_6td3_inner→6td3_inner`) → under `experiments/active_learning/`.
  - `experiments/oracle_validation/` chosen over `validation/` to avoid clashing
    with the top-level `validation/` benchmarking layer.
- **`models/` + `research/preprocessing/test-data/` folded into `data/`** (the
  single inputs dir): `data/models/`, `data/validation-molecules/` (renamed from
  "test-data" — they're validation *inputs*, not fixtures). `data/` keeps its name
  because upstream hardcodes `data/chemistry.xlsx` + `data/targets/`.
- **`scripts/hpc/` removed**; the one generic `submit.sh` → `scripts/submit.sh`.
  Run-specific submit scripts already live with their experiment.

### Rewired
- All `git mv`/`mv` done so git detected pure **renames** (history preserved).
- Code paths: `glue/oracles/docking_6td3_oracle.py` `_DOCK_DIR`,
  `glue/tests/test_oracle_discrimination.py` (`SEED_CSV`/`DOCK_DIR`), the moved
  docking/analysis scripts (`../test-data`→`data/validation-molecules`,
  `docking_gnina`→`docking_crbn`, `models/`→`data/models/`), `clean*.py` in/out dirs.
- Configs: 5 AL `seed_csv` paths repointed to `experiments/active_learning/<run>/`.
- `scripts/active_learning.py`: run_name now groups outputs under
  `experiments/active_learning/<run>/<ts>/` so outputs co-locate with each run's
  committed material.
- `.gitignore`: replaced the blanket `experiments/*` ignore with a **pattern-based**
  scheme — ignore timestamped run-output dirs (`experiments/**/YYYY-MM-DD_*/`),
  force-track `seed_*.csv` / `*_results.csv`; updated `models/`→`data/models/`,
  `test-data`→`data/validation-molecules`, cluster_out/violins paths.

### Verified (balam-login01, `rgfn` env)
- `git status` shows clean renames; no committed file under a timestamped run dir;
  `git check-ignore` confirms run outputs ignored / seeds+results+code tracked.
- `py_compile` of all moved scripts; `import glue`; all 5 AL configs parse **and
  their seed paths resolve**; `pytest glue/tests/test_oracle_discrimination.py` (2
  passed); `bash -n` on all 6 submit scripts.
- READMEs updated/added: `experiments/README.md` (structure guide), per-group/run
  READMEs, `data/README.md` + `data/{models,validation-molecules}/README.md`,
  `scripts/README.md`; plus `CLAUDE.md`, `docs/ARCHITECTURE.md`, README/context.

### NOT done / deferred
- Balam-side `$SCRATCH` paths (e.g. where the user keeps `.cif`/receptors) may need
  the same `models/`→`data/models/` move on Balam; the repo-relative refs are fixed.
- The 6TD3 docking-oracle pdbqt receptors live (git-ignored) at the new
  `experiments/oracle_validation/docking_6td3/` path on this login node; confirm
  they're present at that path on Balam compute nodes before the next 6TD3 run.

---

## 2026-06-27 — `top_k` deliverable sort fix (`glue/datasets/oracle_labeled.py`)

`OracleLabeledDataset.top_k()` sorted `reverse=True` (largest label first), but
oracle labels are *more-negative = better* (`GlueOracle.higher_is_better = False`,
for both the 6TD3 `ddb1_dvina` differential and the sEH Vina energy), so the
"Top-K deliverable" returned the **worst** molecules. Found while reviewing the
first multi-round 6TD3 run (Logs/011, job 69445), whose `top_k.csv` listed +0.68
at rank 1. Fixed to sort ascending. Scope: deliverable file only — training's
reward/proxy path is separate and was correctly oriented (the run's generated
molecules dock *well*). Regenerated `top_k.csv` for that run from
`dataset_round_003.csv` (best now −4.92).

**Verified:** `py_compile` of the module. **Not verified here:** no unit test
covers `top_k`, and gin/rdkit/conda aren't on this node, so the loop wasn't
re-run; the regenerated `top_k.csv` was produced by replicating the (trivial)
ascending-sort logic inline. A `test_top_k_orientation` unit test would be cheap
insurance.

---

## 2026-06-28 — docking oracle sub-step timing (`OracleStepTimer`)

Added per-substep wall-clock timing inside the expensive docking oracle so the
`oracle_score` phase (35% of loop wall-clock, Logs/011) is no longer a black box.

- **New** `glue/oracles/step_timing.py::OracleStepTimer` — disabled-by-default
  (silent no-op) companion to `glue/active_learning/timing.PhaseTimer`. Appends
  `(round, step, seconds, n_molecules)` rows on each sub-step's completion
  (crash-safe), prints a `[dock]` line live.
- `glue/oracles/docking_6td3_oracle.py` — new `enable_step_timing(csv_path)`;
  `score()`'s four sub-steps wrapped: `embed` / `tier2_dock` / `pose_select` /
  `tier1_rescore`. **Docking logic unchanged** (like-for-like with Logs/011).
- `glue/active_learning/loop.py` — opt-in hook: if the oracle has
  `enable_step_timing`, the loop points it at
  `<run>/active_learning/docking_timings.csv`. Loop stays oracle-agnostic; mock /
  sEH oracles (no hook) are unaffected (`getattr` + `callable` guard).

**Verified:** `py_compile` (3 files); in the `rgfn` env `import glue` resolves the
oracle, disabled timer is a confirmed no-op, enabled timer writes the expected
4-column CSV with recoverable s/mol; `bash -n` on the new submit script.
**Not verified here:** a real multi-round run with the timer live — that is
**Logs/012** (job 69450, queued this session). The sEH oracle does **not** yet
implement `enable_step_timing`; add it there when that loop is next run.

---

## 2026-06-29 — FragGFN baseline (first `validation/` entrant) + the oracle bridge

Implemented the **FragGFN** non-synthesizable baseline (Recursion's `gflownet`
fragment environment), the foil for RGFN's synthesizability claim (Objective 4/5;
`Logs/015`). This is the first real code under `validation/`, so it sets the
pattern for future entrants.

**Key constraint that shaped the design.** Recursion's `gflownet` (pinned
`da999404`) hard-pins **python 3.10 / torch 2.1.2 / torch-geometric 2.4.0**,
incompatible with the `rgfn` env (3.11 / torch 2.3 / dgl). So FragGFN runs in its
**own `fraggfn` conda env**, and — since `glue/__init__` eagerly imports the
`rgfn`-dependent registry — it **cannot import `glue`/`rgfn` at all**. The shared
docking oracle is therefore reached **across the env boundary** via a CLI bridge.

**The oracle bridge — `scripts/score_batch.py` (generic, reusable).** Runs in the
`rgfn` env; scores a SMILES file with a named glue oracle and writes the **standard
candidate-dataset format** (`glue.datasets.candidates` + `glue.metrics.dataset_metrics`,
the same code RGFN's `SuggestionLog` uses), `has_route=0` for non-synthesizable
entrants. Per-round shards + `--finalize` give crash-safety across the per-round
subprocess calls. This is now the single shared scoring standard every baseline
will use; it lives in `scripts/` because it's pipeline-wide, not FragGFN-specific.

**Added:**
- `external/setup_fraggfn.sh` — builds the `fraggfn` env (py3.10), clones gflownet
  to `external/gflownet/` (git-ignored via the new `external/*/` rule — clones are
  **not vendored**), installs cu118 torch + pyg wheels + gflownet, pins `numpy<2`.
- `scripts/score_batch.py` — the oracle bridge (above).
- `validation/generators/fraggfn/{proxy,task,al_loop,run_fraggfn_al}.py` + README —
  the thin adapter: `AtomMPNNProxy` (the Bengio-2021 MPNN reused from gflownet's
  own `bengio2021flow`, same architecture as `LearnedGlueProxy`), `FragGFNTrainer`
  over `FragMolBuildingEnvContext`, and `FragGFNActiveLearningLoop` mirroring
  `glue/active_learning/loop.py`. Deliberately does **not** import `glue`/`rgfn`.
- `validation/{__init__,generators/__init__}.py` — make the subtree importable
  (the convention noted in `validation/README.md`).
- `validation/configs/fraggfn_6td3.yaml` (+ `fraggfn_smoke.yaml`) — budget matched
  field-for-field to `configs/glue/active_learning_6td3_gpu.gin`.
- `experiments/active_learning/fraggfn_6td3/{README.md,submit_fraggfn_6td3.sh}` —
  Balam run scaffolding; **reuses** `../6td3/seed_6td3.csv` (identical `D_0`).

**Verified (login node A100, this session):** `fraggfn` env builds + imports
gflownet; `mol2graph` width 71 / `FRAGMENTS` 72; `py_compile` all new Python in
both envs; `bash -n` the submit + setup scripts; YAML configs load; the bridge
mock run produces a conformant standard dataset (`validate_candidate_dataset` →
no issues, `has_route=0`); and a **full end-to-end CPU dry run** of the loop
(2 rounds, mock oracle via the bridge) — fit M → train fragment-GFN → sample →
cross-env score → accumulate → finalize → Top-K.
**Not verified here:** the real 3-round **GPU docking** run (Balam compute node;
`sbatch experiments/active_learning/fraggfn_6td3/submit_fraggfn_6td3.sh`) and the
FragGFN-vs-RGFN comparison numbers — pending, tracked in `Logs/015`.

## Active-learning loop robustness (post job 69481, 2026-06-30)

Two fixes prompted by the entry-014 GPU run, which was killed by a Balam node
failure (balam009 returned all-`no_pose` in round 1, then died mid-round-2):

- **All-NaN-round abort guard** (`glue/active_learning/loop.py`): if an entire
  round's oracle batch comes back NaN (no labels), the loop now raises a clear
  `RuntimeError` instead of silently refitting M on an unchanged D and burning
  another full GFN training run. The round's provenance (`suggestions/`,
  `dataset_round_NNN.csv`) is written *before* the abort, so the failure is fully
  inspectable. Triggers only on a wholesale oracle failure (non-empty batch, zero
  valid scores) — partial failures pass through as before.
- **Manifest provenance** (`SuggestionLog` / `CandidateDataset` / loop / driver):
  `ActiveLearningLoop` gained `system` + `seed` args, forwarded into the candidate
  manifest (were `null`). `scripts/active_learning.py` binds
  `ActiveLearningLoop.seed` from `--seed`; `configs/glue/active_learning_6td3_gpu.gin`
  sets `ActiveLearningLoop.system='6td3'`.

Verified on the Trillium login node (`source ~/bin/rgfn-smoke-env.sh`): `py_compile`;
`glue` import + `ActiveLearningLoop` signature carries `system`/`seed`; gin parse of
`active_learning_6td3_gpu.gin` with the `--seed` binding; and a `SuggestionLog`
round-trip confirming `system`/`seed` land in `manifest.json`. Not verified: a full
multi-round loop run (needs a GPU compute node; Balam submission unavailable).

## RxnFlow synthesizable baseline entrant (exp 016, 2026-06-30)

Added a **second** synthesis-aware benchmark entrant — **RxnFlow** (`[seo2024rxnflow]`,
ICLR 2025; reaction-template + building-block GFlowNet, action-space subsampling) — as
the *synthesizable peer* to RGFN, opposite the non-synthesizable FragGFN foil. (Briefly
scoped as SynFlowNet first, then switched to RxnFlow; the SynFlowNet bib/reference/PDF
work was reverted.) RxnFlow is built on a bundled Recursion `gflownet`, so it drops into
the FragGFN two-env precedent almost verbatim. New / changed:

- `validation/generators/rxnflow/` — thin adapter mirroring `fraggfn/`:
  `proxy.py` (re-exports FragGFN's `AtomMPNNProxy` so `M` is identical across entrants),
  `task.py` (`RxnFlowTask`/`RxnFlowGlueTrainer` over RxnFlow's `BaseTask`/`RxnFlowTrainer`,
  reward = `M`, constant β, `num_workers=0`), `al_loop.py`
  (`RxnFlowActiveLearningLoop` = `[bengio2021gflownet]` Alg.1 + the all-NaN guard +
  `extract_route`), `run_rxnflow_al.py`, `README.md`, `__init__.py`.
- `scripts/score_batch.py` — generalized the shared oracle bridge to be **route-aware**:
  new `--routes` (per-round JSONL `{"smiles":…, …route…}`) stored next to the shards and
  joined onto candidates by SMILES on `--finalize` → synthesizable entrants emit
  `has_route=1` + `routes.jsonl`. FragGFN's path (no `--routes`) is byte-unchanged; the
  hardcoded "FragGFN/non-synthesizable" finalize note is now generator-aware.
- `validation/configs/rxnflow_{6td3,smoke}.yaml` — budget/oracle/proxy matched
  field-for-field to `configs/glue/active_learning_6td3_gpu.gin`; a `rxnflow:` block
  carries the generator knobs (`env_dir`, `max_reactions`, `action_sampling_ratio`).
- `experiments/active_learning/rxnflow_6td3/{README.md,submit_rxnflow_6td3.sh}` — Balam
  run scaffolding; **reuses** `../6td3/seed_6td3.csv` (identical `D_0`).
- `external/setup_rxnflow.sh` — `rxnflow` conda env (py3.12, torch 2.5.1+cu121) + clone +
  `pip install -e` + env-dir prep (public ZINCFrag blocks + `templates/hb_edited.txt`).
- References: reverted SynFlowNet; added `[seo2024rxnflow]` to `references.bib` +
  `Logs/references/README.md`; PDF at `pdfs/seo2024rxnflow.pdf`.

**Verified (this session, no heavy stack locally):** `py_compile` all new Python;
`bash -n` the setup + submit scripts; both YAMLs load and their `loop`/`oracle` blocks
match `fraggfn_6td3.yaml`; the route-aware bridge round-trips with **and** without
`--routes` (`validate_candidate_dataset` → no issues; `has_route` flips 1↔0 correctly).
**Flagged for Balam validation (heavy stack is cluster-only):**
- the RxnFlow upstream API — `rxnflow.config.Config` schema + `env_dir` field path,
  `RxnFlowTrainer`/`BaseTask` signatures (`setup_task`, `set_default_hps`), and the
  per-action attributes `extract_route` reads (block / reaction template / product).
  These are written best-effort from the RxnFlow docs/examples; confirm against the
  cloned repo and tighten `extract_route` so routes are chemically faithful.
- the cu121 (rxnflow) vs cu118 (rgfn bridge) coexistence — separate procs/envs, torch
  cu121 wheels are RPATH-self-contained; needs a GPU driver ≥525 (CUDA 12.1) on the node.
- `external/setup_rxnflow.sh` step 4 (env-dir prep) has a `TODO` for the exact RxnFlow
  building-block preprocessing command (per the cloned repo's `data/README.md`).
- finish: pin `RXNFLOW_COMMIT`, run the CPU mock smoke, then the 3-round GPU run; the
  RGFN-vs-RxnFlow-vs-FragGFN comparison numbers — tracked in `Logs/016`.

---

## 2026-06-30 — Synthesizability metric (AiZynthFinder + SA) in the harness

Added the first `validation/harness/` evaluation metric: a post-hoc
**synthesizability** report over a candidate dataset, the analogue of the `AiZynth`
column in `[koziarski2024rgfn]`/`[gainski2025scent]` and RxnFlow's "Synthesizability %"
(`[seo2024rxnflow]`). It runs uniformly on **every** entrant because they all emit the
one standard candidate-dataset format (`docs/CANDIDATE_DATASET_FORMAT.md`).

Design choices (confirmed with Mark before building):
- **Stock/templates:** AiZynthFinder's standard **public dataset** (USPTO expansion +
  ZINC in-stock) — reproducible and directly comparable to the published RGFN/SCENT
  numbers. Stock/expansion/filter keys are CLI-overridable.
- **Metrics:** the full paper set — fraction-solved (AiZynth success rate), mean #steps
  over solved, and the SA-score distribution.
- **Isolation:** AiZynthFinder in its own `aizynth` conda env
  (`external/setup_aizynthfinder.sh`), invoked post-hoc as a standalone CLI — same
  "one env per tool, one on-disk standard" pattern as the fraggfn/rxnflow oracle bridge.

Added:
- `validation/harness/synthesizability.py` — reads `candidates.csv`/`manifest.json`
  **directly** (csv/json, no `glue` import, so it runs in the lean `aizynth` env);
  parallel AiZynth driver (one finder per worker); dedups by canonical SMILES; writes
  `synthesizability.csv` (per-molecule) + `synthesizability_summary.json` (aggregate,
  incl. the by-construction self-report cross-check). CLI: `--dataset/--config/--nproc/
  --top-k/--no-dedup/--time-limit/--iteration-limit`.
- `validation/harness/__init__.py` (kept import-light on purpose).
- `validation/harness/test_synthesizability.py` — dependency-free unit tests
  (monkeypatch RDKit + AiZynth) for dedup / top-k / scatter-back / aggregation.
- `external/setup_aizynthfinder.sh` — `aizynth` env + `download_public_data` + smoke.
- References: `[genheden2020aizynth]` + `[ertl2009sascore]` in `references.bib` and the
  references `README.md` (new "Evaluation — synthesizability metrics" section).
- `validation/harness/README.md` — documents the landed metric + run command.

**Verified (Mac/login, no heavy stack):** `py_compile` the module/test/init; `bash -n`
the setup script; `python -m unittest validation.harness.test_synthesizability` (3/3
pass — dedup denominator is unique-molecule level, top-k picks best-by-score, files
written).
**Flagged for `aizynth`-env validation (not runnable without the install):**
- the AiZynthFinder ≥4 API surface the driver uses — `AiZynthFinder(configfile=…)`,
  `.stock/.expansion_policy/.filter_policy.select(key)` + `.items`, `target_smiles`,
  `tree_search()`/`build_routes()`, and the `extract_statistics()` keys (`is_solved`,
  `number_of_steps`, `number_of_solved_routes`, `top_score`) — written from the docs;
  confirm against the installed version and adjust key names if the schema differs.
- `download_public_data` output layout (that it writes `config.yml` with `zinc`/`uspto`
  keys) — the `--stock/--expansion/--filter` defaults assume this; the driver falls back
  to the first available key if a name is absent, but verify on a real install.
- timing/throughput on the cluster (retrosynthesis is seconds–minutes/molecule); decide
  whether to default to `--top-k 500` like the papers rather than the full set.

## SCENT cost-aware baseline entrant (exp 017, 2026-06-30)

Added the **cost-aware** benchmark entrant — **SCENT** (`[gainski2025scent]`,
arXiv:2506.19865). SCENT is a **fork of RGFN from the same lab** (its package is named
`rgfn`, same py3.11/torch2.3/dgl/gin stack, same `rgfn.api`/`Trainer`/reaction env) that
adds Recursive Cost Guidance + Exploitation Penalty + Dynamic Library. RGFN is one of
SCENT's *own* baselines, so this is the tightest of the three comparisons.

Design choices (confirmed with Mark before building, via AskUserQuestion):
- **Variant:** *full* SCENT (all three mechanisms ON) — the paper's headline method.
- **Library:** SCENT's **SMALL** building-block set, which *ships* the building-block
  prices (`fragment_to_real_cost.json`) + reaction yields (`templates_yields.csv`) the
  cost model needs — the only way to run *true* cost-aware SCENT without sourcing Enamine
  pricing. Reward/oracle/seed/budget/β stay identical to the RGFN run; only the generator
  + its blocks differ (inherent to a different generator).

Key difference from the FragGFN/RxnFlow precedent — **why the adapter is gin-driven, not
programmatic:** SCENT is configured entirely through its own gin (cost guidance,
exploitation penalty, dynamic library, env, policies are all gin-wired singletons), so
reproducing it programmatically (the `gflownet` `Config` route fraggfn/rxnflow take) is
infeasible. Instead the runner parses a gin AL config that `include`s SCENT's
`scent_base.gin` and swaps the sEH proxy for ours — mirroring `scripts/active_learning.py`.

**Namespace hazard (the load-bearing subtlety):** SCENT's package is *named* `rgfn`, so in
the `scent` env `import rgfn` must hit SCENT's installed package, not our repo-local
`rgfn/`. The runner therefore (a) never puts the repo root on `sys.path` (uses **sibling**
imports `import proxy`/`import al_loop`/`import route`, off the runner's own dir), and (b)
`chdir`s into `external/scent` so SCENT's relative gin includes + `data/small/*` resolve —
with every *our*-side path (config, seed, run dir, the repo root the bridge runs in) made
absolute first. This is why the adapter modules use plain (not package-relative) imports
and are NOT imported by `validation/generators/scent/__init__.py`.

New:
- `validation/generators/scent/` — `proxy.py` (`LearnedDockingProxy`: line-for-line the
  same MPNN as `glue.proxies.LearnedGlueProxy`/fraggfn's `AtomMPNNProxy`, here subclassing
  **SCENT's** `CachedProxyBase` + importing **SCENT's** bundled `MPNNet`/`mol2graph`),
  `route.py` (self-contained copy of `glue.active_learning.route`), `al_loop.py`
  (`ScentActiveLearningLoop` = `[bengio2021gflownet]` Alg.1, gin Trainer + bridge scoring
  + route JSONL + all-NaN abort guard; `LabelStore` = `D`), `run_scent_al.py` (entry
  point), `README.md`, `__init__.py`.
- `validation/configs/scent_6td3.gin` (+ `scent_smoke.gin`) — `include`s `scent_base.gin`
  + `small.gin`; budget/seed/β matched to `configs/glue/active_learning_6td3_gpu.gin`;
  oracle bridge params (the same `Docking6TD3GpuOracle` knobs) passed to the loop.
- `external/setup_scent.sh` — clone `koziarskilab/SCENT@af1fee5` + `scent` conda env
  (py3.11.8/torch2.3/dgl2.2.1, `pip install -e .`) + import smoke test.
- `experiments/active_learning/scent_6td3/submit_scent_6td3.sh` — Balam submit (OpenCL
  gate, `--exclude=balam008`, `$SCRATCH` outputs; `scent` env, bridge re-enters `rgfn`).
- References (`[gainski2025scent]`), generator table, `Logs/017`.
- The shared **route-aware** bridge (`scripts/score_batch.py`) is reused **unchanged**
  (the RxnFlow `--routes` extension already covers synthesizable entrants).

**Verified (Trillium login, no heavy stack):** `py_compile` all SCENT Python; `bash -n`
both shell scripts; static gin-integrity check — every `include` in `scent_{6td3,smoke}.gin`
resolves into the clone, every symbol the adapter imports exists in SCENT's source, and
`data/small/{fragments,templates,fragment_to_real_cost,templates_yields}` are present;
confirmed `CachedProxyBase.compute_proxy_output` wraps a `List[float]` into `ProxyOutput`
(matches `LearnedDockingProxy`'s return), `Trainer.{train,close,train_forward_sampler,
train_batch_size}` + the `Trajectories`/`Reaction*` API the loop binds to all exist.
**Resolved on Balam (job 69513, `COMPLETED` 2 h 03 m) — three bring-up fixes:**
- `import rgfn` died on `pkg_resources` (wandb dep; py3.11 env had setuptools≥81 which
  removed it) → pinned `setuptools<81` (now in `setup_scent.sh`).
- gin "No configurable matching 'Trainer'" → `rgfn/trainer/__init__` doesn't import
  `trainer.py`; runner now does `from rgfn.trainer.trainer import Trainer` (as SCENT's own
  `train.py` does).
- SCENT's sEH-tuned `ScaffoldCost` metric calls `.molecule` on any state with `proxy>8`
  without a terminal-type guard; our lower-is-better proxy gives invalid states `+clip=+10`
  → crash. Overrode `train_metrics` to a sign-safe subset (kept `@TrajectoryCost`); these
  are wandb-only diagnostics, so no effect on generation or the comparison.
- Confirmed working: runner's `chdir`-into-clone + sys.path handling resolves `import rgfn`
  to SCENT (not our repo-local `rgfn/`); the multi-line `oracle_args` dict literal parses;
  the cost-guided graph constructs from `data/small/*`; `has_route=96/96` with tree routes.
- Result: SCENT matches RGFN/FragGFN on glue quality (median dvina −2.12, best −5.81),
  fully synthesizable, and its cost-awareness appears to counteract the size drift — full
  numbers in `Logs/017`.

## sEH benchmark: GPU-pipeline parity + validation-baseline coverage (2026-06-30)

**Why:** sEH is the canonical GFlowNet docking benchmark ([bengio2021gflownet],
RGFN's `configs/rgfn_seh_docking.gin`); we want it as a *reproduction check* before
trusting the harness on the novel 6TD3 glue task. The sEH oracle (`DockingSEHOracle`,
exp 010) and a loop config already existed, but the config lagged the 6TD3 GPU
pipeline and no validation baseline could target sEH. This entry brings sEH to parity.

**Re-validated live (Trillium login H100):** `DockingSEHOracle` (QuickVina2-GPU sEH
docking) reproduces aspirin −6.3 (known value), ibuprofen −6.9, anthracene −8.5,
caffeine −6.5, at ~0.9 s/mol; invalid SMILES → `nan`. The Balam-built QV2-GPU stack
runs unchanged on Trillium (shared FS) via `~/bin/rgfn-smoke-env.sh`. The full
bridge path `scripts/score_batch.py --oracle docking_seh` was also exercised
end-to-end: scores aspirin −6.3 and writes the standard candidate dataset
(`candidates.csv` + `manifest.json`, `has_route=0`).

Changed:
- `configs/glue/active_learning_seh.gin` — GPU-pipeline parity with
  `active_learning_6td3_gpu.gin`: added `ActiveLearningLoop.system='seh'`,
  `Trainer.valid_every_n_iterations`, and a corrected metric block. The inherited
  `@NumScaffoldsFound` (from `rgfn_base.gin`) used positive thresholds `[5,6,7,8]`
  that assume the upstream `SehMoleculeProxy`'s clipped non-negative reward; OUR
  proxy `M` predicts the RAW Vina energy (negative, lower-is-better), so that metric
  was silently inert. Swapped in `@SafeNumScaffoldsFound` with
  `proxy_higher_better=False` and Vina-scale cutoffs `[-8,-9,-10,-11]` grounded in
  the seed `D_0` distribution (median −7.0, Q1 −8.4, strong binders ≤ −10).

New (validation baselines can now target sEH — all three implemented entrants):
- `validation/configs/{fraggfn,rxnflow}_seh.yaml`, `validation/configs/scent_seh.gin`
  — sEH analogues of the `*_6td3` configs, budget/seed/β matched to
  `active_learning_seh.gin`; oracle switched to `docking_seh` (single-target Vina
  binding, `higher_is_better=false`; args `exhaustiveness/docking_batch_size/
  n_conformers/n_gpu`, **no** `num_modes`/dvina), threshold −8.0, seed
  `experiments/active_learning/seh/seed_seh.csv` (250 labels). `scent_seh.gin` is a
  surgical mirror of the validated `scent_6td3.gin` (verified by diff).
- `experiments/active_learning/{fraggfn,rxnflow,scent}_seh/` — submit scripts +
  READMEs, mirroring the `*_6td3` Balam submit scripts (OpenCL gate, `--exclude=balam008`,
  `$SCRATCH` outputs; loop in the generator's env, bridge re-enters `rgfn`).
- Fixed a stale "36-molecule starter" claim in the sEH README + submit script (the
  committed `seed_seh.csv` actually holds 250 labels).

**Verified (Trillium H100 + static):** live `DockingSEHOracle` smoke + live
`docking_seh` bridge run (above); `active_learning_seh.gin` re-parses with all
`@references` resolved (`@SafeNumScaffoldsFound` included); `py_compile` of touched
Python; `bash -n` on all three new submit scripts; YAML schema parity of
`{fraggfn,rxnflow}_seh.yaml` vs their `_6td3` siblings; `scent_seh.gin` body diff vs
`scent_6td3.gin` shows only the intended budget/oracle/threshold/system changes.
**Flagged for `scent`/`rxnflow`-env validation (not runnable without those installs):**
the gin/yaml end-to-end parse under each baseline's own env, and the real multi-round
GPU runs of the RGFN sEH loop + the three baselines on a Balam compute node.

## RxnFlow entrant — Balam validation + first run (exp 016, 2026-06-30, job 69518)

Installed and validated the RxnFlow entrant on Balam (login A100 + compute), resolving
every API/stability uncertainty flagged above. Adapter fixes made against the real
cloned repo:
- **Route extraction rewritten** to consume RxnFlow's real trajectory format
  (`ctx.read_traj` → `("FirstBlock", block)` / `("UniRxn", template)` /
  `("BiRxn", template, block)` tuples), not the guessed action-object attributes; and
  sampling switched to RxnFlow's own `algo.graph_sampler.sample_inference` +
  `ctx.object_to_log_repr`/`read_traj` (the `RxnFlowSampler` path). Verified: 8/8 samples
  render faithful multi-step routes.
- **Config path** `model.num_layers` → `model.graph_transformer.num_layers` (RxnFlow's
  ModelConfig has no `num_layers`).
- **`gdown`** added to `setup_rxnflow.sh` (imported by `rxnflow.utils.misc`, missing from
  RxnFlow's pyproject).

**The load-bearing finding — template set vs. library size.** With `hb_edited.txt` (71
templates) on the bundled 10k ZINCFrag debug library, training goes **non-finite**
(NaN policy logprobs): the library is too sparse, so many (template, reactant) states
have no compatible second block → all-masked action categories → `log(0)`. Confirmed it
was the env, not our code, by reproducing the NaN with RxnFlow's **own** stock QED task
(14/25 steps non-finite), and confirmed the fix by rebuilding with the default
**`real.txt` (109 Enamine-REAL templates)** → QED control **0/25** and our
constant-β=8 proxy loop **0/25**. So: `setup_rxnflow.sh` now defaults to `real.txt`;
switch to `hb_edited.txt` only with the full ~200k ZINCFrag library.
- **RxnFlow defaults are the stable regime**, kept as-is: `action_subsampling.sampling_ratio=0.02`,
  `train_random_action_prob=0.1`. Removed our `RxnFlowGlueTrainer.set_default_hps`
  override that forced `train_random_action_prob=0.0` (the RGFN/FragGFN value) — RxnFlow
  needs positive exploration or its TB loss diverges. Our shared-fairness levers (proxy
  `M`, oracle, seed, budget, constant β=8) are unchanged; only RxnFlow's native
  exploration/subsampling knobs differ (per-generator, as intended).
- **cu121/cu124 vs cu118 coexistence confirmed:** rxnflow torch 2.5.1+cu124 runs GPU fine
  with the `cuda/11.8.0` module + rgfn env's `LD_LIBRARY_PATH` present (needed for the
  bridge's dgl). Driver 580 on balam004 covers CUDA 12.

**Validated end-to-end (login A100):** full cross-env mock smoke — 2 rounds, RxnFlow
train (finite) → sample+routes → bridge (rgfn, mock) → standard dataset. Finalized
dataset conformant, `has_route=1`, `routes.jsonl` = 12/12 with real building blocks +
1–2 reaction steps. **Real run submitted: job 69518** (`al_rxnflow_6td3`, balam004,
3 rounds × 32-mol GPU 6TD3 docking), running alongside the matched RGFN GPU run (69517).
Results → `Logs/016`.

**COMPLETE (job 69518, 2026-07-01 00:40, 51 min, exit 0:0).** All 3 rounds ran on the
real GPU docking oracle: 96/96 candidates scored, **all 96 with 2-step synthesis routes**
(`has_route=1`), conformant standard dataset. Best `dvina` −4.22, median −1.19, 30/96
(31%) ≤ −2.0, diversity 0.86–0.90. Training stayed finite over all 900 steps (the
`hb_edited`→`real.txt` fix held). **Matched-oracle RGFN GPU run (69517) also COMPLETE**
(3h14m; RGFN in-loop GPU training ~1h/round vs the bridge baselines' ~15min), giving a
clean same-oracle GPU three-way in `Logs/016`. Two findings: (1) on glue score all three
are same-league (RGFN median/mean −2.14 best, FragGFN best single hit −4.86 & 54% past
−2.0, RxnFlow behind) — RGFN does not out-dock the non-synth foil; (2) the real split is
drug-likeness — RGFN & FragGFN both bloated/low-QED (MW 665–720, QED 0.11–0.15, RGFN
Lipinski 2/96), only RxnFlow is both synthesizable and drug-like (MW 489, QED 0.36).
Lead: add a QED/property term (or tighter block lib) to RGFN. Full numbers in `Logs/016`.

## GPU-oracle-in-loop fix + pre-flight gate (exp 014, job 69517, 2026-07-01)

The first *complete* GPU-oracle active-learning run (job 69517: 3 rounds, |D|
408→483) required fixing a real bug that three prior attempts hit — round-1
docking returning all `no_pose`:

- **Root cause: GPU-memory contention.** QuickVina2-GPU runs in a subprocess and
  allocates GPU memory via OpenCL; after GFN training, torch's CUDA caching
  allocator holds the whole A100 (free VRAM 40 GB → 1.1 GB), so docking gets 0
  poses. Node-independent (failed on balam004 too, which passed the OpenCL probe).
  Confirmed by a controlled login-A100 reproduction (fresh→docks; torch-holds→0
  poses; `empty_cache()`→docks).
- **Fix** (`glue/active_learning/loop.py`): `_free_torch_gpu_cache()` calls
  `gc.collect()` + `torch.cuda.empty_cache()` after training/sampling and before
  `oracle.score()` each round (prints free VRAM). Guarded; no-op without CUDA.
- **Pre-flight dock gate** (`experiments/active_learning/6td3/preflight_dock.py`,
  wired into `submit_al_6td3_gpu.sh`): docks 2 seed molecules at job startup and
  exits non-zero if 0 poses — catches a genuinely bad node in ~40 s instead of at
  round-1 (67 min). The OpenCL context probe alone was insufficient (balam009
  passed it but couldn't dock). `--exclude` now lists balam008,balam009.

Verified: the completed run freed ~39.7 GB before docking each round and added ~25
molecules/round; docking fell to ~1 min/round (<2% of the loop) vs entry-012's ~31
min (33%). See `Logs/014`. All changes are login-node smoke-tested; the full loop is
now confirmed on a real 3-round Balam run.

## 2026-07-03 — Fixed-reward docking matrix: per-step docking for every generator (Logs/021)

Extended the fixed-reward benchmark to run **per-step GPU docking as a fixed reward for
ALL FOUR generators** (previously RGFN-only), on the shared SMALL library. New structure:

- **`glue/proxies/oracle_reward_proxy.OracleRewardProxy`** — the missing generic adapter
  turning any `glue.oracles.GlueOracle` into an in-loop GFN reward (`clip(-raw/norm)` +
  per-step `torch.cuda.empty_cache()` so a GPU docking oracle isn't starved by torch during
  training — the Logs/014 fix moved into the reward path, the guard the raw upstream
  `DockingMoleculeProxy` lacks). Registered in `glue/proxies/__init__.py`. Symmetric third
  case next to `LearnedGlueProxy` (AL surrogate) and `ExampleGlueProxy` (template).
- **`glue/oracles/docking_seh_oracle.DockingClpPOracle`** — one-line ClpP target subclass;
  added to `scripts/score_batch.py` `ORACLES` as `docking_clpp`.
- **Baseline cross-env bridge** (the "attempt baselines too" work): `DockingBridgeReward` in
  `validation/generators/{fraggfn,rxnflow}/fixed_reward.py` + a `reward.type: docking` branch;
  `DockingBridgeProxy` (SCENT-clone `CachedProxyBase`) in
  `validation/generators/scent/docking_bridge_proxy.py`. Each docks a step's batch by shelling
  to `conda run -n rgfn score_batch.py` (benchmark 69691 showed ~10-13% amortized overhead),
  caches per canonical SMILES, `empty_cache` first. `ScentFixedRewardRun`/the FragGFN/RxnFlow
  runners now also write the raw dvina as a `raw_score` provenance column (backward-compatible).
- **Configs** (`configs/glue/fixed_reward_{6td3,clpp,drd2_stdlib}.gin`; `validation/configs/
  {fraggfn,rxnflow}_{6td3,clpp}_docking_fixed.yaml`, `scent_{6td3,clpp}_fixed.gin`,
  `rxnflow_drd2_fixed_stdlib.yaml`) — all on `glue_standard_v1`, β=4, 400 iters, and **lean
  train_metrics dropping `TanimotoSimilarityModes`** (it re-scores `%train_proxy` every
  eval step → would re-dock). Run-dir names follow `<gen>_<system>` so
  `submit_aizynth_fourway.sh SYSTEM=6td3|clpp` aggregates unchanged.
- **Orchestration**: `experiments/fixed_reward/{6td3,clpp,drd2_stdlib}/*.sh`, parameterized
  `baseline_docking/submit_{fraggfn,rxnflow,scent}_docking.sh CONFIG`, master
  `submit_matrix.sh`, benchmark + smoke dirs.

**Verified**: benchmark (69691) + 2 end-to-end Balam smoke jobs — RGFN in-env 6TD3 (69692) and
FragGFN cross-env (69693) both COMPLETED with conformant candidates and 100% per-step dock
success while torch trains. Imports/gin-parse/py_compile all pass. **NOT YET**: the full matrix
launch (awaiting user go-ahead) and a `git` commit. rgfn/ + upstream configs untouched.

## 2026-07-06 — Oracle-call counting + random-acquisition arm (Objective 1, Fig. 7 curve)

Instrumentation for the top-k-vs-oracle-calls curve reviewers ask for
(`[bengio2021gflownet]` Fig. 7; RESEARCH_CONTEXT Objective 1). The roadmap flags this
data as *"cannot be reconstructed later"* — so it is now recorded for **every** AL run,
regardless of system/oracle/arm. No `rgfn/` edits; the random arm reuses upstream
`RandomSampler` + `UniformPolicy` verbatim.

- **`glue/active_learning/acquisition_trace.py`** — new. `AcquisitionTrace` writes
  `<run>/active_learning/oracle_calls.csv`, one row per round: cumulative oracle calls
  (molecules *submitted* to O, docking failures included), the running Top-K
  `mean`/`best`, and the best molecule's SMILES (so the actual top candidate at each
  budget is recoverable without re-reading the per-round dataset dumps). Honours
  `oracle.higher_is_better` (correct for a future higher-is-better oracle, unlike
  `OracleLabeledDataset.top_k` which hardcodes ascending).
- **`glue/active_learning/loop.py`** — (a) records the trace each round + a **round-0
  baseline** row (Top-K of seed `D_0` at 0 new calls) so both arms anchor at an
  identical start; forwards `al_oracle_calls_*` / `al_topk_*` to the metrics logger.
  (b) New `acquisition` param (`"policy"` | `"random"`). `"random"` swaps the query
  sampler for `RandomSampler(UniformPolicy, env)` — same reaction action space, so the
  baseline stays synthesizable-by-construction — and **skips proxy-fit + GFN-training**
  entirely (the Fig. 7 control: only *how the batch is chosen* differs). `_query_sampler()`
  selects the arm; `_sample_query_batch` (route extraction) is unchanged between arms.
- **`scripts/active_learning.py`** — new `--acquisition {policy,random}` flag → binds
  `ActiveLearningLoop.acquisition`, so one config drives both arms.
- **`validation/harness/acquisition_curve.py`** — new. Reads `oracle_calls.csv` across
  seeds × arms (files self-describe arm+seed), aggregates mean±std per oracle-call
  checkpoint, writes the aggregated CSV + the curve PNG (Okabe–Ito, ±std band). Lives in
  `validation/` (reads only `glue/`-written files — one-way dep rule intact); imports
  nothing from `glue`.
- **`configs/glue/active_learning_6td3_gpu_curve.gin`** — thin `include` of the GPU 6TD3
  config, overriding `n_rounds=10` (320 oracle calls / 10 points) and `top_k=100`.
- **`experiments/active_learning/6td3/{submit_curve_6td3.sh,launch_curve_6td3.sh}`** —
  parameterized submit (`ACQ`/`SEED` env) + launcher for the 3-seed × 2-arm campaign.

**Verified** end-to-end on the login node with the mock-oracle AL config
(`active_learning_mock.gin`), both arms: policy arm runs the full fit→train→sample→score
loop; random arm correctly skips fit/train (no `fit_proxy`/`train_gfn` phases) and samples
uniformly; both log identical oracle-call budgets (16→32) with round-0 baseline;
`AcquisitionTrace` unit tests (lower/higher-is-better, NaN drop) pass; the aggregator reads
both traces → curve CSV+PNG; gin parses the curve config with overrides confirmed
(n_rounds=10, top_k=100). **NOT YET**: the real 3-seed × 10-round × 2-arm 6TD3 GPU
campaign (ready to launch via `launch_curve_6td3.sh`); `git` commit.

**Addendum (pilot, Logs/023, jobs 69945/69946).** Single-seed real-oracle pilot on 6TD3 ran
both arms to COMPLETION and validated the trace on the real GPU oracle; RGFN beat random on the
top-k-mean-vs-oracle-calls curve (−0.68 vs −0.14 over 96 calls). It surfaced a **run-dir
collision**: two arms of the same config launched together resolved `run_name` to one
timestamped dir (second granularity) and clobbered each other's rewrite-each-round artifacts
(the append-only `oracle_calls.csv` survived). **Fixed** in `scripts/active_learning.py`:
`run_name = <config>/<acq>_seed<seed>/<timestamp>` so concurrent arms/seeds never share a dir
(required for the 6-job campaign). Pilot curve committed at
`validation/results/6td3_acquisition_curve_pilot/`.

---

## LSD-Flow phase-1 vertical slice (2026-07-08) — post-hoc hub selection

Built the first slice of **LSD-Flow** (`docs/LSD_FLOW_PROPOSAL.md`): post-hoc extraction of
batchable "synthetic neighborhoods" (hubs) from a trained reaction GFN, split across the two
axes exactly as the proposal §3 prescribes. This is the successor to the removed hub-analysis
pipeline (see the memory note); none of the old code was resurrected.

**Production side — `glue/` (the acquisition primitives; the AL loop imports these):**
- `glue/metrics/lsdflow_flow.py` — the §2 flow recovery in log space:
  `log F_hat(h;x) = logR + logP_B − logP_F(move) − logP_F(stop)`, plus `logsumexp`, the
  median-consensus and total-terminating aggregators, and the reward-free `logZ`-shifted
  visitation estimate.
- `glue/metrics/uncertainty.py` — `U(h)` = population variance of the per-child log-flow
  estimates (the flow-matching residual) + `effective_sample_count`.
- `glue/samplers/lsdflow/` — `records.py` (the model-agnostic `FlowRecord` + the HubDAG-shaped
  duck-type protocols); `dag.py` (`LiteHubDAG`, the lightweight in-loop aggregation; children
  deduped by canonical key so `U(h)` isn't deflated by resampling); `rgfn_extract.py` (the
  rgfn-native trajectory→`FlowRecord` extraction — composes the last reaction's A/B/C
  micro-steps into one move, reads the final stop micro-step as `P_F(stop|x)`, and the C-step
  backward log-prob as the learned `P_B`; shared with the validation RGFN adapter);
  `hub/` (6 strategies — highest_terminating_flow, highest_flow, most_modes, parent_of_topk
  [control], highest_visitation [reward-free], lowest_uncertainty — all `@gin.configurable`,
  protocol-pure, + registry); `molecule/` (topk_reward, prob_weighted [Efraimidis–Spirakis
  weighted-without-replacement], uniform_random + registry); `acquisition.py`
  (`LSDFlowAcquisition`, the AL-facing entry point: trajectories+objective → flat molecule
  batch; `select_grouped` keeps hub→children for the amortized-cost accounting).
- Wired into gin discovery via `glue/samplers/__init__.py` + `glue/metrics/__init__.py`
  (both already on the `glue.registry` path).

**Validation side — `validation/lsdflow/` (analysis; imports `glue/`, never imported back):**
- `adapters/` — `base.py` (`GFNAdapter` ABC + `FlowSample` canonical schema; the §4b
  six-method contract, with the phase-2 enumeration methods raising a clear NotImplementedError
  so an un-wired capability fails loudly); `rgfn_adapter.py` (the in-process RGFN anchor:
  rebuilds objective + the **pure-policy** `valid_sampler` from a gin config + checkpoint,
  loads `last_gfn.pt`, samples, extracts flow records); `registry.py` (declares all four
  targets + env + build status; only RGFN wired); `workers/` (documented placeholder for the
  cross-env SCENT/FragGFN/RxnFlow subprocess workers — phase 3-4).
- `dag/` — `node.py` (canonical stereo-stripped cross-model key, §6); `graph.py` (`HubDAG`:
  wraps `LiteHubDAG` for the strategy-facing duck type, adds a networkx view + per-node stats +
  CSV/JSON/gpickle persistence keyed by model×reward×run); `build.py`.
- `metrics/` — `diversity.py` (Butina modes on ECFP4 @ Tanimoto 0.65 + Bemis-Murcko
  scaffolds, §11); `cost/` (reactions-per-mode PRIMARY: hub batch = depth(h)+k vs independent
  = Σ depth(x_j); amortization-ratio declared for phase 2).
- `harness/` — `config.py` (`LSDFlowRunConfig`) + `run.py` (the vertical-slice driver:
  sample → build DAG → rank hubs under every strategy → run acquisition combos through
  cost/modes → the flow-vs-visitation TB-integrity diagnostic → persist). `matrix.py` (full
  sweep) and `analysis/` (severe tests, hub-coincidence, pareto) are declared, not built.

**Verified on the Balam login node** (GFN inference only, no docking; `rgfn-smoke-env.sh`):
pure-logic unit test of the aggregation + all strategies + acquisition (synthetic records)
passes; the RGFN adapter builds from `configs/glue/fixed_reward_seh_proxy_stdlib.gin` +
the `seh_proxy_stdlib` checkpoint and runs the whole slice end-to-end (300-traj smoke: flow
recovery, `U(h)`, all 6 strategies, acquisition + reactions-per-mode, DAG persistence). One
fix needed: import `Trainer` explicitly in the adapter to register the gin configurable (as
`scripts/*.py` do). **Empirical finding under investigation:** sampled penultimate hubs are
*sparse* — at 300 trajectories only 2/292 hubs had ≥2 terminal children (max 2), because a
hub only accrues a child when a trajectory *stops exactly one reaction later*, and most
trajectories through a hub continue deeper. A 20k-trajectory characterization run is in flight
to see how multichild-hub density scales; this directly bears on whether phase-2
`enumerate_children` (enumerate a selected hub's terminal children rather than waiting for
sampling to hit them) is needed for the thesis. **NOT YET:** git commit; loop integration of
`LSDFlowAcquisition` (needs a small `glue/active_learning/loop.py` change — the current loop
has no pluggable-sampler hook, contra proposal §4a); the matrix/analysis modules; SCENT/
FragGFN/RxnFlow adapters.

**Correction + result (same day, Logs/025).** The 300-traj smoke and 20k run above were on the
wrong checkpoint — `seh_proxy_stdlib/2026-07-02_13-45-03` is a **cancelled 30-iteration** stub,
not the completed run. `seh_proxy_stdlib/` holds three timestamped dirs (jobs 69613/69615/69616);
only **`2026-07-02_14-59-53`** is the completed 5,001-iter model (verified via `metrics['epoch']`
+ `candidates.csv` matching Logs/020). Re-ran the 10k slice on the correct checkpoint: **613/8183
multi-child hubs (max 9), reactions-per-mode 3.55 hub vs 6.24 independent (~40% saving).** Severe-
test caveats: 91% of multi-child hubs sit at the `max_num_reactions` boundary (forced stop); flow
ranking ties the `parent_of_topk` control on cost (3.55 vs 3.47); flow-vs-visitation correlation
~0 (0.086). Added a checkpoint-provenance caution to `validation/lsdflow/README.md`. Results:
`validation/lsdflow/results/seh_rgfn_pilot/`.

**Phase-2 exhaustive enumeration (same day, Logs/025 addendum).** Added the §4b
`enumerate_children` path: `glue/samplers/lsdflow/rgfn_enumerate.py` (DFS the env's A→B→C
action spaces → all one-reaction products → rebuild each `hub→…→stop→Terminal` micro-step
trajectory with the env's own action spaces → reuse `extract_flow_records`, so enumerated flow
terms == sampled), `RGFNAdapter.enumerate_hub_children`, and harness flags
`--enumerate-top-hubs` / `--enumerate-max-children` / `--from-records` (reuse a persisted DAG
so enumeration skips the slow ~30–40 min RGFN re-sample). Persisted-records now carry
`hub_stereo_key`/`child_stereo_key`. **Validated:** a depth-0 fragment hub enumerates to 498
one-reaction children / 309 modes (sampling saw 8), 100% sampled-child recovery — resolving the
boundary-artifact caveat for the cheapest hubs. Guard: children whose `P_B` the env can't invert
(max-depth boundary / stereo-dependent disconnection) are skipped, never assigned a fabricated
`P_B=1`. **Limitation:** hubs with stereocenters reconstructed from the stereo-stripped key
enumerate to 0 (stereo-dependent disconnection fails) — a fresh stereo-keyed DAG fixes it (the
committed 10k `records.csv` predates the stereo columns). Enumeration artifacts:
`validation/lsdflow/results/seh_rgfn_pilot/enumeration.json` + `enumerated_records.csv`.

**Deeper-hub result (Balam job 70140, fresh 30k stereo-keyed DAG, 3h39m, Logs/025).** A fresh run
reconstructs hubs from live stereo SMILES → **fully fixed** deeper-hub enumeration (all 12 hubs
100% sampled-child recovery, incl. depth-3; the limitation was stereo-stripping, not the
max-depth boundary). Landed the reactions-per-mode amortization — costed as **one
representative per mode** (`_per_mode_cost`; hub = depth+n_modes, indep = n_modes·(depth+1),
bounded by trajectory length; an earlier all-children/n_modes denominator inflated it to ~44,
fixed): hub/mode ≈1.0 vs indep/mode = depth+1 → ~(depth+1)× saving (depth-1 ~2×, depth-3 ~4×,
depth-0 fragments free); best enumerated sEH 7.1-7.9. Also added: the harness
persists the DAG BEFORE enumeration (a slow enumeration can't lose the multi-hour sample), the
`--from-records` reuse path, interior (depth 1-2) hub targeting, and
`validation/lsdflow/submit_seh_enum.sh` (compute-node submit). Small artifacts committed to
`validation/lsdflow/results/seh_rgfn_enum/`.

**Paper-comparable modes + dropoff funnel (Logs/026).** Redefined a "mode" in
`validation/lsdflow/metrics/diversity.py` to match upstream `TanimotoSimilarityModes` /
`[bengio2021gflownet]`: reward-gated + best-first greedy sphere-exclusion (ECFP Morgan r=3, 2048,
sim 0.7), replacing the structure-only ECFP4/0.65 Butina; `_per_mode_cost` now costs one
representative per hit-mode; config knobs `mode_similarity_threshold` + `mode_reward_threshold`
(+ CLI). New **modular analysis home `experiments/lsd_hubs/`** (one sub-dir per analysis; reuses
`glue/`+`validation/lsdflow/` primitives) with `dropoff/funnel.py` (per-hub filter funnel) +
committed `funnel_seh_70140_results.csv`/`_summary.json`. Finding (30k sEH): the binding gate is
the dominant dropoff (4586 raw → 3% at sEH≥7, 0% at ≥8; Tanimoto dedup gentle 2.3×) — depth-0
fragment hubs are diverse but hitless, depth-3 hubs carry the hits + amortize ~3.6×. Regenerated
the committed `seh_rgfn_{enum,pilot}` acquisition/enumeration numbers under the gated definition.

**SCENT cross-env adapter — LSD-Flow phase 3 (Logs/027).** Wired SCENT into the hub-analysis
harness as the first cross-env target (proposal §4b/§10.3). SCENT can't co-import (its package
is also named `rgfn`, own `scent` env), so the adapter is a subprocess bridge, NOT one imported
class: `validation/lsdflow/adapters/workers/scent_worker.py` runs *in the scent env* (rebuilds
`objective`+pure `valid_sampler` via the `verify_pb_recovery.py` recipe, loads `last_gfn.pt` +
the `guidance_models.pt` P_B sidecar from Logs/024, samples, and emits the canonical
`records.csv`+`visit_counts.json`+`meta.json`); `validation/lsdflow/adapters/scent_adapter.py`
is the in-process client (rgfn env) that shells to the worker under `conda activate scent` with
the scent env's torch-bundled `nvidia/*/lib` on `LD_LIBRARY_PATH` (the `rgfn-smoke-env.sh`
trick, cluster-agnostic) and reads the files back into a `FlowSample` — identical downstream
contract to `RGFNAdapter`, so **the harness itself is unchanged**. The §2 flow-extraction
algorithm is a self-contained copy of `glue.samplers.lsdflow.rgfn_extract` inside the worker (it
can't be imported — its `import rgfn` would resolve to *our* rgfn; SCENT's fork is API-compatible
so the algorithm transfers verbatim). `registry.py` flipped `get_adapter("scent")` from
`NotImplementedError` to live. `validation/lsdflow/submit_scent_seh.sh` is the compute-node
submit (activates the `rgfn` harness env; the worker self-activates `scent`). **Validated
end-to-end** on the 5,000-iter patched sEH checkpoint (logZ 74.33 = trained value; sidecar loads,
P_B exact): N=2k → 19 multi-child hubs, amortization 2.49 vs 3.92 rxn/mode. **Not built for
SCENT yet:** phase-2 `enumerate_hub_children` (frozen-dynamic-library enumeration) raises
`NotImplementedError`; the hub-coincidence analysis (`validation/lsdflow/analysis/`, §8) is the
next build. Full 30k run = job 70179.

**SCENT frozen-library sampling + enumeration + recipe logging (Logs/027 addendum).** Extended the
SCENT cross-env adapter for the *faithful full* model. (1) **Freeze:** building SCENT from a
checkpoint leaves it restricted to the 418 base fragments (`current_fragments=418`); the worker now
fires `trainer.on_update_fragments_library` from the `fragments_<N>.json` snapshot to grow the env
reactant set + policy embedding + cost proxy to the full trained vocabulary (418 + ~1,600 promoted),
for BOTH sampling and enumeration (`SCENTAdapter.freeze=True` default; `--no-freeze` reverts). (2)
**Enumeration:** `scent_worker.py --mode enumerate` + `SCENTAdapter.enumerate_hub_children` mirror
`glue.samplers.lsdflow.rgfn_enumerate` worker-side on the frozen library — validated exhaustive
(depth-0 hub 3,975 paths, 2/2 sampled recovered at cap 12k; neighborhoods ~10x the RGFN 418-lib
case, so `submit_scent_seh.sh` now runs enumeration with ENUM_MAX=12000). (3) **Recipe logging:**
`validation/generators/scent/recipe_logging.py` monkeypatches `DynamicLibrary` (clone pristine) to
record each promoted fragment's min-reaction synthesis route (reaction SMARTS + reactants + product)
into `fragments_<N>.json`; wired as `run_scent_fixed.py --log-recipes`
(`experiments/fixed_reward/scent_seh/submit_fixed_scent_seh_recipes.sh`, job 70180). Feeds the LSD-Flow
cost model's exact nested dynamic-fragment amortization (each distinct promoted fragment charged once
per costed library; reactions primary + SCENT $-cost secondary) and a future chemist "synthesize
these intermediates" view. **The nested-amortization cost model itself is not built yet** — it needs
per-molecule fragment-composition capture from the analysis trajectories (not in `FlowRecord` today).

---

## 2026-07-14 — Full synthesis-route recording (reconstruction / "how to make it")

**Why:** reconstructing a suggested molecule step-by-step was not fully possible — the reactions
*after the hub attach* were not recorded, and the hub's own build route was never captured (only
promoted-fragment recipes existed, in `fragments_<N>.json`). This closes both gaps so the
chemist-facing `experiments/lsd_hubs/campaign/synthesis_routes.py` (Logs/032) can be a lookup, not an
inference.

**What changed (all in `validation/`, additive + backward-compatible):**
- `adapters/workers/scent_worker.py`:
  - added `_reaction_id` / `_reaction_step` helpers (self-contained copies of the recipe_logging
    schema — the worker can't import our-rgfn-bound modules).
  - **enumerate mode:** every `enum_children.json` child now carries `reaction` = the ground-truth
    final hub→child step(s) `{reaction, reactants, input, product}`, built from the **states**
    (`hub_state.molecule` in, `x_state.molecule` out) because the action's `output_molecule` may be
    unpopulated pre-apply in the enumerate path. `enumerate_terminal_children` / `build_child_trajectory`
    now return a `reaction_by_stereo` map.
  - **sample mode:** new `routes.json` = every product molecule's full min-reaction route (same
    schema), keyed by the **stereo-stripped** product SMILES (matching `records.csv`/enum `hub_key`,
    so the join is exact; steps keep raw SMILES). Gated `routes_out`/`RAC` params on
    `extract_flow_records` (enumerate path passes none → unchanged).
- `adapters/base.py`: `FlowSample.routes` field (empty for adapters that don't emit routes, e.g. RGFN).
- `dag/graph.py`: `HubDAG.save` persists `routes.json` (symmetric to `compositions.json`).
- `adapters/scent_adapter.py`: reads `routes.json` into `FlowSample.routes`.

**Verified (debug job 70526, ~3 min):** `routes.json` populated (814 entries, real RGFN templates);
enum child `reaction` field populated with the ground-truth final step; a depth-1 hub's build route
present in `routes.json` (depth-0 hubs correctly absent — they are stock fragments). Schema/assembly
logic unit-tested separately (nesting, intermediates-first ordering, count-once).

**`synthesis_routes.py` wired to consume it (backward-compatible):** `full_route` now builds the hub
(linearizes `routes` which merges `smiles_to_route` + the optional `--routes routes.json`) and uses
the child's logged `reaction` for the final step, preferring ground truth over `hub_reaction_name`.
Both inputs are optional — with neither present it falls back to the old behavior. Verified: a
backward-compat run on the current (un-repopulated) data runs clean (hub still a "buy" leaf, final
step inferred), and a synthetic test confirms the upgrade path (hub + promoted intermediate built,
then the logged final step) and the fallback path.

**Not done / next:** the **current** committed results predate the recording — `scent_seh_70189` has
no `routes.json` and `campaign_enum_seh_70363`'s `enum_children.json` has no `reaction` field; a
re-run (sample 30k ~3–4 h + enumeration ~5.5 h, both exceed debug's 2 h → `compute`) will populate
them, at which point `synthesis_routes.py --routes routes.json` yields fully-grounded protocols with
the hub built.

---

## 2026-07-14 — Fair count-once cost model for the hub-batching campaign (Logs/033)

**Why.** The campaign's reactions/mode double-counted SCENT's promoted dynamic-library fragments.
SCENT's per-molecule `num_reactions` is **fully nested** — it already includes building every
attached promoted fragment (`external/scent/rgfn/gfns/reaction_gfn/reaction_env.py`: seed carries
`fragment.num_reactions`; each attached reactant adds `fragment.num_reactions`; each coupling +1) —
yet `BestCandidateStrategy`/`HubBatchingStrategy` then *added* the fragment builds again via
`_charge_promoted`. Confirmed empirically: for all 245 promoted fragments present as intermediates,
`compositions.json`'s `num_reactions` equals the nested closure cost exactly. Net: the per-molecule
assembly term was ~2× inflated. Separately, best-candidate got no credit for parent scaffolds its
top-N picks share (user request: "count shared hubs once, like intermediates").

**What changed (all in ours; both concerns fixed on ONE model applied to both strategies):**
- `glue/samplers/lsdflow/campaign.py`:
  - new `shallow_couplings(num_reactions, promoted, cost_table)` = `num_reactions − Σ(nested build
    cost of each attached promoted fragment)` → the true assembly-coupling count (verified 1–4, no
    negatives on top-1000). Both strategies now charge couplings, not the nested `num_reactions`.
  - new `HubAssignmentPolicy` ABC + `MostSharedAssignment` (default) + `NoHubSharing` — swappable;
    assigns each best-candidate mode to the parent hub most reused among accepted modes.
  - `BestCandidateStrategy` is now three-pass (select → assign shared hubs → count-once cost) and
    takes `hub_compositions` (to cost parent hubs) + `assignment_policy`. A **prefix-validity filter**
    (`couplings(hub) < couplings(mode)`) keeps a cross-trajectory parent from ever *raising* a mode's
    cost — sharing is provably ≤ no-sharing (asserted in the isolation test). Fragments are charged
    via each mode's full `promoted` (count-once), so a shared hub's fragments are never double-charged.
  - `HubBatchingStrategy` charges `shallow_couplings(hub)` instead of the nested `depth`.
  - The old "disjoint inputs — best-candidate never touches hub data" contract is dropped: for a fair
    cost comparison best-candidate now reads `records.csv` parent hubs + `compositions`.
- `experiments/lsd_hubs/campaign/run_campaign.py`: `_load_candidates` also collects each terminal's
  observed parent hub keys; new shared `build_strategy()` (used by both drivers) wires the model.
- `experiments/lsd_hubs/campaign/sweep_campaign.py`: threads `compositions` through `build_strategy`.

**Verified (login CPU).** `py_compile` clean; isolation test asserts sharing ≤ no-sharing.
Regenerated `results/scent_seh/` (50-hub, ~42 s) + `results/scent_seh_1kx200/` (200-hub, ~73 s).
sEH cutoff 0.5, 300 modes: best-candidate **1,479 → 949** (double-count fix, −530) **→ 929**
(accidental hubs, −20); hub-batching **819** (≈unchanged; its used hubs are depth-0 base fragments,
0 couplings). Hub-batching's edge **1.81× → 1.13×**. At loose cutoffs (≥0.75, 200-hub)
best-candidate beats hub-batching on reactions.

**Not changed.** `hub_stats.py`, the diversity-pairs/route-trees/synthesis-routes tools, and the
scaffold-concentration ceiling / chemistry-floor findings are cost-model-independent and untouched.

---

## 2026-07-17 — LSD-Flow documentation reconciliation (docs match the code)

**Why.** The LSD-Flow docs described an earlier design that the code had moved past: several READMEs
+ the proposal + auto-memories listed modules that were removed on 2026-07-11 when the count-once
**campaign** superseded the original per-hub acquisition design — `glue/samplers/lsdflow/acquisition.py`
(`LSDFlowAcquisition`), `glue/samplers/lsdflow/molecule/` (the molecule-selection strategy registry),
and `validation/lsdflow/metrics/cost/{base,reactions_per_mode,registry}.py` (the cost-model ABC +
the simple reactions-per-mode metric). That removal was recorded only in Logs/028's addendum, so the
higher-level docs still read as if the deleted pieces existed — which misled an agent into planning
against phantom modules.

**What the pipeline actually is (single, current design).** Two stages joined by a persisted
flow-record DAG: (1) **sampling / flow extraction** — `validation/lsdflow/harness/run.py` drives a
per-model adapter (`rgfn_adapter` in-process; `scent_adapter` → `scent_worker` cross-env), recovers
the §2 flow (`glue/metrics/lsdflow_flow.py` + `uncertainty.py`), builds the DAG
(`validation/lsdflow/dag/` rich; `glue/samplers/lsdflow/dag.py` `LiteHubDAG`), and persists
`records.csv` + `compositions.json` (+ optional `enumerated_records.csv`); (2) **the count-once
library-cost campaign** — `experiments/lsd_hubs/campaign/*.py` run `campaign.py`'s
`BestCandidateStrategy` vs `HubBatchingStrategy` (with `child_select` / `mode_select`) and score them
on the count-once cost (`validation/lsdflow/metrics/cost/{dynamic_amortization,compute_time}.py`).
The hub-selection strategies (`glue/samplers/lsdflow/hub/`) + flow/uncertainty metrics are retained
as the foundation for the deferred AL acquisition + multi-generator benchmark.

**What changed (docs only; no code touched).** Rewrote `validation/lsdflow/README.md` (current
two-stage pipeline + accurate layout + status), `experiments/lsd_hubs/README.md` (added `campaign/`
to the analyses; fixed the primitive routing), the `glue/samplers/lsdflow/__init__.py` docstring
(campaign strategies are AL-*ready*, not AL-wired), the proposal's §3 layout note + §11 cost
definition (count-once, not the `depth(h)+k` sketch), and the `docs/RESEARCH_CONTEXT.md` Logs/025
row. Added the `verify-state-not-docs` auto-memory. No source files were deleted or moved; the
`.pyc` for the removed modules are gitignored (not tracked).

---

## 2026-07-20 — LSD-Flow acquisition wired into active learning (uncertainty-driven, RGFN in-env)

**Why.** The capstone the campaign primitives were built to feed (Logs/037/038 "next steps"): put
LSD-Flow hub selection *inside* the active-learning loop so we can measure how fast a real lab run
finds high-quality candidates per expensive-oracle (docking) call — with the hub-flow **uncertainty
`U(h)`** as the acquisition's exploration signal (proposal §2 phase-2, §4a).

**What (production `glue/`, model-agnostic).**
- **`glue/samplers/lsdflow/hub/ucb.py` — new `UcbHubStrategy`** (registered as `ucb`). Ranks hubs by
  the explore/exploit score `score(h) = z(reward(h)) + λ·z(U(h))`, each term **z-scored across the
  round's eligible hubs** so `λ` is a scale-free relative weight. `reward_fn` (exploitation) and
  `uncertainty_fn` (exploration) are swappable by name (small registries) or callable — the
  researcher's explicit modularity ask. Overrides `rank()` (the score is set-relative, not per-hub);
  `score_breakdown()` emits per-hub provenance (raw + z + blended).
- **`glue/samplers/lsdflow/acquisition.py` — `LSDFlowAcquisition` RE-INTRODUCED** (it and a
  `molecule/` registry were deleted 2026-07-11 when the count-once campaign superseded the *first*
  acquisition design; this is the AL-wired version the proposal §4a always intended, now built **on**
  the campaign primitives, not replacing them). A drop-in batch-selection sampler with two arms:
  `hub_batching` (sample → `LiteHubDAG` → UCB rank → walk hubs best-first, enumerate each hub's
  one-reaction children scored by the reward-generator `M`, pre-select-K + `free_frag` child policy,
  accept reward-gated + Tanimoto-diverse **modes** up to a per-round budget) and `best_candidate`
  (top-`M` sampled terminals under the **same** hit-bar + diversity filter — the control). Returns a
  flat SMILES batch + routes + dual accounting (**oracle calls** = docked modes; **reward-gen
  calls** = `M` evaluations on enumerated children) + `avg mols/hub` + per-hub provenance.
  pre-select-K=20 is **wired but inert on RGFN** (no dynamic library); it becomes load-bearing on the
  SCENT path.
- **`glue/active_learning/loop.py`** — additive: `acquisition` now accepts `hub_batching` /
  `best_candidate` (alongside `policy` / `random`); the two LSD-Flow arms fit `M` + train the GFN
  (like `policy`) then delegate to `LSDFlowAcquisition`; the fit→train→sample→dock→grow structure is
  unchanged (§4a v1). Prefers the pure-policy `valid_sampler` for a clean `U(h)`. Writes a per-round
  `hub_acquisition_round_NNN.csv`.
- **`glue/active_learning/acquisition_trace.py`** — extended `oracle_calls.csv` with
  `reward_gen_calls_{round,cumulative}` + `n_hubs_used` + `avg_mols_per_hub` (both cost axes on the
  Fig.7 substrate). Backward compatible (blank for policy/random).
- **`scripts/active_learning.py`** — `--acquisition` gains `hub_batching` / `best_candidate`.
- **`validation/harness/acquisition_curve.py`** — arm colours/labels for the four arms.
- **`configs/glue/active_learning_6td3_lsdflow.gin`** (new) — RGFN in-env anchor: 6TD3 GPU
  differential docking, 10 rounds, 100 modes/round, hit bar **−1.5** (the glue/decoy discrimination
  cut, Logs/002; Youden −1.58, Logs/006), λ=1, similarity 0.5. Plus
  `experiments/active_learning/6td3/{submit_al_6td3_lsdflow.sh,launch_lsdflow_6td3.sh}`.

**Why RGFN in-env first (build order).** The `glue/` selection primitives can't be imported in the
`scent` env (its `import rgfn` resolves to SCENT's fork — the existing `scent_worker` *vendors* the
extraction/enumeration for exactly this reason), so the clean architecture is the existing harness
pattern: SCENT produces enumeration files in its env, `glue` selection runs in the `rgfn` env on
them. Validating the whole acquisition + curve **in-env on RGFN** (proposal §10 step 2) locks the
modular contract with zero cross-env friction; the SCENT headline run (where pre-select-K=20 is real)
reuses the *same* `LSDFlowAcquisition` via that file bridge — the next build step.

**Compute-time accounting (Logs/039 requirement).** Per-component acquisition wall-clock is measured
live and CUDA-synchronized: `rgfn_enumerate.enumerate_terminal_children` gained a backward-compatible
`timing`/`sync` hook splitting `enumeration_s` / `reward_gen_s` / `flow_extract_s`;
`LSDFlowAcquisition` times `sampling_s` / `flow_extract_s` / `hub_rank_s` / `mode_select_s` around
those, attaches the breakdown to `AcquisitionResult.timing`, and the loop writes it to
`active_learning/acquisition_timings.csv` (one row per round: arm, total, six components) + into the
round metrics. This sits **inside** the loop's `sample_batch` `PhaseTimer` bucket; docking is the
separate `oracle_score` phase — so the run reports both call-count axes (oracle vs reward-gen) AND
the wall-clock breakdown behind "how much longer hub-batching works vs best-candidate."

**Verified (login node, `~/bin/rgfn-smoke-env.sh`).** `py_compile` all touched files; `import glue`
+ `ucb` registered + `LSDFlowAcquisition` gin-configurable; unit smoke of UCB ranking (λ=0 →
best-reward hub first, λ=100 → highest-`U(h)` hub first, λ=1 → z-scored blend) + `score_breakdown` +
the mode hit-bar/diversity gate; timing plumbing present; both configs parse; `bash -n` on all submit
scripts.

**Two compute-node smokes, two real bugs caught (the reason to smoke the full loop, not just imports).**
Both ran clean end-to-end (exit 0) but exposed correctness bugs that compile/import/unit-smoke all
passed:
1. **Smoke 71011 → 0 modes (sampled-`U(h)` starvation).** `U(h)` was computed from *sampled* children
   with a ≥2-child gate; sampling under-counts a hub's children (Logs/025), so the ranker's eligible
   set was empty. **Fix:** two-stage `_hub_batching` — pick candidate hubs by **visit count** (robust
   to sparse sampling), **enumerate** each, compute `U(h)`/`reward(h)` from the *enumerated*
   neighborhood (matches how the campaign/paper compute it), UCB-rank, mode-select. Added knobs
   `n_candidate_hubs` / `min_hub_visits` / `min_hub_depth` / `max_hub_depth`.
2. **Smoke 71025 → still 0 modes (units mismatch).** Two-stage ranking worked (`U(h)` well-defined
   ~22–28), but the `−1.5` **real-ΔVina** hit bar was compared against `M`'s **standardized** output
   (`LearnedGlueProxy.fit` standardizes labels). Best child std `−1.22` = real `−2.35` (a hit) but
   `−1.22 ≤ −1.5` is false → everything rejected. **Fix:** `_mode_selector` maps the bar into `M`'s
   space `(thr−label_mean)/label_std` (the loop passes the proxy's fit stats); z-scored ranking is
   invariant to the affine map so only the gate changes. Unit-tested: real `−1.5` → std `−0.27`, the
   std `−1.22` child now accepts, mean/positive reject.

**Measured compute (smoke 71025, the answer to "where does time go").** Acquisition 5 min:
`enumeration_s` 220 s (73%, RDKit child construction — the dominant cost, per Logs/039), `sampling_s`
56 s, `flow_extract_s` 18 s, `reward_gen_s` 5.7 s (proxy scoring is cheap); `train_gfn` ~13 s/iter.
**Full-run sizing set to these rates:** `Trainer.n_iterations` 300→**150**/round (~5.5 h train),
`max_children_per_hub` 2000→**200** + `n_candidate_hubs` **100** (~17 min/round enum), submit
`--time` 11→**13 h**. CONFIRM on the first post-maintenance run.

**Status.** Fixed smoke **re-queued as job 71114** (`hub_batching`, held `ReqNodeNotAvail` — auto-runs
when nodes return; should now yield modes + a full timing breakdown). **Not yet run:** the full 3-arm
× 10-round run (`launch_lsdflow_6td3.sh 42`, hold until 71114 confirms modes) and the SCENT cross-env
path (pre-select-K live).

## 2026-07-29 — SCENT recipe logging ON by default (so queued campaign links inherit it)

**Why.** A promoted fragment's synthesis route is observable **only while training** — once the
dynamic library promotes it, the policy consumes it atomically and it never reappears as a reaction
product. So a SCENT run trained without `--log-recipes` is permanently stuck on the
`min_num_reactions` cost approximation: `fragments_<N>.json` carries no `smiles_to_route`, and
`dynamic_amortization` cannot charge nested fragment-of-fragment builds exactly (Logs/027/028). Only
the two dedicated recipe re-runs (`scent_{seh,drd2}/2026-07-10_*`, jobs 70180/70184) ever passed the
flag; **every publication-scale 5k cell** (`scent_{seh,drd2,6td3,clpp}_5k/seed*`) lacks routes.
`experiments/fixed_reward/scale5k/submit_baseline.sh` was written 2026-07-14, one day *after* the flag
landed (8ca901f, 07-13), and simply never wired it — checked with `git log -S`: the flag has **never**
been removed from anything, and Logs/030 contains no decision to omit it.

**Where the default lives, and why not in the submit script.** SLURM **snapshots a batch script at
submit time** (`scontrol write batch_script <jobid>` proves it), so editing a `submit_*.sh` cannot
reach an already-queued chain link — and the campaign pre-submits every link up front
(`launch_chain.sh`, `--dependency=afterany`). The stored script does `cd $REPO` and runs
`python validation/generators/scent/run_scent_fixed.py`, which is resolved from the working tree **at
job start**. Flipping the default in that file therefore reaches every link that has not started yet,
with no cancellation and no loss of queue position.

- `validation/generators/scent/run_scent_fixed.py` — `--log-recipes` now `default=True`, with a new
  `--no-log-recipes` opt-out (explicit store_true/store_false pair, not `BooleanOptionalAction`, so it
  is Python-version agnostic). The two existing `submit_fixed_scent_*_recipes.sh` still pass
  `--log-recipes` explicitly; `enable_recipe_logging()` is idempotent, so that is a no-op.
- `experiments/fixed_reward/scale5k/submit_baseline.sh` — passes `--log-recipes` explicitly for
  `GEN=scent`. Redundant with the new default, kept so the campaign script states its own intent.
- Same file — a **partial-coverage warning**: when route logging is on and the run dir already holds
  route-less `fragments_*.json` from earlier links, print that this run's final snapshot will mix
  exact routes with `min_num_reactions` fallbacks. Detection is a chunked byte scan (the snapshots are
  ~60 MB each and `state_dict` appends the key *last*, so neither a full read nor a prefix check
  works); ~0.2 s for 240 MB. Prevents a downstream reader from taking "has `smiles_to_route`" as
  "fully routed" — `reconcile_t15.py --min-recipe-fraction` is the gate that measures coverage.

**Why this cannot perturb training** (the point of the review, since the flag lands on live campaign
runs): `recipe_logging.enable_recipe_logging()` wraps two `DynamicLibrary` methods.
(1) `on_end_sampling` — calls the original **first**, then `capture_routes` inside `try/except`;
`capture_routes` only *reads* (`masked_select` builds a **new** `Trajectories` via
`itertools.compress`, verified in `external/scent/rgfn/api/trajectories.py:337`) and it re-uses the
exact same access pattern SCENT's own hook already performs on the same container. It draws no
randomness, and it runs *after* `seed_everything`, so the RNG stream is unchanged.
(2) `state_dict` — adds one key. That method is consumed in exactly **one** place
(`trainer.py:391 → json.dump` into `fragments_<N>.json`); `DynamicLibrary` has **no**
`load_state_dict` and is **not** part of the torch checkpoint (`make_checkpoint` saves only
model/optimizer/lr_scheduler/metrics/replay_buffer), so the resume path cannot see the extra key.
Cost: routes for ~395k seen molecules ≈ 0.2 GB of JSON-equivalent (nodes have 253 GB) and **+0.7 MB**
per snapshot; the 07-10 recipe run finished *faster* than its route-free 07-07 sibling.

**Pre-existing bug found while auditing, NOT fixed here — the dynamic library resets on requeue.**
`DynamicLibrary` state is not checkpointed and nothing restores it, so a chain requeue restarts it
empty. The promoted-fragment counts prove it happened (`chosen_smiles` per snapshot):

| cell | 1000 | 2000 | 3000 | 4000 | |
|---|---|---|---|---|---|
| `scent_seh_5k/seed42` | 400 | 800 | 1200 | 1600 | clean |
| `scent_drd2_5k/seed42` | 400 | 800 | 1200 | 1600 | clean |
| `scent_clpp_5k/seed42` | 400 | 800 | 1200 | 1600 | clean |
| `scent_6td3_5k/seed42` | 400 | 800 | **400** | 800 | **reset ×2** |
| `scent_clpp_5k/seed43,44` | 400 | **400** | 800 | 1200 | **reset ×1** |
| `scent_6td3_5k/seed43,44` | 400 | 800 | 1200 | *(running)* | clean so far |

Consequences: (a) the final promoted library size varies with requeue timing (800 / 1200 / 1600) — a
confound for any cross-seed cost comparison; (b) `FragmentOneHotEmbedding.weights` is pre-allocated to
`418 + max_additional` and indexed by **position**, so after a reset the re-promoted fragments inherit
the trained embedding rows of the *previous* occupants (no crash, and the content-based
`FragmentFingerprintEmbedding.all_fingerprints` is a plain attribute so it rebuilds correctly — but
the one-hot half carries stale identity). A real fix means persisting/restoring the library state,
which changes training behaviour mid-campaign; left as a decision for the researcher.

---

## 2026-07-30 — Mode-definition ablation seam: `RewardOnlyModeSelector` + `mode_selector_factory` passthrough (Logs/054)

**What changed (two small, additive edits).**

1. `glue/samplers/lsdflow/mode_select.py` — added `RewardOnlyModeSelector` (the **diversity-filter
   ablation**: reward gate + exact canonical-SMILES duplicate suppression, no Tanimoto test, and it
   counts what it suppressed via `n_duplicates_suppressed`), and factored the reward gate out of
   `DiverseThresholdModeSelector._passes_gate` into a module-level `passes_reward_gate()` that both
   selectors call. The gate rule (None ⇒ admit all, NaN never passes, orientation-aware) is now
   defined **once**, so the ablated and unablated arms provably differ only in the similarity test.
   `DiverseThresholdModeSelector` is behaviourally unchanged — the method still exists and delegates.

2. `experiments/lsd_hubs/campaign/run_campaign.py::build_strategy` — added an optional
   `mode_selector_factory` passthrough. Both strategy classes in `glue/samplers/lsdflow/campaign.py`
   have always accepted this argument; **no driver had ever used it**. Default `None` ⇒ each strategy
   builds the canonical `DiverseThresholdModeSelector(reward_threshold, similarity)` exactly as before.

**Why it's here rather than in the experiment dir.** The reward-only selector is a production
component (`glue/`), reusable as a control arm by the AL loop's `LSDFlowAcquisition`, which consumes
the same selector seam. The experiment driver (`experiments/lsd_hubs/filter_ablation/`) only *chooses*
selectors.

**Verified.**

- Default path unchanged: re-ran `run_campaign.py` at the campaign operating point (SCENT×sEH anchor,
  τ=7.0/cutoff 0.5, free_frag + prebuild-K 20) before and after the edit — `summary.json` identical and
  both `curve_*.csv` **byte-identical**.
- All **ten** `build_strategy` callers import cleanly (`sweep_campaign`, `tau_similarity_surface`,
  `preselect_sweep`, `batch_size_distribution`, `dump_frontier_smiles`, `reconcile_t15`,
  `s3gfn_frontier`, `matrix16/gate_curve`, `matrix16/tau_curve_all_generators`, `run_campaign`); every
  one passes keyword arguments after `comps`, so a defaulted keyword is invisible to them.
- Selector unit checks: reward-gate parity with the existing selector across `{7.0, None, −1.5}` ×
  `{8.0, 7.0, 6.9, NaN}` and both orientations; accepts a near-identical molecule; rejects a re-spelled
  duplicate, a below-gate molecule, an unparseable string, and an empty string.
- The ablation driver self-checks the invariant that matters: at the operating point every member of
  the delivered library must re-qualify under the canonical definition (300/300 for both strategies).

**Not done.** No config/gin surface for the new selector (nothing needs it yet), and the AL loop was
not switched over — `LSDFlowAcquisition` can pass the factory when a no-filter control arm is wanted.

---

## 2026-07-30 — INCIDENT: worktree `external/` symlinks were committed and destroyed the clones

**What happened.** The hub-ordering ablation (Logs/053) ran in a git worktree. To give the worktree
access to the upstream baseline clones, `experiments/lsd_hubs/matrix16/link_worktree_data.sh` creates
`external/<repo>` symlinks pointing at the main checkout. A `git add -A` in that worktree committed
six of them (`RxnFlow`, `gflownet`, `multiaiz`, `s3gfn`, `scent`, `sparrow`). Merging the branch then
checked those links out **in the main checkout**, where they pointed at themselves — and git removed
the real clone directories to make room. `external/` fell from ~300 MB to 50 KB, breaking the
`scent` / `rxnflow` / `fraggfn` / `s3gfn` conda envs, whose editable installs (`*.pth`) point into
those paths.

**Why the ignore rule missed it.** The rule was `external/*/`. A trailing slash matches **directories
only**, and a symlink is not a directory — so the links were never ignored. Replaced with a
path-shaped rule that cannot be evaded by file type:

```
external/*
!external/setup_*.sh
```

**Recovery (complete, verified).**
1. Untracked + deleted the six symlinks; fixed `.gitignore` (commit `7b7df2c`).
2. Re-cloned all six at the pinned refs from `external/setup_*.sh`
   (scent `af1fee5`, gflownet `da99940`, RxnFlow/s3gfn/sparrow/multiaiz at `main`).
3. **Re-cloning does not restore downloaded artifacts.** The sEH proxy weights
   (`cache/bengio2021flow_proxy.pkl.gz`, fetched from GitHub at first use) live *inside* the clones
   and were lost. Compute nodes have no internet, so this fails the job at startup with a
   `ConnectTimeout`. Restored into all three copies that need it:
   `external/scent/rgfn/gfns/reaction_gfn/proxies/cache/`,
   `external/gflownet/src/gflownet/models/cache/`, `external/RxnFlow/src/gflownet/models/cache/`.
   (`fpscores.pkl.gz` was unaffected — it is placed in the env's site-packages, not a clone.)
4. Verified by re-running the real SCENT enumeration worker on a compute node (job 72014): output is
   **byte-identical** to the same hubs enumerated before the deletion (474 and 1,399 children).

**No jobs were harmed** — every queued `c5_*` campaign job was still PENDING through the ~11-minute
window, confirmed via `sacct`.

**For future agents.** Never `git add -A` in a worktree that has run `link_worktree_data.sh`; stage
paths explicitly, or confirm `git status --porcelain` shows nothing under `external/`. If a clone
ever has to be re-created, remember it carries first-use downloads that the setup scripts fetch but
`git clone` does not.

---

## 2026-08-14 — New benchmark baselines: REINVENT 4 (Phase 1a) + shared competitor plumbing

**Why.** The external comparison rests on a single route-less baseline (S3-GFN). Adding more
entrants — REINVENT 4 and Saturn (route-less, S3-GFN's class), SynFormer (reaction-aware, the cell
that isolates the *flow field* from mere reaction-grounding), and TANGO — is phased work; this entry
covers Phase 1a and the plumbing every later phase reuses. Surrogate targets only (sEH, DRD2), N=500
pools, 3 seeds, deliverable = 100 modes.

**New — shared, generator-agnostic**
- `experiments/lsd_hubs/campaign/mode_saturation.py` — recovers the pre-flight that Logs/056 ran from
  scratch and never committed, and promotes it to a **gate**: MultiAiZ is ~2.25 h per N=500 pool and
  its cache key is the pool, so a pool that cannot reach 100 modes must be caught in seconds, not
  after the spend. Regression-pinned: reproduces Logs/056's published 41.2% mode rate (206 modes at
  n=500) on the seed-42 S3-GFN sEH pool exactly.
- `experiments/lsd_hubs/campaign/submit_competitor_routes.sh` — `submit_s3gfn_replicate_routes.sh`
  generalized to any route-less entrant (`GENERATOR` × `TARGET` × `SEED`), saturation gate in front.
  `build_s3gfn_pools.py` needed no change; it was already generator-agnostic (only its name is
  S3-GFN-specific).

**New — REINVENT 4 adapter** (`validation/generators/reinvent/`, `validation/configs/reinvent_*.yaml`,
`experiments/lsd_hubs/campaign/submit_reinvent.sh`, `external/setup_reinvent.sh` replacing the stub).
Follows the S3-GFN adapter shape exactly: per-adapter frozen-reward copy, own env, pool crosses to
`rgfn` by subprocess, `has_route=0`.

**Three upstream facts that cost time and are recorded so they cost it once.**

1. **v4.5.11 cannot be seeded through its own interface.** `Reinvent.py` reads `seed` from the TOML
   but gates the call on the *CLI* flag and passes the *config* value, while `ReinventConfig` is
   `extra="forbid"` with **no `seed` field** — so a TOML carrying `seed` is a hard ValidationError,
   `input_config.get("seed")` is always `None`, and `set_seed(None)` returns immediately. Every
   invocation silently fails to seed. `validation/generators/reinvent/_seeded_launcher.py` seeds and
   then defers to `main_script()` verbatim. Without it our three replicates per cell would have been
   real but irreproducible.
2. **Install from the lockfile; `install.py` does not exist at this tag.** v4.5.11 documents
   `pip install -r requirements-linux-64.lock` then `pip install --no-deps .`. Also read the
   **lockfile** for what is installed, not `pyproject.toml`: pyproject says `torch==2.5.1+cu124`, the
   lockfile pins **cu121** (plus `numpy==1.26.4`, Python 3.10). The setup script now asserts the
   installed torch matches the pyg find-links target, since a mismatch would otherwise surface as a
   `torch_sparse` import error inside a job hours later.
3. **The tag is pinned for packaging reasons, not scientific ones** (REINVENT's RL is DAP in every
   4.x release): `main` pins `torch==2.12.0` with no prebuilt `torch-sparse` wheel — which
   `bengio2021flow` imports at module level — requires `numpy>=2` (breaks the legacy DRD2 sklearn
   pickle), and moved the priors to Zenodo (a download that fails on compute nodes).

**Two environment constraints now in force for all later phases.**
- **New conda envs go on `/scratch`, addressed with `conda run -p`**, never `miniconda3/envs`:
  `/home/markymoo` is at ~95 G of its 110 G quota with `conda clean -a` reporting nothing to reclaim.
- **Clone shallow at the tag** (`--branch <tag> --depth 1`). REINVENT4's full history is 1.49 GB of
  git objects because every prior model is versioned in it.

**Two more upstream constraints found by the first real run, both now encoded:**

4. **`max_score` must be `<= 1.0`** — `RLConfig` rejects anything larger, so the "unreachable score"
   trick does not work. It is not needed either: `SimpleTerminator` fires on
   `step > min_steps and score >= max_score`, and `step` never exceeds `max_steps`, so
   **`min_steps == max_steps` is what actually guarantees a fixed budget**. The score bound is
   belt-and-braces.
5. **`gflownet --no-deps` leaves `omegaconf` missing**, and `gflownet/__init__.py` imports it — so
   *any* `from gflownet.models import bengio2021flow` fails. It is not in REINVENT's lockfile, and
   our own driver needs it too. `setuptools<81` is also required or several bundled REINVENT
   components fail to import and the plugin registry silently comes up short (58 vs 59 components),
   which would make the discovery check weaker than it looks.

**Verified.** Statically: `bash -n`, `py_compile`, no `__init__.py` under `plugins/` (which would
break namespace-package discovery), repo-root resolution from the plugin file, and both generated
TOMLs parse with no key `ReinventConfig` forbids. In-env: the setup script's own four checks (sEH
MPNN loads from the pre-placed cache; the component is *registered and typed*, not merely importable;
numpy/torch versions; and real molecules scored through the component with NaN — not 0 — for invalid
input), and the script re-runs idempotently. End-to-end: full RL -> checkpoint -> sampling from the
trained agent -> ingest, conformant `has_route=0` dataset, on **both** sEH and DRD2.

**Two results from that validation worth keeping:**

- **The seeding fix demonstrably works.** Two seed-42 runs produce a **bit-identical** 40/40 pool;
  seed 43 shares **0/40** with them. So the three replicates per cell are both reproducible and
  genuinely independent — which is exactly what would have been silently false without
  `_seeded_launcher.py`.
- **The DRD2 oracle is environment-invariant.** It is an SVC pickled with sklearn 0.23, so every env
  raises `InconsistentVersionWarning`. Measured across `reinvent4` (sklearn 1.7.2), `rgfn` (1.8.0),
  `fraggfn` (1.7.2, rdkit 2026.03.3) and `scent` (1.2.2): **identical probabilities to 12 decimal
  places**. This matters beyond REINVENT — each generator runs in its own env, so a version-dependent
  unpickle would have meant every entrant optimizing a slightly different DRD2, invisibly. Re-run the
  check if the pickle is ever regenerated.

**Still open:** the first full-scale cell (`reinvent_seh` seed 42, job 73605) and everything
downstream of it (saturation gate -> MultiAiZ -> frontiers). Saturn, SynFormer and TANGO not started.

### 2026-08-14 (same day, later) — Phase 1a result + Phase 1b: Saturn

**REINVENT 4 sEH seed 42 ran end-to-end** on an A100 (job 73606, `-p debug`): 1000 RL steps in
**21.2 min** (~1.1 s/step), 2,000 unique valid candidates sampled from the trained agent, ingested
conformant with `has_route=0`. The mode-saturation gate passes and shows the baseline is **not a
strawman**: 1,860 of 2,000 clear the 7.0 gate, and at N=500 the pool holds **201 modes (rate 0.402)**
against S3-GFN's 206 / 0.412 on the same metric — the two route-less entrants are near-identical in
mode density. 100 modes are reachable from 250 candidates. Downstream (MultiAiZ → both frontiers) is
job 73610.

**Saturn adapter built** (`validation/generators/saturn/`, `validation/configs/saturn_*.yaml`,
`experiments/lsd_hubs/campaign/submit_saturn.sh`, `external/setup_saturn.sh`). Validated end-to-end
on both targets; measured **~19 oracle calls/s**, so a full 10,000-call cell is **~10 min**.

`fixed_reward.py` is byte-for-byte the REINVENT copy — verified by comparing ASTs with docstrings
stripped, not by eye. Reward parity is now measured on both targets across five envs: the sEH MPNN
returns 0.0273 for ethanol in `saturn`, `reinvent4` and `s3gfn` alike, and the DRD2 pickle agrees to
12 decimal places in `saturn` / `reinvent4` / `rgfn` / `fraggfn` / `scent` despite spanning sklearn
1.2.2 → 1.8.0. Both checks are now assertions inside `setup_saturn.sh` rather than notes.

**Five upstream facts about Saturn, all found by reading source or by a failing smoke:**

1. **The hash Saturn's own README pins for its paper (`fee0179`) is BROKEN.** Its
   `reinforcement_learning.py` reads `configuration.reinforcement_learning.margin_threshold`, which
   `ReinforcementLearningParameters` does not define, so `ReinforcementLearningAgent` cannot be
   constructed — goal-directed generation cannot run at all. Checked across refs: `fee0179` is the
   *only* one whose RL module mentions `margin_threshold`, and *no* ref defines it, i.e. the line was
   removed right after and that commit caught the repo mid-edit. **We pin `de5cd7f`** (the TANGO
   pre-print hash), the next published-paper pin from the same authors, which runs — and which Phase
   3 needs anyway, so one clone and one pin now serve both arms. `setup_saturn.sh` grew a check that
   every attribute the RL module reads off the dataclass actually exists, so this class of bug fails
   at setup rather than hours into a job.
2. **`ReinforcementLearningAgent.__init__` gained a leading `logging_frequency`** between the two
   hashes. The driver now passes every argument by keyword; a positional call would have silently
   bound the log path to the frequency.
3. **`Oracle.construct_oracle` does `OracleComponentParameters(**component)`**, so `components` must
   be plain dicts — passing the dataclass the signature advertises is a TypeError.
4. **A C compiler is a RUNTIME dependency.** Mamba's layer-norm goes through Triton, which JIT-builds
   a launcher stub on first use and dies with "Failed to find C compiler". This cluster has no
   `/usr/bin/gcc` and `module load gcc` does not populate PATH non-interactively, so `gcc_linux-64`
   goes into the env — which is what Saturn's README recommends anyway. `conda run -p` exports `CC`.
5. **Saturn seeds correctly** via `set_seed_everywhere` — no launcher needed, unlike REINVENT.

**Deliberate deviation from upstream `setup.sh`: torch 2.1.0+cu118, not 1.12.1+cu113.** Two
independent constraints, neither about Saturn's science: `bengio2021flow` needs a prebuilt
`torch_sparse`, and `mamba-ssm`/`causal-conv1d` publish wheels for cu118/cu122 across torch 1.12–2.3
but **none for cu113 at any torch version** — under cu113 both would compile against nvcc 11.3, which
this cluster does not have (that is their documented Issue #1). torch 2.1.0+cu118 is the combination
where every wheel exists, so there is **no CUDA compile at all**. We also skip openbabel and
xtb-python: the only hard import in Saturn's eager oracle chain is `morfeus`, and openbabel is needed
solely by GEAM's docking oracle, which `_stubs.py` stubs (same technique as the S3-GFN adapter's
`unidock_vina` stub).

**Queue etiquette.** Both submit scripts now accept `CELLS="seh:43 drd2:42 ..."` and run the cells
sequentially in one job. The account's QOS caps submitted jobs at 60 and three other agents share it,
so eleven short cells submitted individually would crowd them out for no gain.

### 2026-08-14 (same day, later still) — Phase 2: SynFormer, the reaction-aware entrant

**Why it is the most valuable of the four new baselines.** REINVENT, Saturn and S3-GFN are all
route-LESS. SynFormer generates molecules AS SYNTHETIC PATHWAYS, so its molecules carry a route by
construction (`has_route=1`) exactly as ours do. It is therefore the only cell that separates the two
things the headline conflates — is the advantage the **flow field**, or merely **reaction-grounding**?
No ablation on our own generators can answer it, and `LSD_FLOW_BENCHMARK_PLAN.md` §7 names the gap.
It also skips MultiAiZ entirely (~2.25 h/pool the route-less entrants pay).

**New:** `external/setup_synformer.sh` (replacing the placeholder stub),
`validation/generators/synformer/{__init__,fixed_reward,route_convert,run_synformer_fixed}.py`,
`validation/configs/synformer_{seh,drd2}_fixed.yaml`, `experiments/lsd_hubs/campaign/submit_synformer.sh`,
plus a `--route-source external` branch in `sparrow_select_frontier.py`.

**No Enamine licence needed**, despite the README. That caveat governs re-PREPROCESSING; the
preprocessed `fpindex.pkl` / `matrix.pkl` and the trained checkpoint are on HuggingFace, and the
sampler reads only those (confirmed in source — it never touches the raw SDF).

**The route converter is the substantive piece.** The obvious source, the `synthesis` column, is
`Stack.get_action_string()` — postfix tokens naming the leaves and the reactions but **not the
intermediate products**, and a reaction network is precisely a graph of intermediates. Recovering
them by re-running templates in RDKit would reimplement the model's own bookkeeping with a fresh
chance to be wrong. Instead `route_convert.py` patches `StatePool.get_dataframe` to serialize
`Stack.get_tree()`, which already carries every intermediate. Legitimate because the sampler's
workers are `mp.Process` under Linux's default **fork**, so a parent-side patch is inherited (their
CUDA init happens after the fork, which is what makes fork safe). One trap encoded: `get_tree`
appends children by POPPING a stack, so `children[0]` is the LAST reactant — they are reversed before
emitting `reactant`/`fragments`, the difference between a correct route and a silently transposed one.

**`--route-source external` is additive and validated.** It folds a `routes.jsonl` into the same
`{canonical_smiles: [route, ...]}` shape the multiaiz artifact uses, so `build_network`, the MILP and
the pricing are literally the same code path; `is_native` deliberately excludes it, so no recipe
expansion is attempted (every leaf is purchasable). Verified on a synthetic two-molecule fixture
sharing one intermediate: **5 compound nodes, 2 reaction nodes, the shared acid collapsing to a
single node** — i.e. the merge that the whole cost model depends on actually happens.

**Deviations from upstream, both forced and documented.** We do not install `pytdc`: their GA driver
builds `tdc.Oracle("SA")` at import, and TDC self-downloads into `./oracle`, which fails on a compute
node. Our driver reproduces their GA loop verbatim (`sanitize`, `make_mating_pool`, `reproduce`
copied; `crossover`/`mutate` imported unchanged; same population/offspring/mutation settings) against
our frozen reward, and replaces their patience-based early stop with a fixed oracle budget so the
training budget does not depend on how easy the target is. The starting population is their own
bundled `data/chembl_filtered_1k.txt` rather than TDC's ZINC — no download, and ChEMBL matches
REINVENT's and Saturn's priors.

**Four environment traps, each of which cost a build:**

1. **numpy must be `<2`.** torch 2.1.0 is built against the numpy 1.x C API; numpy 2 makes `import
   torch` warn "Failed to initialize NumPy: _ARRAY_API not found" and then fail at the first
   `torch.tensor(<np array>)` with "Could not infer dtype of numpy.float32" — surfacing while loading
   the sEH weights, far from the cause.
2. **Pin scipy in the SAME pip command as numpy.** Installing numpy alone and letting a later package
   pull scipy yields a numpy-2-built scipy that fails with "numpy._core.multiarray failed to import";
   recovering needs a clean uninstall.
3. **Cap BLAS/OMP threads for the WHOLE script, not just verification.** The login node's
   `RLIMIT_NPROC` is 1024 and OpenBLAS spawns one thread per core, segfaulting the interpreter — and
   this bites during *installation*, because synformer's pyproject uses
   `version = {attr = "synformer.__version__"}`, so `pip install -e .` imports the package.
4. **`ln -sfn TARGET LINK` does not replace LINK when LINK is a real directory** — it creates the link
   *inside* it. The clone ships a tracked `data/trained_weights/`, so the 2.8 GB checkpoint ended up
   one level down and read as "missing". The setup now removes a real directory first, and refuses if
   it holds anything but a `.gitignore`.

**The 6.8 GB lives on `$SCRATCH`** and is symlinked into the clone, because the checkpoint stores its
data paths RELATIVE and resolves them against cwd. Symlinks under `external/` were once genuinely
dangerous here; the ignore rule is now `external/*`, which covers them, and that was re-verified
behaviourally with `git check-ignore` before relying on it.

**Not yet verified:** any SynFormer run at all — the first smoke is in flight. Wall-clock is unknown
and the submit script's 6 h is a guess, because the bottleneck is projection (transformer decode at
search_width 24 over ~200 molecules/generation, 4 GB index per worker), not the oracle.

---

## 2026-08-21 — the route contract: making "no synthesis route" a checked, declared condition

**Problem.** Recovering a synthesis route after a run is over costs a full re-run, and we have paid
that twice. `enum_children.json` `children[].reaction` was omitted by `rgfn_worker` for months and
cost six 24-hour re-enumerations to repair. `routes.json` was worse: empty for **36 of 40** sampled
cell-seeds, and **not repairable at all** — the trajectory is discarded when sampling ends and
`compositions.json` keeps only `num_reactions`, so there is nothing left to reconstruct from. Only
SCENT ever emitted routes. Every non-SCENT cell needs a re-sample before it can feed the competitor
arm.

Neither failure crashed. Both were a **silent default**:

```python
routes: Dict[str, dict] = field(default_factory=dict)
# "Empty for adapters that don't emit routes yet (e.g. RGFN)."
```

Three of four adapters took the default, `rxnflow_worker` and `fraggfn_worker` hardcoded
`json.dump({})`, every run wrote a well-formed `routes.json` containing `{}`, exited 0, and the
harvester promoted it. The defect was never the missing implementation — that is ordinary — it was
that **nothing anywhere asserted the artifact was usable.**

**Change.** A route contract, enforced at write time.

- **`validation/lsdflow/adapters/workers/_routes.py`** (new) — declares per generator whether its
  output is route-bearing, and validates the artifact against that declaration. Writes
  `route_status.json` beside the artifacts (machine-readable, so harvest/frontier can refuse a cell in
  seconds) and raises on unambiguous violation. Stdlib-only, imported by the same sibling convention
  as `_artifacts` so it works unchanged in all four generator envs.
- **"Not applicable" is now a declared answer with a reason, not an empty dict.** FragGFN's move is a
  fragment *attachment*, not a reaction (`docs/LSD_FLOW_PROPOSAL.md` L274), so a FragGFN route is not
  a synthesis plan and SPARROW would price a library nobody can make. That is a different fact from
  "nobody implemented it", and on disk today the two look identical. Now they don't.
- **RGFN route emission implemented** (`rgfn_adapter._collect_routes`) — the machinery already existed
  in `glue/` and was already used by the AL acquisition path; this is plumbing, not new science.
- **RxnFlow route emission implemented** (`rxnflow_worker._collect_routes`) — replaces the
  `json.dump({})` whose comment promised "routes extractable later (`ctx.read_traj`)". That was never
  implemented and "later" was never possible. Everything needed was already in hand: `d["traj"]` is
  the `(state, action)` sequence and every state carries `.smi`.
- **Both emit for EVERY molecule, not just terminals.** A hub is an interior node chosen later by
  `pick_hubs`, and the hub prefix is exactly the half `enum_children.json` cannot supply. SCENT
  already did this (its routes span depths 1–4), and `sparrow_select_frontier.py` L130 measures "64 of
  64 hub_keys" present — so matching it is what makes hub lookup work at all.
- **Keys come from `rgfn_extract._stripped_key`,** the same function the flow records use. A route
  keyed even slightly differently from its hub is *worse* than no route: the lookup returns nothing
  and the caller prices the hub instead of the child, with no error.
- **Canonical schema** (`glue/samplers/lsdflow/route_steps`) for both new emitters, so the competitor
  arm reads one shape across generators.
- **`sparrow_select_frontier.py`** gained the missing half of its own guard. Its `n_no_rxn` aborts are
  gated on `if routes`, so they could not fire when the HUB side was empty: an empty `routes.json`
  skips every hub, leaves `routes` empty, and hands SPARROW a pool it prices as optimal at zero cost —
  the same wrong-and-flattering answer reached from the other direction.
- **`experiments/lsd_hubs/matrix16/check_route_readiness.py`** (new) — one command answers "which
  cell-seeds can feed the competitor arm?". That question previously took a session of manual
  archaeology across three scratch trees.

**Design rule, deliberate and worth keeping:** *a validator must never be why a good run dies.* These
are 24-hour GPU jobs. So the checks raise only on unambiguous evidence (a route-bearing stage that
produced **zero** routes / zero reaction coverage), soft cases are reported rather than enforced, and
every internal error in the check itself is caught and downgraded to a warning. A guard that kills a
good 24-hour job gets switched off within a week, and then protects nothing. Verified: a corrupt
`enum_children.json` yields `state=check_failed` and a warning, not an exception.

**Unknown generators default to route-bearing (REQUIRED),** so a new adapter added without a thought
about routes fails loudly on its first run — a five-minute fix. The opposite default is what produced
this entry.

**Also fixed while in here: `rgfn_worker` declared `--seed` and never applied it** — alone among the
four (scent, rxnflow and fraggfn all call `manual_seed`/`np.random.seed` at setup). So every RGFN
sample was irreproducible while *reporting* a seed, which is the worst of both: a "seed 43" cell was
not reproducibly seed 43, and a lost or route-less sample could not be regenerated even in principle.
That is the same class of defect as the routes gap — the artifact looked complete and wasn't — so it is
fixed rather than noted.

**`--seed` alone does NOT reproduce an RGFN sample — but `--seed` plus `PYTHONHASHSEED=0` does, and
both halves were measured.** Two runs at `--seed 42`, same checkpoint, same trajectories, same device,
gave **377 vs 387 routes sharing only 8 keys**. `Categorical(...).sample()` draws from torch's global
RNG and `manual_seed` does seed it, so the divergence was outside torch: per-process set/dict iteration
order feeding action-space construction, where an identical draw over a differently-ordered action list
picks a different action. Adding `PYTHONHASHSEED=0` closed it completely — `--seed 42` twice on a debug
GPU gave **730/730 byte-identical routes**.

`submit_cell.sh` and `submit_docking_cell.sh` now export `PYTHONHASHSEED=0`, so **future RGFN samples
are regenerable**. (Editing those is safe with jobs queued: SLURM snapshots the batch script at submit
time, so already-queued jobs run the version they were submitted with.)

What this does NOT buy: every sample already on disk was taken before either knob was live, so those
remain one-of-a-kind and recoverable only from backup (TIER 2). It also means re-sampling a route-less
RGFN cell still draws a genuinely different pool — changing its hubs and its published number — so the
23 route-less cell-seeds are not repairable for free even now that the emitter exists.

**Verified, on real runs:**

- All eight validator paths: raise on empty, ok, partial, N/A-with-reason, unknown-generator default,
  enum zero-coverage raise, enum full-coverage ok, and corrupt-input **warn** (not raise).
- **RGFN emission end-to-end**, 200 trajectories against the trained seed-42 sEH checkpoint:
  `routes.json` 524 KB where it had been 2 bytes (`{}`), **667 routes / 667 distinct nodes = 100%
  coverage**, **180/180 hub_keys carry a route**, 615/615 final products equal their key
  (stereo-stripped), 908/908 step-chain integrity (`product[i] == input[i+1]`, so these are connected
  chains and not a bag of steps), 0 canonical-schema violations, and depths 0–4 all populated.
- **The first smoke caught a real gap**: 2 of 193 hub_keys were **depth-0** — bare purchasable
  building blocks with no reaction steps — and had no entry at all. `hub_routes.get(hk)` returns None
  there, and the frontier skips such a hub *and every child hanging off it*, silently. Depth-0
  molecules now get an explicit zero-step route (52 of them in the re-run), which took hub coverage
  99.0% → **100%**. Worth recording because it is exactly the kind of near-miss that only a live run
  surfaces: the code was "working" at 99%.
- **The enumerate-mode check against real enumerations of all four generators**, run as the queued
  slices will run it: rgfn 104,589 children at 100%, scent 644,759 at 100%, rxnflow 232,213 at 100%,
  fraggfn correctly `not_applicable`. (Copied into a tempdir first — never validated into a live run
  dir.) The call site was confirmed by inspection to sit inside `else:  # enumerate`, because the live
  end-to-end enumerate smoke was killed by the login CPU limit after one hub.
- `py_compile` across every touched file; the readiness checker against all three scratch trees
  (10 SPARROW-ready, 23 route-bearing cell-seeds with no routes, 14 control).
- Enumerate-side reaction coverage is **100% wherever an enumeration exists**, matrix-wide — the
  enumeration repairs held, and the entire remaining gap is sample-side.

**RxnFlow emitter: SMOKED AND VERIFIED, twice, independently.** My own debug run gave 611/611 coverage
with 196/196 hub_keys routed after the depth-0 and seed fixes; a co-agent independently ran it in the
`rxnflow` env against the real `rxnflow_seh_5k/seed42` checkpoint (job 74693, 15:02, i.e. after the
14:44 fix) and got **1,150 routes at 100% of terminals** on 400 trajectories. Two environments, two
operators, same verdict.

**A SECOND BLOCKER EXISTS THAT ROUTES DO NOT FIX, and it is worth understanding before anyone plans
native-route work.** A co-agent extended `check_route_readiness.py` with the recipe/snapshot dimension
it had no visibility into, and SPARROW-ready drops **10 → 5**. Six cell-seeds have routes AND 100%
reaction coverage and still cannot be priced natively, because their training snapshot has no
`smiles_to_route` at all — recipe logging was off for runs launched 2026-07-14..07-26:

    seed 42   scent_seh, scent_drd2, scent_clpp, scent_6td3   <- ALL of seed 42
    seed 43   scent_clpp
    seed 44   scent_clpp

This is NOT repairable by re-sampling or re-enumerating: a promoted fragment's recipe is observable only
while it is being BUILT (`recipe_logging.py` writes onto `dl._smiles_to_route` during training), so
those cells would need a re-TRAIN — ~10-15 h each, ~78 GPU-h for the six, and re-training changes the
model, so every published number on those cells moves and the whole cell has to be rebuilt behind it.
Independently confirmed: an earlier audit of mine had already found `smiles_to_route` absent for every
seed-42 cell.

**But it does not block the comparison, and this is the operative point.** Recipe expansion is consulted
only under `--route-source native|enum` (verified: the logic and both new provenance aborts sit inside
`if is_native:`). The DEFAULT source is `multiaiz`, which routes every molecule independently through
AiZynth and needs no snapshot, no recipes, and no `routes.json`. That is the plan's own MVP path, and
T1.3/T1.4 were written for exactly this — "returns `total_reactions` for a MIXED (routed + route-less)
library". So all 18 competitor-scope cells are priceable today at zero GPU cost.

**There is also a FAIRNESS reason to prefer the from-scratch source for the headline, independent of
cost.** Native routes measured 1.22 rxn/mode against from-scratch 1.85. Pricing SCENT natively while
rgfn/rxnflow go from-scratch would let SCENT win on route PROVENANCE rather than chemistry — the same
confound the project already polices on the mode metric (one metric reapplied to all four). One source
across all four generators; native stays a supplementary result. Note the supplementary result can only
be reported on **seeds 43/44** — none of the five route-and-recipe-complete cells is seed 42.

**Not verified:** whether a fresh sample rediscovers a cell's original hubs, which is what the cheap
`enum` route-source repair would depend on. A free proxy is discouraging: 200-trajectory debug samples
of `rgfn_seh` s42 cover **0 of its 200 production hubs** — 1/150th the trajectory count, so not a bound,
but not encouraging either. RxnFlow is the better bet there (its `--seed` was genuinely applied), and a
co-agent is measuring it on one cell before committing the other eight.

**Explicitly NOT fixed by any of this:** the 23 existing route-less cell-seeds. They need a
**re-sample** — re-running the enumeration does nothing, because `routes.json` is written by the
sample stage. And because RGFN sampling was unseeded, an RGFN re-sample draws a *different pool*, so it
would change that cell's hubs and its published number; it is not a free repair. Those cells can still
feed the **from-scratch** SPARROW arm today via AiZynth recovery
(`validation/lsdflow/eval/route_recovery.py`), which is a different and already-planned comparison
(measured: from-scratch 1.85 vs native 1.22 rxn/mode) — but not the native-route arm. Whether to spend
the compute is a decision for the user, not a code change.

**Deliberately not built:** copying the hub routes into `enum/` to make that directory self-contained
against a `$SCRATCH` purge. `sample/` is already covered by the backup script's TIER 2, so the marginal
value is low, and it would mean editing `submit_cell.sh` while 18 jobs are live.

---

## 2026-08-24 — the RGFN reaction repairs verified, and a metric that is not as stable as it looks

**All four completed repairs may stand** — `rgfn_seh` s42 and `rgfn_drd2` s42/s43/s44, each 200/200 hubs
at 100% reaction coverage with the recorded final product reconstructing the child (stereo-stripped) on
400/400 sampled. `rgfn_seh` s42's count-once summary came out bit-identical to the committed value.

**The three DRD2 seeds "failed" check 3, and the failure was uninformative.** Every headline field was
bit-identical — `reactions_per_mode` (1.197 / 1.310 / 1.243), `total_reactions`, `total_modes`,
`case1_modes_at_100rxn` (82 / 84 / 89), `distinct_hubs_used`, `total_reward_gen_calls`. Only
`n_scaffolds` (±1–2 of ~270) and `median_reward` (±0.002) moved.

The chemistry was verified identical rather than assumed: same 200 hubs, same **168,006** distinct
children, **zero** hubs whose child set differs — while child ORDER differed in **200/200** hubs. Greedy
mode selection takes the first of any equally-good candidates, so a reordering swaps membership without
changing count or cost. Recorded as correctness trap 3 in `docs/RESEARCH_CONTEXT.md`, with the reason it
hits DRD2 and not sEH: **51.6%** of DRD2's enumerated children share a reward with another child (top
value ×302) versus **30.4%** on sEH (top value ×6) — a saturated classifier versus a continuous proxy.

**What this says about the verifier, and why I have NOT changed it yet.** Check 3 has only two tiers: a
field is either in `VOLATILE` (ignored) or must be bit-identical. `n_scaffolds` is neither — exempting it
would hide a genuine 50-scaffold regression, while demanding bit-identity cries wolf on a tie. The fix is
a third tier:

    headline     reactions_per_mode, total_reactions, total_modes, case1_modes_at_100rxn,
                 case2_reactions_at_300modes, distinct_hubs_used, total_reward_gen_calls
                 -> bit-identical or FAIL
    descriptive  n_scaffolds, median_reward, best_reward
                 -> print the drift; FAIL only beyond what tie-breaking can produce (±3 / ±0.01)
    volatile     compute_time, wall_s, enum_timings_meta  -> ignore

The bound is the point. It passes today's tie noise and still fails real breakage, which is precisely
what moving these fields into `VOLATILE` would give up. Left unimplemented pending the user's call,
because relaxing a gate so that a currently-failing artifact passes is not a change to make on my own
judgement — the gate exists to be strict, and I have already been over-confident once today.

**Not verified:** whether any published figure or table currently quotes `n_scaffolds` or
`median_reward` for a DRD2 or docking cell. Both appear in eight analysis scripts
(`analyze_matrix.py`, `hub_stats.py`, `preselect_sweep.py`, `strategy_compute_summary.py`,
`run_campaign.py`, `audit_greedy_libraries.py`, `plot_greedy.py`, `compare_hub_order.py`); I checked
that they are referenced, not what consumes their output.

---

## 2026-08-26 — competitor-campaign guards, and the SynFormer control harness

### Three fixes to campaign code (each would have produced a false claim about a baseline)

**`mode_saturation.py` gated docking cells on the training reward.** It accepted `--lower-is-better`
but hardcoded `row.get("score")`; for docking that column is `clip(-vina)`, positive 0..11, so the
ClpP gate of -8.0 admitted nothing and all sixteen ClpP chain-cells aborted as "fewer than 10 modes
— nothing to measure", a message phrased as a POOL-SIZE finding. 645 of 2,000 molecules were in fact
passing on `raw_score`. Now takes `--score-column`, defaulting to `raw_score` under
`--lower-is-better` — the same rule `build_s3gfn_pools.py` already used — and prints the resolved
column every run. Commit `c5f428e`.

**`build_s3gfn_pools.py` skipped any NAIVE pool short of N**, writing no directory, so the cell died
downstream on `FATAL: pool not built`. Eight of thirty-five cells fall short and seven are S3-GFN
(154/65/168 distinct above the sEH gate, against Saturn's ~1,900), so the skip deleted exactly the
cells where a baseline is weakest. Now clamps to availability and names the directory for the size
written (`_N65`), with `pool_limited: true` in `pool_meta.json`; `--strict-naive` restores the old
behaviour. `submit_competitor_routes.sh` resolves a clamped directory for both variants now, not
only pruned. Commit `5ffe0e9`. **This overrides a documented decision** — the original rationale was
that an `_N500` directory holding 300 misstates the pool, which is an argument against misnaming,
answered by naming honestly.

**`DockingServerClient.dock` sent unbounded batches.** `timeout` bounds one round-trip while the
caller's batch is not bounded at all; the final 2,000-molecule pool scoring at ~4 s/mol could never
return inside the 3,600 s socket timeout, and it lost `saturn_clpp` seed 43 after that run had spent
its full 10,000-call training budget. Now chunks at 200 per round-trip in the shared client rather
than in each of the six adapters. Commit `e6b6a1d`.

### SynFormer: diagnosis corrected twice, still one thing open

Nine cells died at hour 9-12 of 72 h. SLURM logs `oom_kill` events on every one. **Two explanations I
published and then had to withdraw**: "blocked in `submit()` on a bounded queue" (the queue is
`task_qsize=0`, i.e. unbounded) and "the fetch guard should have fired" (it was present since commit
`7442376` and never fired). Seven of eight cells stopped INSIDE the projection fetch loop, 16-39
molecules short; one had already degraded to 246 s/molecule against a healthy 3. **The exact frame
the parent blocks in is still unknown** and needs a live stack (`py-spy` is not installed in either
env), not more reasoning.

What IS established: upstream tears the worker pool down every generation and children return to
**0.0-0.1 GiB**, so their cadence bounds the leak completely. Holding the pool open across
generations is our divergence and the cause of the ~260 GiB growth. Our `recycle()` cannot substitute
because it forks after the reward model exists — see `[[fork-after-torch-deadlock]]`; and
`set_start_method("spawn")` is disqualified rather than merely imperfect, because a spawned worker
re-imports synformer and loses `patch_get_dataframe()`, emptying every route column while every other
check passes.

`experiments/synformer_baseline/` is new: upstream's own GraphGA-SF loop with one flag per divergence
(`--workers`, `--routes`, `--torch-in-parent`) so each can be blamed or cleared alone. The route
variant asserts the column is POPULATED, not that the run finished.

**Not verified:** whether memory stays flat beyond three teardown cycles (74999, 16 generations,
queued); whether fork-after-torch blocks under upstream's cadence (75029, on debug); whether routes
survive that cadence (75001, queued). Nine SynFormer cells (~170 GPU-h) stay unsubmitted until those
answer. Four defects were found in this control harness itself — missing `tdc` import, relative
`fpindex` path, a `sys.modules` stub that loky children never saw, and a missing population
truncation that upstream performs at its line 332 — all mine, none SynFormer's.
