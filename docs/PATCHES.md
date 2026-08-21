# Patches to upstream files

We keep `rgfn/` and `configs/` mergeable with upstream RGFN. New functionality is
added via the `glue/` package and `configs/glue/` overlays — **not** by editing
upstream. The exceptions below are small operational overrides we chose to apply
directly to upstream files. They are listed here so they are easy to find, review,
and re-apply after an upstream merge/rebase.

If you pull upstream changes and one of these reverts, re-apply it from here.

---

## 1. `rgfn/trainer/logger/wandb_logger.py` — drop `dir=` from `wandb.init`

**Change:** removed the `dir=self.logdir,` argument from the `wandb.init(...)`
call.

```diff
         return wandb.init(
-            dir=self.logdir,
             project=self.project_name,
             name=self.experiment_name,
             group=group,
```

**Why:** on Balam we run wandb in offline mode with cache/dir env vars
(`WANDB_DIR`, etc., set in `scripts/submit.sh`). Passing `dir=self.logdir`
conflicted with that setup.

**Status:** kept as a one-line patch (intentionally not refactored into a `glue/`
subclass — it's a single line and low-churn).

---

## 2. `configs/loggers/wandb.gin` — default to offline

**Change:**

```diff
-WandbLogger.mode = 'online'
+WandbLogger.mode = 'offline'
```

**Why:** Balam compute nodes have no/limited network; wandb runs offline and is
synced later.

**Note:** could alternatively live as a `configs/glue/` overlay. Left in place for
now because offline is the desired default for all our runs.

---

## 3. `configs/rgfn_seh_docking.gin` — cap docking-run iterations

**Change:** appended

```gin
# GPU docking is ~160s/iter (vs ~10s for the neural proxy), so the base config's
# 5002 iterations would take ~9 days. Cap at 400 (~18h) to complete within the
# 20h SLURM walltime in submit.sh. Override here only; proxy configs keep 5002.
Trainer.n_iterations = 400
```

**Why:** make a docking-oracle run fit inside the Balam walltime.

**Note:** this is a docking-specific config; the cleaner long-term home is a
`configs/glue/` overlay. Left in place to preserve current run behavior.

---

## Convention going forward

Prefer adding new behavior in `glue/` + `configs/glue/`. Only patch upstream when
there is no reasonable override point, and when you do, **add an entry here**.

---

## Baseline-clone patches (added 2026-08-21)

These are edits to third-party clones under `external/`, not to `rgfn/`. Each is recorded here
because a reader must be able to tell what we changed in a competitor's code and why.

### 1. REINVENT — periodic checkpoints (`runmodes/RL/learning.py`)

**What.** A block that writes the agent every `REINVENT_CHECKPOINT_EVERY` steps into
`REINVENT_CHECKPOINT_DIR`. Both env vars unset (the default) means upstream behaviour, unchanged.

**Why.** REINVENT's RL loop has no periodic-save hook. Its multi-stage mechanism looks like one --
each stage writes `chkpt_file` -- but cannot serve here: `optimize()` returns `terminate=True`
whenever a stage ends on `max_steps`, and `run_staged_learning.py:372` then breaks out of the whole
stage loop. Our budget guarantee is `min_steps == max_steps`, so every stage ends that way; a
10-stage run executed stage 1 and stopped at 16 of 157 steps (measured).

**Why it is safe.** The block calls `save_to_file` inside a `try/except` and touches no RNG, no
gradients, no state. It cannot be proved safe by an A/B run, because **REINVENT training is not
reproducible run-to-run at a fixed seed** -- two runs with the patch inactive differed on 8,946 of
10,048 scored molecules and produced 500 vs 560 modes. That nondeterminism is itself recorded as a
benchmark caveat.

**Where it must be applied.** REINVENT is **pip-installed into the env**, so the clone is not what
runs. The patch must be copied to
`/scratch/markymoo/conda_envs/reinvent4/lib/python3.10/site-packages/reinvent/runmodes/RL/learning.py`.
Both copies are kept identical so the repo shows the code that executed. Backups: `learning.py.orig`
beside each.

### 2. Saturn/TANGO — expose `num_top_results` (`oracles/synthesizability/syntheseus.py`)

**What.** `_write_config` used a hardcoded `"num_top_results": 1`; it now reads the oracle's
`specific_parameters`, defaulting to 1 (upstream behaviour preserved). We set 5.

**Why.** The field does double duty in syntheseus: how many routes to return, *and* -- via
`get_model_fn(config, default_num_results=config.num_top_results)` (`cli/search.py:279`) -- how many
disconnections each expansion requests from the reaction model. At 1 the search explores a single
chain and one bad top-1 prediction ends it. Measured on 50 REINVENT sEH molecules with the authors'
own stock and MEGAN: **1 -> 10/50 solved (20%); 5 -> 47/50 (94%)**. Running at the shipped default
would report a solve rate we could see was an artifact. Disclosed in the write-up.

### 3. syntheseus — non-fatal MEGAN log cleanup (`reaction_prediction/inference/megan.py`)

**What.** Wrapped MEGAN's post-load log cleanup in `try/except`.

**Why.** Upstream unlinks its log files then `rmdir`s the directory. The logger can recreate a file
between the two loops, so `rmdir` raises `OSError(39) Directory not empty` and kills model
construction *after* the model has otherwise loaded. Cleanup of MEGAN's own logs has no bearing on
predictions, so it must not be fatal. Backup: `megan.py.orig`. Note MEGAN also writes `logs/` into
the **current working directory**, so every invocation must run from a scratch cwd.

### 4. syntheseus env — `sitecustomize.py` restoring `torch.load(weights_only=False)`

**What.** A `sitecustomize.py` in the dedicated `syntheseus` env only.

**Why.** torch 2.6 flipped `weights_only` to `True`; syntheseus's RootAligned backend is OpenNMT,
whose author checkpoints pickle `TextMultiField` and are refused. The checkpoints are downloaded by
syntheseus itself from the authors' Figshare IDs, so their provenance is the library's. **Scoped to
that one env deliberately** -- re-enabling arbitrary-code-execution-on-load in a general env would be
a real security regression. Do not copy it into `rgfn`/`saturn`.
