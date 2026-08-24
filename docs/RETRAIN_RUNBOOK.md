# Re-train runbook — what to re-run, in what order, and how to know it worked

**Read this before launching any re-training.** It exists because the last time these artifacts were
produced, three separate things were silently missing and nobody found out for weeks. Every check
below corresponds to a failure that actually happened and that produced a confident wrong answer
rather than an error. Full diagnosis: `Logs/070`.

**Nothing here is required for the current headline results.** The matrix (2.93× at R=100) and the
competitor comparison (2.51×) are priced by count-once, which never reads a route. This runbook buys
**dataset breadth** and **the native-route arm**, nothing more. If the paper is accepted, it may never
need to run.

---

## 0. The three artifacts, and which stage makes each

A molecule is only usable for route pricing or a chemist-facing dataset if **all three** exist. They
are produced by different stages, which is why partial failures were so easy to miss.

    TRAIN ─────────────► SAMPLE ─────────────► ENUMERATE
      │                    │                       │
      │ fragments_N.json   │ routes.json           │ enum_children.json
      │ .smiles_to_route   │ (hub prefixes)        │ children[].reaction
      │                    │                       │
      └─ ③ RECIPES         └─ ① HUB ROUTES         └─ ② FINAL STEP
         re-TRAIN only        re-SAMPLE               re-ENUMERATE

A child's full route is assembled at read time as **① prefix + ② final step**, with ③ expanding any
promoted fragment inside them. Miss any one and the route is wrong rather than absent.

---

## 1. What actually needs re-training

**Only SCENT has a dynamic library, so only SCENT needs recipes.** The others build from a fixed
catalogue; their routes already bottom out at purchasable stock and there is nothing to expand.

| cell | recipes | verdict |
|---|---|---|
| `scent_seh` s43, s44 | ✅ | **complete — do not re-train** |
| `scent_drd2` s43, s44 | ✅ | **complete — do not re-train** |
| `scent_6td3` s43 | ✅ | complete (target parked on gate calibration) |
| `scent_seh` **s42** | ❌ | **re-train** |
| `scent_drd2` **s42** | ❌ | **re-train** |
| `scent_6td3` **s42** | ❌ | re-train (only if 6TD3 is unparked) |
| `scent_clpp` **s42, s43, s44** | ❌ | **re-train — all three seeds** |

**6 cells to re-train** (4 if 6TD3 stays parked): every seed-42 SCENT cell, plus ClpP on all seeds.
Cause was a logging regression for runs launched 2026-07-14→07-26; the default has been correct since
07-29, so a re-train today picks it up automatically — **but verify anyway (§4)**, because that is
precisely what nobody did last time.

**Cost:** ~50–100 GPU-h per cell → **300–600 GPU-h** for the six. ClpP and 6TD3 are the expensive
ones (docking reward in the loop). Against a ~1,400 GPU-h/week burn, one to two days of cluster.

### Do NOT re-train these
* **RGFN / RxnFlow (23 cells).** Their gap is ① `routes.json`, a SAMPLE-stage artifact — training is
  not involved. But note it is **not a cheap repair either**: both are `PYTHONHASHSEED`-sensitive
  (measured: same checkpoint, same `--seed`, 61% hub overlap), and the originals ran with that unset.
  A re-sample therefore draws a **different** library — new hubs ⇒ re-enumerate ⇒ re-campaign ⇒ that
  cell's published numbers move. Treat as *replace*, never *repair*, and budget the enumeration
  (~19 GPU-h surrogate, ~104 GPU-h docking per cell), which dwarfs the sampling.
* **FragGFN (12 cells).** Assembles by attachment, not reaction. Its empty `routes.json` is
  **correct**. Do not "fix" it.

---

## 2. Order of operations

1. `python experiments/lsd_hubs/matrix16/check_route_readiness.py` — never skip. It is the one command
   that answers "which cells are usable", and it now covers all three artifacts.
2. **Re-train** only the cells in the table above, with `--log-recipes` (default on — confirm, §4).
3. **Verify recipes immediately** (§4.3). This is the highest-value check in the document: recipes are
   the ONLY artifact that cannot be recovered later, so a failure caught here costs one re-train and a
   failure caught in three weeks costs the same re-train plus everything built on it.
4. Re-sample, re-pick hubs, re-enumerate for those cells (the standard `submit_cell.sh` chain).
5. Re-run the campaign, and **expect the cell's published numbers to change** — a new sample is a new
   library. Diff against the old summary and record the delta rather than quietly replacing it.
6. Re-run `check_route_readiness.py` and confirm the cell now reads READY.

---

## 3. Freeze your inputs before any comparative run

These artifacts live in a shared scratch tree that other agents rewrite. On 2026-08-19 all three
`scent_seh` enumerations were re-run mid-experiment (a correct fix), *after* a comparison arm had read
the old ones — silently turning a same-chemistry comparison into a cross-chemistry one, with both runs
reporting success. The pools differed by 6× and nothing in the outputs showed it.

**Any experiment that claims two arms saw "the same candidates" must read from a frozen copy**, with
the source md5 recorded:

    /scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_snapshot_20260820/<target>_seed<N>/
        enum_children.json  routes.json  enum_children.md5
        meta.json  compositions.json      <- REQUIRED, see below

`meta.json` and `compositions.json` are not optional padding. The provenance and coverage guards
resolve them *relative to their inputs*, so a snapshot that omits them turns both checks into silent
no-ops. That happened on the first attempt at this — the guard I had just written to catch mismatches
was disabled by the very act of moving to snapshots.

---

## 4. Verification checklist — each item is a failure that occurred

### 4.1 Routes were emitted (①)
    python -c "import json;d=json.load(open('<cell>/sample/routes.json'));print(len(d))"
Must be **non-zero** for scent / rgfn / rxnflow. Zero for fraggfn is correct.
*Why:* two workers hardcoded `json.dump({})`. A run wrote a well-formed empty file, exited 0, and was
promoted. 36 of 40 cell-seeds were in this state. Downstream, every hub is skipped and SPARROW prices
the **empty library** as trivially `Optimal` at zero cost.
`_routes.py` now validates this at write time and writes `route_status.json` — check that file exists.

### 4.2 Every enumerated child carries its reaction (②)
    # fraction of children with a non-empty children[].reaction — must be 1.00
Anything **between 0 and 1 is more dangerous than 0**: a partial artifact prices a mixture (some
children at their true molecule, the rest at their hub) and still returns `Optimal`. Measured on merged
artifacts: 56.6%, 24.6%, 99.1%. A 99.1% file is exactly the one a warning gets scrolled past.

### 4.3 Recipes exist AND belong to this run (③) — the two-part check
    # part 1: does the snapshot have smiles_to_route at all?
    # part 2: does it cover the fragments THIS RUN actually used?
**Part 2 is the one that was missing.** Coverage of the snapshot's own `chosen_smiles` is the wrong
question: a snapshot from a *different* model is internally complete, so it scores **100%** while only
**53%** of the run's fragments are expandable — 549 of 1,169 silently *bought* rather than *built*,
which is `Logs/049`'s 62.7% failure with a green light on top. `sparrow_select_frontier.py` now checks
provenance (run checkpoint vs snapshot path) and coverage against `compositions.json`.

### 4.4 Determinism, if you intend the sample to be reproducible
    export PYTHONHASHSEED=0     # load-bearing, not hygiene
`--seed` alone does **not** pin a sample: per-process hash order feeds action-space construction, so an
identical RNG draw over a differently-ordered action list picks a different action. Measured 377 vs 387
routes at the same seed; with `PYTHONHASHSEED=0`, 730/730 byte-identical. `submit_cell.sh` exports it.
Without it, a sample is a one-of-a-kind artifact recoverable only from backup.

### 4.5 Never overwrite a finished cell
`submit_cell.sh` resolves `SAMPLE_DIR` from the manifest via `eval`, which **clobbers any env
override**, and the manifest is not seed-aware (`SCRATCH_ROOT / tag`). Its sample stage will overwrite
the seed-42 tree in place. To sample without destroying the original, call the worker directly with
`--out-dir` elsewhere — see `submit_resample_check.sh`.

### 4.6 Do not trust a solver's status
CBC reports `Optimal` when it has merely run out of time (PuLP's `sol_status` does not distinguish).
Detect by wall-clock. A capped row is a **lower bound on the competitor**, i.e. it flatters us — and on
the corrected (larger) pools the arm does not converge at all, even at a 12 h cap with `gapRel` 1e-3.
Do not quote it as a ratio.

---

## 5. The one-line rule

Every failure above was an **existence check where the real question was a match or a content check**.
The file was present, well-formed, and wrong. When adding a guard, ask what it would take for this
check to pass on bad data — and check that instead.
