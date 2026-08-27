# Re-train runbook — what to re-run, in what order, and how to know it worked

**Read this before launching any re-training.** It exists because the last time these artifacts were
produced, three separate things were silently missing and nobody found out for weeks. Every check
below corresponds to a failure that actually happened and that produced a confident wrong answer
rather than an error. Full diagnosis: `Logs/070`.

**Nothing here is required for the current headline results.** The matrix (2.93× at R=100) and the
competitor comparison (2.51×) are priced by count-once, which never reads a route. This runbook buys
**dataset breadth** and **the native-route arm**, nothing more. If the paper is accepted, it may never
need to run.

**Which submission needs which job (set 2026-08-21).** The near-term target is a **workshop submission
(GEM@NeurIPS, ICLR) with reduced results**; a full **NCS** submission follows only if those are
rejected, incorporating the feedback.

| job | cells | cost | needed for the workshop? |
|---|---|---|---|
| recipe/route repair (§1) | 4 | ~200–400 GPU-h | **no** — buys dataset breadth + the native-route arm |
| **6TD3-B reward swap (§0b)** | **11** | **~1,100–1,700 GPU-h** | **no — this is the NCS path** |

**Do not launch either for the workshop deadline.** The headline results do not depend on them: the
reaction-axis readout (`Logs/065`) contains **zero 6TD3 rows** — the target was already excluded there
on gate grounds — and both competitor comparisons are sEH and DRD2. The 6TD3-B campaign is roughly a
week of cluster and exists to make the CDK12–DDB1 arm *defensible for NCS*, not to unblock a workshop
paper.

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

## 0b. ⛔ 6TD3 cells re-train against **6TD3-B**, not 6TD3 (decided 2026-08-21)

**If you are re-training a CDK12–DDB1 cell, change the reward. Do not reproduce the old one.**

**SCOPE: every 6TD3 cell, for every generator — 11 on disk.** This is the one item in this runbook
that is NOT about routes or recipes, so the usual "only SCENT needs re-training" logic in §1 does not
apply and the "do NOT re-train RGFN / RxnFlow / FragGFN" list below does not exempt their 6TD3 cells.
A reward change invalidates the trained policy regardless of how the generator assembles molecules:

| seed | scent | rxnflow | fraggfn | rgfn |
|---|---|---|---|---|
| 42 | ✓ | ✓ | ✓ | ✓ (no enum yet) |
| 43 | ✓ | ✓ | ✓ | ✓ |
| 44 | ✓ | ✓ | ✓ | — (cell absent) |

FragGFN is included even though its empty `routes.json` is correct (§"Do NOT re-train these") — that
exemption is about ROUTES. It still trained against `ddb1_dvina` and so still learned the exploit.

| | reward column | direction | status |
|---|---|---|---|
| `6TD3` (incumbent) | `ddb1_dvina` (Vina Tier2 − Tier1) | lower better | **do not train against this** |
| **`6TD3-B`** | `cnnsc_t2` (gnina CNNscore, selected Tier-2 pose) | **higher better** | use for all new 6TD3 training |

**Why — measured, not suspected (`Logs/072`).** 400 SCENT candidates were re-scored beside the 160 real
glues and 160 property-matched decoys, all from one docking pass:

* our candidates clear the **6TD3** gate **78%** of the time — real glues clear it 67%;
* our candidates clear the **6TD3-B** gate **0%** of the time (0 of 400) — real glues clear it 66%,
  and even the property-matched decoys manage 1%;
* the generator's reward runs to **−7.97** where the best real glue reaches **−4.09**.

The old reward is a *difference* of two docking scores, so it can be maximised by making Tier 1 worse
rather than Tier 2 better, with no requirement that the pose be physical. The generator found that.
Size does not explain it: inside the real glues' own p10–p90 MW band, **0%** of our candidates clear the
6TD3-B bar against **70.3%** of real glues.

**Practical notes**

* **The docking pass is identical.** 6TD3-B reads a different column of the same gnina output —
  `cnnsc_t2` was always emitted. No new receptor, no extra pass, no re-validation of the docking setup.
* **Keep Tier 1 anyway.** 6TD3-B does not need it, but it is a near-fixed ~4 s model load — 2.3% of a
  400-molecule batch — and keeping it means every run emits BOTH oracles' signals. The whole reason
  `Logs/069` was expensive is that the number we later wanted had never been recorded. Do not repeat
  that to save 2%.
* **Both oracles stay in the tree.** 6TD3-B is a new oracle class and a new `targets.py` entry; the
  `6td3` entry is untouched, so every published number stays reproducible and directly comparable.
* **Existing 6TD3 libraries do not survive re-gating.** Roughly zero of the current candidates clear
  the 6TD3-B bar, so this is not a re-gate — those cells need the full re-train → re-sample →
  re-enumerate chain of §2.
* **6TD3-B is not proven un-gameable.** It is the signal our molecules do not currently exploit. After
  re-training, repeat the `Logs/072` comparison against a signal the new model was NOT trained on. That
  test is now the standing check, not a one-off.

---

## 1. What actually needs re-training

**Only SCENT has a dynamic library, so only SCENT needs recipes.** The others build from a fixed
catalogue; their routes already bottom out at purchasable stock and there is nothing to expand.

| cell | recipes | verdict |
|---|---|---|
| `scent_seh` s43, s44 | ✅ | **complete — do not re-train** |
| `scent_drd2` s43, s44 | ✅ | **complete — do not re-train** |
| `scent_6td3` s43 | ✅ | recipes fine, but **re-train anyway against 6TD3-B** (§0b) — its library was optimised toward an exploitable reward |
| `scent_seh` **s42** | ❌ | **re-train** |
| `scent_drd2` **s42** | ❌ | **re-train** |
| `scent_6td3` **s42** | ❌ | **re-train — against 6TD3-B (§0b), not 6TD3** |
| `scent_clpp` **s42, s43, s44** | ❌ | **re-train — all three seeds** |

**Two separate re-training jobs, and they must not be conflated:**

1. **The recipe/route repair (this section's table): 4 cells** — every seed-42 SCENT cell plus ClpP on
   all three seeds. SCENT-only, because only SCENT has a dynamic library.
2. **The 6TD3-B reward swap (§0b): 11 cells** — every 6TD3 cell of every generator. Independent of
   routes and recipes; driven purely by the reward being exploitable.

`scent_6td3` s42 appears in both and is re-trained once, against 6TD3-B.
Cause was a logging regression for runs launched 2026-07-14→07-26; the default has been correct since
07-29, so a re-train today picks it up automatically — **but verify anyway (§4)**, because that is
precisely what nobody did last time.

**Cost.** Training is ~50–100 GPU-h per cell; ClpP and 6TD3 are the expensive ones (docking reward in
the loop). Re-enumeration is the larger and more often forgotten half — 6TD3 enumeration is **98%
docking** at 0.62–0.98 s per child (`Logs/072`):

| job | cells | training | re-enumeration | total |
|---|---|---|---|---|
| recipe/route repair | 4 | 200–400 GPU-h | (surrogate; cheap) | ~200–400 |
| **6TD3-B reward swap** | **11** | **550–1,100 GPU-h** | **535+ GPU-h** measured over the 8 cells that already have a pool; the 3 without will add more | **~1,100–1,700** |

Against a ~1,400 GPU-h/week burn that is roughly **one week of cluster** for the 6TD3-B swap alone —
not the "one to two days" the recipe repair costs. Budget it as its own campaign.

### Do NOT re-train these

> ⛔ **Except their 6TD3 cells.** Everything in this list is exempt because its gap is a *route*
> artifact, which training does not produce. That reasoning does not survive a REWARD change: every
> 6TD3 cell of every generator trained against `ddb1_dvina` and must be re-trained against 6TD3-B
> (§0b). Read the exemptions below as "for all targets except 6TD3".

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

   **You do NOT need to repeat the reaction repair afterwards.** `children[].reaction` is emitted by
   all four workers on the normal enumerate path, so a re-run through this chain carries it
   automatically — verified end-to-end rather than assumed:

   - all four workers are wired to `_routes.validate_enum_reactions`, which **fails the run** if a
     route-bearing generator enumerates children and none carries a reaction (committed in `176816c`,
     so it travels with the repo rather than with one machine);
   - every enumeration produced in the three days to 2026-08-25 came out at **100%** coverage —
     `rgfn_6td3` 494,195 children, `rgfn_clpp` s42 207,048, `rgfn_clpp` s43 332,056, plus
     `fraggfn_6td3` 238,140 and `rxnflow_clpp` 182,245 **on Trillium**, which is the proof it is a
     property of the code and not of Balam;
   - SCENT is the one worker that does not route through `_artifacts` (it predates it and writes enum
     hubs inline via its own `_reaction_step`), so it is worth naming separately: `scent_6td3` 657,236
     children and `scent_clpp` s43 285,164, both 100%.

   The repairs were needed only for cells enumerated BEFORE that emission landed. Anything the runbook
   produces from here is born with reactions, and if a future change breaks that, the guard stops the
   run instead of shipping a complete-looking pool nothing can route.
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
