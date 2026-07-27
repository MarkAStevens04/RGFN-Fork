# `matrix16-lsdflow` → `Hub-Analysis` : merge handoff

**Status: ready to merge, NOT merged.** Everything is committed. This file is the reconciliation guide.

- **Branch:** `matrix16-lsdflow` (lives in the worktree `.claude/worktrees/matrix16-lsdflow`)
- **Forked from:** `973f721` *"Tier 3 initial"* on `Hub-Analysis`
- **Our commits:** 11 · **Their commits since the fork:** 6 · **Files both sides touched: 1**

---

## 1. The only conflict — and it is trivial

```
validation/lsdflow/adapters/workers/scent_worker.py   (hunk @ line ~312, in _make_enumerator's dfs)
```

**Both branches independently hit the same crash and applied the *same* fix.** The inserted code is
byte-identical; only the comment differs:

```python
            if not isinstance(state, (RSA, RSB, RSC)):
                return
```

*(ours = commit `f3d7d97`, 6 lines incl. comment; theirs = in `Hub-Analysis`, 8 lines incl. a longer
comment.)*

> **Resolution: keep EITHER side — there is no functional difference.** Recommended: take **theirs**
> (the longer comment documents the 200-hub crash context), then verify the guard is present exactly
> once. Nothing else in this file differs.

Verify after resolving:
```bash
grep -n -A1 "isinstance(state, (RSA, RSB, RSC))" validation/lsdflow/adapters/workers/scent_worker.py
python -m py_compile validation/lsdflow/adapters/workers/scent_worker.py
```

**All 20 other changed files are exclusively ours** — `experiments/lsd_hubs/matrix16/**` (new dir),
the three new workers + `_artifacts.py`, and two small additive edits to
`experiments/lsd_hubs/campaign/run_campaign.py`. They did **not** touch `matrix16/`.

---

## 2. Our 11 commits (oldest → newest)

| commit | what |
|---|---|
| `49df406` | **The build.** manifest-driven 16-cell run group + 3 new per-env workers + shared `_artifacts.py` + launchers. |
| `5cbcc23` | fix: activate conda **before** the manifest emit (bare SLURM shell has no `python`). |
| `0e17acc` | fix: use `$SLURM_SUBMIT_DIR` — SLURM spools the batch script, breaking `BASH_SOURCE` repo-root. |
| `13b6888` | fix: `TOPK` 100→1000 (it caps the hub count; `N_HUBS=200` was unreachable). |
| `557296e` | fix: `rxnflow_worker` crash on UniRxn (`RxnAction.block` is an *asserting* property). |
| `f3d7d97` | fix: `scent_worker` enumerate DFS guard vs early-terminal states. ⚠️ **the conflicting one** |
| `01619c3` | `rgfn_worker` enumerate: per-hub loop + progress logging + 10-hub partial flush. |
| `09c8031` | analysis: hub-rank P_B sensitivity + `gate_sweep.sh` (both strategies). |
| `34d9177` | analysis: flow-consistency severe test (enumeration-derived F(h)). |
| `a7c457e` | analysis: heuristic F(h) vs reconstructed `F_true(h)` (4 estimators). |
| `16b421e` | `rxnflow_worker --mode probe_hubs` + well-conditioned flow-conservation test. |

Their 6 commits for context: `3ed80e9` S3-GFN, `b34b5ad`/`bdbe357` MultiAiZ + AL-uncertainty,
`e226ed1` synthesizability, `3888c35` Exp A/B/C, `b27c6f5` ZINC∪SMALL. **Disjoint from our files**
except the one guard above.

---

## 3. What this branch adds (the architecture in one paragraph)

A **manifest-driven** eval group at `experiments/lsd_hubs/matrix16/` running the LSD-Flow pipeline
(**sample → pick_hubs → enumerate → count-once campaign**) uniformly across all four generators via
one CLI per env: `validation/lsdflow/adapters/workers/<gen>_worker.py --mode {sample,enumerate}`.
`manifest.csv` + `targets.py` + `manifest.py` are the single source of truth (per-cell paths, per-target
reward gate + direction, live status). `pick_hubs.py` and the campaign were already model-agnostic and
are reused unchanged. Enumeration is **exhaustive** for all four (the "option 2" decision): RGFN/SCENT/
RxnFlow rebuild the hub state from SMILES; **FragGFN persists `hub_graphs.pkl`** during sampling and
reloads it via `--sample-dir` (its `obj_to_graph` mis-decomposes ~6% of hubs).

---

## 4. Run state at handoff (2026-07-27)

**Committed results** in `results/<cell>/` — 6 of 8 surrogate cells complete at 30k trajectories / 200 hubs:

| cell | best-cand rxn/mode | hub rxn/mode | child policy | note |
|---|---|---|---|---|
| scent_seh | 3.38 | **1.30** | free_frag + K=20 | naive variant kept in `scent_seh_naive/` (2.78) |
| scent_drd2 | 3.67 | **1.16** | free_frag + K=20 | naive in `scent_drd2_naive/` (2.65) |
| rxnflow_seh | 3.00 | 2.45 | naive `reward` | weak edge = the sEH-7.0 gate, see §5 |
| rxnflow_drd2 | 2.94 | 1.83 | naive `reward` | |
| fraggfn_seh | 8.0 | 2.31 | naive `reward` | ⚠️ **CONTROL** — see §5 |
| fraggfn_drd2 | 8.0 | 3.98 | naive `reward` | ⚠️ **CONTROL** |

**Outstanding — the 2 RGFN cells.** Jobs **71766** (`rgfn_seh`, enum-only, reuses its saved 30k sample)
and **71767** (`rgfn_drd2`, full pipeline) were `PENDING` behind the 30-running-job QOS cap at handoff.
RGFN's enumerate is the slow one (timed out twice at 24h) — hence the 3-day walltime and the per-hub
progress logging in `01619c3`. **After they finish:** `bash run_cell_campaign.sh rgfn seh` (and
`rgfn drd2`), then add them to the gate sweep. If they were lost, resubmit:
```bash
sbatch -J m16_rgfn_seh  --time 3-00:00:00 --export=ALL,N_TRAJ=30000,N_HUBS=200,STAGE=enum \
    experiments/lsd_hubs/matrix16/submit_cell.sh rgfn seh
```
`python experiments/lsd_hubs/matrix16/manifest.py` prints live per-cell status any time.

**Docking cells (6TD3/ClpP) remain deliberately deferred** — they need GPU docking to score enumerated
children. Wired in the manifest (`run_stage=deferred`, auto-excluded by `launch_surrogates.sh`) but inert.
Note `targets.py` now carries the **calibrated ClpP gate −8.0** (Logs/045) that landed while we worked.

**Heavy artifacts live on scratch, not in git:**
`$SCRATCH/rgfn_runs/lsdflow/matrix16/<gen>_<target>/{sample,enum}/` (records.csv, enum_children.json
83 MB+, enumerated_records.csv, hub_graphs.pkl, hub_terminal.json).

---

## 5. Reading the numbers — three caveats that must travel with them

1. **FragGFN is a control, not a peer.** Its count-once "reactions" are *fragment attachments*, and its
   molecules have no synthesis route (`has_route=0`). Its 2–3.5× edge is a cost-model artifact; the real
   verdict needs the post-hoc retrosynthesis pass. Also its *enumerated* move ≠ its *sampled* move
   (deferred attachment points), so **never merge FragGFN enumerated and sampled F̂**.
2. **SCENT is on a different child policy than the baselines** (free_frag + pre-select-K=20 vs naive).
   That is deliberate and defensible — pre-select-K needs promoted fragments, which only SCENT has
   (it prebuilds 0 for the others) — but it is **not** an apples-to-apples generator comparison. For
   that, use the `*_naive/` dirs (SCENT naive: 1.22×/1.38×, i.e. in line with RxnFlow's 1.23×/1.60×).
3. **⚠️ The FragGFN results come from the DEPRECATED `max_nodes=9` model.** While we were running, the
   other branch fixed FragGFN's fragment cap (9 → the paper's 6; Logs/046: cap-9 gave MW ~664 and
   collapsed DRD2 to ~0 reward) and **renamed the seed-42 run dirs `seed42` → `seed42_maxnodes9`**. Our
   `fraggfn_{seh,drd2}` numbers were computed against those cap-9 checkpoints — the manifest now points
   explicitly at `seed42_maxnodes9` so the results stay reproducible and the cells resolve, but:
   - **`fraggfn_drd2` should be re-run** against the corrected cap-6 model, which is complete and waiting
     at `$SCRATCH/rgfn_runs/experiments/fixed_reward/fraggfn_drd2_maxfrag6/2026-07-23_16-56-06/`
     (1000 candidates). Its cap-9 predecessor is the one Logs/046 showed was broken on DRD2, so treat the
     committed `fraggfn_drd2` row as provisional.
   - **`fraggfn_seh` has no cap-6 re-run yet** (`fraggfn_seh_maxfrag6` does not exist), so there is
     nothing to re-point it at today.
   Re-running a FragGFN cell means sample + enumerate again (its `hub_graphs.pkl` is checkpoint-specific).

4. **The sEH-7.0 gate starves weak-reward pools.** Only 1.2% of RxnFlow-sEH's 232k enumerated children
   clear 7.0 → 70% of its hubs yield exactly 1 mode. At the Logs/034-calibrated bars it recovers
   normally (gate 6 → 2.07×, gate 5 → 2.31×). **Report the sEH column as a gate sweep**, via
   `gate_sweep.sh` → `results/<cell>_thr<gate>/` + `results/gate_sweep/summary.csv`.

---

## 6. Analysis tools added (all pure-CPU, login-safe, per-cell)

| script | question it answers |
|---|---|
| `gate_sweep.sh` | Both strategies re-scored at each target's `threshold_variants` over the existing enumeration. |
| `hub_rank_sensitivity.py` | Does the muddy P_B drive hub selection? (RxnFlow: **no** — 87.5% same hubs without it; SCENT control: yes it matters.) |
| `flow_consistency.py` | Enumeration-derived `F_true(h)` vs per-child F̂; normalization residual; variance decomposition; sampled-vs-enumerated U(h). |
| `hub_flow_estimators.py` | Four F(h) estimators scored on **bias and rank correlation** vs `F_true`; + the well-conditioned conservation test. |

**Gate-sweep coverage: 16 (cell × gate) points committed** in `results/gate_sweep/summary.csv` — the
cross-cell table. sEH cells at gates 5/6/7 and DRD2 at 0.5/0.7/0.9, **except `fraggfn_drd2`, which has
only its base gate 0.5** (I stopped the sweep to commit before the deadline). To finish it, and to add
the RGFN cells once their enumerations land:
```bash
bash experiments/lsd_hubs/matrix16/gate_sweep.sh fraggfn_drd2          # fills 0.7 / 0.9
bash experiments/lsd_hubs/matrix16/gate_sweep.sh rgfn_seh rgfn_drd2    # after 71766/71767 finish
```
Pure CPU over the cached enumerations, and it re-writes `summary.csv` at the end. Two reads worth
noting from the current table: hub-batching's edge grows monotonically as the gate loosens (sEH
2.60→3.10× for SCENT, 1.23→2.32× for RxnFlow), and **best-candidate's rxn/mode is nearly gate-invariant**
— independently reproducing Logs/035's built-in control (its top modes are all high-reward, so the bar
never binds).

Findings (JSON in `results/gate_sweep/`, full detail in the commit messages):
`Σ_x P_F(x|h) = 0.999` **validates that enumeration is exhaustive and recovered P_F is correctly
normalized**; the production `pick_hubs` heuristic is near-unbiased (+0.43 nats) but only a moderate
*ranker* (ρ≈0.6) because its error spread ≈ the entire across-hub spread of `log F_true`; `U(h)` as
shipped is dominated by `Var[log P_F]` (policy peakedness, not epistemic uncertainty) and the **sampled**
U(h) is 40–55× smaller than the enumerated one and undefined for most hubs — **which undercuts the
planned UCB hub-acquisition signal** in the AL phase; and flow conservation at the hub is violated by
−6.05 nats on RxnFlow (it terminates at hubs ~430× more often than its reward justifies).

---

## 7. Open threads (nothing half-built left behind)

- **Prefix / Z-anchored F(h) reconstruction — designed, NOT built.** `F(h) = Z·∏_{s_0→h} P_F/P_B` is
  exact and its disagreement with our R-anchored estimator *is* the per-trajectory TB residual. We already
  store `log_z`; the missing piece is two columns (`log_pf_prefix`, `log_pb_prefix`) — the workers compute
  the full per-step arrays and discard all but the last move. Needs a re-sample per cell. ⚠️ Caveat found:
  **RxnFlow's learned `logZ` (53.07) is *smaller* than its own single-molecule rewards (max log R 63.7) and
  than one hub's child-flow sum (58.71)** — internally inconsistent, so the Z-anchor is unreliable for that
  cell. SCENT (74.65 > 68.10) and FragGFN (97.17) look sane. Suspect either an under-fit `Z` or the
  untrained retro `P_B` inflating `N(h)`; the prefix-vs-suffix pattern would discriminate (constant offset ⇒ Z).
- **`probe_hubs` exists only on `rxnflow_worker`.** Porting it to `scent_worker` is the highest-value next
  step: SCENT's `S(h)` has real spread (p5 = 0.40), so the `(R+N)` vs `(N/S)` check has genuine power there
  (on RxnFlow it is near-tautological, `R/N ~ 1e-6`).
- **Gate sweep coverage:** run for the 6 real cells; **add the RGFN cells once they land.**
- Untouched by design: docking cells; `sweep_campaign.py` / `validation/lsdflow/eval/*` (the other
  agent's evaluator axis — the enumeration artifact set is our clean handoff seam).

---

## 8. Post-merge verification (fast)

```bash
# 1) the shared-file guard survived exactly once + everything compiles
grep -c "isinstance(state, (RSA, RSB, RSC))" validation/lsdflow/adapters/workers/scent_worker.py   # -> 1
python -m py_compile experiments/lsd_hubs/matrix16/{manifest,targets,flow_consistency,hub_flow_estimators,hub_rank_sensitivity}.py \
    validation/lsdflow/adapters/workers/{_artifacts,rgfn_worker,fraggfn_worker,rxnflow_worker,scent_worker}.py
bash -n experiments/lsd_hubs/matrix16/*.sh

# 2) the manifest resolves all 16 cells against scratch
python experiments/lsd_hubs/matrix16/manifest.py

# 3) a committed result still reproduces from the cached enumeration (pure CPU, seconds)
bash experiments/lsd_hubs/matrix16/run_cell_campaign.sh scent seh    # -> hub 1.303 rxn/mode @ gate 7.0
bash experiments/lsd_hubs/matrix16/run_cell_campaign.sh rxnflow seh  # -> hub 2.446 rxn/mode @ gate 7.0
```

> `run_cell_campaign.sh` and `gate_sweep.sh` both default the child policy **per generator** (SCENT →
> `free_frag` + pre-select-K=20; baselines → naive `reward`), so re-running either reproduces the
> committed numbers rather than overwriting them with a different policy's. SCENT's naive control is
> `CHILD_POLICY=reward PREBUILD_K=0 ...` and is preserved in `results/scent_*_naive/`.

> **Worktree-only step, NOT needed after merging into the main checkout:** `link_worktree_data.sh`
> symlinks the gitignored `data/` + `external/` payloads that `git worktree` omits (the pipeline reads
> them via relative paths; `scent_worker` chdir's into `external/scent`). The main checkout already has them.
