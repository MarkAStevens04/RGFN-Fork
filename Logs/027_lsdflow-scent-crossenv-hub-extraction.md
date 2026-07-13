# sEH / SCENT — LSD-Flow: cross-env hub extraction from the cost-aware flow field
**Date:** 2026-07-10, ~3pm

## Question

Can we run our post-hoc "hub" analysis on SCENT — the cost-aware, synthesizable generator that
lives in its own, incompatible software environment — and does SCENT's internal flow field contain
the same batchable scaffolds we found in RGFN?

## Context & Summary

Entry `025` built **LSD-Flow**: a way to look inside a *already-trained* reaction generator and pull
out "hubs" — pre-terminal scaffolds from which many distinct, high-reward molecules each branch off
in a single reaction, so a chemist could build the scaffold once and diversify late. So far it only
ran on **RGFN**. The paper's central comparison (the design doc's §8) is between RGFN and **SCENT**
(entry `017`): SCENT bakes synthesis-cost awareness into *training*, and the thesis is that the same
batchable structure LSD-Flow harvests *post-hoc* is present even in a model that was optimised for
cost — including whether the hubs LSD-Flow finds coincide with the intermediates SCENT itself
learned to promote. To get there we first need SCENT *inside* the analysis at all, and that has two
obstacles: SCENT can't share a process with our code (its package is also named `rgfn`, in a
separate conda env), and — until entry `024` — its trained backward policy couldn't even be reloaded
from a checkpoint (needed for the flow math). Entry `024` fixed the second obstacle (a sidecar file
that makes the backward policy recover exactly). This entry clears the first: a cross-environment
bridge that lets the analysis sample SCENT in its own environment and hand the results back, then
validates the whole path end-to-end on the trained 5,000-iteration SCENT sEH model.

## Answer

The hub analysis now runs on SCENT with **no changes to the analysis harness itself** — a
subprocess worker samples SCENT in its own environment and returns the same data records RGFN
produces, so every downstream step (hub ranking, batch selection, cost/diversity, the
trajectory-balance consistency check) works unchanged. SCENT's flow field **does** contain batchable
hubs: on a 2,000-molecule validation sample, 19 scaffolds each spawn 2–3 distinct high-reward
products (best sEH up to 8.2), and building a shared hub once instead of each product independently
cuts reactions-per-distinct-compound from 3.9 to 2.5 — the same amortization seen in RGFN. The
trained backward policy reloaded **exactly** (via the entry-024 sidecar), so these numbers reflect
SCENT's real, cost-tilted policy, not a random reload. One preliminary hint worth chasing: SCENT's
internal flow-consistency signal looks *higher* than RGFN's (a flow-vs-visitation correlation of
0.35 at 2k trajectories vs RGFN's near-zero at 10k), but the samples are different sizes so this
isn't yet a real comparison — the full matched-size run is queued to settle it.

## Relevance to our Publication

The RGFN-vs-SCENT hub-coincidence study is the spine of the methods paper (Digital Discovery, or a
NeurIPS AI4Science / ICLR-MLDD workshop): it shows batchable synthetic structure is latent in *any*
trained reaction GFlowNet, not a quirk of one model. A reviewer's first question about a post-hoc
analysis is "does it apply uniformly across generators, or just the one you built it on?" — being
unable to run it on the cost-aware baseline would be a visible gap. This entry makes the comparison
possible and, in passing, shows the analysis is genuinely generator-agnostic (the harness didn't
change to add a second model family). It also sets up the specific, publishable question of whether
cost-aware training preserves or breaks the internal trajectory-balance the flow math assumes.

## Next Experiments

**Refining for publication**
- **Full-scale matched run (queued, job 70179):** 30,000 trajectories, mode definition matched to the
  RGFN headline (sEH ≥ 7.0), so SCENT's hub density and flow-consistency numbers are directly
  comparable to entry `025`'s RGFN DAG at the same size. Copy the small artifacts back to
  `validation/lsdflow/results/scent_seh_pilot/` when it lands.
- **Noise-floor / N controls:** the flow-vs-visitation correlation gap (SCENT 0.35 @ 2k vs RGFN 0.086
  @ 10k) must be re-measured at matched N before it can be claimed as a cost-guidance effect.

**Next steps in project**
- **The hub-coincidence study (§8, the paper's spine):** build `validation/lsdflow/analysis/hub_coincidence.py`
  — compare the hubs LSD-Flow finds on SCENT to the intermediates SCENT promoted into its dynamic
  library (`additional_fragments/fragments_4000.json`, the frozen final snapshot): Jaccard of the
  hub sets, rank-correlation of flow-value vs SCENT's own utility, and the visitation-vs-balance
  agreement as a trajectory-balance integrity test.
- **Phase-2 for SCENT:** exhaustive one-reaction child enumeration on a frozen dynamic library (the
  RGFN `enumerate_children` analogue) so a selected SCENT hub's *full* neighborhood is measured, not
  just what sampling hit.
- **Other reward targets** (DRD2 has a 5,000-iter patched run; 6TD3/ClpP are 400-iter docking runs)
  and the remaining models (FragGFN control, RxnFlow robustness row).

## Addendum — frozen full library, enumeration, and synthesis-recipe logging (2026-07-10, same day)

Building the enumeration surfaced a correctness fork that reshaped the SCENT analysis.

**The base-library restriction (and the fix).** SCENT's cost-aware training promotes high-reward
intermediates into its fragment vocabulary (a **Dynamic Library**): the trained sEH model has 418
base + **1,600 promoted** fragments (the checkpoint reserves 4,418 embedding rows; the
`fragments_4000.json` snapshot filled 1,600). But building the model from a checkpoint alone leaves
it **restricted to the 418 base fragments** (`current_fragments=418`, confirmed live) — the promoted
embeddings load but stay masked off until `on_update_fragments_library` fires. So the first SCENT
sampled run (job 70179, since **cancelled**) was analyzing a crippled model that couldn't use — or
route flow through as reactants — the very intermediates the coincidence study (§8) is about.
**Decision (researcher): analyze the faithful *full* SCENT** — freeze the dynamic library to its
final promoted snapshot for **both** sampling and enumeration. The freeze fires
`trainer.on_update_fragments_library` (the exact hook training uses), which grows the env reactant
set, the forward policy's fragment embedding, and the cost-guided backward policy's cost table
together; verified end-to-end (assign_log_probs finite, sampled trajectories route through promoted
fragments).

**Enumeration (`enumerate_hub_children`) — the thing that actually measures diversification.**
Sampling only credits a hub with a child when a trajectory stops exactly one reaction later, so it
badly under-counts a hub's true one-reaction neighborhood. The SCENT enumerator (worker-side mirror
of `rgfn_enumerate`, on the frozen library) walks the env's A→B→C action spaces to build **every**
one-reaction terminal child. Validated exhaustive: a depth-0 fragment hub has **3,975** one-reaction
paths (≈10× the RGFN 418-library case) and recovers **2/2** sampled children once the per-hub cap
exceeds the neighborhood — so the real run needs a generous `--enumerate-max-children` (12k).

**Cost model — decisions locked (build pending the recipe re-run).** SCENT's promoted fragments are
NOT free like the 418 purchasable base blocks — each took reactions to build. Decisions: **(1)** cost
unit = reactions (primary, RGFN-comparable) + SCENT's $-cost (secondary); **(2)** each *distinct*
promoted fragment used anywhere in a costed library is built + charged **exactly once**, in both the
hub and independent plans (so it can't be double-counted — "you'd make it once whether or not you
batch"); **(3)** **exact nesting** — a fragment built from another fragment expands recursively.

**Synthesis-recipe logging (for exact nesting AND chemist usability).** (2) and (3) — and a future
"synthesize these intermediates" view for chemists — need each promoted fragment's actual synthesis
**route**, which is only observable during training (once promoted, the model uses it atomically).
New `validation/generators/scent/recipe_logging.py` monkeypatches `DynamicLibrary` (clone stays
pristine, like the entry-024 sidecar) to record each promoted fragment's min-reaction route
(ordered reaction steps: reaction SMARTS + reactants + product) into the `fragments_<N>.json`
snapshot; wired as `run_scent_fixed.py --log-recipes`. Validated: a 7-iteration run with early
promotion wrote **20 routes for 20 promoted fragments** (e.g. a correct 2-step benzimidazole-
formation → N-alkylation route). Full recipe-logging sEH re-run launched (**job 70180**, ~12h); it
supersedes `2026-07-07_17-16-09` as the SCENT sEH anchor.

**Next (immediate):** build the nested-amortization cost model (`glue`/`validation` cost metric)
that consumes the routes — requires capturing each analyzed molecule/hub's own dynamic-fragment
composition from the analysis trajectories (not yet in the `FlowRecord` schema). Then run the full
frozen sEH analysis (`validation/lsdflow/submit_scent_seh.sh`, now with enumeration) on job 70180's
checkpoint.

# Re-creation

## Relevant Files

Root: `./` (repo root). GFN **inference only** (no docking); runs on a Balam/Trillium login node
(validation) or a compute node (full run).

**Cross-env SCENT adapter (ours, new — the phase-3 bridge):**
- `./validation/lsdflow/adapters/workers/scent_worker.py` — **new.** Runs *inside the `scent` env*.
  Rebuilds SCENT's `objective` + the pure-policy `valid_sampler` from a gin config + checkpoint
  (the `verify_pb_recovery.py` recipe), loads `last_gfn.pt` (forward policy + logZ) **and** the
  `guidance_models.pt` sidecar so the backward policy P_B is the *trained* one (entry `024`, §5),
  samples trajectories, and writes the canonical `records.csv` (the `FlowRecord` schema the harness
  reads) + `visit_counts.json` + `meta.json`. The §2 flow-extraction algorithm is a self-contained
  copy of `glue.samplers.lsdflow.rgfn_extract` — it can't be *imported* here (its `import rgfn`
  would resolve to our rgfn, not SCENT's), but SCENT's fork exposes the same
  `Trajectories`/`ReactionState*`/`Molecule` API so the algorithm transfers verbatim.
- `./validation/lsdflow/adapters/scent_adapter.py` — **new.** The in-process client mirror (runs in
  the `rgfn` harness env). `sample_flow_records` shells to the worker under `conda activate scent`
  with the scent env's torch-bundled CUDA libs on `LD_LIBRARY_PATH` (the `~/bin/rgfn-smoke-env.sh`
  trick, cluster-agnostic — no `module load`), then reads the worker's files back into a
  `FlowSample` — identical downstream contract to `RGFNAdapter`. Phase-2 `enumerate_hub_children`
  raises `NotImplementedError` (not built for SCENT yet).
- `./validation/lsdflow/adapters/registry.py` — **modified.** `get_adapter("scent")` now constructs
  `SCENTAdapter` (was `NotImplementedError`); status flipped to "live".
- `./validation/lsdflow/submit_scent_seh.sh` — **new.** SLURM script for the full-scale run: activates
  the `rgfn` harness env (the worker self-activates `scent`), 30k trajectories, no enumeration,
  paper-comparable modes at sEH ≥ 7.0.

**Reused (entry 024 / 025, unchanged):**
- `./validation/generators/scent/{guidance_io.py,verify_pb_recovery.py,fixed_reward.py}` — the P_B
  sidecar save/load + the proven in-scent-env build recipe the worker follows.
- `./glue/samplers/lsdflow/rgfn_extract.py` — the model-agnostic flow extractor the worker mirrors.
- `./validation/lsdflow/harness/{run.py,config.py}`, `./validation/lsdflow/dag/**`,
  `./validation/lsdflow/metrics/**`, `./glue/samplers/lsdflow/**` — the analysis machinery, unchanged.
- `./validation/configs/scent_seh_fixed.gin` — the gin config the worker rebuilds SCENT from.

**Model under analysis (entry `024`, job 70066 — the SCENT anchor, matches the RGFN sEH anchor):**
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-07_17-16-09/train/checkpoints/last_gfn.pt`
  — the **5,000-iteration** patched SCENT sEH run (`operative_config.gin: Trainer.n_iterations=5000`;
  `final_paths.csv` 1001 rows; dynamic-library snapshots through `fragments_4000.json`, i.e. no
  promotion in iters 4001–5000, matching the design doc §9). Trained logZ sum = 74.33 (reproduced
  live at load).
- `.../train/checkpoints/guidance_models.pt` — the P_B guidance sidecar (entry `024`), 2 MLPs
  (`policies.0.decomposable_prediction_model`, `policies.1.cost_prediction_model`).

**Results (ours, this entry):**
- Validation runs written to `$SCRATCH/rgfn_runs/lsdflow/scent_seh_smoke{,2}` (throwaway, removed).
- Full-scale run (job 70179) → `$SCRATCH/rgfn_runs/lsdflow/scent_seh_70179/`; small artifacts to be
  copied into `./validation/lsdflow/results/scent_seh_pilot/` on completion.

**Job Logs:** validation ran on the login-node A100 (session scratchpad `scent_seh_2k.log`); the
full run is SLURM job **70179** (`/scratch/markymoo/rgfn_runs/lsdflow_scent_seh-70179.{out,err}`).

## Relevant Versions

Branch `Hub-Analysis` (base commit `6c77a26`, "RGFN Hub analysis, pre-SCENT integration"). The four
files above (`scent_worker.py`, `scent_adapter.py`, the `registry.py` edit, `submit_scent_seh.sh`)
plus this log and the README/RESEARCH_CONTEXT index rows are **not yet committed**.
[TODO — add commit hash after committing.]

## Relevant Resources

**Sources**
- LSD-Flow design spec: `docs/LSD_FLOW_PROPOSAL.md` — §4b (adapter is a subprocess/RPC contract, not
  one imported class), §5 (why P_B must be the trained policy), §8 (the hub-coincidence study), §9
  (SCENT prerequisite: P_B recoverability + retrain), §10.3 (SCENT is build-order step 3).
- Entry `024` — SCENT P_B recovery (the guidance sidecar the worker loads); entry `025` — the
  LSD-Flow machinery the SCENT DAG feeds; entry `017` — the SCENT baseline.
- `[gainski2025scent]` (arXiv:2506.19865) — SCENT (cost-guided backward policy + dynamic library).
- `[malkin2022trajectorybalance]` (arXiv:2201.13259) — the flow-recovery identity the math inverts.
- `[koziarski2024rgfn]` — the reaction micro-step (A/B/C + stop) state machine both models share.

**Packages**
- `scent` conda env (py3.11; SCENT's `rgfn` fork resolves to `external/scent/rgfn`) — the worker's
  `objective`/`valid_sampler`; `guidance_io` for the sidecar; dgl/graphbolt need the env's
  torch-bundled `nvidia/*/lib` on `LD_LIBRARY_PATH`.
- `rgfn` conda env (py3.11) — the harness + client; login-node CUDA via `~/bin/rgfn-smoke-env.sh`.

## Method

1. **Verified transferability before building:** confirmed the SCENT fork's `reaction_api` exposes the
   same `ReactionStateA/B/C/Terminal` micro-step machine and its `Trajectories`/`RewardOutput`/
   `Molecule` expose the exact API `rgfn_extract` uses (`get_forward_log_probs_flat`,
   `get_backward_log_probs_flat`, `get_reward_outputs().{log_reward,proxy}`, `_states_list`,
   `Molecule.{smiles,rdkit_mol,num_reactions}`), and that all four patched SCENT checkpoints carry the
   `guidance_models.pt` sidecar.
2. **Built the bridge:** `scent_worker.py` (scent-env sampler + vendored flow extraction) +
   `scent_adapter.py` (rgfn-env client, subprocess env with the scent CUDA libs) + `registry.py`
   wiring. `py_compile` + an import/construct smoke of the client in the `rgfn` env.
3. **End-to-end smoke (login node, N=60):**
   ```
   source ~/bin/rgfn-smoke-env.sh
   python -m validation.lsdflow.harness.run --model scent \
     --config-path validation/configs/scent_seh_fixed.gin \
     --checkpoint /scratch/.../scent_seh/2026-07-07_17-16-09/train/checkpoints/last_gfn.pt \
     --reward-name seh --n-trajectories 60 --sample-batch-size 30 --out-dir <scratch>
   ```
   Confirmed: forward policy + logZ load clean (real-missing=0), guidance sidecar loads (2 keys),
   logZ=74.3259 (matches the trained value), 60 trajectories → 60 records / 273 nodes, full pipeline
   runs. (0 multi-child hubs at N=60 is expected — a hub only gets a *sampled* child when a trajectory
   stops exactly one reaction later; RGFN needed thousands too.)
4. **Non-degeneracy validation (login node, N=2000):** same command at `--n-trajectories 2000
   --sample-batch-size 200` → 19 multi-child hubs, all six hub strategies populated, acquisition +
   cost + diagnostics computed (below).
5. **Submitted the full-scale run:** `sbatch validation/lsdflow/submit_scent_seh.sh` → job **70179**
   (30k trajectories, modes at sEH ≥ 7.0), pending at time of writing.

## Results

**Build / load (both validation runs, sEH checkpoint 2026-07-07_17-16-09):** forward policy + logZ
load with real-missing = 0; guidance sidecar loads 2 keys (`policies.0.decomposable_prediction_model`,
`policies.1.cost_prediction_model`), unmatched = 0; **logZ = 74.3259** (= entry `024`'s trained value,
confirming the trained checkpoint + P_B loaded correctly).

**HubDAG structure (N = 2,000 trajectories):**

| quantity | value |
|---|---|
| valid terminals | 2,000 |
| total hubs | 1,972 |
| multi-child hubs (≥2 distinct one-reaction products) | **19** |
| max children per hub | 3 |
| mean depth of multi-child hubs | 2.89 |
| distinct molecule nodes | 7,132 |
| logZ | 74.326 |

(RGFN for reference, entry `025` @ 10k: 8,183 hubs, 613 multi-child, mean depth 2.82 — not
size-matched to this 2k run; the 30k job 70179 provides the matched comparison.)

**Acquisition — 96-target batch, reactions-per-mode (modes structure-only at the validation stage;
lower = cheaper):** 39 molecules / 37 modes from 19 hubs; hub-amortized **2.49** rxn/mode vs
independent **3.92** (saved 53 reactions). As with RGFN, the flow strategies tie the `parent_of_topk`
control on cost — the differentiation is a diversity story, not a cost one.

**Diagnostics (TB-integrity, 19 multi-child hubs, N=2k):** Pearson(consensus `log F`, reward-free
visitation `log F`) = **0.353**. Higher than RGFN's 0.086 (entry `025`, but @ 10k) — a candidate
"does cost guidance preserve trajectory balance?" signal, pending the matched-N run before it can be
claimed.
