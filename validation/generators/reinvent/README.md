# REINVENT 4 entrant

A **route-less SMILES baseline** for the LSD-Flow library-efficiency benchmark, in the same class as
S3-GFN: it generates molecules as strings, has no reaction model, and emits `has_route=0`. Any
library built from its pool must have its shared intermediates recovered *post-hoc*
(MultiAiZ → SPARROW). It is the baseline a reviewer names first.

Upstream: [MolecularAI/REINVENT4](https://github.com/MolecularAI/REINVENT4), `[loeffler2024reinvent4]`.
Installed by `external/setup_reinvent.sh`; **not vendored**.

## Layout

| file | role |
|---|---|
| `fixed_reward.py` | The frozen reward providers (`SEHFrozenReward`, `DRD2FrozenReward`) — a per-adapter copy of the classes the RxnFlow / FragGFN / S3-GFN adapters use, so every entrant optimizes the *identical* function. |
| `plugins/reinvent_plugins/components/comp_glue_surrogate.py` | The REINVENT scoring component exposing those providers. |
| `run_reinvent_fixed.py` | The one-shot driver: RL → sample from the trained agent → emit a candidate dataset. |
| `../../configs/reinvent_{seh,drd2}_fixed.yaml` | Run configs (the only per-cell knobs). |
| `../../../experiments/lsd_hubs/campaign/submit_reinvent.sh` | SLURM wrapper (`TARGET=`, `SEED=`). |

## Four things to know before changing anything

**1. The plugin lives here, not in the clone.** `reinvent/scoring/importer.py` *requires*
`reinvent_plugins.components` to be a **namespace package** — it raises outright if the package has a
`__file__` — and then discovers components by walking `sys.path`. So any directory on `PYTHONPATH`
holding `reinvent_plugins/components/comp_*.py` contributes components. That is the upstream-supported
extension point, which is why the clone stays pristine and this file is under version control.
**Never add `__init__.py` under `plugins/`** — that makes it a regular package and REINVENT refuses
to start. `setup_reinvent.sh` verifies discovery (not merely importability), because a component that
imports fine but is not *registered* fails only at run time, hours later.

**2. The pool comes from the trained agent, not the RL trace.** The driver invokes REINVENT twice:
`staged_learning`, then `sampling` from the resulting checkpoint. The RL trace contains every
molecule REINVENT ever tried, including tens of thousands from early untrained steps; harvesting it
would hand the downstream top-500 pool builder a much larger and differently-distributed candidate
set than S3-GFN's 2,000 post-training samples, and the two pools would stop being comparable. This
mirrors `run_s3gfn_fixed.py::_sample_pool`.

**3. Shaping lives in the config, not the code.** The component returns the **raw** oracle value —
the scale the benchmark's mode gate is defined on (sEH `> 7.0`, DRD2 `> 0.5`). REINVENT maps it into
`[0, 1]` with a declarative `transform` written into the generated TOML from `reward.transform` in
the YAML. The sEH transform is deliberately *not* centred on 7.0: transforming at the gate would make
REINVENT optimize the benchmark's own threshold, which is circular. DRD2 gets no transform at all —
it is already a probability.

**4. The diversity filter is on by design.** REINVENT's RL is reward-*maximizing* and will converge
onto one scaffold. A collapsed pool cannot reach the 100-mode deliverable, and we would only discover
that after paying ~2.25 h of MultiAiZ route discovery on it. It is also the charitable setting — what
a chemist chasing a diverse library would switch on — and it is recorded in the config so the choice
is auditable.

## Version pin, and why

`v4.5.11`, not `main`. This is a **packaging** decision, not a scientific one: REINVENT's RL
algorithm is DAP in every 4.x release. Three hard constraints select it:

1. `bengio2021flow` (the frozen sEH MPNN) does `from torch_sparse import coalesce` at *module* level,
   so the env needs a prebuilt `torch-sparse` wheel. Those exist at data.pyg.org for torch 2.5.1
   (`+pt25cu121`) but not for the `torch==2.12.0` that `main` pins, where it would have to be
   compiled from source against CUDA.
2. The DRD2 oracle is a legacy sklearn pickle. v4.5.11's lockfile pins `numpy==1.26.4`; v4.8
   requires `numpy >=2`, under which that pickle is not reliably loadable.
3. v4.5.11 still ships `priors/reinvent.prior` in-repo. Later tags moved every prior to Zenodo, a
   download that would fail on a compute node.

**Install from the lockfile, not `install.py`.** That helper does not exist at this tag — it is a
later addition, and assuming it existed was worth one failed build. v4.5.11 documents
`pip install -r requirements-linux-64.lock` followed by `pip install --no-deps .`, which is strictly
better for a benchmark: every transitive dependency is pinned, so the env is reproducible rather than
merely resolvable. **Read the lockfile, not `pyproject.toml`, for what actually gets installed** —
pyproject says `torch==2.5.1+cu124` while the lockfile pins **cu121**, and it is the lockfile that
runs. Python is **3.10** (the README's version; the lockfile is resolved for it).

cu121 is byte-for-byte the torch line the working `s3gfn` env runs, so the pyg extensions and the sEH
stack are a combination already proven on this cluster. `setup_reinvent.sh` asserts the installed
torch matches the version its pyg find-links URL targets, because a mismatch would otherwise surface
as an import error inside a job hours later. The resolved SHA is written to
`external/reinvent/COMMIT_PINNED.txt` at setup time — a tag can be moved upstream, a SHA cannot.

## Seeding is ours, not REINVENT's

`_seeded_launcher.py` exists because **v4.5.11 cannot be seeded through its own interface.**
`Reinvent.py` reads `seed` from the TOML but gates the call on the *command-line* flag and then
passes the *config* value, while `ReinventConfig` is declared `extra="forbid"` with no `seed` field —
so a TOML carrying `seed` is rejected by pydantic, `input_config.get("seed")` can only be `None`, and
`set_seed(None)` returns immediately. No invocation of REINVENT 4.5.11 seeds its RNGs, and it says
nothing while failing to.

That matters here specifically: every cell runs at three seeds and is reported with error bands.
Unseeded runs would still differ (torch seeds from entropy) so the spread would be real, but they
would not be reproducible, and "3 seeds" in a paper implies a reader can regenerate them. The
launcher calls `set_seed` itself and then defers to `main_script()` verbatim; `PYTHONHASHSEED` is
exported by the driver into the subprocess environment, since an in-process assignment cannot affect
an interpreter that has already started. **Do not add a `seed` key to the generated TOML** — it is a
hard validation error, and the generated file says so.

## The oracle budget is reported, not equalized

1000 RL steps × batch 128 ≈ **128,000 frozen-reward calls**, the same order as S3-GFN's 5,000 × 64.
REINVENT is often benchmarked at PMO's 10,000-call budget instead, which is ~13× less than every
other entrant here receives, so we run it at the generous end: the baseline was over-fed, not
starved. This follows the project's standing convention of reporting asymmetries rather than forcing
a parity constraint (cf. Logs/056's "surrogate calls ~500 vs 155,764").

## Running

```bash
# once, on a LOGIN node (downloads; compute nodes have no internet)
bash external/setup_reinvent.sh

# smoke first — 2 h max on -p debug, near-instant start
TARGET=seh SEED=42 STEPS=5 N_SAMPLES=50 \
  sbatch -p debug -t 00:30:00 experiments/lsd_hubs/campaign/submit_reinvent.sh

# a full cell
TARGET=seh SEED=42 sbatch experiments/lsd_hubs/campaign/submit_reinvent.sh
```

Output: `$SCRATCH/rgfn_runs/experiments/fixed_reward/reinvent_<target>/seed<N>/fixed_reward/candidates/`.

**Next stage:** run the mode-saturation pre-flight on the pool *before* route discovery, then
`submit_competitor_routes.sh`. MultiAiZ is ~2.25 h per N=500 pool and its cache key is the pool, not
the molecule — never spend it on a pool that cannot reach 100 modes.

The env is a **prefix** env on `/scratch` (`/scratch/markymoo/conda_envs/reinvent4`) because
`/home/markymoo` is at 94G of its 110G quota. Address it with `conda run -p`; `-n reinvent4` will not
find it.
