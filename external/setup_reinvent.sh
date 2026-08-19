#!/bin/bash
# setup_reinvent.sh — OURS. Install REINVENT 4, a ROUTE-LESS SMILES baseline for the LSD-Flow
# library-efficiency benchmark (the second entrant in the S3-GFN class).
#
# REINVENT 4 (`[loeffler2024reinvent4]`, github.com/MolecularAI/REINVENT4) is AstraZeneca's
# production de-novo design tool: an RNN over SMILES tokens, pretrained on ChEMBL, fine-tuned by
# reinforcement learning (DAP) against a weighted multi-component score. It has NO reaction model —
# its molecules are strings, carry no route, and are emitted with `has_route=0`, so a *library* built
# from them must have its shared intermediates recovered post-hoc (MultiAiZ -> SPARROW). Same role as
# S3-GFN, different mechanism: S3-GFN induces synthesizability by soft regularization, REINVENT does
# not model it at all. It is the baseline a reviewer names first ("why not REINVENT?").
#
# WHY v4.5.11 AND NOT `main` (a packaging decision, NOT a scientific one — REINVENT's RL algorithm
# is DAP in every 4.x release). Three hard constraints pick this tag:
#   1. `bengio2021flow` — the frozen sEH MPNN every entrant is scored by — does
#      `from torch_sparse import coalesce` at MODULE level, so the env needs a prebuilt torch-sparse
#      wheel. Those exist at data.pyg.org for torch 2.5.1 (`+pt25cu124`) but NOT for the
#      `torch==2.12.0` that `main` pins, where it would have to be compiled from source against CUDA.
#   2. The DRD2 oracle (`oracle/drd2_current.pkl`) is a legacy sklearn pickle. v4.5.11 keeps
#      `numpy >=1.21,<2`; v4.8 requires `numpy >=2`, under which that pickle is not reliably loadable.
#   3. v4.5.11 still ships `priors/reinvent.prior` IN-REPO (23 MB). Later tags moved every prior to
#      Zenodo, which adds a download step that would fail on a compute node (no internet).
#   torch 2.5.1 is also exactly the line the working `s3gfn` env runs, so the pyg extension wheels and
#   the sEH stack are a combination already proven on this cluster.
#
# THE SCORING SEAM — no edits to the clone. `reinvent/scoring/importer.py` requires
# `reinvent_plugins.components` to be a NAMESPACE package (it raises if `__file__` is not None) and
# discovers components with `pkgutil.walk_packages` over `sys.path`. So our component lives in OUR
# repo at validation/generators/reinvent/plugins/reinvent_plugins/components/comp_glue_surrogate.py
# and is picked up by putting that `plugins` dir on PYTHONPATH. Do not add __init__.py files under it:
# that would turn it into a regular package and REINVENT would refuse to start.
#
# Run from the repo root, ON A LOGIN NODE (this downloads; compute nodes have no internet):
#   bash external/setup_reinvent.sh
# Idempotent: re-running skips the clone / env create / model fetch if already present.

set -euo pipefail

# THE ENV LIVES ON $SCRATCH, NOT IN miniconda3/envs — a forced choice, not a preference.
# Measured 2026-08-14: /home/markymoo is at 94G of its 110G quota (17G free) and `conda clean -a`
# reports nothing to reclaim, while /scratch has 28T free. One torch+CUDA env is 8-12G and this
# benchmark adds several, so the existing location cannot hold them. Balam and Trillium share
# /scratch, so a prefix env is reachable from both login nodes exactly like the $HOME ones.
#
# Addressed by explicit `-p <prefix>` rather than by appending to `envs_dirs` in ~/.condarc: this is
# a SHARED machine with other agents, and an explicit path changes nothing for anyone else and shows
# the reader where the env actually is. Every caller (driver, submit script) uses the same `-p`.
ENV_PREFIX="${REINVENT_ENV_PREFIX:-/scratch/markymoo/conda_envs/reinvent4}"
PYVER=3.10                                  # v4.5.11's README; its lockfile is resolved for 3.10

REINVENT_ORG=MolecularAI
REINVENT_REPO=REINVENT4
REINVENT_TAG="${REINVENT_TAG:-v4.5.11}"     # see the version rationale above before changing
CLONE_DIR="external/reinvent"

# Must match what v4.5.11's LOCKFILE actually installs, or the pyg extension wheels won't apply.
# Read from requirements-linux-64.lock, NOT pyproject.toml: pyproject says `torch==2.5.1+cu124`
# while the lockfile — which is what the documented install command uses — pins cu121. It also pins
# numpy==1.26.4 (so the legacy DRD2 sklearn pickle stays loadable) and rdkit==2023.9.5.
# cu121 is a bonus: it is byte-for-byte the torch line the working `s3gfn` env runs.
TORCH_VER=2.5.1
TORCH_CUDA=cu121
PYG_FIND_LINKS="https://data.pyg.org/whl/torch-${TORCH_VER}+${TORCH_CUDA}.html"

# The frozen sEH MPNN weights. `bengio2021flow.load_original_model()` DOWNLOADS these on first call
# and caches them inside the installed package; on a compute node that call would fail with no
# internet. We pre-place them here, preferring a copy from the s3gfn env (same file, no network).
SEH_CACHE_SRC="${SEH_CACHE_SRC:-$HOME/miniconda3/envs/s3gfn/lib/python3.10/site-packages/gflownet/models/cache/bengio2021flow_proxy.pkl.gz}"

echo "[setup_reinvent] env=${ENV_PREFIX} python=${PYVER} tag=${REINVENT_TAG} torch=${TORCH_VER}+${TORCH_CUDA}"

# --- 1. Clone REINVENT4 at the pinned tag (under external/, git-ignored). ----------------------
# SHALLOW, AT THE TAG. The full history is 1.49 GB of git objects for a ~900 MB working tree, because
# every prior model (~800 MB of binaries) is versioned in it. `/home/markymoo` is a 110 GB quota with
# ~17 GB free and this benchmark installs several baselines, so the ~1.5 GB saved here is not
# housekeeping. `--branch <tag> --depth 1` leaves HEAD already at the tag; the later `rev-parse`
# still records the exact SHA.
if [ ! -d "${CLONE_DIR}" ]; then
    echo "[setup_reinvent] cloning ${REINVENT_ORG}/${REINVENT_REPO}@${REINVENT_TAG} (shallow)"
    git -C external clone --branch "${REINVENT_TAG}" --depth 1 \
        "https://github.com/${REINVENT_ORG}/${REINVENT_REPO}" reinvent
else
    echo "[setup_reinvent] ${CLONE_DIR} already present — checking out ${REINVENT_TAG}"
    # A shallow clone has only its own tag, so this fails loudly rather than silently benchmarking
    # a different version than the one this script documents.
    git -C "${CLONE_DIR}" checkout "${REINVENT_TAG}" || {
        echo "[setup_reinvent] FATAL: ${CLONE_DIR} exists but cannot check out ${REINVENT_TAG}" >&2
        echo "  (a shallow clone only carries its own tag). Remove the dir and re-run." >&2
        exit 1
    }
fi
# Record the exact SHA the benchmark ran against; the tag could be moved upstream, a SHA cannot.
git -C "${CLONE_DIR}" rev-parse HEAD > "${CLONE_DIR}/COMMIT_PINNED.txt"
echo "[setup_reinvent] pinned SHA $(cat "${CLONE_DIR}/COMMIT_PINNED.txt")"

# --- 2. Create the dedicated conda env (on $SCRATCH — see the note at the top). -----------------
if [ ! -d "${ENV_PREFIX}" ]; then
    echo "[setup_reinvent] creating conda env at ${ENV_PREFIX}"
    mkdir -p "$(dirname "${ENV_PREFIX}")"
    conda create -y -p "${ENV_PREFIX}" "python=${PYVER}"
else
    echo "[setup_reinvent] conda env ${ENV_PREFIX} already exists — skipping create"
fi

RUN="conda run --no-capture-output -p ${ENV_PREFIX}"

# --- 3. Install REINVENT4 the way v4.5.11 documents: LOCKFILE, then the package --no-deps. -----
# NOT `python install.py` — that helper is a later addition and does not exist at this tag. The
# lockfile is strictly better for a benchmark anyway: it pins every transitive dependency, so the
# env is reproducible rather than merely resolvable.
LOCKFILE="${CLONE_DIR}/requirements-linux-64.lock"
[ -s "${LOCKFILE}" ] || {
    echo "[setup_reinvent] FATAL: ${LOCKFILE} missing — the tag's install method changed." >&2
    echo "  Check the clone's README 'Installation' section before editing this script." >&2
    exit 1; }
echo "[setup_reinvent] installing REINVENT4 dependencies from the lockfile"
${RUN} pip install -r "${LOCKFILE}"
echo "[setup_reinvent] installing the reinvent package (--no-deps; the lockfile already resolved them)"
( cd "${CLONE_DIR}" && ${RUN} pip install --no-deps . )

# The lockfile is the authority on the torch build; if it ever stops matching what the pyg
# find-links URL above targets, the extension wheels silently do not apply and `bengio2021flow`
# fails at import time inside a job rather than here.
ACTUAL_TORCH=$(${RUN} python -c "import torch; print(torch.__version__)" | tr -d '\r')
[ "${ACTUAL_TORCH}" = "${TORCH_VER}+${TORCH_CUDA}" ] || {
    echo "[setup_reinvent] FATAL: lockfile installed torch ${ACTUAL_TORCH}, but this script targets" >&2
    echo "  ${TORCH_VER}+${TORCH_CUDA} for the pyg extension wheels. Update TORCH_VER/TORCH_CUDA." >&2
    exit 1; }
echo "[setup_reinvent] torch ${ACTUAL_TORCH} matches the pyg find-links target"

# --- 4. Add the sEH proxy stack (the FROZEN reward every entrant shares). -----------------------
# torch-geometric is pure python; scatter/sparse/cluster are compiled extensions and MUST come from
# the pyg find-links index for this exact torch build. `gflownet` is installed --no-deps because its
# metadata hard-pins an older torch, which would silently downgrade the stack we just installed
# (the same trap documented in setup_s3gfn.sh).
echo "[setup_reinvent] installing torch-geometric + pyg extensions + recursion gflownet"
${RUN} pip install "torch-geometric==2.6.1"
${RUN} pip install --no-index -f "${PYG_FIND_LINKS}" torch-scatter torch-sparse torch-cluster
${RUN} pip install --no-deps "gflownet @ git+https://github.com/recursionpharma/gflownet.git"
# `pip` will WARN here that gflownet wants torch==2.1.2 / torch-geometric==2.4.0. That warning is the
# --no-deps trap working as intended: honouring those pins would downgrade the stack the lockfile
# just installed. Ignore it.

# Two dependencies --no-deps leaves behind, both genuinely required:
#   omegaconf — `gflownet/__init__.py` imports `.config`, which imports it, so ANY `from
#               gflownet.models import bengio2021flow` fails without it. Our own run driver parses
#               its YAML with OmegaConf too, so it is needed twice over. It is NOT in REINVENT's
#               lockfile.
#   setuptools<81 — several bundled REINVENT components still import `pkg_resources`, removed in
#               setuptools 81. Without it they fail to import and the plugin registry silently comes
#               up short, which makes the discovery check below less meaningful than it looks. Same
#               pin, same reason, as setup_s3gfn.sh.
${RUN} pip install omegaconf "setuptools<81"

# --- 5. Pre-place the sEH weights so no compute-node job ever needs the network. ----------------
GF_MODELS=$(${RUN} python -c "import gflownet.models as m, os; print(os.path.dirname(m.__file__))" | tr -d '\r')
mkdir -p "${GF_MODELS}/cache"
if [ -s "${GF_MODELS}/cache/bengio2021flow_proxy.pkl.gz" ]; then
    echo "[setup_reinvent] sEH weights already cached"
elif [ -s "${SEH_CACHE_SRC}" ]; then
    echo "[setup_reinvent] copying sEH weights from the s3gfn env (no network)"
    cp "${SEH_CACHE_SRC}" "${GF_MODELS}/cache/"
else
    echo "[setup_reinvent] downloading sEH weights (login node only)"
    ${RUN} python -c "from gflownet.models import bengio2021flow; bengio2021flow.load_original_model()"
fi

# --- 6. Verify. Every check below has failed for someone here at least once. --------------------
echo "[setup_reinvent] verifying"
${RUN} reinvent --help > /dev/null
[ -s "${CLONE_DIR}/priors/reinvent.prior" ] || {
    echo "[setup_reinvent] FATAL: priors/reinvent.prior missing — wrong tag?" >&2; exit 1; }

PLUGINS_DIR="$(pwd)/validation/generators/reinvent/plugins"
# Login nodes cap threads per user; torch's OpenMP grabs one per core on sight and dies with
# "libgomp: Thread creation failed: Resource temporarily unavailable". Only affects this check —
# batch jobs get their own allocation.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH="${PLUGINS_DIR}" ${RUN} python - <<'PY'
import numpy as np, torch
torch.set_num_threads(1)
from gflownet.models import bengio2021flow
from rdkit import Chem

# (a) the sEH MPNN loads from cache and produces a finite score
m = bengio2021flow.load_original_model()
g = bengio2021flow.mol2graph(Chem.MolFromSmiles("CCO"))
v = float(m(bengio2021flow.mols2batch([g])).view(-1)[0])
assert np.isfinite(v), "sEH MPNN returned a non-finite score"
print(f"  sEH MPNN ok (ethanol -> {v:.4f})")

# (b) our scoring component is DISCOVERED by REINVENT's plugin importer, not merely importable.
#     A file that imports fine but is not registered would fail only at run time, hours later.
#     The registry key is `component.__name__.lower()` (importer.py), and get_components() looks it
#     up after .lower().replace("-","").replace("_","") -- so the class name GlueSurrogate is
#     addressed in the TOML as [stage.scoring.component.GlueSurrogate] but keyed as "gluesurrogate".
from reinvent.scoring.importer import get_registry
reg = get_registry()
assert "gluesurrogate" in reg, f"GlueSurrogate not registered; found {sorted(reg)[:12]}..."
comp_cls, param_cls = reg["gluesurrogate"]
assert comp_cls.__name__ == "GlueSurrogate", comp_cls
# The param class must carry the three fields the generated TOML sets, as LISTS: config.py's
# collect_params() folds one dict per endpoint into {key: [values]}.
import dataclasses
fields = {f.name for f in dataclasses.fields(param_cls)}
assert {"reward_type", "model_path", "device"} <= fields, fields
print(f"  plugin discovered and typed ({len(reg)} components registered)")

# (c) numpy is <2 so the legacy DRD2 sklearn pickle stays loadable
assert np.__version__.startswith("1."), f"numpy {np.__version__} breaks the DRD2 oracle pickle"
print(f"  numpy {np.__version__} ok, torch {torch.__version__}")

# (d) THE ONE THAT MATTERS: build the component the way REINVENT's get_components() does and score
#     real molecules through it. (a)-(c) can all pass while the component itself is broken, and the
#     next thing to find out would be 1000 RL steps into a job.
comp = comp_cls(param_cls(reward_type=["seh_proxy"], model_path=[""], device=["cpu"]))
s = comp(["CCO", "c1ccccc1C(=O)NC2CCN(CC2)Cc3ccccc3", "not_a_smiles"]).scores[0]
assert np.isfinite(s[0]) and np.isfinite(s[1]), f"valid molecules scored non-finite: {s}"
# ComponentResults is explicit that failure is NaN, never 0 -- scoring an unparseable molecule 0
# would teach the policy it is merely BAD rather than unscoreable.
assert not np.isfinite(s[2]), "an invalid SMILES must score NaN, not 0"
print(f"  scored through the component: {[None if x != x else round(float(x), 3) for x in s]}")
PY

cat <<EOF

[setup_reinvent] DONE.
  env      : ${ENV_PREFIX}   (a PREFIX env — always address it with 'conda run -p', never '-n')
  clone    : ${CLONE_DIR} @ $(cat "${CLONE_DIR}/COMMIT_PINNED.txt")
  prior    : ${CLONE_DIR}/priors/reinvent.prior
  plugins  : ${PLUGINS_DIR}   (export PYTHONPATH to this when running REINVENT)

Next:
  conda run -p ${ENV_PREFIX} python validation/generators/reinvent/run_reinvent_fixed.py \\
      --cfg validation/configs/reinvent_seh_fixed.yaml --root-dir \$SCRATCH/rgfn_runs/experiments
EOF
