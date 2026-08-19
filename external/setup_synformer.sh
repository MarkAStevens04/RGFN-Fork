#!/bin/bash
# setup_synformer.sh — OURS. Install SynFormer, the REACTION-AWARE, non-GFN baseline for the
# LSD-Flow library-efficiency benchmark. Replaces the old placeholder stub.
#
# WHY THIS ENTRANT MATTERS MORE THAN THE OTHERS. REINVENT, Saturn and S3-GFN are all route-LESS: they
# emit molecule strings and a planner has to recover routes afterwards. SynFormer (`[gao2025synformer]`)
# generates molecules AS SYNTHETIC PATHWAYS — postfix notation over reaction templates and purchasable
# building blocks — so its molecules carry a route BY CONSTRUCTION, exactly as ours do
# (`has_route=1`). That makes it the one cell that separates the two things our headline conflates:
# is the win from the FLOW FIELD, or merely from being reaction-grounded at all? No ablation on our
# own generators can answer that; docs/LSD_FLOW_BENCHMARK_PLAN.md section 7 names this gap explicitly.
#
# NO ENAMINE LICENCE NEEDED, despite the README. Upstream says the building blocks are "available only
# upon request" from Enamine — but that applies to re-PREPROCESSING. The preprocessed artifacts
# (fpindex.pkl, matrix.pkl) and the trained encoder-decoder checkpoint are published on HuggingFace,
# and inference reads only those. Confirmed in the source: the sampler worker loads
# `config.chem.{fpindex,rxn_matrix}` pickles and never touches the raw SDF.
#
# ============================ WHERE THE 6.8 GB LIVES, AND WHY ============================
# fpindex.pkl is 4.0 GB and sf_ed_default.ckpt is 2.8 GB. `/home/markymoo` is a 110 GB quota with
# ~16 GB free, so they go to $SCRATCH. That is not simply a matter of pointing a config at them: the
# sampler reads its data paths out of the CHECKPOINT's own saved hyper_parameters
# (`OmegaConf.create(ckpt["hyper_parameters"]["config"])`), where they are stored RELATIVE
# ("data/processed/comp_2048/fpindex.pkl") and resolved against the process's cwd. So the tree has to
# exist at that relative path inside the clone, and we make it a SYMLINK to $SCRATCH.
#
# Symlinks under external/ were once genuinely dangerous here — `git add -A` committed a set of them
# and a later merge deleted the real clones, because the ignore rule was `external/*/`, which matches
# directories only. That rule is now `external/*` (+ `!external/setup_*.sh`), which ignores symlinks
# too; verified behaviourally with `git check-ignore` before writing this. Still: never `git add -A`
# in this repo.
# =========================================================================================
#
# Run from the repo root, ON A LOGIN NODE (downloads ~6.8 GB; compute nodes have no internet):
#   bash external/setup_synformer.sh
# Idempotent: re-running skips the clone / env / download if already present.

set -euo pipefail

# CAP BLAS/OMP THREADS FOR THE WHOLE SCRIPT, not just the verification at the end. The login node's
# RLIMIT_NPROC is 1024 and OpenBLAS spawns one thread per core on first use, which segfaults the
# interpreter. This bites during INSTALLATION, not only at run time: synformer's pyproject declares
# `version = {attr = "synformer.__version__"}`, so `pip install -e .` imports the package — and thus
# numpy/scipy — to read its version.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ENV_PREFIX="${SYNFORMER_ENV_PREFIX:-/scratch/markymoo/conda_envs/synformer}"
PYVER=3.10

SYNFORMER_ORG=wenhao-gao
SYNFORMER_REPO=synformer
SYNFORMER_COMMIT="${SYNFORMER_COMMIT:-bef02bc4481adf3ed792090fc6169331d4d0553a}"
CLONE_DIR="external/synformer"

# The big artifacts, on $SCRATCH. `comp_2048` is the directory name the checkpoint's embedded config
# expects — do not rename it.
DATA_ROOT="${SYNFORMER_DATA_ROOT:-/scratch/markymoo/synformer_data}"
PROCESSED_DIR="${DATA_ROOT}/processed/comp_2048"
WEIGHTS_DIR="${DATA_ROOT}/trained_weights"
HF_REPO=whgao/synformer

# torch 2.1.0+cu118: the same line the saturn env runs, chosen for the same reason — it is the one
# build with prebuilt pyg extension wheels (`bengio2021flow` imports torch_sparse at module level).
# SynFormer itself only asks for torch>=2.0, so this is inside its own requirement.
TORCH_VER=2.1.0
TORCH_CUDA=cu118
PYG_FIND_LINKS="https://data.pyg.org/whl/torch-${TORCH_VER}+${TORCH_CUDA}.html"

SEH_CACHE_SRC="${SEH_CACHE_SRC:-$HOME/miniconda3/envs/s3gfn/lib/python3.10/site-packages/gflownet/models/cache/bengio2021flow_proxy.pkl.gz}"

echo "[setup_synformer] env=${ENV_PREFIX} python=${PYVER} torch=${TORCH_VER}+${TORCH_CUDA}"
echo "[setup_synformer] data=${DATA_ROOT} (6.8 GB, on \$SCRATCH by necessity — see header)"

# --- 1. Clone at the pinned commit. ------------------------------------------------------------
if [ ! -d "${CLONE_DIR}" ]; then
    echo "[setup_synformer] cloning ${SYNFORMER_ORG}/${SYNFORMER_REPO}"
    git -C external clone "https://github.com/${SYNFORMER_ORG}/${SYNFORMER_REPO}" synformer
fi
# A --depth 1 clone cannot check out an arbitrary SHA, so deepen if needed rather than failing.
git -C "${CLONE_DIR}" cat-file -e "${SYNFORMER_COMMIT}^{commit}" 2>/dev/null || \
    git -C "${CLONE_DIR}" fetch --unshallow origin 2>/dev/null || true
git -C "${CLONE_DIR}" checkout -q "${SYNFORMER_COMMIT}"
git -C "${CLONE_DIR}" rev-parse HEAD > "${CLONE_DIR}/COMMIT_PINNED.txt"
echo "[setup_synformer] pinned SHA $(cat "${CLONE_DIR}/COMMIT_PINNED.txt")"

# --- 2. Conda env. -------------------------------------------------------------------------------
if [ ! -d "${ENV_PREFIX}" ]; then
    echo "[setup_synformer] creating conda env at ${ENV_PREFIX}"
    mkdir -p "$(dirname "${ENV_PREFIX}")"
    conda create -y -p "${ENV_PREFIX}" "python=${PYVER}"
fi
RUN="conda run --no-capture-output -p ${ENV_PREFIX}"

# --- 3. SynFormer's dependencies. ----------------------------------------------------------------
# From env.yml / requirements.txt, MINUS pytdc. Their GA driver imports tdc for an SA scorer and a
# diversity metric we do not use, and TDC self-downloads into ./oracle on first use, which fails on a
# compute node ($HOME read-only, no internet). Our driver reimplements their GA loop against our own
# frozen reward instead, so tdc never enters the picture. Also minus jupyter/lint/docs extras.
echo "[setup_synformer] installing torch ${TORCH_VER}+${TORCH_CUDA}"
${RUN} pip install "torch==${TORCH_VER}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "[setup_synformer] installing SynFormer's python deps"
# numpy is PINNED <2 and it is not optional. torch 2.1.0 is built against the numpy 1.x C API, so
# numpy 2 makes `import torch` emit "Failed to initialize NumPy: _ARRAY_API not found" and then fail
# at the first `torch.tensor(<np array>)` with "Could not infer dtype of numpy.float32" — which
# surfaces while loading the sEH weights, far from the cause. It is also the same bound the DRD2
# sklearn pickle needs, and the version the reinvent4 and saturn envs run.
# scipy is pinned in the SAME command, not after: pip resolves the set together, so installing
# numpy alone first and letting a later package pull scipy gives you a scipy wheel built against
# numpy 2, which then fails at import with "numpy._core.multiarray failed to import" (numpy._core is
# the numpy-2 module path). Recovering from that needs a clean uninstall, so avoid creating it.
${RUN} pip install "numpy==1.26.4" "scipy==1.13.1"
${RUN} pip install "rdkit>=2023.09" "pytorch-lightning>=2.0" einops scikit-learn pandas \
    pyyaml joblib omegaconf gitpython cryptography rich tqdm click tensorboard

echo "[setup_synformer] installing the synformer package (--no-deps, editable)"
( cd "${CLONE_DIR}" && ${RUN} pip install --no-deps -e . )

# --- 4. The shared frozen sEH reward stack (identical to the reinvent/saturn envs). --------------
echo "[setup_synformer] installing torch-geometric + pyg extensions + recursion gflownet"
${RUN} pip install "torch-geometric==2.6.1"
${RUN} pip install --no-index -f "${PYG_FIND_LINKS}" torch-scatter torch-sparse torch-cluster
${RUN} pip install --no-deps "gflownet @ git+https://github.com/recursionpharma/gflownet.git"
${RUN} pip install omegaconf   # --no-deps leaves it out and gflownet/__init__.py imports it

# --- 5. Fetch the 6.8 GB of model data to $SCRATCH, then symlink it into the clone. --------------
mkdir -p "${PROCESSED_DIR}" "${WEIGHTS_DIR}"
${RUN} pip install -q "huggingface_hub"
fetch() {  # $1 = filename, $2 = destination dir
    if [ -s "$2/$1" ]; then echo "[setup_synformer] $1 already present"; return; fi
    echo "[setup_synformer] downloading $1 (this is the slow part)"
    ${RUN} python - "$1" "$2" <<'PY'
import sys
from huggingface_hub import hf_hub_download
name, dest = sys.argv[1], sys.argv[2]
p = hf_hub_download(repo_id="whgao/synformer", filename=name, local_dir=dest)
print(f"  -> {p}")
PY
}
fetch fpindex.pkl "${PROCESSED_DIR}"
fetch matrix.pkl  "${PROCESSED_DIR}"
fetch sf_ed_default.ckpt "${WEIGHTS_DIR}"

# The checkpoint's embedded config uses these exact RELATIVE paths, resolved against cwd, so the
# clone needs the tree to exist. Symlink rather than copy: 6.8 GB does not fit in $HOME's quota.
mkdir -p "${CLONE_DIR}/data"
# `ln -sfn TARGET LINK` does NOT replace LINK when LINK is an existing real directory — it creates
# the link INSIDE it (data/trained_weights/trained_weights). The clone ships a tracked, .gitignore-only
# `data/trained_weights/`, so this bites exactly once and then the checkpoint "is missing" while
# sitting one level down. Remove a real directory first; leave an existing symlink to be replaced.
link_scratch() {  # $1 = target on $SCRATCH, $2 = link path in the clone
    if [ -d "$2" ] && [ ! -L "$2" ]; then
        find "$2" -mindepth 1 -maxdepth 1 ! -name '.gitignore' -print -quit | grep -q . && {
            echo "[setup_synformer] FATAL: $2 is a non-empty real directory; refusing to replace it" >&2
            exit 1; }
        rm -rf "$2"
    fi
    ln -sfn "$1" "$2"
}
link_scratch "${DATA_ROOT}/processed" "${CLONE_DIR}/data/processed"
link_scratch "${WEIGHTS_DIR}"         "${CLONE_DIR}/data/trained_weights"
echo "[setup_synformer] linked ${CLONE_DIR}/data/{processed,trained_weights} -> ${DATA_ROOT}"

# --- 6. Pre-place the sEH weights so no compute-node job needs the network. ----------------------
GF_MODELS=$(${RUN} python -c "import gflownet.models as m, os; print(os.path.dirname(m.__file__))" | tr -d '\r')
mkdir -p "${GF_MODELS}/cache"
if [ -s "${GF_MODELS}/cache/bengio2021flow_proxy.pkl.gz" ]; then
    echo "[setup_synformer] sEH weights already cached"
elif [ -s "${SEH_CACHE_SRC}" ]; then
    cp "${SEH_CACHE_SRC}" "${GF_MODELS}/cache/"
    echo "[setup_synformer] copied sEH weights from the s3gfn env (no network)"
else
    ${RUN} python -c "from gflownet.models import bengio2021flow; bengio2021flow.load_original_model()"
fi

# --- 7. Verify. --------------------------------------------------------------------------------
echo "[setup_synformer] verifying"
ACTUAL_NP=$(${RUN} python -c "import numpy; print(numpy.__version__)" | tr -d '\r')
${RUN} python -c "import scipy.linalg, sklearn.svm" || {
    echo "[setup_synformer] FATAL: scipy/sklearn cannot import — almost certainly built against a" >&2
    echo "  different numpy than the one installed. Fix: pip uninstall -y scipy && reinstall pinned." >&2
    exit 1; }
case "${ACTUAL_NP}" in 1.*) echo "[setup_synformer] numpy ${ACTUAL_NP} + scipy ABI ok" ;;
  *) echo "[setup_synformer] FATAL: numpy ${ACTUAL_NP} — torch 2.1.0 needs <2; something upgraded it." >&2; exit 1 ;;
esac
# (thread caps are exported at the top of this script — they cover the installs too)
REPO_ROOT="$(pwd)"
( cd "${CLONE_DIR}" && PYTHONPATH="$(pwd):$(pwd)/experiments:${REPO_ROOT}" ${RUN} python - <<'PY'
import os, numpy as np, torch
torch.set_num_threads(1)
from rdkit import Chem

# (a) the frozen sEH reward agrees with every other env
from gflownet.models import bengio2021flow
m = bengio2021flow.load_original_model()
g = bengio2021flow.mol2graph(Chem.MolFromSmiles("CCO"))
v = float(m(bengio2021flow.mols2batch([g])).view(-1)[0])
assert abs(v - 0.0273) < 1e-3, f"sEH disagrees with the reinvent4/saturn envs ({v:.4f} vs 0.0273)"
print(f"  sEH MPNN ok (ethanol -> {v:.4f}) — matches the other envs")

# (b) synformer imports, including the pieces the adapter depends on
from synformer.chem.mol import Molecule
from synformer.chem.stack import Stack
from synformer.sampler.analog.parallel import run_parallel_sampling
assert hasattr(Stack, "get_action_string") and hasattr(Stack, "get_tree"), dir(Stack)
print("  synformer imports (Molecule, Stack, run_parallel_sampling)")

# (c) the GA helpers we reuse are importable WITHOUT pytdc (their driver script imports tdc at
#     module level; crossover/mutate do not, which is why we import those two directly).
import crossover, mutate  # noqa: F401
print("  GA helpers import (crossover, mutate) with no pytdc")

# (d) the big artifacts resolve at the RELATIVE paths the checkpoint expects, through the symlinks.
ckpt_path = "data/trained_weights/sf_ed_default.ckpt"
assert os.path.exists(ckpt_path), f"missing {ckpt_path}"
from omegaconf import OmegaConf
ck = torch.load(ckpt_path, map_location="cpu")
cfg = OmegaConf.create(ck["hyper_parameters"]["config"])
for key in ("fpindex", "rxn_matrix"):
    p = cfg.chem[key]
    assert os.path.exists(p), (
        f"checkpoint expects {key} at relative path {p!r}, which does not resolve from "
        f"{os.getcwd()}. The symlink into $SCRATCH is missing or wrong.")
    print(f"  {key} resolves: {p} ({os.path.getsize(os.path.realpath(p)) / 1e9:.1f} GB)")
PY
)

cat <<EOF

[setup_synformer] DONE.
  env    : ${ENV_PREFIX}   (a PREFIX env — address it with 'conda run -p', never '-n')
  clone  : ${CLONE_DIR} @ $(cat "${CLONE_DIR}/COMMIT_PINNED.txt")
  data   : ${DATA_ROOT} (symlinked into the clone; the checkpoint resolves it relative to cwd)

Reaction-aware entrant: emits has_route=1, so it needs NO MultiAiZ — it prices through
sparrow_select_frontier.py --route-source external.
EOF
