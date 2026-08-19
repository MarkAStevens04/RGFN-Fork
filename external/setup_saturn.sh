#!/bin/bash
# setup_saturn.sh — OURS. Install Saturn, a ROUTE-LESS SMILES baseline for the LSD-Flow
# library-efficiency benchmark, and the host for the TANGO arms (Phase 3).
#
# Saturn (`[guo2026saturn]`, github.com/schwallergroup/saturn) is a Mamba-based SMILES language model
# trained with the Augmented Memory algorithm, built explicitly for SAMPLE EFFICIENCY — few oracle
# calls per unit of reward. Like REINVENT and S3-GFN it has no reaction model, so its molecules carry
# no route (`has_route=0`) and a library built from them needs its shared intermediates recovered
# post-hoc (MultiAiZ -> SPARROW).
#
# Its sample efficiency is the interesting axis: it is the strongest case for "you do not need many
# oracle calls", which is precisely the axis this benchmark LOSES on and reports (Logs/056's
# candidates-you-must-score trade). A strong showing by Saturn there is a result, not a problem.
#
# ONE CLONE, TWO PHASES. `[guo2026tango]` is not a separate model — TANGO is a reward function inside
# this same repo (`oracles/similarity/tango.py`, plus the syntheseus synthesizability oracle). The
# README pins a git hash per paper: Saturn `fee0179`, TANGO `de5cd7f`. We check out Saturn's hash
# here and Phase 3 adds TANGO's as a `git worktree`, which shares this clone's object store instead
# of costing another 162 MB.
#
# ======================= THE ONE REAL DEVIATION FROM UPSTREAM setup.sh =======================
# Upstream installs `torch==1.12.1+cu113`. WE INSTALL `torch==2.1.0+cu118`. Two independent hard
# constraints make cu113 impossible, and neither is about Saturn's science:
#
#   1. `bengio2021flow` — the frozen sEH MPNN every entrant in this benchmark is scored by — does
#      `from torch_sparse import coalesce` at MODULE level. Prebuilt torch-sparse wheels for cp310
#      exist at data.pyg.org for pt21cu118; building one from source is a long CUDA compile.
#   2. `mamba-ssm==1.2.0.post1` and `causal-conv1d==1.2.0.post2` publish prebuilt wheels for
#      cu118/cu122 across torch 1.12-2.3 — but **not for cu113, at any torch version**. Under cu113
#      both would have to compile against nvcc 11.3, which this cluster does not provide (the only
#      CUDA module is 11.8). This is the documented Saturn install failure (their Issue #1).
#
# torch 2.1.0+cu118 is the combination that satisfies BOTH: mamba-ssm ships
# `+cu118torch2.1cxx11abiFALSE-cp310` and pyg ships `torch_sparse-0.6.18+pt21cu118-cp310`. If Saturn
# turns out to use a torch API removed in 2.x, the fallback that also satisfies both is
# torch 2.0.0+cu118 (`+cu118torch2.0` / `pt20cu118`). Verify Saturn imports and trains before
# trusting this; the smoke in the driver is what checks it.
# =============================================================================================
#
# WHAT WE DELIBERATELY DO NOT INSTALL. Upstream's setup.sh adds openbabel, xtb-python and a
# blas/mkl pin for its xTB electronic-property oracles. `oracles/utils.py` imports every oracle
# eagerly, so the import chain must resolve — but checked file by file, the only hard Python import
# in that chain is `morfeus` (`oracles/xtb/homo.py`: `from morfeus import read_xyz, XTB`);
# `geometry_optimizer.py` invokes xtb as a SUBPROCESS, not an import. So `morfeus-ml` alone is
# enough, and we skip openbabel — which is heavy and is the subject of the GLIBCXX_3.4.29 failure
# their README documents. We never call an xTB oracle.
#
# Run from the repo root, ON A LOGIN NODE (this downloads; compute nodes have no internet):
#   bash external/setup_saturn.sh
# Idempotent: re-running skips the clone / env create / weight copy if already present.

set -euo pipefail

# Prefix env on $SCRATCH, not miniconda3/envs: /home/markymoo is at ~95G of its 110G quota.
# Address it with `conda run -p`; `-n saturn` will not find it.
ENV_PREFIX="${SATURN_ENV_PREFIX:-/scratch/markymoo/conda_envs/saturn}"
PYVER=3.10

SATURN_ORG=schwallergroup
SATURN_REPO=saturn
# NOT fee0179, the hash the upstream README gives for the Saturn pre-print — THAT COMMIT IS BROKEN.
# At fee0179 `goal_directed_generation/reinforcement_learning.py` reads
# `configuration.reinforcement_learning.margin_threshold`, but `ReinforcementLearningParameters` at
# the same commit has no such field, so constructing a ReinforcementLearningAgent raises
# AttributeError and goal-directed generation cannot run at all. Checked across refs: fee0179 is the
# only one whose RL module references `margin_threshold` (de5cd7f, 468b1f4 and master do not), and
# no ref defines it in the dataclass — i.e. the line was removed shortly after, and fee0179 caught
# the repo mid-edit.
#
# de5cd7f is the TANGO pre-print hash, the next published-paper pin from the same authors, and it
# runs. It is also what Phase 3 needs, so one clone and one pin now serve both the Saturn and TANGO
# arms. State in the write-up that Saturn was run at the TANGO hash and why.
SATURN_COMMIT="${SATURN_COMMIT:-de5cd7f}"
CLONE_DIR="external/saturn"

TORCH_VER=2.1.0
TORCH_CUDA=cu118
PYG_FIND_LINKS="https://data.pyg.org/whl/torch-${TORCH_VER}+${TORCH_CUDA}.html"
# cxx11abiFALSE matches PyPI torch wheels, which are built with _GLIBCXX_USE_CXX11_ABI=0.
CCONV_WHL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
MAMBA_WHL="https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

SEH_CACHE_SRC="${SEH_CACHE_SRC:-$HOME/miniconda3/envs/s3gfn/lib/python3.10/site-packages/gflownet/models/cache/bengio2021flow_proxy.pkl.gz}"

echo "[setup_saturn] env=${ENV_PREFIX} python=${PYVER} commit=${SATURN_COMMIT} torch=${TORCH_VER}+${TORCH_CUDA}"

# --- 1. Clone Saturn at the pinned commit. -----------------------------------------------------
# A full clone (162 MB): unlike REINVENT this repo is small, it carries the pretrained Mamba priors
# we need in-tree, and keeping the history lets Phase 3 add TANGO's hash as a worktree for free.
if [ ! -d "${CLONE_DIR}" ]; then
    echo "[setup_saturn] cloning ${SATURN_ORG}/${SATURN_REPO}"
    git -C external clone "https://github.com/${SATURN_ORG}/${SATURN_REPO}" saturn
else
    echo "[setup_saturn] ${CLONE_DIR} already present — skipping clone"
fi
git -C "${CLONE_DIR}" checkout "${SATURN_COMMIT}"
git -C "${CLONE_DIR}" rev-parse HEAD > "${CLONE_DIR}/COMMIT_PINNED.txt"
echo "[setup_saturn] pinned SHA $(cat "${CLONE_DIR}/COMMIT_PINNED.txt")"

# --- 2. Create the dedicated conda env. --------------------------------------------------------
if [ ! -d "${ENV_PREFIX}" ]; then
    echo "[setup_saturn] creating conda env at ${ENV_PREFIX}"
    mkdir -p "$(dirname "${ENV_PREFIX}")"
    conda create -y -p "${ENV_PREFIX}" "python=${PYVER}"
else
    echo "[setup_saturn] conda env ${ENV_PREFIX} already exists — skipping create"
fi

RUN="conda run --no-capture-output -p ${ENV_PREFIX}"

# --- 3. Saturn's own dependencies (versions from upstream setup.sh except torch — see the header).
echo "[setup_saturn] installing torch ${TORCH_VER}+${TORCH_CUDA}"
${RUN} pip install "torch==${TORCH_VER}" "torchvision==0.16.0" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# A C COMPILER IS A RUNTIME DEPENDENCY HERE, not a build-time one. Mamba's layer-norm goes through
# Triton, which JIT-compiles a launcher stub on first use and dies with "Failed to find C compiler.
# Please specify via CC environment variable." if there is none. This cluster has no /usr/bin/gcc and
# its `module load gcc` does not populate PATH for a non-interactive shell, so the compiler goes in
# the env — which is also what Saturn's own README recommends. `conda run -p` runs the activation
# hooks, so CC is exported automatically for every caller.
echo "[setup_saturn] installing a C compiler into the env (Triton JIT needs one at RUNTIME)"
conda install -y -p "${ENV_PREFIX}" -c conda-forge gcc_linux-64 gxx_linux-64

echo "[setup_saturn] installing Saturn's pinned python deps"
${RUN} pip install "rdkit==2023.9.5" "pandas==2.2.1" "numpy==1.26.4" "einops==0.7.0" \
    "scipy==1.10.0" pathos tqdm "morfeus-ml"
# scikit-learn is OURS, not Saturn's: the DRD2 oracle is a pickled sklearn SVC. Installed after the
# pins above and verified not to move them (numpy 1.26.4 / scipy 1.10.0 / pandas 2.2.1 survive).
# Version is unconstrained on purpose — the pickle scores identically across sklearn 1.2.2 / 1.7.2 /
# 1.8.0, which is checked in step 6 rather than assumed.
${RUN} pip install scikit-learn

echo "[setup_saturn] installing causal-conv1d + mamba-ssm (PREBUILT wheels — no CUDA compile)"
${RUN} pip install "${CCONV_WHL}"
${RUN} pip install --no-deps "${MAMBA_WHL}"
# mamba-ssm's generation utils do
#   from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, ...
# at import time, so `import mamba_ssm` fails outright without transformers. Upstream's setup.sh
# omits it (their torch-1.12 pin pulled an older mamba-ssm path). PINNED <4.42: those two symbols
# were removed in transformers 4.42 in favour of GenerateDecoderOnlyOutput, so a current
# transformers breaks mamba-ssm 1.2.0 just as surely as no transformers at all.
${RUN} pip install "transformers<4.42"

# --- 4. The shared frozen sEH reward stack (identical to setup_reinvent.sh). --------------------
echo "[setup_saturn] installing torch-geometric + pyg extensions + recursion gflownet"
${RUN} pip install "torch-geometric==2.6.1"
${RUN} pip install --no-index -f "${PYG_FIND_LINKS}" torch-scatter torch-sparse torch-cluster
${RUN} pip install --no-deps "gflownet @ git+https://github.com/recursionpharma/gflownet.git"
# --no-deps leaves omegaconf out, and `gflownet/__init__.py` imports it, so ANY
# `from gflownet.models import bengio2021flow` fails without this. Not optional.
${RUN} pip install omegaconf

# --- 5. Pre-place the sEH weights so no compute-node job ever needs the network. ----------------
GF_MODELS=$(${RUN} python -c "import gflownet.models as m, os; print(os.path.dirname(m.__file__))" | tr -d '\r')
mkdir -p "${GF_MODELS}/cache"
if [ -s "${GF_MODELS}/cache/bengio2021flow_proxy.pkl.gz" ]; then
    echo "[setup_saturn] sEH weights already cached"
elif [ -s "${SEH_CACHE_SRC}" ]; then
    echo "[setup_saturn] copying sEH weights from the s3gfn env (no network)"
    cp "${SEH_CACHE_SRC}" "${GF_MODELS}/cache/"
else
    echo "[setup_saturn] downloading sEH weights (login node only)"
    ${RUN} python -c "from gflownet.models import bengio2021flow; bengio2021flow.load_original_model()"
fi

# --- 6. Verify. ---------------------------------------------------------------------------------
echo "[setup_saturn] verifying"
# Login nodes cap threads per user; torch's OpenMP otherwise dies with
# "libgomp: Thread creation failed". Only affects this check.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PRIORS=$(ls "${CLONE_DIR}"/experimental_reproduction/checkpoint_models/*.prior 2>/dev/null | head -3)
[ -n "${PRIORS}" ] || {
    echo "[setup_saturn] FATAL: no pretrained priors under ${CLONE_DIR}/experimental_reproduction/checkpoint_models/" >&2
    exit 1; }
echo "[setup_saturn] priors present:"; echo "${PRIORS}" | sed 's/^/    /'

REPO_ROOT="$(pwd)"
( cd "${CLONE_DIR}" && PYTHONPATH="$(pwd):${REPO_ROOT}" ${RUN} python - <<'PY'
import numpy as np, torch
torch.set_num_threads(1)

# (a) the frozen sEH reward stack works in THIS env
from gflownet.models import bengio2021flow
from rdkit import Chem
m = bengio2021flow.load_original_model()
g = bengio2021flow.mol2graph(Chem.MolFromSmiles("CCO"))
v = float(m(bengio2021flow.mols2batch([g])).view(-1)[0])
assert np.isfinite(v), "sEH MPNN returned a non-finite score"
assert abs(v - 0.0273) < 1e-3, f"sEH disagrees with the reinvent4 env ({v:.4f} vs 0.0273)"
print(f"  sEH MPNN ok (ethanol -> {v:.4f}) — matches the reinvent4 env, so reward parity holds")

# (b) Mamba actually imports and runs — the whole reason for the torch deviation. A wheel that
#     installs but cannot load its CUDA kernels fails here rather than mid-training.
from mamba_ssm import Mamba
print(f"  mamba-ssm imports (torch {torch.__version__}, cuda avail={torch.cuda.is_available()})")

# (c) Saturn's oracle registry imports. `oracles/utils.py` pulls in EVERY oracle module eagerly, so
#     this is the check that we got away with skipping openbabel and xtb-python. `morfeus` is a real
#     install (oracles/xtb/homo.py imports it); `openbabel` is stubbed, because the only importer is
#     GEAM's docking oracle, which this adapter never constructs.
from validation.generators.saturn._stubs import stub_unused_oracle_deps
stubbed = stub_unused_oracle_deps()
from oracles.utils import construct_oracle_component
from oracles.oracle import Oracle
from oracles.oracle_component import OracleComponent
print(f"  Saturn's oracle registry imports (morfeus real; stubbed {stubbed})")

# (d) the generator API this adapter depends on. Sampling the candidate pool from the TRAINED agent
#     — matching the S3-GFN/REINVENT protocol — needs exactly these two.
from models.mamba import MambaLMHead
from models.generator import Generator
assert hasattr(Generator, "load_from_file") and hasattr(Generator, "sample_smiles"), dir(Generator)
print("  Saturn generator API present (Generator.load_from_file / .sample_smiles)")

# (e) THE COMMIT IS SELF-CONSISTENT. Every attribute the RL module reads off
#     ReinforcementLearningParameters must actually exist on it. This is not paranoia: the hash the
#     upstream README pins for the Saturn paper (fee0179) fails exactly here — its RL module reads
#     `margin_threshold`, which the dataclass does not define, so the agent cannot be constructed.
#     An import-only check would have sailed past that and failed hours later inside a job.
import dataclasses, re
from goal_directed_generation.dataclass import ReinforcementLearningParameters
src = open("goal_directed_generation/reinforcement_learning.py").read()
used = set(re.findall(r"configuration\.reinforcement_learning\.(\w+)", src))
have = {f.name for f in dataclasses.fields(ReinforcementLearningParameters)}
missing = used - have
assert not missing, (
    f"BROKEN COMMIT: reinforcement_learning.py reads {sorted(missing)} off "
    f"ReinforcementLearningParameters, which defines {sorted(have)}. Goal-directed generation "
    "cannot run at this hash — see the SATURN_COMMIT note at the top of this script.")
print(f"  RL config is self-consistent ({len(used)} attributes read, all defined)")

# (f) the DRD2 oracle scores IDENTICALLY here to every other benchmark env. Each generator runs in
#     its own env with its own sklearn, and the pickle was written with sklearn 0.23 (so every
#     modern env warns that results "might be invalid"). A version-dependent unpickle would mean
#     each entrant silently optimizing a slightly different DRD2. Reference values measured across
#     the reinvent4 / rgfn / fraggfn / scent envs.
import pickle, warnings
warnings.filterwarnings("ignore")
from rdkit.Chem import AllChem
with open("../../oracle/drd2_current.pkl", "rb") as fh:
    drd2 = pickle.load(fh)
def _fp(smi):
    f = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 3, useCounts=True, useFeatures=True)
    n = np.zeros((1, 2048), np.int32)
    for i, c in f.GetNonzeroElements().items():
        n[0, i % 2048] += int(c)
    return n
probe = ["CCO", "c1ccccc1C(=O)NC2CCN(CC2)Cc3ccccc3",
         "CN1CCN(CCCN2c3ccccc3Sc3ccc(Cl)cc32)CC1", "O=C(NCc1ccccc1)c1ccc(N2CCN(C)CC2)cc1"]
got = drd2.predict_proba(np.concatenate([_fp(s) for s in probe]))[:, 1]
ref = [0.005215657677, 0.123834057061, 0.927636834696, 0.049222148471]
assert np.allclose(got, ref, atol=1e-9), f"DRD2 disagrees with the other envs: {list(got)} vs {ref}"
print("  DRD2 oracle matches the other envs to 1e-9 — reward parity holds on both targets")
PY
)

cat <<EOF

[setup_saturn] DONE.
  env    : ${ENV_PREFIX}   (a PREFIX env — always address it with 'conda run -p', never '-n')
  clone  : ${CLONE_DIR} @ $(cat "${CLONE_DIR}/COMMIT_PINNED.txt")
  priors : ${CLONE_DIR}/experimental_reproduction/checkpoint_models/

Note the deviation from upstream setup.sh: torch ${TORCH_VER}+${TORCH_CUDA}, not 1.12.1+cu113.
Read the header before "fixing" it — cu113 has no mamba-ssm and no torch-sparse wheel.
EOF
