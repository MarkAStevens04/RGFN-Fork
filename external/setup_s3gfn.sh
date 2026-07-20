#!/bin/bash
# setup_s3gfn.sh — OURS. Install S3-GFN, the MARQUEE non-reaction baseline for the LSD-Flow
# library-efficiency benchmark (docs/LSD_FLOW_BENCHMARK_PLAN.md T3.1).
#
# S3-GFN (`[kim2026s3gfn]`, github.com/hyeonahkimm/s3gfn) is a SEQUENCE (SMILES) GFlowNet that
# post-trains a pretrained chemical language model (GP-MolFormer) and induces synthesizability by
# SOFT regularization (contrastive replay buffers of synthesizable vs unsynthesizable samples) —
# NOT reaction templates. Its molecules are per-molecule synthesizable but carry NO shared-route
# structure, so a *library* must recover shared intermediates post-hoc (AiZynth→SPARROW). It is the
# foil for the "MDP-necessary-for-library-economics" headline.
#
# FAIRNESS (verified from src/s3gfn/scoring_function.py): the sEH reward is
# `bengio2021flow.load_original_model()` — the SAME Bengio-2021 sEH MPNN our benchmark uses
# (rgfn/gfns/reaction_gfn/proxies/seh_proxy.py: bengio2021flow_proxy.pkl.gz). S3-GFN divides the raw
# proxy value by 8 for its TRAINING reward; our mode gate (>7) is on the RAW value. So we let S3-GFN
# train with its native reward (monotonic in the raw value → same target) and RE-SCORE the output
# pool on our raw proxy for the mode gate — no need to wire a custom reward into its training loop.
#
# Why a DEDICATED conda env (`s3gfn`): it pins python 3.10 + torch 2.5.1+cu121 + transformers 4.32 +
# GP-MolFormer. Same "one env per benchmarked generator, one shared on-disk standard" model as
# rxnflow/fraggfn; the pool is emitted in the standard candidate-dataset format and re-scored across
# the env boundary via scripts/ingest_candidates.py / score_batch.py (run under rgfn). torch
# 2.5.1+cu121 == the rxnflow env's line, so the pyg find-links wheels are reused.
#
# Run from the repo root (login or a CUDA-12-capable node with conda):
#   bash external/setup_s3gfn.sh
# Idempotent: re-running skips the clone / env create if already present.

set -euo pipefail

ENV_NAME="${S3GFN_ENV:-s3gfn}"
PYVER=3.10

S3GFN_ORG=hyeonahkimm
S3GFN_REPO=s3gfn
S3GFN_COMMIT="${S3GFN_COMMIT:-main}"   # TODO: pin to a commit SHA once validated on the cluster
CLONE_DIR="external/${S3GFN_REPO}"

# torch 2.5.1+cu121 (matches requirements.txt); pyg extension wheels via find-links (no nvcc build).
TORCH_VER=2.5.1
PYG_FIND_LINKS="https://data.pyg.org/whl/torch-${TORCH_VER}+cu121.html"

echo "[setup_s3gfn] env=${ENV_NAME} python=${PYVER} torch=${TORCH_VER}+cu121"

# --- 1. Clone S3-GFN at the pinned commit (under external/, git-ignored). -------
if [ ! -d "${CLONE_DIR}" ]; then
    echo "[setup_s3gfn] cloning ${S3GFN_ORG}/${S3GFN_REPO}@${S3GFN_COMMIT}"
    git -C external clone "https://github.com/${S3GFN_ORG}/${S3GFN_REPO}" "${S3GFN_REPO}"
    git -C "${CLONE_DIR}" checkout "${S3GFN_COMMIT}"
else
    echo "[setup_s3gfn] ${CLONE_DIR} already present — skipping clone"
fi

# --- 2. Create the dedicated conda env (python 3.10). --------------------------
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup_s3gfn] creating conda env ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" "python=${PYVER}"
else
    echo "[setup_s3gfn] conda env ${ENV_NAME} already exists — skipping create"
fi

# --- 3. Install S3-GFN requirements + the package (SPLIT install). --------------
# requirements.txt pins `--index-url https://download.pytorch.org/whl/cu121` (EXCLUSIVE); that index
# has torch but NOT pure-PyPI packages (torch-geometric==2.6.1 etc.), and a CLI --extra-index-url does
# not reliably override a requirements-file --index-url. So install in three passes, each from the
# index that actually has the wheels:
#   A) torch stack  -> cu121 index
#   B) pyg C++ extensions (scatter/sparse/cluster, +pt25cu121) -> pyg find-links (torch must exist first)
#   C) everything else (torch-geometric + the rest) -> plain PyPI, from a filtered requirements
#      (the index/find-links directives + the A/B packages stripped out).
TORCH_PKGS="torch==2.5.1+cu121 torchvision torchaudio"
PYG_EXT="torch-scatter==2.1.2+pt25cu121 torch-sparse==0.6.18+pt25cu121 torch-cluster==1.6.3+pt25cu121"
echo "[setup_s3gfn] (A) torch stack from cu121 index"
conda run -n "${ENV_NAME}" pip install ${TORCH_PKGS} --index-url "https://download.pytorch.org/whl/cu121"
echo "[setup_s3gfn] (B) pyg C++ extensions from find-links"
conda run -n "${ENV_NAME}" pip install ${PYG_EXT} --find-links "${PYG_FIND_LINKS}"
echo "[setup_s3gfn] (C) remaining requirements from PyPI (filtered)"
REQ_TMP="$(mktemp)"
grep -viE '^\s*(--index-url|--extra-index-url|-f |torch==|torchvision|torchaudio|torch-scatter|torch-sparse|torch-cluster)' \
    "${CLONE_DIR}/requirements.txt" > "${REQ_TMP}"
conda run -n "${ENV_NAME}" pip install -r "${REQ_TMP}" --find-links "${PYG_FIND_LINKS}"
rm -f "${REQ_TMP}"
echo "[setup_s3gfn] installing S3-GFN (editable)"
conda run -n "${ENV_NAME}" pip install -e "${CLONE_DIR}"

# --- 4. Recursion `gflownet` — provides bengio2021flow (the sEH proxy S3-GFN imports). ----------
# scoring_function.py does `from gflownet.models import bengio2021flow`; the sEH model is the SAME
# one our benchmark uses. requirements.txt does not list gflownet, so install it explicitly (the
# same package the rxnflow/fraggfn envs carry). bengio2021flow.load_original_model() then downloads
# the sEH MPNN weights on first use.
echo "[setup_s3gfn] installing recursion gflownet (for bengio2021flow sEH proxy)"
conda run -n "${ENV_NAME}" pip install "gflownet @ git+https://github.com/recursionpharma/gflownet.git" \
    --find-links "${PYG_FIND_LINKS}" \
    || echo "[setup_s3gfn] WARNING gflownet install failed — pin a version if bengio2021flow import fails (TODO)"

# --- 5. GP-MolFormer prior weights (HuggingFace ibm-research/GP-MoLFormer-Uniq). ----------------
# S3-GFN post-trains this SMILES language model. Pre-fetch so the first training run doesn't block
# on the network (huggingface_hub is already a dep). Cached under $HF_HOME (guard $SCRATCH under
# set -u; fall back to $HOME cache if unset). Run from a TEMP FILE, not `conda run python - <<HEREDOC`
# — conda run does not forward stdin, so a heredoc silently no-ops (verified with the sparrow smoke).
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/.cache/huggingface}"
mkdir -p "$HF_HOME"
echo "[setup_s3gfn] pre-fetching GP-MolFormer weights -> $HF_HOME"
PREFETCH_PY="$(mktemp --suffix=.py)"
cat > "${PREFETCH_PY}" <<'PY'
try:
    from huggingface_hub import snapshot_download
    p = snapshot_download("ibm-research/GP-MoLFormer-Uniq")
    print("[setup_s3gfn] GP-MolFormer at", p)
except Exception as exc:
    print(f"[setup_s3gfn] GP-MolFormer prefetch error: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
PY
conda run -n "${ENV_NAME}" python "${PREFETCH_PY}" \
    || echo "[setup_s3gfn] WARNING GP-MolFormer prefetch failed — check the model id / HF access (TODO)"
rm -f "${PREFETCH_PY}"

# --- 6. Import smoke (report-only; temp file, NOT a heredoc — conda run swallows stdin). --------
echo "[setup_s3gfn] verifying install (imports)"
SMOKE_PY="$(mktemp --suffix=.py)"
cat > "${SMOKE_PY}" <<'PY'
import torch
print("[setup_s3gfn] torch", torch.__version__, "cuda?", torch.cuda.is_available())
for mod in ("s3gfn", "transformers", "rdkit"):
    try:
        __import__(mod); print(f"[setup_s3gfn] import {mod} OK")
    except Exception as exc:
        print(f"[setup_s3gfn] import {mod} FAILED: {type(exc).__name__}: {exc}")
try:
    from gflownet.models import bengio2021flow  # the sEH proxy (same as our benchmark)
    print("[setup_s3gfn] import gflownet.models.bengio2021flow OK (sEH proxy available)")
except Exception as exc:
    print(f"[setup_s3gfn] import bengio2021flow FAILED: {type(exc).__name__}: {exc}")
PY
conda run -n "${ENV_NAME}" python "${SMOKE_PY}" \
    || echo "[setup_s3gfn] WARNING smoke reported an issue — inspect above"
rm -f "${SMOKE_PY}"

echo "[setup_s3gfn] done. Activate with:  conda activate ${ENV_NAME}"
echo "[setup_s3gfn] NEXT (TODO, validate on the cluster):"
echo "  - sEH data prep: S3-GFN follows RxnFlow's pipeline for the retro env (stock_hb = SynFlowNet"
echo "    105 templates + Enamine stock) used for the synthesizability buffers — see the repo README."
echo "  - train:  PYTHONPATH=${CLONE_DIR}/src conda run -n ${ENV_NAME} python -m s3gfn.train \\"
echo "              --task seh --training_mode s3gfn --use_retrosynthesis --retro_env stock_hb --retro_steps 3"
echo "  - then re-score the emitted SMILES pool on OUR raw sEH proxy (>7 gate) + ingest_candidates.py (has_route=0)."
