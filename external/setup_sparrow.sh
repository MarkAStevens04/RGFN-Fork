#!/bin/bash
# setup_sparrow.sh — OURS. Install SPARROW as the library-route-selection evaluator baseline.
#
# SPARROW (Fromer & Coley, `[fromer2023sparrow]`; github.com/coleygroup/sparrow) jointly
# downselects a set of candidate molecules AND plans their synthesis routes so that shared
# intermediates are made once — a batch MILP over a reaction network. In the LSD-Flow
# benchmark it is the **neutral, cross-method evaluator**: hand it the ordered library a
# strategy chose (SMILES) + a pre-computed route network, and it returns the from-scratch
# total-reaction count that every method — including route-less ones (S3-GFN, FragGFN) — is
# scored on. See docs/LSD_FLOW_BENCHMARK_PLAN.md (T1.1) and validation/lsdflow/eval/.
#
# Why a DEDICATED conda env (`sparrow`) instead of reusing `rgfn`:
#   SPARROW pins python 3.12 + its own scientific stack (scikit-learn==1.4, pulp, pathos,
#   configargparse, ...); the `rgfn` env is python 3.11 + torch/dgl/docking. Like every other
#   benchmarked tool (aizynth/rxnflow/scent), SPARROW runs in its own env and is reached across
#   the boundary by subprocess (validation/lsdflow/adapters/workers/sparrow_worker.py), never by
#   import. It is CPU-only.
#
# SOLVER: SPARROW's environment.yml pins Gurobi 11 (commercial, needs `grbgetkey <license>`).
#   We DO NOT install Gurobi — requirements.txt also ships **PuLP**, which bundles the
#   open-source CBC MILP solver (no license). Our worker selects the PuLP/CBC solver. If a
#   Gurobi license is ever available, `conda install -n sparrow gurobi::gurobi=11` + grbgetkey
#   enables it with no code change.
#
# NO ASKCOS: SPARROW can search routes live via a deployed ASKCOS API (`--path-finder api`),
#   but it also accepts a PRE-COMPUTED route network (`--graph trees_w_info.json`,
#   `--path-finder lookup`, `--coster naive`). We use that path exclusively — AiZynth/MultiAiZ
#   builds the routes upstream (T1.3) and validation/lsdflow/eval/network.py merges them into
#   SPARROW's tree.json schema (T1.2). So no ASKCOS deployment is required.
#
# Run from the repo root (login node is fine — CPU only):
#   bash external/setup_sparrow.sh
#
# Idempotent: re-running skips the clone / env create if already present.

set -euo pipefail

ENV_NAME="${SPARROW_ENV:-sparrow}"
PYVER=3.12

# SPARROW upstream — pinned for reproducibility.
SPARROW_ORG=coleygroup
SPARROW_REPO=sparrow
SPARROW_COMMIT="${SPARROW_COMMIT:-main}"   # TODO: pin to a commit SHA once validated on Balam
CLONE_DIR="external/${SPARROW_REPO}"

echo "[setup_sparrow] env=${ENV_NAME} python=${PYVER} solver=PuLP/CBC (no Gurobi)"

# --- 1. Clone SPARROW at the pinned commit (under external/, git-ignored). ------
if [ ! -d "${CLONE_DIR}" ]; then
    echo "[setup_sparrow] cloning ${SPARROW_ORG}/${SPARROW_REPO}@${SPARROW_COMMIT}"
    git -C external clone "https://github.com/${SPARROW_ORG}/${SPARROW_REPO}" "${SPARROW_REPO}"
    git -C "${CLONE_DIR}" checkout "${SPARROW_COMMIT}"
else
    echo "[setup_sparrow] ${CLONE_DIR} already present — skipping clone"
fi

# --- 2. Create the dedicated conda env (python 3.12, no Gurobi). ----------------
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup_sparrow] creating conda env ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" "python=${PYVER}"
else
    echo "[setup_sparrow] conda env ${ENV_NAME} already exists — skipping create"
fi

# --- 3. Install SPARROW's requirements (minus the `pathlib` footgun) + the pkg. -
# `pathlib` in requirements.txt is the ancient PyPI backport; on python>=3.4 it SHADOWS and
# breaks the stdlib pathlib. Strip it. Everything else (numpy, networkx, pulp, rdkit,
# scikit-learn==1.4, scipy, pandas, joblib, configargparse, pathos, ...) is installed as-is.
REQ_SRC="${CLONE_DIR}/requirements.txt"
REQ_TMP="$(mktemp)"
grep -viE '^\s*pathlib\s*$' "${REQ_SRC}" > "${REQ_TMP}"
echo "[setup_sparrow] installing requirements (pathlib stripped) into ${ENV_NAME}"
conda run -n "${ENV_NAME}" pip install -r "${REQ_TMP}"
rm -f "${REQ_TMP}"

# `gurobipy` is imported at MODULE LOAD in sparrow.selector.base/nonlinear, so the selector package
# won't import without it — even though we solve with PuLP/CBC, not Gurobi. The PyPI `gurobipy` wheel
# imports license-free (only large SOLVES need a license; our solves go through the PuLP/CBC linear
# selector), so installing it just satisfies the import. No `grbgetkey` needed.
echo "[setup_sparrow] installing gurobipy (license-free; needed to import sparrow.selector)"
conda run -n "${ENV_NAME}" pip install gurobipy

echo "[setup_sparrow] installing SPARROW (editable, setup.py develop)"
( cd "${CLONE_DIR}" && conda run -n "${ENV_NAME}" python setup.py develop )

# --- 4. Import + solver smoke (report-only; never aborts the install). ----------
# NB: run the smoke from a TEMP FILE, not `conda run python - <<HEREDOC`. `conda run` does not forward
# stdin to the subprocess, so a heredoc silently no-ops (python sees EOF, prints nothing, exits 0).
echo "[setup_sparrow] verifying install (imports + PuLP/CBC solve)"
SMOKE_PY="$(mktemp --suffix=.py)"
cat > "${SMOKE_PY}" <<'PY'
# (a) SPARROW package + the PuLP linear selector (our MILP backend) import.
try:
    import sparrow  # noqa: F401
    print("[setup_sparrow] import sparrow OK ->", getattr(sparrow, "__file__", "?"))
    from sparrow.selector.linear import LinearSelector  # noqa: F401  (uses PuLP/CBC)
    from sparrow.route_graph import RouteGraph          # noqa: F401
    print("[setup_sparrow] import LinearSelector + RouteGraph OK")
except Exception as exc:
    print(f"[setup_sparrow] SPARROW import FAILED: {type(exc).__name__}: {exc}")
# (b) PuLP + the bundled CBC solver actually solve a trivial LP (the MILP backend we use).
import pulp
prob = pulp.LpProblem("smoke", pulp.LpMaximize)
x = pulp.LpVariable("x", 0, 4)
prob += x
prob += x <= 3
status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
print(f"[setup_sparrow] PuLP/CBC solve status={pulp.LpStatus[status]} x*={pulp.value(x)} "
      f"(expect Optimal / 3.0)")
PY
conda run -n "${ENV_NAME}" python "${SMOKE_PY}" \
    || echo "[setup_sparrow] WARNING smoke test reported an issue — inspect above"
rm -f "${SMOKE_PY}"

echo "[setup_sparrow] done. Activate with:  conda activate ${ENV_NAME}"
echo "[setup_sparrow] NOTE: the toy 3-target MILP smoke (BENCHMARK_PLAN T1.1 acceptance) runs"
echo "[setup_sparrow]       via the route-merge converter (T1.2) once that lands — this script"
echo "[setup_sparrow]       only verifies the env + solver are usable."
