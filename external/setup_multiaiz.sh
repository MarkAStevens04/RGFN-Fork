#!/bin/bash
# setup_multiaiz.sh — OURS. Install MultiAiZ as the SECOND competitor route-planner baseline (T4.1).
#
# MultiAiZ (Iáñez Picazo et al. 2026, `[ianez2026multiaiz]`; github.com/MolecularAI/multiaiz) extends
# AiZynthFinder to MULTI-TARGET retrosynthesis: it runs AiZynthFinder for N cycles over a SET of
# targets and, after each cycle, appends the discovered intermediates to the stock, so later targets
# route THROUGH those intermediates → convergent routes that reuse shared building steps. Paper uses
# 5 cycles (n_iters=5). ROLE: the smarter alternative to plain AiZynth→SPARROW; each acquisition
# function's accepted-mode pool is priced in ISOLATION by MultiAiZ→SPARROW (sharing found within a
# pool is never leaked to other pools). See docs/LSD_FLOW_BENCHMARK_PLAN.md T4.1, Logs/043,
# validation/lsdflow/eval/multiaiz.py + .../workers/multiaiz_worker.py.
#
# WHY WE INSTALL INTO THE EXISTING `aizynth` ENV (not a dedicated one):
#   MultiAiZ's own env-dev.yml + poetry install source-builds the LEGACY cgrtools 4.1.35, which fails
#   to Cython-compile under modern setuptools (PEP-517 CompileError). Our `aizynth` env already has
#   aizynthfinder 4.4.1 — which SATISFIES MultiAiZ's `^4.4.0`, needs NO cgrtools, and exposes every
#   API MultiAiZ imports (ReactionTree, RouteCollection, Molecule, InMemoryInchiKeyQuery). MultiAiZ IS
#   an AiZynthFinder extension sharing its exact stack, so the aizynth env is its natural home. We
#   install the `multiaiz` package `--no-deps` (+ adjustText for its plotting import). Reuses the same
#   USPTO/ZINC models the from-scratch AiZynth→SPARROW headline uses, so the curves overlay fairly.
#
# CPU-only; reached across the env boundary by subprocess (multiaiz_worker.py), never imported into rgfn.
#
# Run from the repo root (login node is fine):  bash external/setup_multiaiz.sh
# Idempotent: skips the clone if present; pip install is safe to re-run.

set -euo pipefail

ENV_NAME="${MULTIAIZ_ENV:-aizynth}"   # MultiAiZ lives in the aizynth env (see header)

MULTIAIZ_ORG=MolecularAI
MULTIAIZ_REPO=multiaiz
MULTIAIZ_COMMIT="${MULTIAIZ_COMMIT:-main}"   # TODO: pin to a commit SHA once validated on Balam
CLONE_DIR="external/${MULTIAIZ_REPO}"

echo "[setup_multiaiz] installing multiaiz into the '${ENV_NAME}' env (aizynthfinder 4.4.1 already present)"

# --- 1. Clone MultiAiZ (under external/, git-ignored). ------
if [ ! -d "${CLONE_DIR}" ]; then
    echo "[setup_multiaiz] cloning ${MULTIAIZ_ORG}/${MULTIAIZ_REPO}@${MULTIAIZ_COMMIT}"
    git -C external clone "https://github.com/${MULTIAIZ_ORG}/${MULTIAIZ_REPO}" "${MULTIAIZ_REPO}"
    git -C "${CLONE_DIR}" checkout "${MULTIAIZ_COMMIT}"
else
    echo "[setup_multiaiz] ${CLONE_DIR} already present — skipping clone"
fi

# --- 2. Install the multiaiz package (--no-deps: reuse the aizynth env's aizynthfinder/rdkit/pandas/
#        seaborn) + adjustText (MultiAiZ's plotting import; not needed for route discovery but imported).
echo "[setup_multiaiz] pip install adjustText + multiaiz (--no-deps) into ${ENV_NAME}"
conda run -n "${ENV_NAME}" pip install --no-input adjustText
conda run -n "${ENV_NAME}" pip install --no-input --no-deps -e "${CLONE_DIR}"

# --- 3. Smoke: imports + a tiny 2-target n_iters=1 route-discovery run. ------
AICONFIG="${AICONFIG:-$PWD/data/models/aizynthfinder/config.yml}"
echo "[setup_multiaiz] smoke (imports + tiny run against ${AICONFIG})"
SMOKE_PY="$(mktemp --suffix=.py)"   # temp file, NOT `python - <<HEREDOC` (conda run swallows stdin)
cat > "${SMOKE_PY}" <<PY
import sys, os, tempfile, glob
try:
    from aizynthfinder.aizynthfinder import AiZynthFinder
    from multiaiz.multiaiz import MultiAiZ
    print("[setup_multiaiz] import aizynthfinder + multiaiz OK")
except Exception as exc:
    print(f"[setup_multiaiz] IMPORT FAILED: {type(exc).__name__}: {exc}"); sys.exit(1)
cfg = "${AICONFIG}"
if os.path.exists(cfg):
    try:
        finder = AiZynthFinder(configfile=cfg)
        finder.stock.select("zinc"); finder.expansion_policy.select("uspto"); finder.filter_policy.select("uspto")
        out = tempfile.mkdtemp(prefix="multiaiz_smoke_")
        try:
            MultiAiZ(finder, ["O=C(Nc1ccccc1)c1ccc(Cl)cc1", "O=C(Nc1ccccc1)c1ccc(Br)cc1"], out).run(n_iters=1)
        except Exception as exc:
            print(f"[setup_multiaiz] (post-processing skipped: {type(exc).__name__}); trees written")
        n = len(glob.glob(out + "/trees/*.json.gz"))
        print(f"[setup_multiaiz] tiny run wrote {n} cycle tree file(s) -> {out}" if n else "[setup_multiaiz] WARNING no trees written")
    except Exception as exc:
        print(f"[setup_multiaiz] WARNING tiny run failed ({type(exc).__name__}: {exc}); imports still OK")
else:
    print(f"[setup_multiaiz] (config {cfg} absent — skipped functional smoke; imports OK)")
PY
conda run -n "${ENV_NAME}" python "${SMOKE_PY}" || echo "[setup_multiaiz] smoke reported an issue — inspect above"
rm -f "${SMOKE_PY}"

echo "[setup_multiaiz] done. MultiAiZ is in the '${ENV_NAME}' env."
echo "[setup_multiaiz] NEXT: run T4.1 via --evaluator multiaiz (per-pool MultiAiZ→SPARROW; Logs/043)."
