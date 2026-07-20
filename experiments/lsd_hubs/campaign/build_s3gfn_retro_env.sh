#!/bin/bash
# build_s3gfn_retro_env.sh — OURS. Build S3-GFN's retrosynthesis "env" (the training-time
# synthesizability signal) for the LSD-Flow benchmark's S3-GFN entrant (T3.1).
#
# ============================================================================================
# THE DECISION (read this if debugging S3-GFN synthesizability / why the env is what it is):
# ============================================================================================
# S3-GFN checks "is this generated molecule synthesizable?" by retrosynthetically breaking it
# into (a) purchasable BUILDING BLOCKS via (b) a set of REACTION TEMPLATES. Both are text files
# under external/s3gfn/data/envs/<env>/{building_block.smi, template.txt}. The choice of files
# shapes S3-GFN's `synth_ratio` (fraction of samples judged synthesizable), which its contrastive
# training drives UP — so the env must give a non-trivial STARTING synth_ratio or training can't
# bootstrap (an empty "positive" buffer -> nan loss / no learning).
#
# The PAPER's sEH runs use `stock_hb` = ENAMINE STOCK blocks + the 71-template hb_edited set. We
# CANNOT reproduce that: the Enamine Building-Blocks catalog is a vendor request (not public, not
# in this repo — we only have the templates). S3-GFN ships ZINCFrag as the reproducible PUBLIC
# alternative, and its docs note ZINCFrag is "included in AiZynthFinder's built-in ZINC stock" —
# i.e. the SAME stock our benchmark's neutral referee (AiZynth->SPARROW) routes against, so this
# choice ALIGNS S3-GFN's training signal with our measurement rather than biasing it.
#
# But ZINCFrag is a FRAGMENT set (smaller pieces than Enamine stock), so GP-MolFormer's complex
# molecules are hard to decompose into it. MEASURED starting synth_ratio on the GP-MolFormer prior
# (256 samples), 2026-07-20:
#     ZINCFrag + 71 templates (hb_edited, paper's) -> 0.8%  (2/256)  -> TOO WEAK, won't bootstrap
#     ZINCFrag + 105 templates (hb.txt, fuller)    -> 4.7%  (12/256) -> trains; climbs to 90%+
# A 200-step de-risk run with ZINCFrag+105 climbed synth_ratio 4.7% -> 93.8% and produced a 96%-
# synthesizable pool (T3.1 acceptance >=95%). => WE USE `zincfrag_hb105` (ZINCFrag + hb.txt/105).
#
# This choice affects only S3-GFN's TRAINING; the benchmark score (reactions-per-mode) is computed
# post-hoc by AiZynth->SPARROW, independent of this env. See docs/LSD_FLOW_BENCHMARK_PLAN.md (T3),
# Logs/ (the S3-GFN bring-up entry), and validation/configs/s3gfn_seh_fixed.yaml (retro_env).
#
# CANONICALIZATION GOTCHA: S3-GFN indexes blocks by EXACT string and queries with stereo stripped
# (isomericSmiles=False). So building_block.smi MUST be stereo-stripped canonical SMILES, or every
# block match fails silently (in-stock molecules score 0.0). This script strips + dedups.
#
# Run from repo root (s3gfn env available):  bash experiments/lsd_hubs/campaign/build_s3gfn_retro_env.sh
# Idempotent: skips the gdown if the ZINCFrag archive is already present.

set -euo pipefail

ENV_NAME="${S3GFN_ENV:-s3gfn}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
S3ROOT="${REPO_ROOT}/external/s3gfn"
BB_DIR="${S3ROOT}/data/building_blocks"
ZINCFRAG_GZ="${BB_DIR}/zincfrag.smi.gz"
ZINCFRAG_GDRIVE_ID="16N8Xyxr9a-CifjIofgdH3ssFukC4Eh_V"   # S3-GFN's ZINCFrag (data/README.md)

# The two shipped hb template files. We use the 105 set (see THE DECISION above).
TEMPLATE_SRC="${S3ROOT}/data/templates/hb.txt"           # 105 templates (fuller). 71-set = hb_edited.txt
ENV_DIR="${S3ROOT}/data/envs/zincfrag_hb105"             # <- the env S3-GFN runs point at

echo "[build-env] repo=${REPO_ROOT}"
echo "[build-env] target env: ${ENV_DIR}  (ZINCFrag blocks + $(wc -l < "${TEMPLATE_SRC}") hb.txt templates)"

# --- 1. Fetch ZINCFrag (public, ~200k fragments). ------------------------------------------
mkdir -p "${BB_DIR}"
if [ ! -f "${ZINCFRAG_GZ}" ]; then
    echo "[build-env] downloading ZINCFrag -> ${ZINCFRAG_GZ}"
    conda run --no-capture-output -n "${ENV_NAME}" gdown "${ZINCFRAG_GDRIVE_ID}" -O "${ZINCFRAG_GZ}"
else
    echo "[build-env] ZINCFrag archive already present — skipping download"
fi

# --- 2. Stereo-strip + dedup canonicalize into building_block.smi (exact-match indexing). ---
mkdir -p "${ENV_DIR}"
cp "${TEMPLATE_SRC}" "${ENV_DIR}/template.txt"
BUILD_PY="$(mktemp --suffix=.py)"   # temp file, NOT `python - <<HEREDOC` (conda run swallows stdin)
cat > "${BUILD_PY}" <<PY
import gzip, time
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
src = "${ZINCFRAG_GZ}"; dst = "${ENV_DIR}/building_block.smi"
t0 = time.time(); n_in = n_out = 0; seen = set()
opener = gzip.open if src.endswith(".gz") else open
with opener(src, "rt") as fh, open(dst, "w") as out:
    for line in fh:
        parts = line.split()
        if not parts:
            continue
        n_in += 1
        m = Chem.MolFromSmiles(parts[0])
        if m is None:
            continue
        canon = Chem.MolToSmiles(m, isomericSmiles=False)  # stereo-stripped: matches score() query
        if canon in seen:
            continue
        seen.add(canon)
        bid = parts[1] if len(parts) > 1 else f"ZF{n_out}"
        out.write(f"{canon}\t{bid}\n"); n_out += 1
print(f"[build-env] {n_in} in -> {n_out} unique stereo-stripped blocks in {time.time()-t0:.1f}s")
PY
conda run --no-capture-output -n "${ENV_NAME}" python "${BUILD_PY}"
rm -f "${BUILD_PY}"
echo "[build-env] wrote ${ENV_DIR}/{template.txt ($(wc -l < "${ENV_DIR}/template.txt")), building_block.smi ($(wc -l < "${ENV_DIR}/building_block.smi"))}"

# --- 3. Validate: in-stock blocks must score 1.0 (proves the exact-string index + stereo match). --
VAL_PY="$(mktemp --suffix=.py)"
cat > "${VAL_PY}" <<'PY'
from s3gfn.synthesizability import SynthesizabilityEvaluator
ev = SynthesizabilityEvaluator(use_retrosynthesis=True, env="zincfrag_hb105", max_steps=3, num_workers=4)
blocks = []
with open("data/envs/zincfrag_hb105/building_block.smi") as fh:
    for line in fh:
        blocks.append(line.split()[0])
        if len(blocks) >= 5:
            break
sc = ev.score_batch(blocks)
print("[build-env] in-stock block scores:", sc, "(expect all 1.0)")
assert all(s == 1.0 for s in sc), "in-stock blocks did not score 1.0 — canonicalization mismatch!"
print("[build-env] OK: retro env validated.")
PY
( cd "${S3ROOT}" && conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH="${S3ROOT}/src" python "${VAL_PY}" )
rm -f "${VAL_PY}"

echo "[build-env] done. S3-GFN runs use retro_env=zincfrag_hb105 (validation/configs/s3gfn_seh_fixed.yaml)."
