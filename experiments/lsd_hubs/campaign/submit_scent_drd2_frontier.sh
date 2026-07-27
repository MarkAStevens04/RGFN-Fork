#!/bin/bash
#SBATCH --job-name=scent_drd2_frontier
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
# Target-generality (T4.3) step 2: DRD2 reactions-per-mode frontier, native (count-once) pricing —
# hub-batching (free-frag) vs best-candidate, DRD2 "mode" bar 0.5. The second target for the paper's
# generality claim; overlays the sEH curves. Reads the DRD2 hub enumeration from step 1
# (submit_scent_drd2_enum.sh). Pure CPU (count_once needs no AiZynth/SPARROW) but runs in rgfn
# (sweep_campaign imports glue->dgl).
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

ANALYSIS=${ANALYSIS:-/scratch/markymoo/rgfn_runs/lsdflow/scent_drd2_70190}
ENUM=${ENUM:?set ENUM=<campaign_enum_drd2_*/enum_children.json>}
SNAP=${SNAP:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_drd2/2026-07-10_17-28-06/additional_fragments/fragments_4000.json}
OUT_DIR=${OUT_DIR:-/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results}
TAG=${TAG:-scent_drd2_countonce}
THR=${THR:-0.5}            # DRD2 mode bar (activity prob), Logs/028 + matrix16 targets.py

export TORCH_HOME=$SCRATCH/.cache/torch HF_HOME=$SCRATCH/.cache/huggingface PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR" "$TORCH_HOME"
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/rgfn/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"

echo "host=$(hostname)  ANALYSIS=$ANALYSIS  ENUM=$ENUM  THR=$THR"
if [ ! -s "$ENUM" ]; then echo "[drd2-frontier] FATAL: enum missing ($ENUM)" >&2; exit 1; fi

python experiments/lsd_hubs/campaign/sweep_campaign.py \
    --analysis-dir "$ANALYSIS" --enum-children "$ENUM" --snapshot "$SNAP" \
    --reward-threshold "$THR" --tag "$TAG" --out-dir "$OUT_DIR" \
    --evaluator count_once --child-policy free_frag \
    --snapshot-schedule geometric --snapshot-points 15 \
    --budget-modes 300 --cutoff-min 0.30 --cutoff-max 0.90 --cutoff-step 0.10 \
    --n-hubs 200

echo ""
echo "DONE $TAG -> $OUT_DIR/$TAG  (DRD2 native reactions/mode: hub-batching vs best-candidate; overlay on sEH)"
