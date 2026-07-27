#!/bin/bash
# Fan out submit_cell.sh across all RUNNABLE surrogate cells (active + ready in the manifest) as
# SLURM jobs — the "launch them all at once" driver for the sample+enumerate GPU pipeline. Docking
# cells are excluded automatically (run_stage=deferred). The CPU campaign readout is a separate
# step (run_cell_campaign.sh), run after these finish.
#
# Usage:      bash experiments/lsd_hubs/matrix16/launch_surrogates.sh
# Smoke:      N_TRAJ=10000 N_HUBS=50 bash .../launch_surrogates.sh          # small verification run
# Dry run:    DRY_RUN=1 bash .../launch_surrogates.sh                        # print, don't submit
# Filter:     GENS="rgfn scent" bash .../launch_surrogates.sh               # only these generators
# Stage:      STAGE=sample bash .../launch_surrogates.sh                     # sample only (enum later)
#
# NOTE (current build state): only rgfn + scent have a working `--mode enumerate`. FragGFN/RxnFlow
# enumerate is pending — launch those with STAGE=sample until their enumerate lands.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

N_TRAJ=${N_TRAJ:-30000}
N_HUBS=${N_HUBS:-200}
STAGE=${STAGE:-all}
TIME=${TIME:-}            # sbatch --time override (e.g. 1-00:00:00); empty = the script's 12h default
DRY_RUN=${DRY_RUN:-0}
GENS_FILTER="${GENS:-}"

TIME_ARG=()
[ -n "$TIME" ] && TIME_ARG=(--time "$TIME")

mapfile -t CELLS < <(python experiments/lsd_hubs/matrix16/manifest.py --list active-ready)
if [ "${#CELLS[@]}" -eq 0 ]; then
    echo "No active+ready cells (check: python experiments/lsd_hubs/matrix16/manifest.py)."
    exit 0
fi

echo "Launching ${#CELLS[@]} runnable surrogate cell(s)  N_TRAJ=$N_TRAJ N_HUBS=$N_HUBS STAGE=$STAGE dry=$DRY_RUN"
n=0
for line in "${CELLS[@]}"; do
    gen="${line%%$'\t'*}"
    tgt="${line##*$'\t'}"
    if [ -n "$GENS_FILTER" ] && ! grep -qw -- "$gen" <<<"$GENS_FILTER"; then
        continue
    fi
    cmd=(sbatch -J "m16_${gen}_${tgt}" "${TIME_ARG[@]}"
        "--export=ALL,N_TRAJ=${N_TRAJ},N_HUBS=${N_HUBS},STAGE=${STAGE}"
        experiments/lsd_hubs/matrix16/submit_cell.sh "$gen" "$tgt")
    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY: ${cmd[*]}"
    else
        echo "  submit: $gen $tgt"
        "${cmd[@]}"
    fi
    n=$((n + 1))
done
echo "done ($n cell(s) $([ "$DRY_RUN" = 1 ] && echo listed || echo submitted))."
