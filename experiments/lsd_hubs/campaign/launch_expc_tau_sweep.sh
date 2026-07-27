#!/bin/bash
# Exp C τ-sweep (Logs/048 refinement): complete the reactions-per-mode vs diversity curve under
# MultiAiZ→SPARROW pricing in homogenized chemistry. τ=0.5 already landed (jobs 71529 zincsmall /
# 71531 zinc); this fills the remaining cutoffs.
#
# ONE JOB PER CUTOFF, deliberately: each cutoff selects a DIFFERENT library, so MultiAiZ must re-plan
# that pool from scratch (~14 h measured at τ=0.5, which timed out at a 14 h wall *after* finishing its
# compute — hence 20 h here). Sweeping several cutoffs inside one job would serialize them into a
# multi-day run that the 3-day partition limit would kill; separate jobs also parallelize.
#
# Jobs are CPU-bound (AiZynth + the MILP) but still request 1 GPU because Balam schedules by GPU count
# (see the other submit_*.sh in this dir) — that is the house pattern, not an oversight.
#
# Usage:  bash experiments/lsd_hubs/campaign/launch_expc_tau_sweep.sh [zincsmall|zinc] [cutoffs...]
#   default stock  = zincsmall (the homogenized headline arm; ZINC ∪ the 418 reaction-GFN blocks)
#   default cutoffs= 0.3 0.7 0.9   (0.5 already done)
# Submissions retry on QOSMaxSubmitJobPerUserLimit, so this is safe to fire while the matrix campaign
# is still draining the queue.

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

STOCK=${1:-zincsmall}
shift || true
CUTOFFS=${@:-"0.3 0.7 0.9"}
TIME=${TIME:-20:00:00}
N_ITERS=${N_ITERS:-5}
BUDGET_MODES=${BUDGET_MODES:-300}

if [ "$STOCK" = "zincsmall" ]; then
    AICONFIG=/scratch/markymoo/rgfn_runs/lsdflow_sparrow/config_zincsmall.yml
else
    AICONFIG=data/models/aizynthfinder/config.yml
fi

echo "Exp C τ-sweep | stock=$STOCK | cutoffs=$CUTOFFS | time=$TIME | n_iters=$N_ITERS"

for CUT in $CUTOFFS; do
    TAG="scent_seh_multiaiz_${STOCK}_t${CUT/./}"
    for i in $(seq 1 288); do   # ~24 h of 5-min retries; the queue is shared with the matrix campaign
        JID=$(env TIME="$TIME" BUDGET_MODES="$BUDGET_MODES" \
                  CUTMIN="$CUT" CUTMAX="$CUT" CUTSTEP=0.2 SNAP_POINTS=1 N_ITERS="$N_ITERS" \
                  STOCK="$STOCK" AICONFIG="$AICONFIG" TAG="$TAG" \
              sbatch -p compute --time="$TIME" --parsable \
                  experiments/lsd_hubs/campaign/submit_multiaiz_headline.sh 2>&1)
        if [[ "$JID" =~ ^[0-9]+$ ]]; then
            echo "  SUBMITTED τ=$CUT -> job $JID  (tag $TAG)"
            break
        fi
        [ "$i" = 1 ] && echo "  τ=$CUT queue full (QOS) — retrying every 5 min..."
        sleep 300
    done
done

echo ""
echo "DONE submitting. Results -> /scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/<tag>/"
echo "Combine with the existing τ=0.5 run into one curve once they land."
