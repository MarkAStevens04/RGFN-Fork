#!/bin/bash
#SBATCH --job-name=v2_ledger_verify
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=1
# Absolute $SCRATCH log paths: $HOME is read-only on Balam compute nodes.
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
#
# Periodic drift check on the benchmark_v2 provenance ledger: re-hash every recorded source and
# compare against the digest taken when it was copied.
#
# WHY THIS IS A BATCH JOB AND NOT A COMMAND YOU RUN. `ledger.py verify` re-reads every source run
# directory it has a row for -- about 236 GB here -- and the shared scratch filesystem delivered
# ~17 MB/s under load while the copy was running, so this is roughly four hours of I/O. On a login
# node that is both antisocial and subject to the 3600 s CPU limit that already killed one batch
# gnina run (see CLAUDE.md).
#
# ⛔ NEVER SUBSTITUTE `--quick`. It digests mtime instead of content, so it CANNOT match a
# content-recorded digest -- every row would read as drift, or worse, a future reader would see a
# green `--quick` run and believe the copy was content-verified when nothing was read. A cheap check
# that cannot fail the way the expensive one can is worse than no check, because it looks like one.
#
# WHAT THIS DOES AND DOES NOT PROVE. It proves the v1 SOURCES still match what we digested when we
# copied them -- i.e. nothing has since overwritten a cell we depend on, which is exactly how
# s3gfn_seh/seed43's training history was lost. It does NOT re-verify the v2 copies: those were
# checked file-by-file with md5 at copy time and are now frozen read-only, and each carries a
# .copy_manifest.json recording every file's size and md5 if you want to re-check them directly.
#
# `--strict` is deliberately NOT passed. v1 is a live tree that other agents still write to, so a
# source that has legitimately changed since the copy is expected and is reported rather than failed.
# Pass --strict only when you intend "any change at all is a defect".
#
# A GPU IS REQUESTED FOR HASHING, WHICH SOUNDS ABSURD AND IS NOT OUR CHOICE. Balam's sbatch filter
# rejects CPU-only jobs outright ("Jobs on Balam must use --gpus-per-node=1 or --gpus-per-node=4"),
# and billing is GPU-only (TRESBillingWeights=GRES/gpu=1.0,CPU=0,Mem=0), so the 32 cores arrive free
# either way. Parallelising the hashing would not help regardless: at 17 MB/s this is bound by the
# filesystem, not by md5.
#
# Usage:
#   sbatch experiments/benchmark_v2/tools/submit_ledger_verify.sh
#   BENCHMARK_V2_LEDGER=/path/to/PROVENANCE.csv sbatch .../submit_ledger_verify.sh
#
# Exit 0 = every copied row still matches its source. Non-zero = at least one GONE or DRIFT row,
# named in the log.

set -uo pipefail

# SLURM copies the batch script to a spool dir, so BASH_SOURCE does not point at the repo.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 1; }

TOOLS="$REPO/experiments/benchmark_v2/tools"
[ -f "$TOOLS/ledger.py" ] || { echo "FATAL: no ledger.py under $TOOLS"; exit 2; }

# A bare SLURM batch shell has no python on PATH until an env is activated. ledger.py is stdlib-only,
# so conda base is enough -- no need for the heavy rgfn env.
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate base

export BENCHMARK_V2_LEDGER="${BENCHMARK_V2_LEDGER:-$REPO/experiments/benchmark_v2/PROVENANCE.csv}"

echo "host=$(hostname)  started=$(date '+%F %T')"
echo "ledger=$BENCHMARK_V2_LEDGER"
echo "rows=$(($(wc -l < "$BENCHMARK_V2_LEDGER") - 1))"
echo "NOTE: content digests, not --quick. Expect ~4 h of reads."
echo

START=$(date +%s)
python "$TOOLS/ledger.py" verify
RC=$?
echo
echo "elapsed=$(( ($(date +%s) - START) / 60 )) min  rc=$RC"
if [ "$RC" -ne 0 ]; then
    echo "At least one row is GONE or DRIFTed -- see the lines above. A DRIFT on a v1 source is not"
    echo "automatically a defect (v1 is live), but it means the copy can no longer be reproduced from"
    echo "that source, so the v2 copy is now the only record. Check it against its .copy_manifest.json."
fi
exit $RC
