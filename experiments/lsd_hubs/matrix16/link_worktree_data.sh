#!/bin/bash
# Symlink gitignored data/ + external/ payloads from a source checkout into this worktree.
#
# WHY: `git worktree` only checks out *tracked* files. The pipeline reads large, gitignored payloads
# via RELATIVE paths that a fresh worktree lacks, so runs fail with FileNotFoundError:
#   * data/ files  — `data/libraries/glue_standard_v1/{fragments,reactions}.csv`,
#     `data/models/rxnflow_env*/*`, AiZynth models, ...
#   * external/ clones — the heavy baseline repos (`external/scent`, `gflownet`, `RxnFlow`, `s3gfn`,
#     `sparrow`). e.g. `scent_worker.py` chdir's into `external/scent` at runtime. (FragGFN/RxnFlow
#     import their pkgs via the conda env's editable install pointing at the MAIN checkout, so they
#     don't strictly need the worktree copy — but SCENT does, and linking all is uniform + safe.)
# The symlinks are themselves gitignored, so they never get committed. The eventual real run happens
# from the merged main checkout (which has these), so this helper is only for building + smoke here.
#
# Usage:  bash experiments/lsd_hubs/matrix16/link_worktree_data.sh [SOURCE_CHECKOUT]
#   SOURCE_CHECKOUT defaults to the main repo root inferred from this worktree path.
set -uo pipefail

WT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"           # worktree repo root
SRC="${1:-$(cd "$WT_ROOT/../../.." && pwd)}"                               # main checkout (…/RGFN-Fork)

if [ ! -d "$SRC/data" ]; then
  echo "ERROR: source checkout data/ not found at $SRC/data" >&2
  echo "Pass the main checkout path explicitly: link_worktree_data.sh /path/to/RGFN-Fork" >&2
  exit 1
fi
if [ "$SRC" = "$WT_ROOT" ]; then
  echo "ERROR: source == worktree ($SRC); nothing to link." >&2
  exit 1
fi

# 1) data/ — per-FILE (data/ has a mix of tracked + gitignored files).
echo "linking gitignored data/ files:  $SRC/data  ->  $WT_ROOT/data"
linked=0
while IFS= read -r rel; do
  dst="$WT_ROOT/$rel"
  [ -e "$dst" ] && continue                # already present (tracked or previously linked)
  mkdir -p "$(dirname "$dst")"
  ln -s "$SRC/$rel" "$dst" && linked=$((linked + 1))
done < <(cd "$SRC" && find data -type f | sed 's|^\./||')
echo "  linked $linked missing data/ file(s)."

# 2) external/ — per top-level DIR (the clones are whole git repos; symlink the directory).
if [ -d "$SRC/external" ]; then
  echo "linking gitignored external/ clones: $SRC/external  ->  $WT_ROOT/external"
  mkdir -p "$WT_ROOT/external"
  ext_linked=0
  for d in "$SRC"/external/*/; do
    name="$(basename "$d")"
    dst="$WT_ROOT/external/$name"
    [ -e "$dst" ] && continue              # tracked dir already checked out
    ln -s "${d%/}" "$dst" && ext_linked=$((ext_linked + 1)) && echo "  + external/$name"
  done
  echo "  linked $ext_linked missing external/ clone(s)."
fi

echo "done. data/ + external/ are now complete for pipeline runs in this worktree."
