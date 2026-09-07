#!/usr/bin/env python
"""Print `<target> <gate> higher|lower` for each named target, from matrix16/targets.py.

Extracted so a shell launcher can resolve the 5%-FPR bar without embedding a heredoc. The bar is
never defaulted in a launcher: every gate is the score at which 5% of that target's property-matched
decoys pass, and a stale shell default (the pre-standard 7.0 / 0.5 / -8.0) silently re-prices against
a different question. On ClpP the old -8.0 admitted 23% of decoys against the intended 5%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matrix16"))
from targets import get_target  # noqa: E402

rc = 0
for name in sys.argv[1:]:
    try:
        t = get_target(name)
    except Exception as exc:  # targets.py raises its own message for an unknown target
        print(f"{name} ERROR {exc}", file=sys.stderr)
        rc = 1
        continue
    print(name, t.mode_reward_threshold, "higher" if t.higher_is_better else "lower")
sys.exit(rc)
