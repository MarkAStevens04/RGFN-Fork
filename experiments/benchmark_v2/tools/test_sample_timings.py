#!/usr/bin/env python
"""Unit tests for the shared sample-stage timing writer.

The compute-versus-reactions exhibit quotes whole-pipeline GPU-hours, which makes this file's output
a PUBLISHED number rather than an operational note. Testing it here rather than only in a live run
matters because the failure mode is silent: a missing component reads as a smaller number, not an
error, and the number still looks plausible.

    python experiments/benchmark_v2/tools/test_sample_timings.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # tools -> benchmark_v2 -> experiments -> repo
sys.path.insert(0, str(REPO_ROOT / "validation" / "lsdflow" / "adapters" / "workers"))

import _artifacts as A  # noqa: E402


def test_full_split():
    """SCENT / RxnFlow / FragGFN: components measured separately."""
    with tempfile.TemporaryDirectory() as td:
        m = A.write_sample_timings(
            Path(td) / "sample_timings.json",
            setup_s=44.0,
            totals_s={"sampling_s": 50766.0, "flow_extract_s": 1200.0},
            n_trajectories=30000, n_records=1896,
            device="cuda:0", reward_name="6td3", model="scent", cuda_synchronized=True,
        )
    assert m["component_split"] == "full", m
    assert m["totals_s"] == {"sampling_s": 50766.0, "flow_extract_s": 1200.0}, m
    # setup counts: a cell pays it, so the exhibit should not have to decide.
    assert m["stage_total_s"] == 44.0 + 50766.0 + 1200.0, m
    print(f"  ok  full split -> stage_total {m['stage_total_s']/3600:.1f} h")


def test_lumped():
    """RGFN times the stage as a whole; that must be DECLARED, not implied by a zero column."""
    with tempfile.TemporaryDirectory() as td:
        m = A.write_sample_timings(
            Path(td) / "sample_timings.json",
            setup_s=30.0, totals_s={"unattributed_s": 45179.0},
            n_trajectories=30000, n_records=1900,
            device="cuda:0", reward_name="6td3", model="rgfn", component_split="lumped",
        )
    assert m["component_split"] == "lumped", m
    assert m["stage_total_s"] == 45209.0, m
    print(f"  ok  lumped     -> stage_total {m['stage_total_s']/3600:.1f} h, declared lumped")


def test_zero_components_are_dropped():
    """A component that was not measured must be ABSENT, not present as 0.0.

    A 0.0 column reads as "measured, and it took no time" -- which is how an unmeasured stage
    silently becomes a free one in a GPU-hour total.
    """
    with tempfile.TemporaryDirectory() as td:
        m = A.write_sample_timings(
            Path(td) / "sample_timings.json",
            setup_s=1.0, totals_s={"sampling_s": 5.0, "reward_gen_s": 0.0},
            n_trajectories=10, n_records=10,
        )
    assert "reward_gen_s" not in m["totals_s"], m
    assert m["totals_s"] == {"sampling_s": 5.0}, m
    print("  ok  unmeasured component dropped rather than written as 0.0")


def test_written_file_is_readable():
    import json
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sample_timings.json"
        A.write_sample_timings(p, setup_s=2.0, totals_s={"sampling_s": 3.0}, n_trajectories=7)
        d = json.loads(p.read_text())
    assert "meta" in d and d["meta"]["n_trajectories"] == 7, d
    print("  ok  file round-trips with a `meta` block, like enum_timings.json")


if __name__ == "__main__":
    print("sample timings:")
    test_full_split()
    test_lumped()
    test_zero_components_are_dropped()
    test_written_file_is_readable()
    print("\nall passed")
