#!/usr/bin/env python
"""Unit tests for the two numbers that decide whether a v2 cell is read correctly.

Both are pure functions over small structures, so they are testable without dgl, a GPU, or an
enumeration -- which matters because the surrounding code cannot even be imported on a login node
(``run_campaign`` pulls in rgfn -> dgl -> graphbolt).

    python experiments/benchmark_v2/tools/test_depth_mix.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "lsd_hubs" / "campaign"))


@dataclass
class _Pick:
    cum_reactions: int
    source_hub: str


@dataclass
class _Result:
    accepted: list


def _load_delivered_depth_mix():
    """Import the function without importing run_campaign (which needs dgl)."""
    import ast
    import textwrap

    src = (REPO_ROOT / "experiments" / "lsd_hubs" / "campaign" / "run_campaign.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_delivered_depth_mix":
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {}
            exec(compile(mod, "<extracted>", "exec"), ns)  # noqa: S102 -- our own source
            return ns["_delivered_depth_mix"]
    raise AssertionError("_delivered_depth_mix not found in run_campaign.py")


def test_depth_mix():
    f = _load_delivered_depth_mix()
    hub_depth = {"H0": 0, "H1": 1, "H2": 2}

    # Two depth-0 hubs carrying most of the library -- the v1 shape: few hubs, many modes.
    res = _Result([
        _Pick(10, "H0"), _Pick(20, "H0"), _Pick(30, "H0"),
        _Pick(40, "H1"),
        _Pick(50, "H2"),
        _Pick(200, "H1"),   # beyond the budget; must NOT count
    ])
    got = f(res, hub_depth, 100)
    assert got["n_modes_at_budget"] == 5, got
    assert got["modes_by_hub_depth"] == {0: 3, 1: 1, 2: 1}, got
    assert got["depth0_mode_frac"] == 0.6, got
    assert got["n_unmapped"] == 0, got
    print("  ok  budget is a prefix; depth-0 share is over DELIVERED modes")

    # best-candidate walks no hubs: every pick is unmapped, and the fraction must be null rather
    # than a misleading 0.0 (which would read as "no reliance on bought scaffolds").
    bc = _Result([_Pick(10, ""), _Pick(20, "")])
    got = f(bc, hub_depth, 100)
    assert got["n_from_hubs"] == 0 and got["depth0_mode_frac"] is None, got
    print("  ok  no-hub arm reports null, not 0.0")

    # A hub absent from the enumerated set is counted as unmapped, never silently as depth 0.
    got = f(_Result([_Pick(10, "H0"), _Pick(20, "GHOST")]), hub_depth, 100)
    assert got["n_unmapped"] == 1 and got["modes_by_hub_depth"] == {0: 1}, got
    assert got["depth0_mode_frac"] == 1.0, got
    print("  ok  unknown hub is unmapped, not depth-0")


def test_budget_is_training_rows_only():
    """The gate must count phase=='train' rows, not the shared cumulative counter.

    S3-GFN interleaves a 2,000-molecule eval sample with training on ONE counter, so reading the
    counter makes a cell that is short on training look over budget.
    """
    sys.path.insert(0, str(HERE))
    import csv
    import tempfile

    import verify_cell as V

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "trace.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(V.TRACE_FIELDS)
            n = 0
            # 8,000 training rows and 3,000 eval rows interleaved -> counter 11,000, training 8,000.
            for i in range(8000):
                n += 1
                w.writerow([n, n, "train", i // 64, "CCO", 1.0, 0.1])
                if i % 3 == 0 and n < 11000:
                    n += 1
                    w.writerow([n, n, "eval", i // 64, "CCN", 1.0, 0.1])

        train_rows = sum(1 for r in csv.DictReader(open(p)) if r["phase"] == "train")
        last = int(list(csv.DictReader(open(p)))[-1]["n_scored"])
        assert train_rows == 8000, train_rows
        assert last > 10000, last
        # The point: the counter clears a 10,000 budget while training does not.
        assert last >= 10000 * V.BUDGET_TOLERANCE
        assert not (train_rows >= 10000 * V.BUDGET_TOLERANCE)
        print(f"  ok  counter={last:,} would PASS a 10,000 budget while training={train_rows:,} FAILS")


if __name__ == "__main__":
    print("depth mix:")
    test_depth_mix()
    print("budget measure:")
    test_budget_is_training_rows_only()
    print("\nall passed")
