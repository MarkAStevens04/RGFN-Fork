#!/usr/bin/env python
"""Write a discoverable sidecar recording the best lower bound for each NON-certifiable cell.

Four cells resisted certification at every gap tried (1e-7 default, 1e-2, 1e-6). For those the
honest number is the LARGEST incumbent found across all solves -- and for two of them the main CSV
currently holds a SMALLER value than we have measured, which understates the competitor and
therefore flatters us. merge_longsolve_rows.py deliberately refuses to touch a capped row, so this
cannot be fixed by the merge and has to be recorded where a reader will find it.
"""
import json
import os
from pathlib import Path

RES = Path(os.environ["SCRATCH"]) / "rgfn_runs" / "lsdflow_sparrow" / "results"

BOUNDS = {
    "saturn_seh_seed42_stage2_pruned": {
        "solves": {"1e-7 (default)": [4, "TimeLimit"], "1e-2": [3, "Optimal"],
                   "1e-6": [3, "TimeLimit"]},
        "best_lower_bound": 4,
        "in_main_csv": 4,
        "note": "main CSV already holds the best bound",
    },
    "tango_clpp_seed43_stage2_pruned": {
        "solves": {"1e-7 (default)": [14, "TimeLimit"], "1e-2": [13, "Optimal"],
                   "1e-6": [13, "TimeLimit"]},
        "best_lower_bound": 14,
        "in_main_csv": 14,
        "note": "main CSV already holds the best bound",
    },
    "tango_clpp_seed44_stage2_pruned": {
        "solves": {"1e-7 (default)": [8, "TimeLimit"], "1e-2": [9, "Optimal"],
                   "1e-6": [10, "TimeLimit"]},
        "best_lower_bound": 10,
        "in_main_csv": 8,
        "note": "MAIN CSV UNDERSTATES by 2 modes; quote 10, which is in the _longsolve dir",
    },
    "saturn_seh_seed44_stage2": {
        "solves": {"1e-7 (default)": [4, "TimeLimit"], "1e-2": [5, "TimeLimit"]},
        "best_lower_bound": 5,
        "in_main_csv": 4,
        "note": "MAIN CSV UNDERSTATES by 1 mode; never certified at any gap",
    },
}

DOC = {
    "what": "Cells whose SPARROW-Batching MILP at R=100 could not be certified at any gap tried.",
    "why_it_matters": (
        "A time-capped row is a LOWER BOUND on the competitor, not an optimum. CLAUDE.md forbids "
        "quoting it as a ratio. Understating the competitor flatters us, so the largest incumbent "
        "across solves is the number to quote."
    ),
    "gaps_tried": ["1e-7 (SPARROW default)", "1e-2", "1e-6"],
    "not_explained": (
        "Neither mode count nor route-network size predicts which cells resist: "
        "saturn_seh_42_pruned resists on 4,404 reaction nodes while reinvent_clpp_43_pruned "
        "certifies in 3 s on 4,710, and saturn_drd2_43_pruned holds 9 modes and solves in 2 s."
    ),
    "cells": BOUNDS,
    "see": "Logs/079_reprice-capped-rows-and-coarse-ladder.md",
}

out = RES / "UNCERTIFIED_R100_BOUNDS.json"
out.write_text(json.dumps(DOC, indent=2) + "\n")
print(f"wrote {out}")
for k, v in BOUNDS.items():
    flag = "  <-- UNDERSTATED" if v["in_main_csv"] < v["best_lower_bound"] else ""
    print(f"  {k:<40} best={v['best_lower_bound']:>3}  csv={v['in_main_csv']:>3}{flag}")
