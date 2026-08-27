#!/usr/bin/env python
"""How much of a docking cell's headline number is tie-break luck?

WHY. Docking oracles report to one decimal place, so a docking cell's rewards are heavily QUANTIZED:
`rgfn_clpp` s43 has 30,180 enumerated children sharing only 1,094 distinct rewards (96.8% tied, the
commonest value repeated 1,186 times). Greedy mode selection walks a hub's children in file order and
takes the first of any equally-scoring candidates, so the ORDER of the file decides which of many tied
molecules gets picked. Re-enumerating the same chemistry in a different order is therefore not a no-op
on the selected set -- measured on `rgfn_clpp` s43, `case1_modes_at_100rxn` moved 66 -> 65 while
`reactions_per_mode` stayed bit-identical at 1.623.

That is a reproducibility question a reviewer can ask directly ("we re-ran you and got 65"), so the
honest answer is a measured band rather than a single number. This script shuffles children WITHIN each
hub (which preserves the chemistry exactly -- same hubs, same children, same rewards) and re-runs the
campaign, reporting the spread of each metric across shuffles.

WHAT IT DOES NOT TOUCH. The real artifacts. `run_campaign.py` takes `--enum-children` as an explicit
path, so each shuffle is written to a tempfile and the results go to a tempdir. The cell's own
`enum_children.json` and committed `summary.json` are read-only here. That matters because these are
the files behind published numbers and a shuffled copy must never be able to become the artifact.

Surrogate cells are not the target: sEH's proxy is continuous (22,667 distinct rewards over 27,072
children) and reproduced bit-identically under the same reordering, so the band there is zero.

    python tiebreak_band.py rgfn clpp --seed 43 --shuffles 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# Metrics worth a band. reactions_per_mode and total_reactions are the cost side; the case1/case2
# readings are the budget-anchored ones the paper leads with.
TRACK = [
    "reactions_per_mode",
    "total_reactions",
    "total_modes",
    "case1_modes_at_100rxn",
    "case2_reactions_at_300modes",
    "distinct_hubs_used",
    "n_scaffolds",
]


def emit(gen: str, tgt: str, seed: int) -> dict:
    env = dict(os.environ)
    if seed != 42:
        env["MATRIX16_MANIFEST"] = str(HERE / f"manifest_seed{seed}.csv")
        env["MATRIX16_SCRATCH"] = f"/scratch/markymoo/rgfn_runs/lsdflow/matrix16_seed{seed}"
        env["MATRIX16_RESULTS"] = str(HERE / f"results_seed{seed}")
    else:
        for k in ("MATRIX16_MANIFEST", "MATRIX16_SCRATCH", "MATRIX16_RESULTS"):
            env.pop(k, None)
    out = subprocess.run(
        [sys.executable, str(HERE / "manifest.py"), "--emit", gen, tgt],
        capture_output=True, text=True, env=env, cwd=REPO,
    )
    if out.returncode:
        raise SystemExit(f"manifest emit failed: {out.stderr.strip()}")
    d = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            d[k.strip()] = v.strip().strip("'\"")
    return d


def shuffled_copy(src: Path, dst: Path, rng: random.Random) -> None:
    """Reorder children WITHIN each hub. Hub sequence is left alone: it carries the flow order, which
    is a property of the model rather than an artefact of enumeration, and shuffling it would measure
    something else entirely."""
    d = json.loads(src.read_text())
    for h in d.get("hubs", []):
        ch = h.get("children")
        if ch:
            rng.shuffle(ch)
    dst.write_text(json.dumps(d))


def run_once(gen: str, tgt: str, seed: int, scratch: Path, results: Path) -> dict | None:
    """Run the REAL run_cell_campaign.sh against an isolated scratch tree.

    Deliberately not a re-implementation of its argument list. That script derives the gate, hit
    direction, similarity cutoff, both budgets, the child policy, K, and SCENT's fragment-snapshot
    auto-discovery -- and a band measured with even one of those different from the committed run would
    be a comparison between two experiments rather than a spread within one. Re-deriving them here is
    exactly the drift this project keeps getting bitten by, so instead the cell is redirected via
    MATRIX16_SCRATCH and the script is called unchanged.
    """
    env = dict(os.environ)
    env["MATRIX16_SCRATCH"] = str(scratch)
    env["MATRIX16_RESULTS"] = str(results)
    if seed != 42:
        env["MATRIX16_MANIFEST"] = str(HERE / f"manifest_seed{seed}.csv")
    else:
        env.pop("MATRIX16_MANIFEST", None)
    r = subprocess.run(
        ["bash", str(HERE / "run_cell_campaign.sh"), gen, tgt],
        capture_output=True, text=True, env=env, cwd=REPO,
    )
    hits = sorted(results.glob("*/summary.json"))
    if r.returncode or not hits:
        print(f"      campaign FAILED -- {(r.stderr or r.stdout).strip()[-200:]}")
        return None
    return json.loads(hits[0].read_text())


def stage(spec: dict, seed: int, gen: str, tgt: str, scratch: Path, rng: random.Random) -> None:
    """Build a minimal isolated cell: enum/ holding a SHUFFLED children file plus whatever else the
    campaign reads, and sample/ SYMLINKED to the real one (read-only, and it is the big directory)."""
    cell = Path(spec["ENUM_DIR"]).parent.name
    dst = scratch / cell
    (dst / "enum").mkdir(parents=True, exist_ok=True)
    src_enum = Path(spec["ENUM_DIR"])
    for f in src_enum.iterdir():
        if f.name == "enum_children.json" or not f.is_file():
            continue
        (dst / "enum" / f.name).symlink_to(f)
    shuffled_copy(src_enum / "enum_children.json", dst / "enum" / "enum_children.json", rng)
    link = dst / "sample"
    if not link.exists():
        link.symlink_to(Path(spec["SAMPLE_DIR"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("generator")
    ap.add_argument("target")
    ap.add_argument("--seed", type=int, default=42, help="which matrix seed's cell")
    ap.add_argument("--shuffles", type=int, default=5)
    a = ap.parse_args()

    spec = emit(a.generator, a.target, a.seed)
    enum = Path(spec["ENUM_DIR"]) / "enum_children.json"
    committed = Path(spec["RESULTS_DIR"]) / "summary.json"
    print(f"\n=== {spec['CELL_TAG']} seed {a.seed} — {a.shuffles} shuffles ===")
    print(f"    enum: {enum}")

    base = json.loads(committed.read_text()) if committed.exists() else None
    runs = []
    with tempfile.TemporaryDirectory(dir="/scratch/markymoo/rgfn_runs") as td:
        td = Path(td)
        for i in range(a.shuffles):
            sc = td / f"scratch_{i}"
            stage(spec, a.seed, a.generator, a.target, sc, random.Random(1000 + i))
            s = run_once(a.generator, a.target, a.seed, sc, td / f"res_{i}")
            if s:
                runs.append(s)
                hb = s.get("hub_batching", {})
                print(f"    shuffle {i}: modes@100rxn={hb.get('case1_modes_at_100rxn')}  "
                      f"r/m={hb.get('reactions_per_mode')}  n_scaffolds={hb.get('n_scaffolds')}")

    if not runs:
        raise SystemExit("no shuffle completed")

    print(f"\n    {'metric':30s} {'committed':>10s} {'min':>8s} {'max':>8s} {'spread':>8s} {'CV%':>7s}")
    print("    " + "-" * 76)
    for m in TRACK:
        vals = [r.get("hub_batching", {}).get(m) for r in runs]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            continue
        c = (base or {}).get("hub_batching", {}).get(m)
        cv = 100 * statistics.pstdev(vals) / abs(statistics.mean(vals)) if statistics.mean(vals) else 0
        band = max(vals) - min(vals)
        flag = "  <-- MOVES" if band else ""
        print(f"    {m:30s} {str(c):>10s} {min(vals):>8g} {max(vals):>8g} {band:>8g} {cv:>6.2f}%{flag}")

    # the edge is the published quantity, so band it directly rather than leaving it to be inferred
    edges = [r["best_candidate"]["reactions_per_mode"] / r["hub_batching"]["reactions_per_mode"]
             for r in runs if r.get("best_candidate", {}).get("reactions_per_mode")
             and r.get("hub_batching", {}).get("reactions_per_mode")]
    if edges:
        print(f"\n    EDGE (best r/m / hub r/m): min {min(edges):.4f}x  max {max(edges):.4f}x  "
              f"spread {max(edges)-min(edges):.4f}")
    mode_edges = [r["hub_batching"]["case1_modes_at_100rxn"] / r["best_candidate"]["case1_modes_at_100rxn"]
                  for r in runs if r.get("best_candidate", {}).get("case1_modes_at_100rxn")]
    if mode_edges:
        print(f"    EDGE (hub modes / best modes @100rxn): min {min(mode_edges):.4f}x  "
              f"max {max(mode_edges):.4f}x  spread {max(mode_edges)-min(mode_edges):.4f}")


if __name__ == "__main__":
    main()
