#!/usr/bin/env python
"""OURS. Put hub-batching's deliverable spec on the COMPETITOR's catalogue, without routing a pool
we cannot afford to route.

THE PROBLEM. The depth constraint ("the library must be molecules you cannot just buy and couple")
was measured in two different units: ours counts reactions from our 418-block library, the
competitor's from ZINC's 17.4M compounds. ZINC contains most of our blocks and far more, so the SAME
molecule scores shallower on their axis and the constraint bites them harder at every setting --
which makes the per-method curves readable but any crossover between them meaningless. This driver
re-states OUR side in THEIR unit so the x-axis means one thing.

WHY A LOOP AND NOT A BATCH. 346,762 of one seed's enumerated children clear the sEH gate. At the
measured 11.1 s/molecule that is ~1,070 CPU-hours to route them all, for a library of ~100. But the
depth gate only has to be evaluated on molecules that are actually SELECTED, and selection is cheap.
So: run the campaign, route whatever it chose, drop what comes back shallow or unroutable, re-run --
the freed slots pull in replacements, which are routed on the next pass. Molecules already routed are
cached and shared across every depth rung, so the real cost is ~1,000-2,000 molecules per seed.
Unjudged molecules pass the gate on purpose: that is what makes the loop converge from above rather
than requiring the pool to be pre-planned.

WHAT THIS DOES **NOT** CHANGE. The y-axis stays reactions-per-mode counted by SCENT's own count-once
model. AiZynthFinder is used ONLY as a threshold -- "is this molecule far enough from purchasable
material to count as a deliverable" -- never as a pricer. We do not run SPARROW over hub-batching's
selection; the competitor gets its planner-plus-solver pipeline and we get ours, exactly as before.

TWO ASYMMETRIES, BOTH REPORTED, BOTH AGAINST US.
  * The competitor's pool was routed with MultiAiZ (5 cycles, set-based); ours with plain
    AiZynthFinder. The weaker planner will fail molecules MultiAiZ would have solved, shrinking our
    pool relative to theirs. Accepted deliberately -- MultiAiZ's edge is shared intermediates rather
    than shorter individual routes, and its set-based sharing would make routes depend on the whole
    set, which changes every round of this loop.
  * Dropping unroutable molecules removes exactly the ones FURTHEST from the competitor's catalogue.
    Run `--unroutable deep` as the opposite-biased reading and report the pair.

Usage:
    python zinc_axis_driver.py --analysis-dir <sample> --enum-children <enum.json> \
        --snapshot <fragments_N.json> --cache <zinc_depth.jsonl> --config <ladder config.yml> \
        --gate 5.68 --min-zinc-depth 2 --out-dir <res> --nproc 32
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CAMPAIGN = REPO / "experiments/lsd_hubs/campaign/run_campaign.py"
ROUTER = REPO / "experiments/lsd_hubs/campaign/aiz_stock_ladder.py"


def cached_smiles(cache: Path) -> set:
    out = set()
    if cache.exists():
        for line in cache.open():
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("stock") == "zinc":
                    out.add(r["smiles"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--enum-children", required=True)
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--cache", required=True, help="ZINC depth JSONL; shared across depth rungs")
    ap.add_argument("--config", required=True, help="AiZynth config carrying a `zinc` stock key")
    ap.add_argument("--gate", type=float, required=True)
    ap.add_argument("--higher-is-better", type=lambda v: v.lower() != "false", default=True)
    ap.add_argument("--similarity", type=float, default=0.5)
    ap.add_argument("--min-zinc-depth", type=int, required=True)
    ap.add_argument("--unroutable", default="drop", choices=["drop", "deep"])
    ap.add_argument("--budget-reactions", type=int, default=100)
    ap.add_argument("--budget-modes", type=int, default=300)
    ap.add_argument("--child-policy", default="free_frag")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--aizynth-python", default="", help="python in the aizynth env; else conda run"
    )
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    rounds = []

    for rnd in range(1, a.max_rounds + 1):
        rd = out / f"round{rnd}"
        cmd = [
            a.python,
            str(CAMPAIGN),
            "--analysis-dir",
            a.analysis_dir,
            "--enum-children",
            a.enum_children,
            "--reward-threshold",
            str(a.gate),
            "--higher-is-better",
            "true" if a.higher_is_better else "false",
            "--similarity",
            str(a.similarity),
            "--budget-reactions",
            str(a.budget_reactions),
            "--budget-modes",
            str(a.budget_modes),
            "--child-policy",
            a.child_policy,
            "--zinc-depth-cache",
            str(cache),
            "--min-zinc-depth",
            str(a.min_zinc_depth),
            "--unroutable",
            a.unroutable,
            "--tag",
            f"zinc_d{a.min_zinc_depth}_r{rnd}",
            "--out-dir",
            str(rd),
        ]
        if a.snapshot:
            cmd += ["--snapshot", a.snapshot]
        print(f"\n=== round {rnd}: campaign ===", flush=True)
        if subprocess.call(cmd) != 0:
            raise SystemExit(f"[driver] campaign failed in round {rnd}")

        sel = json.loads((rd / "selection_hub_batching.json").read_text())
        accepted = sel["accepted_smiles"]
        known = cached_smiles(cache)
        todo = [s for s in accepted if s not in known]
        rounds.append(
            {
                "round": rnd,
                "n_accepted": len(accepted),
                "n_unjudged": len(todo),
                "cache_size": len(known),
            }
        )
        print(
            f"[driver] round {rnd}: {len(accepted)} accepted, {len(todo)} of them unrouted, "
            f"cache holds {len(known)}",
            flush=True,
        )

        if not todo:
            print(
                f"[driver] CONVERGED after {rnd} round(s): every delivered molecule has a known "
                f"ZINC depth >= {a.min_zinc_depth}"
            )
            break

        smi_f = rd / "to_route.smi"
        smi_f.write_text("\n".join(todo) + "\n")
        rcmd = [] if a.aizynth_python else ["conda", "run", "--no-capture-output", "-n", "aizynth"]
        rcmd += [
            a.aizynth_python or "python",
            str(ROUTER),
            "--smi",
            str(smi_f),
            "--out",
            str(cache),
            "--config",
            a.config,
            "--stocks",
            "zinc",
            "--nproc",
            str(a.nproc),
        ]
        print(f"=== round {rnd}: routing {len(todo)} molecules ===", flush=True)
        if subprocess.call(rcmd) != 0:
            raise SystemExit(f"[driver] routing failed in round {rnd}")
    else:
        print(
            f"[driver] STOPPED at --max-rounds {a.max_rounds} WITHOUT converging — the last "
            f"round's numbers include unjudged molecules and are an UPPER bound."
        )

    (out / "driver_rounds.json").write_text(
        json.dumps(
            {
                "min_zinc_depth": a.min_zinc_depth,
                "unroutable": a.unroutable,
                "gate": a.gate,
                "rounds": rounds,
            },
            indent=2,
        )
    )
    print(f"[driver] wrote {out}/driver_rounds.json")


if __name__ == "__main__":
    main()
