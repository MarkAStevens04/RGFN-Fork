#!/usr/bin/env python
"""Nested dynamic-fragment amortized cost for a SCENT LSD-Flow analysis (Logs/028).

Takes a completed SCENT hub-analysis dir (``records.csv`` + ``compositions.json``) and the run's
dynamic-library snapshot (``fragments_<N>.json`` with ``smiles_to_route``), selects a diverse
library via the flow hub strategy, and reports the reactions-per-mode cost **with SCENT's promoted
intermediates charged in** — each distinct promoted fragment built exactly once (nested via its
route), in both the hub-amortized and independent plans (see
``validation/lsdflow/metrics/cost/dynamic_amortization.py``). Contrasts the BASE assembly cost
(the entry-025/026 metric, which treats promoted fragments as free) against the AUGMENTED cost.

Run (login node, rgfn env — CPU only, no GFN/GPU):
    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/amortized_cost/cost.py \
        --analysis-dir /scratch/.../lsdflow/scent_seh_70189 \
        --snapshot /scratch/.../scent_seh/<ts>/additional_fragments/fragments_4000.json \
        --log-z 74.33 --reward-threshold 7.0 --tag scent_seh_70189
"""
import argparse
import csv
import json
from pathlib import Path

from glue.samplers.lsdflow.acquisition import LSDFlowAcquisition
from glue.samplers.lsdflow.dag import LiteHubDAG
from glue.samplers.lsdflow.hub.registry import get_hub_strategy
from glue.samplers.lsdflow.molecule.registry import get_molecule_strategy
from glue.samplers.lsdflow.records import FlowRecord
from validation.lsdflow.metrics.cost.dynamic_amortization import (
    amortized_library_cost,
    load_cost_table_from_snapshot,
)
from validation.lsdflow.metrics.diversity import mode_representatives

HERE = Path(__file__).resolve().parent


def load_dag(analysis_dir: str, log_z: float, higher_is_better: bool) -> LiteHubDAG:
    recs = []
    with open(Path(analysis_dir) / "records.csv") as fh:
        for r in csv.DictReader(fh):
            recs.append(
                FlowRecord(
                    hub_key=r["hub_key"],
                    child_key=r["child_key"],
                    reward=float(r["reward"]),
                    log_reward=float(r["log_reward"]),
                    log_pf_move=float(r["log_pf_move"]),
                    log_pb_move=float(r["log_pb_move"]),
                    log_pf_stop=float(r["log_pf_stop"]),
                    hub_depth=int(r["hub_depth"]),
                    hub_stereo_key=r.get("hub_stereo_key") or r["hub_key"],
                    child_stereo_key=r.get("child_stereo_key") or r["child_key"],
                )
            )
    return LiteHubDAG.from_records(
        recs, total_trajectories=0, log_z=log_z, higher_is_better=higher_is_better
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-dir",
        required=True,
        help="SCENT hub-analysis out dir (records.csv + compositions.json)",
    )
    ap.add_argument("--snapshot", required=True, help="fragments_<N>.json with smiles_to_route")
    ap.add_argument("--log-z", type=float, required=True)
    ap.add_argument(
        "--reward-threshold", type=float, required=True, help="mode 'hit' bar (sEH ~7.0; DRD2 ~0.5)"
    )
    ap.add_argument("--similarity", type=float, default=0.7)
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--hub-strategy", default="highest_terminating_flow")
    ap.add_argument("--molecule-strategy", default="topk_reward")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--per-hub", type=int, default=8)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    dag = load_dag(a.analysis_dir, a.log_z, a.higher_is_better)
    comps = json.load(open(Path(a.analysis_dir) / "compositions.json"))
    tab = load_cost_table_from_snapshot(json.load(open(a.snapshot)))

    acq = LSDFlowAcquisition(
        hub_strategy=get_hub_strategy(a.hub_strategy, min_children=2),
        molecule_strategy=get_molecule_strategy(a.molecule_strategy),
        batch_size=a.batch_size,
        per_hub=a.per_hub,
        seed=0,
        higher_is_better=a.higher_is_better,
    )
    items = [
        (c.key, c.reward, hub.key, hub.depth)
        for hub, children in acq.select_grouped(dag)
        for c in children
    ]
    rep_idx = mode_representatives(
        [i[0] for i in items],
        [i[1] for i in items],
        higher_is_better=a.higher_is_better,
        reward_threshold=a.reward_threshold,
        similarity_threshold=a.similarity,
    )
    reps = [(items[i][0], items[i][2], items[i][3]) for i in rep_idx]
    res = amortized_library_cost(reps, comps, tab)
    n = res["n_modes"] or 1

    out = {
        "tag": a.tag,
        "analysis_dir": a.analysis_dir,
        "snapshot": a.snapshot,
        "hub_strategy": a.hub_strategy,
        "molecule_strategy": a.molecule_strategy,
        "reward_threshold": a.reward_threshold,
        "similarity": a.similarity,
        "n_promoted_fragments_in_library": len(tab.promoted_set),
        "recipes": bool(tab.recipes),
        "base_reactions_per_mode_hub": round(res["hub_assembly_reactions"] / n, 3),
        "base_reactions_per_mode_independent": round(res["independent_assembly_reactions"] / n, 3),
        "promoted_build_per_mode": round(res["shared_promoted_build_reactions"] / n, 3),
        **res,
    }
    (HERE / f"cost_{a.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n[amortized_cost] wrote {HERE / f'cost_{a.tag}.json'}")


if __name__ == "__main__":
    main()
