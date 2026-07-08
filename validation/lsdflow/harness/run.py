"""LSD-Flow analysis harness — one (model x reward) run end-to-end (proposal §10 steps 1-2).

Pipeline: adapter samples trajectories -> build the rich HubDAG (flow recovery + ``U(h)``) ->
rank hubs under every registered strategy -> run each (hub x molecule) acquisition combo
through the amortized cost + Butina-mode metrics -> the flow-vs-visitation TB-integrity
diagnostic (§2/§8) -> persist DAG + a readable report.

Run (login node, GFN inference only — no docking; prefix with the smoke env per CLAUDE.md):

    source ~/bin/rgfn-smoke-env.sh
    python -m validation.lsdflow.harness.run \
        --checkpoint /scratch/.../seh_proxy_stdlib/<ts>/train/checkpoints/last_gfn.pt \
        --n-trajectories 2000
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

from glue.samplers.lsdflow.acquisition import LSDFlowAcquisition
from glue.samplers.lsdflow.hub.registry import get_hub_strategy
from glue.samplers.lsdflow.molecule.registry import get_molecule_strategy
from validation.lsdflow.adapters import get_adapter
from validation.lsdflow.dag import build_hub_dag
from validation.lsdflow.harness.config import LSDFlowRunConfig
from validation.lsdflow.metrics.cost import get_cost_model
from validation.lsdflow.metrics.diversity import (
    count_modes,
    mode_counter,
    unique_scaffolds,
)


# ------------------------------------------------------------------ small helpers
def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pts = [
        (x, y)
        for x, y in zip(xs, ys)
        if x == x and y == y and abs(x) != math.inf and abs(y) != math.inf
    ]
    n = len(pts)
    if n < 3:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _round(x: float, n: int = 4):
    return None if (x != x) else round(x, n)


def _hub_row(hub, dag) -> dict:
    u = hub.uncertainty()
    return {
        "hub_key": hub.key,
        "depth": hub.depth,
        "n_children": hub.n_children,
        "visit_count": hub.visit_count,
        "log_flow_consensus": _round(hub.log_flow_consensus()),
        "log_flow_terminating": _round(hub.log_flow_terminating()),
        "uncertainty": _round(u),
        "effective_n": hub.effective_n(),
        "best_reward": _round(hub.best_reward(dag.higher_is_better)),
        "log_visitation": _round(dag.log_visitation(hub)),
    }


def _build_hub_strategy(name: str, cfg: LSDFlowRunConfig):
    kwargs = {"min_children": cfg.min_children_for_hub}
    if name == "lowest_uncertainty":
        kwargs["min_effective_n"] = max(2, cfg.min_children_for_hub)
    if name == "most_modes":
        kwargs["mode_counter"] = mode_counter(cfg.mode_cutoff)
    return get_hub_strategy(name, **kwargs)


def _analyze_combo(dag, hub_name: str, mol_name: str, cfg: LSDFlowRunConfig) -> dict:
    acq = LSDFlowAcquisition(
        hub_strategy=_build_hub_strategy(hub_name, cfg),
        molecule_strategy=get_molecule_strategy(mol_name),
        batch_size=cfg.out_batch_size,
        per_hub=cfg.per_hub,
        seed=cfg.seed,
        higher_is_better=dag.higher_is_better,
    )
    groups = acq.select_grouped(dag)
    cost = get_cost_model("reactions_per_mode")
    batch_rx = sum(cost.batch_reactions(hub.depth, len(children)) for hub, children in groups)
    child_depths = [hub.depth + 1 for hub, children in groups for _ in children]
    indep_rx = cost.independent_reactions(child_depths)
    cross_keys = [c.key for _hub, children in groups for c in children]  # stereo-stripped -> modes
    n_modes = count_modes(cross_keys, cutoff=cfg.mode_cutoff)
    n_scaffolds = unique_scaffolds(cross_keys)
    n_mol = len(cross_keys)
    return {
        "hub_strategy": hub_name,
        "molecule_strategy": mol_name,
        "n_hubs_used": len(groups),
        "n_molecules": n_mol,
        "n_modes": n_modes,
        "n_scaffolds": n_scaffolds,
        "batch_reactions": batch_rx,
        "independent_reactions": indep_rx,
        "reaction_savings": indep_rx - batch_rx,
        "reactions_per_mode_hub": _round(cost.per_mode(batch_rx, n_modes), 3),
        "reactions_per_mode_independent": _round(cost.per_mode(indep_rx, n_modes), 3),
    }


# ------------------------------------------------------------------ driver
def run(cfg: LSDFlowRunConfig) -> dict:
    adapter = get_adapter(
        cfg.model,
        config_path=cfg.config_path,
        checkpoint_path=cfg.checkpoint_path,
        reward_name=cfg.reward_name,
        device=cfg.device,
        batch_size=cfg.sample_batch_size,
    )
    try:
        sample = adapter.sample_flow_records(cfg.n_trajectories)
    finally:
        adapter.close()

    dag = build_hub_dag(sample, run_id=cfg.run_id)
    summary = dag.summary()
    print("\n===== HubDAG summary =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    report = {
        "config": asdict(cfg),
        "summary": summary,
        "strategies": {},
        "acquisitions": [],
        "diagnostics": {},
    }

    print("\n===== Hub rankings (top hubs per strategy) =====")
    for name in cfg.hub_strategies:
        ranked = _build_hub_strategy(name, cfg).rank(dag)[: cfg.report_top_hubs]
        rows = [_hub_row(h, dag) for h in ranked]
        report["strategies"][name] = rows
        print(f"\n-- {name} (top {len(rows)}) --")
        for r in rows:
            print(
                f"   depth={r['depth']} n_ch={r['n_children']:>2} "
                f"U={r['uncertainty']} termF={r['log_flow_terminating']} "
                f"consF={r['log_flow_consensus']} visit={r['visit_count']} "
                f"bestR={r['best_reward']}  {r['hub_key'][:60]}"
            )

    # TB-integrity diagnostic: the DB-recovered flow vs the reward-free visitation estimate
    # should agree on multichild hubs (§2/§8).
    multichild = [h for h in dag.hubs_iter() if h.n_children >= 2 and h.visit_count > 0]
    r_fv = _pearson(
        [h.log_flow_consensus() for h in multichild],
        [dag.log_visitation(h) for h in multichild],
    )
    r_ft = _pearson(
        [h.log_flow_consensus() for h in multichild],
        [h.log_flow_terminating() for h in multichild],
    )
    report["diagnostics"] = {
        "n_multichild_hubs": len(multichild),
        "pearson_flow_vs_visitation": _round(r_fv) if r_fv is not None else None,
        "pearson_consensus_vs_terminating": _round(r_ft) if r_ft is not None else None,
    }
    print("\n===== Diagnostics =====")
    print(f"  multichild hubs: {len(multichild)}")
    print(
        f"  Pearson(consensus logF, visitation logF): {report['diagnostics']['pearson_flow_vs_visitation']}"
    )

    print("\n===== Acquisition combos (batch cost / diversity) =====")
    for hub_name, mol_name in cfg.combos:
        row = _analyze_combo(dag, hub_name, mol_name, cfg)
        report["acquisitions"].append(row)
        print(
            f"  {hub_name:26s} x {mol_name:14s} -> "
            f"{row['n_molecules']} mols / {row['n_modes']} modes from "
            f"{row['n_hubs_used']} hubs | rxn/mode hub={row['reactions_per_mode_hub']} "
            f"vs indep={row['reactions_per_mode_independent']} "
            f"(saved {row['reaction_savings']:.0f} rxns)"
        )

    _save(cfg, dag, report)
    return report


def _save(cfg: LSDFlowRunConfig, dag, report: dict) -> None:
    out = Path(cfg.out_dir)
    dag.save(out)
    with open(out / "report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    # A small, committed-friendly acquisitions table.
    import csv

    with open(out / "acquisitions.csv", "w", newline="") as fh:
        if report["acquisitions"]:
            w = csv.DictWriter(fh, fieldnames=list(report["acquisitions"][0].keys()))
            w.writeheader()
            w.writerows(report["acquisitions"])
    print(f"\n[LSD-Flow] wrote DAG + report to {out}")


def _parse_args(argv: Optional[List[str]] = None) -> LSDFlowRunConfig:
    p = argparse.ArgumentParser(description="LSD-Flow post-hoc hub analysis (one model x reward).")
    p.add_argument("--model", default="rgfn")
    p.add_argument("--config-path", default="configs/glue/fixed_reward_seh_proxy_stdlib.gin")
    p.add_argument("--checkpoint", required=True, help="trained last_gfn.pt")
    p.add_argument("--reward-name", default="seh")
    p.add_argument("--run-id", default=None)
    p.add_argument("--n-trajectories", type=int, default=2000)
    p.add_argument("--sample-batch-size", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-batch-size", type=int, default=96)
    p.add_argument("--per-hub", type=int, default=8)
    p.add_argument("--mode-cutoff", type=float, default=0.65)
    p.add_argument("--min-children", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="validation/lsdflow/results/seh_rgfn_pilot")
    a = p.parse_args(argv)
    return LSDFlowRunConfig(
        model=a.model,
        config_path=a.config_path,
        checkpoint_path=a.checkpoint,
        reward_name=a.reward_name,
        run_id=a.run_id,
        n_trajectories=a.n_trajectories,
        sample_batch_size=a.sample_batch_size,
        device=a.device,
        out_batch_size=a.out_batch_size,
        per_hub=a.per_hub,
        mode_cutoff=a.mode_cutoff,
        min_children_for_hub=a.min_children,
        seed=a.seed,
        out_dir=a.out_dir,
    )


def main(argv: Optional[List[str]] = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
