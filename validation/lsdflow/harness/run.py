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
from glue.samplers.lsdflow.dag import LiteHubDAG
from glue.samplers.lsdflow.hub.registry import get_hub_strategy
from glue.samplers.lsdflow.molecule.registry import get_molecule_strategy
from glue.samplers.lsdflow.records import FlowRecord
from validation.lsdflow.adapters import get_adapter
from validation.lsdflow.dag import build_hub_dag
from validation.lsdflow.harness.config import LSDFlowRunConfig
from validation.lsdflow.metrics.diversity import (
    mode_counter,
    mode_representatives,
    unique_scaffolds,
)


def _per_mode_cost(items, similarity_threshold, reward_threshold, higher_is_better):
    """Reactions to synthesize a **diverse library of one representative per (paper-comparable)
    mode** (§11). Modes follow ``TanimotoSimilarityModes``: reward-gated + greedy ECFP-r3
    sphere-exclusion at ``similarity_threshold``, best-reward-first (so each mode's
    representative is its best binder — see :func:`mode_representatives`).

    ``items``: ``(key, reward, hub_id, hub_depth)`` per candidate child. Then:

      * **hub batch**: each *distinct* owner hub built once (``hub_depth`` reactions) + one final
        reaction per representative — ``Σ_distinct_hub depth + n_modes``.
      * **independent**: each representative built from scratch — ``Σ_rep (hub_depth + 1)``.

    Costing one representative *per mode* (not all children) keeps the per-mode number bounded by
    the trajectory length; redundancy (children per mode) is reported separately. Returns
    ``(n_modes, hub_rx, indep_rx)``.
    """
    rep_idx = mode_representatives(
        [it[0] for it in items],
        [it[1] for it in items],
        higher_is_better=higher_is_better,
        reward_threshold=reward_threshold,
        similarity_threshold=similarity_threshold,
    )
    if not rep_idx:
        return 0, 0.0, 0.0
    reps = [items[i] for i in rep_idx]
    n_modes = len(reps)
    distinct_hub_depth = {it[2]: it[3] for it in reps}
    hub_rx = float(sum(distinct_hub_depth.values()) + n_modes)
    indep_rx = float(sum(it[3] + 1 for it in reps))
    return n_modes, hub_rx, indep_rx


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
        kwargs["mode_counter"] = mode_counter(cfg.mode_similarity_threshold)
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
    # One representative per mode (each child tagged with its owner hub's id + build depth), so
    # reactions/mode is bounded by trajectory length and not inflated by a redundant batch.
    items = [(c.key, c.reward, hub.key, hub.depth) for hub, children in groups for c in children]
    n_modes, batch_rx, indep_rx = _per_mode_cost(
        items, cfg.mode_similarity_threshold, cfg.mode_reward_threshold, dag.higher_is_better
    )
    n_scaffolds = unique_scaffolds([it[0] for it in items])
    n_mol = len(items)
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
        "reactions_per_mode_hub": _round((batch_rx / n_modes) if n_modes else float("nan"), 3),
        "reactions_per_mode_independent": _round(
            (indep_rx / n_modes) if n_modes else float("nan"), 3
        ),
    }


def _load_lite_dag(csv_path: str, *, higher_is_better: bool, log_z: float) -> LiteHubDAG:
    """Rebuild a ``LiteHubDAG`` from a persisted ``records.csv`` (skips re-sampling).

    Reads the §2 log-terms straight back into ``FlowRecord``s. Visit counts are not persisted,
    so the reward-free visitation estimators are unavailable on a loaded DAG (fine for the
    enumeration path). Tolerates old records without the stereo-key columns (falls back to the
    stripped key)."""
    import csv

    recs = []
    with open(csv_path) as fh:
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
    if cfg.from_records:
        dag = _load_lite_dag(
            cfg.from_records, higher_is_better=adapter.higher_is_better, log_z=adapter.log_z
        )
        summary = {
            "model": cfg.model,
            "reward": cfg.reward_name,
            "source": cfg.from_records,
            "n_hubs": len(dag),
            "log_z": dag.log_z,
            "higher_is_better": dag.higher_is_better,
        }
        print(f"\n[LSD-Flow] loaded DAG from {cfg.from_records} (no re-sampling): {len(dag)} hubs")
    else:
        sample = adapter.sample_flow_records(cfg.n_trajectories)
        dag = build_hub_dag(sample, run_id=cfg.run_id)
        summary = dag.summary()
        # Persist the (expensive, CPU-bound) sample BEFORE enumeration, so a slow or failed
        # deep-hub enumeration can never lose a multi-hour compute-node run. `--from-records`
        # can then re-run enumeration on this DAG without re-sampling.
        dag.save(cfg.out_dir)
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

    if cfg.enumerate_top_hubs > 0:
        report["enumeration"] = enumerate_and_report(adapter, dag, cfg)

    if cfg.from_records:
        # Enumeration is additive — don't clobber the source DAG's committed report/artifacts.
        print(f"\n[LSD-Flow] from-records mode: enumeration artifacts written to {cfg.out_dir}")
    else:
        _save(cfg, dag, report)
    adapter.close()
    return report


def enumerate_and_report(adapter, dag, cfg: LSDFlowRunConfig) -> dict:
    """Phase-2: exhaustively enumerate a selected hub's one-reaction children and compare to
    what sampling found (proposal §4b/§6; the Logs/025 boundary-artifact follow-up).

    Selects the union of (a) the top hubs by terminating flow — the method's own picks — and
    (b) the **interior** (depth 1-2) multi-child hubs — the "build-early, diversify-late"
    candidates that (i) sampling under-counts and (ii) show a real reactions-per-mode saving
    (a depth-0 fragment costs 0 reactions, so it has diversity but no amortization; and depth-3
    hubs' children hit the max-reaction boundary where P_B can't be recovered). For each,
    enumerates all terminal children, checks the sampled children are recovered (exhaustiveness
    sanity), and recomputes diversity + amortized cost on the *full* enumerated neighborhood.
    """
    tf_hubs = _build_hub_strategy("highest_terminating_flow", cfg).rank(dag)[
        : cfg.enumerate_top_hubs
    ]
    interior = sorted(
        (h for h in dag.hubs_iter() if 0 <= h.depth <= 2 and h.n_children >= 2),
        key=lambda h: (h.visit_count, h.n_children),  # visit_count is 0 on a loaded DAG
        reverse=True,
    )[: cfg.enumerate_top_hubs]
    selected, seen = [], set()
    for h in list(tf_hubs) + interior:
        if h.key not in seen:
            seen.add(h.key)
            selected.append(h)

    hubs_arg = [(h.stereo_key or h.key, h.depth) for h in selected]
    print(
        f"\n===== Enumerating children of {len(hubs_arg)} selected hubs "
        f"(<= {cfg.enumerate_max_children} each) =====",
        flush=True,
    )
    records, per_hub = adapter.enumerate_hub_children(
        hubs_arg, max_children=cfg.enumerate_max_children
    )
    enum_dag = LiteHubDAG.from_records(
        records, total_trajectories=0, log_z=dag.log_z, higher_is_better=dag.higher_is_better
    )

    rows = []
    for h in selected:
        sampled = dag.hub(h.key)
        sampled_keys = {c.key for c in sampled.children} if sampled else set()
        enum_hub = enum_dag.hub(h.key)
        enum_children = enum_hub.children if enum_hub else []
        enum_keys = {c.key for c in enum_children}
        recovered = len(sampled_keys & enum_keys)
        rewards = [c.reward for c in enum_children if c.reward == c.reward]
        best = (max(rewards) if dag.higher_is_better else min(rewards)) if rewards else float("nan")
        k = len(enum_children)
        # One representative per mode from this single hub (all children share its build depth),
        # so reactions/mode is bounded by trajectory length; redundancy is reported separately.
        items = [(c.key, c.reward, h.key, h.depth) for c in enum_children]
        n_modes, batch_rx, indep_rx = _per_mode_cost(
            items, cfg.mode_similarity_threshold, cfg.mode_reward_threshold, dag.higher_is_better
        )
        row = {
            "hub_key": h.key,
            "depth": h.depth,
            "visit_count": h.visit_count,
            "n_sampled_children": len(sampled_keys),
            "n_enumerated_children": k,
            "children_per_mode": _round(k / n_modes, 2) if n_modes else None,  # redundancy signal
            "sampled_recovered_by_enum": f"{recovered}/{len(sampled_keys)}",
            "enum_modes": n_modes,
            "enum_best_reward": _round(best, 3),
            "enum_reactions_per_mode_hub": _round(
                (batch_rx / n_modes) if n_modes else float("nan"), 3
            ),
            "enum_reactions_per_mode_independent": _round(
                (indep_rx / n_modes) if n_modes else float("nan"), 3
            ),
        }
        rows.append(row)
        print(
            f"   depth={h.depth} sampled={len(sampled_keys)} -> enumerated={k} children "
            f"({n_modes} modes, best R={best:.2f}); sampled recovered "
            f"{recovered}/{len(sampled_keys)}  {h.key[:48]}",
            flush=True,
        )

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "enumeration.json", "w") as fh:
        json.dump({"per_hub_enumeration": per_hub, "enrichment": rows}, fh, indent=2)
    import csv

    with open(out / "enumerated_records.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "hub_key",
                "child_key",
                "reward",
                "log_reward",
                "log_pf_move",
                "log_pb_move",
                "log_pf_stop",
                "hub_depth",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.hub_key,
                    r.child_key,
                    r.reward,
                    r.log_reward,
                    r.log_pf_move,
                    r.log_pb_move,
                    r.log_pf_stop,
                    r.hub_depth,
                ]
            )
    return {
        "n_hubs_enumerated": len(selected),
        "n_enumerated_records": len(records),
        "enrichment": rows,
        "per_hub": per_hub,
    }


def _save(cfg: LSDFlowRunConfig, dag, report: dict) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The rich DAG was already persisted right after sampling (before enumeration); here we only
    # write the report + acquisitions table.
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
    p.add_argument(
        "--mode-similarity",
        type=float,
        default=0.7,
        help="Tanimoto greedy-mode similarity threshold (paper recipe: ECFP r=3, 0.7)",
    )
    p.add_argument(
        "--mode-reward-threshold",
        type=float,
        default=None,
        help="reward/binding 'hit' cutoff for a mode (orientation from the reward); "
        "omit for structure-only modes",
    )
    p.add_argument("--min-children", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--from-records",
        default="",
        help="reuse a persisted records.csv DAG instead of re-sampling (RGFN "
        "sampling is slow); enumeration artifacts are written additively",
    )
    p.add_argument(
        "--enumerate-top-hubs",
        type=int,
        default=0,
        help="phase-2: exhaustively enumerate children of this many selected hubs (0=off)",
    )
    p.add_argument(
        "--enumerate-max-children",
        type=int,
        default=2000,
        help="per-hub enumeration cap (docking budget guard; free for sEH)",
    )
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
        mode_similarity_threshold=a.mode_similarity,
        mode_reward_threshold=a.mode_reward_threshold,
        min_children_for_hub=a.min_children,
        seed=a.seed,
        from_records=a.from_records,
        enumerate_top_hubs=a.enumerate_top_hubs,
        enumerate_max_children=a.enumerate_max_children,
        out_dir=a.out_dir,
    )


def main(argv: Optional[List[str]] = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
