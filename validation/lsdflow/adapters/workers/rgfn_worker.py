#!/usr/bin/env python
"""RGFN per-env worker — sample + enumerate, emitting the uniform LSD-Flow artifact set.

RGFN is native to the ``rgfn`` env, so — unlike the cross-env SCENT/FragGFN/RxnFlow workers — this
imports the in-process :class:`RGFNAdapter` + ``glue`` flow primitives directly (no RPC). It exists
so **all four generators share one** ``<gen>_worker.py --mode {sample,enumerate}`` CLI + one
``matrix16`` submit path; the flow math itself lives in
``glue.samplers.lsdflow.rgfn_extract`` / ``rgfn_enumerate`` (this is a thin wrapper over
:class:`RGFNAdapter`). Output is byte-identical in schema to ``scent_worker.py`` via the shared
stdlib :mod:`_artifacts` writer, so ``pick_hubs.py`` + ``run_campaign.py`` consume every generator
the same way.

RGFN has no promoted-fragment dynamic library, so ``compositions.json`` is empty and each enumerated
child's ``added_promoted`` is ``[]`` (the count-once cost model then falls back to
``min_num_reactions``; the naive ``reward`` child policy ignores it entirely).

Modes:
  * ``--mode sample``    -> records.csv, visit_counts.json, compositions.json, routes.json, meta.json
  * ``--mode enumerate`` -> enum_children.json, enumerated_records.csv, enum_per_hub.json, meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# _artifacts is stdlib-only; import it by path (the shared-worker convention) so this file matches
# the cross-env workers' import style even though it runs in the rgfn env.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifacts as A  # noqa: E402
import _routes  # noqa: E402

from validation.lsdflow.adapters.rgfn_adapter import RGFNAdapter  # noqa: E402


def _rec_to_dict(r) -> dict:
    """FlowRecord dataclass -> the REC_COLS row dict the writers/enum-builder expect."""
    return {c: getattr(r, c) for c in A.REC_COLS}


def _read_hubs(path: str):
    hubs = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            hubs.append((r["smiles"], int(r["depth"])))
    return hubs


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["sample", "enumerate"], default="sample")
    p.add_argument("--config", required=True, help="gin config for the trained RGFN")
    p.add_argument("--checkpoint", required=True, help="trained last_gfn.pt")
    p.add_argument("--reward-name", default="seh")
    p.add_argument("--model-name", default="rgfn")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--n-trajectories", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", default="/tmp/lsdflow_rgfn", help="user_root_dir for gin")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hubs-file", default="", help="enumerate mode: CSV with 'smiles,depth' rows")
    p.add_argument("--enum-max-children", type=int, default=2000, help="per-hub enumeration cap")
    return p.parse_args()


def main() -> None:
    t0 = time.perf_counter()
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SEED THE SAMPLER. This worker declared --seed and never applied it, alone among the four: scent,
    # rxnflow and fraggfn all call manual_seed/np.random.seed at setup. A worker that reports a seed it
    # never used is a trap, so it is applied here.
    #
    # BUT --seed ALONE IS NOT SUFFICIENT, AND BOTH HALVES OF THIS WERE MEASURED. Two runs at --seed 42,
    # same checkpoint, same trajectories, same device, gave 377 vs 387 routes sharing only 8 keys.
    # Categorical(...).sample() draws from torch's global RNG and manual_seed does seed it, so the
    # divergence was outside torch: per-process set/dict iteration order feeding action-space
    # construction, where an identical RNG draw over a differently-ordered action list picks a
    # different action.
    #
    # ADDING PYTHONHASHSEED=0 CLOSES IT COMPLETELY: --seed 42 twice on a debug GPU gave 730/730
    # BYTE-IDENTICAL routes (2026-08-21). So RGFN sampling IS reproducible, but only with BOTH knobs.
    # submit_cell.sh and submit_docking_cell.sh now export PYTHONHASHSEED=0 for exactly this reason --
    # if you invoke this worker by hand and want a regenerable sample, you must set it yourself.
    #
    # Note what this does and does not buy: FUTURE samples are regenerable. Samples already on disk
    # were taken before either knob was live, so they remain one-of-a-kind and recoverable only from
    # backup (scripts/backup_scratch_critical.sh TIER 2). Enumeration is exhaustive and unaffected.
    import random as _random

    import numpy as _np
    import torch as _torch

    _torch.manual_seed(args.seed)
    _np.random.seed(args.seed)
    _random.seed(args.seed)

    adapter = RGFNAdapter(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        reward_name=args.reward_name,
        device=args.device,
        batch_size=args.batch_size,
        run_dir=args.run_dir,
    )
    setup_s = time.perf_counter() - t0

    meta = {
        "setup_s": round(setup_s, 3),
        "reward_name": args.reward_name,
        "model": args.model_name,
        "higher_is_better": adapter.higher_is_better,
        "log_z": adapter.log_z,
        "strip_stereo": True,
        "frozen": False,  # RGFN has no dynamic library
        "n_promoted_fragments": 0,
    }

    if args.mode == "sample":
        _s0 = time.perf_counter()
        sample = adapter.sample_flow_records(args.n_trajectories)
        rows = [_rec_to_dict(r) for r in sample.records]
        A.write_records(out_dir / "records.csv", rows)
        # RGFN has no promoted fragments, but the count-once cost still needs each molecule's flat
        # build depth (num_reactions) so best-candidate is charged correctly (not the =1 fallback).
        comps = A.compositions_from_records(rows)
        json.dump(sample.visit_counts, open(out_dir / "visit_counts.json", "w"))
        json.dump(comps, open(out_dir / "compositions.json", "w"))
        routes = sample.routes or {}
        json.dump(routes, open(out_dir / "routes.json", "w"))
        # Contract check: RGFN is route-bearing, so an empty routes.json here is an unrecoverable
        # defect (the trajectory is gone once sampling ends) and must stop the run rather than be
        # discovered by the competitor arm months later. See _routes.py.
        # Denominator is distinct molecule NODES, not compositions: routes cover every molecule on
        # every trajectory (interior included), which is the visit_counts population. Using
        # compositions gave a nonsensical 178.7% on the first smoke.
        _routes.validate_sample_routes(
            "rgfn", out_dir, routes=routes, n_terminals=len(sample.visit_counts) or None
        )
        meta.update(
            {
                "n_trajectories": sample.n_trajectories,
                "n_records": len(rows),
                "n_compositions": len(comps),
                "n_routes": len(sample.routes or {}),
                "sample_s": round(time.perf_counter() - _s0, 3),
            }
        )
        A.write_json(out_dir / "meta.json", meta)
        print(
            f"[rgfn_worker] sample: {sample.n_trajectories} trajectories -> {len(rows)} records, "
            f"{len(sample.visit_counts)} nodes -> {out_dir}",
            flush=True,
        )
    else:  # enumerate
        if not args.hubs_file:
            raise SystemExit("[rgfn_worker] --mode enumerate requires --hubs-file")
        hubs = _read_hubs(args.hubs_file)
        _s0 = time.perf_counter()
        # Enumerate ONE hub at a time (not a single bulk call) so progress is observable: RGFN's
        # in-process enumerate is slow for high-fanout sEH hubs, and a multi-hour job needs a visible
        # per-hub rate to size walltime / detect stalls. Output is identical (each hub's records are
        # a disjoint group). enum_children.json is rewritten every 10 hubs so a timeout leaves a
        # usable partial (whatever hubs finished), not nothing.
        all_records, per_hub, enum_hubs = [], [], []
        # Measured per-hub compute time (Logs/039). RGFN differs from the other three workers: its
        # enumeration + reward + flow-extraction all happen inside ONE adapter call
        # (glue/samplers/lsdflow/rgfn_enumerate.enumerate_terminal_children), so the per-component
        # split is not observable from here. We record the true per-hub TOTAL under
        # ``unattributed_s`` and mark component_split="lumped" rather than fabricating a breakdown —
        # the head-to-head total stays exact, only the stacked breakdown is coarser for this cell.
        # Splitting it properly means timing inside glue/, which is shared code (coordinate first).
        hub_timings = []
        _use_cuda = str(getattr(adapter, "device", args.device)).startswith("cuda")
        for i, (smiles, depth) in enumerate(hubs):
            _h0 = time.perf_counter()
            # reaction_by_child is NOT optional bookkeeping: enum_children.json children[].reaction
            # is what lets a child's route be assembled as `hub prefix + this step`. RGFN omitted it
            # for months, and the failure is silent — SPARROW prices the hub instead of the child
            # and reports the empty library as Optimal. Every other worker passes it.
            rxn_by_child = {}
            recs_r, ph = adapter.enumerate_hub_children(
                [(smiles, depth)],
                max_children=args.enum_max_children,
                reaction_out=rxn_by_child,
            )
            _hub_s = time.perf_counter() - _h0
            rows = [_rec_to_dict(r) for r in recs_r]
            all_records.extend(rows)
            per_hub.extend(ph)
            hub_key = rows[0]["hub_key"] if rows else smiles
            enum_hubs.append(
                A.build_enum_hub(
                    hub_input=smiles,
                    hub_key=hub_key,
                    depth=depth,
                    recs=rows,
                    reaction_by_child=rxn_by_child,
                )
            )
            hub_timings.append(
                {
                    "hub_input": smiles,  # stereo-aware join key (EnumTimings._hub_id)
                    "hub_key": hub_key,
                    "depth": int(depth),
                    "n_children": len(rows),
                    "unattributed_s": round(_hub_s, 6),
                }
            )
            print(
                f"[rgfn_worker]   hub {i + 1}/{len(hubs)} depth={depth} -> {len(rows)} children "
                f"({time.perf_counter() - _s0:.0f}s elapsed)  {smiles[:44]}",
                flush=True,
            )
            if (i + 1) % 10 == 0:  # periodic partial flush (timeout-safe)
                A.write_enum_children(out_dir / "enum_children.json", enum_hubs)
                A.write_enum_timings(
                    out_dir / "enum_timings.json",
                    per_hub=hub_timings,
                    setup_s=setup_s,
                    device=str(args.device),
                    reward_name=args.reward_name,
                    model=args.model_name,
                    cuda_synchronized=_use_cuda,
                    component_split="lumped",
                )
        records = all_records
        A.write_enum_children(out_dir / "enum_children.json", enum_hubs)
        # Contract check on the FINAL write only (the 10-hub partial flushes above are legitimately
        # incomplete). A child with no reaction cannot be priced: its route stops at the hub, so
        # SPARROW prices the hub and returns an empty library as trivially optimal, silently. That is
        # the defect that cost six 24-hour re-enumerations.
        _routes.validate_enum_reactions("rgfn", out_dir)
        A.write_records(out_dir / "enumerated_records.csv", all_records)
        A.write_json(out_dir / "enum_per_hub.json", {"per_hub": per_hub})
        tmeta = A.write_enum_timings(
            out_dir / "enum_timings.json",
            per_hub=hub_timings,
            setup_s=setup_s,
            device=str(args.device),
            reward_name=args.reward_name,
            model=args.model_name,
            cuda_synchronized=_use_cuda,
            component_split="lumped",
        )
        print(
            f"[rgfn_worker] compute-time: setup {tmeta['setup_s']:.1f}s | "
            f"per-hub total {tmeta['totals_s'].get('unattributed_s', 0):.1f}s (lumped — no component "
            f"split available) over {len(hub_timings)} hubs -> enum_timings.json",
            flush=True,
        )
        meta.update(
            {
                "n_hubs": len(hubs),
                "n_enumerated_records": len(records),
                "n_enum_children": sum(len(h["children"]) for h in enum_hubs),
                "enumerate_s": round(time.perf_counter() - _s0, 3),
            }
        )
        A.write_json(out_dir / "meta.json", meta)
        print(
            f"[rgfn_worker] enumerate: {len(hubs)} hubs -> {len(records)} records + "
            f"enum_children.json -> {out_dir}",
            flush=True,
        )
    adapter.close()


if __name__ == "__main__":
    main()
