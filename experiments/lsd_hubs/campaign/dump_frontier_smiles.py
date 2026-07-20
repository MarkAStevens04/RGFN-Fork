#!/usr/bin/env python
"""Emit the union of accepted-mode SMILES across the whole frontier — the exact molecule set the
from-scratch SPARROW evaluator must route (T1.4/T2.1 prerequisite).

The from-scratch headline re-routes (AiZynth) every molecule the frontier ever prices. That is the
union of accepted modes over BOTH strategies and ALL diversity cutoffs (each cutoff yields a
different accepted set), for whichever hub child-policy / pre-select-K the headline uses. This dumps
that union to a ``.smi`` so a compute-node AiZynth batch (``submit_lsdflow_routes.sh``) can route it
ONCE into the persistent route cache; the fast per-snapshot MILPs then read the cache.

Pure CPU (re-runs the greedy selection over the cached enumeration; no GPU, no re-scoring), same
inputs as ``sweep_campaign.py``. Include multiple hub configs (--configs) so one route batch serves
naive + free-frag + pre-select-K headlines.

    python experiments/lsd_hubs/campaign/dump_frontier_smiles.py \
        --analysis-dir /scratch/.../lsdflow/scent_seh_70189 \
        --enum-children /scratch/.../campaign_enum_seh_70363/enum_children.json \
        --snapshot /scratch/.../fragments_4000.json --reward-threshold 7.0 \
        --configs best_candidate naive preselectk --prebuild-k 100 \
        --out /scratch/.../lsdflow_sparrow/scent_seh/frontier_smiles.smi
"""
import argparse
import json
from pathlib import Path

from run_campaign import _load_candidates, _load_enumerated_hubs, build_strategy

from glue.samplers.lsdflow.campaign import rank_fragments
from glue.samplers.lsdflow.child_select import make_child_policy
from validation.lsdflow.metrics.cost.dynamic_amortization import (
    load_cost_table_from_snapshot,
)


def _cutoffs(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--enum-children", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--reward-threshold", type=float, required=True)
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--budget-modes", type=int, default=300)
    ap.add_argument("--cutoff-min", type=float, default=0.30)
    ap.add_argument("--cutoff-max", type=float, default=0.90)
    ap.add_argument("--cutoff-step", type=float, default=0.05)
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["best_candidate", "naive", "preselectk"],
        choices=["best_candidate", "naive", "freefrag", "preselectk"],
        help="which selection configs to union (so one route batch serves every headline).",
    )
    ap.add_argument("--prebuild-k", type=int, default=100, help="K for the preselectk config")
    ap.add_argument("--rank-by", default="build_score")
    ap.add_argument("--out", required=True, help="output .smi (one SMILES per line)")
    a = ap.parse_args()

    adir = Path(a.analysis_dir)
    cands, comps = _load_candidates(adir, a.higher_is_better)
    enum = _load_enumerated_hubs(Path(a.enum_children), comps)
    cost_table = load_cost_table_from_snapshot(json.load(open(a.snapshot)))
    cutoffs = _cutoffs(a.cutoff_min, a.cutoff_max, a.cutoff_step)
    common = dict(
        target="dump", reward_threshold=a.reward_threshold, higher_is_better=a.higher_is_better
    )

    prebuilt = None
    if "preselectk" in a.configs and a.prebuild_k > 0:
        ranked = rank_fragments(
            enum,
            cost_table,
            a.reward_threshold,
            method=a.rank_by,
            higher_is_better=a.higher_is_better,
        )
        prebuilt = {f for f, _ in ranked[: a.prebuild_k]}

    # (config_name, pool, child_policy_name, prebuilt) — one selection family each.
    plan = []
    if "best_candidate" in a.configs:
        plan.append(("best_candidate", cands, None, None))
    if "naive" in a.configs:
        plan.append(("hub_batching", enum, "reward", None))
    if "freefrag" in a.configs:
        plan.append(("hub_batching", enum, "free_frag", None))
    if "preselectk" in a.configs:
        plan.append(("hub_batching", enum, "free_frag", prebuilt))

    union = set()
    per = {}
    for label, pool, cp_name, pre in plan:
        cp = make_child_policy(cp_name) if cp_name else None
        seen = set()
        for cut in cutoffs:
            strat = build_strategy(
                label,
                pool,
                cost_table,
                comps,
                similarity=cut,
                child_policy=cp,
                prebuilt_fragments=pre,
                **common,
            )
            seen |= {p.smiles for p in strat.run(budget=("modes", a.budget_modes)).accepted}
        key = cp_name or "best_candidate"
        per[f"{label}/{key}" + (f"+K{a.prebuild_k}" if pre else "")] = len(seen)
        union |= seen

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sorted(union)) + "\n")
    print(f"[dump] per-config accepted-mode union: {per}")
    print(f"[dump] wrote {len(union)} unique SMILES -> {out}")


if __name__ == "__main__":
    main()
