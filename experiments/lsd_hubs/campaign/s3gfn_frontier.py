#!/usr/bin/env python
"""T3.2 — put a flat (non-reaction) candidate pool on the LSD-Flow reactions-per-mode frontier.

S3-GFN (and any SMILES-only generator) emits a flat pool of scored molecules with NO shared-route
structure, so the ONLY selection strategy available to it is **best-candidate** — hub-batching needs
a reaction-grounded flow field, which a sequence model lacks. This driver runs best-candidate over the
pool across the diversity (τ) sweep and prices each ordering's snapshots with the SAME evaluator as
``sweep_campaign.py`` (from-scratch AiZynth→SPARROW for the headline), writing ``pareto.csv`` /
``fixed_modes.csv`` in the SAME column format so the S3-GFN curve overlays the reaction-GFN's on the
headline axes (docs/LSD_FLOW_BENCHMARK_PLAN.md T3.2 — the "MDP-necessary-for-library-economics" figure).

It REUSES the shared frontier pieces (``make_evaluator``, ``evaluate_ordering``, the read-time
slicers) from ``sweep_campaign.py`` + ``best_candidate`` from ``run_campaign.py`` — no strategy or
evaluator logic is duplicated. A flat pool needs no reaction structure: each molecule becomes a
``Candidate(num_reactions=1, promoted=(), parents=())`` (its count-once cost is a placeholder the
SPARROW pricer ignores — SPARROW re-prices from scratch), ``cost_table=None``, ``comps={}``.

Runs in the ``rgfn`` env (imports ``glue``→dgl via sweep_campaign). SMOKE: ``--evaluator count_once``
needs no route cache and validates the whole flat-pool path. HEADLINE: ``--evaluator sparrow
--route-source from_scratch --sparrow-cache <S3-GFN route cache from submit_s3gfn_routes.sh>``.

    python experiments/lsd_hubs/campaign/s3gfn_frontier.py \
        --pool-csv <run>/fixed_reward/candidates/candidates.csv --tag s3gfn_seh \
        --reward-threshold 7.0 --evaluator sparrow --route-source from_scratch \
        --sparrow-cache $SCRATCH/rgfn_runs/lsdflow_sparrow/s3gfn_seh/routecache_zinc_uspto.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPO_ROOT = HERE.parents[2]
for _p in (
    str(HERE),
    str(_REPO_ROOT),
):  # HERE: same-dir run_campaign/sweep_campaign; root: glue/validation
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_campaign as rc  # noqa: E402
import sweep_campaign as sw  # noqa: E402

from glue.samplers.lsdflow.campaign import Candidate  # noqa: E402


def load_pool(path: Path, higher_is_better: bool):
    """Flat pool -> best-of-duplicate Candidate list. Accepts the standard candidate dataset
    (``candidates.csv``: ``smiles``,``score``) or a ``pairs.csv`` (``smiles``,``score``)."""
    best: dict = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            smi = (row.get("smiles") or "").strip()
            raw = row.get("score", row.get("reward"))
            if not smi or raw in (None, ""):
                continue
            try:
                reward = float(raw)
            except (TypeError, ValueError):
                continue
            if smi not in best or (reward > best[smi]) == higher_is_better:
                best[smi] = reward
    return [
        Candidate(smiles=s, reward=r, num_reactions=1, promoted=(), parents=())
        for s, r in best.items()
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-csv", required=True, help="candidates.csv / pairs.csv (smiles, score)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=None, help="output root (default: <script dir>/results)")
    ap.add_argument("--reward-threshold", type=float, required=True)
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--budget-reactions", type=int, default=100)
    ap.add_argument("--budget-modes", type=int, default=300)
    ap.add_argument("--cutoff-min", type=float, default=0.30)
    ap.add_argument("--cutoff-max", type=float, default=0.90)
    ap.add_argument("--cutoff-step", type=float, default=0.05)
    ap.add_argument("--baseline-cutoff", type=float, default=0.50)
    ap.add_argument("--label", default="S3-GFN (best-candidate)")
    ap.add_argument("--system-label", default="sEH proxy")
    # ---- evaluator config (mirrors sweep_campaign so make_evaluator works verbatim) ----
    ap.add_argument("--evaluator", default="sparrow")
    ap.add_argument(
        "--snapshot-schedule", default="geometric", choices=["all", "every_k", "geometric"]
    )
    ap.add_argument("--snapshot-k", type=int, default=25)
    ap.add_argument("--snapshot-points", type=int, default=15)
    ap.add_argument("--route-source", default="from_scratch", choices=["from_scratch", "native"])
    ap.add_argument("--milp-objective", default="count", choices=["count", "count_cost"])
    ap.add_argument("--multiaiz-n-iters", type=int, default=5)  # --evaluator multiaiz
    ap.add_argument("--multiaiz-max-routes", type=int, default=0)
    ap.add_argument("--aizynth-env", default="aizynth")
    ap.add_argument("--sparrow-env", default="sparrow")
    ap.add_argument("--aizynth-config", default="data/models/aizynthfinder/config.yml")
    ap.add_argument("--aizynth-stock", default="zinc")
    ap.add_argument("--aizynth-expansion", default="uspto")
    ap.add_argument("--aizynth-filter", default="uspto")
    ap.add_argument("--aizynth-time-limit", type=int, default=60)
    ap.add_argument("--aizynth-nproc", type=int, default=8)
    ap.add_argument("--milp-max-seconds", type=int, default=600)
    ap.add_argument("--sparrow-cache", default=None)
    ap.add_argument("--sparrow-work-dir", default=None)
    ap.add_argument("--price-library", default=None)
    a = ap.parse_args()

    evaluator = sw.make_evaluator(a.evaluator, a)
    if a.evaluator != "count_once" and a.snapshot_schedule == "all":
        a.snapshot_schedule = "geometric"
        print(
            "[s3gfn-frontier] non-count_once evaluator -> snapshot schedule geometric", flush=True
        )

    cands = load_pool(Path(a.pool_csv), a.higher_is_better)
    cutoffs = sw._cutoff_grid(a.cutoff_min, a.cutoff_max, a.cutoff_step)
    out = (Path(a.out_dir) if a.out_dir else HERE / "results") / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[s3gfn-frontier] {len(cands)} unique candidates; evaluator={a.evaluator}; cutoffs={cutoffs}",
        flush=True,
    )

    curves = {}  # cutoff -> [(priced_modes, reactions), ...]
    for cut in cutoffs:
        # best-candidate ONLY (no flow field). cost_table=None + comps={} => count-once = 1/mode
        # placeholder (ignored by SPARROW). similarity=cut is the mode-definition τ.
        strat = rc.build_strategy(
            "best_candidate",
            cands,
            None,
            {},
            similarity=cut,
            target=a.tag,
            reward_threshold=a.reward_threshold,
            higher_is_better=a.higher_is_better,
        )
        result = strat.run(("modes", a.budget_modes))
        curves[cut] = sw.evaluate_ordering(
            result,
            evaluator,
            schedule=a.snapshot_schedule,
            k=a.snapshot_k,
            points=a.snapshot_points,
            provenance={
                "strategy": "s3gfn_best_candidate",
                "cutoff": cut,
                "target": a.tag,
                "evaluator": evaluator.name,
            },
        )
        print(
            f"  cutoff {cut:.2f}: {len(result.accepted)} selected | "
            f"modes@{a.budget_reactions}rxn={sw._modes_at_reactions(curves[cut], a.budget_reactions)} "
            f"rxn@{a.budget_modes}modes={sw._reactions_at_modes(curves[cut], a.budget_modes)}",
            flush=True,
        )

    # Same column format as sweep_campaign's pareto.csv / fixed_modes.csv, strategy='s3gfn' -> overlays.
    with open(out / "pareto.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "cutoff", f"modes_at_{a.budget_reactions}rxn"])
        for cut in cutoffs:
            w.writerow(["s3gfn", cut, sw._modes_at_reactions(curves[cut], a.budget_reactions)])
    with open(out / "fixed_modes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "cutoff", f"reactions_for_{a.budget_modes}modes"])
        for cut in cutoffs:
            w.writerow(["s3gfn", cut, sw._reactions_at_modes(curves[cut], a.budget_modes)])

    json.dump(
        {
            "tag": a.tag,
            "reward_threshold": a.reward_threshold,
            "evaluator": a.evaluator,
            "route_source": a.route_source,
            "budget_reactions": a.budget_reactions,
            "budget_modes": a.budget_modes,
            "baseline_cutoff": a.baseline_cutoff,
            "n_candidates": len(cands),
            "cutoffs": list(cutoffs),
            "curves": {f"{c:.2f}": curves[c] for c in cutoffs},
            "pareto_modes_at_R": {
                f"{c:.2f}": sw._modes_at_reactions(curves[c], a.budget_reactions) for c in cutoffs
            },
            "fixed_modes_reactions_at_M": {
                f"{c:.2f}": sw._reactions_at_modes(curves[c], a.budget_modes) for c in cutoffs
            },
        },
        open(out / "s3gfn_frontier_summary.json", "w"),
        indent=2,
    )
    print(f"[s3gfn-frontier] done -> {out}", flush=True)
    print(
        "  overlay onto the SCENT headline: combine this pareto.csv (strategy=s3gfn) with the "
        "reaction-GFN's results/<scent tag>/pareto.csv (hub_batching/best_candidate).",
        flush=True,
    )


if __name__ == "__main__":
    main()
