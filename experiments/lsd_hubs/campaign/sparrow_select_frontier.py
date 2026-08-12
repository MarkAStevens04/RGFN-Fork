#!/usr/bin/env python
"""The competitor pipeline's frontier: SPARROW SELECTS a library from a pre-routed candidate pool.

THE EXPERIMENT THIS IMPLEMENTS. A chemist wanting 100 diverse compounds likely to bind would (1)
generate candidates with a generative model, (2) retro-plan them ALL with a batch planner, (3) hand
routes + predicted rewards to SPARROW and let it choose the subset worth making. This script is
stages 2->3: it consumes a cached MultiAiZ routes artifact and sweeps SPARROW's reaction budget,
emitting the (modes, reactions) frontier that LSD-Flow's curve is read against at a common mode count.

WHY THIS IS NOT sweep_campaign.py. That driver applies a reward-ranked, tau-diverse greedy selection
FIRST and uses SPARROW only to PRICE the result — SPARROW never decides anything, and the baseline is
handed a diversity-aware selector it would not actually have. Here SPARROW does its own job
(`--select`, maximize selected reward s.t. `--max-rxns`), which is both the honest baseline and the
one that exposes its blind spot: SPARROW optimizes reward and cost, never diversity. We therefore
MEASURE how many of its chosen targets are distinct modes rather than assuming.

PLAN-ONCE / PRICE-MANY. Route discovery (~14 h, `submit_multiaiz_discover.sh`) is done once per pool
and cached; so is the merged reaction network, which does not depend on the budget. Only the MILP
re-runs per budget point (~0.01-1 s), so a whole frontier costs seconds. Changing SPARROW parameters
never re-triggers discovery.

MODE COUNTING uses the project's canonical metric (validation/lsdflow/metrics/diversity: Morgan r=3
/2048, greedy sphere exclusion, best-reward-first) so these numbers are directly comparable to every
reactions-per-mode figure in the benchmark.

Runs in the ``rgfn`` env (imports validation.lsdflow); shells to the ``sparrow`` env for the MILP.

Usage:
  python experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
      --routes  $SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools/s3gfn_seh_N500/multiaiz_routes.json \
      --pool    $SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools/s3gfn_seh_N500/pool_scores.csv \
      --out-dir experiments/lsd_hubs/campaign/results/s3gfn_seh_select \
      --tag s3gfn_seh_multiaiz_select
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from validation.lsdflow.eval.network import (  # noqa: E402
    build_network,
    canonical,
    expand_route_with_recipes,
)
from validation.lsdflow.eval.route_recovery import env_python  # noqa: E402
from validation.lsdflow.metrics.diversity import (  # noqa: E402
    count_modes,
    mode_representatives,
)

SPARROW_WORKER = "validation/lsdflow/adapters/workers/sparrow_worker.py"


def load_pool(path: Path, gate: float, top_n: int = 0):
    """[(smiles, reward)] above the gate, DEDUPLICATED by SMILES, best-reward-first.

    Dedup is load-bearing, not hygiene. A GFlowNet samples the same molecule many times, so a
    ``records.csv`` row is a *sampling event*, not a candidate: the top-500 rows of the sEH run are
    only 115 distinct molecules. Taking rows verbatim would (a) hand SPARROW the same target dozens
    of times and (b) silently shrink "the 500 best candidates" to ~115, making the arm look far
    weaker than it is. Keep each molecule once, at its best observed reward.

    ``top_n`` caps AFTER dedup, so "top-N candidates" means N distinct molecules.
    """
    best = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            smi = r.get("smiles") or r.get("SMILES") or r.get("child_key")
            raw = r.get("score", r.get("reward"))
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if smi and val > gate and (smi not in best or val > best[smi]):
                best[smi] = val
    rows = sorted(best.items(), key=lambda t: -t[1])
    return rows[:top_n] if top_n else rows


def load_enum_pool(enum_path: Path, hub_routes_path: Path, gate: float, top_n: int = 0):
    """BC-Enum-SB's pool: the ENUMERATED CHILDREN of a hub set, with their routes ASSEMBLED.

    Why this cannot reuse the `native` path. An enumerated child is a molecule the generator never
    sampled — it is one reaction past a hub — so it has no entry in ``routes.json`` (measured: 29 of
    500 children present, versus 64 of 64 hub_keys). Its route has to be built the way
    ``reconcile_t15._native_route_for_mode`` builds it for hub-batching:

        child route = the hub's route steps  +  the child's own diversifying reaction

    Getting this wrong silently produces a pool of ~6% of the children, which would make the arm look
    absurdly weak for a reason that has nothing to do with the science.

    Returns ``([(smiles, reward)], {smiles: shallow_route})`` — still SHALLOW, so the caller must
    recipe-expand exactly as for native routes.
    """
    data = json.loads(Path(enum_path).read_text())
    hub_routes = json.loads(Path(hub_routes_path).read_text())
    best, routes, n_missing_hub = {}, {}, 0
    for hub in data.get("hubs", []):
        hk = hub.get("hub_key")
        hr = hub_routes.get(hk) if hk else None
        if hr is None:
            n_missing_hub += 1
            continue  # a hub with no route cannot price its children; count it, never silently drop
        prefix = list(hr.get("steps") or [])
        for c in hub.get("children", []):
            smi, rew = c.get("smiles"), c.get("reward")
            if not smi or rew is None or float(rew) <= gate:
                continue
            rew = float(rew)
            if smi in best and rew <= best[smi]:
                continue
            steps = prefix + list(c.get("reaction") or [])
            best[smi] = rew
            routes[smi] = {"product_smiles": smi, "num_reactions": len(steps), "steps": steps}
    if n_missing_hub:
        print(
            f"[enum] WARNING {n_missing_hub} hub(s) had no route in hub-routes — their children skipped"
        )
    rows = sorted(best.items(), key=lambda t: -t[1])
    if top_n:
        rows = rows[:top_n]
    keep = {s for s, _ in rows}
    return rows, {s: r for s, r in routes.items() if s in keep}


def _run_sparrow(repo, sparrow_env, tree, targets, out, work, max_seconds, extra=()):
    """Invoke the SPARROW worker; returns its result dict (or None on failure)."""
    cmd = [
        *env_python(sparrow_env), str(repo / SPARROW_WORKER),
        "--tree", str(tree), "--targets", str(targets), "--out", str(out),
        "--max-seconds", str(max_seconds), "--work-dir", str(work), *extra,
    ]  # fmt: skip
    proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    if proc.returncode != 0 or not Path(out).exists():
        print(f"    FAILED rc={proc.returncode} {proc.stderr.strip()[-200:]}")
        return None
    return json.loads(Path(out).read_text())


def _greedy_frontier(a, out_dir, pool, routed, entries):
    """The baseline's STRONGEST configuration: a diversity-aware greedy selection picks the modes,
    SPARROW only prices them (pricing mode, every selected mode must be synthesized).

    Contrast with the default path, where SPARROW both selects and prices. Comparing only against
    that would let a reviewer object that SPARROW was judged on diversity, which it does not
    optimize. This arm removes the objection: the competitor gets a selector that IS diversity-aware
    AND the convergent planner's routes.
    """
    rewards = {s: r for s, r in pool}
    routed_list = [s for s in (s for s, _ in pool) if s in routed]  # keep best-reward-first order
    reps = mode_representatives(
        routed_list,
        [rewards[s] for s in routed_list],
        higher_is_better=True,
        reward_threshold=a.gate,
        similarity_threshold=a.cutoff,
    )
    ordered_modes = [routed_list[i] for i in reps]
    print(f"[greedy] {len(ordered_modes)} modes available from {len(routed_list)} routed molecules")

    by_smiles = {}
    for e in entries:
        by_smiles.setdefault(e["smiles"], []).append(e)

    rows = []
    for m in [int(x) for x in a.mode_points.split(",") if x.strip()]:
        if m > len(ordered_modes):
            print(f"  modes={m:<5} SKIP (only {len(ordered_modes)} available)")
            continue
        sel = ordered_modes[:m]
        sub = [e for s in sel for e in by_smiles[s]]
        snap = out_dir / f"network_m{m}"
        net = build_network(sub, strip_stereo=a.strip_stereo)
        tree, targets = net.write(snap)
        res = _run_sparrow(
            REPO, a.sparrow_env, tree, targets, out_dir / f"milp_m{m}.json",
            out_dir / f"sparrow_run_m{m}", a.max_seconds,
        )  # fmt: skip
        if res is None:
            continue
        rx = res.get("total_reactions")
        rows.append(
            {
                "n_modes": m,
                "used_rxns": rx,
                "n_selected": m,  # greedy selects modes directly: every pick IS a distinct mode
                "mode_rate": 1.0,
                "rxn_per_mode": round(rx / m, 4) if rx else None,
                "milp_status": res.get("milp_status"),
                "n_targets_priced": res.get("n_targets_selected"),
            }
        )
        print(
            f"  modes={m:<5} rxns={rx:<5} rxn/mode={rows[-1]['rxn_per_mode']} "
            f"({res.get('milp_status')}, priced {res.get('n_targets_selected')}/{m})"
        )

    with open(out_dir / "greedy_frontier.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["n_modes"])
        w.writeheader()
        w.writerows(rows)
    json.dump(
        {"tag": a.tag, "selection": "greedy", "gate": a.gate, "cutoff": a.cutoff,
         "n_modes_available": len(ordered_modes), "n_routed": len(routed), "rows": rows},
        open(out_dir / "greedy_frontier_summary.json", "w"), indent=2,
    )  # fmt: skip
    print(f"[greedy] wrote {out_dir}/greedy_frontier.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--routes",
        required=True,
        help="multiaiz_routes.json, or the run's routes.json when --route-source native",
    )
    ap.add_argument(
        "--route-source",
        default="multiaiz",
        choices=["multiaiz", "native", "enum"],
        help="multiaiz = the competitor's planned routes (MANY candidate recipes per molecule). "
        "native = the reaction-GFN's own by-construction routes (exactly ONE per molecule, and "
        "SHALLOW, so they must be recipe-expanded — --snapshot becomes required).",
    )
    ap.add_argument(
        "--snapshot",
        default="",
        help="fragments_<N>.json — REQUIRED with --route-source native: supplies the "
        "`smiles_to_route` recipes that turn attached promoted fragments into BUILT ones. Without "
        "it SPARROW buys what count-once builds (the Logs/049 DRD2 62.7% failure).",
    )
    ap.add_argument(
        "--hub-routes",
        default="",
        help="the sample's routes.json — REQUIRED with --route-source enum: supplies each hub's "
        "route prefix. Enumerated children are one reaction past a hub and are absent from "
        "routes.json themselves, so their routes must be assembled rather than looked up.",
    )
    ap.add_argument(
        "--pool",
        required=False,
        default="",
        help="pool CSV (smiles|child_key + score|reward) for the SAME pool",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="cap the pool at the N highest-reward DISTINCT molecules (0 = all above gate). This is "
        "the 'as large as SPARROW can solve' knob.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="sparrow_select")
    ap.add_argument("--gate", type=float, default=7.0)
    ap.add_argument("--cutoff", type=float, default=0.5, help="tau for mode counting")
    ap.add_argument(
        "--budgets", default="", help="comma-separated reaction budgets (default: auto)"
    )
    ap.add_argument(
        "--selection",
        default="sparrow",
        choices=["sparrow", "greedy"],
        help="WHO CHOOSES THE LIBRARY. `sparrow` (default) = SPARROW's own MILP picks the subset "
        "under a reaction budget — the pipeline a chemist actually runs, and the one with no "
        "diversity term. `greedy` = reward-ranked tau-diverse greedy selection picks the modes and "
        "SPARROW only PRICES them — a diversity-AWARE competitor, i.e. the baseline's strongest "
        "configuration. `greedy` exists so the headline cannot be accused of comparing against "
        "SPARROW at the one job it was not built for; it is the conservative comparator.",
    )
    ap.add_argument(
        "--mode-points",
        default="25,50,75,100,125,150",
        help="--selection greedy: mode counts to price (the x-axis is modes, not budget)",
    )
    ap.add_argument("--sparrow-env", default="sparrow")
    ap.add_argument("--max-seconds", type=int, default=600)
    ap.add_argument("--strip-stereo", type=lambda s: s.lower() != "false", default=True)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.route_source in ("native", "enum") and not a.snapshot:
        raise SystemExit(
            f"[select] --route-source {a.route_source} requires --snapshot (recipe expansion)"
        )
    if a.route_source == "enum" and not a.hub_routes:
        raise SystemExit(
            "[select] --route-source enum requires --hub-routes (the sample routes.json)"
        )
    if a.route_source != "enum" and not a.pool:
        raise SystemExit(
            "[select] --pool is required unless --route-source enum (which derives it)"
        )
    if a.route_source == "enum":
        pool, raw = load_enum_pool(Path(a.routes), Path(a.hub_routes), a.gate, a.top_n)
        print(
            f"[enum] {len(pool)} distinct children above gate>{a.gate}, routes assembled "
            "(hub prefix + the child's own reaction)"
        )
    else:
        raw = json.loads(Path(a.routes).read_text())
        pool = load_pool(Path(a.pool), a.gate, a.top_n)
    if not pool:
        raise SystemExit(f"[select] no pool molecules above gate {a.gate} in {a.pool}")

    # TWO ROUTE SOURCES, different shapes and different correctness requirements.
    #
    #   multiaiz : {canonical_smiles: [route, ...]}  — MANY candidate recipes per molecule (~16 for
    #              the N=500 pool). Already bottom out at purchasable stock, so they price directly.
    #
    #   native   : {smiles: {seed, num_reactions, steps}} — exactly ONE recipe per molecule, and
    #              *shallow*: a promoted dynamic-library fragment is ATTACHED in one step, never
    #              synthesized. Feeding those to SPARROW unexpanded is the bug that made the DRD2
    #              reconcile read 62.7% (Logs/049): SPARROW would BUY the promoted fragments while
    #              count-once BUILDS them, so the two price different assumptions. Every native
    #              route must therefore be recipe-expanded first (`expand_route_with_recipes`),
    #              which needs the run's fragment snapshot — hence --snapshot is required here.
    is_native = a.route_source in ("native", "enum")
    recipes, promoted = {}, set()
    if is_native:
        if not a.snapshot:
            raise SystemExit(
                "[select] --route-source native requires --snapshot (recipe expansion)"
            )
        snap = json.loads(Path(a.snapshot).read_text())
        recipes = snap.get("smiles_to_route") or {}
        promoted = set(snap.get("chosen_smiles", []))
        cov = (sum(1 for s in promoted if s in recipes) / len(promoted)) if promoted else 1.0
        if promoted and cov < 0.95:
            raise SystemExit(
                f"[select] ABORT: snapshot has recipes for only {cov:.1%} of its {len(promoted)} "
                "promoted fragments. Native routes cannot be expanded, so SPARROW would treat them "
                "as bought while count-once builds them — the pricing would be meaningless."
            )
        print(
            f"[select] {a.route_source} routes | recipe coverage {cov:.1%} of {len(promoted)} promoted frags"
        )

    # entries: one per (target, candidate route). Duplicate targets are intentional — build_network
    # unions their reactions and the MILP picks the max-sharing combination.
    entries, routed = [], set()
    for smi, rew in pool:
        if is_native:
            rt = raw.get(smi) or (raw.get(canonical(smi, a.strip_stereo)) if smi else None)
            rts = [expand_route_with_recipes(rt, recipes, promoted)] if rt else None
        else:
            c = canonical(smi, a.strip_stereo)
            rts = raw.get(c) if c else None
        if not rts:
            continue
        routed.add(smi)
        for r_ in rts:
            entries.append({"smiles": smi, "reward": rew, "route": r_})
    if not entries:
        raise SystemExit("[select] no pool molecule has a route in the artifact — wrong pairing?")

    print(
        f"[select] pool={len(pool)} above gate>{a.gate} | routed={len(routed)} "
        f"({len(routed)/len(pool):.1%}) | route entries={len(entries)}"
    )

    if a.selection == "greedy":
        return _greedy_frontier(a, out_dir, pool, routed, entries)

    # The network is budget-independent: build + write ONCE, then only the MILP re-runs per budget.
    t0 = time.perf_counter()
    net = build_network(entries, strip_stereo=a.strip_stereo)
    snap = out_dir / "network"
    tree, targets = net.write(snap)
    build_s = time.perf_counter() - t0
    print(f"[select] network built in {build_s:.1f}s -> {tree}")

    if a.budgets:
        budgets = [int(x) for x in a.budgets.split(",") if x.strip()]
    else:
        # geometric-ish ladder; the ceiling is "every routed target", which pricing mode would give
        budgets = [10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 600, 800]

    rows = []
    for R in budgets:
        milp_out = out_dir / f"milp_R{R}.json"
        cmd = [
            *env_python(a.sparrow_env),
            str(REPO / SPARROW_WORKER),
            "--tree", str(tree),
            "--targets", str(targets),
            "--out", str(milp_out),
            "--objective", "reward",
            "--select",
            "--max-rxns", str(R),
            "--max-seconds", str(a.max_seconds),
            "--work-dir", str(out_dir / f"sparrow_run_R{R}"),
        ]  # fmt: skip
        t1 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        solve_s = time.perf_counter() - t1
        if proc.returncode != 0 or not milp_out.exists():
            print(f"  R={R:<5d} FAILED rc={proc.returncode} {proc.stderr.strip()[-160:]}")
            continue
        res = json.loads(milp_out.read_text())
        sel = res.get("selected_target_smiles") or []
        rewards = {s: r for s, r in pool}
        # THE measurement: SPARROW optimized reward+cost with no diversity term, so how many of the
        # molecules it chose are actually distinct modes is an empirical property of its output.
        n_modes = (
            count_modes(
                sel,
                [rewards.get(s, 0.0) for s in sel],
                higher_is_better=True,
                reward_threshold=a.gate,
                similarity_threshold=a.cutoff,
            )
            if sel
            else 0
        )
        n_sel = len(sel)
        rows.append(
            {
                "budget_rxns": R,
                "used_rxns": res.get("total_reactions"),
                "n_selected": n_sel,
                "n_modes": n_modes,
                "mode_rate": round(n_modes / n_sel, 4) if n_sel else None,
                "rxn_per_selected": round(res["total_reactions"] / n_sel, 4)
                if n_sel and res.get("total_reactions") is not None
                else None,
                "rxn_per_mode": round(res["total_reactions"] / n_modes, 4)
                if n_modes and res.get("total_reactions") is not None
                else None,
                "total_reward": res.get("selected_reward_total"),
                "milp_status": res.get("milp_status"),
                "solve_s": round(solve_s, 2),
            }
        )
        print(
            f"  R={R:<5d} used={res.get('total_reactions'):<5} selected={n_sel:<4} "
            f"modes={n_modes:<4} mode_rate={rows[-1]['mode_rate']} "
            f"rxn/mode={rows[-1]['rxn_per_mode']} ({res.get('milp_status')})"
        )

    with open(out_dir / "select_frontier.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["budget_rxns"])
        w.writeheader()
        w.writerows(rows)
    json.dump(
        {
            "tag": a.tag,
            "routes_artifact": str(a.routes),
            "pool": str(a.pool),
            "gate": a.gate,
            "cutoff": a.cutoff,
            "n_pool_above_gate": len(pool),
            "n_routed": len(routed),
            "routed_fraction": round(len(routed) / len(pool), 4),
            "n_route_entries": len(entries),
            "network_build_s": round(build_s, 2),
            "rows": rows,
        },
        open(out_dir / "select_frontier_summary.json", "w"),
        indent=2,
    )
    print(f"[select] wrote {out_dir}/select_frontier.csv + select_frontier_summary.json")


if __name__ == "__main__":
    main()
