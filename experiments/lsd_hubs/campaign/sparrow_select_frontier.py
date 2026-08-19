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
    mean_pairwise_similarity,
    mode_assignments,
    mode_representatives,
    unique_scaffolds,
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


def _price_kept_set(a, out_dir, entries_by_smiles, kept, tag):
    """Reactions actually needed to make ``kept`` — a PRICING solve over just those molecules.

    Why re-price instead of reusing the selection budget R: SPARROW spent R reactions producing a set
    we then prune. You do not pay for compounds you discard, so charging R for the survivors would
    overstate the competitor's cost — and the pruned duplicates take their unique final steps with
    them, leaving only the shared prefixes. R is what it was ALLOWED to spend; this is what the
    deliverable actually costs.

    The same prune is applied to hub-batching, where it is a NO-OP (its picks are distinct by
    construction), so the procedure is symmetric rather than a concession to one side.
    """
    sub = [e for smi in kept for e in entries_by_smiles.get(smi, [])]
    if not sub:
        return None
    snap = out_dir / f"price_{tag}"
    net = build_network(sub, strip_stereo=a.strip_stereo)
    tree, targets = net.write(snap)
    res = _run_sparrow(
        REPO, a.sparrow_env, tree, targets, snap / "milp.json",
        snap / "sparrow_run", a.max_seconds,
    )  # fmt: skip
    return res.get("total_reactions") if res else None


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
        choices=["multiaiz", "native", "enum", "external"],
        help="multiaiz = the competitor's planned routes (MANY candidate recipes per molecule). "
        "native = the reaction-GFN's own by-construction routes (exactly ONE per molecule, and "
        "SHALLOW, so they must be recipe-expanded — --snapshot becomes required). "
        "external = a REACTION-AWARE competitor's own by-construction routes (SynFormer), read from "
        "a routes.jsonl. Like `native` these come free with the molecule, but UNLIKE `native` they "
        "are already deep: every leaf is a purchasable catalogue block, so there is nothing to "
        "recipe-expand and --snapshot must NOT be passed.",
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
        "--lambda-div",
        default="",
        help="comma-separated values of SPARROW's NATIVE diversity weight to sweep (e.g. "
        "'0,0.1,0.5,1,2'). Their Fig 3B traces a Pareto front as this moves, so sweeping and "
        "reporting SPARROW at its BEST value per metric gives it its strongest showing and "
        "forecloses 'you picked a bad lambda'. Requires tau-mode clusters, built automatically.",
    )
    ap.add_argument(
        "--min-clusters",
        type=int,
        default=None,
        help="DIVERSITY-CONSTRAINED SB ([fromer2025diversity] sec 2.2, constraint form): require "
        "every selection to represent at least this many of OUR tau-modes. Turns SB from a "
        "diversity-BLIND competitor into a diversity-AWARE one competing on our own metric. Budgets "
        "below the feasible minimum come back infeasible -- that boundary IS the readout (the "
        "reactions needed to deliver K distinct molecules), directly comparable to hub-batching.",
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
    elif a.route_source == "external":
        # A reaction-aware competitor emits routes.jsonl (one JSON object per line, our route
        # schema). Fold it into the SAME {canonical_smiles: [route, ...]} shape the multiaiz
        # artifact uses, so everything downstream — build_network, the MILP, the pricing — is
        # literally the same code path. One route per molecule here, versus ~16 for multiaiz;
        # that asymmetry favours the multiaiz side and is reported, not corrected.
        raw = {}
        with open(a.routes) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                smi = rec.get("smiles") or rec.get("product_smiles")
                c = canonical(smi, a.strip_stereo) if smi else None
                if c:
                    raw.setdefault(c, []).append(rec)
        print(f"[external] {len(raw)} molecules carry a by-construction route")
        pool = load_pool(Path(a.pool), a.gate, a.top_n)
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

    # Clusters are pool-dependent but budget-independent: compute ONCE, like the network. Using our
    # own tau-modes (not SPARROW's default Butina/count-Morgan) is deliberate -- it makes the arm
    # optimize the exact quantity the benchmark reports, so a win or loss is on our metric rather
    # than on a proxy for it. Legitimate because the paper defines clusters as an arbitrary
    # caller-supplied partition.
    lambdas = [float(x) for x in a.lambda_div.split(",") if x.strip()] or [0.0]
    cluster_args = []
    if a.min_clusters is not None or any(l > 0 for l in lambdas):
        t2 = time.perf_counter()
        rewards_all = {s_: r_ for s_, r_ in pool}
        pool_smis = [s_ for s_, _ in pool if s_ in routed]
        groups = mode_assignments(
            pool_smis,
            [rewards_all[s_] for s_ in pool_smis],
            higher_is_better=True,
            reward_threshold=a.gate,
            similarity_threshold=a.cutoff,
        )
        cl_path = out_dir / "clusters.json"
        cl_path.write_text(json.dumps(groups))
        cluster_args = ["--clusters", str(cl_path)]
        if a.min_clusters is not None:
            cluster_args += ["--min-clusters", str(a.min_clusters)]
        member2cl = {m: c for c, ms in groups.items() for m in ms}
        print(
            f"[select] tau-mode clusters: {len(groups)} over {len(pool_smis)} routed molecules "
            f"({time.perf_counter()-t2:.1f}s)"
            + (
                f" | requiring >= {a.min_clusters} represented"
                if a.min_clusters is not None
                else f" | lambda_div sweep {lambdas}"
            )
        )
        if a.min_clusters is not None and a.min_clusters > len(groups):
            raise SystemExit(
                f"[select] ABORT: --min-clusters {a.min_clusters} exceeds the {len(groups)} clusters "
                "the pool contains — every budget would be infeasible and the sweep would report "
                "that as a solver limit rather than a pool limit."
            )

    if a.budgets:
        budgets = [int(x) for x in a.budgets.split(",") if x.strip()]
    else:
        # geometric-ish ladder; the ceiling is "every routed target", which pricing mode would give
        budgets = [10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 600, 800]

    if not cluster_args:
        member2cl = {}
    entries_by_smiles = {}
    for e in entries:
        entries_by_smiles.setdefault(e["smiles"], []).append(e)

    rows = []
    rewards = {s_: r_ for s_, r_ in pool}
    for LD in lambdas:
        for R in budgets:
            tagLR = f"L{LD:g}_R{R}"
            milp_out = out_dir / f"milp_{tagLR}.json"
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
            "--work-dir", str(out_dir / f"sparrow_run_{tagLR}"),
            *cluster_args,
        ]  # fmt: skip
            if LD > 0:
                cmd += ["--lambda-div", str(LD)]
            t1 = time.perf_counter()
            proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            solve_s = time.perf_counter() - t1
            if proc.returncode != 0 or not milp_out.exists():
                print(
                    f"  L={LD:<5g} R={R:<5d} FAILED rc={proc.returncode} {proc.stderr.strip()[-150:]}"
                )
                continue
            res = json.loads(milp_out.read_text())
            sel = res.get("selected_target_smiles") or []
            n_sel = len(sel)

            # (1) SPARROW's OWN metric -- how many tau-mode families its picks touch. We expect to lose
            #     here: it is optimizing precisely this. Reported anyway.
            touched = len({member2cl.get(x) for x in sel if x in member2cl}) if member2cl else None

            # (2) OUR metric -- prune to a genuinely-distinct subset (each survivor dissimilar to EVERY
            #     other survivor, not merely in a different family), then RE-PRICE the survivors. You do
            #     not pay for what you discard. The identical prune applied to hub-batching is a no-op,
            #     because its picks are distinct by construction -- so this is symmetric, not a handout.
            keep_idx = mode_representatives(
            sel, [rewards.get(x, 0.0) for x in sel],
            higher_is_better=True, reward_threshold=a.gate, similarity_threshold=a.cutoff,
        )  # fmt: skip
            kept = [sel[i] for i in keep_idx]
            cost_kept = (
                _price_kept_set(a, out_dir, entries_by_smiles, kept, tagLR) if kept else None
            )
            mps = mean_pairwise_similarity(sel) if len(sel) > 1 else None

            # FIVE METRICS, on BOTH the unpruned selection and the pruned survivors. No single number
            # captures "diversity": cluster coverage is what SPARROW optimizes but is satisfiable by
            # near-duplicates sitting between adjacent clusters (and one molecule may cover many
            # clusters at once); sphere exclusion demands dissimilarity from EVERYTHING kept but is
            # order-dependent; scaffold counts ignore decoration entirely. All three are proxies for
            # information gain, which is what we would optimize if we could -- out of scope here, and
            # flagged as a limitation. Reporting several, on both output sets, lets the reader see
            # where they disagree rather than trusting our choice of one.
            scaf_all = unique_scaffolds(sel) if sel else 0
            scaf_kept = unique_scaffolds(kept) if kept else 0
            mps_kept = mean_pairwise_similarity(kept) if len(kept) > 1 else None

            rows.append(
                {
                    "lambda_div": LD,
                    "budget_rxns": R,
                    "used_rxns": res.get("total_reactions"),
                    "n_selected": n_sel,
                    "clusters_touched": touched,  # SPARROW's metric
                    "n_modes_kept": len(kept),  # ours, after pruning
                    "cost_kept_rxns": cost_kept,  # reactions for the pruned set ONLY
                    "rxn_per_mode_kept": round(cost_kept / len(kept), 4)
                    if cost_kept and kept
                    else None,
                    "mode_rate": round(len(kept) / n_sel, 4) if n_sel else None,
                    "rxn_per_selected": round(res["total_reactions"] / n_sel, 4)
                    if n_sel and res.get("total_reactions") is not None
                    else None,
                    "mean_pairwise_sim": round(mps, 4) if mps is not None else None,
                    "mean_pairwise_sim_kept": round(mps_kept, 4) if mps_kept is not None else None,
                    "scaffolds_all": scaf_all,  # unpruned: Bemis-Murcko on everything selected
                    "scaffolds_kept": scaf_kept,  # pruned: on the survivors only
                    "scaffold_rate": round(scaf_all / n_sel, 4) if n_sel else None,
                    "total_reward": res.get("selected_reward_total"),
                    "milp_status": res.get("milp_status"),
                    # Carried so a truncated search can never be read as a proven optimum downstream.
                    # A capped row's numbers are a LOWER BOUND on what SPARROW could achieve.
                    "time_capped": res.get("time_capped"),
                    "solve_s": round(solve_s, 2),
                }
            )
            r_ = rows[-1]
            print(
                f"  L={LD:<5g} R={R:<5d} sel={n_sel:<4} touched={str(touched):<5} "
                f"kept={len(kept):<4} cost(kept)={str(cost_kept):<5} "
                f"rxn/mode={r_['rxn_per_mode_kept']} meanSim={r_['mean_pairwise_sim']}"
                + ("  [TIME-CAPPED: lower bound]" if res.get("time_capped") else "")
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
    n_capped = sum(1 for r in rows if r.get("time_capped"))
    if n_capped:
        print(
            f"[select] WARNING {n_capped}/{len(rows)} solves hit the {a.max_seconds}s MILP limit. "
            "Those rows are LOWER BOUNDS on SPARROW's performance, not proven optima — re-run "
            "them with a larger --max-seconds before quoting any of them."
        )
    print(f"[select] wrote {out_dir}/select_frontier.csv + select_frontier_summary.json")


if __name__ == "__main__":
    main()
