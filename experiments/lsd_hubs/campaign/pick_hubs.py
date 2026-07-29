#!/usr/bin/env python
"""Pick + rank the hubs to enumerate for the hub-batching campaign (Logs/028, Logs/053).

Maps a persisted SCENT/RGFN analysis DAG (``records.csv``) to a ranked hub set, and writes
``hubs.csv`` (``smiles,depth``, in **walk order**) for the per-env enumerator. Hub-batching walks
that file top-to-bottom, so the row order *is* the strategy.

Two orthogonal knobs (Logs/053 hub-ordering ablation); the defaults reproduce the original
Logs/028 recipe byte-for-byte:

``--pool`` — which hubs are eligible
  * ``topk_candidates`` (default) — parents of the top-``--top-k-candidates`` candidates by reward.
    A reward pre-filter: the flow ranking only ever sees a few hundred hubs.
  * ``all`` — **every** hub observed in ``records.csv``. The flow estimate then has to do the
    selecting on its own, which is what makes "does the flow signal buy anything?" answerable.

``--order`` — the walk order over the eligible hubs
  * ``flow_desc`` (default) — highest flow first.
  * ``flow_asc`` — lowest flow first (the reverse control).
  * ``random`` — uniform shuffle, ``--seed`` (the no-signal control).
  * ``candidate_reward`` — walk candidates best-reward-first and take their parent hubs in that
    order (the "a great molecule must have a great hub" control). The candidate walk defines the
    selection too, so ``--pool`` is ignored for this order.

``--restrict-to hubs.csv`` keeps only hubs already in an existing set, so an order can be applied
to a *fixed* hub set — isolating ordering from selection at zero enumeration cost.

**The flow estimate.** ``F_hat(h;x) = logR + logP_B - logP_F(move) - logP_F(stop)`` (the §2
log-flow, straight from the record's log-terms — no model needed), aggregated per hub as the **max**
over its observed children. Each child is an independent estimate of the same ``F(h)``; 85% of hubs
have exactly one, so max/median/mean coincide for them, and max keeps the legacy ranking semantics.

Note the two pools aggregate slightly differently, by design:
``topk_candidates`` assigns each candidate to the parent of its own highest-flow record and scores a
hub by the best candidate landing on it (legacy behaviour, kept bit-exact); ``all`` scores a hub by
the max over *every* record naming it, so a child's estimate still counts toward hub ``h`` even when
that child had a better estimate for some other parent.

Writes ``hubs.csv`` (the enumerator contract: ``smiles,depth`` only), plus two sidecars —
``hub_scores.csv`` (rank, score, #estimates, provenance) and ``pick_hubs_timing.json`` (Logs/039).

Pure stdlib (CSV only), so it runs anywhere before the GPU enumeration step.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

POOLS = ("topk_candidates", "all")
ORDERS = ("flow_desc", "flow_asc", "random", "candidate_reward")


def _load_records(path: str):
    """One pass over ``records.csv`` -> the three per-entity views the orders need.

    Returns ``(best_reward, best_flow, hub_flow_all, hub_n_est)``:
      * ``best_reward``  child_key -> best observed reward
      * ``best_flow``    child_key -> (log_F, hub_stereo, hub_depth) of its highest-flow record
      * ``hub_flow_all`` (hub_stereo, depth) -> max log_F over EVERY record naming that hub
      * ``hub_n_est``    (hub_stereo, depth) -> how many finite per-child estimates it has
    """
    best_reward: dict = {}
    best_flow: dict = {}
    hub_flow_all: dict = {}
    hub_n_est: dict = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            child = r["child_key"]
            reward = float(r["reward"])
            if child not in best_reward or reward > best_reward[child]:
                best_reward[child] = reward
            log_f = (
                float(r["log_reward"])
                + float(r["log_pb_move"])
                - float(r["log_pf_move"])
                - float(r["log_pf_stop"])
            )
            if not math.isfinite(log_f):
                continue
            key = (r.get("hub_stereo_key") or r["hub_key"], int(r["hub_depth"]))
            if key not in hub_flow_all or log_f > hub_flow_all[key]:
                hub_flow_all[key] = log_f
            hub_n_est[key] = hub_n_est.get(key, 0) + 1
            if child not in best_flow or log_f > best_flow[child][0]:
                best_flow[child] = (log_f, key[0], key[1])
    return best_reward, best_flow, hub_flow_all, hub_n_est


def _read_hub_keys(path: str) -> set:
    """The ``(smiles, depth)`` keys of an existing hubs.csv (for ``--restrict-to``)."""
    with open(path) as fh:
        return {(r["smiles"], int(r["depth"])) for r in csv.DictReader(fh)}


def _candidate_order(best_reward, best_flow, higher_is_better):
    """Distinct parent hubs in best-candidate order: walk candidates by reward, take each one's
    highest-flow parent, keep first occurrences. Returns ``[(hub_key, candidate_rank, reward)]``."""
    order = sorted(best_reward, key=lambda c: best_reward[c], reverse=higher_is_better)
    out, seen = [], set()
    for rank, child in enumerate(order):
        if child not in best_flow:
            continue
        _, hub_stereo, hub_depth = best_flow[child]
        key = (hub_stereo, hub_depth)
        if key in seen:
            continue
        seen.add(key)
        out.append((key, rank, best_reward[child]))
    return out


def main() -> None:
    _t0 = time.perf_counter()  # Stage-2 hub-pick wall-clock (Logs/039 compute-time accounting)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True, help="a SCENT analysis records.csv")
    ap.add_argument("--out", required=True, help="hubs.csv to write (smiles,depth in walk order)")
    ap.add_argument(
        "--top-k-candidates", type=int, default=100, help="top candidates whose hubs we consider"
    )
    ap.add_argument("--n-hubs", type=int, default=50, help="max hubs to keep (enumeration budget)")
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument(
        "--pool",
        default="topk_candidates",
        choices=POOLS,
        help="eligible hubs: parents of the top-K candidates (default, legacy) | every observed hub",
    )
    ap.add_argument(
        "--order",
        default="flow_desc",
        choices=ORDERS,
        help="walk order (Logs/053): flow_desc (default) | flow_asc | random | candidate_reward",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --order random")
    ap.add_argument(
        "--restrict-to",
        default="",
        help="an existing hubs.csv; keep only its hubs, so --order is applied to a FIXED set "
        "(isolates ordering from selection — no new enumeration needed)",
    )
    a = ap.parse_args()

    best_reward, best_flow, hub_flow_all, hub_n_est = _load_records(a.records)

    # ---- 1. eligible pool -> {hub_key: flow score} -------------------------------------------
    if a.pool == "all":
        hub_flow = dict(hub_flow_all)
    else:  # topk_candidates (legacy): the best candidate landing on each hub defines its score
        top = sorted(best_reward, key=lambda c: best_reward[c], reverse=a.higher_is_better)[
            : a.top_k_candidates
        ]
        hub_flow = {}
        for child in top:
            if child not in best_flow:
                continue
            log_f, hub_stereo, hub_depth = best_flow[child]
            key = (hub_stereo, hub_depth)
            if key not in hub_flow or log_f > hub_flow[key]:
                hub_flow[key] = log_f
    pool_size = len(hub_flow)

    # ---- 2. optional restriction to an existing hub set --------------------------------------
    if a.restrict_to:
        keep = _read_hub_keys(a.restrict_to)
        missing = keep - set(hub_flow)
        if a.pool == "all" and missing:  # every hub in the file should exist in the full pool
            print(f"[pick_hubs] WARNING --restrict-to has {len(missing)} hubs absent from the pool")
        hub_flow = {k: v for k, v in hub_flow.items() if k in keep}

    # ---- 3. walk order ------------------------------------------------------------------------
    cand_meta: dict = {}
    if a.order == "candidate_reward":
        cand = _candidate_order(best_reward, best_flow, a.higher_is_better)
        if a.restrict_to:
            keep = _read_hub_keys(a.restrict_to)
            cand = [c for c in cand if c[0] in keep]
        ranked = [k for k, _, _ in cand][: a.n_hubs]
        cand_meta = {k: (r, rew) for k, r, rew in cand}
        # A restricted set may contain hubs no candidate points at (they entered the set by flow);
        # append them by flow so the arm still walks exactly the set it was given.
        if a.restrict_to and len(ranked) < a.n_hubs:
            tail = sorted(
                (k for k in hub_flow if k not in set(ranked)),
                key=lambda k: hub_flow[k],
                reverse=True,
            )
            ranked += tail[: a.n_hubs - len(ranked)]
    elif a.order == "random":
        keys = sorted(hub_flow)  # sort first so the shuffle is reproducible from the seed alone
        random.Random(a.seed).shuffle(keys)
        ranked = keys[: a.n_hubs]
    else:  # flow_desc | flow_asc
        ranked = sorted(hub_flow, key=lambda k: hub_flow[k], reverse=(a.order == "flow_desc"))[
            : a.n_hubs
        ]

    # ---- 4. write the enumerator contract + provenance sidecar --------------------------------
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "depth"])
        for hub_stereo, hub_depth in ranked:
            w.writerow([hub_stereo, hub_depth])
    scores_path = Path(a.out).parent / "hub_scores.csv"
    with open(scores_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["rank", "smiles", "depth", "log_flow", "n_estimates", "cand_rank", "cand_reward"]
        )
        for i, key in enumerate(ranked):
            cr, crew = cand_meta.get(key, ("", ""))
            w.writerow([i, key[0], key[1], hub_flow.get(key, ""), hub_n_est.get(key, 0), cr, crew])

    elapsed = time.perf_counter() - _t0
    # Stage-2 timing sidecar (Logs/039): the drivers add this to hub-batching's compute-time (it is
    # work best-candidate never does). Written next to hubs.csv so the enum dir carries it.
    timing_path = Path(a.out).parent / "pick_hubs_timing.json"
    json.dump(
        {
            "hub_pick_s": round(elapsed, 3),
            "n_hubs": len(ranked),
            "n_candidates": len(best_reward),
            "top_k_candidates": a.top_k_candidates,
            "pool": a.pool,
            "pool_size": pool_size,
            "order": a.order,
            "seed": a.seed if a.order == "random" else None,
            "restrict_to": a.restrict_to or None,
        },
        open(timing_path, "w"),
        indent=2,
    )
    src = "all observed hubs" if a.pool == "all" else f"top-{a.top_k_candidates} candidates"
    print(
        f"[pick_hubs] {len(best_reward)} candidates -> {src} -> {pool_size} eligible hubs -> "
        f"wrote {len(ranked)} in {a.order} order to {a.out} "
        f"(hub-pick {elapsed:.2f}s -> {timing_path.name}, scores -> {scores_path.name})"
    )


if __name__ == "__main__":
    main()
