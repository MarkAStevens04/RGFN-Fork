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

**Sidecar names follow the ``--out`` basename, and a mismatched overwrite is refused.** Both
sidecars used to be written to FIXED names in ``--out``'s parent, so two invocations differing only
in basename shared one of each and the second silently replaced the first — measured, running
``--pool all`` then ``--pool topk_candidates`` into one directory left two ``hubs_*.csv`` but a
single ranking, of which only **8 of 20** hubs belonged to the run you were about to read. That is
the standing "``hub_scores.csv`` is often stale — never join against it" warning, and it is not
staleness: the next run overwrites it. The canonical ``hubs.csv`` keeps the canonical sidecar names
so every existing reader and v1 artifact is unaffected; any other basename gets prefixed ones. On
top of that, writing over a sidecar whose recorded pool/order/depth-band/``n_hubs`` differs from the
current run aborts with the diff (``--force`` to override); an identical re-run is idempotent and
never trips it.

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


def _depth_hist(keys) -> dict:
    """``{depth: count}`` over hub keys ``(stereo_smiles, depth)``, sorted by depth.

    Reported at every stage because depth is what separates the two things hub-batching can be
    doing: amortising a BUILT intermediate (depth >= 1) versus diversifying off a BOUGHT block
    (depth 0). A hub set whose depth mix is unknown cannot be read as evidence for either.
    """
    h: dict = {}
    for k in keys:
        h[k[1]] = h.get(k[1], 0) + 1
    return dict(sorted(h.items()))


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
        "--force", action="store_true",
        help="overwrite a sidecar that records a DIFFERENT run. Without this, pick_hubs refuses "
        "rather than replacing another run's ranking in place -- the failure that made "
        "'hub_scores.csv is often stale' a standing warning",
    )
    # ---- depth band (added 2026-09-07) --------------------------------------------------------
    # DEFAULT UNFILTERED, DELIBERATELY. This exists so the sensitivity arm is expressible, not to
    # change what hub-batching does; flipping the default would silently redefine every published
    # hub set.
    #
    # WHY IT IS NEEDED NOW. `--pool all` (the v2 standard) removes the reward pre-filter that used
    # to keep bought building blocks out by accident: depth-0 hubs go from 2 to 11 of the top 200,
    # and from 2 to 8 of the first 40 -- the part actually walked. That is legitimate, and priced
    # correctly (a chemist BUYS a depth-0 hub; shallow_couplings(depth=0, promoted=()) == 0), but
    # depth-0 catalogue picking is the metric's own named degenerate optimum, so the comparison
    # "is the win late-stage diversification or catalogue picking?" has to be runnable.
    # `--min-hub-depth 1` is that arm.
    #
    # IT ALSO CLOSES A DISAGREEMENT INSIDE THE CODEBASE. The AL acquisition path
    # (glue/samplers/lsdflow/acquisition.py) has defaulted to min_hub_depth=1 / max_hub_depth=3 all
    # along -- depth 0 is "huge fan-out, zero amortization, not the build-once-diversify signal"
    # (Logs/025), and depth 4 sits at the reaction cap where P_B is unrecoverable. This path had no
    # depth control at all, so two implementations of "what is a hub" disagreed.
    ap.add_argument(
        "--min-hub-depth", type=int, default=None,
        help="drop hubs shallower than this. UNSET = no floor (default). `1` excludes depth-0 "
        "bought building blocks -- the labelled sensitivity arm against --pool all",
    )
    ap.add_argument(
        "--max-hub-depth", type=int, default=None,
        help="drop hubs deeper than this. UNSET = no ceiling (default). `3` matches the AL "
        "acquisition path, which skips depth-4 hubs at the reaction cap (P_B unrecoverable)",
    )
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

    # ---- 1b. depth band ------------------------------------------------------------------------
    # Applied to ELIGIBILITY, before ordering and before --restrict-to, because the band is a
    # statement about what counts as a hub -- not a post-hoc trim of a chosen walk. The dropped
    # count and the surviving histogram are printed and recorded, so a hub set can never be read
    # without knowing which band produced it.
    depth_of = lambda k: k[1]  # noqa: E731 -- hub key is (stereo_smiles, depth)
    pool_depths_before = _depth_hist(hub_flow)
    n_dropped_depth = 0
    if a.min_hub_depth is not None or a.max_hub_depth is not None:
        lo = a.min_hub_depth if a.min_hub_depth is not None else -(1 << 30)
        hi = a.max_hub_depth if a.max_hub_depth is not None else (1 << 30)
        if lo > hi:
            raise SystemExit(f"--min-hub-depth {lo} exceeds --max-hub-depth {hi}")
        kept = {k: v for k, v in hub_flow.items() if lo <= depth_of(k) <= hi}
        n_dropped_depth = len(hub_flow) - len(kept)
        if not kept:
            raise SystemExit(
                f"depth band [{a.min_hub_depth}, {a.max_hub_depth}] removed every one of "
                f"{len(hub_flow)} eligible hubs (depths present: {pool_depths_before}). "
                f"Refusing to write an empty hubs.csv -- downstream an empty hub set prices the "
                f"EMPTY library as trivially optimal at zero cost."
            )
        hub_flow = kept
        print(
            f"[pick_hubs] depth band [{a.min_hub_depth}, {a.max_hub_depth}]: "
            f"{n_dropped_depth} of {pool_size} eligible hubs dropped, {len(hub_flow)} remain "
            f"(before {pool_depths_before} -> after {_depth_hist(hub_flow)})"
        )

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
    # ---- 4a. sidecar naming + collision guard --------------------------------------------------
    # BOTH SIDECARS USED FIXED NAMES IN THE PARENT OF --out, so two invocations differing only in
    # the --out basename shared one hub_scores.csv and one pick_hubs_timing.json, and the second
    # silently clobbered the first. Measured: running `--pool all` then `--pool topk_candidates`
    # into one directory leaves two hubs_*.csv but ONE ranking, and only 8 of its 20 hubs belong to
    # the run whose file you are about to read.
    #
    # THIS IS THE STANDING "hub_scores.csv IS OFTEN STALE" WART, AND IT IS NOT STALENESS. The
    # project warning -- "seed 44's rank-#1 hub is not even in the enum; never join against it" --
    # describes this overwrite. The file does not drift over time; the next pick_hubs run replaces
    # it. So the caveat can become a fix.
    #
    # The canonical basename keeps the canonical sidecar names, so every existing reader and every
    # v1 artifact is untouched. Only a caller who chose a DIFFERENT basename -- which is exactly
    # when a collision is possible -- gets disambiguated names.
    stem = Path(a.out).stem
    _sfx = "" if stem == "hubs" else f"{stem}."
    scores_path = Path(a.out).parent / f"{_sfx}hub_scores.csv"
    timing_path = Path(a.out).parent / f"{_sfx}pick_hubs_timing.json"

    # Identity of THIS run. Two runs sharing a sidecar path must agree on all of it, or one of them
    # is about to read a ranking it did not produce.
    identity = {
        "pool": a.pool,
        "order": a.order,
        "n_hubs": a.n_hubs,
        "top_k_candidates": a.top_k_candidates if a.pool == "topk_candidates" else None,
        "min_hub_depth": a.min_hub_depth,
        "max_hub_depth": a.max_hub_depth,
        "seed": a.seed if a.order == "random" else None,
        "restrict_to": a.restrict_to or None,
    }
    if timing_path.exists() and not a.force:
        try:
            prev = json.load(open(timing_path))
        except Exception:
            prev = {}
        differing = {k: (prev.get(k), v) for k, v in identity.items() if prev.get(k) != v}
        if differing:
            raise SystemExit(
                f"[pick_hubs] REFUSING to overwrite {timing_path.name}: it records a DIFFERENT run.\n"
                + "".join(f"    {k}: {was!r} -> {now!r}\n" for k, (was, now) in differing.items())
                + f"  Its sibling {scores_path.name} is that run's ranking, and overwriting it is how\n"
                  f"  a hub set gets analysed as though it came from another pool -- which has already\n"
                  f"  happened. Write to a different --out basename, or pass --force if you really\n"
                  f"  mean to replace it."
            )
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
    json.dump(
        {
            **identity,  # the fields the collision guard compares, first so they are easy to read
            "hub_pick_s": round(elapsed, 3),
            "n_hubs_written": len(ranked),
            "n_candidates": len(best_reward),
            "pool_size": pool_size,
            "hubs_csv": Path(a.out).name,  # which hubs.csv this sidecar describes
            # The depth band and its effect. `walked_depth_hist` is the headline: it is the depth
            # mix of the hubs this file actually hands the enumerator, and the depth-0 share of it
            # is what says whether a library was diversified off built intermediates or off bought
            # catalogue blocks.
            # (min_hub_depth / max_hub_depth are already in `identity` above -- the guard compares
            # them, so they must live there.)
            "n_dropped_by_depth": n_dropped_depth,
            "pool_depth_hist": pool_depths_before,
            "walked_depth_hist": _depth_hist(ranked),
            "walked_depth0_frac": round(
                sum(1 for k in ranked if k[1] == 0) / max(len(ranked), 1), 4
            ),
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
