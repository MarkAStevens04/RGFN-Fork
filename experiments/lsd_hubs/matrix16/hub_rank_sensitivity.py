#!/usr/bin/env python
"""How much does the hub ranking depend on the backward policy P_B?

``pick_hubs.py`` ranks candidate hubs by the single-candidate flow estimate
``log F_hat = logR + logP_B(h|x) - logP_F(x|h) - logP_F(stop|x)``. For **RxnFlow** that ``logP_B``
is a *retro heuristic* (``do_parameterize_p_b=False``), quantized to ~6 discrete values
(log(1/n_parents), 38% exactly 0) — so a fair question is whether the hub set/order we enumerate is
an artifact of that muddy term. For **SCENT** ``P_B`` is the model's *trained* (cost-tilted) policy
recovered from the guidance sidecar, so it is the natural control: P_B *should* matter there.

Compares three rankings over the same sampled DAG (``records.csv``), reproducing pick_hubs' exact
recipe (top-K candidates by reward -> their parent hubs -> rank hubs):

  A  ``flow``        full ``log F_hat``  (what pick_hubs actually uses)
  B  ``flow_no_pb``  ``log F_hat`` with the ``logP_B`` term DROPPED   (ablates the muddy term)
  C  ``reward``      rank each hub by its best child's raw reward     (P_B/P_F-free baseline)

Reports, for the top-N hub sets: Jaccard/overlap, Spearman rank correlation, and how many children
even *have* a choice of parent hub (if a child has one observed parent, P_B cannot change the
assignment — only the ordering).

Pure stdlib + CPU; login-node safe. Usage:
    python experiments/lsd_hubs/matrix16/hub_rank_sensitivity.py rxnflow_seh [scent_seh ...]
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import get_cell  # noqa: E402

TOP_K_CANDIDATES = 1000  # pick_hubs default (submit_cell.sh TOPK)
N_HUBS = 200  # pick_hubs default (submit_cell.sh N_HUBS)


def _spearman(a: list, b: list) -> float | None:
    """Spearman rho between two equal-length score lists (average ranks for ties)."""
    n = len(a)
    if n < 3:
        return None

    def ranks(x):
        order = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else None


def analyze(cell_tag: str) -> dict:
    gen, tgt = cell_tag.rsplit("_", 1)
    cell = get_cell(gen, tgt)
    recs_path = cell.sample_dir / "records.csv"
    if not recs_path.exists():
        print(f"[{cell_tag}] no records.csv at {recs_path} — skipping")
        return {}
    hib = cell.target.higher_is_better

    # Per child: best reward, and per (child, scoring) the best-scoring parent hub.
    best_reward: dict = {}
    parents_of: dict = defaultdict(set)  # child -> {(hub_stereo, depth)}
    best: dict = {"flow": {}, "flow_no_pb": {}}  # child -> (score, hub_key)
    n_rows = 0
    with open(recs_path) as fh:
        for r in csv.DictReader(fh):
            n_rows += 1
            child = r["child_key"]
            rew = float(r["reward"])
            if child not in best_reward or (rew > best_reward[child]) == hib:
                best_reward[child] = rew
            hub = (r.get("hub_stereo_key") or r["hub_key"], int(r["hub_depth"]))
            parents_of[child].add(hub)
            lr, pb = float(r["log_reward"]), float(r["log_pb_move"])
            pf, ps = float(r["log_pf_move"]), float(r["log_pf_stop"])
            for name, val in (("flow", lr + pb - pf - ps), ("flow_no_pb", lr - pf - ps)):
                if math.isfinite(val) and (child not in best[name] or val > best[name][child][0]):
                    best[name][child] = (val, hub)

    # pick_hubs step 2: the top-K candidates by REWARD (this filter is P_B-independent).
    top = sorted(best_reward, key=lambda c: best_reward[c], reverse=hib)[:TOP_K_CANDIDATES]

    # Build each ranking's hub -> score map over those candidates.
    hub_scores: dict = {"flow": {}, "flow_no_pb": {}, "reward": {}}
    for child in top:
        for name in ("flow", "flow_no_pb"):
            if child in best[name]:
                sc, hub = best[name][child]
                if hub not in hub_scores[name] or sc > hub_scores[name][hub]:
                    hub_scores[name][hub] = sc
        # reward ranking: a hub scores as its best observed child reward (any parent it was seen as)
        for hub in parents_of[child]:
            rew = best_reward[child]
            cur = hub_scores["reward"].get(hub)
            if cur is None or (rew > cur) == hib:
                hub_scores["reward"][hub] = rew

    ranked = {
        name: sorted(sc, key=lambda h: sc[h], reverse=True)[:N_HUBS]
        for name, sc in hub_scores.items()
    }

    multi = sum(1 for c in top if len(parents_of[c]) > 1)
    out = {
        "cell": cell_tag,
        "n_records": n_rows,
        "n_candidates": len(best_reward),
        "top_k": len(top),
        "children_with_multiple_parent_hubs": f"{multi}/{len(top)}",
        "distinct_hubs_available": {k: len(v) for k, v in hub_scores.items()},
    }

    A = ranked["flow"]
    for other in ("flow_no_pb", "reward"):
        B = ranked[other]
        inter = set(A) & set(B)
        union = set(A) | set(B)
        common = list(inter)
        rho = _spearman(
            [hub_scores["flow"][h] for h in common], [hub_scores[other][h] for h in common]
        )
        out[f"vs_{other}"] = {
            "overlap_top200": f"{len(inter)}/{min(len(A), len(B))}",
            "overlap_pct": round(100 * len(inter) / max(1, min(len(A), len(B))), 1),
            "jaccard": round(len(inter) / max(1, len(union)), 3),
            "spearman_on_common": None if rho is None else round(rho, 3),
        }
    return out


def main() -> None:
    cells = sys.argv[1:] or ["rxnflow_seh", "scent_seh"]
    results = []
    for c in cells:
        r = analyze(c)
        if r:
            results.append(r)
            print(json.dumps(r, indent=2))
    if results:
        outp = HERE / "results" / "gate_sweep" / "hub_rank_sensitivity.json"
        os.makedirs(outp.parent, exist_ok=True)
        outp.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {outp}")
        print(
            f"\n{'cell':14}{'P_B-ablated overlap':>22}{'rho':>8}{'reward-rank overlap':>22}{'rho':>8}"
        )
        for r in results:
            a, b = r["vs_flow_no_pb"], r["vs_reward"]
            print(
                f"{r['cell']:14}{a['overlap_top200']:>16} ({a['overlap_pct']:>4}%)"
                f"{str(a['spearman_on_common']):>8}"
                f"{b['overlap_top200']:>16} ({b['overlap_pct']:>4}%){str(b['spearman_on_common']):>8}"
            )


if __name__ == "__main__":
    main()
