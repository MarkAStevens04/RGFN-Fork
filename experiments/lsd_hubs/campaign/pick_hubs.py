#!/usr/bin/env python
"""Pick + rank the hubs to enumerate for the hub-batching campaign (Logs/028).

The first hub strategy (modular — swap this file to try others): take the top-K candidates by
reward from a persisted SCENT DAG (``records.csv``), map each to its parent hub, and rank hubs by
the candidate's naive single-candidate flow estimate ``F_hat(h;x) = logR + logP_B - logP_F(move)
- logP_F(stop)`` (the §2 log-flow, straight from the record's log-terms — no model needed). Writes
``hubs.csv`` (``smiles,depth``, in rank order) for the scent-env enumerator.

Pure stdlib (CSV only), so it runs anywhere before the GPU enumeration step.
"""
import argparse
import csv
import math


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True, help="a SCENT analysis records.csv")
    ap.add_argument("--out", required=True, help="hubs.csv to write (smiles,depth in rank order)")
    ap.add_argument(
        "--top-k-candidates", type=int, default=100, help="top candidates whose hubs we consider"
    )
    ap.add_argument("--n-hubs", type=int, default=50, help="max hubs to keep (enumeration budget)")
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    a = ap.parse_args()

    # Per terminal candidate: its best reward, and the parent hub of its highest-flow record.
    best_reward: dict = {}
    best_flow: dict = {}  # child_key -> (log_F, hub_stereo, hub_depth)
    with open(a.records) as fh:
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
            if child not in best_flow or log_f > best_flow[child][0]:
                best_flow[child] = (
                    log_f,
                    r.get("hub_stereo_key") or r["hub_key"],
                    int(r["hub_depth"]),
                )

    # Top-K candidates by reward -> their hubs -> rank hubs by the candidate's single-candidate flow.
    top = sorted(best_reward, key=lambda c: best_reward[c], reverse=a.higher_is_better)[
        : a.top_k_candidates
    ]
    hub_flow: dict = {}  # (hub_stereo, depth) -> best log_F among top candidates pointing to it
    for child in top:
        if child not in best_flow:
            continue
        log_f, hub_stereo, hub_depth = best_flow[child]
        key = (hub_stereo, hub_depth)
        if key not in hub_flow or log_f > hub_flow[key]:
            hub_flow[key] = log_f

    ranked = sorted(hub_flow, key=lambda k: hub_flow[k], reverse=True)[: a.n_hubs]
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "depth"])
        for hub_stereo, hub_depth in ranked:
            w.writerow([hub_stereo, hub_depth])
    print(
        f"[pick_hubs] {len(best_reward)} candidates -> top-{a.top_k_candidates} -> "
        f"{len(hub_flow)} distinct hubs -> wrote {len(ranked)} (ranked by single-candidate F_hat) to {a.out}"
    )


if __name__ == "__main__":
    main()
