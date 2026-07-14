#!/usr/bin/env python
"""Per-hub statistics table for a campaign enumeration (Logs/029).

Reads a scent-worker enumeration dir (``enumerated_records.csv`` = per-child flow terms;
``hubs.csv`` = the flow-rank order) and writes ``results/<tag>/hub_stats.csv`` — one row per
enumerated hub: rank, depth, #children, #effective (finite-flow) children, ``U(h) = Var_i[log
F_hat]`` (§2 flow-matching residual), mean ``F_hat``, and the child-reward summary (max / median /
#hits). Pure stdlib (CSV + statistics), so it runs anywhere after the GPU enumeration.

    python experiments/lsd_hubs/campaign/hub_stats.py \
        --enum-dir /scratch/.../campaign_enum_seh_70295 --reward-threshold 7.0 --tag scent_seh
"""
import argparse
import csv
import math
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _fhat(r) -> float | None:
    v = (
        float(r["log_reward"])
        + float(r["log_pb_move"])
        - float(r["log_pf_move"])
        - float(r["log_pf_stop"])
    )
    return v if math.isfinite(v) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--enum-dir", required=True, help="campaign_enum_* dir (enumerated_records.csv + hubs.csv)"
    )
    ap.add_argument("--reward-threshold", type=float, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    enum = Path(a.enum_dir)

    rank = OrderedDict()  # hub smiles -> flow rank (1-based) from hubs.csv
    with open(enum / "hubs.csv") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            rank[row["smiles"]] = i + 1

    recs = defaultdict(lambda: {"depth": None, "flows": [], "rewards": []})
    with open(enum / "enumerated_records.csv") as fh:
        for r in csv.DictReader(fh):
            hub = r.get("hub_stereo_key") or r["hub_key"]
            d = recs[hub]
            d["depth"] = int(r["hub_depth"])
            v = _fhat(r)
            if v is not None:
                d["flows"].append(v)
            d["rewards"].append(float(r["reward"]))

    rows = []
    for hub, d in recs.items():
        f = d["flows"]
        rows.append(
            dict(
                rank=rank.get(hub, 999),
                hub=hub,
                depth=d["depth"],
                n_children=len(d["rewards"]),
                n_effective=len(f),
                U_h=round(statistics.pvariance(f), 3) if len(f) > 1 else "",
                mean_Fhat=round(statistics.mean(f), 3) if f else "",
                max_reward=round(max(d["rewards"]), 3),
                median_reward=round(statistics.median(d["rewards"]), 3),
                n_hits=sum(1 for x in d["rewards"] if x >= a.reward_threshold),
            )
        )
    rows.sort(key=lambda r: r["rank"])

    outdir = HERE / "results" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "hub_stats.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[hub_stats] wrote {len(rows)} hubs -> {out}")


if __name__ == "__main__":
    main()
