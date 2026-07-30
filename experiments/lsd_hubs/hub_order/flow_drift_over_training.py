#!/usr/bin/env python
"""Does the flow field move during training? (Logs/053 addendum)

The worry: if the model favours some intermediates early in training and abandons them later, then
any hub signal derived from *training-time* behaviour is stale with respect to the trained model.

This measures the drift directly, with no extra compute. SCENT's `paths.csv` logs every molecule
sampled during training together with its full synthesis path, so the pre-terminal state (the hub)
and the training iteration are both recoverable. Rank all hubs by the final model's flow estimate
(from the post-training sample), then ask, per slice of training: how many of the hubs visited then
still appear in the final sample, and where do they sit in the final ranking?

Two readouts, because each has a different weakness:
  * ``% in final sample`` — survivorship. Confounded by sample size: a hub can be absent from 30k
    post-training trajectories by chance rather than because its flow collapsed.
  * ``median final rank`` — conditions on being present, so it is immune to that, but only describes
    the survivors.
They agree here, which is what makes the drift credible from either alone.

**This does NOT indict the LSD-Flow analyses.** Their candidate pool and flow terms both come from
sampling the FINAL checkpoint, so no training-time staleness enters. It matters for anything that
consumes training-time signal — most concretely SCENT's dynamic library, whose promoted fragments are
chosen by a training-time utility and then frozen into every downstream analysis.

**Limitation.** Only `last_gfn.pt` / `best_gfn.pt` are saved, so we cannot compute flow *under an
early model* — the definitive test ("what was F(h) at iteration 1000, versus now?") needs periodic
checkpoints. Future training runs should keep them; it is cheap and unlocks this measurement.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/hub_order/flow_drift_over_training.py

Needs RDKit (canonicalisation to the stereo-stripped join key). ~4 min for 320k paths on a login
node. Writes ``results/comparison/flow_drift.{csv,png}``.
"""
from __future__ import annotations

import argparse
import ast
import csv
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
csv.field_size_limit(10**9)

DEFAULT_RUN = "/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06"
DEFAULT_ANALYSIS = "/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189"


def final_flow_ranking(records: Path):
    """{stereo-stripped hub_key: rank} under the final model, best = 0. Stereo-stripped because
    `paths.csv` states are canonicalised without stereo for the join."""
    best = defaultdict(lambda: -1e18)
    with open(records) as fh:
        for r in csv.DictReader(fh):
            lf = (
                float(r["log_reward"])
                + float(r["log_pb_move"])
                - float(r["log_pf_move"])
                - float(r["log_pf_stop"])
            )
            if math.isfinite(lf):
                best[r["hub_key"]] = max(best[r["hub_key"]], lf)
    order = sorted(best, key=lambda k: best[k], reverse=True)
    return {k: i for i, k in enumerate(order)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=DEFAULT_RUN, help="training run dir holding paths.csv")
    ap.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS)
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument(
        "--per-step",
        type=int,
        default=64,
        help="molecules logged per training iteration — `iteration` in paths.csv counts MOLECULES, "
        "not steps (320,001 rows = 5,000 iters x 64)",
    )
    ap.add_argument("--steps-per-bin", type=int, default=250)
    a = ap.parse_args()

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    rank = final_flow_ranking(Path(a.analysis_dir) / "records.csv")
    n = len(rank)

    cache: dict = {}

    def canon(s: str) -> str:
        v = cache.get(s)
        if v is None:
            m = Chem.MolFromSmiles(s)
            v = Chem.MolToSmiles(m, isomericSmiles=False) if m else ""
            cache[s] = v
        return v

    width = a.per_step * a.steps_per_bin
    bins: dict = defaultdict(Counter)
    with open(Path(a.run_dir) / "paths.csv") as fh:
        for r in csv.DictReader(fh):
            states = [x for x in ast.literal_eval(r["path"]) if isinstance(x, str)]
            if len(states) < 2:  # a one-state path has no hub
                continue
            bins[int(r["iteration"]) // width][canon(states[-2])] += 1

    rows = []
    for b in sorted(bins):
        hubs = bins[b]
        ranks = [rank[h] for h in hubs if h in rank]
        if not ranks:
            continue
        rows.append(
            {
                "iter_start": b * a.steps_per_bin,
                "iter_end": (b + 1) * a.steps_per_bin,
                "distinct_hubs": len(hubs),
                "pct_in_final_sample": round(100 * len(ranks) / len(hubs), 1),
                "median_final_rank": int(st.median(ranks)),
                "pct_in_top_5pct": round(
                    100 * sum(1 for x in ranks if x < n * 0.05) / len(ranks), 1
                ),
            }
        )

    out = Path(a.results) / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "flow_drift.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in rows[0]}
    print(f"final-model flow ranking: {n} hubs | {len(rows)} bins of {a.steps_per_bin} iters\n")
    print("  ".join(k.ljust(widths[k]) for k in rows[0]))
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in rows[0]))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[drift] plot skipped ({exc})")
        return
    x = [(r["iter_start"] + r["iter_end"]) / 2 for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(
        x,
        [r["pct_in_final_sample"] for r in rows],
        color="#2a9d8f",
        marker="o",
        ms=3.6,
        lw=1.9,
        label="% of that era's hubs still in the final sample",
    )
    ax.set_xlabel("training iteration")
    ax.set_ylabel("% still visited by the trained model", color="#2a9d8f")
    ax.tick_params(axis="y", labelcolor="#2a9d8f")
    ax2 = ax.twinx()
    ax2.plot(
        x,
        [r["median_final_rank"] for r in rows],
        color="#b23a48",
        marker="s",
        ms=3.4,
        lw=1.9,
        ls="--",
        label="median rank in the final flow ordering",
    )
    ax2.set_ylabel("median final flow-rank (lower = higher flow)", color="#b23a48")
    ax2.tick_params(axis="y", labelcolor="#b23a48")
    ax2.invert_yaxis()  # so "better" is up on both axes
    ax.axvline(1000, color="#888", ls=":", lw=1.1)
    ax.text(
        1040, ax.get_ylim()[0] + 0.5, "first dynamic-library\npromotion", fontsize=7, color="#888"
    )
    ax.set_title(
        "The flow field moves during training: hubs favoured early are largely abandoned\n"
        "(both curves up = later-training hubs match the trained model better)",
        fontsize=9.5,
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(out / "flow_drift.png", dpi=140)
    print(f"\n[drift] wrote {out / 'flow_drift.png'} + .csv")


if __name__ == "__main__":
    main()
