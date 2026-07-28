#!/usr/bin/env python
"""Per-trajectory trajectory-balance residual: the Z-anchored vs the R-anchored F(h).

The same sampled trajectory gives **two exact** estimates of a hub's flow, sharing no terms:

    PREFIX (Z-anchored, source -> h):
        log F_pre(h)  = log Z + sum_{t<=k} [ logP_F(s_t|s_{t-1}) - logP_B(s_{t-1}|s_t) ]
    SUFFIX (R-anchored, h -> x -> terminal), what the pipeline ships:
        log F_suf(h)  = log R(x) + logP_B(h|x) - logP_F(x|h) - logP_F(stop|x)

Both follow from chaining detailed balance (``[malkin2022trajectorybalance]`` Eq. 7) from opposite
ends. Multiplying them reproduces the trajectory-balance constraint (their Eq. 13) exactly, so

    log F_pre(h) - log F_suf(h)  ==  TB residual of that trajectory

with **no frequency/visitation estimate anywhere**. A perfectly TB-trained model gives 0.

Inputs (produced by the workers): ``prefix_terms.csv`` + ``records.csv`` + ``meta.json`` (for ``log Z``)
in a cell's sample dir. Pure stdlib/CPU.

INTERPRETING A NON-ZERO RESULT — the two failure modes are distinguishable:
  * a near-CONSTANT offset across hubs  => the learned ``log Z`` scale is off (one global scalar);
  * a SPREAD that varies with hub/depth => the local per-step terms (P_F/P_B) don't balance.
Known caveat: rxnflow_seh's learned ``log Z`` (53.07) is *below* its own maximum single-molecule
reward (63.69), so its Z-anchor is unusable and its residual is dominated by that; scent_seh
(74.65 > 68.10) and fraggfn (97.17) are self-consistent.

    python experiments/lsd_hubs/matrix16/tb_residual.py scent_seh [rxnflow_seh ...]
    python experiments/lsd_hubs/matrix16/tb_residual.py --sample-dir /path/to/sample
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _pct(xs, q):
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))]


def analyze_dir(sample_dir: Path, label: str) -> dict | None:
    pre_p, rec_p, meta_p = (
        sample_dir / "prefix_terms.csv",
        sample_dir / "records.csv",
        sample_dir / "meta.json",
    )
    if not pre_p.exists():
        print(f"[{label}] no prefix_terms.csv in {sample_dir} — re-sample with the patched worker")
        return None
    log_z = float(json.load(open(meta_p)).get("log_z", 0.0)) if meta_p.exists() else 0.0

    # prefix sidecar: (child_stereo, hub_stereo) -> (pf_prefix, pb_prefix)
    pre = {}
    with open(pre_p) as fh:
        for r in csv.DictReader(fh):
            pre[(r["child_stereo_key"], r["hub_stereo_key"])] = (
                float(r["log_pf_prefix"]),
                float(r["log_pb_prefix"]),
            )
    rows, per_depth = [], defaultdict(list)
    with open(rec_p) as fh:
        for r in csv.DictReader(fh):
            key = (
                r.get("child_stereo_key") or r["child_key"],
                r.get("hub_stereo_key") or r["hub_key"],
            )
            if key not in pre:
                continue
            pf_pre, pb_pre = pre[key]
            f_pre = log_z + pf_pre - pb_pre
            f_suf = (
                float(r["log_reward"])
                + float(r["log_pb_move"])
                - float(r["log_pf_move"])
                - float(r["log_pf_stop"])
            )
            if not (math.isfinite(f_pre) and math.isfinite(f_suf)):
                continue
            rows.append((f_pre, f_suf, f_pre - f_suf))
            per_depth[int(r["hub_depth"])].append(f_pre - f_suf)

    if len(rows) < 3:
        print(f"[{label}] only {len(rows)} joinable records — nothing to report")
        return None
    res = [r[2] for r in rows]
    pre_v = [r[0] for r in rows]
    suf_v = [r[1] for r in rows]
    print(f"\n=== {label}  (n={len(rows)} trajectories, log Z = {log_z:.3f}) ===")
    print(
        f"  log F_pre (Z-anchored):  median={st.median(pre_v):>9.2f}  p5={_pct(pre_v, .05):>9.2f}  p95={_pct(pre_v, .95):>9.2f}"
    )
    print(
        f"  log F_suf (R-anchored):  median={st.median(suf_v):>9.2f}  p5={_pct(suf_v, .05):>9.2f}  p95={_pct(suf_v, .95):>9.2f}"
    )
    print(
        f"  TB RESIDUAL (pre-suf):   median={st.median(res):>9.2f}  p5={_pct(res, .05):>9.2f}  p95={_pct(res, .95):>9.2f} nats"
    )
    sd = st.pstdev(res) if len(res) > 1 else 0.0
    print(
        f"  residual spread (sd)={sd:.2f} nats  ->  {'CONSTANT-ish offset => logZ scale' if sd < 2.0 else 'VARIES => local P_F/P_B imbalance'}"
    )
    if len(per_depth) > 1:
        print("  by hub depth:")
        for d in sorted(per_depth):
            v = per_depth[d]
            print(f"    depth {d}: n={len(v):>5}  median residual={st.median(v):>8.2f}")
    return {
        "cell": label,
        "n": len(rows),
        "log_z": log_z,
        "median_log_F_prefix": round(st.median(pre_v), 3),
        "median_log_F_suffix": round(st.median(suf_v), 3),
        "median_tb_residual": round(st.median(res), 3),
        "residual_sd": round(sd, 3),
    }


def main() -> None:
    args = sys.argv[1:]
    out = []
    if args and args[0] == "--sample-dir":
        r = analyze_dir(Path(args[1]), Path(args[1]).parent.name)
        if r:
            out.append(r)
    else:
        from manifest import get_cell  # noqa: E402

        for tag in args or ["scent_seh"]:
            gen, tgt = tag.rsplit("_", 1)
            r = analyze_dir(get_cell(gen, tgt).sample_dir, tag)
            if r:
                out.append(r)
    if out:
        p = HERE / "results" / "gate_sweep" / "tb_residual.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        # UPSERT by cell, don't clobber: cells land one at a time (each needs its own
        # prefix-capable re-sample), and two agents share this file. A plain overwrite
        # silently drops every cell not named in *this* invocation.
        merged = {}
        if p.exists():
            try:
                for r in json.loads(p.read_text()):
                    merged[r["cell"]] = r
            except (ValueError, KeyError, TypeError):
                pass  # unreadable/legacy -> start fresh rather than lose this run
        for r in out:
            merged[r["cell"]] = r
        p.write_text(json.dumps([merged[k] for k in sorted(merged)], indent=2) + "\n")
        print(f"\nwrote {p}  ({len(merged)} cell(s): {', '.join(sorted(merged))})")


if __name__ == "__main__":
    main()
