#!/usr/bin/env python
"""Our heuristic F(h) vs an enumeration-reconstructed "true" F(h).

The pipeline never measures a hub's flow directly — it *estimates* it from one child at a time via
detailed balance (``LSD_FLOW_PROPOSAL`` §2). With exhaustive enumeration we can reconstruct F(h) from
flow conservation and ask how good those estimates are, in absolute terms and — what actually matters
for ``pick_hubs`` — as a *ranking*.

RECONSTRUCTION (``[malkin2022trajectorybalance]`` Eqs. 6-7). Two expressions for the same edge flow,
``F(h->x) = F(h)P_F(x|h)`` and ``F(h->x) = F(x)P_B(h|x)``, summed over ALL children of h:

    F(h) * S(h) = N(h)        S(h) := sum_x P_F(x|h) = 1 - P_F(stop|h)
                              N(h) := sum_x F(x) P_B(h|x)          F(x) = R(x)/P_F(stop|x)
    =>  F_true(h) = N(h) / S(h)

and the terminating-flow identity ``R(h) = F(h) P_F(stop|h)`` gives an INDEPENDENT check:
``R_implied(h) = F_true(h) * (1 - S(h))`` must equal the hub molecule's own reward. That check needs
no model call when the hub also appears as a sampled terminal (then we have its R directly).

ESTIMATORS COMPARED (all in log space, against ``log F_true``):
  * ``pick_hubs``  — what production actually uses: the MAX single-candidate log F_hat over the
                     sampled children of h (see ``campaign/pick_hubs.py``). A one-child heuristic.
  * ``samp_mean``  — mean log F_hat over the *sampled* children only.
  * ``enum_mean``  — mean log F_hat over ALL enumerated children (unweighted).
  * ``enum_wmean`` — P_F-weighted mean over all enumerated children (the statistic consistent with
                     F_true, since F_true is the P_F-weighted arithmetic mean of the per-child F_hat).

Reported per estimator: median bias (nats) and **Spearman rank correlation with log F_true across
hubs** — a biased estimator is still fine for hub *selection* if it ranks correctly.

UNITS: R is the shaped reward the GFN trains against (exp(beta*clip(value))), so log F values are
large (beta=8); divide nats by beta for raw-proxy-unit intuition. Pure stdlib/CPU; login-safe.

    python experiments/lsd_hubs/matrix16/hub_flow_estimators.py rxnflow_seh [scent_seh ...]
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from flow_consistency import _load_terms, _logsumexp, _pct  # noqa: E402
from manifest import get_cell  # noqa: E402

TOP_K_CANDIDATES = 1000  # pick_hubs default, mirrored so we reproduce its heuristic exactly


def _spearman(a: list[float], b: list[float]) -> float | None:
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
    ma, mb = st.fmean(ra), st.fmean(rb)
    num = math.fsum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(math.fsum((x - ma) ** 2 for x in ra))
    db = math.sqrt(math.fsum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else None


def _true_flow(d: dict) -> dict | None:
    """F_true(h) = N/S from flow conservation over ALL enumerated children, + the estimators that
    need the full child set."""
    lr, pf, pb, ps = d["lr"], d["pf"], d["pb"], d["ps"]
    n = len(lr)
    if n < 2:
        return None
    logF_x = [lr[i] - ps[i] for i in range(n)]  # F(x) = R(x)/P_F(stop|x)
    edge = [logF_x[i] + pb[i] for i in range(n)]  # F(h->x) = F(x) P_B(h|x)
    logS, logN = _logsumexp(pf), _logsumexp(edge)
    if not (math.isfinite(logS) and math.isfinite(logN)):
        return None
    logF_true = logN - logS
    fhat = [edge[i] - pf[i] for i in range(n)]
    w = [math.exp(p - logS) for p in pf]
    tw = math.fsum(w)
    out = {
        "n_children": n,
        "S": math.exp(logS),
        "logS": logS,
        "logN": logN,
        "log_F_true": logF_true,
        "enum_mean": st.fmean(fhat),
        "enum_wmean": math.fsum(w[i] * fhat[i] for i in range(n)) / tw if tw else float("nan"),
    }
    if 0 < out["S"] < 1:
        out["log_R_implied"] = logF_true + math.log1p(-out["S"])
    return out


def _sampled_estimators(samp_csv: Path, hib: bool) -> tuple[dict, dict]:
    """Reproduce pick_hubs' heuristic from the sampled DAG.

    Returns (per-hub estimators, hub_reward) where hub_reward maps a hub key -> its own shaped
    log-reward for hubs that ALSO appear as a sampled terminal (the R(h) cross-check, model-free).
    """
    best_reward: dict = {}
    best_flow: dict = {}  # child -> (log_f, hub)
    per_hub_child_flows: dict = defaultdict(list)  # hub -> [log_f of its sampled children]
    child_logreward: dict = {}
    with open(samp_csv) as fh:
        for r in csv.DictReader(fh):
            child = r["child_stereo_key"] or r["child_key"]
            hub = r["hub_stereo_key"] or r["hub_key"]
            rew = float(r["reward"])
            if child not in best_reward or (rew > best_reward[child]) == hib:
                best_reward[child] = rew
            child_logreward[child] = float(r["log_reward"])
            lf = (
                float(r["log_reward"])
                + float(r["log_pb_move"])
                - float(r["log_pf_move"])
                - float(r["log_pf_stop"])
            )
            if not math.isfinite(lf):
                continue
            per_hub_child_flows[hub].append(lf)
            if child not in best_flow or lf > best_flow[child][0]:
                best_flow[child] = (lf, hub)

    top = sorted(best_reward, key=lambda c: best_reward[c], reverse=hib)[:TOP_K_CANDIDATES]
    pick: dict = {}  # hub -> max single-candidate log F_hat among top-K (pick_hubs' value)
    for child in top:
        if child in best_flow:
            lf, hub = best_flow[child]
            if hub not in pick or lf > pick[hub]:
                pick[hub] = lf
    est = {
        h: {"pick_hubs": pick.get(h), "samp_mean": st.fmean(v) if v else None, "n_sampled": len(v)}
        for h, v in per_hub_child_flows.items()
    }
    for h, v in pick.items():
        est.setdefault(h, {"pick_hubs": v, "samp_mean": None, "n_sampled": 0})["pick_hubs"] = v
    return est, child_logreward


def analyze(cell_tag: str) -> dict:
    gen, tgt = cell_tag.rsplit("_", 1)
    cell = get_cell(gen, tgt)
    enum_csv, samp_csv = (
        cell.enum_dir / "enumerated_records.csv",
        cell.sample_dir / "records.csv",
    )
    if not enum_csv.exists():
        print(f"[{cell_tag}] no enumerated_records.csv — skipping")
        return {}
    print(f"\n{'=' * 82}\n=== {cell_tag} : heuristic F(h) vs reconstructed F_true(h) ===")

    truth = {h: t for h, t in ((h, _true_flow(d)) for h, d in _load_terms(enum_csv).items()) if t}
    est, child_logreward = (
        _sampled_estimators(samp_csv, cell.target.higher_is_better)
        if samp_csv.exists()
        else ({}, {})
    )
    print(f"hubs with reconstructed F_true: {len(truth)}")
    lt = [t["log_F_true"] for t in truth.values()]
    print(
        f"  log F_true(h):  median={st.median(lt):.2f}  p5={_pct(lt, .05):.2f}  p95={_pct(lt, .95):.2f}"
        f"   (shaped units; /beta=8 for proxy-value intuition)"
    )

    print(
        f"\n{'estimator':13}{'n':>5}{'median bias':>13}{'p5':>9}{'p95':>9}{'|bias|<1':>10}{'Spearman vs F_true':>20}"
    )
    rows = {}
    for name in ("pick_hubs", "samp_mean", "enum_mean", "enum_wmean"):
        pairs = []
        for h, t in truth.items():
            v = t.get(name) if name.startswith("enum") else (est.get(h, {}) or {}).get(name)
            if v is not None and math.isfinite(v):
                pairs.append((v, t["log_F_true"]))
        if len(pairs) < 3:
            print(f"{name:13}{len(pairs):>5}  (too few hubs)")
            continue
        bias = [a - b for a, b in pairs]
        rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
        near = sum(1 for b in bias if abs(b) < 1.0)
        print(
            f"{name:13}{len(pairs):>5}{st.median(bias):>13.2f}{_pct(bias, .05):>9.2f}"
            f"{_pct(bias, .95):>9.2f}{f'{near}/{len(bias)}':>10}{str(round(rho, 3)) if rho else 'n/a':>20}"
        )
        rows[name] = {
            "n_hubs": len(pairs),
            "median_bias": round(st.median(bias), 3),
            "spearman": round(rho, 3) if rho else None,
        }

    # WELL-CONDITIONED flow-conservation test (needs `--mode probe_hubs` output). Eliminating F(h)
    # from  F(h) = R(h) + N(h)  and  R(h) = F(h) P_F(stop|h)  gives a dimensionless identity with no
    # near-1 cancellation:   R(h)/(R(h)+N(h)) == P_F(stop|h).
    probe_p = cell.enum_dir / "hub_terminal.json"
    if probe_p.exists():
        probe = json.load(open(probe_p))
        rows_p = []
        for h, t in truth.items():
            pr = probe.get(h)
            if not pr or "log_pf_stop_h" not in pr or "log_reward_h" not in pr:
                continue
            lR, lN = pr["log_reward_h"], t["logN"]
            logF_A = max(lR, lN) + math.log1p(math.exp(-abs(lR - lN)))  # log(R + N), stable
            rows_p.append(
                {
                    "implied_log_pf_stop": lR - logF_A,  # log R(h) - log(R(h)+N(h))
                    "measured_log_pf_stop": pr["log_pf_stop_h"],
                    "logF_A": logF_A,  # R(h) + N(h)
                    "logF_B": t["log_F_true"],  # N/S
                }
            )
        print("\n-- WELL-CONDITIONED flow conservation at h:  R/(R+N)  vs  P_F(stop|h) --")
        if rows_p:
            d = [r["implied_log_pf_stop"] - r["measured_log_pf_stop"] for r in rows_p]
            fa = [r["logF_A"] - r["logF_B"] for r in rows_p]
            print(f"  hubs probed: {len(rows_p)}/{len(truth)}")
            print(
                f"  log P_F(stop|h): implied - measured   median={st.median(d):>8.2f}  "
                f"p5={_pct(d, .05):>8.2f}  p95={_pct(d, .95):>8.2f} nats   (0 => conservation holds)"
            )
            print(
                f"  log F: (R+N) vs (N/S)                 median={st.median(fa):>8.3f}  "
                f"p5={_pct(fa, .05):>8.3f}  p95={_pct(fa, .95):>8.3f} nats"
            )
            rows["flow_conservation"] = {
                "n": len(rows_p),
                "median_stop_logprob_gap": round(st.median(d), 3),
                "median_logF_A_minus_B": round(st.median(fa), 4),
            }
        else:
            print("  probe file present but no hub joined (key mismatch?)")
    else:
        print(
            f"\n-- WELL-CONDITIONED test skipped: no {probe_p.name} (run worker --mode probe_hubs) --"
        )

    # INDEPENDENT validation of F_true: does the implied R(h) match the hub's own measured reward?
    checks = [
        (t["log_R_implied"], child_logreward[h])
        for h, t in truth.items()
        if "log_R_implied" in t and h in child_logreward
    ]
    print(
        f"\n-- INDEPENDENT check of F_true: R_implied(h) = F_true(h)(1-S) vs the hub's measured R(h) --"
    )
    if checks:
        diff = [a - b for a, b in checks]
        print(f"  hubs where the hub also appears as a sampled terminal: {len(checks)}")
        print(
            f"  log R_implied - log R_measured:  median={st.median(diff):>8.2f}  "
            f"p5={_pct(diff, .05):>8.2f}  p95={_pct(diff, .95):>8.2f} nats"
        )
        print(f"  (0 => flow conservation reproduces the hub's own reward exactly)")
        rows["R_check"] = {"n": len(checks), "median_diff": round(st.median(diff), 3)}
    else:
        print("  no hub appeared as a sampled terminal -> needs a proxy call (skipped)")
    return {"cell": cell_tag, "n_hubs": len(truth), "estimators": rows}


def main() -> None:
    out = [r for r in (analyze(c) for c in (sys.argv[1:] or ["rxnflow_seh"])) if r]
    if out:
        p = HERE / "results" / "gate_sweep" / "hub_flow_estimators.json"
        os.makedirs(p.parent, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
