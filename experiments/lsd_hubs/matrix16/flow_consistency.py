#!/usr/bin/env python
"""Severe test of the LSD-Flow recovery: per-child F_hat estimates vs an enumeration-derived F(h).

Our hub value comes from the detailed-balance estimator applied to ONE child at a time
(``LSD_FLOW_PROPOSAL`` §2), and ``U(h)`` is the disagreement between those per-child estimates.
Exhaustive enumeration lets us close the loop and ask whether the estimates are *right*, not just
mutually consistent.

Math (grounded in ``[malkin2022trajectorybalance]`` Eqs. 6-7, i.e. Markovian-flow definitions +
detailed balance; ``[bengio2021gflownet]`` flow matching):

    P_F(x|h) = F(h->x)/F(h)        P_B(h|x) = F(h->x)/F(x)        (Eq. 6)
    => F(h) P_F(x|h) = F(x) P_B(h|x)                              (Eq. 7, detailed balance)

  (1) PER-CHILD estimate (what we ship):
        F_hat(h;x) = F(x) P_B(h|x) / P_F(x|h),  with F(x) = R(x)/P_F(stop|x)
      so  log F_hat(h;x) = logR(x) - logP_F(stop|x) + logP_B(h|x) - logP_F(x|h)
      and U(h) = Var_x[log F_hat(h;x)]  (the flow-matching residual we report).

  (2) NORMALIZATION residual (needs exhaustive enumeration):
        S(h) := sum_x P_F(x|h) = 1 - P_F(stop|h)   <= 1
      S << 1 with a small stop-probability means the recovered P_F and/or the enumerated child set
      are inconsistent (missing mass); S > 1 means double counting (e.g. several enumerated paths
      mapping to the same product, or a mis-composed micro-step product).

  (3) AGGREGATE "true" flow (equate the two edge-flow forms and sum over ALL children):
        F(h) * S(h) = sum_x F(x) P_B(h|x) =: N(h)      =>   F_true(h) = N(h)/S(h)
      and the terminating-flow relation R(h) = F(h) P_F(stop|h) = F_true(h)(1 - S(h)) gives an
      implied hub reward as a second, independent consistency read.

  (4) BIAS of the shipped estimator:  mean_x[log F_hat(h;x)] - log F_true(h).
      Under a converged GFN this is 0 and U(h) is 0. Both are measured here.

Also compares the **sampled** U(h) (from the handful of children a training-policy sample happens to
hit -- what the AL loop would see) against the **enumerated** U(h) (all children), i.e. how badly the
cheap estimate misjudges the population value.

NOTE ON UNITS: R(x) is the *shaped* reward the GFN trains against (exp(beta*clip(value))), so
``log_reward`` = log R and absolute log F values are large (beta=8). All tests here are
differences/ratios, so the shaping cancels.

Pure stdlib + CPU (streams the CSV); login-node safe.
    python experiments/lsd_hubs/matrix16/flow_consistency.py rxnflow_seh [scent_seh ...]
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
from manifest import get_cell  # noqa: E402


def _logsumexp(xs: list[float]) -> float:
    if not xs:
        return -math.inf
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.fsum(math.exp(x - m) for x in xs))


def _pct(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[i]


def _load_terms(path: Path, key_col: str = "hub_stereo_key") -> dict:
    """hub -> lists of the four §2 log-terms per child (streamed)."""
    per = defaultdict(lambda: {"lr": [], "pf": [], "pb": [], "ps": []})
    with open(path) as fh:
        for r in csv.DictReader(fh):
            h = r.get(key_col) or r["hub_key"]
            d = per[h]
            d["lr"].append(float(r["log_reward"]))
            d["pf"].append(float(r["log_pf_move"]))
            d["pb"].append(float(r["log_pb_move"]))
            d["ps"].append(float(r["log_pf_stop"]))
    return per


def _hub_stats(d: dict) -> dict | None:
    """Per-hub: normalization S, aggregate F_true, per-child F_hat spread + bias."""
    lr, pf, pb, ps = d["lr"], d["pf"], d["pb"], d["ps"]
    n = len(lr)
    if n < 2:
        return None
    # log F(x) = logR(x) - logP_F(stop|x);  log F(h->x) = log F(x) + logP_B(h|x)
    logF_x = [lr[i] - ps[i] for i in range(n)]
    log_edge = [logF_x[i] + pb[i] for i in range(n)]
    logS = _logsumexp(pf)  # S = sum_x P_F(x|h)
    logN = _logsumexp(log_edge)  # N = sum_x F(x) P_B(h|x)
    logF_true = logN - logS
    fhat = [log_edge[i] - pf[i] for i in range(n)]  # per-child log F_hat
    fhat = [v for v in fhat if math.isfinite(v)]
    if len(fhat) < 2:
        return None
    mean = st.fmean(fhat)
    S = math.exp(logS) if logS < 50 else math.inf
    out = {
        "n_children": n,
        "S": S,  # should be <= 1, ~1 - P_F(stop|h)
        "logS": logS,
        "log_F_true": logF_true,
        "U_h": st.pvariance(fhat),  # what we ship
        "mean_log_Fhat": mean,
        "median_log_Fhat": st.median(fhat),
        "bias_mean_minus_true": mean - logF_true,
        "spread_p5_p95": _pct(fhat, 0.95) - _pct(fhat, 0.05),
    }
    # WHERE does U(h) come from? log F_hat = logF(x) + logP_B - logP_F, so decompose its variance
    # into each term's own variance (the cross-covariances make these not sum to U(h) exactly, but
    # the dominant term identifies what actually drives the spread).
    if n >= 2:
        out["var_logF_x"] = st.pvariance(logF_x)  # reward/terminating spread
        out["var_logP_B"] = st.pvariance(pb)  # backward-policy spread
        out["var_logP_F"] = st.pvariance(
            pf
        )  # forward-policy spread (1/P_F is the estimator's lever)

    # PROPOSED FIX. F_true is the P_F-WEIGHTED arithmetic mean of the per-child F_hat
    # (F_true = sum_x P_F(x|h) F_hat(h;x) / sum_x P_F(x|h)), so the *unweighted* variance we ship
    # lets vanishingly-improbable children (huge 1/P_F) dominate U(h). The consistent statistic is
    # the P_F-weighted variance of log F_hat, which weights each child by the flow it actually
    # carries. Reported alongside so the two definitions can be compared directly.
    w = [math.exp(p - logS) for p in pf]  # normalized weights P_F(x|h)/S, sum to 1
    tw = math.fsum(w)
    if tw > 0 and len(fhat) == n:
        wmean = math.fsum(w[i] * fhat[i] for i in range(n)) / tw
        out["U_h_pf_weighted"] = math.fsum(w[i] * (fhat[i] - wmean) ** 2 for i in range(n)) / tw
        out["wmean_log_Fhat"] = wmean
        out["bias_wmean_minus_true"] = wmean - logF_true
        out["effective_n_children"] = (tw**2) / math.fsum(x * x for x in w)  # Kish ESS
    # implied hub reward from the terminating-flow relation R(h) = F(h)(1-S)
    if 0 < S < 1:
        out["log_R_h_implied"] = logF_true + math.log1p(-S)
    return out


def analyze(cell_tag: str) -> dict:
    gen, tgt = cell_tag.rsplit("_", 1)
    cell = get_cell(gen, tgt)
    enum_csv = cell.enum_dir / "enumerated_records.csv"
    samp_csv = cell.sample_dir / "records.csv"
    if not enum_csv.exists():
        print(f"[{cell_tag}] no enumerated_records.csv — skipping")
        return {}

    print(f"\n{'=' * 78}\n=== {cell_tag} ===")
    enum_per = _load_terms(enum_csv)
    stats = {h: s for h, s in ((h, _hub_stats(d)) for h, d in enum_per.items()) if s}
    print(f"hubs with >=2 enumerated children: {len(stats)}")

    def summarize(field, label, unit=""):
        v = [s[field] for s in stats.values() if field in s and math.isfinite(s[field])]
        if not v:
            return None
        print(
            f"  {label:38} median={st.median(v):>10.3f}  p5={_pct(v, .05):>10.3f}  "
            f"p95={_pct(v, .95):>10.3f}{unit}"
        )
        return st.median(v)

    print("\n-- (2) NORMALIZATION: S(h) = sum_x P_F(x|h)  [must be <= 1; 1-P_F(stop|h)] --")
    medS = summarize("S", "S(h)")
    n_over = sum(1 for s in stats.values() if s["S"] > 1.001)
    n_tiny = sum(1 for s in stats.values() if s["S"] < 0.01)
    print(f"  hubs with S>1 (double counting): {n_over}/{len(stats)}")
    print(f"  hubs with S<0.01 (missing mass): {n_tiny}/{len(stats)}")

    print("\n-- (3)/(4) AGGREGATE F_true vs PER-CHILD F_hat --")
    summarize("log_F_true", "log F_true(h) = log N - log S")
    summarize("mean_log_Fhat", "mean_x log F_hat(h;x)")
    medbias = summarize("bias_mean_minus_true", "BIAS = mean log F_hat - log F_true", " nats")
    summarize("spread_p5_p95", "per-child spread (p95-p5 of log F_hat)", " nats")
    medU = summarize("U_h", "U(h) = Var_x[log F_hat]  (enumerated)")

    print("\n-- WHERE U(h) COMES FROM: per-term variance (dominant term drives the spread) --")
    summarize("var_logF_x", "Var[log F(x)]   (reward / terminating)")
    summarize("var_logP_B", "Var[log P_B(h|x)] (backward policy)")
    summarize("var_logP_F", "Var[log P_F(x|h)] (forward policy)")

    print("\n-- PROPOSED FIX: P_F-weighted U(h) (weight each child by the flow it carries) --")
    summarize("U_h_pf_weighted", "U_w(h) = Var_w[log F_hat]")
    summarize("bias_wmean_minus_true", "BIAS of weighted mean vs log F_true", " nats")
    summarize("effective_n_children", "effective #children (Kish ESS)")

    # sampled vs enumerated U(h)
    samp_med = None
    if samp_csv.exists():
        samp_per = _load_terms(samp_csv)
        pairs = []
        for h, s in stats.items():
            d = samp_per.get(h)
            if not d or len(d["lr"]) < 2:
                continue
            ss = _hub_stats(d)
            if ss:
                pairs.append((ss["U_h"], s["U_h"], len(d["lr"])))
        if pairs:
            print(
                f"\n-- SAMPLED vs ENUMERATED U(h)  ({len(pairs)} hubs with >=2 sampled children) --"
            )
            sv = [p[0] for p in pairs]
            ev = [p[1] for p in pairs]
            nc = [p[2] for p in pairs]
            samp_med = st.median(sv)
            print(f"  sampled children per hub:     median={st.median(nc):.0f}  max={max(nc)}")
            print(f"  U(h) sampled   median={st.median(sv):>10.2f}")
            print(f"  U(h) enumerated median={st.median(ev):>10.2f}  (same hubs)")
            und = sum(1 for a, b, _ in pairs if a < b)
            print(f"  sampled UNDER-estimates enumerated U(h) for {und}/{len(pairs)} hubs")
        else:
            print("\n-- SAMPLED vs ENUMERATED U(h): no hub had >=2 sampled children --")
            n1 = sum(1 for h in stats if len(samp_per.get(h, {"lr": []})["lr"]) == 1)
            print(
                f"   (hubs with exactly 1 sampled child: {n1}/{len(stats)}"
                f" -> U(h) is UNDEFINED from sampling alone)"
            )
    return {
        "cell": cell_tag,
        "n_hubs": len(stats),
        "median_S": medS,
        "hubs_S_gt_1": n_over,
        "median_bias_nats": medbias,
        "median_U_enumerated": medU,
        "median_U_sampled": samp_med,
    }


def main() -> None:
    cells = sys.argv[1:] or ["rxnflow_seh"]
    out = [r for r in (analyze(c) for c in cells) if r]
    if out:
        p = HERE / "results" / "gate_sweep" / "flow_consistency.json"
        os.makedirs(p.parent, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
