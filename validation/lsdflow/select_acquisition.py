#!/usr/bin/env python
"""Cross-env LSD-Flow acquisition selector — runs in the ``rgfn`` env on files a generator's
per-env worker produced (proposal §4a; the cross-env twin of ``glue.samplers.lsdflow.acquisition``).

**Why this exists.** The in-env RGFN loop calls :class:`~glue.samplers.lsdflow.acquisition.LSDFlowAcquisition`
directly. SCENT / FragGFN / RxnFlow run in their own conda envs where ``import glue`` resolves to the
wrong ``rgfn`` (SCENT's fork) — so their loops **cannot** run the glue selection in-process. Instead,
per round, the generator's env enumerates its candidate hubs' children (scoring with the live proxy
``M``) and writes ``enum_children.json`` (+ ``compositions.json`` + a promoted-fragment snapshot); this
script — in the ``rgfn`` env, where ``glue`` imports cleanly — reads those files and does the exact same
selection the in-env acquisition does:

    hub_batching:  UCB-rank hubs by z(reward(h)) + λ·z(U(h))  →  pre-select-K fragments  →
                   HubBatchingStrategy(free_frag, budget=("modes", N))  →  chosen SMILES
    best_candidate: top-M candidates under the same hit-bar + diversity filter

and writes the chosen molecules for the calling loop to dock. It reuses the glue campaign primitives
(``HubBatchingStrategy`` / ``BestCandidateStrategy`` / ``rank_fragments``) + the validation
``FragmentCostTable`` (pre-select-K needs it — SCENT's promoted-fragment recipes), so the cross-env
result matches the in-env one by construction. One selector serves every cross-env generator.

File contract (written by the per-env worker; SCENT's ``scent_worker.py`` already emits all of these):
  --enum-children  enum_children.json   {"hubs":[{hub_key,hub_input,depth,uncertainty(U(h)),
                                          children:[{smiles,reward,added_promoted}]}]}
  --compositions   compositions.json     child_key -> {promoted:[...], num_reactions}
  --snapshot       fragments_<N>.json    promoted-fragment recipes → FragmentCostTable (pre-select-K)
  --records        records.csv           sampled pool (best_candidate arm only)
Outputs --out chosen.csv: smiles[,source_hub] — the round's docking batch.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

from glue.samplers.lsdflow.campaign import (
    BestCandidateStrategy,
    Candidate,
    EnumChild,
    EnumeratedHub,
    HubBatchingStrategy,
    rank_fragments,
)
from glue.samplers.lsdflow.child_select import make_child_policy
from glue.samplers.lsdflow.hub.ucb import (
    _zscore,  # z-score (NaN-safe) — the UCB blend's normalizer
)
from validation.lsdflow.metrics.cost.dynamic_amortization import (
    load_cost_table_from_snapshot,
)


# ----------------------------------------------------------------- loaders (worker file schema)
def _load_enum_hubs(enum_children_path: str, comps: dict) -> List[EnumeratedHub]:
    data = json.load(open(enum_children_path))
    hubs = []
    for h in data.get("hubs", []):
        hub_comp = comps.get(h["hub_key"], {}) or {}
        hubs.append(
            EnumeratedHub(
                hub_key=h["hub_key"],
                hub_input=h.get("hub_input", h["hub_key"]),
                depth=int(h["depth"]),
                promoted=tuple(hub_comp.get("promoted", ())),
                children=[
                    EnumChild(
                        smiles=c["smiles"],
                        reward=float(c["reward"]),
                        added_promoted=tuple(c.get("added_promoted", ())),
                    )
                    for c in h["children"]
                ],
                uncertainty=h.get("uncertainty"),  # U(h), precomputed by the worker
                n_effective=int(h.get("n_effective", 0)),
            )
        )
    return hubs


def _load_candidates(records_csv: str, comps: dict, higher_is_better: bool) -> List[Candidate]:
    best: dict = {}
    parents: dict = {}
    with open(records_csv) as fh:
        for r in csv.DictReader(fh):
            c, reward = r["child_key"], float(r["reward"])
            if c not in best or (reward > best[c]) == higher_is_better:
                best[c] = reward
            hk = r.get("hub_key")
            if hk:
                parents.setdefault(c, set()).add(hk)
    out = []
    for c, reward in best.items():
        comp = comps.get(c, {}) or {}
        out.append(
            Candidate(
                smiles=c,
                reward=reward,
                num_reactions=int(comp.get("num_reactions", 1)),
                promoted=tuple(comp.get("promoted", ())),
                parents=tuple(sorted(parents.get(c, ()))),
            )
        )
    return out


# ----------------------------------------------------------------- UCB hub ranking
def _best_child_reward(hub: EnumeratedHub, higher_is_better: bool) -> float:
    rewards = [c.reward for c in hub.children if c.reward == c.reward]
    if not rewards:
        return float("nan")
    return max(rewards) if higher_is_better else -min(rewards)  # oriented higher = better


def ucb_rank(
    hubs: List[EnumeratedHub], lam: float, higher_is_better: bool, min_children: int = 2
) -> List[EnumeratedHub]:
    """Rank hubs by ``z(reward(h)) + λ·z(U(h))`` — the same explore/exploit blend as
    :class:`~glue.samplers.lsdflow.hub.ucb.UcbHubStrategy`, over the worker's precomputed per-hub
    ``U(h)``. Each term z-scored across the eligible hubs so λ is a scale-free weight; a NaN term
    contributes 0 (never voids an eligible hub). Hubs with < ``min_children`` children (U undefined)
    are dropped."""
    eligible = [h for h in hubs if len(h.children) >= min_children]
    if not eligible:
        return []
    z_r = _zscore([_best_child_reward(h, higher_is_better) for h in eligible])
    z_u = _zscore([h.uncertainty if h.uncertainty is not None else float("nan") for h in eligible])
    scored = [
        (h, (0.0 if zr != zr else zr) + lam * (0.0 if zu != zu else zu))
        for h, zr, zu in zip(eligible, z_r, z_u)
    ]
    scored.sort(key=lambda hs: hs[1], reverse=True)
    return [h for h, _ in scored]


# ----------------------------------------------------------------- driver
def select(
    arm: str,
    *,
    enum_children: Optional[str],
    compositions: Optional[str],
    snapshot: Optional[str],
    records: Optional[str],
    lam: float,
    reward_threshold: Optional[float],
    similarity: float,
    higher_is_better: bool,
    budget_modes: int,
    prebuild_k: int,
    child_policy: str,
    rank_by: str,
    min_children: int,
    proxy_label_mean: float = 0.0,
    proxy_label_std: float = 1.0,
) -> list:
    comps = json.load(open(compositions)) if compositions and Path(compositions).exists() else {}
    cost_table = None
    if snapshot and Path(snapshot).exists():
        cost_table = load_cost_table_from_snapshot(json.load(open(snapshot)))
    # The proxy M standardizes labels at fit time, so worker `reward` fields are STANDARDIZED. Map the
    # real-unit hit bar (e.g. sEH docking −8) into M's output space so the mode gate is correct — the
    # cross-env twin of glue.samplers.lsdflow.acquisition._mode_selector. Ranking (z-scored / rank
    # order) is invariant to this affine map; only the gate needs it. No-op when std==1 & mean==0.
    thr = reward_threshold
    if thr is not None and proxy_label_std and proxy_label_std > 0:
        thr = (thr - proxy_label_mean) / proxy_label_std
    common = dict(
        target="lsdflow",
        reward_threshold=thr,
        similarity=similarity,
        higher_is_better=higher_is_better,
    )

    if arm == "best_candidate":
        cands = _load_candidates(records, comps, higher_is_better)
        result = BestCandidateStrategy(cands, cost_table, hub_compositions=comps, **common).run(
            ("modes", budget_modes)
        )
        return [(p.smiles, p.source_hub) for p in result.accepted]

    # hub_batching
    hubs = _load_enum_hubs(enum_children, comps)
    ranked = ucb_rank(hubs, lam, higher_is_better, min_children=min_children)
    prebuilt = None
    if prebuild_k > 0 and cost_table is not None:
        rf = rank_fragments(
            ranked, cost_table, reward_threshold, method=rank_by, higher_is_better=higher_is_better
        )
        prebuilt = {f for f, _ in rf[:prebuild_k]}
    result = HubBatchingStrategy(
        ranked,
        cost_table,
        child_policy=make_child_policy(child_policy),
        prebuilt_fragments=prebuilt,
        **common,
    ).run(("modes", budget_modes))
    return [(p.smiles, p.source_hub) for p in result.accepted]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["hub_batching", "best_candidate"], default="hub_batching")
    ap.add_argument("--enum-children", default=None)
    ap.add_argument("--compositions", default=None)
    ap.add_argument("--snapshot", default=None, help="fragments_<N>.json (pre-select-K cost table)")
    ap.add_argument("--records", default=None, help="records.csv (best_candidate pool)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--reward-threshold", type=float, default=None)
    ap.add_argument("--similarity", type=float, default=0.5)
    ap.add_argument(
        "--higher-is-better", dest="higher_is_better", action="store_true", default=False
    )
    ap.add_argument("--budget-modes", type=int, default=100)
    ap.add_argument("--prebuild-k", type=int, default=20)
    ap.add_argument("--child-policy", default="free_frag", choices=["free_frag", "reward"])
    ap.add_argument("--rank-by", default="build_score", choices=["build_score", "fanout", "reward"])
    ap.add_argument("--min-children", type=int, default=2)
    # M standardizes labels → worker rewards are standardized; map the real hit bar into that space.
    ap.add_argument("--proxy-label-mean", type=float, default=0.0)
    ap.add_argument("--proxy-label-std", type=float, default=1.0)
    a = ap.parse_args(argv)

    chosen = select(
        a.arm,
        enum_children=a.enum_children,
        compositions=a.compositions,
        snapshot=a.snapshot,
        records=a.records,
        lam=a.lam,
        reward_threshold=a.reward_threshold,
        similarity=a.similarity,
        higher_is_better=a.higher_is_better,
        budget_modes=a.budget_modes,
        prebuild_k=a.prebuild_k,
        child_policy=a.child_policy,
        rank_by=a.rank_by,
        min_children=a.min_children,
        proxy_label_mean=a.proxy_label_mean,
        proxy_label_std=a.proxy_label_std,
    )
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "source_hub"])
        for smi, hub in chosen:
            w.writerow([smi, hub or ""])
    print(f"[select_acquisition] arm={a.arm}: wrote {len(chosen)} chosen molecules -> {a.out}")


if __name__ == "__main__":
    main()
