"""Measured compute-time accounting for hub-batching vs best-candidate (Logs/039).

The reactions-per-mode model (:mod:`dynamic_amortization`) is the *synthesis* (bench) cost;
``reward_gen_calls`` is a *count* of reward-generator invocations. Neither is wall-clock. This module
adds the third axis the paper needs: **how much longer the computer actually works** to do
hub-batching instead of best-candidate, **differentiated by component** so we can see *where* the time
goes.

The times are **measured, not modeled** (see the memory note). The GPU enumeration worker
(``scent_worker.py --mode enumerate``) records, per hub, the real wall-clock of its three components
into ``enum_timings.json``:

  * ``enumeration_s``   — RDKit construction of the hub's one-reaction children (CPU),
  * ``reward_gen_s``    — the reward generator scoring them (proxy or docking oracle — this is the
                          axis that gets big for docking, because it *actually* took longer),
  * ``flow_extract_s``  — ``assign_log_probs`` -> P_F / P_B, the hub-ranking / U(h) signal (GPU),

plus a one-time ``setup_s`` (model load + gin build + dynamic-library freeze). Each campaign strategy
walks a *subset* of the enumerated hubs (best-candidate walks none). :class:`EnumTimings.attribute`
sums the measured per-hub times over the hubs a strategy actually walked
(:attr:`CampaignResult.walked_hub_keys`) — so the reported time is measured-per-hub, summed over the
real walk, never a guessed constant. The driver additionally times the CPU mode-selection live.

Component ownership (per operating point):

| component        | best-candidate            | hub-batching                         |
|------------------|---------------------------|--------------------------------------|
| setup_s          | 0 (reuses sampled pool)   | once, if it walks >=1 hub            |
| hub_pick_s       | 0                         | Stage-2 hub ranking (optional)       |
| enumeration_s    | 0                         | sum over walked hubs (measured)      |
| reward_gen_s     | 0 (marginal; see below)   | sum over walked hubs (measured)      |
| flow_extract_s   | 0                         | sum over walked hubs (measured)      |
| mode_selection_s | measured live in driver   | measured live in driver              |

Best-candidate's ``reward_gen_s`` is **0 marginal**: it reuses rewards computed during Stage-1
sampling (the shared pool both strategies need to discover candidates/hubs). Stage-1 sampling time is
tracked separately (``sample_timings.json``) as shared context; it cancels in the head-to-head. The
head-to-head "extra compute for hub-batching" is therefore Stages 2-3 (setup + hub-pick + enumeration
+ reward-gen + flow-extract) plus any mode-selection delta.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

# The three per-hub measured components the worker records.
_HUB_COMPONENTS = ("enumeration_s", "reward_gen_s", "flow_extract_s")


@dataclass
class EnumTimings:
    """Measured per-hub enumeration timings from ``enum_timings.json`` (the join key is ``hub_key``).

    ``per_hub`` maps ``hub_key -> {enumeration_s, reward_gen_s, flow_extract_s, n_children, depth}``.
    ``setup_s`` is the one-time model-load + library-freeze cost. ``meta`` carries the run context
    (device, reward_name, whether CUDA was synchronized, totals).
    """

    per_hub: Dict[str, dict] = field(default_factory=dict)
    setup_s: float = 0.0
    meta: dict = field(default_factory=dict)

    @staticmethod
    def _hub_id(h: dict) -> str:
        """Unique per-hub identity for the timing join: raw ``hub_input`` (stereo-aware) so
        stereoisomeric hubs — which share the stereo-stripped ``hub_key`` but were enumerated
        separately — are kept distinct. Falls back to ``hub_key`` for older timing files."""
        return h.get("hub_input") or h["hub_key"]

    @classmethod
    def load(cls, path) -> "EnumTimings":
        data = json.load(open(path))
        meta = data.get("meta", {})
        per_hub = {cls._hub_id(h): h for h in data.get("per_hub", [])}
        return cls(per_hub=per_hub, setup_s=float(meta.get("setup_s", 0.0)), meta=meta)

    @classmethod
    def merge(cls, paths: Sequence) -> "EnumTimings":
        """Merge per-hub timing slices (the 200-hub run split across parallel debug jobs) into one.

        Per-hub entries are unioned (each hub timed in exactly one slice); ``setup_s`` is charged
        ONCE (a single production enumeration pays setup once) — we take the **median** across slices
        as the representative single-run setup. ``meta.slice_setups_s`` keeps the raw per-slice values.
        """
        import statistics

        merged: Dict[str, dict] = {}
        setups: List[float] = []
        metas: List[dict] = []
        for p in paths:
            t = cls.load(p)  # already keyed by unique hub_input
            setups.append(t.setup_s)
            metas.append(t.meta)
            for k, v in t.per_hub.items():
                merged[k] = v  # hubs are disjoint across slices (unique hub_input key)
        setup = statistics.median(setups) if setups else 0.0
        totals = {k: round(sum(v.get(k, 0.0) for v in merged.values()), 3) for k in _HUB_COMPONENTS}
        meta = {
            "merged_from": [str(p) for p in paths],
            "n_slices": len(paths),
            "n_hubs": len(merged),
            "n_children": sum(int(v.get("n_children", 0)) for v in merged.values()),
            "setup_s": round(setup, 3),
            "slice_setups_s": [round(s, 3) for s in setups],
            "totals_s": totals,
            "device": (metas[0].get("device") if metas else None),
            "reward_name": (metas[0].get("reward_name") if metas else None),
            "cuda_synchronized": (metas[0].get("cuda_synchronized") if metas else None),
        }
        return cls(per_hub=merged, setup_s=setup, meta=meta)

    def attribute(self, walked_hub_ids: Sequence[str]) -> dict:
        """Sum the measured per-hub components over the hubs a strategy walked (each identified by its
        unique ``hub_input``). Returns the three component sums plus ``n_hubs_walked`` /
        ``n_children_scored`` and ``missing_hub_keys`` (walked hubs with no recorded timing — should be
        empty once all hubs are timed; flagged, not guessed)."""
        out = {k: 0.0 for k in _HUB_COMPONENTS}
        n_children = 0
        missing: List[str] = []
        for hid in walked_hub_ids:
            rec = self.per_hub.get(hid)
            if rec is None:
                missing.append(hid)
                continue
            for k in _HUB_COMPONENTS:
                out[k] += float(rec.get(k, 0.0))
            n_children += int(rec.get("n_children", 0))
        out["n_hubs_walked"] = len(walked_hub_ids)
        out["n_children_scored"] = n_children
        out["missing_hub_keys"] = missing
        return out


@dataclass
class ComputeTimeBreakdown:
    """One strategy's measured compute-time, differentiated by component (all in seconds)."""

    strategy: str
    setup_s: float = 0.0
    hub_pick_s: float = 0.0
    enumeration_s: float = 0.0
    reward_gen_s: float = 0.0
    flow_extract_s: float = 0.0
    mode_selection_s: float = 0.0
    n_hubs_walked: int = 0
    n_children_scored: int = 0
    missing_hub_keys: List[str] = field(default_factory=list)

    _COMPONENTS = (
        "setup_s",
        "hub_pick_s",
        "enumeration_s",
        "reward_gen_s",
        "flow_extract_s",
        "mode_selection_s",
    )

    @property
    def total_s(self) -> float:
        return sum(getattr(self, c) for c in self._COMPONENTS)

    def components(self) -> Dict[str, float]:
        return {c: round(getattr(self, c), 4) for c in self._COMPONENTS}

    def to_dict(self) -> dict:
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}
        d["total_s"] = round(self.total_s, 4)
        return d


def account_strategy(
    result,
    enum_timings: Optional[EnumTimings],
    *,
    selection_s: float,
    hub_pick_s: float = 0.0,
) -> ComputeTimeBreakdown:
    """Build a :class:`ComputeTimeBreakdown` for one :class:`CampaignResult`.

    ``selection_s`` is the driver's live-measured wall-clock of the strategy's ``run()`` (the CPU
    mode-selection + book-keeping over the cached rewards — the real Stage-4 cost). ``hub_pick_s`` is
    the Stage-2 hub-ranking time (0 for best-candidate; measured by ``pick_hubs.py``). The GPU
    components come from attributing ``enum_timings`` over ``result.walked_hub_keys``; a strategy that
    walked no hubs (best-candidate) gets zeros there and pays only ``mode_selection_s``.
    """
    walked = list(getattr(result, "walked_hub_ids", []) or [])
    bd = ComputeTimeBreakdown(strategy=result.strategy, mode_selection_s=float(selection_s))
    if walked and enum_timings is not None:
        att = enum_timings.attribute(walked)
        bd.enumeration_s = att["enumeration_s"]
        bd.reward_gen_s = att["reward_gen_s"]
        bd.flow_extract_s = att["flow_extract_s"]
        bd.n_hubs_walked = att["n_hubs_walked"]
        bd.n_children_scored = att["n_children_scored"]
        bd.missing_hub_keys = att["missing_hub_keys"]
        bd.setup_s = float(enum_timings.setup_s)  # one-time cost of doing hub-batching at all
        bd.hub_pick_s = float(hub_pick_s)
    return bd


def head_to_head(hub_bd: ComputeTimeBreakdown, best_bd: ComputeTimeBreakdown) -> dict:
    """The paper headline: how much longer the computer works for hub-batching vs best-candidate,
    per component and in total. Marginal = hub − best (Stage-1 sampling is shared and excluded)."""
    delta = {
        c: round(getattr(hub_bd, c) - getattr(best_bd, c), 4)
        for c in ComputeTimeBreakdown._COMPONENTS
    }
    extra = round(hub_bd.total_s - best_bd.total_s, 4)
    ratio = (hub_bd.total_s / best_bd.total_s) if best_bd.total_s > 0 else None
    return {
        "hub_total_s": round(hub_bd.total_s, 4),
        "best_total_s": round(best_bd.total_s, 4),
        "extra_compute_s": extra,  # hub-batching's extra wall-clock over best-candidate
        "ratio_hub_over_best": round(ratio, 3) if ratio is not None else None,
        "per_component_delta_s": delta,
        "note": "Stage-1 sampling (shared pool) excluded; it cancels in the marginal.",
    }
