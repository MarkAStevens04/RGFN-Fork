"""Shared, stdlib-only artifact writers for the per-env LSD-Flow workers.

Every cross-env worker (SCENT/FragGFN/RxnFlow) must emit the SAME on-disk contract so everything
downstream — ``pick_hubs.py``, the harness DAG loader, ``run_campaign.py``/``sweep_campaign.py`` —
is model-agnostic. ``scent_worker.py`` predates this module and inlines the same logic; the newer
workers import THIS so the format can never drift between generators (and a future 5th generator
gets it for free).

**Import contract:** pure stdlib (``csv``/``json``/``math``) so it loads unchanged in every conda
env. A worker run as a script imports it via a ``sys.path`` insert of its own directory::

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _artifacts

The record-row shape is a dict keyed by :data:`REC_COLS`; that mirrors the
:class:`glue.samplers.lsdflow.records.FlowRecord` fields and is what ``records.csv`` /
``enumerated_records.csv`` persist. How a worker *computes* those values is model-specific (that is
the flow-extraction recipe); this module only serializes them.
"""

from __future__ import annotations

import csv
import json
import math
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

# records.csv / enumerated_records.csv columns — MUST match scent_worker._REC_COLS and the
# FlowRecord fields the harness/adapters read back (validation/lsdflow/adapters/*_adapter.py).
REC_COLS = [
    "hub_key",
    "child_key",
    "reward",
    "log_reward",
    "log_pf_move",
    "log_pb_move",
    "log_pf_stop",
    "hub_depth",
    "hub_stereo_key",
    "child_stereo_key",
]


def log_flow(rec: Dict) -> float:
    """The §2 single-child log-flow estimate ``log F_hat(h;x) = logR + logP_B(move) -
    logP_F(move) - logP_F(stop)`` from one record's log-terms."""
    return (
        float(rec["log_reward"])
        + float(rec["log_pb_move"])
        - float(rec["log_pf_move"])
        - float(rec["log_pf_stop"])
    )


def hub_uncertainty(recs: Sequence[Dict]) -> Tuple[float, int]:
    """``U(h)`` = population variance of a hub's children's ``log F_hat`` (§2 flow-matching
    residual), plus ``n_effective`` (count of finite estimates). ``U = NaN`` for < 2 finite
    estimates (variance undefined). Matches ``scent_worker._hub_uncertainty``."""
    xs = [log_flow(r) for r in recs]
    xs = [x for x in xs if math.isfinite(x)]
    n = len(xs)
    if n < 2:
        return float("nan"), n
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n  # population variance
    return var, n


def write_records(path, rows: Sequence[Dict]) -> None:
    """Write records.csv / enumerated_records.csv (DictWriter over :data:`REC_COLS`).

    ``extrasaction="ignore"`` so a row may carry additional keys (e.g. the Z-anchored prefix terms,
    written separately by :func:`write_prefix_terms`) without changing this file's schema — every
    downstream reader parses records.csv by column NAME, so the contract stays fixed."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REC_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


PREFIX_COLS = ["child_key", "child_stereo_key", "hub_stereo_key", "log_pf_prefix", "log_pb_prefix"]


def write_prefix_terms(path, rows: Sequence[Dict]) -> None:
    """Write prefix_terms.csv — the source->hub half of the trajectory, for the Z-ANCHORED flow
    reconstruction.

    Chaining detailed balance forward from the source (``F(s_0) = Z``) gives an exact second estimate
    of a hub's flow that shares no terms with the R-anchored one we ship:

        log F_prefix(h) = log Z + sum_{t<=k} [ logP_F(s_t|s_{t-1}) - logP_B(s_{t-1}|s_t) ]

    Multiplying it by the suffix (R-anchored) estimate reproduces trajectory balance exactly, so the
    two agree **iff** TB holds on that trajectory — their log-difference IS the per-trajectory TB
    residual, with no frequency/visitation estimate anywhere. ``log Z`` is already persisted in
    ``meta.json``.

    Kept in a SIDECAR rather than added to ``REC_COLS`` so records.csv's schema (and every existing
    reader) is untouched. Join on ``child_stereo_key`` + ``hub_stereo_key``.

    NOTE (FragGFN): its prefix ends at the last-AddNode *skeleton* state, which is also what its
    suffix estimator is anchored at — so the prefix-vs-suffix comparison is self-consistent, even
    though neither refers to the reconstructed hub *molecule*."""
    rows = [r for r in rows if r]
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PREFIX_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_enum_hub(
    *,
    hub_input: str,
    hub_key: str,
    depth: int,
    recs: Sequence[Dict],
    added_by_child: Optional[Dict[str, List[str]]] = None,
    reaction_by_child: Optional[Dict[str, list]] = None,
    promoted_filter: Optional[set] = None,
) -> Dict:
    """One ``enum_children.json`` hub entry from a hub's enumerated child records.

    ``recs`` are record-dicts (:data:`REC_COLS`) for the hub's one-step children. ``added_by_child``
    / ``reaction_by_child`` (keyed by the child's stereo key) carry the fragment added + the final
    reaction step(s) for the ``free_frag`` child policy + route reconstruction; both empty is fine
    for models with no promoted fragments (the ``reward`` child policy ignores them). Mirrors the
    ``scent_worker`` enumerate output so ``run_campaign.EnumeratedHub`` loads it unchanged."""
    added_by_child = added_by_child or {}
    reaction_by_child = reaction_by_child or {}
    children = []
    for r in recs:
        skey = r.get("child_stereo_key") or r["child_key"]
        added = added_by_child.get(skey, [])
        if promoted_filter is not None:
            added = [f for f in added if f in promoted_filter]
        children.append(
            {
                "smiles": r["child_key"],
                "reward": r["reward"],
                "added_promoted": added,
                "reaction": reaction_by_child.get(skey, []),
            }
        )
    u_h, n_eff = hub_uncertainty(recs)
    return {
        "hub_input": hub_input,
        "hub_key": hub_key,
        "depth": int(depth),
        "uncertainty": None if u_h != u_h else u_h,  # NaN -> null
        "n_effective": n_eff,
        "children": children,
    }


def write_enum_children(path, enum_hubs: Sequence[Dict]) -> None:
    """Write enum_children.json = ``{"hubs": [...]}`` (the campaign's EnumeratedHub list)."""
    json.dump({"hubs": list(enum_hubs)}, open(path, "w"))


def compositions_from_records(records: Sequence[Dict]) -> Dict[str, dict]:
    """Build compositions.json for a generator with NO promoted fragments (RGFN/FragGFN/RxnFlow).

    SCENT populates compositions with each molecule's fully-nested ``num_reactions`` (its dynamic
    library builds). The reaction/fragment baselines have no promotion, so without this the campaign
    falls back to ``num_reactions=1`` per molecule — which under-charges best-candidate and makes the
    count-once comparison meaningless. Here we charge each molecule its FLAT build depth: a terminal
    child is ``hub_depth + 1`` reactions, a hub is ``hub_depth``. Keyed by the cross-model SMILES key;
    keeps the cheapest depth seen (a molecule reachable by a shorter route costs the shorter one).
    ``promoted`` is always empty (no dynamic-library fragments to nest)."""
    comps: Dict[str, dict] = {}

    def put(key: str, nr: int) -> None:
        if key and (key not in comps or nr < comps[key]["num_reactions"]):
            comps[key] = {"num_reactions": int(nr), "promoted": []}

    for r in records:
        put(r["child_key"], int(r["hub_depth"]) + 1)
        put(r["hub_key"], int(r["hub_depth"]))
    return comps


def write_json(path, obj) -> None:
    json.dump(obj, open(path, "w"), indent=2)


# ===================================================== measured compute time (Logs/039)
# The paper's third cost axis is MEASURED wall-clock, differentiated per component, not a model
# (see the lsdflow-compute-time-measured-not-modeled note). ``scent_worker`` inlined this first;
# these two helpers are the portable version so every generator emits the SAME enum_timings.json
# that ``validation/lsdflow/metrics/cost/compute_time.EnumTimings`` reads back.

TIMING_COMPONENTS = ("enumeration_s", "reward_gen_s", "flow_extract_s", "unattributed_s")


class ComponentTimer:
    """Accumulates per-hub wall-clock split across the enumeration pipeline's components.

    The three real components are the ones a strategy's compute is attributed to:

      * ``enumeration_s``  — building the hub's one-step children (RDKit/graph ops, CPU),
      * ``reward_gen_s``   — the reward generator scoring them (proxy now, docking oracle later —
                             the axis that explodes for docking because it *actually* takes longer),
      * ``flow_extract_s`` — the policy forward/backward passes giving P_F / P_B (mostly GPU).

    ``unattributed_s`` is the honest escape hatch for a generator whose enumeration is a single
    opaque call (RGFN goes through ``glue/samplers/lsdflow/rgfn_enumerate``): the total is still
    measured correctly, but we refuse to invent a split. It plots as its own labelled segment.

    ``sync`` is a no-arg callable invoked at both ends of every tracked block — pass
    ``torch.cuda.synchronize`` on GPU so async kernel time lands on the component that launched it,
    and ``None`` on CPU. Without it, GPU work drifts onto whichever component next forces a sync.

    Usage::

        t = ComponentTimer(sync=torch.cuda.synchronize if cuda else None)
        with t.track("enumeration_s"):
            children = build(hub)
        with t.track("reward_gen_s"):
            rewards = proxy.predict(children)
        per_hub.append({..., **t.snapshot()})   # then t.reset() for the next hub
    """

    KEYS = TIMING_COMPONENTS

    def __init__(self, sync=None):
        self._sync = sync if callable(sync) else (lambda: None)
        self.totals: Dict[str, float] = {k: 0.0 for k in self.KEYS}

    @contextmanager
    def track(self, key: str):
        if key not in self.totals:
            raise KeyError(f"unknown timing component {key!r}; expected one of {self.KEYS}")
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.totals[key] += time.perf_counter() - t0

    def add(self, key: str, seconds: float) -> None:
        """Fold in a duration measured elsewhere (e.g. inside a helper that already timed itself)."""
        self.totals[key] = self.totals.get(key, 0.0) + float(seconds)

    def merge(self, other: Dict[str, float]) -> None:
        for k, v in (other or {}).items():
            if k in self.totals:
                self.totals[k] += float(v)

    def snapshot(self) -> Dict[str, float]:
        """The accumulated components, rounded — drops always-zero keys so a fully-attributed
        generator's file has no confusing empty ``unattributed_s`` column."""
        return {k: round(v, 6) for k, v in self.totals.items() if v}

    def reset(self) -> None:
        for k in self.totals:
            self.totals[k] = 0.0


def write_enum_timings(
    path,
    *,
    per_hub: Sequence[Dict],
    setup_s: float,
    device: str = "",
    reward_name: str = "",
    model: str = "",
    cuda_synchronized: bool = False,
    component_split: str = "full",
    extra_meta: Optional[Dict] = None,
) -> Dict:
    """Write enum_timings.json — the measured compute-time sidecar, joined to enum_children by hub.

    ``per_hub`` rows MUST carry ``hub_input`` (the stereo-aware join key ``EnumTimings._hub_id``
    prefers, so stereoisomeric hubs that share a stripped ``hub_key`` stay distinct), ``hub_key``,
    ``depth``, ``n_children``, and whichever of :data:`TIMING_COMPONENTS` were measured.

    ``component_split="full"`` means all three real components were timed separately;
    ``"lumped"`` means the generator could only measure a per-hub total (recorded as
    ``unattributed_s``) — stated in the file rather than silently implied by a zero column.

    When a 200-hub run is split across parallel jobs, merge the slices with
    ``EnumTimings.merge`` (unions ``per_hub``, charges ``setup_s`` once).
    """
    keys = [k for k in TIMING_COMPONENTS if any(h.get(k) for h in per_hub)]
    totals = {k: round(sum(float(h.get(k, 0.0)) for h in per_hub), 3) for k in keys}
    meta = {
        "setup_s": round(float(setup_s), 3),
        "device": str(device),
        "cuda_synchronized": bool(cuda_synchronized),
        "reward_name": reward_name,
        "model": model,
        "component_split": component_split,
        "n_hubs": len(per_hub),
        "n_children": sum(int(h.get("n_children", 0)) for h in per_hub),
        "totals_s": totals,
    }
    meta.update(extra_meta or {})
    write_json(path, {"meta": meta, "per_hub": list(per_hub)})
    return meta
