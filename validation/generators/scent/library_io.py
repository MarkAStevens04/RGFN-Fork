"""Persist / restore SCENT's **dynamic fragment library** across a checkpoint resume.

Why this exists
---------------
SCENT promotes high-value intermediates into a growing fragment vocabulary on a fixed
schedule (``every_n_iterations`` x ``num_additions``). ``Trainer.make_checkpoint`` saves
``model / optimizer / lr_scheduler / metrics / replay_buffer`` and **not** the library, and
``Trainer.__init__``'s resume path restores exactly those five. The library is only ever
serialised to ``additional_fragments/fragments_<N>.json`` for *analysis*, and never read back.

So a requeued run restarts with an **empty** library. Measured in v1: ``scent_6td3_5k/seed42``
progressed 400/800/**400**/800 (two resets) and ``scent_clpp_5k/seed43,44`` 400/**400**/800/1200
(one reset), against a clean 1,600 for cells that happened to fit one walltime. The final
promoted-library size was therefore a function of *requeue timing*, not of the science.

The half that is worse, and silent
----------------------------------
``FragmentOneHotEmbedding.weights`` is preallocated to ``418 + max_num_additional_fragments``
rows and ``_get_embeddings()`` returns ``weights[: current_fragments]`` -- indexed **by
position** -- while ``on_update_fragments_library`` updates only the *count*. After a reset a
re-promoted fragment therefore lands on a slot whose trained row belongs to the slot's previous
occupant. Nothing crashes; the fingerprint half rebuilds correctly, so only the one-hot half
carries stale identity.

**The trained rows were never lost, only made invisible.** ``weights`` is an ``nn.Parameter``
inside the forward policy, so ``model.load_state_dict`` restores all of them on resume; it is
``current_fragments`` -- a plain Python attribute reset to ``initial_fragments`` in
``__init__`` -- that hides everything past row 418. That is why this fix is a restore of
*bookkeeping* rather than a second weight sidecar: put the ordered fragment list back and the
rows it points at are already correct.

This is not a deviation from SCENT's design, it restores it: the one-hot table is sized
``418 + num_additions * n_new_fragments`` = 418 + 4,000 = **4,418** rows, i.e. built for a
complete, never-reset run. The reset is an artifact of our 3-day requeue, not of the method.

How
---
The clone (``external/scent``) is a pinned, git-ignored dependency, so -- exactly as
``guidance_io.py`` does for the ``P_B`` guidance models -- we capture the missing state from our
own adapter rather than patching it: a ``dynamic_library.json`` sidecar beside ``last_gfn.pt``,
written with every checkpoint and read back on resume.

Two fields beyond the library's own ``state_dict()``
----------------------------------------------------
``state_dict()`` already carries what the *restore* needs -- ``chosen_smiles`` is a list, so
promotion **order** (which the positional embedding depends on) is preserved -- but it omits two
things that the *next* promotion needs:

* ``smiles_to_cost`` -- a ``defaultdict(inf)`` accumulated during sampling. It is NOT in
  ``state_dict()``, and ``retrieve_all_additional_fragments`` reads it for the cost of every
  newly promoted fragment. After a resume a candidate can be present in the restored
  ``smiles_to_mean_reward`` (so it is eligible) while never having been re-sampled, which would
  promote it at cost ``inf``. So it is persisted here.
* ``chosen_smiles_fps`` -- rebuilt from ``chosen_smiles`` rather than stored, since it is a pure
  function of them. Only ``_is_molecule_different`` reads it, and that short-circuits when
  ``similarity_threshold == 1.0`` (our configured value), but rebuilding costs milliseconds and
  keeps the restore correct if that threshold ever moves.

Sidecar format::

    {"library": <DynamicLibrary.state_dict()>, "smiles_to_cost": {smiles: float}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Keys of `state_dict()` that are plain containers we can assign straight back.
_LIST_KEYS = ("chosen_smiles", "chosen_smiles_costs")
# Keys that must go back as defaultdicts, not plain dicts: the library indexes them with
# unseen SMILES during sampling and a plain dict would raise KeyError mid-training.
_DEFAULTDICT_KEYS = {
    "smiles_to_mean_reward": 0.0,
    "smiles_to_count": 0,
    "smiles_to_min_num_reactions": 100000000,
}


def save_library_state(dynamic_library, path: Path | str) -> Optional[int]:
    """Write the sidecar. Returns the number of promoted fragments, or ``None`` if skipped.

    Written UNCONDITIONALLY, including when the library is empty. An empty artifact and a
    missing artifact must not look identical -- this project has now been bitten by that three
    times (an empty ``routes.json`` priced as a free library, absent recipes read as complete,
    and a header-only trace). A zero-fragment sidecar says "this run had promoted nothing yet";
    no sidecar says "nobody saved one".
    """
    if dynamic_library is None:
        return None
    path = Path(path)
    try:
        state = dynamic_library.state_dict()
        body = {
            "library": state,
            # See the module docstring: needed by the NEXT promotion, absent from state_dict().
            "smiles_to_cost": dict(getattr(dynamic_library, "smiles_to_cost", {}) or {}),
            "candidate_routes": _candidate_routes(dynamic_library),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body))
        tmp.replace(path)  # atomic: a kill mid-write never leaves a truncated sidecar
        return len(state.get("chosen_smiles", []))
    except Exception as exc:  # noqa: BLE001 - never fail a good training run over provenance
        print(f"[SCENT-FR] WARNING dynamic-library save failed: {exc}", flush=True)
        return None


def _candidate_routes(dynamic_library) -> Dict[str, Any]:
    """Recipe routes for every molecule that could still be PROMOTED, not just those already were.

    `recipe_logging` patches `state_dict()` to embed routes for `chosen_smiles` only. Restoring just
    those is not enough, and the gap is measured rather than theorised: on an 8-iteration resume test
    the uninterrupted run ended 15 promoted / 15 routed, while the resumed run ended 15 promoted /
    **5** routed. Restoring the promoted subset lifts that to 10/15 -- the remaining 5 are molecules
    first SEEN before the stop and promoted after it, whose routes lived only in the full in-memory
    `_smiles_to_route`. A cell like that fails `check_route_readiness`, which is a hard gate.

    So the persisted set is the PROMOTABLE CANDIDATE set: everything the promotion filter could still
    pick, i.e. depth <= `max_num_reactions` and not already an initial fragment. Measured cost ~394 B
    per route and ~53% of seen molecules promotable, so ~39 MB for a 5,000-iteration cell -- against
    the 91 MB `fragments_4000.json` this pipeline already writes at every promotion. Affordable, so
    full coverage is the right call rather than a documented shortfall.
    """
    routes = getattr(dynamic_library, "_smiles_to_route", None)
    if not routes:
        return {}
    try:
        max_rxn = dynamic_library.max_num_reactions
        depth = dynamic_library.smiles_to_min_num_reactions
        initial = dynamic_library.initial_smiles_set
        out = {
            s: r for s, r in routes.items() if s not in initial and depth.get(s, 1 << 30) <= max_rxn
        }
        # Loud rather than silent if this ever grows past what a per-checkpoint write should cost.
        if len(out) > 500_000:
            print(
                f"[SCENT-FR] WARNING candidate_routes is {len(out):,} entries; the sidecar write "
                f"may be slow. Nothing is dropped -- this is a heads-up, not a cap.",
                flush=True,
            )
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[SCENT-FR] WARNING could not collect candidate routes: {exc}", flush=True)
        return {}


def load_library_state(
    dynamic_library, path: Path | str
) -> Optional[Tuple[List[str], List[float]]]:
    """Restore the library's own bookkeeping from the sidecar.

    Returns ``(chosen_smiles, chosen_smiles_costs)`` so the caller can rebuild the model-side
    state, or ``None`` if there was nothing to restore. Does NOT touch the model -- that is
    :func:`reapply_library_to_model`, kept separate because it needs the trainer.
    """
    from collections import defaultdict

    path = Path(path)
    if dynamic_library is None or not path.exists():
        return None
    try:
        body = json.loads(path.read_text())
        state = body.get("library", {})
        chosen = list(state.get("chosen_smiles", []))

        for k in _LIST_KEYS:
            if k in state:
                setattr(dynamic_library, k, list(state[k]))
        for k, default in _DEFAULTDICT_KEYS.items():
            if k in state:
                d = defaultdict(lambda dv=default: dv)
                d.update(state[k])
                setattr(dynamic_library, k, d)
        if "initial_smiles_set" in state:
            dynamic_library.initial_smiles_set = set(state["initial_smiles_set"])
        if body.get("smiles_to_cost"):
            d = defaultdict(lambda: float("inf"))
            d.update(body["smiles_to_cost"])
            dynamic_library.smiles_to_cost = d

        # RECIPE ROUTES for the promoted fragments. `recipe_logging` patches state_dict() to embed
        # `smiles_to_route` for exactly `chosen_smiles`, so the sidecar already carries them -- they
        # were simply never read back. Merged rather than assigned, because a resumed process may
        # already have re-seen some of these molecules and re-derived their routes.
        #
        # WHAT THIS DOES AND DOES NOT COVER, stated because the difference decides whether a cell
        # passes check_route_readiness. It restores routes for every fragment promoted BEFORE the
        # stop, which is what the route dataset ships and what expands a promoted fragment inside a
        # molecule's route. It does NOT restore the full in-memory `_smiles_to_route` (852 entries
        # after four smoke iterations; a 5,000-iteration cell would be hundreds of MB written at
        # every checkpoint), so a molecule first seen pre-resume and promoted post-resume gets a
        # route only if it is re-sampled after the resume. In practice it usually is -- the policy
        # keeps sampling the same high-reward region -- but that is a likelihood, not a guarantee,
        # which is why check_route_readiness stays a hard gate per cell rather than an assumption.
        routes = dict(state.get("smiles_to_route") or {})
        routes.update(body.get("candidate_routes") or {})
        if routes:
            existing = getattr(dynamic_library, "_smiles_to_route", None)
            if not isinstance(existing, dict):
                existing = {}
                dynamic_library._smiles_to_route = existing
            for smi, route in routes.items():
                existing.setdefault(smi, route)
            print(
                f"[SCENT-FR] resume: restored {len(routes)} recipe routes "
                f"(promoted + still-promotable candidates)",
                flush=True,
            )

        # Pure function of chosen_smiles, so rebuilt rather than stored. Order matters: it is
        # zipped against chosen_smiles by index inside _is_molecule_different's similarity list.
        try:
            dynamic_library.chosen_smiles_fps = [dynamic_library._get_fp(s) for s in chosen]
        except Exception as exc:  # noqa: BLE001
            print(f"[SCENT-FR] WARNING could not rebuild chosen_smiles_fps: {exc}", flush=True)

        return chosen, list(state.get("chosen_smiles_costs", []))
    except Exception as exc:  # noqa: BLE001
        print(f"[SCENT-FR] WARNING dynamic-library load failed: {exc}", flush=True)
        return None


def reapply_library_to_model(trainer, dynamic_library, iteration_idx: int = 0) -> int:
    """Replay the promoted fragments into the model, so the restored rows become visible again.

    This calls the SAME hook a live promotion calls, with the SAME arguments, which is what makes
    it faithful rather than a reimplementation: ``on_update_fragments_library`` fans out to the
    environment (rebuilds the fragment list, asserts ``idx == f.idx``, clears the action-space and
    recurrence caches), to the action embeddings (sets ``current_fragments`` and appends the
    fingerprint rows), and to the path-cost proxy (``fragment_to_cost``). Reproducing any of that
    by hand would be a second implementation to keep in sync.

    The ``Molecule`` list is constructed exactly as ``retrieve_all_additional_fragments`` does --
    ``idx = len(initial_smiles_set) + i`` over ``enumerate(chosen_smiles)`` -- because the
    environment asserts positional identity and the one-hot embedding is positionally indexed.
    Restoring the same SET in a different ORDER would reproduce the stale-identity bug from a new
    cause, which is why ``chosen_smiles`` must stay a list all the way through.

    Returns the number of fragments reapplied.
    """
    chosen = list(getattr(dynamic_library, "chosen_smiles", []) or [])
    if not chosen:
        return 0
    from rgfn.gfns.reaction_gfn.api.reaction_api import Molecule

    n_initial = len(dynamic_library.initial_smiles_set)
    molecules = [
        Molecule(
            smiles,
            idx=n_initial + i,
            num_reactions=dynamic_library.smiles_to_min_num_reactions[smiles],
        )
        for i, smiles in enumerate(chosen)
    ]
    costs = list(getattr(dynamic_library, "chosen_smiles_costs", []) or [])
    if len(costs) != len(molecules):  # tolerate a truncated sidecar rather than crash the resume
        print(
            f"[SCENT-FR] WARNING library costs ({len(costs)}) != fragments ({len(molecules)}); "
            f"padding with inf so the resume proceeds",
            flush=True,
        )
        costs = (costs + [float("inf")] * len(molecules))[: len(molecules)]
    trainer.on_update_fragments_library(iteration_idx, molecules, costs)
    _seed_path_cost_cache(dynamic_library, chosen, costs)
    _assert_slot_alignment(trainer, dynamic_library, n_initial, len(molecules))
    return len(molecules)


def _assert_slot_alignment(trainer, dynamic_library, n_initial: int, n_restored: int) -> None:
    """ASSERT the invariant the whole restore rests on: slot order == fingerprint order == list order.

    Documented is not enough, because this is silent when it breaks -- a misaligned restore produces
    the right fragment COUNT pointing at the wrong trained rows, which is the original bug wearing
    the fix's clothes. ``on_update_fragments_library`` appends fingerprints for
    ``fragments[-n_new:]`` where ``n_new = n_initial + len(fragments) - len(all_fingerprints)``, so
    the two counts agreeing is what says the append consumed the whole restored list in order.
    Raises rather than warns: a silently misaligned library is worse than a failed resume.
    """
    expected = n_initial + n_restored
    emb = _find_one_hot(trainer.objective)
    if emb is not None and int(emb.current_fragments) != expected:
        raise RuntimeError(
            f"library restore misaligned: current_fragments={emb.current_fragments}, "
            f"expected {expected} ({n_initial} initial + {n_restored} restored)"
        )
    for m in trainer.objective.modules():
        fps = getattr(m, "all_fingerprints", None)
        if fps is not None and len(fps) != expected:
            raise RuntimeError(
                f"library restore misaligned: {type(m).__name__}.all_fingerprints has "
                f"{len(fps)} rows, expected {expected}"
            )


def _seed_path_cost_cache(dynamic_library, chosen: List[str], costs: List[float]) -> int:
    """Re-seed ``PathCostProxy.molecule_num_reaction_to_cost`` for the restored fragments.

    THE THIRD PIECE OF STATE, and it is not optional -- without it a resumed run CRASHES, which is
    how it was found rather than reasoned about::

        reaction_dynamic_library.py:118, on_end_sampling
        current_cost = self.path_cost_proxy.molecule_num_reaction_to_cost[(smiles, n)]
        self.smiles_to_cost[smiles] = min(self.smiles_to_cost[smiles], current_cost)
        TypeError: '<' not supported between instances of 'NoneType' and 'float'

    ``molecule_num_reaction_to_cost`` is a ``Cache`` that returns **None** on a miss, and
    ``assign_costs`` only fills it for states reached through a reaction action. Restoring the
    library puts the promoted fragments back in the action space, so the policy can immediately
    select one as a *starting* block -- producing a state whose ``(smiles, num_reactions)`` key was
    never populated in THIS process. In an uninterrupted run the entry exists because the fragment
    was built as a product earlier in the same process; across a resume it does not.

    (The clone guards the same lookup two lines away in ``path_cost_proxy`` with ``or float("inf")``
    and does not guard it here. We restore the real value rather than papering over it with inf,
    because the cost is what the NEXT promotion records for that fragment.)
    """
    proxy = getattr(dynamic_library, "path_cost_proxy", None)
    cache = getattr(proxy, "molecule_num_reaction_to_cost", None)
    if cache is None:
        return 0

    install_cache_miss_guard(proxy)
    n = 0
    for smiles, cost in zip(chosen, costs):
        try:
            num_reactions = dynamic_library.smiles_to_min_num_reactions[smiles]
            cache[(smiles, num_reactions)] = float(cost)
            n += 1
        except Exception:  # noqa: BLE001 - one unseeded key must not abort the resume
            continue
    return n


def install_cache_miss_guard(path_cost_proxy) -> bool:
    """Make ``molecule_num_reaction_to_cost`` return ``inf`` on a miss instead of ``None``.

    Installed on EVERY resume, not only when fragments were restored, because "the library happened
    to be empty" is not a reason to leave the landmine armed -- and an empty library is exactly the
    state v1 resumed into, which is the only reason v1 never hit this.

    DEFENDS THE LOOKUP ITSELF, not just the keys we can predict. Seeding the restored fragments'
    # own (smiles, num_reactions) keys is necessary but NOT sufficient, which is how the first
    # attempt at this failed: a restored fragment reappears inside a NEW trajectory at a state whose
    # `num_reactions` is not the depth it was originally built at, so the key we can compute up
    # front does not match the key the library looks up. Rather than enumerate keys we cannot
    # predict, make a miss return `inf` -- which is EXACTLY what the clone itself does for the same
    # cache two lines away in path_cost_proxy (`self.molecule_num_reaction_to_cost[item] or
    # float("inf")`) and simply forgot to do in reaction_dynamic_library:118. So this is an upstream
    # inconsistency we are compensating for from our adapter, not a semantic we invented.
    #
    # `inf` is the right miss value, not a fudge: `smiles_to_cost` is a defaultdict(inf) combined
    # with `min(...)`, so an unknown cost must be the identity for that min. And it cannot mask the
    # real value for the restored fragments, because those keys are seeded below with their true
    # costs, which is what the NEXT promotion records.
    SAFE FOR EVERY OTHER CONSUMER, checked rather than assumed: only ``__getitem__`` is overridden,
    so ``__contains__`` is untouched, and both guidance models gate on ``item in dataset`` before
    indexing (``[dataset[item] for item in ... if item in dataset]``). The one place that indexes
    unguarded already writes ``or float("inf")`` around it, so it sees the same value either way.
    Nothing in the clone distinguishes ``None`` from a value.
    """
    cache = getattr(path_cost_proxy, "molecule_num_reaction_to_cost", None)
    if cache is None or getattr(cache, "_bench_v2_miss_guarded", False):
        return False
    _orig_getitem = type(cache).__getitem__

    def _guarded(self, key, _orig=_orig_getitem):
        val = _orig(self, key)
        return float("inf") if val is None else val

    cache.__class__ = type(
        f"{type(cache).__name__}MissGuarded",
        (type(cache),),
        {"__getitem__": _guarded, "_bench_v2_miss_guarded": True},
    )
    return True


def audit_state(trainer, dynamic_library) -> Dict[str, Any]:
    """A comparable digest of EVERY attribute of the library, the cost proxy and the replay buffer.

    WHY A BLIND WALK RATHER THAN A LIST OF FIELDS. The three pieces of state this restore needs were
    found by two different ad-hoc methods -- two by reading the code, one by crashing into it -- and
    neither method can show there is no fourth. Enumerating the fields we happened to think of and
    comparing those would reproduce exactly that blind spot. So this walks ``__dict__`` and digests
    whatever it finds, and the verifier diffs the digests: anything that differs between an
    uninterrupted run and a resumed one is either state that must be persisted, or state we can then
    say provably does not matter. That turns "we found three by hitting them" into "we checked every
    attribute, and here is the complete list".

    Values are reduced to something order-sensitive and comparable rather than stored whole:
    containers become ``(type, len, sha1)`` over a canonical repr, tensors become a sha1 of their
    bytes, scalars stay themselves, and anything opaque records only its type name -- flagged
    ``opaque`` so a reader knows it was seen and not compared, rather than silently skipped.
    """
    out: Dict[str, Any] = {}
    targets = [("library", dynamic_library)]
    proxy = getattr(dynamic_library, "path_cost_proxy", None)
    if proxy is not None:
        targets.append(("path_cost_proxy", proxy))
    # The replay buffer IS in the checkpoint, so a positional reference to fragment identity there
    # would survive a resume pointing at the wrong fragment -- the same shape of bug as the one in
    # the embedding. Audited so that question is answered by measurement rather than assumption.
    rb = getattr(trainer, "train_replay_buffer", None)
    if rb is not None:
        targets.append(("replay_buffer", rb))

    for group, obj in targets:
        for key, val in sorted(vars(obj).items()):
            out[f"{group}.{key}"] = _digest(val)
    return out


def _digest(val) -> Any:
    """Reduce a value to something comparable across processes. Order-sensitive by construction."""
    import hashlib

    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    try:
        import torch

        if isinstance(val, torch.Tensor):
            arr = val.detach().cpu().contiguous()
            return {
                "type": "Tensor",
                "shape": list(arr.shape),
                "sha1": hashlib.sha1(arr.numpy().tobytes()).hexdigest(),
            }
    except Exception:  # noqa: BLE001
        pass
    if isinstance(val, (list, tuple)):
        # Order-sensitive: a reordered list is a DIFFERENT state here, which is the point.
        return {"type": type(val).__name__, "len": len(val), "sha1": _sha_repr(val)}
    if isinstance(val, set):
        return {"type": "set", "len": len(val), "sha1": _sha_repr(val)}
    if isinstance(val, dict):
        # via _stable_repr, NOT str(): str() of an object is its address, which is what made the
        # first version of this audit report differences that were pure identity noise.
        return {"type": "dict", "len": len(val), "sha1": _sha_repr(val)}
    return {"type": type(val).__name__, "opaque": True}


def _sha_repr(obj) -> str:
    """sha1 of a CONTENT repr -- never one containing a memory address.

    A plain ``repr()`` was the first attempt and it silently compared identities: an RDKit
    fingerprint reprs as ``<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at 0x7feb...>``,
    so two byte-identical fingerprint lists hashed differently on every run. That made the audit
    report permanent false positives on ``chosen_smiles_fps`` and ``anchor_to_reaction`` -- the
    worst failure mode for a check, because a diff that always fires teaches its reader to ignore
    it. Content-bearing reprs are used where the object offers one, and anything left is reported
    as NOT COMPARABLE rather than as differing.
    """
    import hashlib

    try:
        return hashlib.sha1(_stable_repr(obj).encode()).hexdigest()
    except Exception:  # noqa: BLE001
        return "unhashable"


def _stable_repr(obj) -> str:
    """A content-based string for ``obj``, recursing into containers. ``<opaque:Type>`` if none."""
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_stable_repr(x) for x in obj) + "]"
    if isinstance(obj, dict):
        # SORTED: a dict's insertion order is not part of its state, and after a restore it is
        # rebuilt in a different order. Hashing it unsorted made `anchor_to_reaction` report a
        # permanent false difference at identical length and content.
        items = sorted((_stable_repr(k), _stable_repr(v)) for k, v in obj.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(obj, (set, frozenset)):
        return "{" + ",".join(sorted(_stable_repr(x) for x in obj)) + "}"
    # Content accessors, most specific first. ToBase64/ToBitString cover the RDKit bit vectors that
    # made this necessary; `smiles` covers Molecule; `state_dict` covers nn.Modules.
    for attr in ("ToBase64", "ToBitString", "state_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return f"{type(obj).__name__}:{_stable_repr(fn())}"
            except Exception:  # noqa: BLE001
                pass
    for attr in ("smiles", "name", "value"):
        v = getattr(obj, attr, None)
        if isinstance(v, (str, int, float)):
            return f"{type(obj).__name__}:{v}"
    r = repr(obj)
    # The tell for an identity repr. Do not hash it -- say it cannot be compared.
    return f"<opaque:{type(obj).__name__}>" if " at 0x" in r else r


def library_fingerprint(trainer, dynamic_library) -> Dict[str, Any]:
    """A small, comparable summary for the resume verification (and for arm_a.json).

    Deliberately includes a HASH OF THE EMBEDDING ROWS, not just a count. A count check passes on
    exactly the bug this module fixes -- the same number of fragments pointing at the wrong
    trained rows -- so the count alone would certify the defect as healthy.
    """
    import hashlib

    out: Dict[str, Any] = {
        "n_chosen": len(getattr(dynamic_library, "chosen_smiles", []) or []),
        "chosen_smiles_sha1": None,
        "current_fragments": None,
        "embedding_rows_sha1": None,
    }
    chosen = getattr(dynamic_library, "chosen_smiles", []) or []
    if chosen:
        out["chosen_smiles_sha1"] = hashlib.sha1("\n".join(chosen).encode()).hexdigest()
    try:
        emb = _find_one_hot(trainer.objective)
        if emb is not None:
            out["current_fragments"] = int(emb.current_fragments)
            rows = emb.weights[: emb.current_fragments].detach().cpu().contiguous()
            out["embedding_rows_sha1"] = hashlib.sha1(rows.numpy().tobytes()).hexdigest()
    except Exception as exc:  # noqa: BLE001
        print(f"[SCENT-FR] WARNING could not fingerprint embeddings: {exc}", flush=True)
    return out


def _find_one_hot(module):
    """First ``FragmentOneHotEmbedding`` under ``module`` (it is nested inside the fwd policy)."""
    for m in module.modules():
        if type(m).__name__ == "FragmentOneHotEmbedding":
            return m
    return None
