"""Log HOW each SCENT dynamic-library fragment is synthesized (routes), during training.

Motivation (LSD-Flow cost accounting + chemist usability): SCENT promotes high-reward
intermediates into its fragment vocabulary. To (a) charge each promoted fragment's *own* build
cost **exactly once** in the reactions-per-mode / amortization metric — including **nested**
fragments-built-from-fragments — and (b) later show a chemist a "synthesize these intermediates"
route, we must know each promoted fragment's actual synthesis recipe. That recipe is only
observable *during training* (once promoted, the model uses the fragment atomically and its
internal assembly is hidden). The build cost in *reactions* (``min_num_reactions``) and SCENT's
$-cost (``chosen_smiles_costs``) are already saved in ``fragments_<N>.json``; this adds the
**route** (the ordered reaction steps).

Kept out of the pristine SCENT clone by **monkeypatching** ``DynamicLibrary`` from our adapter
layer (same spirit as the entry-024 ``guidance_io`` sidecar): :func:`enable_recipe_logging`
wraps ``on_end_sampling`` (capture the min-reaction route per molecule, from the ``ReactionActionC``
commit actions of the trajectory that built it) and ``state_dict`` (serialize routes for the
promoted ``chosen_smiles`` so they land in the ``fragments_<N>.json`` snapshot). No gin config
change; no clone edit. Call it once before training (see ``run_scent_fixed.py --log-recipes``).

Route schema (per promoted fragment, display-ready for a future "intermediates" tab)::

    { "seed": <starting building-block SMILES>,
      "num_reactions": k,
      "steps": [ {"reaction": <id>, "reactants": [<smiles>...], "input": <smiles>, "product": <smiles>}, ... ] }

Each ``reactants`` entry is a library building block (base or an earlier-promoted fragment); the
post-hoc cost model classifies base vs dynamic (by membership in ``initial_smiles_set`` /
``chosen_smiles``) and recurses into dynamic ones for exact nested cost.
"""

from __future__ import annotations


def _reaction_id(reaction) -> str:
    """A readable, stable id for an AnchoredReaction (name/smarts if available, else repr)."""
    for attr in ("name", "reaction_name", "smarts"):
        v = getattr(reaction, attr, None)
        if v:
            return str(v)
    inner = getattr(reaction, "reaction", None)
    if inner is not None:
        for attr in ("name", "smarts"):
            v = getattr(inner, attr, None)
            if v:
                return str(v)
    return str(reaction)


def capture_routes(dl, trajectories_container) -> None:
    """Record each seen molecule's MIN-reaction synthesis route onto ``dl._smiles_to_route``.

    Module-level (not a closure) so it is unit-testable on a sampled batch independent of the
    base ``on_end_sampling`` bookkeeping. ``dl`` is the ``DynamicLibrary`` instance; it must carry
    ``use_forward_only``, ``initial_smiles_set``, and ``_smiles_to_route``."""
    from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionActionC, ReactionState0

    trajectories = (
        trajectories_container.forward_trajectories
        if dl.use_forward_only
        else trajectories_container.get_all_non_backward_trajectories()
    )
    mask = [isinstance(s, ReactionState0) for s in trajectories.get_source_states_flat()]
    trajectories = trajectories.masked_select(mask)
    routes = dl._smiles_to_route
    for actions in trajectories._actions_list:
        steps = []
        for act in actions:
            if not isinstance(act, ReactionActionC):
                continue
            steps.append(
                {
                    "reaction": _reaction_id(act.input_reaction),
                    "reactants": [f.smiles for f in act.input_fragments],
                    "input": act.input_molecule.smiles,
                    "product": act.output_molecule.smiles,
                }
            )
            product = act.output_molecule.smiles
            nrx = len(steps)
            # Record the MIN-reaction route (matches how smiles_to_min_num_reactions is kept).
            if product not in dl.initial_smiles_set:
                stored = routes.get(product)
                if stored is None or nrx < stored["num_reactions"]:
                    routes[product] = {
                        "seed": steps[0]["input"],
                        "num_reactions": nrx,
                        "steps": list(steps),
                    }


def enable_recipe_logging() -> None:
    """Monkeypatch ``DynamicLibrary`` to record each promoted fragment's synthesis route.

    Idempotent. Route capture is wrapped in try/except so it can never break a training run."""
    from rgfn.gfns.reaction_gfn.dynamic_library.reaction_dynamic_library import (
        DynamicLibrary,
    )

    if getattr(DynamicLibrary, "_recipe_logging_enabled", False):
        return

    _orig_on_end = DynamicLibrary.on_end_sampling
    _orig_state_dict = DynamicLibrary.state_dict

    def on_end_sampling(self, iteration_idx, trajectories_container, recursive=True):
        out = _orig_on_end(self, iteration_idx, trajectories_container, recursive=recursive)
        if not hasattr(self, "_smiles_to_route"):
            self._smiles_to_route = {}
        try:
            capture_routes(self, trajectories_container)
        except Exception as exc:  # noqa: BLE001 - never let route capture break training
            print(f"[recipe_logging] route capture skipped this iteration: {exc}", flush=True)
        return out

    def state_dict(self):
        d = _orig_state_dict(self)
        routes = getattr(self, "_smiles_to_route", {})
        # Only the promoted fragments' routes (bounded ~n_chosen); recursion into dynamic
        # reactants is closed because every chosen fragment has its own route here.
        d["smiles_to_route"] = {s: routes[s] for s in self.chosen_smiles if s in routes}
        return d

    DynamicLibrary.on_end_sampling = on_end_sampling
    DynamicLibrary.state_dict = state_dict
    DynamicLibrary._recipe_logging_enabled = True
    print("[recipe_logging] DynamicLibrary route capture enabled", flush=True)
