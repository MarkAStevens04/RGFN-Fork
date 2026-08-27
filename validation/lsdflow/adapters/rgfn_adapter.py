"""In-process RGFN adapter — the LSD-Flow anchor (``docs/LSD_FLOW_PROPOSAL.md`` §4b, §10 step 1).

RGFN is native to the ``rgfn`` env this analysis runs in, so its worker is in-process (no
RPC). It rebuilds the trained objective + a **pure-policy** sampler from a gin config +
checkpoint (the ``validation/generators/scent/verify_pb_recovery.py`` recipe), samples
trajectories from the trained forward policy with rewards attached, and composes each
terminal transition's §2 log-terms via the shared, rgfn-native
:func:`glue.samplers.lsdflow.rgfn_extract.extract_flow_records`.

Why the ``valid_sampler`` and not the training sampler: the training forward sampler is an
``ExploratoryPolicy`` (95% trained policy + 5% uniform); the ``valid`` sampler uses the
**pure** trained ``forward_policy`` (``configs/samplers/random.gin``), which is the policy
``assign_log_probs`` scores against — so the sampled distribution and the recovered ``P_F``
match, which is what makes ``U(h)`` a clean flow-matching residual (§5).

For RGFN the learned backward policy's ``mlp_c`` is a normal ``nn.Module`` attribute, so it is
in ``last_gfn.pt`` and recovers exactly on ``load_state_dict`` — no sidecar needed (that is a
SCENT-only issue, §5/§9).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import gin
import torch

import glue  # noqa: F401  (registers our gin-configurable components)
from glue.samplers.lsdflow.rgfn_extract import _stripped_key, extract_flow_records
from glue.samplers.lsdflow.route_steps import reaction_step
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionAction0, ReactionActionC
from rgfn.trainer.trainer import (  # noqa: F401  (registers @Trainer, as scripts/*.py do)
    Trainer,
)
from validation.lsdflow.adapters.base import FlowSample, GFNAdapter


class RGFNAdapter(GFNAdapter):
    model_name = "rgfn"

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        *,
        reward_name: str = "seh",
        device: str = "auto",
        batch_size: int = 100,
        extra_bindings: Optional[List[str]] = None,
        run_dir: str = "/tmp/lsdflow_rgfn",
        strip_stereo: bool = True,
    ):
        self.reward_name = reward_name
        self.batch_size = batch_size
        self.strip_stereo = strip_stereo
        # DOCKING targets record the RAW oracle energy in `reward` (what targets.py's calibrated bars
        # are measured in) while `log_reward` keeps the training transform. RGFN reaches its oracle
        # IN-PROCESS via @OracleRewardProxy (same env as glue -- no cross-env bridge), and that proxy
        # emits {"value", "raw_score"} exactly like SCENT's DockingBridgeProxy, so one component name
        # serves both. Surrogate targets leave this None and the proxy value is the gate value.
        # See validation/lsdflow/adapters/workers/_docking for the two-column contract.
        self.gate_component = "raw_score" if reward_name in ("6td3", "clpp") else None
        self.device = _resolve_device(device)

        bindings = [
            f'user_root_dir="{run_dir}"',
            'run_name="lsdflow_analysis"',
            "Trainer.n_iterations=1",  # never trained here; keeps singleton construction cheap
        ]
        if extra_bindings:
            bindings.extend(extra_bindings)
        gin.parse_config_files_and_bindings([config_path], bindings=bindings, finalize_config=False)

        # Build the trained objective (forward+backward policies + logZ) and the pure-policy
        # validation sampler (reward attached).
        self.objective = gin.get_configurable("objective/gin.singleton")()
        self.sampler = gin.get_configurable("valid_sampler/gin.singleton")()

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        result = self.objective.load_state_dict(state, strict=False)
        real_missing = [k for k in result.missing_keys if "_cache" not in k]
        if real_missing:
            print(
                f"[RGFNAdapter] WARNING {len(real_missing)} non-cache keys missing from "
                f"checkpoint (first few: {real_missing[:5]})",
                flush=True,
            )

        self._to_device(self.device)

        # Reward orientation + logZ (the visitation-estimate shift, §2).
        self.higher_is_better = self._reward_higher_is_better()
        self.log_z = self._scalar_log_z()
        print(
            f"[RGFNAdapter] built on {self.device}; higher_is_better={self.higher_is_better}, "
            f"logZ={self.log_z:.4f}, checkpoint={Path(checkpoint_path).name}",
            flush=True,
        )

    # ---------------------------------------------------------------- phase 1 sampling
    @staticmethod
    def _collect_routes(traj, routes: dict, strip_stereo: bool = True) -> None:
        """Record a canonical synthesis route for EVERY molecule on every trajectory in ``traj``.

        WHY EVERY MOLECULE, NOT JUST THE TERMINALS. A hub is an INTERIOR node -- the useful ones sit at
        depth 1-2 of a max-depth-4 tree -- and which molecules become hubs is decided later, by
        ``pick_hubs``, from the sampled pool. Emitting only terminal routes would therefore leave every
        actual hub without one, and the hub prefix is exactly the half that ``enum_children.json``
        cannot supply. SCENT already does it this way (its routes.json spans depths 1-4: 11,780 /
        21,928 / 15,600 / 8,412 on scent_seh), so matching it also keeps the two generators' artifacts
        interchangeable for the competitor arm.

        Keyed by ``_stripped_key`` -- the SAME function ``rgfn_extract`` uses for record and hub keys.
        Reusing it rather than re-canonicalising here is deliberate: a route keyed even slightly
        differently from its hub is worse than no route, because the lookup returns nothing and the
        caller prices the hub instead of the child WITHOUT any error (the failure that cost the six
        rgfn re-enumerations).

        Schema is ``glue.samplers.lsdflow.route_steps`` -- the one shared definition -- not
        ``extract_route``'s AL-flavoured dict. Two schemas for "the step from X to Y" is the exact
        hazard route_steps.py was written to prevent.
        """
        states_list = getattr(traj, "_states_list", None)
        actions_list = getattr(traj, "_actions_list", None)
        if not states_list or not actions_list:
            return
        for i, states in enumerate(states_list):
            if i >= len(actions_list) or not states:
                continue
            seed = None
            steps: list = []
            # A DEPTH-0 hub is a bare purchasable building block, so its route is legitimately EMPTY --
            # but "no entry in routes.json" and "zero-step route" are not the same thing downstream:
            # sparrow_select_frontier does `hub_routes.get(hk)` and on None skips the hub AND every
            # child hanging off it. Measured on the first smoke: 2 of 193 hub_keys were depth-0 and
            # would have silently dropped their whole subtree. Emit the zero-step route explicitly; a
            # purchasable starting material is something SPARROW models natively.
            for act in actions_list[i] or ():
                # ReactionAction0 picks the initial building block; ReactionActionC commits a
                # reaction. Anything else (stop, backward bookkeeping) carries no synthesis step.
                if isinstance(act, ReactionAction0):
                    frag = getattr(act, "fragment", None)
                    seed = getattr(frag, "smiles", None) or seed
                    if seed:
                        key0 = _stripped_key(frag)[0] if strip_stereo else seed
                        if key0 and key0 not in routes:
                            routes[key0] = {"seed": seed, "num_reactions": 0, "steps": []}
                    continue
                if not isinstance(act, ReactionActionC):
                    continue
                step = reaction_step(act)
                if not step.get("product"):
                    continue
                steps.append(step)
                if seed is None:
                    seed = step.get("input")
                mol = getattr(act, "output_molecule", None)
                if mol is None:
                    continue
                key = _stripped_key(mol)[0] if strip_stereo else getattr(mol, "smiles", None)
                # First route wins: identical molecules reached by different paths are the same
                # library item, and re-keying it on every rediscovery would make routes.json depend on
                # sampling order. Cheapest-route selection is the count-once cost model's job, not
                # this writer's.
                if key and key not in routes:
                    routes[key] = {
                        "seed": seed,
                        "num_reactions": len(steps),
                        "steps": list(steps),
                    }

    def sample_flow_records(self, n_trajectories: int) -> FlowSample:
        records = []
        visit_counts: dict = {}
        routes: dict = {}
        total_traj = 0
        for traj in self.sampler.get_trajectories_iterator(n_trajectories, self.batch_size):
            self._collect_routes(traj, routes, strip_stereo=self.strip_stereo)
            recs, visits, n = extract_flow_records(
                self.objective,
                traj,
                strip_stereo=self.strip_stereo,
                gate_component=self.gate_component,
            )
            records.extend(recs)
            for key, count in visits.items():
                visit_counts[key] = visit_counts.get(key, 0) + count
            total_traj += n
        terminal_depths = [r.hub_depth + 1 for r in records]  # x is one reaction past hub h
        print(
            f"[RGFNAdapter] sampled {total_traj} trajectories -> {len(records)} terminal "
            f"transitions, {len(visit_counts)} distinct molecule nodes",
            flush=True,
        )
        return FlowSample(
            records=records,
            visit_counts=visit_counts,
            n_trajectories=total_traj,
            terminal_depths=terminal_depths,
            log_z=self.log_z,
            higher_is_better=self.higher_is_better,
            model=self.model_name,
            reward_name=self.reward_name,
            routes=routes,
        )

    # ---------------------------------------------------------------- phase 2 enumeration
    def enumerate_hub_children(
        self, hubs, *, max_children: int = 2000, reaction_out=None, timing=None, sync=None
    ):
        """Exhaustively enumerate one-reaction terminal children for each hub (§4b, §6).

        Args:
            hubs: iterable of ``(stereo_smiles, depth)`` — the hub's stereo-aware SMILES and its
                observed build depth ``k`` (children land at ``k+1``, which sets stop
                competition vs the ``max_num_reactions`` cap).
            max_children: per-hub enumeration cap (docking budget guard; free for sEH).
            reaction_out: optional dict, populated ``{product_key: [step]}`` with the reaction that
                turns the hub into each child. REQUIRED for any downstream route pricing: without
                it every child's route stops at its hub, so SPARROW prices the hub rather than the
                child and returns an empty library as trivially optimal — with no error. Passing it
                is what ``rgfn_worker`` does to fill ``enum_children.json`` children[].reaction.
            timing: optional dict accumulator for the per-component wall-clock (Logs/039) --
                ``enumeration_s`` / ``reward_gen_s`` / ``flow_extract_s``. WITHOUT it RGFN is the one
                generator with no component breakdown: the worker can only wall-clock the whole call
                and report a lumped per-hub total under ``unattributed_s``, which reads on disk as
                three zero columns beside one huge bucket and has already been reported twice as
                "compute-time attribution is broken". It was never broken -- it was unmeasured, and
                the measurement hooks were sitting unused in ``enumerate_terminal_children`` the whole
                time. Pass this and the split is real, not fabricated.
            sync: optional no-arg callable (``torch.cuda.synchronize``) fired at timing boundaries so
                async CUDA work lands in the component that caused it. Pass it whenever the adapter is
                on a GPU or ``reward_gen_s`` silently absorbs the enumeration's queued kernels.

        Returns:
            ``(records, per_hub_stats)`` — flow records for all enumerated children (mergeable
            with the sampled DAG) + a per-hub dict of enumerated-path / record counts.
        """
        from glue.samplers.lsdflow.rgfn_enumerate import (
            enumerate_terminal_children,
            hub_state_from_smiles,
        )

        env = getattr(self.sampler, "env", None)
        reward = getattr(self.sampler, "reward", None)
        if env is None or reward is None:
            raise RuntimeError(
                "RGFNAdapter.enumerate_hub_children needs sampler.env + sampler.reward."
            )
        all_records = []
        per_hub = []
        for smiles, depth in hubs:
            hub_state = hub_state_from_smiles(smiles, int(depth))
            if hub_state is None:
                per_hub.append(
                    {
                        "hub": smiles,
                        "depth": int(depth),
                        "n_enumerated_paths": 0,
                        "n_records": 0,
                        "error": "invalid_smiles",
                    }
                )
                continue
            recs, n_paths = enumerate_terminal_children(
                env,
                self.objective,
                reward,
                hub_state,
                max_children=max_children,
                strip_stereo=self.strip_stereo,
                gate_component=self.gate_component,
                reaction_out=reaction_out,
                timing=timing,
                sync=sync,
            )
            all_records.extend(recs)
            per_hub.append(
                {
                    "hub": smiles,
                    "depth": int(depth),
                    "n_enumerated_paths": n_paths,
                    "n_records": len(recs),
                }
            )
        return all_records, per_hub

    # ---------------------------------------------------------------- helpers
    def _to_device(self, device: str) -> None:
        self.objective.device = device
        for pol in (self.objective.forward_policy, self.objective.backward_policy):
            if hasattr(pol, "set_device"):
                pol.set_device(device)
        policy = getattr(self.sampler, "policy", None)
        if policy is not None and hasattr(policy, "set_device"):
            policy.set_device(device)
        reward = getattr(self.sampler, "reward", None)
        proxy = getattr(reward, "proxy", None)
        if proxy is not None and hasattr(proxy, "set_device"):
            try:
                proxy.set_device(device)
            except Exception as exc:  # noqa: BLE001 - proxy may pin its own device
                print(f"[RGFNAdapter] proxy.set_device({device}) skipped: {exc}", flush=True)

    def _reward_higher_is_better(self) -> bool:
        reward = getattr(self.sampler, "reward", None)
        proxy = getattr(reward, "proxy", None)
        return bool(getattr(proxy, "higher_is_better", True))

    def _scalar_log_z(self) -> float:
        log_z = getattr(self.objective, "logZ", None)
        if log_z is None:
            return 0.0
        try:
            return float(log_z.detach().sum().item())
        except Exception:  # noqa: BLE001
            return 0.0


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
