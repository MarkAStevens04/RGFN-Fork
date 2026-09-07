"""``ScentFixedRewardRun`` — the SCENT counterpart of ``glue/fixed_reward/pipeline.py``.

The fixed-reward (single-shot) analogue of ``ScentActiveLearningLoop``: train SCENT's
cost-guided reaction-GFN **once** against a **fixed reward generator** — SCENT's own
frozen pretrained ``SehMoleculeProxy`` (the same Bengio-2021 sEH checkpoint RGFN uses) —
then sample a batch with synthesis routes and emit a standard candidate dataset. No
active-learning loop, no oracle, no proxy refit. SCENT's entry in the matched four-way
sEH comparison (the cost-aware synthesizable peer).

Because SCENT is an RGFN fork sharing ``rgfn.api`` (same ``Trainer.train()``, forward
sampler, ``ReactionStateTerminal``), this is almost line-for-line RGFN's
``FixedRewardPipeline`` — it just runs in the ``scent`` env (sibling imports; SCENT's
``rgfn`` must not be shadowed) and, since that env can't import ``glue``, emits candidates
by shelling out to ``scripts/ingest_candidates.py`` under the ``rgfn`` env (like the AL
loop shells to the oracle bridge).
"""

import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gin

# Plain sibling imports (NOT package-relative), matching al_loop.py — the runner keeps
# the repo root OFF sys.path so SCENT's installed `rgfn` fork isn't shadowed.
from guidance_io import load_guidance_models, save_guidance_models  # noqa: E402
from route import extract_route  # noqa: E402

from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal


def _free_gpu_cache() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@gin.configurable()
class ScentFixedRewardRun:
    """Single-shot SCENT: train once against a fixed reward generator, sample, emit."""

    def __init__(
        self,
        trainer,
        reward_generator,
        run_dir: str,
        repo_root: str,
        seed: int = 42,
        n_samples: int = 1000,
        sample_oversample: float = 4.0,
        top_k: int = 100,
        system: str = "seh",
        generator_name: str = "scent",
        reward_name: str = "seh_proxy",
        score_units: str = "seh_mpnn_reward (higher is better)",
        conda_exe: str = "conda",
        ingest_env: str = "rgfn",
        ingest_script: str = "scripts/ingest_candidates.py",
    ):
        """
        Args:
            trainer: a gin-built SCENT ``Trainer`` whose ``Reward`` wraps
                ``reward_generator`` (wire both to ``%train_proxy``).
            reward_generator: the frozen fixed reward generator — SCENT's
                ``@SehMoleculeProxy`` (a ``CachedProxyBase`` with ``compute_proxy_output``).
                Never refit; read as the GFN reward and to score the final batch.
            run_dir: absolute run dir (the runner binds a timestamped path).
            repo_root: RGFN-Fork repo root — the CWD for the ingest subprocess (so
                ``scripts/ingest_candidates.py`` resolves). The run itself has CWD =
                the SCENT clone.
            seed / n_samples / sample_oversample / top_k / system: as in the RGFN
                fixed-reward pipeline. ``generator_name``/``reward_name`` are provenance.
        """
        self.trainer = trainer
        self.reward_generator = reward_generator
        self.run_dir = Path(run_dir)
        self.repo_root = Path(repo_root)
        self.seed = seed
        self.n_samples = n_samples
        self.sample_oversample = sample_oversample
        self.top_k = top_k
        self.system = system
        self.generator_name = generator_name
        self.reward_name = reward_name
        self.score_units = score_units
        self.conda_exe = conda_exe
        self.ingest_env = ingest_env
        self.ingest_script = ingest_script

    # --------------------------------------------------------------------- driver
    def run(self) -> List[Tuple[str, float]]:
        out_dir = self.run_dir / "fixed_reward"
        out_dir.mkdir(parents=True, exist_ok=True)
        higher_is_better = bool(self.reward_generator.higher_is_better)
        print(
            f"[SCENT-FR] start: single-shot training vs frozen {self.reward_name} "
            f"(higher_is_better={higher_is_better}), n_samples={self.n_samples}",
            flush=True,
        )

        # 1. train SCENT's generator ONCE against the frozen reward.
        #    Resume (campaign Logs/030): the Trainer already restored forward policy + logZ +
        #    optimizer from last_gfn.pt in __init__ (if Trainer.resume_path was set); reload the
        #    guidance MLPs (P_B) from the sidecar so cost-guidance survives the requeue. And keep
        #    the sidecar in sync with every periodic checkpoint by saving it alongside each
        #    make_checkpoint (the Trainer's checkpoint holds only the forward state).
        self._load_guidance_models()
        if hasattr(self.trainer, "make_checkpoint"):
            _orig_make_ckpt = self.trainer.make_checkpoint

            def _make_ckpt_with_guidance(*a, **k):
                _orig_make_ckpt(*a, **k)
                self._save_guidance_models()

            self.trainer.make_checkpoint = _make_ckpt_with_guidance

        # 1a. Instrument the reward, and preserve the arm-A checkpoint at 10,000 oracle calls.
        #     Attached AFTER the guidance patch above so the arm-A save goes through it and gets
        #     its P_B sidecar too -- an arm-A checkpoint without one has an UNRECOVERABLE backward
        #     policy (Logs/024), which is the whole reason the sidecar exists.
        self._attach_trace()
        self.trainer.train()

        # 1b. Final guidance-model save — persists the trained P_B (cost + decomposability MLPs)
        #     that objective.state_dict() -> last_gfn.pt silently drops, so it stays exactly
        #     recoverable for flow analysis and for the next chain link. See guidance_io.
        self._save_guidance_models()

        # 1c. Everything from here is EVALUATION, not training signal -- the candidate batch is
        #     scored through the same proxy. Without the flip those rows read as training calls and
        #     inflate the cell's budget. See _trace.attach_proxy_trace.
        if getattr(self, "_trace_handle", None) is not None:
            self._trace_handle.set_phase("eval")

        # 2. sample a batch (unique valid terminals + routes + state objects).
        _free_gpu_cache()
        smiles, routes, states = self._sample_batch()
        print(f"[SCENT-FR] sampled {len(smiles)} unique valid candidates", flush=True)

        # 3. score with the reward generator itself (its proxy value = the score column).
        #    A docking reward generator (@DockingBridgeProxy) also exposes the raw Vina/dvina
        #    as a 'raw_score' component -> kept as a provenance column (matches RGFN/baselines).
        scores, raws = self._score(states)

        # 4. write pairs.csv + routes.jsonl, emit standard dataset via the rgfn env.
        pairs_path = out_dir / "pairs.csv"
        with open(pairs_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["smiles", "score"] + (["raw_score"] if raws is not None else []))
            for i, (smi, sc) in enumerate(zip(smiles, scores)):
                w.writerow([smi, sc] + ([raws[i]] if raws is not None else []))
        routes_path = out_dir / "routes.jsonl"
        with open(routes_path, "w") as fh:
            for smi, route in zip(smiles, routes):
                fh.write(json.dumps({"smiles": smi, **route}) + "\n")

        cand_dir = out_dir / "candidates"
        cmd = [
            self.conda_exe,
            "run",
            "--no-capture-output",
            "-n",
            self.ingest_env,
            "python",
            self.ingest_script,
            "--pairs",
            str(pairs_path),
            "--routes",
            str(routes_path),
            "--out-dir",
            str(cand_dir),
            "--generator",
            self.generator_name,
            "--reward-name",
            self.reward_name,
            "--system",
            self.system,
            "--seed",
            str(self.seed),
            "--score-units",
            self.score_units,
            "--source",
            str(self.run_dir),
        ]
        if higher_is_better:
            cmd.append("--score-higher-is-better")
        print(f"[SCENT-FR] ingest (cwd={self.repo_root}) -> {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(self.repo_root))

        pairs = [(s, sc) for s, sc in zip(smiles, scores) if sc is not None and sc == sc]
        pairs.sort(key=lambda p: p[1], reverse=higher_is_better)
        top = pairs[: self.top_k]
        self._write_top_k(out_dir / "top_k.csv", top)
        self._close_trace()
        print(f"[SCENT-FR] done. candidates at {cand_dir}", flush=True)
        return top

    # ----------------------------------------------------------------- internals
    def _save_guidance_models(self) -> None:
        """Write the backward-policy guidance-model weights beside ``last_gfn.pt``.

        ``last_gfn.pt`` holds only ``objective.state_dict()`` = forward policy + logZ;
        the cost/decomposability guidance MLPs live in a plain Python list on
        ``JointlyGuidedBackwardPolicy`` and are dropped by ``state_dict()``. Saving them
        here (same dir as the checkpoint) makes the trained ``P_B`` exactly recoverable.
        Best-effort: never fail a good training run over provenance."""
        try:
            ckpt_dir = Path(self.trainer.run_dir) / "train" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            gpath = ckpt_dir / "guidance_models.pt"
            keys = save_guidance_models(self.trainer.objective, gpath)
            print(
                f"[SCENT-FR] saved backward-policy guidance models -> {gpath} (keys={keys})",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[SCENT-FR] WARNING guidance-model save failed: {e}", flush=True)

    # ------------------------------------------------------------------- trace / arm A
    def _attach_trace(self) -> None:
        """Record every reward evaluation; preserve the arm-A checkpoint when 10,000 calls land.

        Patches the gin-built ``%train_proxy`` instance in place, because the trainer's Reward and
        this run hold the SAME object -- wrapping here would leave training untraced. Arm A is
        defined on the ORACLE-CALL axis (SCENT is batch 64, so the crossing is near step 156, but
        replay makes that arithmetic unreliable), and lands on a real iteration boundary because
        ``ProxyBase`` inherits ``TrainingHooksMixin`` and is handed the live ``iteration_idx``.
        """
        try:
            # LOADED BY FILE PATH, NOT sys.path. SCENT's own package is ALSO named `rgfn`, and this
            # process has already chdir'd into the clone with it on sys.path -- putting the repo
            # root on sys.path too would make `import rgfn` ambiguous and could silently resolve
            # SCENT's imports to OUR fork. That is the namespace hygiene rule run_scent_fixed.py
            # states explicitly ("never put the repo root on sys.path"), and an earlier draft of
            # this method broke it. importlib gives us the one module we need with no path effects.
            import importlib.util as _ilu

            _mod_path = Path(self.repo_root) / "validation" / "generators" / "_trace.py"
            _spec = _ilu.spec_from_file_location("_benchmark_v2_trace", _mod_path)
            _trace_mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_trace_mod)
            ARM_A_ORACLE_CALLS = _trace_mod.ARM_A_ORACLE_CALLS
            BudgetCheckpointer = _trace_mod.BudgetCheckpointer
            TraceWriter = _trace_mod.TraceWriter
            attach_proxy_trace = _trace_mod.attach_proxy_trace
            self._trace_mod = _trace_mod
        except Exception as exc:  # noqa: BLE001 - instrumentation must never block a run
            print(f"[SCENT-FR] WARNING trace unavailable ({exc}); continuing untraced", flush=True)
            return

        self._trace_t0 = time.time()
        self.trace = TraceWriter(self.run_dir / "trace.csv", t0=self._trace_t0)

        def _save_arm_a(iteration_idx: int) -> None:
            # Goes through the guidance-patched make_checkpoint, so guidance_models.pt is written
            # too -- then BOTH are copied aside. The sidecar path is fixed, so the next periodic
            # save overwrites it with a LATER P_B; without this copy the arm-A weights would end up
            # paired with the arm-B backward policy, silently.
            self.trainer.make_checkpoint("arm_a_10k", {"epoch": int(iteration_idx)})
            ckpt_dir = Path(self.trainer.run_dir) / "train" / "checkpoints"
            src = ckpt_dir / "guidance_models.pt"
            if src.exists():
                shutil.copy2(src, ckpt_dir / "guidance_models_arm_a_10k.pt")
            # Make it resume-clean, same as RGFN: SCENT is an RGFN fork, so its forward policy may
            # carry the same lazily populated `*_cache` buffers that break a strict
            # load_state_dict (Logs/021). SCENT's own runner resumes straight from last_gfn.pt
            # with no strip, so either it has none or nobody has hit it -- stripping if present
            # and no-opping if not is correct under both, and arm A is the checkpoint that will
            # actually be resumed from.
            try:
                import torch

                ckpt = ckpt_dir / "arm_a_10k.pt"
                state = torch.load(ckpt, map_location="cpu")
                dropped = [k for k in list(state.get("model", {})) if k.endswith("_cache")]
                for k in dropped:
                    state["model"].pop(k)
                if dropped:
                    torch.save(state, ckpt)
                    print(
                        f"[SCENT-FR] arm A: stripped {len(dropped)} *_cache keys -> resume-clean",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[SCENT-FR] WARNING arm-A cache strip failed ({exc})", flush=True)

        self.arm_a = BudgetCheckpointer(self.trace, ARM_A_ORACLE_CALLS, _save_arm_a, tag="SCENT-FR")
        self._trace_handle = attach_proxy_trace(
            self.reward_generator, self.trace, tag="SCENT-FR", budget_checkpointer=self.arm_a
        )
        print(
            f"[SCENT-FR] trace -> {self.run_dir / 'trace.csv'} "
            f"(arm A at {ARM_A_ORACLE_CALLS} calls)",
            flush=True,
        )

    def _close_trace(self) -> None:
        """Close the trace; write timing.json + arm_a.json beside it."""
        trace = getattr(self, "trace", None)
        if trace is None:
            return
        try:
            write_timing = self._trace_mod.write_timing

            trace.close()
            elapsed = time.time() - self._trace_t0
            write_timing(
                self.run_dir / "timing.json", {"train_and_sample_s": elapsed}, total_s=elapsed
            )
            arm_a = getattr(self, "arm_a", None)
            if arm_a is not None:
                (self.run_dir / "arm_a.json").write_text(json.dumps(arm_a.manifest(), indent=2))
                if not arm_a.fired:
                    print(
                        f"[SCENT-FR] NOTE arm-A checkpoint never fired: {trace.n_scored} molecules "
                        f"scored, below budget. This cell is arm A in its entirety.",
                        flush=True,
                    )
            print(
                f"[SCENT-FR] trace closed: {trace.n_scored} scored / {trace.n_distinct} distinct "
                f"-> trace.csv; timing -> timing.json",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[SCENT-FR] WARNING closing trace failed ({exc})", flush=True)

    def _load_guidance_models(self) -> bool:
        """Reload the backward-policy guidance MLPs from the sidecar on resume (campaign
        Logs/030). ``Trainer`` resume restores forward policy + logZ + optimizer from
        ``last_gfn.pt`` but reinitialises the guidance MLPs to RANDOM (they aren't in
        ``state_dict()``), which would silently erase SCENT's cost-guidance on every requeue.
        This restores the exact trained ``P_B``. Returns True if a sidecar was loaded."""
        gpath = Path(self.trainer.run_dir) / "train" / "checkpoints" / "guidance_models.pt"
        if not gpath.exists():
            return False
        try:
            loaded, unmatched = load_guidance_models(self.trainer.objective, gpath)
            print(
                f"[SCENT-FR] resume: reloaded guidance models from {gpath} "
                f"(loaded={len(loaded)}, unmatched={len(unmatched)})",
                flush=True,
            )
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[SCENT-FR] WARNING guidance-model reload failed: {e}", flush=True)
            return False

    def _sample_batch(self) -> Tuple[List[str], List[Dict], List]:
        """Sample unique valid terminals from the trained policy; return
        ``(smiles, routes, states)``. Mirrors ``ScentActiveLearningLoop._sample_query_batch``
        but also returns the terminal state objects so we can score them directly."""
        sampler = self.trainer.train_forward_sampler
        n_sample = int(self.n_samples * self.sample_oversample)
        batch_size = self.trainer.train_batch_size
        seen: set = set()
        smiles: List[str] = []
        routes: List[Dict] = []
        states: List = []
        for trajectories in sampler.get_trajectories_iterator(n_sample, batch_size):
            last_states = trajectories.get_last_states_flat()
            for i, state in enumerate(last_states):
                if not isinstance(state, ReactionStateTerminal):
                    continue
                smi = state.molecule.smiles
                if smi in seen:
                    continue
                seen.add(smi)
                smiles.append(smi)
                states.append(state)
                routes.append(
                    extract_route(trajectories._states_list[i], trajectories._actions_list[i])
                )
                if len(smiles) >= self.n_samples:
                    return smiles, routes, states
        return smiles, routes, states

    def _score(self, states: List) -> Tuple[List[float], Optional[List[float]]]:
        """Reward generator's own value per terminal state (the score column), plus the
        raw docking score if the reward generator exposes a ``raw_score`` component
        (docking); ``(scores, None)`` for a plain proxy (sEH/DRD2)."""
        if not states:
            return [], None
        output = self.reward_generator.compute_proxy_output(states)
        scores = [float(v) for v in output.value.detach().cpu().reshape(-1).tolist()]
        raws: Optional[List[float]] = None
        comps = getattr(output, "components", None)
        if comps and "raw_score" in comps:
            raws = [float(v) for v in comps["raw_score"].detach().cpu().reshape(-1).tolist()]
        return scores, raws

    @staticmethod
    def _write_top_k(path: Path, rows: List[Tuple[str, float]]) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "smiles", "reward_score"])
            for rank, (smi, score) in enumerate(rows, start=1):
                w.writerow([rank, smi, score])
