"""SCENT-side LSD-Flow acquisition — the cross-env twin of ``glue.samplers.lsdflow.acquisition``.

Runs **in the scent env, in-process on the live AL trainer** (no per-round SCENT rebuild). It is the
SCENT half of the ``docs/AL_PIPELINE_ARCHITECTURE.md`` acquisition contract: it produces the worker
file contract (``records.csv`` / ``enum_children.json`` / ``compositions.json``) from the trained
SCENT policy + the round's live proxy ``M``, then shells to the ``rgfn``-env selector
(``validation/lsdflow/select_acquisition.py``) which does the UCB-rank + pre-select-K + mode-select and
returns the docking batch. SCENT can't ``import glue`` (its package is also ``rgfn``), so the heavy
glue selection lives across the subprocess boundary; the algorithms here are **reused, not duplicated**
— we import ``scent_worker``'s already-vendored ``extract_flow_records`` / enumerator (its ``rgfn``
imports live inside ``main()``, so importing the module is safe in any env).

Design decisions (see ``docs/AL_PIPELINE_ARCHITECTURE.md`` §7):
  * **Warm-start + frozen library.** :func:`warm_start` loads the sEH checkpoint's forward policy +
    logZ + the ``guidance_models.pt`` sidecar (trained ``P_B``, entry 024) and freezes the dynamic
    library to the checkpoint's promoted-fragment snapshot (~1600 frags) via ``scent_worker._freeze_library``.
    The library is **fixed for the whole run**, so pre-select-K=20 ranks over a stable set and every
    round's enumeration is well-defined.
  * **Candidate hubs by visit count** (depth 1–3), enumerated to compute ``U(h)`` from each hub's
    *enumerated* neighborhood — robust to sparse sampling (the RGFN smoke-71011 lesson).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_WORKER_DIR = _REPO_ROOT / "validation" / "lsdflow" / "adapters" / "workers"

# scent_worker's module-top imports are stdlib-only, so importing it here is safe in the scent env
# (its rgfn imports are inside main()); we reuse its vendored flow-extraction + enumeration verbatim.
if str(_WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKER_DIR))
import scent_worker as _sw  # noqa: E402


def warm_start(
    trainer,
    checkpoint: str,
    guidance: str,
    snapshot: str,
    load_policy: bool = True,
) -> set:
    """Freeze the dynamic library to ``snapshot`` (the rich promoted-fragment vocabulary pre-select-K
    ranks over) and, if ``load_policy``, also load the checkpoint's trained forward policy + logZ +
    guidance ``P_B``. Returns the frozen ``chosen_set``. Mirrors ``scent_worker``'s build recipe.

    **``load_policy`` is the confound knob (docs/AL_PIPELINE_ARCHITECTURE.md §7).** The sEH checkpoint's
    policy was trained on the sEH *proxy*, which DISAGREES with the docking oracle (the known
    proxy≠docking finding) — so warm-starting it biases the from-policy arms toward proxy-good /
    docking-mediocre molecules and random beat them. ``load_policy=False`` freezes ONLY the library
    (keeps pre-select-K meaningful) and leaves the forward policy at fresh init, so the AL loop trains
    it from scratch on the docking proxy ``M`` → docking-aligned, unconfounded, still over the rich
    2018-fragment vocabulary."""
    import torch
    from guidance_io import load_guidance_models  # scent sibling

    from rgfn.gfns.reaction_gfn.api.data_structures import Molecule

    objective = trainer.objective
    if load_policy:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        res = objective.load_state_dict(state, strict=False)
        real_missing = [k for k in res.missing_keys if "_cache" not in k]
        print(
            f"[SCENT-AL] warm-start: loaded forward policy + logZ (real-missing={len(real_missing)})",
            flush=True,
        )
        if guidance and Path(guidance).exists():
            loaded, unmatched = load_guidance_models(
                objective, guidance, map_location="cpu", strict=True
            )
            print(
                f"[SCENT-AL] warm-start: loaded guidance sidecar ({loaded} keys, unmatched={unmatched})",
                flush=True,
            )
        else:
            print(
                f"[SCENT-AL] WARNING warm-start: guidance sidecar missing ({guidance}) — P_B not the trained one",
                flush=True,
            )
    else:
        print(
            "[SCENT-AL] warm-start: LIBRARY-ONLY (fresh docking-aligned policy; not loading the "
            "proxy-trained checkpoint policy — confound fix)",
            flush=True,
        )
    env = (
        trainer.valid_sampler.env
        if getattr(trainer, "valid_sampler", None)
        else trainer.train_forward_sampler.env
    )
    n = _sw._freeze_library(trainer, env, snapshot, Molecule)
    chosen_set = set(json.load(open(snapshot)).get("chosen_smiles", []))
    # scent_worker._freeze_library restores the 2018-frag env the warm-started policy needs, but it is
    # an INFERENCE-time op (the worker never trains after). Continuing to TRAIN then trips SCENT's
    # dynamic-library tracking hook (reaction_dynamic_library.on_end_sampling): the injected promoted
    # fragments have no path_cost_proxy entry, so its `min(smiles_to_cost, current_cost)` hits
    # current_cost=None. Treat the frozen library as this run's *base* library: add the promoted
    # fragments to initial_smiles_set (the hook skips them, line 104) and freeze the vocabulary by
    # disabling further promotion (empty schedule) — a fixed 2018-frag set all run, matching the
    # frozen-library intent + keeping pre-select-K's snapshot consistent with the live library.
    dl = getattr(trainer, "dynamic_fragment_library", None)
    if dl is not None:
        # SCENT's on_end_sampling short-circuits (`if iteration_idx > n_iterations_schedule[-1]:
        # return {}`, line 87) once past the last scheduled promotion iteration — so setting the
        # schedule's last element below every AL iteration disables BOTH the cost-tracking (the
        # line-118 `min(inf, None)` crash on injected frozen fragments) AND further promotion, using
        # SCENT's own mechanism. Result: a fixed 2018-fragment vocabulary all run (the frozen-library
        # intent). (initial_smiles_set update kept as a belt-and-braces guard.)
        if hasattr(dl, "initial_smiles_set"):
            dl.initial_smiles_set |= chosen_set
        if hasattr(dl, "n_iterations_schedule"):
            dl.n_iterations_schedule = [-1]
        print(
            "[SCENT-AL] warm-start: froze dynamic-library vocabulary (no further promotion)",
            flush=True,
        )
    print(f"[SCENT-AL] warm-start: froze library (+{n} promoted fragments)", flush=True)
    return chosen_set


class ScentHubAcquisition:
    """Produce a round's docking batch from the live SCENT trainer + proxy ``M`` (hub_batching /
    best_candidate). Writes the worker file contract, shells to the rgfn-env selector, returns the
    chosen molecules + routes + accounting."""

    def __init__(
        self,
        arm: str,
        *,
        snapshot: str,
        repo_root: str,
        budget_modes: int = 100,
        reward_threshold: Optional[float] = -8.0,
        similarity: float = 0.5,
        higher_is_better: bool = False,
        lam: float = 1.0,
        prebuild_k: int = 20,
        child_policy: str = "free_frag",
        n_candidate_hubs: int = 100,
        min_hub_visits: int = 1,
        min_hub_depth: int = 1,
        max_hub_depth: int = 3,
        max_children_per_hub: int = 200,
        n_sample_trajectories: int = 2000,
        sample_batch_size: int = 100,
        oracle_env: str = "rgfn",
        conda_exe: str = "conda",
        strip_stereo: bool = True,
    ):
        self.arm = arm
        self.snapshot = snapshot
        self.repo_root = Path(repo_root)
        self.budget_modes = budget_modes
        self.reward_threshold = reward_threshold
        self.similarity = similarity
        self.higher_is_better = higher_is_better
        self.lam = lam
        self.prebuild_k = prebuild_k
        self.child_policy = child_policy
        self.n_candidate_hubs = n_candidate_hubs
        self.min_hub_visits = min_hub_visits
        self.min_hub_depth = min_hub_depth
        self.max_hub_depth = max_hub_depth
        self.max_children_per_hub = max_children_per_hub
        self.n_sample_trajectories = n_sample_trajectories
        self.sample_batch_size = sample_batch_size
        self.oracle_env = oracle_env
        self.conda_exe = conda_exe
        self.strip_stereo = strip_stereo

    # ------------------------------------------------------------------ per-round entry
    def select(self, trainer, chosen_set: set, out_dir: Path) -> Tuple[List[str], List[dict], dict]:
        """Return ``(smiles, routes, stats)`` for this round's docking batch."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rgfn_classes = self._rgfn_classes()
        sampler = self._sampler(trainer)
        objective = trainer.objective
        env = sampler.env
        # M standardizes labels; pass its fit stats so the selector maps the real hit bar into M-space.
        _proxy = getattr(sampler.reward, "proxy", None)
        label_mean = float(getattr(_proxy, "_label_mean", 0.0))
        label_std = float(getattr(_proxy, "_label_std", 1.0))

        records, visit_counts, compositions = self._sample(
            objective, sampler, chosen_set, rgfn_classes
        )
        (out_dir / "compositions.json").write_text(json.dumps(compositions))
        self._write_records(out_dir / "records.csv", records)

        stats = {"n_sampled_records": len(records), "reward_gen_calls": 0, "n_candidate_hubs": 0}
        if self.arm == "hub_batching":
            n_enum = self._enumerate(
                env,
                objective,
                sampler.reward,
                chosen_set,
                rgfn_classes,
                records,
                visit_counts,
                compositions,
                out_dir,
            )
            stats["reward_gen_calls"] = n_enum
            stats["n_candidate_hubs"] = self._last_n_candidates

        chosen = self._run_selector(out_dir, label_mean, label_std)
        smiles = [c["smiles"] for c in chosen]
        routes = [{"source_hub": c.get("source_hub", "")} for c in chosen]
        stats["oracle_calls"] = len(smiles)
        n_hubs = len({c.get("source_hub", "") for c in chosen if c.get("source_hub")})
        stats["n_hubs_used"] = n_hubs
        stats["avg_mols_per_hub"] = (len(smiles) / n_hubs) if n_hubs else float("nan")
        return smiles, routes, stats

    # ------------------------------------------------------------------ internals
    def _rgfn_classes(self) -> dict:
        from rgfn.api.trajectories import Trajectories
        from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
        from rgfn.gfns.reaction_gfn.api.reaction_api import (
            ReactionActionC,
            ReactionStateA,
            ReactionStateB,
            ReactionStateC,
            ReactionStateTerminal,
        )

        try:
            from rdkit import Chem
        except Exception:
            Chem = None
        import rgfn as rgfn_api

        return dict(
            rgfn_api=rgfn_api,
            Trajectories=Trajectories,
            Molecule=Molecule,
            Chem=Chem,
            RSA=ReactionStateA,
            RSB=ReactionStateB,
            RSC=ReactionStateC,
            RST=ReactionStateTerminal,
            RAC=ReactionActionC,
        )

    @staticmethod
    def _sampler(trainer):
        for c in (getattr(trainer, "valid_sampler", None), trainer.train_forward_sampler):
            if (
                c is not None
                and getattr(c, "env", None) is not None
                and getattr(c, "reward", None) is not None
            ):
                return c
        return trainer.train_forward_sampler

    def _extract_fn(self, chosen_set, rc):
        """A ``(objective, trajectories, routes_out=None) -> (records, visits, comps, n)`` closure
        bound to SCENT's rgfn classes — the shape ``scent_worker``'s enumerator expects."""

        def _extract(obj, traj, routes_out=None):
            return _sw.extract_flow_records(
                obj,
                traj,
                rc["RSA"],
                rc["RST"],
                rc["RSC"],
                rc["Chem"],
                self.strip_stereo,
                chosen_set,
                RAC=rc["RAC"],
                routes_out=routes_out,
            )

        return _extract

    def _sample(self, objective, sampler, chosen_set, rc):
        """Sample trajectories → merged (records, visit_counts, compositions) via the vendored extractor."""
        _extract = self._extract_fn(chosen_set, rc)
        records: List[dict] = []
        visit_counts: Dict[str, int] = {}
        compositions: Dict[str, dict] = {}
        for traj in sampler.get_trajectories_iterator(
            self.n_sample_trajectories, self.sample_batch_size
        ):
            recs, visits, comps, _n = _extract(objective, traj)
            records.extend(recs)
            for k, c in visits.items():
                visit_counts[k] = visit_counts.get(k, 0) + c
            for k, comp in comps.items():
                prev = compositions.get(k)
                if prev is None or comp["num_reactions"] < prev["num_reactions"]:
                    compositions[k] = comp
        return records, visit_counts, compositions

    def _enumerate(
        self, env, objective, reward, chosen_set, rc, records, visit_counts, compositions, out_dir
    ):
        """Enumerate the most-visited depth-banded hubs → write ``enum_children.json`` (with U(h))."""
        # candidate hubs = most-visited pre-terminal states in the depth band (robust vs sparse sampling)
        depth_by_hub: Dict[str, int] = {}
        for r in records:
            hk = r["hub_key"]
            d = int(r["hub_depth"])
            depth_by_hub[hk] = min(depth_by_hub.get(hk, d), d)
        stereo_by_hub: Dict[str, str] = {
            r["hub_key"]: r.get("hub_stereo_key", r["hub_key"]) for r in records
        }
        cands = [
            hk
            for hk, d in depth_by_hub.items()
            if visit_counts.get(hk, 0) >= self.min_hub_visits
            and self.min_hub_depth <= d <= self.max_hub_depth
        ]
        cands.sort(key=lambda hk: visit_counts.get(hk, 0), reverse=True)
        cands = cands[: self.n_candidate_hubs]
        self._last_n_candidates = len(cands)

        enumerate_children, hub_state_from_smiles, _ = _sw._make_enumerator(
            rc["rgfn_api"],
            rc["Trajectories"],
            rc["RSA"],
            rc["RSB"],
            rc["RSC"],
            rc["RST"],
            rc["RAC"],
            rc["Molecule"],
        )
        _extract = self._extract_fn(chosen_set, rc)
        enum_hubs = []
        n_records = 0
        for hk in cands:
            hub_state = hub_state_from_smiles(stereo_by_hub.get(hk, hk), depth_by_hub[hk])
            if hub_state is None:
                continue
            recs, _n_paths, added_by_stereo, _rxn_by_stereo, _timing = enumerate_children(
                env,
                objective,
                reward,
                hub_state,
                _extract,
                self.max_children_per_hub,
            )
            n_records += len(recs)
            u_h, n_eff = _sw._hub_uncertainty(recs)
            enum_hubs.append(
                {
                    "hub_input": stereo_by_hub.get(hk, hk),
                    "hub_key": recs[0]["hub_key"] if recs else hk,
                    "depth": int(depth_by_hub[hk]),
                    "uncertainty": None if (u_h != u_h) else u_h,
                    "n_effective": n_eff,
                    "children": [
                        {
                            "smiles": r["child_key"],
                            "reward": r["reward"],
                            "added_promoted": [
                                f
                                for f in added_by_stereo.get(r["child_stereo_key"], [])
                                if f in chosen_set
                            ],
                        }
                        for r in recs
                    ],
                }
            )
        (out_dir / "enum_children.json").write_text(json.dumps({"hubs": enum_hubs}))
        print(
            f"[SCENT-AL] enumerated {len(enum_hubs)} candidate hubs -> {n_records} child records",
            flush=True,
        )
        return n_records

    def _run_selector(self, out_dir: Path, label_mean: float, label_std: float) -> List[dict]:
        """Shell to the rgfn-env glue selector on the written files; read back chosen.csv."""
        chosen_path = out_dir / "chosen.csv"
        cmd = [
            self.conda_exe,
            "run",
            "--no-capture-output",
            "-n",
            self.oracle_env,
            "python",
            "-m",
            "validation.lsdflow.select_acquisition",
            "--arm",
            self.arm,
            "--out",
            str(chosen_path),
            "--compositions",
            str(out_dir / "compositions.json"),
            "--budget-modes",
            str(self.budget_modes),
            "--similarity",
            str(self.similarity),
            "--lam",
            str(self.lam),
            "--prebuild-k",
            str(self.prebuild_k),
            "--child-policy",
            self.child_policy,
            "--min-children",
            "2",
            "--proxy-label-mean",
            str(label_mean),
            "--proxy-label-std",
            str(label_std),
        ]
        if self.reward_threshold is not None:
            cmd += ["--reward-threshold", str(self.reward_threshold)]
        if self.higher_is_better:
            cmd += ["--higher-is-better"]
        if self.snapshot:
            cmd += ["--snapshot", str(self.snapshot)]
        if self.arm == "hub_batching":
            cmd += ["--enum-children", str(out_dir / "enum_children.json")]
        else:
            cmd += ["--records", str(out_dir / "records.csv")]
        print(f"[SCENT-AL] selector (cwd={self.repo_root}) -> {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(self.repo_root))
        chosen = []
        if chosen_path.exists():
            with open(chosen_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    chosen.append(row)
        return chosen

    @staticmethod
    def _write_records(path: Path, records: List[dict]) -> None:
        cols = [
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
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(records)
