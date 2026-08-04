"""Shared docking-scoring seam for the per-env LSD-Flow workers.

Surrogate targets score enumerated children with a fast in-process proxy. Docking targets cannot:
the oracle (QuickVina2-GPU + gnina + the prepared receptors) lives in the ``rgfn`` env alongside
``glue``, while each worker runs in its generator's own env, which cannot co-import it. The bridge
already exists per generator — ``validation/generators/<gen>/fixed_reward.DockingBridgeReward``,
which each baseline used to TRAIN against docking — and it already speaks to the persistent
docking server over ``RGFN_DOCK_SOCKET``. This module is deliberately thin: it does not re-implement
any of that. It exists to pin down the one thing that is easy to get silently wrong.

THE TWO-COLUMN CONTRACT
=======================
``records.csv`` carries ``reward`` and ``log_reward``, and for a docking cell **these are different
quantities in different units**. Conflating them is the failure mode this module exists to prevent,
because it produces plausible numbers rather than a crash:

  ``reward``     — the value the MODE GATE is applied to, in the units ``targets.py`` declares
                   alongside its ``higher_is_better``. For docking that is the **RAW** Vina energy
                   (or the raw T2-T1 differential), lower-is-better. ``targets.py``'s ClpP note
                   spells this out: gate on the raw value, NOT on the ``ReLU(-raw)`` reward column.
                   The calibrated bars (-8.0 ClpP, -2.0 6TD3) are raw-energy cutoffs; applied to a
                   ReLU'd column, which is never negative, NOTHING would ever qualify.

  ``log_reward`` — the log-reward the TRAINED MODEL saw, i.e. ``beta * clip(value)`` where ``value``
                   is the generator's own training transform ``max(-raw/norm, 0)``. The flow terms
                   (``log F_hat`` and everything derived from it) are only comparable to the
                   sampled DAG if this matches training exactly, so it must come from the same
                   ``DockingBridgeReward`` the checkpoint was trained with — not be re-derived here.

So one dock feeds two columns, and :class:`DockingChildScorer` hands back both from a single cached
call. Every worker uses it the same way, which is what keeps a fifth oracle or a fifth generator from
having to rediscover this.

ADDING A NEW ORACLE:  declare it in ``matrix16/targets.py`` (``oracle="docking_<x>"``) and register
it in ``glue/oracles/docking_server.py``'s registry. Nothing here changes.

ADDING A NEW GENERATOR: give it a ``DockingBridgeReward``-shaped object in its own env (``.predict``
-> training value, ``.raw_scores`` -> raw) and pass it in. Nothing here changes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


class DockingChildScorer:
    """Wraps a generator's own ``DockingBridgeReward`` and serves the two-column contract.

    Args:
        bridge: the generator's in-env docking reward. Must expose ``predict(smiles) -> values``
            (the training transform, higher-is-better) and ``raw_scores(smiles) -> raws`` (raw
            energies, lower-is-better). ``DockingBridgeReward`` caches per canonical SMILES, so
            asking for both costs ONE dock.
        beta: the reward temperature from the cell's training config (``fixed_reward.beta``).
        clip: the value clip from the cell's training config (``reward.clip``).
        illegal_log_reward: log-reward assigned to a molecule the oracle failed on. Matches the
            workers' existing convention for an unscoreable molecule.

    A failed dock yields ``reward = nan`` (so the gate cannot accept it — a NaN comparison is
    False in either direction) and the floor log-reward. Failures are counted, not hidden: a run
    where the GPU quietly stopped posing shows up as ``n_failed`` rather than as a pool of
    suspiciously bad molecules.
    """

    def __init__(
        self,
        bridge,
        *,
        beta: float,
        clip: float,
        illegal_log_reward: float = -75.0,
    ):
        for m in ("predict", "raw_scores"):
            if not callable(getattr(bridge, m, None)):
                raise TypeError(
                    f"docking bridge {type(bridge).__name__} is missing {m}(); it must expose "
                    "predict() (training transform) and raw_scores() (raw energies)"
                )
        self.bridge = bridge
        self.beta = float(beta)
        self.clip = float(clip)
        self.illegal_log_reward = float(illegal_log_reward)
        self.n_scored = 0
        self.n_failed = 0

    def score(self, smiles: Sequence[str]) -> Tuple[List[float], List[float]]:
        """Return ``(gate_values, log_rewards)`` for a batch — the two record columns.

        ``gate_values`` are RAW energies (what ``targets.py``'s bar is calibrated against);
        ``log_rewards`` are ``beta * clip(training value)`` (what the policy was trained on).
        """
        smiles = list(smiles)
        if not smiles:
            return [], []
        # raw_scores first: it populates the bridge's cache, so predict() then costs no docking.
        raws = list(self.bridge.raw_scores(smiles))
        values = list(self.bridge.predict(smiles))
        if len(raws) != len(smiles) or len(values) != len(smiles):
            raise RuntimeError(
                f"docking bridge returned {len(raws)} raws / {len(values)} values for "
                f"{len(smiles)} molecules"
            )
        gate, logr = [], []
        for raw, val in zip(raws, values):
            ok = raw is not None and float(raw) == float(raw)  # not None, not NaN
            self.n_scored += 1
            if not ok:
                self.n_failed += 1
                gate.append(float("nan"))
                logr.append(self.illegal_log_reward)
                continue
            gate.append(float(raw))
            v = min(float(val), self.clip)
            logr.append(max(self.beta * v, self.illegal_log_reward))
        return gate, logr

    def stats(self) -> Dict:
        """Failure accounting, for the worker's meta.json. A high ``fail_fraction`` is the
        signature of a wedged GPU (Logs/013: the node passes the OpenCL probe but poses nothing),
        not of bad chemistry — which is why the preflight gate runs before any of this."""
        return {
            "n_scored": self.n_scored,
            "n_failed": self.n_failed,
            "fail_fraction": round(self.n_failed / self.n_scored, 6) if self.n_scored else 0.0,
        }


def docking_reward_params(cfg: Dict) -> Dict:
    """Pull the docking transform parameters out of a cell's own training config.

    Deliberately read from the config the CHECKPOINT was trained with rather than from
    ``targets.py``: beta/norm/clip are properties of that training run, and a mismatch would make
    every recovered flow term quietly incomparable to the sampled DAG. Accepts the baselines' YAML
    shape (``fixed_reward.beta`` + ``reward.{oracle,norm,clip,oracle_args}``).
    """
    fr = dict(cfg.get("fixed_reward", {}) or {})
    rw = dict(cfg.get("reward", {}) or {})
    return {
        "oracle": rw.get("oracle", ""),
        "beta": float(fr.get("beta", 4)),
        "norm": float(rw.get("norm", 1.0)),
        "clip": float(rw.get("clip", 10.0)),
        "oracle_args": dict(rw.get("oracle_args", {}) or {}),
    }


def require_socket(env: Optional[Dict] = None) -> str:
    """Return ``RGFN_DOCK_SOCKET``, failing loudly if it is unset.

    Without it ``DockingBridgeReward`` silently falls back to spawning ``score_batch.py`` per
    batch, which re-constructs the oracle (~35-44 s each) — at enumeration scale that fallback is
    thousands of times more expensive than the docking itself, so a missing socket must be an
    error here rather than a slow success.
    """
    import os

    env = env if env is not None else os.environ
    sock = (env.get("RGFN_DOCK_SOCKET") or "").strip()
    if not sock:
        raise SystemExit(
            "RGFN_DOCK_SOCKET is not set. Docking enumeration requires the persistent docking "
            "server (glue.oracles.docking_server); without it every batch would re-construct the "
            "oracle. Launch via matrix16/submit_docking_cell.sh, which starts the server first."
        )
    return sock


def chunked(items: Sequence, size: int):
    """Yield ``size``-sized chunks. Logs/036 measured batch **200** with ONE QuickVina2-GPU process
    as the throughput optimum (3.3x faster per molecule than batch 25; a second concurrent process
    adds nothing because the GPU is already saturated), so docking callers should chunk at 200."""
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield items[i : i + size]


__all__ = [
    "DockingChildScorer",
    "docking_reward_params",
    "require_socket",
    "chunked",
    "DOCK_BATCH",
]

# The measured optimum from Logs/036 Part A. Named so callers don't re-litigate it per worker.
DOCK_BATCH = 200
