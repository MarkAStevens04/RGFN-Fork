#!/usr/bin/env python
"""Pre-flight GPU-docking gate for ANY registered oracle — prove this node can actually
generate poses before a job spends hours on work it cannot score.

Why this exists on top of the raw OpenCL health check (Logs/013/014): balam009 *passes* the
``clCreateContext`` probe (``experiments/ablations/gpu_pose_gen/opencl_healthcheck.c``) yet
QuickVina2-GPU produces **zero poses** for every molecule in real docking — a subtler
post-outage degradation than balam008's hard ``clCreateContext=-5``. The OpenCL gate cannot see
it; only an actual dock can. A docking ENUMERATION is the worst possible place to discover this,
because a wedged node does not crash: it returns a full set of failed molecules, which looks like
a pool of very bad chemistry rather than a broken GPU.

Generalises ``experiments/active_learning/6td3/preflight_dock.py``, which hard-coded the 6TD3
oracle and its two 6TD3 seed molecules. Here the oracle is selected by name from the same registry
the docking server uses, so a fifth oracle is gated for free.

Exit 0 if at least one probe molecule produces a pose (node usable); exit 42 with a clear message
otherwise (add the node to ``#SBATCH --exclude`` and resubmit). Run under the ``rgfn`` env.

    python scripts/preflight_dock.py --oracle docking_clpp
    python scripts/preflight_dock.py --oracle docking_6td3_gpu --oracle-arg num_modes=9
"""

from __future__ import annotations

import argparse
import sys

# Small, chemically unremarkable drug-like probes. The point is only "does the GPU emit a pose",
# so they need to be dockable anywhere rather than active against anything. The 6TD3 pair are the
# validated seeds from entry 002 with known reference scores, kept for continuity with the old gate.
PROBES = {
    "docking_6td3_gpu": [
        "C[C@H](C(=O)Nc1nc2ccccc2[nH]1)N1Cc2ccccc2C1=O",  # ref dvina ~ -2.20
        "C=Cc1ccc(CNc2nc(N[C@H](CC)CO)nc3c2ncn3C(C)C)cc1",  # ref dvina ~ -1.26
    ],
    "_default": [
        "CC(=O)Nc1ccc(O)cc1",  # paracetamol
        "c1ccc(-c2ccccc2)cc1",  # biphenyl
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
    ],
}


def _parse_arg(kv: str):
    k, _, v = kv.partition("=")
    for cast in (int, float):
        try:
            return k, cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return k, v.lower() == "true"
    return k, v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--oracle", required=True, help="oracle name (docking_clpp, docking_6td3_gpu, ...)"
    )
    ap.add_argument(
        "--oracle-arg",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="oracle constructor kwarg (repeatable); pass the SAME args the run will use",
    )
    ap.add_argument("--smiles", action="append", default=[], help="override the probe molecules")
    a = ap.parse_args()

    # Import here (not at module scope) so --help works in an env without glue on the path.
    from glue.oracles.docking_server import _load_oracle as build_oracle

    kwargs = dict(_parse_arg(s) for s in a.oracle_arg)
    probes = a.smiles or PROBES.get(a.oracle, PROBES["_default"])

    print(f"[preflight] oracle={a.oracle} args={kwargs} probes={len(probes)}", flush=True)
    try:
        oracle = build_oracle(a.oracle, kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[preflight] FATAL: could not construct {a.oracle}: {exc}", flush=True)
        return 42

    # score_detailed exposes per-molecule pose counts, which is what distinguishes "wedged GPU"
    # from "bad molecule"; fall back to plain scores for an oracle that lacks it.
    posed = 0
    if hasattr(oracle, "score_detailed"):
        for smi, d in zip(probes, oracle.score_detailed(probes)):
            n = d.get("n_poses") or 0
            print(
                f"[preflight]   n_poses={n:<3} status={d.get('status')} "
                f"score={d.get('dvina', d.get('vina', d.get('score')))}",
                flush=True,
            )
            posed += 1 if n > 0 else 0
    else:
        for smi, s in zip(probes, oracle.score(probes)):
            ok = s is not None and s == s
            print(f"[preflight]   score={s} ok={ok}", flush=True)
            posed += 1 if ok else 0

    if posed == 0:
        print(
            "[preflight] FATAL: the docker produced NO usable output for ANY probe molecule on "
            "this node. It is degraded for docking despite any OpenCL probe passing "
            "(cf. balam009, Logs/014). Add this node to '#SBATCH --exclude' and resubmit.",
            flush=True,
        )
        return 42
    print(f"[preflight] OK: {posed}/{len(probes)} probes docked — node is usable.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
