#!/usr/bin/env python
"""Generate randomly-generated decoy molecules for the sEH proxy benchmark (Logs/034).

The negative control for "can the sEH proxy tell a hit from a miss?" is a set of
molecules that are *not* specifically sEH binders but that live in the space the
generator actually proposes. We draw them exactly the way the project draws its
"random synthesizable molecule" baseline everywhere else: random rollouts from the
**untrained** RGFN forward policy over the standard reaction env — the same sampler
used for the sEH seed set `D_0` (entry `010`, `make_seh_seed.py`) and the
random-acquisition arm (entry `023`), just without the docking label. This makes the
decoys operationally the right control: "among molecules RGFN might generate, what
proxy score separates real sEH inhibitors from random ones?"

We reuse `configs/rgfn_seh_proxy.gin` (the paper's sEH benchmark config: base +
reaction env + `SehMoleculeProxy`), build the `Trainer` purely as a sampler (no GFN
training), collect unique valid terminal SMILES, drop any that collide with the known
actives, and write SMILES + MW.

Run on a GPU login/compute node after `source ~/bin/rgfn-smoke-env.sh` (needs the dgl
CUDA libs on LD_LIBRARY_PATH), from the repo root:

    python experiments/oracle_validation/seh_proxy/make_decoys.py --n 2500
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gin
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

import glue  # noqa: F401  (registers our gin components)
from gin_config import get_time_stamp
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal
from rgfn.trainer.trainer import Trainer  # noqa: F401  (registers @Trainer for gin)
from rgfn.utils.helpers import seed_everything

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sample_unique(trainer, n_target: int, oversample: float, exclude: set[str]) -> list[str]:
    """Unique valid terminal SMILES from the (untrained) forward policy.

    Mirrors ``make_seh_seed._sample_unique`` / ``ActiveLearningLoop._sample_query_batch``
    so the decoys come from the same distribution as the loop's per-round batches.
    """
    sampler = trainer.train_forward_sampler
    n_sample = int(n_target * oversample)
    batch_size = trainer.train_batch_size
    seen: set[str] = set()
    out: list[str] = []
    for trajectories in sampler.get_trajectories_iterator(n_sample, batch_size):
        for state in trajectories.get_last_states_flat():
            if not isinstance(state, ReactionStateTerminal):
                continue
            smi = state.molecule.smiles
            if smi in seen or smi in exclude:
                continue
            seen.add(smi)
            out.append(smi)
            if len(out) >= n_target:
                return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", type=str, default="configs/rgfn_seh_proxy.gin")
    ap.add_argument("--n", type=int, default=2500, help="target number of unique decoys")
    ap.add_argument(
        "--oversample",
        type=float,
        default=3.0,
        help="sample this multiple of --n trajectories to absorb dupes/invalids",
    )
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "seed_decoys_rgfn.csv"
    )
    ap.add_argument(
        "--actives",
        type=Path,
        default=REPO_ROOT / "data" / "validation-molecules" / "sEH_actives_chembl.csv",
        help="actives CSV; decoys colliding with these canonical SMILES are dropped",
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="override gin user_root_dir (sampler run dir); needed on read-only $HOME",
    )
    args = ap.parse_args()

    exclude: set[str] = set()
    if args.actives.exists():
        for row in csv.DictReader(open(args.actives)):
            exclude.add(row["smiles"])
    print(f"[decoys] excluding {len(exclude)} known-active canonical SMILES", flush=True)

    seed_everything(args.seed)
    run_name = f"seh_proxy_decoys/{get_time_stamp()}"
    bindings = [f'run_name="{run_name}"']
    if args.root_dir is not None:
        bindings.append(f'user_root_dir="{args.root_dir}"')
    gin.parse_config_files_and_bindings([args.cfg], bindings=bindings)

    trainer = Trainer()  # untrained forward policy = random in-distribution rollouts
    print(
        f"[decoys] sampling up to {args.n} unique molecules "
        f"(oversample x{args.oversample}) from the untrained policy...",
        flush=True,
    )
    smiles = _sample_unique(trainer, args.n, args.oversample, exclude)
    print(f"[decoys] sampled {len(smiles)} unique valid decoys", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "mw"])
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            mw = round(Descriptors.MolWt(mol), 2) if mol is not None else ""
            w.writerow([smi, mw])
    print(f"[decoys] wrote {args.out} ({len(smiles)} rows)", flush=True)
    trainer.close()


if __name__ == "__main__":
    main()
