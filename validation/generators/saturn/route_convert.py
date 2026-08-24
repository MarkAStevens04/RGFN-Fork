"""Turn TANGO's syntheseus search results into the project's route schema.

WHY THIS EXISTS. TANGO is the second entrant, after SynFormer, whose molecules arrive WITH a route,
so it can be priced by ``sparrow_select_frontier.py --route-source external`` and never needs
MultiAiZ. But the routes are only half-exposed: the oracle persists syntheseus's ``route_0.pkl`` per
solved molecule (``syntheseus_results/output_<oracle_calls>/mol_<i>/``) and the clone ships
``extract_syntheseus_route_data.py``, which reduces each route to ``{smiles, depth}`` per MOLECULE
node -- dropping every reaction. A list of molecules with depths cannot say which reaction consumed
which input, and a reaction network is exactly that. So the shipped extractor is unusable for pricing
and this module reads the pickles directly.

WHAT A SYNTHESEUS ROUTE IS. An unordered ``set`` of nodes forming a bipartite tree:
  * ``OrNode``  -- a molecule (``.mol.smiles``), at EVEN depth; the target is the unique depth-0 node.
  * ``AndNode`` -- a reaction (``.reactants``, ``.product``, ``.reaction_smiles``), at ODD depth.
Depth increases AWAY from the target, so the deepest reaction is the FIRST synthetic step and the
depth-1 reaction is the last. Emitting them in the order the set happens to iterate would produce a
route whose steps are shuffled -- valid-looking and wrong -- so the ordering here is load-bearing.

WHY WE RUN IN THE syntheseus ENV. Unpickling needs the syntheseus classes importable; the pickles are
useless from ``rgfn`` or ``saturn``. This module is therefore invoked as a subprocess under
``conda run -n syntheseus`` (see ``convert_run``), the same cross-env pattern the oracle itself uses.

LEAVES ARE PURCHASABLE, and unlike our own native routes they need no recipe expansion: every leaf is
a molecule from the inventory TANGO was given (the authors' ``frag-reac-zinc-stock.smi``). That is why
the frontier prices these with ``--route-source external`` rather than ``native``.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

SYNTHESEUS_ENV = "syntheseus"


def route_from_pickle(path: Path) -> Optional[dict]:
    """Read one ``route_0.pkl`` into ``{product_smiles, num_reactions, steps}``.

    Returns ``None`` for a route with no reactions -- the target was already purchasable. That is a
    legitimate zero-cost outcome, not a failure, and the caller records it as ``num_reactions: 0``.
    """
    with open(path, "rb") as fh:
        nodes = pickle.load(fh)

    mols, rxns = [], []
    for n in nodes:
        if hasattr(n, "mol"):
            mols.append(n)
        elif hasattr(n, "reaction"):
            rxns.append(n)

    if not mols:
        return None
    root = min(mols, key=lambda n: getattr(n, "depth", 0))
    target = root.mol.smiles

    # Deepest reaction first: depth grows away from the target, so that is synthesis order.
    rxns.sort(key=lambda n: -getattr(n, "depth", 0))
    steps: List[dict] = []
    for i, n in enumerate(rxns):
        rx = n.reaction
        reactants = [m.smiles for m in rx.reactants]
        product = rx.product.smiles
        steps.append(
            {
                "step": i,
                # syntheseus gives a reaction SMILES, not a template index or SMARTS. Keep the key
                # names the schema uses so the frontier needs no special case, and record the real
                # thing under reaction_smarts rather than inventing a template id we do not have.
                "reaction_idx": -1,
                "reaction_smarts": getattr(
                    rx, "reaction_smiles", f"{'.'.join(reactants)}>>{product}"
                ),
                "reactant": reactants[0] if reactants else "",
                "fragments": reactants[1:],
                "product": product,
            }
        )
    return {"product_smiles": target, "num_reactions": len(steps), "steps": steps}


def convert_results_dir(results_dir: Path) -> Dict[str, dict]:
    """Walk ``syntheseus_results/output_*/mol_*/`` and return ``{smiles: route}``.

    The molecule a directory belongs to comes from its own ``stats.json``, NOT from the ``mol_<i>``
    index. The index is a position within the batch that was searched, and the oracle re-uses it
    across batches, so joining on it would silently attach routes to the wrong molecules.

    A molecule searched more than once keeps the CHEAPEST route found, which is the one a chemist
    would run and the one that makes the cost comparison a lower bound on TANGO's own behaviour.
    """
    out: Dict[str, dict] = {}
    for stats_path in results_dir.glob("output_*/mol_*/stats.json"):
        try:
            stats = json.loads(stats_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        smi = (stats.get("smiles") or "").strip()
        if not smi:
            continue
        pkl = stats_path.parent / "route_0.pkl"
        if not pkl.exists():
            continue  # searched but unsolved: no route, so nothing to price
        try:
            r = route_from_pickle(pkl)
        except Exception:  # noqa: BLE001 -- one unreadable pickle must not lose the run
            continue
        if r is None:
            continue
        prev = out.get(smi)
        if prev is None or r["num_reactions"] < prev["num_reactions"]:
            out[smi] = r
    return out


def write_routes_jsonl(routes: Dict[str, dict], pool_smiles: List[str], dest: Path) -> dict:
    """Emit routes.jsonl for the molecules in ``pool_smiles``, in pool order.

    A pooled molecule with no route is OMITTED rather than written with zero steps: a zero-step entry
    means 'purchasable outright, costs nothing', and silently conflating 'unsolved' with 'free' would
    understate the competitor's cost. The frontier reports the routed fraction, so an omission is
    visible; a fake free route would not be.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_missing = 0
    with open(dest, "w") as fh:
        for smi in pool_smiles:
            r = routes.get(smi)
            if r is None:
                n_missing += 1
                continue
            fh.write(json.dumps({"smiles": smi, **r}) + "\n")
            n_written += 1
    return {"written": n_written, "missing": n_missing, "pool": len(pool_smiles)}


def convert_run(run_dir: Path, dest: Optional[Path] = None) -> dict:
    """Cross-env entry point: run this module under the syntheseus env for ``run_dir``."""
    dest = dest or (run_dir / "fixed_reward" / "routes.jsonl")
    cmd = [
        "conda", "run", "--no-capture-output", "-n", SYNTHESEUS_ENV,
        "python", str(Path(__file__).resolve()), str(run_dir), str(dest),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"route conversion failed:\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    run_dir = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    results = run_dir / "syntheseus_results"
    if not results.is_dir():
        raise SystemExit(f"no syntheseus_results under {run_dir}")
    routes = convert_results_dir(results)

    pool_csv = run_dir / "fixed_reward" / "candidates" / "candidates.csv"
    if pool_csv.exists():
        import csv

        pool = [r["smiles"] for r in csv.DictReader(open(pool_csv))]
    else:
        pool = sorted(routes)
    summary = write_routes_jsonl(routes, pool, dest)
    summary["routes_found"] = len(routes)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
