"""The route contract: every stage declares whether its output is route-bearing, and proves it.

WHY THIS EXISTS. Recovering a synthesis route after the fact costs a full re-run. `enum_children.json`
children[].reaction was omitted by ``rgfn_worker`` for months and cost six 24-hour re-enumerations to
repair; ``routes.json`` is still empty for 36 of 40 sampled cell-seeds, and those are NOT repairable at
all -- the trajectory is discarded at sample time, and ``compositions.json`` keeps only ``num_reactions``,
so there is nothing left to reconstruct from. Re-sampling is the only option.

None of that was a crash. It was a SILENT DEFAULT:

    routes: Dict[str, dict] = field(default_factory=dict)
    # "Empty for adapters that don't emit routes yet (e.g. RGFN)."

An adapter that emitted nothing produced a well-formed, non-empty-looking artifact set containing
``{}``, the run exited 0, the harvester promoted it, and the loss surfaced when the competitor arm
needed routes months later. The defect is not that three adapters lacked an implementation -- that is
ordinary -- it is that NOTHING ANYWHERE ASSERTED THE ARTIFACT WAS USABLE. So the fix is not another
implementation; it is making "no routes" a declared, checked, loud condition.

HOW. Each generator declares a contract here, in code, and every worker calls :func:`validate_routes`
before it exits. The check writes ``route_status.json`` beside the artifacts -- machine-readable, so
the harvest and frontier layers can refuse a cell in seconds instead of an agent discovering it in
October -- and raises when the artifact contradicts the declaration.

"Not applicable" is a legitimate answer, but it must be DECLARED WITH A REASON rather than achieved by
writing an empty dict. FragGFN is the real case: its move is a fragment attachment, not a reaction
(``docs/LSD_FLOW_PROPOSAL.md`` L274), so a FragGFN "route" is not a synthesis plan and handing one to
SPARROW would price a library nobody can make. That is a different fact from "nobody implemented it",
and the two must not look identical on disk -- which today they do.

DESIGN RULE, AND IT IS NOT NEGOTIABLE: **a validator must never be why a good run dies.** These
enumerations are 24-hour GPU jobs. So this module raises only on UNAMBIGUOUS evidence (a route-bearing
stage that produced zero routes, or zero reaction coverage), every internal error in the check itself
is caught and downgraded to a warning, and the soft cases are reported rather than enforced. A guard
that kills a good 24-hour job gets switched off within a week, and then protects nothing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------------------------------
# The contract. Keys are generator names as the workers report them (``--model-name`` / meta["model"]).
#
#   "required"       -- this generator's moves ARE synthesis steps, so a sample stage must emit
#                       routes.json and an enumerate stage must emit children[].reaction. SPARROW can
#                       price its output, so a missing route is a defect that costs a re-run.
#   "not_applicable" -- this generator's moves are NOT synthesis steps. An empty routes.json is CORRECT
#                       here and must be recorded as a deliberate answer, with the reason, so it can
#                       never again be confused with an unimplemented emitter.
# ---------------------------------------------------------------------------------------------------
REQUIRED = "required"
NOT_APPLICABLE = "not_applicable"

ROUTE_CONTRACT: Dict[str, str] = {
    "rgfn": REQUIRED,
    "scent": REQUIRED,
    "rxnflow": REQUIRED,
    "multiaiz": REQUIRED,
    "sparrow": REQUIRED,
    "fraggfn": NOT_APPLICABLE,
}

NOT_APPLICABLE_REASON: Dict[str, str] = {
    "fraggfn": (
        "FragGFN's move is a fragment ATTACHMENT, not a reaction (docs/LSD_FLOW_PROPOSAL.md L274), so "
        "its trajectories are not synthesis plans and cannot be priced by SPARROW. FragGFN is the "
        "cost-model CONTROL in this matrix, not a route-bearing generator. An empty routes.json is the "
        "correct output here -- do NOT 'fix' it by inventing reaction steps for attachments."
    ),
}

# Coverage below this fraction of distinct terminals is reported as "partial" rather than "ok". It is
# deliberately NOT an error: a route-bearing stage that produced SOME routes has not suffered the
# failure this module exists to catch, and hard-failing a 24-hour job over a soft ratio is how guards
# get switched off. sparrow_select_frontier.py is the layer that refuses partial coverage, because by
# then the cost of stopping is seconds rather than a day.
PARTIAL_COVERAGE_FLOOR = float(os.environ.get("LSDFLOW_ROUTE_COVERAGE_FLOOR", "0.90"))

# Escape hatch, for the one legitimate case: deliberately sampling a route-bearing generator when you
# only need flow records (a smoke, a timing probe). Must be set explicitly, and it is recorded in
# route_status.json so a cell run this way can never be mistaken for a complete one.
ALLOW_MISSING = os.environ.get("LSDFLOW_ALLOW_MISSING_ROUTES", "") not in ("", "0", "false", "False")


class RouteContractError(RuntimeError):
    """A route-bearing stage produced an artifact set nothing downstream can price."""


def contract_for(generator: str) -> str:
    """The declared contract for a generator, defaulting to REQUIRED for anything unknown.

    Defaulting to REQUIRED is the safe direction: a NEW generator added without a thought about routes
    fails loudly on its first run, which is a five-minute fix. The opposite default reproduces exactly
    the silence this module exists to end.
    """
    return ROUTE_CONTRACT.get((generator or "").strip().lower(), REQUIRED)


def _write_status(out_dir: Path, status: dict) -> None:
    try:
        (out_dir / "route_status.json").write_text(json.dumps(status, indent=2, sort_keys=True))
    except Exception as exc:  # never let bookkeeping sink a finished run
        print(f"[routes] WARNING: could not write route_status.json: {exc}", file=sys.stderr, flush=True)


def _say(msg: str) -> None:
    print(f"[routes] {msg}", flush=True)


def validate_sample_routes(
    generator: str,
    out_dir: Path,
    *,
    routes: Optional[dict] = None,
    n_terminals: Optional[int] = None,
) -> dict:
    """Check a SAMPLE stage's routes.json against the generator's contract; write route_status.json.

    Args:
        generator: generator name (``rgfn`` / ``scent`` / ``rxnflow`` / ``fraggfn`` / ...).
        out_dir: the sample stage's output directory.
        routes: the routes mapping just written. Read back from disk when omitted.
        n_terminals: distinct terminal molecules sampled, for a coverage ratio.

    Raises:
        RouteContractError: a REQUIRED generator emitted ZERO routes. That is the unambiguous case --
            the run kept nothing to reconstruct from, so promoting it silently costs a re-sample.
    """
    out_dir = Path(out_dir)
    contract = contract_for(generator)
    status = {"generator": generator, "mode": "sample", "contract": contract}
    try:
        if routes is None:
            p = out_dir / "routes.json"
            routes = json.loads(p.read_text()) if p.exists() else {}
        n_routes = len(routes or {})
        status["n_routes"] = n_routes
        if n_terminals:
            status["n_terminals"] = int(n_terminals)
            status["coverage"] = round(n_routes / max(int(n_terminals), 1), 4)

        if contract == NOT_APPLICABLE:
            status["state"] = "not_applicable"
            status["reason"] = NOT_APPLICABLE_REASON.get(generator.lower(), "declared not route-bearing")
            if n_routes:
                # Surprising but not fatal: an N/A generator that DID emit something. Say so loudly
                # rather than discarding it -- the contract may simply be out of date.
                status["state"] = "unexpected_routes"
                _say(f"NOTE: {generator} is declared not-route-bearing but emitted {n_routes} routes. "
                     f"If its moves are now real reactions, update ROUTE_CONTRACT in _routes.py.")
            else:
                _say(f"{generator}: routes not applicable by contract -- recorded, not a defect.")
            _write_status(out_dir, status)
            return status

        # -- route-bearing from here ------------------------------------------------------------
        if n_routes == 0:
            status["state"] = "MISSING"
            if ALLOW_MISSING:
                status["state"] = "missing_allowed"
                status["reason"] = "LSDFLOW_ALLOW_MISSING_ROUTES set -- this cell is NOT SPARROW-usable."
                _say(f"WARNING: {generator} emitted NO routes; allowed by "
                     f"LSDFLOW_ALLOW_MISSING_ROUTES. This cell cannot feed the competitor arm.")
                _write_status(out_dir, status)
                return status
            _write_status(out_dir, status)
            raise RouteContractError(
                f"{generator} is declared route-bearing but its sample stage emitted ZERO routes "
                f"({out_dir/'routes.json'}).\n"
                f"This is not recoverable after the fact: the trajectory is gone once sampling ends, "
                f"and compositions.json keeps only num_reactions -- so the competitor arm would need a "
                f"full re-sample of this cell.\n"
                f"Fix the adapter's route emission, or set LSDFLOW_ALLOW_MISSING_ROUTES=1 to record "
                f"deliberately (a smoke or timing probe) that this cell is not SPARROW-usable."
            )

        cov = status.get("coverage")
        if cov is not None and cov < PARTIAL_COVERAGE_FLOOR:
            status["state"] = "partial"
            _say(f"WARNING: {generator} routes cover {cov:.1%} of {n_terminals} terminals "
                 f"(floor {PARTIAL_COVERAGE_FLOOR:.0%}). Recorded; sparrow_select_frontier.py will "
                 f"refuse a partial cell rather than price part of a library.")
        else:
            status["state"] = "ok"
            _say(f"{generator}: {n_routes:,} routes emitted"
                 + (f" ({cov:.1%} of terminals)" if cov is not None else ""))
        _write_status(out_dir, status)
        return status

    except RouteContractError:
        raise
    except Exception as exc:  # a validator must never be why a good 24-hour run dies
        status["state"] = "check_failed"
        status["error"] = repr(exc)
        _say(f"WARNING: route check itself failed ({exc!r}) -- NOT failing the run. "
             f"Treat this cell's route status as unknown.")
        _write_status(out_dir, status)
        return status


def validate_enum_reactions(
    generator: str,
    out_dir: Path,
    *,
    enum_path: Optional[Path] = None,
) -> dict:
    """Check an ENUMERATE stage's children[].reaction coverage; write route_status.json.

    A child without its reaction cannot be priced: the route stops at the hub, so SPARROW prices the
    HUB instead of the child and returns an empty library as trivially optimal -- with no error. That
    is the exact silent failure that cost the rgfn re-enumerations.

    Raises:
        RouteContractError: a REQUIRED generator produced children but ZERO of them carry a reaction.
    """
    out_dir = Path(out_dir)
    contract = contract_for(generator)
    status = {"generator": generator, "mode": "enumerate", "contract": contract}
    try:
        p = Path(enum_path) if enum_path else (out_dir / "enum_children.json")
        if not p.exists():
            status["state"] = "no_output"
            _write_status(out_dir, status)
            return status
        hubs = json.loads(p.read_text()).get("hubs", [])
        n_child = 0
        n_rxn = 0
        for h in hubs:
            for c in h.get("children", []) or []:
                n_child += 1
                if c.get("reaction"):
                    n_rxn += 1
        status.update({"n_hubs": len(hubs), "n_children": n_child, "n_with_reaction": n_rxn})
        status["coverage"] = round(n_rxn / n_child, 4) if n_child else None

        if contract == NOT_APPLICABLE:
            status["state"] = "not_applicable"
            status["reason"] = NOT_APPLICABLE_REASON.get(generator.lower(), "declared not route-bearing")
            _write_status(out_dir, status)
            return status

        if n_child and n_rxn == 0:
            status["state"] = "MISSING"
            if ALLOW_MISSING:
                status["state"] = "missing_allowed"
                _say(f"WARNING: {generator} enumerated {n_child:,} children with NO reactions; "
                     f"allowed by LSDFLOW_ALLOW_MISSING_ROUTES.")
                _write_status(out_dir, status)
                return status
            _write_status(out_dir, status)
            raise RouteContractError(
                f"{generator} enumerated {n_child:,} children and NONE carries children[].reaction "
                f"({p}).\n"
                f"Without it a child's route stops at its hub, so SPARROW prices the hub rather than "
                f"the child and returns an empty library as trivially optimal -- silently. Pass "
                f"reaction_by_child through to the artifact writer (see rgfn_worker's enumerate mode)."
            )
        if n_child and n_rxn < n_child:
            status["state"] = "partial"
            _say(f"WARNING: {generator} reaction coverage {n_rxn:,}/{n_child:,} "
                 f"({status['coverage']:.1%}) -- partial enumerations cannot be priced end-to-end.")
        elif n_child:
            status["state"] = "ok"
            _say(f"{generator}: reaction coverage 100% ({n_rxn:,}/{n_child:,})")
        else:
            status["state"] = "no_children"
        _write_status(out_dir, status)
        return status

    except RouteContractError:
        raise
    except Exception as exc:
        status["state"] = "check_failed"
        status["error"] = repr(exc)
        _say(f"WARNING: route check itself failed ({exc!r}) -- NOT failing the run.")
        _write_status(out_dir, status)
        return status
