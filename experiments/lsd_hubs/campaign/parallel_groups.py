#!/usr/bin/env python
"""Parallelizable groups inside a delivered library (Logs/073 -- CURIOSITY, not a headline).

A hub's batch is already "one plate" in the cost model: build the scaffold once, then run k
diversifying reactions on it. But those k reactions are not necessarily the SAME reaction. A chemist
running a plate wants one set of conditions, so the operationally interesting unit is finer than the
hub and coarser than the molecule. Two groupings, both read off the FINAL step of each selected
molecule:

  **Class A -- (parent, reaction class).** One substrate, one set of conditions, N partners. This is
  a true parallel array: weigh the hub out once into N wells, add a different building block to each.
  Strictly a refinement of the hub batch measured in Logs/038 -- a hub contributing 12 modes across
  3 reaction classes is 3 arrays, not one plate of 12.

  **Class B -- (reaction class) alone.** Same conditions, different substrates. A plate of amide
  couplings that happen to sit on different scaffolds. Weaker (every well needs its own substrate
  weighed out) but much larger, and it is what "parallel chemistry" usually means in a med-chem
  group. Class A groups PARTITION Class B groups, so both are reported together with the
  decomposition.

WHY THE FINAL STEP IS THE RIGHT ONE TO GROUP ON, and the limit of that. Hub-batching charges each
mode exactly one marginal reaction (the diversifying coupling); everything earlier is the shared
prefix already paid for. So the final step IS the per-molecule work, and grouping it is grouping the
work that actually differs. Reactions inside the prefixes could also be run in parallel across hubs
-- that is a separate, larger question this script does not answer.

REACTION CLASS, NOT TEMPLATE ID. The library encodes each transformation several times with the
reactant SMARTS in a different order (ids 136/137 are one acid+amine reaction, anchored on either
reactant), so grouping on the raw id would split chemically identical groups and understate
parallelism. We group on the named class from ``reaction_names.py`` (42 classes over 168 templates,
0.003% of steps unnamed on this corpus) and carry ``template_id`` per molecule so the naming is
auditable and never load-bearing.

THE CONTROL MATTERS MORE THAN THE NUMBER. Best-candidate is run on the identical enumeration, gate
and diversity cutoff -- only the chooser differs -- because "our libraries contain parallel arrays"
is only interesting if the obvious alternative's libraries do not.

TWO OPTIONAL THIRD FILTERS, both off by default, both diagnostic -- neither is part of the published
hub-batching method. They ask whether parallelism can be DEMANDED rather than merely measured, and
what that costs at the same budget. They differ in what they demand, and the difference is the point:

  ``--min-group N`` -- keep children in a group with at least N reward-passing siblings. This gates on
  AVAILABILITY, and on these hubs it is almost never binding: the intermediates hub-batching walks
  have hundreds of qualifying children per reaction class, so groups that go on to deliver two or
  three molecules sail through. Kept because it is the obvious thing to try and because measuring that
  it does not work is worth recording.

  ``--min-modes N`` -- keep children in a group that can itself yield at least N MUTUALLY DISTINCT
  molecules, measured by a micro greedy selection inside the group (same reward gate, same Tanimoto
  test, fresh selector). This gates on DELIVERABLE diversity and does bind.

Even the mode gate is only an upper bound, because the campaign's selector compares each candidate
against every mode already accepted -- from other groups and other hubs -- so a group holding ten
internally-distinct molecules can deliver fewer once the global filter has seen it. That gap is
measured rather than assumed: ``collapse.csv`` reports each used group's local capacity beside what it
actually delivered, excluding the one hub the budget truncated (see :func:`collapse_rows`). And
because interleaving groups is one candidate explanation for the gap, ``--order-modes`` runs the arm
both ways -- groups interleaved by reward, and groups offered contiguously -- so contiguity is tested
instead of argued.

Pure CPU: a re-selection over the cached enumeration, no scoring and no GPU. Reads a FROZEN
enumeration (``docs/ROUTE_DATASET_SCHEMA.md`` §7) so the arms cannot silently be compared across a
mid-flight rewrite.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/parallel_groups.py \
        --analysis-dir /scratch/.../lsdflow/matrix16_seed43/scent_seh/sample \
        --enum-children /scratch/.../_enum_snapshot_20260820/seed43/enum_children.json \
        --snapshot /scratch/.../scent_seh_5k/seed43/additional_fragments/fragments_<N>.json \
        --reward-threshold 5.0 --child-policy free_frag --prebuild-k 20 \
        --tag scent_seh_seed43
"""
import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from batch_size_distribution import _gini, _stats_for
from reaction_names import named_reaction, parse_template
from run_campaign import (
    _load_candidates,
    _load_enumerated_hubs,
    build_strategy,
    run_timed,
)

from glue.samplers.lsdflow.campaign import EnumChild, EnumeratedHub, rank_fragments
from glue.samplers.lsdflow.child_select import (
    ChildSelectionPolicy,
    _reward_key,
    make_child_policy,
)
from glue.samplers.lsdflow.mode_select import (
    DiverseThresholdModeSelector,
    passes_reward_gate,
)
from validation.lsdflow.metrics.cost.dynamic_amortization import (
    load_cost_table_from_snapshot,
)

HERE = Path(__file__).resolve().parent

# Group-size thresholds we report coverage at. 2 = "parallel at all"; 4/8 = a realistic bench array;
# 24/96 = the well counts of the plates these would actually run in.
THRESHOLDS = (2, 4, 8, 10, 24, 96)


# ----------------------------------------------------------------- enumeration with reaction classes
@dataclass(frozen=True)
class RxChild(EnumChild):
    """An enumerated child that also knows the reaction that made it.

    Subclasses rather than replaces ``EnumChild`` so the strategy code (which reads only ``smiles``
    / ``reward`` / ``added_promoted``) is untouched and the baseline selection stays byte-identical
    to the committed campaign. The naming table lives in ``experiments/``, so the class is stamped
    HERE by the driver and never inside ``glue/``.
    """

    named: str = "?"
    template_id: str = ""
    reagents: Tuple[str, ...] = ()  # building blocks consumed by the final step
    # The final step's own ``input`` -- the STEREO-AWARE substrate. ``source_hub`` on an accepted mode
    # is the stereo-STRIPPED key, so this is the identity a policy can key its per-group records by
    # without two stereoisomeric hubs colliding.
    parent_input: str = ""


def load_hubs_with_reactions(enum_path: Path, comps: dict):
    """``_load_enumerated_hubs`` + the final step's reaction class on every child.

    Deliberately re-derives the hub list from the same file rather than patching the objects that
    function returns: one pass, and the child ORDER is preserved, which is what keeps the selection
    identical (both child policies sort with a stable sort, so ties keep JSON order).

    Returns ``(hubs, coverage)`` where coverage reports how many children carried a usable
    single-step reaction -- printed, never assumed.
    """
    data = json.load(open(enum_path))
    hubs: List[EnumeratedHub] = []
    n_child = n_rxn = n_multi = 0
    for h in data.get("hubs", []):
        hub_comp = comps.get(h["hub_key"], {}) or {}
        children = []
        for c in h["children"]:
            steps = c.get("reaction") or []
            n_child += 1
            if len(steps) > 1:
                n_multi += 1
            # The final step is the diversifying reaction. Enumerated children are one reaction off
            # the hub, so len(steps) == 1; take the last regardless, so a multi-step child (which
            # would mean the enumeration changed shape) still names the step that actually differs.
            tmpl = steps[-1].get("reaction", "") if steps else ""
            if tmpl:
                n_rxn += 1
            tid, _ = parse_template(tmpl)
            children.append(
                RxChild(
                    smiles=c["smiles"],
                    reward=float(c["reward"]),
                    added_promoted=tuple(c.get("added_promoted", ())),
                    named=named_reaction(tmpl) if tmpl else "(no reaction recorded)",
                    template_id=tid,
                    reagents=tuple(steps[-1].get("reactants", ()) or ()) if steps else (),
                    parent_input=(steps[-1].get("input") or "") if steps else "",
                )
            )
        hubs.append(
            EnumeratedHub(
                hub_key=h["hub_key"],
                hub_input=h.get("hub_input", h["hub_key"]),
                depth=int(h["depth"]),
                promoted=tuple(hub_comp.get("promoted", ())),
                children=children,
                uncertainty=h.get("uncertainty"),
                n_effective=int(h.get("n_effective", 0)),
            )
        )
    coverage = {
        "n_children": n_child,
        "n_with_reaction": n_rxn,
        "rxn_coverage": round(n_rxn / n_child, 6) if n_child else None,
        "n_multi_step_children": n_multi,
    }
    return hubs, coverage


def parity_check(hubs, comps, enum_path) -> bool:
    """Assert our re-derived hub list matches ``run_campaign``'s on every field the strategy reads.

    The whole point of this script is to describe the COMMITTED selection. If the loader drifted,
    every group below would describe a library we never published, and nothing would error.
    """
    ref = _load_enumerated_hubs(Path(enum_path), comps)
    if len(ref) != len(hubs):
        return False
    for a, b in zip(ref, hubs):
        if (a.hub_key, a.depth, a.promoted, len(a.children)) != (
            b.hub_key,
            b.depth,
            b.promoted,
            len(b.children),
        ):
            return False
        for ca, cb in zip(a.children, b.children):
            if (ca.smiles, ca.reward, ca.added_promoted) != (
                cb.smiles,
                cb.reward,
                cb.added_promoted,
            ):
                return False
    return True


# ----------------------------------------------------------------- the speculative third filter
class ParallelGroupChildPolicy(ChildSelectionPolicy):
    """Offer only children sitting in a (this hub, reaction class) group of at least ``min_group``
    reward-passing siblings; order the survivors with ``inner``.

    The count is taken over children that clear the REWARD gate, before the diversity filter -- that
    is the number of wells you could in principle fill from this hub with one set of conditions.
    How many survive the diversity filter is smaller and is measured downstream, not assumed here;
    ``realized`` group sizes in the output are the honest ones.

    ``count_mode="reward_free"`` additionally requires a counted sibling to survive ``inner``'s own
    filter, so the group size cannot be inflated by children free-frag will drop anyway.
    """

    name = "parallel_group"

    def __init__(
        self,
        inner: ChildSelectionPolicy,
        *,
        reward_threshold: Optional[float],
        higher_is_better: bool = True,
        min_group: int = 10,
        count_mode: str = "reward",
    ):
        self.inner = inner
        self.reward_threshold = reward_threshold
        self.higher_is_better = higher_is_better
        self.min_group = int(min_group)
        self.count_mode = count_mode
        self.dropped_children = 0
        self.dropped_groups = 0
        self.kept_groups = 0

    def order(self, children, *, higher_is_better=True, cost_table=None, available=None):
        pool = children
        if self.count_mode == "reward_free":
            # Count only what the inner policy would actually offer.
            pool = self.inner.order(
                children,
                higher_is_better=higher_is_better,
                cost_table=cost_table,
                available=available,
            )
        counts: Counter = Counter()
        for c in pool:
            if passes_reward_gate(c.reward, self.reward_threshold, self.higher_is_better):
                counts[getattr(c, "named", "?")] += 1
        big = {k for k, v in counts.items() if v >= self.min_group}
        self.kept_groups += len(big)
        self.dropped_groups += len(counts) - len(big)
        kept = [c for c in children if getattr(c, "named", "?") in big]
        self.dropped_children += len(children) - len(kept)
        return self.inner.order(
            kept, higher_is_better=higher_is_better, cost_table=cost_table, available=available
        )


class ModeGroupChildPolicy(ChildSelectionPolicy):
    """Offer only children in a (this hub, reaction class) group that can itself yield at least
    ``min_modes`` MUTUALLY DISTINCT molecules; order the survivors by ``order_mode``.

    WHY THIS REPLACES COUNTING SIBLINGS. :class:`ParallelGroupChildPolicy` gates on how many children
    clear the reward bar, which is *availability*. On these hubs that gate is almost never binding --
    the intermediates hub-batching walks have hundreds of qualifying children per reaction class -- so
    it admits groups that then deliver two or three molecules, and guarantees nothing about the array
    you actually get. This policy instead runs a **micro greedy selection inside each group**: the
    same reward gate and the same Tanimoto test as the campaign, on a FRESH selector, over the group
    alone. ``min_modes`` is then a floor on what the group could deliver *if it had the whole
    diversity budget to itself*.

    IT IS STILL ONLY AN UPPER BOUND, AND MEASURING THE GAP IS THE POINT. The campaign's selector
    compares each candidate against every mode already accepted -- from other groups and other hubs --
    so a group holding ten internally-distinct molecules can deliver three once the global filter has
    seen it. The micro-selection cannot see that (deliberately: it is computed on the group in
    isolation), so ``local_modes`` is recorded per group and compared with what the group actually
    delivered. That comparison, not the gate, is the finding.

    ``order_mode``:
      * ``"reward"`` -- surviving children merged and offered in reward order, groups interleaved
        (identical in shape to the baseline, so only the gate differs).
      * ``"group"`` -- groups offered CONTIGUOUSLY, best-first, each internally in inner order. Groups
        are ranked by their best child's reward, NOT by size: that keeps the reward priority identical
        to ``"reward"`` and isolates the effect of contiguity itself, which is the thing under test.
        (Ranking by size instead would confound the two.)

    ``local_cap`` bounds each micro-selection. The gate only needs to know whether the count reaches
    ``min_modes``, but the count is also reported, so it runs past the gate up to the cap and records
    whether the cap was hit -- an uncapped selection over a 200k-child group would dominate runtime
    for a number we only need the magnitude of.
    """

    name = "mode_group"

    def __init__(
        self,
        inner: ChildSelectionPolicy,
        *,
        reward_threshold: Optional[float],
        similarity: float = 0.5,
        higher_is_better: bool = True,
        min_modes: int = 10,
        local_cap: int = 200,
        order_mode: str = "reward",
    ):
        self.inner = inner
        self.reward_threshold = reward_threshold
        self.similarity = similarity
        self.higher_is_better = higher_is_better
        self.min_modes = int(min_modes)
        self.local_cap = int(local_cap)
        self.order_mode = order_mode
        self.local_modes: Dict[Tuple[str, str], int] = {}  # (parent_input, class) -> local modes
        self.capped: Set[Tuple[str, str]] = set()
        self.dropped_children = 0
        self.kept_groups = 0
        self.dropped_groups = 0

    def _local_modes(self, ordered) -> Tuple[int, bool]:
        """Modes obtainable from this group ALONE, via the campaign's own selector on a fresh state."""
        sel = DiverseThresholdModeSelector(
            self.reward_threshold, self.similarity, self.higher_is_better
        )
        n = 0
        for c in ordered:
            if sel.accept(c.smiles, c.reward):
                n += 1
                if n >= self.local_cap:
                    return n, True
        return n, False

    def order(self, children, *, higher_is_better=True, cost_table=None, available=None):
        groups: Dict[str, list] = defaultdict(list)
        for c in children:
            groups[getattr(c, "named", "?")].append(c)
        kept: list = []
        for cls, members in groups.items():
            # Order INSIDE the group with the inner policy first, so the micro-selection sees exactly
            # what the campaign would be offered (free-frag drops, reward order) -- otherwise the
            # local count would credit children the campaign never gets to consider.
            ordered = self.inner.order(
                members,
                higher_is_better=higher_is_better,
                cost_table=cost_table,
                available=available,
            )
            n, hit_cap = self._local_modes(ordered)
            key = (getattr(members[0], "parent_input", "") or "", cls)
            self.local_modes[key] = max(n, self.local_modes.get(key, 0))
            if hit_cap:
                self.capped.add(key)
            if n >= self.min_modes:
                self.kept_groups += 1
                kept.append((ordered, ordered[0].reward if ordered else float("-inf")))
            else:
                self.dropped_groups += 1
                # Count what the INNER policy would have offered, not every member: children
                # free-frag drops anyway are not this filter's doing, and charging them here would
                # inflate the apparent cost of the gate.
                self.dropped_children += len(ordered)
        if self.order_mode == "group":
            kept.sort(key=lambda gr: _reward_key(gr[1]), reverse=self.higher_is_better)
            return [c for ordered, _ in kept for c in ordered]
        flat = [c for ordered, _ in kept for c in ordered]
        return self.inner.order(
            flat, higher_is_better=higher_is_better, cost_table=cost_table, available=available
        )


def collapse_rows(rows, local_modes, capped, arm, slice_label, cell, min_modes):
    """Per group actually used: what it COULD have delivered alone vs what it DID deliver.

    Keyed on ``parent_input`` (stereo-aware) because that is what the policy recorded. A group gated
    in at >= ``min_modes`` that delivered fewer has COLLAPSED -- its members were distinct from each
    other but not from molecules already accepted elsewhere.

    ONE CONFOUND, AND WHY ``hub_truncated`` EXISTS. A budget stops the campaign mid-walk, so a group
    can deliver less than its local capacity simply because the run ended, not because anything was
    filtered out. Reading that as collapse would overstate it. ``HubBatchingStrategy`` walks hubs
    sequentially and offers EVERY child of a hub to the selector before moving on -- it breaks out of
    the child loop only when the budget stops it, and then leaves the hub loop too -- so exactly one
    hub is ever partially consumed: the one holding the last accepted mode. Every group on an earlier
    hub is CLOSED, and its retention is exact. Groups on the truncated hub are flagged and excluded
    from the retention summary rather than quietly averaged in. (The flag covers every group on that
    hub, which is right for both offering orders: with groups interleaved they are all mid-flight.)
    """
    got: dict = defaultdict(list)
    for r in rows:
        got[(r.get("parent_input", ""), r["named_reaction"])].append(r)
    last = max(rows, key=lambda r: r["step"]) if rows else None
    trunc_parent = last.get("parent_input", "") if last else None
    out = []
    for key, members in sorted(got.items(), key=lambda kv: -len(kv[1])):
        parent, cls = key
        loc = local_modes.get(key)
        truncated = parent == trunc_parent
        out.append(
            {
                "cell": cell,
                "arm": arm,
                "slice": slice_label,
                "named_reaction": cls,
                "parent_input": parent,
                "local_modes": loc,
                "local_capped": key in capped,
                "delivered_modes": len(members),
                "retained_frac": round(len(members) / loc, 3) if loc else None,
                "delivered_ge_min": len(members) >= min_modes,
                # True = the budget stopped mid-walk on this group's hub, so its shortfall is NOT
                # evidence of collapse. Excluded from the retention summary.
                "hub_truncated": truncated,
                "n_wells": len({m["reagents"] for m in members}),
                "first_step": min(m["step"] for m in members),
                "last_step": max(m["step"] for m in members),
            }
        )
    return out


# ----------------------------------------------------------------- per-mode table
def hb_reaction_index(hubs):
    """``(hub_key, child_smiles) -> RxChild``, with genuinely ambiguous keys marked, not resolved.

    ``source_hub`` on an accepted mode is the STEREO-STRIPPED ``hub_key``, so two stereoisomeric
    hubs (distinct ``hub_input``, separately enumerated) collapse onto one key. Almost always their
    child was made by the same reaction and the join is unambiguous; occasionally it was not. Taking
    whichever came first would put a wrong reaction name on a molecule with nothing to show it, so
    those keys are recorded as ambiguous and the modes that land on one are labelled and counted.
    ``ambiguous`` counts KEYS; how many selected modes actually hit one is reported downstream, and
    is the number that matters.
    """
    idx = {}
    ambiguous = set()
    for h in hubs:
        for c in h.children:
            k = (h.hub_key, c.smiles)
            prev = idx.get(k)
            if prev is not None and (prev.named, prev.template_id) != (c.named, c.template_id):
                ambiguous.add(k)
            elif prev is None:
                idx[k] = c
    return idx, ambiguous


def hb_mode_rows(result, idx, ambiguous=frozenset()):
    """One row per accepted hub-batching mode, with its final step's reaction class."""
    rows, missing, n_ambig = [], 0, 0
    for p in result.accepted:
        k = (p.source_hub or "", p.smiles)
        c = idx.get(k)
        if k in ambiguous:
            n_ambig += 1
            c = None
        elif c is None:
            missing += 1
        rows.append(
            {
                "step": p.step,
                "smiles": p.smiles,
                "reward": round(p.reward, 4),
                "cum_reactions": p.cum_reactions,
                "reactions_added": p.reactions_added,
                "parent": p.source_hub or "",
                "parent_input": c.parent_input if c else "",
                "named_reaction": c.named
                if c
                else ("(ambiguous)" if k in ambiguous else "(unjoined)"),
                "template_id": c.template_id if c else "",
                "reagents": ".".join(c.reagents) if c else "",
                "n_reagents": len(c.reagents) if c else 0,
            }
        )
    return rows, missing, n_ambig


def bc_mode_rows(result, routes: dict):
    """Best-candidate modes. Its accepted molecules are sampled terminals, not enumerated children,
    so the final step comes from the SAMPLE stage's ``routes.json``.

    The parent here is the final step's own ``input`` -- the true immediate precursor -- NOT the hub
    the cost model happened to assign it to. Those are different questions: the assigned hub is the
    cheapest valid shared prefix for ACCOUNTING, while parallel chemistry is about what you weigh
    into the well. We record both and report how often they coincide.
    """
    rows, missing, agree = [], 0, 0
    for p in result.accepted:
        rt = routes.get(p.smiles) or {}
        steps = rt.get("steps") or []
        if not steps:
            missing += 1
            rows.append(
                {
                    "step": p.step,
                    "smiles": p.smiles,
                    "reward": round(p.reward, 4),
                    "cum_reactions": p.cum_reactions,
                    "reactions_added": p.reactions_added,
                    "parent": "",
                    "assigned_hub": p.source_hub or "",
                    "named_reaction": "(no route)",
                    "template_id": "",
                    "reagents": "",
                    "n_reagents": 0,
                }
            )
            continue
        last = steps[-1]
        tmpl = last.get("reaction", "") or ""
        tid, _ = parse_template(tmpl)
        parent = last.get("input") or ""
        if parent and parent == (p.source_hub or ""):
            agree += 1
        rows.append(
            {
                "step": p.step,
                "smiles": p.smiles,
                "reward": round(p.reward, 4),
                "cum_reactions": p.cum_reactions,
                "reactions_added": p.reactions_added,
                "parent": parent,
                "assigned_hub": p.source_hub or "",
                "named_reaction": named_reaction(tmpl) if tmpl else "(no reaction recorded)",
                "template_id": tid,
                "reagents": ".".join(last.get("reactants", ()) or ()),
                "n_reagents": len(last.get("reactants", ()) or ()),
            }
        )
    return rows, missing, agree


# ----------------------------------------------------------------- grouping + stats
def group_stats(rows, key_fn, label: str) -> dict:
    """Group-size distribution + how much of the library sits in a group of at least N.

    Reported on TWO units, because they differ and only one of them is the bench workload:

      * **molecules** -- library members, the unit every other readout in this project counts.
      * **wells** -- distinct ``(parent, reagent set)`` pairs within the group. A hub carrying two
        amines of the same class lets one template fire at either site, so one ketone yields two
        regioisomers: two library members out of ONE reaction vessel, separable afterwards. Both
        molecules are real and score differently, so the mode count is not wrong -- but as *parallel
        chemistry* that is one well, not two. Quoting molecules alone would overstate the array width
        exactly where a hub is polyfunctional, which is precisely where hub-batching likes to sit.

    The well key must carry the PARENT, not the reagent alone. Within a Class A group the parent is
    constant so the two are equivalent -- but a Class B group spans parents, and there the same
    amine reacting with four different scaffolds is four wells, not one. Keying on the reagent alone
    collapsed those and made best-candidate's Class B groups read as ~5x more concentrated than they
    are (114 molecules scored as 23 wells on drd2_seed43), which would have understated the control.
    """

    def well(r):
        return (r["parent"], r["reagents"])

    groups: dict = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    wells = sorted((len({well(m) for m in v}) for v in groups.values()), reverse=True)
    n_mol = len(rows)
    n_wells = sum(wells)
    out = {
        "class": label,
        "n_modes": n_mol,
        "n_wells": n_wells,
        "mols_per_well": round(n_mol / n_wells, 3) if n_wells else None,
        "n_groups": len(groups),
        "n_singletons": sum(1 for s in sizes if s == 1),
        "largest_group": sizes[0] if sizes else 0,
        "largest_group_wells": wells[0] if wells else 0,
        "median_group": statistics.median(sizes) if sizes else float("nan"),
        "mean_group": round(n_mol / len(groups), 3) if groups else float("nan"),
        "gini_group": round(_gini(sizes), 3) if sizes else None,
        "largest_group_share": round(sizes[0] / n_mol, 3) if n_mol else None,
    }
    for t in THRESHOLDS:
        n_in = sum(s for s in sizes if s >= t)
        out[f"n_groups_ge{t}"] = sum(1 for s in sizes if s >= t)
        out[f"mols_in_groups_ge{t}"] = n_in
        out[f"frac_in_groups_ge{t}"] = round(n_in / n_mol, 3) if n_mol else None
        w_in = sum(w for w in wells if w >= t)
        out[f"n_wellgroups_ge{t}"] = sum(1 for w in wells if w >= t)
        out[f"frac_wells_in_groups_ge{t}"] = round(w_in / n_wells, 3) if n_wells else None
    return out, groups


def group_rows(groups, class_label: str, arm: str, slice_label: str, cell: str):
    """One CSV row per group, largest first."""
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        parent, rxn = key if isinstance(key, tuple) else ("", key)
        reagents = [m["reagents"] for m in members if m["reagents"]]
        out.append(
            {
                "cell": cell,
                "arm": arm,
                "slice": slice_label,
                "class": class_label,
                "named_reaction": rxn,
                "parent": parent,
                "n_molecules": len(members),
                "n_wells": len({(m["parent"], m["reagents"]) for m in members}),
                "n_multi_product_wells": sum(
                    1
                    for _k, n in Counter((m["parent"], m["reagents"]) for m in members).items()
                    if n > 1
                ),
                "n_distinct_reagent_sets": len(set(reagents)),
                "n_distinct_parents": len({m["parent"] for m in members}),
                "n_distinct_templates": len({m["template_id"] for m in members}),
                "reward_min": round(min(m["reward"] for m in members), 3),
                "reward_median": round(statistics.median([m["reward"] for m in members]), 3),
                "reward_max": round(max(m["reward"] for m in members), 3),
                "first_step": min(m["step"] for m in members),
                "last_step": max(m["step"] for m in members),
            }
        )
    return out


def decomposition(rows):
    """How each Class-B group splits into Class-A arrays: the count of distinct parents per reaction
    class. A Class-B group of 24 spread over 24 parents is a very different plate from one over 2.
    """
    by_rxn = defaultdict(list)
    for r in rows:
        by_rxn[r["named_reaction"]].append(r)
    out = []
    for rxn, members in sorted(by_rxn.items(), key=lambda kv: -len(kv[1])):
        per_parent = Counter(m["parent"] for m in members)
        out.append(
            {
                "named_reaction": rxn,
                "n_molecules": len(members),
                "n_class_a_subgroups": len(per_parent),
                "largest_class_a": max(per_parent.values()),
                "mean_class_a": round(len(members) / len(per_parent), 2),
            }
        )
    return out


def parents_per_mode_stats(rows):
    """Molecules-per-parent, i.e. the Logs/038 batch size, recomputed on this slice so Class A can be
    read against the batch it refines."""
    sizes = list(Counter(r["parent"] for r in rows).values())
    return _stats_for(sizes) if sizes else {}


# ----------------------------------------------------------------- figure
# One colour per arm, reused across both panels so the eye tracks the arm, not the panel.
ARM_COLOURS = {"hub_batching": "#2a6f9e", "best_candidate": "#8a8984"}
PAR_COLOUR = "#b8762a"


def plot_groups(path, arms, budget_reactions: int, tag: str) -> None:
    """Rank-size curves of the group sizes at the reaction budget: Class A beside Class B.

    A rank-size curve rather than a histogram, because the question is "is there a big group at all",
    and the answer is the SHAPE of the tail -- a histogram of a 79-molecule library in 14 groups is
    mostly empty bins. The plate-size guides (4 / 24 wells) are drawn so the y-axis is read against
    bench units instead of against itself.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[pg] plot skipped ({exc})")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, (label, key_fn, blurb) in zip(
        axes,
        [
            (
                "Class A",
                lambda r: (r["parent"], r["named_reaction"]),
                "same parent + same reaction\n(one substrate, N partners)",
            ),
            (
                "Class B",
                lambda r: r["named_reaction"],
                "same reaction, any parent\n(one set of conditions)",
            ),
        ],
    ):
        for arm, rows in arms:
            sub = [r for r in rows if r["cum_reactions"] <= budget_reactions]
            if not sub:
                continue
            sizes = sorted(Counter(key_fn(r) for r in sub).values(), reverse=True)
            colour = ARM_COLOURS.get(arm, PAR_COLOUR)
            ax.plot(
                range(1, len(sizes) + 1),
                sizes,
                "-o",
                ms=4,
                lw=1.6,
                color=colour,
                label=f"{arm} ({len(sub)} molecules)",
            )
        for well, style in ((4, ":"), (24, "--")):
            ax.axhline(well, color="#c9c6c0", ls=style, lw=1.0, zorder=0)
            ax.text(
                0.99,
                well,
                f" {well}-well ",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=7,
                color="#8d8a83",
            )
        ax.set_title(f"{label} — {blurb}", fontsize=9)
        ax.set_xlabel("group rank (largest first)")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("molecules in the group (↑ more parallel)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"{tag}: parallelizable groups at a {budget_reactions}-reaction budget", fontsize=11
    )
    # At a FIXED reaction budget the arms deliver different library sizes (that IS the headline
    # result), so a Class B curve drawn here is partly just the bigger library having more of
    # everything. Class A is immune -- best-candidate's arrays are singletons at any size -- but
    # Class B is not, and the honest comparison is at equal library size. Say so on the figure
    # rather than only in the log, because this panel is exactly the one that would be screenshotted.
    fig.text(
        0.5,
        0.005,
        "Class B is confounded by library size at a fixed budget — compare it at equal library "
        "size (slice=full in group_stats.csv), where the two arms are indistinguishable.",
        ha="center",
        fontsize=7.5,
        color="#8d8a83",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=130)
    print(f"[pg] wrote {path}")


# ----------------------------------------------------------------- driver
def slices(rows, budget_reactions: int):
    """The two library slices: the fixed reaction budget (primary readout, CLAUDE.md) and the full
    mode-budget curve. The first is a PREFIX of the second -- same ordering, read at two points."""
    at_r = [r for r in rows if r["cum_reactions"] <= budget_reactions]
    return [(f"at{budget_reactions}rxn", at_r), ("full", rows)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-dir",
        required=True,
        help="sample dir (records.csv, compositions.json, routes.json)",
    )
    ap.add_argument("--enum-children", required=True, help="FROZEN enum_children.json")
    ap.add_argument("--snapshot", default="", help="SCENT fragments_<N>.json (nested cost model)")
    ap.add_argument("--reward-threshold", type=float, required=True)
    ap.add_argument("--similarity", type=float, default=0.5)
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--budget-reactions", type=int, default=100)
    ap.add_argument("--budget-modes", type=int, default=300)
    ap.add_argument("--child-policy", default="free_frag", choices=["reward", "free_frag"])
    ap.add_argument("--prebuild-k", type=int, default=20)
    ap.add_argument("--rank-by", default="build_score")
    ap.add_argument(
        "--min-group",
        type=int,
        default=0,
        help="speculative third filter: also run a hub-batching arm that keeps only children in a "
        "(parent, reaction class) group of >= N reward-passing siblings. 0 = off.",
    )
    ap.add_argument("--count-mode", default="reward", choices=["reward", "reward_free"])
    ap.add_argument(
        "--min-modes",
        type=int,
        default=0,
        help="MODE-gated parallelization filter: keep only children in a (parent, reaction class) "
        "group that can itself yield >= N mutually distinct molecules, measured by a micro greedy "
        "selection inside the group. Unlike --min-group this gates on deliverable diversity, not "
        "availability. 0 = off. Runs one arm per --order-modes value.",
    )
    ap.add_argument(
        "--order-modes",
        default="reward,group",
        help="comma-separated offering orders for the --min-modes arms: 'reward' (groups interleaved, "
        "same shape as the baseline) and/or 'group' (groups contiguous, best-first). Both by default, "
        "so the effect of contiguity is measured rather than assumed.",
    )
    ap.add_argument(
        "--local-cap",
        type=int,
        default=200,
        help="cap on each within-group micro selection (reported counts saturate here).",
    )
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()

    adir = Path(a.analysis_dir)
    out = Path(a.out_dir) if a.out_dir else HERE / "results" / "parallel_groups" / a.tag
    out.mkdir(parents=True, exist_ok=True)

    cands, comps = _load_candidates(adir, a.higher_is_better)
    hubs, cov = load_hubs_with_reactions(Path(a.enum_children), comps)
    print(
        f"[pg] {a.tag}: {len(cands)} candidates, {len(hubs)} hubs, "
        f"{cov['n_children']:,} children, reaction coverage {cov['rxn_coverage']:.4%}, "
        f"multi-step children {cov['n_multi_step_children']}"
    )
    ok = parity_check(hubs, comps, a.enum_children)
    print(f"[pg] loader parity vs run_campaign._load_enumerated_hubs: {'OK' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit(
            "[pg] ABORT: re-derived hubs differ from the campaign's -- the groups below would "
            "describe a library we never selected."
        )

    snapshot = json.load(open(a.snapshot)) if a.snapshot else {}
    cost_table = load_cost_table_from_snapshot(snapshot)
    common = dict(
        target=a.tag,
        reward_threshold=a.reward_threshold,
        similarity=a.similarity,
        higher_is_better=a.higher_is_better,
    )
    prebuilt = None
    if a.prebuild_k > 0:
        ranked = rank_fragments(
            hubs,
            cost_table,
            a.reward_threshold,
            method=a.rank_by,
            higher_is_better=a.higher_is_better,
        )
        prebuilt = {f for f, _ in ranked[: a.prebuild_k]}
        print(f"[pg] pre-select-K: {len(prebuilt)} fragments pre-built")

    budget = ("modes", a.budget_modes)
    bc, _ = run_timed(build_strategy("best_candidate", cands, cost_table, comps, **common), budget)
    hb, _ = run_timed(
        build_strategy(
            "hub_batching",
            hubs,
            cost_table,
            comps,
            child_policy=make_child_policy(a.child_policy),
            prebuilt_fragments=prebuilt,
            **common,
        ),
        budget,
    )
    print(
        f"[pg] hub_batching: {hb.total_modes} modes / {hb.total_reactions} rxns "
        f"({hb.distinct_hubs_used} hubs) | best_candidate: {bc.total_modes} / {bc.total_reactions}"
    )

    routes = {}
    rp = adir / "routes.json"
    if rp.exists():
        routes = json.loads(rp.read_text())
    print(f"[pg] routes.json: {len(routes):,} molecules")

    idx, ambiguous = hb_reaction_index(hubs)
    hb_rows, hb_missing, hb_ambig = hb_mode_rows(hb, idx, ambiguous)
    bc_rows, bc_missing, bc_agree = bc_mode_rows(bc, routes)
    print(
        f"[pg] join: hub_batching unjoined {hb_missing}/{len(hb_rows)}, "
        f"ambiguous {hb_ambig}/{len(hb_rows)} (of {len(ambiguous)} ambiguous keys) | "
        f"best_candidate no-route {bc_missing}/{len(bc_rows)}, "
        f"final-step input == assigned hub for {bc_agree}/{len(bc_rows)}"
    )

    arms = [("hub_batching", hb_rows), ("best_candidate", bc_rows)]

    par_meta = None
    if a.min_group > 0:
        pol = ParallelGroupChildPolicy(
            make_child_policy(a.child_policy),
            reward_threshold=a.reward_threshold,
            higher_is_better=a.higher_is_better,
            min_group=a.min_group,
            count_mode=a.count_mode,
        )
        hbp, _ = run_timed(
            build_strategy(
                "hub_batching",
                hubs,
                cost_table,
                comps,
                child_policy=pol,
                prebuilt_fragments=prebuilt,
                **common,
            ),
            budget,
        )
        hbp_rows, hbp_missing, hbp_ambig = hb_mode_rows(hbp, idx, ambiguous)
        par_meta = {
            "min_group": a.min_group,
            "count_mode": a.count_mode,
            "total_modes": hbp.total_modes,
            "total_reactions": hbp.total_reactions,
            "distinct_hubs_used": hbp.distinct_hubs_used,
            "reward_gen_calls": hbp.total_reward_gen_calls,
            "children_dropped_by_filter": pol.dropped_children,
            "groups_kept": pol.kept_groups,
            "groups_dropped": pol.dropped_groups,
            "unjoined": hbp_missing,
            "ambiguous": hbp_ambig,
        }
        print(
            f"[pg] parallel-filter arm (min_group={a.min_group}, count={a.count_mode}): "
            f"{hbp.total_modes} modes / {hbp.total_reactions} rxns, "
            f"{pol.dropped_children:,} children dropped"
        )
        arms.append((f"hub_batching_par{a.min_group}", hbp_rows))

    # Mode-gated arms: gate on what a group can DELIVER, then measure how much of that survives the
    # global diversity filter. One arm per offering order, so contiguity is measured not assumed.
    mode_meta, collapse = [], []
    if a.min_modes > 0:
        for om in [x.strip() for x in a.order_modes.split(",") if x.strip()]:
            pol = ModeGroupChildPolicy(
                make_child_policy(a.child_policy),
                reward_threshold=a.reward_threshold,
                similarity=a.similarity,
                higher_is_better=a.higher_is_better,
                min_modes=a.min_modes,
                local_cap=a.local_cap,
                order_mode=om,
            )
            hbm, _ = run_timed(
                build_strategy(
                    "hub_batching",
                    hubs,
                    cost_table,
                    comps,
                    child_policy=pol,
                    prebuilt_fragments=prebuilt,
                    **common,
                ),
                budget,
            )
            rows_m, miss_m, ambig_m = hb_mode_rows(hbm, idx, ambiguous)
            arm = f"hub_batching_modes{a.min_modes}_{om}"
            for slice_label, sub in slices(rows_m, a.budget_reactions):
                collapse += collapse_rows(
                    sub, pol.local_modes, pol.capped, arm, slice_label, a.tag, a.min_modes
                )
            gated = [v for v in pol.local_modes.values() if v >= a.min_modes]
            mode_meta.append(
                {
                    "arm": arm,
                    "min_modes": a.min_modes,
                    "order_mode": om,
                    "local_cap": a.local_cap,
                    "total_modes": hbm.total_modes,
                    "total_reactions": hbm.total_reactions,
                    "distinct_hubs_used": hbm.distinct_hubs_used,
                    "reward_gen_calls": hbm.total_reward_gen_calls,
                    "groups_gated_in": pol.kept_groups,
                    "groups_gated_out": pol.dropped_groups,
                    "children_dropped": pol.dropped_children,
                    "groups_scored": len(pol.local_modes),
                    "groups_local_cap_hit": len(pol.capped),
                    "median_local_modes_of_gated": statistics.median(gated) if gated else None,
                    "unjoined": miss_m,
                    "ambiguous": ambig_m,
                    "stop_reason": hbm.stop_reason,
                }
            )
            print(
                f"[pg] mode-gated arm {arm}: {hbm.total_modes} modes / {hbm.total_reactions} rxns, "
                f"{pol.kept_groups} groups gated in / {pol.dropped_groups} out, "
                f"{pol.dropped_children:,} children dropped, stop={hbm.stop_reason}"
            )
            arms.append((arm, rows_m))

    # ------------------------------------------------------------- write
    stat_rows, all_groups, decomp_rows = [], [], []
    for arm, rows in arms:
        fields = sorted({k for r in rows for k in r})
        with open(out / f"modes_{arm}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        for slice_label, sub in slices(rows, a.budget_reactions):
            a_stats, a_groups = group_stats(sub, lambda r: (r["parent"], r["named_reaction"]), "A")
            b_stats, b_groups = group_stats(sub, lambda r: r["named_reaction"], "B")
            batch = parents_per_mode_stats(sub)
            for st in (a_stats, b_stats):
                st.update(
                    cell=a.tag,
                    arm=arm,
                    slice=slice_label,
                    n_parents=len({r["parent"] for r in sub}),
                    batch_median=batch.get("median"),
                    batch_max=batch.get("max"),
                    batch_gini=batch.get("gini"),
                    reactions_used=max((r["cum_reactions"] for r in sub), default=0),
                )
                stat_rows.append(st)
            all_groups += group_rows(a_groups, "A", arm, slice_label, a.tag)
            all_groups += group_rows(b_groups, "B", arm, slice_label, a.tag)
            for d in decomposition(sub):
                d.update(cell=a.tag, arm=arm, slice=slice_label)
                decomp_rows.append(d)

    for name, rows_ in (
        ("group_stats.csv", stat_rows),
        ("groups.csv", all_groups),
        ("class_b_decomposition.csv", decomp_rows),
        ("collapse.csv", collapse),
    ):
        if not rows_:
            continue
        fields = list(dict.fromkeys(k for r in rows_ for k in r))
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows_)

    (out / "summary.json").write_text(
        json.dumps(
            {
                "tag": a.tag,
                "inputs": {
                    "analysis_dir": str(adir),
                    "enum_children": a.enum_children,
                    "snapshot": a.snapshot,
                    "n_candidates": len(cands),
                    "n_hubs": len(hubs),
                    **cov,
                    "n_routes": len(routes),
                },
                "config": {
                    "reward_threshold": a.reward_threshold,
                    "similarity": a.similarity,
                    "higher_is_better": a.higher_is_better,
                    "child_policy": a.child_policy,
                    "prebuild_k": a.prebuild_k,
                    "budget_reactions": a.budget_reactions,
                    "budget_modes": a.budget_modes,
                },
                "arms": {
                    "hub_batching": {
                        "modes": hb.total_modes,
                        "reactions": hb.total_reactions,
                        "hubs": hb.distinct_hubs_used,
                        "reward_gen_calls": hb.total_reward_gen_calls,
                    },
                    "best_candidate": {
                        "modes": bc.total_modes,
                        "reactions": bc.total_reactions,
                        "hubs": bc.distinct_hubs_used,
                    },
                },
                "joins": {
                    "hb_unjoined": hb_missing,
                    "hb_ambiguous_modes": hb_ambig,
                    "n_ambiguous_hub_keys": len(ambiguous),
                    "bc_no_route": bc_missing,
                    "bc_parent_equals_assigned_hub": bc_agree,
                },
                "parallel_filter": par_meta,
                "mode_gated_arms": mode_meta,
                "group_stats": stat_rows,
            },
            indent=2,
        )
    )
    print(f"[pg] wrote modes_*.csv + group_stats.csv + groups.csv + summary.json -> {out}")

    plot_groups(out / "parallel_groups.png", arms, a.budget_reactions, a.tag)

    for m in mode_meta:
        for sl in (f"at{a.budget_reactions}rxn", "full"):
            rows_c = [r for r in collapse if r["arm"] == m["arm"] and r["slice"] == sl]
            closed = [r for r in rows_c if not r["hub_truncated"]]
            held = sum(1 for r in closed if r["delivered_ge_min"])
            deliv = sum(r["delivered_modes"] for r in closed)
            loc = sum(r["local_modes"] or 0 for r in closed)
            ret = f"{deliv / loc:.1%}" if loc else "n/a"
            print(
                f"  {m['arm']:<34} {sl:<10} {len(rows_c)} groups used "
                f"({len(closed)} closed, {len(rows_c) - len(closed)} truncated) | of the closed: "
                f"{held}/{len(closed)} still >= {a.min_modes} modes, "
                f"{deliv}/{loc} modes retained ({ret})"
            )

    # ------------------------------------------------------------- console readout
    for st in stat_rows:
        print(
            f"  {st['arm']:<22} {st['slice']:<10} class {st['class']}: "
            f"{st['n_modes']:>3} modes / {st['n_wells']:>3} wells in {st['n_groups']:>3} groups "
            f"(largest {st['largest_group']:>3} mol = {st['largest_group_wells']:>3} wells, "
            f"mol>=4: {st['frac_in_groups_ge4']}, mol>=10: {st['frac_in_groups_ge10']}, "
            f"wells>=10: {st['frac_wells_in_groups_ge10']})"
        )


if __name__ == "__main__":
    main()
