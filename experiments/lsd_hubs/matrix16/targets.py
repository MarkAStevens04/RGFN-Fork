"""Per-target science for the 16-cell LSD-Flow matrix — the single source of truth for the
reward gate (value + direction) and the enumeration cost class of each of the four benchmark
systems.

Why a table and not scattered constants: the campaign / sweep drivers used to hard-code
``--reward-threshold 7.0`` (an sEH-only value) and ``higher_is_better`` per call. With four
targets that diverge in scale AND direction (docking is lower-is-better), one authoritative
table keeps every driver honest and makes "add a 5th target" a one-entry edit.

Grounding for each gate:
  * **sEH**  proxy value, higher-is-better. Hit bar **7.0** = the paper-comparable value used
    across Logs/029/031/033. Logs/034 showed 7.0 is optimistic for the proxy scale (real
    inhibitors top ~7.7; empirically useful bar ~5-6), so 5.0/6.0 are offered as variants —
    the bar is a CLI knob, never load-bearing for the cost *comparison* (Logs/035).
  * **DRD2** activity probability in [0,1], higher-is-better. Bar **0.5** (Logs/028 fixed the
    DRD2 hit-bar bug where it inherited sEH's 7.0).
  * **6TD3** neosubstrate differential (Vina T2-T1), **lower-is-better**. Bar **-2.0** = the
    known-glue range (RESEARCH_CONTEXT.md; Logs/011/014). *(provisional — confirm the recorded
    reward column's sign/scale when the docking cells activate; see reward_note.)*
  * **ClpP** docking score, **lower-is-better**. Bar **-2.0**. *(provisional — same caveat.)*

``reward_type`` drives the enumeration cost path (proposal §6): ``surrogate`` targets score
enumerated hub children with a fast in-env proxy (free to enumerate); ``docking`` targets need
fresh GPU docking per enumerated child (cross-env, ~1 s/mol via the persistent docking server)
and are **deferred** in this build — their cells are wired but inert until the docking-enum path
lands and their training finishes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Target:
    """One benchmark system's reward gate + enumeration cost class."""

    name: str  # scratch/dir key: seh | drd2 | 6td3 | clpp
    reward_name: str  # the reward_name the adapters/oracle bridge expect (== name here)
    higher_is_better: bool  # True for surrogate proxies; False for docking (lower energy better)
    mode_reward_threshold: float  # the hit gate before the diversity filter (a "mode" must clear it)
    reward_type: str  # "surrogate" (cheap proxy enum) | "docking" (GPU dock enum; deferred)
    threshold_variants: List[float] = field(default_factory=list)  # extra bars to sweep (Logs/035)
    reward_note: str = ""
    # Docking targets only: the oracle name the persistent docking server / score_batch bridge
    # registers (``glue.oracles.docking_server``'s registry). Declared HERE, beside the gate it is
    # measured against, so adding a fifth target is a one-entry edit rather than a grep across
    # drivers. Empty for surrogate targets, which score in-process with no oracle.
    oracle: str = ""

    @property
    def is_docking(self) -> bool:
        return self.reward_type == "docking"

    @property
    def oracle_name(self) -> str:
        """The oracle to serve for this target, failing loudly rather than silently scoring with
        the wrong one. Oracle *constructor* args (num_modes, exhaustiveness) are NOT here — they
        live in each cell's training config (``reward.oracle_args``), which is the only record of
        what the checkpoint was actually trained against."""
        if not self.is_docking:
            raise ValueError(
                f"target {self.name!r} is a surrogate target; it has no docking oracle"
            )
        if not self.oracle:
            raise ValueError(f"docking target {self.name!r} has no oracle declared in targets.py")
        return self.oracle


# ============================================================================================
# ⛔ THE HIT-GATE STANDARD — 5% FPR against property-matched decoys (decided 2026-08-21)
#
# Every gate below is the score at which **5% of that target's property-matched decoys pass**,
# measured on its own known-actives-vs-decoys set. Nothing else. Do not hand-pick a bar, do not
# reuse a published convention, and do not carry one target's number to another.
#
# WHY A COMMON CRITERION RATHER THAN A COMMON NUMBER. The four rewards live on four incompatible
# scales (an arbitrary proxy value, a probability, kcal/mol, a pK estimate), so "the same bar"
# is meaningless. What CAN be equalised is the effect of the bar. Before this rule the decoy
# pass-rate ranged from 4% (DRD2, 6TD3-B) to 23% (ClpP): a "mode" on ClpP admitted realistic
# non-binders six times more often than a mode on DRD2, which quietly made mode counts
# non-comparable across targets — the one thing a four-target matrix has to get right.
#
# WHY FPR AND NOT TPR OR YOUDEN. The gate answers "does this molecule COUNT", and the failure
# that corrupts the benchmark is decoys inflating a mode count. Fixing FPR bounds that
# contamination identically everywhere. Fixing TPR equalises something we never report and lets
# contamination float (at TPR 75% the sEH gate admits 49% of decoys — barely a gate). Youden's J
# weights a false positive and a false negative equally, which is wrong here: admitting junk
# costs more than missing a hit, because every arm's headline number is a COUNT.
#
# WHY 5% AND NOT 1%. At 1% the weaker oracles starve — sEH would retain 3% of known actives, and
# a gate that discards 97% of actives makes most cells pool-limited, which is a different failure.
#
# The bars are empirical grid points, not round numbers, ON PURPOSE: rounding breaks the exact
# property the standard is defined by. Re-derive with `calibrate_gates.py` if a decoy set changes.
# ============================================================================================

TARGETS: Dict[str, Target] = {
    "seh": Target(
        name="seh",
        reward_name="seh",
        higher_is_better=True,
        # HEADLINE BAR = 5.0 (decided 2026-08-18). Was 7.0, which is the value the RGFN paper's own
        # sEH numbers are quoted at -- but 7.0 sits in the proxy's optimised tail, NOT in the range
        # where the proxy tracks real activity: entry 034 measured only weak enrichment of known
        # actives (AUROC 0.76/0.68) and 0 of 2,315 real actives reach 8.0, and entry 051 found the
        # meaningful range is ~5-6 with a diversity collapse past ~7.
        #
        # The operational reason is what settled it. At 7.0 the comparison is not like-for-like: an
        # arm falls short of the 300-mode budget in rgfn_seh and BOTH rxnflow cells on EVERY one of
        # the three seeds, so the ratio is measured over a starved library and understates
        # hub-batching (entry 055 saw the same thing on one seed; three seeds confirm it is
        # systematic, not seed luck). At 5.0 every clean cell reaches 300/300 and the seed spread is
        # 1.9-3.2%. 7.0 remains in threshold_variants so the paper-comparable number stays quotable.
        mode_reward_threshold=5.68,  # 5% FPR (TPR 13%, 2.5x). WAS 5.0 -> 16% FPR.
        reward_type="surrogate",
        threshold_variants=[5.0, 5.68, 6.0, 7.0],
        reward_note="proxy value; HEADLINE 5.0 (calibrated, Logs/034/051); 7.0 = paper-comparable variant",
    ),
    "drd2": Target(
        name="drd2",
        reward_name="drd2",
        higher_is_better=True,
        mode_reward_threshold=0.345,  # 5% FPR (TPR 74%, 14.9x). WAS 0.5 -> 4% FPR.
        reward_type="surrogate",
        threshold_variants=[0.345, 0.5, 0.7, 0.9],
        reward_note="activity probability in [0,1]; 0.5 = calibrated active cutoff (Logs/028), "
        "0.7/0.9 = stricter bars for the gate-sensitivity sweep (symmetric to sEH 5/6/7)",
    ),
    "6td3": Target(
        name="6td3",
        reward_name="6td3",
        higher_is_better=False,
        mode_reward_threshold=-2.0,
        reward_type="docking",
        oracle="docking_6td3_gpu",  # two-tier differential; num_modes/exhaustiveness from the cfg
        threshold_variants=[-2.0],
        reward_note=(
            "SUPERSEDED for TRAINING by `6td3b` (Logs/072): the generator GAMES this "
            "differential -- its candidates clear this gate 78% of the time against 67% "
            "for real glues, while clearing the CNN gate 0 times out of 400. Retained "
            "UNCHANGED so published numbers stay reproducible, and still emitted by every "
            "run. Separately, -2.0 gives only 2.1x enrichment against property-matched "
            "decoys (Logs/069) versus 82.9x against the older warhead-matched set."
        ),
    ),
    "6td3b": Target(
        name="6td3b",
        reward_name="6td3b",
        higher_is_better=True,
        # 6TD3-B: gnina's CNN_VS = CNNaffinity x CNNscore on the selected Tier-2 pose, HIGHER better.
        # Replaces 6TD3's Vina T2-T1 differential as the reward for all new CDK12-DDB1 runs
        # (Logs/072). Gate on the `cnn_vs` column.
        #
        # WHY THE PRODUCT, not either head. gnina emits two independent numbers whose authors note
        # "need not agree": CNNscore = P(pose within 2 A of true), CNNaffinity = predicted pK.
        #   * cnnaff_t2 ALONE is blind to our failure mode -- AUROC 0.521 (chance) separating real
        #     glues from our own reward-optimised candidates. gnina's affinity head is trained with a
        #     HINGE on high-RMSD poses (penalised only for over-predicting), so an affinity read off a
        #     low-confidence pose is weakly supervised.
        #   * cnnsc_t2 ALONE catches it (0.965) but is bounded [0,1] and saturates (glue p90 0.987),
        #     leaving a GFlowNet no headroom.
        # CNN_VS is gnina's OWN documented virtual-screening metric, so this is published practice
        # rather than our invention.
        #
        #   signal      AUROC vs decoys   AUROC vs our candidates   glues kept @5% FPR
        #   cnnaff_t2       0.804              0.521 (chance)             37.5%
        #   cnnsc_t2        0.923              0.965                      78.8%
        #   CNN_VS          0.917              0.946                      78.1%
        #
        # GATE 6.718 = the 5%-FPR standard (see the header block). Keeps 78.1% of real glues, admits
        # 5.0% of property-matched decoys and 3.5% of our candidates; enrichment 17.9x. Training
        # landscape is wide open -- our candidates median 2.63 vs 7.59 for real glues.
        mode_reward_threshold=6.718,
        reward_type="docking",
        oracle="docking_6td3b_gpu",
        threshold_variants=[6.0, 6.718, 7.0],
        reward_note=(
            "CALIBRATED (Logs/069/072): gnina CNN_VS = CNNaffinity x CNNscore on the selected "
            "Tier-2 pose, HIGHER is better. Gate on `cnn_vs`. 6.718 = the 5%-FPR standard "
            "(78.1% of glues, 5.0% of decoys, 17.9x). Supersedes `6td3` as the training reward; "
            "`6td3` is retained unchanged so published numbers stay reproducible."
        ),
    ),
    "clpp": Target(
        name="clpp",
        reward_name="clpp",
        higher_is_better=False,
        mode_reward_threshold=-9.1,  # 5% FPR (TPR 47%, 9.5x). WAS -8.0 -> 23% FPR, the
        # loosest gate of the four and the reason mode counts were not comparable.
        reward_type="docking",
        oracle="docking_clpp",  # single-target human ClpP 7UVU (Logs/045)
        threshold_variants=[-9.1, -9.0, -8.0],
        reward_note="CALIBRATED (Logs/045): raw QuickVina2-GPU Vina energy vs human ClpP "
        "(7UVU), lower-is-better. Gate on the RAW docking value (candidates.csv `raw_score` / "
        "snapshot `term_raw_score`), NOT the `score`=ReLU(-raw) reward column. -8.0 is the "
        "Youden-optimal cutoff separating 183 ChEMBL ClpP binders from property-matched decoys "
        "(AUROC 0.895, TPR 88%/FPR 23%); the old -2.0 was non-binding (~100% of every generator "
        "cleared it). -9.0 = the 95%-decoy-specificity variant (TPR 48%/FPR 6%). "
        "experiments/oracle_validation/docking_clpp/.",
    ),
}


def get_target(name: str) -> Target:
    if name not in TARGETS:
        raise KeyError(f"Unknown target {name!r}. Known: {sorted(TARGETS)}")
    return TARGETS[name]
