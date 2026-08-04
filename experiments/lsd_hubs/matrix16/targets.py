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


TARGETS: Dict[str, Target] = {
    "seh": Target(
        name="seh",
        reward_name="seh",
        higher_is_better=True,
        mode_reward_threshold=7.0,
        reward_type="surrogate",
        threshold_variants=[5.0, 6.0, 7.0],
        reward_note="proxy value; 7.0 paper-comparable, 5-6 more calibrated (Logs/034)",
    ),
    "drd2": Target(
        name="drd2",
        reward_name="drd2",
        higher_is_better=True,
        mode_reward_threshold=0.5,
        reward_type="surrogate",
        threshold_variants=[0.5, 0.7, 0.9],
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
        reward_note="PROVISIONAL: neosubstrate differential (Vina T2-T1), lower-is-better; "
        "confirm recorded reward-column sign/scale when docking cells activate",
    ),
    "clpp": Target(
        name="clpp",
        reward_name="clpp",
        higher_is_better=False,
        mode_reward_threshold=-8.0,
        reward_type="docking",
        oracle="docking_clpp",  # single-target human ClpP 7UVU (Logs/045)
        threshold_variants=[-8.0, -9.0],
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
