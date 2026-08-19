"""Diversity + mode metrics for LSD-Flow batches (``docs/LSD_FLOW_PROPOSAL.md`` §7 #1, §11).

**Mode (paper-comparable).** Matches the RGFN / ``[bengio2021gflownet]`` mode definition exactly
as implemented by upstream ``rgfn.trainer.metrics.TanimotoSimilarityModes._extract_modes`` (and
the project's ``glue.metrics.dataset_metrics.count_modes``): a molecule is a new mode iff it is

  (a) **above a reward/binding threshold** (``proxy_term_threshold`` upstream — the "hit" bar), and
  (b) **Tanimoto-dissimilar from every mode already accepted** — greedy sphere-exclusion on ECFP
      (Morgan **radius 3**, 2048 bits, no features/chirality) at a fixed similarity threshold
      (default **0.7**).

Candidates are processed **best-reward-first**, so each mode's representative is its
highest-reward member — the molecule you would actually synthesize. This makes "modes" here
directly comparable to the generative-model mode counts RGFN/SCENT report.

Secondary: unique Bemis-Murcko scaffolds, :func:`mean_pairwise_similarity` (mean Tanimoto over all
pairs — the set-level "how alike is this library?" readout, on the same ECFP recipe as the mode
definition so the two are directly comparable), and :func:`count_butina_clusters` (the same
"how many families?" question with reward taken *out* of the clustering order). Lives on the validation axis (needs RDKit); the
production ``most_modes`` strategy takes an injected ``mode_counter`` so it can use this without
importing it.

**Reusing fingerprints across many calls.** A sweep that re-scores overlapping subsets of the same
pool (e.g. ``experiments/lsd_hubs/reward_diversity/``) would otherwise re-fingerprint the same
molecule dozens of times. Every function here accepts a ``fps=`` sequence aligned with ``keys``
(``None`` entries = unparseable, same treatment as a failed parse), and :func:`ecfp` /
:func:`murcko_scaffold` expose the exact recipes so a caller can build and cache them itself.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.ML.Cluster import Butina
except Exception:  # pragma: no cover - RDKit present in-env
    Chem = None
    Butina = None

# ECFP recipe identical to rgfn TanimotoSimilarityModes + glue.metrics.dataset_metrics (r=3, 2048).
_FP_RADIUS = 3
_FP_BITS = 2048
_MODE_SIMILARITY_THRESHOLD = 0.7


def ecfp(smiles: str):
    """The canonical fingerprint for every diversity/mode number in this project (Morgan r=3,
    2048 bits, no features/chirality). Public so callers can pre-compute and cache them; ``None``
    for an empty/unparseable SMILES or a missing RDKit."""
    if Chem is None or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=_FP_RADIUS, nBits=_FP_BITS, useFeatures=False, useChirality=False
    )


_ecfp = ecfp  # back-compat alias for the pre-public name


def murcko_scaffold(smiles: str) -> Optional[str]:
    """Bemis-Murcko scaffold SMILES, or ``None`` if it can't be derived. Public for the same
    caching reason as :func:`ecfp`; :func:`unique_scaffolds` is this plus a ``set``."""
    if Chem is None or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def _fp_at(keys: Sequence[str], fps: Optional[Sequence], i: int):
    """Fingerprint for ``keys[i]`` — from the caller's cache when supplied, else computed."""
    if fps is not None:
        return fps[i]
    return ecfp(keys[i])


def _passes_gate(reward, reward_threshold, higher_is_better) -> bool:
    """The reward/binding gate: ``>= threshold`` (higher-is-better) or ``<= threshold``. A
    missing threshold admits everything; NaN reward never passes a set gate."""
    if reward_threshold is None:
        return True
    if reward != reward:  # NaN
        return False
    return reward >= reward_threshold if higher_is_better else reward <= reward_threshold


def mode_representatives(
    keys: Sequence[str],
    rewards: Optional[Sequence[float]] = None,
    *,
    higher_is_better: bool = True,
    reward_threshold: Optional[float] = None,
    similarity_threshold: float = _MODE_SIMILARITY_THRESHOLD,
    max_modes: Optional[int] = None,
    fps: Optional[Sequence] = None,
) -> List[int]:
    """Indices (into ``keys``) of the mode representatives — the paper's greedy definition.

    Reproduces ``TanimotoSimilarityModes._extract_modes``: (1) drop keys failing the reward gate;
    (2) sort survivors best-reward-first (when ``rewards`` is given); (3) greedily accept a
    molecule as a new mode iff its Tanimoto similarity to every accepted mode is
    ``<= similarity_threshold``. The accepted molecule is that mode's representative (its
    highest-reward member, by the best-first order). Without ``rewards`` it degrades to
    structure-only modes in input order (no gate, no sort).

    ``fps``: optional pre-computed :func:`ecfp` fingerprints aligned with ``keys``. The
    accept/reject test uses ``BulkTanimotoSimilarity`` against the accepted set, which returns the
    same values as the per-pair call — identical modes, one C++ call per candidate instead of one
    per accepted mode.
    """
    idxs = [i for i, k in enumerate(keys) if k]
    if rewards is not None:
        idxs = [i for i in idxs if _passes_gate(rewards[i], reward_threshold, higher_is_better)]
        idxs.sort(
            key=lambda i: (rewards[i] if rewards[i] == rewards[i] else float("-inf")),
            reverse=higher_is_better,
        )
    reps: List[int] = []
    rep_fps: List[object] = []
    for i in idxs:
        fp = _fp_at(keys, fps, i)
        if fp is None:
            continue
        if rep_fps and max(DataStructs.BulkTanimotoSimilarity(fp, rep_fps)) > similarity_threshold:
            continue
        reps.append(i)
        rep_fps.append(fp)
        if max_modes and len(reps) >= max_modes:
            break
    return reps


def mode_assignments(
    keys: Sequence[str],
    rewards: Optional[Sequence[float]] = None,
    *,
    higher_is_better: bool = True,
    reward_threshold: Optional[float] = None,
    similarity_threshold: float = _MODE_SIMILARITY_THRESHOLD,
    fps: Optional[Sequence] = None,
) -> Dict[str, List[str]]:
    """``{representative_smiles: [member_smiles, ...]}`` — the FULL partition behind
    :func:`mode_representatives`, not just its centres.

    :func:`mode_representatives` answers "how many distinct families are here?" by naming one member
    of each. This answers "which family is each molecule in?", which is what an external selector
    needs if it is to be given *our* notion of distinctness rather than its own. Every molecule that
    passes the gate lands in exactly one group, so the group count equals ``count_modes`` on the same
    arguments — the two readouts cannot disagree.

    Assignment rule follows directly from how sphere exclusion rejects: a molecule was excluded
    because it was more similar than ``similarity_threshold`` to some already-accepted mode, so it
    belongs to that mode. Where several qualify it goes to the **most similar**, and a molecule
    similar to none (possible only for the fingerprint-less) is dropped rather than forced.

    Used to hand SPARROW our tau-modes as its clusters ([fromer2025diversity] defines clusters as an
    arbitrary, project-supplied partition precisely so this substitution is legitimate).
    """
    reps = mode_representatives(
        keys,
        rewards,
        higher_is_better=higher_is_better,
        reward_threshold=reward_threshold,
        similarity_threshold=similarity_threshold,
        fps=fps,
    )
    if not reps:
        return {}
    if Chem is None:  # RDKit-unavailable fallback, mirroring count_modes: every key its own group
        return {keys[i]: [keys[i]] for i in reps}

    rep_fps = [_fp_at(keys, fps, i) for i in reps]
    groups: Dict[str, List[str]] = {keys[i]: [] for i in reps}
    rep_keys = [keys[i] for i in reps]
    rep_pos = {i: n for n, i in enumerate(reps)}

    for i, k in enumerate(keys):
        if not k:
            continue
        if i in rep_pos:  # a representative is the first member of its own group
            groups[k].append(k)
            continue
        if rewards is not None and not _passes_gate(rewards[i], reward_threshold, higher_is_better):
            continue
        fp = _fp_at(keys, fps, i)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, rep_fps)
        best = max(range(len(sims)), key=sims.__getitem__)
        if sims[best] > similarity_threshold:
            groups[rep_keys[best]].append(k)
        # else: passed the gate yet resembles no accepted mode -- only reachable when `keys` holds
        # duplicates or the caller passed mismatched fps; dropping is safer than inventing a group.
    return groups


def count_modes(
    keys: Sequence[str],
    rewards: Optional[Sequence[float]] = None,
    *,
    higher_is_better: bool = True,
    reward_threshold: Optional[float] = None,
    similarity_threshold: float = _MODE_SIMILARITY_THRESHOLD,
    fps: Optional[Sequence] = None,
) -> int:
    """Number of paper-comparable modes (see :func:`mode_representatives`). RDKit-unavailable
    fallback: distinct-SMILES count."""
    if Chem is None:
        return len({k for k in keys if k})
    return len(
        mode_representatives(
            keys,
            rewards,
            higher_is_better=higher_is_better,
            reward_threshold=reward_threshold,
            similarity_threshold=similarity_threshold,
            fps=fps,
        )
    )


def mean_pairwise_similarity(
    keys: Sequence[str], *, fps: Optional[Sequence] = None
) -> Optional[float]:
    """Mean Tanimoto similarity over **all distinct pairs** in the set, on the :func:`ecfp` recipe.

    The set-level counterpart to :func:`count_modes`: modes ask "how many distinct families are
    here?", this asks "how alike is an average pair?". Molecules whose fingerprint is unavailable
    are dropped; returns ``None`` for fewer than two usable molecules (no pairs to average).

    Note this is a *pair* average, so it is unbiased under uniform subsampling of the set — an
    estimate from a random size-N subsample targets the same quantity as the full set — but it is
    also insensitive by construction: adding one tight cluster to a diverse set moves it very
    little. Read it alongside a clustering readout, never alone.
    """
    if Chem is None:
        return None
    usable = [f for f in (_fp_at(keys, fps, i) for i in range(len(keys))) if f is not None]
    if len(usable) < 2:
        return None
    total, n_pairs = 0.0, 0
    for i in range(len(usable) - 1):
        sims = DataStructs.BulkTanimotoSimilarity(usable[i], usable[i + 1 :])
        total += sum(sims)
        n_pairs += len(sims)
    return total / n_pairs


def count_butina_clusters(
    keys: Sequence[str],
    *,
    fps: Optional[Sequence] = None,
    similarity_threshold: float = _MODE_SIMILARITY_THRESHOLD,
) -> Optional[int]:
    """Taylor-Butina cluster count — the **reward-blind** counterpart to :func:`count_modes`.

    Why both exist. :func:`count_modes` is greedy sphere exclusion fed **best-reward-first**, which is
    what makes each mode's representative the molecule you would synthesise and what makes our numbers
    comparable to the RGFN/SCENT tables. But greedy exclusion is *order-dependent*, and that order is
    the reward — so it is the wrong instrument for asking whether diversity itself tracks reward.
    Butina picks cluster centres by **neighbourhood density** (most-neighbours-first), never looking at
    reward, so it answers the same "how many distinct families?" question with reward outside the
    procedure. ``docs/paper_planning/lsd-flow-publication-strategy.md`` §2.3 asks for exactly this.

    ``similarity_threshold`` is given as a *similarity* for symmetry with the rest of this module and
    converted to Butina's distance cutoff (``1 - similarity``); the two algorithms are not identical,
    so treat this as a second instrument rather than a drop-in replacement for the mode count.

    Cost: needs the **full lower-triangular distance list** up front, so memory is O(n²) — fine for a
    few thousand molecules (n=2,500 → 3.1M distances, ~0.5 s), impractical for tens of thousands.
    Returns ``None`` when RDKit/Butina is unavailable or fewer than two molecules are usable.
    """
    if Chem is None or Butina is None:
        return None
    usable = [f for f in (_fp_at(keys, fps, i) for i in range(len(keys))) if f is not None]
    if len(usable) < 2:
        return None
    dists: List[float] = []
    for i in range(1, len(usable)):
        dists.extend(1.0 - s for s in DataStructs.BulkTanimotoSimilarity(usable[i], usable[:i]))
    return len(
        Butina.ClusterData(
            dists, len(usable), 1.0 - similarity_threshold, isDistData=True, reordering=False
        )
    )


def unique_scaffolds(smiles: Sequence[str], *, scaffolds: Optional[Sequence] = None) -> int:
    """Number of distinct Bemis-Murcko scaffolds (secondary diversity metric, §11). ``scaffolds``:
    optional pre-computed :func:`murcko_scaffold` values aligned with ``smiles``."""
    valid = [s for s in smiles if s]
    if not valid or Chem is None:
        return len(set(valid))
    if scaffolds is not None:
        return len({s for s in scaffolds if s})
    return len({s for s in (murcko_scaffold(smi) for smi in smiles) if s})


def mode_counter(similarity_threshold: float = _MODE_SIMILARITY_THRESHOLD):
    """A ``keys -> n_modes`` closure for the production ``most_modes`` strategy. Structure-only
    (no reward gate): that strategy ranks hubs by the raw structural diversity of their children."""

    def _count(keys: Sequence[str]) -> int:
        return count_modes(keys, similarity_threshold=similarity_threshold)

    return _count
