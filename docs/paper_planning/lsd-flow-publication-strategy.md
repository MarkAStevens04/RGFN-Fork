# LSD-Flow: Publication Strategy Notes

Working notes for the NeurIPS/ICLR/ICML submission. Organized by decision, with rationale — so the reasoning is recoverable later, not just the conclusion.

---

## 1. Theory: what to include and what to cut

### 1.1 Cut the standalone NP-hardness proof

**Decision:** Do not include a bare "hub selection under budget is NP-hard, by reduction from knapsack" theorem.

**Rationale:** Subset selection with costs and values is the canonical NP-hard setup. Every reviewer in this area knows the reduction; it takes two sentences and demonstrates nothing. Including it invites the Weaknesses line "the theoretical contribution is trivial," converting neutral page space into a negative. The widely-repeated claim that top ML venues require a theorem is a myth — purely empirical papers are accepted routinely, especially in ML4Science. There is no theorem quota; there is a rigor bar, and a weak theorem *lowers* perceived rigor relative to no theorem.

### 1.2 Hardness becomes worth stating when paired with a guarantee

Make hardness load-bearing by attaching an approximation result, so it justifies a design decision rather than decorating one.

> **Proposition.** Hub selection under a reaction budget is an instance of budgeted maximum coverage (hence NP-hard). The objective is monotone submodular, so the cost-benefit greedy of Khuller–Moss–Naor attains a (1 − 1/e)-approximation.

Now the structure is: problem is hard → we use greedy → greedy is provably near-optimal. This directly answers "why this heuristic?"

**Verification required before claiming this:**
- Confirm submodularity against the *actual implemented* objective. Intuition: a terminal reachable from two selected hubs counts once, so marginal gain diminishes — but confirm this holds for the exact objective as coded.
- Confirm the implemented ranking *is* cost-benefit greedy. The guarantee covers only the algorithm actually run. If there's a mismatch, either change the algorithm to match or state the gap explicitly in the text. Reviewers forgive a stated gap; they do not forgive a guarantee that silently doesn't apply.

### 1.3 The flow identity — label as Lemma, then extend

The identity `F(h) = F(x) · P_B(h | x) / P_F(x | h)` is detailed balance at edge h → x, rearranged. It is correct but is a one-line rearrangement of a known condition. **Call it a Lemma, not a Theorem** — labeling it a Theorem reads as overclaiming and costs credibility.

Extend it in one or more of these directions to make it a real contribution:

**(a) Multi-child consistency.** A hub has many terminal children; under a perfectly trained GFlowNet every child yields the same F(h). This gives two things: a variance-reduced estimator by aggregating across children, and a free diagnostic — the *spread* across children measures local training error. This converts a rearrangement into an estimator with a built-in confidence signal.

**(b) Robustness to imperfect training.** *Highest-value option.* If the log-space detailed-balance residual is bounded by ε on every edge, then per-child estimates of log F(h) agree within 2ε, and the induced hub ranking is exact whenever the true log-flow gap exceeds 2ε. This links a number already reported (training loss) to whether the hub ranking can be trusted, and preempts the obvious attack: *"your GFlowNet is approximate, so why believe the hubs?"*

**(c) Reward-mass interpretation.** State explicitly that state flow equals the reward mass of terminal descendants weighted by backward reachability. This reframes "sort hubs by flow" as "sort hubs by the reward mass they capture" — it justifies the entire selection rule. Cheapest theory-to-value conversion available.

### 1.4 Space budget

Main text: DB lemma (short), robustness corollary (short), budgeted-coverage proposition with greedy guarantee (short). **Total main-text theory footprint under half a page.** All full proofs to appendix. If a piece can't fit that budget while remaining non-obvious, it goes to the appendix or gets cut. Main-text real estate is better spent on the metric's external validity (§2).

---

## 2. The metric defense — biggest exposure, biggest opportunity

### 2.1 The problem

The paper proposes reactions-per-mode *and* wins on it. Reviewers are trained to be suspicious of exactly this shape. This is the single largest threat to acceptance, and the Pareto-front analysis is the answer to it — which also solves the "these plots feel out of place" problem. The Pareto section is not a side result; it is the metric's defense.

### 2.2 Threat model — state it explicitly

reactions/mode is a ratio, gameable from either side:

- **Numerator (fewer reactions):** genuinely desirable. Bulk lab cost is researcher time, so reducing reactions is the actual objective, not an exploit.
- **Denominator (more modes):** potentially adversarial. Inflating nominal mode count by loosening what counts as distinct would improve the ratio without improving chemistry.

Critically, mode count is *not* something to minimize. Reward generators are weak discriminators — cite the enrichment analysis (known binders vs. property-matched decoys) — so broad structural diversity is the mechanism by which true hits are found under an unreliable oracle. Suppressing modes would destroy the technique. This asymmetry needs stating plainly: one direction of "gaming" is the goal, the other is the risk.

### 2.3 Denominator defense: the similarity sweep

The Tanimoto-cutoff sweep is the right instrument. The claim to make is **rank invariance**: LSD-Flow's advantage is not an artifact of a particular diversity threshold, because it persists across the full range of mode definitions.

**Instrument robustness:**
- Prefer sphere-exclusion / Butina clustering over raw max-Tanimoto; raw max-similarity mode counting is known to be fragile in this subfield. Sweep the exclusion radius.
- **Normalize for library size.** If compared methods emit different numbers of molecules, the ratio is confounded before the cutoff question arises. Fix this first.
- Consider a second fingerprint (e.g. ECFP4 vs. a pharmacophoric or scaffold-based descriptor) to show conclusions aren't fingerprint-specific.

### 2.4 Numerator defense — currently missing

The sweep defends only the denominator. A method can also lower reactions/mode by producing structurally trivial molecules: shallower routes, cheaper reagents, lower reward. **Add a guard:** report reward distributions and synthetic-depth distributions alongside the ratio, showing reaction savings are not purchased with molecular quality. Without this, the gaming argument is visibly half-complete and a sharp reviewer will notice which half.

### 2.5 Degenerate optimum vs. constrained optimum — keep these separate

Two distinct things, previously conflated. The distinction matters and should be explicit in the text.

**The true degenerate optimum is depth-0.** Purchase N diverse catalog compounds: zero reactions, many modes, reactions/mode → 0. This is the genuine infimum and a real pathology — the metric bottoms out at a strategy that isn't performing the task, and it cannot discriminate among catalog-picking strategies since all score zero. Name this openly as the metric's failure mode.

**One hub with maximally diverse children is the *constrained* optimum** — the best achievable point conditional on actually synthesizing toward a target. This is late-stage diversification, and it is what LSD-Flow is named for. Do not call it degenerate; calling it the constrained optimum is both accurate and a stronger claim.

**Fix: condition the metric on a reward threshold.** The depth-0 exploit is the numerator-side gaming of §2.4 taken to its limit, so one intervention closes both. Catalog fragments do not clear a meaningful reward bar for a specific target — that is precisely why synthesis exists. Reporting reactions/mode conditional on R > τ eliminates the exploit without an ad-hoc rule. Sweeping τ yields a second Pareto axis structurally parallel to the similarity sweep (§2.3), making the two analyses read as one coherent framework rather than two bolted-on defenses.

**Construct validity argument — take this, it's free and it's the strongest paragraph available.** Walking the metric's low end recovers established medicinal chemistry at every depth:

| Depth | Recovered practice |
|---|---|
| 0 | Fragment / catalog screening |
| 1, diverse purchasable SMs | Parallel & combinatorial synthesis |
| Hub limit | Late-stage diversification |

A metric whose optimum recovers practices chemists adopted for reasons entirely independent of this paper is measuring something real. This is a construct-validity claim and it is far more persuasive than "the degenerate case is acceptable." State the LSD-Flow naming coherence explicitly — the method is named for the technique its constrained optimum recovers — rather than leaving it as a pun readers may miss.

### 2.6 External grounding

Anchor reactions/mode outside the paper's own framing so it isn't self-referential: map it to catalog reagent counts, plate/batch counts, and dollar cost from real vendor pricing. A metric that converts to money is much harder to dismiss as invented-to-win.

---

## 3. Own the scaffold-collapse effect

**Observation:** the advantage narrows as the similarity cutoff tightens.

**This is mechanistically predictable, not noise.** LSD-Flow selects hubs *because* many high-reward terminals share a pre-terminal intermediate. Children of a shared hub therefore share substructure by construction. As the cutoff tightens, the library's modes collapse faster than for a method sampling independently across scaffolds. This is the signature of the mechanism working as designed.

**Framing decision — this matters a lot:**

- ❌ "Our method wins everywhere, though margins narrow at strict cutoffs." → reads as hoping nobody asks.
- ✅ "Batching around shared intermediates trades structural diversity for reaction economy. We quantify that trade and report the regime in which it pays." → reads as understanding one's own failure mode.

Same plot, opposite reviewer reaction. This also converts the Pareto front from an odd appendage into the section demonstrating methodological self-awareness.

**Report the crossover explicitly.** If the advantage inverts past some cutoff, state the number: *"LSD-Flow dominates for mode definitions coarser than T = 0.6; beyond that, independent sampling is preferable."* Do not stop the sweep just before the crossover — a reviewer who suspects the sweep was truncated will assume the worst. An honest operating range is a stronger paper than an unblemished one.

---

## 4. "Post-hoc heuristic on existing models" — anticipated criticism

**The attack:** LSD-Flow extracts from already-trained RGFN/SCENT rather than proposing a new training objective. Some reviewer will frame this as a thin contribution.

**Defenses, in order of strength:**

1. **The theory from §1.2–1.3 is the primary answer.** Submodularity + greedy guarantee + robustness-to-ε make the extraction *principled* rather than incidental. This is where theory genuinely buys something — it's the reason to include any theory at all.
2. **Post-hoc is a feature, not a concession.** It applies to any trained GFlowNet without retraining, so it composes with future models. Demonstrate this by running on ≥2 distinct trained models (RGFN *and* SCENT) and showing the gains hold. Generality across backbones is the strongest empirical rebuttal.
3. **Cost framing.** No retraining means no additional compute — quantify it.

---

## 5. Other anticipated reviewer objections

**Hub definition arbitrariness.** Why pre-terminal only? Why not depth-2 or deeper? Have the ablation ready; without it this is a free criticism.

**Baseline strength.** SPARROW is the sharpest foil since it is already synthesis-cost-aware. S3-GFN is the marquee comparison. If LSD-Flow cannot beat SPARROW on an axis it does not own by definition, that is a substantive problem to address before submission, not in rebuttal.

**Oracle reliability.** The enrichment analysis (binders vs. property-matched decoys) does double duty: it justifies why diversity matters (§2.2) and preempts "your reward is meaningless." Make sure it is placed early enough to support both.

---

## 6. Checklist

- [ ] Verify submodularity of the *implemented* objective
- [ ] Verify implemented ranking is cost-benefit greedy (or state the gap)
- [ ] Relabel flow identity: Theorem → Lemma
- [ ] Write robustness-to-ε corollary (§1.3b) — highest-value theory item
- [ ] Add reward + synthetic-depth distributions as numerator guard
- [ ] Report reactions/mode conditional on reward threshold τ; sweep τ as second Pareto axis
- [ ] Write the construct-validity paragraph (depth ladder → established practice)
- [ ] Normalize mode counts for library size
- [ ] Switch/supplement mode definition to sphere-exclusion clustering
- [ ] Extend similarity sweep past the crossover point; report the number
- [ ] Rewrite Pareto section framing around the diversity/economy trade
- [ ] Run LSD-Flow on both RGFN and SCENT for generality claim
- [ ] Hub-depth ablation
- [ ] Vendor-pricing translation of reactions/mode
- [ ] Confirm theory footprint in main text < half a page
