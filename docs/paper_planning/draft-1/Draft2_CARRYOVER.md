# Carry into draft 2

Deferred deliberately during the draft-1 pass, 2026-09-06. **Copy this file into
`draft-2/` when you create it** — everything else in `draft-1/` is a review of a frozen
snapshot and does not travel.

Ordered by how much each changes the paper.

---

## 1. Move Figure 2 out of the body; re-weight the abstract

**Why:** you named **best-candidate** as the primary comparison and demoted SPARROW. But the
body's two figures are Figure 1 (best-candidate, primary) and Figure 2 (SPARROW, secondary),
so the demoted comparison holds half the body's figure real estate. The abstract also
coordinates them as equals — *"median X … **and** Y on identical candidate sets."*

**What to do:**

| | body figures |
|---|---|
| draft 1 | Fig 1 (primary) · Fig 2 (secondary) |
| **draft 2** | Fig 1 (primary) · **method diagram** · Fig 2 → appendix |

Same count, no net page cost, and both body exhibits then serve the primary claim: *here is
the method*, and *here is what it buys against best-candidate*. Figure 2 becomes the
appendix answer to "isn't the generator doing the work?" — a question a reviewer asks
*after* the main claim, not beside it.

**Keep in the body regardless:** the 2.51× *sentence* (the strongest one-line defence
against that objection) and Table 1's pointer. Only the float moves.

**Abstract:** subordinate the second clause rather than coordinating it —
*"…and, where only the selection rule differs, 2.51× against a cost-optimal solver"* — so a
reader knows which number is the paper's.

**Cost:** three edits — move the float, fix two `\ref`s, re-weight one clause.

---

## 2. State the second claim, instead of using it only as support

You named it and called it fluffier: *GFlowNets concentrate on high-reward-mass regions, so
their molecules are close in chemical space **and** good.* It is the **mechanism** behind
the paper, and it is not unsupported — Figure 4's floor shows that ordering hubs by their
single best molecule, which never reads flow, still lands at **median flow rank 546 of
20,874**. High-reward candidates concentrate at high-flow intermediates.

Draft 1 files this as support for the method. Draft 2 should state it as a claim: one
sentence in the intro, with the 546-of-20,874 as its evidence in Appendix B.

---

## 3. The S3-GFN scope finding — see `Draft1_CHECKLIST.md` §9g

The full framing is there, including **the wording to avoid** ("S3-GFN exploits the metric" —
it does not). Headline: the two methods solve different problems and which wins is a
property of the *target*. One-step-from-purchasable rates: S3-GFN **39–80%**, REINVENT
**9–34%**, Saturn **0–16%**.

**The experiment that settles it needs no new generation** — Saturn is already routed on
seven cells. That is the highest-value run available for draft 2.

Draft 1 carries only the three-word scoping (`over its highest-scoring pool`) plus the
pool-construction caveat.

---

## 4. Terminology: standardise on "pre-terminal intermediates" — see §9e

⚠️ **Do not global-replace "intermediate."** Five uses are the *chemistry* sense and are
correct; replacing them would turn the abstract's central argument ("compounds sharing an
intermediate are cheaper together than apart") into a false claim about GFlowNet states.
§9e has the per-line table. Draft 1 does only L297's two lines.

---

## 5. Revisit the threat model's degenerate optimum — see §9j

Once §3 lands, that paragraph is where it goes. S3-GFN's molecules are one step from the
catalogue, i.e. the second exploit at one remove — **and the reward gate does not close
that.** Draft 1's rewrite drops the numerator/denominator machinery; draft 2 should add what
the S3-GFN result teaches about where the metric's limits actually are.

---

## 6. Gates: 6TD3 is the last unstandardised target

sEH **5.68**, DRD2 **0.345**, ClpP **−9.1** are all at the 5%-FPR rule as of draft 1.
6TD3 stays at the provisional **−2.0** because its 5%-FPR gate (**6.718**) belongs to the
**6TD3-B CNN_VS oracle** — a *different reward signal*, so it is a **re-dock, not a
re-gate**, and costs GPU-hours.

Also unresolved: `matrix16/targets.py:175` carries 6.718 while a docstring in
`oracle_validation/calibrate_gates.py:24` still says 7.97. Reconcile before either appears
anywhere.

---

## 7. Two habits that caught real errors — keep them

Both came out of draft 1 and both found things nothing else did.

**Re-derive claims, not just numbers.** Twice a *sentence about the shape* of the data
survived a re-harvest that invalidated it: "the advantage grows with budget" (it peaks at
R=100 and falls) and "budget monotonicity is real". Numbers get re-harvested; derived
statements do not. **Before submitting, list every sentence that asserts a trend, an
ordering, or a comparison, and re-check each against the current artifact.**

**Before writing that two numbers agree, check they are the same quantity.** Three near
misses in draft 1, all the same shape: A2's grid median vs Figure 1's single point; A2's
**cost** ratio vs Figure 1's **mode-count** ratio (identical selections, 0.023 apart); and
`cost_kept_rxns/mode` vs `used_rxns/mode`, where the wrong one made the competitor look
cheaper than us. Similar magnitude is not the same measurement.

---

## Frozen state this review was written against

Artifacts frozen at **`3b15530`** (AC-MedChem-SDL/RGFN-LSD), confirmed 2026-09-06: working
tree clean, nothing queued. Headline **2.48×**, 23 of 24 comparable, range 1.70–3.64.
`draft-1/main.tex` is a snapshot and does not change.
