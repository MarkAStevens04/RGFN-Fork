# Draft 2 → Draft 3 — suggested edits

Review of the rendered `draft-2/main.pdf` (11 pages, no broken refs, no `\pending`, LaTeX
balances). **Nothing here is applied.** When we act on it, edits go in `draft-3/main.tex`.

Line numbers are `draft-2/main.tex`. **House style: US spelling** (decided 2026-09-07).

**Legend** — 🔴 wrong/inconsistent · 🟡 reads poorly · 🔵 addition · 🏗 structural

---

## Part 1 — The structural revision

Two changes that travel together, because the second pays for the first.

### 🏗 S1. Add the method diagram

**Why:** §3's Procedure is six numbered steps and is the densest prose in the paper. Every
current exhibit reports a *result*; none explains the *setup*. A diagram makes hub-batching
legible at a glance.

**Do not draw the spec commented out at L239–266** — that one diagrams the *external*
comparison (route-less vs reaction-grounded), which is the claim you demoted. Draw the two
selection rules side by side:

| track | steps |
|---|---|
| **best-candidate** | sample → sort by reward → take greedily under gate + diversity |
| **hub-batching** | sample → estimate flow at pre-terminal intermediates → rank → **enumerate every one-reaction child** → walk in rank order under gate + diversity |

Two annotations carry the argument:

1. hub-batching adds an **enumeration** stage that *creates* candidates rather than choosing
   among given ones — this is why its candidate count is orders of magnitude larger, and it
   is the honest cost;
2. **both tracks end in the identical greedy acceptance rule**, so the figure shows that
   only the *ordering* differs.

The second annotation is the one to protect. It is the paper's cleanest idea and a reader
currently has to assemble it from step (v) plus §4.2 prose.

**Placement:** §3, before or beside the Procedure list. Half a page.

### 🏗 S2. Move Figure 2 (`fig:selec_strat`) to the appendix

**Why:** you named best-candidate as the primary comparison. The body's two figures are
Figure 1 (primary) and Figure 2 (secondary), so the demoted comparison holds half the body's
figure space — and S1 needs half a page.

| | body figures |
|---|---|
| draft 2 | Fig 1 (primary) · Fig 2 (secondary) |
| **draft 3** | Fig 1 (primary) · **method diagram** · Fig 2 → appendix |

Same count, no net page cost.

**Mechanics:** move the float at **L345–351** into a new appendix section after
`\section{Flow field contributes most advantage}` (L500), titled something like
*"Selection versus enumeration"*. The two `\ref{fig:selec_strat}` sites (L344, L578) keep
working unchanged — LaTeX resolves them across the appendix boundary.

**Keep in the body:** the §4.2 prose, including the 2.51× sentence and the concession. Only
the float moves; the subsection stays where it is and points at the appendix.

### 🏗 S3. Where the page budget should go

§4.2 currently runs ~450 words across three paragraphs on SPARROW; §4.1 runs ~120 on the
primary claim. That is backwards. If the compile runs long, cut here rather than in the
appendices — the Mechanism paragraph can lose its middle third without losing the argument.

---

## Part 2 — Corrections

| # | Loc | Before | After | Why |
|---|---|---|---|---|
| C1 | 🔴 L78 | `\checkval{$2.51\times$} against a cost-optimal solver. A cost-optimal solver beats us on cost per candidate` | `\checkval{$2.51\times$} against a cost-optimal solver. That solver is cheaper per candidate purchased` | the abstract names the same system twice in adjacent sentences, once as beaten and once as beating us; a reader wonders if they are different |
| C2 | 🟡 L324–325 | `…(Figure~\ref{fig:main}). Hub-batching delivers between 2.23 and 2.49× more distinct molecules at every budget from 50 to 200 reactions.` | `…(Figure~\ref{fig:main}); the advantage holds between \checkval{$2.23$} and \checkval{$2.49\times$} at every budget from 50 to 200 reactions.` | two sentences state the headline twice — the second subsumes the first |
| C3 | 🔴 L325 | `2.49×` (raw unicode) | `\checkval{$2.49\times$}` | every other instance uses `$\times$`; the raw glyph typesets in a different font |
| C4 | 🔴 L236 | `(Section~\ref{sec:severe}, Appendix~\ref{app:flow})` | `Appendix~\ref{app:training}` (calibration) or `\ref{app:fair}` (optimality gap) | `app:flow` is the flow-field appendix; neither the oracle failure nor the optimality gap is there |
| C5 | 🟡 L286 | `Our default ($\tau=0.5$, reward at $5\%$ false-positive (FP) rate, details in Appendix~\ref{app:fair}) are reported throughout for legibility, but … (Appendix~\ref{app:fair})` | `Our defaults ($\tau=0.5$; reward at the $5\%$ false-positive rate) are reported throughout for legibility, but …` | `default … are` disagrees; nested parentheses; **Appendix D is referenced twice in one sentence** |
| C6 | 🟡 L286 | `We distinguish this from the \emph{constrained} optimum (one intermediate with maximally diverse children) which is the best achievable point conditional on synthesizing toward a target, which is not degenerate and is a description of late-stage diversification.` | `This is distinct from concentrating on one intermediate with maximally diverse children, which is late-stage diversification and is what the method is for.` | two stacked `which` clauses, and a named term (`constrained optimum`) used exactly once. The clause's *work* — stopping a reader concluding hub-batching is the degenerate case — is preserved |

### Typography

| # | fix | where |
|---|---|---|
| C7 | `1.70 - 3.64` → `1.70--3.64` | L324 |
| C8 | `4.8-5.0\%` → `4.8--5.0\%` | App D, audit paragraph |
| C9 | `0.3-0.7` → `0.3--0.7` | App D, "Default thresholds are strict" |
| C10 | `CDK12-DDB1` → `CDK12--DDB1` | Setup (L320), §5 (L414), App C (L539) — compound of two names, wants an en-dash |

### US spelling — 13 instances

| L | before | after |
|---|---|---|
| 78 | amortised | amortized |
| 202 | amortised | amortized |
| 418 | favour | favor |
| 424 | analyse | analyze |
| 461 | modelling | modeling |
| 517 | neighbourhood | neighborhood |
| 522 | neighbourhood | neighborhood |
| 528 | labelled | labeled |
| 554 | favours | favors |
| 573 | amortising | amortizing |
| 575 | amortisation | amortization |
| 594 | favour | favor |
| 641 | equalising | equalizing |

Already US and correct: `optimiz-` ×10, `labor` ×5, `favorable`, `favorite`. Note L418
(`in our own favour`) and L461 (`modelling`) sit next to US spellings, so they read as slips
rather than as a British house style.

---

## Part 3 — Carried from `draft-1/Draft2_CARRYOVER.md`, now in scope

### 🔵 P1. State the second claim rather than only using it as support

You named it and called it fluffier: *GFlowNets concentrate on high-reward-mass regions, so
their molecules are close in chemical space **and** good.* It is the mechanism behind the
paper and it is **not** unsupported — Appendix B already shows that ordering hubs by their
single best molecule, which never reads flow, still lands at median flow rank
**546 of 20,874**.

One sentence in the intro, with that as its evidence in Appendix B. It also pairs with S1:
the diagram shows *what* the method does, this says *why it works*.

### 🔵 P2. Terminology: standardize on "pre-terminal intermediates"

The abstract and intro already say `pre-terminal intermediates`; §3 steps (ii) and (iii) say
`pre-terminal states`. Two lines to align.

⚠️ **Do not global-replace "intermediate."** Five uses are the *chemistry* sense and are
correct — L202's *"if two target molecules share a late intermediate"* is the paper's central
argument, and replacing it would make it a false claim about GFlowNet states. The per-line
table is in `draft-1/Draft1_CHECKLIST.md` §9e.

Remaining after §3: L217 (`rank intermediates by that quantity` — the intro's echo of step
(iii), same imprecision at lower stakes) and L311.

---

## Part 4 — Deferred past draft 3

Still in `draft-1/Draft2_CARRYOVER.md`, unchanged: the **S3-GFN scope finding** (§3 there,
with the framing to avoid), the **6TD3-B re-dock** and its 6.718-vs-7.97 discrepancy, and the
**threat-model degenerate-optimum revisit** that depends on the S3-GFN work.

---

## What not to touch

§5 is the strongest section in the paper and reads cleanly. Appendix A, the Figure 4 caption,
and the conclusion's scoping plus future-work sentence all land. The headline numbers are
final against artifacts frozen at `3b15530`: **2.48×**, 23 of 24 comparable, range 1.70–3.64.
