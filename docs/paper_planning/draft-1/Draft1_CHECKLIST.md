# Draft 1 — every change, before → after

Working checklist. Full reasoning and sources are in `Draft1_CHANGES.md` (§ refs below).
Line numbers are the `draft-1/main.tex` snapshot and **will drift** from your live file —
the "before" column is written to be searchable.

**Legend** — 🔴 wrong/withdrawn · 🟠 placeholder · 🟡 imprecise or unscoped · 🔵 addition
· ⏭ **skippable** (nice-to-have; cutting it loses nothing a reviewer will miss)

**Deferred to draft 2:** `Draft2_CARRYOVER.md` — seven items, ordered by impact. Copy that
file into `draft-2/` when you make it; nothing else in this directory travels.

**Reading this top to bottom.** Rows are in *paper order*, so they follow your scroll.
Three things do not:

1. **Do the headline numbers first** (A1–A4, R2–R5). Several later rows quote them, and R5
   is a deletion whose replacement text lives in §9k.
2. **G4a/G4b are a pair** — the gate *rule* goes in §2, the four *values* in Appendix C.
   Decide once, edit twice.
3. **⏭ rows are safe to skip.** Six of them, ~150 words. Every 🔴 and 🟠 row is not
   optional: 🔴 is wrong and 🟠 is visible in the compiled PDF.

**Verified 2026-09-06:** every *Before* string in this file is findable verbatim in
`draft-1/main.tex`, and every line number points at it. If a search fails, tell me — that is
a bug in this document, not in your copy.

---

## 0. Prerequisites — the runs (no cluster time)

| # | What | Status | § |
|---|---|---|---|
| P1 | Figure 2: drop seed 42 | ✅ done (`8b56866`) — λ=0 **4.71× → 3.26×**, λ=1 **2.51×** unchanged | 0.1 |
| P2 | Table 1 → n=3 | ✅ done (`8b56866`) — **1.41×** mean, worst **1.26×** | 0.5 |
| P3 | 1-mode re-price, seeds 42 + 44 | ✅ done (`198e1ba`) — 36 mode-points each, all `Optimal`; brackets gone, values **exact** | 8.0 |
| P4 | "conservative end" sign error in `table1_external.py` + `harvest/external.py` | ✅ fixed — both strings now name **hi** as the conservative end | 8.0 |
| P5 | `table1_external.py` printed `K=` blank | ✅ fixed — reads `K=0`; `prebuild_k` now its own column | 8.0 |
| P6 | **Figure 1 re-gated** to the 5%-FPR rule | ✅ done — headline **2.48×**, 23 of 24, range 1.70–3.64; **the budget-monotonicity claim FAILED** (§9i) | 9i |

> **Headline confirmed: 1.41× mean, worst seed 1.26×, from exact values.** My earlier
> alternative of 1.35× / 1.19× is **withdrawn — it was wrong.** The 1-mode grid shows the
> bracket's upper ends were never purchasable inside the budget: on seed 42, 55 modes cost
> 99 reactions and 56 cost **103**; on seed 44, 65 cost 100 and 66 cost **101**. So 55 and 65
> are measurements at R=100, not the optimistic edge of a range, and crediting the competitor
> with 59 or 69 would credit it with modes it cannot buy. The sign *principle* still holds
> for any coarse grid — quoting `lo` understates the competitor and inflates our ratio — which
> is why both strings were fixed; it simply does not bind here.

---

## 1. Title & global

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| G1 | 🔴 L29 | `...Reaction-GFlowNets"}` | `...Reaction-GFlowNets}` | stray quote, renders in the PDF | 2.7 |
| G2 | 🟠 all | `\pending{...}` × 4 (L344, 475, 479, 513) | resolve all four, then delete the macro (L26) | L344, 475, 479, 513 | 2 |
| G3 | 🟠 all | `\checkval{...}` everywhere | delete the macro (L27) | it is blue text in the PDF | 2.10 |
| G4a | 🔵 **§2**, by the mode definition | *(nothing)* | **the rule, one sentence:** every gate is the 5%-FPR point on that oracle's own actives-vs-decoys set | State which exhibits read which bar — **Fig 1 + Appendix A1** at the rule; **Fig 2 + Table 1** at **7.0**, the paper-comparable bar, where the competitor's pool is threshold-non-binding. Two regimes, not three: A2 *sweeps* both and marks them, so it needs no separate justification | 8.1 |
| G4b | 🔵 **App C** | *(nothing)* | **the four values, with sources:** sEH **5.68**, DRD2 **0.345**, ClpP **−9.1**; 6TD3 stays at the provisional **−2.0** and is excluded from every ratio (a re-dock, not a re-gate) | **Quote neither 6.718 nor 7.97** | 8.1, 7.6 |

---

## 2. Abstract (L76)

| # | Before | After | Why | § |
|---|---|---|---|---|
| A1 | 🔴 `median 2.93×` | **`2.48×`** | was 2.73× before the Fig 1 re-gate | 0.2 |
| A2 | 🔴 `30 of 31 comparable conditions` | **`23 of 24`** | not 31 of 32, which is the all-four-generator count | 0.2 |
| A3 | 🔴 `2.27× on identical candidate sets` | **`2.51×`** | 2.27× is **withdrawn**. Quote λ=1 alone: it is SPARROW *with* its diversity term, i.e. its strongest setting, so it is the conservative end (λ=0 gives 3.26×). **Do not write "2.5–3.3×"** — those are two different competitor configurations, not a spread | 0.1 |
| A4 | 🟡 `three reaction-grounded generators, one fragment-based control` | keep the phrasing | but the ratio must now exclude the control | 0.2 |
| A5 | ✅ `one scoring oracle failed calibration` | correct | consider adding the exploitation finding | 3.3 |

---

## 3. §2 Library cost

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| M1 | 🟡 L274 | `τ = 0.5 to match ... \citep{shen2025cgflow}` | **keep | verified in the source**: *"Tanimoto similarity < 0.5 to any other mode"* | 7.2 |
| M2 | ⏭ 🟡 L274 | *(nothing)* | optional: our own Logs/032 gallery | at 0.75 the closest still-distinct pair is an oxazole↔isoxazole swap | 7.2 |
| M3 | 🟡 Threat model | `We therefore report the full sweep over $\tau$ rather than one cutoff` | **CORRECTED 2026-09-06.** My earlier "no sweep appears in the paper" was wrong | A2 sweeps **13 τ values × 9 reward bars**. But the sentence still overstates: every *reported* number is at one cutoff. Use **`we fix τ = 0.5 for every reported number and give the full sweep over both thresholds in Appendix~\ref{app:fair}`** | 3.5, 9b |
| M4 | 🔴 Threat model | the whole paragraph — see **§9j**. It asks the reader to hold **ten** things and is framed on the readout §2 calls *secondary* | rewrite to ~5 concepts | drop `numerator`/`denominator` and the `constrained optimum` label; keep its content as a clause. See §9j | 9j |
| M4b | ⏭ 🟡 Threat model | `144 families against 153` | scope it or cut it | it is the **diversity-off** library at cutoff 0.7; at 0.5 it is 39 vs 31. §9d lists it as safe to cut | 3.5 |
| M5 | 🟡 L276 | `Three run statuses, and why they matter.` + three labels | **drop to two.** Retitle *`Only runs that spent the budget are comparable.`* and define **budget-binding** / **pool-exhausted** only | Both are properties of our own metric and both are exercised in Fig 1; `solver-truncated` is a fact about a competitor's MILP, which has not been introduced as an arm yet. **This row owns the DELETION of the solver-truncated clause** — M6a/M6b only add its replacements elsewhere | 4.3, 9f, 9j |
| M6a | 🔵 **L362**, §4.2 | `…answers the same problem on the same pool in about a second.` | append: *"We exclude non-converged rows, since each bounds the competitor from below rather than measuring it."* | nothing else on that line changes; the deletion at L277 belongs to M5 | 4.3, 9f |
| M6b | 🔵 **L527**, App D | end of `\paragraph{Run status affects reported cost.}` | append: *"Non-convergence is detected by wall-clock rather than by the solver's status: CBC reports whichever incumbent it holds when the limit stops it as optimal, so a truncated run is indistinguishable from a converged one on status alone."* | this paragraph already says "budget-binding regime", so it is where that surviving label earns its second use | 4.3, 9f, 9j |
| M7 | ⏭ 🔵 L276 | *(nothing)* | if you want it, add `redundant` **inside M5's rewrite** — do not append a second paragraph | elegant, not load-bearing; it is your thesis stated as a run label | 0.5 |

---

## 4. §3 Hub-batching

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| H1 | 🔴 L297 | `rank intermediates by the resulting estimate and select the top 200 as hubs` | **`rank the pre-terminal states of the highest-reward candidates by that estimated flow and keep the top 200 as \emph{hubs}`** | The "pre-terminal states" / "that estimated flow" wording is an improvement — it matches step (ii)'s notation and the abstract. The load-bearing addition is **of the highest-reward candidates**: without it the sentence reads as ranking *all* pre-terminal states, which is not what was run. Exact K goes to App C (H4), not the body | 0.3, 9c |
| H4 | 🔵 App C | *(nothing)* | the K **values**, one clause, **armed against the cherry-pick reading**: *"Hub selection ranks the parents of the highest-reward sampled candidates; we expand the top 200 (64 in the head-to-head cell) | Expanding 200 rather than 64 there moves our count by −2 and +1 modes across the two seeds and leaves the baseline unchanged at 27, since it selects from the model's samples rather than from our enumeration."* **Do not discuss why K, or pre-select-K's break-even — omit** | 0.3, 9c |
| H2 | ✅ L297 | `sample 30,000 trajectories` | no change | verified: `sample/meta.json: n_trajectories = 30000` | 4.4 |
| H3 | ⏭ 🔵 L303 | *(nothing)* | optional: **85%** of hubs have exactly one estimate, so max/median/mean coincide | defuses "why max?" | 4.4 |

---

## 5. §4 Results

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| R1 | ✅ L318 | `docking scores for ... sEH, DRD2, ClpP, DDB1` | **already fixed by you.** DRD2 confirmed `sklearn.svm.SVC`, RBF, C=128 | not an MPNN, not a docking surrogate | 1.5 |
| R2 | 🔴 **L322** — one sentence, three numbers (**R2+R3+R4 merged**) | `Hub-batching delivers a median \checkval{$2.93\times$} more distinct high-reward molecules than best-candidate selection at a 100-reaction budget, over \checkval{30 of 31} comparable cells, range \checkval{1.67--4.80$\times$}` | `…a median \checkval{$2.48\times$} … over \checkval{23 of 24} comparable cells, range \checkval{1.70--3.64$\times$}` | post-re-gate, reaction-grounded scope. Edit once, not three times | 0.2, 9i |
| R3 | — | *(folded into R2)* | — | same sentence | 0.2 |
| R4 | — | *(folded into R2)* | — | same sentence | 0.2 |
| R5 | 🔴 **L323–324** | `The 100-reaction budget is a conservative choice rather than a tuned one: … advantage \emph{grows} with budget … second-lowest of four measured values.` | **delete the whole sentence**, replace per §9k | one sentence, three false claims — was split across R5/R14, now **merged**. Verbatim text in **§9k** | 0.2, 9i, 9k |
| R6 | 🔴 L354 | four separate quantities all printed `2.51×` | λ=0 **3.26×**, λ=1 **2.51×**; rxn/distinct **3.25×** and **2.5×**; candidates **1.21×** | four different quantities are all printed as `2.51×` | 2.2 |
| R7 | 🟠 L354 | `81.7 ± 0.6 ... while SPARROW ... 81.7 ± 0.6` | **delete the sentence.** Post-drop, λ=0 is 24/26 | tight, not collapsing | 2.1 |
| R8 | 🟡 L358 | `the solver returns 98 compounds amounting to 2 distinct molecules` | `at $\lambda = 0$ the solver returns \checkval{99} compounds amounting to \checkval{24} and \checkval{26} distinct molecules on the two seeds.` | the `98`/`2` pair is **seed 42**, which the figure no longer plots. The collapse is 4× rather than 49×, so the sentence keeps its point with less force | 1.8 |
| R9 | 🔴 L361 | `raising $\lambda$ from 0 to 1 takes the solver from \checkval{2} to \checkval{24} distinct molecules` | **folded into R10** — same sentence, edit once | the old `2`→`24` pairs seed 42's λ=0 with a `TimeLimit` λ=1 row the figure excludes | 1.8 |
| R10 | 🔴 **L361–362** | `Their knob is real but saturates: raising $\lambda$ from 0 to 1 takes the solver from \checkval{2} to \checkval{24} distinct molecules---so an earlier comparison of ours against a diversity-blind selector was genuinely an unfair fight---while $\lambda = 10$ buys nothing further and pushes solve time past \checkval{7{,}000\,s}.` | **delete `but saturates` and the whole λ=10 clause**; fix the numbers per R9 → `Their knob is real: raising $\lambda$ from 0 to 1 takes the solver from \checkval{24} to \checkval{33} distinct molecules on one seed and \checkval{26} to \checkval{32} on the other---so an earlier comparison of ours against a diversity-blind selector was genuinely an unfair fight.` | λ=10 exists on **one row — seed 42, `TimeLimit`** — compared against another capped row, on the seed Fig 2 no longer plots. The next sentence already makes the effort claim at n=36. `saturates` goes because λ=10 was its only support on this arm. **R9 and R10 are one sentence** | 1.8 |
| R11 | 🔴 L364 | `the competitor is \emph{cheaper per mode} than we are (\checkval{1.03} against \checkval{1.22})` | **`cheaper per candidate purchased (1.01 against 1.23)`** | our cost *per distinct molecule* is lower, 1.23 vs 3.08 | 1.2 |
| R12 | 🔴 **L363–368**, and the lead-in | `Two results cut against us and we report both.` … `And handing the solver candidates enumerated from our flow-ranked hubs rather than reward-ranked ones shows \emph{no detectable advantage} (\checkval{30.0} against \checkval{32.5}, direction flipping between seeds); at $n=3$ against a \checkval{29\%} coefficient of variation this is no evidence of a difference rather than evidence of no difference.` | **delete the second result, and change the lead-in to `One result cuts against us.`** Do **not** substitute the non-convergence finding — it cuts in our *favour*, so it cannot sit under this heading, and the paper already states it at **L362** | Result 13 is **withdrawn** (Logs/062, 2026-08-23). My earlier "replace with the non-convergence result" was wrong on both counts | 1.1 |
| R13 | 🟡 **end of L359** (not L365 — that is R11/R12's paragraph) | `This also predicts the fade we observe: the advantage is large at $R = 100$ and nearly gone by \checkval{$\sim$290} reactions (\checkval{$1.07\times$}), where our pool is close to exhausted and theirs is still climbing.` | **delete the sentence.** Verbatim reasoning in **§9k** | the row it rests on is `TimeLimit`-capped, and R5's deletion removes the claim it was contrasting with | 3.1, 9k |

---

## 6. §5 Severe testing

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| S1 | ⏭ 🟡 L417 | `The DDB1 oracle returns AUROC 0.688 ...` | keep the numbers, add the mechanism | add reward exploitation (clears the gate 78% vs 67% for real glues, 0/400 on an independent gate) and that a calibrated replacement exists (AUROC 0.917) | 3.3 |
| S2 | 🔴 L421 | `from ~2.34× to ~1.47×` | **`1.81× → 1.13×`** (Logs/033) | and 1.47× collides with Table 1's ratio | 1.4 |
| S3 | 🔴 L422 | `verified against SPARROW within 5%` | **`0.75%`** | the re-audit superseded Logs/049's figure | 1.4 |
| S4 | 🔴 L426 | `roughly 16 alternative recipes ... against 1.07 on ours` | **`13.1 (median 12, range 2–40) against 1`** | measured on Table 1's own pool | 1.6 |
| S5 | ✅ L426 | `−6.05 to −14.03 nats` | no change | verified against Logs/050 | 4.6 |
---

## 7. §6 Conclusion

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| C1 | 🟡 L461+ | `...post-hoc planning and cost-optimal selection` | add **`over its highest-scoring pool`** | 3 words, stops the footnote falsifying the sentence | 3.2 |
| C2 | 🔵 L461+ | *(nothing)* | add the future-work sentence | *"give cost-aware selectors pools that are diverse at construction time, so the selector is tested on its choices rather than on what it was handed"* | 3.2 |
---

## 8. Figures & tables

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| F1 | 🔴 L340 | Fig 2 caption: `30,000 ... while SPARROW was given 40,000 ... high-flow hubs` | full rewrite | Those are **seed 42 and 43 sizes of the same arm**. Must state: same pool; **two seeds** (41,817/64 hubs on s43, **39,969/73 on s44**); what differs (hard constraint vs soft penalty, λ=1/λ=0 + `\citep{fromer2025diversity}`); **reward > 7.0, τ < 0.5**; all solves `Optimal` | 1.7 |
| F2 | 🟠 L344 | `\pending{Something along the lines of...}` | the DRD2 control text | already written in `external/provenance.json` footnote [2] | 2.3 |
| F3 | 🟡 L538 | Table 1 header `Vs ours` | **`Ours ÷ theirs`** | `1.76x` on a competitor row reads backwards | 1.3 |
| F4 | 🔴 L542 | `Hub-batching (ours) & 82` | **`81`** | the table's own ratios prove it (81/46=1.76, 81/55=1.47) | 1.3 |
| F5 | 🔵 L536 | Table 1 caption: no seeds | **seeds 42/43/44**, headline **1.41× (worst 1.26×)**, exact | no bracket note needed | 0.5 |
| F6 | 🟠 L479 | Fig 3 (A1) caption `\pending{FILL IN CAPTION}` | write it | lead with: mean step yield **0.730–0.765** for both arms on every cell, so the difference is route *length*, not chemistry | 4.7 |
| F7 | 🔴 **L494** — the whole Fig 4 caption (**F7+F8+F9 merged**) | `Floor and ceiling for the hub ordering` … `an exhaustive strategy of counting the total number of children that pass both filters, and choosing to visit the hub with the highest number. Number reported is a percentage of the ` | **rewrite the caption once**: the ceiling arm is the **adaptive** greedy (*re-scores every remaining hub at every step, argmax marginal modes-per-reaction*) — the counting version is `greedy_oracle_static`, the regression control; finish the truncated final sentence; fix `$R \ge 7.0, \tau \le 0.5$` to `$R > 7.0$, $\tau < 0.5$`; add **one cell, one seed, a different checkpoint from Fig 1** | one caption, four problems | 1.9, 4.8 |
| F8 | — | *(folded into F7)* | — | same caption | 1.9 |
| F9 | — | *(folded into F7)* | — | same caption | 4.8 |
---

## 9. Appendices

### A — Yield

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| Y1 | 🟠 L475 | `\pending{Need to write this section more thoroughly}` | write the section | see §4.7 of `Draft1_CHANGES.md` for the content | 4.7 |
| Y2 | 🟡 | `extracted the average yield from each reaction class` | **per-template**, joined by **exact SMARTS**, from SCENT's own library | not per-class averages | 4.7 |
| Y3 | 🔵 | *(nothing)* | lead with: **step yield 0.730–0.765 for both arms on every cell** | chemistry identical, only length differs | 4.7 |
| Y4 | 🔵 | *(nothing)* | state that the direction **flips by target** | shorter on DRD2 (**2.66** vs 3.37 steps), *longer* on sEH (**2.00** vs 1.82) | 4.7 |
| Y6 | 🔵 | *(nothing)* | **A1 was rebuilt 2026-09-06 at the 5%-FPR rule** (sEH 5.68, DRD2 0.345) | The previous version was not reproducible — only **8–71%** of its molecules survive in the current enumerations. Numbers in Y3/Y4 are the rebuilt ones | — |
| Y5 | 🟡 | commented-out risk block (L400–408) | **do not restore** | every number rests on a Bernoulli assumption you dropped | 4.7 |

### B — Flow field

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| B1 | 🟡 | `unable to build the library at all` | **`unable to complete the library`** | it reaches 279 of 300 modes | 4.8 |
| B2 | 🔵 | *(nothing)* | add the scope | **one cell, one seed**, and a *different* SCENT checkpoint from Fig 1 — `PROVENANCE.md` is explicit that the two landing on similar numbers is coincidence | 4.8 |
| B3 | 🔵 | *(nothing)* | the `--pool all` defence, now sharper: **flow ranking is free** (0.199 s), so letting it select from all **20,874** observed intermediates costs nothing and gives **73 vs 72** | The reward pre-filter is a **legacy default** (bit-compat with the original recipe), not a compute saving at fixed hub count. Carry the caveats: one cell, one seed, and `all` aggregates differently | 0.3, 9c |
| B4 | ⏭ 🟡 | `\checkval{1.0--26.8$\times$} more molecule scorings` | give the true low end and say which population | wall-clock low end is **0.6×** (greedy was faster on one cell); say whether the figure shows all-14 (0.946) or budget-bound-only (0.952, n=12) | 4.8 |
### C — Training details

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| T1 | 🔴 **L509** — one sentence (**T1+T2 merged**) | `Of training and target , trained for \checkval{5{,}000} iterations with a reaction cap of 4 over a shared vocabulary of \checkval{418} building blocks and \checkval{112} reaction templates; \checkval{39} generator--target--seed cells.` | finish the opening clause **and** fix the count to **40** cell-seeds (32 calibrated, 24 reaction-grounded); say the design is **unbalanced** — sEH/DRD2 have 3 seeds, ClpP/6TD3 have 2 | unfinished edit plus a wrong count, same sentence | 2.8, 4.9 |
| T2 | — | *(folded into T1)* | — | same sentence | 4.9 |
| T3 | 🔴 L509 | `We set the reward gate at \checkval{5.0} rather than the \checkval{7.0} used in earlier work because` | **rewrite around the 5%-FPR rule** | the clause and its unverifiable denominator both go | 7.3 |
| T4 | 🟡 L511 | `DDB1` (throughout) | **`CDK12–DDB1 (PDB 6TD3)`** on first mention | DDB1 is the E3 adaptor, not the target; the metric is the DDB1 bonus | 1.5 |
| T5 | 🟠 L513 | `\pending{Discuss why DDB1 is shaded}` | *"6TD3 cells are hatched: its gate failed calibration (§5) and they are excluded from every ratio."* | resolves the `\pending` at L513 | 2.6 |
| T6 | 🔵 | *(nothing)* | state that 6TD3 stays at the provisional **−2.0** | **quote neither 6.718 nor 7.97** — they belong to the 6TD3-B oracle, which is a re-dock, not a re-gate | 7.6 |
| T7 | ✅ L511 | DRD2 97.5% saturation | no change | the DRD2-saturation self-criticism is correct and load-bearing | 4.9 |
### D — Metric validation

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| D1 | 🟡 L519 | `those apparent savings cost 9.8×, 6.8×, and 29× with both filters off` | **`cost 9.8× and 6.8× respectively, and 29× with both removed`** | three multipliers listed for two filters | 4.10 |
| D2 | 🔴 L523 | `0.0--3.1% above provably optimal and the baseline 4.8--5.0%` | **`0.75%`** on the headline config | keep the DRD2 K=20 caveat: 12.69%, traced entirely to pre-select-K; at K=0 it is exactly 0.00% | 1.4 |
| D3 | ✅ L519 | `The baseline is exactly indifferent to the reward gate` | no change | best-candidate is *exactly* reward-gate-invariant — a genuinely strong result | 4.10 |
### E — External pipeline

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| E1 | 🔴 **L554** — the whole asymmetry paragraph (**E1+E2+E3+E4 merged**) | `We report the asymmetries rather than equalizing them: the competitor's selector saw \checkval{2,000} candidates to our \checkval{26{,}069} and spent \checkval{2.25\,h} on retrosynthesis and route planning against our our zero.` | **rewrite the paragraph once** — see §4.11 of `Draft1_CHANGES.md` for the drop-in text: **500** routed candidates vs our **41,817** above-gate children; the **10,271 / 41,346 / 10,342** eval-phase oracle calls against a **10,048**-call training budget, per seed; "our zero" is true only of *route planning*; recompute the 4×/2× figures; fix the `our our` typo | the printed numbers belong to a **different experiment** (Logs/056, not Logs/075). One paragraph, not four edits | 4.11, 0.5 |
| E2 | — | *(folded into E1)* | — | same paragraph | 4.11 |
| E3 | — | *(folded into E1)* | — | same paragraph | 4.11 |
| E4 | — | *(folded into E1)* | — | same paragraph | 4.11 |
| E5 | 🔵 | *(nothing)* | **pool-construction caveat** | a diversity-enforced pool of **270** yields **100** distinct vs **46** from the 500. *Your call whether to name the numbers; see §3.2* | 3.2 |
| E6 | 🔵 | *(nothing)* | cite `genheden2020aizynthfinder` | it is in the bib and currently uncited | 4.2 |
---

## 9b. The two-knob heatmap — **not in the paper at all**

Full section text, caption and placement: **`Draft1_A2_SECTION.md`**.

| # | Loc | Before | After | Why | § |
|---|---|---|---|---|---|
| X1 | 🔵 App D | *(figure absent from `main.tex`)* | insert after "Both filters are load-bearing" | it is the **joint** version of that one-at-a-time ablation. Do **not** make a sixth appendix | A2 §1 |
| X2 | ✅ **figure code, not `main.tex`** — already fixed | `The advantage does not depend on where the two thresholds are set` | **already fixed** → *"Hub-batching is cheaper across a wide range of molecule requirements"* | But the new title states a *result*, not the figure's *job*, so the **caption must now carry**: *"Sweeping both thresholds over their full range leaves the conclusion unchanged, so the result is not an artifact of where the thresholds were set."* | A2 §2 |
| X6 | 🟡 caption | *(nothing)* | avoid *"cheaper at every setting"* | 3 cells are exact **ties**. **"Never more expensive"** covers all 452. And all three ties are at the strictest *diversity* settings; one is at bar **6.5**, so *"ties at very high reward"* is wrong | A2 §2 |
| X7 | 🔵 layout | *(nothing)* | `--layout grid` (2×2, ~8.4 in) fits a column-spanning slot; `--layout row` (1×4, ~14.1 in) needs a landscape plate | Identical content — **take `grid`** unless you have a spare page | A2 §3 |
| X3 | 🔵 App D | *(nothing)* | own the τ falloff as a **prediction of the method** | children of a shared parent are similar by construction, so demanding extreme dissimilarity forces the walk off the hub | A2 §2 |
| X4 | 🔵 caption | *(nothing)* | **0 of 452** comparable settings favour the baseline | 397 of 400 drawn are strictly cheaper, 3 exact ties | A2 §4 |
| X5 | 🟡 — | *(nothing)* | panels do **not** share one config (`child_policy` `reward` for RGFN, `free_frag`+K=20 for SCENT) | each is at its own operating point. Do not imply otherwise | A2 §4 |

---

## 9c. What the top-K knob actually is (so nobody re-opens it)

Established 2026-09-06. Record it here because it is the first thing a careful reviewer —
or the next agent — will ask.

**Picking hubs is free; expanding them is not.**

| step | cost | what it is |
|---|---|---|
| pick (flow estimate per hub) | **0.199 s** | arithmetic over `records.csv`; no model forward pass |
| expand (enumerate + score children) | **3,125 s** + 203 s | 170,724 children of 64 hubs |

~15,000×. So **the reward pre-filter is not what saves compute.** At a fixed hub count both
settings expand the same number of hubs; the pre-filter changes *which*, not *how many*.
What did save compute on the head-to-head cell is the smaller **hub count** — 64 expanded
rather than 200, hence 170,724 children rather than 702,574.

**We have already run K=all.** The ablation's `flow_top` arm ranks all **20,874** observed
intermediates and returns **73** modes against the reward-filtered arm's **72**. Flow selects
perfectly well unaided; the pre-filter is a legacy default that `pick_hubs.py` keeps because
"the defaults reproduce the original recipe byte-for-byte."

**Why not just switch everything to `all`?** Not the picking — the **re-enumeration**. A
different ranking selects different hubs, so every cell would have to be enumerated again:
~3,000 s × 40 cell-seeds ≈ **33 GPU-hours**. That is the answer to "can we just do it", and
it is not a tonight job.

**One open hypothesis, not a finding.** 85% of observed hubs have exactly one flow estimate,
so taking the top 200 by max-over-one-sample from 20,874 mostly-singleton hubs is partly
selecting on noise; the reward pre-filter implicitly favours hubs the sampler visited more,
whose estimates are better supported. Whether `flow_top`'s hubs are actually better-sampled
is **unmeasured** — do not assert it.

**Caveats on the 73-vs-72:** one cell, one seed, on the ablation checkpoint rather than the
matrix cells; and `all` aggregates a hub's score differently (max over *every* record naming
it, versus the best candidate landing on it), so it is not purely a change of eligible set.

---

## 9d. If you are over length — drop these first

Eighty-odd clauses is a different paper. These rows are **nice-to-have**; cutting every one
of them loses nothing a reviewer will miss, and buys ~150 words.

| drop | why it is safe to lose |
|---|---|
| M2 (Logs/032 gallery) | τ=0.5 is already supported by the verified `shen2025cgflow` citation |
| M4 (144 vs 153) | needs two clauses of scoping to be accurate; the filter ablation already carries the point |
| M7 (`redundant` status) | elegant, not load-bearing |
| H3 (85% of hubs single-estimate) | defuses a question only a careful reader asks |
| B3 (`--pool all`: 73 vs 72) | **promoted — I would now keep this.** It is the direct answer to "why not rank all hubs?", which §9c shows is the obvious question. Drop H3 instead |
| B4 (0.6× wall-clock low end) | a range endpoint, not a claim |
| R13 (the ~290 fade) | needs a "capped" caveat to be honest; cheaper to cut than to qualify |
| S1's exploitation detail | genuinely interesting, but the calibration failure alone carries §5 |

**Never drop:** the 🔴 rows — those are wrong, not merely absent. And of the 🔵 additions,
**G4** (which exhibit at which gate), **E3** (the sampling asymmetry) and **C1** (three-word
scoping) are disclosures, not enrichment: dropping them buys words with the reader's trust.

---

## 9e. Terminology: standardise on "pre-terminal intermediates" — **DRAFT 2**

Decided 2026-09-06; deferred. **For draft 1 do only the two rows marked DRAFT 1** — the
rest is consistency, not correctness, and draft 1's job is to be right and readable.

### ⚠️ Do not global-replace "intermediate"

Most uses are the **chemistry** sense — a shared *synthetic* intermediate — and that is
correct as written. A find-and-replace would turn the paper's central argument
("compounds sharing an intermediate are cheaper together than apart") into a claim about
GFlowNet states, which is false: a shared synthetic intermediate is not necessarily a
pre-terminal state of the policy.

| L | text | sense | action |
|---|---|---|---|---|
| 76 | `compounds sharing an intermediate are cheaper` | chemistry | **leave** |
| 200 | `if two target molecules share a late intermediate` | chemistry | **leave** |
| 207 | `pricing shared intermediates explicitly` (SPARROW's own term) | chemistry | **leave** |
| 310 | `any shared hubs or intermediates it happens to find` | chemistry / cost | **leave** |
| 358 | `an analogue hanging off an intermediate already paid for` | chemistry | **leave** |
| 76, abstract | `which cores are worth diversifying around`, `a high-value core` | med-chem framing | **leave** — "cores" is doing real work in the scaffold-decoration analogy |

### The formal object — the state flow is estimated at

| L | before | after | when |
|---|---|---|---|---|
| **297 (iii)** | `rank intermediates by the resulting estimate` | `rank the pre-terminal intermediates of the highest-reward candidates by that estimated flow` | **DRAFT 1** (this is H1 — an accuracy fix, not terminology) |
| **297 (ii)** | `at each pre-terminal state $h$` | `at each pre-terminal intermediate $h$` | **DRAFT 1** — free, and it is the sentence H1 sits next to |
| 217 | `rank intermediates by that quantity` | `rank pre-terminal intermediates by that quantity` | draft 2 — the intro's echo of step (iii); same inaccuracy, lower stakes |
| 307 | `the reward mass of an intermediate's terminal descendants` | `…of a pre-terminal intermediate's…` | draft 2 |
| 203 | `which intermediates are worth diversifying around` | fine as an anaphor — L202 says "pre-terminal intermediates" one clause earlier | draft 2, optional |
| 285 | `one intermediate with maximally diverse children` | reads fine; formal but unambiguous | draft 2, optional |
| 499, 500 | `high-flow intermediates` | "high-flow" already disambiguates | draft 2, optional |

Already correct: **L202** and **L215** both say `pre-terminal intermediates`. Standardising
means matching *them*, not inventing a term.

---

## 9f. §2's status paragraph — restructured for load, not length

The concern is how many separate things §2 asks the reader to hold, not its word count.
As drafted it carries **three**: an objection, three labels with three definitions, and a
comparison rule — and one of the three labels is about a solver the paper has not yet
introduced as an experimental arm.

**Split it by where each idea does work.**

| idea | goes | why there |
|---|---|---|
| budget-binding / pool-exhausted | **§2** | properties of *our* metric, and both are exercised in Figure 1, the next thing the reader sees |
| "we exclude non-converged solver rows" | **§4.2, L362** | the first solver arm in the paper, and L362 *already* discusses non-convergence — the clause attaches to a sentence that exists |
| CBC reports a truncated incumbent as optimal | **Appendix D** | an implementation caveat only an artifact auditor needs |

**§2, rewritten (~48 words, two concepts):**

> \paragraph{Only runs that spent the budget are comparable.}
> A fixed-budget comparison invites an obvious objection: an arm that runs out of
> candidates before spending its budget looks thrifty. We therefore record whether each
> run was \textbf{budget-binding} --- it spent the budget and could have spent more ---
> or \textbf{pool-exhausted}, and compare only budget-binding pairs
> (Appendix~\ref{app:fair}).

**§4.2, one clause onto L362:** *"…answers the same problem on the same pool in about a
second. We exclude non-converged rows, since each bounds the competitor from below rather
than measuring it."*

**Appendix D, one sentence:** *"Non-convergence is detected by wall-clock rather than by the
solver's status: CBC reports whichever incumbent it holds when the limit stops it as
optimal, so a truncated run is indistinguishable from a converged one on status alone."*

Nothing is lost — the paper still earns every exclusion, including seed 42's λ=1 row in
Figure 2 and the HB-Enum-SB bound in §4.2 — but no section carries a concept it does not
use. See §9c for the same treatment applied to top-K.

---

## 9g. Deferred to DRAFT 2 — the S3-GFN scope finding

Decided 2026-09-06. **Not in draft 1.** Draft 1 carries only C1's three-word scoping
("over its highest-scoring pool") so the conclusion cannot be falsified by the footnote
beneath the table.

**The framing to build in draft 2, and the one to avoid.**

❌ *"S3-GFN exploits the metric."* It does not. §2 defines the degenerate optimum as
*purchasing* catalogue compounds — modes without making anything. S3-GFN's molecules do not
exist until made, clear the reward gate, and are mutually Tanimoto-distant. That is
satisfying the metric, not gaming it, and calling it degenerate is reaching for a word
because the result is inconvenient.

✅ **The two methods solve different problems, and which wins is a property of the target.**
S3-GFN + a diversity-enforced pool finds molecules that sit one step from purchasable
material; hub-batching builds a core and decorates it. The first wins wherever high-reward
chemistry is one-step-reachable. Measured one-step-from-purchasable rates (Logs/075):

| generator | one-step rate |
|---|---|
| S3-GFN | **39–80%** |
| REINVENT | 9–34% |
| Saturn | **0–16%** |

So sEH under S3-GFN's chemistry is a one-step regime, and Saturn's is not. **The experiment
that settles it needs no new generation** — Saturn is already routed on seven cells.

**A supporting observation, not a defence.** One building block appears in 37 of their 100
molecules — amortisation of a shared parent, with a *purchased* parent. That generalises the
paper's mechanism rather than refuting it: the cheapest hub is sometimes one you can buy.

**Do not use** the starting-material count as an argument. The 1,065 / 1,198 figures are
network-wide, not per-selection, and do not support a claim either way without being
measured properly.

---

## 9h. The method diagram — DRAFT 1, user is drawing it

Replaces the commented-out spec at L239–266, which diagrams the **external** comparison
(route-less vs reaction-grounded) — the claim now demoted to secondary. What the primary
claim needs is a **method** diagram:

| track | steps |
|---|---|
| **best-candidate** | sample → sort by reward → take greedily under gate + diversity |
| **hub-batching** | sample → estimate flow at pre-terminal intermediates → rank → **enumerate every one-reaction child** → walk in rank order under gate + diversity |

The two annotations that carry the argument: (a) hub-batching adds an **enumeration** stage
that *creates* candidates rather than selecting among given ones — which is why its
candidate count is orders of magnitude larger; (b) both arms end in the *same* greedy
acceptance rule, so the figure makes visible that **only the ordering differs**.

Why it earns half a page: §3's Procedure is six numbered steps and is the densest prose in
the paper. A diagram makes hub-batching legible at a glance, which is draft 1's stated goal.

---

## 9i. The budget-monotonicity claim FAILED the re-gate — DRAFT 1

Verified independently 2026-09-06. Second time a *claim* rather than a number has broken
under checking, and the more serious of the two.

| R | median | n comparable | range |
|---|---|---|---|---|
| 50 | 2.23× | 24/24 | 1.77–3.75 |
| **100** | **2.48×** | 23/24 | 1.70–3.64 |
| 150 | 2.46× | 23/24 | 1.69–3.49 |
| 200 | 2.35× | 23/24 | 1.66–3.33 |

Exact: 2.231 / **2.485** / 2.460 / 2.353 — **rises to R=100 and falls.** At the old gates it
read 2.33 / 2.73 / 2.73 / 2.81, which is where "the advantage grows with budget" came from.
**With FragGFN included it still rises** (2.39 → 3.10), so the monotonicity was resting on
the arm already removed from the headline.

Per target: DRD2 rises strongly (2.23 → 3.21); sEH and ClpP are flat (~2.2–2.3 throughout).
The pooled median peaks while DRD2's rise leads, then falls once the two flat targets take
the median position.

### The part that needs a sentence, not just a deletion

**R=100 is now the maximum of the four.** Quoting it is the most favourable choice available,
and a reviewer who plots the other three will see it. Disclose it.

**But "we picked the peak" overstates it too**, and the spread says why:

| | spread |
|---|---|
| across budgets (medians) | 2.23–2.48× — **11%** |
| across cells at R=100 (IQR alone) | 2.14–2.93× — **37%** |
| R=100 vs R=150 | **1.0% apart** |

R=100 and R=150 are effectively tied; the ratio is **flat in budget** on this scope and the
apparent shape sits inside cell-to-cell variation. Do not replace one trend claim with
another.

### Suggested replacement for L322–323

> Hub-batching delivers between \checkval{2.23} and \checkval{2.49$\times$} more distinct
> molecules than best-candidate selection at every budget from \checkval{50} to
> \checkval{200} reactions, with no cell in which the baseline wins or ties at any budget.
> We report \checkval{100} because it is the constraint a chemist actually has and the
> budget at which the competitor's solver reliably converges
> (Appendix~\ref{app:fair}), not because it is the most favourable point.

The reasons for choosing 100 are unchanged and independent of the ratio — about one plate,
and the MILP converges at 100 but not at 200–300. That is what makes the disclosure
survivable: the choice predates the measurement.

---

## 9j. The class M5 belongs to: named machinery that does not earn its place

Raised 2026-09-06 — and it *is* a class, not three incidents. I had been fixing instances
(M5's statuses, §9c's K, §9f's solver caveat) without naming the pattern.

**The test.** A name earns its place if its *later* uses depend on it and each saves words.
Defined once and never reused, a name costs the reader a definition and buys nothing.

| term | uses | where | verdict |
|---|---|---|---|---|
| `count-once` | 3 | §2, §3(vi), App D | ✅ **earns it** — compresses a 20-word rule, twice |
| `hub-batching`, `mode`, `hub`, `best-candidate` | 6–44 | throughout | ✅ necessary |
| `pre-terminal intermediate` | 4 | abstract, intro, §3 | ✅ necessary (§9e) |
| `budget-binding` | 3 | but two are the *same sentence*; one remote (App D L527) | ⚠ borderline |
| `pool-exhausted` | 2 | §2 + Fig 1 caption | ⚠ borderline |
| `solver-truncated` | **1** | defined at L277, **never used again** | ❌ pure cost |
| `numerator` / `denominator` | 2 each | threat model only | ❌ see below |
| `constrained optimum` | 1 | threat model only | ❌ label a clause instead |

### M5, revised again — the deeper move is *zero* labels, not two

I said "drop to two." Your framing — *we exclude runs that ran out of candidates before
they ran out of budget* — is one idea with **no** labels, and it works everywhere the labels
are used:

| use | with labels | without |
|---|---|---|
| §2 | "label every run budget-binding / pool-exhausted…" | "we compare only cells where **both** arms spent the budget" |
| Fig 1 caption | "Pool-exhausted cells are hatched" | "cells where an arm ran out of candidates are hatched" |
| App D L527 | "Outside the budget-binding regime…" | "where an arm did not spend its budget…" |

**The honest tradeoff.** Zero labels is lighter to read; two labels *signal rigour* — a
reviewer seeing an explicit comparability condition reads it as care. That is rhetorical,
not informational, and it is your call which the venue rewards. **My recommendation: keep
`budget-binding` only** (it names the comparability condition, which is the paper's central
fairness move, and App D reuses it), drop `pool-exhausted` to plain words, drop
`solver-truncated` entirely per M6. One new term instead of three.

### The threat model is the worst case — and is framed on the wrong readout

⚠ **§2 declares the primary readout is "modes delivered at a fixed budget of 100 reactions."
The threat model then analyses "a ratio, reactions per mode (lower is better)" — the
direction §2 itself calls secondary.** So the numerator/denominator machinery maps onto an
axis the paper does not lead with, and the reader has to hold both orientations at once.

Ten things in one paragraph: the ratio; that it is gameable both ways; denominator = modes;
the τ sweep; a score-blind clustering cross-check (144 vs 153); numerator = reactions;
reward and depth distributions; the degenerate optimum; the reward filter; the constrained
optimum and that it is *not* degenerate.

**Rewrite on the primary readout (~105 words, 5 concepts):**

> \paragraph{Threat model.}
> The readout rewards anything that makes distinct high-reward molecules cheaper to reach,
> so it is gameable in two ways. Loosening what counts as *distinct* inflates the count
> without improving the library; we fix $\tau = 0.5$ for every reported number and give the
> full sweep over both thresholds in Appendix~\ref{app:fair}. Choosing structurally trivial
> molecules is cheaper per mode; we report reward and synthetic-depth distributions beside
> every ratio (Appendix~\ref{app:yield}). At the limit the second exploit is buying
> catalogue compounds, which costs no reactions at all, and the reward gate is what closes
> it --- distinct from concentrating on one intermediate with maximally diverse children,
> which is late-stage diversification and is what the method is for.

That last clause replaces the "constrained optimum" label while keeping its work: without
it a reader may think hub-batching *is* the degenerate case.

**Dropped from the paragraph:** numerator/denominator as named positions, the
constrained-optimum label, and the 144-vs-153 cross-check (M4b — it needs two clauses of
scoping to be accurate, and §9d already lists it as safe to cut).

**For draft 2:** the degenerate-optimum discussion is where the S3-GFN finding lands (§9g).
Its molecules are one step from the catalogue, which is the *second* exploit at one remove —
and the reward gate does not close that. Worth reopening this paragraph then.

---

## 9k. R5 and R13, verbatim

Both are deletions, and R13's is now a *consequence* of R5's.

### R5 — L323–324. One sentence, three problems.

**BEFORE** (verbatim):

> The 100-reaction budget is a conservative choice rather than a tuned one: Hub-batching's
> advantage \emph{grows} with budget (\checkval{2.53}, \checkval{2.93},
> \checkval{3.07}, \checkval{3.15}$\times$ at $R = 50, 100, 150, 200$), and we report the
> second-lowest of four measured values.

Three false claims in one sentence: **(a)** "conservative choice" — R=100 is now the
*maximum* of the four; **(b)** "advantage grows with budget" — it rises to R=100 and falls;
**(c)** "second-lowest of four" — it is the highest. This is why R5 and R14 merged: there is
one sentence to delete, not two edits.

**AFTER** (replaces it):

> Hub-batching delivers between \checkval{2.23} and \checkval{2.49$\times$} more distinct
> molecules at every budget from \checkval{50} to \checkval{200} reactions. We report
> \checkval{100} because it is the constraint a chemist actually has and the budget at which
> the competitor's solver reliably converges (Appendix~\ref{app:fair}), not because it is
> the most favourable point.

Why phrase it as a range rather than a corrected trend: the spread across budgets is 11%
(2.23–2.48) while the IQR across cells at R=100 alone is 37% (2.14–2.93), and R=100 and
R=150 sit **1.0% apart**. There is no trend to report — the ratio is flat in budget on this
scope, and claiming a peak would be as unsupported as claiming growth was.

The second sentence is the disclosure. It is survivable because both reasons for choosing
100 predate the measurement and are independent of the ratio.

### R13 — end of L359. Delete.

**BEFORE** (verbatim, the final sentence of the Mechanism paragraph):

> This also predicts the fade we observe: the advantage is large at $R = 100$ and nearly
> gone by \checkval{$\sim$290} reactions (\checkval{$1.07\times$}), where our pool is close
> to exhausted and theirs is still climbing.

**AFTER:** delete it. Nothing replaces it. Three reasons, in order of weight:

1. **The 1.07× row is `TimeLimit`-capped**, so it is a *lower bound on the competitor*. The
   true convergence could be flatter still, and a capped row cannot carry a ratio — the same
   rule that excludes seed 42 from Figure 2 and reduces HB-Enum-SB to a bound.
2. **R5's deletion removes its partner.** The sentence exists to reconcile "the advantage
   grows with budget" (§4.1) with a fade at large budgets. With the growth claim gone there
   is nothing to reconcile, and this dissolves §3.1's apparent contradiction rather than
   fixing it.
3. **It is measured against the solver, not against best-candidate** — a different
   comparison from the one §4.1 makes, two pages earlier, with no signposting.

If you want to keep the large-budget behaviour, App D is the place and it needs the "capped,
therefore a lower bound" caveat spelled out — which costs more words than the observation is
worth in a five-page paper. §9d already lists this as safe to cut.

---

## 10. Length

Body is **2,379 words**. Cuts ≈ **365**; the additions above cost ≈ **235**. Net ≈ **130 freed**,
plus ~0.5 in from Figure 2's new aspect ratio. Enough — but not enough to add anything else.

| Cut | From → to | Saves |
|---|---|---|
| §4.2 "Mechanism" paragraph | 343 → ~190 | ~150 |
| Intro "This work" | 235 → ~160 | ~75 |
| §2 Threat model (use the commented version at L233) | 152 → ~90 | ~60 |
| Intro opening | 204 → ~155 | ~50 |
| §2 Definitions | 152 → ~120 | ~30 |

**Do not cut:** run statuses · "no cell in which the baseline wins" · the "what hub-batching
is not" paragraph · §5.

---

## 11. Do NOT change these (verified correct)

`2.51×` λ=1 · **25 of 36** unconverged · greedy in **~1 s** ·
ceiling **94.6% / 9.2% / 10.9% / 12 of 14 / ~37%** · floor **1.53×**, within **9%**, flow rank
**546 of 20,874** vs **11,043** · filter ablation **13% / 91% / 12% / 39** · 6TD3 **0.688 / 31% /
2.1× / 82.9×** · DRD2 **0.949 / 0.961 / +0.012 / 18×** · ClpP **0.895** · sEH **0.76 / 0.68** ·
**30,000** trajectories · **418** blocks / **112** templates / **5,000** iters / cap 4 ·
τ=0.5 stricter than 0.7 · bibliography keys all resolve · anonymisation renders correctly.

**⚠️ Two items were REMOVED from this list on 2026-09-06** because the Figure 1 re-gate
falsified them. Do not restore them from an earlier copy of this file:

- ~~"budget monotonicity is real"~~ — **it is not**, on the reaction-grounded scope.
  2.23 / 2.48 / 2.46 / 2.35 rises to R=100 then falls. It still rises with FragGFN included,
  which means the claim was resting on the arm you removed from the headline. See R5.
- ~~"median 2.93×, range 1.67–4.80"~~ — those are the **old-gate, all-four-generator**
  values. The headline is now **2.48×, 23 of 24, range 1.70–3.64**.

Still verified after the re-gate: **zero** cells where the baseline wins or ties, at every
budget; comparable counts **24/24** at R=50 and **23/24** at 100/150/200 (better than the old
gates' 22/24 at R=200); the one non-comparable cell is `rxnflow_drd2` seed 44, where
*best-candidate* is pool-exhausted.

Full list with sources: `Draft1_CHANGES.md` §6.
