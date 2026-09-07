# The heatmap — where it goes, and the text to paste

The two-knob surface (`figA2_two_knob_surface.pdf`) is **not in `main.tex` at all**. Below:
where to put it, drop-in LaTeX and caption, and the one sentence the caption must carry now
that the figure's title states a result rather than its job (§2).

---

## 1. Where it goes: fold into Appendix D, do not make a sixth appendix

Appendix D is *"Metric validation and cost model audit"*, and its first paragraph is
**"Both filters are load-bearing"** — the filter ablation, which removes the reward gate and
the diversity cutoff **one at a time**. The heatmap is the **joint** version of that exact
question: it sweeps both at once. It belongs directly after that paragraph.

Putting it in its own appendix would separate two halves of one argument and cost a section
heading for nothing. It also sits badly anywhere else: it is not yield (A), not the flow
field (B), not training (C), and not the external pipeline (E).

Suggested Appendix D order:

1. Both filters are load-bearing *(existing)*
2. **← the heatmap here**
3. The cost model is audited externally *(existing)*
4. Run status affects reported cost *(existing)*

---

## 2. The title has changed — and that moves work onto the caption

**Updated 2026-09-06.** The overclaiming title I flagged is already fixed:

| | |
|---|---|
| was | *"The advantage does not depend on where the two thresholds are set"* |
| **now** | *"Hub-batching is cheaper across a wide range of molecule requirements"* |
| subtitle | *"sEH · budget 100 reactions · ↗ stricter"* |
| colorbar | *"× cheaper per distinct molecule than best-candidate (1.0 = parity)"* |

**But the fix creates a gap the caption must close.** The old title, whatever its faults,
stated the figure's *job*: this is a robustness check. The new title states a *result*.
Without one sentence restoring the job, the panel reads as one more finding rather than as
the answer to an objection. Put this in the caption, near the front:

> Sweeping both thresholds over their full range leaves the conclusion unchanged, so the
> result is not an artifact of where the thresholds were set.

That objection — *"you picked the thresholds you win at"* — is the standard one against any
ratio metric, and it is asked directly at this kind of venue. The figure exists to answer
it, and nothing else in the paper does.

### What the surface actually shows, and why the asymmetry is worth keeping

Pooling all 452 comparable cells:

| reward bar | median | | diversity cutoff | median |
|---|---|---|---|---|
| 4.0 | 2.79× | | 0.90 | 2.73× |
| 5.68 | 2.74× | | 0.70 | 2.86× |
| 7.0 | 2.56× | | 0.50 | **2.49×** |
| 8.0 | 2.05× | | **0.30** | **1.19×** |

Near-flat in the reward bar; falls off a knee in the diversity cutoff. **Own that rather
than smooth it**, because it is what the method predicts: hub-batching earns its saving by
amortising one intermediate across many children, and children of a shared intermediate are
similar *by construction*, so demanding extreme dissimilarity forces the walk off the hub
and the amortisation stops paying. A surface that stayed flat at τ=0.3 would be evidence the
cost model was not measuring what we claim.

τ=0.5 also sits on the **falling edge, not the peak** — the advantage is larger at looser
cutoffs — which makes the operating point look chosen rather than convenient.

### Three things the caption must not overstate

1. **↗ means stricter, not better.** The advantage *narrows* toward that corner. Never
   describe up-and-right as desirable.
2. **"Pool limited" is one flag over two facts** — an arm ran short of library, *or* nothing
   cleared the threshold at all. The artifact records which; the figure merges them because
   the paper does not investigate *why* a cell is invalid, only that it is. Do not gloss it
   as a single mechanism.
3. **This sweeps thresholds, not seeds — and the one seed is 42.** sEH only, one cached
   enumeration per generator. **Name the seed**, because 42 is the most favourable one for
   RGFN: its ratio at the operating point is **2.28 on seed 42 against 1.885 on both 43 and
   44**. The threshold claim is unaffected (it is a statement about one surface, held
   consistently) but a reader must not infer the magnitudes are typical.
4. **Do not write that A2 and Figure 1 "agree".** They are computed from the same cells and
   the **mode counts are identical** (57/25, 73/33, 76/28, 80/20) — that cross-check is real
   and worth stating. But the *ratios* differ in the third digit because they are different
   quantities: A2's colour is a **cost** ratio (best-candidate's reactions-per-mode ÷ ours),
   Figure 1's is a **mode-count** ratio (our modes ÷ theirs). They coincide only when both
   arms spend the full budget — SCENT and FragGFN do (2.714, 4.000 identical); RGFN and
   RxnFlow spend 99 on one arm, giving 2.257 vs 2.280 and 2.190 vs 2.212. Safe phrasing:
   *"computed from the same cells"*, then name the two quantities. If the paper claims
   agreement, a reviewer recomputing either finds a 0.023 gap and reasonably suspects a bug.

### Where the three ties sit — verified

| generator | reward bar | diversity cutoff |
|---|---|---|
| RxnFlow | **6.5** | 0.30 |
| FragGFN | 8.0 | 0.30 |
| FragGFN | 8.0 | 0.35 |

All three are at the **two strictest diversity cutoffs**; **two of three** are also at the
highest reward bar. Writing *"the ties are at very high reward"* would be wrong — the
RxnFlow tie is at 6.5. The safe phrasing is *"all three ties fall at the strictest diversity
settings."*

Note this also means *"cheaper at every setting"* is slightly wrong: 397 are strictly
cheaper, 3 are exact ties, 0 are losses. **"Never more expensive"** is the phrase that
covers all 452.

---

## 3. Drop-in LaTeX

Paste after the "Both filters are load-bearing" paragraph in
`\section{Metric validation and cost model audit}`.

```latex
\paragraph{The result is not an artifact of where the two thresholds sit.}
Every headline number is quoted at one setting of two knobs a reader can reasonably
argue with: the reward bar, and the diversity cutoff. We therefore measure the whole
grid rather than defend a point --- \checkval{9} reward bars $\times$ \checkval{13}
cutoffs, all four generators, at the paper's own 100-reaction budget
(Figure~\ref{fig:twoknob}). Each cell is the ratio of the two arms' reactions per
distinct molecule, so $1.0$ is parity; because the budget is fixed this is monotone in
modes-at-100-reactions, the primary readout swept rather than a different metric.
\textbf{No setting favours the baseline}: of \checkval{452} comparable cells,
\checkval{0} fall below parity, and of the \checkval{400} drawn, \checkval{397} are
strictly cheaper with \checkval{3} exact ties --- all three at the strictest diversity
settings. Median advantage is \checkval{2.57--3.06$\times$} across the three
reaction-grounded generators (\checkval{4.40$\times$} for the fragment-based control,
which is not a peer; Appendix~\ref{app:training}).

Two classes of cell are deliberately left uncoloured, because counting them would
flatter us. \emph{Pool-limited} cells (hatched) are ones where an arm could not spend
the budget --- either it ran short of library, or nothing cleared the bar at all --- so
its cost is measured over a shorter run and is not comparable. Both causes concentrate
at the strict end, which is exactly where an unflagged surface would look most
impressive.

The surface is close to flat in the reward bar (median \checkval{2.79$\times$} at
\checkval{4.0} against \checkval{2.56$\times$} at \checkval{7.0}) and narrows as the
diversity requirement tightens (\checkval{2.86$\times$} at $\tau=0.7$,
\checkval{2.49$\times$} at our \checkval{0.5}, \checkval{1.19$\times$} at
\checkval{0.3}). That asymmetry is what the method predicts rather than a weakness of
it: hub-batching earns its saving by amortising one intermediate across many children,
and children of a shared intermediate are similar by construction, so demanding extreme
dissimilarity forces the walk off the hub and the amortisation stops paying. Our
operating point sits on the falling edge of that curve, not at its peak. This sweep
varies thresholds on one cached enumeration per generator; seed variance is
Figure~\ref{fig:selec_strat}'s question.
```

**Figure environment.** Two layouts render from one code path — `--layout grid` (2$\times$2,
~8.4 in, fits an ordinary column-spanning slot) and `--layout row` (1$\times$4, ~14.1 in,
needs a full-page landscape plate). Content is identical; choose on page budget. **Take
`grid` unless you have a spare landscape page.**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{Formatting_Instructions_For_NeurIPS_2026/figures/figA2_two_knob_surface.pdf}
\caption{\textbf{Hub-batching is cheaper across a wide range of molecule requirements.}
Sweeping both thresholds over their full range leaves the conclusion unchanged, so the
result is not an artifact of where the thresholds were set. Reactions per distinct
molecule, baseline $\div$ hub-batching, at a 100-reaction budget on sEH; $1.0$ is parity.
Rows are the reward bar, columns the diversity cutoff (Tanimoto), oriented so that
up-and-right is \emph{stricter} on both axes --- the advantage narrows in that direction.
Filled circle: \checkval{5.68}, the 5\%-FPR threshold. Hollow circle: \checkval{7.0},
where the head-to-head comparison is quoted; both at cutoff \checkval{0.5}. Hatched cells
are pool-limited --- an arm could not spend the budget --- and are excluded, as are cells
where no molecule clears the bar for either arm. Of \checkval{452} comparable settings,
\checkval{0} favour the baseline. This varies thresholds, not seeds: one enumeration per
generator. The fragment-based generator is a cost-model control, not a peer.}
\label{fig:twoknob}
\end{figure}
```

---

## 4. Numbers used above — all verified

| claim | value | source |
|---|---|---|
| grid | 10 bars × 13 cutoffs measured; **9 × 13 shown** (5.5 suppressed, 5.68 replaces it) | `artifacts/two_knob/surface.csv` |
| comparable cells | **452** | `cell_class` counts |
| pool-limited / no-modes | **29 / 39** | same |
| strictly cheaper | **397 of 400** shown, **3** ties, **0** losses | `figA2` stdout |
| cells below parity | **0 of 452** | computed |
| medians | RGFN **3.06×**, SCENT **2.71×**, RxnFlow **2.57×**, FragGFN **4.40×** | `figA2` stdout |
| ranges | 1.01–3.84 / 1.15–2.98 / 1.00–2.80 / 1.00–4.80 | same |
| bar gradient | 2.79× @4.0 → 2.56× @7.0 → 2.05× @8.0 | computed |
| cutoff gradient | 2.73× @0.9 → 2.49× @0.5 → **1.19× @0.3** | computed |

**One caveat to keep in mind:** the panels do **not** share one configuration —
`artifacts/two_knob/provenance.json` records `child_policy` `reward` for RGFN and
`free_frag` with `prebuild_k=20` for SCENT. Each panel uses its own headline config. That
is defensible, but if a reviewer asks "same settings?", the answer is "each generator at its
own operating point", so do not write a sentence implying otherwise.

---

## 5. Checklist rows

Add to `Draft1_CHECKLIST.md`:

| # | Loc | Before | After |
|---|---|---|---|
| X1 | 🔵 App D | *(figure absent from the paper)* | insert `\paragraph{The advantage is not an artifact...}` + figure after "Both filters are load-bearing" |
| X2 | 🟡 figure title | `The advantage does not depend on where the two thresholds are set` | overclaims — the cutoff gradient is 2.73× → 1.19×. Use *"cheaper at every threshold setting we measured"* |
| X3 | 🔵 App D | *(nothing)* | own the τ falloff as a **prediction of the method**, not a weakness |
