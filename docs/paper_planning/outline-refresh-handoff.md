# Outline refresh — what changed, and what it does to the structure

**Written:** 2026-08-25. **For:** the agent rewriting `docs/paper_planning/current-outline.md`.

**This is deliberately not an outline.** It is the *delta* between the outline as written and where
the experiments actually are, plus an argument about which deltas are structural (they move or create
sections) and which are cosmetic (they change a number or a sentence). Use it to rewrite the outline;
do not paste it into the outline.

**Read alongside, in this order:** `current-outline.md` (the thing being revised) → this file →
`lsd-flow-iclr-evidence-handoff.md` (**13 days stale — see §5 before trusting any number in it**) →
`lsd-flow-publication-strategy.md` (reviewer defence; still largely valid, one item now dead) →
`docs/RESEARCH_CONTEXT.md` §"How library cost is measured" and §"Key terminology" (**definitions —
read before touching any number**) → the individual `Logs/NNN_*.md` cited per item.

**Provenance rule for the rewrite.** Every number below carries its log entry. Cite the log, not this
file. Where a number has moved, the superseded value is named explicitly so it can be grepped out of
older drafts.

---

## 0. The venue decision is the biggest single reorganiser

Confirmed by the researcher 2026-08-25: **a tiered submission.**

| tier | venue | status |
|---|---|---|
| 1 (now) | **GEM @ NeurIPS workshop** | non-archival / non-exclusive — this is the immediate target |
| 2 | **ICLR** | the full paper |
| 3 | **Nature Computational Science** | only if ICLR rejects, rewritten using ICLR's feedback |

Two consequences that should drive the rewrite:

1. **The outline as written is the ICLR-tier paper.** It is close to right for that tier. What does
   not exist yet is the workshop-tier cut. The most useful artifact is **one outline with a per-item
   tier marker** (`[W]` workshop / `[I]` ICLR / `[N]` NCS-only), not two documents that will drift.
   Tier 1 and tier 2 share a spine; NCS is where the chemistry assets (construct-validity ladder,
   chemistry gallery, vendor pricing, the glue-oracle contribution) get promoted.

2. **The researcher's constraint on tier 1 is explicit: "our headline should be something that is
   already complete."** That single sentence decides §1.2 below and most of §6. The five-entrant
   competitor table is the strongest thing in the pipeline and it is *not* complete at corrected
   budget. It cannot be the workshop headline. It should be the ICLR headline.

⟦PENDING: GEM @ NeurIPS page limit and deadline — researcher. This sets how much of the workshop cut
survives, and I have not verified it.⟧

---

## 1. The five results that most reshape the outline

Ranked by how much of the outline they move.

### 1.1 The primary readout flipped axis — and the outline still states the old one

**What.** The benchmark's headline stopping condition is **modes delivered at a fixed budget of 100
reactions**, not reactions needed to reach a fixed mode target. Decided 2026-08-17, recorded in
`CLAUDE.md` as a standing rule; re-read across the whole matrix in **Logs/065**.

**Why we did it.** Four reasons, all worth a sentence in the paper: it was the benchmark plan's own
designated headline all along (the drift to a mode target began at Logs/056 and was never a decision);
it is the constraint a chemist actually has; ~100 reactions is about one plate; and the competitor's
MILP converges at R=100 where it does not at 200–300, so 100 is the budget most likely to yield a
certified optimum rather than a lower bound.

**What it found.** Median **2.93×** more distinct molecules than best-candidate at R=100, over 30 of
31 comparable cell-seeds (verified against `experiments/lsd_hubs/matrix16/results/reaction_axis/
reaction_axis_ratio.csv`, 2026-08-20). Range 1.67–4.80×; no cell where best-candidate wins or ties.
The advantage **grows with budget** — 2.53× / 2.93× / 3.07× / 3.15× at R = 50 / 100 / 150 / 200 — so
the headline budget is the second-most-conservative of the four measured, which is a point worth
making rather than hiding. Seed scatter roughly doubles on this axis (median CV 7.0% vs 3.2% on the
mode axis); three cells carry nearly all of it.

**What it does to the outline.**

- **`Experiments → Setup` currently says "Goal: 100 modes."** That is now the *secondary* readout.
  This is the single most load-bearing correction in this document.
- The **`Library-cost` section needs a new subsection** the outline has no slot for: **the three-way
  status label.** A fixed budget can end three ways, and only one is comparable — *budget-binding*
  (like-for-like), *pool-exhausted* (the arm ran out of qualifying candidates and left budget
  unspent), *solver-truncated* (SB arm only: CBC hit its time limit and still reports `Optimal`).
  This is not bookkeeping; it is the pre-emption of the sharpest available attack on a fixed-budget
  claim ("a method that runs out of things to buy will *look* thrifty"). It also carries a real
  measurement: outside the budget-binding regime the SB arm's reported `used_rxns` **inflates** —
  Saturn's pruned sEH cell holds 65 molecules priced at 247 reactions while `used_rxns` reads
  300→387 across R=300…1000, up to 140 reactions of slack. So quote `cost_kept_rxns` outside that
  regime; the gap between the two *is* the exhaustion detector.
- **A live inconsistency the draft must not inherit.** The internal matrix is on the primary axis
  (modes at R=100). The committed external headline figure
  (`campaign/results/paper_pipeline_headline/`) is still on the **secondary** axis (reactions for 100
  modes: 131 vs 264). `CLAUDE.md` forbids mixing the two in one table or figure — a co-agent already
  lost a result that way. **The fix is cheap and should happen before drafting:** the SB arm already
  has a native R=100 row in `s3gfn_seh_select_N500/select_frontier.csv` (24 modes at
  `used_rxns`=100), and the diversity-aware greedy arm is prefix-stable so `greedy_frontier.csv` can
  be re-read exactly at R=100 rather than interpolated. CPU-minutes, no re-run. Note the trap:
  re-slicing is exact for greedy and for our own hub-batching, but **not** for SPARROW, whose
  selection is jointly optimised — trimming an SB solve gives a feasible but suboptimal set. Re-solve
  SB, re-read the rest.

**Evidence:** Logs/065; `CLAUDE.md` §"THE BENCHMARK'S PRIMARY READOUT";
`matrix16/results/reaction_axis/{reaction_axis.png,reaction_axis_ratio.csv}`.

---

### 1.2 The competitor field went from one entrant to five — and a budget audit currently blocks the table

**What.** When the outline was drafted the external comparison was S3-GFN. It is now five entrants,
and they populate exactly the 2×2 the outline asks for in `Experiments → Hub-Batching beats other
models`:

| | **GFlowNet** | **not a GFlowNet** |
|---|---|---|
| **reaction-grounded** | RGFN, RxnFlow, SCENT *(ours)* | **SynFormer**, **TANGO** *(new)* |
| **not reaction-grounded** | FragGFN *(cost-model control)*, **S3-GFN** | **REINVENT**, **Saturn** *(new)* |

That top-right cell did not previously exist as evidence. It matters: it is the cell that stops the
paper reading as "GFlowNets vs everything else" and makes it read as "reaction-grounded generation vs
everything else", which is the actual claim.

**Why we did it.** The outline's own framing — the S3-GFN open debate — invites the reply "S3-GFN is
one model". Five entrants across four design classes turns a rebuttal of one paper into a statement
about a *class* of generator. Three of the additions also close specific holes: Saturn is the
sample-efficient extreme (and mode-collapses, which is a result); TANGO is a *constrained*-
synthesizability objective that yields routes natively, i.e. the closest thing to a peer; SynFormer
likewise ships routes by construction and so skips MultiAiZ.

**What it found so far, and why it cannot be tabled yet.** The **2026-08-20 baseline-config audit**
(`RESEARCH_CONTEXT.md` §"Baseline configs audited against their authors' defaults") found **three
oracle-call budget conventions running side by side**: ~10,000 (Saturn, SynFormer — their paper
defaults), ~128,000 (REINVENT), ~320,000 (S3-GFN, matched to *our* generators' 5,000 steps rather
than to its own paper). Every external generator's own default sits between 1,000 and 10,000.

Three things follow, and the paper should say all three:

- **The asymmetry is not in our favour.** The two entrants handed extra budget — REINVENT and S3-GFN
  — are precisely the two that perform best against us. Correcting should *improve* the headline,
  which is the reason to trust the correction rather than hesitate over it.
- **Quote no baseline-vs-baseline row until the correction lands.** That is a standing instruction in
  `RESEARCH_CONTEXT.md`, and the correction is in flight right now (`pilot_rnv42_naive` /
  `pilot_rnv42_pruned` are PENDING in the queue as of 2026-08-25).
- **One published number is already withdrawn because of it.** Logs/067's "pruning is worth 2.3–2.9×"
  and its "our advantage reads ~3.7× naive / ~1.6× pruned" both measured REINVENT at **12.4× its
  authors' budget**. At its own budget, pruning buys 1.0–1.36×. **Do not cite either figure.**

**A whole reporting convention the outline does not mention: the two pools.** Every competitor cell is
a 2×2 — *naive pool* (top 500 by reward, whatever diversity the generator's own machinery produced)
vs *pruned pool* (500 guaranteed-distinct molecules, our own sphere-exclusion rule applied to their
output), each read two ways (reactions/candidate, which the competitor wins by design, and
reactions/mode, which is what the claim is about). Reporting only the naive pool invites "you never
let it be diverse"; reporting only the pruned pool invites "you did the hard part for it". Running
both closes each objection with the other.

The re-measurement turned that design into a sharper finding than it started as: **the pruning lift is
a collapse detector, not a general property.** At authors' budgets it tracks mode collapse and nothing
else — Saturn 13.7× (sEH) / 5.2× (DRD2) / 4.0× (ClpP), TANGO 7.4 / 2.7 / 3.3, against REINVENT ~1.2
and S3-GFN 1.00 on sEH and ClpP. Saturn's top-500 sEH pool holds **18 distinct molecules**; its full
2,000 samples hold 338. So the collapse is concentrated in the highest-reward band — *the pressure
that makes Saturn sample-efficient is what makes its best molecules one family.* That is a genuinely
interesting statement about reward-maximising generators and it belongs in the paper, not a footnote.

**What it does to the outline.**

- `Introduction → Existing techniques` should name the five entrants and the 2×2 axes, replacing the
  current bare list (RGFN / SynFlowNet / RxnFlow / SCENT / CGFlow).
- `Experiments → Hub-Batching beats other models` is the section this builds. **Tier it:** the 2×2 is
  `[I]`, not `[W]`.
- `Experiments → Setup` needs the budget-parity paragraph and the two-pool paragraph. Both are
  fairness statements a reviewer will otherwise raise, and both are cheap to state.

**Evidence:** `RESEARCH_CONTEXT.md` §"Baseline configs audited…", §"The two pools and the two
numbers", §"TANGO: a reaction-aware entrant"; Logs/067 (**read its correction banner first**).
Trained candidates on disk: REINVENT and S3-GFN at 3 targets × 3 seeds; Saturn 8 of 9 (ClpP s43 resuming); TANGO 7 of 9 (DRD2 seeds 42/44 running);
SynFormer 1 of 9 and running.

---

### 1.3 The greedy-oracle ceiling kills the planned theorem and replaces it with a measurement

**What.** **Logs/060** built the step-by-step optimiser the outline's Proposition was meant to bound —
re-scoring every remaining hub at every step by marginal (new modes)/(marginal reactions) — and ran it
against our fixed flow ranking on all 14 cells.

**Why we did it.** Everything we had about the ordering was a *floor* (Logs/053: reverse it and the
method fails outright, randomise it and it costs 1.53×). "Better than nothing" is not the question a
reviewer asks. The question is the **ceiling**.

**What it found.**

- Our one-shot flow ordering recovers a **median 94.6%** (range 82–99%) of the distance from the
  standing baseline to the adaptive optimiser. In absolute terms flow costs a median **9.2%** more
  reactions, or delivers a median **10.9%** fewer modes at R=100 (the harsher reading, and the one
  the headline metric uses).
- The remainder is bought at a price the optimiser does not advertise: **1.0–26.8× more molecule
  scorings** (it cannot rank a hub it has not scored, so it pays for the whole pool up front) and up
  to **80×** more selection wall-clock (80.1 s vs 1.0 s on `scent_seh`).
- **Part of the remaining gap is not a saving at all.** The optimiser's libraries sit closer to the
  acceptance bar on 12 of 14 cells, and under a reward-blind Butina clustering at a strict 0.35 cutoff
  its libraries are the *most* structurally collapsed of the three arms. Forcing quality parity on the
  worst cell removes ~37% of its apparent advantage (20.2% → 11.5%).
- **The optimiser does not rediscover our hubs.** Median flow-rank 76 against a uniform null of 100;
  Jaccard overlap 0.00–0.19 on the easy cells. The pool contains many near-equivalent good choices —
  which is Logs/053's "flow finds the neighbourhood, not the rank" arriving from the opposite
  direction.

**What it does to the outline — this is a required deletion, not an optional one.** The outline's
`Methods → Formalism` reads: *"Proposition — only if submodularity is verified against campaign.py,
otherwise cut."* **It was verified, and it is false.** A "mode" is decided by greedy sphere exclusion
fed in acceptance order, so the objective is coverage over an **order-dependent set system**, which is
not the structure the budgeted-maximum-coverage result applies to. Two further mismatches compound it:
the shipped algorithm is a fixed ranking, not cost-benefit greedy, so the guarantee would not cover
the algorithm we run; and Khuller–Moss–Naor's (1−1/e) needs partial enumeration of starting triples,
which we do not do — the plain cost-benefit greedy is (1−1/√e).

So: **cut the Proposition, and replace it with Logs/060 as a measured optimality gap.** Two arguments
for why that is a stronger paper, both worth making explicit in the rewrite: a measured gap of ~5% of
the achievable gain is a *tighter* statement than a (1−1/e) worst-case bound; and the theory section
gets shorter, which the strategy notes already argue for on their own grounds.

The surviving theory is: **Lemma** (the detailed-balance rearrangement), **reward-mass reading**
(state flow = reward mass of terminal descendants weighted by backward reachability, so "sort hubs by
flow" is "sort hubs by captured reward mass"), and the **ε-robustness corollary** — which must be
written against what Logs/050 measured, i.e. the residual is *not* small in our models, so the honest
statement is the ranking-error bound, which is exactly why the claim is "neighbourhood, not rank".

`Experiments → Hub-Batching across reaction-GFNs` should now present **floor and ceiling in one
figure**: reverse-flow / random / flow / best-molecule-order / adaptive-greedy on the two cost axes.

**Evidence:** Logs/060; Logs/053; `experiments/lsd_hubs/greedy_oracle/results/`.

---

### 1.4 Selection vs enumeration — the contribution is narrower, better measured, and better defended

**What.** **Logs/062** switched on SPARROW's *own* diversity mechanism (`[fromer2025diversity]`, same
group as SPARROW) and swept its weight across four orders of magnitude, on both competitor arms.

**Why we did it.** Logs/059 compared against a diversity-blind selector. A reviewer who reads
`[fromer2025diversity]` asks immediately whether the advantage is an artifact of that. Using *their*
mechanism rather than a rule of our own means the comparison is against the published method.

**What it found.** The entry revised itself twice; read the numbers below, not the entry's own Answer
section.

- **Headline: 2.51×** distinct molecules at R=100 against BC-Enum-SB, on the two seeds where the
  competitor's solver converged. Our own arm is remarkably stable across seeds (81.7 ± 0.6, CV 0.7%);
  **the spread in the ratio is the competitor's, not ours** (BC-Enum-SB 23/32/33, CV 29%). Say it
  that way.
- **Selection alone is worth 2.27×** (n=3, sd 0.76) on *identical* candidates — same enumeration, same
  gate, same budget, only the chooser differs. This is the number that answers "your generator makes
  the molecules, why do we need your selection?", and it is the honest one.
- **The mechanism, and it is the best paragraph in the entry: a hard constraint versus a soft
  penalty.** SPARROW maximises reward under a reaction cap with diversity as a *penalty*. When the cap
  is tight, the cheapest available reward is another analogue hanging off an intermediate already paid
  for, and no affordable diversity weight outbids that — at λ=0 it returns **98 compounds that amount
  to 2 distinct ones**. We *refuse* any molecule within τ of one already taken, so we pay to move
  elsewhere from the first step. **A soft penalty loses to a hard constraint precisely when the budget
  is small — which is the regime a chemist with one plate is in.** This explains the shape of the
  result rather than restating it, and it is the reason the advantage is large at R=100 and nearly
  vanishes at ~290 reactions (1.07×), where our own pool is nearly exhausted and theirs is still
  climbing.
- **Their knob is load-bearing but saturates.** λ 0→1 takes BC-Enum-SB from 2 to 24 distinct
  molecules (12× on their own mechanism, so Logs/059 really was an unfair fight); λ=10 buys nothing
  further while pushing solve time past 7,000 s.
- **The selection-effort axis is a first-class result and the outline has no slot for it.** At a 2 h
  cap, **25 of 36** points did not converge. Raising to 12 h and relaxing the optimality tolerance did
  not fix it. On the corrected (uncapped) enumerations the HB-Enum-SB arm does not converge at all, so
  it is reportable **only as a bound — in the direction that flatters us**. Our greedy answers the
  same problem in ~1 s on the same pool. *"The optimizer cannot solve the selection problem at the
  scale our enumeration produces"* is a claim, not an apology.
- **The candidate-count confound is closed from both sides.** BC-Enum-SB saw **1.8–2.4× our oracle
  budget** and delivered under half the molecules — a measured point already past us on the x-axis,
  no extrapolation needed. And a BC-SB prefix sweep (k = 2k…30k sampling events) shows a 15× increase
  in sampling buys ~+35% distinct molecules on a near-flat, noisy curve.
- **A null result worth reporting.** HB-Enum-SB — SPARROW handed candidates enumerated from *our*
  flow-ranked hubs instead of reward-picked ones — shows **no detectable advantage** (HB 30.0 vs BC
  32.5 on converged seeds, direction flipping between seeds). Report it as "no evidence of a
  difference", not "no difference": it is n=3 against a 29% CV, and there is a live confound (the arms
  used 200 hubs / top-1000 vs 64 / top-100, so the net width differs as well as the ranking).

**What it does to the outline.** `Experiments → Hub-Batching across reaction-GFNs → BC-SB and
BC-Enum-SB` currently reads *"concede the loss on const-per-candidate / acknowledge (potential) win on
diversity."* Both halves need upgrading:

- The loss on cost-per-candidate is **real and now sharper than "concede"** — with a converged solve
  the competitor is cheaper per mode than we are (1.03 vs 1.22). The surviving claim is specifically
  cost per *distinct* molecule. State the reversal; it is cheaper than being caught.
- The win on diversity is no longer "potential". It is measured, replicated, and mechanistically
  explained.
- Add the **selection-effort axis** and the **hard-constraint-vs-soft-penalty** paragraph as their own
  outline items. My view is the mechanism paragraph is strong enough to sit in the metric section
  rather than the results — it is the argument for why a *constraint* formulation is the right one,
  which is also the outline's own "separation guarantee as a constraint rather than a measurement"
  ambition.

**Evidence:** Logs/062 (Results 6, 9–15 and both update banners — note Results 13–15 carry a
withdrawal banner for a *different* reason and the 2.51× is unaffected); Logs/059;
`RESEARCH_CONTEXT.md` §"How to report diversity whenever SPARROW selects".

---

### 1.5 Oracle validation now splits three ways, and one of our four scoring systems failed

**What.** **Logs/066** and **Logs/069** closed the last calibration gaps. The result is that the four
targets are no longer interchangeable and the paper must stop treating them as one axis.

| target | verdict | what it can carry |
|---|---|---|
| **ClpP** | calibrated, AUROC 0.895 (Logs/045) | full hit-count claims; also the cell where TANGO's authors dock against the same receptor (7UVU) |
| **DRD2** | **passes, and the memorisation control passes too** — AUROC 0.949 on held-out actives vs property-matched decoys, 0.961 on training-era molecules, 18× enrichment at the gate (Logs/069) | full hit-count claims — but see the saturation caveat below |
| **sEH** | weak but honestly reported — AUROC 0.76 vs random, **0.68** vs property-matched decoys (Logs/034) | a *scale*, not an activity claim. "modes ≥ 8.0" is not an activity claim |
| **6TD3** | **FAILS** — 0.688 AUROC and 31% of property-matched decoys pass, i.e. 2.1× enrichment where warhead-matched decoys gave 82.9× (Logs/069) | see below |

**Two things here are more interesting than a pass/fail table.**

- **DRD2 is saturated, not soft.** 97.5% of generated molecules clear the 0.5 gate, and none of 3,408
  unrelated drug-like molecules do. So the bar is defensible — but the DRD2 cells are effectively a
  **diversity-only** comparison, where the advantage comes entirely from picking varied molecules
  rather than high-scoring ones. sEH's bar removes a meaningful fraction. **The two targets are not
  measuring the same thing and must not be averaged into one number without saying so.** The outline
  currently has no place where that is said.
- **The 6TD3 failure contradicts our own earlier entries** (Logs/005, 007), and that is exactly why
  reporting it is worth more than it costs. The differential wins decisively against warhead-matched
  decoys and *loses to absolute Tier-2* against property-matched ones.

**Researcher's decision (2026-08-25): include the failure, and state that we are coming back to it —
a fix is ready and will ship with the large re-train.** So 6TD3 appears in severe testing as a
scoring system that did not hold up, with a named forward path, rather than being quietly dropped.

⟦PENDING: what the 6TD3 fix actually is — researcher. It is not written down anywhere in the repo that
I could find, and the drafting agent will need one or two sentences of it.⟧

**What it does to the outline.**

- `Experiments → Setup` needs a short "what each target is for" table. Right now the outline implies
  four equivalent systems.
- `Severe Testing` gains an item: *"we calibrated all four of our own scoring systems against
  property-matched negatives and one of them failed."* A benchmark that reports which of its own
  oracles does not hold up is much harder to dismiss than one reporting four successes.
- `Limitations` should carry the DRD2 memorisation-holdout leak (excluded by date in one of two source
  databases, not by structure) — because we volunteered the control, which is unusual, and the
  remaining hole is small enough to name.
- **NCS consequence:** the old handoff's §10 proposed promoting the glue-oracle work (the Tier2−Tier1
  neosubstrate differential) to a contribution for the Nature-tier version. Logs/069 substantially
  weakens that hook as written. Flag it for the NCS tier rather than deleting it, and pair it with
  the fix.

---

## 2. Second tier — real results that change wording, not structure

### 2.1 Yield-blindness and risk concentration (Logs/071)

**Researcher's decision: keep in Limitations, as the outline has it.** I would keep the placement and
**upgrade the wording**, because the outline's current phrasing ("cost accounting ignores probability
of reaction success / yield") describes an omission, and what we have is a measurement with a
mechanism and a proposed fix.

- The model picks below-average-yield reactions in **77–87%** of steps — but it is *reward-following,
  not careless*. On DRD2 the low-yield product scores 0.56 against the high-yield alternative's 0.35
  (amine preferred in 94% of matched pairs) because DRD2 ligands need a basic amine an amide cannot
  provide. On sEH, where the reward mildly prefers the amide, the model uses the low-yield reaction
  *less* than chance. Compounded, the cost is 8–11% of expected yield on sEH, 16–17% on DRD2.
- **The sharper half is about hub-batching specifically, and it is new.** Batching around shared
  prefixes means failures are no longer independent: one bad step in a shared prefix loses the whole
  batch. Expected molecule count is unchanged (the maths cancels), but the **spread rises 1.8–4.8×**,
  with 94–100% of the library behind a handful of prefixes, the largest single batch holding 63 of 96
  molecules, and the worst prefix failing with probability 0.30–0.44. Best-candidate shows no such
  concentration. We still win 2.7–5.3× on *expected* molecules delivered — so this is a risk caveat on
  a real win, not a reversal.

That last bullet is a quantified statement about what parallel-chemistry batching costs you, on the
same axis SPARROW built its method around. Even inside Limitations it should read as a finding.

### 2.2 The route-artifact audit (Logs/070) — the dataset is smaller than planned

Only **5 of 36** (generator × target × seed) cells hold a complete recipe. Of the rest, 23 are missing
an artifact that can only be regenerated by producing a *different* library, and 6 are missing one
that exists only during training. **Nothing is repairable by re-running a cheap stage.**

The reason recovery is not cheap is itself worth a line: a fresh draw from a frozen model does **not**
reproduce the molecules it drew before — RGFN sampling depends on `PYTHONHASHSEED`, which was never
recorded, so two runs of the same model with the same `--seed` agree on only **61%** of scaffolds.

**The published results survive** because they never used the recipes: count-once reads reaction
counts from separate accounting that was always complete. What the gap costs is dataset breadth and
the native-route arm's scope. If the rewrite gives the route dataset a contribution slot, it must say
"four cells, and here is the exclusion rule" rather than implying matrix-wide coverage.

### 2.3 The enumeration-cap correction (Logs/068)

A per-hub cap of 4,000 children was silently truncating SCENT (and only SCENT — it promotes
intermediates into a growing vocabulary, so one scaffold can reach far more children; the other three
generators never exceed ~3,435). Lifting it gave the six affected cells 8–61% more molecules and moved
the headline numbers **+0.1% to +3.3%, every one in our favour**. So published results were right and
slightly conservative.

The mechanism is worth one sentence because it predicts the size: the correction only bites if a
truncated scaffold is one the walk actually reaches, and at the headline bar the walk needs only 6–23
scaffolds. Raise the bar, the walk reaches further, and the correction grows with it (+4.7% at the
strictest bar, +10.6% on the docking cell that triggered the investigation). **Consequence for the
rewrite:** the "cheaper in 380 of 380 cells" claim from Logs/055 is intact but its surfaces were
regenerated on corrected inputs — re-read before quoting, and the count is now against the corrected
enumerations.

### 2.4 The competitor now has error bars (Logs/061)

The external head-to-head was one training run of the baseline against three of ours. It is now three
each: competitor **264 ± 25** reactions for the 100-mode library against our 131 (seed 42, our
*worst* seed) or 124.3 ± 6.1 (3-seed mean). **Seed 42 was the competitor's luckiest run of the
three** (235 vs 278/280), so the number the draft was quoting favoured them.

**The conservative comparison moves 1.79× → 2.02×.** Replication moved the claim in our favour and put
a band on it. Anywhere the draft says 1.79×, it should say 2.02×.

---

## 3. State of the evidence — what a claim can rest on today

Verified against disk and the live queue, 2026-08-25.

**Complete and committed (safe to build a workshop headline on):**

| evidence | scope | where |
|---|---|---|
| Internal matrix at R=100 | 16/16 cells have ≥1 seed; **39 cell-seeds**; 8 cells n=3, 7 at n=2, `rgfn_6td3` at n=1. Median **2.93×** over 30/31 comparable at R=100 | `matrix16/results*/` , `results/reaction_axis/` |
| External head-to-head, sEH | 3 seeds both sides; **2.02×** conservative, 3.21× as-run; both pricing regimes | `campaign/results/paper_pipeline_headline/` |
| Diversity-aware SPARROW | **2.51×** vs BC-Enum-SB; **2.27×** selection-only; effort axis; confound closed | Logs/062; `paper_sample_efficiency/`, `paper_pair_gallery/` |
| Ordering floor **and** ceiling | 6-arm ablation + 14-cell adaptive-greedy ceiling | Logs/053, Logs/060 |
| Filters load-bearing | dropping either flatters us by 12–13% and costs 9.8× / 6.8× / 29× per keeper | Logs/054 |
| Cost model audited | count-once vs independent SPARROW MILP: hub-batching 0.0–3.1% off optimal, baseline 4.8–5.0% | Logs/042, Logs/049 |
| Two-knob surfaces | 4 generators × 9 bars × 13 cutoffs, no crossover | Logs/055, re-run on corrected enums (Logs/068) |
| Real GPU docking | 2.02M docked children, 412.5 GPU-h, 6 cells + 42 gate points | Logs/058 |
| Oracle calibration | all four systems, incl. one failure | Logs/034, 045, 066, 069 |

**In flight (do not build the workshop headline on these):** REINVENT budget-corrected route
discovery (queued now); SynFormer cells (8 of 9 missing — the last commit diagnosed the cause as
worker OOM, not slowness); TANGO DRD2 seeds 42/44 and the sEH arm-2; `rgfn_6td3` seed 42 enumeration;
`rgfn_clpp` seed-43 reaction re-enumeration; six seed-44 docking cells on Trillium (explicitly *not*
on anyone's critical path — Trillium is delivering ~9–19 GPU-h/day).

**Will not exist without new compute, and should be declared in Limitations:** S3-GFN on any target
but sEH (needs retraining, not re-planning); error bars on the competitor side of the docking cells;
a hub-depth ablation; the native-route arm on the 23 route-less cell-seeds; multi-recipe route choice
on our side (SPARROW gets ~16 alternative recipes per molecule on the competitor's side and **1.07**
on ours — this asymmetry cuts *against* us, so say so).

---

## 4. Line-by-line corrections keyed to the current outline

| outline location | issue | fix |
|---|---|---|
| `Experiments → Setup`, "Goal: 100 modes" | **Wrong axis.** 100 modes is the secondary readout | "Goal: 100 reactions — how many distinct molecules?" See §1.1 |
| `Experiments → Setup`, "why reward is 5.0 not 7.0" | Correct, and the reason is stronger than "calibrated" | At 7.0 an arm falls short of budget in 3 of 8 cells on *every* seed, so those ratios were computed over starved libraries and **understated us**; 5.0 lifts RGFN-sEH 1.34× → 2.32× and makes all four cells full-budget. Keep 7.0 reported as the paper-comparable variant (Logs/064) |
| `Experiments → Setup`, "MultiAiZ requires retraining for SMALL library, so didn't do it lol" | The real reason is better and is measured | Logs/048 Exp A: a fragments-only transfer **cannot** homogenize a template-based synthesizability signal, so "give the baseline the same blocks" is not a well-posed fairness test for S3-GFN. It *is* well-posed for AiZynth's broad USPTO templates, which is why Logs/047 worked. Also: TANGO's `enforced_building_blocks_file` gives the 418-block arm a natural home using the authors' own mechanism |
| `Methods → Formalism → Proposition` | **Dead.** Objective is not submodular; shipped algorithm is not cost-benefit greedy | Cut it; replace with Logs/060's measured ceiling. See §1.3 |
| `Methods → Formalism → E-robustness corollary` | Still the highest-value theory item, still unwritten | Must be written against Logs/050's measurement — the residual is *not* small, so the honest form is the ranking-error bound |
| `Methods → Models we're using` | Lists only S3-GFN + FragGFN as non-reaction-aware | Five entrants, 2×2. See §1.2 |
| `Library-cost → How to game the metric → Denominator` | Butina/sphere-exclusion and second fingerprint listed as to-do | Partly done: Logs/054's score-blind clustering agrees and is slightly harsher (144 families vs our 153); Logs/060 uses Butina at 0.5 and 0.35 as a severe test. Second fingerprint still open |
| `Library-cost` | No slot for the budget status labels | Add. See §1.1 |
| `Experiments → Hub-Batching beats other models` | 2×2 has no evidence at matched budget yet | Tier `[I]`. See §1.2 |
| `Experiments → …other selection strategies → BC-SB / BC-Enum-SB` | "concede the loss" / "acknowledge (potential) win" both understate | See §1.4 |
| `Experiments → Failure modes`, "4 diff reward generators, docked 2m children" | Correct — 2,022,682 docked children, 412.5 GPU-h | Keep, cite Logs/058 |
| `Severe Testing → conservation law` | Numbers correct (−6.05 nats RxnFlow, −14.03 SCENT) | Keep. Note the violation is **larger in the model whose backward policy is trained and exactly recovered**, so it is a property of trained models, not one model's approximations |
| `Severe Testing` | Missing the oracle-calibration failure and the two withdrawn results | Add 6TD3 (§1.5) and the Logs/067 budget-error withdrawal (§1.2) |
| `Limitations → error bars` | Now partly closed | Ours n=3 on the surrogate half and on the sEH external arm; competitor n=3 on sEH; docking cells mostly n=2, one at n=1 |
| `Limitations → yield` | Understates what we measured | See §2.1 |
| Everywhere | Venue tiering absent | Mark every item `[W]` / `[I]` / `[N]`. See §0 |

---

## 5. What in the OLD evidence handoff is now wrong

`lsd-flow-iclr-evidence-handoff.md` is dated 2026-08-12 and the drafting agent **will** read it. Its
structure is still good; these specific numbers and statuses have moved.

| in the old handoff | now |
|---|---|
| "1.79× against the competitor's strongest configuration. **Lead with 1.79×**" | **2.02×** — competitor replicated to n=3, and seed 42 was its best run (Logs/061) |
| "3.13× as-run" | **3.21×** on the current committed CSV (the SB arm re-read at 420 rather than 411) |
| Deliverable framed as "100 modes" throughout; §3 says every table predates the axis decision | The axis decision is now executed matrix-wide. Use Logs/065, not a re-slice |
| §5.3 BC-SB / BC-Enum-SB table (14 and 47.7 distinct) | Diversity-blind, λ=0 only. Logs/062 says that was not a fair fight. Use the λ-swept numbers |
| §11 Proposition / submodularity / (1−1/e) | **Dead.** See §1.3 |
| §4 "Missing: `rgfn_6td3`, and `rgfn_clpp`" | Both exist; `rgfn_clpp` n=2, `rgfn_6td3` n=1. Matrix is 16/16 cells, 39 cell-seeds |
| §6.5 "`scent_clpp` re-enumeration in flight" | Landed; every cell uncapped. Effect +0.1% to +3.3%, all in our favour (Logs/068) |
| F1 "publication-ready, needs the 3-seed band" | Band is in. But F1 is on the **secondary axis** — see §1.1 |
| §8 gap 7 "run the competitor on a docking target" | Superseded in scope by the five-entrant campaign, which covers ClpP |
| §10 NCS hook: promote the glue oracle | Weakened by Logs/069. Keep for the NCS tier, pair it with the fix |
| §12 do-not-cite list | Still valid; **add**: 3.4× (BC-Enum-SB, capped row), 1.79×, "we win on both axes at once", Logs/062 Results 13–15, Logs/067's 2.3–2.9× pruning lift and its 3.7×/1.6× framing, and any baseline-vs-baseline row before the budget correction |

---

## 6. My recommendation for the workshop cut  *(opinion — the researcher has not signed off on this)*

Given "the headline should be something already complete", I would build the GEM version on the two
finished pillars and hold the five-entrant table for ICLR.

**Claim for tier 1, one sentence:** *a reaction-grounded generator's own trained flow field tells you
which intermediates to batch around, and reading it post-hoc buys ~3× more distinct high-reward
molecules per reaction than either picking the best candidates or handing the same molecules to a
cost-optimal MILP.*

**Keep `[W]`:** the metric definition + the three budget status labels; the internal matrix at R=100
(2.93×, 4 generators × 3–4 targets); the sEH external head-to-head re-read on the primary axis; the
BC-Enum-SB comparison with the hard-constraint-vs-soft-penalty mechanism and the selection-effort
axis; the filter ablation; the ordering floor+ceiling as one figure; the cost-model audit against
SPARROW's MILP; one severe test.

**Hold `[I]`:** the five-entrant 2×2 and the two-pool design; the two-knob surfaces; the conservation
violation and the `U(h)` autopsy; the construct-validity ladder; the compute-frontier attribution.

**Hold `[N]`:** the glue-oracle contribution, the chemistry gallery, vendor pricing.

**Two things I would do before drafting either tier**, both cheap:

1. **Re-read the external head-to-head at R=100** so the paper never mixes axes (§1.1). CPU-minutes.
2. **Pick one severe test to lead with.** My preference is the cost-model self-correction (Logs/029 →
   Logs/033: we found a fragment double-count in our own baseline, fixed it, and our claim dropped
   from ~1.8× to ~1.13× on the naive policy). It is the cheapest possible demonstration that the
   numbers are not self-serving, and it costs half a paragraph.

**One framing note I would carry into every tier**, because it is the difference between a defensive
paper and a confident one: the scaffold-collapse effect is *predicted by the mechanism*, not a
weakness to survive. Children of a shared hub share substructure **by construction**. The right
sentence is "batching around shared intermediates trades structural diversity for reaction economy;
we quantify the trade and report the regime in which it pays" — never "we win everywhere, though
margins narrow." Same plot, opposite reviewer reaction. Same discipline applies to the two places we
genuinely lose: cost per *candidate* (BC-Enum-SB at 1.03 vs our 1.22) and the from-scratch pricing
regime (305 vs 269). Lead with both.

---

## 7. Open questions for the researcher

Ordered by how much they block the rewrite.

1. **GEM page limit and deadline?** Sets how much of §6's `[W]` list survives.
2. **What is the 6TD3 fix?** §1.5 needs one or two sentences, and "we have a fix ready for the
   re-train" is not recoverable from the repo.
3. **Is the route dataset a contribution, or supporting material?** Logs/070 caps it at four cells
   with a stated exclusion rule. If it is a contribution it needs an outline slot it does not have;
   if it is supporting material, `Limitations` should carry the coverage statement.
4. **Which single number leads the abstract?** Candidates: **2.93×** (internal matrix, R=100, 30/31
   cell-seeds — broadest), **2.51×** (vs a cost-optimal MILP given our own enumerated molecules —
   hardest competitor), **2.27×** (selection alone on identical candidates — most conservative and
   most precisely the contribution), **2.02×** (vs the full external pipeline — most external). My
   preference is to lead with 2.93× for breadth and immediately give 2.27× as the like-for-like
   selection-only figure, because the second pre-empts the first's obvious objection.
5. **Does the paper still name the method "LSD-Flow"?** The old handoff flagged this as an open
   decision. The pun is load-bearing for the construct-validity argument but reads as the drug on
   first contact. The mechanism name ("hub-batching") is unaffected either way, and a workshop paper
   is a cheap place to test a name.
6. **Do we report the DRD2 cells separately from sEH** on the grounds that DRD2's gate is saturated
   (diversity-only) while sEH's binds? I think yes, and that averaging them into one number without
   saying so is the kind of thing a careful reviewer catches.
