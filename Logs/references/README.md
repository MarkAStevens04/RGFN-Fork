# References — orientation sheet

**Purpose:** a fast way for an agent (or a new contributor) to orient itself on the
handful of papers this project is *directly* built on — what the method is, why we
chose these systems, and which paper to open when a question comes up. This is **not**
an exhaustive bibliography; it's the short shelf of things you should actually know.

If you find yourself unsure what RGFN is, how a GFlowNet works, how the training loop is
supposed to run, or why we use the 6TD3 system, the answer is here — read the relevant
entry before reasoning further.

## Conventions

- **Cite by key, don't paraphrase.** In logs/docs write `[koziarski2024rgfn]`. The one
  authoritative citation string lives in `references.bib`.
- **PDFs are in `pdfs/<key>.pdf`** and are **git-ignored** (copyrighted + binary; keeps
  the repo clean for the code release). Open them locally by path. If one is
  missing, fetch it from the arXiv/DOI link below and name it after its key.

---

## What we're building (in one paragraph)

We fork **RGFN** (`[koziarski2024rgfn]`) as the generator and train it through the
**multi-round active-learning loop** from the original GFlowNet paper
(`[bengio2021gflownet]`, §4.3 / Alg. 1): RGFN samples molecules proportional to a fast
**proxy** reward; an expensive **oracle** (our docking-based *neosubstrate differential*,
MD later) scores only the per-round query batch; the proxy is refit on those labels;
repeat. The novel piece is **the oracle itself** — a two-tier, same-pose docking
differential that scores whether a candidate's *arm* recruits the second protein of the
ternary complex. Code: the generator lives in RGFN's `rgfn/gfns/reaction_gfn/`; our
oracle/proxy/sampler code lives in the `glue/` package and plugs into RGFN's reward
interface; configs in `configs/glue/`. See `docs/RESEARCH_CONTEXT.md` for the full picture.

---

## Method — what we're building with

### `[bengio2021gflownet]` — Flow Network based Generative Models (NeurIPS 2021)
The original **GFlowNet**. Two ideas to internalize:
1. A GFlowNet learns to *build an object step by step* and sample it with probability
   **proportional to a reward `R(x)`** — so you get many diverse high-reward samples, not
   one optimum.
2. **The training loop we use is theirs.** §4.3 ("Multi-Round Experiments") and
   **Algorithm 1** in Appendix A.5 define the active-learning loop: a learned **proxy `M`**
   (warm-started on a seed set `D_0` of true-oracle labels) is the in-loop reward, RGFN
   trains against `M(x)^β`, a query batch is sampled and scored by the expensive
   **oracle `O`**, the labels accumulate (`D_i = D̂_i ∪ D_{i-1}`), and `M` is refit on the
   full history each round. Their molecule instantiation (A.5.2) is an MPNN proxy predicting
   AutoDock scores, refit on ~200 freshly docked molecules per round — the direct template
   for our setup, with our docking differential in the role of `O`. **`O` scores enter only
   by retraining `M`, never as a direct RGFN reward.** Full pseudocode is transcribed in
   `docs/RESEARCH_CONTEXT.md` ("How the model learns"). &nbsp;`pdfs/bengio2021gflownet.pdf` · arXiv:2106.04399

### `[koziarski2024rgfn]` — RGFN: Synthesizable Molecular Generation (NeurIPS 2024)
**The paper this whole fork builds on.** RGFN = Reaction-GFlowNet: instead of growing a
molecule atom-by-atom, it assembles it through a **DAG of chemical reactions over a
building-block library**, so every generated molecule is synthesizable by construction.
The entire `rgfn/gfns/reaction_gfn/` package implements this; `data/chemistry.xlsx` is the
building-block/reaction library; our `glue/` oracles plug into its proxy/reward interface.
When extending the model, "does this match RGFN?" is answered here.
&nbsp;`pdfs/koziarski2024rgfn.pdf` · arXiv:2406.08506

### `[malkin2022trajectorybalance]` — Trajectory Balance: Improved Credit Assignment in GFlowNets (NeurIPS 2022)
**The GFlowNet training objective RGFN actually optimizes** (`configs/objectives/trajectory_balance.gin`
→ `@TrajectoryBalanceObjective`). TB learns the forward policy `P_F`, backward policy `P_B`, and a
single **global** scalar `Z = F(s0)` (the `logZ` parameter) by matching, over each complete
trajectory, `Z·∏P_F = R(x)·∏P_B` (their Eq. 13; Prop. 1 proves a global minimizer samples ∝ reward).
Crucially it learns **no per-state flow `F(s)`** — unlike flow-matching/detailed-balance — which is why
per-state flow must be recovered *post-hoc*: as a forward-sampling visit count, *or* from the balance
condition as `F(h)=R(x)·P_B(h|x)/P_F(x|h)` (the two are the `Z·∏P_F` and `R·∏P_B` sides of the TB loss,
so their agreement is a training-quality check).
&nbsp;`pdfs/malkin2022trajectorybalance.pdf` · arXiv:2201.13259

---

## Baselines — what we compare against

### `[seo2024rxnflow]` — RxnFlow: Generative Flows on Synthetic Pathway for Drug Design (2024, ICLR 2025)
Our primary **synthesis-aware** baseline — the synthesizable peer to RGFN (FragGFN
is the *non*-synthesizable foil). Like RGFN, RxnFlow is a GFlowNet that assembles
molecules along a **synthetic pathway** — picking a building block, then applying a
**reaction template** to a chosen reactant — so every sampled molecule carries a
forward-synthesis route. Its headline contribution is an **action-space subsampling**
trick that lets it learn over a huge action space (~1.2M building blocks × 71
reaction templates) without retraining when the library changes. It is built on
Recursion's `gflownet` (bundled, v0.2.0) — the *same* base as our FragGFN entrant —
so it drops into the same two-env pattern. We run it through the **same**
active-learning loop, the **same** 6TD3 docking oracle, the **same** seed/budget/β,
and the **same** proxy `M` as the RGFN entrant, so the comparison isolates the
generator. The standard candidate dataset records its routes (`has_route=1`,
`routes.jsonl`) — the differentiator vs. FragGFN. Heavy upstream code is installed
via `external/setup_rxnflow.sh` (not vendored); the thin adapter lives in
`validation/generators/rxnflow/`. &nbsp;`pdfs/seo2024rxnflow.pdf` · arXiv:2410.04542

### `[gainski2025scent]` — SCENT: Scalable and Cost-Efficient de Novo Template-Based Molecular Generation (2025)
Our **cost-aware** baseline — and the closest relative of all of them: SCENT is a
**fork of RGFN from the same lab** (its package is literally named `rgfn`; same
stack — py3.11/torch2.3/dgl/gin — same `rgfn.api`, same `train.py --cfg ….gin`),
so RGFN is one of *its* own baselines. It keeps RGFN's reaction-template,
synthesizable action space and adds three things on top: **Recursive Cost Guidance**
(auxiliary models that estimate synthesis cost from building-block prices + reaction
yields, steering the backward policy toward cheap routes), an **Exploitation Penalty**
(visitation-count term that keeps cost guidance from collapsing diversity), and a
**Dynamic Library** (promotes high-value intermediates to building blocks, enabling
tree-structured routes). Because the package name `rgfn` collides with ours, it runs
in its **own `scent` conda env** and reaches the shared 6TD3 oracle across the env
boundary via `scripts/score_batch.py` — the *same* two-env bridge pattern as FragGFN
/ RxnFlow, just forced by a namespace clash rather than a version clash. We run it
through the **same** active-learning loop, oracle, seed/budget/β, and proxy `M` as the
RGFN entrant, so the comparison isolates what SCENT's cost-awareness buys. It is
synthesizable (`has_route=1`, `routes.jsonl`). Heavy upstream code installed via
`external/setup_scent.sh` (not vendored); thin adapter in
`validation/generators/scent/`. &nbsp;`pdfs/gainski2025scent.pdf` · arXiv:2506.19865

### `[kim2026s3gfn]` — S3-GFN: Synthesizable Molecular Generation via Soft-constrained GFlowNets (2026)
The **marquee foil** for the LSD-Flow *library-efficiency* benchmark (`docs/LSD_FLOW_BENCHMARK_PLAN.md`
T3.1–T3.2). Unlike every other baseline here, S3-GFN is **not reaction-grounded**: it is a
**sequence (SMILES) GFlowNet** that induces synthesizability **softly** — off-policy replay training
with a **contrastive signal** from separate buffers of synthesizable vs. unsynthesizable samples,
plus rich priors from large SMILES corpora — rather than by construction from reaction templates. It
reports **>95% per-molecule synthesizability** with higher rewards, and argues you therefore *don't
need* a reaction-based decision process. Our benchmark concedes the per-molecule point and targets the
part it misses: **library** economics. S3-GFN's molecules carry **no shared-route structure**, so
assembling a *library* from them requires recovering shared intermediates **post-hoc** (AiZynth →
SPARROW, `has_route=0`, routes found not by-construction). The headline figure shows S3-GFN + the best
batch planner still loses to a reaction-GFN + hub-batching on **reactions-per-mode** — the
"MDP-necessary-for-library-economics" result. Mila/Bengio lineage (same as `[bengio2021gflownet]`).
Runs in its own conda env (`external/setup_s3gfn.sh`); pool ingested via
`scripts/ingest_candidates.py` (`has_route=0`) and routed by T1.3. &nbsp;`pdfs/kim2026s3gfn.pdf` ·
arXiv:2602.04119

### `[malik2023batchgfn]` — BatchGFN: Generative Flow Networks for Batch Active Learning (ICML 2023 workshop)
Where the **joint mutual information (JMI)** batch objective enters our world. A GFlowNet whose
*state* is the query batch under construction and whose *actions* add pool points, trained to sample
batches proportional to `exp(JMI/T)` — JMI being BatchBALD's `I[y₁:B, θ | x₁:B, D]`, closed-form
under an exact GP. Read it for the objective, not the results: it is a **10-page workshop paper
evaluated only on toy 1D regression** (pool 2000, query 10). **The line that matters for us** is
their §4.2–4.3: BatchGFN is *"on par with BatchBALD"* and beats BALD/random — its contribution is
**amortizing** the greedy objective, not improving it. Hence greedy BatchBALD **upper-bounds** it,
which is why `docs/LSD_FLOW_BENCHMARK_PLAN.md` §11 implements greedy BatchBALD rather than a
GFlowNet over subsets. Note it has **no cost model** — every pool point costs the same to label,
which is precisely the axis LSD-Flow owns. &nbsp;`pdfs/malik2023batchgfn.pdf` · arXiv:2306.15058 ·
[code](https://github.com/s-a-malik/batchgfn)

### `[zhang2025baldgfn]` — BALD-GFlowNet: Why Pool When You Can Flow? (2025)
The **stronger of the two information-theoretic benchmark targets** (`LSD_FLOW_BENCHMARK_PLAN.md`
§11) — the *generative* successor to pool-based acquisition, and the one actually run on molecules.
It swaps "which pool point is most informative?" for "what does an informative sample **look**
like?": a GFlowNet is trained to sample proportional to the BALD reward, so acquisition cost becomes
**independent of pool size**. Three details we build on: (1) it uses **single-point BALD**
`I(y; ω | x, D)` over an **ensemble**, *not* BatchBALD's joint MI — so it inherits BALD's
batch-redundancy weakness and offsets it with GFlowNet diversity; (2) its reward is
**multiplicative** — `MI · TPSA · QED · SAS · Rings` — our precedent for giving the information
baseline a quality term rather than beating a pure-exploration strawman (§11.2 decision 7); (3) the
ensemble surrogate is why our `M` becomes an ensemble (§11.2 decision 3). Results: comparable F1 to
the BALD baseline at a fraction of the oracle cost, 12.5% runtime reduction at a 12M library, more
diverse molecules (JAK2 / Enamine REAL, atom-level graph-Transformer GFlowNet). Its generator is
**atom-level with no reaction grounding and no synthesis cost** — the same library-economics blind
spot as `[kim2026s3gfn]`. &nbsp;`pdfs/zhang2025baldgfn.pdf` · arXiv:2509.00704

---

## Evaluation — synthesizability metrics & synthesis-cost tooling

### `[genheden2020aizynth]` — AiZynthFinder: a fast, robust retrosynthesis tool
The retrosynthesis engine behind the **synthesizability metric we report on every
entrant** (`validation/harness/synthesizability.py`). Given a target SMILES it runs a
Monte-Carlo-tree search over USPTO reaction templates back toward a **building-block
stock** (we use the standard public ZINC in-stock set); a molecule is **"solved"** iff a
full route to in-stock precursors is found. The headline number is the **fraction
solved** ("AiZynth success rate") — exactly the `AiZynth` column in `[koziarski2024rgfn]`
Table 1 (RGFN ≈ 0.56) and `[gainski2025scent]` Table 1 (up to ≈ 0.75), and RxnFlow's
"Synthesizability %" (`[seo2024rxnflow]`). Note these papers stress it is **noisy and
conservative** (RGFN molecules an expert confirmed synthesizable score below 1.0), so it
is a *post-hoc validation* metric, never an in-loop reward. Installed in its own
`aizynth` conda env via `external/setup_aizynthfinder.sh`. &nbsp;DOI:10.1186/s13321-020-00472-1

### `[ertl2009sascore]` — Synthetic Accessibility (SA) score
The cheap, RDKit-native companion to AiZynth that the same papers also report (a 1 = easy
… 10 = hard heuristic from fragment contributions + complexity penalties). We compute it
alongside the AiZynth verdict in the same evaluator. &nbsp;DOI:10.1186/1758-2946-1-8

### `[fromer2024sparrow]` — SPARROW: synthetic cost-aware decision making in molecular design (Nat Comput Sci 2024)
The **cost model everything in the library benchmark is ultimately priced against**, and the one
paper to read before touching a reactions/mode number. SPARROW ("Synthesis Planning And
Rewards-based Route Optimization Workflow", [`coleygroup/sparrow`](https://github.com/coleygroup/sparrow))
is a **mixed-integer linear program over a merged retrosynthetic graph** that jointly picks *which*
candidates to make and *which routes* to make them by, trading three scalarized terms: cumulative
reward of the selected candidates (maximized) against starting-material cost and a per-reaction
penalty inversely proportional to success probability (minimized). Compounds are deduplicated by
canonical SMILES, so **shared intermediates collapse to one node and amortize automatically** —
which is exactly why it can price a hub-batched library fairly.

**Read this next bit before citing it, because we use SPARROW for two opposite jobs** and
conflating them is the easiest way to misread a result (convention adopted 2026-08-04; see
`docs/RESEARCH_CONTEXT.md`, "How library cost is measured"):
- **SPARROW-Verifier (SV)** — *independent auditor.* Hand it a library we already chose plus its
  recipes and ask for the cheapest way to make **all** of it (`constrain_all_targets=True`, reward
  weight 0). We *expect* agreement; this is the source of "our count-once estimate is within
  0.0–3.1% of provably optimal, and hub-batching is closer to the optimum than the baseline"
  (`Logs/042` gate, `Logs/049`).
- **SPARROW-Batching (SB)** — *a genuine competitor.* Hand it a pool + a hard reaction budget and
  it chooses **which** molecules to make (`constrain_all_targets=False`, reward-bearing objective,
  `--max-rxns`). The competitor's selector in the headline (`Logs/056`) and in the BC-SB /
  BC-Enum-SB arms (`Logs/059`). Always name the pool with it.

**The single most load-bearing fact:** the objective has **no diversity term**, and the authors say
so themselves — *"SPARROW currently does not consider marginal information gain related to
molecular diversity and matched molecular pairs."* Quote them, not our source-code reading, when
the paper needs it. That limitation is the mechanism behind `Logs/059`: a cost-only optimizer
concentrates its picks on a handful of intermediates (78% from one at the tightest budget), so the
distinctness of an SB selection is always **measured, never assumed**. Sign convention verified
against `LinearSelector.set_objective` — it *minimizes*
`-w₀·Σ(reward×selected) + w₁·Σ(SM cost) + w₂·Σ(reaction penalty)`, so `weights=[1,0,0,0,0]` plus a
hard `max_rxns` maximizes selected reward under a reaction budget. The MILP is superlinear in pool
size (~1 s per budget point at 500 targets, >110 s at 2,000), so a pool that fails to solve is
itself a reportable datapoint. Own `sparrow` conda env (`external/setup_sparrow.sh`) + PuLP/CBC,
crossed by subprocess via `validation/lsdflow/adapters/workers/sparrow_worker.py`.
&nbsp;DOI:10.1038/s43588-024-00639-y · arXiv:2311.02187 · `pdfs/fromer2024sparrow.pdf`


### `[fromer2025diversity]` — SPARROW v2: optimal downselection for diversity and parallel chemistry (JCIM 2025)
The **same group's follow-up to [fromer2024sparrow]**, and the paper that answers the sharpest
objection to our competitor arms: that we only ever raced against a *diversity-blind* optimizer
(Logs/059 — handed our own enumerated children, SPARROW took 98 candidates off 3 hubs and produced
2 distinct molecules). Three additions, all already present in our clone:

1. **Expected cumulative reward.** Instead of a linear weighted sum, maximize
   `Σ_t U_t·c_t·Π_i L_i^(u_i,t)` — discount each candidate's reward by the success probability of
   *every step in its route*. A risky reaction is then penalized **once per route that uses it**
   rather than once globally. This is **nonlinear**; the authors state it needs **Gurobi**, is not
   guaranteed to reach a global optimum, and they therefore use the **linear** formulation (with an
   iterative `λ_rew` scheme, SI S1.3) for all of their own analysis. We run **PuLP/CBC with no
   Gurobi licence**, so the linear path is the one available to us — which is also the one they
   recommend.
2. **Cluster diversity.** Add `λ_div ×(number of clusters represented)` to the scalarized objective,
   or impose clusters-represented as a *constraint*. Clusters are deliberately arbitrary: they use
   Butina on count-Morgan at 0.8, but note they can be scaffolds, predicted protein interactions, or
   any project-specific criterion — so **our own τ-mode definition can be dropped in directly**,
   making the comparison like-for-like on our metric rather than on theirs.
3. **Parallel chemistry.** An inequality constraint capping the number of distinct **reaction
   classes** selected, so the batch can be run in parallel.

Their Fig. 3C result matters for how we should expect our arms to move: raising `λ_div` *reduced*
the number of selected reactions and *raised* mean reaction score — in their case study diversity
was **not** bought with extra steps. They caution this does not hold on every candidate set (SI S4).

**Status in this repo: no upgrade needed.** `external/sparrow` is already at this version (clone
dated 2025-06-30; `selector/{linear,nonlinear,bayesian}`), and `LinearSelector` already exposes
`clusters` / `N_per_cluster` / `rxn_classes` / `max_rxn_classes` with
`weights = [reward, start_cost, reaction, diversity, class]`. Our worker simply hardcodes the last
two weights to zero and never passes clusters — so enabling this is **configuration, not
re-implementation**.
&nbsp;DOI:10.1021/acs.jcim.5c00606 · `pdfs/fromer2025diversity.pdf`

### `[ianez2026multiaiz]` — MultiAiZ: joint synthesis planning by leveraging common intermediates
The **second competitor route-planner** for the LSD-Flow library benchmark (T4.1), a smarter
alternative to plain AiZynth→SPARROW. Where AiZynth routes each target *independently*,
MultiAiZ ([`MolecularAI/multiaiz`](https://github.com/MolecularAI/multiaiz)) plans over a
**set** of targets: it runs AiZynthFinder for `n_iters` **cycles** (paper: **5**), and after
each cycle **appends the discovered intermediates to the stock** so later targets route
*through* them → convergent routes that reuse shared intermediates. It ranks intermediates by a
novel **"intermediate score"** (amortized subtree cost × reaction-class utility). We use it as a
per-pool pricer: run MultiAiZ on *one* acquisition function's accepted-mode pool in **isolation**
(sharing found within a pool is **not** leaked to other pools), feed the resulting route trees to
the same SPARROW MILP over the base ZINC stock (so a shared intermediate is **built once and
amortized**, not free), and ask whether smarter discovery lowers **reactions-per-mode** vs plain
AiZynth→SPARROW, or SPARROW's MILP already captures the sharing. Own `multiaiz` conda env
(`external/setup_multiaiz.sh`); Molecular AI / AstraZeneca, same group as AiZynthFinder.
&nbsp;DOI:10.1016/j.ailsci.2026.100175 · `pdfs/ianez2026multiaiz.pdf`

---

## Domain — systems, glue design & evaluation

### `[koziarski2024rgfn]` is method; these explain the *chemistry* we're scoring.

### `[bengeoffrey2025molde]` — Molecular Glue-Design-Evaluator (ACS Omega 2025)
In-silico method for **designing and scoring molecular glues**. Consult this when
reasoning about glue design or oracle scoring choices — and note it's the reference
behind our choice of the **6TD3** system as the testbed (`Logs/002_6td3-cr8-validation-and-discrimination.md`).
&nbsp;`pdfs/bengeoffrey2025molde.pdf` · doi:10.1021/acsomega.4c08049

### `[slabicki2020cr8]` — CR8 is a molecular glue degrader of cyclin K (Nature 2020)
Source of the **6TD3** system: DDB1·CDK12–cyclinK·CR8 ternary complex — our **validated
oracle** (78-pp separation on the neosubstrate differential). Cited in logs 002, 003, 005.
&nbsp;doi:10.1038/s41586-020-2133-z &nbsp;*(no PDF — paywalled; drop in if obtained)*

### `[matyskiela2016cc885]` — Cereblon modulator recruits GSPT1 (Nature 2016)
Source of the **5HXB** system: CRBN·DDB1·GSPT1·CC-885 — the **ceiling-hit** system where
docking can't separate real glues from decoys. Cited in logs 001, 003, 005.
&nbsp;doi:10.1038/nature18611 &nbsp;*(no PDF — paywalled; drop in if obtained)*
> ⚠️ Logs 001 & 003 cite this as "Science 2016" — it is **Nature 535:252–257 (2016)**.
> Fix when those logs are next touched.

---

## Adding a paper

Keep this sheet short — add a paper only if work genuinely builds on it.
1. Drop the PDF in `pdfs/` named `<citekey>.pdf` (won't be committed — fine).
2. Add the BibTeX entry to `references.bib` (key = `<firstauthor><year><tag>`).
3. Add a 2–4 line entry here: what it is, why we care, and a pointer if useful.
4. Cite it by key from logs/docs — never restate the citation inline.
