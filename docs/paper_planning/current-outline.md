# Current outline
Written Aug 25, 2026
This was originally intended for an ICLR publication.

1. Abstract  
2. Introduction / Related Work  
   1. Why does Synthesis Space even matter?  
      1. Labs build libraries to hedge bets (not just 1 candidate)  
      2. Cost in lab is time, which is non-additive.  
      3. Late-Stage Diversification exists because chemists know this.  
   2. Existing techniques  
      1. Moving towards synthesis aware (why)  
         1) RGFN, SynFlowNet, RxnFlow, SCENT, CGFlow  
      2. Moving towards diversity aware (why)  
         1) Whole reason GFlowNets were made  
      3. Open Debate: S3-GFN argues reaction MDP is unnecessarily rigid. A sequence model with soft synthesizability regularization reaches \>= 95% synthesizable w/ more flexibility and scale.  
         1) Use exact quote / phrasing from publication here  
   2. Our Case  
      1. Learning synthesis space inherently makes molecules that are batchable.  
      2. reaction-MDPs might not be necessary for every case, but at library level they are currently unmatched.  
      3. Flow field a reaction-GFN already learned tells us which intermediates to batch around.  
         1) High level, why we think this might work, more fluffy and abstract.  
   3. Our Contributions  
      1. Axis (rxns / mode)  
      2. Mechanism (hub-batching)  
         1) Fixed charge to reach hub, marginal cost per child.  
         2) See little part under Lemma about guarantees.  
      3. Generality (across multiple reaction-GFNs)  
      4. Severe tests as findings  
   4. To Flag:  
      1. Different from SCENT dynamic library  
         1) SCENT caches high-reward intermediates as reusable structures.  
         2) Framed as library learning  
         3) Training-time reuse, built to expand block set  
         4) Ours is inference time, prices a delivered library.  
         5) They work well together and build off of each other.  
      2. Different from SPARROW \+ Diversity  
         1) Cite when we introduce Sparrow Batching, say what formulation we ran.  
2. Library-cost (reactions / mode)  
   1. Definitions, verbatim (from H3)  
      1. Chemistry library vs library, mode, reactions/mode, count-once, native vs from-scratch  
   2. Why per-molecule synthesizability underdetermines library cost  
      1. Short worked example  
   3. How to game the metric, how we closed it  
      1. Denominator (inflate mode count \-\> adversarial \-\> tau sweep, Butina / sphere-exclusion \+ second fingerprint, library-size normalization  
      2. Numerator (produce trivially cheap molecules) \-\> report reward AND synthetic-depth distributions.  
      3. Limiting case: depth-0 catalog picking is the true degenerate optimum; reward gate closes it. One hub with maximally diverse children is constrained optimum, not degenerate.  
   4. Construct Validity  
      1. Depth 0 \= fragment / catalog screening  
      2. Depth 1 w/ diverse purchasable starting materials \= parallel & combinatorial synthesis  
      3. Hub Limit \= late-stage diversification  
      4. Our limiting cases are already well documented as the goal for synthetic chemists.  
   5. Cost model audited against a third party  
      1. Count-once vs SPARROW MILP optimum: 0.0-3.1% (hub-batching) vs 4.8-5.0% (baseline); conservative where they disagree.  
   6. Filters are load-bearing  
      1. Dropping either filter flatters us.  
3. Methods  
   1. Pipeline w/ generative chemistry that chemists would actually use  
      1. Generator \-\> enumeration (sometimes) \-\> retrosynthesis (sometimes) \-\> selection  
      2. Retrosynthesis needed when models don’t produce synthesis steps  
   2. Hub-batching  
      1. 7 steps, include model diagram  
      2. State two unusual choices as choices: selection \+ evaluation cost models are decoupled (fair accounting), hub estimator is max over sampled children (near-unbiased one)  
      3. Formalism, \< half a page  
         1) Lemma \- DB rearrangement (hub flow calculation)  
         2) Reward-mass reading \- sort hubs by flow \= sort hubs by captured reward mass.  
         3) E-robustness corollary \- residual not small, ranking is error-bound  
         4) Proposition \- only if submodularity is verified against [campaign.py](http://campaign.py), otherwise cut  
         5) Select hubs H and children C maximizing the number of **pairwise t-separated** compounds, subject to a budget B≥∑h∈H​fh​+∑j∈C​cj.  
         6) It’s maximum independent set under a fixed-charge knapsack constraint, in a metric space, on a generative model's own reaction DAG. Every ingredient exists; the combination doesn't. Note precisely what's missing from the neighbors: \#Circles is post-hoc evaluation with no cost model. SPARROW optimizes cluster coverage exactly, but the clusters are precomputed and carry no geometric separation guarantee. Briem minimizes synthesis effort for coverage, but coverage ≠ separation. You'd be the first to make the separation guarantee a *constraint* rather than a *measurement*.  
      4. Models we’re using & comparing against  
         1) S3-GFn \+ FragGFN : non-reaction-aware GFNs  
         2) Non-reaction aware GFNs  
         3) Reaction-aware GFNs

4. Experiments  
   1. Setup  
      1. Goal: 100 modes. Briefly why reward is 5.0 not 7.0  
      2. SMALL library vs ZINC (advantage to competitor).  
      3. MutliAiZ requires retraining for SMALL library, so didn’t do it lol.  
   2. Hub-Batching beats other models (reaction & non-reaction aware. Non-reaction GFNs go here)  
      1. Clear distinction between reaction-aware & non-reaction aware  
      2. Clean 2x2 table (reaction-aware vs non-reaction-aware TIMES GFNs vs non-GFNs).  
      3. MutliAiZ \+ Greedy/SPARROW for everything EXCEPT reaction-GFNs. There put hub batching.  
      4. Comparison of SPARROW to hub-batching in next cell  
      5. Mentioned in H5.1  
   3. Hub-Batching across reaction-GFNs against other selection strategies. Flow field is doing the work.  
      1. Ordering ablation (reverse flow fails outright, random is okay)  
      2. BC-SB and BC-Enum-SB  
         1) Concede the loss on const-per-candidate  
         2) Acknowledge (potential) win on diversity  
   4. Failure modes & performance of hub-batching  
      1. LSD-Flow’s performance as the definition of “mode” changes  
      2. Heatmap, pareto front, LSD-Flow vs Best-Candidate  
      3. Filters on Diversity & Reward  
      4. 4 diff reward generators, docked 2m children, etc.  
   5. Compute Time comparison  
      1. Longest is Enumeration when surrogate quicker than 1ms  
      2. Longest is Reward when we do docking  
      3. More expensive the oracle, smaller hub-batching’s overhead\!  
      4. Each phase adds compute time.  
         1) Model Training \+ enumeration (sometimes) \+ retrosynthesis (sometimes) \+ selection (sparrow / greedy)  
         2) Also count oracle calls  
5. Severe Testing  
   1. Earlier cost model inflated result.  
      1. Found double-count, fixed, and added SPARROW verification just in case.  
   2. Trained GFlowNets violate conservation law at interior states  
      1. Flow into hub \!= flow out of hub  
      2. \-6.05 nats (RxnFlow), \-14.03 nats (SCENT).  
      3. Pure ML finding, really good for paper\!  
   3. Scaffold collapse as tau tightens, quantified tradeoff  
   4. Look at other “severe testing” from doc  
6. Conclusion  
   1. “In the regimes we measure, reaction-grounded generation delivers cheaper libraries at matched diversity, while costing more candidates to score.”  
   2. Decision process matters at library level, trained flow field already contains the selection signal.  
7. Limitations  
   1. Difficulty incorporating rich chemical priors.  
      1. Need to learn which synthesis steps lead to molecules with better children, not just which molecules have better performance.  
   2. Wet-Lab Validation, Active Learning Loop (diversity is loose proxy for information gain), Uncertainty modelling (these 3 each caused by the next)  
   3. Cost accounting ignores probability of reaction success / yield, cost of fragments, some parallelization is better than others (same reaction but different reagents is cheaper than diff reaction with diff reagents, because you can perform in 1 plate).  
   4. Error bars, route-choice asymmetry (\~16 alternative recipes vs 1.07 for us).
