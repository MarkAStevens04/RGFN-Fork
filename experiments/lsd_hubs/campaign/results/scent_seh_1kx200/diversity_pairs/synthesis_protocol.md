# Synthesis protocols — closest diversity pairs (SCENT sEH hub-batching, scent_seh_1kx200)

Step-by-step routes for the two most-similar-yet-distinct molecules at each cutoff. Reactions are the SCENT dynamic-library assembly steps we logged (named from their reaction templates); **buy** = a base building block, **make** = an intermediate built in an earlier step. The final step attaches the hub building block — its reaction is the **logged** template when available (else inferred from the structural change), and the hub's own build steps are included when a sampled route (`routes.json`) is supplied. This is the logged assembly route, not a claim of an optimal retrosynthesis.

## Cutoff 0.30 — closest pair (Tanimoto 0.3)
Same hub reagent (`ClCC1CC1`) and the same reaction *sequence* from mostly-shared building blocks, but they **diverge at the first reaction** — no reusable built intermediate (they share purchases, not an intermediate; see the buy list).

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `Nc1ccc(Br)cc1N` | A/B | shared |
| `O=C(O)C1CNC1` | A/B | shared |
| `C#C` | B | diverging |
| `NCc1c[nH]c2ccccc12` | A | diverging |

**Molecule A (sEH 8.325)**

1. **Benzimidazole formation (o-diamine + acid) [t183]** — `O=C(O)C1CNC1` (buy) + `Nc1ccc(Br)cc1N` (buy) → `Brc1ccc2[nH]c(C3CNC3)nc2c1`
2. **Buchwald–Hartwig amination (aryl halide + amine) [t113]** — `Brc1ccc2[nH]c(C3CNC3)nc2c1` (make) + `NCc1c[nH]c2ccccc12` (buy) → `c1ccc2c(CNc3ccc4[nH]c(C5CNC5)nc4c3)c[nH]c2c1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `c1ccc2c(CNc3ccc4[nH]c(C5CNC5)nc4c3)c[nH]c2c1` (make) + `ClCC1CC1` (buy) → `c1ccc2c(c1)c(CNc1ccc3[nH]c(C4CNC4)nc3c1)cn2CC1CC1`

**Molecule B (sEH 8.121)**

1. **Benzimidazole formation (o-diamine + acid) [t183]** — `O=C(O)C1CNC1` (buy) + `Nc1ccc(Br)cc1N` (buy) → `Brc1ccc2nc(C3CNC3)[nH]c2c1`
2. **Sonogashira coupling (aryl halide + alkyne) [t120]** — `C#C` (buy) + `Brc1ccc2nc(C3CNC3)[nH]c2c1` (make) → `C#Cc1ccc2nc(C3CNC3)[nH]c2c1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `C#Cc1ccc2nc(C3CNC3)[nH]c2c1` (make) + `ClCC1CC1` (buy) → `c1cc2nc(C3CNC3)[nH]c2cc1-c1cn(CC2CC2)nn1`

## Cutoff 0.35 — closest pair (Tanimoto 0.35)
**Different hubs** — independently synthesised (`ClCC1CC1` vs `O=C(O)C1Cc2ccccc2CN1`).

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `Brc1c[nH]cn1` | A | diverging |
| `Ic1c[nH]cn1` | B | diverging |
| `Nc1ccc(B(O)O)cc1` | A | diverging |
| `Nc1cccc(B(O)O)c1` | B | diverging |
| `O=C(O)C1CCN1` | A | diverging |
| `O=C(O)C1Cc2ccccc2CN1` | B | diverging |

**Molecule A (sEH 8.136)**

1. **Suzuki coupling (biaryl bond) [t87]** — `Brc1c[nH]cn1` (buy) + `Nc1ccc(B(O)O)cc1` (buy) → `Nc1ccc(-c2c[nH]cn2)cc1`
2. **Amide coupling (acid + amine) [t137]** — `Nc1ccc(-c2c[nH]cn2)cc1` (make) + `O=C(O)C1CCN1` (buy) → `c1nc(-c2ccc(NCC3CCN3)cc2)c[nH]1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `c1nc(-c2ccc(NCC3CCN3)cc2)c[nH]1` (make) + `ClCC1CC1` (buy) → `c1cc(-c2cn(CC3CC3)cn2)ccc1NCC1CCN1`

**Molecule B (sEH 8.029)**

1. **Suzuki coupling (biaryl bond) [t84]** — `Nc1cccc(B(O)O)c1` (buy) + `Ic1c[nH]cn1` (buy) → `Nc1cccc(-c2c[nH]cn2)c1`
2. **N-alkylation (ring N–H + alkyl halide) [t200]** — `Nc1cccc(-c2c[nH]cn2)c1` (make) + `ClCC1CC1` (buy) → `Nc1cccc(-c2cn(CC3CC3)cn2)c1`
3. **Amide coupling (amine + hub carboxylic acid)** — `Nc1cccc(-c2cn(CC3CC3)cn2)c1` (make) + `O=C(O)C1Cc2ccccc2CN1` (buy) → `O=C(Nc1cccc(-c2cn(CC3CC3)cn2)c1)C1Cc2ccccc2CN1`

## Cutoff 0.40 — closest pair (Tanimoto 0.4)
Same hub reagent (`ClCC1CC1`) and the same reaction *sequence* from mostly-shared building blocks, but they **diverge at the first reaction** — no reusable built intermediate (they share purchases, not an intermediate; see the buy list).

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `NCc1c[nH]c2ccccc12` | A/B | shared |
| `C1CNCCN1` | A | diverging |
| `Nc1cc(Br)n[nH]1` | B | diverging |
| `O=C(O)[C@@H]1Cc2ccccc2CN1` | A | diverging |
| `O=C(O)[C@H]1CCCNC1` | B | diverging |

**Molecule A (sEH 7.744)**

1. **Amide coupling (acid + secondary amine → tertiary amide) [t139]** — `C1CNCCN1` (buy) + `O=C(O)[C@@H]1Cc2ccccc2CN1` (buy) → `c1ccc2c(c1)CN[C@H](CN1CCNCC1)C2`
2. **Amide coupling (acid + secondary amine → tertiary amide) [t126]** — `c1ccc2c(c1)CN[C@H](CN1CCNCC1)C2` (make) + `NCc1c[nH]c2ccccc12` (buy) → `O=C(NCc1c[nH]c2ccccc12)N1CCN(C[C@@H]2Cc3ccccc3CN2)CC1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `O=C(NCc1c[nH]c2ccccc12)N1CCN(C[C@@H]2Cc3ccccc3CN2)CC1` (make) + `ClCC1CC1` (buy) → `O=C(NCc1cn(CC2CC2)c2ccccc12)N1CCN(CC2Cc3ccccc3CN2)CC1`

**Molecule B (sEH 7.118)**

1. **Amide coupling (acid + secondary amine → tertiary amide) [t127]** — `Nc1cc(Br)n[nH]1` (buy) + `O=C(O)[C@H]1CCCNC1` (buy) → `O=C(O)[C@H]1CCCN(C(=O)Nc2cc(Br)n[nH]2)C1`
2. **Amide coupling (acid + amine) [t76]** — `O=C(O)[C@H]1CCCN(C(=O)Nc2cc(Br)n[nH]2)C1` (make) + `NCc1c[nH]c2ccccc12` (buy) → `O=C(NCc1c[nH]c2ccccc12)[C@H]1CCCN(C(=O)Nc2cc(Br)n[nH]2)C1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `O=C(NCc1c[nH]c2ccccc12)[C@H]1CCCN(C(=O)Nc2cc(Br)n[nH]2)C1` (make) + `ClCC1CC1` (buy) → `O=C(NCc1cn(CC2CC2)c2ccccc12)C1CCCN(C(=O)Nc2cc(Br)n[nH]2)C1`

## Cutoff 0.45 — closest pair (Tanimoto 0.45)
**Different hubs** — independently synthesised (`ClCC1CC1` vs `NCCc1cn(CC2CC2)c2ccccc12`).

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `NCCc1c[nH]c2ccccc12` | A/B | shared |
| `NC1CC1` | A | diverging |
| `Nc1ccc(Br)cc1` | B | diverging |
| `O=C(O)c1cccc(B(O)O)c1` | B | diverging |
| `O=C=Nc1cccc2ccccc12` | B | diverging |
| `O=Cc1ccc(C(=O)O)cc1` | A | diverging |

**Molecule A (sEH 7.986)**

1. **Amide coupling (acid + amine) [t77]** — `NC1CC1` (buy) + `O=Cc1ccc(C(=O)O)cc1` (buy) → `O=Cc1ccc(C(=O)NC2CC2)cc1`
2. **Reductive amination (aldehyde + amine) [t68]** — `NCCc1c[nH]c2ccccc12` (buy) + `O=Cc1ccc(C(=O)NC2CC2)cc1` (make) → `O=C(NC1CC1)c1ccc(CNCCc2c[nH]c3ccccc23)cc1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `O=C(NC1CC1)c1ccc(CNCCc2c[nH]c3ccccc23)cc1` (make) + `ClCC1CC1` (buy) → `O=C(NC1CC1)c1ccc(CNCCc2cn(CC3CC3)c3ccccc23)cc1`

**Molecule B (sEH 7.713)**

1. **N-alkylation (ring N–H + alkyl halide) [t200]** — `NCCc1c[nH]c2ccccc12` (buy) + `ClCC1CC1` (buy) → `NCCc1cn(CC2CC2)c2ccccc12`
2. **reaction (RGFN template 80) [t80]** — `Nc1ccc(Br)cc1` (buy) + `O=C=Nc1cccc2ccccc12` (buy) → `O=C(Nc1ccc(Br)cc1)Nc1cccc2ccccc12`
3. **Suzuki coupling (biaryl bond) [t87]** — `O=C(Nc1ccc(Br)cc1)Nc1cccc2ccccc12` (make) + `O=C(O)c1cccc(B(O)O)c1` (buy) → `O=C(Nc1ccc(-c2cccc(C(=O)O)c2)cc1)Nc1cccc2ccccc12`
4. **Final assembly (attach hub building block)** — `O=C(Nc1ccc(-c2cccc(C(=O)O)c2)cc1)Nc1cccc2ccccc12` (make) + `NCCc1cn(CC2CC2)c2ccccc12` (make) → `O=C(Nc1ccc(-c2cccc(CNCCc3cn(CC4CC4)c4ccccc34)c2)cc1)Nc1cccc2ccccc12`

## Cutoff 0.50 — closest pair (Tanimoto 0.5)
Same hub reagent (`ClCC1CC1`) and the same reaction *sequence* from mostly-shared building blocks, but they **diverge at the first reaction** — no reusable built intermediate (they share purchases, not an intermediate; see the buy list).

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `N#Cc1ccc(CBr)cc1` | A/B | shared |
| `O=C(O)C1CNC1` | A/B | shared |
| `NCCc1c[nH]c2ccccc12` | B | diverging |
| `NCc1c[nH]c2ccccc12` | A | diverging |

**Molecule A (sEH 8.302)**

1. **N-alkylation (ring N–H + alkyl halide) [t202]** — `NCc1c[nH]c2ccccc12` (buy) + `N#Cc1ccc(CBr)cc1` (buy) → `N#Cc1ccc(Cn2cc(CN)c3ccccc32)cc1`
2. **Amide coupling (acid + amine) [t77]** — `N#Cc1ccc(Cn2cc(CN)c3ccccc32)cc1` (make) + `O=C(O)C1CNC1` (buy) → `N#Cc1ccc(Cn2cc(CNC(=O)C3CNC3)c3ccccc32)cc1`
3. **Tetrazole formation + N-alkylation (nitrile + hub)** — `N#Cc1ccc(Cn2cc(CNC(=O)C3CNC3)c3ccccc32)cc1` (make) + `ClCC1CC1` (buy) → `O=C(NCc1cn(Cc2ccc(-c3nnn(CC4CC4)n3)cc2)c2ccccc12)C1CNC1`

**Molecule B (sEH 7.761)**

1. **N-alkylation (ring N–H + alkyl halide) [t203]** — `N#Cc1ccc(CBr)cc1` (buy) + `NCCc1c[nH]c2ccccc12` (buy) → `N#Cc1ccc(Cn2cc(CCN)c3ccccc32)cc1`
2. **Amide coupling (acid + amine) [t77]** — `N#Cc1ccc(Cn2cc(CCN)c3ccccc32)cc1` (make) + `O=C(O)C1CNC1` (buy) → `N#Cc1ccc(Cn2cc(CCNC(=O)C3CNC3)c3ccccc32)cc1`
3. **Tetrazole formation + N-alkylation (nitrile + hub)** — `N#Cc1ccc(Cn2cc(CCNC(=O)C3CNC3)c3ccccc32)cc1` (make) + `ClCC1CC1` (buy) → `O=C(NCCc1cn(Cc2ccc(-c3nnnn3CC3CC3)cc2)c2ccccc12)C1CNC1`

## Cutoff 0.55 — closest pair (Tanimoto 0.5496)
Same hub reagent (`ClCC1CC1`) and the same reaction *sequence* from mostly-shared building blocks, but they **diverge at the first reaction** — no reusable built intermediate (they share purchases, not an intermediate; see the buy list).

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `N#Cc1ccc(C(=O)O)cn1` | A/B | shared |
| `NCCc1c[nH]c2ccccc12` | A/B | shared |
| `Nc1c(O)cccc1C(=O)O` | B | diverging |
| `O=C(O)C1CCNCC1` | A | diverging |

**Molecule A (sEH 7.858)**

1. **Amide coupling (acid + secondary amine → tertiary amide) [t139]** — `O=C(O)C1CCNCC1` (buy) + `N#Cc1ccc(C(=O)O)cn1` (buy) → `N#Cc1ccc(CN2CCC(C(=O)O)CC2)cn1`
2. **Amide coupling (acid + amine) [t76]** — `N#Cc1ccc(CN2CCC(C(=O)O)CC2)cn1` (make) + `NCCc1c[nH]c2ccccc12` (buy) → `N#Cc1ccc(CN2CCC(C(=O)NCCc3c[nH]c4ccccc34)CC2)cn1`
3. **Tetrazole formation + N-alkylation (nitrile + hub)** — `N#Cc1ccc(CN2CCC(C(=O)NCCc3c[nH]c4ccccc34)CC2)cn1` (make) + `ClCC1CC1` (buy) → `O=C(NCCc1c[nH]c2ccccc12)C1CCN(Cc2ccc(-c3nnn(CC4CC4)n3)nc2)CC1`

**Molecule B (sEH 7.51)**

1. **Benzoxazole formation (2-aminophenol + acid/aldehyde) [t181]** — `N#Cc1ccc(C(=O)O)cn1` (buy) + `Nc1c(O)cccc1C(=O)O` (buy) → `N#Cc1ccc(-c2nc3c(C(=O)O)cccc3o2)cn1`
2. **Amide coupling (acid + amine) [t76]** — `N#Cc1ccc(-c2nc3c(C(=O)O)cccc3o2)cn1` (make) + `NCCc1c[nH]c2ccccc12` (buy) → `N#Cc1ccc(-c2nc3c(C(=O)NCCc4c[nH]c5ccccc45)cccc3o2)cn1`
3. **Tetrazole formation + N-alkylation (nitrile + hub)** — `N#Cc1ccc(-c2nc3c(C(=O)NCCc4c[nH]c5ccccc45)cccc3o2)cn1` (make) + `ClCC1CC1` (buy) → `O=C(NCCc1c[nH]c2ccccc12)c1cccc2oc(-c3ccc(-c4nnn(CC5CC5)n4)nc3)nc12`

## Cutoff 0.60 — closest pair (Tanimoto 0.6)
Both molecules are built from the **same hub** (`ClCC1CC1`) and share **1** fully-built intermediate step(s) (made once), then diverge at the final decoration.

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `NCc1c[nH]c2ccccc12` | A/B | shared |
| `Nc1cc(Br)ccc1O` | A/B | shared |
| `Nc1cc(C(=O)O)ccc1O` | B | diverging |
| `O=C(O)C1CCN1` | A | diverging |

**Shared steps (built once)**

1. **Buchwald–Hartwig amination (aryl halide + amine) [t113]** — `Nc1cc(Br)ccc1O` (buy) + `NCc1c[nH]c2ccccc12` (buy) → `Nc1cc(NCc2c[nH]c3ccccc23)ccc1O`

**Molecule A (sEH 8.137) — continues from the shared intermediate**

2. **Benzoxazole formation (2-aminophenol + acid/aldehyde) [t181]** — `O=C(O)C1CCN1` (buy) + `Nc1cc(NCc2c[nH]c3ccccc23)ccc1O` (make) → `c1ccc2c(CNc3ccc4oc(C5CCN5)nc4c3)c[nH]c2c1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `c1ccc2c(CNc3ccc4oc(C5CCN5)nc4c3)c[nH]c2c1` (make) + `ClCC1CC1` (buy) → `c1ccc2c(c1)c(CNc1ccc3oc(C4CCN4)nc3c1)cn2CC1CC1`

**Molecule B (sEH 7.13) — continues from the shared intermediate**

2. **Benzoxazole formation (2-aminophenol + acid/aldehyde) [t181]** — `Nc1cc(C(=O)O)ccc1O` (buy) + `Nc1cc(NCc2c[nH]c3ccccc23)ccc1O` (make) → `Nc1cc(-c2nc3cc(NCc4c[nH]c5ccccc45)ccc3o2)ccc1O`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `Nc1cc(-c2nc3cc(NCc4c[nH]c5ccccc45)ccc3o2)ccc1O` (make) + `ClCC1CC1` (buy) → `Nc1cc(-c2nc3cc(NCc4cn(CC5CC5)c5ccccc45)ccc3o2)ccc1O`

## Cutoff 0.65 — closest pair (Tanimoto 0.65)
Same hub reagent (`ClCC1CC1`) and the same reaction *sequence* from mostly-shared building blocks, but they **diverge at the first reaction** — no reusable built intermediate (they share purchases, not an intermediate; see the buy list).

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `N#Cc1cccc(B(O)O)c1` | A/B | shared |
| `O=Cc1nc2ccccc2[nH]1` | A/B | shared |
| `Nc1cc(Br)n[nH]1` | A | diverging |
| `Nc1ccc(Br)cc1` | B | diverging |

**Molecule A (sEH 7.724)**

1. **Suzuki coupling (biaryl bond) [t86]** — `N#Cc1cccc(B(O)O)c1` (buy) + `Nc1cc(Br)n[nH]1` (buy) → `N#Cc1cccc(-c2cc(N)[nH]n2)c1`
2. **Reductive amination (aldehyde + amine) [t68]** — `N#Cc1cccc(-c2cc(N)[nH]n2)c1` (make) + `O=Cc1nc2ccccc2[nH]1` (buy) → `N#Cc1cccc(-c2cc(NCc3nc4ccccc4[nH]3)[nH]n2)c1`
3. **Tetrazole formation + N-alkylation (nitrile + hub)** — `N#Cc1cccc(-c2cc(NCc3nc4ccccc4[nH]3)[nH]n2)c1` (make) + `ClCC1CC1` (buy) → `c1cc(-c2cc(NCc3nc4ccccc4[nH]3)[nH]n2)cc(-c2nnn(CC3CC3)n2)c1`

**Molecule B (sEH 7.676)**

1. **Reductive amination (aldehyde + amine) [t69]** — `O=Cc1nc2ccccc2[nH]1` (buy) + `Nc1ccc(Br)cc1` (buy) → `Brc1ccc(NCc2nc3ccccc3[nH]2)cc1`
2. **Suzuki coupling (biaryl bond) [t86]** — `N#Cc1cccc(B(O)O)c1` (buy) + `Brc1ccc(NCc2nc3ccccc3[nH]2)cc1` (make) → `N#Cc1cccc(-c2ccc(NCc3nc4ccccc4[nH]3)cc2)c1`
3. **Tetrazole formation + N-alkylation (nitrile + hub)** — `N#Cc1cccc(-c2ccc(NCc3nc4ccccc4[nH]3)cc2)c1` (make) + `ClCC1CC1` (buy) → `c1cc(-c2ccc(NCc3nc4ccccc4[nH]3)cc2)cc(-c2nnn(CC3CC3)n2)c1`

## Cutoff 0.70 — closest pair (Tanimoto 0.7)
Same hub reagent (`ClCC1CC1`) and the same reaction *sequence* from mostly-shared building blocks, but they **diverge at the first reaction** — no reusable built intermediate (they share purchases, not an intermediate; see the buy list).

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `O=C(O)C1CNC1` | A/B | shared |
| `OB(O)c1ccc2[nH]ccc2c1` | A/B | shared |
| `Nc1cc(Br)ccc1O` | A | diverging |
| `Nc1ccc(Br)cc1O` | B | diverging |

**Molecule A (sEH 7.981)**

1. **Benzoxazole formation (2-aminophenol + acid/aldehyde) [t181]** — `O=C(O)C1CNC1` (buy) + `Nc1cc(Br)ccc1O` (buy) → `Brc1ccc2oc(C3CNC3)nc2c1`
2. **Suzuki coupling (biaryl bond) [t86]** — `OB(O)c1ccc2[nH]ccc2c1` (buy) + `Brc1ccc2oc(C3CNC3)nc2c1` (make) → `c1cc2cc(-c3ccc4oc(C5CNC5)nc4c3)ccc2[nH]1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `c1cc2cc(-c3ccc4oc(C5CNC5)nc4c3)ccc2[nH]1` (make) + `ClCC1CC1` (buy) → `c1cc2c(ccn2CC2CC2)cc1-c1ccc2oc(C3CNC3)nc2c1`

**Molecule B (sEH 7.954)**

1. **Benzoxazole formation (2-aminophenol + acid/aldehyde) [t180]** — `Nc1ccc(Br)cc1O` (buy) + `O=C(O)C1CNC1` (buy) → `Brc1ccc2nc(C3CNC3)oc2c1`
2. **Suzuki coupling (biaryl bond) [t86]** — `OB(O)c1ccc2[nH]ccc2c1` (buy) + `Brc1ccc2nc(C3CNC3)oc2c1` (make) → `c1cc2cc(-c3ccc4nc(C5CNC5)oc4c3)ccc2[nH]1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `c1cc2cc(-c3ccc4nc(C5CNC5)oc4c3)ccc2[nH]1` (make) + `ClCC1CC1` (buy) → `c1cc2c(ccn2CC2CC2)cc1-c1ccc2nc(C3CNC3)oc2c1`

## Cutoff 0.75 — closest pair (Tanimoto 0.75)
Same hub reagent (`ClCC1CC1`) and the same reaction *sequence* from mostly-shared building blocks, but they **diverge at the first reaction** — no reusable built intermediate (they share purchases, not an intermediate; see the buy list).

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `NCc1c[nH]c2ccccc12` | A/B | shared |
| `O=C(O)C1CNC1` | A/B | shared |
| `Nc1cc(C(=O)O)ccc1O` | B | diverging |
| `Nc1ccc(C(=O)O)cc1O` | A | diverging |

**Molecule A (sEH 8.271)**

1. **Benzoxazole formation (2-aminophenol + acid/aldehyde) [t181]** — `O=C(O)C1CNC1` (buy) + `Nc1ccc(C(=O)O)cc1O` (buy) → `O=C(O)c1ccc2nc(C3CNC3)oc2c1`
2. **Amide coupling (acid + amine) [t76]** — `O=C(O)c1ccc2nc(C3CNC3)oc2c1` (make) + `NCc1c[nH]c2ccccc12` (buy) → `O=C(NCc1c[nH]c2ccccc12)c1ccc2nc(C3CNC3)oc2c1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `O=C(NCc1c[nH]c2ccccc12)c1ccc2nc(C3CNC3)oc2c1` (make) + `ClCC1CC1` (buy) → `O=C(NCc1cn(CC2CC2)c2ccccc12)c1ccc2nc(C3CNC3)oc2c1`

**Molecule B (sEH 8.228)**

1. **Benzoxazole formation (2-aminophenol + acid/aldehyde) [t181]** — `O=C(O)C1CNC1` (buy) + `Nc1cc(C(=O)O)ccc1O` (buy) → `O=C(O)c1ccc2oc(C3CNC3)nc2c1`
2. **Amide coupling (acid + amine) [t76]** — `O=C(O)c1ccc2oc(C3CNC3)nc2c1` (make) + `NCc1c[nH]c2ccccc12` (buy) → `O=C(NCc1c[nH]c2ccccc12)c1ccc2oc(C3CNC3)nc2c1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `O=C(NCc1c[nH]c2ccccc12)c1ccc2oc(C3CNC3)nc2c1` (make) + `ClCC1CC1` (buy) → `O=C(NCc1cn(CC2CC2)c2ccccc12)c1ccc2oc(C3CNC3)nc2c1`

## Cutoff 0.80 — closest pair (Tanimoto 0.8)
Same hub reagent (`ClCC1CC1`) and the same reaction *sequence* from mostly-shared building blocks, but they **diverge at the first reaction** — no reusable built intermediate (they share purchases, not an intermediate; see the buy list).

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `N#Cc1ccc(C(=O)O)cn1` | A/B | shared |
| `Nc1cccc2cc[nH]c12` | A/B | shared |
| `Nc1cc(C(=O)O)ccc1O` | A | diverging |
| `Nc1ccc(C(=O)O)cc1O` | B | diverging |

**Molecule A (sEH 7.72)**

1. **Benzoxazole formation (2-aminophenol + acid/aldehyde) [t181]** — `N#Cc1ccc(C(=O)O)cn1` (buy) + `Nc1cc(C(=O)O)ccc1O` (buy) → `N#Cc1ccc(-c2nc3cc(C(=O)O)ccc3o2)cn1`
2. **Amide coupling (acid + amine) [t137]** — `Nc1cccc2cc[nH]c12` (buy) + `N#Cc1ccc(-c2nc3cc(C(=O)O)ccc3o2)cn1` (make) → `N#Cc1ccc(-c2nc3cc(CNc4cccc5cc[nH]c45)ccc3o2)cn1`
3. **Tetrazole formation + N-alkylation (nitrile + hub)** — `N#Cc1ccc(-c2nc3cc(CNc4cccc5cc[nH]c45)ccc3o2)cn1` (make) + `ClCC1CC1` (buy) → `c1cc(NCc2ccc3oc(-c4ccc(-c5nnn(CC6CC6)n5)nc4)nc3c2)c2[nH]ccc2c1`

**Molecule B (sEH 7.656)**

1. **Benzoxazole formation (2-aminophenol + acid/aldehyde) [t181]** — `N#Cc1ccc(C(=O)O)cn1` (buy) + `Nc1ccc(C(=O)O)cc1O` (buy) → `N#Cc1ccc(-c2nc3ccc(C(=O)O)cc3o2)cn1`
2. **Amide coupling (acid + amine) [t136]** — `N#Cc1ccc(-c2nc3ccc(C(=O)O)cc3o2)cn1` (make) + `Nc1cccc2cc[nH]c12` (buy) → `N#Cc1ccc(-c2nc3ccc(CNc4cccc5cc[nH]c45)cc3o2)cn1`
3. **Tetrazole formation + N-alkylation (nitrile + hub)** — `N#Cc1ccc(-c2nc3ccc(CNc4cccc5cc[nH]c45)cc3o2)cn1` (make) + `ClCC1CC1` (buy) → `c1cc(NCc2ccc3nc(-c4ccc(-c5nnn(CC6CC6)n5)nc4)oc3c2)c2[nH]ccc2c1`

## Cutoff 0.85 — closest pair (Tanimoto 0.8438)
Both molecules are built from the **same hub** (`ClCC1CC1`) and share **1** fully-built intermediate step(s) (made once), then diverge at the final decoration.

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `N#Cc1ccc(CBr)cc1` | A/B | shared |
| `NCc1c[nH]c2ccccc12` | A/B | shared |
| `O=C(O)C1CCNCC1` | B | diverging |
| `O=C(O)C1CNC1` | A | diverging |

**Shared steps (built once)**

1. **N-alkylation (ring N–H + alkyl halide) [t202]** — `NCc1c[nH]c2ccccc12` (buy) + `N#Cc1ccc(CBr)cc1` (buy) → `N#Cc1ccc(Cn2cc(CN)c3ccccc32)cc1`

**Molecule A (sEH 7.909) — continues from the shared intermediate**

2. **Amide coupling (acid + amine) [t77]** — `N#Cc1ccc(Cn2cc(CN)c3ccccc32)cc1` (make) + `O=C(O)C1CNC1` (buy) → `N#Cc1ccc(Cn2cc(CNC(=O)C3CNC3)c3ccccc32)cc1`
3. **Tetrazole formation + N-alkylation (nitrile + hub)** — `N#Cc1ccc(Cn2cc(CNC(=O)C3CNC3)c3ccccc32)cc1` (make) + `ClCC1CC1` (buy) → `O=C(NCc1cn(Cc2ccc(-c3nnnn3CC3CC3)cc2)c2ccccc12)C1CNC1`

**Molecule B (sEH 7.602) — continues from the shared intermediate**

2. **Amide coupling (acid + amine) [t76]** — `O=C(O)C1CCNCC1` (buy) + `N#Cc1ccc(Cn2cc(CN)c3ccccc32)cc1` (make) → `N#Cc1ccc(Cn2cc(CNC(=O)C3CCNCC3)c3ccccc32)cc1`
3. **Tetrazole formation + N-alkylation (nitrile + hub)** — `N#Cc1ccc(Cn2cc(CNC(=O)C3CCNCC3)c3ccccc32)cc1` (make) + `ClCC1CC1` (buy) → `O=C(NCc1cn(Cc2ccc(-c3nnnn3CC3CC3)cc2)c2ccccc12)C1CCNCC1`

## Cutoff 0.90 — closest pair (Tanimoto 0.8837)
Both molecules are built from the **same hub** (`ClCC1CC1`) and share **1** fully-built intermediate step(s) (made once), then diverge at the final decoration.

**Building blocks to buy**

| SMILES | for | note |
|---|---|---|
| `ClCC1CC1` | A/B | shared |
| `NCc1c[nH]c2ccccc12` | A/B | shared |
| `Nc1cc(Br)ccc1O` | A/B | shared |
| `O=C(O)C1CCN1` | A | diverging |
| `O=C(O)[C@@H]1CCCN1` | B | diverging |

**Shared steps (built once)**

1. **Buchwald–Hartwig amination (aryl halide + amine) [t113]** — `Nc1cc(Br)ccc1O` (buy) + `NCc1c[nH]c2ccccc12` (buy) → `Nc1cc(NCc2c[nH]c3ccccc23)ccc1O`

**Molecule A (sEH 8.137) — continues from the shared intermediate**

2. **Benzoxazole formation (2-aminophenol + acid/aldehyde) [t181]** — `O=C(O)C1CCN1` (buy) + `Nc1cc(NCc2c[nH]c3ccccc23)ccc1O` (make) → `c1ccc2c(CNc3ccc4oc(C5CCN5)nc4c3)c[nH]c2c1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `c1ccc2c(CNc3ccc4oc(C5CCN5)nc4c3)c[nH]c2c1` (make) + `ClCC1CC1` (buy) → `c1ccc2c(c1)c(CNc1ccc3oc(C4CCN4)nc3c1)cn2CC1CC1`

**Molecule B (sEH 7.937) — continues from the shared intermediate**

2. **Benzoxazole formation (2-aminophenol + acid/aldehyde) [t180]** — `Nc1cc(NCc2c[nH]c3ccccc23)ccc1O` (make) + `O=C(O)[C@@H]1CCCN1` (buy) → `c1ccc2c(CNc3ccc4oc([C@@H]5CCCN5)nc4c3)c[nH]c2c1`
3. **N-alkylation (ring N–H + hub alkyl halide)** — `c1ccc2c(CNc3ccc4oc([C@@H]5CCCN5)nc4c3)c[nH]c2c1` (make) + `ClCC1CC1` (buy) → `c1ccc2c(c1)c(CNc1ccc3oc(C4CCCN4)nc3c1)cn2CC1CC1`
