# Diagram molecules

Schematic molecules for the method diagram. **Illustrative, not run outputs** — they show
the *shape* of hub-batching, not anything the pipeline produced. Regenerate with
`python render_molecules.py` (needs the `rgfn` env for rdkit).

## The set

One **origin** → two **hubs** → five **candidates**.

| file | what it is | atoms | derived from |
|---|---|---|---|
| `00_origin` | aniline | 7 | — |
| `01_hubA` | acetanilide — **stick** acyl | 10 | origin |
| `02_hubB` | cyclopropyl anilide — **triangle** acyl | 12 | origin |
| `A1` | + methyl | 11 | hub A |
| `A2` | + ethyl | 12 | hub A |
| `A3` | + cyclopropyl | 13 | hub A |
| `B1` | + chloro | 13 | hub B |
| `B2` | + morpholino | 18 | hub B |

Both hubs contain the *whole* aniline origin, so they read as siblings; they differ only in
the shape of the acyl group — a stick against a triangle — which is what makes the two
families distinguishable at a glance.

7–18 heavy atoms each (96 total), down from 9–21 (136 total) in the first version.

## The similarity story, and why it is real

The molecules are made up; **the similarity geometry is not.** Measured with the paper's own
metric (Morgan r=3, 2048 bits), against its own $\tau = 0.50$:

| pair | Tanimoto | |
|---|---|---|
| **A1 – A2** | **0.514** | **above τ → A2 is the duplicate the greedy rejects** |
| A1 – A3 | 0.474 | under τ → kept |
| A2 – A3 | 0.439 | under τ → kept |
| B1 – B2 | 0.471 | kept |
| all cross-family | 0.18–0.29 | different hubs, no collision |

A1 and A2 differ by **one bond** — methyl against ethyl — which is the point. At diagram
scale they read as the same molecule, which is what makes "this one gets thrown away"
legible without a caption. A3 puts a triangle on the ring, so it is obviously different.

**A caution if you want to print the number.** A1–A2 at 0.514 clears τ=0.50 by very
little, and A1–A3 at 0.474 misses it by about as much. That is honest — small molecules
share most of their fingerprint bits, so a schematic at this size cannot also have wide
margins. Label the *relation* ("too similar — rejected") rather than the number, unless you
are happy showing 0.51 against 0.50.

`render_molecules.py` asserts every row of that table on each run. If someone edits a SMILES
and breaks the story, it fails instead of shipping a diagram that contradicts the text.

## Which files to use

| dir | use for |
|---|---|
| `svg/` | **first choice** if Canva accepts SVG — vector, scales to any size |
| `png/` | ~2000 px wide, transparent, heavy strokes |
| `overview.png` | all eight in one sheet, for checking the layout before assembly |

All PNGs are auto-cropped to the ink, so they drop in without manual trimming, and
transparent so they sit on any background. Bond strokes are deliberately heavy
(`bondLineWidth = width/135`) so they hold up when scaled down in a slide.

Note `overview.png` is drawn with RDKit's grid defaults — thin lines, for layout checking
only. The files in `png/` and `svg/` are the ones to use.

## Suggested layout

```
                    ┌── A1  ●────────────┐
                    │                     │  family A
   origin ── hub A ─┼── A2  ● (greyed)    │  2 kept, 1 rejected
        │           │                     │
        │           └── A3  ●────────────┘
        │
        └───── hub B ─┬── B1  ●──────────┐  family B
                      └── B2  ●──────────┘  2 kept
```

Two things worth showing explicitly, because they are the argument:

1. **The hub is drawn once per family.** That is the cost being amortized — the reader
   should see one parent serving several children.
2. **A2 greyed out or struck through**, with the reason on the edge (`Tanimoto 0.68 > 0.50`).
   The rejection is the mechanism, not an afterthought: it is why hub-batching pays to move
   to a new hub instead of taking another cheap analogue.
