# stage 4 — selection (COMPETITORS ONLY)

SPARROW's MILP under a hard reaction budget, plus the diversity-aware greedy arm (the baseline's
strongest configuration).

**Check `time_capped` on every SB row before reading it as an optimum.** CBC reports `Optimal` for
whatever it holds when a time limit stops it, so status alone is not enough — detect by wall-clock. A
capped row is a LOWER BOUND on the competitor, i.e. it flatters us, and cannot carry a ratio.
