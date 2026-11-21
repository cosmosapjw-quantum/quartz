# R7. Reproducible finite exact counterexample and Wolfram/Python check

For fixed original root `P=(1/2,1/3,1/6)`, MCTS state at n_root=8 with `N=(8,0,0)`, `Q=(2/5,0,0)`, c=2, unnormalized fixed-root-prior PUCT computes:

```text
score(0) = 2/5 + 2sqrt(2)/9,
score(1) = 4sqrt(2)/3,
score(2) = 2sqrt(2)/3.
score(1)-score(0) = 10sqrt(2)/9 - 2/5 > 0.
Σ full P = 1; Σ first-two P = 5/6; Σ N = N_root = 8.
```

**Exact Wolfram Language calculation executed on 2026-11-21**, in the Wolfram tool: `{{2/5 + (2*Sqrt[2])/9, (4*Sqrt[2])/3, (2*Sqrt[2])/3},{5/6,1},True,{8,8}}`. Independent local `finite_counterexample13.py` uses exact `fractions.Fraction` and 70-digit Decimal square root, saves `FINITE_COUNTEREXAMPLE13.json`. Neither computation accesses actual hidden WDL labels, and neither is a simulation of the full engine.

The calculation proves a **finite scoring rank switch** after a candidate is enabled; it does not prove that the candidate is actually a worse legal Connect Four move for a specific board. A demonstration with real Connect Four boards requires R9/C7 oracle-audited search. General mathematical result: no deterministic per-step regret monotonicity from counter preservation alone.

R7 status `derived + independently symbolically checked`. The exact program output SHA is recorded by R7 publication.
