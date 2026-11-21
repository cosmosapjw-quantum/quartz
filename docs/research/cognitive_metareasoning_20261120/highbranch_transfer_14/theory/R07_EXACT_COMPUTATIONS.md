# R7. Exact symbolic crosscheck and auditable finite computation

Wolfram Language exact computation executed on 2026-11-21:
- B7,k3,r2,m1: 1/2.
- B225,k16,r4,m1: 4/209≈0.019138755980861244.
- B225,k16,r4,m5: 2393375/25746292≈0.092959988180045499.
- PUCT under c2,p1/100, N32 gives 0.11313708498984760 < 1/5.
- Number of all subsets of 225 possible cells 2^225, much too large to enumerate.

Pure Python source theory/combinatorial_reopening14.py independently recomputes the rational results using math.comb and fractions.Fraction, validates a complete seven-action small-domain hypergeometric count by combinations, and checks all singleton action indices 0..224 plus boundary masks across the four u64 words. It writes finite-check JSON to accompany R7 commit. This validates math-only finite statements, not Rust behavior or measured game strength.

The exact disparity 0.5 (small 7-action case) versus ~0.0191 (one omitted decisive in 225-action case) gives a genuine structural prediction: for a reliable tactical witness, fixed-size targeted opening may be much more efficient than uninformed candidate release in rare-target high-branch settings. This still does NOT imply superiority over an all-legal MCTS baseline or a stronger independent Rapfi reference.
