# R4. Competing constructive representations and experiment

**Four technical constructions compared (divergent investigation, then selected):**
1. **Full legal edge allocation + opt-in monotone root selection mask**: materialize all ≤7 legal moves at init using existing arena, keep candidate array/prior and TT unmodified; `select_core` skips disabled root edges. At eight visits, enable tactical subset and keep old counters. *Chosen* because safe and source-local. Weakness: no root edge-allocation saving and extra mask branch.
2. Replace OnceLock candidate slab and relocate arena edges at checkpoint: potentially reduces upfront edge materialization but pointer publication + lock-free readers make it **high risk** under existing memory-contract invariants. Not chosen.
3. Use a `PwConfig` with k=3 and switch to full after visit8: enables some root widening but **also prunes descendants**, changing game-search semantics and introducing depth confounding. Not chosen.
4. COG03-12 `Connect4::with_root_candidates` and new search tree: valid prior baseline, but **does not preserve 8 root visits or Q-state**. Kept as previous control only.

**Chosen active mask:** `M_t = Σ_{a∈A_t}2^{idx(a)}` (u64; game has 7 columns), sentinel 0 in implementation means *unrestricted*, not empty mask. Mask change at serial checkpoint, never while worker threads search. No game-state modification, legal columns and edge priors remain those from full prior. Root child TT key is never recomputed.

**Selection:** `a_t = argmax_{a∈A_t} [Qhat_a + c_PUCT P(a|s) sqrt(N_root)/(1+N_a)]`, using same original `P` regardless of active-set mass; no per-mask prior renormalization. The restricted prior masses do not sum to one, and this is an explicit algorithmic convention. Consequently the active-set PUCT may differ from rerunning a masked-evaluator full-normalized model; do not claim byte-identical COG03-12 behavior.

**Observables:** source root node hash, address/identity, per-action visit/value records at visit8 and immediately after activation, new action visit N=0 at activation, no new NN root forward, total24 root visits, per-iteration selected action legality, per-policy end-to-end µs, number of actually activated/selected decisive exact-WDL moves, posthoc final exact regret. All controls include a *sham* one-ply tactical probe.

**Pre-registered main comparison:** resident root top3 first8 + new tactical actions until24 vs unrestricted all-legal root 24 visits from outset, with per-group mean wall CPU mismatch ≤10%; also resident-closed, same-count oracle-blind random activate three fixed seeds, resident all-open at8. Primary game utility is exact WDL rather than policy KL. Stop if no meaningful resident advantage; high branching-factor Gomoku/Renju is a separate scope.
