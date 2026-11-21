# C3. Numerical and computational contract, frozen before code outcomes

All probabilities and likelihood rows are finite in [0,1] with absolute normalization error <=1e-10. Empty action/state/observation dimensions, mismatched matrices, non-finite utilities/cost, negative/zero priced compute cost, invalid λ and invalid belief should return explicit errors, not silently normalized to another scientific model. Positive-probability observations use Bayes update; zero-mass outcomes are omitted from expectation and cannot cause division by zero.

Numerical decision comparison: f64 with 1e-10 absolute criterion for exact rational example and robust comparative tests, STOP on nonpositive net gain within a 1e-12 comparison allowance; this allowance is for floating rounding near an exact tie, not post-hoc scientific threshold fitting. H limited to <=4 in the initial reference implementation because brute-force belief-tree enumeration scales exponentially in observations. The finite R5 theorem applies to any finite H but implementation is deliberately capped. Caller explicitly supplies λ in utility/resource, no magic default game frame.

Required tests:
- Correct regret, EVSI, ratio and net for 3-state two-computation exact R4.
- Positive one-step stopping equivalence for κ>0 but divergent dispatch.
- Horizon values H0..H3, skip impossible observations.
- Relabelling terminal actions/states and observation outcomes does not change Bayes utility, expected regret or optimal H value.
- Uninformative-but-entropy-bearing observation can have gross VOC0.
- Full/grooved partial/no-information channel obeys information monotonicity before cost.
- Costs rescaled by factor1000 with λ inversely rescaled preserve policy and objective.
- At least one additional non-identity signed/reward matrix and a zero-value two-action state confirm code not tailored to one chess/Gomoku example.
- Unit tests in Python and Rust, exact Fraction standalone check, no production MCTS mutation, full existing Rust regression before claiming cross-domain computational substrate verified.

C8 descriptive finite simulation objective: charged realized cumulative κ plus realized *true-state* decision simple regret of final selected action. Compare STOP, per-step maximal ratio, per-step maximal net and H-horizon Bellman on frozen hypothetical-state tasks, exact prior/channels and common seeded pseudo-random streams. Empirical mean ± MC standard error descriptive, absolute comparison against exact analytical objective where available. Do not call the toy model a live neural-MCTS demonstration.