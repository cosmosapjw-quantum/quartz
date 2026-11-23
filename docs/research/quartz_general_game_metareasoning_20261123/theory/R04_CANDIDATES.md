# R04 — candidate constructions and a discriminating example

The original research links utility, uncertainty, cost, and bounded deliberation. Compare actual expressions:

1. Legacy local surrogate: p_flip*estimated_gain minus time cost; useful when its uncertainty/observation model is explicit. A ranking-error probability alone is not the probability that a particular computation changes the recommendation.
2. C25 interval certificate: known lower bound retention and residual worst-case gap. It supplies safety and a maximum possible gain, but the gap is not an expected attainable gain.
3. Exact-query recognition model: g_a(b)=E[(U_a-b)_+], calibrated from game-agnostic cues; net Q_a=g_a-lambda*c_a. Exact STOP and reservation-value ordering can be proved under additional independent fixed-cost assumptions.
4. KL-regularized computation choice: maximize q.Q-tau*KL(q||pi); sample the resulting Gibbs law. This directly implements the original variational motif on metacomputations. It optimizes a regularized local objective and need not maximize raw net utility.
5. Cumulant-informed terminal free energy: expected free-energy correction contains contrast covariance; it is not automatically the hard-max query value in (3). Retain it as a measured theory diagnostic until the terminal objective changes explicitly.
6. Joint-belief information-only queries: finite joint-posterior Bellman optimization can account for revealing another candidate's value. This provides the constructive follow-up when the product-prior approximation fails.

## First exact example

Initial b=0. Candidate A returns 1/2 surely, costs 1/10; candidate B returns 1 with probability 2/5 and 0 otherwise, costs 1/20. Greedy one-step nets are 2/5 and 7/20, so A is first. The complete greedy strategy has net 11/20=0.55. Reservation indices are z_A=2/5 and z_B=7/8. Opening B first, followed by A only after B=0, gives 2/5+(3/5)(1/2)-1/20-(3/5)(1/10)=59/100=0.59. Both retain the best observed utility. If only one query is allowed, A instead wins (0.4 versus 0.35). This separates safety, stopping, ordering and budget assumptions.

## Selection

Implement (3) and (4) alongside common-initial STOP, ALL_ACTIONS and RANDOM_ORDER controls. Keep GREEDY_VOC and PANDORA separate. Derive (5) and a corrected reversible Markov realization, then verify them with small exact/numeric calculations. Preserve (6) as a concrete next-cycle extension with its already constructed correlation counterexample. A single cycle will not inject all legacy motifs as unvalidated terms into an existing controller.

The mechanism selection preserves actual source motives: statistical mechanics supplies a variational choice law and controlled fluctuation calculus; cognitive resource rationality supplies recognition–verification and paid computation; economic sequential search supplies a provable ordering rule. Information bottleneck, RG flow, critical transitions and a full Friston generative model require additional concrete state variables/objectives and are not renamed into existence.
