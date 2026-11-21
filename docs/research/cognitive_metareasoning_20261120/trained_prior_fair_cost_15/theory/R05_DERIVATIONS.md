# R5. Exact learning gradients, search support and CPU-budget decomposition

## Lemma A: legal-action maximum likelihood

Fix finite nonempty legal set A(s), explicit feature phi_a∈R^d and theta. Define Z(theta)=sum_{a∈A}exp(theta·phi_a), p_a=exp(theta·phi_a)/Z, and label y∈A. One-source cross-entropy ell(theta;s,y)=log Z(theta)-theta·phi_y. Differentiation gives grad ell=sum_a p_a phi_a - phi_y and Hessian=sum_a p_a (phi_a-mu)(phi_a-mu)^T, mu=sum_a p_a phi_a. Thus Hessian is positive semidefinite for all theta, and empirical loss plus lambda||theta||²/2 is convex (strict if lambda>0). This is a well-defined supervised inference objective, not a reinforcement-learning solution or proof y is optimal. A learned policy-only scorer uses value=0, so any benefit in all-legal runs is from move priors and consequential changed trajectories, not a learned value estimate.

To avoid overflows compute z'=z-max_A z; softmax p_a=exp(z'_a)/sum exp(z'_b). Illegal actions receive exactly zero and never enter normalization. For finite theta and bounded features ||phi||∞<=1, |z_a|<=||theta||1=S and p_a>=exp(-2S)/|A|>0 for legal a. However root candidate masking makes p=0 *for excluded actions at selection*, overriding support; this precisely distinguishes attention from model.

## Lemma B: root-reopening discovery and lost guarantee

For B legal actions, k initially admitted, m decisive omitted, and r uniformly sampled distinct candidates from B-k, exact recovery probability is 1 - choose(B-k-m,r)/choose(B-k,r). For B=225,k=16,m=1,r=4 this is 4/209; for a deterministic learned ranking, discovery is the indicator 1{rank_theta(y)≤k} before reopening. This rank coverage can be estimated on independent teacher labels but is NOT a proof of winning probability. For weighted with-replacement draws of critical candidate mass q, probability at least one hit is 1-(1-q)^r; it does not equal without-replacement weighted PPS without additional sampling assumptions.

## Lemma C: equal visits do not imply equal CPU

Write observed wall time tau_i=setup_i+prior_i+probe_i+sum_{j=1}^{N_i} t_{ij}+mask_i. At a fixed N, unequal evaluator costs t_{ij} or outcome-dependent stop can produce arbitrarily different tau. Therefore the primary comparison must fix a physical target time tau_target for each arm (with predeclared operational overshoot tolerance), measure full online work by a monotonic clock, and separately audit achieved mean tau by stratum. The C14 pooled cancellation counterexample directly demonstrates why a pooled time match alone is invalid.

No theorem here guarantees improved policy accuracy, causal identification across game families, Elo, or a cognitive mechanism. R7 evaluates numerical support and finite differences.