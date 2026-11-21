# R5. Constructive finite Bayesian computation-control derivation

Assumptions: finite world Θ, terminal actions A and experiment channels C. Posterior b is normalized and faithful to the observation likelihood conditional on latent θ; utility U finite; costs κ_c>0 deterministic and priced by λ>0; experiment observations conditionally independent given θ when experiments are repeated. All optimality statements below are relative to this finite model, not adaptive neural-MCTS.

**Lemma 1: posterior simple regret is nonnegative.** Let Q_θ=max_a U(a,θ), M(b)=max_a Σθ bθ U(a,θ), K(b)=Σθ bθ Qθ. Then R(b)=K(b)-M(b). Since max and expectation satisfy E[max_a U(a,θ)]≥max_a E[U(a,θ)], R(b)≥0. This is a *decision gap*, not epistemic entropy and not realized regret of an arbitrarily selected action.

**Lemma 2: expected value of a calibrated observation is nonnegative.** For computation c, define q(y)=Σθ bθ L_c(y|θ) and b_y(θ)=bθ L_c(y|θ)/q(y) if q(y)>0. Posterior calibration gives Σ_y q(y)b_y=b. Thus
G_c(b)=R(b)-Σ_y q(y)R(b_y) = Σ_y q(y)M(b_y)-M(b).
The K terms cancel because Σ_yq(y)K(b_y)=K(b). Since M(b)=max_a b·U_a is convex in b, Jensen implies G_c≥0. Equality occurs whenever the same terminal action achieves M(b_y) for every positive-probability observation outcome (ties allowed); data can reduce entropy while G=0 if no decision policy is improved. This is a statement about a *faithful Bayesian* channel and appropriate terminal utility, not arbitrary stochastic scores.

**Lemma 3: one-step dispatch and stop semantics.** With a single affordable opportunity and utility/resource exchange λ, the net gain of c is N_c=G_c−λκ_c. Choose STOP iff max_c N_c≤0 (zero ties stop by convention); otherwise choose argmax N. The ratio index I_c=G_c/κ_c satisfies I_c>λ iff N_c>0 because κ_c>0. Thus the **sign of the one-step stopping predicate** is unchanged by ratio representation, but the identity argmax_c I_c generally differs from argmax_c N_c (R4 exact counterexample). Ratios do not by themselves constitute the same finite one-step *selection policy*.

**Proposition 4: finite-horizon sequential Bellman equation.** Let V_0(b)=R(b). For integer h≥1, define
V_h(b)=min{R(b), min_c[λκ_c+Σ_{y:q(y)>0}q(y)V_{h−1}(b_y)]}.
Proof: stop immediately incurs posterior Bayes regret R(b). Continuing with c incurs priced compute λκ_c and then optimally the expected residual Bayes regret/cost with at most h−1 further experiments. By backward induction on h this recurrence equals the optimum value for all policies using ≤h computations. It follows V_h(b)≤R(b) and V_{h+1}(b)≤V_h(b), since additional computations are optional, not mandatory. This DOES NOT imply one-step myopic choices are globally optimal at h>1.

**Corollary: garbling monotonicity of gross EVSI.** If observation Z is a stochastic garbling of Y conditional only on Y, posterior b_Z is the average of b_Y under Z. Convexity of M yields E[M(b_Y)]≥E[M(b_Z)], hence G_Y≥G_Z≥0, without costs. The *net* priority may reverse when Y costs more than Z. Domain and action relabelling leave R,G,V invariant when payoff/likelihood rows and labels are permuted consistently.

**Bound on posterior perturbation:** If |U(a,θ)|≤M and distributions b,b' have L1 distance D, then |K(b)-K(b')|≤M D and |M(b)-M(b')|≤M D, so |R(b)-R(b')|≤2M D. This is a finite-model continuity bound, not a statistical calibration guarantee.

**Failure to transport without extra assumptions:** neural MCTS leaf statistics generally violate the simple repeated conditional-IID observation channel; evaluator drift, correlated backups, candidate omissions and new tree structures alter the belief state and likelihood. The theorem still supplies a reference **semantics**, but an adaptive neural MCTS agent needs a separately calibrated belief/observation model before claiming normative or empirical optimality.