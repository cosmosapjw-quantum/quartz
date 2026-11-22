# R5. Direct derivation: what a calibrated computation-observation model can and cannot prove

Assume terminal utility U(a,theta) bounded in [u_min,u_max] for all admissible actions a and actual states theta. Set D=u_max-u_min. Fix a pre-computation action a0, an action a1 selected after c, and revision event Y=1[a1!=a0]. For pathwise signed change Z=U(a1,theta)-U(a0,theta):
- Y=0 implies Z=0 exactly (if physical action truly identical and utility evaluated at same theta).
- |Z|<=D Y pathwise, hence E[|Z||X]<=D q(X) and |E[Z|X]|<=D q(X). A correct calibrated upper confidence bound on q may bound absolute expected terminal utility change, but q alone does not determine its SIGN.
- E[Z|X]=q(X) E[Z|Y=1,X] if conditional event has positive mass. Two models may have identical q(X) and opposite utility impact; without a utility-aware conditional effect model, positive computation value is NOT_IDENTIFIED.

Counterexample: prior action0, prospective action1. With probability q=1/4 action changes, otherwise not. In world A action1 has utility +1 relative to a0 at changed outcomes; world B action1 has utility -1. Same q=1/4 but E Z=+1/4 vs -1/4. A flip-only forecast cannot conclude expected gain.

Proper Brier score: given initial-only features X, true p=P(Y=1|X) and forecast qhat in[0,1], E[(Y-qhat)^2|X]=p(1-p)+(qhat-p)^2. Proof: substitute Y in {0,1}, expand, complete square. Minimum score E[p(1-p)] irreducible. This conditional-expectation identity does not require independent observations; statistical confidence for finite dataset does.

Grouped dependence: if m exchangeable predictions per independent parent have variance sigma^2 and pairwise intracluster correlation rho, Var(grand mean)=sigma^2/[n*m]*[1+(m-1)rho]. Proof by sum of n groups with m diagonal and m(m-1) covariance terms. Only homogeneous exchangeable clusters; descriptive sensitivity, not a guarantee for adaptive MCTS. Resample parent groups, not individual calls.

Safe STOP: if utility-aware signed improvement unknown, revision q alone is insufficient to declare safe beneficial STOP or positive net EVSI. With known utility range D, upper bound D q on gross potential change may support a conservative shadow rule, but it is not a lower bound of gain, and calibration uncertainty must be included. For Gomoku15 true utility range with respect to optimal play is not established by the instrumented source; output is scoped predictive revision forecast, not utility guarantee.

The finite Bayesian C17 math remains authoritative under its own assumptions. This is a genuine mathematical interpretation of what C18 can estimate, not a novel metareasoning theorem or empirical P2/P3 success.