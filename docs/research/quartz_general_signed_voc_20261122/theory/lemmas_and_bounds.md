# R5 | Direct derivation and interval-valued decision rule

**1. Exact two-part identity.** Since Y=0 iff a1=a0, Z=0 on Y=0 for identical true theta and action utility. Hence E[Z|X,c]=P(Y=1|X,c) E[Z|Y=1,X,c] with the q=0 case assigned g=0 without defining conditional m on a null event. This is a probabilistic identity, not evidence for a good predictive model.

**2. Sign and robust interval rule.** If q∈[qL,qU]⊂[0,1], m∈[mL,mU]⊂[-1,1], and expected positive cost c∈[cL,cU], then the feasible product range is obtained by all four endpoint products because qm is bilinear and extrema over a rectangle occur at corners:
gL=min(qL mL,qL mU,qU mL,qU mU), gU=max(...).
With price lambda>0, nL=gL−lambda cU and nU=gU−lambda cL. Therefore nL>0 is sufficient for compute optimality relative to STOP in the **one-action one-step** decision, and nU<0 sufficient for STOP, under valid simultaneous input coverage and correctly modelled conditional means. Otherwise ABSTAIN from strong claims, not a forced 'safe STOP'. These are mathematical conditional implications; arbitrary uncalibrated intervals do not give coverage.

**3. Two-predictor error bound.** For q,qhat∈[0,1], m,mhat∈[-1,1], |qm−qhat mhat|≤|q−qhat|·|m|+|m−mhat|·qhat≤|q−qhat|+|m−mhat|. If forecast cost discrepancy |c−chat|≤εc, then net-error ≤εq+εm+lambda εc. Consequently, a genuine bound needs probability calibration, signed effect calibration and cost prediction jointly, not just q Brier.

**4. Bayesian EVSI consistency.** If a0 and a1 maximize posterior expected U under the correct prior/likelihood for the same task and future observation conditionally follows it, then E[Z|H0,c]=E_y[max_a E(U(a,theta)|H0,y)]−max_a E(U(a,theta)|H0)≥0 by convexity of maximum. A negative individual Z path does NOT contradict this. With priced compute the expected net gain can nevertheless be <0. A learned q/m surrogate may output negative gain; no sign guarantee transfers from exact Bayes to a misspecified fit.

**5. Frozen hypothesis and falsifier.** A feature-based signed two-part model may rank some computation opportunities better than an event-only q heuristic on heldout groups, but the model's empirical performance and statistical uncertainty are unresolved. The admitted mathematical output is a clear conditional decision-value estimate and identifiable failure modes, not a new universal search improvement theorem.