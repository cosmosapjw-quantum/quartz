# R04 — candidate constructions and discriminating observables

Exploration considered three explicit laws on m categorical utilities.

1. Full joint table P(U_1,...,U_m) has up to3^m cells. It is an independent exact oracle for small fixtures, but estimating a root-varying table from90 training roots is not supported.
2. An unconstrained two-component mixture sum_z w_z product p_{az} adds separate component probability fits. It confounds changed marginal fits with dependence and increases data demands.
3. A balanced latent tilt p_a^z=p_a(1+z rho f_a), f centered/scaled as R03, preserves all original cue marginals, needs one nonnegative coupling, and has an exact scalar posterior. We select this family for C27. Its nonnegative covariance restriction is an empirical modeling risk.

The product law rho=0 is nested exactly. Initial conditional productization is retained as a secondary control. We will not silently swap to a larger latent family after CAL/TEST.

## A representative calculation already executed

Use balanced A in{0,1/2}, B in{0,1}, incumbent b=1/2, rho=4/5, query costs d_A=1/100 and d_B=3/10. A cannot directly improve the incumbent. Its high observation raises P(B=1) to(1+rho^2)/2=.82. Then opening B has gain .5*.82-.30=.11; the low A branch stops. Therefore opening A first gives .5-.01+.5*.11=.545=109/200. Product marginals give no positive query and value.5.

An independent Wolfram symbolic evaluation returned query-A value (4-8c-4epsilon+rho^2)/8 and improvement (rho^2-4epsilon-8c)/8 in the high-branch-active regime, plus109/200 for this fixture. Full proof and exhaustive rational checks are R05–R07, not inferred from one CAS expression.

## Physics and cognition transported as mathematics

For exact observations define ell_h=sum_s atanh(rho f_{a_s}(u_s)). The latent posterior is P(Z=z|h)=exp(z ell_h)/(2 cosh ell_h), so m_h=tanh ell_h. This is a finite two-state exponential-family partition function, logZ=log(2 cosh ell), with m=d logZ/dell and susceptibility dm/dell=1-m^2. ell and logZ are dimensionless. It provides stable computational coordinates; it is not evidence of a physical temperature or thermodynamic phase transition.

The original variational identity E_q[F]-beta^-1 KL(q||pi) has an exact Gibbs optimizer on computation actions. C27 does not need to add a tunable softmax action policy to implement the latent posterior. Deterministic Bellman selection with a STOP action is the finite controller under study.

The cognitive transfer is conditional information value: an observation can redirect the next computation even when it cannot improve the current known action. H1 cannot use pure cross-candidate information under the known-only terminal rule; H2 can. Human skill/signature claims remain untested.

## Empirical observables chosen before outcome access

Primary: paired JOINT_H2 minus PRODUCT_H2 net utility including measured full native runtime, on a frozen new source cohort. Secondary: comparison with INITIAL_PROJECTED_H2, root-level realized value, exact nodes, planner/full timing, selected-query predictive error and conditional candidate log score/Brier score.

Diagnostic controls: INITIAL_STOP and RANDOM_TWO. H1 is a Bellman subroutine and mathematical diagnostic, not a separately tuned empirical primary. Query limit is2 after common initialization, no repeated horizon resets. One primary comparison prevents post-hoc baseline selection.

TRAIN-only tuning: a predeclared rho grid and root-level crossvalidation with inside-fold marginal refitting; final base marginals remain the exact C26 full-TRAIN model. Timing-to-utility conversion is independently measured on TRAIN before CAL/TEST. Final details are frozen in R09.
