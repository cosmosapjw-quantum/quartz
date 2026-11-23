# C26 R07: constructive Gibbs relaxation and computational fluctuation checks

Status: **72 assertions passed** in one bounded Python execution. Exact algebra uses SymPy rationals. KL values on rational trajectories use 70-digit log evaluation; the separate contraction proof below supplies the universal statement. Numeric Gibbs checks use fixed tolerances; no project module, game input, production binary or campaign was run.

## 1. Units, definitions and target

There are m=3 states. pi_a>0 is a normalized prior, Q_a is utility, and tau>0 has the same utility units. Define q*_a=pi_a exp(Q_a/tau)/Z. KL is dimensionless and measured in nats. kappa has inverse algorithmic-time units; delta has algorithmic-time units. This is a mathematical stochastic construction. No identification with physical temperature, k_B, hbar, or actual MCTS dynamics is assumed. In the Lindblad construction H=0, so no natural-unit convention is needed.

The variational objective J(p)=p.Q-tau KL(p||pi) obeys exactly

    tau log Z - J(p) = tau KL(p||q*) >= 0.

This follows by substituting log(q*_a/pi_a)=Q_a/tau-log Z. The target is therefore the unique variational maximizer on the positive prior support. The rational target q*=(1/6,1/3,1/2), uniform pi and tau=1/5 is realized by Q_a=tau log(q*_a/pi_a), with Z=1. Both its exact symbolic identity and the requested numeric Q=(0,.2,-.1), tau=.1 identity were checked.

## 2. Constructive population dynamics

For a!=b, set W_ab=kappa/(m-1) min(1,q*_a/q*_b), where column b is the departure state. Set W_bb=-sum_(a!=b) W_ab. Then

    W_ab q*_b = kappa/(m-1) min(q*_a,q*_b) = W_ba q*_a.

Detailed balance plus zero column sums yields W q*=0. All offdiagonal rates are positive; the finite chain is irreducible. Its exit rates are <=kappa. For 0<=delta<=1/kappa, P=I+delta W is column stochastic and nonnegative; Pq*=q*. With delta strictly below 1/kappa it has positive diagonal and all entries are positive. P is an exact selected discrete-time transition, not the continuous semigroup exp(delta W).

Exact W at kappa=1:

    [['-1', '1/4', '1/6'], ['1/2', '-3/4', '1/3'], ['1/2', '1/2', '-1/2']]

Exact P at delta=1/2:

    [['1/2', '1/8', '1/12'], ['1/4', '5/8', '1/6'], ['1/4', '1/4', '3/4']]

For any distribution p, the log-sum inequality applied to each output row gives D(Pp||Pq*)<=D(p||q*); stationarity makes this D(Pp||q*)<=D(p||q*). For the strictly positive P and p!=q*, equality is excluded by the equality condition of log-sum. The supplied nonstationary rational trajectory satisfied strict contraction for all 12 inspected transitions. The numeric trajectory satisfied contraction over 80 transitions.

## 3. Lindblad correction and finite CPTP realization

Set L_ab=sqrt(W_ab)|a><b| for a!=b and use d rho/dt=sum_ab [L_ab rho L_ab^dagger-{L_ab^dagger L_ab,rho}/2], with H=0. For arbitrary matrix rho its populations obey

    d rho_aa/dt = sum_(b!=a) W_ab rho_bb - sum_(b!=a) W_ba rho_aa = (W p)_a.

For a!=b its coherences satisfy d rho_ab/dt=-(exit_a+exit_b)rho_ab/2. Thus diag(q*) is stationary. This equality was checked on symbolic entries of a general 3x3 matrix, not just on a special diagonal density matrix.

In contrast, diagonal jumps L_a=sqrt(gamma_a)|a><a| give d rho_aa/dt=0 and d rho_ab/dt=-(gamma_a+gamma_b)rho_ab/2. They only dephase. They cannot supply the population-balance/Gibbs derivation claimed for such jumps in the legacy notes.

A finite completely-positive trace-preserving channel with exactly the selected diagonal transition P is provided by K_ab=sqrt(delta W_ab)|a><b| and K_0=diag(sqrt(1-delta exit_b)). Directly sum K_j^dagger K_j=I, and the diagonal of sum K_j rho K_j^dagger is Pp. The check does NOT claim the naive full-superoperator Euler map I+delta L is CPTP, or that this Kraus channel equals exp(delta L).

## 4. Susceptibility and computation-specific covariance

Let F_tau(mu)=tau log sum_a pi_a exp(mu_a/tau) and q=softmax(log pi+mu/tau). Then grad F=q and H=(diag(q)-q q^T)/tau, a positive-semidefinite susceptibility matrix. H*1=0 exactly. For a calibrated observation Y, mu^Y=E[U|Y] has E mu^Y=mu. Its exact smoothed information value is

    G_tau=E[F_tau(mu^Y)]-F_tau(mu)>=0,

and its second-order term is (1/2) tr[H Cov(mu^Y)]. The expectation is over the *specified computation's outcomes*. It is not generally correct to insert the total latent Cov(U). The exact decomposition is

    Cov(U)=E Cov(U|Y)+Cov(E[U|Y]).

The script verifies this for theta in {0,1} with equal prior and U(theta)=(theta,1-theta,1/4). At channel accuracy alpha=3/4, Cov(mu^Y) has leading block [[1/16,-1/16],[-1/16,1/16]], whereas Cov(U)'s block is four times larger. Exact terminal EVSI is 1/4. At alpha=1/2 the channel is uninformative: Cov(mu^Y)=0, EVSI=0 and G_tau=0, despite nonzero Cov(U). Substituting latent variance gives a spurious positive second-order value of 1.2007194719.

A useful finite remainder bound follows from f(t)=F_tau(mu+t v): f'''(t)=tau^(-2) E_qt[(v-E_qt v)^3]. If R(v)=max(v)-min(v), then |f'''(t)|<=R(v)^3/tau^2 and

    |G_tau - (1/2)tr[H Cov(mu^Y)]| <= E[R(mu^Y-mu)^3]/(6 tau^2).

It is a conservative controlled bound, not an asymptotic equality. The actual finite enumerations at alpha=1/2,3/4 and small excess accuracies 1/64,1/32,1/16,1/8 satisfy it. No approximation guarantee is inferred merely from a successful example. beta=1/tau -> infinity is not a uniform validity limit for this local expansion.

## 5. Positive information with zero decision value

Take theta in {-1/4,+1/4} equiprobably and U(theta)=(2+theta,1+theta,theta). Perfectly reveal theta. Mutual information is log(2)=0.6931471805599453 nats, but action 0 is always strictly optimal and terminal EVSI is exactly zero. F_tau is common-shift equivariant, F_tau(mu+x*1)=F_tau(mu)+x, so its exact information value is also zero. Cov(mu^Y)=(1/16)11^T is nonzero; the susceptibility annihilates it. The same example tests why entropy reduction alone is not a decision-value estimator.

## 6. Reproducible execution and outcome

Command: `python3 /tmp/quartz_c26_statphys.py`.
Environment: Python 3.12.14, SymPy 1.14.0, NumPy 2.3.5, SciPy 1.17.0. Script SHA256: `c61d1d32ef08fc6fadec3fec70923ce00c61851cdddd26f1348e91a52c354c39`. JSON stores all matrices, finite outcomes, trajectory rows, fixed tolerances through named assertions, and the complete check list. The invoking host records the actual exit code separately.

Requested numeric Gibbs target: [0.11419519938459448, 0.8437947344813395, 0.042010066134066035]. Initial KL: 0.7719002946995058; step-80 KL: 4.192334491911877e-15; variational-gap residual: 0. Floating-point final KL may have a last-bit signed roundoff and is accepted only under the stated absolute tolerance; it is not evidence for negative mathematical KL.

Evidence classification: exact matrix and covariance identities **DERIVED + SYMBOLICALLY CHECKED**; finite high-precision and float trajectories **NUMERICALLY CHECKED**; general contraction and remainder bound **DERIVED** from the displayed inequalities. Physical equivalence to MCTS, real-game benefit, cognitive-brain validity and production runtime efficiency remain **UNRESOLVED / NOT TESTED HERE**. The exact classical/Gibbs mechanism is a valid constructive replacement for the defective diagonal-jump argument without claiming a physical identity.

## Execution receipt

The successful host execution returned exit code **0** (exec_command chunk `d0cadd`, 0.623 s) and printed 72 passing scientific assertions. The initial invocation returned exit code 1 (chunk `feff3c`) before running any calculation: a Markdown f-string delimiter collided with the third-derivative notation. Only that string delimiter was repaired before the successful run. This is a document-generation syntax failure, not a failed scientific check.
