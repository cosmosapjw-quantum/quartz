# C24 R4–R6 constructive handoff: ordinary arithmetic with proved scalar envelopes

Author: native child /root/c24_theory. Scope: R4 candidate equations, R5 constructive proof, R6 finite arithmetic/domain check. No production repository edits, benchmark, or publication performed.

## Skill context actually used

Read cloud PhysMath Research Loop 1.3.0 SKILL.md plus references/THEORY_LOOP.md, POLICY_INTEGRATION.md and SUBAGENT_MODE.md. Applied R4 actual candidates, R5 explicit derivation, R6 conventions/limits and bounded checking. No recursive delegation or review-of-review. Parent owns final admission.

Read the complete 496-line C23 source at /home/cosmosapjw/quantum/research/quartz_general_meta23_20261123/src/bin/quartz_c23_residual_gate.rs, parent HEAD a4417b081037c4d5b49976ddcaeaa3f0771773b2. The inherited ordinary operator performs one diagonal multiplication and, in fixed north/south/west/east order, up to four neighbour multiplications plus four subtractions.

Primary arithmetic source actually opened: Siegfried M. Rump, Error Estimation of Floating-Point Summation and Dot Product, author PDF https://www.tuhh.de/ti3/paper/rump/Ru11.pdf . Section2 equations2.2/2.4 give relative addition and relative-plus-absolute multiplication models with underflow; Section3 equation3.1 gives gamma notation. The particular deliberately loose envelope below is derived here, not attributed verbatim to that paper.

## R4: candidates and chosen contract

Candidate A, selected by parent before new cohort outcomes: ordinary stencil/dot accumulation, exact max magnitudes, conservative gamma envelopes, and outward rounding only for a fixed number of scalar budget operations. It avoids a per-element interval stencil. Its conservatism can delay certification and its speed remains an empirical question.

Candidate B, retained as literature alternative: Rump Algorithm4.2/Corollary4.4 accumulates the sum of absolute rounded products in the same order as the dot, then uses ufp and a realmin floor. It is sharper, but adds accumulator/ufp implementation obligations. It is not selected for this cycle. Do not silently swap candidates after seeing TEST.

The mathematical problem and original choice theorem remain unchanged:
A=I+rho L_D, zero-exterior 5-point stencil, n=m², alpha=1, beta=1+8rho;
q=b^T A^-1 b, ell=2b^T x-x^T A x, r=b-Ax;
q-ell=r^T A^-1 r and ell+D/beta <= q <= ell+D, D=||r||².

All quantities are nondimensional. This is a finite discrete linear algebra problem, not a continuum convergence claim. Supported arithmetic is IEEE binary64 round-to-nearest with gradual underflow; no FTZ/DAZ, no unaccounted mixed precision, and no overflow. Stored inputs and iterates are treated as exact reals.

## R5.1: primitive error and finite-operation lemma

Let u=2^-53 and sigma=2^-1022 (f64::MIN_POSITIVE, the smallest positive NORMAL number).
For a finite elementary result z, round-to-nearest satisfies
    fl(z) = z(1+delta)+epsilon, |delta|<=u, |epsilon|<=sigma.
This is conservative: gradual-underflow rounding error is at most half the smallest subnormal, which is vastly smaller than sigma. The normal-result relative model supplies delta; the underflow case can use delta=0. Addition/subtraction are exact sufficiently near underflow, but charging sigma for them is still valid.

For k u < 1 define exact-real constants
    g_k = k u / (1-k u),
    F_k = k sigma / (1-k u).

A linear sum of input products using at most k elementary roundings can be expanded as a sum of exact input terms multiplied by products of at most k factors (1+delta), plus at most k additive epsilon terms multiplied by such factors. The factors obey
    |product(1+delta_j)-1| <= g_k,
    |product(1+delta_j)| <= 1/(1-k u).
Proof: the product is between (1-u)^k and (1+u)^k. Bernoulli bounds the lower deviation by ku; the binomial series is bounded by sum_j(ku)^j =1/(1-ku), bounding the upper deviation by g_k. Thus
    |computed linear sum - exact linear sum|
      <= g_k * sum(abs(exact input terms)) + F_k.

This proof accounts for every normal relative rounding and every additive underflow error. It uses no empirical epsilon and no CG orthogonality.

Compute UPPER floating bounds g>=g_k,F>=F_k using scalar directed enclosures; do not evaluate a mathematical error budget with untracked ordinary rounding. n<=16384 gives 2 n u<=2^-38 and g_2n<1, far from a singular denominator.

## R5.2: ordinary pass and exact maxima

For the stored b,x, compute using the same fixed ordinary order as the baseline:
    yhat_i = ordinary_stencil(A,x)_i
    rhat_i = fl(b_i-yhat_i)
    Phat = sequential dot(b,x)
    Qhat = sequential dot(x,yhat)
    Dhat = sequential dot(rhat,rhat).

Let
    B=max_i |b_i|, X=max_i |x_i|,
    Y=max_i |yhat_i|, R=max_i |rhat_i|.
Absolute value and max on finite stored binary64 values are exact; B can be cached from validation. No vector norms or exact oracle labels are supplied to the controller.

The stencil has at most9 elementary operations. Its exact absolute input sum is
    (1+4rho)|x_i| + rho sum_neighbours |x_j| <= beta X.
Therefore, componentwise,
    |yhat_i-(Ax)_i| <= Ey = g_9 beta X + F_9.

The fresh residual satisfies
    |rhat_i-r_i| <= Er = Ey + u(B+Y) + sigma.
The second contribution bounds the ordinary subtraction b_i-yhat_i. Even tiny/subnormal intermediate products are covered.

## R5.3: variational-score enclosure

A sequential n-term dot can be overcounted as n multiplications plus n additions, k=2n. Define
    EP = g_2n n B X + F_2n,
    EQ = g_2n n X Y + F_2n + n X Ey.
Then
    |Phat-b^T x| <= EP,
    |Qhat-x^T A x| <= EQ.
For EQ, first bound the ordinary dot with yhat, then add
    |x^T(yhat-Ax)| <= n X Ey.

With outward scalar endpoints,
    P in [down(Phat-EP), up(Phat+EP)],
    Q in [down(Qhat-EQ), up(Qhat+EQ)],
    ell in 2 P - Q.
The deterministic fallback score remains the identical ordinary
    ell_hat = fl(fl(2*Phat-Qhat)/S).
The certificate uses the enclosing ell interval, never ell_hat as a rigorous lower bound. Baselines and gated fallback must share the same ordinary order/state.

## R5.4: positive-dot inversion and fresh residual norm

Let D0=sum_i rhat_i² as an exact real. Since D0>=0, the dot lemma gives
    |Dhat-D0| <= g D0+F, g=g_2n, F=F_2n.
Rearranging yields
    D0L=max(0,(Dhat-F)/(1+g)),
    D0U=(Dhat+F)/(1-g).

For delta_i=r_i-rhat_i, |delta_i|<=Er,
    |D-D0|=|sum_i(2 rhat_i delta_i+delta_i²)|
            <= T = n(2 R Er+Er²).
Consequently
    DL=max(0,D0L-T), DU=D0U+T
enclose the exact norm² of the true residual at the stored iterate.

Recommended outward implementation:
    D0L=max(0, down_div(down_sub(Dhat,F), up_add(1,g))),
    D0U=up_div(up_add(Dhat,F), down_sub(1,g)),
    DL=max(0,down_sub(D0L,T)), DU=up_add(D0U,T).
Every nonnegative error expression Ey,Er,EP,EQ,T is computed upward. F,g are already upper bounds, so substitution enlarges rather than narrows the interval.
Using the positive dot rather than g*n*R² gives a tighter residual-square error envelope at essentially the same scalar cost.

## R5.5: normalize only after raw clipping

Use scalar outward arithmetic to form
    qL=max(0, down(ell_L+DL/beta)),
    qU=min(S, up(ell_U+DU)).
Check qL<=qU and finiteness. Then
    L=max(0,down(qL/S)), U=min(1,up(qU/S)).
Clip to [0,S] BEFORE dividing by S: a loose raw upper bound divided by a tiny S can overflow even when true u=q/S lies in[0,1]. Raw clipping is justified by A>=I and S>=||b||² and loses no valid true value.

Historical intersection and strict separation are unchanged:
    Lbar=max(Lbar,L), Ubar=min(Ubar,U);
    return certified i iff Lbar_i>max_(j!=i) Ubar_j.
Thus the C23 proof of exact optimality at certification and fixed-fallback noninferiority is reused under this new enclosure theorem.

## R5.6: input norm, tiny-input rejection and honest residual reporting

Bound input norm² by the same positive-dot lemma, using
    Bhat=sequential sum b_i²,
    BL=max(0, down((Bhat-F_2n)/(1+g_2n))),
    BU=up((Bhat+F_2n)/(1-g_2n)).
Each displayed operation must be outward in the indicated direction. The common scale validation is S>=BU.

Admitted input domain:
- side2..128, integer rho0..64; 2..8 actions; finite vectors of length side².
- positive normal power-of-two S.
- each RHS is exactly zero OR its max B lies in[2^-400,2^100].
- for nonzero RHS require computed BL>0.
- at response evaluation require finite stored state, X<=2^200 and X<=2^100*B for nonzero RHS; zero RHS requires x=0.
- reject any nonfinite scalar arithmetic or nonfinite normalized fallback score.

This explicitly rejects the previous nonzero b=2^-600, S=2^-1000 reproducer as UNSUPPORTED_RHS_NORM_RANGE before solver initialization. Do not label rejection as successful convergence. This is a deliberate supported-domain restriction; it is not arbitrary-input solver repair.

Why it prevents initial norm underflow: B>=2^-400 means at least one squared component is about2^-800, far above the smallest normal. Bhat is positive and BL>2^-801. The shared recursive stopping threshold Bhat*(1e-14)² also stays far above normal underflow. This does not make a recursive residual identical to the true residual; fresh certification remains necessary.

For FAST and OUTWARD evaluations, a conservative upper bound on the true relative residual is
    relative_upper = up_div(up_sqrt(DU), down_sqrt(BL)).
Do not form DU/BL before sqrt: that intermediate ratio may overflow although the final norm ratio is finite. BL>0 is established by validation. Report residual_goal_met=true only if this upper bound<=the specified f64 tolerance. For OUTWARD use its actual outward residual-square upper endpoint in place of DU.
For zero b and x=0 the exact residual is zero; define relative residual0 as the explicit zero-problem convention. A nonzero x for b=0 needs an error/undefined-relative handling, not a zero-norm success shortcut.

PLAIN_STOP/PLAIN_ALWAYS should not incur a hidden full enclosure cost just to report a certified residual. They may report a clearly labelled ordinary residual estimate, with certified upper/goal=null or UNVERIFIED. Approximate ordinary diagnostics must not be labelled a mathematical residual certificate.

## R6: no-overflow envelope and adopted state guard

Within B<=2^100, X<=2^200, n<=2^14, beta<2^10, conservative magnitudes are
    Y<2^211, R<2^212,
    |Phat|<2^315, |Qhat|<2^426, Dhat<2^439.
All error envelopes and unnormalized ell/q endpoints remain below2^450. The proof budgets use only positive products/sums of quantities of these orders, g_2n<2^-37 and normal F_k. Thus the raw ordinary/envelope construction has ample binary64 headroom. After raw clipping, q/S lies in[0,1], so normalized certificate endpoints cannot overflow.

Normalized ell_hat could overflow for an arbitrary huge x and tiny S with only the absolute x cap. The parent adopted the following additional sufficient guard before R5 publication:
    X <= 2^100 * B for nonzero RHS,
with zero RHS requiring x=0.
The threshold is an exact power-of-two scaling in the admitted B range. Using S>=B² gives
    |ell_hat| < n*(2*2^100 + 2*beta*2^200) < 2^226,
so this very loose guard guarantees finite scores. Retain the explicit nonfinite-score rejection as defense against implementation/runtime violations. The guard is not expected to constrain the manufactured CG runs.

For the certified relative upper, BL>2^-801 and DU<2^450 imply the sqrt-then-divide evaluation is finite (less than about2^625).

Counterexamples/limits checked:
- zero RHS: exact zero response, no false unique winner among equal zero actions;
- rho0: A=I, beta1, the energy identity remains exact;
- ties: strict separation cannot certify a unique optimal action;
- arbitrary non-Galerkin iterates: proof uses no CG orthogonality;
- extreme initial norm underflow: reject the known nonzero tiny fixture;
- tiny components mixed into a normal-max RHS: permitted; sigma terms cover their underflow;
- huge x/tiny S: clip raw certificate before normalization and preserve score-finiteness rejection;
- final exact solution with rounded stencil: the residual envelope must still include true D=0.

## Actual finite arithmetic check (constructor check, not independent model review)

Executed a fresh local Python3 calculation through exec_command, process exit0. It implemented the selected scalar envelopes with math.nextafter and compared each intermediate against fractions.Fraction computations from the exact stored binary64 b and x. This did not import production code.

For side2,rho1,S1, b1=(1,0,0,0), b2=(0,1/2,1/2,0), x_a=fl(b_a/5):
- Exact oracle q1=23/105, q2=5/42.
- Envelope1: [0.20888888888888474,0.28000000000000724].
- Envelope2: [0.10888888888888769,0.1800000000000028].
- L1>U2: true.
- For both cases, exact stored-iterate stencil error, P error, Q error, D interval and ell interval checks all passed.

Case1 scalar errors: Ey=1.798561299892757e-15, Er=2.020605904817789e-15, EP=7.105427357601016e-16, EQ=2.149391775674308e-15.
Case2: Ey=8.992806499463785e-16, Er=1.0103029524088946e-15, EP=1.776356839400254e-16, EQ=5.37347943918577e-16.

This finite check supports the formula/representative construction; it is not a universal floating-point implementation proof or performance measurement. Parent R7/C6/C7 must bind the actual Rust operation path and independent exact oracle checks.

## Handoff

The selected theorem is derived under explicit arithmetic/domain assumptions. Ordinary computation is O(n), with additional max reductions and O(1) outward scalar error-budget work per action. Original per-element outward evaluation is also O(n), so only measured constants determine the runtime result. Preserve the full outward path as a reference, use fresh frozen inputs, keep plain efficient controls, and do not claim speedup before actual total-time measurement.
