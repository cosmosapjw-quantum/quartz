# C27 R03–R07: a marginal-preserving joint posterior and exact two-query metareasoning

## Authority, scope, and evidence status

Authority is the user-specified quartz commit `87c15dc828d6eea519c219fd8a412fa21dfc464d`, read on device `edd1b263-56a0-4391-8a2a-677dd00eabf5` in `/home/cosmosapjw/quantum/research/quartz_general_meta26_20261123`. The observed HEAD matched that commit and the observed working tree was clean. This lane creates only scratch artifacts; it does not modify a repository, fit TRAIN, open CAL/TEST, or publish.

The model below is a **conjectural statistical closure** selected for identifiability with a small root sample. Its normalization, posterior update, Bellman decomposition, threshold, and stated bounds are **derived**. A separate complete-joint enumerator and the latent closed form agree in the finite cases recorded below: **exact-finite checked** and **implementation-verified for the supplied oracle**, not for a production controller. Empirical utility, fitted coupling, and wall-time advantage remain **unresolved**.

The controlling C26 deployment contract is `theory/R03_DEFINITIONS.md`: initial choice uses ordinary raw rollout mean, followed by an exact solve, and terminal deployment chooses only exact-known candidates. Do not restore the older prior-mean initialization appearing in an intermediate constructor narrative. The action STOP here is the known incumbent value b; it is not the maximum posterior mean over unrevealed candidates. General metareasoning papers allowing deployment of unqueried alternatives therefore need an explicit contract translation.

Source identities, relative to `docs/research/quartz_general_game_metareasoning_20261123/`:

| Source | SHA-256 |
|---|---|
| NEXT_C27_PROPOSAL.md | 50012cf2fc3e033bbaf323f381dfbb1597a555fa17cd2b77eedb3191f110830f |
| theory/CONSTRUCTOR_DERIVATION.md | 2c33c06caa02f0f684c82d534a6bff2c41f068027ab2bcefa11df662b5f63dc6 |
| theory/STATPHYS_DERIVATION.md | d717a973a529bd339a259b824ddaa80e88cd054fc4baee44102a9f14360fc512 |
| theory/R03_DEFINITIONS.md | 7bd7f3718693ce872275827ea3c4607a5f411b63b1f5f22a50dad494813d8d31 |
| coding/CODE_CONTRACT.md | 1821e46989ab21f9259ee9c61e4935824a362da5139d08db13d10fb86c1d0985 |

## R03. State, observations, costs, and the comparison contract

At a fixed root, U_a is the exact root-player utility of candidate a, with support {0,1/2,1}. X denotes the cues available before exact queries. Observed history h contains distinct exact queries and their outcomes. The state consists of h, remaining candidates S, and the best exact-known utility b. Querying a reveals U_a without observation noise, pays a fixed nonnegative modeled cost d_a, removes a, and updates b to max(b,U_a). Exact-known terminal actions are retained. No repeated query is legal.

All utilities and d_a have the same utility units. A node price lambda_N converts predicted nodes to utility cost. A wall price lambda_t has units utility/time. A comparison involving actual wall time must include planner overhead, not only exact-solver query time. The finite Bellman below prices queries only and is an approximation to full computation economics. The common initial solve is sunk at h_0 in the displayed continuation value; end-to-end wall-net evaluation must still account for it consistently.

The operative start h_0 includes the common mandatory initial exact solve and all other exact-known information. For a strict same-starting-marginals comparison, construct

    Q_{h_0}(u_S) = product_{j in S} P_rho(U_j=u_j | X,h_0).

Then both arms start with the same b, S, costs, and candidate marginals. The joint arm updates the remaining candidates after later queries; the product arm keeps their factors independent. Re-productizing after each actual joint update would erase the intended independent control. Conversely, comparing a conditioned joint law with the original unconditioned product law mixes the benefit of initial-reveal conditioning with the benefit of subsequent dependence. Such an arm answers a different question. The parent integration has selected that broader comparison as primary PRODUCT_H2 (common cue prior; initial plus later conditioning as the combined mechanism), and the strict post-initial productization as secondary INITIAL_PROJECTED_H2 (later conditioning only). Both estimands must retain these labels; equality of prior marginals must not be misreported as equality after initial conditioning.

## R04. A normalized one-parameter closure with exactly fixed prior marginals

Let p_a(u) be a previously fitted categorical marginal at cues X. For nondegenerate p_a define

    mu_a = sum_u p_a(u)u,
    s_a = max_{u:p_a(u)>0} |u-mu_a|,
    f_a(u) = (u-mu_a)/s_a.

For a degenerate marginal define f_a=0 and delta_a=0. Otherwise delta_a=Var_{p_a}(U_a)/s_a. Only f on positive-probability support matters. Then E_p[f_a]=0 and |f_a|<=1 on that support. The scale convention is invariant under a common positive affine change of utility units.

Introduce a binary latent variable Z with equal prior masses and a single global 0<=rho<1:

    p_a^z(u) = p_a(u)[1+z rho f_a(u)],  z in {-1,+1},
    P_rho(u_1,...,u_m | X) = (1/2) product_a p_a^+(u_a)
                             + (1/2) product_a p_a^-(u_a).

Each component is normalized because sum p_a f_a=0, and nonnegative because |rho f_a|<1. Summing all coordinates except a gives exactly p_a. Therefore coupling cannot improve the initial marginal fit by secretly replacing the C26 marginal model. rho=0 recovers the independent product. Changing rho to -rho only relabels Z, so theta=rho^2 is the identifiable nonnegative coupling coordinate.

This is a two-component conditional-independence mixture with **one additional fitted scalar**, not a separate latent fit in each of the 41 existing bins. It is estimable in principle from the already available complete root-level TRAIN utility vectors. A root log-likelihood separates as

    log P_rho(u) = sum_a log p_a(u_a) + log R_rho(u),
    R_rho = [product_a(1+rho f_a(u_a)) + product_a(1-rho f_a(u_a))]/2.

Single-candidate marginal observations do not identify rho. Complete vectors can. Near theta=0,

    log R_rho = theta sum_{a<b} f_a(u_a)f_b(u_b) + O(theta^2).

Thus the new parameter learns root-level cross-cumulants while leaving each original marginal unchanged. A fixed finite rho grid, endpoint strictly below one, deterministic tie rule favoring smaller rho, and any regularizer must be frozen before CAL. Reusing 90 TRAIN roots for this one-parameter fit does not by itself establish good generalization. If root cross-validation is reported, p_a must also be refitted inside each training fold; marginal fits made from all 90 labels cannot be called held-out predictions.

### Exact posterior and computational form

Write m_h=E[Z|h], initially zero, and w_h^+=(1+m_h)/2. For an unrevealed a,

    p_a(u|h)=p_a(u)[1+rho m_h f_a(u)].

After observing (a,u), Bayes' rule gives

    m_{h+(a,u)} = [m_h+rho f_a(u)]/[1+rho m_h f_a(u)].

The denominator is the observation likelihood divided by p_a(u), and is positive on supported observations for rho<1. A stable implementation uses the latent log odds:

    ell' = ell + log1p(rho f_a(u)) - log1p(-rho f_a(u)),
    m = tanh(ell/2).

Multiplication of likelihood factors commutes, hence conditioning order does not matter. Deterministic known terminal outcomes have f=0 and convey no additional latent information under this closure. Positive smoothing in the base p_a is needed if all physically possible exact outcomes must be supported; rho<1 does not repair a zero base probability.

Since E[U_a|Z=z]=mu_a+z rho delta_a, conditional independence gives, for a!=j,

    Cov(U_a,U_j)=rho^2 delta_a delta_j,
    Cov(U_a,U_j|h)=rho^2(1-m_h^2) delta_a delta_j.

Only the off-diagonal covariance has this rank-one form; the full covariance also contains conditional variances. The family represents nonnegative common-strength dependence. It cannot express arbitrary antagonistic candidate relationships or general transposition-induced dependence. A negative or negligible fit is grounds to select rho=0 or reject this closure, not to conclude that all correlation-aware metareasoning fails. The earlier three-candidate 0.69 example lies outside this restricted positive-common-strength family and remains a general joint-law unit fixture.

### What is imported from statistical physics, and what is not

This construction introduces an explicit two-state shared mode and its induced cross-cumulants. Each positive component can also be represented as a normalized finite Gibbs law with dimensionless energy

    E_z(u) = -sum_a log p_a(u_a) -sum_a log(1+z rho f_a(u_a)).

Here the component partition sum is one by the centering identity. This representation is a statistical model; no thermodynamic temperature, physical phase transition, or information-to-utility identity is asserted. Integrating out Z generates higher even-order interactions, not merely a fitted pairwise Ising Hamiltonian.

Separately, a KL-regularized action problem max_pi sum_a pi_a Q_a - tau KL(pi||r) produces pi_a proportional to r_a exp(Q_a/tau). That Gibbs action law concerns randomized decisions. It is not the joint generative law of utility outcomes. tau has utility units, whereas rho and f are dimensionless. C27 can use a hard Bellman maximum without enabling that action randomization. Any Gibbs-action ablation requires its own contract and must not be silently conflated with the new utility coupling.

## R05. Exact two-query value and the role of information

Define the conditional improvement

    g_j(h,b) = E[(U_j-b)_+ | h]

and the last-query option

    M(h,b,S) = max(0, max_{j in S}[g_j(h,b)-d_j]).

With no further query V_0=b. With at most one, V_1=b+M. Conditioning on the first query a and using V_1 for the remaining opportunity yields

    K_a(h,b,S) = g_a(h,b)-d_a
                 + sum_u p_a(u|h) M(h+(a,u),max(b,u),S\{a}),
    V_2(h,b,S) = b + max(0, max_{a in S} K_a(h,b,S)).

This follows directly from E[max(b,U_a)]=b+g_a and the law of total expectation. It requires no candidate independence. For the three-point support,

    g_j(h,0)   = p_j(1/2|h)/2 + p_j(1|h),
    g_j(h,1/2) = p_j(1|h)/2,
    g_j(h,1)   = 0.

The mixture posterior makes each predictive probability and update O(1). A direct H2 evaluation takes O(|S|^2 |U|^2) arithmetic, hence O(|S|^2) for fixed three-point support, with O(|S|) model storage. It does not enumerate 3^|S| complete worlds. A production constant-factor optimization is separate from the oracle, which deliberately enumerates small worlds for verification.

The parent integration plans a cap of two additional queries, using H2 then H1; this matches the finite-horizon theorem. If H2 is instead recomputed after every observation while the global query budget permits more than two queries, the resulting policy is receding-horizon H2. The displayed V_2 is the exact value of a commitment to at most two additional queries. It is not a proof of full-horizon optimality or an exact prediction of the longer receding-horizon policy's return.

### Direct incumbent gain versus posterior-dependent option

The first term g_a is direct incumbent gain. The second term is the option to choose or decline a final query after observing a. Define M_frozen(b_u) using current remaining marginals p_j(.|h), updated threshold b_u=max(b,u), and unchanged costs. Let

    I_a = E_u[M_updated(h+(a,u),b_u) - M_frozen(b_u)].

Then K_a=g_a-d_a+E M_frozen+I_a. This correction need not be nonnegative in general, because the threshold b_u and posterior information are coupled through the same observation. Calling I_a a universally nonnegative value of information would be an error.

For a purely informative query U_a<=b almost surely, b_u=b and g_a=0. The posterior expectations of each fixed improvement function form a martingale. Convexity of max(0,x_1-d_1,...) therefore gives E M_after >= M_before, so I_a>=0. Such a query beats STOP exactly when E M_after>d_a. It beats the best one-query value when I_a>d_a; it is the optimal H2 first action only if its K_a also dominates every other first-action K_j. If all initial one-query net gains are nonpositive, M_before=0 and its positive-value condition reduces to I_a>d_a.

For any a, even with a moving incumbent,

    |I_a| <= E_u[(1-b_u) max_{j!=a} TV(p_j(.|h,a=u),p_j(.|h))].

For fixed u, the improvement function lies in [0,1-b_u]. The bounded-function total-variation inequality controls each gain difference, and the maximum is 1-Lipschitz in the sup norm. Taking expectations and the triangle inequality proves the result. For the pure-information case combine it with I_a>=0.

### A new discriminating exact family

Let A have masses 1/2 on 0 and 1/2, and B have masses 1/2 on 0 and 1. Use the centered two-component law above, b=1/2, d_A=c>=0, and d_B=1/4+epsilon with epsilon>=0. Both initial one-query net improvements are nonpositive. A can never improve the incumbent, but

    P(B=1|A=1/2)=(1+rho^2)/2,
    P(B=1|A=0)=(1-rho^2)/2.

Querying B first has no positive continuation through A. Querying A first has one profitable branch when rho^2/4>epsilon. Consequently the exact optimum is

    V_2 = 1/2 + [-c + (rho^2/4-epsilon)_+/2]_+.

For c,epsilon>=0, a strictly profitable information-only first query exists exactly when

    rho^2 > 4 epsilon + 8 c.

This supplies a phase boundary in dimensionless coupling and utility costs that can discriminate the implementation, not merely a single favorable number. At rho=4/5, epsilon=1/20, c=1/100, d_B=3/10:

| A | B | Joint mass |
|---|---|---|
| 0 | 0 | 41/100 |
| 0 | 1 | 9/100 |
| 1/2 | 0 | 9/100 |
| 1/2 | 1 | 41/100 |

The high-A branch has B-win probability 41/50 and last-query net gain 11/100. Paying 1/100 for A gives gain (1/2)(11/100)-1/100=9/200. Thus V_2=109/200=0.545. The same-marginal product controller stops at 1/2. This is a mechanistic existence result, not evidence that quartz TRAIN roots exhibit this structure.

### A sign counterexample that prevents an information claim from drifting

Take A,B binary 0/1 with equal marginals, rho=9/10, b=0 and both costs 1/10. If A=0, then P(B=1|A=0)=19/200<1/10, so the actual last-query option is zero. If A=1, the incumbent already equals one, also giving zero option. A frozen marginal would assign option 1/2-1/10=2/5 after A=0, occurring with probability 1/2. Hence I_A=-1/5. This is a negative correction to a misspecified frozen-marginal valuation at a random threshold. It does not assert that a calibrated decision maker is harmed by an additional freely ignorable observation.

## R06. Belief error, finite checks, and scope of conclusions

Let P and Q be two complete joint laws on the same utility vectors at a common known history. Let policies have at most H further exact queries, the same legal actions, incumbent b, and nonnegative fixed costs. Let D_H be the sum of the largest min(H,|S|) query costs. Any fixed policy payoff F_pi lies in [b-D_H,1], so its range width is B_H=1-b+D_H. With TV(P,Q)=(1/2)sum|P-Q|,

    |J_P(pi)-J_Q(pi)| <= B_H TV(P,Q).

Taking a supremum over the same policy class on both sides gives

    |V_H(P)-V_H(Q)| <= B_H TV(P,Q).

If pi_Q is Q-optimal, insert and subtract V_H(Q)=J_Q(pi_Q) to obtain

    0 <= V_H(P)-J_P(pi_Q) <= 2 B_H TV(P,Q).

A policy must be defined on every P-supported observation history. Positive shared support ensures this; otherwise fix an explicit action extension on Q-zero histories before applying the bound. The conditional version uses TV(P(.|h),Q(.|h)); a small unconditional TV does not guarantee a small error after a rare conditioning event. Uniform cost errors |d_j-dhat_j|<=eta add at most H eta to a fixed-policy discrepancy and twice that to the standard transferred-optimum regret argument. History-dependent wall costs and planner CPU require additional explicit bounds; they are not automatically covered by this fixed-cost theorem.

### Exact computation audit actually executed

The supplied oracle uses `fractions.Fraction`, no RNG, and no floating tolerance. It compares a full joint-world recursive Bellman calculation with the independently implemented latent H2 closed form. Independence here means two algorithmic representations, not a second author's review or a production-code equivalence proof.

| Finite evidence | Cases |
|---|---:|
| Two/three-candidate joint models | 336 |
| Positive-probability histories, including 336 empty histories | 5,076 |
| Initial H2 value comparisons | 3,024 |
| Conditional H2 comparisons, including 4,740 nonempty histories | 5,076 |
| Information-only threshold/formula cases | 120 |
| Belief/policy transfer cases | 81 |
| Total successful assertions in final run | 53,851 |

The laws include uniform ternary marginals, binary 0/1 marginals, 0 versus 1/2 marginals, 1/2 versus 1 marginals, and deterministic zero/one cases. The threshold family additionally includes rho=1 as a mathematical boundary, while a fitted production family should stay below one. The checks cover normalization, unchanged initial marginals, posterior order invariance, full-joint versus latent posterior marginals, initial and posterior covariance formulas, b=1 stopping, horizon monotonicity, posterior-option TV bounds, positivity under pure information, productization after initial reveals, the new 109/200 example, the previous 69/100 general-joint fixture, and the -1/5 sign counterexample.

Execution used Mathbox 4.0.0 `computation-audit` bounded runner, Python 3.12.14, Linux x86_64, 120-second wall limit, 100-second CPU limit, 512 MiB address-space limit, one core and one conventional numerical-library thread. No external mathematical package is required.

- Baseline check01: 34,783 assertions; child exit 0; completed; 7.997255 seconds; started `2026-11-23T11:24:02.268056+00:00` as recorded by the executor.
- Final check02 added the concrete initial-reveal comparison risk: 53,851 assertions; child exit 0; completed; 6.042682 seconds; started `2026-11-23T11:25:34.149039+00:00` as recorded by the executor.
- Both `validate_manifest.py ... --root ...` calls exited 0 and returned `valid evidence record (run status: completed); mathematical interpretation requires review`.
- Final executed input SHA-256: `420149645d7f482cb908f8d319fa09017bdb817b8b15173789fc2812d1d6f7a9`.
- Final raw result SHA-256: `450f9f8351affc811eb7d876bbe106f5902e7435f687b20dc67bbca634142c47`.

The exact command, contract, manifest, logs and raw result are bundled in `quartz_c27_theory.json`. The execution root was `/workspace/scratch/3ddaad4e019e/c27_theory_local`, deliberately outside the repository. Its manifest therefore says repository commit unavailable; that is not the source-authority SHA. The source repository identity is separately recorded above. The delivered Python file is byte-identical to the final executed input, renamed only for handoff.

To rerun the exact oracle directly, use `python3 quartz_c27_theory.py /absolute/new/result.json`; create the result's parent directory first. To reproduce the bounded audit, use the recorded contract and Mathbox runner command with a new output directory. A runner status does not turn finite checking into a universal proof: the preceding algebra establishes the scoped universal statements.

## R07. Constructive decision and concrete falsifiers

Proceed with a one-parameter coupling pilot on the authorized TRAIN evidence, keeping the existing exact query/deployment mechanism and shared initial history. Freeze parameter fitting, the productization point, tie rules, costs, query cap, planner-price procedure and CAL decision rule before new outcomes are inspected. The current lane has not performed that fit.

The diagnostic sequence is: verify whether TRAIN complete vectors support a nonzero shared-mode coupling; verify conditional predictive calibration after exact reveals; verify that coupling changes the H2 first-action decision in the intended information-only situations; then measure actual paired wall-net outcomes with the planner included. If rho=0 is selected, the family has constructively collapsed to the independent limit. If the fitted posterior predicts poorly, reject or revise this closure before making a utility claim. If query-only gain is outweighed by planner time, record an economic failure of this implementation at the frozen price. These are distinct outcomes.

The principal open modeling risk is that quartz alternatives may have heterogeneous or negative dependence, which the one-parameter common-strength family excludes. This is a local restriction imposed for a small-data, reproducible first pilot, not a theorem against the original motivation. The original motivation is preserved by making computation an action, retaining exact evidence, updating beliefs about remaining computations after an observation, and measuring resource-priced deployment value. No information entropy or posterior sharpening is substituted for actual utility gain.
