# Quartz C26 R02 — original-theory import and primary-source synthesis
Date checked: 2026-11-23 UTC
Owner: native subagent /root/c26_literature. Selected scope: R02 only; no recursive agent, no repository edit, no build, benchmark, or code implementation.
Observed repository HEAD: d6019d674ae76524b1895016f49b82d4f7b001b2.
Repository: /home/cosmosapjw/quantum/research/quartz_general_meta25_20261123.
Evidence: LITERATURE_SUPPORTED for source statements; DERIVED for the translated operators and examples; implementation and empirical game-performance claims NOT_RUN here.

## Decision for the parent

A faithful C26 import can combine (a) original Quartz's KL/Gibbs bounded-rational selection and priced metareasoning, with (b) Weitzman's independent exact-reveal search theorem as a sharply defined global reference. The latter supplies something absent from merely repeating the finite Bayesian reference: a reservation-index ordering and an exact global STOP criterion for a restricted query model. In a game adapter with learned categorical priors and measured/predicted CPU costs, this theorem supplies a model-based policy and a conditional benchmark, not unconditional game optimality.

Reuse the existing Python quartz/experiments/metalevel_decision.py and Rust src/mcts/policy/finite_metalevel.rs. Both were read. They already have finite posterior transitions, terminal expected utility, Bayes simple regret and finite-horizon computation control; the Rust module explicitly does not implement SearchPolicy. Add the concrete game query bridge or new operator/measurements rather than a second copy of that reference.

## Original-source identity and intent

Read docs/legacy/QUARTZ_paper.pdf through pdftotext (21 substantive pages), especially:
- PDF p.3: deterministic two-player zero-sum perfect-information games; original terminal utilities {-1,0,+1}; consumer-CPU time targets; comparable baseline.
- PDF p.5 Eq.(1), Proposition 1: expected utility minus beta^-1 KL(q||pi), with Gibbs optimizer and log-partition.
- PDF pp.5–6 Eq.(2): FOCUS / EXPAND / MERGE / STOP and expected root decision improvement minus compute cost.
- PDF pp.7–11: heuristic log1p(scale) "one-loop" bonus, NS-inspired hidden-mode gate, transposition/graph curvature, and KL(tree visits||prior) surprise.
- PDF pp.20–21: original references include Russell–Wefald, Skilling, Friston, Ollivier, and Forman-related work.

Thus finite-temperature choice, metareasoning, uncertainty corrections and rare-mode exploration are original project directions. Coarse-graining/renormalization is not selected merely from a fashionable name. Parent/history agent independently confirmed the same source intent and identified existing heuristic implementations.

Prior verified C25 record docs/research/quartz_general_incumbent_price_20261123/theory/LITERATURE.md was read and reused. Its Hay/LUCB/simple-regret comparisons do not need a complete repeat.

## Six source records and exact applicability

### S1 — Weitzman: independent exact-reveal search

Martin L. Weitzman, Optimal Search for the Best Alternative, Econometrica 47(3), 641–654 (1979), DOI 10.2307/1910412.
Published author copy: https://scholar.harvard.edu/files/weitzman/files/optimalsearchbestalternative.pdf
Primary earlier version actually acquired: MIT-EL-78-008, May 1978, https://www.osti.gov/servlets/purl/6795538 .
MIT authentication record: https://dspace.mit.edu/entities/publication/6eca68ca-2a67-4e4c-b2f3-797103523e50 .

Published pp.643–644 state independent rewards, fixed opening fees, recall, fallback and Bellman Eq.(2). The 1978 PDF's printed p.9 / PDF p.13 Eq.(7), visually verified, is
c_i = beta_i integral_{z_i}^infinity (x_i-z_i) dF_i(x_i) - (1-beta_i) z_i,
where beta_i=exp(-r t_i). Printed p.10 / PDF p.14 states highest-reservation-first and stop above all remaining reservations. Proof begins printed p.16a; pp.14–16 explicitly discuss limitations, including correlated learning, partial research, parallel search and binding horizons.
Set beta_i=1 and k_i=lambda c_i to obtain E[(X_i-z_i)_+]=k_i.
Verdict: VERIFIED for the source model; CONDITIONAL for the proposed learned game policy. Tie-stop at equality is a weakly optimal convention. The source theorem is known after notation translation, not a new Quartz theorem.

### S2 — Ortega–Braun: thermodynamic bounded rationality

Pedro A. Ortega and Daniel A. Braun, Thermodynamics as a theory of decision-making with information-processing costs, Proc. R. Soc. A 469, 20120683 (2013), DOI 10.1098/rspa.2012.0683.
Actually read author manuscript dated July 31, 2012: https://adaptiveagents.org/_media/papers/thermodynamicsdecision.pdf .
Theorem 2 / Eq.(8), PDF p.6, gives expected gain minus beta^-1 relative entropy. Prefer the internally explicit Eqs.(9)–(11), PDF p.9:
q(x) = p0(x) exp(beta U(x))/Z,
Z = sum_x p0(x) exp(beta U(x)),
V = beta^-1 log Z.
These equations avoid a missing p0 factor in the normalizer displayed earlier in Corollary 3 of this manuscript. The corrected factor is present explicitly in its Eq.(9). Finite support and beta>0 suffice here; options with p0=0 are excluded.
Verdict: VERIFIED finite-choice operator; CONDITIONAL as a physical CPU-cost theory. The KL price and observed runtime price require separate meanings. Author-manuscript/publication byte equivalence was not asserted.

### S3 — Lieder–Griffiths: resource-rational cognition

Falk Lieder and Thomas L. Griffiths, Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources, Behavioral and Brain Sciences 43, e1 (2020), DOI 10.1017/S0140525X1900061X.
Primary target-article PDF: https://perception.jhu.edu/files/PDFs/20_ResourceRationality/LiederGriffiths_ResourceRationality_BBS.pdf .
PDF/printed p.5, Eqs.(3)–(4): choose a feasible heuristic to maximize expected result utility minus expected opportunity cost cost(t_h,rho,lambda); Eq.(4) averages over environment uncertainty given limited information. The paragraph below Eq.(3) explains that the optimization may be effected by evolution/development/learning, rather than solved online by the actor.
Verdict: VERIFIED objective/interpretation; CONDITIONAL transfer of a particular learned heuristic. Measured execution, controller and inference overhead belong in its resource accounting. Empirical human-cognition claims are not required for the software result.

### S4 — Friston: variational free energy and surprise

Karl Friston, The free-energy principle: a unified brain theory?, Nature Reviews Neuroscience 11, 127–138 (2010), DOI 10.1038/nrn2787.
Author PDF: https://www.fil.ion.ucl.ac.uk/~karl/NRN.pdf .
PDF p.2, Box 1, explicitly gives
F[q] = E_q[log q(theta)-log p(o,theta)]
     = KL(q(theta)||p(theta|o))-log p(o).
This is the original project's cited theoretical exposition (a review by the theory's author, not a new empirical study). It also separates prior divergence from likelihood accuracy.
Verdict: VERIFIED identity; the original Quartz S=KL(q_tree||pi) alone is not this complete variational free energy. Its use as a policy-mismatch observable is available without claiming an FEP derivation of expected improvement. No generic value-of-computation bound follows from that scalar alone.

### S5 — Hay et al.: reuse of verified priced metareasoning

Nicholas Hay, Stuart Russell, David Tolpin, Solomon Eyal Shimony, Selecting Computations: Theory and Applications, UAI 2012.
https://www.auai.org/uai2012/papers/123.pdf .
Reused C25's already verified locations: Definition 3, PDF p.3; Theorems 4–5 and Definition 6/Theorem 7, PDF p.4. The policy objective is E[-cN+max_a mu_a(S_N)]; VPI bounds optimal expected computation count divided by cost. A positive myopic gain supports continuation; nonpositive one-step gain generally does not certify all multistep policies as unprofitable.
Verdict: VERIFIED_REUSED in its preserved C25 scope; not freshly re-fetched here. Weitzman's exact-reveal independent-box model is a special structure that admits a stronger global stopping/order rule.

### S6 — Skilling: original rare-mode direction and its actual operator

John Skilling, Nested Sampling for General Bayesian Computation, Bayesian Analysis 1(4), 833–860 (2006), DOI 10.1214/06-BA127.
https://projecteuclid.org/journals/bayesian-analysis/volume-1/issue-4/Nested-sampling-for-general-Bayesian-computation/10.1214/06-BA127.pdf .
Printed p.834 Eqs.(3)–(4) define X(ell)=integral_{L>ell}pi and Z=integral_0^1 L(X)dX. Printed p.839 Eqs.(14)–(15) give X_i=t_i X_{i-1}, density N t_i^(N-1), E log t=-1/N. Constrained-prior sampling and sufficiently independent live points are essential.
Verdict: VERIFIED source equations through extracted primary text; the original Quartz Poisson hidden-mode proxy is adjacent inspiration, not a nested-sampling implementation. C26 need not add NS before the exact query bridge is tested.

## Concrete executable translation A: game root queries and Pandora indices

The following is a direct specialization, not a claim that all game values are independent.

Let a finite set A of unopened root-child computations remain. Opening a fully solves that child's root-perspective payoff X_a in {0,1/2,1}. A retained exact incumbent gives y. The admissible terminal recommendation chooses only among the known fallback and fully revealed children. This restriction matters: classic Pandora does not grant a free choice of an unopened action at its posterior mean.

Let categorical p_a=(p_a0,p_ad,p_a1) be known model probabilities, independent across unopened a, and let k_a=lambda c_a>0 be fixed additive utility costs. Each action is revealed at most once, no binding cap prohibits otherwise feasible openings, and all explored results are retained. Then
V(A,y)=max{y, max_{a in A}[-k_a+sum_x p_a(x)V(A\{a},max(y,x))]},
V(empty,y)=y.
Pandora's index z_a uniquely solves sum_x p_a(x)(x-z_a)_+=k_a. Open a maximal z_a while z_a>y; otherwise stop. Degenerate distributions are allowed. With positive cost the root lies below the essential maximum; it may be negative and must not be silently clipped.

For the three-point support define mu_a=(1/2)p_ad+p_a1 and s_a=p_ad+p_a1:
- If 0<k_a<=(1/2)p_a1 with p_a1>0: z_a=1-k_a/p_a1.
- If (1/2)p_a1<k_a<=mu_a with s_a>0: z_a=(mu_a-k_a)/s_a.
- If k_a>mu_a: z_a=mu_a-k_a.
Boundary branches agree; skip branches with zero denominator. For X=0 surely, z=-k.

The one-step gross value is g_a(y)=sum_x p_a(x)(x-y)_+. Its net gain is g_a(y)-k_a.
For the specified independent exact-reveal model, all nets<=0 is equivalent to y>=max z_a and is globally sufficient to stop. The ordering argmax net is generally not argmax z. That distinction is the substantive new source import.

Analytic discriminator, not a numerical run:
y=0. A reveals 1 with probability .2 and 0 otherwise, k_A=.05. B reveals .5 surely, k_B=.1.
Myopic nets: .15 and .4; greedy chooses B.
Reservations: z_A=.75, z_B=.4; Pandora chooses A.
A-first adaptive value=.2*1+.8*(.5-.1)-.05=.47.
B-first value=.5-.1+(.2*.5-.05)=.45.
This gives an exact counterexample to identifying local myopic selection with global resource-optimal ordering.

Implementation status and measurement contract:
- Train-only categorical predictions are subjective model inputs, not deterministic enclosures or automatically calibrated posteriors.
- True solved game payoffs may be dependent. Conditional independence of the fitted product model must be named.
- Predicted/EMA cost is a model expectation; actual wall time can depend on the hidden payoff, transposition cache and search history. Such costs do not establish Weitzman's fixed-fee assumptions.
- A hard deadline or fixed maximum number of openings changes the admissible policy class and can break index optimality. Do not label a budget-capped index policy globally optimal under the original theorem.
- Retaining the maximum of exact revealed utilities independently guarantees sample-path nondecrease of the incumbent, even when the probability model is poor. This deterministic property is separate from expected net-value optimality.
- Record true incumbent payoff, remaining possible gain ceiling, predicted g/k/z, actual incremental and controller runtime, query order, stopping reason, terminal utility and total priced utility.
- Original v in {-1,0,1} maps to u=(v+1)/2. Utility differences halve, so lambda must be specified in the chosen utility scale.

## Concrete executable translation B: KL/Gibbs choice, with precise scope

For a fixed state and already-defined scalar compute values Q_c in utility units, including STOP, and a fixed normalized prior pi0 with support on admissible choices, define
J_tau(q)=sum_c q_c Q_c-tau KL(q||pi0), tau>0.
Then
q_tau(c)=pi0(c)exp(Q_c/tau)/sum_d pi0(d)exp(Q_d/tau),
F_tau=tau log sum_c pi0(c)exp(Q_c/tau).
The exact check is
F_tau-J_tau(q)=tau KL(q||q_tau)>=0.
This is an executable variational optimum and measurable identity. Stable log-sum-exp is needed. tau has utility units; beta=1/tau has inverse-utility units. KL uses natural logs (nats). Neither tau nor beta is a literal bath temperature without a physical utility-energy map. Physical inverse temperature would be 1/(k_B T) for energy measured in joules.

If Q_c=g_c-lambda c_c, this is a local one-step selection objective. It does not inherit the global Pandora ordering theorem and, at finite tau, assigns probability to inferior choices on positive prior support. It may continue with negative expected raw net gain. The example above remains a counterexample as tau tends to zero.

If global KL-regularized finite-horizon control is wanted, use a consistent recursion:
F_0(b)=B(b)=max_a E_b[U_a],
Q_h(b,STOP)=B(b),
Q_h(b,c)=-lambda cost_c+E_y F_{h-1}(b^{c,y}),
F_h(b)=tau log sum_c pi0(c|b,h)exp(Q_h(b,c)/tau).
The action prior and feasible set are specified before evaluating the objective; at h=0 STOP is forced. Reusing unregularized Bellman values inside a repeated soft selector is only a local optimum, not this recursive optimum. A hard certificate gate may remove forbidden continuations before normalizing pi0; state that changed admissible set explicitly.

Information divergence and physical/runtime costs should not be double-counted as identical measured expenditures. Treat KL as a stated behavioral regularizer unless its cost calibration is separately supported. Evaluate raw utility minus CPU cost separately from the regularized objective.

## Optional original-channel repair: covariance, information gain and VOC

These are direct calculus identities, not a claimed new literature theorem and not an instruction to implement every channel in C26.

For F_beta(u)=beta^-1 log sum_a pi_a exp(beta u_a),
gradient F=q, Hessian F=beta[diag(q)-qq^T].
For U=mu+epsilon, E epsilon=0, Cov(epsilon)=Sigma:
E F_beta(U)=F_beta(mu)+(beta/2)[sum_a q_a Sigma_aa-q^T Sigma q]+R_3.
A sufficient remainder bound is |R_3| <= (4/3) beta^2 E||epsilon||_infinity^3, obtained from the third centered moment along the Taylor path. This explicitly needs the finite third absolute moment. The bound may be loose; no asymptotic-smallness claim follows merely from existence of moments.

A common-mode fluctuation Sigma=sigma^2 11^T gives zero quadratic correction. Independent per-action variance bonuses fail this invariance. Also E log Z and log E Z are different objects; the annealed expression generally includes common-mode uncertainty and may fail to exist for heavy-tailed priors.

Actual computation gain depends on its observation channel. If M'=E[U|Y], then Cov(M')=Sigma-E[Cov(U|Y)] and
E F_beta(M')-F_beta(mu)
approximately equals (beta/2)Tr{[diag(q)-qq^T][Sigma-E Sigma']}.
This is a soft-terminal-decision gain. It is not automatically the hard-choice EVSI or a certified bound. Existing finite posterior transitions can calculate the exact comparison rather than treating a variance proxy as a theorem.

A separate useful diagnostic is expected Bayesian information gain:
I(Theta;Y|b,c)=sum_y P(y|b,c) KL(b^{c,y}||b).
It does not equal decision improvement. If U(a0)=1 and U(a1)=0 in every hidden state, an experiment may identify the hidden state perfectly while its EVSI is exactly zero. This distinguishes a model-surprise monitor from the priced action-value criterion.

## Verification status, current obligations and stop boundary

Completed: original intent inspection; existing implementation/reference reuse; six-source bounded map; exact equations and applicability; Weitzman Eq.(7) visual verification; concrete three-outcome reservation implementation contract; local-versus-global Gibbs distinction; analytic discriminator.
No implementation, benchmark, native test/build or full proof audit was run by this child. Nothing here establishes game win-rate superiority or the original consumer-CPU timing targets.
Parent owns selecting one concrete C26 import, integrating code, testing the adapter and admitting empirical claims.
No new strong no-go or novelty claim is made. Formula-level components are known after translation or elementary derived consequences.

## Actual tool provenance and failures

- Exa: 2 searches (Ortega–Braun; Weitzman), 3 fetch calls (Ortega/Skilling; Weitzman published copy; Weitzman MIT1978/OSTI). Success returned useful primary text. The published Weitzman fetch exposed only pp.641–644 despite requested character budget; it did not verify Eq.(7).
- SciSpace: 2 searches. First broad search returned adjacent material; refined search found exact Lieder–Griffiths target article. One extract_data call returned SUCCESS with an explicit inability to extract equations/pages/online-optimization scope. Its formula extraction was INSUFFICIENT; direct PDF reading supplied pp.5 Eq.(3)(4).
- Parallel: one search, search_b42b5aba1c3269ab917a48f52846ab27, session_b42b5aba1c3269ab917a48f52846ab27. Three fetch calls:
  1. extract_3ee3cbd4440b5a5385a8c7f08ae980be: Skilling/BayeSys excerpts success, exact formulas insufficient.
  2. extract_9938b5479764266453063048d2ead2a7: full Skilling text success; exact pp.834/839 equations verified.
  3. extract_76a0ca57400542cfb0fc09668881c6bd: returned journal table of contents instead of requested Weitzman paper; CONTENT_INVALID, not evidence of Eq.(7). No feedback was sent; parent can offer to report this request ID if user wishes.
- Search-service primary PDF reading succeeded for Ortega author manuscript, Friston and Lieder–Griffiths. Publisher Ortega returned403, exact arXiv v2 endpoint DisabledError, Harvard Weitzman503, Euclid Skilling anti-bot/service page, MIT download429, OSTI timeout then502. None of those failures was represented as successful access.
- Remote public OSTI download then succeeded (exit0); pdftotext and rendering of PDF p.13 resolved the formula OCR gap. No credentials or private repository content were submitted to public searches.
- Weitzman temporary primary PDF: /tmp/quartz_c26_weitzman1978.pdf; SHA256 546b8b0344acd58b96e2ea9ab753a3b70dd87803d9ecd4d7b69b7fdb15373303.
- Formula image actually viewed: /tmp/quartz_c26_weitzman_eq7.png. Primary PDF/text were not copied into repository or report.
- Useful root-readable source refs from this child: Ortega author Eqs.(9)–(11) turn464611453view0; Lieder Eq.(3)(4) turn435053133view4 and turn746319631view0; Friston Box1 turn746319631view1; Weitzman published-primary search turn417565186search6; MIT metadata turn422427132view0. Parent must open/retrieve its own web refs before user-facing citations.

## skill_context_used

Actually read PhysMath Research Loop1.3.0 c43/physmath-research-loop SKILL.md, references/THEORY_LOOP.md (R02), references/POLICY_INTEGRATION.md, references/SUBAGENT_MODE.md.
Actually read Mathbox4.0.0 c42/literature-check SKILL.md and references/source-cache.md. No cache command or .mathbox source-evidence mutation was performed; this is an unapplied R02 packet.
Actually read c5/parallel-web-search and used the connected MCP, without CLI/account provisioning.
The selected-stage scope, bounded source repair, genuine source evidence and no-review-recursion instructions were applied.
