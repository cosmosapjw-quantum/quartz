# Quartz C27 R02: bounded primary-literature reuse

Date: 2026-11-23 UTC. Agent: /root/c27_literature.
SSOT: https://github.com/cosmosapjw-quantum/quartz/commit/87c15dc828d6eea519c219fd8a412fa21dfc464d
Read-only project input: docs/research/quartz_general_game_metareasoning_20261123/NEXT_C27_PROPOSAL.md at SSOT checkout /home/cosmosapjw/quantum/research/quartz_general_meta26_20261123.
Artifacts are on remote host cosmosapjw, device edd1b263-56a0-4391-8a2a-677dd00eabf5, under /tmp. No repository edit or experiment was performed by this agent.

## Decision

Four primary sources cover the load-bearing claims. Reuse Bayesian metalevel control and correlated information value; derive the categorical latent update and finite-horizon recursion explicitly for Quartz. Do not present Bayes' rule or Bellman recursion as theoretical novelty. The empirical claim that C26 excess continuation is caused by dependence remains conjectural.

The model under examination has utility support {0, 0.5, 1}, exact paid reveals, a known incumbent, and deployment restricted to already known candidates. Correlated utilities can make the next query depend on earlier outcomes. A tiny explicitly parameterized joint law supports exact finite-horizon evaluation; arbitrary-distribution sample-access hardness does not rule this out.

## S1 — Weitzman: independent exact-reveal search

**Primary original read:** Martin Weitzman, Optimal Search for the Best Alternative, MIT Energy Laboratory report MIT-EL-78-008, May 1978.
URL: https://www.osti.gov/servlets/purl/6795538
This is the original report in the lineage of the 1979 published paper; report pagination/equation numbers are used here.
Author-hosted published-paper URL retained from C26 provenance: https://scholar.harvard.edu/files/weitzman/files/optimalsearchbestalternative.pdf (not newly fetched in this subtask).

**Locations:** “Pandora's Problem,” printed pp. 4–5; reservation equation (7), printed p. 9; selection/stopping rule, printed p. 10.
**Assumptions:** independent box rewards; known reward distributions; fixed opening costs and delays; exact revelation; recall; terminal reward is the maximum revealed reward and fallback.
**Reusable computational form:** with zero delay/no discount, Eq. (7) specializes to d_a = E[(U_a-z_a)_+]. Pandora selects the highest remaining reservation value and stops when the incumbent dominates all remaining reservation values. This displayed specialization is algebra, not a quotation of the discounted formula.
**C27 boundary:** reuse as the independent control/reference. Under a correlated joint model, an observed box changes beliefs about other boxes; the independence premise of this theorem is absent. The theorem therefore supplies no fixed-index optimality guarantee for C27.
**Evidence:** Exa fetched the original report text. Existing remote PDF /tmp/quartz_c26_weitzman1978.pdf was checked: 975398 bytes; SHA-256 546b8b0344acd58b96e2ea9ab753a3b70dd87803d9ecd4d7b69b7fdb15373303.
**Status:** literature-supported; no new optimality proof.

## S2 — Hay et al.: the general metalevel decision problem

Nicholas Hay, Stuart Russell, David Tolpin, Solomon Eyal Shimony, Selecting Computations: Theory and Applications, UAI 2012.
Version read: arXiv:1207.5879v1 (2012-07-25).
URLs: https://arxiv.org/html/1207.5879v1 ; https://aima.cs.berkeley.edu/~russell/papers/uai12-meta.pdf
Proceedings URL already used by C26: https://www.auai.org/uai2012/papers/123.pdf

**Locations:** §2, Definitions 1 and 3; Theorem 4; Definition 6 and Theorem 7.
**Exact ingredients:** Definition 1 explicitly uses jointly distributed utilities and computational outcomes. Definition 3 sets T(s,E,s') = P(E=e | earlier outcomes), computation reward -c, and STOP reward max_i E[U_i | s]. Theorem 4 writes policy value as E[-c N + max_i E[U_i | S_N]]. Definition 6 is the one-further-computation approximation. Theorem 7 says that when this myopic policy computes, the optimal policy also computes, under that decision-problem contract.
**C27 boundary:** joint dependence is already admitted. Quartz's STOP=b is a restricted terminal-action contract: only revealed candidates may be deployed. Variable d_a(h), remaining-query caps, and that restriction require writing the appropriate MDP rather than quoting Hay's constant-cost theorem verbatim. Use the framework; do not assert automatic transfer of every theorem.
**Status:** literature-supported framework; Quartz specialization derived separately.

## S3 — Chawla et al.: correlated Pandora and adaptivity

Shuchi Chawla, Evangelia Gergatsouli, Yifeng Teng, Christos Tzamos, Ruimin Zhang, Pandora's Box with Correlations: Learning and Approximation, FOCS 2020.
Version read: arXiv:1911.01632v2, dated 16 April 2020 in the PDF.
URLs: https://arxiv.org/pdf/1911.01632v2 ; https://doi.org/10.1109/FOCS46700.2020.00116

**Locations:** §1 pp. 1–2; §2 Definition 2.1 pp. 5–6; Theorem 3.4 p. 8.
**Assumptions/objective:** minimization of selected revealed cost plus probing time, arbitrary correlated scenario distribution available through samples. Fully adaptive policies adapt both probing order and stopping; partially adaptive policies fix order and adapt stopping.
**Specific theorem:** Theorem 3.4 converts an alpha-approximate scenario-aware partially adaptive algorithm into a partially adaptive algorithm with ratio e/(e-1)(1+epsilon)alpha and polynomial n,1/epsilon complexity under its stated premise.
**C27 boundary:** supports distinguishing fixed ordering from outcome-dependent ordering. The introductory arbitrary-mapping/sample-access obstruction concerns an unrestricted distribution class and an approximation benchmark; it is not a no-go for exact enumeration of Quartz's small, explicit joint model. Its minimization ratios and PA benchmark are not transferred to Quartz reward-minus-cost performance.
**Status:** literature-supported model distinction; no approximation/hardness claim made for C27.

## S4 — Frazier, Powell, Dayanik: correlated information value

Peter Frazier, Warren Powell, Savas Dayanik, The Knowledge-Gradient Policy for Correlated Normal Beliefs, INFORMS Journal on Computing 21(4), 599–613 (2009), DOI 10.1287/ijoc.1080.0314.
Published author-hosted PDF read:
https://optimallearning.princeton.edu/Papers/Frazier%20Powell%20Dayanik%20CorrelatedKnowledgeGradientJOC.pdf
Author manuscript crosscheck:
https://people.orie.cornell.edu/pfrazier/pub/CorrelatedKG.pdf

**Locations:** §2.1 Eqs. (5)–(6), p. 602; §3 Eq. (11), p. 603; Remark 1 after Eq. (11).
**Assumptions:** correlated multivariate-normal prior over unknown alternative means; conditionally independent normal measurement noise of known variance; finite measurement budget; final choice by largest posterior mean.
**Equations in clear notation:** after measuring x with noise variance lambda_x,
mu' = mu + Sigma e_x (y-mu_x)/(lambda_x+Sigma_xx);
Sigma' = Sigma - Sigma e_x e_x^T Sigma/(lambda_x+Sigma_xx).
Eq. (11) maximizes E[max_i mu'_i]-max_i mu_i. It is optimal when only one measurement remains.
**C27 boundary:** demonstrates that a measurement updates other alternatives and that information value is evaluated through the eventual decision. Gaussian updates and one-step optimality do not transfer to categorical exact reveals, paid stopping, or a known-only terminal set. Use as conceptual/mathematical precedent, not as C27's algorithm.
**Status:** literature-supported; C27's categorical update derived below.

## Derived C27 mapping: no novelty claim

### 1. Joint latent model and all available conditioning

Let X include all free cues available to the controller; let Z have K components. A proposed model is
q(U_1,...,U_m | X) = sum_z w_z(X) product_a q_z(U_a | X).
Using q_z(U_a | x_a) is an additional modeling restriction, not a consequence of Bayes' rule.
Given exact observed history h={(a_s,u_s)}, conditional independence given Z,X yields

w_z(h) = w_z(X) product_s q_z(u_s | X,a_s)
         / sum_k w_k(X) product_s q_k(u_s | X,a_s),

q(U_b=u | X,h) = sum_z w_z(h) q_z(u | X,b), for unobserved b.

Initial incumbent observations and already-known terminal values belong in h before planning. Every exact value is included once. Log-space updates/normalization and a declared zero-evidence behavior are implementation requirements; support smoothing changes the fitted model and must be frozen before TEST.

If query actions are functions only of available X,h and fresh randomization independent of Z conditional on X,h, their action-selection factors cancel from the posterior ratio. This is a direct factorization argument. If the acquisition procedure uses unrecorded information about U or Z, that cancellation does not apply. Model fit and calibration under the real collection/selection process remain empirical questions.

### 2. Correct finite-horizon target

Let b_h be the best known deployable reward, S_h the remaining admissible exact-reveal queries, and d_a(h) a utility-denominated charge. Under that terminal-action contract,

V_0(h)=b_h,
V_H(h)=max{b_h, max_{a in S_h}[-d_a(h)+sum_u q(U_a=u | X,h) V_{H-1}(h union {(a,u)})]}.

This is finite backward induction. H=2 is exact only for the specified model and a decision problem with at most two additional queries. Replanning a two-step horizon repeatedly while permitting more than two total queries is a receding-horizon policy, not an exact global H=2 optimum. Cost already incurred in reaching h is sunk; future comparison still needs all future query/control charges under a consistent accounting convention.

For H=1, pure information about other candidates cannot increase the known-only terminal reward if no second reveal is allowed. H=2 is the first horizon at which revealing an initially unattractive candidate can be valuable by redirecting the next reveal. This follows from the terminal set and recursion; it is not a claim that any particular Quartz case exhibits this effect.

### 3. Product controls and their different estimands

**Primary PRODUCT_H2:** construct the product of joint-model marginals at the common public-cue prior, before the paid initial reveal:
q_prod(U | X) = product_a q_joint(U_a | X).
The joint controller propagates information from both the first exact incumbent reveal and later reveals; the frozen product controller updates the revealed candidate itself but not others. With the common cue prior, charges, admissible actions, terminal rule, and tie-breaking, this is a legitimate combined contrast for information propagation at the initial and later stages. Posterior marginals differ after the initial reveal by design.

**Secondary INITIAL_PROJECTED_H2:** after the common initial exact observation h0, form
q_projected(U_remaining | X,h0) = product_a q_joint(U_a | X,h0),
and freeze cross-candidate updating thereafter. Comparing joint H2 with this controller isolates information propagation from subsequent queries, holding the initial conditional marginals equal.

These controls answer different questions. Neither is the sole legitimate baseline. Record the projection time and avoid attributing the primary combined effect solely to later-query information. Separately fitted marginal and joint models add a model-fitting difference and are a further ablation.

### 4. Selection-conditional calibration: exact statement and limitation

For the true predictive probability p_a,t(u)=P(U_a=u | F_t) and any F_t-measurable selection indicator S_a,t,
E[S_a,t (1{U_a=u}-p_a,t(u))]=0,
by conditional expectation. Here F_t contains the full free-cue vector, observed history, and permissible policy randomization.

A fitted predictor q does not inherit this property merely because it is marginally calibrated across all candidates, or because a latent-mixture model is used. The proposed conditional diagnostics should therefore stratify or condition on the decision-relevant history (including incumbent and selection); finite-sample power and strata construction must be stated. Predictive improvement, joint calibration, and policy net utility are separate empirical outcomes.

## Source-to-target claim gate

| Target | Support | Current status |
|---|---|---|
| Independent fixed reservation ordering | S1 under its assumptions | literature-supported reference |
| History-conditioned transition kernel | S2 Def. 3 | literature-supported |
| Correlated observation can change later query choice | S3 adaptivity distinction + direct joint update | literature-supported mechanism; no Quartz performance claim |
| One observation can change other alternatives' posterior | S4 Eqs. 5–6 and direct latent derivation | literature-supported precedent / derived categorical form |
| Tiny latent-mixture posterior | Bayes factorization above | derived conditional on proposed model |
| H2 Bellman controller | backward induction above | derived for model/cap; implementation unverified by this agent |
| Selection-conditional residual identity | conditional expectation above | derived for true conditional law; fitted-law validity unresolved |
| C26 over-continuation is caused by dependence | no identification yet | conjectural |
| C27 improves net utility or generalizes to more games | no execution in this subtask | unresolved |

## Search and verification receipt

Skills read: physmath-research-loop 1.3.0 SKILL.md, THEORY_LOOP.md (R02), POLICY_INTEGRATION.md, SUBAGENT_MODE.md; Exa Search with searching/source-quality/papers references; Parallel web-search.
Exa: one paper-focused discovery query (requested 5 results); fulltext fetches for Hay, Chawla, Weitzman and Frazier et al. Four targeted primary documents inspected, not an exhaustive review.
SciSpace: one query, 8 returned discovery records, including Hay and correlated ranking-and-selection work. SciSpace summaries were not used as proof of equations.
Parallel: two primary-source searches, metalevel MDP and correlated KG, session session_96e5c0a3bc00e0a9733df91d4c81e4f1.
Search-service: opened original arXiv/author PDFs and inspected targeted theorem/equation text; published KG PDF pages 602–603 additionally requested as screenshots.
One endpoint failure: search-service opening OSTI PDF returned HTTP 502. Alternative Exa extraction succeeded and the already available remote PDF's SHA-256 matched the C26 provenance.
Extraction limitations: initial Exa fetches capped each text at 32000 characters; long displayed tool output was truncated in the interface. Only located passages are claimed as inspected. KG plain-text extraction loses some symbols; Eq. (11) was independently recovered from Parallel's author-PDF excerpt.
No environment, permission, calculation, or implementation failure was observed because no research code was executed here.
No state-of-the-field priority or latest-best bound claim is made. A 2025 correlated-Pandora approximation paper appeared in discovery but was not required for this mapping or used as a premise.
No independent calibration theorem was imported: the limited conditional-expectation identity is derived above.

## Web citation handles for the parent

Parent must reopen/search these sources before final citation:
- Hay original HTML: turn659083251view0 / Theorem 4 find turn457162856view3.
- Chawla original PDF: turn659083251view1 / targeted findings turn457162856view1.
- Published correlated KG PDF: turn457162856view0.
- Cornell author manuscript: turn659083251view2.
- OSTI failed web handle turn659083251view3 must NOT be cited as successful retrieval; use the URL plus verified original-PDF provenance or reopen successfully.

Final review owner: parent/root. Suggested R02 conclusion: accept these four references; move to direct derivation and the smallest discriminating exact example, with model and horizon boundaries recorded.
