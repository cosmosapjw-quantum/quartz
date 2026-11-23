# C28 R06 independent mathematical audit

## Normalized claim and primary verdict

**Primary verdict: correct only after a stated restriction.**

The audited core claim concerns a finite candidate set, utility support contained in `{0, 1/2, 1}`, exact revelation, nonnegative deterministic query costs, a terminal choice restricted to verified candidates, no repeated queries, a moving verified incumbent, and at most two remaining queries. With fixed base distributions, signed loadings `|r_i|<1`, costs, incumbent and remaining set, the raw proof's posterior formula, H1 score and retention certificate, exact H2 lift, convex piecewise-affine values, symmetric-family cost partitions, numerical examples, information-bijection equality, and reference belief threshold are correct.

The literal strict covariance-sign sentence needs the restriction stated below. The finite-plan argument proves that grouping plans by first action yields a union of intervals; it does not itself exhibit a disconnected region in this particular model. The fully warranted replacement is: **convexity alone does not imply connected first-action regions; no connectedness assumption is used.** No search for an optional disconnectedness example was performed.

This verdict is for the immutable raw proof identified below. The parent's reported correction to its canonical manuscript was not read or independently certified in this pass.

## Claim card and provenance

- Review mode: independent fresh subagent; one bounded R06 pass, without recursive review.
- Input actually read: `/workspace/scratch/486e36161bdd/c28_theory/RAW_PROOF_FOR_REVIEW.md`.
- Raw proof SHA-256: `ff46d109c223d08dd4fd77df5b1de04a9dab4f8e7b216634553ece89e23c94d4`.
- Supplied source identity: Quartz commit `87c15dc828d6eea519c219fd8a412fa21dfc464d`.
- Supplied manuscript identity: C28 commit `cf306571c5e7a701cdbd0b6cc85ffa8260c270ff`.
- The two commit identities are provenance supplied in the task packet; this audit binds its findings directly to the raw-proof bytes, not to an independently inspected Git tree.
- Constructor verdict, constructor result JSON, and parent history were not read. No repository source was modified. No external model/API calls or purchases were made.
- Arithmetic domain for exact checks: Python `fractions.Fraction`; mutual information alone uses floating-point logarithms.
- Utility and query cost have the same units. Belief, loading, normalized feature, posterior odds and information in nats are dimensionless. No physical-equivalence claim is involved.

## Dependency graph

1. Normalized base probabilities and bounded centered features imply normalized positive conditional factors on the operative support.
2. The two-component product mixture gives marginal likelihoods and supported-observation Bayes updates. Products of likelihood factors establish order independence.
3. Conditional independence plus total covariance gives equation (4), including zero covariance at certain-latent endpoints and for degenerate marginals.
4. Known-only terminal deployment gives the positive-part improvement payoff. Its expectation gives H1 affine scores at a fixed state.
5. H1 affine score differences give the exact retention inequalities and interval optimality regions.
6. Exact first-query revelation updates both incumbent and remaining set. Applying H1 to that updated state gives equation (9).
7. Positive supported-branch likelihood factors distribute through the maximum and cancel the posterior denominator, giving equation (10).
8. A finite maximum of affine functions and nonnegative weighted sums give convex piecewise-affine H2 values. Finite contingent plans give a second, equivalent proof.
9. The symmetric three-candidate construction specializes these formulas to the complete score table. Comparisons with every target and STOP give equations (12)–(13).
10. The target-coordinate bijection proves equal entropy and mutual information, while the unchanged numerical payoff rule explains differing decision values.
11. Specializing the exact lift to the reference contrast model gives the continuation kinks and first-action crossing in equations (16)–(18).

All load-bearing leaves above are finite internal algebra or definitions. The POMDP/Blackwell historical attribution is not needed for these finite derivations and was not independently literature-audited.

## Obligation matrix

| Obligation | Status | Evidence and scope |
|---|---|---|
| Conditional-factor normalization and positivity | passed | `E_p f_i=0`, `|r_i f_i|<1`; zero base probabilities remain outside support. |
| Point masses and zero probabilities | passed | `s_i=0` gives `f_i=0`, no latent information; unsupported observations are not repaired by Bayes' formula. |
| Bayes formula and endpoints | passed | Denominator is positive for supported observations even at `m=±1`; the endpoint remains fixed. |
| Observation-order independence | passed | Multiplicative latent likelihoods commute; exact two-observation checks agree with direct likelihood normalization. |
| Off-diagonal covariance formula | passed | Exact derivation and independent full-world covariance checks agree. |
| Strictly positive product of three covariance signs from nonzero loadings alone | failed | Certain-latent endpoints or point masses yield zero covariances. Minimal restriction given below. |
| H1 interval geometry | passed | Each competitor inequality is affine; deterministic tie priorities merely select open/closed endpoints. |
| Exact H1 action-retention certificate | passed | For each competing action `j≠k`, the score gap is exactly `gamma_kj + (beta_k-beta_j) Delta m`; tie precedence must be applied. |
| H1 same-state limitation | passed | The statement explicitly holds `p,r,d,b,S` fixed. It does not carry across a reveal that changes `b` or `S`. |
| H2 no-repeat and moving incumbent | passed | Continuation uses `S\{i}` and `max(b,u)`; these are also imposed in the independent complete-world calculation. |
| Exact H2 denominator cancellation | passed | The factor is positive before moving it inside the branch maximum; no approximation is introduced. |
| Empty remaining set, one candidate, and `b=1` | passed | Empty continuation maximum is zero; at `b=1`, all scores are nonpositive and STOP wins the prescribed zero-cost ties. |
| Convex piecewise-affine fixed-first-action and root values | passed | Finite maxima and nonnegative sums, or finite contingent-plan enumeration. |
| Existence of disconnected first-action regions in this exact model | not addressed | The raw union-of-plan-interval argument establishes no connectedness inference, but contains no concrete model witness. The weaker stated replacement is sufficient for the controller. |
| Common/contrast score table and all first alternatives | passed | STOP and both payoff-relevant first queries are included, not only the information-only query. |
| Strict cost regions and boundary ties | passed | Direct piecewise comparisons plus the complete score table; checks include `r=0`, zero costs, `x=0`, `x=±y`, `3x+y=0`, and equality at the A-comparison boundaries. |
| Product and zero-cost limits | passed | Values reduce to the expressions given in the proof; zero-cost equalities use STOP-first tie handling. |
| Equal-information bijection and interpretation | passed | Exact worldwise equality under `C→1-C`; utility is not relabeled, so the paid decision problem changes. |
| Reference belief kinks and crossing | passed | Exact analytic reconstruction below and 193 exact belief states. |
| General speedup, empirical loading signs, novelty, physical equivalence | out of scope | None follows from the audited algebra, and this review makes none of those claims. |

## Decisive derivations and finding

### 1. Covariance restriction: actual counterexample and smallest repair

For three candidates, take nondegenerate Bernoulli base probabilities `p_i(0)=p_i(1)=1/2`, loadings `r_i=1/2`, and `m=1`. Every loading is nonzero, but the latent variable is certain and the three candidates are independent conditional draws. Hence all three off-diagonal covariances are zero, and the product of their signs is zero, not positive.

More generally, the product of the three off-diagonal covariances is

`(r_1 r_2 r_3)^2 (delta_1 delta_2 delta_3)^2 (1-m^2)^3 >= 0`.

It is strictly positive precisely when all three loadings and all three marginal variances are nonzero and `|m|<1`. Equivalently, strict positivity holds when all three pairwise covariances are nonzero. This local restriction preserves the substantive statement that the family cannot realize three strictly negative pairwise covariances. The exact covariance formula, H1/H2 equations and phase calculations are unaffected.

Finding classification: a real endpoint/degeneracy error in one sentence; minimal forced local correction, not a controller blocker or a reason to reopen the construction.

### 2. H2 from the actual terminal decision problem

For a complete utility world `u`, a first query `i` produces incumbent `b_i=max(b,u_i)` and incurs `d_i`. For each first observation, the expected gain of continuing with `j≠i` is the branch-weighted expectation of `(u_j-b_i)_+ - d_j`. Maximizing this branch value against zero and summing over first observations gives equation (9), with the direct first gain outside the continuation sum. Multiplication of every branch alternative by the strictly positive likelihood `1+m t_iu` gives the affine expression

`(1+m t_iu) alpha_j(b_i) + (m+t_iu) beta_j(b_i)`.

This reconstructs equation (10) directly, with both the moving incumbent and the no-repeat restriction present. The independent checker uses complete-world joint probabilities and branch sums; it does not use the posterior update in its raw-world evaluation.

### 3. Cost partitions and ties

For the common model, when `-y<x<y`, the A gain is `(x+y)/2-c` and the target gain is `x`. Being strictly above both a target and STOP is therefore equivalent to `2c+|x|<y`. When `x>=y`, a target weakly dominates A; when `x<=-y`, STOP weakly dominates A. These cases cover the whole allowed domain `x<=1/4`, `0<=y<1/4`, `c>=0`.

For the contrast model, A can have positive gain only if `x+y>0`, in which case its gain is `x+y-c`. The target gain is `(3x+y)/2`. Comparing A with STOP and the target gives exactly `c<min(x+y,(y-x)/2)`. The target is positive exactly when `3x+y>0`.

At equalities, the raw score table is essential. For example, `c=0` and `x<-y` produces an A/STOP tie even though the strict-A boundary `2c+|x|=y` is not met. At `x=0`, target/STOP ties occur over a region of costs. These cases are correctly resolved by the score table together with STOP before A before B before C; the strict inequalities alone are not the entire tie specification. The raw proof explicitly retains that score table and tie rule, so this is a clarification, not a defect.

### 4. Belief threshold reconstructed without a posterior-change proxy

For the reference contrast model, the fixed first-target scores reduce to

`K_B(m) = -1/20 + m/5 + (1/2)[11/100 - 4m/25]_+`,

`K_C(m) = -1/20 - m/5 + (1/2)[11/100 + 4m/25]_+`.

The information-only score is

`K_A(m) = -1/100 + (1/2)[-1/20-m/25+|m+4/5|/5]_+ + (1/2)[-1/20+m/25+|m-4/5|/5]_+`.

These expressions give the kinks `|m|=11/16` and `|m|=7/8` in equation (16), and the target expression in equation (17). In the intermediate interval the A-minus-best-target gap is `19/200 - 3|m|/25`; it vanishes at `|m|=19/24`. The common winning gain is `13/120`. A precedes either target in the stated tie order, so both crossing endpoints belong to A's selection region. At `m=±1`, A has gain `7/50` and the favorable target has gain `3/20`, as expected when the latent state is already known.

### 5. Information equality

The checker verifies world by world that the contrast mass at `(a,b,c)` equals the common mass at `(a,b,1-c)`. Therefore both the target-pair entropy and its entropy conditional on A are unchanged. Direct summation gives, in both cases, `I(A;(B,C)) = 0.284656509301053` nats at the reference parameters, consistent with the displayed approximation. The numerical utility of C remains its observed value; carrying the distribution through a coordinate bijection without carrying the payoff through the same map changes the decision problem. Thus the differing paid values do not purport to compare two experiments under an identical relabeled payoff function.

## Independent computations and actual commands

Checker: `/workspace/scratch/486e36161bdd/c28_independent_math_check.py`.

Checker SHA-256: `b15f1cb55f50455fff835895ba72e8e0a34004589379394b67a3a309b2a1a794`.

The final executed command was:

```bash
python /workspace/scratch/486e36161bdd/c28_independent_math_check.py
```

Final exit code: **0**. Output:

```text
RANDOM_EXACT_MODELS 120
EXACT_COST_PARTITION_CASES 290
EXACT_BELIEF_STATES 193
REFERENCE_SCORES -1 ['7/50', '-23/200', '3/20']
REFERENCE_SCORES -19/24 ['13/120', '-9/100', '13/120']
REFERENCE_SCORES 0 ['1/10', '1/200', '1/200']
REFERENCE_SCORES 19/24 ['13/120', '13/120', '-9/100']
REFERENCE_SCORES 1 ['7/50', '3/20', '-23/200']
MUTUAL_INFORMATION_NATS 1 0.284656509301053
MUTUAL_INFORMATION_NATS -1 0.284656509301053
ENDPOINT_COVARIANCE_SIGN_COUNTEREXAMPLE ['0', '0', '0']
ALL_ASSERTIONS_PASSED
```

The 120 seeded rational model cases use candidate counts 0–4, zero-probability support entries and point masses, signed or zero loadings, endpoint/interior beliefs, grid/off-grid incumbents, and nonnegative costs. They compare every H2 first-action score with independently summed complete-world branches and also check off-diagonal covariance and two-observation posterior order independence. These are bounded computational checks, not exhaustive proofs of all admissible inputs; the universal conclusions rest on the analytic arguments above.

The first checker invocation exited **1** during the cost-table comparison because an expected-value helper used integer zero with `/2`, introducing a floating zero into otherwise rational arithmetic. The helper was locally corrected to use `Fraction(0)` throughout; the unchanged model formulas then passed the full check. This was a checker implementation issue, not a mathematical counterexample. It is recorded rather than discarded.

## Exact remaining scope and cheapest next action

The core finite-model proof requires only the covariance strictness restriction identified above. For the H2 region wording, retain the demonstrated union-of-plan-interval statement and say that convexity alone gives no connectedness guarantee; an existential claim of a disconnected region in this exact family would need a separately supplied valid witness. No such claim is required by the formulas, finite controller or reference family.

The smallest action is to apply those wording restrictions in the canonical manuscript. No repeated numerical campaign or broader search is warranted by the findings of this pass. This review does not certify the parent's subsequent edit or its separate runtime results.

## skill_context_used

Actually loaded and applied:

1. `skill://mathbox@openai-curated-remote/home/agent/.codex/plugins/cache/openai-curated-remote/mathbox/4.0.0/skills/proof-audit/SKILL.md`.
2. The same package's `references/obligation-checklists.md`, using only general logic/typing, computation-dependent claims, and applicable symmetry/external-dependency obligations. Graded, topological and spectral-sequence checklists were inapplicable.
3. `skill://physmath-research-loop@created-by-me-remote/home/agent/.codex/plugins/cache/created-by-me-remote/physmath-research-loop/1.3.0/skills/physmath-research-loop/SKILL.md`.
4. The same package's `references/POLICY_INTEGRATION.md`, `references/THEORY_LOOP.md` (R6), `STAGE_SELECTION.md`, and `references/SUBAGENT_MODE.md`.

Applied behavior: exact selected-stage scope; reconstruction from definitions; boundary and degenerate checks; distinct analytic and finite-computational evidence; one bounded independent pass; no recursive review; a concrete counterexample and smallest correction; no invented blocker, performance, novelty or physical-equivalence claim.
