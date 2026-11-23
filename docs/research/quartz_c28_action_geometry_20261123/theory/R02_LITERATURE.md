# Quartz C28 R2 — literature transport for action-sensitive belief geometry

**Research anchor:** `cosmosapjw-quantum/quartz` commit `87c15dc828d6eea519c219fd8a412fa21dfc464d` (the user-selected C26 SSOT). This lane is read-only with respect to the workstation and Git repositories. It supplies source-based mathematics to the parent; it neither validates C27 campaign results nor changes the next-cycle empirical contract.

**Finding:** finite-horizon belief-space piecewise linearity, policy partitions, and decision-specific value of information are established results. C28's useful contribution can be the explicit signed latent construction, a correctly specialized two-query formula, exact discriminating examples, and its measured implementation. This search does not establish publication novelty.

## 1. Primary sources actually read

### S1. Kaelbling, Littman, and Cassandra (1998)

*Planning and acting in partially observable stochastic domains*, Artificial Intelligence 101, 99–134. DOI: `10.1016/S0004-3702(98)00023-X`.

Author-hosted full text: <https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf>.

**Checked location:** §4.1, especially journal pp.109–112; the unnumbered equations on p.110 give the value of a fixed policy tree as a belief-weighted state-value vector, followed by the maximum over policy trees. The resulting upper envelope and policy-region projection appear on pp.110–112. §4.4 p.114 explicitly states the analogous property for fixed-first-action Q functions.

**Transport:** keep the latent law and initial observed state fixed and vary the two-state prior, represented by `m`. Every finite contingent policy then has value affine in `m`; maximizing yields convex piecewise-affine V and Q functions. The result depends on a finite policy family and a fixed reward/transition contract.

**Non-transfer:** the paper does not prove that C28's special evaluation formula is faster, that fitted beliefs are accurate, or that each first-action label has one connected optimal interval. A first action may correspond to several contingent policy trees.

### S2. Hay, Russell, Tolpin, and Shimony (2012)

*Selecting Computations: Theory and Applications*, UAI 2012, 346–355.

Author-hosted full text: <https://aima.eecs.berkeley.edu/~russell/papers/uai12-meta.pdf>.
Proceedings record: <https://proceedings.mlr.press/r10/hay12a.html>.

**Checked location:** §2, Definition 3, PDF p.3: computational outcomes form the state; computation costs `c`; stopping returns `max_i E[U_i|h]`. Definition 6 and Theorem 7, PDF p.4, define the myopic approximation and its relation to optimal stopping. The paper also explains how a finite budget can enter the state after Definition 3.

**Transport:** computations are chosen for downstream decision improvement after charging their cost, and one-step stopping can miss multi-computation opportunities. This supports explicit Bellman reasoning rather than an entropy proxy.

**Non-transfer:** the source permits selection of an unopened action by posterior mean. Quartz's terminal set is restricted to exact-known actions and STOP equals the known incumbent `b`; its Bellman equations must be rederived with that terminal condition. Independent-action simplifications in the paper do not automatically apply to a shared latent mode. Modeled query costs also do not constitute a measurement of planner wall time.

### S3. de Oliveira (2018)

*Blackwell's informativeness theorem using diagrams*, Games and Economic Behavior 109, 126–131. DOI: `10.1016/j.geb.2017.12.008`.

Author-hosted preprint, dated 2017-11-30: <https://henriquedeoliveira.com/wp-content/uploads/2020/12/blackwells-informativeness-theorem-using-diagrams.pdf>.

**Checked location:** Theorem 1, preprint p.2; stochastic-map composition in §2 p.3; expected-utility functional in §3 p.4; proof in §4 pp.4–5. All sets are finite under the paper's introductory convention. The theorem equates garbling, inclusion of feasible conditional action distributions, and preference in every Bayesian decision problem.

**Transport:** for a common underlying state and payoff/action contract, an observer can ignore or garble extra information. This is the appropriate all-decision-problems comparison and explains why posterior information has value only through possible decisions.

**Non-transfer:** Blackwell dominance does not say that paying for more information is profitable, or that a particular decision must change. Query costs, horizon, and available actions still matter. It also does not compare changed utility meanings merely because the resulting distributions are related by a coordinate permutation.

### S4. Rauh, Banerjee, Olbrich, Jost, Bertschinger, and Wolpert (2017)

*Coarse-Graining and the Blackwell Order*, Entropy 19(10), 527. DOI: `10.3390/e19100527`.

Full manuscript checked: <https://arxiv.org/pdf/1701.07602>, version `1701.07602v3`, dated 2017-11-10. Publisher: <https://www.mdpi.com/1099-4300/19/10/527>.

**Checked location:** Introduction pp.1–2, Theorem 1 p.2, and §II Example 1 pp.4–5. The introduction distinguishes Blackwell ordering from mutual-information ordering, even when the latter is required for every input distribution. Example 1 has equal mutual information but different decision performance; §IV p.9 explicitly contrasts it with another example.

**Transport:** equal or greater scalar mutual information does not alone determine a fixed downstream decision's value. A useful Quartz counterexample can hold the informational quantity fixed while changing alignment with the selectable payoffs.

**Non-transfer:** that establishes a limitation of a scalar criterion, not a global impossibility result for information-based methods, latent models, or resource-rational search. C28 should report its own exact finite example and the precise payoff mapping.

## 2. Historical attribution and retrieval limitations

The original finite-horizon result is Smallwood and Sondik (1973), *The Optimal Control of Partially Observable Markov Processes over a Finite Horizon*, Operations Research 21(5), 1071–1088, DOI `10.1287/opre.21.5.1071`: <https://pubsonline.informs.org/doi/10.1287/opre.21.5.1071>. Its publisher abstract was verified, but its complete proof was not retrieved in this lane. S1 supplies the directly inspected policy-tree derivation, so no uninspected theorem number is attributed to the 1973 text.

Blackwell (1953), *Equivalent Comparisons of Experiments*, Annals of Mathematical Statistics 24(2), 265–272, DOI `10.1214/aoms/1177729032`, was verified at the publisher-record level. The direct publisher PDF extraction returned a network error; an alternate fetch returned only metadata/abstract. S3 supplies an actually read original proof of the finite theorem. The 1953 proof is therefore not reported as directly read.

## 3. Exact source-to-Quartz dictionary

The following is a project-specific translation, not a quotation from a source.

One local source-notation correction is needed if S3's proof is reused literally: its preprint p.4 forward implication prints the composition in reversed order. For `gamma:S -> S'` and `alpha':S' -> A`, the well-typed strategy is `alpha = alpha' composed with gamma`. The page image was checked; the theorem and its simulation argument remain valid after this elementary correction.

For fixed candidate marginals, signed loadings, costs, known incumbent and remaining set, write the latent prior as

\[
w_+=(1+m)/2,\qquad w_-=(1-m)/2,\qquad -1\leq m\leq1.
\]

The observable state must include the remaining candidates and incumbent; `m` is only the unresolved latent belief coordinate. It is not sufficient after silently changing those observable quantities. A fixed contingent policy's conditional values `v_+` and `v_-` give

\[
V_\pi(m)=\tfrac12(v_++v_-)+\tfrac m2(v_+-v_-).
\]

This is the explicit bridge from S1 to C28. The two-query Bayes-denominator cancellation proposed by the parent is an algebraic specialization of the same expectation identity. Its normalization, support, moving incumbent, no-repeat rule and tie handling still require the project's direct derivation and implementation checks.

At H1, each action score is affine, so the set where a fixed action dominates all others is an interval (possibly empty or a point). At H2, convex piecewise-affine scores do not imply monotone differences between actions. Several separate intervals can share the same first-action label. A plot must not infer one threshold per action unless a special-family proof supplies it.

For the proposed common-versus-signed Bernoulli construction, changing `C` to `1-C` is a bijection on the target utility vector and preserves `I(A;(B,C))`. However, selecting the larger numeric utility is not invariant unless the payoff semantics are transformed too. Equivalently, the experiment `A` about latent `Z` can be identical while the downstream target payoffs differ. This is a decision-relevance example consistent with S3, not a violation of Blackwell's theorem.

The actual new-to-project claims should therefore be separated as follows:

| Claim | Evidence class / appropriate support |
|---|---|
| Finite belief-space V and fixed-first-action Q are convex piecewise-affine | Established; S1, with fixed contract |
| Metacomputation has priced decision value and explicit STOP | Literature-supported; S2, with terminal-set correction |
| Scalar information does not determine arbitrary decision value | Literature-supported; S3–S4 |
| Signed latent normalization, posterior update and off-diagonal covariance | Direct project derivation; no novelty implied |
| Explicit lifted H2 formula and family phase boundaries | Direct derivation plus exact finite oracle |
| Reduced native compute time at identical value/decisions | Implementation and timing evidence required |
| Higher actual game utility/Elo or cognitive validity | Unresolved until separately tested |

## 4. Retrieval and skill receipt

- Date inspected: 2026-11-23 (Asia/Seoul).
- Exa: two search calls requesting five candidates each (`sources_reviewed=10` under the skill's counting convention), plus one attempted original Blackwell fetch. Similarity hits were filtered; copied course-material sites and generic tutorials were not used as proof.
- Parallel: three bounded search calls and two full-source fetch batches. Four full research PDFs were successfully extracted. One original Blackwell PDF failed as stated above.
- SciSpace: one six-result scholarly search was executed. Its broad mixed-question results were useful for discovery, but did not directly recover the requested classic proofs; none was promoted to primary proof evidence.
- The four accepted URLs were also opened with the web retrieval tool, and relevant definitions/sections located there, so independent source references exist. No claim depends solely on a generated search summary.
- `skill_context_used`: PhysMath Research Loop 1.3.0 main instructions, THEORY_LOOP R2 and POLICY_INTEGRATION; Exa Search 4.0.1 plus `references/searching.md` and `references/patterns-papers.md`; Parallel web-search 1.2.4. Child aliases were unavailable; full supplied skill locators were resolved successfully.
- Deliverable only: this note and its small source record. No repository mutation, publication, empirical run, fitted parameter, or fresh reviewer was performed by this lane.
