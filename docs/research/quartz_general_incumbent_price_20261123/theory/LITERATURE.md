# Quartz C25 R02 — bounded primary-literature comparison
Date: 2026-11-23 UTC
Owner: native subagent /root/c25_literature. Scope: R02 literature comparison only; no repository edits, publication, or child agents.
Parent-supplied project context: Quartz computation allocation/stopping, originally neural MCTS with CPU and actual game-quality targets; C24 certified SPD utility intervals are cheap, but further computation can worsen the selected action and cost more than its small utility gain at the locked price 10^-6 utility/microsecond.
Evidence status: LITERATURE_SUPPORTED for the paper claims below; DERIVED for the explicit comparison and transport arguments; no experiment or implementation performed by this subagent.

## Main finding

C25 is properly positioned as a deterministic, incumbent-preserving specialization of established computation-value and confidence-separation ideas. Neither pairwise interval dominance nor the rectangular worst-case simple-regret ceiling should carry a novelty claim. The useful project contribution is the correct guarantee contract, its realization with the C24 certified enclosures and a defensible cost model, and eventual validation against the original MCTS CPU/game-quality objective. An expected cost predictor does not furnish the deterministic lower bound needed by the proposed universal stopping argument.

## Four load-bearing primary sources

### S1. Hay, Russell, Tolpin, Shimony (2012), Selecting Computations: Theory and Applications
Primary PDF: https://www.auai.org/uai2012/papers/123.pdf
UAI 2012, pp. 346–355. Page positions below use PDF pages (1-based).
Verified relevant locations: Definition 3, PDF p.3; Theorems 4–5 and Definition 6/Theorem 7, PDF p.4; Section 6.1 and Eq. (11), PDF p.8.
Theorem 4 writes policy value as E[-cN + max_a mu_a(S_N) | S_0=s]. Theorem 5 bounds the optimal expected computation count by VPI(s)/c, with VPI(s)=E[max_a U_a|s]-max_a E[U_a|s]. It bounds expected count, not worst-case count; Example 3 allows arbitrarily long finite computation sequences.
Definition 6 is one-step/myopic computation value; Theorem 7 supports continuation when a myopic computation is worthwhile, without the converse stopping implication.
Section 6.1 compares an intrinsic-VOI upper bound with sample cost, Eq. (11). Section 6 acknowledges adaptive MCTS rollouts can violate stationary sampling assumptions; its tree policy is heuristic.
Reuse: direct prior art for priced computation and upper-bound stopping. Transport requires an explicit uncertainty model and cost in the same utility units; its calibrated sample cost is not a universal wall-clock lower bound.

### S2. Kalyanakrishnan, Tewari, Auer, Stone (2012), PAC Subset Selection in Stochastic Multi-armed Bandits
Primary PDF: https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/ICML12-shivaram.pdf
ICML 2012, pp. 655–662.
Verified relevant locations: Section 3.1, PDF p.3 / proceedings p.657, unnumbered stopping display; Theorem 1 and Eq. (4) on that page.
With h* the smallest lower-confidence bound among the empirical top-m arms and l* the largest upper-confidence bound outside that set, LUCB stops when U_l*(t)-L_h*(t)<epsilon. For m=1, this is a global epsilon-optimality certificate for the empirical leader on the confidence event.
Theorem 1 uses a summable probability bound, sum_t sum_{u<=t} exp[-2u beta(u,t)^2] <= delta/n, to control incorrect returns across adaptively selected times. It is a frequentist probability guarantee, not deterministic containment.
Reuse: confidence separation, independently chosen sampling and stopping rules, and the need for joint/time-uniform validity. C25's pairwise SWITCH condition L_j>U_i compares a challenger with the retained incumbent; LUCB's STOP condition compares all challengers with its empirical leader. They are distinct operations.

### S3. Kaufmann, Cappé, Garivier (2016), On the Complexity of Best-Arm Identification in Multi-Armed Bandit Models
Primary PDF: https://www.jmlr.org/papers/volume17/kaufman16a/kaufman16a.pdf
JMLR 17(1), pp. 1–42.
Verified relevant locations: Section 1, p.2, strategy triple and fixed-confidence/fixed-budget definitions; Eq. (1) defines their respective complexities.
The strategy separates sampling, stopping time, and recommendation. Fixed-confidence identification fixes delta and requires the correct arm/subset with probability at least 1-delta while minimizing expected samples. Fixed-budget identification fixes the draw count and minimizes recommendation-error probability; consistency only requires that error tend to zero as the budget grows.
Reuse: a fixed-budget recommendation contract is not interchangeable with a fixed-confidence stopping contract or a deterministic incumbent-preservation contract. The paper's definitions do not impose sample-path monotonicity of recommended true utility.
Applicability: stationary stochastic arms with specified model class and identifiable best set. Do not transfer sample-complexity constants to deterministic SPD residual contractions or nonstationary MCTS statistics.

### S4. Bubeck, Munos, Stoltz (2011), Pure Exploration in Finitely-Armed and Continuous-Armed Bandits
Primary author-copy PDF: https://chercheurs.lille.inria.fr/munos/papers/files/TCS2011.pdf
TCS 412, pp.1832–1852. DOI 10.1016/j.tcs.2010.12.059.
Verified relevant locations: Section 2, printed p.1834 / PDF p.4 (the PDF includes a publisher cover), unnumbered simple-regret display r_n=mu*-mu_{J_n}; Section 3, Theorem 1, printed p.1835 / PDF p.5.
The paper separates sampling/allocation I_t from recommendation J_t and evaluates the final recommendation through simple regret. Its motivating resource budget includes CPU time. Its cumulative-regret/simple-regret comparison explains why an allocation strategy optimized for cumulative reward is a different objective.
Reuse: C25 G_i bounds simple regret of the retained action, rather than cumulative regret of sampled computations. Expected regret bounds across random histories are not realized-history safety guarantees.

## Direct comparison and transport conditions (DERIVED, not attributed as a new paper theorem)

Let q=(q_1,...,q_K) be fixed real action utilities, with simultaneously valid deterministic enclosures q_a in [L_a,U_a]. Define B=product_a [L_a,U_a], and let i be the retained incumbent.

1. Pairwise switch:
   L_j>U_i implies q_j-q_i >= L_j-U_i>0.
   Applied only to the current incumbent, this makes every accepted switch strictly improve its true utility. It gives no guarantee that useful switches will eventually be certified within a finite budget. If the action utility itself changes during computation, the fixed-q premise has to be restored or the theorem does not apply.

2. Robust regret ceiling:
   G_i=max(0,max_{j != i}(U_j-L_i)).
   For every q in B, max_j q_j-q_i <= G_i.
   The supremum over the full rectangular set B equals G_i; if physical/model correlations restrict the feasible set to F subset B, G_i remains an upper bound but need not be tight. The incumbent's own U_i-L_i must not be mistaken for actionable regret.

3. Bayesian relation:
   Suppose, as an additional model, a posterior over q is supported inside B and i is a Bayes action maximizing E[q_i|s]. Then
   VPI(s)=E[max_j q_j|s]-max_j E[q_j|s]
         =E[max_j q_j-q_i|s] <= G_i.
   For any future information/computation policy, its expected gross decision improvement is at most VPI, provided the final action is selected from the same set and utilities/information are modeled coherently.
   Thus G_i is a distribution-free outer ceiling under support containment; it is not a posterior expectation, a predicted marginal gain, or a claim that any computation can realize that gain.
   Hay et al.'s random utilities U_a should not be identified silently with a deterministic value or a posterior over a mean; the mapping must specify which object is enclosed.

4. Rigorous price stopping:
   Let lambda>=0 be the price in utility/cost-unit. If every allowed nonempty continuation pi and every realization omega has remaining total cost C_pi(omega)>=C_min>0, then relative to stopping now with incumbent i:
   gain_net(pi,omega) <= G_i-lambda*C_min.
   Consequently G_i<=lambda*C_min makes stopping weakly dominate all such continuations.
   C_min must be a universal lower bound over the admissible remaining continuations, not an average, a fitted prediction, an observed historical minimum promoted without justification, or a lower bound only on one chosen next action.
   Costs already spent are sunk; all unavoidable future computation/decision overhead relevant to the comparison belongs in remaining cost. A nonnegative-cost model suffices for the special case G_i=0.
   When no positive floor is justified, a wall-time predictor may still support a heuristic economic rule, clearly separated from the deterministic certificate.

5. Why fixed-budget recommendation does not imply incumbent safety:
   A direct two-arm example has true Bernoulli means (0.6,0.4). Pull arm 1 and see 1, arm 2 and see 0; the empirical leader is arm 1. Then obtain arm 1:0 and arm 2:1,1; means become 1/2 and 2/3, so empirical-best recommendation switches to the worse arm. This history has positive probability 0.6*0.6*0.4*0.4*0.4=0.02304. Simple regret grows from 0 to 0.2 on that history. This is a constructed illustration, not an empirical result or an assertion quoted from S3/S4.

## Novelty and claim boundaries

- Established ingredients: computation value minus cost; perfect-information upper bounds; confidence-bound separation for identification; simple regret as terminal recommendation loss.
- Elementary direct derivations: pairwise deterministic dominance and rectangular worst-case regret G_i. State them as lemmas/contract, not a new general theory.
- C25-specific useful result: prevent C24's observed selected-action regression by retaining the incumbent unless a strict certified improvement is available; attach a correct sufficiency condition for price stopping.
- Still unresolved by this literature task: actual certificate correctness and numerical rounding in C24; whether a deterministic positive cost floor is available; empirical cost/quality advantage under the original neural-MCTS target. SPD surrogate performance does not establish game win-rate or original CPU target completion.
- Avoid inferring that a nonpositive one-step estimated VOC proves every adaptive multistep continuation unprofitable. S1's Theorem 7 supports only the forward continuation implication; the global G_i/cost-floor proof has a different basis.

## Actual tool execution provenance

All listed services were actually invoked; none is simulated.
- Exa: web_search_exa, 1 call, SUCCESS; located Hay et al. primary PDF and arXiv material. web_fetch_exa, 1 call, SUCCESS; retrieved the Hay primary PDF text (about 43,424 characters).
- SciSpace: search, 2 calls, SUCCESS; first found fixed-confidence/fixed-budget papers, second found the original LUCB paper. extract_data, 1 call, response SUCCESS but EXACT FORMULA/PAGE EXTRACTION INSUFFICIENT: it explicitly reported missing/corrupted LUCB stopping display and unavailable exact locations. Those fields were not treated as verified evidence; direct primary PDFs resolved them.
- Parallel: web_search, 1 call, SUCCESS; complementary search located Bubeck–Munos–Stoltz author PDFs and fixed-budget papers. search_id=search_ba7947d0ca3d312d0ef25644b7fa5d34; session_id=session_ba7947d0ca3d312d0ef25644b7fa5d34. No Parallel Research run or Fetch call was made.
- Primary verification: search_service system1 search for author/proceedings/JMLR sources, followed by direct PDF open/find/read. Four PDFs and the above formula locations were read. Exact-string finds for LUCB "Algorithm 1" and "stopping condition" returned no match; Section 3.1 direct reading supplied the actual unnumbered stopping display.
- Root must open these URLs before citing the sources with its own web references. Subagent web refs: Hay turn384840500view0 / turn559984856view1 / turn559984856view2; LUCB turn384840500view1 / turn559984856view0; KCG turn384840500view2; BMS turn384840500view3 / turn480520976view6.

## skill_context_used

Actually read physmath-research-loop 1.3.0 from cloud package c43/physmath-research-loop:
- skill://plugins_6acdfbba921881919e0c17843706aef4/physmath-research-loop/SKILL.md
- same package references/THEORY_LOOP.md (R02 contract: primary sources, exact formula/theorem location, reuse conditions)
- same package references/POLICY_INTEGRATION.md (bounded objective, real evidence, no review recursion)
- same package references/SUBAGENT_MODE.md (child scope/write ownership; no recursive delegation)
Also read c5/parallel-web-search SKILL.md; used connected Parallel MCP, not a CLI or alternate account.
Boundary honored: no repository file changed, no child launched, no publication, no claim of numerical or implementation verification.
