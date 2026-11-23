# C26 R01–R03 — Original statistical-physics and cognitive-science mechanisms

Status: completed source investigation; no repository mutation, no experiment, no build, no campaign, no publishing. The report is a bounded primary-source map, not a new efficacy result. Exact source byte identities are in `/tmp/quartz_c26_history.json`.

## Question and scientific contract

Which statistical-physics and cognitive-science ideas were actually intended in QUARTZ, what executable mechanisms survive, and what can C26 introduce constructively without erasing either original goal?

The dual original goal remains (i) general, uncertainty-sensitive allocation of computation and (ii) better actual game decisions under a fair resource envelope on consumer CPUs. The internally January-dated paper already contains the general metareasoning principle. The archival PRD independently requires better Elo under the same mean time budget. Original document dates are not independent January Git dates. Earliest reachable Git root is `15f3a1986a00d548d3da56dab8233b336e931aee` (2026-04-09); legacy inclusion is `3bcaa435b6bdca2f821f8990230ea05ddf251a2f` (2026-07-12).

Inspected current state: remote device `cosmosapjw`, device ID `edd1b263-56a0-4391-8a2a-677dd00eabf5`; repository `/home/cosmosapjw/quantum/research/quartz_general_meta25_20261123`, initial HEAD `d6019d674ae76524b1895016f49b82d4f7b001b2`, initially clean. The existing C25 ORIGIN_WATCHDOG/ORIGIN_PROVENANCE are useful secondary synthesis; the findings below were checked against actual PDF, archive members, current source and root-commit source.

## Source key and identity

| ID | Source and locator | Identity / provenance |
|---|---|---|
| P | `docs/legacy/QUARTZ_paper.pdf`, PDF pages 5–11 and references pp20–21 | SHA256 `74d8848012f1ad95fbf2320a8f0392654c800c641d3d37ea90b81dead154547b`; title page January 29, 2026 |
| D | `docs/legacy/QUARTZ_docs.zip` | SHA256 `927d9fd0201aeb3e8a052b9a5e55943d8cf21608320ac93753ec0595ce133fa2` |
| D-PRD | D member `QUARTZ/PRD.md`, §§1–2,7–8 | SHA256 `0d84ea7f2a30532d14a33a4685a47192e2bf8a1ab76eb2d5b02f1b9b50e64c50` |
| D-QFT | D member `QUARTZ/references/user_uploads/quantum_mcts_foundation.md` | SHA256 `fc8bb41be8b8ec43a924d17c41a9a7528bc185a7df96a0d7da1a059d065389c6` |
| D-EA | D member `QUARTZ/references/user_uploads/effective-action-mcts.md` | SHA256 `d8638991645cc3803ab363e3c81f9f843d5dc5b0d2afe865c29f4bc80b38386b` |
| Z | `docs/legacy/quartz.zip`, including `docs/QUARTZ_THEORY.md` | archive SHA256 `18585cddf2fed20377cb9507025af886c5e891ba4981d5b24f2b4657ecc17c4f`; theory member SHA256 `f0e69d54e2e947b6be70f05babc5a58e00f09ce8e377364a6c1c6b272c3cfed3` |
| B | `docs/BQ_PLUS_PLUS_DESIGN.md` | first commit `00d3deb8ed70cfe7cf49afb40d04c5009be10237` (2026-05-04); current SHA256 `3782424f9f1def15ee7a0bb3b4f4bfef9f3460079477c8fb512b2684c44fe5ef` |
| T | `docs/THESIS.md` | first commit `7241ad0cfe1e92f28bd8cded5ad5f9fe7d4b4eab` (2026-07-10; inherited C25 provenance); current SHA256 `02ffa31af60ecbb1fa46f8bfebe4590caa365aca22f3c3d17817017a8ef5cb8f` |
| M | `quartz/experiments/metalevel_decision.py` | first commit `777031f6dbf7ff2225f8087a9eb0b9633eb5b3a8` (2026-11-22); current SHA256 `47f33420223c3c05e495a705ef0ef838825781c3fe9b989b0a351ab479a0c47f` |
| H | `quartz/phase15_one_loop.py` | first commit `af6b35a86f9a45ce46c8a38262be9fed04d91387` (2026-07-10); current SHA256 `44204eb2a94bf5456fdc5838c9f75a1fcb897fc6f0e3e546b38cfbba4cd7993e` |

All current paths in this report are relative to the exact C25 repository above. JSON also records Git blob OIDs and SHA256 for current Rust sources and root-commit source. Text page extraction was performed with `pdftotext -layout`, retained temporarily as `/tmp/quartz_c26_paper.txt`. The original PDF remains unchanged.

## Theory-to-mechanism map

| Concept | Actual original content | Current reuse and precise status |
|---|---|---|
| KL variational free energy / Gibbs choice | P p5 Eq(1): L_beta(q)=sum q_a mu_a−beta^{-1} KL(q||pi); q_beta proportional to pi exp(beta mu); optimum beta^{-1}log Z. The page gives the Lagrange-multiplier derivation. | Original mathematically explicit mechanism, not later ornament. Under finite utilities and positive prior support it is an exact variational identity. The inspected legacy Rust contains KL surprise and heuristic bonuses, not evidence that its complete PUCT selection exactly optimizes this objective. |
| Rational metareasoning / VOC | P pp5–6: FOCUS, EXPAND, MERGE, STOP; Eq(2) expected root decision improvement minus compute cost; Eq(3) visit-ratio impact weighting. | Original cognitive/computational theory. Root and current Rust implement FOCUS/EXPAND/MERGE scores; M supplies exact finite Bayesian posterior update, EVSI, net versus ratio decisions and finite-horizon reference. These are source-inspected; historical engine efficacy was not rerun. |
| Free-energy principle / surprise | P p8 §6.4: S=KL(q_T||pi_epsilon); p9 gain modulation sigma_Delta(1+S/S0); p20 ref[7] is Friston 2010. | The original explicitly invokes FEP, but its actual scalar is action-prior discrepancy. Current `src/mcts/quartz.rs:847–857` computes that KL. It does not establish a latent generative-model ELBO, active-inference epistemic value, or brain-level theory. |
| Fluctuation / cumulant / one-loop idea | P p5 §4.2 discusses random utilities, log partition, Gaussian/Laplace fluctuations; p7 §6.1.2 implements lambda log(1+robust sigma); p11 §8.2 proposes higher cumulants for Gaussian flip correction. | Original proposal with a real mathematical bridge still needing precision. Current Rust `1288–1294` uses a gated log1p scale bonus, `1452–1457` negative inverse-N penalty. H is a later positive inverse-N log-policy readout whose coefficients are explicitly not first principles. These are distinct formulas, not interchangeable derivations. |
| Nested sampling / hidden modes | P p7: outside prior mass, ideal hidden-max probability, Poisson proxy 1−exp(−m_out p_tail), gate using multimodality/surprise/heavy tails. | Original intended EXPAND channel. B §§2.4 specifies a later nested reservoir with quantile pruning and cooldown. This is NS-inspired search, not proof that the controller implements Skilling evidence quadrature. Existing hidden-candidate and reservoir lanes should be reused before inventing another synonymous subsystem. |
| Discrete geometry | P p8 §§6.3.1–2: variance of incoming TT estimates as holonomy proxy; optional Ollivier/Forman curvature on sparse-TT trees. | Original mechanism proposal for MERGE. C25 independent fixed SPD responses lack this transposition graph/curvature channel. A scalar response Hessian is not automatically that graph geometry. |
| RG, open quantum system, criticality | D-QFT contains explicit lattice/action, Wick-rotation, Lindblad, RG, redundancy and phase-transition claims; D-EA §§2,5 contains Tr log effective action and quantum-critical fixed points. | Genuinely present in the original intuition corpus, but exact physical equivalence is not established. Several displayed derivation steps are defective, as detailed below. Preserve the sources and construct corrected statistical mechanisms instead of importing their theorem labels. |
| Bounded rationality / expert signatures | B title explicitly says bounded-rational Bayesian free-energy search, first committed May4. T July10 articulates de Groot macro-invariance and Russek-style VOC-tightness as a double-pattern prediction. | Bounded-rational language is later formalization of already-original rational metareasoning. The detailed human-GM battery is later, and remains a hypothesis. This investigation does not verify the external psychological claims in T. |
| Rate distortion / information bottleneck | No named formulation was found in the bounded inspected PDF and text members of D and Z. | Treat a rate-distortion/IB formulation as a proposed new connection unless another exact historical source is located. Absence from this bounded corpus is not global proof of absence. |

### Actual source path continuity

At root commit `15f3a…`, `src/mcts/quartz.rs:607–617` already computes smoothed KL(q_T||prior); line945 computes `voc_focus=p_flip*e_delta−cost_focus`; lines1064–1068 use the log1p uncertainty bonus; lines1226–1232 return `−min(hbar_eff,cap)/N_a`. Current corresponding formulas occur at lines847–857,1171,1288–1294,1452–1457. This establishes source-level continuity, not fresh test success.

Current `src/mcts/select.rs` imports and calls the scoring helpers; the historical root limits the special scoring to root depth (as directly inspected in C25 provenance). `src/mcts/gvoc.rs` is the actual legacy compute-allocation module. `src/mcts/policy/kg_stop.rs:76–82` contains Gaussian expected improvement `s phi(delta/s)−delta Phi(−delta/s)`; B §2.3 identifies it as the later knowledge-gradient primitive.

M lines107–161 explicitly separates terminal value, Bayes simple regret and posterior-predictive EVSI. Lines163–175 distinguish gross gain/cost from gross gain−lambda*cost. Its opening contract says it is a finite Bayesian reference, not a trained MCTS posterior or wired production policy. This is the cleanest reusable validation substrate for C26 theoretical introduction.

## Source discrepancies and transported assumptions

1. **Path normalization in D-QFT, lines115–124.** The proposed path-integral proof writes log P(gamma)=sum beta Score−sum log Z_s, then calls node-dependent Z_s path-independent. Distinct paths visit distinct states, so this requires a special assumption not stated. Smallest correction: keep the path-specific sum log Z_s in the action, or prove a restricted constant-normalizer case. This is a local formula defect; it does not rule out useful Gibbs models.

2. **Diagonal Lindblad jumps in D-QFT, lines194–221.** The stated jumps are sqrt(Gamma)|a><a|. For diagonal rho, each associated dissipator has zero diagonal and only dephases off-diagonal elements. The displayed population-balance/Gibbs claim does not follow from those jumps. Smallest correction: derive a classical detailed-balance Markov generator with off-diagonal transition jumps, or explicitly retain an analogy. No repaired quantum theorem is claimed here.

3. **Cumulants must identify the random object.** P §4.2 moves between random log partition, mean-utility log partition and a heuristic exploration bonus without deriving their equality. `E log Z(U)`, `log E Z(U)`, and `log Z(E U)` differ. A C26 approximation must name which expectation and covariance it approximates; low-order cumulants alone do not justify the legacy log1p or inverse-N law.

4. **Gaussian ranking error versus effect of a computation.** P p6 Phi(−mu_Delta/sigma_Delta) is exact for the stated Gaussian ordering error. It is not automatically the probability that a specified future computation changes the chosen action. Posterior transition information is required. M already supplies that transition structure and should be reused.

5. **Correlation bound in P p6.** The statement that setting rho=0 is conservative when correlation is unknown requires rho>=0 for nonnegative gap. Negative rho increases difference variance. Keep this qualification when reusing the original Gaussian channel.

6. **Original VOC notation and units.** P Eq(2) must be read as post-computation information affecting terminal expected value; evaluating every future selected arm by the same fixed current mean cannot give positive expected gain over the current argmax. M corrects this with posterior-predictive updated terminal values. A price lambda with utility/time units is needed before subtracting wall time; C25 already preserves that distinction.

These are direct algebraic/semantic observations from source formulas, not simulation findings or blanket no-go results.

## Grounded policy correction and origin watchdog

**Object:** T lines177–180 says all surviving physical terms are merely names/schema aliases. **Defect:** this goes beyond barring unsupported literal thermodynamic superiority and excludes the valid variational mechanism P Eq(1), despite the present request to actually introduce statistical physics. **Impact:** C26 risks replacing theory with renamed heuristics. **Smallest correction:** allow precise KL/Gibbs, fluctuation and information-theoretic constructions under explicit definitions and validation; retain the prohibition on unsupported physical equivalence or performance superiority. Preserve the frozen historical source and record the scoped current-policy correction rather than rewriting its history.

**Object:** T lines124–129 says the engine never computes or consumes VOC. **Defect:** read literally this conflicts with P1's computational-value controller, P's original Eq(2), and actual KG/VOC source. **Smallest correction:** prohibit using offline oracle VOC-tightness labels or signature evaluation targets as runtime inputs; allow runtime estimated VOI/VOC derived from permitted observations. Independent high-budget labels must remain evaluation-only.

**Object:** original dual goal. **Observed drift:** the C23–C25 SPD lane isolates priced STOP/FOCUS and removes learned game values, candidate omission and graph reconciliation. **Correction:** identify it as a tractable mechanism study, connect the theory to utility and actual charged compute, and retain a concrete future MCTS bridge. Neither a general finite model nor an SPD certificate establishes Elo, human expertise, literal thermodynamics, or the complete four-action controller.

No current result is discarded merely because its domain differs. No historical campaign is required to finish C26. No new audit hierarchy is proposed.

## Two feasible C26 constructions

### A. Exact Gibbs allocation of actual metalevel compute actions

Reuse M's explicit posterior transitions and finite-horizon/net values. For c in {STOP, compute actions}, let Q_c be terminal-utility gain after charged resource cost, with Q_STOP=0. Introduce a declared prior r(c)>0 and solve

max_rho sum_c rho(c) Q_c − tau KL(rho||r).

Then rho*(c)=r(c)exp(Q_c/tau)/Z and the optimum is tau log Z. This directly reintroduces the original free-energy mechanism at the metalevel. An information-budget formulation KL<=kappa gives tau as a dual variable; it should not be hidden as a signature-tuned knob. tau has utility units, whereas lambda has utility/time units; they are distinct costs.

Compare against exact hard Bellman choice, immediate STOP and the already-tested baseline on exactly the same finite tasks. Report both the regularized objective and unregularized actual terminal utility minus all charged compute. Stochastic softening is not predicted to outperform the exact hard optimum for the latter objective. The scientific question is the cost/benefit of a bounded information policy and a usable computation-allocation mechanism, not a free win from temperature.

### B. Derive the original fluctuation channel on posterior-mean transitions

For fixed positive action prior pi, define F_beta(mu)=beta^{-1}log sum_a pi_a exp(beta mu_a). A specified computation produces posterior means mu^Y. Under the calibrated finite model E[mu^Y]=mu. The exact smoothed information value is G_beta(c)=E F_beta(mu^Y)−F_beta(mu)>=0.

Writing q=softmax(log pi+beta mu) and C_c=Cov(mu^Y), its second-order term is

G_beta(c) approximately (beta/2) tr[(diag(q)−q q^T) C_c].

The relevant covariance is of the *future posterior mean*, not the current total latent utility variance. This directly connects a statistical-physics susceptibility/Hessian to the cognitive value of a computation. Use exact finite outcome enumeration as the reference and establish an explicit remainder/domain before calling the approximation controlled. For finite positive pi, F_beta approaches max utility as beta grows, recovering the original terminal-value functional; the local expansion need not remain accurate in that limit.

This candidate is the strongest faithful extension if C26 needs actual theory rather than another heuristic: it makes P §4.2 computable and falsifiable, reuses M, and exposes whether the extra controller work is worthwhile. It is not already implemented by the legacy inverse-N or log1p bonuses. A subsequent game adapter must estimate the corresponding transition covariance and charge its runtime; that bridge remains distinct.

## Evidence status, limitations and next step

- Historical presence and source continuity: established by inspected primary project files and Git identities.
- KL/Gibbs variational statement: exact under the stated finite-support assumptions; original P supplies a derivation.
- Defects listed above: direct algebra/semantics; no external literature validation or numerical experiment was performed in this delegated lane.
- C26 constructions: proposed calculable mechanisms; the parent/theory lane must complete derivation, choose acceptance and execute requested R04–R10/C01–C10 work.
- Current build, tests, CPU efficiency, MCTS play quality and cognitive signatures: not tested here.
- This was a bounded corpus read. Named-theory absence is not asserted for all branches or every legacy archive.
- Recommended next module: parent research integration and constructive implementation using the exact finite model; no additional context-survey gate.

## Skill context used

Actually read PhysMath Research Loop 1.3.0 `SKILL.md`, `references/THEORY_LOOP.md` (R01–R03 contracts), `references/POLICY_INTEGRATION.md`, `references/SUBAGENT_MODE.md`, and Keystone 2.0.4 context-survey. Applied a bounded source read, preserved original scientific targets, separated source/derivation/implementation claims, and proposed only the smallest evidenced policy corrections. Parent owns stage acceptance and publication. No child agents were created.

Checkpoint: objective completed; artifacts `/tmp/quartz_c26_history.md` and `/tmp/quartz_c26_history.json`; assumptions and source limitations above; no current-task blocker; next action is parent read/integrate and select one calculable construction.

## Addendum — Actual COG sources and the parent's selected C26 direction

A targeted tracked-file search found `docs/research/cognitive_metareasoning_20261120/` and its actual COG01, resident continuation, reopening and high-branch follow-ups. It did not locate the exact named independent `QUARTZ_COGNITIVE_R1_R10_REPORT_KO.md`. The actual COG01 contract explicitly says that the independent COG R1–R10 report was supplied in the preceding conversation and is not assumed tracked in that branch. Therefore chunk/template/replay formulations known only through conversation retrieval must not be presented as inspected repository-source content.

Concrete primary anchors:

- `COG01_CONTRACT.md` (SHA256 `24b3645d5c9c2926d56f1e10651d5e9110748ce84cbf3768fd7a9acf9c9c9f71`), first commit `c183b7210a451ba3445c65b86672609d8807f954`, 2026-11-20. It binds TRAIN-only posterior q(S), a binary recognizer and fixed fallback utility u, perfect inspection, gross value g(q)=min[q(1−u),u(1−q)], and inspection iff g(q)>c. It explicitly separates cue-quality calibration from signed computation-value calibration and preserves cue-inversion harm.
- `COG01_RESULTS_AND_NEXT_PLAN_KO.md` (SHA256 `6f7932ac67dd2a1495c5b80a0cf37237bd0beb95d8dac4b2d764f3e6e40fe9a4`), first commit `cc3aa1771460f5a0f608375d985fd02998861364`, 2026-11-20. Lines71–79 request actual frozen-checkpoint traces, resident-root forks, signed utility change, all real cost channels, held-out families, and COG-02 candidate reopening. The explicit cognitive target is appealing but suboptimal high-prior candidates versus low-prior decisive refutations, assessed by matched-cost causal regret.
- `cognitive_exact.py` (SHA256 `4f4c55bc16dc2f0efdf1b26b0867ec4f2795aeb9bf65b651d2b36e45ae79dd61`), first commit `8a624e69978a6df18bbd07dfd6b4b0465fc4c371`, provides the actual finite recognizer/verification implementation source. No test was rerun here.
- `resident_reopening_13/theory/R10_HANDOFF_TO_CODING.md` (SHA256 `5ed4131915d704bb97ba7ccc0e1a4989a96ebd2b9d8a371c4298b9e2528a9c47`) records full root candidate preservation with monotone mask activation and N/W/M2 continuity. It explicitly credits prior progressive unpruning and does not claim a human Einstellung theory.
- `highbranch_transfer_14/RESULTS_AND_NEXT_KO.md` (SHA256 `7bbe4c5f020365981df1e97d411664d16aa820b9744082ff68bac15e38eff60e`) reports actual 225-action / resident-root source work; source-specific tactical recovery is not a general matched-CPU play-strength or human-mechanism claim. These are historical report claims, not revalidated measurements.
- `docs/METACOGNITIVE_EXPERIMENTS.md:135–140` already calls for actual resident-root continuation and WIDEN/CHALLENGE/DEEPEN/PROVE/REPLAY/STOP actions, warning that a policy object alone does not wire morphology actions. The July13 document predates the November COG series.

Parent's newly selected candidate is **actual GameState full-root action queries + TRAIN-only categorical belief + Gibbs net-VOC + Pandora reservation index**. This is consistent with original KL/Gibbs and cognitive computation-choice intent and with the concrete COG01B direction. It is a stronger direct game bridge than repeating the SPD-only lane. Source attribution must distinguish:
1. Original: KL/Gibbs, net computation value, root decisions, real resource accounting.
2. Existing reuse: TRAIN-only recognition, categorical/exact finite Bayesian transitions, full candidate list / resident query infrastructure.
3. New C26 contribution: the explicit Pandora mapping and its derived reservation-index semantics.

The Pandora mapping must state whether unrevealed action utilities are modeled conditionally independent, whether inspection cost is fixed, whether inspected values remain selectable (recall), and how STOP's known incumbent enters. A learned factorized categorical model is an explicit model approximation for coupled game actions; it is not a proved empirical independence property. Any theorem optimality belongs to that stated model. Real GameState outcomes and all actual controller/query cost remain the external validation target. This is an assumption ledger entry, not a prohibition on the requested experiment.
