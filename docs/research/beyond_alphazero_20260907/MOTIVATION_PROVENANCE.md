# Quartz motivation and a defensible statistical-physics translation

Prepared from repository `main` at `3e303aef0a0a534d1ccc8b72289a72b7fd0dc901`. This is an analysis memo, not a production change or evidence of measured gains. Source line numbers refer to this checkout. Mathematical constructions below are explicitly proposed reconstructions unless stated otherwise.

## 1. Preserve the problem the user was trying to solve

The faithful research question is broader than “stop a conventional MCTS run earlier.” A compact learned evaluator supplies pattern-based expectations; computation should build and revise a structured belief about competing continuations, seek consequential surprises across multiple plausible regions, and allocate further simulation according to its prospective effect on the final decision. Physics supplies candidate methods for representing and integrating correlated fluctuations, transporting probability across separated regions, and comparing descriptions at different resolutions. Metareasoning supplies the decision objective. Neither language alone exhausts the original idea.

The current request adds an explicit large, repeated simulation-budget requirement. It should not be silently replaced with the repository's most recent short-budget target.

### Provenance map

| Intuition | Direct historical support | Faithful interpretation and limit |
|---|---|---|
| Parallel paths as a statistical-field representation | `docs/legacy/mcts_study/v6.0/mcts_conversation.txt:1` (user Q1) explicitly says this is **not quantum computing**, and proposes an analogy to statistical field theory/QFT. | Preserve weighted alternatives and shared structure; do not ascribe literal complex-amplitude interference to a batch of classical evaluations. |
| Diversity and selection of one decision | Same file Q2 at line 7 explicitly names virtual loss, path diversity, MinHash, simulation index as time, and a possible approximately Gaussian slice. | Redundancy-aware computation and a stochastic process on search histories are concrete targets. A single selected move is not evidence of quantum decoherence. |
| Demand for rigorous discrete construction | Q7 at line 431 requires a discrete, acyclic construction before saddle points/loops and explicitly rejects importing gauge structure without justification. | The user repeatedly asks for the bridge to be made correct, not for terminology to be defended at any cost. |
| Parameters must come from search | Q16 at 1086 objects that effective potentials depend on new mass/hopping parameters not derived from MCTS; Q18 at 1255 asks deliberate interpretation; Q20 at 1407 asks for information distances derived from dynamics. | Observable or learnable covariance, likelihood, compute cost, and transition kernels are preferable to freely named physical constants. |
| Bayesian-brain perspective | Q23 at 1648 explicitly asks to retain this viewpoint; the response develops prior/search distributions and a free-energy expression at 1654–1708. | This supports the intent, not the response's claimed exact equivalence. A visit distribution is not automatically a posterior; reward is not automatically log-likelihood. |
| Concrete corrections, not descriptive metaphors | Q6 at 349 and Q24 at 1710 request an effective action and a corrected implementable selection rule. | Require a specified measure, quantity to integrate, approximation, and empirical benefit. |
| Biological and information efficiency | `thermodynamic-mcts-research-proposal_4.md:43–70,85–148` frames search as resource-constrained information extraction and explicitly discusses finite-system statistical mechanics. | The numerical claims about brains/engines there are historical motivation, not independently validated measurements. The finite size of a search tree is not a reason to ban partition functions. |
| Practical parallel performance | `vectorized-mcts-research-proposal_1.md:16–46` asks for wave processing, diversity, hardware-aware evaluation, and fair optimized baselines. | Both decision efficiency and wall-clock throughput belong in the programme. |
| Latent utility belief and VOC | `docs/BQ_PLUS_PLUS_DESIGN.md:14–32,42–74` specifies posterior simple regret, calibrated beliefs, computation value, and a candidate reservoir. `docs/THESIS.md:34–59` makes these approximations to a common objective. | This is strong later support for belief-guided computation, but the composed policy is explicitly not integrated. |
| Information geometry | The v6 conversation Q20 and the Fisher-metric discussion at 1757–1762; v5 `new_quantum_mcts.md:74–90` attempts a distance-based action. | The Fisher geometry is real mathematics; the particular derivative identities and “Hellinger” label in the drafts require checking. |
| Coarse-to-fine multimodal search / Wang–Landau | Coarse-graining language appears in v6 Q8 at 497 and 1789–1799; reservoir/local-minimum protection appears in BQ++ at 64–74. **The exact Wang–Landau, small-NN pattern-recognition, and latent-utility phrases were not found in searched local `.md`/`.txt` records.** | Attribute these explicit formulations to the current user's clarification, with historical adjacent support. Do not invent a dated original Wang–Landau design. |

The legacy corpus contains competing rewrites, including `thermodynamic-mcts-research-proposal_1.md:21–33,45–56`, which strips away statistical physics almost completely, and `quantum-mcts-article.md:3–13`, which overclaims exact quantum foundations and very large speedups. Neither should override the user's first-person questions. Git records the corpus in one import (`3bcaa43`, 2026-07-12); filename versions establish conceptual lineage, not reliable original creation dates.

The current thesis narrows its performance target to ≥30% fewer NN evaluations on short-budget Gomoku7 (`THESIS.md:63–91`) and its engineering gate to 8–64 visits (`142–147`). Part B explicitly calls its scope low-budget (`1–7`). Those are historical operational choices, not a refutation of the user's wider ambition.

## 2. A valid bridge: define the distribution before naming the physics

For a finite set of alternatives `x`, a base measure `m(x)>0`, energy `E(x)`, and positive temperature `τ`, define

`Zτ = Σ_x m(x) exp(-E(x)/τ)` and `qτ(x)=m(x)exp(-E(x)/τ)/Zτ`.

Then the Gibbs variational identity is exact:

`-τ log Zτ = min_q { E_q[E] + τ KL(q || m) }`

when `m` is normalized; an unnormalized measure requires the corresponding normalization convention. This is legitimate finite-system statistical mechanics. It requires neither a thermodynamic limit, actual thermal energy, nor quantum hardware. If `x` is a root-to-leaf trajectory, the same object is a discrete path sum. The hard work is choosing an energy and reference path law that encode the desired decision problem, and sampling it accurately at useful cost.

Three different distributions must remain distinct:

1. `B_t(θ)=p(θ | history_t)`: a belief over unknown utility quantities, possibly correlated and multimodal.
2. `π_t(a)`: the policy used to allocate simulations or choose a move.
3. `q_t(history)`: the law induced by adaptive selection, evaluation, and backup over computation histories.

A policy prior from a small NN can help define all three, but they are not interchangeable. A Bayesian model needs a likelihood for observations. Deterministic NN evaluation on an already-seen leaf is not a new independent noisy observation of the root's true minimax utility. Adaptive backups from a changing subtree are generally nonstationary and correlated. A belief model must encode or conservatively estimate those effects; Welford variance divided by a visit count does not establish calibration.

For policy improvement with fixed utilities, `E(π)= -π·Q + τ KL(π || P)` yields `π_a∝P_a exp(Q_a/τ)`. This is an exact log-partition identity, but it is not exact PUCT. Reversing the KL gives a different optimizer: maximizing `π·Q - λ KL(P || π)` has stationary solution `π_a=λP_a/(α-Q_a)`, with `α` chosen for normalization. The direction matters. Grill et al. establish a regularized-optimization relationship to AlphaZero heuristics, not a license to equate every KL-based action with PUCT: [primary paper](https://proceedings.mlr.press/v119/grill20a.html).

A genuine path-integral control identity can exist under particular control/noise structure; Kappen's work derives such a class by linearizing the relevant optimal-control equation. It does not imply that an arbitrary adversarial tree or PUCT dynamics inherits the same identity: [primary paper](https://arxiv.org/abs/physics/0505066).

## 3. What remains useful in one-loop and geometric reasoning

### Controlled integration, not an arbitrary bonus

Given a genuine integral `Z(τ)=∫m(x) exp(-E(x)/τ) dx`, an isolated interior minimum `x*`, and positive-definite constrained Hessian `H`, Laplace gives

`-τ log Z ≈ E(x*) + (τ/2)log det H - (dτ/2)log(2πτ) - τ log m(x*)`.

This is the valid Gaussian fluctuation correction. On a simplex, use tangent coordinates or an explicitly constrained measure; include the Jacobian under a coordinate change. A global scalar log determinant cannot simply become a per-arm score without a derivation of the decision or marginal quantity that score represents.

For multiple separated minima, the approximation must sum their contributions **before** taking the log. A one-basin approximation misses a second basin regardless of its internal Hessian accuracy. At boundaries, zero or negative curvature, discrete counts near zero, or changing model dimension, the ordinary local formula can fail. Replacing a negative eigenvalue by its absolute value does not turn an unstable Gaussian integral into a convergent one; v6 at 1767 performs precisely this unsupported substitution.

Acyclic graph topology does not imply a diagonal Hessian. On a two-node tree, `E=(x_parent-x_child)^2/2` already gives a nonzero off-diagonal Hessian. Nor does an acyclic graph forbid perturbative “loops”: diagrammatic loop order counts fluctuation contractions, not cycles in the game graph. Thus v5 `new_quantum_mcts.md:46–50,114–120` is mistaken, while a graph-based fluctuation theory remains entirely possible.

### Uncertainty should concern contrasts

Let `θ~B_t` have mean `μ` and covariance `Σ`, and define the smooth value

`gβ(θ)=β⁻¹ log Σ_a P_a exp(βθ_a)`, `π_a=P_a exp(βμ_a)/Σ_bP_bexp(βμ_b)`.

A second-order posterior expansion gives

`E_B[gβ(θ)] ≈ gβ(μ) + (β/2)tr[(diag π-ππᵀ)Σ]`

`= gβ(μ) + (β/4)Σ_ab π_aπ_b Var_B(θ_a-θ_b)`.

This is a constructive fluctuation correction that retains the physics intuition. It naturally removes irrelevant common-mode uncertainty. It is a correction to a posterior expected soft value, **not yet a proven acquisition bonus or VOC**. The expansion needs controlled higher-order terms; near ties a useful rough diagnostic is that `β` times relevant contrast uncertainty be small. “Many simulations” alone does not guarantee this when temperature is simultaneously annealed, new branches are introduced, or alternative modes remain unresolved.

`QUARTZ_THEORY.md:152–158` instead calls dispersion across estimated action Qs the primary uncertainty signal. Counterexamples are immediate: exactly known values `(0,1)` have high cross-action dispersion and no uncertainty; equal means `(0.5,0.5)` with wide independent posteriors have zero dispersion and substantial decision risk. Common noise `θ_a=μ_a+Z` can make marginal uncertainty enormous while every ranking is certain. Pairwise posterior contrast uncertainty is the relevant object.

The log-partition Hessian with respect to action scores is a Fisher matrix of the **policy exponential family**. It measures policy sensitivity; it is not automatically the covariance of unknown utility estimation error. Geometry becomes actionable when the model says which information is being measured and which parameters a computation can change.

## 4. Three concrete mechanisms worth falsifying

### A. Correlated latent-utility belief with decision-sensitive computation

Treat a proposed computation `c` as a leaf evaluation, a deeper refutation search, a subtree comparison, or a batch. In a linear-Gaussian pilot, `Y_c=h_cᵀθ+ε`, `Var ε=r_c`. The posterior mean update is `μ⁺=μ+u_c Z` for `u_c=Σh_c/sqrt(h_cᵀΣh_c+r_c)` and `Z~N(0,1)`.

For two contenders with current mean gap `Δ≥0`, the update scale of the **difference** is `s=|u_ca-u_cb|`, and expected decision improvement is `s φ(Δ/s)-Δ Φ(-Δ/s)`, with zero limit at `s=0`. Under a calibrated Bayesian model and a posterior-mean-optimal current decision, expected reduction in posterior simple regret equals the knowledge gradient by the tower property. This connects fluctuations and VOC without inventing a thermodynamic law.

**Large-budget hypothesis:** structured uncertainty directs later evaluations toward unresolved counterarguments and avoids repeatedly paying for common-mode information. Compare diagonal belief, learned/fitted covariance, shuffled covariance, and ordinary search at logarithmically increasing budgets, both equal NN evaluations and equal wall time. Measure decision regret and calibration of pairwise reversals, not just policy entropy.

**Falsifiers/counterexamples:** wrongly inferred shared structure suppresses a real independent refutation; all uncertainty is common-mode and no computation should be purchased; many repeat evaluations with truly independent noise are useful, so a blanket duplicate penalty is wrong. Gain must survive covariance ablation and amortization costs.

### B. Coarse-to-fine multimodal exploration with a density-aware reservoir

Maintain coarse classes of strategically distinct candidate continuations; retain representatives outside the incumbent basin; refine a class when its potential decision value warrants cost. For a specified microscopic energy and measure, the exact effective energy of class `C` is `E_eff(C)=-τlog Σ_{x∈C}m(x)exp(-E(x)/τ)`. This is genuine coarse-graining. Approximation error can be measured against exact tiny trees before using it in games.

Wang–Landau's directly relevant idea is estimating density while flattening exploration across energy ranges, including multiple restricted windows: [original paper](https://arxiv.org/abs/cond-mat/0011174). In Quartz, use that idea for an explicit candidate-generation **proposal**, with frozen epochs or other safeguards against the energy bins moving under the sampler. Record proposal probabilities; if it estimates a target integral, preserve the correct weights. A growing adaptive tree is not automatically an ergodic Wang–Landau chain. Root visits also are not a density-of-states estimate.

**Large-budget hypothesis:** the broad phase prevents persistent prior lock-in, while later refinement amortizes its overhead and focuses on a few consequential basins. Test adversarially low prior on a winning/refuting continuation, vary mode separation and prior quality, and include reservoir-without-density and density-without-refinement ablations. Report discovery delay, late-budget regret, and false commitments.

**Falsifiers/counterexamples:** a unimodal landscape may make flattening pure waste; the chosen coarse descriptor may group the hidden winning continuation with irrelevant branches; a completely uninformative prior plus an exponentially rare needle supplies no generic shortcut. Entropic multiplicity is another trap: one genuinely best action must not lose solely because a worse basin contains many duplicate continuations. Opponent replies require minimax/negamax or a specified opponent model; summing paths as if their multiplicity were utility changes the game objective.

### C. Decision-relevant batch diversity from posterior innovations

Use a covariance or overlap model to estimate how much a proposed batch teaches about contender contrasts. A Gaussian log determinant gives an exact information gain for a specified linear observation model. Project or weight the information toward root contrasts that can change the decision, and divide by measured batch cost. This translates “interference between paths” into redundancy-aware experimental design; MinHash may be a cheap feature, but not the ground truth covariance.

**Large-budget hypothesis:** when many workers repeatedly sample overlapping continuations, innovation-aware batches deliver more useful information per NN batch than edge-local virtual loss. Compare ordinary VL, exact-state deduplication, path sketches, and covariance-aware batches across worker count and budget. Require same-evaluation decision improvement or same-quality throughput gain after sketch/model overhead.

**Falsifiers/counterexamples:** disjoint paths can share the same evaluator error and be statistically redundant; overlapping paths may test distinct tactical refutations; a forced line should not be abandoned merely to increase diversity. If gains disappear against exact-state deduplication, the mechanism was cache hygiene, not a deeper correlation model. If they appear only at higher worker counts, report that limited regime.

These are complementary, but start with separately falsifiable pieces. The distinctive project contribution could eventually be their successful composition around one calibrated latent model; renaming existing individual methods is not evidence of novelty.

## 5. Conditions for the large-budget programme

Per-arm finite-visit corrections that vanish as `1/N_a` should not be the sole long-budget mechanism. Part B's B13 is explicitly designed to vanish (`103–118`). That does not preclude earlier corrections saving computation, but it offers no automatic persistent large-budget advantage. Persistent gains must come from a better allocation rate, discovering neglected modes, calibrating evaluator bias, reducing correlated batches, or amortizing learned structure over repeated related decisions.

Separate two empirical questions: at fixed per-position compute, does decision regret fall faster; and across a fixed total workload budget, does stopping/reallocation improve final decisions? Early stopping cannot claim better equal-compute search unless its saved budget is accounted for and meaningfully reassigned. “Large budget” should include logarithmic budget sweeps and tail positions, not just extending one favorable short-budget curve.

There is a wording conflict in the current thesis: `THESIS.md:124–128` and Part B `50–55` say the engine never computes or consumes VOC, while the governing principle and BQ++ KG module explicitly require runtime estimates of computation value. The valid non-circularity rule is that a held-out high-budget oracle proxy remains independent of runtime inputs and tuning. An engine may estimate its own VOC; correlation with an independent oracle is then a calibration/generalization result, not automatically a tautology. Avoid importing the document's literal prohibition into the reconstructed objective.

The defensible stance is neither “the old derivations prove QFT-MCTS” nor “discard all physics.” Retain exact statistical identities, explicit stochastic models, constrained fluctuation approximations, and density-aware exploration as testable mechanisms. Reject unsupported equivalences and use the saved conceptual precision to design experiments that could succeed at the scale the user actually wants.
