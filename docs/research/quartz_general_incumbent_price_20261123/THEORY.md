# C25 theory research record

## R01 — Question, scope and origin

Given valid deterministic utility intervals, derive a recommendation rule with true-utility nondecrease relative to the initial four-step decision. Derive the strongest affordable-stop statement justified by the available interval information and a genuine minimum future charge. Implement that rule and separately test a time-predicted stopping heuristic at the carried primary wall-time price 1/1,000,000 utility per microsecond.

The SPD contract, dimensionless normalization, finite input domain, sequential binary64 arithmetic, zero-exterior boundary conditions and C24 rigorous envelopes are reused with identical dependencies. No continuum or physical-unit claim is introduced.

The original-history watchdog records two preserved commitments: computation should serve external decision quality under finite resources, and the controller should use general search/computation state. C25 studies recommendation safety and priced stopping. Its bounds are deterministic, not a posterior expectation model; tree allocation, learned priors and actual game-strength targets remain distinct obligations.

Exploration candidates: unchanged C24 fresh-score fallback; certified incumbent retention; retention with deterministic charged-work stopping; retention with TRAIN-only predicted wall-time stopping. The coding contract will fix the selected rules and all primary statistics before TEST outcomes.

Prospective success is not assumed: false certificates, initial-decision deterioration, conservative missed improvements, charged net utility and actual wall-time net utility will all be reported. No strong no-go is proposed.

## R02 — Evidence and reuse

The primary-source comparison in theory/LITERATURE.md reads Hay et al. (2012) Theorems 4–5 and Eq.(11), LUCB (2012) §3.1, Kaufmann et al. (2016) p.2, and Bubeck et al. (2011) simple regret. It separates incumbent SWITCH from global identification STOP, and deterministic containment from probabilistic confidence. No novelty is claimed for pairwise dominance or the rectangular regret bound.

C24's scalar envelope derivation and native sequential floating-point evaluator are reused unchanged. C24's adverse selection fixture becomes regression evidence; fresh C25 seeds are required for new primary estimates.

## R03 — Definitions, assumptions and claim alignment

Let K>=2 fixed actions have fixed unknown dimensionless utility u_a in [0,1]. At a checkpoint, finite valid intervals satisfy L_a<=u_a<=U_a. Historical intersection is the only contraction rule. Current incumbent i initially equals the ordinary four-step score recommendation. Terminal loss is max_a u_a-u_i.

Define G_i=max(0,max_{j!=i} U_j-L_i). This is a worst-case ceiling, not expected VOC. A separate Bayesian interpretation requires a posterior supported on the box; identifying it with VPI additionally requires the incumbent to maximize posterior mean.

The formal cost C is remaining charged cell-stencil work, not time. W=n*(solver matvecs+FAST evaluations+ordinary evaluations) is a declared accounting metric, not total CPU operations. Every admitted continued block unconditionally evaluates fresh FAST bounds for all K actions, so c_min=K*n. The formal price is exactly 2^-30 utility per charged cell-stencil. It is fixed prospectively as an accounting example, not fitted from economic outcomes.

The empirical time price stays 1/1,000,000 utility per microsecond. A TRAIN-only per-side next-block cost prediction is a heuristic input. Total measured times include validation, all solver and certificate work, control, and workspace destruction. Sunk work is excluded from the local future comparison but included in all policy-level net utility estimates.

No game claim follows from this finite SPD model. The claim ledger binds each assertion to its additional assumptions and evidence.
