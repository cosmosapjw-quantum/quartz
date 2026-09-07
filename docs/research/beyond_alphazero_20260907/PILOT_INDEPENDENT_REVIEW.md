# Independent C01 pre-run review — 2026-09-07

Target: `docs/research/beyond_alphazero_20260907/{gaussian_batch_pilot.py,test_research_math.py,RESEARCH_RECORD.md}`. Reviewer did not write these files and made no changes to them. Scope: mathematical correctness, budget/CRN protocol, experimental interpretation, and frozen-plan coverage. No full pilot executed by this reviewer.

Verdict: **no blocking code or experimental-validity defect found for the explicitly exploratory Gaussian diagnostic.** This is not approval of an AlphaZero performance claim or an asynchronous scheduler deployment. The pilot is suitable for its one planned frozen execution with the claim limits below.

## Reviewed identity

Review began with the files untracked on base `a74adb4e0176b0e7f0ad6cb7c1870a483b787749`. During review the author committed them as `253df550031d816249dac0a414c74596f27142b0` (`research: freeze Quartz Gaussian diagnostic and source counterexamples`). Final reviewed SHA256 values:

- `gaussian_batch_pilot.py`: `536e319ab2e10dd7b0e0b9100a13da47aec37ba70f08d674d6735f9270333109`
- `test_research_math.py`: `80413d8df794a2b98227606dc70657e0414ccd113d6484bfc33d8bba61eded19`
- `RESEARCH_RECORD.md`: `07225f2477bbeeb2b99bd1a3b312efa77c043e542dcaebab104ec3e0f8597ac7`

## Positive findings and independent evidence

1. **Completed logical budget fairness.** `allocate` uses partial final batches and increments exactly one coordinate per observation (`gaussian_batch_pilot.py:54-80`). I independently traversed all 4 scenarios × 3 widths × 5 budgets × 4 methods = 240 designs: all coordinate counts were nonnegative, summed to the budget, and the sequential covariance update matched the direct precision posterior to rtol 1e-10 / atol 1e-12. Different priors in the misspecified case do not change the number of completed observations.

2. **Gaussian posterior update and risk identity are correct for this experiment.** `posterior` correctly combines independent coordinate observation sufficient statistics with the assumed multivariate normal prior (`:83-87`). The exact risk (`:90-93`) equals prior expected maximum minus expected maximum posterior mean, using the law of total expectation. For a correctly specified deterministic design, posterior covariance is constant and posterior means have covariance `prior_cov - posterior_cov`, so the formula applies. I verified it through a separate direct integration of true-gap magnitude times conditional probability that the posterior-mean sign is wrong. Nine representative designs across rho={0,.9,-.6}, including an unobserved coordinate, matched with maximum absolute error **1.70003e-16**. Exact risk is appropriately set to `None` in the misspecified scenario (`:153`); no incorrect Bayes identity is applied there.

3. **CRN has the correct unit and coordinate indexing.** Each scenario gets an independent truth-world bank and arm-specific normal noise tape (`:123-131`). Every design uses the same world's truth and consumes each arm's prefix by its own count (`:142-145`). This preserves marginal experiment distributions while enabling paired differences. Design rows are dependent; the code only forms paired mean differences at fixed scenario/width/budget, and the frozen record explicitly labels their intervals descriptive. No independence of the 240 design rows is claimed.

4. **Raw-data deduplication is valid for the actual algorithms.** Allocation uses covariance and counts only; no method conditions future allocation on realized observation values. Consequently, same scenario and final coordinate counts imply the same sufficient statistics and posterior decision under this CRN scheme. A shared raw_key for these designs is exact statistical identity, not selective deletion. This would need revisiting if future policies depend on observed values, stopping times, or execution order; it is valid here.

5. **Pending versus stale semantics are internally consistent.** `stale_batch` picks one arm from covariance available at batch start and commits all batch observations to it. `pending_greedy` updates covariance after each planned observation, without inventing pending observation values (`:66-79`). In the two-action normal model, exact one-step KG ordering is monotone in contrast innovation variance for every current gap, so outcome-free covariance planning is justified. The serial-width=1 equality test passes.

6. **Known-noise control and misspecification are included before results.** Neyman allocation uses standard deviations rather than variances (`:65,73`), approaching count ratio 2:5 for noise variances .04:.25. It addresses the obvious confound that beating equal allocation may only exploit known unequal noise. The misspecified rho=-.6/+.9 world tests a concrete wrong covariance assumption, and empirical coverage remains available there. Coverage uses the assumed posterior interval as it should; bad coverage in the misspecified scenario is a meaningful failure rather than a code bug.

7. **Existing focused tests pass.** Ran `python docs/research/beyond_alphazero_20260907/test_research_math.py`: **12 tests passed**, exit 0, reported 0.012 s test execution. These include quadrature for EI, covariance/precision equality, common-mode and leader counterexamples, source-proxy disagreement, zero-support limitation, free-energy Hessian finite differences, serial batch identity, and exact-risk endpoints/monotonicity. The test suite does not falsely treat a reproduced legacy defect as successful legacy behavior.

## Interpretation limits to preserve in final analysis

- **Neyman is an asymptotically optimal known-noise baseline**, not an exact finite-budget/prior-aware Bayes allocation oracle. Avoid saying pending-greedy beats an optimal finite-prior design if it only beats this baseline. An exhaustive allocation optimum would be easy for two arms but is not required for the stated illustrative experiment and should not be added after results to manufacture a claim.
- **Pending-greedy here reproduces serial greedy covariance planning inside a synchronous batch.** It does not model actual queue wait, throughput, cancellation, stragglers, overlapping roots, or wall-clock costs. Width-dependent improvement demonstrates the price of repeatedly applying stale allocation information in this chosen stale baseline, not measured production scheduler benefit.
- **All observation costs are equal and logical.** The noise differs by arm, but cost does not. The pilot measures no NN evaluations, energy, hardware speedup, or training benefit. “Matched evaluations” should be described as matched synthetic coordinate observations.
- **The prior is part of the correctly specified world generator.** Prior-predictive risk and 95% coverage are averages over independent prior worlds. They do not establish frequentist conditional coverage for every fixed latent value or tree-search state.
- **No shared-tree or temporally correlated observation process is simulated.** Correlation is covariance of latent action values; observation noise is independent across samples and coordinates. This is appropriate for isolating the derivation but is not an ESS/drift/bias solution for actual MCTS.
- **Descriptive intervals are not a multiplicity-adjusted or sequential confirmatory test.** The pre-frozen source and parameters strengthen reproducibility, but do not turn an explicitly exploratory small model into a confirmatory search-strength result. Exact expected risks should carry the primary model comparison when available.
- The source “acyclic Hessian” check is a generic counterexample to assuming acyclicity implies diagonality. It is not a derivation of Quartz's actual tree Hessian and should not be represented as one.

## Preregistration coverage and nonblocking notes

The record fixes the model, noise, prior/misspecification scenarios, budgets, widths, methods, independent unit count, scenario seeds, output, paired descriptive intervals, cost interpretation, and execution stop before the full run. This is adequate for the proposed exploratory C01.

The record lists “myopic stop need not be globally optimal” among failure probes (`RESEARCH_RECORD.md:27`), but that probe is not implemented in the two reviewed Python files. Supply the counterexample in the accompanying derivation or explicitly mark it unperformed; do not claim all listed probes were numerically checked. This does not block the allocation pilot, which contains no stopping policy.

The pilot's tracked-dirty guard ignores untracked files. That is acceptable for the concrete run because the reviewed source has now been committed and hashes are recorded; it is not a general guarantee that arbitrary future untracked code cannot affect execution. No further provenance infrastructure is needed for this isolated trusted-local diagnostic.

No mutation, new test, dependency installation, or full C01 run performed by reviewer. Verification consisted of the existing 12 tests and the independent 240-design/9-integral bounded read-only checks described above.
