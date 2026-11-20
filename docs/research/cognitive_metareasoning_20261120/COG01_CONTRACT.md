# QUARTZ COG-01: recognition/value-of-computation calibration contract

Date: 2026-11-20. Scientific scope: exact finite Bayesian surrogate; NO Rust/NN/real-game strength claim.
Parent research commit: `6d279096c8b125c8af8d222784fac31a2d36b187`.
Branch: `research/quartz-cog01-recognition-calibration-20261120`.
Previous source: `docs/research/beyond_alphazero_20260907/W1_RESULTS_AND_NEXT_PLAN_KO.md`.
Related independent cognitive report: QUARTZ COG R1–R10 (2026-11-20), provided in preceding conversation; not assumed tracked in this branch.

## Scientific question
Can a small recognizer's calibrated assessment of an option also calibrate the *marginal value of verifying it*, and what fails when prior frequency or cue likelihood changes? A finite exact-screen is an implementable reference, NOT a result about learned QUARTZ checkpoints. Keep recognition, selection, and computation accounting distinct.

## Exact generative world and conventions
- Action A has known utility `u=3/5`. Action B has utility `X∈{0,1}`.
- `P(X=1)=p`. A binary recognizer cue `S∈{+,-}` has `P(+|X=1)=se` and `P(+|X=0)=fpr`.
- A perfect verification computation observes X, then selects the truly better action, and has cost `c=1/10` in the same utility units. It is a *counterfactual computation model*, not an MCTS/NN call.
- Recognition always happens first. Once its cue is observed, the agent outputs `q(S)=P_train(X=1|S)`, with exact rational training values.
- If no verification is performed, the agent selects B iff `q(S)>u` (ties choose A). Otherwise it selects A.
- Predicted *gross* verification value is
  `g(q)=q(1-u)` when `q≤u`, and `g(q)=u(1-q)` when `q>u`.
  This equals `E[max(u,X)|S]-max(u,q)` **only when q is the true conditional probability**.
- Recognition+VOC metacontrol: verify iff `g(q)>c`; ties stop. Realized world utility of verification is `max(u,X)-c`, irrespective of predicted q.
- Stop-only uses the same `q` and selects according to the same threshold; no change of policy prior across treatments.

## Frozen training identity and deployment scenario bank
Training: `p=1/5, se=17/20, fpr=1/5`, implying
`P_train(+)=33/100`, `q(+)=17/33`, `q(-)=3/67`.
All scenarios are **ex ante** definitions with exact rational arithmetic.
1. `in_distribution`: (1/5, 17/20, 1/5).
2. `uninformative_cue`: (1/5, 1/2, 1/2).
3. `inverted_cue`: (1/5, 1/5, 17/20).
4. `prevalence_shift`: (2/5, 17/20, 1/5).
5. `perfect_cue`: (1/5, 1, 0).
No parameter is fitted or reselected on the scenario outcomes.

## Four comparisons and independent outcome measures
- `stop_only`: cue-conditioned action without inspection.
- `learned_selective`: source-trained q and g; cost-triggered verification.
- `random_matched_rate`: independent Bernoulli verification at the *realized marginal verification rate* of learned_selective in that scenario. This rate depends on the deployment law: **diagnostic matched-cost counterfactual, NOT a deployable baseline**; never describe it as prospectively budget matched.
- `oracle_recalibrated`: the correct `P_deploy(X=1|S)`, allowing a new action and verification choice. **Diagnostic upper comparator, not a fair learned algorithm.**
Also report `always_verify` and `full_information_no_cost` references.

Primary numerical estimands per scenario:
`mean_net_utility`, `verification_rate`, `mean_quality_brier=E[(q-X)^2]`,
`quality_ECE_exact=E_S|q(S)-P(X=1|S)|`,
`gain_ECE_exact=E_S|g(q(S))-E[U_verify_no_cost-U_stop|S]|`.
The gain term is action-dependent and *must be computed with the action selected by q*, not by the true deployment posterior. This is a calibration diagnostic, not a confidence sequence.
Use exact enumeration of four atomic worlds `(X,S)` with `Fraction`, no PRNG, no Monte Carlo confidence intervals. All source q values frozen; no oracle label fed into learned_selective.

## Decision criterion and falsifiers
Scientific mechanism admissible when the exact oracle and a separately implemented joint-atom enumerator agree; all expected joint masses sum to 1; all payoffs and rates are in bounds.
Expect in-distribution: selective utility `127/200`, verification rate `33/100`, random matched utility `2967/5000`, improvement `26/625`. A deliberate cue-inversion should reveal harm; do not silently suppress it. The four other frozen scenarios test calibration/shift and idealization. None can establish neural-network MCTS efficacy.

The executable test must (a) compute source Bayes q exactly, (b) compare exact-policy and independent enumeration, (c) reject impossible probabilities, (d) handle zero-probability cues without dividing by zero, (e) record both quality and computation-value calibration, (f) preserve false results, (g) avoid creating or overwriting existing evidence, (h) keep result tuples and public schema explicit.

## Constructive integration and negative conclusion
The exact test yields a *conditional* reference for QUARTZ COG-01. In the real engine, `q` may be derived from learned features and `X` must be a target-independent held-out continuation/reference, not a claimed perfect binary outcome. First validate whether a model of late reference reversal predicts signed gain under fixed search budgets; only then add live SAMPLE/REOPEN and train/test calibration.
Frozen observation source W1 semantics must apply: same NN deterministic leaf is not repeated independent evidence, and model/representation epoch changes invalidate an unqualified posterior.

## Sources (original theory and human motivation; no algorithmic superiority claim)
- Callaway et al., *Learning to Select Computations*, UAI 2018: https://arxiv.org/abs/1711.06892
- Sezener & Dayan, *Static and Dynamic Values of Computation in MCTS*, UAI 2020: https://proceedings.mlr.press/v124/sezener20a.html
- Russek et al., *Time Spent Thinking in Online Chess Reflects the Value of Computation*, *Cognitive Science* 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12553403/
- Mattar & Daw, *Prioritized memory access explains planning and hippocampal replay*, *Nature Neuroscience* 2018: https://pmc.ncbi.nlm.nih.gov/articles/PMC6203620/
- Original QUARTZ: `docs/THESIS.md` P1/P3; `docs/METACOGNITIVE_EXPERIMENTS.md`.

## Execution and publication rules
Stage A: commit this contract and publish a GitHub branch, with a separate local backup.
Stage B: test-first implement exact enumerator and runner; publish implementation and raw exact evidence, preserving negative shift rows and red/green tests.
Stage C: publish a bounded interpretation and next COG-01B handoff, plus a fresh verification command and evidence identity. Never update main or the original development branch.
