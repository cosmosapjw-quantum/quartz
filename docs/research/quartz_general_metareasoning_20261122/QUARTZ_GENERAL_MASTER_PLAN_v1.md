# QUARTZ General Metareasoning · Master Research Plan v1.0

Date: 2026-11-22 (KST). This is a NEW interpretive research companion, NOT a rewrite of the frozen original THESIS.md. Source manifest: evidence/REPOSITORY_PROVENANCE.json. Exact mathematical results: theory/R03–R07. Code/evaluation: quartz/experiments/metalevel_decision.py, src/mcts/policy/finite_metalevel.rs, evidence/C08, evidence/C09_INDEPENDENT_FRACTION_AUDIT.json.

## Identity and historical authority

QUARTZ originally abbreviates Q-value Uncertainty–Adaptive Root-risk Tree search, Zero-tunable (README). Its research goal is a bounded-rational computation selection, allocation, and stopping controller. The multi-game Rust MCTS engine and Python selfplay are experimental environments rather than the scientific objective. Specific Gomoku, Connect4, Go and Chess optimizations are auxiliary benchmark studies and must never become the governing law.

In the original frozen THESIS.md, P1 states one computation-value principle, P2 is a specific testable efficiency target, P3 human-expert-style signatures are emergent, nonoptimized predictions, P4 actual runtime CPU-friendliness is an engineering gate. The same document forbids game-specific pattern injection into the generic controller. The old physics-inspired terminology is exploratory analogy, not proof of physical superiority.

Historical provenance: 2026-04-09 first Git commit and Rust/Python multi-game MCTS; May BQ++ posterior confidence, KG and stop primitives; July explicit frozen P1–P4 and separate Beta-Bernoulli ranking, candidate-omission labs; August local idea foundry and 26-axis synthetic research; September beyond-AlphaZero plan; November COG01–COG06 Connect4 and Freestyle15 practical CPU experiments. These stages include valuable finite positive AND negative evidence. None overrides the original frozen project mission.

The complete available GitHub snapshot was 413 Git commits and 30 origin refs at the survey, agent7/audit4/research18/main1. Actual GitHub REST PR objects returned zero. PR-like branch names do not prove those commits were reviewed or merged through PRs. No merges to main, invented PR or protected developer worktree edits are part of this plan.

## Mathematical nucleus

Let b be a probability distribution over unknown utility-relevant states, A the physical terminal action set, U(a,theta) the payoff, and computation c a priced observation channel with conditional law L_c(y|theta). Define terminal Bayes decision value M(b)=max_a E_b U(a,theta), perfect-information value K(b)=E_b max_a U(a,theta), and decision simple regret R(b)=K(b)-M(b).

Gross one-step value of computation G_c=E_y M(b_after_y)-M(b) is nonnegative in a calibrated Bayesian model. The originally proposed throughput index I_c=G_c/k_c and the one-step priced *net* value N_c=G_c-lambda*k_c are distinct choices when computation costs differ. A finite-horizon STOP-or-compute decision is governed by V_0(b)=R(b), V_h(b)=min{R(b), min_c [lambda*k_c + E_y V_{h-1}(b_after_y)]}. A fixed-budget constrained objective is another separate problem.

This cycle executed an exact finite counterexample: three hidden states, identity terminal payoffs, partial/full information gains 1/5 and 1/2, costs 1/20 and 1/5. Ratio chooses partial (4>5/2); net chooses full (3/10>3/20). This is a local correction to original objective *interpretation*, not an original claim of the Bellman equation. The one-step STOP sign equivalence is preserved for positive cost. The original THESIS bytes and claims remain historically unchanged.

The meta-action menu must be explicit, with cost units and a calibrated observation model. Observation is not the same thing as a neural-MCTS value backup. Correlated backups, evaluator drift and omitted candidates require an augmented inference model, not silent conditional-IID assumptions.

## Layered research architecture

1. Object-level task adapter: legal terminal choices, payoff/loss convention and outcome, finite horizon, independent of the generic selector.
2. Belief and computation-observation model: posterior, uncertainty, state-dependent channel, calibration and drift; needs real neural search data before application.
3. Computation menu: STOP, MORE_SAMPLES, WIDEN, CHALLENGE, DEEPEN, VERIFY, REPLAY, each with measured cost, effect and completeness conditions. Do not invent unimplemented actions.
4. Metalevel controller: risk, EVSI, net, rate, bounded Bellman / approximation, typed evidence, objective selected in advance.
5. Search runtime adapter: Rust SearchPolicy, SearchSnapshot/EdgeView, immutable cache, actor/learner boundaries and true time/NN-eval cost.
6. Offline evaluation: source-grouped paired comparisons, calibrated reference, decision regret, observational alignment, physical compute cost and uncertainty. Game WDL remains secondary evidence.

A compiled reference module is not a deployed SearchPolicy; do not retrofit the PUCT engine without measuring real observation-model calibration. Existing 26 foundry axes are candidates and comparators, not 26 sequential acceptance barriers.

## Revised research DAG and next deliverables

T0. **Mathematical semantics and finite proof-of-computability.** CURRENT C17 scope. Read original source authority; derive/verify probability and cost consistency, information value, ratio/net mismatch and finite Bellman. Produce independent Rust/Python model, exact numeric witness and reproducible five-task lab. Exit status IMPLEMENTATION_VERIFIED for finite synthetic reference only, NOT real MCTS.

T1. **Calibrated computation-observation model.** NEXT C18 priority. Freeze actual Rust root continuing-search traces and an independently different non-game planning/decision environment, with exact cost and parent-family provenance. Treat continuation outcomes as correlated, nonstationary conditional observations unless formally justified. Fit predictive likelihood/uncertainty, evaluate heldout reliability, calibrated risk and wrong-stop error under grouped source/checkpoint splits. Reject or recalibrate invalid observation models before production activation. Negative calibration results are legitimate science.

T2. **Actual generic computation-action semantics and native integration.** Use existing SearchPolicy and immutable cache for a shadow controller that proposes named typed meta-actions and STOP with no live mutation. Compare one-step rate, net, learned approximation and bounded-lookahead against legacy fixed and risk controls using the identical evaluator and independently measured CPU/NN costs. Add functional handlers for a meta-action only when it has real MCTS or task-adapter semantics, and preserve full root action/prior statistics. Enable a gated live experimental policy only after C18 calibration and demonstrated non-inferior terminal decision quality.

T3. **Cross-domain falsification rather than Elo leaderboard.** Require at least two structurally different domains and utility families with independent task/source grouping, comparable cost units, fixed shared evaluator where appropriate, common-random-number or paired data, decision regret and uncertainty. P2's historical >=30% NN-evals saving on Gomoku7 is one specific benchmark target, not a universal research objective. Positive single-domain outcomes do not establish transfer.

T4. **Emergent cognitive and physical analogy claims as distinct hypotheses.** Original P3 predicts a conjunction: macro search signatures approximately skill-invariant while offline oracle VOC-tightness rises with evaluator skill. No optimization of these signatures as control targets, and no human-general cognitive mechanism from model search geometry alone. Legacy quantum/thermodynamic notes remain idea sources pending rigorous mathematical correspondence.

## Evidence and stop conditions

Each scoped result carries exact source/commit/input identities, assumptions, measured law or numerical derivation, actual executable commands, source-family independence and uncertainty. Separate ESTABLISHED, LITERATURE-SUPPORTED, DERIVED, NUMERICALLY CHECKED, IMPLEMENTATION-VERIFIED, EXPLORATORY, UNRESOLVED and BLOCKED.

For the core P1: finite Bayes mathematical consistency is verified, but real adaptive neural-MCTS belief/likelihood calibration is NOT verified. For P2: benchmark savings and terminal noninferiority NOT TESTED this cycle. For P3: the emergent signature double pattern NOT TESTED. P4: actual production hot-path CPU overhead NOT TESTED; CLI reference timing is not P4.

If observation calibration fails, do not promote the controller into production. It is still scientifically productive to compare failure modes or alternate observations, but not to tune decision signatures retrospectively. Never adjust safety/cost thresholds merely to make a negative result look positive. No unbounded review recursion.

## Next new independent research node

QUARTZ-GENERAL-18-CALIBRATED-COMPUTATION-OBSERVATION.

Theory R1–R10: explicit correlated finite-observation model and source-conditional assumptions; derive consistent decision value, posterior sensitivity and finite-cost optimization with actual MCTS and non-game task observations; original primary literature and adversarial counterexamples; numerical checks; fixed implementation contract.

Coding C1–C10: immutable source locks, true root continuation and non-game traces, heldout grouped calibration/testing, physically measured computation costs, shadow controller proposals, exact reference comparisons, independent root and predictive model checks, local backup restore and remote Git publication at each stage. At least two structurally distinct domains are needed before cross-domain claims. If data insufficient label NOT_ESTIMABLE, not PASS.

Do not automatically begin C18 inside this C17 cycle. Existing original developer branch, current main, frozen THESIS and CLAIM_LEDGER remain unchanged.
