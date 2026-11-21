# R10. Reconstructed cross-domain QUARTZ research program and coding handoff

This cycle concludes its 10-stage THEORY scope with actual source/history survey, exact Bayes counterexample, Bellman derivation and numerical checks. It reanchors the laboratory without modifying frozen 2026 THESIS.md or inventing main/PR merges.

## Foundational mission (not a game AI leaderboard)
A resource-bounded decision agent with state-derived beliefs, posterior terminal simple decision risk and explicit opportunities to buy computations. Inputs: prior/current computation traces, uncertainty calibration, possible meta-action observations, constrained resources and utility/cost exchange. Output: COMPUTE(choice c) or STOP, with predicted decision loss/cost and provenance. Games, theorem-proving, planning and scientific computation are **test domains**, not target-specific controller logics. An optional game-domain forced move is a safety constraint at the GameState adapter boundary, not a hidden selector inside the general controller.

## Four priority research tracks after this cycle
T0 (current): mathematical semantics and falsifiers. Separate posterior risk, gross EVSI, cost index, net gain, finite-horizon Bellman and action omission; deliver exact and floating-game-neutral referential code. This directly addresses the May BQ++/July P1 consistency question.

T1: calibrated observation models. Fit and validate probability of future root-statistics or decision changes under actual continuing MCTS computations, with decision group/time family splits, out-of-fold reliability and calibration drift monitoring. Distinguish observation-model accuracy from search performance. If calibration fails, keep controller in SHADOW, never ship a self-certifying stopping policy.

T2: SearchPolicy integration and computation morphology. Use existing generic SearchSnapshot/EdgeView/immutable cache; separate STOP, MORE_SAMPLES, WIDEN, CHALLENGE, DEEPEN, VERIFY and REPLAY as available meta-actions with actual cost. No game feature/pattern leakage into generic policy. Benchmark O(1) hot path (P4) and observable state/backups. Evaluate against legacy controller and cost-normalized ablations; stop on actual adverse quality.

T3: cross-domain transfer and emergent statistics. Benchmark at least two structurally distinct test domains and utility families before generalization; frozen evaluator, matched compute budget and grouped source sampling. The original P2 >=30% nn-evals reduction on Gomoku7 is a historical targeted benchmark, not an excuse to optimize Gomoku exclusively; it must not be promoted from one example to a game-independent gain. P3 human-expert-like macro/signature dissociation remains a secondary *non-circular* empirical prediction, never tuned directly.

## Revised acceptance and stop logic
- A principled computation rule must explicitly say which objective it optimizes (rate, net priced, fixed budget or multi-horizon), in what units, under what posterior model.
- Claiming general controller superiority requires independent evaluated task families, real CPU/NN costs, exact evaluator/checkpoint identities, paired seeds, and uncertainty; negative evidence preserved without gate inflation.
- A test with one or more gameplay domains is not a universal law; zero action change is not a positive value-of-computation result. Human cognitive interpretation requires independent cognitive evidence.
- Historical physics analogies in docs/legacy/mcts_study are *hypothesis-generation sources*, not automatically physically derived algorithms. Their rigorous crosswalk requires separate mathematical proof if promoted.
- No unnecessary global refactor, no deletion of frozen prior research, no new 26-axis meta-gates. Preserve existing legacy and foundry modules as comparators, not mutually exclusive failures.

## Transfer to coding C1-C10
Deliver the small finite-Bayes metalevel reference implementation with typed input validation, a real Rust source module and Python numeric/semantic tests. Compute exact G, net and finite-horizon V for generic synthetic decision tasks, then inspect noninferiority claims separately. Do not change the deployed MCTS control law yet; production integration awaits later evidence.

Preservation: individual stage archives under /home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta17_backups; new isolated branch research/quartz-general-metareasoning-identity-20261122; original developer worktree contains an untracked context file which must remain untouched. No native delegated subagent is exposed: NOT_RUN, serial fallback.

**Next independent research after C10:** calibrate a genuine computation observation channel from frozen root continuation traces and test if the finite Bayesian model's predicted terminal risk change matches observed outcomes *across distinct environments*, then propose guarded SearchPolicy integration. This step is not attempted/claimed in this reference-scope cycle.