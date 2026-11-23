# Quartz C30 next full research cycle: calibrated metacomputation costs

## Scope and authority

The immutable user SSOT remains 87c15dc828d6eea519c219fd8a412fa21dfc464d, with C28 C10 and C29 C10 as precisely tagged scientific predecessors. C29 C09 corrected-source snapshot, C09_REAGGREGATION.json and C09_MEASURED_BINARY.json are the authoritative scientific data for continuation. C29 C08 historical measurements are superseded for current game conclusions. Separate branches, original PRD, C26/C27 and C28 histories must be preserved; never overwrite those worktrees.

C29 proved the exact finite noisy repeated-observation posterior/Bellman and native equivalence, but CAL realised zero average action utility change while extra rollouts had positive cost and full controller time was >4x baseline. The next problem is not more mathematics bookkeeping: get **a calibrated computational observation model and a cheaper policy with real CPU budget** that can actually produce downstream decisions worth their cost.

## Candidate scientific improvements

First, test whether a global likelihood C(Y|U) is insufficient after conditioning on cue histogram, legal move/remaining depth, candidate selection and rollout reuse. Compare global, feature-conditioned regularized and nonparametric bounded channel families using TRAIN-only; evaluate calibration and predictive log loss on a fixed CAL group, never retrain on it. Explicitly analyze repeated noisy rollouts: their independence conditional on exact minimax utility may fail because board geometry, shared seeds and cue history remain latent.

Second, model actual computing costs. Node prices 1/1024 utility/node are insufficient for additional H2 planner CPU time. Prefer enforcing a **fixed overall wall-time budget**, comparing policies with the same total computation time, including cue generation, likelihood updates, planning and requested rollout work. If a utility price for CPU time is used, calibrate it independently from an authorized measured opportunity-cost model and keep its units explicit; no arbitrary ns-to-utility scale.

Third, derive cheap stopping/selection certificates from action-gap geometry and a valid VOI upper bound, then implement a bounded policy that materially reduces H2 overhead on actual roots. Product/common/signed belief law ablations must share identical pre-observation candidate marginal inputs and selection budgets. Do not claim latent factor physically models TT geometry without a mathematical map.

## C30 theory R01–R10

| Step | Concrete theory output |
|---|---|
| R01 | Freeze C29 corrected source/input identity, original-origin watchdog, independent root policy and measurable success criterion at matched real CPU time |
| R02 | Identify exact propositions in original metareasoning and calibration/active-selection literature needed for these channel and policy hypotheses |
| R03 | Derive observation law conditional on cues, selected query and hidden board state; show when old C29 global likelihood is invalid |
| R04 | Derive an actual CPU-budget Bellman or constrained policy and a cheap valid VOI/STOP bound, with proper units |
| R05 | Construct exact small counterexamples that separate log-loss improvement, action crossing and negative priced utility |
| R06 | One bounded adversarial proof review of valid bound, selected covariates and conditional-independence assumptions |
| R07 | Rational/CAS minimal cases and independent finite-world oracle for the new observation/cost model |
| R08 | Decide which mechanism survives and preserve any negative or inconclusive cases without new global gates |
| R09 | Freeze measurement/fit/control/data partition, exact code mapping and fail criteria before CAL/TEST |
| R10 | Publish complete derivation with a runnable C01–C10 implementation handoff, preserve original motivation and open P2/P3 |

## C30 coding C01–C10

| Step | Concrete implementation and evidence |
|---|---|
| C01 | Current repo/branch/source/candidate root provenance and machine-resource identity; no edits before inspecting |
| C02 | Map observation likelihood, posterior state, STOP and budgeted action to native structures and exact inputs |
| C03 | Plan stable log-likelihoods, zero-support, probabilistic ties, numerical precision and actual CPU measurement boundaries |
| C04 | Fix bounded resources, training split, untouched CAL and genuinely new TEST availability; publish backup |
| C05 | Implement real cue-conditioned calibrated channel and inexpensive planner/STOP strategy in isolated binary |
| C06 | Run locked build, small exact world oracle, limits and malformed-input regression, preserve real stdout/stderr |
| C07 | Validate true GameState rollout observations, no oracle leakage and the frozen CAL calibration before outcomes comparison |
| C08 | Compare actual realized utility and CPU cost at matched total wall-time, record raw per-root and paired outcomes |
| C09 | One bounded science/source audit and actual restore of earlier C30 cumulative backup; fix real defects only |
| C10 | Publish final corrected report, unambiguous negative or positive admission, all-stage remote receipts and exact FINAL restoration audit |

Requirements: all selected R01–R10 first, C01–C10 after, actual verification per stage, push/readback and history-preserving backup before any previous C30-only duplicate deletion. New independent TEST is mandatory for an out-of-sample performance or Elo claim. With only the old C26 cohort, status remains retrospective. Distinguish true planning cost improvement from mere kernel microbenchmark speedup. Do not let the watchdog grow into process-only review work.
