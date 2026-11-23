# R08 — per-claim decisions

| Claim / mechanism | Evidence and exact domain | Decision |
|---|---|---|
| Incumbent utility never decreases | Direct max-update proof for exact-known actions | Adopt as hard production invariant; empirical oracle check still required |
| One-step all-nonpositive iff global STOP | Direct finite proof, 9,495 exact cases, independent full-reveal fixed-cost product model | Adopt conditional theorem; actual games use a model-based decision |
| Reservation ordering is optimal | Established Weitzman rule, direct capped-value proof, exact Bellman agreement | Adopt primary full-horizon model policy; binding caps are comparisons, not theorem coverage |
| Greedy ordering is generally optimal | Exact two-box and three-box counterexamples | Reject this unrestricted statement; keep GREEDY_VOC as distinct control |
| Gibbs realizes KL objective | Exact variational identity, symbolic/numeric checks | Implement actual seeded Gibbs sampling with explicit b=1 certificate mask |
| Finite-tau Gibbs maximizes raw net | Negative-query support and exact recursive values | Do not admit; assess regularized behavior and actual costs separately |
| Statistical dynamics can realize Gibbs populations | Constructed detailed-balanced rate, off-diagonal Lindblad and CPTP Kraus map | Admit mathematical realization, retain algorithmic-time units; runtime uses direct sampling |
| Legacy diagonal jumps redistribute populations | Exact general-matrix diagonal derivative zero | Correct this specific construction using transition jumps |
| All uncertainty or information is decision value | Uninformative-channel and common-mode counterexamples | Use computation-specific posterior-mean covariance; keep exact query gain separate |
| Quadratic fluctuation term is uniformly accurate | Exact strong-signal gain .178002 versus .300180 approximation | Admit controlled small-noise expansion and explicit remainder only |
| Learned product model is calibrated in GameState | No held-out runtime evidence yet | Test in coding loop with TRAIN-only fit, calibration and correlation diagnostics |
| Actual positive priced game improvement | No production run yet | Prospectively freeze price and primary comparison; decide after held-out evidence |
| Original Gomoku P2 / cognitive P3 is closed | C26 exact-query TicTacToe model does not measure those targets | Remain open; no promotion or threshold change |

## Constructive next obligations

Use reachable game states, complete legal candidate sets, paid generic random-play cues, exact minimax queries and independent offline oracles. Freeze source groups and code contract before viewing policy outcomes. The correlated information-only-query example is an executable next-theory target, not a reason to stop the present exact-query implementation.

No new strong no-go is claimed, so NG1–NG7 promotion is not invoked. The two counterexamples defeat precise universal statements and supply replacements.
