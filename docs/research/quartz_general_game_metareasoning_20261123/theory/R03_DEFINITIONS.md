# R03 — definitions, objective and admissible claims

## State, computations and signs

Use the existing GameState trait and TicTacToe implementation unchanged. A root has player p_r in {-1,+1}; at a terminal descendant s, outcome(s) is current-player-relative. Set v_r(s)=p_r*p_s*outcome(s), then minimax in the root frame maximizes on root-player turns and minimizes on opposing turns. Candidate utility u_a=(v_a+1)/2 is 0, 1/2 or 1. No natural-units convention is used: utility is dimensionless normalized game reward; node price has utility/node, wall price utility/microsecond, and tau utility.

A FOCUS is the complete, uncached minimax traversal of one root child. It reveals a fixed exact utility, counts every visited state once per traversal, and is never repeated. This is a deliberately tractable actual-GameState bridge, not an MCTS policy replacement or Gomoku benchmark.

## Common paid initialization

Generate four uniform random playout cues per nonterminal root child. A terminal child is evaluated once, its exact value is marked known, and its feature counts encode four identical outcomes without pretending four traversals occurred. These feature counts are a representation, not four independent observations. Count actual evaluations. All legal root actions remain listed.

If a known terminal win exists, use the lowest-index such action as incumbent and skip the initial exact query. Otherwise choose the nonterminal candidate with largest ordinary rollout sample mean (ties smallest action index), solve it exactly, and retain the maximum of that result and already known terminal children. If all children are terminal, choose the best known value. Ties retain the lowest-index initial known action. This is the operative initialization, superseding the constructor packet's earlier prior-mean description.

All policies pay this same initialization. Deployment is restricted to exact-known candidates; further reveals update b to max(b,u). This proves pathwise nondegradation independently of model calibration. It does not certify positive net value after charging computation.

## Learned laws and closures

Training labels are exact minimax utilities. The empirical predictor uses the child's remaining legal count and four-cue loss/draw/win counts. Cue frequencies alone are not minimax probabilities. TRAIN-only smoothed categorical laws are frozen before CAL/TEST. The product over unqueried actions and fixed predicted node charges define a model, not true game independence or exact cost knowledge.

The model one-step gain is g_a(b)=p_a,h max(1/2-b,0)+p_a,1 max(1-b,0). Q_a=g_a-lambda*predicted_nodes_a; Q_STOP=0. Predicted and actual costs are both retained. Root values share game constraints; reveal-conditioned correlations can violate the model. No offline root-child oracle fields enter the runtime request.

## Claim classes

- Established / literature-supported: Weitzman index optimality under independence, fixed charges, exact reveal and unrestricted remaining-query horizon; KL/Gibbs variational identity.
- Derived here: explicit three-point index, safe known-incumbent mapping, exact STOP theorem and capped-value proof, consistent source corrections.
- Numerically checked: finite exact oracle and statphys realizations, only after R07 execution evidence.
- Conjectural until held-out execution: empirical cue calibration, useful query ordering and positive priced game gain.
- Unresolved: correlated posterior updating, partial computation, large branching MCTS integration, original P2/P3 empirical targets.
- No strong no-go is asserted. Counterexamples delimit specific transported theorems and guide constructive extensions.

A one-query or other binding additional-query cap preserves safety and the sufficient STOP condition; it can invalidate unrestricted Pandora ordering optimality. The full-horizon cohort is primary and finite caps are labeled secondary.
