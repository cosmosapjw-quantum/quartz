# R6. Adversarial assumptions, counterexamples and limits

1. Unknown utilities: real Gomoku15 C16 reports MCTS actions, not solved/minimax utility; revision event Y is not a decision improvement label. External finite-time Rapfi move is not exact minimax. No signed VOC claim from flip alone.
2. Non-IID shared tree: subsequent MCTS backups reuse root, evaluator and adaptive policy. Independence of samples conditional on stationary theta is not established. Source unit is parent played-game trajectory; within-parent root snapshots and repeats are clustered.
3. 20ms C16 endpoint includes root setup and all searches from first8 through end, excluding model disk load and output JSON. This does NOT implement exactly 20ms extra after 8 visits. Report actual wall and test only that frozen budget intervention.
4. Feature leakage: only initial first8 root counts/prior and predeclared source metadata may feed predictor. final_mcts_move, final_root_visits and achieved wall time are label/post-outcome fields; never features.
5. Parent leakage: position snapshots or repeated queries from same played history cannot straddle train, calibration or test. D4 rotations/reflections must not generate duplicate cross-split source. Freeze exact identity.
6. Non-game oracle: simulator-known true Bernoulli arm means unavailable online; used only to score signed utility after the computation. A posterior can change without changing selected action; information and revision are distinct.
7. Pooled calibration vs transfer: reports are per-domain, per-split plus pooled; domain-specific id coefficients are disallowed in main joint predictor. Passing one domain says nothing universal.
8. Falsifier: if all Y=0, event-conditioned value/correlation NOT_ESTIMABLE; no threshold/source rerouting after heldout reveal. Declare degenerate prevalence instead.
9. Shadow-only: a forecast with low Brier changes NO MCTS action, hence no CPU saving, gameplay quality or utility improvement claim. Any active policy intervention belongs to a later stage.
10. Verification: Python rule replay independently checks source histories; source-only Rust diagnostics are not themselves proof of regret. One bounded independent audit pass, no recursive reviews.

No strong novel no-go is claimed. Two-world signed-utility counterexample is narrow and constructive; game-specific MCTS is merely an instrumented source for a broader meta-observation experiment.