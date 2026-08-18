# QUARTZ Invariants & Safety Rules

1. **Feature Isolation**: Gated Rust modules under `src/mcts/foundry/` must compile with `--features idea-foundry` and remain unreferenced in non-feature builds.
2. **Deterministic Reproducibility**: Seed enforcement must be preserved across multi-threaded MCTS runs.
3. **No Mock Results**: Scientific tests and experiment logs must derive from real engine executions.
4. **Subprocess Resilience**: Checkpoints must serialize full metadata (`model_state_dict`, `cfg`) to allow seamless resumption.
