# Search Controller Ablation Guide

## Three Ablation Levels

### Level 1: Search-Only (Rust, no training)
Fixed positions, fixed evaluator, fixed budget. Measures controller effect on search stability.

### Level 2: Virtual Loss Ablation (Rust, parallel search)
Measures adaptive VL effect on parallel search quality and duplicate suppression.

Key metrics (all printed by default):
- Agreement: move match vs serial reference
- DupRate: fraction of selections hitting already-pending edges
- MaxPending: peak concurrent pending leaves (contention indicator)
- AvgVV: mean vvalue applied (should be ~sigma_Q for Adaptive, 1.0 for Fixed)
- Entropy: root visit diversity
- NPS: throughput

### Level 3: Training-Level (Rust+NN, full pipeline)
Same NN, same game, different controller. Measures controller effect on training quality.
All training uses Rust+NN (search_nn protocol).

Recommended profile matrix for current runs:

- `--search-profile quartz`
- `--search-profile baseline`
- `--search-profile baseline_strict`

Use the same systems substrate and monitor settings across all three.

## What These Experiments Measure
- Search-only (Level 1): Controller effect under stylized evaluators
- VL ablation (Level 2): Parallel search quality, duplicate suppression, pending control
- Training-level (Level 3): Controller effect on NN training quality via Rust+NN self-play
- c_puct sweep: Whether controller gains survive exploration-strength normalization
- Arena: Direct model strength comparison with Wilson CI

### Current execution hygiene for fair ablation

- Keep runtime autotune off unless explicitly profiling tuner behavior.
- Keep eval/self-play isolation on for reproducible checkpoint-eval timing.
- Compare profiles on identical command shape except `--search-profile`.

## Known Limitations
- Training uses Rust+NN only (search_nn protocol)
- ShortRollout is for Level 1 search-only ablation, not training
- Adaptive VL control law: vvalue = sigma_Q × depth_decay × entropy_factor × contention_amplifier. State-derived with fixed constants (no learned or user-tuned hyperparameters). Inputs: sigma_Q, root_entropy, dup_rate, max_pending, n_threads.
- Arena uses SPRT early stopping with Wilson CI

## Legacy Components (not used in training)
- TreeMCTS: lightweight arena evaluation helper (QUARTZ-lite penalty modes)
- selfplay_rust(): Tier 1 ShortRollout, search engine smoke testing only
