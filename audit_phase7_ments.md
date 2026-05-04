# Audit — BQ++ Phase 7: MENTS soft-Bellman primitives

**Date:** 2026-05-04
**Scope:** Maximum Entropy MCTS primitives (Xiao et al. NeurIPS 2019).
Soft-value, soft-policy with exploration-floor smoothing, and KL
convergence criterion. Useful for single-agent / non-zero-sum
contexts; **NOT the default** for AlphaZero zero-sum games.

## What changed

### New file `src/mcts/policy/ments.rs`

- `soft_value(q_values, tau) -> f32`: V_soft(s) = τ · log Σ_a exp(Q_a / τ).
  Numerically stable via log-sum-exp shift by max(Q). Edge cases:
  τ ≤ 0 ⇒ max(Q); empty input ⇒ −∞.
- `soft_policy(q_values, n_state, tau, epsilon) -> SmallVec<[f32; 32]>`:
  Xiao et al. 2019 Algorithm 1's mixture:
    λ_s = ε · K / log(2 + N(s))
    π_soft(a | s) = (1 − λ_s) · softmax(Q / τ) + λ_s / K
  τ ≤ 0 ⇒ delta on argmax. Sum-to-one preserved by construction.
- `kl_visit_to_soft(visit_counts, soft_policy) -> f32`: KL(π_visit ‖
  π_soft) = Σ_a p_a · log(p_a / q_a). Convergence criterion: halt
  when this is below `kl_threshold` (default 1e-3 per the audit's
  Phase 7 spec). Zero-visit terms contribute 0.

### `src/mcts/policy/mod.rs`

Re-exports `kl_visit_to_soft, soft_policy, soft_value`.

## Tests added (11)

1. `test_phase7_soft_value_at_zero_tau`: τ → 0 ⇒ V_soft = max(Q).
2. `test_phase7_soft_value_at_unit_tau`: V_soft = log_sum_exp(Q).
   Hand-derived: Q=[0,0,0], τ=1 ⇒ V_soft = log(3).
3. `test_phase7_soft_value_entropy_bonus`: nearly-tied Q ⇒ V_soft
   > max(Q) (entropy contribution).
4. `test_phase7_soft_value_empty`: empty input → -∞.
5. `test_phase7_soft_policy_delta_at_zero_tau`: τ=0 ⇒ delta on argmax.
6. `test_phase7_soft_policy_sums_to_one`: probability mass = 1.
7. `test_phase7_soft_policy_high_epsilon_is_near_uniform`: ε ↑ ⇒
   distribution closer to uniform (relative comparison).
8. `test_phase7_kl_visit_to_soft_convergence`: Q=[0.5, 0.495], τ=0.01,
   visits=[60, 40] ⇒ KL < 0.005.
9. `test_phase7_kl_zero_on_match`: π_visit = π_soft ⇒ KL = 0.
10. `test_phase7_kl_positive_on_divergence`: π_visit=[0.9, 0.1],
    π_soft=[0.5, 0.5] ⇒ KL ≈ 0.368 (hand-derived).
11. `test_phase7_kl_zero_visit_arm`: zero-visit term contributes 0
    correctly. Hand-derived: KL([0.5, 0.5, 0] ‖ [0.5, 0.4, 0.1]) ≈ 0.112.

## Test results

- Phase 7 tests: 11/11.
- `cargo test --release`: **539 passed** (was 528 + 11 from Phase 7).
- All P01-P08 + Phase 0-6 tests still pass.

## Adversarial review

### When MENTS is appropriate

The Xiao 2019 paper proves exponential best-arm convergence for
single-agent / fully-cooperative RL settings. AlphaZero's zero-sum
games (chess, Go, Gomoku) are NOT a clean fit — soft-Bellman backups
introduce a temperature-dependent bias that doesn't match the
optimal-strategy structure of zero-sum minimax. MENTS is therefore
shipped as **opt-in** (`--policy=ments` in Phase 8 CLI), not as a
default.

Use cases where MENTS shines:
- Single-agent puzzle games (Sokoban, mazes).
- Cooperative multi-agent (Hanabi-like settings).
- AlphaGo Zero variant where the value head is well-calibrated and
  the entropy regularization explicitly rewards exploration.

For Gomoku 7×7 / 15×15 / Go 9×9 / Chess, EB-cert + KG-stop +
Gumbel SH (Phases 2-4) are the canonical primitives.

### Numerical stability

Soft-value with f32 arithmetic: at large |Q/τ|, the exp can overflow.
The log-sum-exp shift handles this — we subtract max(Q) before
exponentiating, so the largest exponent is 0. f32 can hold exp(0) =
1 trivially. The smallest exponent is `(min_Q - max_Q) / τ`; if
this is below ≈ −86 (where exp underflows to 0), the contribution
is silently dropped. Acceptable for our purposes.

KL with clamp on q at 1e-12: prevents log(0) crashes when soft_policy
has a zero entry. In practice soft_policy never produces exact
zeros (the λ_s/K floor adds at least 1/(K · log(2 + N))), but the
clamp is defensive.

### What is NOT in this patch

- **Soft-Bellman backup integration with the search tree.** This
  module ships only the per-state soft-policy / KL primitives. The
  full integration — replacing the standard PUCT backup with a
  soft-Bellman backup at every internal node — is a significant
  engine refactor that the plan defers to "MENTS opt-in" phase.
  For Phase 7 this is just the math primitive; Phase 8 will add
  a thin policy wrapper for `--policy=ments`.
- **Temperature scheduling.** The plan calls for τ to be tied to
  an entropy target. Static default τ=0.01 is used in the tests;
  Phase 8's policy wrapper will plumb the schedule.

## Files touched

- `src/mcts/policy/ments.rs` (NEW; 220 LOC incl. tests)
- `src/mcts/policy/mod.rs` (+2 LOC; module + re-exports)

Net delta: **+222 / 0 LOC**.

## What unblocks next

- **Phase 8 (battle)**: composes Gumbel SH + EB cert + KG-stop +
  tactical sentinel + nested reservoir into the production BQPP
  policy. MENTS gets a thin wrapper as `--policy=ments` for the
  experimental matrix. Then runs the 7-system comparison.
