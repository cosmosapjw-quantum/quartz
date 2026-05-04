# Audit — BQ++ Phase 6: Nested-reservoir live-set

**Date:** 2026-05-04
**Scope:** Rust port of the nested-reservoir live-set primitive from
the Phase 1 Python prototype (commit `32e5ea9`). "Nested-reservoir
search" — borrowing the live-set + threshold maintenance idea from
Skilling 2006 nested sampling, **explicitly NOT** nested sampling
for evidence estimation. Per audit §6.4 / §13.6.

## What changed

### New file `src/mcts/policy/reservoir.rs`

- `lambda_score(upper_ci, kg, log_prior_smoothed, rho, tau) -> f32`:
  computes `Λ_a = U_a + ρ·KG_a + τ·log π̃_0(a)`. The 3-component
  ranking score combines the EB upper CI ("could plausibly become
  best"), the Knowledge Gradient ("worth investigating further"),
  and the network's prior (smoothed log).
- `quantile(values, q) -> f32`: numpy-compatible linear-interpolation
  quantile.
- `Reservoir` struct: live-set with cooldown hysteresis.
  - `live: SmallVec<[u16; 32]>`: currently-live edge-local positions.
  - `cooldown_until: HashMap<u16, u32>`: per-arm cooldown expiry.
  - `add(idx, current_iter) -> bool`: returns false if already
    live, max_size reached, or in cooldown.
  - `remove(idx, current_iter) -> bool`: starts cooldown for
    `cooldown_iters` iterations.
  - `prune_below_quantile(scores, q, current_iter) -> Vec<u16>`:
    removes arms with score below the q-th quantile of the explicitly-
    scored set; returns the list of removed arms. Arms without an
    explicit score entry are **automatically** pruned (NOT
    `f32::NEG_INFINITY`-treated) to avoid the degenerate case where
    a -inf collapses the quantile threshold and prevents any strict-
    less-than match.

### `src/mcts/policy/mod.rs`

Re-exports `lambda_score, quantile, Reservoir`.

## Tests added (11)

1. `test_phase6_lambda_score_components`: Λ = U + ρ·KG + τ·log π
   hand-derived: U=0.6, KG=0.05, log_prior=−1.0, ρ=1.0, τ=0.1 →
   0.55.
2. `test_phase6_quantile_single_value`: |[5]|=1 → q(0.25) = 5.
3. `test_phase6_quantile_two_values`: |[0, 10]|=2 → q(0.5) = 5
   (linear interpolation).
4. `test_phase6_quantile_clean_25th`: [1,2,3,4] → q(0.25) = 1.75
   (hand-derived: pos = 0.25·3 = 0.75, lo=0, hi=1, frac=0.75 →
   1·0.25 + 2·0.75 = 1.75).
5. `test_phase6_quantile_empty`: empty → 0.
6. `test_phase6_reservoir_add_respects_max_size`: 3rd `add` to
   max_size=2 reservoir returns false.
7. `test_phase6_reservoir_remove_starts_cooldown`: removed at
   iter=5, cooldown=10 ⇒ ineligible until iter ≥ 15.
8. `test_phase6_reservoir_quantile_pruning`: 4 arms with scores
   {0:0.9, 1:0.7, 2:0.5, 3:0.3}; q=0.25 quantile = 0.45; arm 3
   (score 0.3 < 0.45) is pruned.
9. `test_phase6_reservoir_no_thrashing`: removed at iter=100,
   cooldown=200 ⇒ cannot re-enter at iter=150 (still in cooldown);
   can re-enter at iter=301 (past cooldown).
10. `test_phase6_reservoir_empty_prune_noop`: empty live set →
    no removal.
11. `test_phase6_reservoir_missing_score_pruned`: arm without an
    explicit score entry is auto-pruned regardless of threshold.

## Test results

- Phase 6 tests: 11/11.
- `cargo test --release`: **528 passed** (was 517 + 11 from Phase 6).
- All P01-P08 + Phase 0-5 tests still pass.

## Bug caught during port

Initial `prune_below_quantile` used `scores.get(i).unwrap_or(&NEG_INFINITY)`
which produced a degenerate behavior: when at least one live arm had
no score, the quantile threshold collapsed to NEG_INFINITY, and the
strict-less-than comparison `s < NEG_INFINITY` was false for *every*
finite score, so nothing was pruned.

Fix: compute the threshold using only arms with explicit scores; treat
missing-score arms as auto-prune. This keeps the threshold at a
meaningful value AND ensures no-score arms (which represent unranked /
unsupported candidates) leave the live set deterministically.

The Python prototype's behavior is the same; the Rust port now matches.

## What this does NOT do yet

- **No engine integration.** The reservoir is a pure data structure;
  no policy currently uses it. Phase 8 will compose it with Gumbel SH,
  KG-stop, EB cert, and the tactical sentinel into the BQPP policy.
- **No replenishment logic.** When the live set is below `max_size`
  (e.g. after pruning), the BQPP policy will use Gumbel sampling
  from `unexplored ∪ low_prior_high_uncertainty` to fill the gap.
  That logic lives in the BQPP integration, not in the reservoir
  primitive.

## Adversarial review

### Why HashMap and not BTreeMap?

`HashMap<u16, u32>` for cooldowns: O(1) amortized lookup. The number
of unique arms ever in cooldown is bounded by total search iterations
÷ check_interval, typically ≤ 100 entries. HashMap memory overhead
(~48 bytes/entry on 64-bit) × 100 = 4.8 KB. Acceptable.

### Cooldown semantics

The audit specifies "cooldown_iters" as a single scalar (default
200 = 2×check_interval). A future enhancement could decay this
exponentially based on how often the arm has been removed in the
past — but the current scalar is sufficient for the audit's
"prevent thrashing on borderline candidates" goal.

### Threshold semantics: strict-less-than vs less-than-or-equal

The Python prototype uses strict `<`: arms with score *equal* to
the quantile are kept. The Rust port preserves this. Important
because the quantile is itself a value from the score set when
`q` falls exactly on an interpolation node; arm with that score
should not be auto-pruned.

### Missing-score auto-prune

This is a deliberate semantic: "if you can't score this arm, remove
it from the live set." This catches the case where an arm's
metadata becomes stale (e.g. its `KG` is no longer computable
because it was bumped from the candidate set). Better to remove
than to keep ranking against `NEG_INFINITY`.

The alternative ("keep arms without scores at the bottom of the
ranking") would mean the live set retains stale arms forever. Not
desirable.

## Files touched

- `src/mcts/policy/reservoir.rs` (NEW; 270 LOC)
- `src/mcts/policy/mod.rs` (+2 LOC; module + re-exports)

Net delta: **+272 / 0 LOC**.

## What unblocks next

- **Phase 7 (MENTS)**: opt-in soft-Bellman policy. Independent of
  the reservoir; can land in any order.
- **Phase 8 (battle)**: composes Gumbel SH + EB cert + KG-stop +
  tactical sentinel + reservoir into the production BQPP policy,
  wires it into the engine, runs the 7-system experiment matrix.
