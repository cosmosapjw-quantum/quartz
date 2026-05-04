# Audit — BQ++ Phase 3: Gumbel SH root candidate scheduler

**Date:** 2026-05-04
**Scope:** Rust port of Gumbel without-replacement sampling +
Sequential Halving (Danihelka et al. ICLR 2022). Direct match for
the user's primary objective — reduce NN-evals per move while
preserving (or improving) play quality. The Python prototype at
`prototype/bqpp_prototype/gumbel_sh.py` (Phase 1, commit `32e5ea9`)
validated the math; this patch ports to Rust with cross-checked
expected values.

## Why Gumbel SH

Pure PUCT at root visits all candidates roughly proportional to prior.
For low-budget self-play (few NN evals), this means many "wasted"
visits on the lowest-prior arms even when their value is clearly
worse than the empirical best. Sequential Halving (Karnin-Koren-
Somekh 2013) is the optimal pure-exploration allocator: it
concentrates budget on candidates that survive successive halving
rounds.

Combining SH with Gumbel-top-m candidate selection at the start
(rather than fixed top-prior) gives us **policy improvement
guarantees with far fewer simulations** — the headline result of
Danihelka et al. 2022.

## What changed

### New file: `src/mcts/policy/gumbel_sh.rs`

- `sample_gumbel<R: Rng>(rng) -> f32`: inverse-CDF Gumbel(0, 1) via
  `-ln(-ln(U))`. U clamped to `(1e-12, 1 - 1e-12)`.
- `gumbel_top_m(log_priors, m, rng) -> SmallVec<[u16; 32]>`:
  without-replacement Plackett-Luce sampling. Returns the m largest
  perturbed log-prior indices in descending order. Stable: same
  RNG state ⇒ same selection.
- `SequentialHalvingBracket`:
  - `candidates: SmallVec<[u16; 32]>` — currently-live edge-local
    positions.
  - `n_total_rounds()`: `⌈log₂(m₀)⌉` for `m₀ ≥ 2`, else `1`.
  - `round_budget()`: `⌊B / (m_r * total_rounds)⌋` per arm per round.
  - `is_done()`: `len ≤ 1` or all rounds completed.
  - `advance_round(arm_means) -> Self`: drops bottom half by mean,
    returns new bracket. Anytime-resumable.
  - `select_winner(arm_means) -> u16`: argmax over live candidates.
- `initial_bracket(log_priors, m_initial, budget, rng) -> Bracket`:
  convenience constructor.

### Re-exports

`src/mcts/policy/mod.rs` re-exports:
`gumbel_top_m, initial_bracket, sample_gumbel, SequentialHalvingBracket`.

## Tests added (10)

1. `test_phase3_gumbel_sample_mean`: 200K samples, mean ≈ 0.5772
   (Euler-Mascheroni constant).
2. `test_phase3_gumbel_top_m_count`: returns exactly m, no duplicates.
3. `test_phase3_gumbel_top_m_concentrates_on_strong_prior`: π=0.97
   ⇒ top-1 picks arm 0 with prob > 93% over 1000 runs.
4. `test_phase3_gumbel_top_m_uniform_distributes`: uniform prior
   gives 25%±5pp per arm over 4000 runs.
5. `test_phase3_gumbel_top_m_empty_input`: empty in → empty out.
6. `test_phase3_sh_bracket_n_total_rounds`: ⌈log₂(m₀)⌉ for several
   m₀ values.
7. `test_phase3_sh_advance_halves_candidates`: 4 arms with means
   [0.9, 0.7, 0.5, 0.3] → [0, 1] after round 1 → [0] after round 2.
8. `test_phase3_sh_select_winner`: argmax over live set.
9. `test_phase3_sh_resumable_property`: same RNG seed produces
   identical winner regardless of pause-and-resume timing.
10. `test_phase3_sh_round_budget_at_least_one`: 4 arms, budget 8
    ⇒ 1 visit per arm per round.

Cross-validation: tests 1, 3, 4, 9 mirror the Python prototype
tests by name (`test_sample_gumbel_distribution_mean`,
`test_gumbel_top_m_concentration_on_strong_prior`,
`test_gumbel_top_m_uniform_distributes`, `test_sh_resumable_property`)
with the same hand-derived expected values.

## Test results

- `cargo test --release`: **495 passed** (was 485 + 10 from Phase 3).
- All P01-P08 + Phase 0-2 tests still pass.

## What this does NOT do yet

- **No engine integration.** The `SearchPolicy` trait does not yet
  expose an `allocate(snap, edges) -> Option<u16>` method. That
  integration is part of BQ++ Phase 4 (KG-stop) or later when the
  scheduler is plumbed through the engine's selection loop.
- **No `MctsConfig` flag.** `--policy=gumbel_sh` is not yet a valid
  CLI choice. Adding it would be premature without engine
  integration.

This patch ships the **scheduler primitive**. Phase 4 will be the
first to consume it inside a live policy.

## Adversarial review

### What if log_priors contains NaN / -inf?

A zero prior gives `log_prior = -inf`. Adding a Gumbel sample yields
`-inf`, which sorts last. `argmax` then never picks the zero-prior
arm — correct behavior. NaN priors would corrupt the sort; the
caller is responsible for clamping priors to a small ε before passing
to `gumbel_top_m`. The audit's §6.4 candidate-reservoir formula
includes the clamp `π̃₀(a) = (1 - ε) π₀(a) + ε / K` precisely for
this reason; the Phase 3 module trusts the caller to apply it.

### Anytime property

The resumable test pins the property: if you compute the bracket in
two halves (round 1, then rounds 2+) you get the same final winner
as if you compute it all at once, given the same RNG seed and
arm_means. This is a deterministic-given-inputs property, not a
probabilistic one — a stronger guarantee than the "anytime in
expectation" version.

### SmallVec spill

`SmallVec<[u16; 32]>` is stack-allocated up to 32 candidates;
spills to heap above. For Gomoku 7×7 (49 max actions), Gomoku 15×15
(225 max), Go 9×9 (82 max) typical n_children stays under 32 because
Phase 5's tactical sentinel and the candidate-reservoir cap further
trim the set. Chess (4672 action slots, but typically ~30 legal
moves at any position) also stays under 32. The spill case is
correct, just slower.

### Float comparison in `select_winner`

`partial_cmp` returns None for NaN. The fallback is `Ordering::Equal`.
A NaN arm_mean would be treated as tied with everything else, which
is benign. Caller is responsible for avoiding NaN means.

## Files touched

- `src/mcts/policy/gumbel_sh.rs` (NEW; 290 LOC)
- `src/mcts/policy/mod.rs` (+4 LOC; module + re-exports)

Net delta: **+294 / 0 LOC**.

## What unblocks next

- **Phase 4 (KG-stop)** is the first phase that integrates Gumbel SH
  into a live policy. Phase 4's policy will:
  1. Use `gumbel_top_m` to build the candidate reservoir at observe
     time.
  2. Use SequentialHalvingBracket to drive the per-round visit
     allocation.
  3. Use the empirical-Bernstein certificate (already in
     `cache::PolicyCache.cert_gap`) for halt.
- **Phase 6 (nested-reservoir)** will use the `gumbel_top_m` primitive
  for replenishing the live set when arms drop below the quantile
  threshold.

## Cross-language verification

The Python prototype's `test_gumbel_top_m_concentration_on_strong_prior`
expects ≥93% top-1 on `[0.97, 0.01, 0.01, 0.01]` over 1000 runs.
The Rust port's `test_phase3_gumbel_top_m_concentrates_on_strong_prior`
asserts the same threshold. Empirically the rate hovers around 95-97%
in both languages — well above the 93% lower bound — confirming the
cross-port behavioral parity.
