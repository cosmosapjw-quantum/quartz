# Audit — BQ++ Phase 4: KG-per-cost stop rule

**Date:** 2026-05-04
**Scope:** Rust port of the Knowledge Gradient stop rule. The
mathematical foundation is from Phase 1 Python prototype
(`prototype/bqpp_prototype/voi.py` + `kg.py`, commit `32e5ea9`).
This phase is the first that ships the audit's §1.2 correction in
production-path code: the FULL `s · φ(Δ/s) − Δ · Φ(−Δ/s)` formula,
**not** the wrong `s · φ(z)` overestimate from the cancelled P09
plan.

## What changed

### New file `src/mcts/policy/kg_stop.rs`

- `standard_normal_pdf(z)`: φ(z) = (1/√(2π)) · exp(−z²/2).
- `standard_normal_cdf(z)`: Φ(z) via `erf_f32` (Abramowitz-Stegun
  7.1.26 approximation, max error 1.5e-7).
- `expected_improvement(delta, s) -> f32`: full EI formula with
  edge-case clamping. **Floors at 0** because f32 erf approximation
  noise can produce small negative values for large |z| where φ
  and Φ both → 0; clamping preserves the monotonicity property.
- `kg_per_arm(mu_a, n_a, sigma2_a, mu_b, n_b, sigma2_b, lambda0)`:
  Knowledge Gradient for one challenger arm.
- `compute_kg_array(...) -> SmallVec<[f32; 32]>`: per-arm KG with
  `kg[best_pos] = 0` convention.
- `should_halt_by_kg(kg_array, n_total, min_total, kg_threshold,
  cost_per_pull_ms) -> bool`: stop iff `n_total >= min_total` AND
  `max_a kg_a < kg_threshold * cost_per_pull_ms`.

### `src/mcts/policy/mod.rs`

Re-exports: `compute_kg_array, expected_improvement, kg_per_arm,
should_halt_by_kg`.

## Tests added (15)

1-2. φ(0) ≈ 0.3989; Φ(z) + Φ(−z) = 1 (sanity).
3. EI at Δ=0 = s · φ(0) (hand-derived).
4. EI is **non-increasing** in Δ for fixed s. Note: uses ≤ rather
   than strict < because f32 erf noise makes consecutive large-Δ
   values both round to 0.
5. EI decays to 0 at Δ → ∞ (clear-loss arm, 100σ).
6. EI = 0 at s = 0 (no uncertainty ⇒ no improvement).
7. EI clamps negative Δ (caller bug protection).
8. KG[best] = 0 by convention.
9. KG > 1e-6 for sub-optimal arms (test inputs chosen above f32
   erf-approximation noise floor: μ=[0.55, 0.50, 0.45], σ²=0.04,
   n=50 each ⇒ EI ≈ 0.0017).
10. KG monotone in σ_a (variance-adaptive).
11. KG monotone in 1/n_a (less-pulled ⇒ larger KG).
12. should_halt_by_kg respects min_total (no halt below).
13. should_halt_by_kg fires at low KG.
14. should_halt_by_kg does NOT fire at high KG.
15. KG cross-check against Python prototype at less-extreme inputs
    (μ_a=0.6, μ_b=0.7, σ²=0.04, n=50/100, λ₀=4 ⇒ KG ≈ 1e-5).

## Bugs caught and fixed during port

### A. Rust doesn't support keyword args

My initial test file used `should_halt_by_kg(&kg, n_total=200,
min_total=100, kg_threshold=1e-3, cost_per_pull_ms=1.0)` which is
Python syntax. Rust expressions like `n_total=200` are interpreted
as **assignments to local variables** that don't exist, producing
`E0425: cannot find value 'n_total' in this scope`. Fixed by
converting to positional args.

### B. f32 erf approximation noise

My initial `expected_improvement` returned the raw `s·φ(z) −
Δ·Φ(−z)` formula. For large |z| (Δ ≫ s), both terms approach 0
and the f32 erf approximation produces small *negative* values
like −3.4e-10 due to the truncated polynomial. The
"non-monotone at delta=1" test failure was the symptom.

Fix: clamp EI at 0 since it's mathematically non-negative.
Documented in code comments as the "f32 noise floor" phenomenon.
The Python prototype uses scipy's f64 erf and doesn't have this
issue — but the Rust port uses Abramowitz-Stegun's f32 approximation
for performance, so the clamp is the right tradeoff.

### C. Test inputs in the f32 underflow regime

My initial `test_phase4_kg_positive_for_subopt_arms` used Δ=0.3, s≈0.034
which gives z ≈ 8.9 — well into the regime where φ(z) ≈ 0 in f32.
KG was `0.0` after the EI clamp, and the assertion `kg > 0` failed.

Fix: choose less-extreme inputs (Δ=0.05, s≈0.039 ⇒ z≈1.3, EI≈0.0017
which is well above the f32 noise floor). Documented why in the
test comment.

## Test results

- Phase 4 tests: 15/15 passing.
- `cargo test --release`: **510 passed** (was 495 + 15 = 510).
- All P01-P08 + Phase 0-3 tests still pass.

## What this does NOT do yet

- **No engine integration.** `should_halt_by_kg` is a pure function;
  no policy currently calls it inside the engine's selection loop.
  Phase 8 will add a `BQPP` policy that combines:
  1. `gumbel_top_m` from Phase 3 for candidate selection.
  2. `SequentialHalvingBracket` from Phase 3 for visit allocation.
  3. `expected_improvement` + `compute_kg_array` from this phase
     for VOI tracking.
  4. The EB certificate from `cache::PolicyCache.cert_gap` (Phase 2).
  5. `should_halt_by_kg` for the value-of-information stop.
  6. Tactical sentinel from Phase 5 for forced-move detection.
- **No marginal-Elo-per-ms calibration.** The plan calls for setting
  `kg_threshold = marginal_elo_per_ms / 2` derived from offline
  self-play data. Phase 8's experimental matrix will calibrate this
  per-game.

## Adversarial review

### KG-stop vs EB certificate priority

The audit's §6.4 module structure makes EB certificate the
**primary** halt rule for NN-driven value backups, because the
KK13 PAC guarantee assumes iid Bernoulli samples (which MCTS
violates). The KG-stop is a **secondary** halt rule for "the
remaining computation is no longer worth the cost."

Phase 8 will compose them:
```
if cert_gap > 0          ⇒ Stop(EmpBernsteinCertified)  (primary)
else if max_kg < threshold ⇒ Stop(PolicyConverged)      (secondary)
else if n >= max_visits  ⇒ Stop(MaxVisits)              (hard cap)
else                     ⇒ Continue
```

### `kg_threshold * cost_per_pull_ms` semantic

The product has units of "value units per ms × ms = value units."
This is the marginal value gained from one more pull. Comparing
`max_kg` to it makes dimensional sense: stop when the best
remaining computation has value below the per-pull cost.

In practice this requires `kg_threshold` to be in units of
"value-units-per-ms" — which depends on the game and the
network's value calibration. The Phase 8 calibration will set
this from offline self-play.

### What if `max_kg` is itself in the f32 noise floor?

If all KG values round to 0 (very clear-lead, large gaps), then
`max_kg = 0 < kg_threshold * cost_per_pull_ms` for any positive
threshold, and `should_halt_by_kg` returns true. This is the
correct behavior — there's literally no value left to gain.

The corner case to watch: if `min_total` is set very low (e.g. 10)
AND the search starts with all-zero KG (not enough pulls to even
estimate σ_a), the policy could halt prematurely. Mitigation: the
caller should set `min_total >= 100` AND require all arms to have
at least `min_pulls` (≥ 30) visits before consulting KG-stop.
This is precisely the structure in the Phase 1 prototype's
`run_controller` function.

## Files touched

- `src/mcts/policy/kg_stop.rs` (NEW, 280 LOC)
- `src/mcts/policy/mod.rs` (+5 LOC; module + re-exports)

Net delta: **+285 / 0 LOC**.

## What unblocks next

- **Phase 5 (tactical sentinel)**: writes `forced_move_pos` into
  the cache. The KG-stop logic from this phase will need to be
  *suppressed* when a forced move is detected (Phase 8 composition).
- **Phase 6 (nested-reservoir)**: uses `kg_per_arm` to compute the
  `Λ_a = U_a + ρ · KG_a + τ · log π̃_0(a)` ranking score for the
  live-set quantile pruning.
- **Phase 8 (battle)**: composes Gumbel SH + EB certificate + KG-stop
  + tactical sentinel into the production `BQPP` policy. This is
  the first Phase that wires everything into the engine.
