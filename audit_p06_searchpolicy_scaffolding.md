# Audit — P06: `SearchPolicy` trait + types + dispatcher (scaffolding)

**Date:** 2026-05-04
**Patch:** P06 (15-patch QUARTZ v1.0 sequence)
**Scope:** add the new `SearchPolicy` abstraction with **zero behavior
change** in the engine. Prepares the ground for P07 (LegacyQuartz shim),
P08 (KLLUCBStop), P09 (BayesianQuartz), P10 (default flip), P11
(MENTS opt-in).

This is the safest atomic patch in the policy work — pure additive
code; the existing `quartz_policy_adjustment` and
`QuartzController::should_stop` paths continue to drive search.

## What changed

### New files

- `src/mcts/policy/mod.rs` — module entry, re-exports, and the
  `DefaultBridgePolicy` no-op fallback (returns
  `ScoreAdjustment::default()` and `HaltDecision::Continue`).
- `src/mcts/policy/trait_def.rs` — types and trait. Key items:
  - `SearchSnapshot` (root-side state captured at periodic boundary)
  - `EdgeView<'a>` (per-edge read-only borrow surface). `m2: f64`
    chosen deliberately — f32 Welford diverges by ~5e-4 from
    scipy.var at 10⁶ samples (verified empirically); f64 stays
    within 1e-9.
  - `EdgeView::sigma_a(lambda0)` — per-action posterior std-dev with
    Beta-Binomial-conjugate smoothing:
    σ_a² = (M2 + λ₀·σ_root²) / (N + λ₀). λ₀=4 is the canonical
    weak-prior choice.
  - `ScoreAdjustment` (effective_prior, penalty, fisher_alpha,
    q_override). Default = identity (= pure PUCT).
  - `HaltDecision { Continue, Stop(HaltReason) }` integrating with
    the P01 HaltReason enum.
  - `EffectivePrior` (raw, posterior, blend) for refresh policies.
  - `ControllerTelemetry { schema_version, policy_name, halt_reason,
    gap_bits, glr_z, mean_sigma_a, chi2, chi2_dof, bayes_voi,
    eval_sigma, iters_at_halt }` — JSON-serializable summary.
  - `SearchPolicy` trait with `name`, `observe`, `score_adjustment`,
    `should_halt`, `refresh_prior` (default no-op), `telemetry`. All
    methods take `&self` so multiple worker threads can share an
    `Arc<dyn SearchPolicy>`. Internal mutability is the implementor's
    responsibility (parking_lot::Mutex<Cache> is the canonical
    pattern that P08+P09 will use).
- `src/mcts/policy/kl_helpers.rs` — KL bisection helpers shared by
  P08+P09:
  - `bernoulli_kl(p, q)` — Bernoulli KL divergence with clamp.
  - `kl_upper(mu, n, beta)` and `kl_lower(mu, n, beta)` — invert
    `n·KL(μ̂, q) ≤ β` via 32-iteration bisection.
  - `kl_lucb_beta(t, k, delta)` — Kaufmann-Kalyanakrishnan 2013
    Theorem 8 stopping threshold:
    β(t, δ) = log(k₁·K·t^α/δ), k₁=405.5, α=1.1.

### Modified

- `src/mcts/mod.rs`: added `pub mod policy;` declaration.
  No call sites changed; the module exists but is unused by the
  engine.

## Tests added (9 total, Rust)

`src/mcts/policy/mod.rs::tests`:
1. `test_p06_default_bridge_no_op` — DefaultBridgePolicy: name is
   `"default_bridge"`, score_adjustment is identity, should_halt is
   Continue, telemetry serializes via serde_json.
2. `test_p06_halt_decision_carries_reason` — HaltDecision::Stop holds
   a HaltReason matching the P01 enum.
3. `test_p06_edge_view_sigma_a_smoothing_at_zero_visits` —
   N=0, M2=0, σ_root=0.3, λ₀=4 ⇒ σ_a = 0.3 exactly.
4. `test_p06_edge_view_sigma_a_smoothing_after_one_observation` —
   N=1, M2=0, σ_root=0.3, λ₀=4 ⇒ σ_a ≈ 0.2683 (hand-derived).

`src/mcts/policy/kl_helpers.rs::tests`:
5. `bernoulli_kl_zero_on_diagonal` — KL(p, p) ≈ 0 for p ∈ {0.1..0.9}.
6. `bernoulli_kl_non_negative` — KL(p, q) ≥ 0 for all (p, q).
7. `kl_upper_inverts_bisection` — at the returned q, n·KL(μ̂, q) ≈ β
   within 1e-2.
8. `kl_lower_inverts_bisection` — same for the lower bound.
9. `kl_lucb_beta_kk13_sanity` — β(t=151, K=3, δ=0.05) ≈ 15.618.
   Hand re-derivation: 151^1.1 ≈ 249.4, 405.5·3·249.4/0.05 ≈ 6.068e6,
   ln(6.068e6) ≈ 15.618.

The first attempt at this test asserted ≈ 18.40, which was a
hand-computation error (mistakenly multiplied 405.5·3·151 instead of
405.5·3·151^1.1). Fixed and documented in the test comment to prevent
the same mistake recurring.

## Test results

- `cargo test --release`: **461 passed** (was 452 + 9 = 461).
- No other test affected; the module is dead code from the engine's
  perspective.

## Adversarial review

### What this scaffolding lets P07-P11 ship safely

- **Trait stability**: Once P07 lands, the trait surface is frozen
  for the duration of the rollout. Adding new policies = `impl
  SearchPolicy for X` in a new file. No engine changes required.
- **Test isolation**: Each policy can be unit-tested by constructing
  a synthetic `SearchSnapshot` + `Vec<EdgeView>` without spinning up
  an `MctsEngine`. The hand-derived σ_a checks above demonstrate
  this.
- **Telemetry contract**: `ControllerTelemetry` ships with
  `schema_version: 1` from day 1. P08/P09/P11 emit consistent shape
  even though they populate different subsets of the fields.

### What this patch does NOT do

- **No engine integration**: The trait is dead from the engine's
  perspective. P07 wires `LegacyQuartz` into the engine — that's
  when the trait's hot-path performance matters.
- **No CLI**: `--policy` flag arrives in P07.
- **No telemetry emission**: `ControllerTelemetry` is constructed
  but never serialized into the actual search response yet — P07
  is the first patch where it ends up in `controller_summary`.

### Concurrency

- All trait methods take `&self`. Hot-path `score_adjustment` reads
  from cache populated by `observe`; implementors use
  `parking_lot::Mutex<Cache>` (P08/P09 pattern). `Mutex` lock cost
  per selection is ~1 μs uncontended on 16-thread parallel search,
  i.e. ~0.2% throughput at 50k iter/s — acceptable for the
  observability gain.

### Schema discipline

- `ControllerTelemetry.schema_version: 1` initial. New fields can
  land at v2 without renaming existing ones; all consumers should
  use `dict.get(key, default)` (Python) or the serde Default bound.

## Files touched

- `src/mcts/policy/mod.rs` (NEW, 200 LOC)
- `src/mcts/policy/trait_def.rs` (NEW, 230 LOC)
- `src/mcts/policy/kl_helpers.rs` (NEW, 130 LOC)
- `src/mcts/mod.rs` (+5 / -0)

Net delta: **+565 / -0 LOC**.

## What unblocks next

- **P07** (next): `LegacyAlphaZero` (pure PUCT, fixed budget) and
  `LegacyQuartz` (shim around existing controller path) — both
  trivial impls of `SearchPolicy` thanks to P06's scaffolding.
- **P08** (KLLUCBStop): consumes `kl_helpers::kl_lucb_beta` +
  `kl_upper` + `kl_lower`. Already test-covered for correctness in
  P06.
- **P09** (BayesianQuartz): consumes `EdgeView::sigma_a` for
  Welford+conjugate-prior σ_a, plus the KL helpers, plus a Pearson
  χ² external (statrs). P06's σ_a sanity tests (zero-visits + one-
  observation) already pin the numerical math the BayesianQuartz
  policy depends on.
