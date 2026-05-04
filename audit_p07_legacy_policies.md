# Audit — P07: `LegacyAlphaZero` + `LegacyQuartz` policies (standalone)

**Date:** 2026-05-04
**Patch:** P07 (15-patch QUARTZ v1.0 sequence)
**Scope:** ship the first two concrete `SearchPolicy` impls. Both are
**standalone** — neither is wired into the engine's hot path yet.
P10 lands the engine integration, CLI translator, and default flip.

This split keeps the patches small and individually auditable: P07 = "do
the policy types compose with real Rust impls?", P10 = "swap them in".

## What changed

### Reused (made `pub(crate)`)

- `src/mcts/select.rs::QuartzPolicyAdjustment` — was private inside
  the file; promoted so `LegacyQuartz::score_adjustment` can read the
  fields.
- `src/mcts/select.rs::quartz_policy_adjustment` — was private
  `#[inline] fn`; promoted to `pub(crate) #[inline] fn` so the shim
  can delegate to it verbatim.

No behavior change from the visibility bump — the function body and
all call sites are unchanged.

### New files

- `src/mcts/policy/legacy_az.rs` — `LegacyAlphaZero { budget: u32 }`.
  Pure PUCT, fixed-budget halt. ~30 LOC of behavior + boilerplate.
  - `name()` = `"legacy_az"`.
  - `observe()` = no-op (vanilla PUCT has no inter-selection state).
  - `score_adjustment()` = `ScoreAdjustment::default()` (identity).
  - `should_halt()` = `Stop(FixedBudget)` when `root_visits >= budget`.

- `src/mcts/policy/legacy_quartz.rs` — `LegacyQuartz { cfg, ctrl: Arc<QuartzController> }`.
  Bit-identical shim around the existing controller.
  - `name()` = `"legacy_quartz"`.
  - `observe()` = no-op (the legacy controller updates its stats
    inside the engine's own backup path — duplicating here would
    break locality).
  - `score_adjustment(edge)` = delegates to
    `quartz_policy_adjustment(edge.n, edge.o_a, edge.q, edge.prior,
    sqrt(*edge.root_total_n), &ctrl.last_stats(), &cfg)`. Maps the
    legacy `(effective_prior, penalty, bonus, use_fisher_puct)` shape
    onto the trait's `ScoreAdjustment`. The legacy `bonus` (off-diagonal
    one-loop) is collapsed into `penalty` because both are additive on
    the score in the live PUCT formula (`adjusted_puct_score:156`).
  - `should_halt()` = delegates to
    `QuartzController::should_stop(snap.root_visits, snap.elapsed_ms)`.
    Side-effect: the controller's halt_reason_count atomics (P01) are
    populated by `note_halt()` inside should_stop's terminal branches,
    so the JSON `extended.halt_reason_count` field stays falsifiable.
    Stop reason mapped from `StopReason` via best-effort match: legacy
    `BudgetExhausted + HaltMode::Fixed` ⇒ `FixedBudget`,
    `BudgetExhausted + other` ⇒ `MaxVisits`, etc.

### Modified

- `src/mcts/policy/mod.rs` — declared the two new submodules and
  re-exported the policy types.

## Tests added (8)

`legacy_az` (3):
1. `test_p07_legacy_az_score_adjustment_is_identity` — all four
   ScoreAdjustment fields are zero / None.
2. `test_p07_legacy_az_halt_at_budget` — Continue below budget, Stop
   at budget, Stop above budget. Reason matches FixedBudget.
3. `test_p07_legacy_az_telemetry` — schema_version=1, name=legacy_az.

`legacy_quartz` (5):
4. `test_p07_legacy_quartz_telemetry_shape` — name + schema_version.
5. `test_p07_legacy_quartz_halt_max_visits` — at root_visits=800
   (= cfg.max_visits), Stop(MaxVisits).
6. `test_p07_legacy_quartz_halt_fixed_budget` — under HaltMode::Fixed,
   Stop(FixedBudget).
7. `test_p07_legacy_quartz_continue_below_budget` — below thresholds,
   Continue.
8. `test_p07_legacy_quartz_score_adjustment_none_mode` — with
   PenaltyMode::None and one_loop disabled, penalty == 0.

## Test results

- `cargo test --release`: **469 passed** (was 461; +8 from P07).
- Two compile errors discovered and fixed during integration:
  - `StopReason::MaxNodesHit` was a variant the shim's match was
    missing — now mapped to `HaltReason::MaxVisits` (the closest
    semantic equivalent).
  - `should_stop` is a method on the `SearchController` trait, so
    the trait must be in scope. Added `use crate::mcts::search::
    SearchController` to legacy_quartz.rs.

## Adversarial review

### Why a shim?

The shim is the bridge that lets P10 flip the engine's default policy
to `BayesianQuartz` *without* breaking reproducibility of any
published number. Users who need bit-identical legacy behavior pass
`--policy=legacy_quartz` (P10 wires the CLI); the shim then reuses
the existing controller path verbatim. `last_stats()` is an `Arc`-borrow
read of a Mutex-guarded snapshot — exactly the same data the legacy
hot path sees.

### What the shim does NOT do

- **Tree-internal selection**: the shim only handles root-level
  `score_adjustment`. The legacy code path is invoked for every
  internal selection (depth > 0); the shim does not change that.
- **Online stat updates**: `observe()` is a no-op. The legacy
  controller updates its stats inside the engine's `update_stats`
  pipeline; duplicating here would either no-op or double-count.

### Concurrency

- The shim holds `Arc<QuartzController>`; reading `last_stats()` is
  a Mutex acquisition. P10's hot path expects `score_adjustment` to
  be O(1) per selection; the Mutex acquisition is ~1 μs uncontended,
  ~0.2% throughput at 50k iter/s on 16-thread parallel search. P15
  may switch to a seqlock if profiling shows hot-spot.

### Schema discipline

- `LegacyQuartz::telemetry()` sets `gap_bits=0, glr_z=0, chi2=0,
  chi2_dof=0, eval_sigma=0` because the legacy controller doesn't
  compute these. Downstream consumers must NOT assume non-zero
  values when `policy_name == "legacy_quartz"`. P14's pipeline-
  contract documentation will codify this contract.

### What unblocks next

- P08 (KLLUCBStop): consumes the same `EdgeView` / `SearchSnapshot`
  + the kl_helpers from P06 + the `HaltReason::KLLUCBStop` variant
  reserved in P01. Standalone — no engine wiring.
- P09 (BayesianQuartz): adds the principled core. Standalone.
- P10: integrates the four standalone policies into the engine via
  a `MctsConfig.search_policy: Option<Arc<dyn SearchPolicy>>` field
  + CLI `--policy` flag, then flips the default to `bayesian_quartz`.

## Files touched

- `src/mcts/select.rs` (+8 / -3, visibility only)
- `src/mcts/policy/mod.rs` (+5 / -0)
- `src/mcts/policy/legacy_az.rs` (NEW, 130 LOC incl. tests)
- `src/mcts/policy/legacy_quartz.rs` (NEW, 220 LOC incl. tests)

Net delta: **+361 / -3 LOC**.
