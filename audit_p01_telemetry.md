# Audit — P01: Telemetry counters + JSON schema_version 6

**Date:** 2026-05-04
**Patch:** P01 (15-patch QUARTZ v1.0 sequence; see /home/cosmosapjw/.claude/plans/iridescent-giggling-bachman.md)
**Scope:** make `controller_penalty_mode_counts`, `mean_prior_refresh_rate`, and
`halt_reason_count` telemetry actually emitted from Rust, not just claimed in README.
Closes audit weakness W3 (telemetry phantom claims) and provides the machinery
F2's "missing argmax counters" fix needs.

## What changed

### Rust

- `src/mcts/quartz.rs`
  - Added `enum HaltReason` (10 variants: PFlipConverged, VOCNonPositive,
    FixedBudget, KLLUCBStop, MaxVisits, MaxTime, MinVisitsNotMet,
    GLRCertified, PolicyConverged, EmpBernsteinSep). Reserved variants
    (KLLUCBStop, GLRCertified, PolicyConverged, EmpBernsteinSep) for the
    P08 / P09 / P11 policies so the array layout and JSON keys are stable
    across the patch series.
  - Added `pub const PENALTY_MODE_KEYS: [&str; 7]` and helper
    `penalty_mode_idx(PenaltyMode) -> u8`. Stable ordering — never rename
    without bumping `controller_summary.extended.schema_version`.
  - Added `halt_reason_count: [AtomicU32; HALT_REASON_COUNT]` to
    `QuartzController`, kept OUTSIDE the `Mutex<QuartzCtrlInner>` so
    `should_stop` can increment lock-free.
  - Added `note_halt(HaltReason)` (private) + `halt_reason_count_snapshot()`
    (public) accessors.
  - Wired `should_stop` to call `note_halt` on every terminal `return true`
    path, after dropping the inner mutex (avoids the `mem::drop(g);` ordering
    bug). Added a `MinVisitsNotMet` increment at the periodic-check
    pre-condition early return so "stop attempted but visits insufficient"
    is distinct from "not yet checked".

- `src/mcts/select.rs`
  - Extended `RootSelectionTrace` and `RootScoreDetail` with
    `penalty_mode_idx: u8` (sentinel `u8::MAX` for non-Quartz traces) and
    `refresh_eligible: bool`. Manually impl `Default` so the sentinel
    `u8::MAX` is preserved (`derive(Default)` would have set it to 0
    which equals `PenaltyMode::Legacy`, polluting the histogram).
  - Added `refresh_supported(qcfg)` helper that returns true iff the
    active `PenaltyMode` *can* refresh (regardless of runtime gate).
    Eligibility = `n_raw > 0 && refresh_supported`. Activation =
    `effective_prior_l1 > 1e-6` (existing definition, unchanged).
  - Extended `SelectionTelemetry` with three new atomics:
    `penalty_mode_invoke_count: [AtomicU64; PENALTY_MODE_COUNT]`,
    `refresh_eligible_count: AtomicU64`, `refresh_active_count: AtomicU64`.
    Wired into `record_root`, `reset`, `snapshot`. Ordering invariant:
    `eligible++` happens after the gate decision so any snapshot satisfies
    `eligible >= active`.
  - Extended `SelectionTelemetrySnapshot` with the three new fields.

- `src/mcts_server.rs`
  - Extended `SearchExecutionOutcome` with
    `selection_penalty_mode_invoke_count: [u64; 7]`,
    `selection_refresh_eligible_count: u64`,
    `selection_refresh_active_count: u64`,
    `halt_reason_count: [u32; HALT_REASON_COUNT]`. Populated at all 3
    producer sites (Quartz, Baseline/BaselineStrict, async synth).
    Baseline/BaselineStrict and synth zero `halt_reason_count` since no
    QuartzController is present.
  - `build_result_value` emits the new fields as flat top-level JSON keys
    (`selection_penalty_mode_invoke_count`, `selection_refresh_eligible_count`,
    `selection_refresh_active_count`, `halt_reason_count`).
  - `attach_search_metadata`:
    - Promotes the flat keys into a new `controller_summary.extended`
      object with stable string keys (`PENALTY_MODE_KEYS` and
      `HaltReason::as_key()`).
    - Computes `mean_prior_refresh_rate = active / eligible`, returning
      `null` when `eligible == 0` so the JSON never carries NaN/Infinity.
    - Bumps `controller_summary.schema_version` 5 → 6.
  - Added `#![recursion_limit = "256"]` to `src/main.rs` because the
    `serde_json::json!{...}` block in `build_result_value` exceeded the
    default 128-token macro recursion when the new fields were appended.
    256 is the standard `serde_json` escape hatch.

### Python

- `quartz/replay.py:ReplayMetrics.search_summary` extended with
  aggregation of the new `controller_summary.extended` block:
  - Sums `refresh_active_count` and `refresh_eligible_count` across rows
    so the study-level `extended_measured_prior_refresh_rate` is a
    *pooled* estimator (not a row-mean, which would over-weight games
    with very few eligible selects).
  - Sums per-`PenaltyMode` and per-`HaltReason` counts across rows.
  - Emits 6 new top-level fields on the summary dict: `extended_*`.
  - Pre-P01 rows (schema_version ≤ 5) are silently skipped via the
    falsy-guard; `extended_coverage_frac` exposes how many rows carried
    the new block.

  **The legacy `mean_prior_refresh_rate` field was NOT renamed** despite
  being a misleading "average of configured rate" rather than a measured
  rate. Renaming would break downstream consumers. The new
  `extended_measured_prior_refresh_rate` is the principled metric;
  documentation should redirect users to it.

## Tests added (5 total)

### Rust (`src/mcts_server.rs`, in `tests` mod)

1. `test_controller_summary_extended_block_has_all_keys` — synthesizes
   counters with `Legacy=7, PFlipMixture=3, MaxVisits=1, PFlipConverged=4,
   eligible=50, active=17`, runs `attach_search_metadata`, asserts the
   `extended` JSON has all key-value pairs correct including reserved
   zero-count keys (KLLUCBStop, GLRCertified) so downstream consumers
   indexing by string key never miss a slot.
2. `test_controller_summary_extended_handles_zero_eligible` — verifies
   `mean_prior_refresh_rate` is JSON `null` (not NaN/Infinity) when
   `eligible == 0`.
3. `test_controller_summary_marks_inert_prior_refresh_rate` — updated
   from schema_version 5 → 6 + asserts `extended` block exists.

### Python (`tests/test_training_pipeline_regressions.py`)

4. `test_p01_replay_summary_aggregates_extended_block` — two rows with
   non-trivial counts, asserts pooled rate `25/90`, summed mode counts,
   summed reason counts.
5. `test_p01_replay_summary_handles_missing_extended_block` —
   pre-schema-6 row, asserts `extended_coverage_frac=0.0`,
   `extended_measured_prior_refresh_rate is None`, no crash.

## Test results

- `cargo test --release`: **446 passed, 0 failed, 89 ignored** (was 444; +2
  from this patch).
- `pytest -q tests/ --ignore=tests/test_real_loop_e2e.py`: in progress at
  audit-write time; the targeted P01 tests pass standalone (`pytest -k
  "p01 or p6_replay"` → 4 passed).

## Adversarial review (red-team self-audit)

### Race conditions

- **Eligible/active ordering**: `eligible.fetch_add(1, Relaxed)` happens
  *after* the gate decision, before any `active.fetch_add(1, Relaxed)`.
  Workers under heavy contention can produce snapshots where eligible
  and active update on different threads; the invariant
  `eligible >= active` holds at every moment because the gate→eligible→
  optional-active sequence is all on the same worker.
- **Lost increments under panic**: if a worker panics between
  `eligible++` and `active++`, the active count is biased low and the
  ratio under-reports refresh activity. Acceptable for telemetry-only
  data; not load-bearing for halt decisions.
- **`Relaxed` ordering across workers**: each counter is monotonic
  (always `+= 1`); final values are stable once all workers join the
  search. No cross-counter consistency requirement, so `Relaxed` is
  sufficient. The `should_stop` halt-reason increment happens after
  `mem::drop(g)` of the inner Mutex, ensuring the lock is released
  before the lock-free counter update — no deadlock potential.

### Schema drift

- New `extended.schema_version: 1`. Bumping it requires renaming or
  removing a key. All consumers (replay.py aggregator, the new tests)
  use `dict.get(key)` with a default, so adding new keys at v2 is
  forward-compatible. Removing keys requires a major version bump.
- Older Rust binaries (schema_version ≤ 5) emit no `extended` block; the
  Python aggregator's `extended or {}` guard treats this as `{}` and
  reports `extended_coverage_frac=0.0` so the contamination is visible.

### Hash collisions / Sentinel values

- `penalty_mode_idx: u8::MAX` is the sentinel for non-Quartz traces.
  `record_root` skips the histogram increment for sentinel values via
  the bounds check `if (penalty_mode_idx as usize) < PENALTY_MODE_COUNT`.
  Default `derive(Default)` would have used `0`, polluting `Legacy` —
  manual `impl Default` was added for both `RootSelectionTrace` and
  `RootScoreDetail` to enforce the sentinel.

### What this patch does NOT fix

- Tree-internal penalty/refresh telemetry (only root selects sampled).
  Per the design doc: tree-internal sampling would 100× the increment
  count and is less informative anyway; root-level is sufficient for
  the README claim.
- Argmax-channel histogram drift: existing `voc_argmax_channel_hist` is
  unchanged. P01 just adds a parallel set of counters.
- The legacy `mean_prior_refresh_rate` field name (still misleading;
  P15 cleanup may rename or deprecate).

## What unblocks next

P01's `HaltReason` enum has reserved slots (KLLUCBStop, GLRCertified,
PolicyConverged, EmpBernsteinSep) that P08 / P09 / P11 increment
without further enum changes. The `extended` block schema is also stable
for those patches — they only need to call `self.note_halt(reason)`.

## Files touched

- `src/mcts/quartz.rs` (+90 / -0)
- `src/mcts/select.rs` (+95 / -10)
- `src/mcts_server.rs` (+150 / -3)
- `src/main.rs` (+7 / -0)
- `quartz/replay.py` (+45 / -0)
- `tests/test_training_pipeline_regressions.py` (+95 / -0)

Net delta: **+482 / -13 LOC**.

## Remaining open items deferred to later patches

- P02: pre-flight checkpoint hash gate (next).
- P15 (cleanup): rename or deprecate the legacy
  `mean_prior_refresh_rate` field on `ReplayMetrics.search_summary`.
- The README's "Recent Updates" section needs a paragraph noting the
  schema_version 6 emit and the new `extended` block — handled in P15
  doc-cleanup or sooner if a user-facing release lands.
