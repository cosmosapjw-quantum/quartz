# Audit — P05: σ_0 calibration auto-load + EvalStrength bins

**Date:** 2026-05-04
**Patch:** P05 (15-patch QUARTZ v1.0 sequence)
**Scope:** make `sigma_0_recommendation.json` (already produced by
`src/calibration.rs::write_sigma_recommendation`) actually load at
runtime. Closes audit weaknesses W10 (calibration silent skip) and F6.

## What changed

### Rust — src/mcts/quartz.rs

- Added `EvalStrength` enum (Weak, Medium, Strong) with `tag()` and
  `from_value_loss(vl: f32)` helpers. Stratifies σ₀ recommendations by
  evaluator quality regime per QUARTZ_THEORY.md §3 thresholds (vl > 0.7
  ⇒ weak, 0.3-0.7 medium, < 0.3 strong).
- Added `CalibrationDiagnostic` enum (Info, Warn) — typed diagnostic
  output that the caller can render as eprintln WARN or pass-through.
- Added `QuartzConfig::with_calibration(self, calibration_dir,
  game_label, eval_strength, warn_factor) -> (Self, Vec<CalibrationDiagnostic>)`.
  Search order:
  1. `<dir>/<strength>.json` if `eval_strength` is set.
  2. `<dir>/sigma_0_recommendation.json` (default).
  
  First file that parses cleanly wins. If `schema_version != 1`,
  emit a WARN and continue to the next candidate (forward-compat).
  Per-game key takes priority over `__cross_game__` fallback.
  Invalid sigma_0 (non-finite, ≤ 0) silently skipped with WARN.
  Override differing by more than `warn_factor` (default 2.0) from
  base ⇒ additional WARN diagnostic.

### Rust — src/mcts_server.rs

- `apply_search_profile` now reads three env vars:
  - `QUARTZ_CALIBRATION_DIR` (path to calibration JSON dir; if unset,
    no auto-load)
  - `QUARTZ_CALIBRATION_GAME` (game label for per-game lookup)
  - `QUARTZ_CALIBRATION_STRENGTH` (one of "weak"/"medium"/"strong";
    defaults to Strong if unset)
- Diagnostics from `with_calibration` are eprintln'd with `[quartz]
  [calibration] INFO/WARN:` prefix so they surface in the train_log
  without artifact inspection.
- Only fires when the profile retains a Quartz config (Baseline*
  profiles already disabled it earlier in the function).

### Python — scripts/ablation_study.py

- Added `--calibration-dir` CLI arg.
- When set, the train subprocess gets `QUARTZ_CALIBRATION_DIR` and
  `QUARTZ_CALIBRATION_GAME` env vars in addition to the inherited
  process environment.

### Tests added (6, Rust)

1. `test_p05_with_calibration_applies_per_game_recommendation` — JSON
   file with `gomoku7: 0.18, __cross_game__: 0.22`, ask for `gomoku7`
   ⇒ sigma_0 == 0.18, exactly one INFO diagnostic.
2. `test_p05_with_calibration_falls_back_to_cross_game` — per-game key
   missing ⇒ falls back to `__cross_game__` value.
3. `test_p05_with_calibration_warns_on_large_divergence` — base 0.3,
   override 1.5 (5× ratio) ⇒ ≥1 WARN diagnostic.
4. `test_p05_with_calibration_missing_file_keeps_default` — empty dir
   ⇒ sigma_0 unchanged, single INFO emitted.
5. `test_p05_with_calibration_strength_stratified_file_wins` — both
   `weak.json` and `sigma_0_recommendation.json` present, with `Weak`
   passed ⇒ takes the weak.json value.
6. `test_p05_with_calibration_skips_unknown_schema_version` — file
   with `schema_version=99` ⇒ skipped, sigma_0 unchanged, WARN
   diagnostic mentions `schema_version`.

### Cargo.toml

- Added `[dev-dependencies] tempfile = "3"` for the calibration loader
  tests (creates ephemeral JSON fixtures in tmpdir).

## Test results

- `cargo test --release p05`: 6 passed.
- `cargo test --release` full: **452 passed** (was 446; +6 from P05).
- `pytest tests/test_ablation_study.py -q`: 66 passed (no regressions).

## Adversarial review

### What this patch enables

- **Closed the calibration loop**: previously the
  `evaluator_calibration.py` script wrote `sigma_0_recommendation.json`
  to a directory and then nothing read it. Users had to hand-edit
  config.json. Now: `--calibration-dir results/calibration` ⇒ the Rust
  server picks up the recommendation per-process.
- **Per-game stratification**: `recommendations` JSON includes
  per-game keys; the Rust loader picks the game-specific override
  (or falls back to `__cross_game__`). Useful when calibration was
  done across multiple games but the runtime is single-game.
- **Eval-strength stratification**: foundation for P09's BayesianQuartz
  policy (which benefits from σ₀ tuned to the evaluator's training
  state). Currently optional; user must opt in via env var.

### What this patch does NOT do

- **Calibration regeneration**: this patch loads existing files; it
  does NOT regenerate them when they're stale. Users must re-run
  `evaluator_calibration.py` manually.
- **EvalStrength auto-detection**: the env var is `QUARTZ_CALIBRATION_STRENGTH`
  (string). Future P09 work will plumb the runtime value-loss into this
  automatically.
- **Multi-game runtime**: env vars are per-server-process; the same
  process can't serve two games with different σ₀. Multi-game runs
  must use separate server invocations (which is the existing model).

### Concurrency

- `apply_search_profile` is called per-search-request; each call reads
  env vars, opens JSON, applies override. Negligible overhead (~µs)
  for the JSON parse on a 1KB file.
- No mutation of shared state — each request gets a fresh QuartzConfig.

### Schema discipline

- Reuses the existing schema_version=1 emitted by `write_sigma_recommendation`.
- Forward-compat: schema_version != 1 silently skipped with WARN, so
  a v2 file written by a newer calibration tool won't crash older
  servers.

## Files touched

- `src/mcts/quartz.rs` (+205 / -0)
- `src/mcts_server.rs` (+38 / -0)
- `scripts/ablation_study.py` (+22 / -1)
- `Cargo.toml` (+4 / -0)

Net delta: **+269 / -1 LOC**.

## What unblocks next

- P09 (BayesianQuartz): with σ₀ now load-able at runtime, the
  per-evaluator-strength σ₀ becomes a meaningful axis instead of a
  dead recommendation file.
- P14 (pipeline contracts): the calibration env vars become part of
  the pipeline contract metadata.
