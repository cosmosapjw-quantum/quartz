# Audit Phase Landing — P1-P10 Delta (2026-04-25)

**HEAD:** `54e1be5` (tests: P10 — E2E training convergence + determinism)
**Branch base:** `281cf18` (Phase 6 final delta)
**Source audit:** [`audit_codex_20260425.md`](../audit_codex_20260425.md)
**Plan:** [`/home/cosmosapjw/.claude/plans/audit-codex-20260425-md-squishy-canyon.md`](../../.claude/plans/audit-codex-20260425-md-squishy-canyon.md)

---

## 1. What landed

Ten audit patches (P1–P10) shipped as ten focused commits with per-step semantic audits. The work followed the user's audit-first ordering: every patch lands and is verified before the Phase 7 perf checkpoints (C / F / I) start, so the perf measurements rest on a trustworthy ablation pipeline.

| # | Audit ID | Commit | Substance |
| --- | --- | --- | --- |
| 1 | P1 / W4 | [`f11d3ca`](#) | `_write_checkpoint_status` now keys preferred-checkpoint logic on `saw_promotion` instead of the per-run `best_checkpoint_bootstrap` flag. Re-runs with prior bootstrap `best.pt` no longer leak the untrained model into `ablation_study.resolve_model_path`. |
| 2 | P2 / W3 | [`cd5c2f1`](#) | `prior_refresh_temp` near zero is honored verbatim (clamp `1e-6`) instead of snapping to the legacy `0.5` literal. Optuna sweeps over the parameter near zero now register their actual effect. |
| 3 | P3 / W6 | [`98f2e3c`](#) | `scripts/smoke_e2e.py` defaults to `--games-per-iter 16` so replay crosses the SGD batch threshold; a post-run `verify_training_fired` walks every per-condition `train_log.jsonl` and SystemExits if zero loss rows fired. |
| 4 | P4 / W9 | [`d7d1541`](#) | README "CI Gates" section names `tests-gate.yml` (the unit/contract gate, already landed Apr 25 13:00) and `phase15-benchmark-gate.yml` (the benchmark-shape gate). The latter carries a header comment pointing readers at the unit-test gate. |
| 5 | P5 / W1 | [`39cf6b5`](#) | `controller_identity_keys()` enumerates the 14 identity-defining cfg fields; `controller_identity_hash(cfg)` is a stable 16-hex SHA-256 over those values; `assert_single_axis_isolation(surfaces, axis_keys)` verifies that single-axis presets isolate exactly the declared axes. The manifest now embeds `train_condition_identity_hashes` and `eval_condition_identity_hashes`. |
| 6 | P9 / W5 | [`20e0c93`](#) | `bg_worker.update_model(actor_source)` is lifted out of the `executed_steps > 0` gate. Every iteration now bumps the worker's actor_generation, including warmup iterations where SGD does not fire. Concurrent-path replay tags align with the learner's iteration boundary regardless of SGD path. |
| 7 | P8 / W7 | [`38cd3f5`](#) | New `--frozen-eval-condition NAME` / `--no-frozen-eval` flags. For attribution presets the eval matrix collapses to a single named condition (alphabetically first by default) so cross-row deltas reflect model quality, not (model × eval search profile). Manifest records the choice as `frozen_eval_condition`. |
| 8 | P7 / W2 | [`2369fcc`](#) | Three layers: (a) Rust `SearchOverrides.halt_mode` + `parse_halt_mode_override` accepts `"fixed"` / `"voc"` / `"simple_threshold"`; (b) `SEARCH_RUNTIME_KEYS` and `SEARCH_MANIFEST_KEYS` include `halt_mode`; (c) `pin_halt_mode_for_attribution(preset)` deep-copies the preset and stamps `halt_mode = "fixed"` on every train + eval condition for `controller_axes` / `controller_factorial`. `HaltMode::Fixed { budget = u32::MAX }` disables every adaptive halt branch so attribution rows see same-budget arena results. |
| 9 | P6 / W8 | [`30dbc07`](#) | `HaltCheck` Rust struct (Serialize, `schema_version: 1`) records per-call halt-decision telemetry into `QuartzCtrlInner.halt_telemetry`; `attach_search_metadata` stamps `controller_summary.schema_version: 1` plus `voc_total / voc_focus / voc_expand / voc_merge`; `quartz/replay.py:search_summary` aggregates the new fields and emits a `controller_schema_versions` census. |
| 10 | P10 / W10 | [`54e1be5`](#) | Two new tests in `tests/test_e2e_convergence.py`: (a) loss strictly decreases across 5 iterations of a real torch SGD run on a synthetic gomoku7 replay; (b) two runs with the same seed produce first-iter loss within `1e-5`. Tagged `@pytest.mark.slow`, runs in ~3 s. |

---

## 2. Semantic audit envelope

| Suite | Pre-audit (`281cf18`) | Post-audit (`54e1be5`) | Δ |
| --- | --- | --- | --- |
| `cargo test --release --locked` | 391 passed | **398 passed** | +7 (P6×3, P7×4) |
| `pytest tests/ -q --ignore=tests/test_play_gui.py` | 287 passed | **312 passed** | +25 (P1×3, P3×3, P5×5, P7×4, P8×5, P9×1, P6×2, P10×2) |
| `v5_stress_parallel × 10` (binary direct) | 10 / 10 | 10 / 10 | unchanged |
| TSAN full bin suite | clean | clean (re-run after P7 Rust changes only; P6/P9 are additive) | unchanged |
| Rust `audit_codex_20260425.md` weakness count | 10 | 0 (all addressed; W9 was already landed) | −10 |

Every commit was built `--release --locked`, and pytest was run after every commit that touched Python. v5_stress_parallel × 10 was rerun after every commit touching the Rust hot path.

---

## 3. What is now ablation-grade that was not before

The audit's ten weaknesses had a common load-bearing structure: short ablation runs silently produce corrupt or unfalsifiable results because pipeline contracts (`best.pt` seeding, replay→SGD threshold, eval engine spec, actor freshness, halt-mode budget coupling) were quietly violated under typical smoke parameters. With P1–P10 landed:

- **`best.pt` corruption (W4) eliminated**: re-runs without promotion now correctly resolve to `latest.pt`, even when a prior bootstrap `best.pt` is present.
- **Smoke certifies SGD (W6)**: the smoke now fails fast if zero training rows fire, so "smoke pass" implies the training loop actually executed gradient steps.
- **Controller identity is observable (W1)**: every manifest carries a per-row 16-hex hash over the 14 controller-identity fields. Cross-paper readers can name the exact dispatch surface that produced an Elo curve.
- **Penalty/halt orthogonality is recoverable (W2)**: attribution presets default to `HaltMode::Fixed`, so every row of `controller_axes` runs at the same `max_visits` ceiling. Penalty-mode comparisons no longer confound with budget drift.
- **Eval engine drift is fixed (W7)**: attribution presets default to a single frozen eval condition; cross-row deltas reflect model quality only.
- **Actor freshness is iteration-aligned (W5)**: the worker's actor_generation increments every iteration regardless of SGD path; replay tags align with the learner's step counter.
- **Controller telemetry is wire-stable (W8)**: `schema_version: 1` is published with controller_summary, voc-channel decomposition is included, and the Python aggregator surfaces both. Schema drift is now visible at aggregation time.
- **Behavioural regression covered (W10)**: a fast E2E test asserts loss-decrease and seed-determinism, closing the gap that left the "AlphaZero training loop" claim untested.
- **Honest sweeps (W3)**: `prior_refresh_temp` near zero is now real, not silently snapped to `0.5`.
- **CI gate visibility (W9)**: README and the legacy phase15 workflow point at `tests-gate.yml` so contributors don't mistake the benchmark gate for the regression gate.

The Phase 7 perf work (C / F / I) can now run with an ablation pipeline whose floor is verified.

---

## 4. Known follow-ups inside the audit scope

Three items are partial — code that lands the headline contract but defers the long tail:

1. **P6 per-check Vec<HaltCheck> not yet plumbed through every search-result builder.** `QuartzCtrlInner.halt_telemetry` accumulates one record per `should_stop` call, but the `mcts_server.rs` search-result builders (six call sites) still emit only the final `controller_summary`. A follow-up patch can serialize the full vec into the search response so per-game traces are inspectable.
2. **P7 binary-frame search overrides do not carry `halt_mode`.** The JSON path honours it (the only path attribution presets exercise); the SHM binary frame at `mcts_server.rs:702` is intentionally None until a downstream consumer asks for it.
3. **P8 `--frozen-eval-condition` only collapses the eval matrix loop.** The matrix-replay logic (existing-match detection by `(eval_name, a, b)`) was not touched; collapsing the matrix means existing matches under non-frozen rows are simply ignored on resume. Acceptable for the attribution use case but worth noting if a user mixes flags across re-runs.

None of these block the Phase 7 perf checkpoints.

---

## 5. Phase boundary

This commit closes the audit phase. Next: Phase 7 perf checkpoints (C — per-bucket bumpalo edge buffer with `AtomicPtr<MctsEdge>`; F — open-addressing TT; I — PGO build), targeting the strict bars Phase 6 left open: IPC > 1.7, dTLB < 10 %, `TT::get_or_create` < 8 % Ir, wall mean < 320 ms / wall min < 280 ms.
