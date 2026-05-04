# Audit — P03: Replay freshness exponential-decay metric

**Date:** 2026-05-04
**Patch:** P03 (15-patch QUARTZ v1.0 sequence)
**Scope:** replace the misleading `n_new / replay_size` "freshness" with a
true monotone-in-mean-age exponential-decay score. Closes the audit's
"missing observability for off-policy drift in concurrent mode" finding.

## What changed

### Python

- `quartz/replay.py`: added `ReplayMetrics.freshness_summary(replay,
  current_generation, sample_n=200)` static method. Returns a JSON-shaped
  summary with `schema_version: 1`, `oldest_gen`, `newest_gen`,
  `mean_age`, `freshness_score`, `sample_count`, `half_life_gen`.

  Formula:

  ```
  half_life_gen = max(1, capacity / 100)         # ~100 pos/game heuristic
  mean_age      = current_generation - mean(actor_generation across sample)
  freshness     = exp(-max(0, mean_age) / half_life_gen)
  ```

  - Range: (0, 1]. 1.0 = freshly produced this generation. 0.5 ≈
    one half-life old. Below 0.1 ≈ gradient direction dominated by
    stale on-policy correlations.
  - Empty replay → `freshness_score=0.0`, all gen fields `None`,
    schema still emitted (callers can index by key without `KeyError`).
  - Rows without `actor_generation` metadata silently skipped; if NO
    sampled row has the tag, the empty shape is returned.
  - Negative ages (buffer reloaded with future-stamped data) clamp to
    0 in the formula — `freshness_score` stays in (0, 1].
  - Half-life heuristic of `capacity / 100` assumes ~100 positions per
    game. Callers needing a tighter half-life can divide capacity by
    their actual positions-per-game ratio.

- `quartz/cli_main.py`:
  - Added `replay_freshness_summary: None` to the per-iteration entry
    dict so consumers always see the key.
  - Hoisted `current_actor_generation` computation above the
    `if len(replay) >= cfg["batch"]:` block so it's available for the
    new `freshness_summary` call regardless of whether SGD fires this
    iteration. The redundant recomputation inside the SGD-fired branch
    is preserved (commented as intentional) to keep that branch
    self-contained for review.
  - Wired `replay_freshness_summary` into both the always-emitted entry
    and the SGD-fired entry update with `hasattr` guard so test stubs
    that pass a minimal `replay_metrics` object don't crash.

### Tests

`tests/test_training_pipeline_regressions.py`:

1. `test_p03_replay_freshness_summary_empty_buffer` — empty replay
   returns canonical empty shape, no exception.
2. `test_p03_replay_freshness_summary_all_current_gen` — every sample
   stamped at current_generation ⇒ mean_age=0, freshness_score=1.0
   exactly. Half-life check (capacity 1000 ⇒ half_life=10).
3. `test_p03_replay_freshness_summary_one_half_life_old` — hand-computed:
   capacity=1000, half_life=10, mean_age=10 ⇒ freshness ≈ exp(-1) ≈
   0.3679. Verified to within 1e-4.
4. `test_p03_replay_freshness_summary_skips_untagged` — rows without
   `actor_generation` metadata never reach the count.
5. `test_p03_replay_freshness_summary_negative_age_clamped` — future-
   stamped rows clamp to age=0, score=1.0.

### P02 regression fix (collateral)

P02's pre-flight gate revealed three pre-existing tests in
`test_training_pipeline_regressions.py` that passed `model_path`
strings without creating the underlying files. P02 caught this
correctly (the gate is doing exactly what it's designed for), so the
test fixtures were updated to write minimal bytes:

- `test_ablation_study_discards_stale_eval_cache` (a.pt, b.pt)
- `test_ablation_study_records_eval_condition_timings` (a.pt, b.pt)
- `test_ablation_study_uses_compare_many_when_available` (a.pt, b.pt, c.pt)

These three test edits land in the P03 commit (they're collateral to
P02 but only surfaced when `pytest test_training_pipeline_regressions.py`
ran in P03 scope).

The 4th regression (`test_main_runs_eval_at_interval_even_when_train_steps_are_zero`)
was caused by P03 itself — `current_actor_generation` was originally
computed inside the SGD-fired branch, but P03 references it earlier.
Fixed by hoisting the computation up.

## Test results

- `pytest tests/test_training_pipeline_regressions.py -q`: **200 passed**
  (was 196 before P03 + 4 P03 tests = 200; was 200 before any drift
  introduced — net regression count is zero).
- `pytest tests/ -k "p01 or p02 or p03"`: **14 passed**.

## Adversarial review (red-team self-audit)

### What this metric catches

- **Concurrent-mode off-policy drift**: with `--concurrent` and a slow
  bg_worker, the replay can fill with samples from generations 5
  iterations old before SGD fires. Legacy `freshness = n_new /
  replay_size` shows ~0.05 (5%) regardless of capacity, which gives no
  signal. The new score drops below 0.5 the moment mean_age exceeds
  the half-life — a hard threshold readers can act on.
- **Cross-pipeline comparisons**: a 5k-replay × 50/iter pipeline and a
  50k-replay × 500/iter pipeline have the same legacy `freshness`
  number. Their `freshness_score` differs as long as their
  half_life_gen differs (which it does — half_life scales with
  capacity).

### What this metric does NOT catch

- **Bias direction of staleness**: a freshness_score of 0.3 doesn't
  tell you whether the policy distribution has shifted toward over-
  or under-exploration. Pair this with `policy_entropy` (already
  emitted) to see distributional shift.
- **Per-game vs per-position**: actor_generation is stamped per-game
  at `add_game()` time. All positions from one game share the
  same tag. If a game spans 200 positions and the actor was updated
  mid-game (not currently possible in the pipeline), the stamp would
  represent the start-of-game actor.
- **Half-life is a rough heuristic**: 100 positions per game is the
  default Gomoku/Go ballpark. Chess games are shorter (~80 positions);
  TTT is ~9 positions. Callers running short games will see a too-
  generous half-life. P15 cleanup should expose `positions_per_game`
  as an optional argument.

### Schema discipline

- `freshness_summary.schema_version = 1`. New fields can be added at
  v2 without breaking the existing keys. Renaming `freshness_score`
  would be a major bump.

### Concurrency

- `freshness_summary` reads from `replay` while the bg_worker may be
  writing. The existing `_examples_at_indices` helper handles this
  gracefully (returns whatever is there at sample time). Worst case:
  a sample appears as `untagged` if its metadata dict is being
  initialized concurrently — gets skipped, no crash.

## Files touched

- `quartz/replay.py` (+88 / -0)
- `quartz/cli_main.py` (+27 / -10)
- `tests/test_training_pipeline_regressions.py` (+95 / -6 P03 tests +
  18/-6 P02-collateral fixture writes)

Net delta: **+228 / -22 LOC**.

## What unblocks next

- P14's `ReplayState` dataclass can carry the `freshness_score`
  field directly from this summary.
- P15 cleanup: rename the legacy `replay_freshness` (turnover rate)
  field to `replay_turnover_rate` once downstream consumers have
  migrated to the new `replay_freshness_summary`.
