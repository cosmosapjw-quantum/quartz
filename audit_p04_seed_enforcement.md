# Audit — P04: Multi-seed enforcement + concurrent-stall warning

**Date:** 2026-05-04
**Patch:** P04 (15-patch QUARTZ v1.0 sequence)
**Scope:** convert `--research-grade` from a passive readiness flag into
a hard pre-flight gate enforcing the multi-seed and paired-seed
protocols. Add a runtime warning for concurrent-mode SGD starvation.
Closes audit weakness W8 (passive seed protocol) and F4 (silent zero-SGD).

## What changed

### Python — scripts/ablation_study.py

- Added CLI arg `--min-seeds-for-research-grade` (default 3, matching
  RESEARCH_READINESS.md).
- `enforce_research_grade(args, report)` now performs three gates in
  order:

  1. **Soft warning** (always fires, regardless of `--research-grade`):
     if `len(seeds) < 3`, print to stderr that the run does not meet
     the research-readiness convention. This is a teaching tool, not
     a block.
  2. **Hard gate 1** (only under `--research-grade`): seed count
     `< min_seeds_for_research_grade` ⇒ SystemExit BEFORE training
     starts. Avoids spending hours of compute on a campaign that will
     be rejected post-hoc.
  3. **Hard gate 2** (only under `--research-grade --paired-seed-eval`):
     conditions in `report["runs"]` must share an identical seed set.
     Mismatch ⇒ SystemExit with the per-condition seed map. Pre-train,
     `report` is `None` and this gate is skipped.
  4. **Hard gate 3** (existing): `report["research_readiness"]
     ["research_grade_ready"]` must be True; if False, SystemExit with
     the unmet criteria list.

### Python — quartz/cli_main.py

- Added `zero_sgd_streak = 0` counter outside the iteration loop.
- Inside the `if len(replay) >= cfg["batch"]:` block, when
  `train_steps <= 0 and args.concurrent`, increment the streak. After
  2 consecutive zero-SGD iterations, print a stderr WARN with the
  replay-size / batch-size / min_new ratio so users immediately see
  the backpressure starvation that P02 / W8 warned about.
- `train_steps > 0` resets the streak. Bootstrap iterations
  (`len(replay) < cfg["batch"]`) do NOT count toward the streak — that
  regime is expected to skip SGD.

### Tests

`tests/test_ablation_study.py`:

1. `test_p04_research_grade_blocks_single_seed` — `--research-grade`
   with `seeds=42` ⇒ SystemExit("at least 3 seeds").
2. `test_p04_research_grade_passes_with_three_seeds_and_ready_report` —
   3 seeds + ready readiness ⇒ no raise.
3. `test_p04_research_grade_paired_seed_mismatch_blocks` —
   `--paired-seed-eval` with conditions that disagree on seeds ⇒
   SystemExit("conditions disagree on seeds").
4. `test_p04_soft_warning_on_single_seed_without_research_grade` —
   verifies the soft warning prints to stderr without raising.

The existing `test_research_grade_gate_fails_on_incomplete_report`
test was updated to provide 3 seeds so the gate it was testing
(readiness) still fires; otherwise P04's seed-count gate would
short-circuit it.

## Test results

- `pytest tests/test_ablation_study.py tests/test_training_pipeline_regressions.py -q`:
  266 passed (was 262 before P04 + 4 new = 266).

## Adversarial review

### What this patch catches

- **Single-seed `--research-grade` runs**: previously completed and
  emitted a "ready" readiness report in some pipelines (the readiness
  computation could mark single-seed as compliant if other criteria
  passed). Now blocked at CLI parse time.
- **Paired-seed runs with mismatched seed sets**: e.g. condition A
  trained with seeds [11, 22, 33] and condition B trained with seeds
  [11, 22, 44]. Previously the eval matrix would silently pair
  (A_11, B_11) and (A_22, B_22) and skip the mismatch — leaving the
  eval coverage incomplete and the leaderboard misleading. Now blocked.
- **Concurrent zero-SGD starvation**: the smoke_e2e workaround
  (forcing `--no-pipeline`) was the user-facing visible answer. Without
  the workaround, a full training run in `--concurrent` mode could
  silently fail to make any SGD progress for hours. The warning
  surfaces this on the second offending iteration.

### What it does NOT catch

- **Different seeds per condition that are all sufficient in count**:
  e.g. A=[11,22,33], B=[44,55,66]. Both meet `min_seeds=3`, so hard
  gate 1 passes. Under `--paired-seed-eval`, hard gate 2 fires (seed
  sets differ). Without `--paired-seed-eval`, this is a legitimate
  configuration (unpaired eval) and is correctly allowed.
- **One-shot debug runs that would benefit from a single seed**: yes,
  blocked under `--research-grade`. The intended workflow is to drop
  `--research-grade` while iterating, then add it for the claim run.
- **Hardware-unique stalls**: the concurrent-stall warning fires only
  on `args.concurrent`. A non-concurrent stall (e.g. selfplay queue
  starvation in inline mode) is not captured here — that's P14's
  pipeline-contract scope.

### Schema discipline

- No new schema fields. CLI argparse is the only public surface
  change. The new gates raise `SystemExit` with prefix-stable error
  messages (`"at least N seeds"`, `"conditions disagree on seeds"`,
  `"research-grade gate failed"`) so downstream consumers parsing
  stderr can match on a stable substring.

## Files touched

- `scripts/ablation_study.py` (+62 / -3)
- `quartz/cli_main.py` (+30 / -0)
- `tests/test_ablation_study.py` (+78 / -1)

Net delta: **+170 / -4 LOC**.

## What unblocks next

P04's enforcement primitives (`zero_sgd_streak`, the per-condition
seed-set check) unblock P10's flip-default-to-BayesianQuartz: when
the new default lands, multi-seed enforcement gives users a fast
fail if they accidentally run a single-seed regression test against
the new policy.
