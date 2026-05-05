# Audit — BQ++ Phase 8c: SearchPolicy halt in parallel + async batched paths

**Date:** 2026-05-05
**Scope:** extend the Phase 8b single-threaded `MctsEngine::run` policy
halt integration to the three remaining hot paths used by training and
evaluation: `run_par`, `run_par_quartz`, and the async batched
self-play path (`process_job_tick` + `run_multi_async_batch_tags`).

This commit closes the gap diagnosed in the 3-policy toy ablation
(none / legacy_az / kl_lucb_stop) where all three policies produced
**identical champions** and KLLUCBStop fired **0 times** despite being
attached. Root cause: Phase 8b only wired `policy_halt_check` into the
single-threaded `run()` loop; the actual training/eval search uses
parallel and async paths, all of which bypassed the policy.

## What changed

### `src/mcts/mod.rs` (~80 LOC)

**1. `policy_halt_check` exposed as `pub`.**

Was `fn policy_halt_check`; now `pub fn` so `mcts_server.rs` can call it
from the async batched path. No behavior change for in-crate callers.

**2. `run_par` policy halt (~40 LOC across two branches).**

A shared `AtomicBool policy_halted` is created outside the `rayon::scope`.
Thread 0 polls `policy_halt_check` once per `qcfg.check_interval` (or
every 64 launched iters when no quartz config is attached); when the
policy returns `Stop(_)`, thread 0 sets the flag and breaks. Other
threads observe the flag at chunk boundary (fast path) or per-iteration
(slow path) and break. The `has_policy` flag short-circuits the atomic
read entirely when no policy is attached, so the no-policy hot path
adds zero overhead beyond a single bool branch on `Option::is_some`.

**3. `run_par_quartz` policy halt (~25 LOC across two branches).**

Same shape as `run_par`: shared `AtomicBool` + thread-0-driven poll.
The poll piggybacks on the existing `tid == 0 && local_it %
check_interval == 0` quartz-stats refresh block, so adds at most one
`policy_halt_check` call per check_interval iters per worker pool.

**4. Tests added (2).**

- `test_phase8c_run_par_honors_search_policy_halt`:
  LegacyAlphaZero(budget=50) + FixedIterations(1000) + 2 worker
  threads. Asserts `iterations < 800` (loose bound to absorb cross-
  thread propagation lag through the AtomicBool flag) and
  `root_visits >= 50` (the policy budget).

- `test_phase8c_run_par_quartz_honors_search_policy_halt`:
  Same shape but goes through QuartzController(1000, qcfg). Pins
  the quartz path's policy integration.

### `src/mcts_server.rs` (~30 LOC)

**1. `AsyncBatchJob` schema extension.**

New fields:
- `policy_halted: bool` — latched once the policy fires for this job.
- `policy_check_tick: u32` — last `launched` count at which the policy
  was polled, used to throttle the check.

The latched flag stops the gather phase from launching new iterations
on the next `process_job_tick` call. The reap phase continues running
so in-flight tickets drain cleanly — no `_permit` leaks because the
RAII `AsyncBatchPending` is consumed on `try_take()`.

**2. `process_job_tick` policy poll.**

Added at the head of the function: when a policy is attached and at
least `check_interval` (or 64) launched iters have passed since the
last poll, call `engine.policy_halt_check`. On `Stop(_)`, latch
`policy_halted = true`. The gather `while` loop now also gates on
`!job.policy_halted`.

**3. Outer-loop completion conditions updated.**

Both the single-threaded and multi-threaded paths in
`run_multi_async_batch_tags` updated their loop conditions from:

```rust
job.completed < iters || !job.pending.is_empty()
```

to:

```rust
(!job.policy_halted && job.completed < iters) || !job.pending.is_empty()
```

This lets a policy-halted job exit cleanly the moment its pending
queue drains, without waiting for `completed == iters`.

## Test results

- Phase 8c tests: 2/2.
- `cargo test --release`: **544 passed** (was 542 + 2 = 544).
- All P01-P08 + Phase 0-8b tests still pass (back-compat preserved).

## Bug caught during integration

**Initial wiring used `engine.config()` accessor that didn't exist.**
`MctsEngine.config` is a `pub` field, not an accessor method. Compile
error fixed by switching to direct field access.

## Concurrency considerations

**AtomicBool ordering: `Relaxed` is sufficient.** The flag is purely
advisory — readers may take one or two extra iterations after thread 0
publishes, but cannot miss the publish indefinitely (the flag is
checked at every chunk/iteration boundary, and Relaxed loads on the
same memory location see writes within a few cycles on x86/ARM). The
worst-case overshoot is bounded by `ticket_chunk × n_threads` for the
fast path or one observation interval for the slow path. Tests cap
`iterations < 800` (vs the policy's budget of 50) to absorb this
slack with margin.

**Gather/reap separation in `process_job_tick`.** The gather phase is
gated on `!job.policy_halted`; the reap phase is unconditional. This
ensures pending tickets always drain. If reap-only execution can make
no progress (no pending), the outer loop's updated condition exits.

**Single-thread-0 caller for `policy.observe`.** In all parallel paths,
only thread 0 invokes `policy_halt_check` (which internally calls
`policy.observe` periodically). This matches the single-thread design
of the policy implementations (KLLUCBStop holds a `parking_lot::Mutex`
on its cache; thread 0 contention is zero).

## What this commit does NOT do (still deferred)

This is the **halt-side-only** integration. Per-edge `score_adjustment`
plumbing through `select.rs` is still pending — that is what unblocks
LegacyQuartz, BayesianQuartz, BQPP composed, and MENTS to drive
selection (not just halt). Halt-only policies (LegacyAlphaZero,
KLLUCBStop) are now fully integrated across all hot paths.

## Files touched

- `src/mcts/mod.rs` (+86 / -8, including 2 new tests + atomic import +
  pub on `policy_halt_check`)
- `src/mcts_server.rs` (+34 / -4)

Net delta: **+120 / -12 LOC**.

## What unblocks next

The 3-policy toy ablation can now be re-run; KLLUCBStop should fire
non-zero times in `halt_reason_count` (not just legacy quartz's
VocNonPositive). Differentiation between policies in
`nn_evals_per_move` and per-game iteration counts becomes observable.
The original ablation that produced identical champions across
none/legacy_az/kl_lucb_stop should now produce policy-specific
behavior at least in the budget_fairness halt distribution.
