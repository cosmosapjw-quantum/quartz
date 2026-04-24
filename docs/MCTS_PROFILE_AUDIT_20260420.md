# MCTS Profile Audit — 2026-04-20

This audit focuses on the Rust MCTS core rather than per-game encoder work.

## Scope

- Static audit of:
  - `src/mcts/mod.rs`
  - `src/mcts/select.rs`
  - `src/mcts/expand.rs`
  - `src/mcts/backup.rs`
  - `src/mcts/parallel.rs`
  - `src/mcts/tt.rs`
  - `src/mcts/node.rs`
  - `src/mcts/eval.rs`
- Dynamic profiling of:
  - `mcts::tests::bench_search_controller_fixed_iterations_fast_path`
  - `mcts::select::tests::bench_linear_scan_vs_reference`
  - `mcts::parallel::tests::bench_parallel_vl_hot_path`
  - `mcts::tests::test_run_par_quartz_four_threads`

## Artifacts

- `tmp/mcts_profiles/search_controller.callgrind`
- `tmp/mcts_profiles/search_controller.summary.txt`
- `tmp/mcts_profiles/search_controller.heaptrack.zst`
- `tmp/mcts_profiles/search_controller_flame.svg`
- `tmp/mcts_profiles/select_linear.callgrind`
- `tmp/mcts_profiles/parallel_vl.callgrind`
- `tmp/mcts_profiles/par_quartz_four_threads.callgrind`
- `tmp/mcts_profiles/par_quartz_four_threads_flame.svg`

## Baseline Bench Results

- `bench_search_controller_fixed_iterations_fast_path`
  - native warm run: `reference=34014 nps optimized=35047 nps speedup=1.030x`
- `bench_parallel_vl_hot_path`
  - native warm run: `speedup=1.04x`
  - callgrind run: `speedup=0.96x`
- `bench_linear_scan_vs_reference`
  - native run: `speedup=1.65x`
  - callgrind run: `speedup=1.71x`

Interpretation:

- `select` micro-optimization wins are real, but mostly micro-bench local.
- `parallel VL` is not a major current bottleneck.
- the actual fixed-iterations search loop only improved by about `1.03x`, which means the remaining cost is outside the optimized loop-control logic.

## Static Audit Findings

### 1. Hot-path timing and contention instrumentation is always on

Relevant code:

- `src/mcts/mod.rs:342-400`
- `src/mcts/tt.rs:100-144`
- `src/mcts/node.rs:47-59`
- `src/mcts/node.rs:310-316`
- `src/mcts/expand.rs:105-107`

Observed pattern:

- `iterate()` always records phase timings with `Instant::now()` / `elapsed()`.
- TT access always records lock wait with `Instant::now()` / `elapsed()`.
- edge snapshot and edge materialization also record lock wait on every call.

Risk:

- this is benchmark-safe but not performance-free.
- the profiling path is currently mixed into production hot paths.

### 2. TT hashing is paying full `HashMap` hasher cost on already-hashed keys

Relevant code:

- `src/mcts/tt.rs:36-38`
- `src/mcts/tt.rs:100-130`

Observed pattern:

- TT stores `u64 -> Arc<MctsNode<_>>` inside `std::collections::HashMap`.
- each lookup/insert hashes an already-computed Zobrist-style `u64`.

Risk:

- this is semantically safe but expensive.
- the table is paying `RandomState`/SipHash cost that is not buying correctness.

### 3. `materialize_edges()` is the real structural center of cost

Relevant code:

- `src/mcts/expand.rs:87-120`
- `src/mcts/node.rs:275-278`

Observed pattern:

- for each materialized edge:
  - `state.apply_move(mv)`
  - `child_state.tt_hash()`
  - `tt.get_or_create(...)`
  - `child.set_parent(node)`
  - push `Arc<MctsEdge<_>>`

Risk:

- this is not a select-formula bottleneck.
- it is a state-transition + TT + allocation bottleneck.

### 4. QUARTZ bookkeeping still allocates and clones at check boundaries

Relevant code:

- `src/mcts/mod.rs:574-625`
- `src/mcts/mod.rs:876-879`

Observed pattern:

- `run_quartz()` periodically calls `root_priors()`.
- `root_priors()` calls `edge_snapshot()`, which clones `Arc`s into a fresh `Vec`.

Risk:

- not the first-order bottleneck in the current fixed-iterations benchmark.
- still a real source of overhead in QUARTZ-heavy runs.

### 5. Fallback evaluator path is intentionally serialized

Relevant code:

- `src/mcts/eval.rs:1229-1231`

Observed pattern:

- `StdioCallbackEval` protects request/response with a single `Mutex`.

Risk:

- acceptable as fallback.
- not benchmark-safe for threaded search.

### 6. Broker loops have heavy timing/timeout bookkeeping inside the collection loop

Relevant code:

- `src/mcts/eval.rs:1749-1857`
- `src/mcts/eval.rs:2016-2230`

Observed pattern:

- multiple `Instant::now()` / `elapsed()` calls per batch cycle.
- adaptive timeout retuning and queue metrics are always active.

Risk:

- probably secondary relative to actual transport cost.
- still worth isolating from strict performance runs.

## Dynamic Profiling Findings

### A. Fixed-iterations core search (`search_controller.callgrind`)

Total:

- `Ir = 3,481,456,191`

Top flat costs:

- `__memcpy_avx_unaligned_erms`: `10.02%`
- `MctsEngine::iterate`: `9.70%`
- `_int_malloc`: `8.86%`
- `Gomoku::check_win_at`: `8.86%`
- `BuildHasher::hash_one`: `7.29%`
- `_int_free`: `7.25%`
- `TranspositionTable::get_or_create`: `6.15%`
- `malloc`: `5.76%`
- `DefaultHasher::write`: `4.92%`
- `Gomoku::apply_move`: `4.64%`
- `materialize_edges`: `3.02%`
- `MctsNode::set_parent`: `2.10%`
- `Timespec::now`: `1.39%`
- `Instant::elapsed`: `1.13%`

Important tree view conclusions:

- `expand_and_evaluate` accounts for most of the true work under `iterate`.
- `materialize_edges` fans directly into:
  - `apply_move`
  - `tt.get_or_create`
  - `malloc/free`
  - `set_parent`
- `select` is not the dominant cost in the real search loop.

### B. Heap profile of fixed-iterations search

From `heaptrack --analyze`:

- `1,470,098` allocation calls from `Gomoku::apply_move`
- `1,383,232` allocation calls from `materialize_edges`
- `materialize_edges` peak heap contribution around `99.59 MB`

Interpretation:

- actual search cost is dominated by materialization/state-copy/allocation churn.
- this matches the callgrind result and rules out “PUCT math” as the main current problem.

### C. `select` micro-bench (`select_linear.callgrind`)

Total:

- `Ir = 10,425,685,556`

Top flat costs:

- `ablation_puct_score_with_parent_sqrt`: `41.72%`
- `__logf_fma`: `24.18%`

Interpretation:

- in QUARTZ/ablation-heavy selection, log/exp math inside refresh blending is the dominant micro-cost.
- this matters for controller-heavy runs, but not enough to explain the fixed-iterations full-search profile by itself.

### D. Parallel VL micro-bench (`parallel_vl.callgrind`)

Total:

- `Ir = 167,068,498`

Interpretation:

- too small and too harness-dominated to justify more local tuning right now.
- current adaptive VL math is not where the main search budget is going.

### E. Parallel QUARTZ smoke (`par_quartz_four_threads.callgrind`)

Total:

- `Ir = 52,084,573`

Top flat costs:

- `__memcpy_avx_unaligned_erms`: `11.11%`
- `Gomoku::check_win_at`: `9.22%`
- `_int_malloc`: `9.01%`
- `_int_free`: `7.27%`
- `BuildHasher::hash_one`: `6.94%`
- `MctsEngine::iterate`: `5.70%`
- `tt.get_or_create`: `5.64%`
- `apply_move`: `4.83%`
- `DefaultHasher::write`: `4.63%`
- `materialize_edges`: `2.76%`
- `ShortRollout::evaluate`: `1.84%`
- `crossbeam_deque::Stealer::steal`: `1.54%`

Interpretation:

- even in threaded QUARTZ search, the core picture does not change:
  - state transition
  - TT hashing
  - allocation/free
  - edge materialization
  dominate.
- work-stealing overhead is visible but still secondary.

### F. Flamegraph notes

- `search_controller_flame.svg` was captured successfully and is consistent with callgrind: `materialize_edges`, `apply_move`, TT lookup/insert, and allocation dominate the stack.
- `par_quartz_four_threads_flame.svg` was captured with sample loss:
  - processed sample loss reported around `23%`
  - acceptable as a qualitative check, not as a precise percentage source

## Strict, Semantics-Safe Optimization Targets

These are the best candidates before touching search semantics.

### Priority 0

1. Gate or compile-switch hot-path timers and contention counters
   - `iterate()` phase timers
   - TT lock wait timing
   - edge lock wait timing

2. Replace TT hasher with an identity/nohash hasher for `u64`
   - correctness is unchanged because the external TT key is already a finalized hash

### Priority 1

3. Reduce `materialize_edges()` allocation churn
   - especially edge object creation and parent-link bookkeeping

4. Replace `parent: RwLock<Option<Weak<_>>>` with a write-once parent mechanism
   - current `set_parent()` cost is visible and semantically simple

### Priority 2

5. Remove `root_priors()` edge snapshot allocation from QUARTZ check path
   - reuse a scratch buffer or iterate under the read lock without cloning `Arc`s

6. Separate “strict benchmark” eval path from “fully instrumented profile” eval path
   - especially in `eval.rs`

## Non-Priorities For Now

These currently look like poor targets.

- more micro-tuning of `ParallelismController`
- more micro-tuning of the basic fixed-iterations stop loop
- further select-loop algebra tweaks without addressing TT/materialization/state-copy

## Bottom Line

The current MCTS bottleneck is not the tree policy formula.

The current MCTS bottleneck is:

1. edge materialization
2. state transition during materialization
3. TT hashing and TT insertion
4. allocation/free churn
5. always-on profiling and contention accounting in production hot paths

Any next optimization round should target those first.
