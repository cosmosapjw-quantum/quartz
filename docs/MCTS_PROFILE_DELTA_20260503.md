# MCTS Profile Delta - 2026-05-03

This note records the current game-agnostic CPU optimization pass for the Rust
MCTS production path. It is not a strength claim for QUARTZ search quality; it
only documents execution behavior and semantic guardrails.

## Landed changes

1. Parallel fixed-budget searches now reserve visit tickets in chunks when
   `n_threads > 1`.
   - Applies to both `run_par(FixedIterations)` and
     `run_par_quartz(HaltMode::Fixed)`.
   - Each worker executes only ticket IDs below the requested limit, so
     `root_visits` remains exact even when the budget is not divisible by the
     ticket chunk.
   - Controller scoring, QUARTZ stats refresh, and halt telemetry are
     unchanged.

2. Best-effort progressive widening now has a per-node materialization owner
   claim.
   - The claim is used only after at least one edge is already published.
   - Serial materialization and first-edge publication remain blocking.
   - A busy materialization lock is checked before child-hash / TT preparation,
     preserving the previous "skip instead of prepare-and-wait" behavior.
   - The claim prevents duplicate preparation work when multiple parallel
     selectors widen the same node concurrently.

3. Edge materialization now resolves larger TT child batches by bucket.
   - Batches smaller than four use the existing scalar `get_or_create` path.
   - Larger batches keep the same hit/miss accounting and duplicate-hash
     behavior while taking fewer repeated TT bucket locks.
   - The materialized edge order remains the candidate order; only the child
     lookup work is coalesced.

4. Uniform/simple rollout evaluators now delegate root policy construction to
   `GameState::uniform_eval(value)` after computing the rollout value.
   - Gomoku and Gomoku15 already provide direct uniform-policy builders, so
     ShortRollout avoids a redundant root legal-move vector on those games.
   - The rollout value path is unchanged.

5. Automatic thread selection is now available as an opt-in production path.
   - Explicit `run_par(..., n_threads)` and `run_par_quartz(..., n_threads)`
     are unchanged for reproducible ablations.
   - `MctsEngine::run_auto` and `MctsEngine::run_quartz_auto` choose an
     effective thread count from host parallelism, remaining fixed-visit
     budget, root legal count, PW status, and select-scratch support.
   - `AutoThreadPolicy::throughput()` maximizes raw NPS when the visit budget
     can absorb scheduling overhead.
   - `AutoThreadPolicy::quality()` caps low-branching or small-budget searches
     more aggressively to reduce duplicate virtual-loss churn.
   - `QUARTZ_PROFILE_THREADS=auto|throughput|quality` exercises the same path
     in the supported-game profile harness and prints `thread_policy` plus
     `auto_reason` on each result line.

6. The single-position Rust NN search server path now accepts opt-in automatic
   thread selection.
   - `search_nn` requests can set `"n_threads":"auto"` or
     `"thread_policy":"quality"` / `"auto_thread_policy":"quality"`.
   - `"thread_cap"`, `"max_threads"`, or `"n_threads_cap"` bound the host cap.
   - Search results and `search_manifest` now distinguish
     `requested_threads`, `effective_threads`, `thread_policy`, and
     `auto_thread_reason`.
   - Multi-position/session search remains on explicit `n_threads` for now so
     existing batched-eval ablations do not silently change worker scheduling.

## Validation

Commands used an isolated target directory because the default `target/` was
held by stale cargo sessions during this pass.

| Command | Result |
|---|---|
| `cargo fmt --check` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target cargo test --release --locked test_run_par_fixed_budget_chunking_is_exact_when_not_divisible -- --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target cargo test --release --locked test_run_par_quartz_fixed_budget_chunking_is_exact_when_not_divisible -- --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target cargo test --release --locked test_best_effort_materialization -- --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target cargo test --release --locked batch_get_or_create -- --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target cargo test --release --locked test_run_par_quartz_basic -- --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target cargo test --release --locked test_auto -- --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target cargo test --release --locked run_auto -- --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target cargo test --release --locked run_quartz_auto -- --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target cargo test --release --locked nps_baseline -- --ignored --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target QUARTZ_PROFILE_REPEATS=2 QUARTZ_PROFILE_SKIP_OPS=1 cargo test --release --locked profile_mcts_parallel_all_supported_games -- --ignored --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target QUARTZ_MCTS_HOTPATH_METRICS=1 QUARTZ_PROFILE_REPEATS=2 QUARTZ_PROFILE_SKIP_OPS=1 cargo test --release --locked profile_mcts_parallel_gomoku15_freestyle -- --ignored --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target QUARTZ_PROFILE_THREADS=auto QUARTZ_PROFILE_REPEATS=2 QUARTZ_PROFILE_SKIP_OPS=1 cargo test --release --locked profile_mcts_parallel_all_supported_games -- --ignored --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target QUARTZ_PROFILE_THREADS=quality QUARTZ_PROFILE_REPEATS=1 QUARTZ_PROFILE_SKIP_OPS=1 cargo test --release --locked profile_mcts_parallel_all_supported_games -- --ignored --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target cargo test --release --locked search_thread_spec -- --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target cargo test --release --locked execute_search_auto -- --nocapture` | pass |
| `CARGO_TARGET_DIR=/tmp/quartz-target cargo test --release --locked attach_search_metadata_preserves_auto_thread_manifest -- --nocapture` | pass |
| `pytest -q tests/test_training_pipeline_regressions.py -k "rust_search_options or system_runtime_api or search_manifest_key"` | pass |

## Current local profile read

Thread-scaling was rerun across all supported games after the TT-batch and
fixed-ticket passes. The earlier 4-thread result was not stable once the full
supported-game matrix and longer scale=10 runs were considered. On this host
(`available_parallelism() = 24`), raw NPS generally improves through the
logical-thread cap, while duplicate-selection telemetry rises on low-branching
or short-budget searches.

Longer scale=10 validation, median NPS:

| Game | 12 threads | 24 threads | Interpretation |
|---|---:|---:|---|
| tictactoe | 1.24M | 1.45M | throughput wins, but dup-rate is high |
| gomoku7 | 279k | 305k | 24t slightly better; TT/materialization pressure remains |
| gomoku15_freestyle | 752k | 1.09M | 24t wins |
| gomoku15_standard | 965k | 1.28M | 24t wins |
| gomoku15_omok | 746k | 1.12M | 24t wins |
| gomoku15_renju | 599k | 915k | 24t wins |
| gomoku15_caro | 1.01M | 1.28M | 24t wins |
| chess | 1.43M | 2.22M | 24t wins |
| go9 | 1.21M | 1.63M | 24t wins |
| go13 | 930k | 1.11M | 24t wins |
| go19 | 714k | 938k | 24t wins, but short-budget noise is visible |

After the TT-batch pass, hotpath instrumentation for 4 threads reported
median:

| Metric | Median |
|---|---:|
| select ns / iter | 5,923.9 |
| expand ns / iter | 8,274.7 |
| backprop ns / iter | 368.0 |
| TT lock wait ns / iter | 316.1 |
| edge lock wait ns / iter | 529.0 |
| best-effort busy skips / iter | 0.069 |

The opt-in throughput policy selected host-cap threads for the default
supported-game profile except Go19, where the 1500-visit budget limited the
decision to 23 threads. Representative median NPS from
`QUARTZ_PROFILE_THREADS=auto`, `QUARTZ_PROFILE_REPEATS=2`:

| Game | Median NPS |
|---|---:|
| tictactoe | 1.20M |
| gomoku7 | 277k |
| gomoku15_freestyle | 473k |
| gomoku15_standard | 1.18M |
| gomoku15_omok | 882k |
| gomoku15_renju | 688k |
| gomoku15_caro | 1.05M |
| chess | 1.47M |
| go9 | 1.60M |
| go13 | 1.07M |
| go19 | 1.50M |

The quality policy is intentionally lower-throughput on small or short-budget
searches. In the same harness it selected 4 threads for TicTacToe, 12 for
Gomoku7, 8 for Chess, 20 for Go9, 10 for Go13, and 3 for Go19. This path is
meant for lower duplicate-selection pressure and study discipline, not maximum
node throughput.

## Interpretation

The kept changes are semantic-preserving execution optimizations. They reduce
coordination overhead, duplicate materialization preparation, repeated TT bucket
locking, and redundant root uniform-policy allocation without changing search
policy, first-edge visibility, or fixed-budget accounting.

The profile still does not justify a broad lock-free rewrite or a new external
concurrency library. `rayon`, `parking_lot`, `crossbeam-channel`, `smallvec`,
and `bumpalo` are already present where they help. The remaining bottlenecks
are code-structural:

- low-branching searches can convert extra threads into high duplicate-path
  telemetry rather than better search diversity
- Gomoku7/no-PW still spends disproportionate work in TT get-or-create and
  materialization
- short fixed budgets are noisy enough that raw NPS alone should not select a
  publication ablation setting

Next high-leverage work should be code-structural and measured:

1. Add a duplicate-rate feedback preset that can downgrade from throughput to
   quality mode after a warmup window.
2. Revisit lock-free TT only if slot footprint stays at 16 B or the workload is
   demonstrably hit-heavy.
3. Consider a no-allocation scratch path for dense NN fallback policy
   normalization; that is outside the rollout-only evaluator path optimized
   here.
