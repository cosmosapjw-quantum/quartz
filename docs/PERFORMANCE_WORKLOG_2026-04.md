# QUARTZ Performance Worklog — 2026-04

## Purpose

This document is a durable performance investigation log for the current
`gomoku7` training path on:

- CPU: Ryzen 9 5900X
- GPU: AMD Radeon RX 6950 XT
- Backend: PyTorch ROCm
- Training entrypoint: `python -m quartz.train`

It records:

- what was attempted
- what helped
- what did not help
- what regressed or was rolled back
- what artifacts were kept as reference

This is intentionally written as a systems worklog, not as a polished guide.
It exists so later optimization rounds can avoid repeating dead ends.

## Current Reference Artifacts

These were intentionally kept after cleanup and are the current reference set:

- `models/alphazero_gomoku7/autotune_profile.json`
- `models/alphazero_gomoku7/eval_autotune.json`
- `models/alphazero_gomoku7/best.pt`
- `models/alphazero_gomoku7/latest.pt`
- `models/alphazero_gomoku7/champion.json`
- `models/alphazero_gomoku7/glicko2_ladder.json`
- `artifacts/repro_selfplay_probe/gomoku7_20260408_174021`
- `artifacts/repro_selfplay_probe/gomoku7_20260408_174128`

Short meaning:

- `autotune_profile.json`: latest self-play/train batch autotune outcome
- `eval_autotune.json`: latest evaluation worker microbenchmark result
- `174021`: repro for `parallel=2, batch_games=2, n_threads=1`
- `174128`: repro for `parallel=4, batch_games=4, n_threads=1`

## Executive Summary

### What is now known

1. The dominant bottleneck is still **Rust search waiting for Python-side NN
   responses**, not feature serialization.
2. Shared-memory payload transport helped, but it did **not** remove the main
   bottleneck.
3. Multi-game `n_threads=1` was originally much worse than expected because it
   still fell back to effectively single-call evaluation. That specific issue
   was fixed.
4. Evaluation became functionally correct again after fixing the Python
   `select()` file-descriptor limit problem, but it is still too slow.
5. CPU underutilization is mostly a **pipeline fill / callback density / wait**
   problem, not raw Rust compute inefficiency.
6. Some aggressive expansions were useful as experiments but are not safe
   default-path optimizations yet.

### What remains true after many patches

- GPU is healthier than before, but still under-filled.
- CPU is still too idle relative to expected MCTS workload.
- Evaluation is still too expensive relative to game complexity.
- Self-play supply is still bursty rather than smooth.
- The newest batched evaluation attempt is currently not trustworthy as a
  benchmark source because later eval outputs contain void games with
  `error: "'gi'"`.

## Environment + Constraint Notes

- Search semantics should remain stable for ablation work.
- Systems changes are allowed; algorithm changes should be explicit and
  separable.
- We want fair comparison between QUARTZ-controlled search and a simpler
  baseline MCTS, ideally on the same execution infrastructure.

## Chronological Work Log

### 1. Initial diagnosis: CPU utilization too low, autotune suspicious

#### Observation

- Default concurrent training often chose `n_threads=1` with many processes.
- CPU utilization looked poor.
- Early logs suggested the learner was not the main bottleneck.

#### Action

- Collected hardware-aware autotune results.
- Added more candidate shapes for concurrent self-play.
- Re-measured self-play throughput directly instead of trusting prior defaults.

#### Result

- **Success**: `n_threads=1` was no longer treated as automatically optimal.
- **Success**: autotune started preferring more reasonable shapes such as
  multi-threaded / fewer-process configurations.
- **Evidence**: latest `models/alphazero_gomoku7/autotune_profile.json`
  currently resolves to:
  - `selfplay_parallel=4`
  - `bg_parallel=4`
  - `bg_batch_games=4`
  - `n_threads=4`
  - `batch=192`
  - `batch_size=18`

#### Remaining issue

- Better autotune selection did not fully solve low utilization, which implied
  the root bottleneck was not just bad parameter search.

---

### 2. Python-side replay/training plumbing cleanup

#### Action

- Moved replay sampling to a `DataLoader`.
- Removed duplicated Python package layout and kept one canonical `quartz/`
  code path.
- Added safer checkpoint loading and backend selection behavior.

#### Result

- **Success** for maintainability and training-path clarity.
- **Success** for reducing confusion and backend mismatch issues.
- **Not a major throughput win** by itself.

#### Why it still mattered

- It made later profiling easier by removing duplicated implementations and
  backend confusion.

---

### 3. Rust engine hot-path optimization pass

#### Action

- Reduced heap allocations and redundant work in:
  - Go legality / capture / scoring paths
  - Gomoku move generation / forbidden filtering
  - Chess legal move checking and terminal detection
- Added targeted evaluator-side micro-optimizations.

#### Result

- **Success** as local engine optimization.
- **Kept**.
- **Not sufficient** to solve end-to-end training wall-clock.

#### Interpretation

- These changes improved inner loops, but end-to-end time was still dominated by
  cross-process waiting and evaluation orchestration.

---

### 4. Binary QIPC and shared-memory payload transport

#### Action

- Replaced the hottest JSON payload paths with binary framing.
- Added shared-memory transport for eval request/response payloads.

#### Result

- **Partial success**.
- `codec_time` dropped substantially.
- The transport kind in later traces is consistently `shm`.

#### Evidence

- In the preserved repro artifacts:
  - `gomoku7_20260408_174021/summary.json`
  - `gomoku7_20260408_174128/summary.json`
- Rust QIPC summaries show `transport: shm`.

#### Why it was not enough

- `read_s` / waiting still dominated.
- This proved payload copy was not the main bottleneck anymore.

#### Decision

- **Keep** the SHM transport.
- **Do not** treat transport rewriting alone as the main optimization path.

---

### 5. Cross-process batch merge on the Python side

#### Action

- Modified the Python-side request handling so multiple Rust eval groups could
  be merged into a single model forward.

#### Result

- **Success**.
- Helped batch filling and reduced the worst-case single-call flood.

#### Evidence

- The latest repros still show under-filled batches, but they are no longer
  always `1.0`.

#### Limitation

- It still leaves Python as the broker for the control plane.
- This appears to be a structural limit.

---

### 6. Detailed repro harness for self-play stall localization

#### Action

- Added `scripts/repro_selfplay_probe.py`.
- Added Python-side stall trace output.

---

### 9. Evaluation batching phase-1: mixed result, current correctness regression

#### Action

- Added a batched Rust-backed evaluation path so evaluation could use
  `select_moves_batch(...)` instead of many independent `select_move(...)`
  calls.
- Added `search_profile=quartz|baseline` routing on the same Rust request path,
  so algorithm ablations can share one systems stack.

#### Result

- **Partial success**:
  - the direction is right
  - it avoids the old Python worker-cap bottleneck
  - it matches the intended systems direction better than the old path
- **Current failure**:
  - the newest evaluation outputs are not fully correct
  - latest `train_log.jsonl` rows record evals with `games: 0`
  - latest `eval_matches.jsonl` tail contains repeated void games with
    `error: "'gi'"`

#### Interpretation

- This means the newest evaluation wall-clock observations are **contaminated**.
- We should not use them as final proof that the architecture is right or wrong.
- The bug is in the new batched evaluation implementation, not evidence by
  itself that Rust-side evaluation is the wrong direction.

#### Decision

- **Keep the direction under investigation**.
- **Do not trust current phase-1 evaluation timing as a benchmark**.
- Fix correctness first, then re-measure.

---

### 10. 2026-04-09 integrated async-core run (`gomoku7_async_core_full`)

Reference artifact:

- `artifacts/runtime_monitor/gomoku7_async_core_full/summary.json`
- `artifacts/runtime_monitor/gomoku7_async_core_full/stdout.log`
- `artifacts/runtime_monitor/gomoku7_async_core_full/rust_qipc.jsonl`
- `artifacts/runtime_monitor/gomoku7_async_core_full/rust_server_trace.jsonl`

#### Run settings

- command: `venv/bin/python -m quartz.train --game gomoku7 --iterations 20 --retune --search-profile quartz`
- runtime tuner: `off`
- eval/self-play isolation: `on`
- return code: `0`
- wall-clock: `1580.18s`

#### What worked

1. End-to-end run completed cleanly with no eval correctness regression.
   - `last_eval.valid_eval=true`, `games=200`, `errors=0`, `voids=0`
2. Replay starvation symptom was removed in this run shape.
   - `training_wait_count=0`, `training_wait_total_s=0.0`
3. New async runner paths were actually active.
   - `run_multi_async_batch_start/done` events are present at high volume
   - `selfplay_runner_wave/progress/done` and `eval_runner_wave/progress/done` all present
4. Batch fill was acceptable.
   - `mean_batch_weighted=15.598` with `batch_size=18`
   - `target_batch_reached` dominates `max_wait_reached`

#### What did not work

1. Effective CPU utilization remained low.
   - `training cpu_thr_mean=0.517`
   - `evaluation cpu_thr_mean=0.628`
2. Wait remained dominant by a large margin.
   - `result_wait_s=317276.322`
   - `queue_wait_s=26747.628`
   - ratio `result/queue ≈ 11.86`
3. Async batch loop still produced many null results.
   - `async_batch_null_results_sum=5037`
   - `run_multi_async_batch_done` nulls remain non-trivial
4. Per-wave self-play yield is still low relative to wall-time.
   - `selfplay_wave_count=1972`
   - `selfplay_positions_emitted=11592` (~5.88 positions/wave)
   - `selfplay_wave_elapsed_ms_sum=846096.852` (~429ms/wave)

#### Interpretation

- This run confirms we fixed major correctness issues and improved pipeline
  stability, but not the core throughput bottleneck.
- Bottleneck remains the eval-result handshake boundary, not serialization.
- TT/edge lock is still not indicated as primary in this run.
- The remaining performance cap is structural:
  - search work submission is async
  - but completion/application still behaves like high-latency wait-bound flow

#### Decision after this run

Keep:

- async runner substrate
- runtime tuner off by default
- eval/self-play isolation for audit runs
- expanded trace/monitor summaries

Do not do next:

- more ad-hoc timeout tuning
- TT-first optimization
- codec transport churn

Do next:

- reduce null-result rate in async batch completion
- move toward true pending-eval/apply separation in core execution loop
- keep strict baseline profiles on same substrate for fair comparison

---

## Consolidated Progress Summary (this chat window)

### Major attempts

1. correctness recovery for eval pipeline (`gi`, zero-game pollution, monitor parsing)
2. shared-session Rust eval path
3. deadlock fix in batch collector lock scope
4. bounded low-lock broker channel migration
5. async batch runner introduction for `eval_nn_run` and `selfplay_nn_run`
6. monitor upgrade with bottleneck/stage summaries
7. runtime hygiene defaults:
   - runtime autotune off
   - eval/self-play isolation on
8. shipped-path review fixes (install quoting, MPS auto, play cfg preservation, jax extra)

### Confirmed successes

- evaluation correctness restored (`valid_eval=true`, no void/error in latest run)
- deadlock class in session open path removed
- monitor now exposes async-stage signals and runtime hygiene settings
- 20-iteration audit run completes reliably

### Confirmed failures / incomplete areas

- effective CPU usage still low despite async substrate introduction
- result-wait remains dominant
- async batch null-result volume still high
- true non-blocking pending/apply state machine is not complete yet

### Current primary bottleneck

- `sync_eval_handshake_orchestration` is still the top bottleneck class, now
  with stronger evidence under cleaner run conditions.

#### Cross-reference

- See `docs/CONCURRENT_MCTS_CRAG_2026-04.md` for the external baseline used to
  judge whether the next move should be more Rust-native orchestration or a
  rethink of MCTS itself.

---

### 10. Stronger audit finding: current Rust multi-search still fans out too aggressively

#### Observation

After code reading of `src/mcts_server.rs`, the current `search_nn_multi`
implementation still expands one incoming multi-search request into:

- one OS thread per active state/job
- each of which may then run `n_threads` internal MCTS threads

This is particularly problematic for evaluation, where many games can be active
at once.

#### Meaning

- This explains the observed "evaluation thread count spikes high" symptom.
- It also means the current Rust path is still not a true global inference or
  search broker.
- In practice, the system is still closer to "many independent searches packed
  into one RPC" than to the bounded concurrent analysis designs used by engines
  like KataGo.

#### Decision

- Treat this as a structural critique, not a minor tuning issue.
- Future Rust work should prefer:
  - bounded worker pools
  - a central broker / scheduler
  - explicit separation between:
    - positions in flight
    - search threads per position
    - NN batches in flight
- Added hard timeout and process-group kill so one hung candidate would not
  block the whole experiment.

#### Result

- **Major diagnostic success**.
- This was one of the most useful additions.

#### Key findings from the repro harness

##### Repro A

Artifact: `artifacts/repro_selfplay_probe/gomoku7_20260408_174021`

- configuration: `parallel=2, batch_games=2, n_threads=1`
- `positions_per_s ≈ 3.31`
- `eval_messages = 665`
- `model_calls = 665`
- `python_trace.max_read_wait_s ≈ 0.051`
- `rust_qipc.batch_mean_weighted ≈ 1.96`

---

### 7. Evaluation-path restructuring: batched Rust evaluator path

#### Observation

- Evaluation remained much slower than self-play.
- Even after the `select()` FD bug was fixed, `eval_matches.jsonl` still showed
  very small games taking several seconds each.
- The old evaluation path was still:
  - Python `TrainingEvaluator`
  - Python `ThreadPoolExecutor`
  - per-game `select_move()`
  - per-move Rust search calls

This meant evaluation was paying orchestration overhead on every single game
and every single move.

#### Action

- Added a batched evaluation path in `quartz/evaluation.py`.
- Added `select_moves_batch()` to the Rust-backed evaluator engine.
- Added `search_nn_multi` support to `NNSearchClient`, reusing the same
  bidirectional eval callback protocol but for many states at once.
- Changed `TrainingEvaluator.evaluate_checkpoint()` to prefer the batched path
  whenever both engines support `select_moves_batch()`.

#### Result

- **Success** as an architectural shift.
- This does **not** yet move the full evaluation runner into Rust, but it does
  remove the worst per-game / per-move orchestration pattern from the default
  Rust-backed evaluation path.
- This is now the main bridge toward a future Rust-native evaluation runner.

#### Decision

- **Keep**.
- Treat this as stage 1 of evaluation migration.
- The next larger step is still a Rust-owned tournament runner / broker.

---

### 8. Shared execution profile for QUARTZ vs baseline

#### Observation

- Ablation work needs a fair systems baseline.
- Up to this point, systems work and algorithm work were too entangled.

#### Action

- Added `search_profile = quartz | baseline` to the Rust-backed search request
  path.
- `baseline` now strips `QUARTZ` control and disables adaptive virtual loss,
  while keeping the same Rust server, transport, batching, and request path.

#### Result

- **Success** as an ablation-safety change.
- This does not directly improve throughput, but it makes later performance and
  algorithm studies comparable on the same systems substrate.

#### Decision

- **Keep**.
- Use this for future QUARTZ-vs-baseline comparisons after the new evaluation
  and broker infrastructure stabilizes.
- `rust_qipc.single_calls = 233`

##### Repro B

Artifact: `artifacts/repro_selfplay_probe/gomoku7_20260408_174128`

- configuration: `parallel=4, batch_games=4, n_threads=1`
- `positions_per_s ≈ 7.18`
- `eval_messages = 902`
- `model_calls = 902`
- `rust_qipc.batch_mean_weighted ≈ 3.68`
- `rust_qipc.single_calls = 63`

#### Interpretation

- Multi-game `n_threads=1` is now better than before, but still far too
  callback-heavy.
- Batching improved from catastrophic to merely insufficient.
- This strongly suggests the next frontier is **callback density reduction**, not
  more micro-optimizing the transport codec.

---

### 7. Resident-session Rust search attempt

#### Action

- Added a resident Rust session path for self-play search reuse.

#### Result

- **Failed as default-path optimization**.
- It caused autotune coarse candidates to stall.

#### Decision

- **Rolled back from the default path**.
- Left only as an optional / experimental path before later removal from the
  default workflow.

#### Why this matters

- This was the clearest example of a technically plausible optimization that
  became overengineering in practice.

---

### 8. JAX-on-Radeon attempt

#### Action

- Tried making the JAX backend usable for local Radeon training.
- Added explicit backend handling and prewarm logic.

#### Result

- **Failed for practical local training**.
- JAX devices were visible, but training backend initialization failed on
  real convolution paths.

#### Decision

- Default backend selection now prefers PyTorch.
- JAX remains explicit opt-in only.

#### Why it matters

- This removed a large source of confusion during performance work.

---

### 9. Evaluation bug: `games: 0`

#### Observation

- Evaluation results were sometimes logged as if no games had been scored.

#### Root cause

- Python-side evaluation client used `select.select()`.
- With enough file descriptors, it hit the `FD_SETSIZE` limit.
- The failure path manifested as effectively void/failed evaluations.

#### Action

- Replaced the relevant waits with a `poll`-based helper.

#### Result

- **Success**.
- Evaluation is now functionally progressing again.

#### Evidence

- The current `models/alphazero_gomoku7/eval_matches.jsonl` contains real
  completed games instead of an empty/void tally.

#### Remaining problem

- Evaluation is still much too slow.

---

### 10. Evaluation worker autotune

#### Action

- Replaced static worker-count recommendation with a measured pilot benchmark.
- Stored the result in `models/alphazero_gomoku7/eval_autotune.json`.

#### Current result

- Chosen worker count: `2`
- Benchmarked candidates:
  - `1 worker -> 0.1142 games/s`
  - `2 workers -> 0.2209 games/s`
  - `3 workers -> 0.2184 games/s`

#### Result

- **Success**, but only partial.
- It removed an arbitrary cap and replaced it with a measured choice.

#### Remaining problem

- Even after this fix, evaluation still takes on the order of seconds per tiny
  game, which means the orchestration model itself is still too expensive.

---

### 11. Dual-level early stopping

#### Action

- Split early stopping into:
  - outer iteration-level stopper
  - loose inner train-step stopper
- Required minimum fraction of planned steps before inner stopping can fire.

#### Result

- **Success** as a learning-efficiency change.
- **Not a primary throughput fix**.

#### Evidence

- Recent `train_log.jsonl` rows now include:
  - `planned_train_steps`
  - `inner_stop`

#### Current behavior snapshot

- Example from the current log:
  - iteration 6 executed `72/100` steps
  - inner stopper triggered only after `min_steps=70`

---

### 12. Runtime smoothing of background self-play

#### Action

- Made self-play telemetry chunk-based instead of cycle-only.
- Runtime tuning now uses rolling metrics and burstiness.

#### Result

- **Partial success**.
- This is better than the old cycle-only view.

#### Remaining issue

- Replay supply is still visibly bursty in the latest `train_log.jsonl`.
- Example:
  - iteration 6: `new_pos = 5506`
  - iterations 7–9: `new_pos ≈ 34–40`

This means the learner is still not being fed smoothly.

## Current State of the System

### What is now stable enough to keep

- binary/shm eval payload transport
- improved autotune candidate search
- measured evaluation worker selection
- dual-level early stopping
- poll-based evaluation wait handling
- cross-process eval merge
- current multi-game `n_threads=1` batch-eval fix
- repro harness + stall tracing

### What is explicitly not trusted as a final solution

- Python-owned evaluation orchestration
- Python acting as the main inference broker
- current callback density
- current replay supply smoothness
- any conclusion based only on CPU utilization percentage

### What has been rejected or downgraded

- resident-session default path
- “just keep tuning hardcoded values” as a strategy
- JAX local training path on current Radeon stack

## Open Bottlenecks

### 1. Evaluation orchestration

Current `eval_matches.jsonl` still shows tiny games taking roughly:

- `~6.5s` for some 7-move games
- `~8.5–9.0s` for many 9-move games
- `~10s` for some 10-move games

This is too high for the game size and search budget, even after worker
autotune.

### 2. Callback density

Even the improved repro still shows:

- hundreds of model calls per short probe
- batch means well below the configured batch ceiling

This indicates the system still flushes too often relative to useful batch fill.

### 3. Bursty self-play supply

The learner still sees large refresh bursts followed by thin trickles. That
under-fills hardware and destabilizes runtime tuning.

## Why the Next Step Needs to Be Bigger

At this point, the evidence no longer supports another round of small local
patches as the main strategy.

The main remaining bottlenecks are structural:

- Python still owns too much orchestration
- evaluation is still scheduled in Python
- inference requests are still too callback-heavy

The next serious step is therefore architectural:

1. move evaluation orchestration into Rust
2. add a Rust global inference broker for self-play + evaluation
3. keep Python as learner/checkpoint/monitor layer
4. support a controller-free baseline MCTS on the same transport infrastructure

## Kept Metrics That Matter

These are the metrics that should continue to gate future optimization work:

- self-play `positions/s`
- evaluation `games/s`
- mean batch size
- callback count
- Rust-side wait time vs encode/decode time
- replay burstiness
- iteration wall-clock

These should matter more than:

- raw CPU usage percentage
- raw VRAM usage percentage
- one-off microbenchmark wins in codec-only code

## Files and Paths Most Relevant For The Next Round

Python:

- `quartz/alphazero_train.py`
- `quartz/evaluation.py`

Rust:

- `src/mcts/eval.rs`
- `src/mcts_server.rs`
- `src/mcts/mod.rs`
- `src/mcts/search.rs`
- `src/mcts/parallel.rs`
- `src/mcts/quartz.rs`

These are the loci for:

- brokering
- evaluation execution
- baseline/quartz split
- batching and flush policy

## Bottom-Line Judgement

### Real optimization work that succeeded

- hot-path Rust engine cleanup
- better autotune search
- binary/shm payload transport
- measured evaluation worker tuning
- poll-based FD-safe evaluation fix
- repro/stall instrumentation
- current multi-game batch-eval fix

### Work that was useful diagnostically but not a keeper as default-path design

- resident search sessions
- deeper JAX local training work
- repeated Python-side heuristic layering as the main answer

### Most likely next high-value change

Move orchestration and batching ownership into Rust.

Not because Rust is inherently faster in all cases, but because the current
remaining bottleneck is the shape of the control plane, not just the speed of
individual functions.

## Second-Pass Audit Addendum

This addendum records issues found by a stricter adversarial pass that kept
reading through the original code and artifacts instead of stopping at the
first plausible explanation.

### A. Batched evaluation regression has a concrete protocol root cause

The current `error: "'gi'"` evaluation failures are not vague instability.

They come from a specific contract mismatch in `quartz/alphazero_train.py`:

- `_run_batched_eval_groups(...)` requires every group to contain `gi`
- the self-play batched path provides `gi`
- `NNSearchClient._exchange_search_request(...).parse_eval_group(...)` for
  `search_nn_multi` does **not** provide `gi`

This explains the current corrupted evaluation outputs:

- `models/alphazero_gomoku7/eval_matches.jsonl`
- `models/alphazero_gomoku7/train_log.jsonl`

So the current batched evaluation path is not just slow or noisy, it is
correctness-broken.

### B. Current evaluation tests do not cover the real Rust-backed fast path

The current tests cover:

- `_run_batched_eval_groups(...)` with synthetic groups that already contain
  `gi`
- `play_match_tally_batched(...)` with toy batched engines

They do **not** cover the integrated path:

- `RustNNEvaluatorEngine.select_moves_batch(...)`
- `NNSearchClient.search_moves_multi(...)`
- `_exchange_search_request(...)`
- `_run_batched_eval_groups(...)`

That missing integration coverage is why the current evaluation bug slipped
through.

### C. The current `baseline` path is useful but not a pure minimal-MCTS baseline

`search_profile=baseline` disables:

- QUARTZ controller
- adaptive virtual loss

but it still shares the same engine substrate and still inherits other engine
behavior such as root-forced-win and related engine-level defaults.

So future ablation writeups must describe this path accurately as:

- same systems substrate, QUARTZ/VL removed

not as:

- pure minimal MCTS with all extra search heuristics removed

### D. Monitor summaries have schema drift

The live monitor writes sample keys like:

- `cpu_percent_total`
- `proc_tree.total_threads`

Some earlier summary logic expected older names.

This is not the main performance problem, but it contaminates diagnosis and
should be treated as a measurement bug.

### E. Thread fanout remains the strongest structural critique

Even after the recent evaluation-stage work, the core Rust multi-search
topology is still:

- one host thread per job
- times `n_threads` internal search threads

That is still the most likely explanation for:

- evaluation thread spikes
- low CPU efficiency despite high GPU load
- weak scaling at higher concurrent counts

So the next architectural step still needs to be:

- bounded worker pool
- global inference broker

not more functionality layered onto the current per-job fanout path.

### F. Current batched evaluation is still split across engine-local clients

The new batched evaluation path is not globally brokered yet.

- `RustNNEvaluatorEngine` creates its own `NNSearchClient`
- each `NNSearchClient` owns its own Rust server process
- batched match running groups sessions by mover engine

So candidate and champion requests still do not share one inference broker.
This helps explain why evaluation can still show high GPU activity with poor
overall efficiency and elevated thread counts.

### G. Broken evaluations can still perturb published Elo state

`TrainingEvaluator.evaluate_checkpoint(...)` currently advances the ladder
period and recalibrates published Elo even when `tally.scored == 0`.

That matches the current observed symptom:

- `train_log.jsonl` eval rows with `games: 0`
- but nontrivial `published_elo` / `elo_gap` changes

So evaluation correctness bugs are currently capable of polluting later
training metadata, not just wasting time.

### H. Autotune/eval-autotune cache signatures remain under-specified

The current cache validation is still too narrow for systems work:

- self-play autotune validates only hardware signature + version
- eval autotune validates hardware/game/eval count/iters/thread count/batch

They still omit key semantics such as:

- `search_profile`
- QUARTZ/penalty-mode differences
- execution-topology changes

That means stale performance decisions can survive across meaningful systems
changes unless a manual `--retune` is forced.

### I. Monitor phase tagging can overstate evaluation residency

`profile_training_monitor.py` marks phase as `evaluation` on the evaluation
start line and returns to `training` only when the next iteration line is
observed. So monitor output can visually over-attribute time to evaluation.

This is a diagnosis issue, not the root bottleneck, but it matters when
reading long-running monitor sessions.

### J. Stop-reason logs are category-level, not always exact measurements

Current controller metadata uses convenient category outputs such as:

- `BudgetExhausted { iterations: limit }`
- `TimeCapHit { elapsed_ms: budget_ms }`

These are useful labels, but they should not be treated as exact measured stop
times in later ablation summaries.

## Update: 2026-04-08 Fix-First Pass 2

This pass addressed the most immediate correctness and topology gaps without
changing search semantics.

Implemented:

- expanded QIPC eval payloads so requests can carry a `model_tag`
- taught Python batch-eval decoding to route mixed-model requests by tag
- added a shared `BatchStdioEval` collector path so two evaluators can share
  one batch collector / Rust server connection
- routed `search_nn_multi` through tagged jobs, so multi-game and dual-model
  evaluation can use one Rust subprocess rather than separate candidate and
  champion subprocesses
- tightened self-play and eval autotune signatures to include search/topology
  semantics
- fixed live monitor phase transitions so `evaluation_result` moves the phase
  out of `evaluation`

Validated:

- `cargo build --release`
- `cargo test --quiet mcts_server::tests`
- `pytest tests/test_training_pipeline_regressions.py tests/test_evaluation_pipeline_regressions.py`

Still intentionally not solved in this pass:

- there is still no true global Rust inference broker shared across self-play
  and evaluation
- evaluation orchestration still lives partly in Python even though candidate
  and champion now share one Rust client in the batched path
- warning cleanup remains limited to touched hot-path modules only

## Update: 2026-04-08 Utilization Audit Addendum

After re-reading the latest monitor and repro artifacts, the "GPU high, CPU low"
pattern should be interpreted carefully.

Confirmed normal factors:

- the model is small, so low VRAM use is expected on a 16GB GPU
- CPU and GPU utilization do not need to be symmetric in NN-guided MCTS

Confirmed abnormal factors:

- QIPC time remains wait-dominant (`io_time_s >> codec_time_s`)
- thread counts can rise sharply without corresponding CPU saturation
- evaluation can still look expensive even when GPU usage is high

Working conclusion:

- low VRAM by itself is not a problem
- low CPU by itself is not a problem
- but the current combination of low CPU, high thread count, and wait-dominant
  QIPC remains evidence of an execution-topology bottleneck rather than a
  hardware-imposed limit

So future optimization work should continue to focus on:

- Rust-side coordination
- shared/global batching
- evaluation orchestration

not on utilization numbers in isolation.

## Update: 2026-04-09 Chess Session Step 1

Resident-style Rust search sessions now support chess state updates in the
server and in the shared Rust evaluation path.

Implemented:

- `search_nn_multi_session_*` now accepts chess jobs expressed as `fen`
- Rust `SearchSessionAny` has a chess variant and can apply chess action/deactivate
  updates
- `RustNNEvaluatorEngine.play_match_tally_against(...)` now uses session payloads
  for chess too, with `fen`-based jobs instead of board arrays

Validated:

- Rust unit smoke for chess session action/deactivate update
- Python regression that shared chess evaluation opens a session with `fen`
  payloads and steps/closes it successfully
- `cargo build --release`
- `cargo test --quiet mcts_server::tests`
- targeted Python evaluation/training regressions

Intentionally deferred:

- chess self-play resident auto-enable remains off
- reason: chess self-play resident mode still needs a dedicated audit of the
  update loop before enabling by default
- current training-side chess resident self-play path would need a separate
  move/result update contract review, so this step only opens the safer
  evaluation/session substrate first

## Update: 2026-04-09 Async Core Full (latest rerun)

Reference:

- `artifacts/runtime_monitor/alphazero_gomoku7_20260409_231945/summary.json`

Observed:

- run completed (`returncode=0`)
- `runtime_tuner_enabled=false`, `eval_selfplay_isolated=true`
- effective CPU remains low:
  - training `cpu_thr_mean=0.523`
  - evaluation `cpu_thr_mean=0.633`
- result-side wait still dominates:
  - `result_wait_s=400905.907`
  - `queue_wait_s=31310.545`
  - ratio ≈ `12.8`
- batch fill remains decent:
  - `mean_batch_weighted=15.688` (target 18)
- codec remains minor:
  - `io_time_s=1356.319`
  - `codec_time_s=2.017`

Interpretation:

- the dominant bottleneck remains orchestration/completion wait, not codec or
  obvious lock contention
- current async substrate is active but still not delivering high effective CPU
  utilization

Important correction:

- `run_multi_async_batch_done.null_results` currently counts null entries in
  the full slot-aligned result vector, which includes inactive slots.
- so `async_batch_null_results_sum` should not be treated as a pure "failed
  eval completion" metric without normalization.

Stronger structural signal:

- `max_active_waiters=1200` aligns with large-wave launch geometry
  (`jobs=200`, `max_inflight_per_job=6`).
- this points to a missing global inflight admission control/credit scheduler.

Current non-heuristic next step:

1. introduce a global inflight credit budget across jobs
2. gate new submissions on global credits, not only per-job pending
3. separate "inactive-slot null" from real completion misses in telemetry
4. keep runtime tuner off until completion path is stabilized
