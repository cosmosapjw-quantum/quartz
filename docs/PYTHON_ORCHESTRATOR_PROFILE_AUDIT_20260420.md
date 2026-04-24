# Python Orchestrator Profile Audit (2026-04-20)

This audit covers the Python-side orchestration path rather than game rules or the Rust MCTS core.

Scope:
- `quartz/cli_main.py`
- `quartz/selfplay_runtime.py`
- `quartz/evaluator_runtime.py`
- `quartz/runtime_support.py`
- `quartz/qipc.py`
- `quartz/replay.py`
- `quartz/torch_training_runtime.py`
- `quartz/jax_training_runtime.py`

Method:
- static code audit of load-bearing orchestration paths
- `cProfile` on representative self-play, replay, and evaluator runs
- `-X importtime` for startup/import cost
- `tracemalloc` for Python heap attribution

Representative artifacts:
- [tmp/profile_python_orchestrator.py](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/profile_python_orchestrator.py)
- [tmp/profile_python_evaluator.py](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/profile_python_evaluator.py)
- [tmp/python_orchestrator_profiles_final/selfplay_profile.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_orchestrator_profiles_final/selfplay_profile.txt)
- [tmp/python_orchestrator_profiles_final/replay_profile.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_orchestrator_profiles_final/replay_profile.txt)
- [tmp/python_evaluator_profiles_smoke/evaluator_profile.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_evaluator_profiles_smoke/evaluator_profile.txt)
- [tmp/python_orchestrator_importtime_alphazero.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_orchestrator_importtime_alphazero.txt)
- [tmp/python_orchestrator_importtime_torch_runtime.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_orchestrator_importtime_torch_runtime.txt)
- [tmp/python_orchestrator_tracemalloc.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_orchestrator_tracemalloc.txt)

## 1. Static Audit

### 1.1 `cli_main.py`

`prepare_training_context()` and `run_training_main()` are still the real orchestration spine.

Findings:
- The module is not algorithmically expensive by itself.
- It does duplicate runtime setup fields a few times:
  - `_selfplay_topology_version`
  - `_shared_eval_session`
  - `_broker_enabled`
  - `_eval_runner_mode`
  - `_selfplay_runner_mode`
- The duplication is mostly hygiene debt, not a runtime hotspot.
- The more meaningful Python runtime cost starts after `run_training_main()` delegates into:
  - `selfplay_rust_nn_batched()`
  - `wait_for_worker_progress()`
  - `ReplayBuffer.build_dataloader()`
  - evaluator/arena entrypoints

Conclusion:
- `cli_main.py` is mostly coordination overhead, not hot-path compute.

### 1.2 `selfplay_runtime.py`

This is the main Python orchestration hotspot.

Findings:
- `NNSearchClient._exchange_search_request()` duplicates logic that also exists in the batched self-play inner `exchange_search_request()`.
- Both paths implement:
  - read/collect/model/write duty cycle
  - deferred terminal handling
  - pipeline optionality
  - adaptive collect timeout
- This duplication raises maintenance cost and makes profiling less honest, because the same conceptual broker loop exists twice.
- `selfplay_rust_nn_batched()` still does substantial Python work per result:
  - sparse policy decode
  - encoder copies
  - trace dict construction
  - per-game state bookkeeping
- `wait_for_worker_progress()` is not expensive, but it uses polling and sleep by design.

Conclusion:
- The dominant Python orchestration cost sits in the SHM/broker exchange loop and per-result bookkeeping, not in planner helpers like `plan_selfplay_runner_chunk()`.

### 1.3 `evaluator_runtime.py`

Findings:
- `shm_eval_loop()` mirrors the self-play-side broker pattern:
  - scan slots
  - optional pipeline
  - collect/write loop
  - sleep-based idle backoff
- `play_match_tally_against()` is structurally sound, but it pays startup and session management tax per evaluation run.
- Same-model arena/eval is already optimized to reuse the same model tag and shared client, which is correct and important.

Conclusion:
- The heavy Python-side evaluator cost is still `shm_eval_loop()`, not the match bookkeeping itself.

### 1.4 `qipc.py`

Findings:
- Before this audit round, `ShmRingBuffer` used `ctypes.from_buffer` for repeated byte loads/stores.
- That showed up in profile as repeated low-value Python overhead.
- The SHM API is still polling-heavy:
  - repeated `r2p_try_read_meta()`
  - repeated `proc.poll()`
  - sleep-based backoff
- This is now the clearest remaining Python-side low-level bottleneck.

Conclusion:
- `qipc.py` is the correct place to continue if Python-side orchestration remains a priority.
- The next meaningful step is likely not more Python micro-tuning, but reducing polling/sleep structure itself.

### 1.5 `replay.py`

Findings:
- `collate_replay_samples()` is a real hotspot in replay-heavy workloads because it densifies sparse policies row by row.
- `build_dataloader()` and `_torch_module()` had unnecessary import/lazy-loader overhead before caching.
- Dense policy materialization remains the dominant replay-side Python cost even after optimization.

Conclusion:
- Replay is not the biggest system bottleneck overall, but it is the main Python-side training-loop hotspot once self-play is hidden behind Rust.

## 2. Profiling Results

### 2.1 Representative Self-Play Profile

Harness:
- `gomoku7`
- `cpu`
- `n_games=4`
- `parallel=4`
- `iters=16`

Latest summary:
- `selfplay_elapsed_s = 0.1425`

Top cumulative frames from [selfplay_profile.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_orchestrator_profiles_final/selfplay_profile.txt):
- `torch_training_runtime.selfplay_rust_nn_batched`
- `selfplay_runtime.selfplay_rust_nn_batched`
- `selfplay_runtime.exchange_search_request`
- `runtime_support._shm_eval_loop`
- `evaluator_runtime.shm_eval_loop`
- `runtime_support._run_model_batch`

Important observations:
- After the low-concurrency pipeline policy change, `InferencePipelineThread.collect()` and `queue.get()` dropped out of the top frames on CPU.
- Remaining dominant Python cost is now:
  - SHM polling loop
  - model forward itself
  - `time.sleep()` idle backoff
  - `subprocess.poll()`
  - `qipc.r2p_try_read_meta()`

### 2.2 Pipeline On vs Off

Before policy gating:
- `parallel=4`, CPU self-play was around `0.201s`

After policy gating:
- same harness dropped to `0.142s`

Interpretation:
- On CPU and low-concurrency settings, async pipeline threading was negative value.
- Keeping pipeline enabled only when the backend/device can actually overlap useful work is the correct default.

### 2.3 Representative Replay Profile

Harness:
- synthetic replay if streamed self-play emits no positions
- `batch_size=4`
- `n_steps=16`
- `replay_repeats=16`

Latest summary:
- `replay_elapsed_s = 0.0127`

Top cumulative frames from [replay_profile.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_orchestrator_profiles_final/replay_profile.txt):
- `ReplayBuffer.build_dataloader`
- `collate_replay_samples`
- `torch.from_numpy`
- `torch profiler record_function` wrappers

Important observations:
- `collate_replay_samples()` is still the main replay hotspot.
- Dense policy materialization is the real cost.
- Import/lazy-loader overhead shrank after module-level caching.

### 2.4 Evaluator Profile

Harness:
- same-model `RustNNEvaluatorEngine`
- `gomoku7`
- `num_games=2`

Result:
- [evaluator_profile.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_evaluator_profiles_smoke/evaluator_profile.txt)
- `elapsed_s = 0.0692`

Important observations:
- Evaluator path hits the same broker loop shape:
  - `play_match_tally_against`
  - `run_shared_session`
  - `open_search_session`
  - `_exchange_search_request`
  - `_shm_eval_loop`
- The Python evaluator orchestration is not fundamentally different from self-play; it shares the same polling and SHM structure.

## 3. Import-Time Audit

### 3.1 `quartz.alphazero_train`

From [tmp/python_orchestrator_importtime_alphazero.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_orchestrator_importtime_alphazero.txt):
- total import wall was about `1.384s`
- `torch` dominated import cost at roughly `1.234s`

Conclusion:
- Import-time startup cost is overwhelmingly framework startup, not local Python code.

### 3.2 `quartz.torch_training_runtime`

From [tmp/python_orchestrator_importtime_torch_runtime.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_orchestrator_importtime_torch_runtime.txt):
- total import wall was about `1.437s`
- again dominated by `torch`
- local modules like `runtime_support`, `evaluation`, `qipc`, `selfplay_runtime` were much smaller than Torch startup

Conclusion:
- Python import startup is real, but not the place to chase large wins unless avoiding Torch import entirely is possible for a path.

## 4. Tracemalloc Audit

From [tmp/python_orchestrator_tracemalloc.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_orchestrator_tracemalloc.txt):

Self-play top allocations:
- import/bootstrap noise
- multiprocessing synchronization objects
- eval runtime request-group bookkeeping
- some qipc payload handling

Replay delta top allocations:
- synthetic replay state creation from the harness
- `replay.py` sparse-policy conversion and metadata objects
- `collate_replay_samples()` dense arrays

Conclusion:
- The main Python heap pressure in replay comes from sparse-policy normalization and dense batch assembly, not from replay deque mechanics.

## 5. Changes Kept During This Audit

These were semantics-preserving changes kept after validation:

1. Async pipeline policy
   - new `should_use_async_pipeline()` in `runtime_support.py`
   - CPU / `batch_size <= 1` defaults to no async pipeline
   - override remains available with `QUARTZ_FORCE_ASYNC_PIPELINE=1`

2. Replay collation fast path
   - `ReplayExample` batches avoid extra wrapping
   - states/values are preallocated directly

3. Replay import caching
   - `_torch_module()` and `_data_loader_cls()` now cache loaded objects

4. SHM ring access cleanup
   - `ShmRingBuffer` now uses direct memoryview/`struct` access instead of repeated `ctypes.from_buffer`

5. Wider default SHM ring topology
   - Rust server launch now uses a wider default ring (`8x8` slots, env-overridable)
   - representative comparison showed `2x2` slots were materially worse than the wider default on the current orchestrator path

## 6. Validation

Targeted Python regressions passed:
- `13 passed`

Covered behaviors:
- replay dense/sparse roundtrip
- replay metadata preservation
- dataloader shape/type contract
- async pipeline policy behavior
- SHM frame decode/write roundtrip
- SHM ring epoch/seq/payload roundtrip
- self-play state-machine payload handling
- evaluator shared Rust path behavior
- sparse search-response decode

## 7. Finalization Audit

One additional experimental branch was tried after the main optimizations:
- more aggressive SHM idle-backoff and process-liveness throttling in `shm_eval_loop()`
- a Rust→Python ring-wakeup pipe so Python could wait for ring activity instead of pure slot scanning

Those branches were **fully reverted** because the representative harness regressed badly:
- self-play moved from the optimized `~0.14s-0.20s` range to about `~1.14s`
- evaluator smoke also regressed to about `~1.11s`
- the ring-wakeup experiment also regressed representative self-play/evaluator timings enough to fail the keep threshold

This matters because it closes off the most obvious remaining Python-side micro-tuning idea:
- the current polling/sleep structure is fragile
- naive backoff tweaks are easy to make worse
- even Rust-side wakeup hints are not automatically wins without a larger broker redesign
- the next meaningful improvement is architectural, not another tiny Python loop edit

Latest re-verification after reverting that branch:
- representative self-play rerun: `selfplay_elapsed_s = 0.2253`
- representative replay rerun: `replay_elapsed_s = 0.0090`
- targeted regressions: `13 passed`

Interpretation:
- replay remains improved and stable
- self-play remains materially better than the original pre-audit CPU path, but it is noisy enough that small delta chasing in Python is no longer trustworthy
- no additional production optimization was kept after this point

## 8. Final Findings

### What is no longer worth chasing in Python
- import micro-optimizations inside orchestrator modules
- more tiny replay object cleanups
- more queue-thread tuning on CPU now that low-concurrency pipeline is disabled
- more SHM sleep/backoff micro-tuning in Python

### What is still meaningfully hot
1. `evaluator_runtime.shm_eval_loop()` / `selfplay_runtime.exchange_search_request()`
   - slot scanning
   - sleep-based backoff
   - `proc.poll()`
   - SHM metadata reads

2. `runtime_support._run_model_batch()`
   - once Python broker tax is reduced, actual inference dominates

3. `replay.collate_replay_samples()`
   - dense policy materialization remains costly

## 8. Recommended Next Steps

Priority order:
1. Push SHM/broker wakeup logic closer to Rust or use a more blocking/event-driven handoff.
2. Reduce repeated `proc.poll()` and slot-scan churn in Python loops.
3. If replay becomes training-visible, consider a sparse-policy training path or a denser pack format that avoids full dense policy expansion in Python.

The main conclusion is simple:

**The Python orchestrator is no longer dominated by obvious micro-inefficiencies. The remaining large costs are the SHM polling architecture and the unavoidable model-forward path.**

Operational conclusion:

**The Python orchestrator optimization loop should be considered closed for now. Further meaningful wins should come from Rust-side broker/event-notification changes, not more Python micro-tuning.**
