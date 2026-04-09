# QUARTZ Handoff Packet — 2026-04-09 (Async Core Round)

## 1) Scope of this handoff

This packet summarizes:

- what was implemented in this round
- what was validated
- what is still failing / incomplete
- current dominant bottleneck
- exact next execution steps

Primary reference run for current state:

- `artifacts/runtime_monitor/gomoku7_async_core_full/summary.json`

---

## 2) Implemented changes (high signal)

### A. Shipped-path correctness fixes

1. GPU installer command parsing fixed (`shlex.split`):
   - `quartz/gpu_detect.py`
2. Play session now preserves checkpoint-tuned config:
   - `quartz/play_gui.py`
3. `--device auto` supports MPS on Apple Silicon:
   - `quartz/alphazero_train.py`
   - `quartz/play_gui.py`
4. `jax` extra made runnable with current train entrypoint (includes torch):
   - `pyproject.toml`

### B. Runtime hygiene defaults

1. Runtime autotune is opt-in:
   - default off, explicit `--runtime-autotune` required
2. Evaluation/self-play isolation is default on:
   - explicit `--no-eval-selfplay-isolation` to disable

### C. Async substrate progression

1. Added async iteration primitives in Rust core:
   - `src/mcts/mod.rs`
2. Added async eval ticket submission path:
   - `src/mcts/eval.rs`
3. Added async multi-job batch runner and wired it into:
   - `eval_nn_run` (`gomoku7`)
   - `selfplay_nn_run` (`gomoku7`)
   - `src/mcts_server.rs`

### D. Monitor and telemetry upgrades

1. Added command setting capture:
   - runtime tuner on/off
   - eval isolation on/off
   - search profile
2. Added async batch/stage summaries:
   - async runs/jobs/null-results/max inflight
   - selfplay/eval runner done durations
   - wave elapsed/frontier/active-game aggregates
3. Added bottleneck report expansion:
   - async underfeed/null-result/flush timeout signals

---

## 3) Validation status

### Test status

- `cargo build --release`: pass
- `cargo test --quiet mcts::eval::tests`: pass
- `venv/bin/python -m pytest -q tests/test_training_pipeline_regressions.py tests/test_evaluation_pipeline_regressions.py`: pass (`107 passed`)

### Smoke status

- eval deadlock/debug smoke: pass
  - `artifacts/eval_deadlock_debug/gomoku7_async_ticket_smoke/summary.json`

### Full monitor run status

- 20-iteration run completed, return code `0`
  - `artifacts/runtime_monitor/gomoku7_async_core_full/summary.json`

---

## 4) What succeeded vs failed

### Succeeded

1. Evaluation correctness now stable in clean run:
   - `valid_eval=true`, `games=200`, `errors=0`, `voids=0`
2. Replay starvation symptom removed in this run shape:
   - `training_wait_total_s=0.0`
3. Async runner events active at scale:
   - `run_multi_async_batch_*`, `selfplay_runner_*`, `eval_runner_*` all present
4. Batch fill is generally healthy:
   - weighted mean batch `~15.6` (target `18`)

### Failed / still incomplete

1. Effective CPU usage still low:
   - training `cpu_thr_mean=0.517`
   - evaluation `cpu_thr_mean=0.628`
2. Result-side waiting still dominates:
   - `result_wait_s=317276.322`
   - `queue_wait_s=26747.628`
3. Async completion null-result churn still high:
   - `async_batch_null_results_sum=5037`
4. True non-blocking pending/apply execution is not fully achieved yet.

---

## 5) Current primary bottleneck (detailed)

Current top bottleneck remains:

- **sync eval-handshake behavior inside async substrate completion path**

Evidence:

1. Wait ratios:
   - `result_wait_s / queue_wait_s ≈ 11.86`
2. Transport not primary:
   - `io_time_s=1199.157` vs `codec_time_s=1.907`
3. Batch fill not primary:
   - target-batch flush count dominates timeout flush count
4. Lock contention not evidenced as primary in this run:
   - TT/edge lock aggregates are not dominant signals

Interpretation:

- We now have async submission and good batching,
- but completion/application path still behaves too wait-bound.

---

## 6) Highest-priority next tasks

1. Reduce null-result churn in `run_multi_async_batch_done` path.
2. Complete pending-eval/apply separation to avoid result-bound idle.
3. Add explicit null-result cause telemetry tags (per batch run):
   - no legal
   - terminal race
   - selection miss
   - eval response miss
4. Keep runtime tuner off for benchmark runs until completion path stabilizes.
5. Keep strict profile comparison on same substrate:
   - `quartz`
   - `baseline`
   - `baseline_strict`

---

## 7) Recommended next run commands

Primary benchmark:

```bash
venv/bin/python scripts/profile_training_monitor.py \
  --model-dir models/alphazero_gomoku7 \
  --output-dir artifacts/runtime_monitor/gomoku7_async_core_full_next \
  --interval-s 0.5 \
  --run "venv/bin/python -m quartz.train --game gomoku7 --iterations 20 --retune --search-profile quartz" \
  --print-summary
```

Strict baseline comparison:

```bash
venv/bin/python scripts/profile_training_monitor.py \
  --model-dir models/alphazero_gomoku7 \
  --output-dir artifacts/runtime_monitor/gomoku7_async_core_baseline_strict \
  --interval-s 0.5 \
  --run "venv/bin/python -m quartz.train --game gomoku7 --iterations 20 --retune --search-profile baseline_strict" \
  --print-summary
```

---

## 8) Quick chat-window summary

This window moved from broad tuning to architecture-first remediation:

1. fixed shipped-path correctness regressions
2. turned off unstable runtime heuristic control for clean measurements
3. added async core primitives and async batch runner wiring
4. upgraded monitor to report async-stage bottleneck signals
5. validated correctness and stability on full 20-iteration run
6. confirmed the remaining bottleneck is completion-side wait/null-result churn

Current status:

- stable and measurable
- not yet throughput-optimal
- next gains require completion-path redesign, not more heuristic tuning
