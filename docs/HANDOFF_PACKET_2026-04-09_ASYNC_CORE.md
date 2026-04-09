# QUARTZ Handoff Packet — 2026-04-09 (Async Core, latest)

## 1) Canonical reference run

- `artifacts/runtime_monitor/alphazero_gomoku7_20260409_231945/summary.json`
- command:
  - `venv/bin/python -m quartz.train --game gomoku7 --iterations 20 --retune --search-profile quartz`
- settings:
  - `runtime_tuner_enabled=false`
  - `eval_selfplay_isolated=true`
- status:
  - `returncode=0`
  - 20-iteration run completed

## 2) What succeeded

1. Evaluation correctness is stable in this run shape.
   - `valid_eval=true`, `games=200`, `errors=0`, `voids=0`
2. Replay starvation symptom is absent here.
   - `training_wait_total_s=0.0`
3. Async runners are active at scale.
   - `run_multi_async_batch_*`, `selfplay_runner_*`, `eval_runner_*` present
4. Batch fill is acceptable.
   - `mean_batch_weighted=15.688` (target 18)

## 3) What failed / remains incomplete

1. Effective CPU remains low.
   - training `cpu_thr_mean=0.523`
   - evaluation `cpu_thr_mean=0.633`
2. Result-side waiting dominates.
   - `result_wait_s=400905.907`
   - `queue_wait_s=31310.545`
   - ratio ≈ `12.8`
3. Global pending pressure is too high in large waves.
   - `max_active_waiters=1200`
4. True non-blocking pending/apply flow is not complete yet.

## 4) Current primary bottleneck

- **sync eval-handshake behavior inside async completion path, amplified by missing global inflight admission control**

Evidence:

1. Wait ratio is extremely skewed to result wait.
2. Codec cost is tiny relative to wait.
3. Batch fill is already decent.
4. Large-wave geometry (`jobs=200`, `max_inflight_per_job=6`) can create
   broker-level pending spikes.

## 5) Important interpretation correction

- `run_multi_async_batch_done.null_results` currently counts slot-alignment nulls
  for inactive slots.
- Therefore, aggregate `async_batch_null_results_sum` is not a pure failure
  signal and must be normalized before root-cause attribution.

## 6) Latest `/review` findings

1. Inflight launch policy in `run_multi_async_batch_tags` scales with job count
   and can over-queue waiters in large eval waves.
2. `null_results` metric currently mixes structural nulls with actual misses.
3. Previously reported chess terminal shared-eval and promotion significance
   inconsistencies are addressed in current code and are not current top blockers.

## 7) Highest-priority next tasks (non-heuristic)

1. Add global inflight credit scheduler across jobs.
2. Complete pending-eval/apply separation to reduce result-bound idle.
3. Split null metrics:
   - inactive-slot null
   - terminal-immediate
   - selection miss
   - eval response miss
4. Keep runtime tuner off for benchmark runs until completion path stabilizes.
5. Keep strict profile comparisons on same substrate:
   - `quartz`
   - `baseline`
   - `baseline_strict`

## 8) Full chat-window progress summary

This chat window progressed through:

1. shipped-path correctness fixes (installer quoting, MPS auto, play cfg, jax extra)
2. runtime hygiene hardening (runtime tuner off by default, eval isolation on)
3. async core primitives + async batch runner wiring
4. monitor/telemetry expansion with bottleneck reporting
5. repeated clean monitor runs and adversarial audits

Net state:

- system is now stable and measurable
- correctness regressions are mostly contained
- throughput remains capped by completion-side wait and inflight admission
- next breakthrough requires scheduler/flow redesign, not additional heuristic tuning

## 9) Recommended next run commands

Primary:

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
