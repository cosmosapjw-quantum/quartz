# Training Guide

For the current 2026 performance investigation history, failures, kept fixes,
and surviving benchmark artifacts, see
[`docs/PERFORMANCE_WORKLOG_2026-04.md`](./PERFORMANCE_WORKLOG_2026-04.md).

## Architecture: Rust MCTS + Python NN (sole training engine)

All training self-play uses the Rust MCTS engine with Python NN evaluation
via the bidirectional IPC protocol (`search_nn*`, `selfplay_nn_run`, `eval_nn_run`).

### Search Features (all active during training)

- Transposition table (Zobrist hashing)
- Virtual loss (adaptive split vvisit/vvalue via ParallelismController)
- Progressive widening
- QUARTZ controller (6 penalty modes)
- FEN tracking (chess)

### Self-Play / Eval Flow (current)

1. Python launches Rust server process(es).
2. Rust self-play/eval runners drive game/session loops.
3. Rust batches NN eval requests and exchanges responses through QIPC.
4. Python owns model forward/backend and learner bookkeeping.
5. Progress and bottleneck telemetry are emitted to monitor artifacts.

Training-time `search_nn` requests rebuild search from the current position.
Tree reuse via `advance_root()` exists in the server self-play path, but it is
not the current Rust+NN training path contract.

### Background Actor (--concurrent)

Main thread: training (SGD on replay buffer)
Background: SelfPlayWorker (Rust+NN, frozen model snapshot)
- Uses selfplay_rust_nn_batched(parallel=2)
- Backpressure: pauses at 80% replay capacity
- Model snapshot refreshed after each checkpoint
- Evaluation isolation (default): background self-play is paused during
  checkpoint evaluation and resumed after evaluation ends.

### Runtime tuning and isolation defaults

- Online runtime autotune is **off by default**.
  - Enable explicitly with `--runtime-autotune`.
- Evaluation/self-play isolation is **on by default**.
  - Disable explicitly with `--no-eval-selfplay-isolation`.
- Search profile options:
  - `quartz`
  - `baseline` (shared substrate)
  - `baseline_strict`

## Requirements

- Rust binary required: cargo build --release (training exits if not found)
- PyTorch: for NN model (training + evaluation)
- Recommended GPU: for NN forward pass throughput

## Legacy Components

- TreeMCTS: retained for arena evaluation (lightweight model comparison)
- selfplay_rust(): Tier 1 ShortRollout, internal function for search-only ablation (not exposed as standalone CLI)

## Checkpoint Evaluation

Training-time checkpoint evaluation uses RustNNEvaluatorEngine (same Rust+NN stack as training self-play)
when the Rust binary is available. This ensures evaluation semantics match training semantics.
Falls back to TreeMCTSEngine only when Rust binary is missing (with explicit warning).

The search_nn result payload includes controller telemetry (partial; core stats):
- p_flip, sigma_q, hbar_eff: QUARTZ controller state
- stop_reason: why search terminated (Budget/VOC/Threshold/ConfAdaptive)
- iterations: actual visit count
- dup_rate, max_pending, avg_vvalue: parallelism-controller telemetry

For current performance diagnosis and handoff status, see:

- [`docs/PERFORMANCE_WORKLOG_2026-04.md`](./PERFORMANCE_WORKLOG_2026-04.md)
- [`docs/HANDOFF_PACKET_2026-04-09_ASYNC_CORE.md`](./HANDOFF_PACKET_2026-04-09_ASYNC_CORE.md)
