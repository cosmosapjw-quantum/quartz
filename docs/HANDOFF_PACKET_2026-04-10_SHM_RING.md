# QUARTZ Handoff Packet — 2026-04-10 (SHM Ring Buffer)

## 1) Session Summary

This session progressed through:

1. **Phase 1a GlobalBroker** (completed): single process-wide eval broker replacing per-instance collector threads
2. **Bug fixes** (completed): GlobalBroker lifecycle, terminal chess eval, promotion significance
3. **Monitor upgrades** (completed): GPU merge, broker health, promotion audit, null metric split, duty-cycle instrumentation, regex fixes
4. **Rust worker pool** (completed, PR 1-5): multi-threaded job processing with GlobalInflightCredit (RAII permits)
5. **Python inference pipeline** (completed, PR 8-9): InferencePipelineThread + deadlock fix — works but **no throughput gain** due to broker stdin/stdout serialization
6. **Rust broker split-IO** (attempted, reverted): deadlocked because serve() and broker share stdin
7. **SHM ring buffer infrastructure** (completed, PR 11): data structures on both Rust and Python sides, integrated into launch_rust_server

## 2) What Works Now

- 338 Rust tests + 119 Python tests passing
- Worker pool active (multi-threaded job processing)
- GlobalInflightCredit with RAII CreditPermit
- Split null metrics (null_inactive_slot vs null_result_miss)
- DutyCycle instrumentation in both eval loop sites
- Monitor regex fixes (void false-positive, eval counter format)
- InferencePipelineThread (functional but no overlap due to broker serialization)
- ShmRingBuffer data structures created at launch, passed via env vars

## 3) What Failed and Why

### Broker split-IO deadlock (reverted)
- **Attempt**: Split broker_loop into writer thread (stdout) + reader thread (stdin)
- **Failure**: serve() main loop (mcts_server.rs:813) reads JSON commands from stdin; broker reader thread also reads QIPC responses from stdin → deadlock on stdin.lock()
- **Learning**: stdin/stdout are multiplexed between JSON commands and QIPC eval frames — cannot be partially separated

### Python pipeline no-overlap
- **Attempt**: InferencePipelineThread runs GPU inference while collector reads next batch
- **Failure**: Broker does write(stdout)→read(stdin) synchronously per batch, so Python's stdin has no pending data during inference — collector sleeps in proc_read_message()
- **DutyCycle proof**: model=55%, read+collect=45% — exactly the serial pattern, no overlap achieved
- **Learning**: True pipelining requires removing broker from stdin/stdout entirely

## 4) Current Primary Bottleneck

**stdin/stdout serialization**: All IPC between Rust and Python goes through a single pipe pair. serve() reads commands, broker reads/writes eval frames, and emit_json_message writes selfplay_chunk/progress — all on the same stdin/stdout.

Evidence from DutyCycle (15k cycles, eval phase):
- Per-cycle: 9.57ms total (model 4.92ms + IPC 4.65ms)
- GPU utilization: 42% (idle between batches)
- CPU: 0.68/24 cores effective
- Theoretical with zero IPC: 5ms/cycle → 1.9x speedup

## 5) Next Step: SHM Ring Buffer Integration (PR 12-13)

### What's Already Built (PR 11, merged)

**Rust** (`src/mcts/eval.rs`):
- `ShmRingBuffer` struct with `open()`, `epoch()`, `bump_epoch()`, `cmd_done()`, `set_cmd_done()`, `r2p_try_write()`, `r2p_reclaim()`, `p2r_try_read()`, `slot_payload_capacity()`
- Layout: 256-byte header + 2 r2p slots + 2 p2r slots, each ~4MB
- Atomic state machine: EMPTY(0) → WRITTEN(1) → DONE(2)
- Message types: EVAL_BATCH_REQ(1), EVAL_BATCH_RESP(2), JSON_MSG(3)

**Python** (`quartz/alphazero_train.py`):
- `ShmRingBuffer` class with `create()`, `open()`, `r2p_try_read()`, `r2p_mark_done()`, `p2r_try_write()`, `epoch()`, `cmd_done()`
- Created in `launch_rust_server()`, attached as `proc._quartz_ring_buffer`
- Env vars: `QUARTZ_QIPC_RING_SHM_NAME`, `QUARTZ_QIPC_RING_SHM_SIZE`

### PR 12: Rust Broker SHM Mode

**Goal**: When ring buffer is available, broker writes eval requests to r2p slots and reads responses from p2r slots — **never touches stdout/stdin**. Also: `emit_json_message` writes JSON to r2p slots.

Key changes:
1. New `broker_loop_shm()` in eval.rs — uses `ShmRingBuffer` instead of stdout/stdin
2. `emit_json_message()` in mcts_server.rs — writes to r2p slot with `SHM_MSG_JSON` type when ring available
3. `serve()` — bumps epoch at command start, sets cmd_done at command end
4. `GlobalBroker::new()` — loads ring buffer from env alongside existing SHM transport
5. **Shutdown handling**: all inflight slot batches get `send_uniform_fallback()`

Critical constraints:
- serve() still reads initial JSON command from stdin and writes final result to stdout
- Only hot-path messages (eval frames + selfplay_chunk/progress JSON) go through SHM ring
- Ring buffer has per-command epoch to prevent stale slot reads across commands
- Fallback to existing stdin/stdout path when ring buffer is not available

### PR 13: Python SHM Unified Eval Loop

**Goal**: Python reads eval requests AND JSON messages from r2p SHM slots, writes eval responses to p2r slots.

Key changes:
1. New `_shm_eval_loop(ring, model, device, cfg, proc, on_json=None)` function
2. Integration into **all 3 eval loop sites**:
   - `NNSearchClient.selfplay_run()` (line ~1800) — `on_json` handles selfplay_chunk/progress
   - `NNSearchClient._exchange_search_request()` (line ~1879) — no JSON interleaving
   - `selfplay_rust_nn_batched()` local `exchange_search_request()` (line ~5060) — `on_json` handles selfplay_chunk/progress
3. InferencePipelineThread integration — **now truly overlaps** because SHM polling doesn't block on stdin
4. Terminal detection: Rust sets `cmd_done=1`, Python exits SHM loop, reads final JSON from stdout

## 6) Review Findings That Must Be Addressed

1. **selfplay_chunk/progress interleaving**: SHM slots carry msg_type field — JSON_MSG(3) for non-eval messages
2. **done flag as sticky global**: Use epoch counter — reset at each command start; Python ignores stale-epoch slots
3. **Memory ordering**: Rust uses real AtomicU8/AtomicU32 on mmap'd memory; Python uses ctypes
4. **Shutdown fallback**: All inflight batches must get uniform fallback on shutdown/timeout
5. **Slot capacity**: Separate 16MB ring buffer (not splitting existing 8MB req/resp regions)
6. **All 3 eval loop sites**: selfplay_run(), _exchange_search_request(), selfplay_rust_nn_batched()

## 7) Files Modified in This Session

| File | Changes |
|------|---------|
| `src/mcts/eval.rs` | GlobalBroker, broker_loop (sync, reverted split-IO), GlobalInflightCredit, CreditPermit, ShmRingBuffer |
| `src/mcts_server.rs` | process_job_tick, adaptive_backoff, worker pool, split null metrics, credit tests |
| `quartz/alphazero_train.py` | Terminal chess fix, InferencePipelineThread, duty-cycle instrumentation, ShmRingBuffer, launch_rust_server ring buffer |
| `quartz/evaluation.py` | Promotion significance (scored = wins + 0.5*draws) |
| `scripts/profile_training_monitor.py` | GPU merge, broker health, promotion audit, null split, void regex fix, DutyCycle parsing, worker telemetry |

## 8) Test Commands

```bash
cargo test --release --quiet   # 338 pass
venv/bin/python -m pytest tests/ -q   # 119 pass
venv/bin/python -m quartz.train --game gomoku7 --iterations 5 --retune --search-profile quartz
```

## 9) Key Monitoring Baselines

| Metric | Pre-session (04/09) | Post-session (latest) |
|--------|--------------------|-----------------------|
| Wall time (20 iter) | 1760s | 1620s (-8%) |
| result_vs_queue ratio | 12.8 | 8.35 (-35%) |
| broker_result_wait_s | 103,569 | 75,132 (-27%) |
| CPU effective | 0.52 | 0.68 |
| GPU utilization | 42% | 42% (unchanged — blocked by IPC) |
| DutyCycle model% | n/a | 55% |
| null_result_miss | n/a | 0 |
| fallback_count | 0 | 0 |
