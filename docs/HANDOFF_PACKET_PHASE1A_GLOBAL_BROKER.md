# QUARTZ Phase 1a Handoff: Global Inference Broker

## 1) What this task is

Replace the current per-instance `BatchStdioEval` collector thread with a
**single GlobalBroker** that owns all eval I/O. This is the last remaining
Tier-1 bottleneck (B1 + B5) from the adversarial audit.

### Why it matters

Current architecture: Each `BatchStdioEval` instance spawns its own
`collector_loop` thread. Even with `new_shared_pair()`, there is still only
one collector per Rust server process, but the design couples batch assembly
to a single consumer thread that:

- Locks stdout for every batch write
- Locks stdin for every response read
- Uses FIFO queue with no per-job priority
- Has no way to coordinate across self-play and evaluation workloads

The result: `result_wait_s / queue_wait_s ≈ 11.86x` — workers spend 12x
longer waiting for the collector to finish I/O than waiting for the queue to
accept their request.

---

## 2) Current architecture (what exists now)

### Key types in `src/mcts/eval.rs`

```
BatchStdioEval<M>
  └── shared: Arc<BatchEvalShared<M>>
        ├── request_tx: channel::Sender<BatchRequest<M>>  (bounded, cap = max_batch_size * 2)
        ├── shutdown: Arc<AtomicBool>
        ├── io_handle: Mutex<Option<JoinHandle>>           (collector thread)
        └── stats: Arc<BatchBrokerStats>

AsyncEvalTicket<M>
  ├── result_rx: channel::Receiver<EvalResult<M>>          (bounded(1), per-request)
  ├── legal_moves: Vec<M>
  ├── wait_started_at: Instant
  └── stats: Arc<BatchBrokerStats>

BatchRequest<M>
  ├── features: Vec<f32>
  ├── legal_moves_idx: Vec<(M, usize)>
  ├── n_actions: usize
  ├── model_tag: u32
  ├── enqueued_at: Instant
  └── result_tx: channel::Sender<EvalResult<M>>
```

### Data flow

```
MCTS thread  ──submit()──►  request_tx  ──►  collector_loop
                                                  │
                                            encode_batch_eval_req_payload()
                                                  │
                                            write QIPC frame → stdout → Python
                                            read QIPC frame ← stdin ← Python
                                                  │
                                            distribute_binary_batch()
                                                  │
                                            result_tx.send() per request
                                                  │
MCTS thread  ◄──try_take()──  result_rx  ◄────────┘
```

### QIPC protocol (binary framing on stdin/stdout)

```
Frame: [MAGIC: 4B "QIPC"][KIND: 1B][LEN: 4B LE][PAYLOAD: LEN bytes]

Kinds:
  BATCH_EVAL_REQ     = 3   (Rust → Python: batch of encoded states)
  BATCH_EVAL_RESP    = 4   (Python → Rust: batch of (policy, value))
  BATCH_EVAL_REQ_SHM = 7   (SHM variant: payload is 4-byte length, data in SHM)
  BATCH_EVAL_RESP_SHM = 8  (SHM variant)

Request payload: [batch_size: u32][per item: model_tag:u32, n_actions:u32, feat_len:u32, features:f32*]
Response payload: [batch_size: u32][per item: policy_len:u32, probs:f32*, value:f32]
```

### Key file locations

| File | Lines | What |
|------|-------|------|
| `src/mcts/eval.rs:942-956` | `BatchConfig` | max_batch_size, timeout_us |
| `src/mcts/eval.rs:959-967` | `BatchRequest` | per-request data |
| `src/mcts/eval.rs:970-1055` | `BatchBrokerStats` | telemetry counters |
| `src/mcts/eval.rs:1106-1214` | `BatchEvalShared` | shared state + constructor |
| `src/mcts/eval.rs:1113-1175` | `AsyncEvalTicket` | ticket + try_take/recv_blocking |
| `src/mcts/eval.rs:1178-1242` | `BatchStdioEval` | public API: new, new_shared_pair |
| `src/mcts/eval.rs:1244-1486` | `collector_loop` | THE MAIN TARGET for replacement |
| `src/mcts/eval.rs:1488-1524` | `distribute_binary_batch` | response → per-request channels |
| `src/mcts/eval.rs:1616-1662` | `submit()` | request creation + channel send |
| `src/mcts/eval.rs:1665-1692` | `Clone + Drop` | lifecycle management |
| `src/mcts_server.rs:1244-1289` | `make_eval_pair` | factory for evaluators |
| `src/mcts_server.rs:1385-1524` | `run_multi_async_batch_tags` | async job loop (consumer) |

### Shared memory transport

`SharedMemTransport` (eval.rs:450-485) loads SHM regions from env vars
(`QUARTZ_QIPC_REQ_SHM_NAME`, `QUARTZ_QIPC_RESP_SHM_NAME`) set by
Python's `launch_rust_server()` in `alphazero_train.py:560-565`.

When SHM is available, the collector writes only a 4-byte length to
stdout/stdin and the actual payload goes through SHM. The GlobalBroker
MUST preserve this optimization.

---

## 3) Design for GlobalBroker

### Core idea

Instead of `BatchEvalShared` spawning a `collector_loop` per instance,
introduce a **single `GlobalBroker`** that:

1. Owns stdin/stdout I/O exclusively
2. Receives requests from ALL `BatchStdioEval` instances via one shared channel
3. Assembles batches with **priority awareness** (optional: deadline-based)
4. Performs I/O (write request, read response)
5. Distributes results back via per-request `result_tx` channels

### Struct sketch

```rust
pub struct GlobalBroker<M: Copy + Eq + Hash + Debug + Send + 'static> {
    shared: Arc<GlobalBrokerShared<M>>,
}

struct GlobalBrokerShared<M: ...> {
    request_tx: channel::Sender<BatchRequest<M>>,
    shutdown: Arc<AtomicBool>,
    io_handle: Mutex<Option<JoinHandle<()>>>,
    stats: Arc<BatchBrokerStats>,
}
```

### Changes needed

1. **`GlobalBroker::new(config)`** — creates one broker per Rust server process
   - Spawns exactly one collector thread (owns stdin/stdout)
   - Stores `Arc<GlobalBrokerShared>` for sharing

2. **`BatchStdioEval`** becomes a thin wrapper:
   ```rust
   pub struct BatchStdioEval<M> {
       broker: Arc<GlobalBrokerShared<M>>,
       model_tag: u32,
   }
   ```
   - `submit()` sends to `broker.request_tx` (unchanged API)
   - `new()` and `new_shared_pair()` accept a `&GlobalBroker` reference
   - Drop no longer shuts down the broker (broker outlives all evaluators)

3. **`make_eval_pair()`** in `mcts_server.rs`:
   - Accepts a `&GlobalBroker` parameter
   - Creates `BatchStdioEval` instances that reference the shared broker
   - No more per-instance collector threads

4. **Broker lifecycle in `serve()`** (`mcts_server.rs`):
   - Create one `GlobalBroker` at server startup
   - Pass to all `make_eval_pair()` calls
   - Shut down on `quit` command

### What MUST NOT change

- `AsyncEvalTicket` API (try_take, recv_blocking)
- QIPC frame format (Python compatibility)
- SHM transport behavior
- `BatchRequest` and `EvalResult` types
- `BatchBrokerStats` telemetry schema
- Per-request result channels (bounded(1))
- `Evaluator<G>` trait implementation on `BatchStdioEval`

### What CAN change

- `BatchEvalShared` → replaced by `GlobalBrokerShared`
- `collector_loop` → moved to `GlobalBroker::broker_loop` (same logic, new owner)
- `BatchStdioEval::new()` → takes `&GlobalBroker` instead of spawning thread
- Drop behavior → evaluators don't own the broker thread

---

## 4) Implementation steps

### Step 1: Extract broker_loop from collector_loop

Move the body of `BatchStdioEval::collector_loop()` (eval.rs:1244-1486) into
a standalone function `fn broker_loop<M>(...)` with the same signature.
Verify: all tests pass (logic unchanged, just moved).

### Step 2: Create GlobalBroker struct

Add `GlobalBroker<M>` and `GlobalBrokerShared<M>` structs.
`GlobalBroker::new()` spawns one thread running `broker_loop`.
Verify: compiles, not yet used.

### Step 3: Refactor BatchStdioEval to use GlobalBroker

Change `BatchStdioEval.shared` from `Arc<BatchEvalShared>` to
`Arc<GlobalBrokerShared>`. Update `new()`, `new_shared_pair()`, `submit()`,
Clone, and Drop.
Verify: all eval.rs tests pass.

### Step 4: Wire GlobalBroker into mcts_server.rs

- Create broker in `serve()` function (one per process)
- Pass to `make_eval_pair()` and `run_multi_async_batch_tags()`
- Remove per-instance collector thread spawning
Verify: `cargo test --release -- mcts_server::tests`

### Step 5: Validate end-to-end

```bash
cargo test --release --quiet
venv/bin/python -m pytest tests/ -q
venv/bin/python -m quartz.train --game gomoku7 --iterations 2 --retune
```

---

## 5) Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Broker thread crash kills all evals | Keep existing uniform-fallback pattern on I/O error |
| Stdin/stdout lifetime issues | Broker thread owns both; same as current collector |
| SHM transport race | Single broker thread = no race (same as now) |
| Python-side protocol change | None needed — QIPC frames identical |
| Performance regression | Measure `result_wait_s / queue_wait_s` ratio before/after |

---

## 6) Test plan

1. `cargo test --release` — 335+ tests pass
2. `pytest tests/` — 107+ tests pass
3. Smoke: `venv/bin/python -m quartz.train --game gomoku7 --iterations 2 --retune --search-profile quartz`
4. Monitor run: compare `result_wait_s / queue_wait_s` ratio (target: < 8.0, current: 11.86)
5. Ablation integrity: `--search-profile baseline` on same infrastructure

---

## 7) Already completed in this session (DO NOT redo)

The following changes are already applied and tested (335 Rust + 107 Python tests passing):

- **B6**: `edges: Mutex` → `RwLock` (14 files)
- **B7**: `atomic_f64_add` CAS backoff (node.rs)
- **B9**: Results per-slot `Mutex` instead of global (mcts_server.rs, 2 functions)
- **B11**: Depth-extended QUARTZ scoring, depth≤3 blending (select.rs)
- **B12**: TT eviction, MAX_ENTRIES_PER_BUCKET=4096 (tt.rs)
- **B13**: PW depth-aware narrowing (select.rs)
- **B14**: ENVAR_CONST scaling O(1/√N) (quartz.rs)
- **B15**: Welford race fix, local w computation (backup.rs)
- **B18**: TimeManager records actual elapsed_ms (search.rs)
- **B20**: IPC resync, JSON parse protection (alphazero_train.py)
- **B2**: multi_job_execution_plan hard thread cap (mcts_server.rs)
- **B3**: Load shedding at 75% aggregate pending + ImmediateReason telemetry (mcts_server.rs, mod.rs)
- **B1 partial**: Adaptive idle backoff spin→yield→50µs→200µs (mcts_server.rs)
