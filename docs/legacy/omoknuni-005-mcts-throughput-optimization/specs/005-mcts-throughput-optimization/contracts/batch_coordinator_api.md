# API Contract: Batch Inference Coordinator

**Feature**: MCTS Throughput Optimization
**Component**: Batch Inference Coordinator (Phase 1C)
**File**: `cpp_extensions/mcts/batch_inference_coordinator.{hpp,cpp}`
**Reference**: [plan.md](../plan.md#phase-1c-coordinator-simplification)

---

## Overview

This contract defines the simplified API for `BatchInferenceCoordinator` after eliminating state cloning and feature extraction. The coordinator's role is reduced to:
1. Collecting pre-extracted features from queue
2. Assembling them into contiguous batches
3. Invoking Python neural network inference
4. Distributing results back to tree nodes

**Key Changes from Baseline**:
- Remove all state cloning and feature extraction logic
- Accept pre-extracted features from `AsyncInferenceQueue`
- Zero memory allocations in hot path (pre-reserve all buffers)
- Support condition variable wake-up (eliminate polling)

---

## Interface Specification

### BatchInferenceCoordinator Class

**Purpose**: Coordinate neural network inference for batches of MCTS nodes with zero-copy feature aggregation

**C++ API**:
```cpp
class BatchInferenceCoordinator {
public:
    // Constructor
    // PRECONDITION: callback must remain valid for coordinator lifetime
    // PRECONDITION: max_batch_size > 0 && max_batch_size <= 256
    // PRECONDITION: max_action_space > 0 && max_action_space <= 512
    // PRECONDITION: min_batch_size > 0 && min_batch_size <= max_batch_size
    // POSTCONDITION: All internal buffers pre-allocated (batch_policy_ sized for worst-case)
    explicit BatchInferenceCoordinator(
        PyInferenceCallback* callback,
        AsyncInferenceQueue* queue,
        int max_batch_size = 64,
        int max_action_space = 512,
        int min_batch_size = 16,
        std::chrono::microseconds batch_timeout = std::chrono::microseconds(500)
    );

    // MODIFIED: Run coordinator loop (blocks until shutdown)
    // THREAD-SAFE: Should run in dedicated coordinator thread
    // EFFECTS: Repeatedly collects batches, runs inference, distributes results
    // RETURNS: When queue signals shutdown
    void run();

    // NEW: Shutdown coordinator gracefully
    // THREAD-SAFE: Called by main thread during teardown
    // EFFECTS: Signals queue shutdown, wakes coordinator, waits for exit
    void shutdown();

    // EXISTING: Get performance metrics (for profiling)
    // THREAD-SAFE: Called by telemetry threads
    // RETURNS: Snapshot of current metrics
    CoordinatorMetrics get_metrics() const;

private:
    // Core loop iteration (internal)
    // EFFECTS: Wait for requests → dequeue batch → run inference → distribute
    // RETURNS: False if shutdown requested, true to continue
    bool run_iteration();

    // MODIFIED: Collect batch from queue (zero allocation)
    // USES: AsyncInferenceQueue::dequeue_batch_blocking() with encapsulated lock management
    // PRECONDITION: batch_requests_ is empty (cleared from previous iteration)
    // POSTCONDITION: batch_requests_ contains up to max_batch_size requests
    // RETURNS: Number of requests collected (0 if timeout or shutdown)
    size_t collect_batch();

    // MODIFIED: Run inference on collected batch (zero copy via pinned tensor bridge)
    // EXECUTES: On per-coordinator CUDA stream (NO synchronize() in hot path)
    // PRECONDITION: batch_requests_.size() > 0
    // POSTCONDITION: batch_policy_ and batch_value_ filled with inference results
    // EFFECTS: Calls callback->infer_batch() with pinned CPU→GPU tensors + event handoff
    void run_inference();

    // MODIFIED: Distribute inference results to tree nodes (atomic updates)
    // OPERATIONS: expand_if_unexpanded() + backup() + remove_virtual_loss()
    // PRECONDITION: run_inference() completed successfully
    // POSTCONDITION: Tree nodes updated with policy/value, virtual loss removed
    // EFFECTS: Calls node->expand(), node->backup() with thread-safe atomics
    void distribute_results();

    // Dependencies (non-owned)
    PyInferenceCallback* callback_;      // Python neural network bridge
    AsyncInferenceQueue* queue_;         // Inference request queue

    // Configuration
    int max_batch_size_;                 // Maximum batch size (tunable, default: 64)
    int max_action_space_;               // Maximum action space (512 for Go/Chess/Gomoku)
    int min_batch_size_;                 // Minimum batch size before timeout (default: 16)
    std::chrono::microseconds batch_timeout_;  // Batch collection timeout (500μs)

    // Pre-allocated buffers (Phase 1C optimization)
    std::vector<InferenceRequest> batch_requests_;  // Current batch
    std::vector<float> batch_policy_;    // Policy outputs (max_batch × max_action_space)
    std::vector<float> batch_value_;     // Value outputs (max_batch × 1)

    // Tree reference (for result distribution)
    MctsTree* tree_ = nullptr;           // MCTS tree (set after construction via set_tree())

    // Metrics
    mutable std::mutex metrics_mutex_;   // Protects metrics_
    CoordinatorMetrics metrics_;         // Performance counters
};
```

**Stream Handoff Contract** (NO `synchronize()` in hot path):

The coordinator uses a dedicated CUDA stream for asynchronous H2D transfer and inference execution:

1. **Tensor Creation**: `callback->create_batch_tensor()` returns `(gpu_tensor, event, stream)` tuple
   - Pinned CPU buffer → GPU buffer via `copy_(non_blocking=True)` on the stream
   - Event recorded after H2D transfer completes
   - NO `stream.synchronize()` call (defeats async overlap)

2. **Inference Execution**: Model forward runs on the same stream or waits on event
   - **Option A** (simplest): `with torch.cuda.stream(stream): model(gpu_tensor)`
   - **Option B** (multi-stream): `torch.cuda.current_stream().wait_event(event); model(gpu_tensor)`
   - Both avoid explicit `synchronize()` in hot path

3. **Result Retrieval**: Results automatically synchronized when accessed on CPU
   - `policy.cpu()` implicitly waits for GPU work to complete
   - Or use events for explicit synchronization if needed for metrics

**Pinned Memory vs DLPack**:

- **Current Approach (Pinned Tensor Bridge)**:
  - Pre-allocated pinned CPU buffer (3.3MB default for batch=64, 36 planes, 19×19)
  - Pre-allocated GPU buffer (same size)
  - Feature data copied from C++ vectors → pinned buffer → GPU buffer
  - Buffer reuse (zero allocation per batch)
  - Lifetime: Managed by Python bridge object (RAII)

- **DLPack Alternative** (not implemented):
  - Would provide zero-copy view of C++ `std::vector<float>` data
  - Requires capsule with correct deleter to prevent use-after-free
  - C++ vector must outlive DLPack capsule (complex lifetime management)
  - Still needs copy to GPU (DLPack doesn't avoid H2D transfer)
  - **Not used**: Pinned buffer approach is simpler and equally fast

**Rationale**: Pinned buffer + reuse is simpler, safer (no lifetime issues), and provides same performance as DLPack (both require one H2D copy). The `DLPackInferenceBridge` naming is historical; functionally it's a pinned tensor bridge.

---

## Data Structures

### CoordinatorMetrics

**Purpose**: Track coordinator performance for profiling and tuning

**C++ Definition**:
```cpp
struct CoordinatorMetrics {
    // Throughput metrics
    uint64_t total_batches = 0;          // Total batches processed
    uint64_t total_requests = 0;         // Total requests processed
    uint64_t total_iterations = 0;       // Total run_iteration() calls

    // Batch size distribution
    double avg_batch_size = 0.0;         // Mean requests per batch
    uint32_t max_batch_size_seen = 0;    // Maximum batch size observed
    uint32_t min_batch_size_seen = 0;    // Minimum batch size observed

    // Timing metrics (microseconds)
    double avg_collect_us = 0.0;         // Mean time to collect batch
    double avg_inference_us = 0.0;       // Mean time for inference
    double avg_distribute_us = 0.0;      // Mean time to distribute results
    double avg_iteration_us = 0.0;       // Mean total iteration time

    // Wait metrics
    uint64_t timeout_count = 0;          // Number of timeouts (no requests)
    uint64_t wakeup_count = 0;           // Number of condition variable wake-ups
    double avg_wait_us = 0.0;            // Mean time waiting for requests

    // Error tracking
    uint64_t inference_errors = 0;       // Number of failed inference calls
    uint64_t distribution_errors = 0;    // Number of failed result distributions
};
```

---

## Method Contracts

### Constructor

**Signature**:
```cpp
BatchInferenceCoordinator(
    PyInferenceCallback* callback,
    AsyncInferenceQueue* queue,
    int max_batch_size = 64,
    std::chrono::microseconds batch_timeout = std::chrono::microseconds(500)
);
```

**Preconditions**:
- `callback != nullptr`
- `queue != nullptr`
- `max_batch_size > 0 && max_batch_size <= 256`
- `batch_timeout > 0`

**Effects**:
1. Stores references to `callback` and `queue`
2. Pre-allocates internal buffers:
   ```cpp
   batch_requests_.reserve(max_batch_size);
   batch_policy_.reserve(max_batch_size * 512);  // Max action space
   batch_value_.reserve(max_batch_size);
   ```

**Postconditions**:
- All internal buffers have capacity >= `max_batch_size`
- Metrics initialized to zero

**Performance**:
- One-time allocation overhead: ~1-2ms
- Zero allocations after construction

---

### run

**Signature**:
```cpp
void run();
```

**Preconditions**: None

**Effects**:
1. Enters infinite loop calling `run_iteration()`
2. Exits when `run_iteration()` returns `false` (shutdown signal)

**Postconditions**:
- Coordinator thread exits gracefully
- All pending requests processed before exit

**Thread Safety**:
- Should run in dedicated thread (NOT safe for concurrent calls)

**Performance**:
- Blocks indefinitely until shutdown
- Per-iteration latency: <1ms (target)

---

### run_iteration (Private)

**Signature**:
```cpp
bool run_iteration();
```

**Preconditions**: None

**Effects**:
1. **Collect Phase**:
   ```cpp
   size_t count = collect_batch();  // Blocks until timeout or batch ready
   if (count == 0) {
       if (queue_->is_shutdown()) return false;  // Exit signal
       return true;  // Timeout, retry
   }
   ```

2. **Inference Phase**:
   ```cpp
   run_inference();  // Call Python neural network
   ```

3. **Distribution Phase**:
   ```cpp
   distribute_results();  // Update tree nodes
   ```

4. **Cleanup**:
   ```cpp
   batch_requests_.clear();  // Prepare for next batch (no deallocation)
   ```

**Postconditions**:
- Tree nodes updated with inference results
- Virtual loss removed from expanded nodes
- Metrics updated

**Returns**:
- `false`: Shutdown requested, exit loop
- `true`: Continue to next iteration

**Performance**:
- Target latency: <1ms per iteration (for batch of 64)
- Breakdown: collect (~10μs) + inference (~300-500μs) + distribute (~50μs)

---

### collect_batch (Private)

**Signature**:
```cpp
size_t collect_batch();
```

**Preconditions**:
- `batch_requests_` is empty (cleared from previous iteration)

**Effects**:
1. Acquire queue lock:
   ```cpp
   std::unique_lock<std::mutex> lock(queue_->mutex_);
   ```

2. Wait for requests with timeout:
   ```cpp
   bool got_request = queue_->wait_for_request(lock, batch_timeout_);
   if (!got_request) {
       metrics_.timeout_count++;
       return 0;  // Timeout or shutdown
   }
   metrics_.wakeup_count++;
   ```

3. Dequeue batch (move requests):
   ```cpp
   size_t count = queue_->dequeue_batch(batch_requests_, max_batch_size_, lock);
   lock.unlock();  // Release lock immediately
   ```

4. Update metrics:
   ```cpp
   metrics_.total_requests += count;
   metrics_.avg_batch_size = update_running_average(metrics_.avg_batch_size, count);
   ```

**Postconditions**:
- `batch_requests_.size() == count`
- `count ∈ [0, max_batch_size_]`
- Queue lock is released

**Returns**: Number of requests collected

**Performance**:
- Target latency: <10μs (lock acquisition + dequeue)
- Zero memory allocations (move only)

---

### run_inference (Private)

**Signature**:
```cpp
void run_inference();
```

**Preconditions**:
- `batch_requests_.size() > 0`
- All requests have valid feature buffers

**Effects**:
1. **Build batch tensor** via pinned tensor bridge (Phase 2):
   ```cpp
   // Python callback returns (gpu_tensor, event, stream) tuple
   // NO synchronize() call - maintains async overlap
   auto [batch_tensor, xfer_event, xfer_stream] = callback_->create_batch_tensor(batch_requests_);
   ```
   - Features copied from C++ vectors → pinned CPU buffer → GPU buffer
   - H2D transfer uses `non_blocking=True` on dedicated stream
   - Event recorded after transfer completes
   - Pre-allocated buffers (zero allocation per batch)

2. **Run inference** on coordinator's CUDA stream:
   ```cpp
   // Option A: Execute on same stream (simplest, recommended)
   with torch.cuda.stream(xfer_stream):
       with torch.cuda.amp.autocast(enabled=use_fp16):
           policy, value = model(batch_tensor)

   // Option B: Wait on event if using different stream
   torch.cuda.current_stream().wait_event(xfer_event)
   policy, value = model(batch_tensor)
   ```
   - NO explicit `synchronize()` in hot path
   - FP16 autocast for Tensor Core utilization

3. **Copy results** to pre-allocated buffers:
   ```cpp
   // Results implicitly synchronized when accessed on CPU
   copy_policy_to_buffer(policy.cpu(), batch_policy_);  // memcpy
   copy_value_to_buffer(value.cpu(), batch_value_);     // memcpy
   ```

**Postconditions**:
- `batch_policy_` contains policy outputs (batch_size × max_action_space)
- `batch_value_` contains value outputs (batch_size × 1)
- GPU work may still be in flight (synchronized via implicit wait on .cpu())

**Thread Safety**:
- Re-acquires GIL for Python callback
- Releases GIL after results copied
- Stream-ordered operations prevent race conditions

**Error Handling**:
- On inference error: Log error, increment `metrics_.inference_errors`, continue
- Invalid results: Use fallback uniform policy + value=0.0
- Stream errors: Fall back to synchronous path with warning

**Performance**:
- **Target latency**: <500μs for batch=64 (p95, Phase 2 optimizations)
- **Breakdown** (p95 targets):
  - Tensor creation (Python-side): ≤2.0ms (pinned buffer + H2D)
  - GPU inference kernel: Model-dependent (baseline + 20% headroom)
  - Result copy: ~50μs (small buffer, already on CPU after .cpu())
- **Total coordinator iteration**: Target <3ms (p95) including collection + inference + distribution

---

### distribute_results (Private)

**Signature**:
```cpp
void distribute_results();
```

**Preconditions**:
- `run_inference()` completed successfully
- `batch_policy_` and `batch_value_` filled with valid results

**Effects**:
For each request in `batch_requests_`:
1. Extract policy/value slice:
   ```cpp
   int32_t node_idx = batch_requests_[i].node_index;
   int32_t action_count = batch_requests_[i].action_space_size;

   float* policy_slice = batch_policy_.data() + (i * max_action_space);
   float value = batch_value_[i];
   ```

2. Update tree node (atomic operations):
   ```cpp
   Node* node = tree_->get_node(node_idx);
   node->expand(policy_slice, action_count);  // Set prior probabilities
   node->backup(value);                       // Backpropagate value
   node->remove_virtual_loss();               // Allow re-selection
   ```

**Postconditions**:
- All nodes in batch are expanded and backed up
- Virtual loss removed (nodes available for selection)

**Thread Safety**:
- Uses atomic operations for visit counts and values
- No lock required (lock-free tree operations)

**Error Handling**:
- If node already expanded: Skip expansion, still backup value
- If node invalid: Increment `metrics_.distribution_errors`, continue

**Performance**:
- Target latency: <50μs for batch of 64
- Breakdown: <1μs per node (pointer lookup + atomic ops)

---

## Usage Example

### Main Thread: Coordinator Lifecycle

```cpp
int main() {
    // Initialize dependencies
    PyInferenceCallback callback(neural_network);
    AsyncInferenceQueue queue(4096);

    // Create coordinator
    BatchInferenceCoordinator coordinator(
        &callback,
        &queue,
        64,                                      // max batch size
        std::chrono::microseconds(500)           // 500μs timeout
    );

    // Start coordinator thread
    std::thread coordinator_thread([&]() {
        coordinator.run();  // Blocks until shutdown
    });

    // Run MCTS search in simulation threads
    run_mcts_search(800);  // 800 simulations

    // Shutdown coordinator
    coordinator.shutdown();
    coordinator_thread.join();

    // Print metrics
    auto metrics = coordinator.get_metrics();
    std::cout << "Total batches: " << metrics.total_batches << "\n";
    std::cout << "Avg batch size: " << metrics.avg_batch_size << "\n";
    std::cout << "Avg iteration: " << metrics.avg_iteration_us << "μs\n";
}
```

### Coordinator Thread: Internal Loop

```cpp
void BatchInferenceCoordinator::run() {
    while (true) {
        bool should_continue = run_iteration();
        if (!should_continue) break;  // Shutdown signal
    }
}

bool BatchInferenceCoordinator::run_iteration() {
    auto start = std::chrono::high_resolution_clock::now();

    // Phase 1: Collect batch (blocks until timeout or batch ready)
    size_t count = collect_batch();
    if (count == 0) {
        if (queue_->is_shutdown()) return false;  // Exit
        return true;  // Timeout, retry
    }

    // Phase 2: Run inference (re-acquire GIL)
    run_inference();

    // Phase 3: Distribute results (lock-free)
    distribute_results();

    // Update metrics
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start
    );
    metrics_.avg_iteration_us = update_running_average(
        metrics_.avg_iteration_us,
        duration.count()
    );
    metrics_.total_iterations++;

    return true;  // Continue
}
```

---

## Testing Contract

### Test Cases

**T-CONTRACT-5: Zero Allocation in Hot Path**
```cpp
TEST(BatchInferenceCoordinator, ZeroAllocationHotPath) {
    // Setup
    MockPyInferenceCallback callback;
    AsyncInferenceQueue queue;
    BatchInferenceCoordinator coordinator(&callback, &queue, 64);

    // Submit 64 requests to queue
    for (int i = 0; i < 64; ++i) {
        InferenceRequest req;
        req.features.resize(100);
        req.node_index = i;
        queue.submit_request(std::move(req));
    }

    // Run one iteration with allocation tracking
    size_t allocations_before = get_allocation_count();
    coordinator.run_iteration();
    size_t allocations_after = get_allocation_count();

    // Verify zero allocations (only moves/copies)
    EXPECT_EQ(allocations_after, allocations_before);
}
```

**T-CONTRACT-6: Batch Size Limits**
```cpp
TEST(BatchInferenceCoordinator, BatchSizeLimits) {
    MockPyInferenceCallback callback;
    AsyncInferenceQueue queue;
    BatchInferenceCoordinator coordinator(&callback, &queue, 32);  // Max 32

    // Submit 100 requests
    for (int i = 0; i < 100; ++i) {
        InferenceRequest req;
        req.features.resize(100);
        queue.submit_request(std::move(req));
    }

    // Run one iteration
    coordinator.run_iteration();

    // Verify batch size capped at 32
    auto metrics = coordinator.get_metrics();
    EXPECT_EQ(metrics.total_requests, 32);  // Only 32 processed
    EXPECT_EQ(queue.size(), 68);            // 68 remaining
}
```

**T-CONTRACT-7: Graceful Shutdown**
```cpp
TEST(BatchInferenceCoordinator, GracefulShutdown) {
    MockPyInferenceCallback callback;
    AsyncInferenceQueue queue;
    BatchInferenceCoordinator coordinator(&callback, &queue, 64);

    // Start coordinator thread
    std::atomic<bool> exited{false};
    std::thread coordinator_thread([&]() {
        coordinator.run();
        exited = true;
    });

    // Shutdown after 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    coordinator.shutdown();

    // Wait for exit (max 5 seconds)
    coordinator_thread.join();
    EXPECT_TRUE(exited);
}
```

**T-CONTRACT-8: Metrics Accuracy**
```cpp
TEST(BatchInferenceCoordinator, MetricsAccuracy) {
    MockPyInferenceCallback callback;
    AsyncInferenceQueue queue;
    BatchInferenceCoordinator coordinator(&callback, &queue, 64);

    // Submit 3 batches (64 + 64 + 32 = 160 requests)
    for (int batch = 0; batch < 3; ++batch) {
        int count = (batch == 2) ? 32 : 64;
        for (int i = 0; i < count; ++i) {
            InferenceRequest req;
            req.features.resize(100);
            queue.submit_request(std::move(req));
        }
        coordinator.run_iteration();
    }

    // Verify metrics
    auto metrics = coordinator.get_metrics();
    EXPECT_EQ(metrics.total_batches, 3);
    EXPECT_EQ(metrics.total_requests, 160);
    EXPECT_NEAR(metrics.avg_batch_size, 53.33, 0.1);  // (64+64+32)/3
}
```

---

## Performance Acceptance Criteria

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| `collect_batch()` latency | <10μs | Benchmark with pre-filled queue |
| `run_inference()` latency | <500μs (batch 64) | Benchmark with mock GPU (Phase 2: <300μs) |
| `distribute_results()` latency | <50μs (batch 64) | Benchmark with real tree |
| Total `run_iteration()` latency | <1ms (batch 64) | End-to-end benchmark |
| Memory allocations in hot path | 0 | Run with allocation profiler |
| Thread safety | TSan clean | Run all tests with `-fsanitize=thread` |

---

## Phase 2 Extensions

### Pinned Memory Tensor Creation (Phase 2B)

**Modified `run_inference()` with Pinned Tensor Bridge**:
```cpp
void BatchInferenceCoordinator::run_inference() {
    // NEW: Use pinned tensor bridge with stream handoff (NO synchronize)
    // Returns (gpu_tensor, event, stream) tuple for async overlap
    auto [batch_tensor, xfer_event, xfer_stream] = callback_->create_batch_tensor(batch_requests_);

    // Execute inference on the same stream (maintains async)
    // Python side: with torch.cuda.stream(xfer_stream): model(batch_tensor)
    auto [policy, value] = callback_->infer_batch(batch_tensor, xfer_stream);

    // Result copying (implicitly synchronized via .cpu())
    copy_policy_to_buffer(policy, batch_policy_);
    copy_value_to_buffer(value, batch_value_);
}
```

**Stream Handoff Requirements**:
1. **Tensor creation** returns tuple: `(gpu_tensor, event, stream)`
2. **Inference execution** uses same stream: `with torch.cuda.stream(stream): model(tensor)`
3. **NO `stream.synchronize()`** in hot path (defeats async overlap)
4. **Event-based sync** for metrics: `event.synchronize()` only when measuring latency

**Expected Improvement** (Phase 2):
- **Tensor creation**: 7.5ms → ≤2.0ms (p95, 3.75× faster)
- **Python-side overhead**: 37ms → ≤2.0ms (p95, 18.5× reduction)
- **GIL hold time**: 37ms → <0.5ms (74× reduction)
- **GPU utilization**: ~68% → 80%+ (better async overlap)

**Validation**:
- Run 100 trials, measure p95 tensor creation latency
- Assert ≤2.0ms per batch (batch_size=64, pinned memory enabled)
- Verify zero buffer reallocations (same memory address across iterations)

---

## References

- [plan.md](../plan.md): Phase 1C implementation details
- [async_inference_queue_api.md](async_inference_queue_api.md): Queue interface
- [data-model.md](../data-model.md): CoordinatorMetrics structure
- [profiling_api.md](profiling_api.md): Profiling integration
