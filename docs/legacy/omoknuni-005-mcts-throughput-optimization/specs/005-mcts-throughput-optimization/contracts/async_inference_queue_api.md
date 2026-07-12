# API Contract: AsyncInferenceQueue

**Feature**: MCTS Throughput Optimization
**Component**: Asynchronous Inference Queue (Phase 1B)
**File**: `cpp_extensions/mcts/async_inference_queue.{hpp,cpp}`
**Reference**: [plan.md](../plan.md#phase-1b-asyncinferencequeue-api-shift)

---

## Overview

This contract defines the modified API for `AsyncInferenceQueue` to support zero-copy feature submission via move semantics and efficient coordinator wake-up via condition variables.

**Key Changes from Baseline**:
- Accept `InferenceRequest&&` (rvalue reference) instead of state clones
- Add condition variable notification for coordinator wake-up
- Remove state storage (queue only stores features + metadata)

---

## Data Structures

### InferenceRequest (Move-Only)

**Purpose**: Encapsulate neural network inference request with ownership of feature buffer

**C++ Definition**:
```cpp
struct InferenceRequest {
    // OWNED feature buffer (moved from thread-local storage)
    std::vector<float> features;        // Size: planes × board_size²

    // Metadata for batch construction and result routing
    int32_t node_index;                 // Tree node requiring evaluation
    int32_t action_space_size;          // Number of legal moves at this node
    int16_t board_size;                 // Board dimension (8, 9, 15, or 19)
    int16_t planes;                     // Feature plane count (25-36)
    std::vector<int16_t> path;          // Move sequence from root (for fallback reconstruction)
    uint64_t request_id;                // Unique identifier for request tracking

    // Move-only semantics (NO COPY ALLOWED)
    InferenceRequest() = default;
    InferenceRequest(InferenceRequest&&) = default;
    InferenceRequest& operator=(InferenceRequest&&) = default;
    InferenceRequest(const InferenceRequest&) = delete;
    InferenceRequest& operator=(const InferenceRequest&) = delete;
};
```

**Validation Rules**:
- `features.size() == planes × board_size × board_size`
- `action_space_size > 0 && action_space_size <= 512`
- `planes >= 17 && planes <= 36` (range covers all 3 games)
- `board_size ∈ {8, 9, 15, 19}`
- `node_index >= 0`
- `request_id` must be unique across all active requests

---

## Interface Specification

### AsyncInferenceQueue Class

**Purpose**: Thread-safe MPMC queue for transferring feature buffers from simulation threads to coordinator

**C++ API**:
```cpp
class AsyncInferenceQueue {
public:
    // Backpressure policy when queue reaches max_size_
    enum class SubmitPolicy {
        Block,              // Block until space available (default)
        BlockWithTimeout,   // Block with timeout, return false on timeout
        DropOldest,         // Remove oldest request, insert new one
        DropNewest          // Drop new request, return false
    };

    // Constructor
    explicit AsyncInferenceQueue(size_t max_size = 4096);

    // Producer API: Submit request with backpressure handling
    // THREAD-SAFE: Called by multiple simulation threads concurrently
    // PRECONDITION: request.features must be non-empty and valid
    // POSTCONDITION: request is moved-from (empty state), coordinator is notified
    // RETURNS: False only on timeout (BlockWithTimeout) or drop (DropNewest); true otherwise
    // EFFECTS: May block if queue is full (policy-dependent)
    bool submit_request(InferenceRequest&& request,
                        SubmitPolicy policy = SubmitPolicy::Block,
                        std::chrono::microseconds timeout = std::chrono::microseconds{0});

    // Consumer API: Block until requests available or timeout/shutdown
    // THREAD-SAFE: Called by coordinator thread(s) - supports K coordinators with notify_one
    // PREDICATE: Uses predicate to defend against spurious wakeups
    // POSTCONDITION: Moves up to max_batch requests into output vector
    // RETURNS: Number of requests dequeued (0 if timeout or shutdown)
    // EFFECTS: Locks internally, waits with predicate, unlocks on return
    // NOTE: Replaces wait_for_request + dequeue_batch (encapsulated lock management)
    size_t dequeue_batch_blocking(std::vector<InferenceRequest>& output,
                                  size_t max_batch,
                                  std::chrono::microseconds timeout);

    // Single request dequeue (non-blocking, for testing/fallback)
    // THREAD-SAFE: Called by coordinator thread(s)
    // RETURNS: True if request was dequeued, false if queue empty
    bool try_dequeue(InferenceRequest& request);

    // Query approximate queue size (atomic read, may be stale)
    // THREAD-SAFE: Called by any thread
    // RETURNS: Approximate queue size (updated atomically under lock)
    size_t size() const;

    // Shutdown signal for graceful coordinator exit
    // THREAD-SAFE: Called by main thread during teardown
    // EFFECTS: Wakes all waiting coordinators via cv_request_ready_.notify_all()
    void shutdown();

    // Check if queue is in shutdown state
    // THREAD-SAFE: Called by coordinator threads
    bool is_shutdown() const;

private:
    std::deque<InferenceRequest> requests_;      // Deque for efficient pop_front/push_back
    mutable std::mutex mutex_;                   // Protects requests_, shutdown_, and size_
    std::condition_variable cv_request_ready_;   // Coordinator wake-up (requests available)
    std::condition_variable cv_not_full_;        // Producer wake-up (space available)
    size_t max_size_;                            // Maximum queue capacity (backpressure threshold)
    bool shutdown_ = false;                      // Shutdown flag
    std::atomic<size_t> size_{0};                // Approximate size (atomic for lock-free reads)
};
```

**Memory Implications**:
- Each `InferenceRequest` owns a `std::vector<float>` of features (~52KB for Go 19×19)
- At full capacity (4096 entries), worst-case memory: **~212 MB** of in-flight feature payloads
- **Recommendation**: Either (a) reduce `max_size_` to 1024-2048 for memory-constrained systems, or (b) refactor queue to hold handles into a fixed-size feature pool with recycling (e.g., 256 pre-allocated buffers = 13MB fixed)

**Multi-Coordinator Semantics**:
- Supports **K coordinator threads** (K∈{1,2,3,4}) operating concurrently
- Uses `notify_one()` to wake a single waiting coordinator (avoids thundering herd)
- Fairness is best-effort; sustained imbalance triggers auto-tuner adjustment
- **CUDA streams do NOT partition SMs**; concurrency is empirically tuned
- Each coordinator should call `dequeue_batch_blocking()` in a loop

**Spurious Wakeups**:
- All waits use predicates to defend against spurious wakeups
- Callers do NOT manage locks directly (encapsulated within `dequeue_batch_blocking`)

---

## Method Contracts

### submit_request

**Signature**:
```cpp
bool submit_request(InferenceRequest&& request,
                    SubmitPolicy policy = SubmitPolicy::Block,
                    std::chrono::microseconds timeout = std::chrono::microseconds{0});
```

**Preconditions**:
- `request.features.size() > 0` (non-empty feature buffer)
- `request.features.size() == request.planes × request.board_size × request.board_size`
- All request metadata fields are valid (see validation rules above)
- If `policy == BlockWithTimeout`, `timeout` must be > 0

**Effects**:
1. Acquires `mutex_` via `std::unique_lock`
2. **If queue is full (`requests_.size() >= max_size_`)**:
   - `Block`: Waits on `cv_not_full_` until space available
   - `BlockWithTimeout`: Waits with timeout; returns `false` if timeout expires
   - `DropOldest`: Removes oldest request (`requests_.pop_front()`), continues
   - `DropNewest`: Returns `false` immediately without inserting
3. Moves `request` into `requests_` deque: `requests_.push_back(std::move(request))`
4. Updates `size_` atomically
5. Notifies ONE waiting coordinator: `cv_request_ready_.notify_one()`
6. Releases `mutex_` at scope exit

**Postconditions**:
- `request` is in moved-from state (implementation-defined, typically empty) if returned `true`
- Queue size increased by 1 if returned `true` (or unchanged if DropOldest replaced entry)
- Returns `true` if request was enqueued, `false` only if:
  - `BlockWithTimeout` policy and timeout expired
  - `DropNewest` policy and queue was full
- At least one coordinator thread is woken if request was enqueued

**Thread Safety**:
- Safe to call concurrently from multiple simulation threads
- Lock acquisition order: always `mutex_` (no lock ordering issues)
- Backpressure via `cv_not_full_` prevents unbounded queue growth

**Performance**:
- Lock hold time: O(1) in fast path (just deque push + notify)
- May block if queue full and policy is `Block` or `BlockWithTimeout`
- Expected latency: <1μs in fast path (queue not full)

---

### dequeue_batch_blocking

**Signature**:
```cpp
size_t dequeue_batch_blocking(std::vector<InferenceRequest>& output,
                              size_t max_batch,
                              std::chrono::microseconds timeout);
```

**Preconditions**:
- `output` must be empty or pre-reserved (will be appended to)
- `max_batch > 0`
- `timeout` must be > 0

**Effects**:
1. Acquires `mutex_` internally via `std::unique_lock`
2. **Waits with predicate** on `cv_request_ready_` until:
   - Predicate: `[&]{ return shutdown_ || !requests_.empty(); }`
   - `submit_request()` calls `notify_one()` (new request available), OR
   - `timeout` expires, OR
   - `shutdown()` called
3. **If woken by predicate (requests available or shutdown)**:
   - Determines batch size: `n = min(requests_.size(), max_batch)`
   - Moves first `n` requests from `requests_` to `output`:
     ```cpp
     for (size_t i = 0; i < n; ++i) {
         output.push_back(std::move(requests_.front()));
         requests_.pop_front();
     }
     ```
   - Updates `size_` atomically
   - Notifies waiting producers if queue was full: `cv_not_full_.notify_one()`
4. **If timeout or shutdown**:
   - Returns 0 without dequeuing
5. Releases `mutex_` at scope exit
6. Returns `n` (number of requests dequeued)

**Postconditions**:
- Queue size reduced by `n` where `n` is the return value
- `output.size()` increased by `n`
- Each request in `output` owns its feature buffer (moved from queue)
- Returns 0 if timeout expired or shutdown triggered
- If returned > 0: At least one producer may be woken if queue had backpressure

**Thread Safety**:
- Safe to call concurrently from multiple coordinator threads
- Lock is managed internally (callers do NOT hold locks)
- Predicate defends against spurious wakeups
- Multi-coordinator fairness via `notify_one()` (no thundering herd)

**Performance**:
- Wake latency: <5μs (measured with FUTEX on Linux)
- Batch extraction: <10μs for 64 requests
- Zero memory allocations (moves only)
- No CPU spinning (thread sleeps in kernel while waiting)

**Spurious Wakeup Defense**:
The implementation uses `cv_request_ready_.wait_until(lock, deadline, [&]{ return shutdown_ || !requests_.empty(); })` to automatically re-check the predicate after spurious wakeups. Callers do not need to implement retry loops.

**Example Implementation Pattern**:
```cpp
size_t AsyncInferenceQueue::dequeue_batch_blocking(
    std::vector<InferenceRequest>& output,
    size_t max_batch,
    std::chrono::microseconds timeout)
{
    std::unique_lock<std::mutex> lock(mutex_);
    auto deadline = std::chrono::steady_clock::now() + timeout;

    // Wait with predicate (defends against spurious wakeups)
    if (!cv_request_ready_.wait_until(lock, deadline,
        [&]{ return shutdown_ || !requests_.empty(); }))
    {
        return 0;  // Timeout
    }

    if (shutdown_ && requests_.empty()) {
        return 0;  // Shutdown with no requests
    }

    // Dequeue batch
    size_t n = std::min(requests_.size(), max_batch);
    bool was_full = (requests_.size() >= max_size_);

    for (size_t i = 0; i < n; ++i) {
        output.push_back(std::move(requests_.front()));
        requests_.pop_front();
    }

    size_.store(requests_.size(), std::memory_order_relaxed);

    // Wake blocked producers if queue had backpressure
    if (was_full) {
        cv_not_full_.notify_one();
    }

    return n;
}

---

### shutdown

**Signature**:
```cpp
void shutdown();
```

**Preconditions**: None

**Effects**:
1. Acquires `mutex_`
2. Sets `shutdown_ = true`
3. Wakes ALL waiting coordinators: `cv_request_ready_.notify_all()`
4. Wakes ALL waiting producers: `cv_not_full_.notify_all()`
5. Releases `mutex_`

**Postconditions**:
- All future `dequeue_batch_blocking()` calls return 0 immediately (after checking predicate)
- All currently blocked coordinators wake and return 0
- All currently blocked producers (waiting on full queue) wake and can check shutdown state
- `is_shutdown()` returns `true`

**Thread Safety**:
- Safe to call once during teardown
- NOT safe to call concurrently from multiple threads (undefined behavior)

**Performance**:
- Latency: <10μs (wake all threads)

**Usage Pattern**:
```cpp
// Main thread shutdown sequence
queue.shutdown();  // Signal all threads to stop

// Coordinator threads should detect shutdown via dequeue_batch_blocking() returning 0
while (true) {
    size_t count = queue.dequeue_batch_blocking(batch, 64, std::chrono::microseconds(500));
    if (count == 0 && queue.is_shutdown()) {
        break;  // Graceful exit
    }
    // ... process batch ...
}
```

---

## Usage Example

### Simulation Thread (Producer)

```cpp
void run_simulation(ThreadLocalState& tls, AsyncInferenceQueue& queue) {
    // Select leaf node
    Node* leaf = select_leaf(root, tls);

    if (leaf->is_leaf() && !leaf->is_terminal()) {
        // Extract features in-place into thread-local buffer
        game->extract_features_to_buffer(
            current_state,
            tls.feature_buffer.data()
        );

        // Build request (prepare for move)
        InferenceRequest request;
        request.features = std::move(tls.feature_buffer);  // Move buffer ownership
        request.node_index = leaf->index;
        request.action_space_size = game->get_action_space_size();
        request.board_size = game->get_board_size();
        request.planes = game->get_feature_planes();
        request.path = get_path_to_node(leaf);
        request.request_id = generate_unique_id();

        // Submit to queue (transfer ownership)
        queue.submit_request(std::move(request));  // ✅ Zero copy

        // NOTE: request is now empty, tls.feature_buffer is empty
        // They will be resized/refilled on next simulation
    }
}
```

### Coordinator Thread (Consumer)

```cpp
void coordinator_loop(AsyncInferenceQueue& queue) {
    std::vector<InferenceRequest> batch;
    batch.reserve(64);  // Pre-allocate for max batch size

    while (true) {
        // Wait for requests and dequeue batch (encapsulated lock management)
        // Blocks until requests available, timeout expires, or shutdown signaled
        size_t count = queue.dequeue_batch_blocking(
            batch,
            64,  // max_batch
            std::chrono::microseconds(500)  // 500μs timeout
        );

        // Check for graceful shutdown
        if (count == 0) {
            if (queue.is_shutdown()) break;  // Graceful exit
            continue;  // Timeout, retry
        }

        // Process batch (create tensor, run inference, distribute results)
        // NO locks held during GPU work - queue manages locks internally
        process_batch(batch);

        // Clear batch for reuse (feature buffers are moved out in process_batch)
        batch.clear();
    }
}
```

**Key Improvements**:
- **No lock exposure**: Callers don't manage `std::unique_lock` objects
- **Encapsulated synchronization**: All mutex/CV operations hidden inside `dequeue_batch_blocking()`
- **Predicate-based waits**: Spurious wakeups handled internally
- **Simpler control flow**: Single call replaces wait + dequeue sequence
- **Multi-coordinator safe**: Multiple threads can call `dequeue_batch_blocking()` concurrently

---

## Testing Contract

### Test Cases

**T-CONTRACT-1: Move Semantics**
```cpp
TEST(AsyncInferenceQueue, MoveSemantics) {
    InferenceRequest req;
    req.features.resize(1000);
    req.node_index = 42;

    size_t orig_size = req.features.size();
    InferenceRequest moved = std::move(req);

    // Original is moved-from (empty or unspecified state)
    EXPECT_TRUE(req.features.empty() || req.features.size() == 0);

    // Moved-to has ownership
    EXPECT_EQ(moved.features.size(), orig_size);
    EXPECT_EQ(moved.node_index, 42);
}
```

**T-CONTRACT-2: Condition Variable Wake**
```cpp
TEST(AsyncInferenceQueue, ConditionVariableWake) {
    AsyncInferenceQueue queue;
    std::atomic<size_t> dequeued_count{0};

    // Coordinator thread: wait for request using encapsulated API
    std::thread coordinator([&]() {
        std::vector<InferenceRequest> batch;
        size_t count = queue.dequeue_batch_blocking(
            batch,
            64,  // max_batch
            std::chrono::seconds(5)  // timeout
        );
        dequeued_count = count;
    });

    // Simulation thread: submit request after 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    InferenceRequest req;
    req.features.resize(100);
    bool submitted = queue.submit_request(std::move(req));

    coordinator.join();
    EXPECT_TRUE(submitted);  // Request should be enqueued
    EXPECT_EQ(dequeued_count, 1);  // Coordinator should dequeue 1 request
}
```

**T-CONTRACT-3: Batch Dequeue**
```cpp
TEST(AsyncInferenceQueue, BatchDequeue) {
    AsyncInferenceQueue queue;

    // Submit 100 requests
    for (int i = 0; i < 100; ++i) {
        InferenceRequest req;
        req.features.resize(100);
        req.node_index = i;
        bool submitted = queue.submit_request(std::move(req));
        EXPECT_TRUE(submitted);
    }

    // Dequeue batch of 64 using encapsulated API
    std::vector<InferenceRequest> batch;
    size_t count = queue.dequeue_batch_blocking(
        batch,
        64,  // max_batch
        std::chrono::milliseconds(100)  // timeout (should not expire)
    );

    EXPECT_EQ(count, 64);
    EXPECT_EQ(batch.size(), 64);
    EXPECT_EQ(queue.size(), 36);  // 100 - 64 = 36 remaining

    // Verify request order preserved (FIFO)
    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(batch[i].node_index, i);
    }
}
```

**T-CONTRACT-4: Shutdown Grace**
```cpp
TEST(AsyncInferenceQueue, ShutdownGraceful) {
    AsyncInferenceQueue queue;
    std::atomic<int> shutdown_exit_count{0};

    // Start 3 coordinator threads using encapsulated API
    std::vector<std::thread> coordinators;
    for (int i = 0; i < 3; ++i) {
        coordinators.emplace_back([&]() {
            std::vector<InferenceRequest> batch;
            while (true) {
                size_t count = queue.dequeue_batch_blocking(
                    batch,
                    64,
                    std::chrono::seconds(10)
                );
                if (count == 0 && queue.is_shutdown()) {
                    shutdown_exit_count++;
                    break;  // Graceful exit on shutdown
                }
                batch.clear();
            }
        });
    }

    // Shutdown queue after brief delay
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    queue.shutdown();

    // All coordinators should exit gracefully
    for (auto& t : coordinators) t.join();
    EXPECT_EQ(shutdown_exit_count, 3);  // All 3 threads exited via shutdown path
}
```

---

## Performance Acceptance Criteria

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| `submit_request()` latency (fast path) | <1μs (p95) | Benchmark with 8 threads, 1M submissions, queue not full |
| `submit_request()` with backpressure | Variable (policy-dependent) | Test Block/BlockWithTimeout/Drop policies under saturation |
| `dequeue_batch_blocking()` wake latency | <5μs (p95) | Measure time from `submit_request()` to coordinator wake |
| `dequeue_batch_blocking()` batch extraction | <10μs for 64 requests | Benchmark with pre-filled queue |
| Memory allocations in hot path | 0 | Run with allocation profiler, verify zero malloc/free |
| Thread safety | TSan clean | Run all tests with `-fsanitize=thread` |
| Spurious wakeup resilience | 100% handled | Inject spurious wakeups, verify correct behavior |
| Multi-coordinator fairness | Within 20% of ideal | Run K=4 coordinators, measure throughput imbalance |

---

## References

- [plan.md](../plan.md): Phase 1B implementation details
- [data-model.md](../data-model.md): InferenceRequest structure definition
- [C++ Condition Variables](https://en.cppreference.com/w/cpp/thread/condition_variable)
- [Move Semantics](https://en.cppreference.com/w/cpp/language/move_constructor)
