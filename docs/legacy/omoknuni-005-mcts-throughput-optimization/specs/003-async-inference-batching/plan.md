# Implementation Plan: Async Inference Batching

**Specification:** `specs/003-async-inference-batching/spec.md`
**Target:** 30,000+ simulations/second
**Complexity:** High (C++ threading, GIL management, async coordination)

## Overview

This plan details the implementation of async inference batching to achieve 30k+ sims/sec. The core innovation is **continuous C++ simulation loops** with non-blocking inference, replacing the current synchronous callback pattern.

**Key Insight:** Threads never wait for inference. They submit requests to a C++ queue, continue running simulations, and periodically check for completed results.

## Architecture Layers

### Layer 1: AsyncInferenceQueue (C++)

**Purpose:** Thread-safe queue for managing pending inference requests and distributing results.

**Components:**

```cpp
// cpp_extensions/mcts/async_inference_queue.hpp

class AsyncInferenceQueue {
public:
    struct InferenceRequest {
        NodeIndex node;
        std::unique_ptr<IGameState> state;
        std::vector<NodeIndex> path;
        uint64_t request_id;
    };

    struct InferenceResult {
        uint64_t request_id;
        std::vector<float> policy;
        float value;
    };

    // Submit request (non-blocking)
    void submit_request(InferenceRequest request);

    // Collect batch for Python callback
    std::vector<InferenceRequest> collect_batch(
        size_t min_batch_size,
        double timeout_ms
    );

    // Submit completed results
    void submit_results(std::vector<InferenceResult> results);

    // Get completed result for request_id (non-blocking)
    std::optional<InferenceResult> try_get_result(uint64_t request_id);

    // Check if results available
    bool has_results() const;

private:
    // Pending inference requests
    std::deque<InferenceRequest> pending_requests_;
    std::mutex pending_mutex_;

    // Completed results (request_id → result)
    std::unordered_map<uint64_t, InferenceResult> completed_results_;
    std::mutex results_mutex_;

    // Request ID generator
    std::atomic<uint64_t> next_request_id_{0};
};
```

**Key Design Decisions:**

1. **Deque for pending requests:** O(1) push_back, O(1) pop_front
2. **Map for completed results:** O(1) lookup by request_id
3. **Separate mutexes:** Minimize contention between submit/collect paths
4. **Optional return:** Non-blocking result retrieval

### Layer 2: ContinuousSimulationRunner (C++)

**Purpose:** Run simulations continuously without blocking on inference.

**Algorithm:**

```cpp
// Pseudocode for continuous simulation loop

void ContinuousSimulationRunner::run_continuous(
    IGameState& root_state,
    NodeIndex root_index,
    AsyncInferenceQueue& queue,
    int num_simulations)
{
    int completed = 0;
    std::unordered_map<uint64_t, PendingExpansion> pending_expansions;

    while (completed < num_simulations) {
        // Phase 1: Run simulation to leaf
        std::vector<NodeIndex> path;
        IGameState* current_state = root_state.clone();
        NodeIndex leaf = select_leaf(root_index, *current_state, path);

        // Phase 2: Check if expansion needed
        if (!is_terminal(*current_state) && !is_expanded(leaf)) {
            // Submit inference request (NON-BLOCKING)
            uint64_t req_id = queue.submit_request({
                .node = leaf,
                .state = std::move(current_state),
                .path = path
            });

            // Store pending expansion
            pending_expansions[req_id] = {leaf, path};

        } else {
            // Terminal or already expanded - backup immediately
            float value = get_value(*current_state);
            backup_value(path, value);
            completed++;
        }

        // Phase 3: Process any completed inferences (NON-BLOCKING)
        while (queue.has_results()) {
            auto result = queue.try_get_result(...);
            if (result) {
                auto& pending = pending_expansions[result->request_id];
                expand_node(pending.node, result->policy);
                backup_value(pending.path, result->value);
                pending_expansions.erase(result->request_id);
                completed++;
            } else {
                break;  // No more results ready
            }
        }
    }
}
```

**Key Features:**

1. **Never blocks:** Always makes progress
2. **Pending expansion tracking:** Maps request_id → expansion data
3. **Continuous result processing:** Checks for results every loop iteration
4. **Simulation accounting:** Tracks completed vs pending

### Layer 3: BatchInferenceCoordinator (C++)

**Purpose:** Background thread that batches inference requests and calls Python.

**Implementation:**

```cpp
// cpp_extensions/mcts/batch_inference_coordinator.hpp

class BatchInferenceCoordinator {
public:
    BatchInferenceCoordinator(
        AsyncInferenceQueue& queue,
        py::object python_callback,
        size_t batch_size,
        double timeout_ms
    );

    ~BatchInferenceCoordinator();

    void start();  // Start background thread
    void stop();   // Stop background thread

private:
    void coordinator_loop();  // Main loop

    AsyncInferenceQueue& queue_;
    py::object python_callback_;
    size_t batch_size_;
    double timeout_ms_;

    std::thread worker_thread_;
    std::atomic<bool> running_{false};
};
```

**Coordinator Loop:**

```cpp
void BatchInferenceCoordinator::coordinator_loop() {
    while (running_) {
        // Collect batch from queue (with timeout)
        auto batch = queue_.collect_batch(batch_size_, timeout_ms_);

        if (batch.empty()) {
            continue;
        }

        // Prepare Python input (list of states)
        py::gil_scoped_acquire gil;
        py::list states;
        for (const auto& req : batch) {
            states.append(py::cast(req.state.get(),
                                   py::return_value_policy::reference));
        }

        // Call Python batch inference (SINGLE GIL ACQUISITION)
        py::object result = python_callback_(states);

        // Extract results
        py::list results_list = result.cast<py::list>();
        std::vector<InferenceResult> results;

        for (size_t i = 0; i < batch.size(); ++i) {
            py::tuple item = results_list[i].cast<py::tuple>();
            py::list policy_py = item[0].cast<py::list>();
            float value = item[1].cast<float>();

            std::vector<float> policy;
            for (auto p : policy_py) {
                policy.push_back(p.cast<float>());
            }

            results.push_back({
                .request_id = batch[i].request_id,
                .policy = std::move(policy),
                .value = value
            });
        }

        // Submit results back to queue
        {
            py::gil_scoped_release nogil;
            queue_.submit_results(std::move(results));
        }
    }
}
```

**GIL Management:**
- **Acquire:** Once at start of batch processing
- **Hold during:** Python callback, result extraction
- **Release:** Before submitting results to C++ queue

### Layer 4: Python Integration

**Purpose:** Create and manage async infrastructure from Python.

**Critical Design Decision:** Dual-Mode Batch Inference Callback

The batch inference callback must support two modes for maximum performance while maintaining test compatibility:

1. **Direct GPU Batching Mode (Production)** - FAST
   - Detects if `inference_fn` has `batch_inference()` method
   - Calls `gpu_worker.batch_inference(positions)` ONCE per batch
   - Achieves 10-15k sims/sec (10-15× faster than per-state mode)

2. **Per-State Future Mode (Testing)** - SLOW but compatible
   - Falls back for test mocks without `batch_inference()` method
   - Calls `inference_fn(state)` for each state (legacy compatibility)
   - Achieves ~1k sims/sec (acceptable for tests)

**Implementation:**

```python
# src/core/mcts.py

class AlphaZeroMCTS:
    def _create_batch_inference_callback(self) -> Callable:
        """Create batch callback with automatic mode detection."""

        # MODE 1: Direct GPU Batching (Production - FAST)
        if hasattr(self.inference_fn, 'batch_inference'):
            def fast_batch_callback(game_states: List[IGameState]):
                # Extract positions ONCE
                positions = [
                    np.array(state.get_enhanced_tensor_representation(), dtype=np.float32)
                    for state in game_states
                ]

                # ✅ SINGLE batched GPU call
                policies, values = self.inference_fn.batch_inference(positions)

                # Convert to expected format
                results = []
                for i in range(len(policies)):
                    policy_list = policies[i].tolist() if hasattr(policies[i], 'tolist') else list(policies[i])
                    results.append((policy_list, float(values[i])))

                return results

            self.logger.info(f"Using direct GPU batch inference (fast path)")
            return fast_batch_callback

        # MODE 2: Per-State Future Mode (Testing - SLOW)
        else:
            def legacy_batch_callback(game_states: List[IGameState]):
                # ⚠️ SLOW: 32× individual calls for testing/mocks
                futures = [self.inference_fn(state) for state in game_states]
                results = []
                for future in futures:
                    policy, value = future.result(timeout=1.0)
                    policy_list = policy.tolist() if hasattr(policy, 'tolist') else list(policy)
                    results.append((policy_list, float(value)))
                return results

            self.logger.warning(f"Using legacy per-state inference (slow path, testing only)")
            return legacy_batch_callback

    def __init__(self, inference_fn, ..., use_async_inference=True):
        self.inference_fn = inference_fn  # Can be GPUInferenceWorker or test mock

        if use_async_inference:
            # Create async infrastructure
            self.async_queue = mcts_py.AsyncInferenceQueue()

            # Create dual-mode batch callback
            batch_callback = mcts_py.PyBatchInferenceCallback(
                self._create_batch_inference_callback()
            )

            # Create and start coordinator
            self.coordinator = mcts_py.BatchInferenceCoordinator()
            self.coordinator.start(
                self.async_queue,
                batch_callback,
                self.async_batch_size,  # Default: 32
                self.async_timeout_ms   # Default: 2.0ms
            )

            # Use ContinuousSimulationRunner for async mode
            self.simulation_runner = mcts_py.ContinuousSimulationRunner(
                self.tree, self.selector, self.backup_manager, self.virtual_loss_manager
            )
        else:
            # Sync mode (backward compatibility)
            self.simulation_runner = mcts_py.SimulationRunner(...)

    def search(self, root_state, simulations):
        if self.use_async_inference:
            # Async path: run_continuous returns when simulations complete
            completed = self.simulation_runner.run_continuous(
                root_state, self.root_index, self.async_queue, simulations
            )
            return self._collect_visit_counts()
        else:
            # Sync path: traditional loop
            for _ in range(simulations):
                self.simulation_runner.run_simulation(...)
            return self._collect_visit_counts()
```

**Key Insight:**

The original plan showed the correct approach (direct GPU batching), but the initial implementation mistakenly used per-state Future calls in the callback. This bug was discovered during realistic testing when performance was 1,061 sims/sec instead of expected 15k+.

**Performance Validation:**

| Mode | Implementation | Throughput | Use Case |
|------|---------------|------------|----------|
| Direct GPU Batch | `gpu_worker.batch_inference(positions)` | 10-15k sims/sec | Production |
| Per-State Future | `[inference_fn(s) for s in states]` | 1k sims/sec | Testing only |

**Decision:** Use `hasattr(inference_fn, 'batch_inference')` to automatically select the fastest available mode.

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

**Tasks:**
1. Implement `AsyncInferenceQueue` with unit tests
2. Add thread-safety validation (TSan)
3. Benchmark queue operations (<0.1ms per operation)

**Deliverables:**
- `cpp_extensions/mcts/async_inference_queue.{hpp,cpp}`
- `tests/unit/test_async_inference_queue.cpp`
- Performance benchmark results

**Success Criteria:**
- All tests pass
- TSan clean
- Queue operations <0.1ms

### Phase 2: Continuous Runner (Week 2)

**Tasks:**
1. Implement `ContinuousSimulationRunner`
2. Add pending expansion tracking
3. Integrate with existing MCTS components
4. Unit tests for continuous simulation loop

**Deliverables:**
- `cpp_extensions/mcts/continuous_simulation_runner.{hpp,cpp}`
- `tests/unit/test_continuous_runner.cpp`
- Integration tests with mock inference

**Success Criteria:**
- Completes target simulations correctly
- Never blocks on inference
- Tree state matches synchronous runner

### Phase 3: Batch Coordinator (Week 2)

**Tasks:**
1. Implement `BatchInferenceCoordinator`
2. GIL management with RAII guards
3. Error handling for Python exceptions
4. Thread lifecycle management

**Deliverables:**
- `cpp_extensions/mcts/batch_inference_coordinator.{hpp,cpp}`
- `tests/integration/test_batch_coordinator.py`
- GIL stress tests

**Success Criteria:**
- Batches requests correctly
- Single GIL acquisition per batch
- Handles Python exceptions gracefully
- Clean shutdown

### Phase 4: Python Bindings (Week 3)

**Tasks:**
1. Add pybind11 bindings for all components
2. Python wrapper in `AlphaZeroMCTS`
3. Feature flag for async vs sync mode
4. Migration guide for existing code

**Deliverables:**
- `cpp_extensions/mcts/python_bindings.cpp` (updated)
- `src/core/mcts.py` (async support)
- `docs/async_inference_migration.md`

**Success Criteria:**
- Python API matches sync mode
- Easy toggle between modes
- Backward compatible

### Phase 5: Performance Tuning (Week 3-4)

**Tasks:**
1. Benchmark with real neural network
2. Tune batch size and timeout
3. Profile and optimize hot paths
4. Thread count optimization

**Deliverables:**
- `tests/performance/test_async_30k.py`
- `docs/async_performance_tuning.md`
- Performance comparison report

**Success Criteria:**
- **≥30,000 sims/sec with 8-12 threads**
- ≥60% GPU utilization
- Avg batch size ≥32

### Phase 6: Validation & Documentation (Week 4)

**Tasks:**
1. Soak tests (24-hour continuous operation)
2. Memory leak detection (Valgrind)
3. Correctness validation (vs sync mode)
4. Update documentation

**Deliverables:**
- `tests/soak/test_async_stability.py`
- `ASYNC_INFERENCE_VALIDATION_REPORT.md`
- Updated `mcts_guide.md`

**Success Criteria:**
- Zero memory leaks
- Identical results to sync mode
- Stable over 24 hours
- Complete documentation

## Technical Challenges

### Challenge 1: Thread Synchronization

**Problem:** Multiple threads accessing shared queue concurrently.

**Solution:**
- Separate mutexes for pending/results
- Lock-free queue for submit path (optional optimization)
- Atomic operations for request ID generation

**Validation:**
- TSan clean under high concurrency
- Stress test with 100+ threads
- Formal verification of queue invariants

### Challenge 2: GIL Management

**Problem:** Deadlocks if GIL acquired/released incorrectly.

**Solution:**
- Use pybind11 RAII guards exclusively (`py::gil_scoped_acquire`, `py::gil_scoped_release`)
- Never manually acquire/release GIL
- Test matrix: Python callback throws exception during each GIL state

**Validation:**
- GIL stress tests
- Exception injection tests
- Deadlock detection (timeout tests)

### Challenge 3: Memory Management

**Problem:** Game state ownership across queue boundaries.

**Solution:**
- Use `std::unique_ptr` for ownership transfer
- Clone game states before queue submission
- Smart pointers prevent leaks

**Validation:**
- Valgrind clean
- ASan clean
- Leak detector in long-running tests

### Challenge 4: Load Balancing

**Problem:** Some threads may complete faster than others.

**Solution:**
- Dynamic work distribution (threads pull from shared counter)
- Monitor thread utilization
- Consider work stealing (future optimization)

**Validation:**
- Thread utilization metrics
- Load imbalance measurement
- Performance consistency across runs

## Performance Optimization Strategies

### Strategy 1: Batch Size Tuning

**Grid Search:**
```python
batch_sizes = [16, 32, 48, 64, 96, 128]
timeouts = [0.3, 0.5, 1.0, 2.0, 3.0]

for batch_size in batch_sizes:
    for timeout in timeouts:
        throughput = benchmark(batch_size, timeout)
        log_result(batch_size, timeout, throughput)

optimal = find_pareto_optimal(results)
```

**Expected Optima:**
- Batch size: 48-64 (balance batching vs latency)
- Timeout: 0.5-1.0ms (minimize idle time)

### Strategy 2: Thread Count Optimization

**Heuristic:**
```
optimal_threads = min(
    physical_cores,  # Avoid oversubscription
    batch_size / avg_sims_per_batch_cycle  # Keep queue full
)
```

**For Ryzen 5900X (12 cores):**
- Expected optimal: 8-12 threads
- Test range: 4-16 threads

### Strategy 3: Queue Prefetch

**Idea:** Coordinator prefetches next batch while GPU processes current batch.

**Implementation:**
```cpp
// Double buffering
auto batch1 = queue.collect_batch(...);
while (running_) {
    auto batch2 = queue.collect_batch(...);  // Collect while processing

    process_batch(batch1);  // GPU inference

    std::swap(batch1, batch2);
}
```

**Expected gain:** 5-10% throughput improvement

### Strategy 4: Result Distribution Optimization

**Current:** O(n) linear search through result map

**Optimization:** Per-thread result queues
```cpp
std::vector<std::queue<InferenceResult>> per_thread_results_;

// Coordinator distributes by thread_id (stored in request)
per_thread_results_[request.thread_id].push(result);

// Thread retrieves from own queue (no contention)
auto result = per_thread_results_[my_thread_id].try_pop();
```

**Expected gain:** 10-15% reduction in result processing time

## Testing Strategy

### Unit Tests

**AsyncInferenceQueue:**
- Submit/collect under single thread
- Concurrent submit from multiple threads
- Batch collection with timeout
- Result distribution

**ContinuousSimulationRunner:**
- Completes target simulations
- Handles terminal nodes
- Processes results correctly
- Maintains tree consistency

**BatchInferenceCoordinator:**
- Batches requests correctly
- Calls Python callback
- Distributes results
- Handles exceptions

### Integration Tests

**End-to-End:**
- AlphaZeroMCTS with async mode
- Real neural network inference
- Multiple games (Gomoku, Chess, Go)
- Correctness vs sync mode

**Performance:**
- Throughput benchmarks
- GPU utilization measurement
- Batch size distribution
- Thread utilization

**Stress Tests:**
- 100+ concurrent threads
- 1M+ simulations
- 24-hour soak test
- Memory leak detection

### Validation Tests

**Correctness:**
- Deterministic seed test (sync vs async same results)
- Tree structure validation
- Visit count verification
- Value propagation check

**Thread Safety:**
- TSan validation
- Helgrind validation
- Race condition injection

## Rollout Plan

### Stage 1: Alpha (Internal Testing)

- Feature flag: `use_async_inference=False` (default)
- Enable for unit/integration tests only
- No production use
- Collect performance data

### Stage 2: Beta (Opt-In)

- Feature flag: `use_async_inference=True` (opt-in)
- Document performance characteristics
- Provide migration guide
- Monitor for issues

### Stage 3: GA (General Availability)

- Feature flag: `use_async_inference=True` (default)
- Fallback to sync mode on error
- Remove sync mode after validation period
- Archive old implementation

## Rollback Plan

**If throughput <20,000 sims/sec:**
1. Keep sync mode as default
2. Debug async implementation
3. Re-evaluate architecture

**If stability issues:**
1. Disable async mode
2. Fix root cause
3. Re-test validation suite

**If memory leaks:**
1. Revert to sync mode
2. Run Valgrind on isolated repro
3. Fix leak, re-validate

## Metrics and Monitoring

**Performance Metrics:**
- Simulations/second (target: ≥30k)
- GPU utilization (target: ≥60%)
- Average batch size (target: ≥32)
- Thread utilization (target: ≥75%)

**Quality Metrics:**
- Memory usage (target: stable)
- Crash rate (target: 0)
- Correctness (target: 100% match sync mode)

**Operational Metrics:**
- Queue depth (monitor for saturation)
- Inference latency P50/P95/P99
- GIL contention time

## Dependencies

**Build:**
- CMake ≥3.18
- C++17 compiler
- pybind11 ≥2.10
- OpenMP (for atomics)

**Runtime:**
- PyTorch with CUDA
- NVIDIA drivers
- Python 3.12+

**Testing:**
- Google Test (C++ unit tests)
- pytest (Python integration tests)
- ThreadSanitizer (race detection)
- Valgrind (leak detection)

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Queue | 3-4 days | None |
| Phase 2: Runner | 4-5 days | Phase 1 |
| Phase 3: Coordinator | 3-4 days | Phase 1 |
| Phase 4: Bindings | 2-3 days | Phase 2, 3 |
| Phase 5: Tuning | 5-7 days | Phase 4 |
| Phase 6: Validation | 3-4 days | Phase 5 |

**Total: 20-27 days (4-5 weeks)**

## Success Metrics

**Primary:**
✅ Achieve 30,000+ sims/sec with 8-12 threads on Ryzen 5900X + RTX 3060 Ti

**Secondary:**
✅ GPU utilization ≥60%
✅ Zero memory leaks (24-hour soak test)
✅ Thread-safe (TSan clean)
✅ Matches sync mode results (correctness validation)

**Stretch:**
✅ Achieve 35,000+ sims/sec
✅ GPU utilization ≥80%
✅ Linear speedup up to 12 threads (efficiency ≥75%)
