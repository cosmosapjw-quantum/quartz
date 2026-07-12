# API Contract: Profiling & Metrics Collection

**Feature**: MCTS Throughput Optimization
**Component**: Profiling Infrastructure (Phase Validation)
**Files**: `src/telemetry/profiling.py`, `cpp_extensions/mcts/profiling_metrics.{hpp,cpp}`
**Reference**: [plan.md](../plan.md#phase-validation)

---

## Overview

This contract defines the profiling API for collecting performance metrics during optimization phases. Metrics are used for:
1. **Phase acceptance gates** (Constitution Principle VI)
2. **Regression detection** (automated rollback)
3. **Bottleneck identification** (profiling campaigns)

**Key Design Principles**:
- **Minimal overhead**: Disabled path uses macro-based no-ops (compile-time elimination); runtime APIs short-circuit in ~tens of nanoseconds
- **Thread-safe**: Integer metric IDs with thread-local storage (TLS) aggregation; per-thread buffers flushed periodically to avoid contention
- **High-resolution timing**: Nanosecond precision via `std::chrono::high_resolution_clock`
- **Structured output**: JSON/CSV export for automated analysis and regression detection

---

## Data Structures

### ProfilingMetrics (C++)

**Purpose**: Aggregate performance metrics from MCTS search and inference pipeline

**C++ Definition**:
```cpp
struct ProfilingMetrics {
    // ==== BASELINE METRICS (existing) ====
    uint64_t simulations_completed = 0;      // Total MCTS simulations
    uint64_t tree_nodes_allocated = 0;       // Total nodes allocated
    double search_duration_ms = 0.0;         // Total search time (milliseconds)

    // ==== PHASE 1 METRICS (State Cloning Elimination) ====
    double state_cloning_us = 0.0;           // Time in state cloning (target: ~0 μs)
    uint64_t state_clone_count = 0;          // Count of clone() calls (target: 0)
    double feature_extraction_us = 0.0;      // Time in in-place extraction
    uint64_t feature_move_count = 0;         // Count of std::move(features) (should == simulations)

    // Coordinator metrics
    double coordinator_wait_us = 0.0;        // Time coordinator waits for requests
    double coordinator_collect_us = 0.0;     // Time collecting batch from queue
    double coordinator_inference_us = 0.0;   // Time in inference (including tensor creation)
    double coordinator_distribute_us = 0.0;  // Time distributing results to tree
    uint64_t batch_count = 0;                // Total batches processed
    double avg_batch_size = 0.0;             // Mean requests per batch

    // Virtual loss metrics
    uint64_t virtual_loss_restarts = 0;      // Count of immediate restarts on contention
    double virtual_loss_success_rate = 0.0;  // % of selections without contention

    // ==== PHASE 2 METRICS (Tensor + OpenMP) ====
    double tensor_creation_ms = 0.0;         // Time to create batch tensor (target: ≤2.0ms)
    double h2d_transfer_ms = 0.0;            // Host-to-device transfer time (target: ≤1.0ms)
    double gpu_inference_ms = 0.0;           // Pure GPU inference time (no transfer)
    double d2h_transfer_ms = 0.0;            // Device-to-host transfer time

    // OpenMP metrics
    int32_t openmp_thread_count = 0;         // Actual OpenMP threads used (target: >1)
    bool openmp_enabled = false;             // True if OpenMP linked and active
    double feature_extraction_parallel_us = 0.0;  // Time in parallel extraction (should be <serial)

    // Pinned memory metrics
    double pinned_buffer_reuse_pct = 0.0;    // % of batches using pre-allocated buffer (target: 100%)
    uint64_t pinned_buffer_reallocations = 0;  // Count of unexpected reallocations (target: 0)

    // ==== PHASE 3A METRICS (Multi-Coordinator) ====
    int32_t active_coordinators = 0;         // Number of active coordinator threads
    double coordinator_blocking_pct = 0.0;   // % of time coordinators block (target: <10%)
    double gpu_utilization_pct = 0.0;        // GPU utilization percentage (target: ≥80%)

    // ==== DERIVED METRICS (computed) ====
    double sims_per_second = 0.0;            // Throughput (computed: simulations / search_duration)
    double avg_iteration_ms = 0.0;           // Mean coordinator iteration time
    double state_clone_overhead_pct = 0.0;   // % of time in state cloning (target: <1%)

    // ==== MEMORY METRICS ====
    size_t peak_memory_bytes = 0;            // Peak memory usage (bytes)
    size_t tree_memory_bytes = 0;            // MCTS tree memory footprint
    size_t feature_buffer_memory_bytes = 0;  // Thread-local feature buffers

    // ==== TIMESTAMP ====
    uint64_t timestamp_ns = 0;               // Nanosecond timestamp when metrics collected
};
```

---

## Interface Specification

### ProfilingSession (Python)

**Purpose**: Manage profiling campaigns with automated acceptance gates

**Python API**:
```python
class ProfilingSession:
    """
    Manage a profiling campaign for phase acceptance testing.

    Example:
        session = ProfilingSession(
            phase="Phase 1A - State Cloning Elimination",
            min_trials=100,
            acceptance_criteria={
                "sims_per_second": (1500, 3000),  # (min, max) range
                "state_clone_overhead_pct": (0.0, 1.0),
                "state_clone_count": (0, 0),  # Must be exactly 0
            }
        )

        for trial in range(100):
            metrics = session.run_trial(config)
            session.record_metrics(metrics)

        report = session.generate_report()
        if session.passes_acceptance():
            print("Phase accepted ✅")
        else:
            print("Phase rejected ❌ - initiating rollback")
    """

    def __init__(
        self,
        phase: str,
        min_trials: int = 100,
        acceptance_criteria: Dict[str, Tuple[float, float]],
        output_dir: Path = Path("profiling_results"),
    ):
        """
        Initialize profiling session.

        Args:
            phase: Phase name (e.g., "Phase 1A - State Cloning Elimination")
            min_trials: Minimum trials required for statistical significance
            acceptance_criteria: Dict mapping metric name to (min, max) range
            output_dir: Directory to save profiling results
        """
        ...

    def run_trial(self, config: SearchConfig) -> ProfilingMetrics:
        """
        Run single MCTS search trial and collect metrics.

        Args:
            config: Search configuration (simulations, threads, etc.)

        Returns:
            ProfilingMetrics object with collected data

        Raises:
            ProfilingError: If trial fails or metrics invalid
        """
        ...

    def record_metrics(self, metrics: ProfilingMetrics) -> None:
        """
        Record metrics from trial for aggregation.

        Args:
            metrics: Metrics collected from run_trial()

        Effects:
            - Appends metrics to internal trial list
            - Updates running statistics (mean, std, p50, p95, p99)
        """
        ...

    def generate_report(self) -> ProfilingReport:
        """
        Generate statistical report from collected trials.

        Returns:
            ProfilingReport with:
            - Summary statistics (mean, std, percentiles)
            - Acceptance gate results (pass/fail per criterion)
            - Regression analysis (vs baseline)
            - Bottleneck ranking (time attribution)

        Precondition:
            - At least min_trials recorded
        """
        ...

    def passes_acceptance(self) -> bool:
        """
        Check if all acceptance criteria met.

        Returns:
            True if ALL criteria pass, False otherwise

        Criteria evaluation:
            - For each metric in acceptance_criteria:
              * **Throughput metrics** (sims_per_second, etc.): Use **p50 (median)** across all trials
              * **Latency metrics** (tensor_creation_ms, h2d_transfer_ms, etc.): Use **p95 (95th percentile)** across all trials
              * Check if statistic ∈ [min, max] range
            - All criteria must pass for overall PASS
            - **Rationale**: p50 for throughput (central tendency); p95 for latency (tail behavior, robustness)
        """
        ...

    def save_results(self, path: Path) -> None:
        """
        Save raw metrics and report to disk.

        Args:
            path: Output path (JSON format)

        Effects:
            - Writes all trial metrics (CSV format)
            - Writes summary report (JSON format)
            - Writes plots (PNG format): throughput distribution, timeline
        """
        ...
```

---

### ProfilingMetricsCollector (C++)

**Purpose**: Collect timing and counter metrics from C++ code with zero overhead when disabled

**C++ API**:
```cpp
class ProfilingMetricsCollector {
public:
    // Singleton access (thread-safe)
    static ProfilingMetricsCollector& instance();

    // Enable/disable profiling (default: disabled for zero overhead)
    void enable(bool enabled);
    bool is_enabled() const;

    // ==== TIMING MEASUREMENTS ====

    // Start timing region (returns handle for end_timing)
    // Usage: auto handle = collector.start_timing("feature_extraction");
    TimingHandle start_timing(const std::string& region_name);

    // End timing region and record duration
    // Usage: collector.end_timing(handle);
    void end_timing(TimingHandle handle);

    // RAII wrapper for automatic timing
    // Usage: { auto _timer = collector.scoped_timer("collect_batch"); ... }
    class ScopedTimer {
    public:
        ScopedTimer(const std::string& region_name);
        ~ScopedTimer();
    private:
        TimingHandle handle_;
    };

    ScopedTimer scoped_timer(const std::string& region_name);

    // ==== COUNTER OPERATIONS ====

    // Increment counter (atomic, thread-safe)
    // Usage: collector.increment("state_clone_count");
    void increment(const std::string& counter_name, uint64_t delta = 1);

    // Record value (atomic, thread-safe)
    // Usage: collector.record("batch_size", 64);
    void record(const std::string& metric_name, double value);

    // ==== METRICS EXTRACTION ====

    // Get current metrics snapshot (thread-safe)
    ProfilingMetrics get_metrics() const;

    // Reset all metrics (for new profiling session)
    void reset();

private:
    std::atomic<bool> enabled_{false};

    // ==== INTEGER METRIC IDs (enum-based, TLS aggregation) ====
    // Metric identifiers (compile-time constants for fast lookup)
    enum class MetricID : uint32_t {
        STATE_CLONING_US = 0,
        FEATURE_EXTRACTION_US,
        TENSOR_CREATION_MS,
        H2D_TRANSFER_MS,
        GPU_INFERENCE_MS,
        COORDINATOR_WAIT_US,
        // ... (complete enum in implementation)
        METRIC_COUNT
    };

    // Per-thread metric buffers (TLS, flushed periodically to avoid contention)
    struct ThreadLocalMetrics {
        std::array<double, static_cast<size_t>(MetricID::METRIC_COUNT)> timings{};
        std::array<uint64_t, static_cast<size_t>(MetricID::METRIC_COUNT)> counters{};
    };
    static thread_local ThreadLocalMetrics tls_metrics_;

    // Global aggregated metrics (flushed from TLS periodically or on get_metrics())
    std::array<std::atomic<double>, static_cast<size_t>(MetricID::METRIC_COUNT)> aggregated_timings_{};
    std::array<std::atomic<uint64_t>, static_cast<size_t>(MetricID::METRIC_COUNT)> aggregated_counters_{};

    mutable std::mutex flush_mutex_;  // Protects flush operations
};

// Convenience macro for scoped timing
#define PROFILE_SCOPE(name) \
    auto _scoped_timer_##__LINE__ = \
        ProfilingMetricsCollector::instance().scoped_timer(name)
```

---

## Usage Examples

### Phase 1 Profiling Campaign (Python)

```python
from src.telemetry.profiling import ProfilingSession, SearchConfig

# Define Phase 1 acceptance criteria (from constitution)
acceptance_criteria = {
    "sims_per_second": (1500, 3000),         # 10-25× baseline (120 sims/sec)
    "state_clone_overhead_pct": (0.0, 1.0),  # <1% of total time
    "state_clone_count": (0, 0),              # Must be exactly 0
    "feature_move_count": (800, 800),         # Must equal simulation count
    "avg_batch_size": (32, 64),               # Target batch size range
}

# Create profiling session
session = ProfilingSession(
    phase="Phase 1 - State Cloning Elimination",
    min_trials=100,
    acceptance_criteria=acceptance_criteria,
    output_dir=Path("profiling_results/phase_1"),
)

# Run 100 trials
config = SearchConfig(
    game="gomoku",
    simulations=800,
    threads=8,
    batch_size=64,
    timeout_us=500,
)

for trial_id in range(100):
    print(f"Trial {trial_id + 1}/100...", end="")
    metrics = session.run_trial(config)
    session.record_metrics(metrics)
    print(f" {metrics.sims_per_second:.1f} sims/sec")

# Generate report
report = session.generate_report()
print("\n" + "="*60)
print(f"Phase 1 Profiling Report")
print("="*60)
print(f"Trials: {report.trial_count}")
print(f"Mean throughput: {report.mean_sims_per_sec:.1f} ± {report.std_sims_per_sec:.1f}")
print(f"Percentiles: p50={report.p50_sims_per_sec:.1f}, p95={report.p95_sims_per_sec:.1f}")
print(f"\nAcceptance Gates:")
for criterion, result in report.acceptance_results.items():
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"  {criterion}: {result.value:.2f} ∈ [{result.min}, {result.max}] {status}")

# Check overall acceptance
if session.passes_acceptance():
    print("\n✅ Phase 1 ACCEPTED - proceeding to Phase 2")
    session.save_results(Path("profiling_results/phase_1/final_report.json"))
else:
    print("\n❌ Phase 1 REJECTED - initiating rollback")
    print("Failed criteria:")
    for criterion, result in report.acceptance_results.items():
        if not result.passed:
            print(f"  - {criterion}: {result.value:.2f} (expected [{result.min}, {result.max}])")
```

### C++ Profiling Instrumentation

```cpp
// continuous_simulation_runner.cpp

void ContinuousSimulationRunner::run_simulation(ThreadLocalState& tls) {
    auto& profiler = ProfilingMetricsCollector::instance();

    // Phase 1A: Feature extraction timing
    {
        PROFILE_SCOPE("feature_extraction");
        game->extract_features_to_buffer(current_state, tls.feature_buffer.data());
    }

    // Phase 1B: Queue submission (should be negligible)
    {
        PROFILE_SCOPE("queue_submission");

        InferenceRequest request;
        request.features = std::move(tls.feature_buffer);  // Move, not copy
        request.node_index = leaf->index;
        // ... fill metadata ...

        queue->submit_request(std::move(request));
        profiler.increment("feature_move_count");  // Should equal simulation count
    }

    // Virtual loss tracking
    if (selection_failed_due_to_contention) {
        profiler.increment("virtual_loss_restarts");
    }
}

// batch_inference_coordinator.cpp

void BatchInferenceCoordinator::run_iteration() {
    auto& profiler = ProfilingMetricsCollector::instance();

    // Collect batch
    size_t count;
    {
        PROFILE_SCOPE("coordinator_collect");
        count = collect_batch();
    }

    if (count == 0) return;  // Timeout

    profiler.record("batch_size", count);
    profiler.increment("batch_count");

    // Inference
    {
        PROFILE_SCOPE("coordinator_inference");
        run_inference();
    }

    // Distribution
    {
        PROFILE_SCOPE("coordinator_distribute");
        distribute_results();
    }
}
```

---

## Testing Contract

### Test Cases

**T-CONTRACT-9: Minimal Overhead**
```cpp
TEST(ProfilingMetricsCollector, MinimalOverhead) {
    auto& profiler = ProfilingMetricsCollector::instance();

    // Test 1: Macro-based no-ops (compile-time elimination when disabled)
    profiler.enable(false);
    auto start_noop = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1'000'000; ++i) {
        PROFILE_INCREMENT(STATE_CLONE_COUNT);  // Macro expands to no-op if disabled
        PROFILE_SCOPE("test_region");           // Macro expands to no-op if disabled
    }
    auto noop_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_noop
    ).count();
    // Verify negligible overhead: 1M no-op calls < 5ms (O2, LTO, Ryzen 5900X)
    EXPECT_LT(noop_us, 5000);  // <5ms for compile-time eliminated calls

    // Test 2: Runtime short-circuit (enabled=false, runtime check)
    profiler.enable(false);
    auto start_runtime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1'000'000; ++i) {
        if (profiler.is_enabled()) {  // Runtime check (not compile-time)
            profiler.increment(ProfilingMetricsCollector::MetricID::STATE_CLONE_COUNT);
            auto handle = profiler.start_timing(ProfilingMetricsCollector::MetricID::TENSOR_CREATION_MS);
            profiler.end_timing(handle);
        }
    }
    auto runtime_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_runtime
    ).count();
    // Verify minimal overhead: 1M runtime short-circuits < 100ms (O2, LTO, Ryzen 5900X)
    EXPECT_LT(runtime_us, 100'000);  // <100ms for runtime-checked calls

    // Test 3: Actual profiling overhead (enabled, TLS aggregation)
    profiler.enable(true);
    profiler.reset();
    auto start_enabled = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1'000'000; ++i) {
        profiler.increment(ProfilingMetricsCollector::MetricID::STATE_CLONE_COUNT);
        auto handle = profiler.start_timing(ProfilingMetricsCollector::MetricID::TENSOR_CREATION_MS);
        profiler.end_timing(handle);
    }
    auto enabled_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_enabled
    ).count();
    // TLS aggregation should keep overhead reasonable: <500ms for 1M actual measurements
    EXPECT_LT(enabled_us, 500'000);  // <500ms with TLS aggregation
}
```

**T-CONTRACT-10: Timing Accuracy**
```cpp
TEST(ProfilingMetricsCollector, TimingAccuracy) {
    auto& profiler = ProfilingMetricsCollector::instance();
    profiler.enable(true);
    profiler.reset();

    // Record known sleep duration
    {
        auto timer = profiler.scoped_timer("test_sleep");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto metrics = profiler.get_metrics();
    double recorded_ms = metrics.timings_["test_sleep"] / 1000.0;  // μs → ms

    // Verify accuracy within 10% (system sleep precision)
    EXPECT_NEAR(recorded_ms, 100.0, 10.0);
}
```

**T-CONTRACT-11: Thread Safety**
```cpp
TEST(ProfilingMetricsCollector, ThreadSafety) {
    auto& profiler = ProfilingMetricsCollector::instance();
    profiler.enable(true);
    profiler.reset();

    // 8 threads incrementing same counter
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 10'000; ++i) {
                profiler.increment("thread_safe_counter");
            }
        });
    }

    for (auto& t : threads) t.join();

    auto metrics = profiler.get_metrics();
    EXPECT_EQ(metrics.counters_["thread_safe_counter"], 80'000);  // 8 × 10k
}
```

**T-CONTRACT-12: Profiling Session Acceptance**
```python
def test_profiling_session_acceptance():
    """Verify acceptance gates work correctly"""
    session = ProfilingSession(
        phase="Test Phase",
        min_trials=10,
        acceptance_criteria={
            "sims_per_second": (100, 200),
            "state_clone_count": (0, 0),
        }
    )

    # Record 10 trials within range
    for _ in range(10):
        metrics = ProfilingMetrics()
        metrics.sims_per_second = 150.0  # Within [100, 200]
        metrics.state_clone_count = 0     # Exactly 0
        session.record_metrics(metrics)

    # Should pass
    assert session.passes_acceptance()

    # Record one trial outside range
    metrics = ProfilingMetrics()
    metrics.sims_per_second = 250.0  # Outside [100, 200]
    metrics.state_clone_count = 0
    session.record_metrics(metrics)

    # Should still pass (median still in range)
    assert session.passes_acceptance()

    # Record 10 more trials outside range
    for _ in range(10):
        metrics = ProfilingMetrics()
        metrics.sims_per_second = 300.0  # Outside [100, 200]
        metrics.state_clone_count = 0
        session.record_metrics(metrics)

    # Should fail now (median shifted above max)
    assert not session.passes_acceptance()
```

---

## Acceptance Criteria

### Phase 1: State Cloning Elimination

| Metric | Target Range | Validation |
|--------|--------------|------------|
| `sims_per_second` (throughput) | [1,500, 3,000] | 100 trials, **p50** must be in range |
| `state_clone_overhead_pct` (latency) | [0.0, 1.0] | **p95** <1% of total time |
| `state_clone_count` (counter) | [0, 0] | Must be exactly 0 (no cloning) |
| `feature_move_count` (counter) | [simulations, simulations] | Must equal simulation count |
| `avg_batch_size` (throughput) | [32, 64] | **p50** in target range |

### Phase 2: Tensor + OpenMP

| Metric | Target Range | Validation |
|--------|--------------|------------|
| `sims_per_second` (throughput) | [7,000, 9,000] | 100 trials, **p50** must be in range |
| `tensor_creation_ms` (latency) | [0.0, 2.0] | **p95** ≤2.0ms per batch (batch_size=64) |
| `h2d_transfer_ms` (latency) | [0.0, 1.0] | **p95** ≤1.0ms per batch (batch_size=64) |
| `openmp_enabled` (boolean) | [True, True] | Must be True (verify linkage) |
| `openmp_thread_count` (counter) | [2, 12] | >1 thread (parallel extraction) |
| `pinned_buffer_reuse_pct` (counter) | [100.0, 100.0] | Must be 100% (zero reallocations) |
| `pinned_buffer_reallocations` (counter) | [0, 0] | Must be 0 (assert stable address) |

### Phase 3A: Multi-Coordinator (Stretch)

| Metric | Target Range | Validation |
|--------|--------------|------------|
| `sims_per_second` (throughput) | [12,000, 20,000] | 100 trials, **p50** must be in range |
| `coordinator_blocking_pct` (latency) | [0.0, 10.0] | **p95** <10% blocking time |
| `gpu_utilization_pct` (throughput) | [80.0, 100.0] | **p50** ≥80% GPU utilization |
| `active_coordinators` (counter) | [2, 4] | Auto-tuned K ∈ {1,2,3,4}, default 3 on RTX 3060 Ti |

---

## References

- [plan.md](../plan.md): Phase validation procedures
- [CLAUDE.md](../../../CLAUDE.md): Constitution Principle VI (Evidence-Based Gates)
- [research.md](../research.md): Baseline profiling results
- [data-model.md](../data-model.md): ProfilingMetrics structure
