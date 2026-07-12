/**
 * @file example_usage.cpp
 * @brief Example usage of enhanced profiling instrumentation
 *
 * Demonstrates how to integrate profiling into MCTS code.
 */

#include "profiler.hpp"
#include "hardware_counters.hpp"
#include "statistical_analyzer.hpp"
#include "contention_tracker.hpp"
#include "export.hpp"

namespace mcts {
namespace profiling {

// ============================================================
// Example 1: Basic scoped profiling
// ============================================================

void example_basic_profiling() {
    // Enable profiling
    Profiler::instance().enable();
    Profiler::instance().start_session("BasicExample");

    // Profile a function scope
    {
        PROFILE_SCOPE(ProfileMetric::SelectionTotal);

        // Simulate work
        for (int i = 0; i < 1000; ++i) {
            // Do selection work...
        }
    }

    // Profile nested operations
    {
        PROFILE_SCOPE(ProfileMetric::ExpansionTotal);

        {
            PROFILE_SCOPE(ProfileMetric::ExpansionInferenceRequest);
            // Submit inference request...
        }

        {
            PROFILE_SCOPE(ProfileMetric::ExpansionAllocateNodes);
            // Allocate child nodes...
        }
    }

    // Stop session and export
    Profiler::instance().stop_session();
    Profiler::instance().export_json("basic_profile.json");
}

// ============================================================
// Example 2: Hardware counter integration
// ============================================================

void example_hardware_counters() {
    Profiler::instance().enable();
    Profiler::instance().enable_hardware_counters();
    Profiler::instance().start_session("HardwareCountersExample");

    HardwareCounterReader hw;
    if (hw.initialize()) {
        hw.start();

        // Run MCTS simulations
        {
            PROFILE_SCOPE(ProfileMetric::SelectionTotal);
            // ... MCTS work ...
        }

        hw.stop();

        // Read hardware counters
        auto cycles = hw.read(HWCounterType::CPUCycles);
        auto instructions = hw.read(HWCounterType::Instructions);
        auto cache_misses = hw.read(HWCounterType::CacheMisses);

        // Record as gauges
        PROFILE_GAUGE(ProfileMetric::HWCPUCycles, cycles);
        PROFILE_GAUGE(ProfileMetric::HWInstructions, instructions);
        PROFILE_GAUGE(ProfileMetric::HWCacheMisses, cache_misses);
    }

    Profiler::instance().stop_session();
    Profiler::instance().export_markdown("hw_counters_report.md");
}

// ============================================================
// Example 3: Contention tracking
// ============================================================

void example_contention_tracking() {
    Profiler::instance().enable();
    Profiler::instance().start_session("ContentionExample");

    // Use tracked atomic for CAS operations
    TrackedAtomic<uint64_t> visit_count(0);

    for (int i = 0; i < 1000; ++i) {
        uint64_t expected = visit_count.load();
        uint64_t desired = expected + 1;

        // Track CAS retries
        while (!visit_count.compare_exchange_weak(
            expected, desired,
            std::memory_order_release,
            std::memory_order_acquire,
            ProfileMetric::VirtualLossCASSuccess
        )) {
            PROFILE_COUNTER(ProfileMetric::VirtualLossCASRetries);
            expected = visit_count.load();
            desired = expected + 1;
        }
    }

    // Use tracked mutex
    TrackedMutex mutex;
    {
        std::lock_guard<TrackedMutex> lock(mutex);
        // Critical section...
    }

    Profiler::instance().stop_session();
    Profiler::instance().export_chrome_trace("contention_trace.json");
}

// ============================================================
// Example 4: Statistical analysis
// ============================================================

void example_statistical_analysis() {
    Profiler::instance().enable();
    Profiler::instance().start_session("StatisticsExample");

    // Run profiling session
    for (int i = 0; i < 10000; ++i) {
        PROFILE_SCOPE(ProfileMetric::SelectionTotal);
        // ... work ...
    }

    // Collect thread metrics
    auto thread_metrics = Profiler::instance().get_all_thread_metrics();

    // Aggregate statistics
    MetricAggregator aggregator;
    auto metrics = aggregator.aggregate(thread_metrics);

    // Analyze specific metric
    StatisticalAnalyzer analyzer;
    for (const auto& metric_stats : metrics) {
        if (metric_stats.count > 0) {
            // Print statistics
            printf("Metric: %s\n", metric_name(static_cast<ProfileMetric>(0)));  // TODO: Get metric enum
            printf("  Mean: %.2f ns\n", metric_stats.mean);
            printf("  Std Dev: %.2f ns\n", metric_stats.std_dev);
            printf("  P50: %.2f ns\n", metric_stats.p50);
            printf("  P95: %.2f ns\n", metric_stats.p95);
            printf("  P99: %.2f ns\n", metric_stats.p99);
            printf("  Min: %lu ns\n", metric_stats.min);
            printf("  Max: %lu ns\n", metric_stats.max);
        }
    }

    // Detect bottlenecks
    BottleneckDetector detector;
    auto bottlenecks = detector.detect(metrics);

    for (const auto& bottleneck : bottlenecks) {
        printf("Bottleneck: %s (severity: %.2f)\n",
               metric_name(bottleneck.metric),
               bottleneck.severity);
    }

    Profiler::instance().stop_session();
}

// ============================================================
// Example 5: Instrumented MCTS simulation
// ============================================================

class InstrumentedSimulationRunner {
public:
    void run_simulation() {
        PROFILE_SCOPE(ProfileMetric::SelectionTotal);

        // Selection phase
        auto leaf = select_leaf();

        // Expansion phase
        expand_node(leaf);

        // Backup phase
        backup_value(leaf);
    }

private:
    int select_leaf() {
        PROFILE_SCOPE(ProfileMetric::SelectionPUCT);

        // Track tree depth
        int depth = 0;

        while (/* not leaf */) {
            depth++;

            // Track atomic loads
            {
                PROFILE_SCOPE(ProfileMetric::SelectionAtomicLoad);
                // Load visit counts atomically...
            }

            // Track PUCT computation
            {
                PROFILE_SCOPE_CYCLES(ProfileMetric::SelectionPUCT);
                // Compute PUCT scores...
            }

            // Track busy-edge skips
            if (/* node is expanding */) {
                PROFILE_COUNTER(ProfileMetric::SelectionBusyEdgeSkip);
                continue;
            }

            // Apply virtual loss
            {
                PROFILE_SCOPE(ProfileMetric::VirtualLossApply);
                // Apply virtual loss...
            }
        }

        PROFILE_GAUGE(ProfileMetric::SelectionDepth, depth);
        return 0;  // Return leaf index
    }

    void expand_node(int leaf) {
        PROFILE_SCOPE(ProfileMetric::ExpansionTotal);

        // Terminal check
        {
            PROFILE_SCOPE(ProfileMetric::ExpansionTerminalCheck);
            // Check if terminal...
        }

        // Legal moves
        {
            PROFILE_SCOPE(ProfileMetric::ExpansionLegalMoves);
            // Generate legal moves...
        }

        // Inference request
        {
            PROFILE_SCOPE(ProfileMetric::ExpansionInferenceRequest);
            // Submit to inference queue...
        }

        // Allocate nodes
        {
            PROFILE_SCOPE(ProfileMetric::ExpansionAllocateNodes);
            PROFILE_MEMORY_ALLOC(/* node_size * num_children */);
            // Allocate child nodes...
        }

        // CAS expanded flag
        {
            PROFILE_SCOPE(ProfileMetric::ExpansionCASExpanded);
            // Atomic CAS to mark expanded...
        }
    }

    void backup_value(int leaf) {
        PROFILE_SCOPE(ProfileMetric::BackupTotal);

        // Path traversal
        {
            PROFILE_SCOPE(ProfileMetric::BackupPathTraversal);
            // Traverse path from leaf to root...
        }

        // Atomic updates
        for (int i = 0; i < /* path length */; ++i) {
            {
                PROFILE_SCOPE(ProfileMetric::BackupAtomicVisitUpdate);
                // Atomic increment visit count...
            }

            {
                PROFILE_SCOPE(ProfileMetric::BackupAtomicValueUpdate);
                // Atomic add to value...
            }

            {
                PROFILE_SCOPE(ProfileMetric::BackupVirtualLossRemove);
                // Remove virtual loss...
            }
        }
    }
};

// ============================================================
// Example 6: Queue profiling
// ============================================================

void example_queue_profiling() {
    PROFILE_SCOPE(ProfileMetric::QueueSubmit);

    // Submit to MPMC queue
    {
        PROFILE_SCOPE(ProfileMetric::QueueSubmitMPMCPush);
        // Push to lock-free queue...
    }

    // Update queue depth gauge
    PROFILE_UPDATE_GAUGE(ProfileMetric::QueuePendingDepth, 1);

    // Collect batch (with timeout)
    {
        PROFILE_SCOPE(ProfileMetric::QueueCollect);

        {
            PROFILE_SCOPE(ProfileMetric::QueueCollectWait);
            // Wait on condition variable...
        }

        // Record batch size
        int batch_size = 64;
        PROFILE_GAUGE(ProfileMetric::QueueBatchSize, batch_size);
    }

    // Try get result
    {
        PROFILE_SCOPE(ProfileMetric::QueueTryGetResult);

        {
            PROFILE_SCOPE(ProfileMetric::QueueResultLookup);
            // Lookup in ring buffer...
        }
    }
}

// ============================================================
// Example 7: Full profiling session with export
// ============================================================

void example_full_session() {
    // Configure profiling
    Profiler::instance().enable();
    Profiler::instance().set_sampling_rate(1);  // Profile all operations
    Profiler::instance().enable_hardware_counters();

    // Start session
    Profiler::instance().start_session("MCTSOptimizationRun");

    // Run MCTS (simulated)
    for (int i = 0; i < 10000; ++i) {
        InstrumentedSimulationRunner runner;
        runner.run_simulation();
    }

    // Stop session
    Profiler::instance().stop_session();

    // Get metrics
    auto thread_metrics = Profiler::instance().get_all_thread_metrics();

    // Aggregate
    MetricAggregator aggregator;
    auto metrics = aggregator.aggregate(thread_metrics);

    // Detect bottlenecks
    BottleneckDetector detector;
    auto bottlenecks = detector.detect(metrics);

    // Build comprehensive report
    SessionMetadata metadata;
    metadata.session_name = "MCTSOptimizationRun";
    metadata.num_threads = 4;
    metadata.total_samples = 40000;

    ReportBuilder()
        .add_session(metadata)
        .add_metrics(metrics)
        .add_bottlenecks(bottlenecks)
        .export_json("full_report.json")
        .export_chrome_trace("full_trace.json")
        .export_markdown("full_report.md")
        .export_html("full_report.html");
}

// ============================================================
// Example 8: Regression detection
// ============================================================

void example_regression_detection() {
    // Load baseline from previous run
    SessionMetadata baseline_metadata;
    std::vector<MetricStatistics> baseline_metrics;
    DataLoader::load_json("baseline.json", baseline_metadata, baseline_metrics);

    // Run current profiling
    Profiler::instance().enable();
    Profiler::instance().start_session("RegressionTest");
    // ... run MCTS ...
    Profiler::instance().stop_session();

    // Get current metrics
    auto thread_metrics = Profiler::instance().get_all_thread_metrics();
    MetricAggregator aggregator;
    auto current_metrics = aggregator.aggregate(thread_metrics);

    // Compare
    SessionComparator comparator;
    auto diffs = comparator.compare(baseline_metrics, current_metrics, 5.0);

    // Find regressions
    auto regressions = comparator.find_regressions(diffs);

    if (!regressions.empty()) {
        printf("PERFORMANCE REGRESSIONS DETECTED:\n");
        for (const auto& diff : regressions) {
            printf("  %s: %.2f%% slower\n",
                   metric_name(diff.metric),
                   diff.change_percentage);
        }
    }

    // Generate comparison report
    std::string report = comparator.generate_comparison_report(diffs);
    std::ofstream out("regression_report.md");
    out << report;
}

} // namespace profiling
} // namespace mcts
