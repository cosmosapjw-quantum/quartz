#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace mcts {

/**
 * @brief Metrics tracked by the instrumentation subsystem.
 */
enum class InstrumentationMetric : std::uint8_t {
    TreeClear = 0,
    TreeAllocateNode,
    TreeAllocateNodes,
    Selection,
    Expansion,
    Backup,
    VirtualLossApply,
    VirtualLossRemove,
    QueueSubmit,
    QueueCollect,
    QueueProcessResults,
    QueueTryGetResult,
    ExpansionConflict,        // Node already being expanded by another thread
    BusyEdgeMasked,           // Node skipped due to expanding flag (busy-edge masking)
    UniqueBatchPositions,     // Number of unique positions in each inference batch
    SelectionRetry,           // Selection restarted due to conflicts
    // Phase 1 Metrics (State Cloning Elimination)
    StateCloning,             // Time spent in state cloning operations
    FeatureExtraction,        // Time spent in in-place feature extraction
    StateCloneCount,          // Counter: number of clone() calls (target: 0)
    FeatureMoveCount,         // Counter: number of std::move(features) to queue
    // Phase 2 Metrics (Tensor Pipeline + OpenMP)
    TensorCreation,           // Time spent creating batch tensors
    H2DTransfer,              // Time spent in host-to-device transfer
    OpenMPThreadCount,        // Gauge: actual OpenMP threads used (target: >1)
    OpenMPEnabled,            // Boolean flag: true if OpenMP linked
    PinnedBufferReuse,        // Counter: batches using pre-allocated pinned buffer
    PinnedBufferAllocation,   // Counter: new pinned buffer allocations (target: 0)
    Count
};

struct MetricSnapshot {
    std::uint64_t call_count = 0;
    std::uint64_t total_elapsed_ns = 0;
};

class Instrumentation {
public:
    static Instrumentation& instance();

    void set_enabled(bool enabled);
    bool is_enabled() const;

    void record_duration(InstrumentationMetric metric, std::uint64_t elapsed_ns);
    void increment_counter(InstrumentationMetric metric, std::uint64_t value = 1);

    std::unordered_map<InstrumentationMetric, MetricSnapshot> snapshot() const;

    void reset();

private:
    Instrumentation() = default;

    struct MetricData {
        std::atomic<std::uint64_t> call_count{0};
        std::atomic<std::uint64_t> total_elapsed_ns{0};
    };

    std::atomic<bool> enabled_{false};
    std::array<MetricData, static_cast<std::size_t>(InstrumentationMetric::Count)> metrics_{};
};

class ScopedMetric {
public:
    explicit ScopedMetric(InstrumentationMetric metric);
    ScopedMetric(const ScopedMetric&) = delete;
    ScopedMetric& operator=(const ScopedMetric&) = delete;
    ~ScopedMetric();

private:
    InstrumentationMetric metric_;
    bool enabled_;
    std::chrono::steady_clock::time_point start_;
};

std::string_view metric_to_string(InstrumentationMetric metric);

} // namespace mcts
