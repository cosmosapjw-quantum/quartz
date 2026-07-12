#include "instrumentation.hpp"

#include <mutex>

namespace mcts {

Instrumentation& Instrumentation::instance() {
    static Instrumentation instance;
    return instance;
}

void Instrumentation::set_enabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_release);
}

bool Instrumentation::is_enabled() const {
    return enabled_.load(std::memory_order_acquire);
}

void Instrumentation::record_duration(InstrumentationMetric metric, std::uint64_t elapsed_ns) {
    if (!is_enabled()) {
        return;
    }

    auto index = static_cast<std::size_t>(metric);
    auto& data = metrics_[index];
    data.call_count.fetch_add(1, std::memory_order_relaxed);
    data.total_elapsed_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
}

void Instrumentation::increment_counter(InstrumentationMetric metric, std::uint64_t value) {
    if (!is_enabled()) {
        return;
    }

    auto index = static_cast<std::size_t>(metric);
    auto& data = metrics_[index];
    data.call_count.fetch_add(value, std::memory_order_relaxed);
}

std::unordered_map<InstrumentationMetric, MetricSnapshot> Instrumentation::snapshot() const {
    std::unordered_map<InstrumentationMetric, MetricSnapshot> result;
    if (!is_enabled()) {
        return result;
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(InstrumentationMetric::Count); ++i) {
        MetricSnapshot snapshot{};
        snapshot.call_count = metrics_[i].call_count.load(std::memory_order_relaxed);
        snapshot.total_elapsed_ns = metrics_[i].total_elapsed_ns.load(std::memory_order_relaxed);
        if (snapshot.call_count == 0 && snapshot.total_elapsed_ns == 0) {
            continue;
        }
        result.emplace(static_cast<InstrumentationMetric>(i), snapshot);
    }

    return result;
}

void Instrumentation::reset() {
    for (auto& metric : metrics_) {
        metric.call_count.store(0, std::memory_order_relaxed);
        metric.total_elapsed_ns.store(0, std::memory_order_relaxed);
    }
}

ScopedMetric::ScopedMetric(InstrumentationMetric metric)
    : metric_(metric),
      enabled_(Instrumentation::instance().is_enabled()),
      start_(enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point()) {
}

ScopedMetric::~ScopedMetric() {
    if (!enabled_) {
        return;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
    Instrumentation::instance().record_duration(metric_, static_cast<std::uint64_t>(elapsed));
}

std::string_view metric_to_string(InstrumentationMetric metric) {
    switch (metric) {
        case InstrumentationMetric::TreeClear:
            return "tree_clear";
        case InstrumentationMetric::TreeAllocateNode:
            return "tree_allocate_node";
        case InstrumentationMetric::TreeAllocateNodes:
            return "tree_allocate_nodes";
        case InstrumentationMetric::Selection:
            return "selection";
        case InstrumentationMetric::Expansion:
            return "expansion";
        case InstrumentationMetric::Backup:
            return "backup";
        case InstrumentationMetric::VirtualLossApply:
            return "virtual_loss_apply";
        case InstrumentationMetric::VirtualLossRemove:
            return "virtual_loss_remove";
        case InstrumentationMetric::QueueSubmit:
            return "queue_submit";
        case InstrumentationMetric::QueueCollect:
            return "queue_collect";
        case InstrumentationMetric::QueueProcessResults:
            return "queue_process_results";
        case InstrumentationMetric::QueueTryGetResult:
            return "queue_try_get_result";
        case InstrumentationMetric::ExpansionConflict:
            return "expansion_conflict";
        case InstrumentationMetric::BusyEdgeMasked:
            return "busy_edge_masked";
        case InstrumentationMetric::UniqueBatchPositions:
            return "unique_batch_positions";
        case InstrumentationMetric::SelectionRetry:
            return "selection_retry";
        // Phase 1 Metrics
        case InstrumentationMetric::StateCloning:
            return "state_cloning";
        case InstrumentationMetric::FeatureExtraction:
            return "feature_extraction";
        case InstrumentationMetric::StateCloneCount:
            return "state_clone_count";
        case InstrumentationMetric::FeatureMoveCount:
            return "feature_move_count";
        // Phase 2 Metrics
        case InstrumentationMetric::TensorCreation:
            return "tensor_creation";
        case InstrumentationMetric::H2DTransfer:
            return "h2d_transfer";
        case InstrumentationMetric::OpenMPThreadCount:
            return "openmp_thread_count";
        case InstrumentationMetric::OpenMPEnabled:
            return "openmp_enabled";
        case InstrumentationMetric::PinnedBufferReuse:
            return "pinned_buffer_reuse";
        case InstrumentationMetric::PinnedBufferAllocation:
            return "pinned_buffer_allocation";
        case InstrumentationMetric::Count:
        default:
            return "unknown";
    }
}

} // namespace mcts
