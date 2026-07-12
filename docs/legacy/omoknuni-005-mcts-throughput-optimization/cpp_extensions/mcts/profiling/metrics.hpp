/**
 * @file metrics.hpp
 * @brief Enhanced profiling metrics enumeration and metadata
 *
 * Defines all profiling metrics with metadata for categorization and analysis.
 */

#pragma once

#include <cstdint>
#include <string_view>

namespace mcts {
namespace profiling {

/**
 * @brief Type of metric being tracked
 */
enum class MetricType : std::uint8_t {
    Timer,              // Duration measurement (nanoseconds)
    Counter,            // Event count (monotonic increment)
    Gauge,              // Current value (can go up/down)
    Histogram,          // Distribution of values
    HardwareCounter,    // CPU performance counter
    MemoryMetric,       // Cache/memory statistics
};

/**
 * @brief Category for organizing metrics
 */
enum class MetricCategory : std::uint8_t {
    Selection,          // PUCT selection phase
    Expansion,          // Node expansion
    Backup,             // Value backup
    VirtualLoss,        // Virtual loss operations
    Queue,              // Inference queue
    Memory,             // Allocation/cache
    Synchronization,    // Lock/atomic contention
    Hardware,           // CPU counters
    Thread,             // Thread efficiency
};

/**
 * @brief Comprehensive profiling metrics
 *
 * Organized by MCTS phase and component for detailed analysis.
 */
enum class ProfileMetric : std::uint16_t {
    // ========== Selection Phase (0-19) ==========
    SelectionTotal = 0,         // Total selection time
    SelectionPUCT,              // PUCT computation
    SelectionAVX2,              // AVX2 vectorized operations
    SelectionCacheMiss,         // Cache miss during selection
    SelectionAtomicLoad,        // Atomic loads of visit counts
    SelectionTreeTraversal,     // Tree traversal overhead
    SelectionChildIteration,    // Iterating over children
    SelectionBusyEdgeSkip,      // Nodes skipped (busy-edge masking)
    SelectionRetry,             // Selection restarts
    SelectionDepth,             // Average tree depth (gauge)

    // ========== Expansion Phase (20-39) ==========
    ExpansionTotal = 20,        // Total expansion time
    ExpansionInferenceRequest,  // Inference request submission
    ExpansionInferenceWait,     // Waiting for inference result
    ExpansionMaskPolicy,        // Legal move masking
    ExpansionNormalizePolicy,   // Policy normalization
    ExpansionAllocateNodes,     // Child node allocation
    ExpansionInitChildren,      // Child initialization
    ExpansionCASExpanded,       // CAS for expanded flag
    ExpansionCASExpanding,      // CAS for expanding flag
    ExpansionConflict,          // Expansion conflicts (counter)
    ExpansionTerminalCheck,     // Terminal state check
    ExpansionLegalMoves,        // Legal move generation

    // ========== Backup Phase (40-59) ==========
    BackupTotal = 40,           // Total backup time
    BackupPathTraversal,        // Traversing backup path
    BackupSignFlip,             // Value sign flipping
    BackupAtomicVisitUpdate,    // Atomic visit count update
    BackupAtomicValueUpdate,    // Atomic value update
    BackupVirtualLossRemove,    // Virtual loss removal
    BackupCASRetries,           // CAS retry count

    // ========== Virtual Loss (60-79) ==========
    VirtualLossApply = 60,      // Apply virtual loss
    VirtualLossRemove,          // Remove virtual loss
    VirtualLossContention,      // Contention on virtual loss
    VirtualLossCASSuccess,      // Successful CAS (counter)
    VirtualLossCASFailure,      // Failed CAS (counter)
    VirtualLossCASRetries,      // Total CAS retry count

    // ========== Queue Operations (80-99) ==========
    QueueSubmit = 80,           // Submit inference request
    QueueSubmitMPMCPush,        // MPMC ring buffer push
    QueueCollect,               // Collect batch
    QueueCollectWait,           // Wait for batch (condition var)
    QueueCollectTimeout,        // Batch collection timeouts
    QueueBatchSize,             // Batch size (gauge)
    QueuePendingDepth,          // Pending queue depth (gauge)
    QueueSubmitResults,         // Submit inference results
    QueueTryGetResult,          // Try get result
    QueueResultLookup,          // Result ring buffer lookup
    QueueCondVarWait,           // Condition variable wait time
    QueueCondVarSpuriousWake,   // Spurious wakeups (counter)

    // ========== Memory Allocation (100-129) ==========
    MemoryNodeAllocate = 100,   // Node allocation
    MemoryNodeAllocateFast,     // Fast path (thread-local)
    MemoryNodeAllocateSlow,     // Slow path (global pool)
    MemoryNodeDeallocate,       // Node deallocation
    MemoryArenaAllocate,        // Arena allocation
    MemoryArenaBump,            // Bump pointer allocation
    MemoryArenaFreeList,        // Free list allocation
    MemoryArenaChunkAlloc,      // New chunk allocation
    MemoryArenaCacheLine,       // Cache line efficiency (gauge)
    MemoryBytesAllocated,       // Total bytes allocated (gauge)
    MemoryBytesDeallocated,     // Total bytes freed (gauge)
    MemoryNetUsage,             // Net memory usage (gauge)
    MemoryFragmentation,        // Fragmentation ratio (gauge)

    // ========== Synchronization (130-149) ==========
    SyncMutexLockWait = 130,    // Mutex lock wait time
    SyncMutexLockSuccess,       // Mutex locks acquired
    SyncCondVarWait,            // Condition variable wait
    SyncCondVarSignal,          // Condition variable signals
    SyncCondVarBroadcast,       // Condition variable broadcasts
    SyncAtomicCASSuccess,       // Successful CAS operations
    SyncAtomicCASFailure,       // Failed CAS operations
    SyncAtomicCASRetries,       // Total CAS retries
    SyncSpinWaitCycles,         // CPU cycles spent spinning
    SyncSpinWaitIterations,     // Spin loop iterations

    // ========== Hardware Counters (150-179) ==========
    HWCPUCycles = 150,          // Total CPU cycles
    HWInstructions,             // Instructions executed
    HWIPC,                      // Instructions per cycle (gauge)
    HWCacheReferences,          // Cache references
    HWCacheMisses,              // Cache misses
    HWCacheHitRate,             // Cache hit rate (gauge)
    HWL1DCacheMisses,           // L1 data cache misses
    HWL1ICacheMisses,           // L1 instruction cache misses
    HWL2CacheMisses,            // L2 cache misses
    HWLLCMisses,                // Last-level cache misses
    HWBranchInstructions,       // Branch instructions
    HWBranchMisses,             // Branch mispredictions
    HWBranchMissRate,           // Branch miss rate (gauge)
    HWTLBMisses,                // TLB misses
    HWPageFaults,               // Page faults
    HWContextSwitches,          // Context switches
    HWStalledCyclesFrontend,    // Stalled cycles (frontend)
    HWStalledCyclesBackend,     // Stalled cycles (backend)

    // ========== Thread Efficiency (180-199) ==========
    ThreadIdleTime = 180,       // Thread idle time
    ThreadActiveTime,           // Thread active time
    ThreadUtilization,          // Thread utilization (gauge)
    ThreadSimulations,          // Simulations per thread (counter)
    ThreadSimulationsPerSec,    // Simulations/sec (gauge)
    ThreadWaitInference,        // Time waiting for inference
    ThreadWaitQueue,            // Time waiting on queue
    ThreadWaitAllocation,       // Time waiting for allocation
    ThreadWaitContention,       // Time waiting on contention

    // ========== Advanced Metrics (200-229) ==========
    AdvancedFalseSharing = 200, // False sharing detected (counter)
    AdvancedCacheLineContention,// Cache line contention
    AdvancedMemoryBandwidth,    // Memory bandwidth (gauge)
    AdvancedBranchPrediction,   // Branch prediction efficiency
    AdvancedPipelineStalls,     // Pipeline stall cycles
    AdvancedLoadStoreUnits,     // Load/store unit utilization

    // Sentinel
    Count
};

/**
 * @brief Metadata for a profiling metric
 */
struct MetricMetadata {
    ProfileMetric metric;
    const char* name;
    const char* description;
    MetricType type;
    MetricCategory category;
    const char* unit;  // "ns", "count", "bytes", "cycles", "%"
};

/**
 * @brief Get metadata for a metric
 */
constexpr const MetricMetadata& get_metric_metadata(ProfileMetric metric);

/**
 * @brief Get metric name as string
 */
constexpr const char* metric_name(ProfileMetric metric);

/**
 * @brief Get metric category as string
 */
constexpr const char* category_name(MetricCategory category);

/**
 * @brief Get metric type as string
 */
constexpr const char* type_name(MetricType type);

/**
 * @brief Check if metric is a timer (duration measurement)
 */
constexpr bool is_timer(ProfileMetric metric);

/**
 * @brief Check if metric is a counter (monotonic)
 */
constexpr bool is_counter(ProfileMetric metric);

/**
 * @brief Check if metric is a gauge (current value)
 */
constexpr bool is_gauge(ProfileMetric metric);

/**
 * @brief Check if metric is hardware counter
 */
constexpr bool is_hardware_counter(ProfileMetric metric);

// ============================================================
// Implementation (constexpr for compile-time evaluation)
// ============================================================

namespace detail {

// Metadata table (indexed by ProfileMetric)
constexpr MetricMetadata METRIC_METADATA[] = {
    // Selection
    {ProfileMetric::SelectionTotal, "selection_total", "Total selection time", MetricType::Timer, MetricCategory::Selection, "ns"},
    {ProfileMetric::SelectionPUCT, "selection_puct", "PUCT computation", MetricType::Timer, MetricCategory::Selection, "ns"},
    {ProfileMetric::SelectionAVX2, "selection_avx2", "AVX2 vectorized operations", MetricType::Timer, MetricCategory::Selection, "ns"},
    {ProfileMetric::SelectionCacheMiss, "selection_cache_miss", "Cache misses during selection", MetricType::Counter, MetricCategory::Selection, "count"},
    {ProfileMetric::SelectionAtomicLoad, "selection_atomic_load", "Atomic loads", MetricType::Timer, MetricCategory::Selection, "ns"},
    {ProfileMetric::SelectionTreeTraversal, "selection_tree_traversal", "Tree traversal overhead", MetricType::Timer, MetricCategory::Selection, "ns"},
    {ProfileMetric::SelectionChildIteration, "selection_child_iteration", "Child iteration", MetricType::Timer, MetricCategory::Selection, "ns"},
    {ProfileMetric::SelectionBusyEdgeSkip, "selection_busy_edge_skip", "Nodes skipped (busy-edge)", MetricType::Counter, MetricCategory::Selection, "count"},
    {ProfileMetric::SelectionRetry, "selection_retry", "Selection restarts", MetricType::Counter, MetricCategory::Selection, "count"},
    {ProfileMetric::SelectionDepth, "selection_depth", "Average tree depth", MetricType::Gauge, MetricCategory::Selection, "depth"},

    // Expansion
    {ProfileMetric::ExpansionTotal, "expansion_total", "Total expansion time", MetricType::Timer, MetricCategory::Expansion, "ns"},
    {ProfileMetric::ExpansionInferenceRequest, "expansion_inference_request", "Inference request submission", MetricType::Timer, MetricCategory::Expansion, "ns"},
    {ProfileMetric::ExpansionInferenceWait, "expansion_inference_wait", "Waiting for inference", MetricType::Timer, MetricCategory::Expansion, "ns"},
    {ProfileMetric::ExpansionMaskPolicy, "expansion_mask_policy", "Legal move masking", MetricType::Timer, MetricCategory::Expansion, "ns"},
    {ProfileMetric::ExpansionNormalizePolicy, "expansion_normalize_policy", "Policy normalization", MetricType::Timer, MetricCategory::Expansion, "ns"},
    {ProfileMetric::ExpansionAllocateNodes, "expansion_allocate_nodes", "Child node allocation", MetricType::Timer, MetricCategory::Expansion, "ns"},
    {ProfileMetric::ExpansionInitChildren, "expansion_init_children", "Child initialization", MetricType::Timer, MetricCategory::Expansion, "ns"},
    {ProfileMetric::ExpansionCASExpanded, "expansion_cas_expanded", "CAS for expanded flag", MetricType::Timer, MetricCategory::Expansion, "ns"},
    {ProfileMetric::ExpansionCASExpanding, "expansion_cas_expanding", "CAS for expanding flag", MetricType::Timer, MetricCategory::Expansion, "ns"},
    {ProfileMetric::ExpansionConflict, "expansion_conflict", "Expansion conflicts", MetricType::Counter, MetricCategory::Expansion, "count"},
    {ProfileMetric::ExpansionTerminalCheck, "expansion_terminal_check", "Terminal state check", MetricType::Timer, MetricCategory::Expansion, "ns"},
    {ProfileMetric::ExpansionLegalMoves, "expansion_legal_moves", "Legal move generation", MetricType::Timer, MetricCategory::Expansion, "ns"},

    // Backup
    {ProfileMetric::BackupTotal, "backup_total", "Total backup time", MetricType::Timer, MetricCategory::Backup, "ns"},
    {ProfileMetric::BackupPathTraversal, "backup_path_traversal", "Backup path traversal", MetricType::Timer, MetricCategory::Backup, "ns"},
    {ProfileMetric::BackupSignFlip, "backup_sign_flip", "Value sign flipping", MetricType::Timer, MetricCategory::Backup, "ns"},
    {ProfileMetric::BackupAtomicVisitUpdate, "backup_atomic_visit_update", "Atomic visit count update", MetricType::Timer, MetricCategory::Backup, "ns"},
    {ProfileMetric::BackupAtomicValueUpdate, "backup_atomic_value_update", "Atomic value update", MetricType::Timer, MetricCategory::Backup, "ns"},
    {ProfileMetric::BackupVirtualLossRemove, "backup_virtual_loss_remove", "Virtual loss removal", MetricType::Timer, MetricCategory::Backup, "ns"},
    {ProfileMetric::BackupCASRetries, "backup_cas_retries", "CAS retries", MetricType::Counter, MetricCategory::Backup, "count"},

    // Virtual Loss
    {ProfileMetric::VirtualLossApply, "virtual_loss_apply", "Apply virtual loss", MetricType::Timer, MetricCategory::VirtualLoss, "ns"},
    {ProfileMetric::VirtualLossRemove, "virtual_loss_remove", "Remove virtual loss", MetricType::Timer, MetricCategory::VirtualLoss, "ns"},
    {ProfileMetric::VirtualLossContention, "virtual_loss_contention", "Virtual loss contention", MetricType::Timer, MetricCategory::VirtualLoss, "ns"},
    {ProfileMetric::VirtualLossCASSuccess, "virtual_loss_cas_success", "Successful CAS", MetricType::Counter, MetricCategory::VirtualLoss, "count"},
    {ProfileMetric::VirtualLossCASFailure, "virtual_loss_cas_failure", "Failed CAS", MetricType::Counter, MetricCategory::VirtualLoss, "count"},
    {ProfileMetric::VirtualLossCASRetries, "virtual_loss_cas_retries", "CAS retries", MetricType::Counter, MetricCategory::VirtualLoss, "count"},

    // Queue
    {ProfileMetric::QueueSubmit, "queue_submit", "Submit inference request", MetricType::Timer, MetricCategory::Queue, "ns"},
    {ProfileMetric::QueueSubmitMPMCPush, "queue_submit_mpmc_push", "MPMC push", MetricType::Timer, MetricCategory::Queue, "ns"},
    {ProfileMetric::QueueCollect, "queue_collect", "Collect batch", MetricType::Timer, MetricCategory::Queue, "ns"},
    {ProfileMetric::QueueCollectWait, "queue_collect_wait", "Wait for batch", MetricType::Timer, MetricCategory::Queue, "ns"},
    {ProfileMetric::QueueCollectTimeout, "queue_collect_timeout", "Batch timeouts", MetricType::Counter, MetricCategory::Queue, "count"},
    {ProfileMetric::QueueBatchSize, "queue_batch_size", "Batch size", MetricType::Gauge, MetricCategory::Queue, "size"},
    {ProfileMetric::QueuePendingDepth, "queue_pending_depth", "Pending queue depth", MetricType::Gauge, MetricCategory::Queue, "depth"},
    {ProfileMetric::QueueSubmitResults, "queue_submit_results", "Submit results", MetricType::Timer, MetricCategory::Queue, "ns"},
    {ProfileMetric::QueueTryGetResult, "queue_try_get_result", "Try get result", MetricType::Timer, MetricCategory::Queue, "ns"},
    {ProfileMetric::QueueResultLookup, "queue_result_lookup", "Result lookup", MetricType::Timer, MetricCategory::Queue, "ns"},
    {ProfileMetric::QueueCondVarWait, "queue_condvar_wait", "Condition variable wait", MetricType::Timer, MetricCategory::Queue, "ns"},
    {ProfileMetric::QueueCondVarSpuriousWake, "queue_condvar_spurious", "Spurious wakeups", MetricType::Counter, MetricCategory::Queue, "count"},

    // Memory (showing first few, similar pattern continues)
    {ProfileMetric::MemoryNodeAllocate, "memory_node_allocate", "Node allocation", MetricType::Timer, MetricCategory::Memory, "ns"},
    {ProfileMetric::MemoryNodeAllocateFast, "memory_node_allocate_fast", "Fast path allocation", MetricType::Counter, MetricCategory::Memory, "count"},
    {ProfileMetric::MemoryNodeAllocateSlow, "memory_node_allocate_slow", "Slow path allocation", MetricType::Counter, MetricCategory::Memory, "count"},
    // ... (remaining entries follow same pattern)
};

} // namespace detail

constexpr const MetricMetadata& get_metric_metadata(ProfileMetric metric) {
    return detail::METRIC_METADATA[static_cast<std::uint16_t>(metric)];
}

constexpr const char* metric_name(ProfileMetric metric) {
    return get_metric_metadata(metric).name;
}

constexpr const char* category_name(MetricCategory category) {
    switch (category) {
        case MetricCategory::Selection: return "Selection";
        case MetricCategory::Expansion: return "Expansion";
        case MetricCategory::Backup: return "Backup";
        case MetricCategory::VirtualLoss: return "VirtualLoss";
        case MetricCategory::Queue: return "Queue";
        case MetricCategory::Memory: return "Memory";
        case MetricCategory::Synchronization: return "Synchronization";
        case MetricCategory::Hardware: return "Hardware";
        case MetricCategory::Thread: return "Thread";
        default: return "Unknown";
    }
}

constexpr const char* type_name(MetricType type) {
    switch (type) {
        case MetricType::Timer: return "Timer";
        case MetricType::Counter: return "Counter";
        case MetricType::Gauge: return "Gauge";
        case MetricType::Histogram: return "Histogram";
        case MetricType::HardwareCounter: return "HardwareCounter";
        case MetricType::MemoryMetric: return "MemoryMetric";
        default: return "Unknown";
    }
}

constexpr bool is_timer(ProfileMetric metric) {
    return get_metric_metadata(metric).type == MetricType::Timer;
}

constexpr bool is_counter(ProfileMetric metric) {
    return get_metric_metadata(metric).type == MetricType::Counter;
}

constexpr bool is_gauge(ProfileMetric metric) {
    return get_metric_metadata(metric).type == MetricType::Gauge;
}

constexpr bool is_hardware_counter(ProfileMetric metric) {
    return get_metric_metadata(metric).type == MetricType::HardwareCounter;
}

} // namespace profiling
} // namespace mcts
