/**
 * @file enhanced_metrics.hpp
 * @brief Comprehensive metric definitions for MCTS profiling
 * @author MCTS Performance Team
 * @date 2024
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <array>
#include <unordered_map>

namespace mcts {
namespace profiling {

/**
 * @brief Types of profiling metrics
 */
enum class MetricType : uint8_t {
    Timer,          // Duration measurement
    Counter,        // Event counting
    Gauge,          // Current value measurement
    Histogram,      // Distribution measurement
    HardwareCounter // CPU performance counter
};

/**
 * @brief Categories for organizing metrics
 */
enum class MetricCategory : uint8_t {
    Selection,
    Expansion,
    Backup,
    VirtualLoss,
    Queue,
    Memory,
    Synchronization,
    Hardware,
    Thread,
    Pipeline,
    GPU,
    Network
};

/**
 * @brief Comprehensive metric enumeration
 */
enum class ProfileMetric : uint16_t {
    // Selection Phase (0-19)
    SelectionTotal = 0,
    SelectionPUCT,
    SelectionTreeTraversal,
    SelectionChildSelection,
    SelectionVirtualLossWait,
    SelectionBusyEdgeSkip,
    SelectionDepth,
    SelectionRetries,
    SelectionCacheHits,
    SelectionCacheMisses,
    SelectionAVX2Operations,
    SelectionBranchMispredictions,
    // T024f-6: Make/Unmake pattern metrics
    SelectionMakeMove,           // Time for make_move() calls during selection
    SelectionUnmakeMove,         // Time for unmake_move() calls during unwind
    SelectionMakeMoveCount,      // Counter: number of make_move calls
    SelectionUnmakeMoveCount,    // Counter: number of unmake_move calls
    SelectionThreadLocalInit,    // Time for thread-local state initialization
    SelectionActualStateClone,   // Time for actual state.clone() calls (should be rare)

    // Expansion Phase (20-39)
    ExpansionTotal = 20,
    ExpansionStateClone,
    ExpansionLegalMoveGen,
    ExpansionNeuralNetRequest,
    ExpansionNeuralNetWait,
    ExpansionPolicyMasking,
    ExpansionNodeAllocation,
    ExpansionChildInit,
    ExpansionConflicts,
    ExpansionMemoryAlloc,
    ExpansionDLPackConversion,
    ExpansionTensorCreation,

    // Backup Phase (40-54)
    BackupTotal = 40,
    BackupPathTraversal,
    BackupValueUpdate,
    BackupVirtualLossRemoval,
    BackupAtomicOperations,
    BackupSignFlipping,
    BackupCASRetries,

    // Virtual Loss (55-64)
    VirtualLossApply = 55,
    VirtualLossRemove,
    VirtualLossContention,
    VirtualLossCASRetries,
    VirtualLossSpinWait,

    // Queue Operations (65-84)
    QueueSubmitTotal = 65,
    QueueSubmitEnqueue,
    QueueSubmitRetries,
    QueueCollectTotal,
    QueueCollectWait,
    QueueCollectBatchSize,
    QueueProcessResults,
    QueueTryGetResult,
    QueueConditionWait,
    QueueSpuriousWakeups,
    QueueRingBufferFull,
    QueueDLPackSubmit,

    // Memory Operations (85-104)
    MemoryNodeAllocation = 85,
    MemoryNodeDeallocation,
    MemoryArenaAllocation,
    MemoryArenaContention,
    MemoryFragmentation,
    MemoryPageFaults,
    MemoryCacheLineLoads,
    MemoryCacheLineEvictions,
    MemoryFalseSharing,
    MemoryBandwidthUsed,
    MemoryTLBMisses,
    MemoryPrefetchHits,

    // Synchronization (105-119)
    SyncMutexLockWait = 105,
    SyncMutexLockAcquired,
    SyncAtomicCAS,
    SyncAtomicCASRetries,
    SyncSpinlockSpins,
    SyncCondvarWait,
    SyncCondvarSignal,
    SyncBarrierWait,
    SyncFutexWait,
    SyncRWLockRead,
    SyncRWLockWrite,

    // Hardware Counters (120-144)
    HWCPUCycles = 120,
    HWInstructions,
    HWIPC,
    HWCacheMissesL1D,
    HWCacheMissesL1I,
    HWCacheMissesL2,
    HWCacheMissesLLC,
    HWBranchMisses,
    HWBranchInstructions,
    HWPageFaults,
    HWContextSwitches,
    HWCPUMigrations,
    HWTLBMissesData,
    HWTLBMissesInstruction,
    HWStalledCyclesFrontend,
    HWStalledCyclesBackend,
    HWBusTransactions,
    HWRemoteAccesses,

    // Thread Metrics (145-164)
    ThreadActiveTime = 145,
    ThreadIdleTime,
    ThreadWaitTime,
    ThreadUserTime,
    ThreadSystemTime,
    ThreadVoluntaryCtxSwitch,
    ThreadInvoluntaryCtxSwitch,
    ThreadCPUMigrations,
    ThreadWorkStolen,
    ThreadTasksCompleted,
    ThreadQueueDepth,
    ThreadAffinity,

    // Pipeline Metrics (165-184)
    PipelineE2ELatency = 165,
    PipelineStageSelection,
    PipelineStageExpansion,
    PipelineStageBackup,
    PipelineStageInference,
    PipelineQueueDepth,
    PipelineThroughput,
    PipelineUtilization,
    PipelineStalls,
    PipelineBottleneck,

    // GPU Metrics (185-204)
    GPUInferenceTime = 185,
    GPUKernelTime,
    GPUH2DTransfer,
    GPUD2HTransfer,
    GPUMemoryUsed,
    GPUUtilization,
    GPUSMOccupancy,
    GPUTensorCoreUsage,
    GPUBatchSize,
    GPUQueueDepth,
    GPUPowerUsage,
    GPUTemperature,
    GPUThrottling,

    // Network/Communication (205-219)
    NetInterThreadComm = 205,
    NetCacheSyncs,
    NetMemoryBarriers,
    NetAtomicBroadcasts,
    NetDMATransfers,
    NetPCIeBandwidth,

    // Aggregate/Derived Metrics (220-239)
    TotalSimulations = 220,
    SimulationsPerSecond,
    EffectiveBranchingFactor,
    TreeEfficiency,
    ParallelEfficiency,
    ContentionFactor,
    CacheEfficiency,
    MemoryEfficiency,

    // === EXTENDED METRICS FOR BOTTLENECK ANALYSIS (240-299) ===
    // Based on review.txt bottleneck analysis and profiling gap analysis

    // State Management (240-249) - CRITICAL from review.txt lines 37-54
    StateCloneStart = 240,
    StateCloneTotal,
    StateCloneCount,
    StateCloneBytes,
    StatePoolHit,
    StatePoolMiss,
    StatePoolAllocation,
    StateCopyFrom,
    StateDestructorTime,

    // Feature Extraction (250-259) - CRITICAL from review.txt lines 22-34
    FeatureExtractionTotal = 250,
    FeatureExtractionPerState,
    FeatureExtractionOpenMP,
    FeatureExtractionSerial,
    OMP_ThreadCount,
    OMP_WorkDistribution,
    OMP_BarrierWait,
    TensorCreationOverhead,

    // Thread Idle Time (260-269) - CRITICAL from review.txt lines 71-136
    ThreadIdleTotal = 260,
    ThreadWaitingForResults,
    ThreadSleepCycles,
    ThreadSpinWaitCycles,
    ThreadYieldCount,
    ThreadBlockedOnMutex,
    ThreadBlockedOnCondVar,
    ThreadBlockedOnAtomic,

    // Synchronization Contention (270-279) - HIGH from review.txt lines 225-236
    MutexLockWaitTime = 270,
    MutexContentionEvents,
    CAS_SuccessCount,
    CAS_FailureCount,
    CAS_RetryCount,
    CAS_MaxRetriesPerOp,
    AtomicLoadStalls,
    AtomicStoreStalls,

    // Allocation Contention (280-284) - HIGH from review.txt
    AllocationMutexWait = 280,
    AllocationFastPath,
    AllocationSlowPath,
    AllocationContentionRatio,

    // Python Bridge (285-294) - HIGH from review.txt lines 258-307
    PythonCallbackEntry = 285,
    PythonCallbackExit,
    PythonCallbackTotal,
    GIL_AcquisitionTime,
    GIL_HoldTime,
    GIL_ReleaseTime,
    PythonObjectCreation,
    PythonObjectDestruction,
    DLPackCapsuleCreation,

    // === PROFILING FRAMEWORK UPGRADE (295-340) ===
    // Added 2025-10-17 to eliminate 81-98% "unknown" time
    // See: PROFILING_FRAMEWORK_UPGRADE_SPEC.md

    // Coordinator Thread Metrics (295-310) - CRITICAL (separate thread, ZERO previous instrumentation)
    CoordinatorLoopTotal = 295,          // Total time in coordinator_loop() - CRITICAL
    CoordinatorLoopIteration,            // Single loop iteration time
    CoordinatorCollectBatch,             // Time blocked in queue.collect_batch()
    CoordinatorCollectBatchEmpty,        // Iterations where batch was empty
    CoordinatorFeatureExtraction,        // Time extracting features (total)
    CoordinatorFeatureExtractionOMP,     // Time in OpenMP parallel region
    CoordinatorFeatureExtractionSerial,  // Time in serial extraction fallback
    CoordinatorFeatureAllocation,        // Time allocating/resizing feature vectors
    CoordinatorPythonCallback,           // Time in callback->batch_inference_features() - CRITICAL
    CoordinatorGILWait,                  // Time waiting for GIL (if measurable)
    CoordinatorResultSubmit,             // Time submitting results to queue
    CoordinatorIdleTime,                 // Time when no requests available
    CoordinatorFallbackPath,             // Time in fallback result generation
    CoordinatorOMPThreadCount,           // Number of threads used by OpenMP
    CoordinatorOMPOverhead,              // OpenMP parallelization overhead
    CoordinatorBatchCount,               // Counter: Number of batches processed

    // ContinuousRunner Main Loop (311-330) - CRITICAL (main execution, minimal previous instrumentation)
    RunContinuousLoopTotal = 311,        // Total loop iteration time - CRITICAL
    RunContinuousLoopIteration,          // Single loop iteration
    RunContinuousPhase1,                 // Phase 1: Select to leaf and submit
    RunContinuousPhase2,                 // Phase 2: Process completed results
    RunContinuousQueueClone,             // State cloning for queue submission - CRITICAL SUSPECTED BOTTLENECK
    RunContinuousQueueSubmit,            // Queue.submit_request() call time
    RunContinuousPendingBuffer,          // Pending buffer management time
    RunContinuousBackoffLoop,            // Time in backoff/wait loops when queue full
    RunContinuousSleepYield,             // Time sleeping/yielding - CRITICAL (idle time)
    RunContinuousThreadLocalInit,        // Thread-local state initialization
    RunContinuousStateRestore,           // Time restoring state to root via unwind
    RunContinuousSubmitReady,            // Time preparing submission
    RunContinuousQueueFullWait,          // Time waiting when queue is full
    RunContinuousTerminalBackup,         // Time backing up terminal nodes
    RunContinuousExpansionConflict,      // Time handling expansion conflicts
    RunContinuousLoopIdleCount,          // Counter: Iterations with no work done
    RunContinuousLoopWorkCount,          // Counter: Iterations with work done

    // Root Expansion (331-335) - HIGH (synchronous wait, minimal previous instrumentation)
    RootExpansionTotal = 331,            // Total root expansion time
    RootExpansionWaitInference,          // Synchronous wait for inference result
    RootExpansionClone,                  // Root state cloning
    RootExpansionDirichlet,              // Dirichlet noise generation
    RootExpansionAtomicRace,             // Time handling atomic flag races

    // Node Expansion Details (336-342) - MEDIUM (called frequently, minimal breakdown)
    NodeExpansionPolicyMask = 336,       // Policy masking/normalization time
    NodeExpansionPolicyNormalize,        // Policy normalization specifically
    NodeExpansionChildAlloc,             // Child node allocation time
    NodeExpansionChildInit,              // Child initialization loop time
    NodeExpansionAtomicFlag,             // Atomic flag operations time
    NodeExpansionFlagCheck,              // Time checking if already expanded
    NodeExpansionLegalMoves,             // Time getting legal moves

    // Batch Result Processing Phases (343-350) - HIGH (called frequently, NO phase breakdown)
    BatchResultsTotal = 343,             // Total process_completed_results() time
    BatchResultsCollect,                 // Phase 1: Collect ready results from buffer
    BatchResultsExpand,                  // Phase 2: Expand all nodes
    BatchResultsBackupPrep,              // Phase 3: Prepare batched backup data
    BatchResultsAtomicUpdate,            // Phase 4: Apply atomic tree updates
    BatchResultsClearFlags,              // Phase 5: Clear expanding flags
    BatchResultsReturnStates,            // Phase 6: Return states to pool
    BatchResultsPathReversal,            // Time reversing paths for backup

    // State Management Detail (351-355) - CRITICAL (suspected primary bottleneck)
    StateCloneForQueue = 351,            // State cloning specifically for queue submission
    StateCloneForRoot,                   // State cloning specifically for root expansion
    StateCloneForPool,                   // State cloning for pool allocation
    StateUnwindPath,                     // Time unwinding path (make/unmake)
    StateThreadLocalClone,               // One-time thread-local state initialization clone

    // Async Coordination Detail (356-360) - MEDIUM (async overhead tracking)
    AsyncSubmissionOverhead = 356,       // Overhead of async submission vs direct call
    AsyncResultWaitTime,                 // Time waiting for async results
    AsyncQueueLatency,                   // Request submission to result ready latency
    AsyncCoordinatorLatency,             // Coordinator processing latency (batch collection to result submit)
    AsyncPipelineBubbles,                // Pipeline stalls/bubbles (idle time in pipeline)

    // Sentinel
    MetricCount = 361
};

/**
 * @brief Metadata for each metric
 */
struct MetricMetadata {
    const char* name;
    const char* description;
    MetricType type;
    MetricCategory category;
    const char* unit;
    bool is_derived;  // Computed from other metrics
};

/**
 * @brief Get metadata for a metric
 */
constexpr MetricMetadata get_metric_metadata(ProfileMetric metric) {
    switch (metric) {
        // Selection metrics
        case ProfileMetric::SelectionTotal:
            return {"selection_total", "Total selection time", MetricType::Timer, MetricCategory::Selection, "ns", false};
        case ProfileMetric::SelectionPUCT:
            return {"selection_puct", "PUCT computation time", MetricType::Timer, MetricCategory::Selection, "ns", false};
        case ProfileMetric::SelectionTreeTraversal:
            return {"selection_traversal", "Tree traversal time", MetricType::Timer, MetricCategory::Selection, "ns", false};
        case ProfileMetric::SelectionBusyEdgeSkip:
            return {"selection_busy_skip", "Nodes skipped due to busy edges", MetricType::Counter, MetricCategory::Selection, "count", false};
        case ProfileMetric::SelectionDepth:
            return {"selection_depth", "Selection depth reached", MetricType::Gauge, MetricCategory::Selection, "levels", false};

        // T024f-6: Make/unmake pattern metrics
        case ProfileMetric::SelectionMakeMove:
            return {"selection_make_move", "make_move() call time", MetricType::Timer, MetricCategory::Selection, "ns", false};
        case ProfileMetric::SelectionUnmakeMove:
            return {"selection_unmake_move", "unmake_move() call time", MetricType::Timer, MetricCategory::Selection, "ns", false};
        case ProfileMetric::SelectionMakeMoveCount:
            return {"selection_make_move_count", "Number of make_move calls", MetricType::Counter, MetricCategory::Selection, "count", false};
        case ProfileMetric::SelectionUnmakeMoveCount:
            return {"selection_unmake_move_count", "Number of unmake_move calls", MetricType::Counter, MetricCategory::Selection, "count", false};
        case ProfileMetric::SelectionThreadLocalInit:
            return {"selection_thread_local_init", "Thread-local state initialization", MetricType::Timer, MetricCategory::Selection, "ns", false};
        case ProfileMetric::SelectionActualStateClone:
            return {"selection_actual_state_clone", "Actual state.clone() calls", MetricType::Timer, MetricCategory::Selection, "ns", false};

        // Expansion metrics
        case ProfileMetric::ExpansionTotal:
            return {"expansion_total", "Total expansion time", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::ExpansionNeuralNetWait:
            return {"expansion_nn_wait", "Neural network wait time", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::ExpansionConflicts:
            return {"expansion_conflicts", "Expansion conflicts detected", MetricType::Counter, MetricCategory::Expansion, "count", false};

        // Backup metrics
        case ProfileMetric::BackupTotal:
            return {"backup_total", "Total backup time", MetricType::Timer, MetricCategory::Backup, "ns", false};
        case ProfileMetric::BackupCASRetries:
            return {"backup_cas_retries", "CAS retry count during backup", MetricType::Counter, MetricCategory::Backup, "count", false};

        // Virtual Loss metrics
        case ProfileMetric::VirtualLossContention:
            return {"vl_contention", "Virtual loss contention events", MetricType::Counter, MetricCategory::VirtualLoss, "count", false};

        // Queue metrics
        case ProfileMetric::QueueCollectBatchSize:
            return {"queue_batch_size", "Collected batch size", MetricType::Histogram, MetricCategory::Queue, "count", false};
        case ProfileMetric::QueueConditionWait:
            return {"queue_cv_wait", "Condition variable wait time", MetricType::Timer, MetricCategory::Queue, "ns", false};

        // Memory metrics
        case ProfileMetric::MemoryBandwidthUsed:
            return {"memory_bandwidth", "Memory bandwidth utilization", MetricType::Gauge, MetricCategory::Memory, "GB/s", false};
        case ProfileMetric::MemoryFalseSharing:
            return {"memory_false_sharing", "False sharing events detected", MetricType::Counter, MetricCategory::Memory, "count", false};

        // Hardware counter metrics
        case ProfileMetric::HWIPC:
            return {"hw_ipc", "Instructions per cycle", MetricType::Gauge, MetricCategory::Hardware, "IPC", true};
        case ProfileMetric::HWCacheMissesL1D:
            return {"hw_l1d_misses", "L1 data cache misses", MetricType::Counter, MetricCategory::Hardware, "count", false};

        // Thread metrics
        case ProfileMetric::ThreadActiveTime:
            return {"thread_active", "Thread active time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::ThreadWorkStolen:
            return {"thread_work_stolen", "Work stealing events", MetricType::Counter, MetricCategory::Thread, "count", false};

        // GPU metrics
        case ProfileMetric::GPUInferenceTime:
            return {"gpu_inference", "GPU inference time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::GPUUtilization:
            return {"gpu_utilization", "GPU SM utilization", MetricType::Gauge, MetricCategory::GPU, "%", false};

        // Derived metrics
        case ProfileMetric::SimulationsPerSecond:
            return {"sims_per_sec", "Simulations per second", MetricType::Gauge, MetricCategory::Pipeline, "sims/s", true};
        case ProfileMetric::ParallelEfficiency:
            return {"parallel_efficiency", "Parallel execution efficiency", MetricType::Gauge, MetricCategory::Thread, "%", true};

        // === Extended Metrics for Bottleneck Analysis ===

        // State Management (review.txt lines 37-54)
        case ProfileMetric::StateCloneStart:
            return {"state_clone_start", "State clone start timestamp", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::StateCloneTotal:
            return {"state_clone_total", "Total state cloning time", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::StateCloneCount:
            return {"state_clone_count", "Number of state clones", MetricType::Counter, MetricCategory::Memory, "count", false};
        case ProfileMetric::StateCloneBytes:
            return {"state_clone_bytes", "Bytes cloned per state", MetricType::Gauge, MetricCategory::Memory, "bytes", false};
        case ProfileMetric::StatePoolHit:
            return {"state_pool_hit", "State pool cache hits", MetricType::Counter, MetricCategory::Memory, "count", false};
        case ProfileMetric::StatePoolMiss:
            return {"state_pool_miss", "State pool cache misses", MetricType::Counter, MetricCategory::Memory, "count", false};
        case ProfileMetric::StatePoolAllocation:
            return {"state_pool_alloc", "State pool allocations", MetricType::Counter, MetricCategory::Memory, "count", false};
        case ProfileMetric::StateCopyFrom:
            return {"state_copy_from", "State copy_from() calls", MetricType::Counter, MetricCategory::Memory, "count", false};
        case ProfileMetric::StateDestructorTime:
            return {"state_destructor", "State destructor time", MetricType::Timer, MetricCategory::Memory, "ns", false};

        // Feature Extraction (review.txt lines 22-34)
        case ProfileMetric::FeatureExtractionTotal:
            return {"feature_extract_total", "Total feature extraction time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::FeatureExtractionPerState:
            return {"feature_extract_per_state", "Per-state extraction time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::FeatureExtractionOpenMP:
            return {"feature_extract_omp", "OpenMP parallelized extraction", MetricType::Counter, MetricCategory::GPU, "bool", false};
        case ProfileMetric::FeatureExtractionSerial:
            return {"feature_extract_serial", "Serial extraction time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::OMP_ThreadCount:
            return {"omp_thread_count", "OpenMP thread count", MetricType::Gauge, MetricCategory::Thread, "count", false};
        case ProfileMetric::OMP_WorkDistribution:
            return {"omp_work_dist", "OpenMP work distribution variance", MetricType::Gauge, MetricCategory::Thread, "%", false};
        case ProfileMetric::OMP_BarrierWait:
            return {"omp_barrier_wait", "OpenMP barrier wait time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::TensorCreationOverhead:
            return {"tensor_creation_overhead", "DLPack tensor creation time", MetricType::Timer, MetricCategory::GPU, "ns", false};

        // Thread Idle Time (review.txt lines 71-136)
        case ProfileMetric::ThreadIdleTotal:
            return {"thread_idle_total", "Total thread idle time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::ThreadWaitingForResults:
            return {"thread_wait_results", "Thread waiting for inference results", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::ThreadSleepCycles:
            return {"thread_sleep_cycles", "Thread sleep cycles", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::ThreadSpinWaitCycles:
            return {"thread_spin_wait", "Thread spin-wait cycles", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::ThreadYieldCount:
            return {"thread_yield_count", "Thread yield count", MetricType::Counter, MetricCategory::Thread, "count", false};
        case ProfileMetric::ThreadBlockedOnMutex:
            return {"thread_blocked_mutex", "Thread blocked on mutex", MetricType::Timer, MetricCategory::Synchronization, "ns", false};
        case ProfileMetric::ThreadBlockedOnCondVar:
            return {"thread_blocked_condvar", "Thread blocked on condition variable", MetricType::Timer, MetricCategory::Synchronization, "ns", false};
        case ProfileMetric::ThreadBlockedOnAtomic:
            return {"thread_blocked_atomic", "Thread blocked on atomic operation", MetricType::Timer, MetricCategory::Synchronization, "ns", false};

        // Synchronization Contention (review.txt lines 225-236)
        case ProfileMetric::MutexLockWaitTime:
            return {"mutex_lock_wait", "Mutex lock wait time", MetricType::Timer, MetricCategory::Synchronization, "ns", false};
        case ProfileMetric::MutexContentionEvents:
            return {"mutex_contention", "Mutex contention events", MetricType::Counter, MetricCategory::Synchronization, "count", false};
        case ProfileMetric::CAS_SuccessCount:
            return {"cas_success", "CAS operations succeeded", MetricType::Counter, MetricCategory::Synchronization, "count", false};
        case ProfileMetric::CAS_FailureCount:
            return {"cas_failure", "CAS operations failed", MetricType::Counter, MetricCategory::Synchronization, "count", false};
        case ProfileMetric::CAS_RetryCount:
            return {"cas_retry", "CAS retry count", MetricType::Counter, MetricCategory::Synchronization, "count", false};
        case ProfileMetric::CAS_MaxRetriesPerOp:
            return {"cas_max_retries", "CAS max retries per operation", MetricType::Gauge, MetricCategory::Synchronization, "count", false};
        case ProfileMetric::AtomicLoadStalls:
            return {"atomic_load_stalls", "Atomic load stalls", MetricType::Counter, MetricCategory::Synchronization, "count", false};
        case ProfileMetric::AtomicStoreStalls:
            return {"atomic_store_stalls", "Atomic store stalls", MetricType::Counter, MetricCategory::Synchronization, "count", false};

        // Allocation Contention (review.txt)
        case ProfileMetric::AllocationMutexWait:
            return {"alloc_mutex_wait", "Allocation mutex wait time", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::AllocationFastPath:
            return {"alloc_fast_path", "Fast path allocations (thread-local)", MetricType::Counter, MetricCategory::Memory, "count", false};
        case ProfileMetric::AllocationSlowPath:
            return {"alloc_slow_path", "Slow path allocations (global mutex)", MetricType::Counter, MetricCategory::Memory, "count", false};
        case ProfileMetric::AllocationContentionRatio:
            return {"alloc_contention_ratio", "Allocation contention ratio", MetricType::Gauge, MetricCategory::Memory, "%", true};

        // Python Bridge (review.txt lines 258-307)
        case ProfileMetric::PythonCallbackEntry:
            return {"python_callback_entry", "Python callback entry count", MetricType::Counter, MetricCategory::GPU, "count", false};
        case ProfileMetric::PythonCallbackExit:
            return {"python_callback_exit", "Python callback exit count", MetricType::Counter, MetricCategory::GPU, "count", false};
        case ProfileMetric::PythonCallbackTotal:
            return {"python_callback_total", "Total Python callback time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::GIL_AcquisitionTime:
            return {"gil_acquire_time", "GIL acquisition time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::GIL_HoldTime:
            return {"gil_hold_time", "GIL hold time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::GIL_ReleaseTime:
            return {"gil_release_time", "GIL release time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::PythonObjectCreation:
            return {"py_object_creation", "Python object creation count", MetricType::Counter, MetricCategory::GPU, "count", false};
        case ProfileMetric::PythonObjectDestruction:
            return {"py_object_destruction", "Python object destruction count", MetricType::Counter, MetricCategory::GPU, "count", false};
        case ProfileMetric::DLPackCapsuleCreation:
            return {"dlpack_capsule_creation", "DLPack capsule creation time", MetricType::Timer, MetricCategory::GPU, "ns", false};

        // === PROFILING FRAMEWORK UPGRADE METADATA (295-360) ===
        // Coordinator Thread Metrics
        case ProfileMetric::CoordinatorLoopTotal:
            return {"coordinator_loop_total", "Coordinator thread loop total time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::CoordinatorLoopIteration:
            return {"coordinator_loop_iteration", "Single coordinator loop iteration", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::CoordinatorCollectBatch:
            return {"coordinator_collect_batch", "Coordinator batch collection time", MetricType::Timer, MetricCategory::Queue, "ns", false};
        case ProfileMetric::CoordinatorCollectBatchEmpty:
            return {"coordinator_batch_empty", "Empty batch iterations", MetricType::Counter, MetricCategory::Queue, "count", false};
        case ProfileMetric::CoordinatorFeatureExtraction:
            return {"coordinator_feature_extraction", "Feature extraction total time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::CoordinatorFeatureExtractionOMP:
            return {"coordinator_feature_omp", "OpenMP feature extraction time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::CoordinatorFeatureExtractionSerial:
            return {"coordinator_feature_serial", "Serial feature extraction time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::CoordinatorFeatureAllocation:
            return {"coordinator_feature_alloc", "Feature buffer allocation time", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::CoordinatorPythonCallback:
            return {"coordinator_python_callback", "Python inference callback time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::CoordinatorGILWait:
            return {"coordinator_gil_wait", "GIL wait time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::CoordinatorResultSubmit:
            return {"coordinator_result_submit", "Result submission time", MetricType::Timer, MetricCategory::Queue, "ns", false};
        case ProfileMetric::CoordinatorIdleTime:
            return {"coordinator_idle_time", "Coordinator idle time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::CoordinatorFallbackPath:
            return {"coordinator_fallback", "Fallback result generation time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::CoordinatorOMPThreadCount:
            return {"coordinator_omp_threads", "OpenMP thread count", MetricType::Gauge, MetricCategory::Thread, "count", false};
        case ProfileMetric::CoordinatorOMPOverhead:
            return {"coordinator_omp_overhead", "OpenMP overhead time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::CoordinatorBatchCount:
            return {"coordinator_batch_count", "Batches processed", MetricType::Counter, MetricCategory::Queue, "count", false};

        // ContinuousRunner Main Loop
        case ProfileMetric::RunContinuousLoopTotal:
            return {"run_continuous_loop_total", "Main loop total time", MetricType::Timer, MetricCategory::Pipeline, "ns", false};
        case ProfileMetric::RunContinuousLoopIteration:
            return {"run_continuous_loop_iter", "Single loop iteration time", MetricType::Timer, MetricCategory::Pipeline, "ns", false};
        case ProfileMetric::RunContinuousPhase1:
            return {"run_continuous_phase1", "Phase 1: Select and submit", MetricType::Timer, MetricCategory::Pipeline, "ns", false};
        case ProfileMetric::RunContinuousPhase2:
            return {"run_continuous_phase2", "Phase 2: Process results", MetricType::Timer, MetricCategory::Pipeline, "ns", false};
        case ProfileMetric::RunContinuousQueueClone:
            return {"run_continuous_queue_clone", "State cloning for queue", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::RunContinuousQueueSubmit:
            return {"run_continuous_queue_submit", "Queue submission time", MetricType::Timer, MetricCategory::Queue, "ns", false};
        case ProfileMetric::RunContinuousPendingBuffer:
            return {"run_continuous_pending_buffer", "Pending buffer management", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::RunContinuousBackoffLoop:
            return {"run_continuous_backoff", "Backoff loop time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::RunContinuousSleepYield:
            return {"run_continuous_sleep", "Sleep/yield time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::RunContinuousThreadLocalInit:
            return {"run_continuous_tls_init", "Thread-local init time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::RunContinuousStateRestore:
            return {"run_continuous_state_restore", "State restore time", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::RunContinuousSubmitReady:
            return {"run_continuous_submit_ready", "Submission prep time", MetricType::Timer, MetricCategory::Queue, "ns", false};
        case ProfileMetric::RunContinuousQueueFullWait:
            return {"run_continuous_queue_full", "Queue full wait time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::RunContinuousTerminalBackup:
            return {"run_continuous_terminal_backup", "Terminal node backup time", MetricType::Timer, MetricCategory::Backup, "ns", false};
        case ProfileMetric::RunContinuousExpansionConflict:
            return {"run_continuous_expansion_conflict", "Expansion conflict handling", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::RunContinuousLoopIdleCount:
            return {"run_continuous_idle_count", "Idle iterations", MetricType::Counter, MetricCategory::Thread, "count", false};
        case ProfileMetric::RunContinuousLoopWorkCount:
            return {"run_continuous_work_count", "Work iterations", MetricType::Counter, MetricCategory::Pipeline, "count", false};

        // Root Expansion
        case ProfileMetric::RootExpansionTotal:
            return {"root_expansion_total", "Root expansion total time", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::RootExpansionWaitInference:
            return {"root_expansion_wait", "Root inference wait time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::RootExpansionClone:
            return {"root_expansion_clone", "Root state cloning", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::RootExpansionDirichlet:
            return {"root_expansion_dirichlet", "Dirichlet noise generation", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::RootExpansionAtomicRace:
            return {"root_expansion_atomic_race", "Atomic flag race handling", MetricType::Timer, MetricCategory::Synchronization, "ns", false};

        // Node Expansion Details
        case ProfileMetric::NodeExpansionPolicyMask:
            return {"node_expansion_policy_mask", "Policy masking time", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::NodeExpansionPolicyNormalize:
            return {"node_expansion_policy_norm", "Policy normalization time", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::NodeExpansionChildAlloc:
            return {"node_expansion_child_alloc", "Child allocation time", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::NodeExpansionChildInit:
            return {"node_expansion_child_init", "Child initialization time", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::NodeExpansionAtomicFlag:
            return {"node_expansion_atomic_flag", "Atomic flag operations", MetricType::Timer, MetricCategory::Synchronization, "ns", false};
        case ProfileMetric::NodeExpansionFlagCheck:
            return {"node_expansion_flag_check", "Expanded flag check time", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::NodeExpansionLegalMoves:
            return {"node_expansion_legal_moves", "Legal moves generation", MetricType::Timer, MetricCategory::Expansion, "ns", false};

        // Batch Result Processing
        case ProfileMetric::BatchResultsTotal:
            return {"batch_results_total", "Batch processing total time", MetricType::Timer, MetricCategory::Pipeline, "ns", false};
        case ProfileMetric::BatchResultsCollect:
            return {"batch_results_collect", "Phase 1: Collect results", MetricType::Timer, MetricCategory::Queue, "ns", false};
        case ProfileMetric::BatchResultsExpand:
            return {"batch_results_expand", "Phase 2: Expand nodes", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::BatchResultsBackupPrep:
            return {"batch_results_backup_prep", "Phase 3: Backup prep", MetricType::Timer, MetricCategory::Backup, "ns", false};
        case ProfileMetric::BatchResultsAtomicUpdate:
            return {"batch_results_atomic_update", "Phase 4: Atomic updates", MetricType::Timer, MetricCategory::Backup, "ns", false};
        case ProfileMetric::BatchResultsClearFlags:
            return {"batch_results_clear_flags", "Phase 5: Clear flags", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::BatchResultsReturnStates:
            return {"batch_results_return_states", "Phase 6: Return states", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::BatchResultsPathReversal:
            return {"batch_results_path_reverse", "Path reversal time", MetricType::Timer, MetricCategory::Backup, "ns", false};

        // State Management Detail
        case ProfileMetric::StateCloneForQueue:
            return {"state_clone_for_queue", "State clone for queue", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::StateCloneForRoot:
            return {"state_clone_for_root", "State clone for root", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::StateCloneForPool:
            return {"state_clone_for_pool", "State clone for pool", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::StateUnwindPath:
            return {"state_unwind_path", "Path unwinding time", MetricType::Timer, MetricCategory::Memory, "ns", false};
        case ProfileMetric::StateThreadLocalClone:
            return {"state_tls_clone", "Thread-local init clone", MetricType::Timer, MetricCategory::Memory, "ns", false};

        // Async Coordination
        case ProfileMetric::AsyncSubmissionOverhead:
            return {"async_submission_overhead", "Async submission overhead", MetricType::Timer, MetricCategory::Queue, "ns", false};
        case ProfileMetric::AsyncResultWaitTime:
            return {"async_result_wait", "Async result wait time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::AsyncQueueLatency:
            return {"async_queue_latency", "Request-to-result latency", MetricType::Timer, MetricCategory::Queue, "ns", false};
        case ProfileMetric::AsyncCoordinatorLatency:
            return {"async_coordinator_latency", "Coordinator processing latency", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::AsyncPipelineBubbles:
            return {"async_pipeline_bubbles", "Pipeline bubble time", MetricType::Timer, MetricCategory::Pipeline, "ns", false};

        default:
            return {"unknown", "Unknown metric", MetricType::Counter, MetricCategory::Selection, "?", false};
    }
}

/**
 * @brief Convert metric to string name
 */
constexpr const char* metric_to_string(ProfileMetric metric) {
    return get_metric_metadata(metric).name;
}

/**
 * @brief Convert category to string
 */
constexpr const char* category_to_string(MetricCategory category) {
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
        case MetricCategory::Pipeline: return "Pipeline";
        case MetricCategory::GPU: return "GPU";
        case MetricCategory::Network: return "Network";
        default: return "Unknown";
    }
}

/**
 * @brief Profiling levels for compile-time configuration
 */
enum class ProfileLevel : uint8_t {
    None = 0,      // No profiling (0% overhead)
    Basic = 1,     // Timers only (<0.1% overhead)
    Detailed = 2,  // Timers + hardware counters (<0.5% overhead)
    Full = 3       // Everything including memory tracking (<1% overhead)
};

// Compile-time profiling level (use integer constants for preprocessor)
#ifndef PROFILE_LEVEL_VALUE
    #ifdef NDEBUG
        #define PROFILE_LEVEL_VALUE 0  // None
    #else
        #define PROFILE_LEVEL_VALUE 1  // Basic
    #endif
#endif

} // namespace profiling
} // namespace mcts