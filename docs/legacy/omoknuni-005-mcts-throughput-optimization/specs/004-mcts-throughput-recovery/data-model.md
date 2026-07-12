# Data Model: MCTS Throughput Recovery

## Overview

This document specifies the data structures and memory layouts for the optimized MCTS implementation. All structures are designed for cache efficiency, lock-free operation where possible, and minimal memory footprint.

## Core Data Structures

### 1. WUUCTVirtualLossManager

Manages in-flight simulation counts for WU-UCT style virtual loss without distorting Q-values.

```cpp
class WUUCTVirtualLossManager {
private:
    // Separate array for in-flight counts (not mixed with visit counts)
    std::vector<std::atomic<uint32_t>> in_flight_counts_;

    // Virtual loss magnitude (tunable, default 1.0)
    float virtual_loss_magnitude_;

    // Statistics for monitoring
    std::atomic<uint64_t> total_applications_{0};
    std::atomic<uint64_t> total_removals_{0};
    std::atomic<uint64_t> collision_count_{0};

public:
    // Thread-safe operations
    void add_in_flight(NodeIndex node);
    void remove_in_flight(NodeIndex node);
    float get_exploration_adjustment(NodeIndex node) const;
    bool is_busy(NodeIndex node) const;  // For busy-edge masking
};

// Memory layout (per node):
// - 4 bytes: atomic<uint32_t> in_flight count
// Total: 4 bytes per node (40MB for 10M nodes)
```

### 2. MPMCRingBuffer (Lock-Free Queue)

Multi-producer multi-consumer ring buffer for inference requests.

```cpp
template<typename T, size_t Capacity = 4096>
class MPMCRingBuffer {
private:
    struct Cell {
        std::atomic<uint64_t> sequence;
        T data;
        char padding[64 - sizeof(T) - 8];  // Cache line padding
    };

    alignas(64) std::array<Cell, Capacity> buffer_;
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) std::atomic<uint64_t> tail_{0};

    // Condition variable for blocking wait
    std::condition_variable cv_;
    std::mutex cv_mutex_;
    std::atomic<bool> has_waiters_{false};

public:
    bool try_enqueue(T&& item);
    bool try_dequeue(T& item);
    bool wait_dequeue(T& item, std::chrono::milliseconds timeout);
    size_t size() const;
};

// Memory layout:
// - 4096 cells × 64 bytes = 256KB per queue
// - Cache-line aligned to prevent false sharing
// - Sequence numbers prevent ABA problem
```

### 3. DLPackTensorBridge

Zero-copy tensor bridge between C++ and Python using DLPack protocol.

**⚠️ PERFORMANCE NOTE (2025-10-16 - Profiling-Validated):**
- Feature extraction loop: OpenMP NOT active (0/560 trials in profiling campaign)
- OpenMP benchmark: 8.64ms → 1.57ms @ 12 threads ✅ WORKING when active
- Current impact: Secondary bottleneck (state cloning is 86.6% of time, GPU inference 2.1%)
- Priority: Low (state pooling alone achieves 8k target via T018)
- Investigation: T019 (optional enhancement for 14k+ stretch goal)
- Note: Feature extraction overhead amortized across batches, not per-simulation

```cpp
class DLPackTensorBridge {
private:
    // Pinned CPU memory pool (NOT GPU device memory)
    // DLPack tensors use kDLCUDAHost device type (pinned host memory)
    struct PinnedBuffer {
        void* data;
        size_t size;
        bool in_use;
    };
    std::vector<PinnedBuffer> pinned_pool_;
    std::mutex pool_mutex_;

    // DLPack tensor descriptors
    std::vector<DLTensor> tensor_descriptors_;
    std::vector<DLManagedTensor> managed_tensors_;

public:
    // Create DLPack tensor from game states (zero-copy)
    DLManagedTensor* create_batch_tensor(
        const std::vector<std::unique_ptr<IGameState>>& states,
        int batch_size,
        int channels,
        int height,
        int width
    );

    // Direct feature extraction into pinned memory
    // TODO: Parallelize this loop with OpenMP (7.5ms bottleneck)
    void extract_features_direct(
        const IGameState* state,
        float* output_buffer,
        int buffer_offset
    );

    // Memory management
    void* allocate_pinned(size_t size);
    void free_pinned(void* ptr);
};

// Memory layout per batch:
// - Batch 64 × 36 channels × 15×15 = 518,400 floats = 2MB
// - Pre-allocated pinned CPU memory pool: 8MB (4 buffers)
// - Double buffering for async GPU transfers (H2D copy ~0.24ms)
```

### 4. ThreadLocalArena

Per-thread memory arena to eliminate allocation contention.

```cpp
class ThreadLocalArena {
private:
    struct Block {
        char data[1 << 20];  // 1MB blocks
        std::atomic<size_t> offset{0};
    };

    // Per-thread storage
    static thread_local Block* current_block_;
    static thread_local std::vector<std::unique_ptr<Block>> blocks_;

    // Free list for reuse
    struct FreeNode {
        FreeNode* next;
        size_t size;
    };
    static thread_local FreeNode* free_list_;

public:
    // Allocation (lock-free for thread-local access)
    void* allocate(size_t size, size_t alignment = 8);
    void deallocate(void* ptr, size_t size);

    // Bulk operations
    void reset();  // Reset all allocations
    size_t memory_used() const;
};

// Memory layout:
// - 1MB per block × N threads × M blocks = ~100MB typical
// - O(1) allocation from current block
// - Free list for deallocated memory
```

### 5. OptimizedTreeNode

Enhanced tree node structure with busy-edge masking support.

```cpp
struct alignas(64) OptimizedTreeNode {
    // Core MCTS data (unchanged)
    float total_value;           // 4 bytes
    float prior;                  // 4 bytes
    std::atomic<uint32_t> visit_count;  // 4 bytes
    std::atomic<float> virtual_loss;     // 4 bytes

    // Relationships
    NodeIndex parent;             // 4 bytes
    NodeIndex first_child;        // 4 bytes
    NodeIndex next_sibling;       // 4 bytes

    // Move storage
    uint16_t move;                // 2 bytes

    // NEW: Expansion state for busy-edge masking
    std::atomic<uint8_t> expansion_state;  // 1 byte
    // States: UNEXPANDED=0, EXPANDING=1, EXPANDED=2

    // Padding to 32 bytes
    uint8_t padding[1];
};

// Memory layout:
// - 32 bytes per node (aligned to half cache line)
// - 320MB for 10M nodes
// - Atomic fields for thread safety
```

### 6. BatchInferenceRequest

Optimized batch request structure.

```cpp
struct BatchInferenceRequest {
    // Request metadata
    uint64_t batch_id;
    std::chrono::steady_clock::time_point creation_time;

    // Batch data (using DLPack)
    DLManagedTensor* input_tensor;  // Zero-copy tensor

    // Request tracking
    std::vector<uint64_t> request_ids;
    std::vector<NodeIndex> node_indices;
    std::vector<std::vector<NodeIndex>> paths;

    // Result placeholders
    std::vector<std::vector<float>> policies;  // Pre-allocated
    std::vector<float> values;                 // Pre-allocated
    std::atomic<bool> completed{false};
};

// Memory layout per batch:
// - Input tensor: 2MB (batch 64)
// - Metadata: ~4KB
// - Results: ~500KB (policies + values)
// Total: ~2.5MB per in-flight batch
```

### 7. CollisionMetrics

Performance monitoring structure.

```cpp
struct alignas(64) CollisionMetrics {
    // Selection collisions
    std::atomic<uint64_t> selection_retries{0};
    std::atomic<uint64_t> duplicate_paths{0};
    std::atomic<uint64_t> busy_edge_blocks{0};

    // Expansion conflicts
    std::atomic<uint64_t> expansion_conflicts{0};
    std::atomic<uint64_t> successful_expansions{0};

    // Batch efficiency
    std::atomic<uint64_t> total_batches{0};
    std::atomic<uint64_t> total_positions{0};
    std::atomic<uint64_t> unique_positions{0};

    // Timing
    std::atomic<uint64_t> total_selection_ns{0};
    std::atomic<uint64_t> total_expansion_ns{0};
    std::atomic<uint64_t> total_backup_ns{0};

    // Methods
    float collision_rate() const;
    float batch_efficiency() const;
    void reset();
};

// Memory layout:
// - 64 bytes (single cache line)
// - All metrics are atomic for thread safety
```

## Memory Layout Optimizations

### Cache Line Alignment Strategy

```cpp
// Group hot data on same cache line (64 bytes)
struct alignas(64) HotData {
    float values[8];      // Frequently accessed together
    uint32_t counts[8];   // Frequently accessed together
};

// Separate cold data to different cache lines
struct alignas(64) ColdData {
    char metadata[64];    // Rarely accessed
};
```

### SIMD-Friendly Layout

```cpp
// Structure of Arrays for vectorized operations
struct TreeSoA {
    // Hot arrays (accessed during PUCT calculation)
    alignas(64) float* q_values;        // All Q-values contiguous
    alignas(64) float* priors;          // All priors contiguous
    alignas(64) uint32_t* visit_counts; // All visits contiguous
    alignas(64) uint32_t* in_flight;    // All in-flight contiguous

    // Cold arrays (accessed during expansion/backup)
    alignas(64) NodeIndex* parents;     // Parent indices
    alignas(64) NodeIndex* children;    // Child indices
    alignas(64) uint16_t* moves;        // Move encoding

    // AVX2 operations on hot data
    void compute_puct_batch_avx2(
        const NodeIndex* nodes,
        float* scores_out,
        size_t count
    );
};
```

### Memory Pool Sizes

```cpp
// Pre-allocated pool sizes based on target usage
constexpr size_t TREE_NODE_POOL_SIZE = 10'000'000;  // 10M nodes
constexpr size_t PINNED_BUFFER_SIZE = 8'388'608;    // 8MB pinned
constexpr size_t ARENA_BLOCK_SIZE = 1'048'576;      // 1MB blocks
constexpr size_t RING_BUFFER_SIZE = 4096;           // 4K entries
constexpr size_t BATCH_POOL_SIZE = 8;               // 8 concurrent batches

// Total memory budget
// Tree nodes: 320MB
// Virtual loss: 40MB
// Pinned buffers: 32MB (4 × 8MB)
// Thread arenas: 100MB (estimated)
// Ring buffers: 1MB
// Total: ~500MB (well under 1GB target)
```

## Atomic Operations Ordering

### Relaxed Ordering Usage

```cpp
// Safe for independent counters
visit_count.fetch_add(1, std::memory_order_relaxed);
in_flight.fetch_add(1, std::memory_order_relaxed);

// Safe for statistics
collision_count.fetch_add(1, std::memory_order_relaxed);
```

### Acquire-Release Ordering

```cpp
// Required for expansion state transitions
uint8_t expected = UNEXPANDED;
if (expansion_state.compare_exchange_strong(
    expected, EXPANDING,
    std::memory_order_acquire,
    std::memory_order_relaxed)) {
    // Perform expansion...
    expansion_state.store(EXPANDED, std::memory_order_release);
}
```

### Sequential Consistency

```cpp
// Required for queue head/tail updates (avoided where possible)
head.fetch_add(1, std::memory_order_seq_cst);  // Only if necessary
```

## Data Flow Optimization

### Zero-Copy Pipeline

```
C++ Game State
    ↓ (extract features directly to pinned memory)
DLPack Tensor (pinned)
    ↓ (zero-copy view in PyTorch)
PyTorch Tensor
    ↓ (async GPU transfer)
GPU Memory
    ↓ (inference)
GPU Results
    ↓ (async transfer back)
Pinned Results Buffer
    ↓ (direct access from C++)
Tree Update
```

### Batch Assembly Strategy

```cpp
// Efficient batch assembly without copies
class BatchAssembler {
    DLPackTensorBridge bridge_;
    float* current_buffer_;
    int current_offset_;

    void add_state(const IGameState* state) {
        // Direct extraction into batch buffer
        bridge_.extract_features_direct(
            state,
            current_buffer_,
            current_offset_
        );
        current_offset_ += state->feature_size();
    }
};
```

## Thread Safety Guarantees

### Lock-Free Operations
- Node visit count updates
- Virtual loss application/removal
- Statistics collection
- Ring buffer enqueue/dequeue (with CAS)

### Mutex-Protected Operations
- Tree node allocation (per-pool mutex)
- Pinned memory allocation
- Batch result association

### Thread-Local Operations
- Arena allocation (no synchronization)
- Selection scratch buffers
- Random number generation

## Performance Characteristics

### Operation Costs

| Operation | Current | Optimized | Improvement |
|-----------|---------|-----------|-------------|
| Node allocation | 150ns (mutex) | 10ns (arena) | 15× |
| Queue enqueue | 500ns (mutex) | 50ns (CAS) | 10× |
| State→Tensor | 1000ns (Python) | 100ns (direct) | 10× |
| Batch assembly | 5μs (copies) | 0.5μs (zero-copy) | 10× |
| Result retrieval | 200ns (map) | 20ns (index) | 10× |

### Memory Bandwidth

```cpp
// Selection phase (read-heavy)
// 4 threads × 1000 nodes/sec × 32 bytes = 128KB/sec

// Expansion phase (write-heavy)
// 100 expansions/sec × 320 bytes = 32KB/sec

// Backup phase (atomic updates)
// 100 backups/sec × 20 nodes × 8 bytes = 16KB/sec

// Total: <200KB/sec (negligible vs DDR4 bandwidth)
```

## Validation Requirements

### Correctness Tests
- Thread sanitizer clean (no data races)
- Memory sanitizer clean (no leaks)
- Value conservation in backup
- Policy normalization after masking

### Performance Tests
- 10M node allocation in <1 second
- 1M queue operations in <100ms
- Zero-copy tensor creation <100μs
- Batch assembly for 64 states <1ms

### Stress Tests
- 24-hour continuous operation
- 100M simulations without memory growth
- Thread scaling from 1 to 16
- Queue overflow/underflow handling

## Migration Path

### Phase 1: Non-Breaking Additions
```cpp
// Add new structures alongside existing
class Tree {
    // Existing AoS nodes
    std::vector<TreeNode> nodes_;

    // NEW: WU-UCT manager
    WUUCTVirtualLossManager vu_manager_;

    // NEW: Collision metrics
    CollisionMetrics metrics_;
};
```

### Phase 2: Interface Updates
```cpp
// Update selection to use WU-UCT
float Tree::get_puct_score(NodeIndex node) {
    float q = get_q_value(node);  // True Q
    float adjustment = vu_manager_.get_exploration_adjustment(node);
    return q + c_puct * prior * sqrt_term / (1 + N + adjustment);
}
```

### Phase 3: Full Integration
```cpp
// Replace queue, add DLPack bridge
class AsyncInferenceQueue {
    MPMCRingBuffer<InferenceRequest> requests_;
    DLPackTensorBridge bridge_;
    // Remove mutex-based implementation
};
```

## Future Extensions

### GPU-Accelerated Selection
```cuda
__global__ void compute_puct_kernel(
    const float* q_values,
    const float* priors,
    const uint32_t* visit_counts,
    const uint32_t* in_flight,
    float* scores_out,
    int num_nodes
);
```

### Compressed Tree Representation
```cpp
// For very large trees (100M+ nodes)
struct CompressedNode {
    uint16_t q_value_quantized;  // 2 bytes (vs 4)
    uint16_t prior_quantized;    // 2 bytes (vs 4)
    uint16_t visit_count_log;    // 2 bytes (vs 4)
    // Total: 6 bytes (vs 32)
};
```

### Distributed MCTS
```cpp
// For multi-machine training
class DistributedTree {
    TreeShard local_shard_;
    RemoteShardProxy remote_shards_[N];
    ConsistencyProtocol protocol_;
};
```