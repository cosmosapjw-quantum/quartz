# ThreadLocalArena API Design

**Version**: 1.0
**Author**: Claude Code
**Date**: 2025-10-09
**Status**: Design Complete

## Executive Summary

ThreadLocalArena provides a high-performance, thread-local memory allocator optimized for MCTS tree node allocation. By eliminating malloc contention and leveraging bump-pointer allocation with free lists, we target **10× faster allocation** compared to system malloc, enabling **1.1× overall MCTS throughput improvement**.

Key characteristics:
- **Thread-local**: Zero contention, no locks in allocation paths
- **Bump-pointer fast path**: O(1) allocation for new nodes (~2-5 CPU cycles)
- **Free list reuse**: Efficient recycling of deallocated nodes
- **Cache-aligned**: 64-byte alignment prevents false sharing
- **Fixed memory budget**: Pre-allocated chunks prevent unbounded growth
- **MCTS-optimized**: Size classes matched to node structures (27-64 bytes)

## 1. Background and Motivation

### 1.1 Problem Statement

Current MCTS implementation uses global malloc/free for node allocation:
- **Contention**: Multiple threads compete for malloc locks (~50ns per allocation)
- **Fragmentation**: System allocator not optimized for uniform-size objects
- **Cache misses**: Nodes scattered across memory pages
- **Overhead**: malloc metadata adds 8-16 bytes per allocation

With 8k simulations/second target (realistic) and ~100 node allocations per simulation, we perform **800k allocations/second**. At 50ns per malloc, this is **40ms/second = 4% overhead**.

### 1.2 Research: Existing Arena Allocators

**jemalloc** (Facebook):
- Per-thread caching with size-class segregation
- 4KB page-aligned chunks
- Extent-based allocation with metadata
- **Insight**: Size classes are crucial for reuse

**tcmalloc** (Google):
- Thread-local caches (freelists per size class)
- Central page heap for large allocations
- Span-based memory management
- **Insight**: Separate fast path (thread-local) from slow path (central heap)

**mimalloc** (Microsoft Research):
- Sharded heaps to reduce contention
- Free-list per page with LIFO ordering
- Delayed free handling for cross-thread deallocations
- **Insight**: LIFO improves cache locality

### 1.3 MCTS-Specific Requirements

Unlike general-purpose allocators, MCTS has predictable allocation patterns:
- **Dominant size**: 27-64 bytes (MCTSNode with SoA fields)
- **Allocation rate**: 3M/sec during search
- **Deallocation pattern**: Tree clear (epoch-based) or individual node reuse
- **Thread locality**: Nodes allocated/freed by same thread 99% of the time
- **Memory budget**: <1GB for 10M nodes (27 bytes each = 270MB)

These characteristics allow for a **simpler, faster allocator** than general-purpose solutions.

## 2. Architecture Design

### 2.1 Core Components

```
ThreadLocalArena
├── ChunkList (linked list of memory chunks)
│   ├── Chunk 0: [64KB pre-allocated]
│   ├── Chunk 1: [64KB pre-allocated]
│   └── Chunk N: [on-demand allocation]
├── BumpPointer (current allocation offset)
├── FreeLists (per size class)
│   ├── FreeList[0]: 32 bytes (LIFO linked list)
│   ├── FreeList[1]: 64 bytes
│   ├── FreeList[2]: 128 bytes
│   └── FreeList[3]: 256 bytes
└── Statistics (allocation counters, bytes used)
```

### 2.2 Memory Layout

#### Chunk Structure (64KB each)
```
┌─────────────────────────────────────────┐
│ ChunkHeader (64 bytes, cache-aligned)   │
│  - next_chunk: Chunk*                   │
│  - chunk_size: size_t                   │
│  - used_bytes: size_t                   │
│  - chunk_id: uint32_t                   │
│  - padding: [align to 64 bytes]         │
├─────────────────────────────────────────┤
│ Allocation Space (65472 bytes)          │
│  - Bump-pointer allocations grow →     │
│  - 64-byte aligned allocations          │
│  - Freed nodes added to free lists      │
└─────────────────────────────────────────┘
```

**Chunk Size Selection**: 64KB
- Reasoning:
  - Fits 2048 nodes @ 32 bytes each (typical MCTS node)
  - Aligns with Linux huge page size (2MB = 32 chunks)
  - Small enough for L2 cache (256KB on Ryzen 5900X)
  - Reduces mmap overhead (1 syscall per 2048 nodes vs 2048 mallocs)

#### Allocated Block Layout
```
┌─────────────────────────────────────────┐
│ User Data (requested_size)              │
│  - No metadata header (arena tracks)    │
│  - 64-byte aligned start address        │
│  - Padding to next 64-byte boundary     │
└─────────────────────────────────────────┘
```

**No Metadata**: Unlike malloc, we don't store size in the block. The arena tracks allocations via bump pointer and free lists. This saves 8-16 bytes per node.

#### Free List Node (Intrusive)
```
┌─────────────────────────────────────────┐
│ next: FreeNode* (8 bytes)               │
│ [rest of block available for reuse]     │
└─────────────────────────────────────────┘
```

Freed blocks store only a `next` pointer in the first 8 bytes. When allocated, the entire block is available.

### 2.3 Size Classes

Based on MCTS node sizes:

| Class | Size (bytes) | Use Case | Alignment |
|-------|-------------|----------|-----------|
| 0 | 32 | Minimal node (visit count + value) | 64 |
| 1 | 64 | Standard node (N, W, Q, P, VL) | 64 |
| 2 | 128 | Node with metadata | 64 |
| 3 | 256 | Large node or batch allocation | 64 |

**Rationale**:
- Class 1 (64 bytes) matches MCTSNode structure (27 bytes rounded up)
- 64-byte alignment prevents false sharing between cores
- Powers of 2 simplify size calculations

### 2.4 Thread-Local Storage Strategy

```cpp
// Global thread-local storage
thread_local ThreadLocalArena* g_thread_arena = nullptr;

// Lazy initialization on first use
ThreadLocalArena* get_thread_arena() {
    if (!g_thread_arena) {
        g_thread_arena = new ThreadLocalArena(
            /*initial_chunks=*/2,
            /*chunk_size=*/64 * 1024,
            /*max_chunks=*/128  // 8MB limit per thread
        );
    }
    return g_thread_arena;
}
```

**Design Decisions**:
- **Lazy initialization**: Avoid allocation for unused threads
- **No destruction**: Arenas persist for thread lifetime (avoid cleanup overhead)
- **Max chunks limit**: Prevent unbounded growth (8MB = 128 chunks)
- **No cross-thread sharing**: Each thread has independent arena

## 3. Allocation Algorithms

### 3.1 Fast Path: Bump Pointer Allocation

```cpp
void* allocate(size_t size) {
    // Round up to size class
    size_t aligned_size = round_up_to_size_class(size);

    // FAST PATH: Try free list first (LIFO for cache locality)
    if (FreeNode* node = pop_from_freelist(aligned_size)) {
        stats_.allocations_from_freelist++;
        return node;
    }

    // FAST PATH: Bump pointer allocation
    size_t offset = current_offset_;
    size_t new_offset = align_up(offset + aligned_size, 64);

    if (new_offset <= current_chunk_->chunk_size) {
        current_offset_ = new_offset;
        stats_.allocations_from_bump++;
        return current_chunk_->data + offset;
    }

    // SLOW PATH: Need new chunk
    return allocate_from_new_chunk(aligned_size);
}
```

**Performance**:
- Free list pop: 2-3 CPU cycles (single pointer load + store)
- Bump pointer: 4-5 CPU cycles (add + compare + conditional move)
- **Total: ~5 CPU cycles = 1.5ns @ 3.5 GHz** (vs 50ns malloc = 33× faster)

### 3.2 Free List Management

```cpp
void deallocate(void* ptr, size_t size) {
    if (!ptr) return;

    // Round up to size class
    size_t aligned_size = round_up_to_size_class(size);

    // Add to free list (LIFO for cache locality)
    FreeNode* node = static_cast<FreeNode*>(ptr);
    size_t class_idx = size_to_class_index(aligned_size);

    node->next = freelists_[class_idx];
    freelists_[class_idx] = node;

    stats_.deallocations++;
}
```

**LIFO Rationale**: Recently freed nodes are more likely to be in cache (L1/L2). LIFO maximizes cache hit rate.

### 3.3 Slow Path: Chunk Allocation

```cpp
void* allocate_from_new_chunk(size_t size) {
    // Try to allocate new chunk
    if (num_chunks_ >= max_chunks_) {
        // Fallback to malloc if we hit limit
        stats_.fallback_to_malloc++;
        return std::malloc(size);
    }

    // Allocate new chunk via mmap or malloc
    Chunk* new_chunk = allocate_chunk(chunk_size_);
    new_chunk->next = current_chunk_;
    current_chunk_ = new_chunk;
    num_chunks_++;

    // Allocate from new chunk
    current_offset_ = align_up(size, 64);
    stats_.chunks_allocated++;
    return new_chunk->data;
}
```

**Chunk Allocation Strategy**:
- Use `mmap(MAP_ANONYMOUS)` on Linux for large pages (2MB hugepages if available)
- Fallback to `std::malloc` on other platforms
- Pre-fault pages to avoid page faults during search

## 4. API Specification

### 4.1 ThreadLocalArena Class

```cpp
namespace alphazero {
namespace core {

class ThreadLocalArena {
public:
    // Constructor
    // initial_chunks: Number of chunks to pre-allocate (default 2)
    // chunk_size: Size of each chunk in bytes (default 64KB)
    // max_chunks: Maximum chunks before fallback to malloc (default 128 = 8MB)
    explicit ThreadLocalArena(
        size_t initial_chunks = 2,
        size_t chunk_size = 64 * 1024,
        size_t max_chunks = 128
    );

    // Destructor - frees all chunks
    ~ThreadLocalArena();

    // Allocate memory (64-byte aligned)
    // Returns nullptr if out of memory
    void* allocate(size_t size);

    // Deallocate memory (adds to free list)
    // ptr must have been returned by allocate()
    // size must match the original allocation size
    void deallocate(void* ptr, size_t size);

    // Reset arena (invalidates all allocations, resets to initial state)
    // Fast O(1) operation - just resets bump pointers and clears free lists
    void reset();

    // Get statistics
    struct Statistics {
        size_t allocations_from_bump;      // Bump pointer allocations
        size_t allocations_from_freelist;  // Free list reuse
        size_t deallocations;              // Total frees
        size_t chunks_allocated;           // Chunks allocated
        size_t bytes_allocated;            // Total bytes allocated
        size_t bytes_in_freelists;         // Bytes in free lists
        size_t fallback_to_malloc;         // Overflow allocations
    };

    Statistics get_statistics() const { return stats_; }

    // Non-copyable, non-movable
    ThreadLocalArena(const ThreadLocalArena&) = delete;
    ThreadLocalArena& operator=(const ThreadLocalArena&) = delete;

private:
    struct alignas(64) Chunk {
        Chunk* next;
        size_t chunk_size;
        size_t used_bytes;
        uint32_t chunk_id;
        uint8_t data[0];  // Flexible array member
    };

    struct FreeNode {
        FreeNode* next;
    };

    static constexpr size_t NUM_SIZE_CLASSES = 4;
    static constexpr size_t SIZE_CLASSES[NUM_SIZE_CLASSES] = {32, 64, 128, 256};

    Chunk* current_chunk_;
    size_t current_offset_;
    size_t chunk_size_;
    size_t max_chunks_;
    size_t num_chunks_;

    FreeNode* freelists_[NUM_SIZE_CLASSES];
    Statistics stats_;

    // Helper methods
    Chunk* allocate_chunk(size_t size);
    void free_chunk(Chunk* chunk);
    size_t round_up_to_size_class(size_t size) const;
    size_t size_to_class_index(size_t size) const;
    FreeNode* pop_from_freelist(size_t size);
};

}} // namespace alphazero::core
```

### 4.2 Global Thread-Local Access

```cpp
// Get or create thread-local arena
ThreadLocalArena* get_thread_arena();

// Destroy thread-local arena (call on thread exit)
void destroy_thread_arena();
```

### 4.3 Integration with MCTS Tree

```cpp
// In MCTSTree class
class MCTSTree {
    // ...

    Node* allocate_node() {
        void* memory = get_thread_arena()->allocate(sizeof(Node));
        return new (memory) Node();  // Placement new
    }

    void deallocate_node(Node* node) {
        node->~Node();  // Explicit destructor call
        get_thread_arena()->deallocate(node, sizeof(Node));
    }

    void clear_tree() {
        // Fast O(1) reset instead of individual deallocations
        get_thread_arena()->reset();
        root_ = nullptr;
        node_count_ = 0;
    }
};
```

## 5. Memory Layout and Alignment

### 5.1 Alignment Requirements

All allocations are 64-byte aligned for:
1. **Cache line alignment**: Prevents false sharing between CPU cores
2. **SIMD operations**: AVX2/AVX-512 require aligned loads
3. **Prefetching**: Hardware prefetcher works best with aligned access

### 5.2 Memory Overhead Analysis

**Per Node (64-byte allocation)**:
- User data: 27 bytes (MCTSNode fields)
- Alignment padding: 37 bytes (round up to 64)
- Arena overhead: 0 bytes (no per-allocation metadata)
- **Total: 64 bytes** (vs 43 bytes with malloc = 48% overhead reduction)

**Per Chunk (64KB)**:
- Chunk header: 64 bytes
- Usable space: 65,472 bytes
- Overhead: 0.1% (vs malloc's 10-20% metadata overhead)

**Per Thread (8MB max)**:
- 128 chunks × 64KB = 8,192 KB = 8 MB
- Actual usage: ~2-4 chunks typical (128-256 KB)
- Footprint: **256 KB typical, 8 MB max**

### 5.3 Total Memory Budget

For 12 threads with 10M nodes:
- Nodes: 10M × 64 bytes = 640 MB
- Arena overhead: 12 threads × 256 KB = 3 MB
- **Total: 643 MB** (well under 1GB target)

## 6. Lifecycle Management

### 6.1 Initialization

```
Thread Start
    ↓
First allocate() call
    ↓
get_thread_arena() (lazy init)
    ↓
new ThreadLocalArena(2 chunks, 64KB each)
    ↓
Allocate 2 initial chunks via mmap
    ↓
Ready for allocations
```

### 6.2 Allocation Flow

```
allocate(size)
    ↓
Round up to size class (32/64/128/256)
    ↓
Check free list for size class
    ├─ Found → Pop from list (LIFO) → Return
    └─ Empty ↓
Check bump pointer space in current chunk
    ├─ Fits → Bump pointer → Return
    └─ Full ↓
Allocate new chunk (or fallback to malloc)
    ↓
Return allocation from new chunk
```

### 6.3 Deallocation Flow

```
deallocate(ptr, size)
    ↓
Round up to size class
    ↓
Add to free list (LIFO, intrusive)
    ↓
Update statistics
```

### 6.4 Reset Flow

```
reset()
    ↓
For each chunk:
    └─ Set used_bytes = 0
    ↓
Reset bump pointer to first chunk
    ↓
Clear all free lists
    ↓
Reset statistics
```

**Performance**: O(1) - just pointer updates, no memory freeing

### 6.5 Destruction

```
~ThreadLocalArena()
    ↓
For each chunk in linked list:
    ├─ munmap(chunk) or free(chunk)
    └─ next chunk
    ↓
Clear statistics
```

## 7. Performance Characteristics

### 7.1 Time Complexity

| Operation | Best Case | Average Case | Worst Case |
|-----------|-----------|--------------|------------|
| allocate() | O(1) - free list | O(1) - bump pointer | O(1) - new chunk |
| deallocate() | O(1) - free list push | O(1) | O(1) |
| reset() | O(1) - pointer updates | O(1) | O(1) |
| ~ThreadLocalArena() | O(n) chunks | O(n) chunks | O(n) chunks |

### 7.2 Space Complexity

- Per thread: O(1) - fixed 8MB max
- Per allocation: O(1) - no metadata
- Total: O(threads × max_chunks) = O(12 × 128 × 64KB) = 96 MB max

### 7.3 Benchmark Targets

| Metric | Target | Baseline (malloc) | Speedup |
|--------|--------|------------------|---------|
| Allocation latency | 1.5 ns | 50 ns | 33× |
| Free list reuse | <2 ns | 50 ns | 25× |
| Memory overhead | <1% | 10-20% | 10-20× |
| Cache miss rate | <5% | 20-30% | 4-6× |

### 7.4 Expected Impact on MCTS

**NOTE**: Memory arenas (T009) already implemented. Current bottleneck is state cloning (86.6%), not allocation.

- Allocation time: Optimized via thread-local arenas (99.93% fast-path)
- Overall throughput baseline: 2,659 sims/sec (profiling_suite_20251016_124134)
- After state pooling (T018): 9,838 sims/sec = **3.7× speedup** (exceeds 8k target)
- Memory footprint: <1GB (27 bytes per node, 270MB for 10M nodes) ✅ ACHIEVED

## 8. Error Handling

### 8.1 Out of Memory

```cpp
void* allocate(size_t size) {
    // ... normal allocation ...

    // If max chunks reached, fallback to malloc
    if (num_chunks_ >= max_chunks_) {
        void* ptr = std::malloc(size);
        if (!ptr) {
            // Out of memory - cannot recover
            throw std::bad_alloc();
        }
        stats_.fallback_to_malloc++;
        return ptr;
    }

    // ... allocate new chunk ...
}
```

**Fallback Strategy**: If arena exhausted, fall back to system malloc. This prevents hard failures but loses performance benefits.

### 8.2 Invalid Deallocation

```cpp
void deallocate(void* ptr, size_t size) {
    if (!ptr) return;  // nullptr is valid

    // Note: We do NOT validate that ptr came from this arena
    // This is by design for performance - user must ensure correctness

    // Add to free list
    // ...
}
```

**Design Decision**: No validation of `ptr` ownership. This is a performance allocator, not a debugging allocator. Use AddressSanitizer/Valgrind for debugging.

### 8.3 Double Free

Not detected - user must ensure correctness. Using the same pointer twice in `deallocate()` will corrupt the free list.

**Mitigation**: Use RAII wrappers or smart pointers to prevent double free.

## 9. Testing Strategy

### 9.1 Unit Tests

**Test Suite**: `tests/unit/test_thread_local_arena.cpp`

1. **Basic Operations**
   - Allocate single object
   - Deallocate single object
   - Allocate multiple objects
   - Free list reuse

2. **Alignment**
   - Verify 64-byte alignment for all allocations
   - Test with various sizes (1, 16, 32, 64, 128, 256 bytes)

3. **Chunk Management**
   - Allocate until chunk overflow
   - Verify new chunk allocated
   - Test max_chunks limit
   - Test fallback to malloc

4. **Free List**
   - LIFO ordering verification
   - Reuse correctness
   - Multiple size classes

5. **Reset**
   - Reset clears allocations
   - Subsequent allocations work
   - Statistics reset correctly

6. **Statistics**
   - Track allocations/deallocations
   - Bytes allocated/freed
   - Chunk count

7. **Thread Safety**
   - Multiple threads with separate arenas
   - No cross-thread interference
   - Thread-local storage correctness

### 9.2 Performance Benchmarks

**Test Suite**: `tests/performance/test_arena_vs_malloc.cpp`

1. **Allocation Speed**
   - Measure cycles per allocation
   - Compare vs malloc/free
   - Target: 10× faster

2. **Cache Locality**
   - Measure L1/L2 cache hit rate
   - Compare sequential allocations
   - Target: 90%+ hit rate

3. **Fragmentation**
   - Allocate/free patterns
   - Measure memory efficiency
   - Target: <5% fragmentation

4. **Scalability**
   - Test with 1-12 threads
   - Measure throughput
   - Verify linear scaling

### 9.3 Integration Tests

**Test Suite**: `tests/integration/test_mcts_with_arena.cpp`

1. **MCTS Tree Allocation**
   - Full search with arena allocator
   - Verify correctness vs baseline
   - Measure performance improvement

2. **Memory Leak Detection**
   - Run 1000 searches
   - Verify no memory growth
   - Check arena statistics

3. **Stress Test**
   - 24-hour soak test
   - Monitor memory usage
   - Verify stability

## 10. Implementation Phases

### Phase 1: Core Arena (T009b)
- Chunk allocation
- Bump pointer allocation
- Basic statistics

### Phase 2: Free Lists (T009c)
- Size class management
- LIFO free lists
- Allocation from free list

### Phase 3: Optimization (T009d)
- Lock-free operations (if needed for cross-thread)
- Cache optimization
- SIMD-aligned allocations

### Phase 4: Integration (T009e)
- MCTS tree integration
- Thread-local storage
- Performance validation

### Phase 5: Testing (T009f)
- Comprehensive test suite
- Benchmarks
- Documentation

## 11. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Memory leaks | Low | High | Comprehensive testing, ASAN |
| Fragmentation | Medium | Medium | Reset on tree clear, size classes |
| Thread safety bugs | Low | High | Thread-local design, TSan |
| Performance regression | Low | Medium | Benchmarks, A/B testing |
| Integration issues | Medium | Low | Incremental rollout |

## 12. Alternatives Considered

### 12.1 Object Pool

**Pros**: Simpler, type-safe
**Cons**: Less flexible, harder to extend
**Decision**: Arena is more general-purpose

### 12.2 Custom Slab Allocator

**Pros**: Zero fragmentation for fixed sizes
**Cons**: Harder to implement, less flexible
**Decision**: Arena with size classes achieves similar benefits

### 12.3 Use jemalloc/tcmalloc

**Pros**: Battle-tested, full-featured
**Cons**: External dependency, general-purpose (slower)
**Decision**: Custom arena is simpler and faster for MCTS

## 13. Future Enhancements

### 13.1 Huge Page Support
Use `madvise(MADV_HUGEPAGE)` for 2MB pages on Linux to reduce TLB misses.

### 13.2 NUMA-Aware Allocation
Allocate chunks from NUMA node local to thread for multi-socket systems.

### 13.3 Cross-Thread Deallocation
Handle case where thread A allocates and thread B deallocates (rare in MCTS but possible).

### 13.4 Compaction
Compact live objects to reduce fragmentation (low priority - reset() handles this).

## 14. Summary

ThreadLocalArena provides a **simple, fast, and safe** memory allocator for MCTS node allocation:

✅ **10× faster allocation** via bump pointers
✅ **Zero contention** via thread-local design
✅ **Low overhead** via metadata-free allocations
✅ **Cache-friendly** via 64-byte alignment and LIFO free lists
✅ **Predictable memory** via fixed chunk budget
✅ **Easy integration** via simple API

**Expected impact**: **1.1× MCTS throughput improvement** by eliminating 145ms/sec of allocation overhead.

## 15. References

1. **jemalloc**: http://jemalloc.net/jemalloc.3.html
2. **tcmalloc**: https://google.github.io/tcmalloc/design.html
3. **mimalloc**: https://www.microsoft.com/en-us/research/publication/mimalloc-free-list-sharding-in-action/
4. **Linux mmap**: man 2 mmap
5. **Cache-line alignment**: Intel Optimization Manual, Section 3.6
6. **AlphaZero**: Silver et al., "Mastering Chess and Shogi by Self-Play"

---

**End of Design Document**
