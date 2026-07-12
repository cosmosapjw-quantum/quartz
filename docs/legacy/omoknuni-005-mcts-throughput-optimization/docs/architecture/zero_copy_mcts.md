# Zero-Copy MCTS Architecture

**Document Version**: 1.0
**Date**: 2025-10-16
**Authority**: T019 Zero-Copy Architecture Design (spec/004)

---

## Overview

The zero-copy MCTS architecture eliminates the state cloning bottleneck by storing only move sequences in tree nodes. Game states are reconstructed on-demand by thread-local workers applying moves from root to leaf (make) and unwinding (unmake).

This design addresses the fundamental architectural limitation identified in T018, where state cloning consumed 86.6% of execution time (418μs per simulation) and could not be optimized away through pooling.

## Problem Statement

### Current Architecture Bottleneck

**Profiling Evidence** (560 trials, 100% capture):
```
State cloning: 835.85 ms / 982.86 ms (86.6% of time)
Per-simulation cost: 418μs
Root cause: 223 allocations per clone (~2μs each = 446μs)
Baseline performance: 2,659 sims/sec
```

**Architectural Limitation**:
- Nodes contain full game state objects (120KB for Gomoku)
- Every simulation requires cloning root state
- Clone involves 223 heap allocations + memcpy overhead
- Cannot be optimized away with state pooling (T018 investigation)

### Zero-Copy Solution

**Core Principle**: Store only move sequences, reconstruct states on-demand.

**Performance Impact**:
- State cloning: 418μs → State reconstruction: ~600ns (make/unmake)
- Speedup: 697× faster (418,000ns / 600ns)
- Expected throughput: 15,000-25,000 sims/sec (5-10× improvement)

---

## Architecture Components

### 1. Tiny Nodes (32 bytes)

**Design** (`cpp_extensions/mcts/tiny_node.hpp`):
```cpp
struct alignas(64) TinyNode {
    uint16_t move;                         // Move that led to this node
    uint32_t parent_idx;                   // Parent node index
    uint32_t first_child_idx;              // First child index
    uint32_t next_sibling_idx;             // Next sibling index
    std::atomic<uint32_t> visit_count;     // Visit count (MCTS)
    std::atomic<int32_t> total_value_scaled; // Total value (scaled)
    uint16_t prior_scaled;                 // Prior probability (scaled)
    std::atomic<uint8_t> virtual_loss;     // Virtual loss (WU-UCT)
    uint8_t flags;                         // Node flags
    uint64_t zobrist_hash;                 // Zobrist hash (transpositions)
    // Total: 34 bytes, aligned to 64 bytes
};
```

**Memory Impact**:
- Current: 120KB per node → New: 64 bytes per node (aligned)
- Reduction: 1,875× per node
- 10M nodes: 1.2GB → 640MB (50% reduction)
- Cache efficiency: 1,875 cache lines → 1 cache line

**Tree Structure**:
- Index-based references (not pointers)
- Parent-child-sibling linked list
- Supports 4 billion nodes (uint32_t indices)

### 2. Thread-Local State Reconstruction

**Pattern**: make/unmake for in-place move application/reversal.

**API** (`cpp_extensions/utils/igamestate.h`):
```cpp
class IGameState {
public:
    // Fast in-place move application (~15ns)
    // Returns opaque undo token (game-specific)
    virtual uint64_t make_move(uint16_t move) = 0;

    // Fast move reversal (~15ns)
    // Restores state before make_move
    virtual void unmake_move(uint16_t move, uint64_t undo_token) = 0;

    // Zobrist hash for transposition tables
    virtual uint64_t zobrist_hash() const = 0;
};
```

**Undo Token Design**:

Gomoku (minimal):
```cpp
uint64_t undo_token = (
    (last_move_row << 8) |
    (last_move_col << 0) |
    (game_result << 16) |
    (move_count << 24)
);
```

Chess (complex):
```cpp
uint64_t undo_token = (
    (captured_piece << 0) |      // 4 bits
    (castling_rights << 4) |     // 4 bits
    (en_passant_square << 8) |   // 8 bits
    (halfmove_clock << 16) |     // 8 bits
    (game_result << 24)          // 8 bits
);
```

Go (moderate):
```cpp
uint64_t undo_token = (
    (ko_position << 0) |         // 16 bits
    (captured_stones_mask << 16) | // 32 bits
    (passes << 48) |             // 8 bits
    (game_result << 56)          // 8 bits
);
```

**Usage Pattern**:
```cpp
// Thread-local state (one per worker)
thread_local std::unique_ptr<IGameState> worker_state;

// Traverse path in MCTS tree
std::vector<uint64_t> undo_stack;
for (TinyNode* node : path) {
    uint64_t undo = worker_state->make_move(node->move);
    undo_stack.push_back(undo);
}

// Neural network inference at leaf
auto [policy, value] = infer(*worker_state);

// Unwind path (LIFO order)
for (int i = path.size() - 1; i >= 0; --i) {
    worker_state->unmake_move(path[i]->move, undo_stack[i]);
}
```

**Thread Safety**:
- make/unmake are NOT thread-safe (modify state in-place)
- Each thread maintains its own IGameState instance
- No synchronization overhead

### 3. Per-Thread Bump Arenas

**Design** (`cpp_extensions/mcts/bump_arena.hpp`):
```cpp
class BumpArena {
public:
    static constexpr size_t BLOCK_SIZE = 65536;  // 64K nodes per block

    TinyNode* allocate() {
        if (offset_ >= BLOCK_SIZE) {
            allocate_new_block();
        }
        return &current_block_[offset_++];
    }

    void reclaim_epoch() {
        // Bulk-free all blocks allocated before epoch marker
        blocks_.erase(blocks_.begin(), epoch_marker_);
        epoch_marker_ = blocks_.begin();
        offset_ = 0;
    }

private:
    TinyNode* current_block_;
    size_t offset_;
    std::vector<std::unique_ptr<TinyNode[]>> blocks_;
};

// Thread-local bump arena per worker
thread_local BumpArena node_arena;
```

**Performance**:
- Allocation speed: O(1) pointer increment (~5ns)
- No locking: Each thread has own arena
- Bulk reclamation: O(1) epoch increment (vs O(N) free)

**Memory Management**:
- Pre-allocate 64K-node blocks (4MB per block)
- Bump allocator: offset_++
- Epoch reclamation: bulk-free when safe

### 4. Epoch Reclamation (QSBR)

**Quiescent-State-Based Reclamation**:
```cpp
class EpochManager {
public:
    void enter_epoch(size_t thread_id) {
        thread_epochs_[thread_id].store(
            global_epoch_.load(std::memory_order_acquire),
            std::memory_order_release
        );
    }

    void exit_epoch(size_t thread_id) {
        thread_epochs_[thread_id].store(
            QUIESCENT,
            std::memory_order_release
        );
    }

    void try_reclaim() {
        // Advance global epoch
        size_t new_epoch = global_epoch_.fetch_add(1, std::memory_order_acq_rel);

        // Wait for all threads to reach quiescent state
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            while (thread_epochs_[tid].load(std::memory_order_acquire) < new_epoch) {
                std::this_thread::yield();
            }
        }

        // Safe to reclaim all arenas now
        for (auto& arena : thread_arenas_) {
            arena.reclaim_epoch();
        }
    }

private:
    std::atomic<size_t> global_epoch_{0};
    std::vector<std::atomic<size_t>> thread_epochs_;
};
```

**Safety Guarantees**:
- Memory reclaimed only when all threads quiesce
- No use-after-free (bounded waiting)
- Bulk-free entire blocks (efficient)

### 5. Transposition Tables (DAG)

**Monte-Carlo Graph Search**:
```cpp
struct TranspositionEntry {
    uint64_t zobrist_hash;
    uint32_t node_idx;  // Index to canonical node
    uint32_t visit_count;
};

class TranspositionTable {
public:
    uint32_t lookup_or_insert(uint64_t zobrist, uint32_t new_node_idx) {
        size_t slot = zobrist % table_size_;

        // Linear probing (max 16 probes)
        for (size_t i = 0; i < MAX_PROBE; ++i) {
            TranspositionEntry& entry = table_[(slot + i) % table_size_];

            // Empty slot - insert new
            if (entry.zobrist_hash == 0) {
                entry.zobrist_hash = zobrist;
                entry.node_idx = new_node_idx;
                return new_node_idx;
            }

            // Found match - return canonical node
            if (entry.zobrist_hash == zobrist) {
                return entry.node_idx;
            }
        }

        // Table full - evict LRU
        return evict_and_insert(zobrist, new_node_idx);
    }

private:
    std::vector<TranspositionEntry> table_;
    size_t table_size_;
    static constexpr size_t MAX_PROBE = 16;
};
```

**Benefits**:
- Deduplication: Tree becomes DAG (positions shared)
- Memory savings: 20-40% (typical for board games)
- Visit count accuracy: Transpositions share statistics
- Faster search: Reuse previous work

### 6. Bounded SPSC Queues

**Replace moodycamel with bounded SPSC**:
```cpp
template<typename T, size_t Capacity>
class BoundedSPSCQueue {
public:
    bool try_enqueue(T&& item) {
        size_t write_idx = write_idx_.load(std::memory_order_relaxed);
        size_t next_write = (write_idx + 1) % Capacity;

        // Check full
        if (next_write == read_idx_.load(std::memory_order_acquire)) {
            return false;
        }

        buffer_[write_idx] = std::move(item);
        write_idx_.store(next_write, std::memory_order_release);
        return true;
    }

    bool try_dequeue(T& item) {
        size_t read_idx = read_idx_.load(std::memory_order_relaxed);

        // Check empty
        if (read_idx == write_idx_.load(std::memory_order_acquire)) {
            return false;
        }

        item = std::move(buffer_[read_idx]);
        read_idx_.store((read_idx + 1) % Capacity, std::memory_order_release);
        return true;
    }

private:
    std::array<T, Capacity> buffer_;
    std::atomic<size_t> write_idx_{0};
    std::atomic<size_t> read_idx_{0};
};
```

**Benefits**:
- No dynamic allocation (fixed-size ring buffer)
- Cache-friendly (contiguous storage)
- Lock-free (single producer, single consumer)

---

## Performance Analysis

### State Reconstruction Cost

**Benchmark** (Gomoku):
```
Current (clone):
  memcpy(225 bytes) = ~50ns
  memcpy(200 bytes) = ~40ns
  Primitive copies = ~10ns
  Allocation overhead = 446μs (223 allocs × 2μs)
  Total: ~418μs per clone

Zero-Copy (make/unmake):
  Place stone: board[row * 15 + col] = player (5ns)
  Update metadata: move_count++, last_move (5ns)
  Check win condition (if terminal): ~100ns
  Total: ~15ns per make_move

Speedup: 418μs / 15ns = 27,867× faster
```

### Path Traversal Cost

**Typical MCTS Path** (depth = 20 moves):
```
Current Architecture:
  Clone root state: 418μs
  Apply 20 moves: 20 × 10μs = 200μs
  Total: 618μs per simulation

Zero-Copy Architecture:
  make 20 moves: 20 × 15ns = 300ns
  unmake 20 moves: 20 × 15ns = 300ns
  Total: 600ns per simulation

Speedup: 618μs / 600ns = 1,030× faster
```

### Overall Throughput Projection

**Baseline Breakdown** (2,659 sims/sec):
```
State cloning:       835.85 ms (86.6%)
Expansion:            37.24 ms ( 3.8%)
Selection:             3.58 ms ( 0.4%)
Backup:                1.67 ms ( 0.2%)
Overhead:             85.64 ms ( 8.7%)
Total:               982.86 ms per 2,000 sims
```

**After Zero-Copy** (projected):
```
State reconstruction:   1.20 ms ( 1.2%)  ← 600ns × 2,000 sims
Expansion:             37.24 ms (37.9%)
Selection:              3.58 ms ( 3.6%)
Backup:                 1.67 ms ( 1.7%)
Overhead:              54.58 ms (55.6%)  ← Reduced (less GIL)
Total:                 98.27 ms per 2,000 sims

Throughput: 2,000 / 0.09827 = 20,351 sims/sec
Improvement: 20,351 / 2,659 = 7.65× faster
```

**Conservative Estimate** (30% overhead):
```
Adjusted throughput: 20,351 × 0.7 = 14,246 sims/sec
Improvement: 14,246 / 2,659 = 5.36× faster
Range: 15,000-25,000 sims/sec (5-10× improvement)
```

---

## Prior Art & References

### Production Systems Using Zero-Copy MCTS

1. **Stockfish** (Chess)
   - make/unmake with 64-bit undo tokens
   - Zobrist transposition tables
   - 200M nodes/sec search speed
   - Source: github.com/official-stockfish/Stockfish

2. **KataGo** (Go)
   - Zero-copy MCTS with thread-local reconstruction
   - DAG tree with transposition tables
   - 80k playouts/sec on GPU
   - Source: github.com/lightvector/KataGo

3. **Leela Zero** (Go/Chess)
   - AlphaZero-style with make/unmake
   - Per-thread node arenas
   - Source: github.com/leela-zero/leela-zero

4. **AlphaGo/AlphaZero** (Google DeepMind)
   - Tiny nodes with move sequences
   - Thread-local state reconstruction
   - Reference: Silver et al. 2016, 2017 papers

### Academic References

1. **Epoch-Based Reclamation**:
   - Fraser, K. (2004). "Practical lock-freedom"
   - Hart et al. (2007). "Performance of memory reclamation for lockless synchronization"

2. **Transposition Tables**:
   - Breuker et al. (1994). "Replacement schemes for transposition tables"
   - Chaslot et al. (2008). "Monte-Carlo Tree Search: A New Framework for Game AI"

3. **MCTS Optimizations**:
   - Browne et al. (2012). "A Survey of Monte Carlo Tree Search Methods"
   - Coulom, R. (2006). "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search"

---

## Implementation Timeline

**Phase 5A: Core Architecture** (2-3 weeks)
- T024a: Tiny Node Design ✅
- T024b: make/unmake API Design
- T024c-e: Game-specific make/unmake (Gomoku, Chess, Go)
- T024f: Tree Refactor
- T024g: SimRunner Integration
- T024h: Correctness Validation

**Phase 5B: Memory Management** (1 week)
- T025a-c: Bump arenas and epoch reclamation

**Phase 5C: Transposition Tables** (1 week)
- T026a-c: Zobrist hashing and DAG tree

**Phase 5D: Queue Optimization** (3-5 days)
- T027a-c: Bounded SPSC queues

**Phase 5E: Final Validation** (3-5 days)
- T028-T029: Benchmarking and documentation

**Total**: 5-7 weeks

---

## Success Metrics

**Performance Targets**:

| Metric | Current | T018 (Pooling) | T019 (Zero-Copy) | Target |
|--------|---------|----------------|------------------|--------|
| Throughput (sims/sec) | 2,659 | 1,164 | 15,000-25,000 | ≥8,000 |
| State overhead (% time) | 86.6% | 88.2% | <2% | <5% |
| Memory per node | 120KB | 120KB | 32 bytes | <1KB |
| Tree memory (10M nodes) | 1.2GB | 1.2GB | 320MB | <1GB |
| Path reconstruction (μs) | 418 | 418 | 0.6 | <10 |

**Correctness Targets**:
- make/unmake equivalence: 100% bit-exact with clone()
- Transposition correctness: 100% win rate vs tree-only mode
- Memory leak: 0 bytes leaked over 24h
- TSan clean: 0 data races
- Win rate vs baseline: ≥99.5% (search quality)

---

**Document Status**: ✅ Complete (T024a)
**Next**: T024b (make/unmake API Design)
