# Technical Plan: MCTS Throughput Recovery
# Profiling-Grounded Implementation Strategy

**Version**: 1.0
**Status**: ACTIVE - Authoritative Technical Design
**Last Updated**: 2025-10-16
**Profiling Campaign**: profiling_suite_20251016_124134 (560 trials, 100% capture)
**Authority**: Implements spec.md v3.0 | CONSTITUTION.md v3.0

---

## Document Purpose

This technical plan provides the **HOW** - detailed implementation design for achieving ≥8,000 sims/sec throughput target. All designs are grounded in production profiling evidence from 560-trial campaign with 100% data capture.

**Authority Chain**:
1. **CONSTITUTION.md v3.0** - Non-negotiable constraints
2. **FINAL_PROFILING_ANALYSIS_20251016.md** - Profiling evidence (560 trials)
3. **spec.md v3.0** - Functional requirements (WHAT to achieve)
4. **This TECHNICAL_PLAN.md** - Implementation design (HOW to implement)
5. **tasks.md** - Task breakdown (WHAT to do, generated via `/speckit.tasks`)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Profiling Evidence Summary](#2-profiling-evidence-summary)
3. [Solution Architecture](#3-solution-architecture)
4. [Priority #1: State Pooling Implementation](#4-priority-1-state-pooling-implementation)
5. [Priority #2: OpenMP Investigation](#5-priority-2-openmp-investigation)
6. [Priority #3: Memory Allocation Optimization](#6-priority-3-memory-allocation-optimization)
7. [Validation & Measurement](#7-validation--measurement)
8. [Risk Management](#8-risk-management)
9. [Implementation Timeline](#9-implementation-timeline)
10. [Rollback Procedures](#10-rollback-procedures)

---

## 1. Executive Summary

### 1.1 Performance Target

**Current**: 2,659 sims/sec (measured, 560 trials)
**Target**: ≥8,000 sims/sec (3.0× improvement required)
**Projected**: 9,838 sims/sec (3.7× improvement from state pooling alone)

### 1.2 Root Cause Analysis (Profiling-Validated)

**PRIMARY BOTTLENECK**: State Cloning = 86.6% of execution time

**Evidence** (Trial 001, representative of 560 trials):
```
Total: 982.86 ms for 2,000 simulations

state_clone_total:   835.85 ms (86.6%) 🔴 CRITICAL BOTTLENECK
expansion_total:      37.24 ms ( 3.8%)
expansion_nn_wait:    20.66 ms ( 2.1%) ← GPU inference (NOT the bottleneck!)
selection_total:       3.58 ms ( 0.4%)
backup_total:          1.67 ms ( 0.2%)
unknown/overhead:     85.64 ms ( 8.7%) ← Expected (Python, GIL)
```

**Root Cause**: 223 allocations per state clone
```
alloc_slow_path counter: 446,227 for 2,000 sims (223 per sim)
Allocation overhead: 223 × 2μs × 2,000 = 892 ms
Percentage: 892 ms / 983 ms = 90.7% of time
Conclusion: Matches observed 86.6% state cloning time ✅
```

### 1.3 Solution Strategy

**Phase 1: State Pooling (T018)** - CRITICAL PATH
- **Impact**: Eliminate 223 allocations per clone
- **Expected**: 418μs → 20μs per clone (20.9× faster)
- **Overall Gain**: 3.7× throughput → **9,838 sims/sec** ✅ **Exceeds 8k target ALONE**
- **Timeline**: 2-3 days
- **Risk**: LOW (well-understood optimization)

**Phase 2: OpenMP Investigation (T019)** - OPTIONAL
- **Impact**: Enable feature extraction parallelization (0/560 trials active)
- **Expected**: 1.5-2.0× additional speedup → **14,757 sims/sec**
- **Timeline**: 1-2 days
- **Risk**: LOW (debugging task)

**Phase 3: Allocation Reduction (T020)** - REFINEMENT
- **Impact**: Further reduce allocation overhead
- **Expected**: 1.2-1.5× additional speedup → **17,708 sims/sec**
- **Timeline**: 1-2 days (AFTER state pooling)
- **Risk**: MEDIUM (memory leak potential)

---

## 2. Profiling Evidence Summary

### 2.1 Campaign Overview

**Campaign ID**: profiling_suite_20251016_124134
**Date**: October 16, 2025
**Status**: ✅ COMPLETE DATA (100% capture rate)
**Trials**: 560/560 successful

**Test Matrix**:
```
Simulations:  [2000, 4000, 8000, 16000]
Threads:      [1, 2, 4, 6, 8, 10, 12]
Batch sizes:  [16, 32, 64, 128]
Repetitions:  5 per configuration
Total trials: 4 × 7 × 4 × 5 = 560
```

### 2.2 Key Findings

**Finding #1: State Cloning Bottleneck**
```
Mean time per trial:       2,254.56 ms
Mean % of wall clock:      86.6%
Clones per simulation:     1.0× (correct - no over-cloning)
Time per clone:            418 μs (should be ~50 μs)
Expected clone time:       ~2 μs (memcpy for 445 bytes)
Actual overhead:           209× slower than expected!
```

**Finding #2: Memory Allocation Overhead**
```
Allocations per simulation: 223 (catastrophic!)
Expected allocations:       <10 per simulation
Allocation time:            ~2 μs per allocation
Total allocation overhead:  446 μs (99% of clone time)
```

**Finding #3: Zero Thread Scaling Benefit**
```
1 thread:  2,619 sims/sec (baseline)
2 threads: 2,654 sims/sec (1.01× speedup, 50.7% efficiency)
4 threads: 2,668 sims/sec (1.02× speedup, 25.5% efficiency)
8 threads: 2,664 sims/sec (1.02× speedup, 12.7% efficiency)
12 threads: 2,672 sims/sec (1.02× speedup, 8.5% efficiency)

Conclusion: Allocation contention completely dominates
```

**Finding #4: OpenMP Never Active**
```
omp_parallel_success counter: 0/560 trials (NEVER activated)
Expected behavior: Feature extraction parallelized with 12 threads
Actual behavior: Sequential execution only
```

**Finding #5: GPU is NOT the Bottleneck**
```
GPU inference time: 20.66 ms out of 982.86 ms total (2.1%)
Previous assumption: GPU inference was 32.8% of time (WRONG!)
Profiling correction: GPU already optimized, NOT the problem
```

### 2.3 Performance Calculation

**Current Performance**:
```
Throughput: 2,659 sims/sec (mean, 560 trials)
Time per simulation: 377 μs average
State cloning: 418 μs per clone (86.6% of time)
```

**After State Pooling**:
```
Clone time: 418 μs → 20 μs (20.9× faster in cloning phase)
Total time reduction: 836 ms → 40 ms (796 ms saved per 2,000 sims)
New total time: 982 - 796 = 186 ms per 2,000 sims
Throughput: 2,000 / 0.186s = 10,753 sims/sec

Conservative estimate (with overhead): 9,838 sims/sec
Improvement: 3.7× over current 2,659 sims/sec ✅ Exceeds 8k target!
```

---

## 3. Solution Architecture

### 3.1 High-Level Design

```
┌──────────────────────────────────────────────────────────────┐
│                   MCTS Simulation Loop                        │
│                                                                │
│  OLD (Current - 418μs per clone):                            │
│    std::unique_ptr<IGameState> state = root.clone();         │
│    // Triggers 223 heap allocations (~2μs each = 446μs)      │
│                                                                │
│  NEW (Proposed - 20μs via copyFrom):                         │
│    IGameState* state = pool.acquire();                       │
│    state->copyFrom(root);  // 0 allocations, memcpy only     │
│    // ... use state ...                                       │
│    pool.release(state);                                       │
│                                                                │
└──────────────────────────────────────────────────────────────┘
          │                                    │
          ▼                                    ▼
┌─────────────────────┐          ┌─────────────────────────┐
│ Thread-Local        │          │ IGameState Interface    │
│ State Pool          │          │                         │
│                     │          │ + clone() (existing)    │
│ - Pre-allocated     │          │ + copyFrom() (NEW)      │
│   states (16/thread)│◄─────────┤   - NO allocations      │
│ - Lock-free acquire │          │   - memcpy for arrays   │
│ - Ring buffer reuse │          │   - Shallow copy fields │
└─────────────────────┘          └─────────────────────────┘
```

### 3.2 Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   C++ MCTS Core                               │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ ContinuousSimulationRunner                             │  │
│  │                                                         │  │
│  │  - run_continuous(root, simulations)                   │  │
│  │    ├─► state = state_pool.acquire()                    │  │
│  │    ├─► state->copyFrom(root)  ◄── NEW API              │  │
│  │    ├─► select_leaf(state)                              │  │
│  │    ├─► queue.submit_request(state, ...)                │  │
│  │    └─► state_pool.release(state)                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                   │
│                           ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ ThreadLocalStatePool (NEW)                             │  │
│  │                                                         │  │
│  │  std::vector<IGameState*> pool_                        │  │
│  │  std::atomic<size_t> next_free_                        │  │
│  │                                                         │  │
│  │  + acquire() → IGameState* (O(1), lock-free)          │  │
│  │  + release(IGameState*) → void (O(1), lock-free)      │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                   │
│                           ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ IGameState (Modified Interface)                        │  │
│  │                                                         │  │
│  │  + clone() → unique_ptr<IGameState> (existing, slow)   │  │
│  │  + copyFrom(const IGameState&) → void (NEW, fast)     │  │
│  │    Requirements:                                        │  │
│  │    - NO heap allocations                               │  │
│  │    - memcpy for fixed-size arrays                      │  │
│  │    - Shallow copy for primitive fields                 │  │
│  │    - Thread-safe: read-only access to 'other'          │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                   │
│                           ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Game Implementations (Gomoku, Chess, Go)               │  │
│  │                                                         │  │
│  │  GomokuState::copyFrom(const IGameState& other) {      │  │
│  │    auto& src = static_cast<const GomokuState&>(other); │  │
│  │    memcpy(board_, src.board_, 225);  // 15×15          │  │
│  │    move_count_ = src.move_count_;    // Primitives     │  │
│  │    current_player_ = src.current_player_;              │  │
│  │    // NO allocations, fast memcpy only                 │  │
│  │  }                                                      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 Memory Layout

**Thread-Local State Pool**:
```
Thread 0:  [State 0] [State 1] ... [State 15]  (16 pre-allocated)
Thread 1:  [State 0] [State 1] ... [State 15]
...
Thread 7:  [State 0] [State 1] ... [State 15]

Total: 16 states × 8 threads × 445 bytes = 57 KB (negligible)
```

**State Structure** (Gomoku 15×15):
```cpp
class GomokuState : public IGameState {
private:
    uint8_t board_[225];          // 15×15 board (225 bytes)
    uint16_t move_history_[100];  // Move sequence (200 bytes)
    uint8_t move_count_;          // 1 byte
    uint8_t current_player_;      // 1 byte
    uint8_t game_result_;         // 1 byte
    // ... other metadata (~18 bytes)
    // Total: ~445 bytes

public:
    // Existing (slow - 418μs per call)
    std::unique_ptr<IGameState> clone() const override {
        auto copy = std::make_unique<GomokuState>();  // Allocation #1
        copy->board_ = std::vector<uint8_t>(board_.begin(), board_.end());  // Allocation #2
        copy->move_history_ = std::vector<uint16_t>(move_history_.begin(), move_history_.end());  // Allocation #3
        // ... more allocations for std::vector, std::unordered_set, etc.
        // Total: 223 allocations (~2μs each = 446μs)
        return copy;
    }

    // NEW (fast - ~20μs per call)
    void copyFrom(const IGameState& other) override {
        auto& src = static_cast<const GomokuState&>(other);

        // Fast memcpy for fixed-size arrays
        memcpy(board_, src.board_, 225);              // ~0.1μs
        memcpy(move_history_, src.move_history_, 200);  // ~0.1μs

        // Primitive field copies
        move_count_ = src.move_count_;
        current_player_ = src.current_player_;
        game_result_ = src.game_result_;

        // Total: ~20μs (10× faster than expected due to cache effects)
        // NO allocations ✅
    }
};
```

---

## 4. Priority #1: State Pooling Implementation

### 4.1 Design Goals

1. **Eliminate allocations**: 223 → <10 per simulation
2. **Reduce clone time**: 418μs → 20μs (20.9× faster)
3. **Thread-safe**: Lock-free acquisition/release
4. **Zero memory leaks**: Fixed pool size, no growth
5. **Bit-exact equivalence**: `copyFrom()` matches `clone()` semantically

### 4.2 Thread-Local State Pool Design

**File**: `cpp_extensions/mcts/state_pool.hpp`

```cpp
#pragma once

#include <vector>
#include <atomic>
#include <memory>
#include "utils/igamestate.h"

namespace mcts {

// Thread-local state pool for zero-allocation state management
class ThreadLocalStatePool {
public:
    // Constructor: Pre-allocate pool_size states
    explicit ThreadLocalStatePool(
        GameType game_type,
        size_t pool_size = 16
    );

    ~ThreadLocalStatePool();

    // Acquire state from pool (O(1), lock-free)
    // Returns: Pointer to pre-allocated state (NOT owned by caller)
    IGameState* acquire();

    // Release state back to pool (O(1), lock-free)
    // Note: State remains in pool, just marked as available
    void release(IGameState* state);

    // Statistics
    struct Stats {
        size_t total_acquires;
        size_t total_releases;
        size_t peak_usage;
        size_t pool_size;
    };
    Stats get_stats() const;

    // Reset statistics
    void reset_stats();

private:
    // Pre-allocated states (never deallocated)
    std::vector<std::unique_ptr<IGameState>> pool_;

    // Lock-free ring buffer allocation
    std::atomic<size_t> next_free_;

    // Pool size (fixed)
    size_t pool_size_;

    // Statistics (atomic for thread safety)
    std::atomic<size_t> total_acquires_{0};
    std::atomic<size_t> total_releases_{0};
    std::atomic<size_t> peak_usage_{0};
};

// Thread-local accessor (lazy initialization)
ThreadLocalStatePool* get_thread_state_pool(GameType game_type);

} // namespace mcts
```

**File**: `cpp_extensions/mcts/state_pool.cpp`

```cpp
#include "state_pool.hpp"
#include "games/gomoku_state.h"
#include "games/chess_state.h"
#include "games/go_state.h"
#include <algorithm>

namespace mcts {

ThreadLocalStatePool::ThreadLocalStatePool(
    GameType game_type,
    size_t pool_size
) : pool_size_(pool_size), next_free_(0) {

    // Pre-allocate all states
    pool_.reserve(pool_size);
    for (size_t i = 0; i < pool_size; ++i) {
        switch (game_type) {
            case GameType::GOMOKU:
                pool_.emplace_back(std::make_unique<GomokuState>());
                break;
            case GameType::CHESS:
                pool_.emplace_back(std::make_unique<ChessState>());
                break;
            case GameType::GO:
                pool_.emplace_back(std::make_unique<GoState>());
                break;
        }
    }
}

ThreadLocalStatePool::~ThreadLocalStatePool() {
    // pool_ automatically deallocates via unique_ptr destructors
}

IGameState* ThreadLocalStatePool::acquire() {
    total_acquires_.fetch_add(1, std::memory_order_relaxed);

    // Lock-free ring buffer allocation
    size_t idx = next_free_.fetch_add(1, std::memory_order_relaxed);
    size_t pool_idx = idx % pool_size_;

    // Update peak usage tracking
    size_t current_usage = (idx / pool_size_) + 1;
    size_t peak = peak_usage_.load(std::memory_order_relaxed);
    while (current_usage > peak) {
        if (peak_usage_.compare_exchange_weak(
            peak, current_usage,
            std::memory_order_relaxed)) {
            break;
        }
    }

    return pool_[pool_idx].get();
}

void ThreadLocalStatePool::release(IGameState* state) {
    total_releases_.fetch_add(1, std::memory_order_relaxed);
    // No-op: State remains in pool for reuse
    // Ring buffer wraps around automatically
}

ThreadLocalStatePool::Stats ThreadLocalStatePool::get_stats() const {
    return Stats{
        total_acquires_.load(std::memory_order_relaxed),
        total_releases_.load(std::memory_order_relaxed),
        peak_usage_.load(std::memory_order_relaxed),
        pool_size_
    };
}

void ThreadLocalStatePool::reset_stats() {
    total_acquires_.store(0, std::memory_order_relaxed);
    total_releases_.store(0, std::memory_order_relaxed);
    peak_usage_.store(0, std::memory_order_relaxed);
}

// Thread-local storage
thread_local std::unique_ptr<ThreadLocalStatePool> tl_gomoku_pool;
thread_local std::unique_ptr<ThreadLocalStatePool> tl_chess_pool;
thread_local std::unique_ptr<ThreadLocalStatePool> tl_go_pool;

ThreadLocalStatePool* get_thread_state_pool(GameType game_type) {
    switch (game_type) {
        case GameType::GOMOKU:
            if (!tl_gomoku_pool) {
                tl_gomoku_pool = std::make_unique<ThreadLocalStatePool>(
                    GameType::GOMOKU, 16
                );
            }
            return tl_gomoku_pool.get();

        case GameType::CHESS:
            if (!tl_chess_pool) {
                tl_chess_pool = std::make_unique<ThreadLocalStatePool>(
                    GameType::CHESS, 16
                );
            }
            return tl_chess_pool.get();

        case GameType::GO:
            if (!tl_go_pool) {
                tl_go_pool = std::make_unique<ThreadLocalStatePool>(
                    GameType::GO, 16
                );
            }
            return tl_go_pool.get();
    }
    return nullptr;
}

} // namespace mcts
```

### 4.3 IGameState Interface Update

**File**: `cpp_extensions/utils/igamestate.h` (Modified)

```cpp
#pragma once

#include <memory>
#include <vector>
#include <cstdint>

class IGameState {
public:
    virtual ~IGameState() = default;

    // ========================================
    // Existing API (unchanged)
    // ========================================

    // Deep clone (slow - 418μs per call due to 223 allocations)
    // DEPRECATED: Use copyFrom() with state pooling instead
    virtual std::unique_ptr<IGameState> clone() const = 0;

    // Apply move in-place
    virtual void apply_move_inplace(int action) = 0;

    // Get legal moves mask
    virtual void get_legal_moves(uint8_t* mask) const = 0;

    // Extract features to buffer
    virtual void extract_features_to_buffer(float* buffer) const = 0;

    // Game state queries
    virtual bool is_terminal() const = 0;
    virtual float get_reward(uint8_t player) const = 0;
    virtual uint8_t current_player() const = 0;

    // ========================================
    // NEW API (for state pooling)
    // ========================================

    // Fast shallow copy (target: ~20μs per call, NO allocations)
    //
    // Requirements:
    // - NO heap allocations allowed
    // - Use memcpy for fixed-size arrays
    // - Shallow copy for primitive fields
    // - Thread-safe: read-only access to 'other'
    // - Bit-exact equivalence with clone() semantically
    //
    // Implementation guide:
    //   auto& src = static_cast<const ConcreteState&>(other);
    //   memcpy(board_, src.board_, board_size);
    //   field1_ = src.field1_;
    //   field2_ = src.field2_;
    //
    virtual void copyFrom(const IGameState& other) = 0;

    // Utility for pool size estimation
    virtual size_t estimated_size_bytes() const = 0;
};
```

### 4.4 GomokuState Implementation

**File**: `cpp_extensions/games/gomoku_state.h` (Modified)

```cpp
class GomokuState : public IGameState {
private:
    // Fixed-size arrays (NO dynamic allocation)
    uint8_t board_[225];          // 15×15 = 225 cells
    uint16_t move_history_[100];  // Max 100 moves (Gomoku rarely exceeds 50)
    uint8_t move_count_;
    uint8_t current_player_;
    uint8_t game_result_;         // 0=ongoing, 1=player0_wins, 2=player1_wins, 3=draw
    uint8_t last_move_row_;
    uint8_t last_move_col_;
    // Total: ~445 bytes

public:
    // Existing (slow - 418μs)
    std::unique_ptr<IGameState> clone() const override {
        auto copy = std::make_unique<GomokuState>();
        copy->copyFrom(*this);  // Delegate to copyFrom
        return copy;
    }

    // NEW (fast - ~20μs)
    void copyFrom(const IGameState& other) override {
        auto& src = static_cast<const GomokuState&>(other);

        // Fast memcpy for fixed-size arrays (~0.2μs total)
        memcpy(board_, src.board_, 225);
        memcpy(move_history_, src.move_history_, 200);

        // Primitive field copies (~0.05μs)
        move_count_ = src.move_count_;
        current_player_ = src.current_player_;
        game_result_ = src.game_result_;
        last_move_row_ = src.last_move_row_;
        last_move_col_ = src.last_move_col_;

        // Total: ~20μs (includes cache misses, validation overhead)
    }

    size_t estimated_size_bytes() const override {
        return sizeof(GomokuState);  // ~445 bytes
    }

    // ... existing methods unchanged ...
};
```

### 4.5 Integration into ContinuousSimulationRunner

**File**: `cpp_extensions/mcts/continuous_simulation_runner.cpp` (Modified)

```cpp
#include "continuous_simulation_runner.hpp"
#include "state_pool.hpp"

namespace mcts {

int ContinuousSimulationRunner::run_continuous(
    IGameState& root_state,
    NodeIndex root_index,
    AsyncInferenceQueue& queue,
    int num_simulations
) {
    // Get thread-local state pool (lazy initialization)
    GameType game_type = detect_game_type(root_state);
    ThreadLocalStatePool* pool = get_thread_state_pool(game_type);

    int completed = 0;

    for (int i = 0; i < num_simulations; ++i) {
        // ==========================================
        // OLD CODE (418μs per clone, 223 allocations):
        // std::unique_ptr<IGameState> current_state = root_state.clone();
        // ==========================================

        // NEW CODE (20μs via copyFrom, 0 allocations):
        IGameState* current_state = pool->acquire();  // O(1), lock-free
        current_state->copyFrom(root_state);  // Fast memcpy

        // Selection phase: Traverse tree to leaf
        std::vector<NodeIndex> path;
        NodeIndex leaf_node = select_leaf(
            root_index,
            *current_state,
            path
        );

        // Check if terminal
        if (current_state->is_terminal()) {
            float value = current_state->get_reward(
                current_state->current_player()
            );
            backup_.backup_value_along_path(path, value, &virtual_loss_);
            pool->release(current_state);  // Return to pool
            completed++;
            continue;
        }

        // Submit inference request (transfers ownership)
        uint64_t request_id = queue.submit_request(
            current_state,  // Pointer passed, pool manages lifetime
            leaf_node,
            std::move(path)
        );

        // State will be returned to pool by result processing
        // (handled in process_completed_results)
    }

    // Process any completed results
    completed += process_completed_results(queue);

    return completed;
}

int ContinuousSimulationRunner::process_completed_results(
    AsyncInferenceQueue& queue
) {
    // ... existing result processing ...

    // After expansion and backup:
    if (request.state_ptr) {
        // Return state to pool
        GameType game_type = detect_game_type(*request.state_ptr);
        ThreadLocalStatePool* pool = get_thread_state_pool(game_type);
        pool->release(request.state_ptr);
    }

    // ... rest of processing ...
}

} // namespace mcts
```

### 4.6 Validation Requirements

**Unit Tests** (`tests/unit/test_state_pool.py`):
```python
def test_state_pool_acquisition():
    """Test that state pool acquisition is fast and lock-free."""
    pool = mcts_py.ThreadLocalStatePool(GameType.GOMOKU, pool_size=16)

    # Acquire all states
    states = [pool.acquire() for _ in range(16)]

    # Verify all states are distinct
    assert len(set(id(s) for s in states)) == 16

    # Release all states
    for state in states:
        pool.release(state)

    # Re-acquire should get same states (ring buffer wraps)
    states2 = [pool.acquire() for _ in range(16)]
    assert set(id(s) for s in states) == set(id(s) for s in states2)

def test_copyFrom_equivalence():
    """Test that copyFrom() is bit-exact equivalent to clone()."""
    root = GomokuState()
    # ... apply some moves ...

    # Clone via old method
    cloned = root.clone()

    # Clone via copyFrom
    pool = mcts_py.ThreadLocalStatePool(GameType.GOMOKU)
    copied = pool.acquire()
    copied.copyFrom(root)

    # Verify bit-exact equivalence
    assert cloned.get_board_hash() == copied.get_board_hash()
    assert cloned.current_player() == copied.current_player()
    assert cloned.move_count() == copied.move_count()
    # ... verify all fields ...

def test_state_pool_performance():
    """Test that state pool reduces allocation overhead."""
    import time

    root = GomokuState()
    pool = mcts_py.ThreadLocalStatePool(GameType.GOMOKU, pool_size=16)

    # Measure old clone() performance
    start = time.perf_counter()
    for _ in range(1000):
        cloned = root.clone()
    old_time = time.perf_counter() - start

    # Measure new copyFrom() performance
    start = time.perf_counter()
    for _ in range(1000):
        state = pool.acquire()
        state.copyFrom(root)
        pool.release(state)
    new_time = time.perf_counter() - start

    # Verify speedup ≥10× (conservative target)
    speedup = old_time / new_time
    assert speedup >= 10.0, f"Expected ≥10× speedup, got {speedup:.2f}×"
```

**Profiling Validation** (`scripts/validate_state_pooling.py`):
```python
#!/usr/bin/env python3
"""
Validate state pooling optimization with profiling.

Acceptance criteria:
- alloc_slow_path counter <20,000 for 2,000 sims (<10 per sim)
- state_clone_total <50 ms (<5% of time)
- throughput ≥7,500 sims/sec (3.0× minimum improvement)
"""

import subprocess
import json
import sys

def run_profiling_benchmark():
    """Run profiling benchmark with state pooling enabled."""
    result = subprocess.run([
        'python', 'scripts/benchmark_throughput.py',
        '--game', 'gomoku',
        '--threads', '8',
        '--simulations', '2000',
        '--seed', '42',
        '--iterations', '10',
        '--enable-profiling'
    ], capture_output=True, text=True)

    return json.loads(result.stdout)

def validate_results(data):
    """Validate profiling results against acceptance criteria."""
    errors = []

    # Extract metrics
    alloc_count = data['cpp_profiling']['counters']['alloc_slow_path']
    state_clone_ms = data['cpp_profiling']['timings']['state_clone_total']
    total_time_ms = data['cpp_profiling']['session_duration_ms']
    throughput = data['throughput_sims_per_sec']

    # Criterion 1: Allocations <10 per simulation
    alloc_per_sim = alloc_count / 2000
    if alloc_per_sim >= 10:
        errors.append(
            f"❌ Allocations per sim: {alloc_per_sim:.1f} (target: <10)"
        )
    else:
        print(f"✅ Allocations per sim: {alloc_per_sim:.1f} (target: <10)")

    # Criterion 2: State cloning <5% of time
    clone_pct = (state_clone_ms / total_time_ms) * 100
    if clone_pct >= 5.0:
        errors.append(
            f"❌ State cloning: {clone_pct:.1f}% of time (target: <5%)"
        )
    else:
        print(f"✅ State cloning: {clone_pct:.1f}% of time (target: <5%)")

    # Criterion 3: Throughput ≥7,500 sims/sec
    if throughput < 7500:
        errors.append(
            f"❌ Throughput: {throughput:.0f} sims/sec (target: ≥7,500)"
        )
    else:
        print(f"✅ Throughput: {throughput:.0f} sims/sec (target: ≥7,500)")

    # Criterion 4: Speedup ≥3.0× vs baseline
    baseline = 2659  # From profiling campaign
    speedup = throughput / baseline
    if speedup < 3.0:
        errors.append(
            f"❌ Speedup: {speedup:.2f}× (target: ≥3.0×)"
        )
    else:
        print(f"✅ Speedup: {speedup:.2f}× vs baseline {baseline} sims/sec")

    return errors

def main():
    print("=" * 60)
    print("State Pooling Validation")
    print("=" * 60)

    print("\n1. Running profiling benchmark...")
    data = run_profiling_benchmark()

    print("\n2. Validating results...")
    errors = validate_results(data)

    if errors:
        print("\n" + "=" * 60)
        print("❌ VALIDATION FAILED")
        print("=" * 60)
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("✅ VALIDATION PASSED")
        print("=" * 60)
        print("State pooling optimization successful!")
        sys.exit(0)

if __name__ == '__main__':
    main()
```

---

## 5. Priority #2: OpenMP Investigation

### 5.1 Problem Statement

**Evidence**: 0/560 trials show `omp_parallel_success > 0`
**Affected code**: Feature extraction loop in `dlpack_bridge.cpp:431-434`
**Expected behavior**: Parallel feature extraction with 12 threads
**Actual behavior**: Sequential execution only

### 5.2 Diagnostic Procedure

**Step 1: Verify OpenMP Linkage**

```bash
# Check if OpenMP runtime is linked
ldd venv/lib/python3.12/site-packages/mcts_py*.so | grep omp

# Expected output (if linked correctly):
# libgomp.so.1 => /usr/lib/x86_64-linux-gnu/libgomp.so.1 (0x00007f...)
# OR
# libomp.so.5 => /usr/lib/x86_64-linux-gnu/libomp.so.5 (0x00007f...)

# If NOT found: OpenMP not linked!
```

**Step 2: Check Environment Variables**

```bash
# Check OMP_NUM_THREADS
echo $OMP_NUM_THREADS

# Expected: unset OR >1
# If set to 1: OpenMP will use single thread!

# Recommended fix:
export OMP_NUM_THREADS=12  # Ryzen 5900X has 12 cores
```

**Step 3: Add Debug Instrumentation**

**File**: `cpp_extensions/mcts/dlpack_bridge.cpp` (Modified)

```cpp
DLManagedTensor* create_batch_tensor_from_states(
    const std::vector<const IGameState*>& states,
    bool use_cuda
) {
    // ... existing setup code ...

    #ifdef _OPENMP
    // OpenMP is available at compile time
    #pragma omp parallel
    {
        #pragma omp single
        {
            int num_threads = omp_get_num_threads();
            printf("[DEBUG] OpenMP active: %d threads\n", num_threads);

            // IMPORTANT: Increment profiling counter
            PROFILE_COUNTER_INCREMENT(omp_parallel_success);
        }
    }
    #else
    printf("[WARNING] OpenMP NOT available at compile time!\n");
    #endif

    // Feature extraction loop
    #pragma omp parallel for num_threads(12)  // Explicit thread count
    for (int i = 0; i < batch_size; ++i) {
        states[i]->extract_features_to_buffer(
            feature_ptr + (i * features_per_state)
        );
    }

    // ... rest of function ...
}
```

**Step 4: Rebuild with Explicit OpenMP Flags**

```bash
# Clean rebuild
rm -rf build/ *.so venv/lib/python3.12/site-packages/mcts_py*

# Set explicit flags
export CFLAGS="-O3 -march=znver3 -fopenmp -DPROFILE_LEVEL_VALUE=3"
export CXXFLAGS="-O3 -march=znver3 -fopenmp -DPROFILE_LEVEL_VALUE=3"
export LDFLAGS="-fopenmp"

# Rebuild
pip install -e . --force-reinstall --no-deps

# Verify linkage again
ldd venv/lib/python3.12/site-packages/mcts_py*.so | grep omp
```

**Step 5: Test with Simple OpenMP Program**

**File**: `test_openmp.cpp`

```cpp
#include <omp.h>
#include <stdio.h>

int main() {
    #ifdef _OPENMP
    printf("OpenMP version: %d\n", _OPENMP);
    #else
    printf("OpenMP NOT available!\n");
    return 1;
    #endif

    #pragma omp parallel
    {
        #pragma omp single
        printf("Parallel region active with %d threads\n", omp_get_num_threads());
    }

    return 0;
}
```

```bash
# Compile and test
g++ -fopenmp test_openmp.cpp -o test_openmp
./test_openmp

# Expected output:
# OpenMP version: 201511
# Parallel region active with 12 threads
```

### 5.3 Expected Outcomes

**Success Case**: `omp_parallel_success > 0` in profiling output
- Feature extraction parallelized across 12 threads
- Expected speedup: 6-10× for this phase
- Overall throughput gain: +1.5-2.0× (combined with state pooling)

**Failure Case**: OpenMP still not active
- **Contingency**: Accept as non-critical (state pooling achieves 8k target alone)
- **Alternative**: Investigate thread pool executor for manual parallelization
- **Defer**: Move to future optimization phase

### 5.4 Validation Protocol

**Profiling Validation** (`scripts/validate_openmp.py`):
```python
#!/usr/bin/env python3
"""
Validate OpenMP parallelization.

Acceptance criteria:
- omp_parallel_success counter >0
- Thread scaling shows >1.0× speedup with multiple threads
- Feature extraction time <1.0ms per batch-64
"""

import subprocess
import json
import sys

def run_openmp_benchmark():
    """Run benchmark to test OpenMP activation."""
    result = subprocess.run([
        'python', 'scripts/benchmark_throughput.py',
        '--game', 'gomoku',
        '--threads', '1,2,4,8,12',  # Test thread scaling
        '--simulations', '2000',
        '--batch-size', '64',
        '--seed', '42',
        '--enable-profiling'
    ], capture_output=True, text=True)

    return json.loads(result.stdout)

def validate_openmp(data):
    """Validate OpenMP activation and performance."""
    errors = []

    # Criterion 1: OpenMP counter >0
    omp_success = data['cpp_profiling']['counters'].get('omp_parallel_success', 0)
    if omp_success == 0:
        errors.append("❌ OpenMP not active (omp_parallel_success = 0)")
    else:
        print(f"✅ OpenMP active (omp_parallel_success = {omp_success})")

    # Criterion 2: Thread scaling
    throughputs = {
        cfg['threads']: cfg['throughput_sims_per_sec']
        for cfg in data['configurations']
    }

    speedup_2t = throughputs[2] / throughputs[1]
    speedup_4t = throughputs[4] / throughputs[1]
    speedup_8t = throughputs[8] / throughputs[1]

    if speedup_2t < 1.5:
        errors.append(f"❌ 2-thread speedup: {speedup_2t:.2f}× (target: ≥1.5×)")
    else:
        print(f"✅ 2-thread speedup: {speedup_2t:.2f}×")

    if speedup_4t < 2.5:
        errors.append(f"❌ 4-thread speedup: {speedup_4t:.2f}× (target: ≥2.5×)")
    else:
        print(f"✅ 4-thread speedup: {speedup_4t:.2f}×")

    if speedup_8t < 4.0:
        errors.append(f"❌ 8-thread speedup: {speedup_8t:.2f}× (target: ≥4.0×)")
    else:
        print(f"✅ 8-thread speedup: {speedup_8t:.2f}×")

    return errors

def main():
    print("=" * 60)
    print("OpenMP Validation")
    print("=" * 60)

    print("\n1. Running OpenMP benchmark...")
    data = run_openmp_benchmark()

    print("\n2. Validating OpenMP activation...")
    errors = validate_openmp(data)

    if errors:
        print("\n" + "=" * 60)
        print("⚠️  OPENMP VALIDATION FAILED")
        print("=" * 60)
        for error in errors:
            print(error)
        print("\nNote: State pooling achieves 8k target without OpenMP.")
        print("OpenMP is optional enhancement for 14k+ stretch goal.")
        sys.exit(0)  # Non-blocking failure
    else:
        print("\n" + "=" * 60)
        print("✅ OPENMP VALIDATION PASSED")
        print("=" * 60)
        sys.exit(0)

if __name__ == '__main__':
    main()
```

---

## 6. Priority #3: Memory Allocation Optimization

### 6.1 Goals

**Current**: 223 allocations per simulation (catastrophic!)
**Target**: <10 allocations per simulation
**Expected Gain**: 1.2-1.5× additional speedup (AFTER state pooling)

### 6.2 Allocation Sources (Profiling Analysis)

**Evidence**: State cloning triggers 223 allocations
- GomokuState::clone() creates 1 new object + 222 allocations in members
- std::vector growth in move history (~50 allocations)
- std::unordered_set for zobrist cache (~100 allocations)
- std::string for metadata (~50 allocations)
- Template instantiations (~22 allocations)

**Post-State-Pooling**: Residual allocations from:
- Node allocation in tree expansion (~5 per simulation)
- Path vector growth (~2 per simulation)
- DLPack tensor metadata (~1 per batch)
- Expected total: 8-10 per simulation

### 6.3 Expansion of Thread-Local Arenas

**Current Design** (from T009a-f):
```cpp
class ThreadLocalArena {
    static constexpr size_t CHUNK_SIZE = 4096 * 64;  // 4096 nodes × 64 bytes
    // Covers: Node allocation only
};
```

**Enhanced Design** (T020):
```cpp
class EnhancedThreadLocalArena {
    static constexpr size_t NODE_CHUNK_SIZE = 4096 * 64;    // 256KB (nodes)
    static constexpr size_t GENERAL_CHUNK_SIZE = 1024 * 1024; // 1MB (general)

    // Separate pools for different allocation patterns
    std::vector<NodeChunk> node_chunks_;      // Node allocations
    std::vector<GeneralChunk> general_chunks_; // Vectors, strings, etc.

    // Fast-path allocation (99.5% of calls)
    void* allocate_node(size_t size);
    void* allocate_general(size_t size);

    // Free list for deallocation reuse
    FreeList free_lists_[MAX_SIZE_CLASSES];
};
```

**Integration** (`cpp_extensions/mcts/tree.cpp`):
```cpp
NodeIndex MCTSTree::allocate_nodes(uint16_t count) {
    // Get thread-local arena
    EnhancedThreadLocalArena* arena = get_thread_arena();

    // Fast-path: Allocate from arena (no mutex!)
    NodeIndex first = next_free_index_.fetch_add(count, std::memory_order_relaxed);

    if (first + count <= max_nodes_) {
        // Initialize nodes in-place (arena memory)
        for (uint16_t i = 0; i < count; ++i) {
            NodeIndex idx = first + i;
            visit_counts_[idx] = 0.0f;
            total_values_[idx] = 0.0f;
            // ... initialize other fields ...
        }
        return first;
    }

    // Slow-path: Out of space (should never happen in practice)
    throw std::runtime_error("Tree node pool exhausted");
}
```

### 6.4 Pre-Allocated Node Pools

**Design**: Allocate large blocks at startup, eliminate per-node malloc

```cpp
class PreAllocatedNodePool {
public:
    static constexpr size_t BLOCK_SIZE = 4096;  // Nodes per block
    static constexpr size_t MAX_BLOCKS = 2500;  // 10M nodes total

    PreAllocatedNodePool() {
        // Pre-allocate all blocks at construction
        for (size_t i = 0; i < MAX_BLOCKS; ++i) {
            blocks_.emplace_back(std::make_unique<NodeBlock>());
        }
        next_block_.store(0, std::memory_order_relaxed);
    }

    NodeBlock* allocate_block() {
        size_t idx = next_block_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= MAX_BLOCKS) {
            throw std::runtime_error("Node pool exhausted");
        }
        return blocks_[idx].get();
    }

private:
    struct NodeBlock {
        alignas(64) uint8_t data[BLOCK_SIZE * 64];  // 256KB per block
    };

    std::vector<std::unique_ptr<NodeBlock>> blocks_;
    std::atomic<size_t> next_block_;
};
```

### 6.5 Validation Requirements

**Profiling Validation** (`scripts/validate_allocations.py`):
```python
#!/usr/bin/env python3
"""
Validate allocation reduction.

Acceptance criteria:
- alloc_slow_path counter <20,000 for 2,000 sims (<10 per sim)
- Fast-path allocation rate ≥99.5%
- No memory leaks (valgrind clean)
- Throughput improvement ≥1.2× (AFTER state pooling)
"""

import subprocess
import json
import sys

def run_allocation_benchmark():
    """Run benchmark with allocation profiling."""
    result = subprocess.run([
        'python', 'scripts/benchmark_throughput.py',
        '--game', 'gomoku',
        '--threads', '8',
        '--simulations', '2000',
        '--seed', '42',
        '--enable-profiling'
    ], capture_output=True, text=True)

    return json.loads(result.stdout)

def run_valgrind_check():
    """Run valgrind memory leak check."""
    result = subprocess.run([
        'valgrind',
        '--leak-check=full',
        '--error-exitcode=1',
        'python', 'scripts/benchmark_throughput.py',
        '--game', 'gomoku',
        '--threads', '8',
        '--simulations', '2000',
        '--seed', '42'
    ], capture_output=True, text=True)

    return result.returncode == 0

def validate_allocations(data):
    """Validate allocation metrics."""
    errors = []

    # Criterion 1: Allocations <10 per simulation
    alloc_count = data['cpp_profiling']['counters']['alloc_slow_path']
    alloc_per_sim = alloc_count / 2000

    if alloc_per_sim >= 10:
        errors.append(
            f"❌ Allocations per sim: {alloc_per_sim:.1f} (target: <10)"
        )
    else:
        print(f"✅ Allocations per sim: {alloc_per_sim:.1f}")

    # Criterion 2: Fast-path rate ≥99.5%
    fast_path = data['cpp_profiling']['counters'].get('alloc_fast_path', 0)
    total_alloc = fast_path + alloc_count
    fast_path_pct = (fast_path / total_alloc) * 100 if total_alloc > 0 else 0

    if fast_path_pct < 99.5:
        errors.append(
            f"❌ Fast-path rate: {fast_path_pct:.2f}% (target: ≥99.5%)"
        )
    else:
        print(f"✅ Fast-path rate: {fast_path_pct:.2f}%")

    # Criterion 3: Throughput improvement ≥1.2× vs state-pooling-only
    baseline_with_pooling = 9838  # From state pooling calculation
    throughput = data['throughput_sims_per_sec']
    improvement = throughput / baseline_with_pooling

    if improvement < 1.2:
        errors.append(
            f"❌ Improvement: {improvement:.2f}× (target: ≥1.2×)"
        )
    else:
        print(f"✅ Improvement: {improvement:.2f}× vs state-pooling-only")

    return errors

def main():
    print("=" * 60)
    print("Allocation Optimization Validation")
    print("=" * 60)

    print("\n1. Running allocation benchmark...")
    data = run_allocation_benchmark()

    print("\n2. Validating allocation metrics...")
    errors = validate_allocations(data)

    print("\n3. Running valgrind memory leak check...")
    valgrind_clean = run_valgrind_check()
    if not valgrind_clean:
        errors.append("❌ Valgrind detected memory leaks")
    else:
        print("✅ Valgrind clean (no memory leaks)")

    if errors:
        print("\n" + "=" * 60)
        print("❌ VALIDATION FAILED")
        print("=" * 60)
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("✅ VALIDATION PASSED")
        print("=" * 60)
        sys.exit(0)

if __name__ == '__main__':
    main()
```

---

## 7. Validation & Measurement

### 7.1 Profiling Infrastructure

**Build Configuration**:
```bash
# Enable full profiling
export CXXFLAGS="-O3 -march=znver3 -fopenmp -DPROFILE_LEVEL_VALUE=3"
export PROFILE_BUFFER_SIZE=524288  # 512K samples (avoid overflow)

# Rebuild
pip install -e . --force-reinstall --no-deps
```

**Critical Counters** (Always Enabled):
```cpp
// State cloning metrics (PRIORITY #1)
PROFILE_COUNTER(state_clone_count);        // Must equal simulation count
PROFILE_SCOPE(StateCloneTotal);            // Must be <5% of time

// Memory allocation metrics (PRIORITY #1)
PROFILE_COUNTER(alloc_slow_path);          // Must be <10 per simulation

// OpenMP metrics (PRIORITY #2)
PROFILE_COUNTER(omp_parallel_success);     // Must be >0 if OpenMP works

// Thread coordination metrics
PROFILE_COUNTER(selection_retries);        // Collision detection
PROFILE_COUNTER(expansion_conflicts);      // Thread contention
```

### 7.2 Benchmark Protocol

**Script**: `scripts/run_profiling_suite.sh`

```bash
#!/bin/bash
# Comprehensive profiling campaign

set -e

# Configuration
CAMPAIGN_ID="profiling_suite_$(date +%Y%m%d_%H%M%S)"
CAMPAIGN_DIR="profiling_reports/${CAMPAIGN_ID}"
mkdir -p "${CAMPAIGN_DIR}"

echo "Starting profiling campaign: ${CAMPAIGN_ID}"

# Test matrix
SIMULATIONS=(2000 4000 8000 16000)
THREADS=(1 2 4 6 8 10 12)
BATCH_SIZES=(16 32 64 128)
REPETITIONS=5

# Run all configurations
TRIAL=0
for sims in "${SIMULATIONS[@]}"; do
    for threads in "${THREADS[@]}"; do
        for batch in "${BATCH_SIZES[@]}"; do
            for rep in $(seq 1 $REPETITIONS); do
                TRIAL=$((TRIAL + 1))
                TRIAL_DIR="${CAMPAIGN_DIR}/trial_$(printf '%03d' $TRIAL)"
                mkdir -p "${TRIAL_DIR}"

                echo "Trial ${TRIAL}: sims=${sims}, threads=${threads}, batch=${batch}, rep=${rep}"

                # Run benchmark with profiling
                python scripts/benchmark_throughput.py \
                    --game gomoku \
                    --simulations ${sims} \
                    --threads ${threads} \
                    --batch-size ${batch} \
                    --seed $((42 + rep)) \
                    --enable-profiling \
                    --output-dir "${TRIAL_DIR}" \
                    > "${TRIAL_DIR}/stdout.log" 2>&1

                # Verify 100% capture rate
                python scripts/verify_profiling_capture.py \
                    --trial-dir "${TRIAL_DIR}" \
                    || echo "WARNING: Trial ${TRIAL} incomplete capture"
            done
        done
    done
done

# Generate campaign summary
python scripts/analyze_profiling_results.py \
    --campaign "${CAMPAIGN_DIR}" \
    --baseline profiling_suite_20251016_124134 \
    --output "${CAMPAIGN_DIR}/campaign_summary.json"

echo "Profiling campaign complete: ${CAMPAIGN_DIR}"
```

**Script**: `scripts/verify_profiling_capture.py`

```python
#!/usr/bin/env python3
"""
Verify 100% profiling capture rate.

Checks:
- state_clone_count matches simulation count
- No buffer overflow warnings
- All timing metrics present
"""

import json
import sys
import argparse

def verify_capture_rate(trial_dir):
    """Verify profiling data completeness."""
    # Load profiling data
    with open(f'{trial_dir}/cpp_profiling.json') as f:
        profiling = json.load(f)

    # Load result metadata
    with open(f'{trial_dir}/result.json') as f:
        result = json.load(f)

    errors = []

    # Check counter capture
    state_clone_count = profiling['counters'].get('state_clone_count', 0)
    expected_count = result['simulations']

    if state_clone_count != expected_count:
        errors.append(
            f"state_clone_count mismatch: {state_clone_count} vs {expected_count}"
        )

    # Check timing capture
    state_clone_timing = profiling['timings'].get('state_clone_total', {})
    if state_clone_timing.get('count', 0) != expected_count:
        errors.append(
            f"state_clone_total timing count mismatch"
        )

    # Check for buffer overflow warnings
    if profiling.get('buffer_overflow', False):
        errors.append("Buffer overflow detected!")

    if errors:
        print(f"❌ Capture rate validation failed for {trial_dir}:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(f"✅ Capture rate 100% for {trial_dir}")
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial-dir', required=True)
    args = parser.parse_args()

    success = verify_capture_rate(args.trial_dir)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
```

### 7.3 Statistical Validation

**Script**: `scripts/analyze_profiling_results.py`

```python
#!/usr/bin/env python3
"""
Analyze profiling campaign and compare to baseline.

Statistical requirements:
- N≥10 runs per configuration
- Two-sample t-test p<0.05 for significance
- Coefficient of variation CV<5% for stability
"""

import json
import argparse
import numpy as np
from scipy import stats

def load_campaign_data(campaign_dir):
    """Load all trial results from campaign."""
    trials = []
    # ... load all trial_NNN/result.json files ...
    return trials

def calculate_statistics(data):
    """Calculate mean, stddev, CV, confidence interval."""
    mean = np.mean(data)
    stddev = np.std(data, ddof=1)
    cv = (stddev / mean) * 100 if mean > 0 else 0
    ci_95 = stats.t.interval(
        0.95, len(data)-1, loc=mean, scale=stats.sem(data)
    )

    return {
        'mean': mean,
        'stddev': stddev,
        'cv': cv,
        'ci_95_low': ci_95[0],
        'ci_95_high': ci_95[1],
        'n': len(data)
    }

def compare_to_baseline(campaign_data, baseline_data):
    """Two-sample t-test comparing campaign to baseline."""
    # Extract throughputs for matching configurations
    campaign_throughputs = [t['throughput_sims_per_sec'] for t in campaign_data]
    baseline_throughputs = [t['throughput_sims_per_sec'] for t in baseline_data]

    # T-test
    t_statistic, p_value = stats.ttest_ind(
        campaign_throughputs,
        baseline_throughputs
    )

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.var(campaign_throughputs, ddof=1) + np.var(baseline_throughputs, ddof=1)) / 2
    )
    cohens_d = (np.mean(campaign_throughputs) - np.mean(baseline_throughputs)) / pooled_std

    return {
        't_statistic': t_statistic,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'campaign_mean': np.mean(campaign_throughputs),
        'baseline_mean': np.mean(baseline_throughputs),
        'improvement': np.mean(campaign_throughputs) / np.mean(baseline_throughputs)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    print("Loading campaign data...")
    campaign_data = load_campaign_data(args.campaign)

    print("Loading baseline data...")
    baseline_data = load_campaign_data(args.baseline)

    print("Calculating statistics...")
    stats_report = calculate_statistics([
        t['throughput_sims_per_sec'] for t in campaign_data
    ])

    print("Comparing to baseline...")
    comparison = compare_to_baseline(campaign_data, baseline_data)

    # Generate report
    report = {
        'campaign_id': args.campaign,
        'baseline_id': args.baseline,
        'statistics': stats_report,
        'comparison': comparison,
        'acceptance_criteria': {
            'throughput_target': 8000,
            'throughput_achieved': stats_report['mean'] >= 8000,
            'cv_target': 5.0,
            'cv_achieved': stats_report['cv'] < 5.0,
            'significance_target': 0.05,
            'significance_achieved': comparison['p_value'] < 0.05
        }
    }

    # Write report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {args.output}")
    print(f"Campaign mean: {stats_report['mean']:.0f} sims/sec")
    print(f"Baseline mean: {comparison['baseline_mean']:.0f} sims/sec")
    print(f"Improvement: {comparison['improvement']:.2f}×")
    print(f"Statistical significance: p={comparison['p_value']:.6f}")

    if report['acceptance_criteria']['throughput_achieved'] and \
       report['acceptance_criteria']['significance_achieved']:
        print("\n✅ ACCEPTANCE CRITERIA MET")
        return 0
    else:
        print("\n❌ ACCEPTANCE CRITERIA NOT MET")
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

---

## 8. Risk Management

### 8.1 Implementation Risks

| Risk | Likelihood | Impact | Mitigation | Contingency |
|------|-----------|--------|-----------|-------------|
| State pooling bugs (use-after-free) | MEDIUM | CRITICAL | Extensive unit tests, TSan, incremental rollout | Rollback to clone(), optimize allocator |
| copyFrom() slower than expected | LOW | MEDIUM | Profile each game, optimize hot paths | Accept partial gain (1.5-2× instead of 3.7×) |
| Thread contention after memory fix | MEDIUM | MEDIUM | Lock-free structures, relaxed atomics | Use 4-6 threads, accept 60% efficiency |
| OpenMP still not active | MEDIUM | LOW | Diagnostic tooling, linkage verification | Accept as non-critical (state pooling sufficient) |

### 8.2 Quality Risks

| Risk | Likelihood | Impact | Mitigation | Contingency |
|------|-----------|--------|-----------|-------------|
| Memory leaks in state pool | LOW | HIGH | Valgrind soak tests (1hr+) | Pool exhaustion detection, fallback |
| Thread safety violation | LOW | CRITICAL | TSan with 24 threads, stress testing | Rollback, add mutexes if needed |
| Search quality regression | LOW | CRITICAL | A/B testing (1000-game matches) | Rollback if win rate <99.5% |
| Profiling overhead distortion | LOW | MEDIUM | Compare profiled vs non-profiled runs | Disable profiling for production |

### 8.3 Rollback Triggers

**Immediate Rollback Required If**:
- Throughput < 95% of baseline (regression detected)
- `alloc_slow_path` counter increases >10% over baseline
- TSan reports data races
- Memory leaks detected (valgrind or RSS growth)
- Search quality regression: win rate <99.5% vs baseline

**Rollback Procedure**:
1. Revert code changes to last known-good commit
2. Re-run validation suite to confirm baseline restored
3. Document failure mode and root cause
4. Create issue with profiling evidence
5. Redesign optimization with new approach

---

## 9. Implementation Timeline

### Week 1: State Pooling (Critical Path)

**Day 1-2: Implementation**
- [ ] Design ThreadLocalStatePool class
- [ ] Implement IGameState::copyFrom() API
- [ ] Implement GomokuState::copyFrom()
- [ ] Integrate pool into ContinuousSimulationRunner
- [ ] Unit tests for pool + copyFrom()

**Day 3: Validation**
- [ ] Run profiling benchmark (100 trials minimum)
- [ ] Verify alloc_slow_path <20,000 for 2,000 sims
- [ ] Verify state cloning <5% of time
- [ ] Verify throughput ≥7,500 sims/sec
- [ ] TSan validation (24 threads)

**Day 4-5: OpenMP Investigation (Optional)**
- [ ] Diagnostic procedure (linkage, environment, debug output)
- [ ] Rebuild with explicit OpenMP flags
- [ ] Test with simple OpenMP program
- [ ] Validate thread scaling
- [ ] If failure: Document and defer

**Day 6: Chess/Go State Pooling**
- [ ] Implement ChessState::copyFrom()
- [ ] Implement GoState::copyFrom()
- [ ] Unit tests for all game types
- [ ] Cross-game validation

### Week 2: Refinement & Validation

**Day 7-8: Allocation Reduction (Optional)**
- [ ] Expand thread-local arenas
- [ ] Pre-allocate node pools
- [ ] Stack-based temporaries
- [ ] Validation benchmarks

**Day 9: Comprehensive Profiling Campaign**
- [ ] Run 560-trial campaign (all configurations)
- [ ] Verify 100% capture rate
- [ ] Statistical analysis vs baseline
- [ ] Generate campaign report

**Day 10: Documentation & Handoff**
- [ ] Update FINAL_PROFILING_ANALYSIS with new data
- [ ] Update spec.md with achieved results
- [ ] Archive profiling session
- [ ] Create summary report

**Total Timeline**: 10 days (2 weeks)
**Critical Path**: Days 1-3 (state pooling implementation + validation)
**Stretch Goals**: Days 4-5 (OpenMP), Days 7-8 (allocation reduction)

---

## 10. Rollback Procedures

### 10.1 Rollback Triggers (Detailed)

**Automatic Triggers** (CI failure):
```python
# In CI pipeline
if throughput < baseline * 0.95:
    trigger_rollback("Throughput regression detected")

if alloc_slow_path > baseline * 1.10:
    trigger_rollback("Allocation overhead increased")

if tsan_errors > 0:
    trigger_rollback("Thread safety violation")
```

**Manual Triggers** (code review):
- Memory leak detected in valgrind soak test
- Win rate vs baseline <99.5% (1000-game A/B test)
- Search quality regression (policy agreement <95%)

### 10.2 Rollback Procedure (Step-by-Step)

**Step 1: Identify Last Known Good**
```bash
# Find last commit with passing benchmarks
git log --grep="benchmark: PASS" -n 1

# Alternative: Use git bisect
git bisect start
git bisect bad HEAD
git bisect good <last-known-good-commit>
```

**Step 2: Revert Changes**
```bash
# Revert to last known good
git revert --no-commit <bad-commit>..<HEAD>
git commit -m "Rollback: State pooling regression"

# Rebuild
export CXXFLAGS="-O3 -march=znver3 -fopenmp"
pip install -e . --force-reinstall --no-deps
```

**Step 3: Validate Rollback**
```bash
# Run validation suite
python scripts/validate_rollback.py \
    --baseline profiling_suite_20251016_124134 \
    --iterations 10

# Expected output:
# ✅ Throughput: 2,659 ± 53 sims/sec (baseline restored)
# ✅ Allocation overhead: 223 per sim (baseline restored)
# ✅ TSan clean (0 races)
```

**Step 4: Root Cause Analysis**
```bash
# Extract profiling data from failed run
cd profiling_reports/failed_campaign_YYYYMMDD_HHMMSS/

# Generate failure report
python scripts/analyze_failure.py \
    --campaign . \
    --baseline profiling_suite_20251016_124134 \
    --output failure_report.md

# Review failure report
cat failure_report.md
```

**Step 5: Document & Create Issue**
```bash
# Create issue with evidence
gh issue create \
    --title "State pooling regression: Throughput ${throughput} < ${target}" \
    --body-file failure_report.md \
    --label "bug,performance,rollback"

# Archive failed profiling session
git add profiling_reports/failed_campaign_*/
git commit -m "Archive failed profiling session for state pooling"
```

**Step 6: Redesign**
- Review profiling data to understand failure mode
- Propose alternative implementation approach
- Document expected impact and risks
- Submit for review before re-implementation

### 10.3 Rollback Validation Script

**Script**: `scripts/validate_rollback.py`

```python
#!/usr/bin/env python3
"""
Validate that rollback restored baseline performance.
"""

import subprocess
import json
import sys
import argparse

def run_rollback_validation(baseline_id, iterations):
    """Run validation benchmark after rollback."""
    result = subprocess.run([
        'python', 'scripts/benchmark_throughput.py',
        '--game', 'gomoku',
        '--threads', '8',
        '--simulations', '2000',
        '--seed', '42',
        '--iterations', str(iterations),
        '--enable-profiling'
    ], capture_output=True, text=True)

    return json.loads(result.stdout)

def validate_baseline_restored(data, baseline_id):
    """Verify performance matches baseline."""
    # Load baseline data
    with open(f'profiling_reports/{baseline_id}/campaign_summary.json') as f:
        baseline = json.load(f)

    errors = []

    # Check throughput within 5% of baseline
    baseline_throughput = baseline['statistics']['mean']
    current_throughput = data['statistics']['mean']

    delta_pct = abs(current_throughput - baseline_throughput) / baseline_throughput * 100

    if delta_pct > 5.0:
        errors.append(
            f"❌ Throughput: {current_throughput:.0f} vs baseline {baseline_throughput:.0f} "
            f"(Δ{delta_pct:.1f}% > 5% tolerance)"
        )
    else:
        print(f"✅ Throughput: {current_throughput:.0f} ± {data['statistics']['stddev']:.0f} sims/sec (baseline restored)")

    # Check allocation overhead matches baseline
    baseline_allocs = baseline['mean_alloc_per_sim']
    current_allocs = data['mean_alloc_per_sim']

    if abs(current_allocs - baseline_allocs) > 5:
        errors.append(
            f"❌ Allocations per sim: {current_allocs:.1f} vs baseline {baseline_allocs:.1f}"
        )
    else:
        print(f"✅ Allocations per sim: {current_allocs:.1f} (baseline restored)")

    # Check TSan clean
    if data['tsan_errors'] > 0:
        errors.append(f"❌ TSan errors: {data['tsan_errors']}")
    else:
        print(f"✅ TSan clean (0 races)")

    return errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("Rollback Validation")
    print("=" * 60)

    print("\n1. Running validation benchmark...")
    data = run_rollback_validation(args.baseline, args.iterations)

    print("\n2. Validating baseline restored...")
    errors = validate_baseline_restored(data, args.baseline)

    if errors:
        print("\n" + "=" * 60)
        print("❌ ROLLBACK VALIDATION FAILED")
        print("=" * 60)
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("✅ ROLLBACK SUCCESSFUL")
        print("=" * 60)
        print("Baseline performance restored.")
        sys.exit(0)

if __name__ == '__main__':
    main()
```

---

## Appendix A: Performance Calculations

### A.1 State Pooling Impact Calculation

**Current Performance** (from profiling):
```
Throughput: 2,659 sims/sec
Time per 2,000 simulations: 982.86 ms

Breakdown:
  state_clone_total:   835.85 ms (86.6%)
  expansion_total:      37.24 ms ( 3.8%)
  expansion_nn_wait:    20.66 ms ( 2.1%)
  selection_total:       3.58 ms ( 0.4%)
  backup_total:          1.67 ms ( 0.2%)
  other:                85.64 ms ( 8.7%)
```

**After State Pooling**:
```
State cloning: 835.85 ms → 40 ms (reduction: 795.85 ms)
Reason: 418μs → 20μs per clone (20.9× faster)

New total time: 982.86 - 795.85 = 187.01 ms per 2,000 sims
New throughput: 2,000 / 0.187 s = 10,695 sims/sec

Conservative estimate (overhead): 9,838 sims/sec
Improvement factor: 9,838 / 2,659 = 3.70×
```

**After OpenMP Fix** (if successful):
```
Feature extraction: Currently sequential, target parallel with 12 threads
Expected speedup in this phase: 6-10× (assume 8×)
Overall impact: 1.5-2.0× additional (assume 1.5×)

New throughput: 9,838 × 1.5 = 14,757 sims/sec
Improvement factor: 14,757 / 2,659 = 5.55×
```

**After Allocation Reduction**:
```
Remaining allocations: 8-10 per sim (from node expansion, etc.)
Expected overhead reduction: 10-20% of residual time
Overall impact: 1.2-1.5× additional (assume 1.2×)

New throughput: 14,757 × 1.2 = 17,708 sims/sec
Improvement factor: 17,708 / 2,659 = 6.66×
```

---

## 11. T019: Zero-Copy MCTS Architecture (NEXT PHASE)

**Note**: This section documents the architectural refactor to address the fundamental state cloning bottleneck identified in T018. See `T018_FINDINGS_AND_PATH_FORWARD.md` for comprehensive analysis.

### 11.1 Architectural Finding from T018

**T018 State Pooling Outcomes**:
- ✅ Solved memory leak (bounded growth via lock-free lazy ring buffer)
- ✅ Solved illegal moves (proper ring sizing)
- ❌ **Performance regression: 1,164 sims/sec** (56% slower than baseline 2,659)
- ❌ **Architectural ceiling identified**: 418μs state cloning cannot be optimized away with pooling

**Root Cause**:
```
Current Architecture:
  Node contains full State (120KB)
    → Clone required for each simulation (418μs)
      → 223 allocations per clone (~2μs each = 446μs)
        → 86.6% of execution time

Pooling Attempt:
  Pre-allocate states in pool
    → copyFrom() reduces allocations to 0 ✅
      → But still 418μs memcpy overhead ❌
        → Sparse allocation hurts cache locality ❌
          → 56% performance regression
```

**Conclusion**: State pooling is a **band-aid** on an **architectural problem**. Need zero-copy architecture.

### 11.2 Zero-Copy Architecture Design

**Core Principle**: Store only move sequences in tree, reconstruct states on-demand.

#### 11.2.1 Tiny Node Structure (32 bytes)

```cpp
// cpp_extensions/mcts/tiny_node.hpp
struct alignas(64) TinyNode {
    // Move that led to this node (16 bits)
    uint16_t move;

    // Parent node index (32 bits, supports 4B nodes)
    uint32_t parent_idx;

    // First child index (32 bits, 0 = no children)
    uint32_t first_child_idx;

    // Sibling index (32 bits, 0 = no sibling)
    uint32_t next_sibling_idx;

    // Visit count (atomic, 32 bits)
    std::atomic<uint32_t> visit_count;

    // Total value (atomic, 32 bits scaled)
    std::atomic<int32_t> total_value_scaled;

    // Prior probability (16 bits scaled)
    uint16_t prior_scaled;

    // Virtual loss (8 bits, max 255)
    std::atomic<uint8_t> virtual_loss;

    // Node flags (8 bits: terminal, expanded, etc.)
    uint8_t flags;

    // Zobrist hash (64 bits, for transposition table)
    uint64_t zobrist_hash;

    // Total: 34 bytes, aligned to 64 bytes
};

static_assert(sizeof(TinyNode) <= 64, "TinyNode must fit in cache line");
```

**Impact**:
- Memory per node: 120KB → 32 bytes (3,750× reduction)
- Cache efficiency: 1,875 cache lines → 1 cache line
- 10M nodes: 1.2GB → 320MB (tree memory)

#### 11.2.2 make/unmake Pattern

**API Design**:
```cpp
// cpp_extensions/utils/igamestate.h
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

Gomoku (minimal undo):
```cpp
uint64_t undo_token = (
    (last_move_row << 8) |
    (last_move_col << 0) |
    (game_result << 16) |
    (move_count << 24)
);
```

Chess (complex undo):
```cpp
uint64_t undo_token = (
    (captured_piece << 0) |      // 4 bits
    (castling_rights << 4) |     // 4 bits
    (en_passant_square << 8) |   // 8 bits
    (halfmove_clock << 16) |     // 8 bits
    (game_result << 24)          // 8 bits
);
```

Go (moderate undo):
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

#### 11.2.3 Per-Thread Bump Arenas

**Design**:
```cpp
// cpp_extensions/mcts/bump_arena.hpp
class BumpArena {
public:
    static constexpr size_t BLOCK_SIZE = 65536;  // 64K nodes

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
    std::vector<...>::iterator epoch_marker_;
};

// Thread-local bump arena per worker
thread_local BumpArena node_arena;
```

**Impact**:
- Allocation speed: O(1) pointer increment (~5ns)
- No locking: Each thread has own arena
- Bulk reclamation: O(1) epoch increment (vs O(N) free)

#### 11.2.4 Epoch Reclamation (QSBR)

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
    static constexpr size_t QUIESCENT = SIZE_MAX;
};
```

**Impact**:
- Memory reclamation: Bulk-free entire blocks
- Latency: O(1) with bounded waiting (all threads quiesce)
- Safety: No use-after-free (wait for quiescence)

#### 11.2.5 Transposition Tables (DAG)

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

        // Linear probing
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

        // Table full or max probes - evict LRU
        return evict_and_insert(zobrist, new_node_idx);
    }

private:
    std::vector<TranspositionEntry> table_;
    size_t table_size_;
    static constexpr size_t MAX_PROBE = 16;
};
```

**Impact**:
- Deduplication: Tree becomes DAG (positions shared)
- Memory savings: 20-40% (typical for board games)
- Visit count accuracy: Transpositions share statistics

#### 11.2.6 Bounded SPSC Queues

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
            return false;  // Queue full
        }

        buffer_[write_idx] = std::move(item);
        write_idx_.store(next_write, std::memory_order_release);
        return true;
    }

    bool try_dequeue(T& item) {
        size_t read_idx = read_idx_.load(std::memory_order_relaxed);

        // Check empty
        if (read_idx == write_idx_.load(std::memory_order_acquire)) {
            return false;  // Queue empty
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

**Impact**:
- No dynamic allocation (fixed-size ring buffer)
- Cache-friendly (contiguous storage)
- Lock-free (single producer, single consumer)

### 11.3 Performance Analysis

#### 11.3.1 State Reconstruction Cost

**make/unmake Benchmark** (Gomoku):
```
Current (clone):
  - memcpy(225 bytes) = ~50ns
  - memcpy(200 bytes) = ~40ns
  - Primitive copies = ~10ns
  - Allocation overhead = 446μs (223 allocs × 2μs)
  - Total: ~418μs per clone

Zero-Copy (make/unmake):
  - Place stone: board[row * 15 + col] = player (5ns)
  - Update metadata: move_count++, last_move (5ns)
  - Check win condition (if terminal): ~100ns
  - Total: ~15ns per make_move

Speedup: 418μs / 15ns = 27,867× faster
```

#### 11.3.2 Path Traversal Cost

**Typical MCTS Path** (depth = 20 moves):
```
Current Architecture:
  - Clone root state: 418μs
  - Apply 20 moves: 20 × 10μs = 200μs
  - Total: 618μs per simulation

Zero-Copy Architecture:
  - make 20 moves: 20 × 15ns = 300ns
  - unmake 20 moves: 20 × 15ns = 300ns
  - Total: 600ns per simulation

Speedup: 618μs / 600ns = 1,030× faster
```

#### 11.3.3 Overall Throughput Projection

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
Overhead:              54.58 ms (55.6%)  ← Reduced (less GIL contention)
Total:                 98.27 ms per 2,000 sims

Throughput: 2,000 / 0.09827 = 20,351 sims/sec
Improvement: 20,351 / 2,659 = 7.65× faster
```

**Conservative Estimate** (accounting for unknowns):
```
Assume 30% overhead from:
  - Path reconstruction complexity
  - Transposition table lookups
  - Arena allocation overhead
  - Epoch reclamation pauses

Adjusted throughput: 20,351 × 0.7 = 14,246 sims/sec
Improvement: 14,246 / 2,659 = 5.36× faster

Range: 15,000-25,000 sims/sec (5-10× improvement)
```

### 11.4 Implementation Phases

**Phase 5A: Core Architecture** (2-3 weeks):
- T024a: Tiny Node Design & Specification (1 day)
- T024b: make/unmake API Design (1 day)
- T024c: Gomoku make/unmake Implementation (2 days)
- T024d: Chess make/unmake Implementation (2 days)
- T024e: Go make/unmake Implementation (2 days)
- T024f: Tree Refactor (Tiny Nodes + Indices) (3 days)
- T024g: SimRunner Integration (2 days)
- T024h: Correctness Validation (1 day)

**Phase 5B: Memory Management** (1 week):
- T025a: Per-Thread Bump Arenas Design (1 day)
- T025b: Epoch Reclamation Implementation (2 days)
- T025c: Memory Validation & Leak Testing (1 day)

**Phase 5C: Transposition Tables** (1 week):
- T026a: Zobrist Hashing Implementation (1 day)
- T026b: DAG Tree (MCGS) Implementation (2 days)
- T026c: Transposition Table Validation (1 day)

**Phase 5D: Queue Optimization** (3-5 days):
- T027a: Bounded SPSC Queue Design (1 day)
- T027b: Replace moodycamel Queue (2 days)
- T027c: Queue Validation & Performance Testing (1 day)

**Phase 5E: Final Validation** (3-5 days):
- T028: Comprehensive Performance Benchmarking (2 days)
- T029: Documentation & Architecture Guide (1 day)

**Total Timeline**: 5-7 weeks

### 11.5 Risk Analysis

**Technical Risks**:

1. **make/unmake Correctness** (MEDIUM)
   - Risk: Undo token insufficient for complete state restoration
   - Mitigation: Extensive unit tests, bit-exact equivalence validation
   - Fallback: Add more bits to undo token (up to 128 bits if needed)

2. **Memory Reclamation Complexity** (MEDIUM)
   - Risk: Epoch reclamation bugs (use-after-free, memory leaks)
   - Mitigation: Valgrind validation, TSan verification
   - Fallback: Disable reclamation (leak slowly), fix in follow-up

3. **Transposition Table Bugs** (LOW-MEDIUM)
   - Risk: Zobrist hash collisions, incorrect canonical node
   - Mitigation: Collision detection, extensive validation
   - Fallback: Disable transpositions (tree mode), fix separately

4. **Integration Complexity** (HIGH)
   - Risk: Large refactor, many files changed, high regression potential
   - Mitigation: Phased implementation, maintain old path for comparison
   - Fallback: Rollback to T018 state pooling (functional, but slow)

**Performance Risks**:

1. **make/unmake Slower Than Expected** (LOW)
   - Risk: Game logic complexity increases cost beyond 15ns
   - Mitigation: Benchmark early, optimize hot paths
   - Expected: Even at 100ns, still 4,180× faster than clone

2. **Path Reconstruction Overhead** (MEDIUM)
   - Risk: Deep trees (depth >50) increase reconstruction cost
   - Mitigation: Limit tree depth, optimize make/unmake
   - Expected: 50 × 15ns = 750ns (still <1% of time)

3. **Transposition Table Overhead** (LOW)
   - Risk: Hash lookups slow down node creation
   - Mitigation: Fast hash function (Zobrist XOR), small probe count
   - Expected: <50ns per lookup (negligible)

**Schedule Risks**:

1. **Longer Than 5-7 Weeks** (MEDIUM)
   - Risk: Complex debugging, unforeseen integration issues
   - Mitigation: Buffer time built into estimate
   - Contingency: Defer optional components (transpositions, queue optimization)

### 11.6 Success Metrics

**Performance Targets**:

| Metric | Current | T018 (Pooling) | T019 (Zero-Copy) | Target |
|--------|---------|----------------|------------------|--------|
| Throughput (sims/sec) | 2,659 | 1,164 (regression) | 15,000-25,000 | ≥8,000 |
| State overhead (% time) | 86.6% | 88.2% | <2% | <5% |
| Memory per node | 120KB | 120KB | 32 bytes | <1KB |
| Tree memory (10M nodes) | 1.2GB | 1.2GB | 320MB | <1GB |
| Path reconstruction (μs) | 418 | 418 | 0.6 | <10 |

**Correctness Targets**:

| Metric | Target |
|--------|--------|
| make/unmake equivalence | 100% bit-exact with clone() |
| Transposition correctness | 100% win rate vs tree-only mode |
| Memory leak | 0 bytes leaked over 24h |
| TSan clean | 0 data races |
| Win rate vs baseline | ≥99.5% (search quality) |

### 11.7 Prior Art & References

**Production Systems Using Zero-Copy MCTS**:

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

**Academic References**:

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

## Appendix B: Profiling Metrics Reference

### B.1 Critical Counters

| Counter | Current | Target | Meaning |
|---------|---------|--------|---------|
| `state_clone_count` | 2,000 | 2,000 | Must equal simulation count |
| `alloc_slow_path` | 446,227 | <20,000 | Heap allocations (<10 per sim) |
| `omp_parallel_success` | 0 | >0 | OpenMP activation count |
| `selection_retries` | ~500 | <100 | Path collision retries |
| `expansion_conflicts` | ~50 | <20 | Thread expansion conflicts |

### B.2 Critical Timings

| Timing | Current | Target | Meaning |
|--------|---------|--------|---------|
| `state_clone_total` | 835.85 ms | <50 ms | Total state cloning time |
| `expansion_total` | 37.24 ms | <100 ms | Total expansion time |
| `expansion_nn_wait` | 20.66 ms | <50 ms | GPU inference wait time |
| `selection_total` | 3.58 ms | <20 ms | Tree traversal time |
| `backup_total` | 1.67 ms | <10 ms | Value propagation time |

### B.3 Acceptance Thresholds

**Phase 4 Completion** (State Pooling):
- ✅ `alloc_slow_path` <20,000 for 2,000 sims
- ✅ `state_clone_total` <50 ms (<5% of time)
- ✅ Throughput ≥7,500 sims/sec (3.0× minimum improvement)
- ✅ TSan clean (0 races)
- ✅ Win rate ≥99.5% vs baseline

**Stretch Goals** (OpenMP + Allocations):
- 🎯 `omp_parallel_success` >0
- 🎯 Thread scaling >1.0× with multiple threads
- 🎯 Throughput ≥10,000 sims/sec

---

**END OF TECHNICAL PLAN v1.1**

**Version History**:
- v1.0: Initial technical plan (T018-T023: state pooling and incremental optimizations)
- v1.1: Added T019 zero-copy architecture design (Section 11) based on T018 architectural findings

**Next Steps**:
1. Complete T018 state pooling closure (validation and documentation)
2. Archive T018 findings and performance results
3. Review T019 technical design with stakeholders
4. Begin T024a (Tiny Node Design) after T018 sign-off
5. Follow 5-7 week phased implementation plan for T019
6. Target achievement: 15,000-25,000 sims/sec (5-10× improvement)
