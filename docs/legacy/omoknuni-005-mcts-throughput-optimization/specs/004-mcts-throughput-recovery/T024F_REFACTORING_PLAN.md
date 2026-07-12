# T024f: Tree Refactor (Tiny Nodes) - Detailed Implementation Plan

**Status**: Planning Phase
**Created**: 2025-10-17
**Dependencies**: T024a-e (complete)

## Executive Summary

Refactor MCTSTree from Structure-of-Arrays (SoA) to TinyNode array-of-structs (AoS) to enable zero-copy MCTS via make/unmake pattern. This is a **critical 3-day task** that must be broken into small, testable subjobs.

## Current Architecture Analysis

### Current MCTSTree (SoA)
```cpp
// Separate arrays for each field (~32-40 bytes/node)
float* visit_counts_;          // 4 bytes
float* total_values_;          // 4 bytes
float* prior_probs_;           // 4 bytes
float* virtual_losses_;        // 4 bytes
NodeIndex* parent_indices_;    // 4 bytes
NodeIndex* first_child_indices_; // 4 bytes
uint16_t* num_children_;       // 2 bytes
NodeFlags* flags_;             // 1 byte
uint16_t* moves_;              // 2 bytes
```

**Strengths**:
- Good cache locality for traversal (related fields together)
- SIMD-friendly (64-byte aligned arrays)
- Already quite efficient (32-40 bytes/node)

**Limitations**:
- No zobrist hash (needed for transpositions)
- Uses array-based children (first_child + num_children)
- Cannot store move sequences efficiently

### Target TinyNode (AoS)
```cpp
// Single struct (34 bytes actual, 64 bytes aligned)
struct alignas(64) TinyNode {
    uint16_t move;                    // 2 bytes
    uint32_t parent_idx;              // 4 bytes
    uint32_t first_child_idx;         // 4 bytes
    uint32_t next_sibling_idx;        // 4 bytes (NEW: sibling linking)
    atomic<uint32_t> visit_count;     // 4 bytes
    atomic<int32_t> total_value_scaled; // 4 bytes
    uint16_t prior_scaled;            // 2 bytes
    atomic<uint8_t> virtual_loss;     // 1 byte
    uint8_t flags;                    // 1 byte
    uint64_t zobrist_hash;            // 8 bytes (NEW: transpositions)
    // 30 bytes padding to 64
};
```

**Key Changes**:
1. **Sibling-linked children**: `next_sibling_idx` instead of `num_children`
2. **Zobrist hash**: Added for transposition table support
3. **Unified structure**: All fields in one cache line
4. **Scaled integers**: prior_scaled, total_value_scaled for precision

### Current Simulation Flow
```cpp
// Current flow (with state pooling - T018)
1. acquire pooled state
2. select_leaf(state, path) - traverse tree, apply moves to state
3. extract features from state
4. inference
5. backup(path, value)
6. release pooled state
```

### Target Simulation Flow
```cpp
// Target flow (with make/unmake - T024f)
1. thread-local state (persistent)
2. traverse path via make_move (apply moves incrementally)
3. extract features from thread-local state
4. inference
5. backup(path, value)
6. unwind path via unmake_move (restore to root)
```

## Refactoring Strategy

### Phased Migration (Safe)

**Phase 1: Parallel Implementation (T024f-1 to T024f-4)**
- Implement new TinyNode tree alongside existing MCTSTree
- Keep both versions running
- Validate equivalence continuously
- Zero production impact

**Phase 2: Integration (T024f-5 to T024f-7)**
- Add adapter layer
- Switch simulation runner to TinyNode
- Performance validation
- Remove old code after validation

**Phase 3: Cleanup (T024f-8)**
- Remove old MCTSTree
- Update documentation
- Final benchmarks

## Subjob Breakdown

### T024f-1: TinyNode Storage Layer (Day 1 Morning)
**Goal**: Implement TinyNode array with allocation

**Scope**:
- Create `TinyNodeTree` class (new file)
- Implement bump arena allocator
- Basic allocation/deallocation
- Unit tests for allocation

**Files**:
- `cpp_extensions/mcts/tiny_node_tree.hpp` (NEW)
- `cpp_extensions/mcts/tiny_node_tree.cpp` (NEW)
- `tests/unit/test_tiny_node_tree.cpp` (NEW)

**Acceptance Criteria**:
- ✓ Allocate/deallocate nodes
- ✓ O(1) bump allocation
- ✓ Tree capacity management
- ✓ Memory leak tests
- ✓ Thread safety (basic)

**Estimated**: 3-4 hours

---

### T024f-2: Sibling-Linked Children (Day 1 Afternoon)
**Goal**: Implement child management with sibling links

**Scope**:
- Add child nodes (sibling linking)
- Iterate children
- Expand node with policy
- Unit tests for child operations

**Files**:
- `cpp_extensions/mcts/tiny_node_tree.cpp` (modify)
- `tests/unit/test_tiny_node_tree.cpp` (extend)

**Acceptance Criteria**:
- ✓ Add single child
- ✓ Add multiple children (expand)
- ✓ Iterate children correctly
- ✓ Child count calculation
- ✓ Validate tree structure

**Estimated**: 3-4 hours

---

### T024f-3: Path Traversal Methods (Day 2 Morning)
**Goal**: Implement tree traversal with move storage

**Scope**:
- Path to node (collect moves)
- Select leaf (with virtual loss)
- Backup value (accumulate statistics)
- Unit tests for traversal

**Files**:
- `cpp_extensions/mcts/tiny_node_tree.cpp` (modify)
- `tests/unit/test_tiny_node_tree.cpp` (extend)

**Acceptance Criteria**:
- ✓ Collect path from root to node
- ✓ Select leaf with PUCT
- ✓ Apply virtual loss correctly
- ✓ Backup value updates
- ✓ Multi-threaded backup test

**Estimated**: 4 hours

---

### T024f-4: Zobrist Hash Integration (Day 2 Afternoon)
**Goal**: Add incremental Zobrist hashing

**Scope**:
- Initialize Zobrist tables
- Update hash on expand
- Validate hash consistency
- Unit tests for hashing

**Files**:
- `cpp_extensions/mcts/tiny_node_tree.cpp` (modify)
- `tests/unit/test_tiny_node_tree.cpp` (extend)

**Acceptance Criteria**:
- ✓ Initialize root hash
- ✓ Update child hashes
- ✓ Verify hash incremental = full recompute
- ✓ Collision detection tests

**Estimated**: 2-3 hours

---

### T024f-5: Adapter Layer (Day 2 Evening)
**Goal**: Create compatibility layer for simulation runner

**Scope**:
- Wrapper for TinyNodeTree
- Match MCTSTree interface
- Enable A/B testing
- Unit tests for adapter

**Files**:
- `cpp_extensions/mcts/tree_adapter.hpp` (NEW)
- `cpp_extensions/mcts/tree_adapter.cpp` (NEW)
- `tests/unit/test_tree_adapter.cpp` (NEW)

**Acceptance Criteria**:
- ✓ Expose same API as MCTSTree
- ✓ Convert between representations
- ✓ Feature flag for switching
- ✓ No regression in existing tests

**Estimated**: 2-3 hours

---

### T024f-6: SimRunner Integration (Day 3 Morning)
**Goal**: Switch simulation runner to use TinyNodeTree + make/unmake

**Scope**:
- Replace state pooling with thread-local state
- Use make_move during select_leaf
- Use unmake_move for unwind
- Integration tests

**Files**:
- `cpp_extensions/mcts/continuous_simulation_runner.cpp` (modify)
- `tests/integration/test_tiny_node_mcts.cpp` (NEW)

**Acceptance Criteria**:
- ✓ Thread-local state per worker
- ✓ Path traversal with make/unmake
- ✓ No state cloning
- ✓ Equivalent results to old path
- ✓ Performance improvement

**Estimated**: 4-5 hours

---

### T024f-7: Correctness Validation (Day 3 Afternoon)
**Goal**: Comprehensive validation against old implementation

**Scope**:
- A/B comparison tests
- Gomoku/Chess/Go all games
- Equivalence validation
- Performance benchmarks

**Files**:
- `tests/validation/test_tiny_node_equivalence.cpp` (NEW)
- `tests/performance/benchmark_tiny_node.cpp` (NEW)

**Acceptance Criteria**:
- ✓ Identical tree structure (1000 sims)
- ✓ Identical policy (within tolerance)
- ✓ Identical value (within tolerance)
- ✓ Performance ≥8,000 sims/sec
- ✓ Memory usage ≤ old implementation

**Estimated**: 3-4 hours

---

### T024f-8: Cleanup & Documentation (Day 3 Evening)
**Goal**: Remove old code, update docs

**Scope**:
- Remove state pooling (T018)
- Remove old MCTSTree (if validated)
- Update CLAUDE.md
- Update quickstart.md

**Files**:
- Multiple files (cleanup)
- Documentation updates

**Acceptance Criteria**:
- ✓ No dead code
- ✓ Documentation current
- ✓ Examples updated
- ✓ All tests passing

**Estimated**: 2 hours

---

## Risk Mitigation

### Technical Risks

**Risk 1: Child Iteration Bugs** (MEDIUM)
- Current: Array-based (first_child + num_children)
- New: Sibling-linked (first_child + next_sibling)
- Mitigation: Extensive unit tests, validation against old

**Risk 2: Thread Safety** (HIGH)
- Atomics must be correct
- Virtual loss must be thread-safe
- Mitigation: TSan validation, stress tests

**Risk 3: Performance Regression** (MEDIUM)
- AoS vs SoA cache behavior
- Mitigation: Benchmarks at each step, rollback plan

**Risk 4: Integration Complexity** (HIGH)
- Many files touched
- Complex state management
- Mitigation: Adapter layer, feature flags, incremental rollout

### Rollback Strategy

If **any subjob** fails critical acceptance criteria:
1. **STOP** immediately
2. Document failure in `NEEDS_DECISION` section of TASKS.md
3. Keep existing code working
4. Analyze root cause
5. Either fix or rollback subjob

If **T024f overall** fails to meet KPIs:
1. Keep adapter layer
2. Disable TinyNode via feature flag
3. Fall back to T018 state pooling (functional, ~3k sims/sec)
4. Document findings
5. Plan remediation

## Success Criteria

### Functional
- ✓ All existing tests pass
- ✓ New unit tests pass (50+ tests)
- ✓ Integration tests pass (3 games)
- ✓ Equivalence validation pass (A/B comparison)
- ✓ No memory leaks (Valgrind clean)
- ✓ No race conditions (TSan clean)

### Performance
- ✓ Throughput ≥ 8,000 sims/sec (target met)
- ✓ Memory ≤ 1GB for 10M nodes
- ✓ Latency ≤ 15ns per make/unmake
- ✓ Path traversal ≤ 600ns (20 moves)

### Code Quality
- ✓ Clean architecture
- ✓ Well-documented
- ✓ Maintainable
- ✓ No dead code

## Timeline

| Day | Subjobs | Duration | Cumulative |
|-----|---------|----------|------------|
| Day 1 AM | T024f-1 (Storage) | 3-4h | 4h |
| Day 1 PM | T024f-2 (Children) | 3-4h | 8h |
| Day 2 AM | T024f-3 (Traversal) | 4h | 12h |
| Day 2 PM | T024f-4 (Zobrist) | 2-3h | 15h |
| Day 2 Eve | T024f-5 (Adapter) | 2-3h | 18h |
| Day 3 AM | T024f-6 (Integration) | 4-5h | 23h |
| Day 3 PM | T024f-7 (Validation) | 3-4h | 27h |
| Day 3 Eve | T024f-8 (Cleanup) | 2h | 29h |

**Total**: ~29 hours (3.6 days with breaks)

## Next Steps

1. **Approve this plan** - Review subjob breakdown
2. **Start T024f-1** - Implement TinyNode storage layer
3. **Track progress** - Use TodoWrite for each subjob
4. **Commit frequently** - After each subjob passes tests
5. **Validate continuously** - Run tests after every change

**Ready to begin**: All prerequisites (T024a-e) complete and validated (25/25 tests passing).
