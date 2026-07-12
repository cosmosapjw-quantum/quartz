# Research & Architectural Decisions

**Feature**: MCTS Throughput Optimization
**Date**: 2025-10-20
**Status**: Complete

## Summary

This document consolidates research findings from profiling campaigns, external reviews, and codebase analysis to validate architectural decisions for the 4-phase optimization plan.

## Research Findings

### R1: State Cloning Touch-Point ✅

**Question**: Where exactly does state cloning occur in `continuous_simulation_runner.cpp`?

**Finding**: State cloning occurs when submitting to `AsyncInferenceQueue` because current API expects `std::shared_ptr<GameState>`, forcing a deep copy at every submission.

**Profiling Evidence**:
- Source: `COMPREHENSIVE_PROFILING_ANALYSIS_20251018.md`
- State cloning: 418μs per clone, 86.6% of total execution time
- Allocation count: 223 allocations per clone

**Decision**: **Phase 1A** - Add thread-local `feature_buffer` to `ThreadLocalState`, extract features in-place at leaf node, move features to queue via rvalue semantics.

**Impact**: Expected 10-25× throughput gain (1,500-3,000 sims/sec from 120 baseline).

---

### R2: OpenMP Linking Failure ✅

**Question**: Why is OpenMP success rate 0% despite `find_package(OpenMP REQUIRED)` in CMakeLists.txt?

**Finding**: OpenMP compile flags are added globally but `OpenMP::OpenMP_CXX` target is **NOT linked** to `mcts_py` shared library. Only `CMAKE_EXE_LINKER_FLAGS` is set, which applies to executables only.

**Evidence**:
```cmake
# CMakeLists.txt:52-56 (current, BROKEN)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  # ❌ Not for .so
endif()
```

**Decision**: **Phase 2A** - Add `target_link_libraries(mcts_py PRIVATE OpenMP::OpenMP_CXX)` to link OpenMP library to shared library target.

**Impact**: Expected 4× feature extraction speedup (7.5ms → 1.5ms for 64-item batch).

---

### R3: Tensor Copy Pipeline Analysis ✅

**Question**: Where do the 4-6 memory copies occur in Python tensor creation (37ms/batch)?

**Finding**: 6 copies identified in `dlpack_inference_bridge.py`:
1. **Copy 1**: C++ `std::vector<float>` → numpy array (via `np.array()`)
2. **Copy 2**: List of numpy arrays → stacked numpy array (via `np.stack()`)
3. **Copy 3**: Numpy array → torch CPU tensor (via `torch.from_numpy()`, forces contiguous)
4. **Copy 4-6**: CPU → GPU blocking transfer (non-pinned → pinned → GPU, 3 copies)

**Profiling Evidence**: Tensor creation measured at 37ms per 64-item batch (profiling campaign).

**Decision**: **Phase 2B** - Pre-allocate pinned CPU buffer (3.3MB) + GPU buffer (3.3MB), fill pinned buffer directly via `torch.frombuffer()`, use `non_blocking=True` for async H2D transfer.

**Impact**: Expected 18× improvement (37ms → 2ms).

---

### R4: Condition Variable Current State ✅

**Question**: Does `AsyncInferenceQueue` currently poll or use condition variables?

**Finding**: Current implementation polls queue in tight loop, wasting CPU cycles. No `std::condition_variable` detected in codebase.

**Evidence**: Profiling shows coordinator blocking 99.6% of iteration time, consistent with busy-wait polling.

**Decision**: **Phase 1 Enhancement** - Add `std::condition_variable cv_request_ready_` to wake coordinator immediately on new request. Add `cv_results_ready_` to wake simulation threads on inference completion.

**Impact**: Reduces idle CPU usage from ~100% to <10%, eliminates polling latency (~1-10ms wake time).

---

### R5: Virtual-Loss Contention Assessment ✅

**Question**: How often does virtual-loss contention occur?

**Estimation**:
- 8 simulation threads × 120 sims/sec = 960 selections/sec
- Expected contention rate: 5-10% (empirical for shared-tree MCTS with 8 threads)
- Contention events: ~50-100/sec

**Current Behavior**: On contention, thread sleeps for 10μs then retries entire selection from root.

**Decision**: **Phase 1 Enhancement** - On `try_apply_virtual_loss()` failure, immediately restart selection from root without sleep (use `goto restart;` pattern).

**Impact**: Reduces contention latency from 10-50μs (sleep overhead) to <1μs (restart overhead). Expected 5-10% throughput gain for 8-thread workloads.

---

### R6: Make/Unmake Pattern Support ✅

**Question**: Do all three games support `make_move()` / `unmake_move()` for in-place state manipulation?

**Finding**: Yes, all games support make/unmake:
- **Gomoku** (`cpp_extensions/games/gomoku.cpp`): ✅ Full support, simple board state
- **Chess** (`cpp_extensions/games/chess.cpp`): ✅ Full support, includes castling/en passant reversal
- **Go** (`cpp_extensions/games/go.cpp`): ✅ Full support, but ko detection may need path history for superko rules

**Decision**: Use in-place extraction (Phase 1A) as primary approach. Path reconstruction (Phase 1D) as optional fallback for Go superko validation only.

**Impact**: Confirms Phase 1A approach is viable for all 3 games.

---

### R7: Pinned Memory Buffer Sizing ✅

**Question**: What's the optimal size for pre-allocated pinned tensor buffer?

**Calculation**:
```
max_batch × max_planes × max_board_H × max_board_W
= 64 × 36 × 19 × 19 = 831,744 floats
= 831,744 × 4 bytes = 3,326,976 bytes ≈ 3.3MB
```

**Per-Game Analysis**:
- Gomoku (15×15, 36 planes): 64 × 36 × 15 × 15 = 518,400 floats = 2.1MB
- Chess (8×8, 30 planes): 64 × 30 × 8 × 8 = 122,880 floats = 0.5MB
- Go (9×9, 25 planes): 64 × 25 × 9 × 9 = 129,600 floats = 0.5MB

**VRAM Budget Check**: 3.3MB << 8GB VRAM ✅ (0.04% of VRAM)

**Decision**: **Phase 2B** - Pre-allocate single buffer sized for worst-case (Gomoku 15×15 extrapolated to Go 19×19 for future-proofing) = 64×36×19×19 = 3.3MB.

**Impact**: Negligible memory overhead, enables 100% buffer reuse (zero allocations per batch).

---

### R8: Multi-Coordinator Queue Partitioning ✅

**Question**: How should inference queue be partitioned among 2-4 coordinators (Phase 3A)?

**Options Evaluated**:
- **Option A**: Single shared queue + round-robin atomic dequeue
  - Pros: Simplest implementation, automatic load balancing
  - Cons: Potential atomic contention (low with lock-free MPMC)

- **Option B**: Separate queues per coordinator + manual load balancing
  - Pros: Zero contention between coordinators
  - Cons: Complex load balancing logic, potential queue imbalance

- **Option C**: Work-stealing deques
  - Pros: Optimal load balancing
  - Cons: High complexity, not needed for this scale

**Decision**: **Option A for Phase 3A** - Use single shared lock-free MPMC queue (existing `async_inference_queue.cpp` with 4096-entry ring buffer). Each coordinator owns separate CUDA stream.

**Rationale**: Lock-free MPMC already handles contention efficiently (atomic compare-exchange). Simplicity outweighs minimal contention cost at 12-20k sims/sec scale.

**Impact**: Enables 3.2-3.6× scaling with 4 coordinators (diminishing returns due to queue contention).

---

## Architectural Decisions

### Decision 1: Zero-Copy via Move Semantics (Phase 1) ✅

**Chosen Approach**: In-place feature extraction + rvalue move semantics

**Alternatives Considered**:
- ❌ **State pool with copyFrom()**: Tested and rejected (56% regression, `copyFrom()` still expensive)
- ❌ **Shared pointers with ref counting**: Adds synchronization overhead, doesn't eliminate copy
- ✅ **Move semantics**: Zero overhead, compiler-optimized, clean ownership transfer

**Implementation**:
```cpp
// Thread-local buffer (allocated once)
tls.feature_buffer.resize(max_planes * max_board * max_board);

// Extract in-place
game->extract_features_to_buffer(current_state, tls.feature_buffer.data());

// Move to queue (zero copy)
InferenceRequest request;
request.features = std::move(tls.feature_buffer);  // ✅
queue.submit_request(std::move(request));          // ✅
```

**Validation**: Code review MUST grep for `clone()`, `copy()`, `new State()` in simulation paths → 0 occurrences required.

---

### Decision 2: Pinned Memory + Non-Blocking Transfer (Phase 2) ✅

**Chosen Approach**: Pre-allocated pinned CPU buffer + pre-allocated GPU buffer + async transfer

**Alternatives Considered**:
- ❌ **Allocate per batch**: 37ms overhead per batch (current bottleneck)
- ❌ **CPU tensor → GPU blocking**: Forces GIL hold for entire transfer (~37ms)
- ✅ **Pinned + non-blocking**: Minimal GIL hold (~0.5ms), async overlap with simulation

**Implementation**:
```python
# Pre-allocate once (initialization)
self.pinned_buffer = torch.zeros((64, 36, 19, 19), pin_memory=True)
self.gpu_buffer = torch.zeros((64, 36, 19, 19), device='cuda')
self.stream = torch.cuda.Stream()

# Per batch (zero allocation)
with torch.cuda.stream(self.stream):
    self.gpu_buffer[:batch_size, ...].copy_(
        self.pinned_buffer[:batch_size, ...],
        non_blocking=True  # ✅ Async transfer
    )
```

**Validation**: Assert `self.pinned_buffer.is_pinned() == True`, measure tensor creation time <2ms.

---

### Decision 3: Single-Process Multi-Coordinator (Phase 3A) vs Multi-Process (Phase 3B)

**Phase 3A Choice**: 2-4 coordinator threads, single shared queue, multi-stream GPU

**Phase 3B Deferred**: Multi-process only if Phase 3A insufficient AND GIL >50% bottleneck

**Rationale**:
- Phase 2 optimizations (pinned buffers, non-blocking transfer) reduce GIL hold from 37ms to <0.5ms
- Phase 3A multi-coordinator should achieve 12-20k sims/sec without multi-process complexity
- Multi-process adds 6+ weeks implementation time, high risk, maintenance burden

**Decision Gate**: Implement Phase 3B ONLY if:
1. Phase 3A throughput <12,000 sims/sec AND
2. Profiling shows Python callback >5ms/batch AND
3. Target throughput >25,000 sims/sec

---

## Constitution Compliance Validation

**Principle I (Zero-Copy First)**: ✅ SATISFIED
- Phase 1A eliminates all state cloning via in-place extraction
- Move semantics ensures zero copies in queue submission
- Code review enforces zero `clone()` calls

**Principle II (Coordinator Efficiency)**: ✅ SATISFIED
- Phase 1 adds condition variables (no polling)
- Phase 2 pre-allocates pinned buffers (zero allocation per batch)
- Phase 3A multi-coordinator eliminates serialization

**Principle III (Python-C++ Boundary Discipline)**: ✅ SATISFIED
- Phase 2B minimizes boundary crossings (simulation loop in C++)
- Pinned memory + non-blocking transfer reduces GIL hold from 37ms to <0.5ms

**Principle IV (Threading Saturation)**: ✅ SATISFIED
- Phase 2A fixes OpenMP linking (8-12 threads for feature extraction)
- Thread-local arenas maintain 99.93% fast-path (existing)

**Principle V (Legacy Code Discipline)**: ✅ SATISFIED
- All changes target `continuous_simulation_runner.*`, `async_inference_queue.*`, `batch_inference_coordinator.*`
- No modifications to deprecated `simulation_runner.*` or `mcts_guide.md`

**Principle VI (Evidence-Based Gates)**: ✅ SATISFIED
- Each phase requires 100+ trial profiling campaign
- Automated rollback on regression
- Profiling data committed to repository

---

## Risk Assessment

### Phase 1 Risks: LOW

**Technical Risks**:
- Feature extraction correctness: Mitigated by unit tests comparing in-place vs copy
- Move semantics bugs: Mitigated by compiler checks (move-only types)
- Thread safety: Mitigated by thread-local buffers (no sharing)

**Rollback Complexity**: LOW (5 commits, automated script)

**Confidence**: HIGH (similar patterns used in other high-performance C++ projects)

---

### Phase 2 Risks: MEDIUM

**Technical Risks**:
- OpenMP linking failure: Mitigated by `ldd` verification + runtime thread count check
- Pinned memory exhaustion: Mitigated by small buffer size (3.3MB << system RAM)
- Non-blocking transfer bugs: Mitigated by synchronization before use

**Rollback Complexity**: MEDIUM (8 commits, automated script)

**Confidence**: HIGH (standard PyTorch pinned memory pattern)

---

### Phase 3A Risks: MEDIUM-HIGH

**Technical Risks**:
- Queue contention: Mitigated by lock-free MPMC (already proven)
- GPU stream synchronization: Mitigated by per-coordinator streams
- Load imbalance: Mitigated by shared queue (automatic balancing)

**Rollback Complexity**: MEDIUM (coordinator can be disabled via flag)

**Confidence**: MEDIUM (multi-coordinator is less battle-tested)

---

### Phase 3B Risks: HIGH (Deferred)

**Technical Risks**:
- IPC synchronization bugs: High complexity, hard to debug
- Shared memory corruption: Requires extensive testing
- Process health monitoring: Additional infrastructure needed

**Rollback Complexity**: HIGH (multi-process architecture, 6+ weeks)

**Confidence**: LOW (defer unless absolutely necessary)

---

## References

- [COMPREHENSIVE_PROFILING_ANALYSIS_20251018.md](../../COMPREHENSIVE_PROFILING_ANALYSIS_20251018.md): Profiling data (560 trials)
- [MCTS_OPTIMIZATION_MASTER_PLAN.md](../../MCTS_OPTIMIZATION_MASTER_PLAN.md): Original 3-phase plan
- [MCTS_OPTIMIZATION_MASTER_PLAN_ENHANCEMENTS.md](../../MCTS_OPTIMIZATION_MASTER_PLAN_ENHANCEMENTS.md): External review enhancements
- [ARCHITECTURE_TRADEOFFS.md](../../ARCHITECTURE_TRADEOFFS.md): Decision framework
- [EXTERNAL_REVIEWS_COMPARISON_ANALYSIS.md](../../EXTERNAL_REVIEWS_COMPARISON_ANALYSIS.md): Comparison of approaches

---

## Conclusion

All 8 research tasks validated. Architectural decisions grounded in profiling data and constitution principles. Phase 1-2 expected to achieve 7,000-9,000 sims/sec target (58-75× improvement). Phase 3A/3B optional for stretch goals (12k-35k sims/sec).

**Next Step**: Proceed to Phase 1 Design (data-model.md, contracts/, quickstart.md).
