# Implementation Checklists: MCTS Throughput Optimization

**Feature**: MCTS Throughput Optimization (Zero-Copy + Tensor Pipeline)
**Branch**: `005-mcts-throughput-optimization`
**Purpose**: High-signal validation checklists for rapid phase acceptance

---

## A) "No-Clone Hot Path" Checklist ✅

**Constitution Principle I: Zero-Copy First**

**Code Review** (run `scripts/audit_state_cloning.sh`):
- [ ] No `state->clone()` calls in `cpp_extensions/mcts/continuous_simulation_runner.cpp`
- [ ] No `new GameState()` or `std::make_shared<GameState>()` in simulation hot path
- [ ] No root expansion clone (pre-expansion uses existing node without copy)

**Data Structure Validation**:
- [ ] `AsyncInferenceQueue::InferenceRequest` owns `std::vector<float> features` (not `IGameState*` or shared_ptr)
- [ ] `InferenceRequest` has deleted copy constructor/assignment (move-only semantics enforced)
- [ ] `ThreadLocalState` contains pre-allocated `std::vector<float> feature_buffer` (52KB per thread)

**Coordinator Validation**:
- [ ] `BatchInferenceCoordinator::form_batch()` contains NO feature extraction logic (grep shows zero `extract_features` calls)
- [ ] Coordinator only collects and moves pre-extracted features (batch formed via `std::move(request)`)
- [ ] Zero state cloning in coordinator (grep `batch_inference_coordinator.cpp` for `clone()` returns empty)

**Runtime Validation**:
```bash
# Expected: 0 occurrences
scripts/audit_state_cloning.sh
# Expected output: "✅ Zero state cloning detected in hot paths"

# Expected: feature_buffer exists in ThreadLocalState
grep "feature_buffer" cpp_extensions/mcts/continuous_simulation_runner.hpp
# Expected: std::vector<float> feature_buffer;
```

**Profiling Validation** (Phase 1 campaign):
- [ ] `state_cloning_us / total_time_us < 0.01` (<1% of execution time)
- [ ] `state_clone_count == 0` (zero clones in 100+ trial campaign)
- [ ] `feature_move_count == simulation_count` (all features moved, not copied)

**DoD**: All checks pass + profiling shows <1% state cloning + throughput ≥1,500 sims/sec

---

## B) "Coordinator Throughput" Checklist ⚡

**Constitution Principle II: Coordinator Efficiency**

**Condition Variables (No Polling)**:
- [ ] `AsyncInferenceQueue` has `std::condition_variable cv_request_ready_` declared in header
- [ ] `submit_request()` calls `cv_request_ready_.notify_one()` after enqueue
- [ ] `form_batch()` uses `cv_request_ready_.wait_until()` instead of polling loop
- [ ] No `while(true) { if(queue.size() > 0) break; }` busy-wait patterns in coordinator
- [ ] CPU usage during idle <10% (down from 100% with polling)

**Runtime Validation**:
```bash
# Expected: cv_request_ready_ exists
grep "std::condition_variable" cpp_extensions/mcts/async_inference_queue.hpp
# Expected output: std::condition_variable cv_request_ready_;

# Expected: notify_one() called after enqueue
grep "notify_one()" cpp_extensions/mcts/async_inference_queue.cpp
# Expected: cv_request_ready_.notify_one();
```

**Zero Allocation in Hot Loop**:
- [ ] `BatchInferenceCoordinator::form_batch()` pre-reserves vectors in constructor
- [ ] No `std::vector::push_back()` without prior `reserve()` in hot path
- [ ] Batch assembly uses `std::move()` for all transfers (zero copies)
- [ ] Allocation profiler shows 0 malloc/free calls per iteration

**Runtime Validation**:
```bash
# Run with allocation profiler
python -m pytest tests/contract/test_coordinator_api.py::test_zero_allocation_in_batching
# Expected: 0 allocations detected in hot loop
```

**Python Callback Performance**:
- [ ] Tensor creation via `DLPackInferenceBridge::create_batch_tensor()` completes in ≤2ms per batch
- [ ] Pinned CPU buffer pre-allocated at initialization (3.3MB, `pin_memory=True`)
- [ ] GPU buffer pre-allocated at initialization (3.3MB, `device='cuda'`)
- [ ] Non-blocking transfer used (`copy_(pinned_buffer, non_blocking=True)`)
- [ ] Profiling shows `python_callback_ms p95 ≤ 2.0` (down from 37ms baseline)

**Runtime Validation**:
```python
# Verify pinned buffer exists and is reused
python -c "
from src.core.dlpack_inference_bridge import DLPackInferenceBridge
bridge = DLPackInferenceBridge()
print(f'Pinned: {bridge.pinned_buffer.is_pinned()}')  # Expected: True
print(f'Size: {bridge.pinned_buffer.numel() * 4 / 1e6:.1f}MB')  # Expected: ~3.3MB
"
```

**OpenMP Parallelization**:
- [ ] CMakeLists.txt includes `target_link_libraries(mcts_py PRIVATE OpenMP::OpenMP_CXX)`
- [ ] `ldd build/lib.*/mcts_py*.so | grep -i omp` shows `libomp.so` or `libgomp.so` linked
- [ ] Runtime reports `openmp_thread_count > 1` (measured via profiling)
- [ ] Feature extraction parallelized across cores (8-12 threads observed)

**Runtime Validation**:
```bash
# Expected: libgomp.so.1 or libomp.so linked
scripts/verify_openmp.sh
# Expected output: "✅ OpenMP linked successfully"

# Expected: >1 thread
python -c "import mcts_py; print(f'OpenMP threads: {mcts_py.get_openmp_threads()}')"
# Expected output: OpenMP threads: 8 (or 12, depending on OMP_NUM_THREADS)
```

**DoD**: All checks pass + profiling shows batch assembly <0.3ms, Python callback p95 ≤2ms, OpenMP >1 thread

---

## C) "Phase Gate Benchmarks" Checklist 🎯

**Constitution Principle VI: Evidence-Based Gates**

### Phase 1 Gate: State Cloning Elimination (SC-001 to SC-004)

**Profiling Campaign** (100+ trials):
```bash
python scripts/benchmark_phase1.py --trials 100 --output profiling_phase1_$(date +%Y%m%d)
python scripts/profiling/analyze_campaign.py profiling_phase1_*/ --compare-to-baseline
```

**Acceptance Criteria**:
- [ ] **Throughput**: 1,500-3,000 sims/sec (mean, p50) ✅ **10-25× baseline**
- [ ] **State Cloning Overhead**: <1% of total execution time (down from 86.6%)
- [ ] **State Clone Count**: 0 (zero clones detected across all trials)
- [ ] **Memory Allocations**: 0 in hot path (measured via allocation profiler)
- [ ] **Feature Move Count**: Equals simulation count (all features moved, not copied)

**Rollback If**:
- Throughput <1,500 sims/sec
- State cloning >1% of execution time
- Any clone() calls detected in hot path

**Rollback Procedure**:
```bash
git revert HEAD~10..HEAD  # Revert Phase 1 commits
pip install -e . --force-reinstall --no-deps
scripts/validate_all_phases.sh --verify-baseline  # Should restore 120 sims/sec
```

---

### Phase 2 Gate: Tensor Pipeline + OpenMP (SC-005 to SC-009) ✅ PRIMARY TARGET

**Profiling Campaign** (100+ trials):
```bash
python scripts/benchmark_phase2.py --trials 100 --output profiling_phase2_$(date +%Y%m%d)
python scripts/profiling/analyze_campaign.py profiling_phase2_*/ --compare-to-baseline --compare-to-phase1
```

**Acceptance Criteria**:
- [ ] **Throughput**: 7,000-9,000 sims/sec (mean, p50) ✅ **58-75× baseline, PRIMARY GOAL**
- [ ] **OpenMP Enabled**: `openmp_thread_count > 1` in execution logs (confirms parallelization)
- [ ] **Tensor Creation**: ≤2.0ms per batch (down from 37ms, 18× improvement)
- [ ] **GPU Utilization**: ≥80% during search operations (up from ~68%)
- [ ] **Pinned Buffer Reuse**: 100% (zero allocations per batch, measured via profiling)
- [ ] **H2D Transfer**: ≤1.0ms per batch (non-blocking, async)

**Rollback If**:
- Throughput <7,000 sims/sec
- OpenMP thread count = 1 (not linked)
- Tensor creation >2.0ms per batch
- GPU utilization <70%

**Rollback Procedure**:
```bash
git revert HEAD~15..HEAD  # Revert Phase 2 commits
pip install -e . --force-reinstall --no-deps
scripts/validate_all_phases.sh --verify-phase1  # Should restore 1,500-3,000 sims/sec
```

---

### Phase 3A Gate: Multi-Coordinator (SC-010 to SC-012) 🎯 STRETCH (Optional)

**⚠️ ONLY RUN IF**: Phase 2 meets 7k-9k target AND stretch goal of 12k+ is desired

**Profiling Campaign** (100+ trials):
```bash
python scripts/benchmark_phase3a.py --coordinators 4 --trials 100 --output profiling_phase3a_$(date +%Y%m%d)
python scripts/profiling/analyze_campaign.py profiling_phase3a_*/ --compare-to-phase2
```

**Acceptance Criteria**:
- [ ] **Throughput**: 12,000-20,000 sims/sec (mean, p50) ✅ **100-166× baseline**
- [ ] **Coordinator Blocking**: <10% of iteration time (down from 99.6%)
- [ ] **Linear Scaling**: 4 coordinators → 3.2-3.6× throughput vs 1 coordinator
- [ ] **GPU Utilization**: ≥85% during search (multi-stream inference)

**Decision Gate**:
- If throughput ≥12,000: **SUCCESS**, Phase 3B not needed
- If throughput <12,000 AND Python callback >5ms: Consider Phase 3B (multi-process)
- If throughput <12,000 AND Python callback ≤5ms: Investigate other bottlenecks

---

## Profiling Metrics Validation

**Dominant Bottleneck Shift** (expected across phases):

| Metric | Baseline | Phase 1 Target | Phase 2 Target | Phase 3A Target |
|--------|----------|----------------|----------------|-----------------|
| **Throughput** | 120 sims/sec | 1,500-3,000 | 7,000-9,000 ✅ | 12,000-20,000 |
| **State Cloning** | 86.6% | <1% | <1% | <1% |
| **Coordinator Blocking** | 99.6% | ~95% | ~50% | <10% |
| **Tensor Creation** | 37ms | 37ms | ≤2ms | ≤2ms |
| **OpenMP Threads** | 0 (broken) | 0 (deferred) | >1 ✅ | >1 |
| **GPU Utilization** | ~68% | ~70% | ≥80% ✅ | ≥85% |

**Phase Transition Validation**:
- [ ] **Baseline → Phase 1**: State cloning drops from 86.6% to <1%, throughput increases 10-25×
- [ ] **Phase 1 → Phase 2**: Tensor creation drops from 37ms to ≤2ms, OpenMP enabled, throughput increases 4-6× (total 58-75× baseline)
- [ ] **Phase 2 → Phase 3A** (optional): Coordinator blocking drops from ~50% to <10%, throughput increases 1.7-2.2× (total 100-166× baseline)

---

## Quick Validation Commands

### A) No-Clone Hot Path
```bash
# Audit for state cloning (should find 0 occurrences)
scripts/audit_state_cloning.sh

# Verify InferenceRequest owns features
grep "std::vector<float> features" cpp_extensions/mcts/async_inference_queue.hpp

# Verify move-only semantics
grep "InferenceRequest(const InferenceRequest&) = delete" cpp_extensions/mcts/async_inference_queue.hpp
```

### B) Coordinator Throughput
```bash
# Verify OpenMP linked
scripts/verify_openmp.sh

# Verify pinned memory
python -c "from src.core.dlpack_inference_bridge import DLPackInferenceBridge; bridge = DLPackInferenceBridge(); print(f'Pinned: {bridge.pinned_buffer.is_pinned()}')"

# Run zero-allocation test
python -m pytest tests/contract/test_coordinator_api.py::test_zero_allocation_in_batching -v
```

### C) Phase Gate Benchmarks
```bash
# Phase 1 validation (expect 1.5k-3k sims/sec)
python scripts/benchmark_phase1.py --trials 100

# Phase 2 validation (expect 7k-9k sims/sec) ✅
python scripts/benchmark_phase2.py --trials 100

# Phase 3A validation (optional, expect 12k-20k sims/sec)
python scripts/benchmark_phase3a.py --coordinators 4 --trials 100
```

---

## Automated Validation Script

**Run All Checklists**:
```bash
#!/bin/bash
set -e

echo "=== A) No-Clone Hot Path Checklist ==="
scripts/audit_state_cloning.sh || { echo "❌ State cloning detected"; exit 1; }
echo "✅ Zero state cloning in hot paths"

echo "=== B) Coordinator Throughput Checklist ==="
scripts/verify_openmp.sh || { echo "❌ OpenMP not linked"; exit 1; }
echo "✅ OpenMP linked successfully"

python -m pytest tests/contract/test_coordinator_api.py::test_zero_allocation_in_batching -v || { echo "❌ Allocations detected in hot loop"; exit 1; }
echo "✅ Zero allocations in coordinator hot loop"

python -c "from src.core.dlpack_inference_bridge import DLPackInferenceBridge; bridge = DLPackInferenceBridge(); assert bridge.pinned_buffer.is_pinned(), 'Buffer not pinned'" || { echo "❌ Pinned buffer not allocated"; exit 1; }
echo "✅ Pinned memory buffers allocated"

echo "=== C) Phase Gate Benchmarks ==="
# Run appropriate phase validation based on current state
if [ -f "profiling_phase2_*/results.json" ]; then
    echo "Running Phase 2 validation..."
    python scripts/benchmark_phase2.py --trials 100
    # Analyze and verify 7k-9k sims/sec target
else
    echo "Running Phase 1 validation..."
    python scripts/benchmark_phase1.py --trials 100
    # Analyze and verify 1.5k-3k sims/sec target
fi

echo "✅ All checklists passed"
```

**Save as**: `scripts/validate_checklists.sh`

---

## References

- [Constitution](../.specify/memory/constitution.md): 6 core principles (Zero-Copy First, Coordinator Efficiency, etc.)
- [Spec](../spec.md): User stories and success criteria (SC-001 to SC-015)
- [Plan](../plan.md): Implementation details with exact code touch-points
- [Tasks](../tasks.md): Task list with DoD for each sub-phase
- [Contracts](../contracts/): API interface specifications

---

## Summary

**Three high-signal checklists for rapid validation**:

✅ **A) No-Clone Hot Path**: Verify zero state cloning via code review + profiling (<1% overhead, 0 clones)

✅ **B) Coordinator Throughput**: Verify condition variables, zero allocation, Python callback ≤2ms, OpenMP >1 thread

✅ **C) Phase Gate Benchmarks**: Verify Phase 1 (1.5k-3k sims/sec), Phase 2 (7k-9k sims/sec ✅ TARGET), Phase 3A (12k-20k sims/sec optional)

**Each checklist maps to constitution principles and includes automated validation commands for immediate execution.**
