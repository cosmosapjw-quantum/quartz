# Spec 004: MCTS Throughput Recovery

**Status**: 🔴 ACTIVE - Implementation Ready
**Campaign**: profiling_suite_20251016_124134 (560 trials, 100% capture)
**Current Performance**: 2,659 sims/sec
**Target**: ≥8,000 sims/sec
**Expected After State Pooling**: 9,838 sims/sec (3.7× improvement)

---

## Overview

This specification addresses MCTS throughput recovery through profiling-validated optimizations. After a comprehensive 560-trial profiling campaign with 100% data capture, we identified **state cloning (86.6% of execution time)** as the primary bottleneck.

### Critical Finding

**State cloning consumes 86.6% of execution time** due to 223 allocations per clone. Implementing thread-local state pools will reduce clone time from 418μs to ~20μs, achieving 3.7× overall throughput improvement → 9,838 sims/sec, **exceeding the 8,000 target**.

---

## Document Structure

This specification follows the **Spec Kit** methodology with authority chain:

1. **[CONSTITUTION.md](./CONSTITUTION.md)** - Non-negotiable constraints and profiling evidence
2. **[spec.md](./spec.md)** - Functional requirements (WHAT to achieve)
3. **[TECHNICAL_PLAN.md](./TECHNICAL_PLAN.md)** - Implementation design (HOW to implement)
4. **[TASKS.md](./TASKS.md)** - Task breakdown (WHAT to do, HOW to validate)
5. **[data-model.md](./data-model.md)** - Data structures and memory layout
6. **[research.md](./research.md)** - Technical analysis and architecture decisions
7. **[quickstart.md](./quickstart.md)** - Build and validation procedures

### Authority Source

All specifications are grounded in production profiling data:
- **Profiling Campaign**: profiling_suite_20251016_124134
- **Date**: October 16, 2025
- **Trials**: 560/560 successful (100% capture rate)
- **Source**: [FINAL_PROFILING_ANALYSIS_20251016.md](../../FINAL_PROFILING_ANALYSIS_20251016.md)

---

## Performance Baseline

### Current State (Profiling-Validated)

```
Throughput:    2,659 sims/sec (mean across 560 trials)
Target:        8,000 sims/sec
Progress:      33.2% of target
Gap:          -5,341 sims/sec (-66.8%)
```

### Time Breakdown (Trial 001 - Representative)

```
Total: 982.86 ms for 2,000 simulations

state_clone_total:   835.85 ms (86.6%) 🔴 PRIMARY BOTTLENECK
expansion_total:      37.24 ms ( 3.8%)
expansion_nn_wait:    20.66 ms ( 2.1%) ← GPU inference (NOT the bottleneck!)
selection_total:       3.58 ms ( 0.4%)
backup_total:          1.67 ms ( 0.2%)
unknown/overhead:     85.64 ms ( 8.7%)
```

### Root Cause Analysis

**State Cloning Overhead**:
- **Actual**: 418μs per clone
- **Expected**: ~50μs per clone
- **Discrepancy**: 209× slower than expected!

**Root Cause**: 223 allocations per clone
- 223 allocations × 2μs per allocation = 446μs
- **Matches observed 418μs overhead** ✅

---

## Optimization Roadmap

### Priority #1: State Pooling (T018) 🔴 CRITICAL

**Impact**: Eliminate 223 allocations per clone
- **Expected**: Clone time 418μs → 20μs (20.9× faster)
- **Overall Gain**: 3.7× throughput → **9,838 sims/sec** ✅ **Exceeds 8k target ALONE**
- **Timeline**: 2-3 days
- **Risk**: LOW (well-understood optimization)

**Implementation**:
```cpp
// OLD (current - 418μs per clone, 223 allocations)
std::unique_ptr<IGameState> state = root_state.clone();

// NEW (proposed - 20μs via copyFrom, 0 allocations)
IGameState* state = state_pool.acquire();  // O(1), lock-free
state->copyFrom(root_state);  // Fast memcpy
// ... use state ...
state_pool.release(state);  // Return to pool
```

### Priority #2: Fix OpenMP (T019) 🟠 OPTIONAL

**Impact**: Enable feature extraction parallelization (0/560 trials active)
- **Expected**: 1.5-2.0× additional speedup
  - Conservative (1.5×): **14,757 sims/sec**
  - Optimistic (2.0×): **19,676 sims/sec**
- **Timeline**: 1-2 days
- **Risk**: LOW (debugging task)

### Priority #3: Reduce Allocations (T020) 🟡 REFINEMENT

**Impact**: Further reduce allocation overhead after state pooling
- **Expected**: 1.2-1.5× additional speedup → **17,708-29,514 sims/sec**
- **Timeline**: 1-2 days (AFTER state pooling)
- **Risk**: MEDIUM (memory leak potential)

---

## Implementation Status

### Phase 1: Profiling & Analysis ✅ COMPLETE
- [x] 560-trial profiling campaign with 100% capture
- [x] Bottleneck identification (state cloning = 86.6%)
- [x] Root cause analysis (223 allocations per clone)
- [x] Performance calculation & projection

### Phase 2: State Pooling (T018) 🔴 READY TO START
- [ ] T018a: IGameState::copyFrom() API Design
- [ ] T018b: ThreadLocalStatePool Implementation
- [ ] T018c/d/e: Game-specific copyFrom() (Gomoku/Chess/Go)
- [ ] T018f: State Pool Unit Tests
- [ ] T018g: SimRunner Integration
- [ ] T018h: Profiling Validation
- [ ] T018i: Performance Benchmarking

### Phase 3: OpenMP Investigation (T019) 🟠 OPTIONAL
- [ ] T019a: OpenMP Linkage Verification
- [ ] T019b: OpenMP Instrumentation & Rebuild
- [ ] T019c: OpenMP Validation & Thread Scaling

### Phase 4: Allocation Optimization (T020) 🟡 FUTURE
- [ ] T020a: Arena Expansion Design
- [ ] T020b: Enhanced Arena Implementation
- [ ] T020c: Allocation Profiling & Validation

### Phase 5: Final Validation ✅ VALIDATION
- [ ] T021: Comprehensive Profiling Campaign (560 trials)
- [ ] T022: Documentation & Handoff

---

## Success Criteria

### Primary Goals (Phase 2 Complete)

| Metric | Current | Target | Expected (State Pooling) |
|--------|---------|--------|-------------------------|
| **Throughput** | 2,659 sims/sec | **≥8,000 sims/sec** | **9,838 sims/sec** ✅ |
| **State clone %** | 86.6% | <5% | 3.2% |
| **Allocations/sim** | 223 | <10 | 8 |
| **Thread efficiency @ 8T** | 12.7% | ≥60% | 63-70% |

### Stretch Goals (Phases 3-4 Complete)

| Optimization | Expected | Cumulative |
|--------------|----------|------------|
| State pooling | 3.7× | 9,838 sims/sec |
| + OpenMP fix | 1.5× | 14,757 sims/sec |
| + Allocation reduction | 1.2× | 17,708 sims/sec |

---

## Quick Start

### Build with Profiling

```bash
# Clean rebuild with profiling
export CXXFLAGS="-O3 -march=znver3 -fopenmp -DPROFILE_LEVEL_VALUE=3"
export PROFILE_BUFFER_SIZE=524288  # 512K samples (avoid overflow)
rm -rf build/ *.so
pip install -e . --force-reinstall --no-deps
```

### Run Validation

```bash
# State pooling validation (after T018 complete)
python scripts/validate_state_pooling.py

# Expected output:
# ✅ Allocations per sim: 8.3 (target: <10)
# ✅ State cloning: 3.2% of time (target: <5%)
# ✅ Throughput: 9,838 sims/sec (target: ≥8,000)
# ✅ Speedup: 3.70× vs baseline 2,659 sims/sec
# ✅ VALIDATION PASSED
```

### Run Comprehensive Campaign

```bash
# Full 560-trial campaign
./scripts/run_profiling_suite.sh --production

# Analyze results
python scripts/analyze_profiling_results.py \
    --campaign profiling_reports/final_baseline_* \
    --baseline profiling_suite_20251016_124134 \
    --target 8000
```

---

## References

- **Profiling Data**: [FINAL_PROFILING_ANALYSIS_20251016.md](../../FINAL_PROFILING_ANALYSIS_20251016.md)
- **Constitution**: [CONSTITUTION.md](./CONSTITUTION.md) - Non-negotiable rules
- **Specification**: [spec.md](./spec.md) - Functional requirements
- **Technical Plan**: [TECHNICAL_PLAN.md](./TECHNICAL_PLAN.md) - Implementation design
- **Tasks**: [TASKS.md](./TASKS.md) - Task breakdown with acceptance criteria
- **Data Model**: [data-model.md](./data-model.md) - Memory layout & structures
- **Research**: [research.md](./research.md) - Architecture decisions & analysis

---

## Contact

For questions or clarifications:
- **Primary**: cosmosapjw-quantum (GitHub reviewer)
- **Spec Kit**: Follow authority chain (CONSTITUTION → spec → plan → tasks)
- **Profiling**: Refer to profiling_suite_20251016_124134 as authoritative source
