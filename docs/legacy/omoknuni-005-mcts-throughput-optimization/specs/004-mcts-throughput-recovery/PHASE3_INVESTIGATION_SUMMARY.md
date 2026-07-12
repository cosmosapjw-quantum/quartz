# Phase 3 Investigation Summary

**Date**: 2025-10-21
**Session**: Comprehensive analysis of T024f-6 performance
**Status**: Investigation complete, root cause partially identified
**Throughput**: 864-1,363 sims/sec (benchmark), target 3,500+ sims/sec

---

## Executive Summary

We conducted a comprehensive investigation into Phase 3 (T024f-6 make/unmake pattern) performance. The implementation is **functionally correct** and achieves its design goal of 50% state cloning reduction, but reveals a **2× overall throughput regression** that needs to be addressed.

### Key Findings

1. **T024f-6 Works As Designed** ✅
   - State cloning reduced from 836ms → 172ms (50% reduction achieved)
   - Make/unmake pattern functional (0.068μs per move)
   - Thread-local state management working correctly

2. **Real Performance** (from `benchmark_throughput.py`)
   - Current: 864-1,363 sims/sec (mean ~1,034 sims/sec)
   - Baseline: 2,659 sims/sec
   - **Regression: 2.57× slower**

3. **GPU Inference is NOT the Problem** ✅
   - Isolated benchmark shows 5-16ms per batch (normal)
   - Profiler showed 148ms, but that was measuring dummy numpy array creation
   - Real GPU model works fine

4. **Profiler Limitations** ⚠️
   - `unified_profiler.py` uses incompatible callback interface
   - Cannot easily integrate real GPU model with ContinuousSimulationRunner
   - Use `benchmark_throughput.py` as source of truth instead

---

## What We Know

### Baseline Performance (Oct 16, 2025)
```
Total time:        982.86ms for 2,000 sims
Throughput:        2,659 sims/sec
Primary bottleneck: State cloning (836ms, 86.6%)
GPU inference:     20.66ms (2.1%)
```

### Current Performance (T024f-6)
```
Total time:        ~1,930ms for 2,000 sims (estimated)
Throughput:        1,034 sims/sec (mean of 3 runs)
State cloning:     172ms (4.5% - IMPROVED ✅)
Unknown overhead:  ~1,758ms (91% - NEW BOTTLENECK 🔴)
```

### Performance Gap Analysis
```
Expected (with 50% state cloning reduction):
  Remove 418ms overhead per 2000 sims
  982ms → 564ms per 2000 sims
  = 3,546 sims/sec

Actual:
  1,034 sims/sec

Gap: 3,546 - 1,034 = 2,512 sims/sec missing
Missing time: ~1,366ms per 2000 sims
Per-sim overhead: ~0.68ms extra per simulation
```

---

## Root Cause Hypotheses

### Hypothesis A: Coordinator Inefficiency (MOST LIKELY)
**Evidence**:
- Profiler shows `coordinator_python_callback` taking 33% of time
- Profiler shows `run_continuous_idle_count: 34,466` (excessive!)
- Threads spending time sleeping/waiting

**Possible causes**:
- Coordinator not batching efficiently
- Too many small batches (1, 7, 47) before reaching batch size 64
- Queue backpressure causing simulation threads to wait
- Synchronization overhead between C++ and Python

**Impact**: ~1,000-1,500ms of extra overhead

### Hypothesis B: Make/Unmake Integration Overhead
**Evidence**:
- Thread-local state management adds bookkeeping
- Undo token collection/storage
- Path buffer management

**Possible causes**:
- Thread-local state initialization overhead
- Extra memory allocations for undo tokens
- Cache misses due to different memory access patterns

**Impact**: ~200-400ms of extra overhead

### Hypothesis C: Queue/Memory Changes
**Evidence**:
- T024f-6 modified queue submission logic
- State cloning still happens once (for queue)

**Possible causes**:
- Different queue filling patterns
- Memory allocation patterns changed
- Lock contention in queue operations

**Impact**: ~200-400ms of extra overhead

---

## Investigation Steps Completed

1. ✅ **Ran T024f-6 tests** - All passing (4/4 equivalence, 5/5 integration except performance)
2. ✅ **Ran throughput benchmark** - Confirmed 2× regression
3. ✅ **Ran isolated GPU benchmark** - Confirmed GPU is normal (5-16ms)
4. ✅ **Attempted profiler with real GPU** - Hit interface incompatibilities
5. ✅ **Analyzed profiling data** - Identified coordinator as suspected bottleneck
6. ✅ **Created analysis documents** - Documented findings thoroughly

---

## Recommended Next Steps

### Step 1: Compare Baseline vs T024f-6 Coordinator Behavior
**Goal**: Identify what changed in coordinator/queue interaction

**Method**:
```bash
# Profile baseline (without make/unmake)
git stash  # Temporarily revert T024f-6 changes
python scripts/benchmark_throughput.py --simulations 2000 --threads 1 --iterations 5

# Profile T024f-6
git stash pop
python scripts/benchmark_throughput.py --simulations 2000 --threads 1 --iterations 5

# Compare results
```

### Step 2: Add Targeted Instrumentation
**Goal**: Measure where the extra 0.68ms per simulation is spent

**Add timing points in**:
- `continuous_simulation_runner.cpp:run_continuous()`
- Queue submission (`queue.submit_request`)
- Result processing (`process_completed_results`)
- Thread-local state operations

**Method**:
```cpp
// In run_continuous loop
auto loop_start = std::chrono::high_resolution_clock::now();
// ... simulation code ...
auto loop_end = std::chrono::high_resolution_clock::now();
auto loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
std::cout << "Loop iteration: " << loop_duration.count() << "μs\n";
```

### Step 3: Test Without Async Queue
**Goal**: Isolate whether the problem is make/unmake or queue/coordinator

**Method**:
Create a synchronous version of ContinuousSimulationRunner that doesn't use the async queue and directly calls a blocking inference function. If this is still slow, the problem is in the make/unmake implementation. If it's fast, the problem is in the queue/coordinator.

### Step 4: Profile Coordinator Batch Collection
**Goal**: Understand why coordinator is slow

**Check**:
- How long does `collect_batch()` take?
- Are batches being collected efficiently?
- Is there blocking/waiting happening?

### Step 5: Reduce Coordinator Timeout
**Current**: 5ms timeout
**Try**: 0.5-1.0ms timeout (match benchmark configuration)

This might reduce idle time and improve batching efficiency.

---

## Alternative Approach: Bypass the Issue

If the coordinator is fundamentally inefficient with the current architecture, consider:

### Option A: Use Old SimulationRunner with State Pools
- Keep state pooling optimization
- Use the old synchronous runner
- Expected: 2× improvement from state pooling alone

### Option B: Optimize Coordinator Separately
- Make coordinator more efficient as a separate task
- Fix batching logic
- Reduce synchronization overhead

### Option C: Move to Full Zero-Copy (T019)
- Skip incremental fixes
- Implement complete zero-copy architecture
- Expected: 5-10× improvement
- Timeline: 5-7 weeks

---

## Performance Targets

### T024f-6 Original Goal
- **Target**: 4,700 sims/sec (1.77× improvement)
- **Current**: 1,034 sims/sec (0.39× - regression!)
- **Gap**: 4.5× improvement needed

### Minimum Acceptable
- **Target**: 3,500 sims/sec (better than baseline)
- **Current**: 1,034 sims/sec
- **Gap**: 3.4× improvement needed

### Ultimate Goal (Spec 004)
- **Target**: 8,000 sims/sec
- **Current**: 1,034 sims/sec
- **Gap**: 7.7× improvement needed

---

## Conclusion

**What Worked**:
- ✅ State cloning reduction (50% achieved)
- ✅ Make/unmake pattern implementation (0.068μs per move)
- ✅ Thread-local state management
- ✅ Integration tests passing

**What Didn't Work**:
- ❌ Overall throughput (2.57× regression)
- ❌ Coordinator efficiency
- ❌ Unknown overhead consuming 91% of time

**Critical Finding**:
The 50% state cloning reduction should have given us 3,546 sims/sec, but we're only seeing 1,034 sims/sec. There's **~1,366ms of extra overhead** somewhere that's eating all the gains and then some.

**Next Action**:
Run Step 1 (baseline vs T024f-6 comparison) to identify what changed in the coordinator/queue behavior. This is the highest-priority diagnostic step.

---

**Documents Created**:
1. [T024F6_PERFORMANCE_REGRESSION_ANALYSIS.md](T024F6_PERFORMANCE_REGRESSION_ANALYSIS.md) - Initial (misleading) analysis
2. [T024F6_CORRECTED_ANALYSIS.md](T024F6_CORRECTED_ANALYSIS.md) - Corrected analysis
3. [PHASE3_INVESTIGATION_SUMMARY.md](PHASE3_INVESTIGATION_SUMMARY.md) - This document

**Status**: Ready for next investigation phase
**Owner**: cosmosapjw-quantum
**Recommended Next Step**: Baseline vs T024f-6 comparison (Step 1)
