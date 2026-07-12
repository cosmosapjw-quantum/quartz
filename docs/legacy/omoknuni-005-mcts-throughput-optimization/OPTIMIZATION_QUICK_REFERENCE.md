# MCTS Optimization Quick Reference

**Last Updated**: 2025-10-24
**Full Report**: [MCTS_PERFORMANCE_ANALYSIS_REPORT.md](MCTS_PERFORMANCE_ANALYSIS_REPORT.md)
**Profiling Overhead Analysis**: [PROFILING_OVERHEAD_ANALYSIS_FINAL.md](PROFILING_OVERHEAD_ANALYSIS_FINAL.md)

---

## ⚠️ CRITICAL: Profiling Overhead Impact

**Campaign 1** (Profiling Mostly Disabled): **2,303 sims/sec** ← Production baseline
**Campaign 2** (ALL Profiling Enabled): **602 sims/sec** ← Debug only (3.83× slowdown!)

**All numbers below are from Campaign 1 (production configuration)**

---

## Current Performance (560 Trials - Production Config)

```
Mean Throughput:  2,303 sims/sec (28.8% of 8k target)
Peak Throughput:  3,420 sims/sec (42.7% of 8k target)
Target:           8,000 sims/sec

Status: 🔴 CRITICAL - Coordinator serialization bottleneck
```

---

## Critical Finding: Zero Thread Scaling!

| Threads | Mean Throughput | Improvement |
|---------|-----------------|-------------|
| 1       | 2,297 sims/sec  | Baseline    |
| 2       | 2,309 sims/sec  | +0.5%       |
| 4       | 2,304 sims/sec  | +0.3%       |
| 8       | 2,302 sims/sec  | +0.2%       |
| 12      | 2,307 sims/sec  | +0.4%       |

**Expected**: 6-7× improvement with 8 threads
**Actual**: 0.2% improvement
**Conclusion**: Coordinator is serializing all work!

---

## Root Causes (Profiling-Validated)

### 1. Coordinator Serialization (PRIMARY - 89.8% of trials)
```
Coordinator loop iteration: 35.3ms mean
  ├─ Collect batch:        4.8ms (13.6%)
  ├─ Feature extraction:   0.2ms (0.6%)
  └─ Python callback:     30.3ms (85.8%)  ← BOTTLENECK

Python callback breakdown:
  ├─ Tensor creation:      3.8ms (12.5%)
  ├─ GPU inference:       15.0ms (49.5%)
  ├─ D2H transfer:         8.0ms (26.4%)
  └─ Python overhead:      2.2ms (7.3%)
```

### 2. GPU Transfer Inefficiency
```
Transfer size:    ~5 MB (batch of 128)
Transfer time:    8.0 ms
Actual bandwidth: 625 MB/s (2.5% of PCIe theoretical!)
Expected:         2,500 MB/s (10% of PCIe)
```

### 3. Tensor Creation Overhead
```
Feature extraction: 1.3 ms
Tensor creation:    3.8 ms (2.9× feature extraction!)

Expected: <0.5ms (zero-copy with DLPack)
Actual: 3.8ms (indicates copies happening)
```

---

## Optimization Path to 8k Target

### Phase 1: Tensor + Python (1-2 weeks)
**Target**: 2,303 → 2,971 sims/sec (+29%)
- Optimize tensor creation: 3.8ms → 0.5ms
- Reduce Python overhead: 2.2ms → 0.5ms

### Phase 2: GPU Transfer (2-3 weeks)
**Target**: 2,971 → 4,457 sims/sec (+50% more, +94% total)
- Verify non-blocking transfers working
- Increase batch size to 256
- Pipeline transfers with multiple streams
- D2H transfer: 8ms → 2ms

### Phase 3: Parallel Coordinators ⭐ CRITICAL ⭐
**Target**: 4,457 → 9,212 sims/sec (+107% more) **← ACHIEVES 8K TARGET**
- Implement 4 parallel coordinators
- Shard inference queue by thread ID
- Independent GPU streams per coordinator
- Expected scaling: 4× improvement

### Phase 4: Advanced (Optional)
**Target**: 9,212 → 10,000-14,000 sims/sec (+9-52% more)
- Re-investigate OpenMP feature extraction
- Further reduce Python overhead

---

## Optimal Configuration (From 560 Trials)

### Current Best (Trial #479)
```yaml
Simulations:      16,000
Thread Count:     4
Batch Size:       128
Timeout:          0.5-1.0ms
Mixed Precision:  FP16 enabled
CUDA Graphs:      Disabled (avoid fallback)

Result: 3,420 sims/sec (peak performance)
```

### Recommended for Production
```yaml
Simulations:      800 per move
Thread Count:     4 (until parallel coordinators)
Batch Size:       128
Timeout:          1.0ms
Mixed Precision:  FP16 enabled
CUDA Graphs:      Disabled

Expected: 2,500-2,800 sims/sec
```

### After Parallel Coordinator Implementation
```yaml
Simulations:      800 per move
Thread Count:     8-12 (will scale with coordinators)
Batch Size:       256 (larger batches)
Timeout:          1.0ms
Coordinators:     4 (parallel inference)
Mixed Precision:  FP16 enabled
CUDA Graphs:      Re-enable

Expected: 9,000-13,000 sims/sec ✅ TARGET ACHIEVED
```

---

## Profiling Results Summary

```
Campaign:    profiling_suite_20251024_055901
Trials:      560 (100% success rate)
Duration:    28.2 minutes

Overhead:    2.6% (profiling framework working correctly ✅)

Bottleneck Frequency:
  - coordinator_loop_iteration: 503 trials (89.8%)
  - unknown:                     57 trials (10.2%)

Performance by Batch Size:
  - Batch 16:  1,848 sims/sec
  - Batch 32:  2,362 sims/sec (+27.8%)
  - Batch 64:  2,389 sims/sec (+29.3%)
  - Batch 128: 2,614 sims/sec (+41.5%) ← OPTIMAL
```

---

## Next Steps (Priority Order)

1. **This Week**:
   - ✅ Comprehensive profiling analysis complete
   - ✅ Polished report generated
   - ✅ Outdated documents removed
   - 🔲 Establish baseline metrics with current codebase
   - 🔲 Create detailed design doc for parallel coordinators

2. **Next 2 Weeks (Phase 1)**:
   - 🔲 Optimize tensor creation (3.8ms → 0.5ms)
   - 🔲 Reduce Python overhead (2.2ms → 0.5ms)
   - 🔲 Validate +29% improvement

3. **Weeks 3-6 (Phase 2)**:
   - 🔲 Optimize GPU transfers (8ms → 2ms)
   - 🔲 Increase batch size to 256
   - 🔲 Validate +94% total improvement

4. **Weeks 7-10 (Phase 3) ⭐ CRITICAL ⭐**:
   - 🔲 Implement parallel coordinators (4× scaling)
   - 🔲 Validate thread scaling (1→8 threads)
   - 🔲 **ACHIEVE 8,000+ sims/sec target** ✅

---

## Key Metrics to Track

### Performance Metrics
- **Throughput**: sims/sec (target: 8,000)
- **Thread Scaling**: X× improvement from 1→8 threads (target: 4-6×)
- **GPU Utilization**: % (target: 80%)
- **Coordinator Wait**: ms (target: <20ms, currently 139ms)

### Timing Metrics
- **Callback Time**: ms (target: <10ms, currently 30.3ms)
- **Tensor Creation**: ms (target: <0.5ms, currently 3.8ms)
- **D2H Transfer**: ms (target: <2ms, currently 8ms)
- **Python Overhead**: ms (target: <0.5ms, currently 2.2ms)

### Quality Metrics
- **Profiling Overhead**: % (target: <5%, currently 2.6% ✅)
- **Memory Footprint**: MB (target: <1GB, currently 270MB ✅)
- **Thread Safety**: TSan clean (currently ✅)

---

## Files & Documentation

### Primary Documents
- 📊 **[MCTS_PERFORMANCE_ANALYSIS_REPORT.md](MCTS_PERFORMANCE_ANALYSIS_REPORT.md)** - Complete 662-line analysis
- 📌 **This File** - Quick reference for optimization work

### Profiling Data
- 📁 `profiling_suite_20251024_055901/` - 560 trial results
- 📈 `profiling_suite_20251024_055901/campaign/results.csv` - Aggregate data
- 🔍 `profiling_suite_20251024_055901/campaign/trial_479/` - Best performer (3,420 sims/sec)

### Implementation Files
- 🔧 `scripts/wall_clock_validation.py` - Baseline measurement (now uses GPU)
- 🔧 `scripts/unified_profiler.py` - Comprehensive profiling (CUDA graphs disabled)
- 🔧 `cpp_extensions/mcts/dlpack_bridge.cpp` - DLPack zero-copy (fixed kDLCPU)
- 🔧 `src/core/dlpack_inference_bridge.py` - CUDA streams (non-blocking transfers)

---

**Status**: ✅ Analysis complete, optimization path clear
**Timeline**: 8-10 weeks to 8,000+ sims/sec target
**Critical Path**: Parallel coordinator implementation (Phase 3)
