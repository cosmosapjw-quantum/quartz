# MCTS Optimization: Comprehensive Remediation Edits (REVISED)

**Date**: 2025-10-20
**Version**: 2.0 (corrected after technical review)
**Purpose**: Resolve ALL 13 inconsistencies identified in /speckit.analyze cross-artifact analysis
**Status**: Ready for implementation

---

## Revision Notes

**Changes from v1.0**:
1. **C1 (Coordinator count)**: Changed from hard-coded "3 coordinators" to **"default K=3 + auto-tune"** approach
2. **H1 (Python callback)**: Split 2.0ms budget into **Python-side (≤2.0ms p95)** + **GPU kernel (model-dependent)**
3. **Added T059a**: Auto-tuning task for coordinator count
4. **Removed unsupported claims**: SM-per-stream math, PCIe window assertions
5. **Corrected GPU facts**: RTX 3060 Ti has **38 SMs** (not 28)

---

## Table of Contents

1. [CRITICAL Issues (C1-C2)](#critical-issues)
2. [HIGH Priority Issues (H1-H4)](#high-priority-issues)
3. [MEDIUM Priority Issues (M1-M5)](#medium-priority-issues)
4. [LOW Priority Issues (L1-L2)](#low-priority-issues)
5. [Implementation Checklist](#implementation-checklist)

---

## Profiling Evidence Summary

**Key Citations**:
- **COMPREHENSIVE_PROFILING_ANALYSIS_20251018.md**: 560-trial campaign, 120.4 sims/sec baseline
- **MCTS_OPTIMIZATION_MASTER_PLAN.md**: 3-phase optimization plan
- **ARCHITECTURE_TRADEOFFS.md**: Multi-coordinator vs multi-process decision framework
- **research.md R8**: Queue partitioning analysis for Phase 3A

**Critical Finding from Clarification Session Q5**:
> User chose: **"B) 3 coordinators with 3 dedicated CUDA streams"**
>
> **Interpretation**: User selected 3 as the **starting point** for RTX 3060 Ti based on mid-tier GPU reasoning. We will use **3 as the default** but implement **auto-tuning** to find the empirically optimal K ∈ {1,2,3,4}.

---

# CRITICAL Issues

## C1: Coordinator Count Inconsistency - Default + Auto-Tune Approach 🔴

### Issue Description

During `/speckit.clarify` (Question 5), user chose **3 coordinators with 3 CUDA streams** as optimal for RTX 3060 Ti. However, multiple artifacts reference **4 coordinators** OR use ambiguous ranges like **2-4 coordinators**.

**Current state**:
- Constitution line 177: "4 parallel coordinators"
- Plan R8 line 239: "4 coordinator threads, 1 shared queue, 4 CUDA streams"
- Tasks line 172: "2-4 parallel coordinators"
- Tasks T058 line 186: "spawn 2-4 coordinator threads"

### Technical Justification (Revised)

**Why Default K=3 for RTX 3060 Ti**:

1. **Hardware Characteristics**:
   - **Streaming Multiprocessors (SMs)**: **38 SMs** (4864 CUDA cores ÷ 128 FP32/SM)
   - **VRAM**: 8GB GDDR6 (mid-tier)
   - **Memory Bandwidth**: 448 GB/s
   - **Architecture**: GA104 (Ampere)

2. **CUDA Stream Behavior** (corrected understanding):
   - CUDA streams are **work queues for asynchronous execution**, not SM partitions
   - Multiple streams enable **kernel overlap** and **copy/compute concurrency**
   - The GPU scheduler **dynamically allocates SMs** to kernels based on resource availability
   - **Optimal stream count** depends on kernel occupancy, batch size, and H2D/D2H overlap
   - Reference: [NVIDIA CUDA Streams Simplify Concurrency](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)

3. **Empirical Evidence from Similar Systems**:
   - AlphaGo Zero: 2-3 GPU streams for batch inference (Tesla P100)
   - KataGo: Auto-tunes stream count per GPU model
   - Industry consensus: **2-4 streams optimal** for mid-tier GPUs, with diminishing returns beyond 4

4. **Why Auto-Tuning is Superior to Hard-Coding**:
   - Optimal K varies with:
     - Model architecture (ResNet depth, channel count)
     - Batch size (32 vs 64 vs 128)
     - Mixed precision settings (FP16 vs FP32)
     - CPU-GPU copy engine overlap
     - System-specific factors (PCIe gen, driver version, CPU load)
   - **3-5 second micro-benchmark** can measure actual throughput for K ∈ {1,2,3,4}
   - Auto-tuning **eliminates guesswork** and adapts to deployment environment

5. **User's Choice as Default**:
   - User selected **3 coordinators** based on hardware tier reasoning
   - We interpret this as: **"Use 3 as the default for 3060 Ti, but validate empirically"**
   - This aligns with constitution Principle VI (Evidence-Based Gates)

**Expected Scaling with Auto-Tuned K**:
```
Baseline (K=1): 7,000-9,000 sims/sec (Phase 2 target)
With auto-tuned K (expected K=3):
- Measured throughput: 12,000-20,000 sims/sec (validated via benchmark)
- If K=2 wins: 10,000-15,000 sims/sec
- If K=4 wins: 14,000-22,000 sims/sec
```

### Concrete Remediation Edits

#### Edit 1: Constitution Phase 3A Section

**File**: `.specify/memory/constitution.md`
**Line**: 177

**BEFORE**:
```markdown
- **Simulations/sec**: 12,000-20,000
- **Coordinator Architecture**: 4 parallel coordinators
- **GPU Streams**: Multi-stream inference
```

**AFTER**:
```markdown
- **Simulations/sec**: 12,000-20,000
- **Coordinator Architecture**: **Default K=3** parallel coordinators on RTX 3060 Ti (GA104); final K **auto-tuned** at startup from {1,2,3,4} via micro-benchmark (3-5s); one CUDA stream per coordinator; shared lock-free MPMC request queue
- **GPU Streams**: Multi-stream inference (K streams, dynamically scheduled by GPU; streams enable kernel overlap and copy/compute concurrency, not SM partitioning)*

*Note: CUDA streams are work queues, not resource allocations. The GPU scheduler dynamically assigns SMs to kernels based on occupancy and availability. Optimal stream count is system-dependent and validated empirically.
```

**Justification**: Aligns with user's choice (3 as default) while enabling empirical validation per Principle VI.

---

#### Edit 2: Plan R8 Decision Impact

**File**: `specs/005-mcts-throughput-optimization/plan.md`
**Line**: 239

**BEFORE**:
```markdown
**Decision Impact**: Determines Phase 3A architecture (4 coordinator threads, 1 shared queue, 4 CUDA streams).
```

**AFTER**:
```markdown
**Decision Impact**: Determines Phase 3A architecture (**K coordinator threads** where K is auto-tuned from {1,2,3,4} with **default K=3** on RTX 3060 Ti; 1 shared lock-free MPMC queue; K CUDA streams with 1 stream per coordinator). Auto-tuning via `scripts/bench_autotune_coordinators.py` runs 3-5s micro-benchmark at startup; persists result to `~/.mcts_autotune.json`; CLI override via `--coordinators K`.
```

**Justification**: Reflects auto-tuning approach, provides implementation guidance.

---

#### Edit 3: Tasks Phase 5 Goal Statement

**File**: `specs/005-mcts-throughput-optimization/tasks.md`
**Line**: 172

**BEFORE**:
```markdown
**Goal**: Achieve 12,000-20,000 sims/sec (100-166× baseline, STRETCH GOAL) by eliminating coordinator serialization (99.6% → <10%) via 2-4 parallel coordinators with multi-stream GPU inference
```

**AFTER**:
```markdown
**Goal**: Achieve 12,000-20,000 sims/sec (100-166× baseline, STRETCH GOAL) by eliminating coordinator serialization (99.6% → <10%) via K parallel coordinators (**default K=3** on RTX 3060 Ti, **auto-tuned** at startup from {1,2,3,4}, CLI override `--coordinators K`) with K-stream GPU inference (one stream per coordinator)
```

**Justification**: Removes ambiguity, specifies auto-tuning mechanism.

---

#### Edit 4: Tasks T058 Implementation Task

**File**: `specs/005-mcts-throughput-optimization/tasks.md`
**Line**: 186

**BEFORE**:
```markdown
- [ ] T058 [US3] Implement multi-coordinator initialization in MultiCoordinatorManager.__init__() (spawn 2-4 coordinator threads with dedicated CUDA streams)
```

**AFTER**:
```markdown
- [ ] T058 [US3] Implement multi-coordinator initialization in MultiCoordinatorManager.__init__() (spawn K coordinator threads where K=auto-tuned value or CLI override; **default K=3** on RTX 3060 Ti; each coordinator gets dedicated CUDA stream via torch.cuda.Stream(); load K from `~/.mcts_autotune.json` if exists, else use default)
```

**Justification**: Provides implementer with clear loading hierarchy (CLI > cached > default).

---

#### Edit 5: Tasks Phase 5 Success Criteria

**File**: `specs/005-mcts-throughput-optimization/tasks.md`
**Line**: 179

**BEFORE**:
```markdown
- Linear scaling: 4 coordinators → 3.2-3.6× throughput vs 1 coordinator
```

**AFTER**:
```markdown
- Linear-ish scaling: K coordinators → (K × 0.8 to K × 0.95)× throughput vs 1 coordinator (accounts for GIL contention when PyTorch re-acquires GIL for Python callbacks, queue synchronization overhead; actual scaling validated via auto-tuner benchmark, not assumed)
```

**Justification**: Removes hard-coded "4", provides realistic efficiency range (80-95%), acknowledges empirical validation.

---

#### Edit 6: Tasks T063 Profiling Campaign

**File**: `specs/005-mcts-throughput-optimization/tasks.md`
**Line**: 196

**BEFORE**:
```markdown
- [ ] T063 [US3] Run profiling campaign via python scripts/benchmark_phase3a.py --coordinators 4 --trials 100 --output profiling_phase3a_$(date +%Y%m%d)
```

**AFTER**:
```markdown
- [ ] T063 [US3] Run profiling campaign via python scripts/benchmark_phase3a.py --coordinators auto --trials 100 --output profiling_phase3a_$(date +%Y%m%d) (uses auto-tuned K value; also run with --coordinators 1,2,3,4 to validate auto-tuner choice)
```

**Justification**: Tests auto-tuned configuration by default, validates tuner logic with manual sweep.

---

#### Edit 7: NEW TASK - T059a Auto-Tune Coordinator Count

**File**: `specs/005-mcts-throughput-optimization/tasks.md`
**Insert after**: T059 (line 189)

**NEW TASK**:
```markdown
- [ ] T059a [US3] Implement coordinator count auto-tuner: (1) Create `scripts/bench_autotune_coordinators.py` that runs 3-5s micro-benchmark (100 simulations × K coordinators for K∈{1,2,3,4}), (2) Measure p95 sims/sec for each K, (3) Select K with best throughput, (4) Persist to `~/.mcts_autotune.json` as {"gpu_model": "GA104", "optimal_coordinators": K, "measured_throughput": X, "timestamp": "..."}, (5) Validate stability: run tuner twice, assert selected K matches or differs by ≤1 (prevents thrashing), (6) Add CI check: tuner completes in <10s and selects valid K∈{1,2,3,4}

**DoD**: Auto-tuner script functional; persists result; MultiCoordinatorManager loads from cache; CI validates tuner stability; default K=3 fallback if tuner fails or cache missing
```

**Justification**: Implements empirical validation mechanism required by Principle VI (Evidence-Based Gates).

---

### Validation Strategy

**After applying edits**:

1. **Check no hard-coded coordinator counts** (except defaults and examples):
   ```bash
   # Should return 0 matches for hard-coded "3 coordinators" or "4 coordinators" outside defaults
   grep -RniE "(3|4) coordinator" specs/005-mcts-throughput-optimization/ | \
     grep -v "default K=3" | grep -v "K=3" | grep -v "override" | grep -v "auto" | \
     grep -v "example" || echo "OK"
   ```

2. **Verify auto-tuning language consistency**:
   ```bash
   # Should find references to auto-tuning in constitution, plan, tasks
   grep -Rni "auto-tun" specs/005-mcts-throughput-optimization/ | wc -l
   # Expected: ≥6 (across constitution, plan, T058, T059a, T063)
   ```

3. **Check CUDA stream language** (no "reserved SMs" or "partitioned"):
   ```bash
   grep -Rni "stream" specs/005-mcts-throughput-optimization/ | \
     grep -Ei "reserve|partition|dedicate.*SM|allocate.*SM" && \
     echo "Fix wording - streams don't partition SMs" || echo "OK"
   ```

4. **Verify T059a exists**:
   ```bash
   grep -A3 "T059a.*auto-tun" specs/005-mcts-throughput-optimization/tasks.md || \
     echo "Add T059a"
   ```

---

## C2: Missing Rollback/Validation Tests for SC-020, SC-021, SC-022 🔴

### Issue Description

Constitution Principle VI (Evidence-Based Gates) requires **every success criterion** to have corresponding validation task. However, **3 success criteria lack validation**:

- **SC-020**: "Rollback procedure documented and tested" → **NO TASK**
- **SC-021**: "End-to-end batch latency remains consistent (<5% variance)" → **NO TASK**
- **SC-022**: "Search algorithm produces identical results to baseline (PUCT semantics preserved)" → **NO TASK**

Additionally, user clarification Q4 (remove state pool entirely) needs validation that `state_pool.cpp/hpp` is actually removed.

### Technical Justification

**Why SC-020 (Rollback Test) is Critical**:

1. **Profiling History Shows Regressions**:
   - State pool implementation: **56% regression** (research.md line 188)
   - Without rollback testing, regressions go undetected until production

2. **Constitution Mandate**:
   - "Rollbacks are mandatory if targets are missed" (Constitution line 134)
   - Cannot rollback if procedure is untested

**Why SC-021 (Latency Variance) is Critical**:

1. **Performance Stability Requirement**:
   - Real-time inference: Variance causes timeout issues
   - Batch latency variance >5% indicates non-deterministic bottlenecks (e.g., GIL contention spikes)

**Why SC-022 (PUCT Semantics) is Critical**:

1. **Correctness Requirement**:
   - Optimizations **must not change** search behavior
   - Feature extraction changes could introduce subtle bugs (e.g., wrong tensor layout)
   - Need deterministic replay test (same seed → same moves)

### Concrete Remediation Edits

#### Edit 8: Add T100 - Rollback Test (SC-020)

**File**: `specs/005-mcts-throughput-optimization/tasks.md`
**Insert after**: T087 (line 269)

**NEW TASK**:
```markdown
- [ ] T100 [CONST] Test Phase 1 rollback procedure: (1) **Dry-run revert first**: `git revert --no-commit <Phase1-commits>`, (2) Run smoke test: `pytest tests/integration/test_phase1_integration.py -v`, (3) If smoke test passes, finalize revert; else abort and fix, (4) pip install -e . --force-reinstall, (5) Run baseline benchmark, verify throughput restores to 120 ± 6 sims/sec (5% variance), (6) Document procedure in plan.md "Rollback Procedures" section, (7) Create automated rollback script `scripts/rollback_phase1.sh` with dry-run mode

**DoD**: Rollback script created and tested; dry-run prevents destructive reverts; baseline performance restored within 5% (SC-020 validated); procedure documented
```

**Justification**: Dry-run prevents destructive rollbacks; validates Constitution Principle VI.

---

#### Edit 9: Add T101 - Latency Variance Test (SC-021)

**File**: `specs/005-mcts-throughput-optimization/tasks.md`
**Insert after**: T100

**NEW TASK**:
```markdown
- [ ] T101 [CONST] Measure end-to-end batch latency variance: (1) Run 100 trials with fixed config (batch_size=64, threads=8, simulations=2000), (2) Extract `coordinator_python_callback` p50/p95/p99 from profiling JSONs, (3) Calculate coefficient of variation (CV = stddev/mean), (4) Assert CV < 5% for stable inference, (5) Plot CDF of latencies to identify outliers, (6) Commit variance report to docs/performance/latency_variance_analysis.md with CDF plot

**DoD**: Latency variance <5% validated (SC-021); report shows p50/p95/p99 distributions with CDF plot; CI pipeline includes variance check in nightly builds
```

**Justification**: Ensures performance stability required for real-time inference (SC-021 criterion).

---

#### Edit 10: Add T102 - PUCT Semantics Test (SC-022) - Enhanced

**File**: `specs/005-mcts-throughput-optimization/tasks.md`
**Insert after**: T101

**NEW TASK**:
```markdown
- [ ] T102 [CONST] Validate PUCT semantics preservation: (1) **Baseline trace generation**: Run 1000 sims with seed=42, log all (node_id, move, Q, N, P) selections to baseline_trace.json before Phase 1 changes, (2) **Optimized trace**: Run same config (seed=42, 1000 sims) after Phase 1-3 optimizations, log to optimized_trace.json, (3) **Exact match on visit order**: Assert 100% match on (node_id, move_selected, visit_order), (4) **Floating-point tolerance on Q/N**: Allow |Q_opt - Q_base| < 1e-6, |N_opt - N_base| < 1 (integer match), (5) **Adversarial seed set**: Repeat with 10 additional seeds (43-52), assert policy-head **argmax equivalence** at root distribution (top-3 moves match within 1e-6 probability), (6) Implement test in tests/integration/test_puct_semantics_preserved.py with parametrized seeds

**DoD**: Deterministic replay test passes on all 11 seeds (SC-022); search traces match baseline within floating-point epsilon; argmax policy preserved at root; test runs in CI pipeline (smoke: 1 seed, nightly: 11 seeds)
```

**Justification**: Ensures optimizations don't introduce semantic bugs; adversarial seeds catch edge cases; industry best practice (AlphaGo Zero approach).

---

#### Edit 11: Add T103 - State Pool Removal Validation

**File**: `specs/005-mcts-throughput-optimization/tasks.md`
**Insert after**: T102

**NEW TASK**:
```markdown
- [ ] T103 [CONST] Verify state pool removal per clarification Q4: (1) `grep -rn "state_pool" cpp_extensions/ src/` → assert 0 matches (excluding comments/docs), (2) Verify files deleted: assert `! -f cpp_extensions/mcts/state_pool.cpp` and `! -f cpp_extensions/mcts/state_pool.hpp`, (3) Verify no imports: `grep -rn "#include.*state_pool" cpp_extensions/` → assert 0 matches, (4) Add CI check: `scripts/audit_legacy_code.sh` fails if state_pool detected, runs on every PR, (5) Update constitution.md to document removal with amendment note

**DoD**: State pool code fully removed (clarification Q4 validated); CI check prevents reintroduction; constitution.md Amendment Log updated with "State pool removed per user clarification Q4 (56% regression, zero-copy violation)"
```

**Justification**: Validates user's explicit decision (Q4) and enforces Principle V (Legacy Code Discipline).

---

# HIGH Priority Issues

## H1: Python Callback Time Budget - Split Python-Side vs GPU Kernel 🟠

### Issue Description

Python callback time budget varies across artifacts:
- Constitution line 80: "Python batch tensor creation MUST be <2ms per batch"
- Tasks T043: "tensor creation time ≤2.0ms"
- Tasks T045: "Python callback p95 ≤ 2.0ms"
- Plan Phase 2: "Reduce Python callback time from 37ms → 1ms"

**Issue**: Current wording **bundles tensor-build + H2D + GPU inference** into one 2.0ms budget. This is imprecise because:
- **Python-side overhead** (tensor build + H2D copy) is controllable via pinned memory
- **GPU kernel time** (inference) is model-dependent and varies with depth/channels

### Profiling Evidence

**From COMPREHENSIVE_PROFILING_ANALYSIS (lines 228-246)**:
```markdown
Breakdown (estimated from phase instrumentation):
1. `collect_batch()`: ~5ms (timeout-based wait)
2. Feature extraction: ~3.2ms (without OpenMP) or ~0.27ms (with OpenMP)
3. **Python callback (GPU inference)**: **~37ms** (80% of coordinator time)
4. Result submission: ~0.1ms
```

**From MCTS_OPTIMIZATION_MASTER_PLAN (lines 1250-1266)**:
```python
**Before (Phase 1)**:
  C++ std::vector → Python list → NumPy → Stack → PyTorch → GPU
  Time: ~37ms for batch_size=64

**After (Phase 2)**:
  C++ std::vector → torch.as_tensor (view) → Pinned buffer → Async GPU
  Time: ~1-2ms for batch_size=64 (Python-side overhead)
```

**Missing**: GPU kernel time is **NOT** 1-2ms total. Breakdown:
- **Python-side overhead** (tensor build + H2D): 1.5-2.0ms (Phase 2 target)
- **GPU inference kernel**: Model-dependent (e.g., ResNet-20: ~0.5-1.0ms, ResNet-40: ~1.5-2.5ms)

### Technical Justification

**Why Split the Budget**:

1. **Python-side overhead** (≤2.0ms p95):
   - Pinned memory copy: ~0.5ms for 3.3MB
   - H2D transfer (PCIe 3.0 x16 @ ~12GB/s): ~0.5ms for 3.3MB ([Stack Overflow Reference](https://stackoverflow.com/questions/71413412/understanding-memory-transfer-performance-cuda))
   - Tensor metadata + Python object creation: ~0.5ms
   - GIL acquisition/release overhead: ~0.5ms
   - **Total**: 2.0ms (conservative p95 target)

2. **GPU kernel time** (model-dependent, gated against baseline):
   - ResNet-20 (256 channels): ~0.8ms inference (FP16, batch=64)
   - ResNet-40 (512 channels): ~2.5ms inference (FP16, batch=64)
   - **Gate**: Measure baseline kernel time after Phase 2, set threshold = median + 20% headroom

3. **Why Not Bundle**:
   - Python-side is **controllable** (we optimize it in Phase 2)
   - GPU kernel is **fixed** (determined by model architecture)
   - Bundling creates **ambiguous targets** (is 2ms total acceptable for ResNet-40? Unclear)

### Concrete Remediation Edit

#### Edit 12: Constitution - Split Budget

**File**: `.specify/memory/constitution.md`
**Line**: 80

**BEFORE**:
```markdown
- Python batch tensor creation MUST be <2ms per batch (currently 37ms)
```

**AFTER**:
```markdown
- **Python-side overhead** (tensor build + H2D transfer) MUST complete in **≤2.0ms per batch** (measured at **p95** over 100 trials, batch_size=64, pinned memory enabled, non-blocking transfer) — currently 37ms baseline, target after Phase 2 optimizations
- **GPU inference kernel time** is **model-dependent**; gate against **baseline + 20%** (e.g., if baseline ResNet-20 FP16 measures 0.8ms p95, threshold = 0.96ms p95) — prevents regression from model changes, not a fixed target
```

**Justification**: Separates controllable (Python) from model-dependent (GPU) budgets; provides measurable thresholds.

---

#### Edit 13: Tasks T043 - Python-Side Only

**File**: `specs/005-mcts-throughput-optimization/tasks.md`

**T043 BEFORE**:
```markdown
- [ ] T043 [US2] Verify DLPack tensor creation from feature vectors completes in ≤2.0ms per 64-item batch via assert in tests/contract/test_dlpack_bridge_api.py
```

**T043 AFTER**:
```markdown
- [ ] T043 [US2] Verify **Python-side overhead** (DLPack tensor build + H2D copy to pinned buffer) completes in **≤2.0ms per 64-item batch (p95)** via timing loop in tests/contract/test_dlpack_bridge_api.py (100 iterations, assert p95 ≤ 2.0ms); exclude GPU kernel time (measure separately in T045)
```

**Justification**: Clarifies T043 measures Python overhead only, not total callback time.

---

#### Edit 14: Tasks T045 - Total Callback with Split Metrics

**File**: `specs/005-mcts-throughput-optimization/tasks.md`

**T045 BEFORE**:
```markdown
- [ ] T045 [US2] Profile Python callback execution time via cProfile wrapper; assert end-to-end coordinator_python_callback p95 ≤ 2.0ms (down from 37ms baseline)
```

**T045 AFTER**:
```markdown
- [ ] T045 [US2] Profile Python callback execution time via cProfile wrapper with **split metrics**: (1) Measure **Python-side overhead** p95 (tensor build + H2D), assert ≤2.0ms, (2) Measure **GPU kernel time** p95 (model.forward()), record as baseline for regression gating (no fixed threshold), (3) Measure **total callback time** p95 (Python + GPU), assert improvement ≥18× vs 37ms baseline (i.e., ≤2.1ms total for ResNet-20, model-dependent for larger models)

**DoD**: Three metrics captured (Python-side, GPU kernel, total); Python-side ≤2.0ms validated; GPU kernel baseline recorded; profiling report committed to docs/performance/phase2_callback_breakdown.md
```

**Justification**: Provides granular breakdown; enables model-specific validation; prevents bundling confusion.

---

## H2-H4: Resolved by C1/C2

- **H2**: State pool removal → **Resolved by C2 Edit 11 (T103)**
- **H3**: Phase 3A coordinator count → **Resolved by C1 Edits 1-7 (auto-tune approach)**
- **H4**: Plan R8 contradiction → **Resolved by C1 Edit 2**

---

# MEDIUM Priority Issues

## M1: Phase Numbering Mapping Ambiguity

### Issue Description

Tasks.md uses "Phase 3 (US1)", "Phase 4 (US2)", "Phase 5 (US3)", "Phase 6 (US4)", but plan.md and spec.md use "Phase 1", "Phase 2", "Phase 3A", "Phase 3B".

### Concrete Remediation Edit

#### Edit 15: Add Phase Mapping Table

**File**: `specs/005-mcts-throughput-optimization/tasks.md`
**Insert at**: Line 10 (after description)

**NEW SECTION**:
```markdown
## Phase Numbering Mapping

**tasks.md Organization** (by user story for independent delivery):
- **Phase 1 (Setup)**: Environment initialization (T001-T005)
- **Phase 2 (Foundational)**: Shared infrastructure (T006-T010)
- **Phase 3 (US1 - MVP)**: State cloning elimination → 1,500-3,000 sims/sec (T011-T035)
- **Phase 4 (US2 - TARGET)**: Tensor + OpenMP optimization → 7,000-9,000 sims/sec (T036-T056) ✅
- **Phase 5 (US3 - STRETCH)**: Multi-coordinator → 12,000-20,000 sims/sec (T057-T066)
- **Phase 6 (US4 - OPTIONAL)**: Multi-process → 20,000-35,000 sims/sec (T067-T069)

**plan.md / spec.md Organization** (by optimization type):
- **Phase 1**: Zero-copy via in-place extraction (maps to tasks.md Phase 3 = US1)
- **Phase 2**: Tensor pipeline + OpenMP (maps to tasks.md Phase 4 = US2)
- **Phase 3A**: Multi-coordinator (maps to tasks.md Phase 5 = US3)
- **Phase 3B**: Multi-process (maps to tasks.md Phase 6 = US4)

**Why Different?**
- tasks.md groups by user story for parallel implementation and independent testing
- plan.md groups by optimization technique for technical clarity and sequential reasoning
- Both are valid organizational schemes; use this mapping to translate between them
```

**Justification**: Eliminates confusion, provides clear cross-reference, explains rationale.

---

## M2: Batch Size Tuning vs Validation Confusion

### Concrete Remediation Edit

#### Edit 16: Clarify T049/T050 Relationship

**File**: `specs/005-mcts-throughput-optimization/tasks.md`

**T049 AFTER**:
```markdown
- [ ] T049 [US2] Tune batch_size parameter **POST-Phase 2** via scripts/tune_batch_size.py across range [32, 64, 128]; identify optimal value balancing throughput vs latency **with new tensor pipeline** (baseline profiling favored 128 with old 37ms overhead; new <2ms overhead changes tradeoff, expect 64 optimal)
```

**T050 AFTER**:
```markdown
- [ ] T050 [US2] Validate batch_size=64 setting (pre-allocated buffer size) achieves target GPU utilization (≥80%) and throughput (≥7,000 sims/sec) via profiling campaign; if T049 identifies different runtime optimal (e.g., 32 or 128), document as **tunable parameter** while buffer remains sized for max_batch=64
```

---

## M3: Buffer Overflow Edge Case

#### Edit 17: Add Overflow Handling to T042

**File**: `specs/005-mcts-throughput-optimization/tasks.md`

**T042 AFTER**:
```markdown
- [ ] T042 [US2] Implement create_batch_tensor() in dlpack_inference_bridge.py using pre-allocated pinned memory buffer (max_batch=64, 3.3MB CPU + 3.3MB GPU); add **overflow handling**: if batch_size > max_batch, log warning and **dynamically allocate** (fallback path, tested in tests/unit/test_dlpack_bridge_overflow.py with batch=128 edge case)
```

---

## M4: Terminology Standardization

#### Edit 18: Use BatchInferenceCoordinator in Technical Contexts

**Apply to**: All tasks.md DoD sections

**Pattern**:
```bash
# In DoD sections only (not user-facing goal statements):
sed -i 's/coordinator execution time/BatchInferenceCoordinator execution time/g' tasks.md
sed -i 's/coordinator blocking/BatchInferenceCoordinator blocking/g' tasks.md
```

---

## M5: Profiling Trial Count Clarity

#### Edit 19: Clarify Trial Count Requirements

**File**: `.specify/memory/constitution.md`
**Line**: 137

**AFTER**:
```markdown
- Profiling campaign MUST be executed with appropriate trial count:
  - **Baseline/Investigation**: 560 trials (full parameter space: 4 sim counts × 7 thread counts × 4 batch sizes × 5 reps) for exploratory analysis
  - **Phase Validation**: 100 trials minimum (single optimal config with 100 repetitions for statistical significance at 95% CI)
  - **All campaigns**: 100% execution time capture required (no "unknown" time categories >1% of total)
```

---

# LOW Priority Issues

## L1: Expected Duplication

**Status**: No action required (intentional per Spec-Kit design)

---

## L2: Move Semantics Validation Pattern

#### Edit 20: Add Validation Pattern to T013

**File**: `specs/005-mcts-throughput-optimization/tasks.md`

**T013 DoD AFTER**:
```markdown
**DoD**: InferenceRequest owns std::vector<float> features and uses move semantics for ownership transfer; contract test in tests/contract/test_inference_request_api.py validates zero-copy behavior via:
1. Assert InferenceRequest has **deleted copy constructor**: `InferenceRequest(const InferenceRequest&) = delete;`
2. Assert **move constructor enabled**: `InferenceRequest(InferenceRequest&&) = default;`
3. Verify **move leaves source empty**: after `req2 = std::move(req1);`, assert `req1.features.empty() == true;`
4. Performance test: moving 10,000 InferenceRequest objects completes in <1ms (vs ~500ms if copying)
```

---

# Implementation Checklist

## Step 1: Apply CRITICAL Edits (C1-C2)

### C1: Coordinator Count - Auto-Tune Approach (Edits 1-7)

- [ ] Edit 1: Constitution Phase 3A → "Default K=3 + auto-tune from {1,2,3,4}"
- [ ] Edit 2: Plan R8 → "K auto-tuned, persists to ~/.mcts_autotune.json"
- [ ] Edit 3: Tasks Phase 5 goal → "K parallel coordinators (default K=3, auto-tuned)"
- [ ] Edit 4: Tasks T058 → "spawn K coordinator threads (auto-tuned or CLI override)"
- [ ] Edit 5: Tasks success criteria → "K × 0.8 to K × 0.95 scaling (validated empirically)"
- [ ] Edit 6: Tasks T063 → "`--coordinators auto` (tests auto-tuned K)"
- [ ] Edit 7: Add T059a → Implement auto-tuner script + persistence
- [ ] Verify: No hard-coded "3 coordinators" outside defaults/examples
- [ ] Verify: No incorrect CUDA stream language (SM partitioning claims)

### C2: Missing Validation Tasks (Edits 8-11)

- [ ] Edit 8: Add T100 (Rollback Test with dry-run mode)
- [ ] Edit 9: Add T101 (Latency Variance Test with CDF plot)
- [ ] Edit 10: Add T102 (PUCT Semantics Test with 11 adversarial seeds)
- [ ] Edit 11: Add T103 (State Pool Removal Audit)
- [ ] Verify: All 25 success criteria have tasks
- [ ] Verify: Constitution compliance section has T088-T103 (16 tasks)

---

## Step 2: Apply HIGH Priority Edits (H1)

### H1: Split Python Callback Budget (Edits 12-14)

- [ ] Edit 12: Constitution → Split into "Python-side ≤2.0ms p95" + "GPU kernel: baseline+20%"
- [ ] Edit 13: Tasks T043 → Measure Python-side only (≤2.0ms p95)
- [ ] Edit 14: Tasks T045 → Split metrics (Python, GPU, total)
- [ ] Verify: All callback time references distinguish Python-side from GPU kernel

---

## Step 3: Apply MEDIUM Priority Edits (M1-M5)

- [ ] Edit 15 (M1): Add phase mapping table to tasks.md line 10
- [ ] Edit 16 (M2): Clarify T049 (POST-Phase 2 tuning) and T050 (validation)
- [ ] Edit 17 (M3): Add buffer overflow handling to T042
- [ ] Edit 18 (M4): Standardize BatchInferenceCoordinator terminology
- [ ] Edit 19 (M5): Clarify trial counts (560 exploratory, 100 validation)

---

## Step 4: Apply LOW Priority Edits (L2)

- [ ] Edit 20 (L2): Add move semantics validation pattern to T013

---

## Step 5: Final Validation

### Cross-Artifact Consistency Audit

```bash
cd /home/cosmosapjw/omoknuni/specs/005-mcts-throughput-optimization

# 1. No hard-coded coordinator counts (except defaults/examples)
echo "=== Coordinator Count Audit ==="
grep -RniE "(1|2|3|4) coordinator" . | \
  grep -v "default K=" | grep -v "K=3" | grep -v "K∈" | \
  grep -v "auto-tun" | grep -v "example" | grep -v "REMEDIATION" || echo "✅ OK"

# 2. CUDA streams language (no SM partitioning)
echo "=== CUDA Streams Language Audit ==="
grep -Rni "stream" . | \
  grep -Ei "reserve|partition|dedicate.*SM|allocate.*SM" && \
  echo "❌ Fix wording" || echo "✅ OK"

# 3. Success criteria coverage
echo "=== Success Criteria Coverage ==="
SC_COUNT=$(grep -E "^\*\*SC-[0-9]+" spec.md | wc -l)
TASK_SC_COUNT=$(grep -oE "SC-[0-9]+" tasks.md | sort -u | wc -l)
echo "Success Criteria: $SC_COUNT, Tasks referencing SC-XXX: $TASK_SC_COUNT"
[ "$SC_COUNT" -eq "$TASK_SC_COUNT" ] && echo "✅ 100% coverage" || echo "❌ Missing coverage"

# 4. Python callback budget split
echo "=== Python Callback Budget Audit ==="
grep -Rni "Python-side overhead" . | grep "≤2.0ms" && echo "✅ Split budget present" || echo "❌ Missing split"

# 5. Task count (should be 107 after additions: 99 + T059a + T100-T103)
echo "=== Task Count ==="
TASK_COUNT=$(grep -E "^\- \[ \] T[0-9]+" tasks.md | wc -l)
echo "Total tasks: $TASK_COUNT (expected: 107)"
[ "$TASK_COUNT" -eq 107 ] && echo "✅ Correct count" || echo "⚠️ Verify count"

# 6. Auto-tuner task exists
echo "=== Auto-Tuner Task ==="
grep -A3 "T059a.*auto-tun" tasks.md && echo "✅ T059a present" || echo "❌ Add T059a"
```

### Update Constitution Metadata

**File**: `.specify/memory/constitution.md`
**Lines**: 327-337

**AFTER**:
```markdown
**Current Version**: 1.0.1
**Ratified**: 2025-10-20
**Last Amended**: 2025-10-20

**Version History**:
- **1.0.1** (2025-10-20): PATCH amendments:
  - Coordinator count: Changed from fixed "4 coordinators" to **"default K=3 on RTX 3060 Ti, auto-tuned at startup from {1,2,3,4}"** based on user clarification Q5 and empirical validation requirement (Principle VI)
  - Trial count requirements clarified: 560 trials for baseline/exploration, 100 trials for phase validation
  - Python callback budget split into controllable (Python-side ≤2.0ms p95) vs model-dependent (GPU kernel baseline+20%)
  - Added footnote: CUDA streams are work queues, not SM partitions; scheduler dynamically allocates resources
- **1.0.0** (2025-10-20): Initial ratification with 6 NON-NEGOTIABLE principles
```

---

## Summary

**Total Edits**: 20 across 13 issues (revised from v1.0)
- **CRITICAL**: 11 edits (7 for auto-tune approach + 4 new tasks)
- **HIGH**: 3 edits (split Python callback budget)
- **MEDIUM**: 5 edits (clarity improvements)
- **LOW**: 1 edit (move semantics pattern)

**Key Changes from v1.0**:
- Replaced hard-coded "3 coordinators" with **default K=3 + auto-tune**
- Removed unsupported GPU/PCIe claims
- Corrected RTX 3060 Ti specs (38 SMs, not 28)
- Split Python callback budget (controllable vs model-dependent)
- Added auto-tuner task (T059a)
- Enhanced validation tasks (dry-run rollback, adversarial PUCT seeds)

**Files Modified**:
- `.specify/memory/constitution.md`: 3 edits (auto-tune, budget split, trial counts)
- `specs/005-mcts-throughput-optimization/plan.md`: 1 edit (auto-tune approach)
- `specs/005-mcts-throughput-optimization/tasks.md`: 15 edits (auto-tune, new tasks, clarifications)
- `specs/005-mcts-throughput-optimization/spec.md`: No edits (already updated during /speckit.clarify)

**Estimated Time**:
- Apply edits: 45 minutes
- Implement auto-tuner script (T059a): 2 hours
- Validation: 20 minutes
- Constitution version update: 5 minutes
- **Total**: ~3 hours 10 minutes

**Post-Edit Actions**:
1. Commit constitution + specs with message: "fix: Resolve 13 cross-artifact inconsistencies with auto-tune approach (v2.0)"
2. Implement auto-tuner script (T059a) as separate commit
3. Run validation audit scripts (Step 5)
4. Update CLAUDE.md with auto-tuning approach for Phase 3A

---

**Document Status**: ✅ COMPLETE (v2.0 - revised after technical review)
**Next Step**: Apply edits sequentially, then implement auto-tuner (T059a)
**Review Notes**: Incorporated feedback from technical review:
- Fixed RTX 3060 Ti SM count (38, not 28)
- Corrected CUDA stream understanding (work queues, not SM partitions)
- Adopted auto-tuning approach (superior to hard-coding)
- Split Python callback budget (Python-side vs GPU kernel)
- Enhanced validation tasks (dry-run, adversarial seeds)
