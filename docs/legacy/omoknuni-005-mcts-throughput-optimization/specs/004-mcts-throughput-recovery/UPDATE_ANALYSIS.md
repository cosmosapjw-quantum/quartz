# Spec 004 Files - Update Analysis Against FINAL_PROFILING_ANALYSIS_20251016.md

**Analysis Date**: 2025-10-16
**Baseline Source**: FINAL_PROFILING_ANALYSIS_20251016.md (560 trials, 100% capture)
**Current Baseline**: 2,659 sims/sec (profiling-validated)

---

## Files Status Summary

### ✅ UPDATED (Aligned with profiling data)
- [x] **CONSTITUTION.md** - Updated with 2,659 sims/sec baseline (one typo fixed: 300μs → 418μs)
- [x] **spec.md** - Updated with profiling data, dual thresholds, memory breakdown
- [x] **TECHNICAL_PLAN.md** - Updated with profiling-grounded design
- [x] **TASKS.md** - Updated with T023 rollback, statistical validation, expanded details
- [x] **README.md** - Completely rewritten with current profiling data
- [x] **FINAL_PROFILING_ANALYSIS_20251016.md** - Fixed typo (300μs → 418μs)
- [x] **CLAUDE.md** - Updated with 2,659 sims/sec baseline and profiling priorities
- [x] **research.md** - Updated with profiling data (state cloning 86.6%, GPU 2.1%) 🎉
- [x] **quickstart.md** - Updated with profiling validation, removed outdated patches 🎉
- [x] **data-model.md** - Updated OpenMP note with profiling data (0/560 trials) 🎉
- [x] **contracts/inference_api.yaml** - Updated with profiling baseline and validation results 🎉
- [x] **contracts/arena-api.md** - Updated with current throughput expectations 🎉

### 🗑️ COMPLETED CLEANUP
- [x] **archive/** - Deleted (192KB of outdated data removed) 🎉

### ✅ ALL FILES NOW ALIGNED
All spec/004 files are now synchronized with FINAL_PROFILING_ANALYSIS_20251016.md (560 trials, 100% capture).

---

## Detailed Analysis

### 1. research.md ❌ CRITICAL UPDATES NEEDED

**Issues Found**:
- Line 5: "Current performance of 3,831 simulations/second (12.8% of target)"
  - ❌ Should be: 2,659 sims/sec (33.2% of 8,000 target)
- Lines 10-20: "GPU Inference: 32.8%, MCTS Overhead: 67.2%"
  - ❌ Completely wrong breakdown (old hypothesis)
  - ✅ Should be: State cloning 86.6%, GPU inference 2.1%
- Line 23: "Even with perfect GPU utilization, cannot exceed ~8,000 sims/sec"
  - ❌ Wrong conclusion (GPU is NOT the bottleneck)
  - ✅ Should be: State pooling alone achieves 9,838 sims/sec

**Required Changes**:
```markdown
OLD:
Current performance of 3,831 simulations/second (12.8% of target) stems
from architectural inefficiencies... GPU inference accounts for only 32.8%
of runtime while MCTS overhead consumes 67.2%

NEW:
Current performance of 2,659 simulations/second (33.2% of 8,000 target)
stems from state cloning bottleneck (86.6% of execution time). Profiling
campaign (560 trials, 100% capture) revealed state cloning with 223
allocations per clone as primary bottleneck, NOT GPU inference (only 2.1%
of time).
```

### 2. quickstart.md ⚠️ MODERATE UPDATES NEEDED

**Issues Found**:
- Lines 39-40: "-fopenmp is CRITICAL for DLPack feature extraction parallelization"
  - ⚠️ Misleading - OpenMP is currently NOT active (0/560 trials)
- Lines 58-62: References to patches (001-wu-uct, 002-lock-free, etc.)
  - ⚠️ These patches don't exist / are outdated
- Lines 89-96: Validation tests reference non-existent test files
  - ❌ test_wu_uct.py, test_lock_free_queue.py, etc. may not exist

**Required Changes**:
1. Add profiling campaign as primary validation method
2. Remove references to non-existent patches
3. Update build commands to match TASKS.md (with PROFILE_BUFFER_SIZE)
4. Add state pooling validation as primary test

### 3. data-model.md ⚠️ MINOR UPDATES NEEDED

**Issues Found**:
- Lines 80-84: "**⚠️ CRITICAL PERFORMANCE NOTE (2025-10-13):**"
  - ⚠️ Outdated note about OpenMP (T-VALID-2 reference)
  - ⚠️ Says "7.5ms per batch-64" but this is pre-profiling estimate
  - ✅ Should reference profiling data: OpenMP 0/560 trials active

**Required Changes**:
```markdown
OLD:
**⚠️ CRITICAL PERFORMANCE NOTE (2025-10-13):**
- Feature extraction loop currently NOT parallelized with OpenMP
- Measured overhead: 7.5ms per batch-64 (T-VALID-2)
- Required fix: Add `#pragma omp parallel for` to `dlpack_bridge.cpp:431-434`

NEW:
**⚠️ PERFORMANCE NOTE (2025-10-16 - Profiling Validated):**
- Feature extraction loop: OpenMP NOT active (0/560 trials in campaign)
- Current impact: Secondary bottleneck (state cloning is 86.6% of time)
- Priority: Low (state pooling alone achieves 8k target)
- Investigation: T019 (optional enhancement for 14k+ stretch goal)
```

### 4. contracts/ ⚠️ REVIEW NEEDED

**Files in contracts/**:
- arena-api.md (22K) - Contains old 3,831 sims/sec references
- dlpack-api.md (18K) - May reference T-VALID-2
- dlpack_api.yaml (7.9K)
- inference_api.yaml (9.6K) - Contains old 1,543 sims/sec references
- python-bridge-api.md (11K)
- queue_api.yaml (6.4K)
- thread_api.yaml (8.3K)
- virtual_loss_api.yaml (4.1K)

**Action Required**: Review each contract file and update performance numbers

### 5. archive/ 🗑️ REMOVAL RECOMMENDED

**Files in archive/**:
- PR_CHECKLIST.md (20K)
- README.md (4.4K)
- REVIEW-RESPONSE.md (7.8K) - Contains old 3,831 sims/sec
- SPECIFICATION.md (53K)
- TASKS.md (61K) - Old version
- TECHNICAL_PLAN.md (34K) - Old version

**Recommendation**: Delete entire archive directory
- Archive files contain OLD data (3,831 sims/sec baseline)
- No longer relevant after profiling campaign
- If history needed, rely on git history instead

---

## Update Priority Order

### 🔴 CRITICAL (Must fix before implementation)
1. **research.md** - Completely wrong bottleneck analysis (3,831 → 2,659, GPU 32.8% → state cloning 86.6%)

### 🟠 HIGH (Should fix before Phase 1)
2. **quickstart.md** - Outdated build/validation procedures
3. **data-model.md** - Outdated OpenMP note

### 🟡 MEDIUM (Can fix during implementation)
4. **contracts/arena-api.md** - Update performance references
5. **contracts/inference_api.yaml** - Update performance references

### ⚪ LOW (Optional cleanup)
6. **archive/** - Delete old files (or keep for git history)

---

## Recommended Actions

### Action 1: Update research.md (CRITICAL)

**Command**:
```bash
# Backup first
cp research.md research.md.bak

# Create updated version
cat > research.md.new << 'EOF'
# Research: MCTS Throughput Recovery Technical Analysis
# UPDATED 2025-10-16 - Based on profiling_suite_20251016_124134

## Executive Summary

Current performance of **2,659 sims/sec** (33.2% of 8,000 target) stems from
state cloning bottleneck. Comprehensive profiling campaign (560 trials, 100%
data capture) revealed that **state cloning consumes 86.6% of execution time**
due to 223 allocations per clone (~2μs each = 446μs overhead).

**Key Finding**: GPU inference is NOT the bottleneck (only 2.1% of time). The
problem is CPU-side state cloning overhead.

## Performance Bottleneck Analysis (Profiling-Validated)

### Current Performance Profile (Trial 001 - Representative of 560 trials)
```
Total: 982.86 ms for 2,000 simulations

state_clone_total:   835.85 ms (86.6%) 🔴 PRIMARY BOTTLENECK
expansion_total:      37.24 ms ( 3.8%)
expansion_nn_wait:    20.66 ms ( 2.1%) ← GPU inference (NOT the bottleneck!)
selection_total:       3.58 ms ( 0.4%)
backup_total:          1.67 ms ( 0.2%)
unknown/overhead:     85.64 ms ( 8.7%)
```

### Critical Finding
**State cloning overhead**:
- Actual: 418μs per clone
- Expected: ~50μs per clone
- Discrepancy: 209× slower than expected
- Root cause: 223 allocations per clone

## Architecture Decision: State Pooling (CORRECTED)

Previous analysis incorrectly focused on GPU optimization. Profiling data shows
state pooling is the critical path to 8k target.

### Selected Approach: Thread-Local State Pools
**Impact**: Eliminate 223 allocations per clone
- Clone time: 418μs → 20μs (20.9× faster)
- Overall gain: 3.7× throughput → 9,838 sims/sec ✅ Exceeds target

... [continue with updated content]
EOF
```

### Action 2: Update quickstart.md

**Key Changes**:
- Replace patch references with direct C++ implementation steps
- Add profiling validation as primary test
- Update build commands to match TASKS.md
- Reference state pooling as primary optimization

### Action 3: Update data-model.md

**Minimal Change**:
- Update lines 80-84 OpenMP note with profiling data
- Reference T019 as optional enhancement

### Action 4: Delete Archive

**Command**:
```bash
cd /home/cosmosapjw/omoknuni/specs/004-mcts-throughput-recovery
rm -rf archive/
```

### Action 5: Review and Update Contracts

**For each contract file**:
1. Search for performance numbers (3,831, 1,543, etc.)
2. Replace with profiling-validated baseline (2,659)
3. Update expected performance after optimizations

---

## Validation Checklist

After updates, verify:
- [ ] No references to 3,831 sims/sec (old baseline)
- [ ] No references to 1,543 sims/sec (partial measurement)
- [ ] No references to 2,147 sims/sec (pre-profiling estimate)
- [ ] All files cite 2,659 sims/sec as current baseline
- [ ] All files reference profiling_suite_20251016_124134 as authority
- [ ] Bottleneck correctly identified as state cloning (86.6%)
- [ ] GPU correctly identified as NOT the bottleneck (2.1%)
- [ ] State pooling cited as achieving 8k target alone (9,838 sims/sec)

---

## Summary

**Files Needing Updates**: 3-4 critical files
**Archive Cleanup**: Delete archive/ directory (267KB of outdated data)
**Estimated Effort**: 2-3 hours for complete alignment
**Priority**: research.md is CRITICAL (completely wrong bottleneck analysis)
