# Rollback Procedure: Spec 004 Optimizations

**Version**: 1.0
**Last Updated**: 2025-10-12
**Purpose**: Emergency rollback procedures if Spec 004 optimizations cause regressions

---

## Quick Rollback (Environment Variables)

If optimizations cause issues, disable them via environment variables:

```bash
# Disable specific optimizations
export MCTS_DISABLE_LOCK_FREE_QUEUE=1       # Revert T006/T006b (lock-free queue)
export MCTS_DISABLE_CONDITION_VARS=1        # Revert T006c (condition variables)
export MCTS_DISABLE_DLPACK=1                # Revert T007 (DLPack zero-copy)
export MCTS_DISABLE_FP16=1                  # Revert T008f (FP16 mixed precision)
export MCTS_DISABLE_THREAD_AFFINITY=1       # Revert T004 (thread pinning)
export MCTS_DISABLE_WU_UCT=1                # Revert T001 (WU-UCT virtual loss)
export MCTS_DISABLE_BUSY_EDGE_MASKING=1     # Revert T002 (busy-edge masking)

# Verify rollback works
python scripts/benchmark_throughput.py \
  --game gomoku \
  --threads 8 \
  --simulations 10000 \
  --verify-rollback

# Expected: Behavior reverts to pre-optimization state
```

---

## Full Rollback (Git Revert)

Revert to baseline commit (Spec 003 complete):

```bash
# Find baseline commit
git log --oneline --all | grep -i "baseline\|spec.*003"

# Checkout baseline (example commit hash)
git checkout <baseline-commit-hash>

# Alternative: Create rollback branch
git checkout -b rollback-spec-004
git revert <first-spec-004-commit>..<last-spec-004-commit>

# Rebuild C++ extensions
python -m pip install -e . --force-reinstall --no-deps

# Verify baseline performance restored
python scripts/verify_baseline.py
# Expected: 3,831 ±10% sims/sec (3,448-4,214 range)
```

---

## Partial Rollback (Specific Optimization)

Revert individual optimizations while keeping others:

### Example: Revert T006c (Condition Variables)

```bash
# Find commit hash for T006c
git log --oneline --all | grep -i "T006c\|condition.*variable"

# Revert specific commit
git revert <commit-2253a97>

# Rebuild
python -m pip install -e . --force-reinstall --no-deps

# Verify
python scripts/benchmark_throughput.py --game gomoku --threads 8
# Expected: Throughput without T006c gains (polling overhead returns)
```

### Example: Revert T008f (FP16 Mixed Precision)

```bash
# Disable FP16 via config (no code change needed)
# Edit src/core/config.py:
# use_mixed_precision = False

# Or via environment variable
export MCTS_DISABLE_FP16=1

# Verify
python scripts/validate_fp16_inference.py --model models/gomoku_10M.pth
# Expected: FP32 inference only (no FP16 speedup)
```

---

## Validation After Rollback

Always validate that rollback succeeded:

### 1. Quality Tests (Non-Negotiable)

```bash
# Policy agreement test
python scripts/test_policy_agreement.py \
  --baseline models/baseline_v003.pth \
  --candidate models/current.pth \
  --threshold 0.95

# Expected: ≥95% agreement (no quality regression)

# Win rate test
python scripts/test_win_rate.py \
  --baseline models/baseline_v003.pth \
  --candidate models/current.pth \
  --games 1000 \
  --threshold 0.995

# Expected: ≥99.5% win rate (no strength loss)
```

### 2. Performance Tests

```bash
# Throughput benchmark
python scripts/benchmark_throughput.py \
  --game gomoku \
  --threads 8 \
  --simulations 10000 \
  --runs 5

# Expected: Known baseline performance (3,831 sims/sec for full rollback)

# Memory stability
python tests/soak/test_memory_stability.py --duration 3600

# Expected: No memory leaks, RSS growth <10MB/hour
```

### 3. Stability Tests

```bash
# Thread safety (TSan)
cmake -B build -DSANITIZE_THREAD=ON
cmake --build build --parallel 12
python -m pytest tests/ -v

# Expected: Zero TSan warnings

# Memory safety (Valgrind)
valgrind --leak-check=full --error-exitcode=1 \
  python -m pytest tests/unit/ tests/integration/ -v

# Expected: Zero memory leaks
```

---

## Rollback Decision Tree

```
Performance regression detected
│
├─ Throughput <20k sims/sec?
│  ├─ YES → Full rollback to baseline
│  └─ NO → Partial rollback (disable suspect optimization)
│
├─ Quality regression (win rate <99%)?
│  ├─ YES → IMMEDIATE full rollback (non-negotiable)
│  └─ NO → Investigate further
│
├─ Memory leak detected?
│  ├─ YES → Full rollback + investigate (use Valgrind)
│  └─ NO → Continue
│
└─ Thread safety violation (TSan)?
   ├─ YES → Full rollback + investigate (critical bug)
   └─ NO → Partial rollback acceptable
```

---

## Emergency Contact

If rollback fails or issues persist:

1. **Create GitHub Issue**: Tag as `critical` + `performance-regression`
2. **Include Diagnostics**:
   - Git commit hash
   - Performance benchmark results
   - Error logs / stack traces
   - Hardware configuration
3. **Rollback Status**: Document which rollback steps succeeded/failed

---

## Post-Rollback Action Items

After successful rollback:

1. **Document Root Cause**: Why did optimization fail?
2. **Add Regression Test**: Prevent future occurrences
3. **Update KPIs**: Adjust expectations if necessary
4. **Plan Re-Attempt**: Can optimization be salvaged with fixes?

---

**END OF ROLLBACK PROCEDURE**
