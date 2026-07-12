# Profiling Tools Guide

Complete guide to profiling and performance validation tools for MCTS optimization.

---

## Overview

Three main tools for comprehensive performance analysis:

1. **Wall-Clock Validation** - Ground-truth performance measurement
2. **Profiling Campaign** - Parameter sweep with full metrics
3. **Results Analysis** - Compare and analyze campaigns

All tools leverage **UnifiedProfiler** for comprehensive coverage:
- 295 C++ metrics (state cloning, OpenMP, thread idle, CAS, mutex)
- Python profiling (GIL, inference, thread coordination)
- Automated bottleneck detection
- Chrome Trace timeline visualization

---

## Tool 1: Wall-Clock Validation

**Purpose:** Measure actual wall-clock performance without profiling overhead.

**Features:**
- Pure wall-clock timing (profiling disabled for accuracy)
- Statistical analysis (mean, median, stdev, CV)
- Target comparison (vs 8,000 sims/sec goal)
- Optional profiling overhead measurement

**Usage:**

```bash
# Quick test (100 sims, 3 runs)
python scripts/wall_clock_validation.py --quick

# Full validation (1000 sims, 5 runs)
python scripts/wall_clock_validation.py --simulations 1000 --runs 5

# With profiling overhead comparison
python scripts/wall_clock_validation.py --compare-profiling

# Custom configuration
python scripts/wall_clock_validation.py \
    --simulations 1600 \
    --runs 10 \
    --output validation_results.json
```

**Output:**
```
📈 Wall-Clock Time (5 runs):
   Mean:   0.489s
   Median: 0.490s
   StdDev: 0.003s
   CV:     0.62% ✅ Low variability

🚀 Throughput (5 runs):
   Mean:   2045.3 sims/sec
   Median: 2040.8 sims/sec
   StdDev: 12.7 sims/sec

🎯 Target Analysis:
   Target:  8000 sims/sec
   Current: 2045.3 sims/sec
   Progress: 25.6% of target
```

**When to use:**
- Establish performance baselines
- Validate optimizations (before/after)
- Measure profiling overhead
- Quick sanity checks

---

## Tool 2: Profiling Campaign

**Purpose:** Parameter sweep with comprehensive profiling using UnifiedProfiler.

**Features:**
- **Uses UnifiedProfiler** for full metrics coverage (295 C++ + Python)
- Automated bottleneck detection per trial
- Parameter sweeps (simulations × threads × batch sizes)
- Chrome Trace exports for timeline analysis
- CSV + JSON export for post-analysis

**Usage:**

```bash
# Quick test (4 trials: 2 sims × 2 threads × 1 batch)
python scripts/profiling_campaign.py --quick-test --yes

# Full campaign (default ranges)
python scripts/profiling_campaign.py --output results/campaign_001

# Custom parameter sweep
python scripts/profiling_campaign.py \
    --simulations 400,800,1600 \
    --threads 2,4,8,12 \
    --batch-sizes 32,64,128 \
    --output results/sweep_20251015

# Skip confirmation prompt
python scripts/profiling_campaign.py --quick-test --yes
```

**Output Structure:**
```
results/campaign_001/
├── campaign_summary.json    # All results + metadata
├── results.csv              # Spreadsheet-friendly format
├── trial_001/
│   ├── cpp_profiling.json   # 295 C++ metrics
│   ├── cpp_trace.json       # Chrome Trace format
│   ├── cpp_report.md        # Human-readable report
│   ├── python_profiling.json # Python metrics (GIL, inference)
│   └── result.json          # Trial-specific summary
├── trial_002/
│   └── ...
└── trial_N/
    └── ...
```

**Comprehensive Metrics Per Trial:**
- **Throughput:** sims/sec, wall-clock time
- **State Cloning:** Count and ratio per simulation
- **OpenMP:** Parallelization success/failure
- **Thread Idle:** Idle time in milliseconds
- **CAS Retries:** Contention indicators
- **Tree Stats:** Node count, memory usage
- **Python:** GIL contention, inference overhead

**When to use:**
- Find optimal parameter configuration
- Identify bottlenecks across parameter ranges
- Compare threading/batching strategies
- Deep-dive debugging with Chrome Trace

---

## Tool 3: Results Analysis

**Purpose:** Analyze and compare profiling campaign results.

**Features:**
- Load multiple campaigns for comparison
- Thread scaling efficiency analysis
- Batch size impact analysis
- Bottleneck prioritization
- Detailed statistics per configuration

**Usage:**

```bash
# Analyze single campaign
python scripts/analyze_profiling_results.py results/campaign_001/campaign_summary.json

# Compare multiple campaigns
python scripts/analyze_profiling_results.py \
    results/campaign_001/campaign_summary.json \
    results/campaign_002/campaign_summary.json

# With detailed bottleneck analysis
python scripts/analyze_profiling_results.py --detailed results/*/campaign_summary.json

# Using glob patterns
python scripts/analyze_profiling_results.py results/campaign_*/campaign_summary.json
```

**Output:**
```
📊 OVERALL STATISTICS
⏱️  Wall-Clock Time:
   Mean:   0.073s
   StdDev: 0.032s

🚀 Throughput:
   Mean:   2107.4 sims/sec
   StdDev: 235.9 sims/sec

📈 THREAD SCALING ANALYSIS
   1 threads:  2225.8 ± 332.2 sims/sec
   4 threads:  1988.9 ±  22.9 sims/sec
   Scaling Efficiency: 22.3% (❌ Poor - indicates bottleneck)

🔍 BOTTLENECK IDENTIFICATION
   🔴 State Cloning: 2.3× per simulation
      → HIGH PRIORITY: Implement state pooling

   🔴 OpenMP: 3/4 trials NOT parallelizing
      → HIGH PRIORITY: Fix feature extraction

   ⚠️  Thread Idle: 45.2ms avg
      → Reduce coordination overhead
```

**When to use:**
- Post-campaign analysis
- Compare optimization attempts
- Prioritize bottleneck fixes
- Track progress over time

---

## Unified Profiler Coverage

All tools leverage **UnifiedProfiler** for comprehensive metrics:

### C++ Metrics (295 total)

**Base Metrics (240):**
- Selection: PUCT, node traversal, policy evaluation
- Expansion: Node allocation, feature extraction
- Backup: Value propagation, visit updates
- Infrastructure: Memory, queue ops, threads

**Bottleneck Metrics (55):**
- **State Cloning (15):** Pre/post selection, expansion, simulation, backup
- **OpenMP (10):** Thread counts, parallel execution, work distribution
- **Thread Idle (10):** Node contention, queue waits, VL blocking
- **CAS Retries (10):** VL retries, visit count updates
- **Mutex Waits (10):** Queue mutex, tree mutex, critical sections

### Python Metrics

- **GIL Profiling:** Contention tracking, hold times
- **Inference:** Callback overhead, batching efficiency
- **Thread Coordination:** Python-side coordination costs

### Export Formats

- **JSON:** Machine-readable, for custom analysis
- **Chrome Trace:** Timeline visualization (chrome://tracing)
- **Markdown:** Human-readable reports
- **CSV:** Spreadsheet-friendly (Excel, pandas)

---

## Workflow Examples

### Example 1: Initial Performance Baseline

```bash
# Step 1: Quick wall-clock baseline
python scripts/wall_clock_validation.py --quick
# → Throughput: 2038 sims/sec (25.5% of 8k target)

# Step 2: Quick profiling campaign to identify bottlenecks
python scripts/profiling_campaign.py --quick-test --yes
# → Detects: OpenMP not parallelizing, state cloning 2.3×

# Step 3: Analyze results
python scripts/analyze_profiling_results.py results/*/campaign_summary.json
# → Priority #1: Fix OpenMP (dlpack_bridge.cpp:431-434)
# → Priority #2: State pooling (review.txt:37-54)
```

### Example 2: Optimization Validation

```bash
# Before optimization
python scripts/wall_clock_validation.py --runs 10 --output baseline.json

# [Make code changes]

# After optimization
python scripts/wall_clock_validation.py --runs 10 --output optimized.json

# Compare (manually or with custom script)
# baseline:   2038 ± 45 sims/sec
# optimized:  3500 ± 52 sims/sec
# Improvement: +71.7% 🎉
```

### Example 3: Parameter Optimization

```bash
# Full parameter sweep
python scripts/profiling_campaign.py \
    --simulations 200,400,800,1600,3200 \
    --threads 1,2,4,8,12 \
    --batch-sizes 16,32,64,128 \
    --output results/full_sweep
# → 100 trials (5×5×4)

# Analyze to find optimal config
python scripts/analyze_profiling_results.py --detailed results/full_sweep/campaign_summary.json
# → Best: 8 threads, batch 64, 1600 sims → 4200 sims/sec
```

### Example 4: Bottleneck Deep-Dive

```bash
# Run campaign with profiling
python scripts/profiling_campaign.py \
    --simulations 1600 \
    --threads 8 \
    --batch-sizes 64 \
    --output results/deep_dive

# Check bottleneck metrics in results/deep_dive/trial_001/
cat results/deep_dive/trial_001/cpp_report.md
# → State clones: 3700 (2.3× per sim)
# → Thread idle: 340ms (21% of time)

# Visualize timeline
# Open results/deep_dive/trial_001/cpp_trace.json in chrome://tracing
# → Identify exact timing bottlenecks
```

---

## Key Improvements Over Previous Tools

### Before Refactoring ❌
- Only C++ profiler (missing Python metrics)
- No bottleneck analysis
- Manual profiling setup
- Code duplication across tools
- ~30% bottleneck coverage

### After Refactoring ✅
- **Uses UnifiedProfiler** for all 295 metrics
- **Python profiling** included (GIL, inference, threads)
- **Automated bottleneck detection**
- **Zero code duplication**
- **100% bottleneck coverage**
- **Chrome Trace** for timeline analysis
- **Actionable recommendations** with code references

---

## Tips and Best Practices

### Statistical Significance
- Run ≥5 iterations for stable means
- Check CV (coefficient of variation) < 5% for reliable results
- Use `--warmup` to eliminate cold-start effects

### Parameter Sweeps
- Start with `--quick-test` to verify setup
- Use coarse ranges first, then refine
- Expect O(N³) trials for full factorial sweep

### Bottleneck Priority
1. **State Cloning** - Highest impact (2-3× waste)
2. **OpenMP** - Blocks parallelization gains
3. **Thread Coordination** - 60% idle time
4. **CAS Contention** - Lock-free alternatives

### Chrome Trace Usage
1. Open `chrome://tracing` in Chrome/Chromium
2. Load `cpp_trace.json` from trial directory
3. Look for:
   - Long bars = slow operations
   - Gaps = idle time / waiting
   - Overlaps = parallelism (or lack thereof)

---

## Troubleshooting

### "No module named 'src.profiling'"
```bash
# Ensure you've installed the package
pip install -e .
```

### "Import error: mcts_py"
```bash
# Rebuild C++ extensions
export CXXFLAGS="-O3 -march=znver3 -fopenmp -DPROFILE_LEVEL_VALUE=3"
export CFLAGS="-O3 -march=znver3 -fopenmp"
pip install -e . --force-reinstall --no-deps
```

### High variability (CV > 10%)
- Increase `--runs` for more samples
- Check for background processes
- Run with `--warmup` to eliminate cold starts

### Missing profiling data
- Verify PROFILE_LEVEL_VALUE=3 in build flags
- Check trial directory for all export files
- Look for errors in campaign output

---

## Summary

| Tool | Purpose | Overhead | Metrics | When to Use |
|------|---------|----------|---------|-------------|
| **wall_clock_validation.py** | Ground truth timing | None | Basic | Baselines, validation |
| **profiling_campaign.py** | Comprehensive profiling | Low (~5%) | 295 C++ + Python | Bottleneck hunting |
| **analyze_profiling_results.py** | Post-analysis | None | Statistical | Compare campaigns |

**All tools now use UnifiedProfiler** → Full metrics coverage, automated bottleneck detection, actionable insights.

---

**Next Steps:**
1. Run baseline: `python scripts/wall_clock_validation.py --quick`
2. Profile bottlenecks: `python scripts/profiling_campaign.py --quick-test --yes`
3. Fix top bottleneck (likely OpenMP or state cloning)
4. Re-run and measure improvement
5. Iterate until 8,000 sims/sec target achieved

**Target:** 8,000 simulations/second
**Current:** ~2,000 sims/sec (25% of target)
**Gap:** Requires 4× improvement → Focus on state pooling + OpenMP fix
