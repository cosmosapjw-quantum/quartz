# Enhanced Profiling System - Quick Reference

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Common Patterns](#common-patterns)
- [Macro Reference](#macro-reference)
- [Metric Categories](#metric-categories)
- [Export Formats](#export-formats)
- [Troubleshooting](#troubleshooting)

## Installation

### Compile with Profiling

```bash
# CMakeLists.txt or compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPROFILE_LEVEL=1")

# Levels:
# -DPROFILE_LEVEL=0  # Disabled (0% overhead)
# -DPROFILE_LEVEL=1  # Basic timers (<0.1% overhead)
# -DPROFILE_LEVEL=2  # + Hardware counters (<0.5% overhead)
# -DPROFILE_LEVEL=3  # + Memory tracking (<1.0% overhead)
```

### Include Headers

```cpp
#include "profiling/profiler.hpp"
#include "profiling/hardware_counters.hpp"  // Optional
#include "profiling/contention_tracker.hpp"  // Optional
#include "profiling/export.hpp"             // Optional

using namespace mcts::profiling;
```

## Basic Usage

### 1. Simple Timing

```cpp
void my_function() {
    PROFILE_SCOPE(ProfileMetric::SelectionTotal);
    // Your code here...
}
```

### 2. Nested Profiling

```cpp
void outer_function() {
    PROFILE_SCOPE(ProfileMetric::SelectionTotal);

    {
        PROFILE_SCOPE(ProfileMetric::SelectionPUCT);
        // PUCT computation...
    }

    {
        PROFILE_SCOPE(ProfileMetric::SelectionAtomicLoad);
        // Atomic loads...
    }
}
```

### 3. Counter Tracking

```cpp
if (node_expanding) {
    PROFILE_COUNTER(ProfileMetric::SelectionBusyEdgeSkip);
}

for (int i = 0; i < retry_count; ++i) {
    PROFILE_COUNTER(ProfileMetric::VirtualLossCASRetries);
}
```

### 4. Gauge Tracking

```cpp
// Set absolute value
PROFILE_GAUGE(ProfileMetric::QueueBatchSize, batch.size());
PROFILE_GAUGE(ProfileMetric::SelectionDepth, depth);

// Update by delta
PROFILE_UPDATE_GAUGE(ProfileMetric::QueuePendingDepth, +1);  // Increment
PROFILE_UPDATE_GAUGE(ProfileMetric::QueuePendingDepth, -1);  // Decrement
```

### 5. Complete Session

```cpp
// Start profiling
Profiler::instance().enable();
Profiler::instance().start_session("MyBenchmark");

// Run workload
for (int i = 0; i < 10000; ++i) {
    PROFILE_SCOPE(ProfileMetric::SelectionTotal);
    // ... work ...
}

// Stop and export
Profiler::instance().stop_session();
Profiler::instance().export_json("profile.json");
Profiler::instance().export_chrome_trace("trace.json");
```

## Common Patterns

### Pattern 1: Function-Level Profiling

```cpp
void select_leaf() {
    PROFILE_FUNCTION(ProfileMetric::SelectionTotal);
    // Entire function is profiled
}
```

### Pattern 2: Conditional Profiling

```cpp
void process_node() {
    #if PROFILE_LEVEL >= 2
        PROFILE_SCOPE(ProfileMetric::SelectionAtomicLoad);
    #endif
    // Code that should only be profiled at detailed level
}
```

### Pattern 3: High-Precision Timing

```cpp
void short_operation() {
    PROFILE_SCOPE_CYCLES(ProfileMetric::SelectionAVX2);
    // Uses RDTSC for sub-nanosecond precision
}
```

### Pattern 4: Manual Control

```cpp
ManualProfiler profiler;

profiler.start(ProfileMetric::ExpansionTotal);
// ... do work ...
profiler.stop();

// Restart later
profiler.start(ProfileMetric::ExpansionTotal);
// ... more work ...
profiler.stop();
```

### Pattern 5: Sampling Mode

```cpp
// Profile only 1 in 100 operations
Profiler::instance().set_sampling_rate(100);
Profiler::instance().enable();

for (int i = 0; i < 1000000; ++i) {
    PROFILE_SCOPE(ProfileMetric::SelectionTotal);
    // Only ~10,000 samples recorded
}
```

### Pattern 6: Hardware Counters

```cpp
Profiler::instance().enable_hardware_counters();

HardwareCounterReader hw;
if (hw.initialize()) {
    hw.start();

    // Run benchmark
    PROFILE_SCOPE(ProfileMetric::SelectionTotal);
    // ... work ...

    hw.stop();

    auto cycles = hw.read(HWCounterType::CPUCycles);
    auto cache_misses = hw.read(HWCounterType::CacheMisses);

    printf("Cycles: %lu, Cache Misses: %lu\n", cycles, cache_misses);
}
```

### Pattern 7: Contention Tracking

```cpp
TrackedAtomic<uint64_t> visit_count(0);

uint64_t expected = visit_count.load();
uint64_t desired = expected + 1;

while (!visit_count.compare_exchange_weak(
    expected, desired,
    std::memory_order_release,
    std::memory_order_acquire,
    ProfileMetric::VirtualLossCASSuccess
)) {
    PROFILE_COUNTER(ProfileMetric::VirtualLossCASRetries);
    expected = visit_count.load();
    desired = expected + 1;
}
```

### Pattern 8: Statistical Analysis

```cpp
// Get metrics
auto thread_metrics = Profiler::instance().get_all_thread_metrics();

// Aggregate
MetricAggregator aggregator;
auto metrics = aggregator.aggregate(thread_metrics);

// Analyze
for (const auto& metric_stats : metrics) {
    if (metric_stats.count > 0) {
        printf("%s: mean=%.2fns, p95=%.2fns, p99=%.2fns\n",
               metric_name(/* metric_id */),
               metric_stats.mean,
               metric_stats.p95,
               metric_stats.p99);
    }
}

// Detect bottlenecks
BottleneckDetector detector;
auto bottlenecks = detector.detect(metrics);
```

## Macro Reference

### Profiling Macros

| Macro | Description | Overhead |
|-------|-------------|----------|
| `PROFILE_SCOPE(metric)` | Profile current scope | ~20ns |
| `PROFILE_SCOPE_CYCLES(metric)` | High-precision cycle-based timing | ~50ns |
| `PROFILE_FUNCTION(metric)` | Profile entire function | ~20ns |
| `PROFILE_NAMED_SCOPE(name, metric)` | Named scope for visualization | ~20ns |
| `PROFILE_COUNTER(metric, value)` | Increment counter | ~5ns |
| `PROFILE_GAUGE(metric, value)` | Set gauge value | ~5ns |
| `PROFILE_UPDATE_GAUGE(metric, delta)` | Update gauge by delta | ~5ns |

### Conditional Macros

```cpp
#if PROFILE_LEVEL >= 1
    PROFILE_SCOPE(metric)
#else
    // Compiles to nothing
#endif
```

### VTune Macros (Optional)

```cpp
#ifdef USE_VTUNE
    VTUNE_TASK("Task Name")
    // Appears in VTune timeline
#endif
```

## Metric Categories

### Selection Phase
- `SelectionTotal` - Total selection time
- `SelectionPUCT` - PUCT computation
- `SelectionAVX2` - AVX2 vectorized ops
- `SelectionAtomicLoad` - Atomic loads
- `SelectionBusyEdgeSkip` - Nodes skipped
- `SelectionRetry` - Selection restarts
- `SelectionDepth` - Tree depth (gauge)

### Expansion Phase
- `ExpansionTotal` - Total expansion time
- `ExpansionInferenceRequest` - Inference submission
- `ExpansionInferenceWait` - Waiting for inference
- `ExpansionMaskPolicy` - Legal move masking
- `ExpansionAllocateNodes` - Child allocation
- `ExpansionConflict` - Expansion conflicts

### Backup Phase
- `BackupTotal` - Total backup time
- `BackupPathTraversal` - Path traversal
- `BackupAtomicVisitUpdate` - Visit count update
- `BackupAtomicValueUpdate` - Value update
- `BackupVirtualLossRemove` - VL removal

### Virtual Loss
- `VirtualLossApply` - Apply VL
- `VirtualLossRemove` - Remove VL
- `VirtualLossCASSuccess` - Successful CAS
- `VirtualLossCASFailure` - Failed CAS
- `VirtualLossCASRetries` - Retry count

### Queue Operations
- `QueueSubmit` - Submit request
- `QueueCollect` - Collect batch
- `QueueCondVarWait` - Condition variable wait
- `QueueBatchSize` - Batch size (gauge)
- `QueuePendingDepth` - Queue depth (gauge)

### Memory
- `MemoryNodeAllocate` - Node allocation
- `MemoryNodeAllocateFast` - Fast path
- `MemoryNodeAllocateSlow` - Slow path
- `MemoryArenaAllocate` - Arena allocation

### Synchronization
- `SyncMutexLockWait` - Mutex wait time
- `SyncAtomicCASSuccess` - CAS success
- `SyncAtomicCASFailure` - CAS failure
- `SyncSpinWaitCycles` - Spin cycles

### Hardware Counters
- `HWCPUCycles` - CPU cycles
- `HWInstructions` - Instructions
- `HWIPC` - Instructions per cycle
- `HWCacheMisses` - Cache misses
- `HWBranchMisses` - Branch misses

**Full list:** See `metrics.hpp` for all 230+ metrics

## Export Formats

### JSON

```cpp
Profiler::instance().export_json("profile.json");
```

**Output:**
```json
{
  "session": {
    "name": "Benchmark",
    "duration_ns": 60000000000,
    "threads": 12
  },
  "metrics": [
    {
      "name": "selection_total",
      "count": 1234567,
      "mean_ns": 850,
      "std_dev_ns": 120,
      "p50_ns": 820,
      "p95_ns": 1100,
      "p99_ns": 1500
    }
  ]
}
```

### Chrome Trace

```cpp
Profiler::instance().export_chrome_trace("trace.json");
```

**Visualization:**
1. Open Chrome/Chromium
2. Navigate to `chrome://tracing`
3. Click "Load" and select `trace.json`
4. Use WASD keys to navigate timeline

### Markdown

```cpp
Profiler::instance().export_markdown("report.md");
```

**Output:**
```markdown
# Performance Report

## Summary
- Duration: 60.0s
- Threads: 12
- Total Samples: 1,234,567

## Top Metrics
| Metric | Count | Mean (ns) | P95 (ns) | P99 (ns) |
|--------|-------|-----------|----------|----------|
| selection_total | 1234567 | 850 | 1100 | 1500 |
```

### CSV

```cpp
CSVExporter::export_metrics("metrics.csv", metrics);
```

**Output:**
```csv
metric,count,mean_ns,std_dev_ns,p50_ns,p95_ns,p99_ns
selection_total,1234567,850,120,820,1100,1500
expansion_total,1234567,320,45,310,410,520
```

### Comprehensive Report

```cpp
SessionMetadata metadata;
metadata.session_name = "OptimizationRun";
metadata.num_threads = 12;

ReportBuilder()
    .add_session(metadata)
    .add_metrics(metrics)
    .add_bottlenecks(bottlenecks)
    .export_json("report.json")
    .export_chrome_trace("trace.json")
    .export_markdown("report.md")
    .export_html("report.html");
```

## Troubleshooting

### Problem: High Overhead

**Symptoms:**
- Profiling slows down code by >5%

**Solutions:**
```cpp
// 1. Reduce sampling rate
Profiler::instance().set_sampling_rate(100);  // Profile 1 in 100

// 2. Lower profiling level
// In CMakeLists.txt:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPROFILE_LEVEL=1")

// 3. Disable hardware counters
Profiler::instance().disable_hardware_counters();

// 4. Profile less frequently
#if PROFILE_LEVEL >= 2
    PROFILE_SCOPE(metric);
#endif
```

### Problem: Ring Buffer Overflow

**Symptoms:**
- `samples_dropped > 0` in thread metrics

**Solutions:**
```cpp
// In thread_local_metrics.hpp, increase buffer size:
static constexpr size_t SAMPLE_BUFFER_SIZE = 8192;  // Default: 4096

// Or increase sampling rate:
Profiler::instance().set_sampling_rate(10);
```

### Problem: Hardware Counters Not Working

**Symptoms:**
- `hw.initialize()` returns `false`
- `/proc/sys/kernel/perf_event_paranoid` errors

**Solutions:**
```bash
# 1. Check kernel permissions
cat /proc/sys/kernel/perf_event_paranoid
# Should be <= 2

# 2. Allow non-root perf access
sudo sysctl -w kernel.perf_event_paranoid=1

# 3. Or run with capabilities
sudo setcap cap_perfmon=ep ./your_binary

# 4. Or run as root (not recommended)
sudo ./your_binary
```

### Problem: Chrome Trace File Too Large

**Symptoms:**
- Chrome Trace export > 100 MB
- Chrome can't load trace file

**Solutions:**
```cpp
// 1. Use sampling
Profiler::instance().set_sampling_rate(1000);

// 2. Profile shorter duration
Profiler::instance().start_session("ShortRun");
// ... run for 10s instead of 60s ...
Profiler::instance().stop_session();

// 3. Use binary format for storage
Profiler::instance().export_binary("profile.bin");
// Then convert subset to Chrome Trace
```

### Problem: Missing Metrics

**Symptoms:**
- Some metrics show 0 samples
- Expected metrics not appearing

**Solutions:**
```cpp
// 1. Check if profiling is enabled
if (!Profiler::instance().is_enabled()) {
    Profiler::instance().enable();
}

// 2. Check profiling level
#if PROFILE_LEVEL < 2
    // Hardware counters won't be available
#endif

// 3. Verify macro usage
PROFILE_SCOPE(ProfileMetric::SelectionTotal);  // Correct
// Not: PROFILE_SCOPE("SelectionTotal");        // Wrong (string)

// 4. Check sampling rate
Profiler::instance().set_sampling_rate(1);  // Profile all
```

## Performance Tips

### 1. Use Appropriate Profiling Level

```cpp
// Development: Full profiling
#define PROFILE_LEVEL 3

// Testing: Basic profiling
#define PROFILE_LEVEL 1

// Production: Disabled
#define PROFILE_LEVEL 0
```

### 2. Profile Hot Paths Only

```cpp
void hot_path() {
    PROFILE_SCOPE(ProfileMetric::SelectionTotal);
    // Called 1M times per second - profile it
}

void cold_path() {
    // Called once per minute - don't profile
}
```

### 3. Use Gauges for State

```cpp
// Don't do this:
for (int i = 0; i < count; ++i) {
    PROFILE_COUNTER(ProfileMetric::QueuePendingDepth);
}

// Do this:
PROFILE_GAUGE(ProfileMetric::QueuePendingDepth, count);
```

### 4. Batch Counter Updates

```cpp
// Don't do this:
for (int i = 0; i < n; ++i) {
    PROFILE_COUNTER(ProfileMetric::ExpansionConflict, 1);
}

// Do this:
int conflicts = count_conflicts();
PROFILE_COUNTER(ProfileMetric::ExpansionConflict, conflicts);
```

## Quick Commands

```bash
# Enable profiling
export PROFILE_LEVEL=1

# Build with profiling
cmake -DPROFILE_LEVEL=1 ..
make

# Run with profiling
./mcts_benchmark

# View Chrome Trace
google-chrome chrome://tracing
# Load trace.json

# Analyze JSON
python scripts/analyze_profile.py profile.json

# Compare sessions
python scripts/compare_profiles.py baseline.json current.json
```

## Further Reading

- **Full Documentation:** See `README.md`
- **Architecture:** See `ARCHITECTURE.md`
- **Examples:** See `example_usage.cpp`
- **Design:** See `docs/performance/enhanced_profiling_design.md`

## Support

For issues or questions:
1. Check `ARCHITECTURE.md` for design details
2. Review `example_usage.cpp` for patterns
3. See main project documentation
4. File issue with profiling logs and system info
