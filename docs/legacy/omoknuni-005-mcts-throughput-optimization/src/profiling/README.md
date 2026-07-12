# Python Profiling Framework

Comprehensive profiling system for analyzing Python coordination overhead in the MCTS engine.

## Quick Start

```python
from src.profiling import ProfilingSession, ProfilerConfig

# Create profiling session
config = ProfilerConfig(
    enable_gil_profiling=True,
    enable_inference_profiling=True,
    enable_thread_profiling=True,
    enable_memory_profiling=True,
    auto_save_reports=True
)

# Profile your workload
with ProfilingSession(config) as session:
    # ... run MCTS searches ...
    mcts.search(root_state, 800)

# Reports automatically saved to profiling_reports/
```

## Features

### 1. GIL Profiler
Track Python Global Interpreter Lock usage and contention:
- Time with/without GIL per thread
- GIL wait times and hotspots
- Contention events between threads

### 2. Inference Pipeline Profiler
Profile neural network inference end-to-end:
- Request latency breakdown (queue → batch → GPU → results)
- Batch collection efficiency
- DLPack zero-copy effectiveness

### 3. Thread Coordinator Profiler
Measure ThreadPoolExecutor and Future overhead:
- Future lifecycle tracking
- Thread pool utilization
- Task submission and execution latency

### 4. Memory Profiler
Track memory allocations and garbage collection:
- Memory timeline with periodic snapshots
- GC event monitoring
- Memory leak detection
- Allocation hotspots (via tracemalloc)

## Modules

- `gil_profiler.py` - GIL profiling
- `inference_profiler.py` - Inference pipeline profiling
- `thread_profiler.py` - Thread coordination profiling
- `memory_profiler.py` - Memory and GC profiling
- `profiling_session.py` - Unified session manager
- `report_generator.py` - Report generation (JSON/HTML/Flamegraph)

## Examples

See `examples/` directory:
- `profiling_demo.py` - Basic demonstrations
- `profile_mcts_search.py` - Real MCTS profiling

Run examples:
```bash
python examples/profiling_demo.py
python examples/profile_mcts_search.py --simulations 800 --threads 4
```

## Documentation

Complete documentation:
- `docs/profiling_framework.md` - User guide
- `PROFILING_DESIGN.md` - Design document

## Performance Impact

Typical overhead: 4-6%
- GIL Profiler: <2%
- Inference Profiler: <1%
- Thread Profiler: <1%
- Memory Profiler: 1-2% (5-10% with tracemalloc)

## Report Formats

Three report formats generated automatically:

1. **JSON**: Machine-readable structured data
2. **HTML**: Interactive dashboard with charts
3. **Markdown**: Summary report

## Integration

The framework integrates seamlessly with:
- MCTS engine (`src/core/mcts.py`)
- Inference worker (`src/neural/inference_worker.py`)
- DLPack bridge (`src/core/dlpack_inference_bridge.py`)
- Search coordinator (`src/core/search_coordinator.py`)
- C++ instrumentation (via `mcts_py`)

## Common Use Cases

### Diagnose Low Throughput
```python
with ProfilingSession(config) as session:
    mcts.search(root_state, 800)

metrics = session.get_all_metrics()
if metrics['gil_metrics']['summary']['gil_efficiency'] < 70:
    print("WARNING: Poor GIL efficiency!")
```

### Optimize Batch Collection
```python
config = ProfilerConfig(enable_inference_profiling=True)
with ProfilingSession(config) as session:
    # ... run searches ...

# Analyze stage breakdown
for stage, stats in metrics['inference_metrics']['stage_breakdown'].items():
    print(f"{stage}: {stats['percentage']:.1f}%")
```

### Detect Memory Leaks
```python
config = ProfilerConfig(enable_memory_profiling=True, memory_enable_tracemalloc=True)
with ProfilingSession(config) as session:
    # ... long workload ...

if metrics['memory_metrics']['leak_candidates']:
    print("WARNING: Potential memory leaks detected!")
```

## License

See project LICENSE file.
