# How to Run Performance Benchmarks

## Performance Regression Suite (T043)

The performance regression suite provides automated benchmarking with regression detection to ensure the AlphaZero engine maintains target performance levels.

### Quick Commands

```bash
# Run all performance benchmarks
python -m pytest tests/performance/test_benchmarks.py -v

# Run specific performance categories
python -m pytest -m "performance" -v
python -m pytest -m "benchmark" -v

# Run benchmarks directly (without pytest)
python tests/performance/test_benchmarks.py

# Run benchmark framework unit tests
python -m pytest tests/unit/test_benchmark_framework.py -v
```

### Benchmark Categories

1. **MCTS Simulation Rate**: Tests search throughput (target: 30k-40k sims/sec)
2. **Neural Inference Throughput**: Tests NN inference performance (target: 1k+ inf/sec)
3. **GPU Utilization**: Tests GPU efficiency (target: 80-92%)
4. **Memory Efficiency**: Tests tree memory usage (target: <1GB for 10M nodes)
5. **Search Coordinator**: Tests coordination performance (target: 50k+ ops/sec)

### Performance Targets

The benchmarks validate these key performance requirements:

- **30,000-40,000 simulations/second** including neural network inference
- **80-92% GPU utilization** during search operations
- **<1GB memory usage** for 10M node MCTS trees
- **85-95% CPU utilization** optimal thread saturation
- **32-64 average batch size** for efficient GPU occupancy

### Regression Detection

The system automatically detects performance regressions by:

1. **Baseline Comparison**: Stores baseline performance metrics
2. **Threshold Analysis**: Flags >5% performance drops as regressions
3. **Statistical Validation**: Uses multiple iterations for reliable measurements
4. **Automated Reporting**: Generates detailed regression reports

### Usage Examples

```bash
# Run with specific GPU tests (requires CUDA)
python -m pytest -m "gpu and performance" -v

# Run without slow tests
python -m pytest tests/performance/ -m "not slow" -v

# Generate baseline for regression detection
python tests/performance/test_benchmarks.py

# Check for regressions against baseline
python -m pytest tests/performance/test_benchmarks.py::test_performance_regression_detection -v
```

### CI Integration

The benchmarks are designed to run in CI environments:

- **Automated on PRs**: Detects performance regressions
- **Baseline Updates**: Stores performance baselines per branch
- **Failure Thresholds**: Fails CI on >20% regressions
- **Performance Reports**: Generates detailed performance metrics

### Output Files

Results are saved to `results/benchmarks/`:

- `benchmarks_<timestamp>.json` - Latest benchmark results
- `baseline.json` - Performance baseline for regression detection
- Detailed metrics including system resource usage

### System Requirements

- **Python 3.12+** with pytest and numpy
- **PyTorch 2.x** for GPU inference tests (optional)
- **CUDA support** for GPU utilization tests (optional)
- **psutil** for system monitoring
- **pynvml** for GPU monitoring (optional)

### Docker Testing

Run tests in containerized environments for consistent results:

```bash
# Run all tests in Docker
docker-compose run --rm benchmark

# Run specific test categories in Docker
docker-compose run --rm dev python -m pytest tests/performance/ -v

# Run tests in development environment
./scripts/docker/run.sh dev
# Then inside container: python -m pytest tests/unit/ -v
```

### Docker Container Validation

Test Docker functionality and configuration:

```bash
# Test Docker setup
python -m pytest tests/unit/test_docker_functionality.py -v

# Validate Dockerfile syntax
docker build --target runtime -t test-build .

# Test all Docker stages
./scripts/docker/build.sh -t all
```

### Troubleshooting

**No GPU detected**: GPU benchmarks fall back to CPU simulation
**High variance**: Increase iteration count for more stable results
**Memory errors**: Reduce batch sizes in memory efficiency tests
**Permission errors**: Ensure write access to `results/` directory
**Docker issues**: Check NVIDIA Docker runtime with `docker run --rm --gpus all nvidia/cuda:12.2-base nvidia-smi`