# CI/CD Pipeline Documentation

## Overview

The project uses GitHub Actions for continuous integration and deployment. The pipeline is designed to ensure code quality, performance, and compatibility across different environments.

## Pipeline Stages

### 1. Lint and Type Check
- **Runs on**: Ubuntu Latest
- **Tools**: Black, isort, flake8, mypy
- **Purpose**: Code formatting, import sorting, linting, and type checking

### 2. CPU Tests
- **Runs on**: Ubuntu Latest
- **Tests**: Unit tests, contract tests
- **Features**:
  - Build artifact caching
  - Code coverage reporting
  - CPU-only package build

### 3. GPU Tests (Self-hosted)
- **Runs on**: Self-hosted runner with RTX 3060 Ti
- **Triggered**: On push to main or PR with 'gpu-test' label
- **Features**:
  - CUDA environment validation
  - GPU-specific test execution
  - Performance benchmarking

### 4. Performance Regression Detection
- **Runs on**: Self-hosted GPU runner
- **Purpose**: Compare current benchmarks with baseline
- **Output**: Performance regression alerts

### 5. Integration Tests
- **Runs on**: Ubuntu Latest
- **Purpose**: End-to-end functionality testing

### 6. Memory Leak Detection
- **Runs on**: Self-hosted GPU runner (weekly)
- **Tools**: Python tracemalloc, valgrind, CUDA memory profiling
- **Purpose**: Long-term stability validation and memory leak detection
- **Duration**: Configurable (default: 30-60 minutes)

### 7. Security Scanning
- **Tools**: Bandit, Safety
- **Purpose**: Security vulnerability detection

### 8. Documentation Build
- **Runs on**: Ubuntu Latest (main branch only)
- **Purpose**: Generate and validate documentation

## Configuration Files

### Workflow Configuration
- **File**: `.github/workflows/ci.yml`
- **Triggers**: Push to main/develop, PRs to main/develop
- **Manual trigger**: `workflow_dispatch`

### Test Configuration
- **File**: `pytest.ini`
- **Markers**: `slow`, `gpu`, `integration`, `contract`, `performance`, `benchmark`

### Performance Monitoring
- **Script**: `scripts/compare_benchmarks.py`
- **Purpose**: Automated performance regression detection
- **Baseline**: `.benchmarks/baseline.json`

### Memory Leak Detection
- **Script**: `scripts/check_memory_leaks.py`
- **Purpose**: Comprehensive memory leak detection and analysis
- **Output**: JSON reports with leak analysis and recommendations

## Test Structure

```
tests/
├── unit/               # Unit tests (fast, isolated)
├── contract/           # API contract tests (TDD approach)
├── integration/        # End-to-end integration tests
└── performance/        # Benchmark tests for regression detection
```

## Running Tests Locally

### Basic Test Run
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-benchmark

# Run all tests
pytest

# Run specific test types
pytest tests/unit/          # Unit tests only
pytest tests/contract/      # Contract tests only
pytest -m "not slow"        # Skip slow tests
pytest -m gpu              # GPU tests only
```

### Performance Benchmarks
```bash
# Run benchmarks
pytest tests/performance/ --benchmark-only

# Compare with baseline
python scripts/compare_benchmarks.py benchmark_results.json
```

### Memory Leak Detection
```bash
# Comprehensive leak detection (all methods)
python scripts/check_memory_leaks.py --all --duration 1800 --output leak_report.json

# Python-only profiling
python scripts/check_memory_leaks.py --python --threshold 5.0

# C++ component analysis with valgrind
python scripts/check_memory_leaks.py --valgrind --component mcts --component games

# GPU memory monitoring
python scripts/check_memory_leaks.py --gpu --verbose
```

### Code Quality Checks
```bash
# Code formatting
black --check src/ tests/

# Import sorting
isort --check-only src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

## Self-hosted Runner Setup

For GPU testing, a self-hosted runner should be configured with:

1. **Hardware**: NVIDIA RTX 3060 Ti or compatible GPU
2. **Software**:
   - CUDA Toolkit 11.8+
   - Docker (optional)
   - Python 3.12
3. **Labels**: `self-hosted`, `gpu`, `rtx-3060-ti`

### Runner Configuration
```bash
# Install runner
./config.sh --url https://github.com/USER/REPO --token TOKEN

# Add labels
./config.sh --labels self-hosted,gpu,rtx-3060-ti

# Start runner
./run.sh
```

## Performance Regression Detection

The pipeline automatically compares benchmark results with a baseline:

1. **Threshold**: 20% slower triggers regression alert
2. **Baseline Update**: Automatically updated on main branch
3. **Reporting**: Detailed performance comparison in CI output

### Benchmark Metrics
- Computation performance
- Memory allocation patterns
- GPU utilization (when available)

## Troubleshooting

### Common Issues

1. **Test Failures**: Check test output for specific error messages
2. **Build Failures**: Verify dependencies in requirements.txt
3. **GPU Tests Skipped**: Ensure self-hosted runner is available and labeled correctly
4. **Performance Regressions**: Review code changes for performance impact

### Debug Commands
```bash
# Verbose test output
pytest -v -s

# Coverage report
pytest --cov=src --cov-report=html

# Benchmark details
pytest tests/performance/ --benchmark-verbose
```

## Contributing

When contributing:

1. **All tests must pass** before merging
2. **Add tests** for new functionality
3. **Update benchmarks** if adding performance-critical code
4. **Use appropriate test markers** for categorization