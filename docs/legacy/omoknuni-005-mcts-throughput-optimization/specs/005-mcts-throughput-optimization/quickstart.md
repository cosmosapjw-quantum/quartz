# Quick Start: MCTS Throughput Optimization

**Feature**: MCTS Throughput Optimization (Zero-Copy + Tensor Pipeline)
**Branch**: `005-mcts-throughput-optimization`
**Reference**: [plan.md](plan.md), [spec.md](spec.md)

---

## Overview

This guide provides step-by-step instructions for building, validating, and deploying the MCTS throughput optimizations. Follow these instructions to reproduce performance results and verify acceptance criteria.

**Optimization Phases**:
- **Phase 1**: State cloning elimination → 1.5k-3k sims/sec (10-25× gain)
- **Phase 2**: Tensor pipeline + OpenMP → 7k-9k sims/sec (58-75× gain) ✅ **PRIMARY TARGET**
- **Phase 3A**: Multi-coordinator (stretch) → 12k-20k sims/sec (100-166× gain)
- **Phase 3B**: Multi-process (optional) → 20k-35k sims/sec (166-291× gain)

---

## Prerequisites

### Hardware Requirements

**Minimum** (for Phase 1-2):
- CPU: 8-core/16-thread (e.g., AMD Ryzen 5 3700X, Intel i7-10700K)
- GPU: NVIDIA RTX 2060 or better (6GB VRAM, Turing+ architecture)
- RAM: 16GB
- Storage: 10GB free space

**Recommended** (for Phase 3):
- CPU: 12-core/24-thread (e.g., AMD Ryzen 9 5900X, Intel i9-12900K)
- GPU: NVIDIA RTX 3060 Ti or better (8GB VRAM, Ampere+ architecture)
- RAM: 32GB
- Storage: 20GB free space

### Software Requirements

**Operating System**:
- Linux: Ubuntu 20.04+ (recommended), Debian 11+, Arch Linux
- macOS: 12.0+ (CPU-only, limited GPU support)
- Windows: WSL2 with Ubuntu 20.04+ (native Windows not tested)

**Compiler & Build Tools**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    g++-10 \
    libomp-dev \
    git

# macOS (Homebrew)
brew install cmake ninja libomp

# Verify compiler supports C++17 and OpenMP
g++ --version  # Should be ≥9.0
g++ -fopenmp -dM -E - </dev/null | grep -i openmp  # Should show OpenMP version
```

**Python Environment**:
```bash
# Python 3.10-3.12 (tested). Recommended: 3.11/3.12 for fastest CPython
python3 --version  # Should be 3.10, 3.11, or 3.12

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR: venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

**CUDA Toolkit** (for GPU support):
```bash
# Ubuntu/Debian: Install CUDA 11.8+ or 12.x
# Visit: https://developer.nvidia.com/cuda-downloads

# Verify CUDA installation
nvcc --version  # Should show CUDA 11.8+
nvidia-smi     # Should show driver version ≥520.61.05 for CUDA 11.8
```

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/cosmosapjw/omoknuni.git
cd omoknuni
git checkout 005-mcts-throughput-optimization
```

### Step 2: Install Python Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected output: "CUDA available: True"
```

### Step 3: Build C++ Extensions

**Phase 1 Build** (State Cloning Elimination):
```bash
# Set compiler flags for Ryzen 5900X (adjust -march for your CPU)
export CFLAGS="-O3 -march=znver3 -fopenmp"
export CXXFLAGS="-O3 -march=znver3 -fopenmp"

# Build with CMake
python -m pip install -e . --config-settings build-dir=build --verbose

# Verify OpenMP linkage
ldd build/lib.*/mcts_py*.so | grep omp
# Expected output: libomp.so or libgomp.so

# Verify build success
python -c "import mcts_py; print('Build successful!')"
```

**Phase 2 Build** (Tensor Pipeline + OpenMP):
```bash
# Same as Phase 1 (OpenMP fix applied in CMakeLists.txt)
# No additional flags required

# Rebuild after CMakeLists.txt changes
pip install -e . --force-reinstall --no-deps --config-settings build-dir=build
```

### Step 4: Verify Installation

```bash
# Run unit tests
python -m pytest tests/unit/ -v

# Run contract tests
python -m pytest tests/contract/ -v

# Run quick performance test
python scripts/test_mcts.py --game gomoku --simulations 100 --threads 4
# Expected output: Throughput > 1000 sims/sec (Phase 1)
```

---

## Validation

### Phase 1 Validation (State Cloning Elimination)

**Expected Results**:
- Throughput: 1,500-3,000 sims/sec (10-25× baseline)
- State cloning overhead: <1% of total time
- Zero `clone()` calls in hot path

**Validation Benchmark**:
```bash
# Run Phase 1 profiling campaign (100 trials)
python scripts/validate_phase1.py \
    --game gomoku \
    --simulations 800 \
    --threads 8 \
    --trials 100 \
    --output profiling_results/phase_1

# Expected output:
# ✅ Phase 1 ACCEPTED
# Mean throughput: 2,247 ± 134 sims/sec (target: 1,500-3,000)
# State cloning overhead: 0.2% (target: <1%)
# State clone count: 0 (target: 0)
```

**Manual Verification**:
```bash
# Check for state cloning in code (should be zero matches)
grep -r "\.clone()" cpp_extensions/mcts/continuous_simulation_runner.cpp
grep -r "new.*State" cpp_extensions/mcts/continuous_simulation_runner.cpp

# Run with verbose profiling
python scripts/test_mcts.py \
    --game gomoku \
    --simulations 800 \
    --threads 8 \
    --profiling-enabled \
    --verbose

# Check profiling output for:
# - "State cloning time: 0.0 ms" (should be ~0)
# - "Feature move count: 800" (should equal simulations)
```

---

### Phase 2 Validation (Tensor Pipeline + OpenMP)

**Expected Results**:
- Throughput: 7,000-9,000 sims/sec (58-75× baseline)
- Tensor creation: <2.0ms per batch
- OpenMP enabled with >1 thread
- Pinned buffer reuse: 100%

**Validation Benchmark**:
```bash
# Run Phase 2 profiling campaign (100 trials)
python scripts/validate_phase2.py \
    --game gomoku \
    --simulations 800 \
    --threads 8 \
    --batch-size 64 \
    --trials 100 \
    --output profiling_results/phase_2

# Expected output:
# ✅ Phase 2 ACCEPTED
# Mean throughput: 8,124 ± 412 sims/sec (target: 7,000-9,000)
# Tensor creation: 1.8ms (target: <2.0ms)
# H2D transfer: 0.7ms (target: <1.0ms)
# OpenMP enabled: True (target: True)
# OpenMP threads: 8 (target: >1)
# Pinned buffer reuse: 100.0% (target: 100%)
```

**Manual Verification**:
```bash
# Verify OpenMP linkage
python -c "
import mcts_py
import ctypes
import os

# Check OpenMP symbols in shared library
lib_path = mcts_py.__file__
print(f'Library: {lib_path}')

# On Linux
os.system(f'nm -D {lib_path} | grep omp')
"

# Run with OpenMP environment variables
export OMP_NUM_THREADS=8
export OMP_DISPLAY_ENV=TRUE

python scripts/test_mcts.py \
    --game gomoku \
    --simulations 800 \
    --threads 8 \
    --profiling-enabled

# Check output for OpenMP initialization messages
```

---

### Phase 3A Validation (Multi-Coordinator - Stretch)

**Expected Results**:
- Throughput: 12,000-20,000 sims/sec (100-166× baseline)
- Coordinator blocking: <10%
- GPU utilization: ≥80%

**Validation Benchmark**:
```bash
# Run Phase 3A profiling campaign
python scripts/validate_phase3a.py \
    --game gomoku \
    --simulations 800 \
    --threads 8 \
    --coordinators 3 \
    --batch-size 64 \
    --trials 100 \
    --output profiling_results/phase_3a

# Expected output:
# ✅ Phase 3A ACCEPTED
# Mean throughput: 15,623 ± 891 sims/sec (target: 12,000-20,000)
# Coordinator blocking: 7.3% (target: <10%)
# GPU utilization: 84.2% (target: ≥80%)
```

---

## Performance Tuning

### Thread Count Optimization

```bash
# Find optimal thread count for your CPU
python scripts/tune_threads.py \
    --game gomoku \
    --simulations 800 \
    --min-threads 2 \
    --max-threads 16 \
    --iterations 50

# Expected output (Ryzen 5900X):
# Optimal thread count: 8
# Throughput at 8 threads: 8,247 sims/sec
# Thread efficiency: 78%
```

**Guidance**:
- Start with physical core count (not including SMT/HT)
- Dual-CCD CPUs (Ryzen 5900X): Use 8-12 threads
- Single-CCD CPUs (Ryzen 5600X): Use 4-6 threads
- Intel CPUs: Use 4-8 threads depending on cache topology

### Batch Size Optimization

```bash
# Find optimal batch size for your GPU
python scripts/tune_batch_size.py \
    --game gomoku \
    --simulations 800 \
    --threads 8 \
    --min-batch 16 \
    --max-batch 128 \
    --iterations 100

# Expected output (RTX 3060 Ti):
# Optimal batch size: 64
# Throughput at batch 64: 8,124 sims/sec
# GPU utilization: 82%
```

**Guidance**:
- RTX 2060 (6GB): batch_size = 32-48
- RTX 3060 Ti (8GB): batch_size = 64
- RTX 3090 (24GB): batch_size = 96-128

### Timeout Tuning

```bash
# Find optimal batch timeout
python scripts/tune_timeout.py \
    --game gomoku \
    --simulations 800 \
    --threads 8 \
    --batch-size 64 \
    --min-timeout 100 \
    --max-timeout 2000 \
    --iterations 100

# Expected output:
# Optimal timeout: 500 μs
# Throughput at 500μs: 8,247 sims/sec
# Avg batch size: 58.3
```

**Guidance**:
- Faster GPUs (RTX 3080+): 300-500μs
- Mid-range GPUs (RTX 3060): 500-800μs
- Slower GPUs (GTX 1660): 1000-1500μs

---

## Troubleshooting

### Issue: Low Throughput (< 1,500 sims/sec after Phase 1)

**Symptoms**:
```bash
python scripts/test_mcts.py --game gomoku --simulations 800 --threads 8
# Output: Throughput: 842 sims/sec (expected: 1,500-3,000)
```

**Diagnosis**:
```bash
# Check for state cloning (should be 0)
python scripts/test_mcts.py --profiling-enabled --verbose 2>&1 | grep "State cloning"
# If "State cloning time: > 0.0 ms" → Phase 1 not fully implemented
```

**Fix**:
1. Verify thread-local feature buffers allocated:
   ```bash
   grep "feature_buffer" cpp_extensions/mcts/continuous_simulation_runner.cpp
   # Should show: std::vector<float> feature_buffer;
   ```

2. Verify move semantics in queue submission:
   ```bash
   grep "std::move(request)" cpp_extensions/mcts/async_inference_queue.cpp
   # Should show: requests_.push_back(std::move(request));
   ```

3. Verify coordinator doesn't clone states:
   ```bash
   grep "clone()" cpp_extensions/mcts/batch_inference_coordinator.cpp
   # Should return no matches (or only in comments)
   ```

---

### Issue: OpenMP Not Working (Phase 2)

**Symptoms**:
```bash
python scripts/test_mcts.py --profiling-enabled 2>&1 | grep "OpenMP"
# Output: OpenMP enabled: False (expected: True)
```

**Diagnosis**:
```bash
# Check if OpenMP linked to shared library
ldd build/lib.*/mcts_py*.so | grep omp
# If no output → OpenMP not linked
```

**Fix**:
1. Verify CMakeLists.txt fix applied:
   ```bash
   grep "target_link_libraries(mcts_py" CMakeLists.txt
   # Should show: OpenMP::OpenMP_CXX
   ```

2. Rebuild with forced relink:
   ```bash
   rm -rf build/
   pip install -e . --force-reinstall --no-deps --config-settings build-dir=build
   ```

3. Set OpenMP environment variables:
   ```bash
   export OMP_NUM_THREADS=8
   export OMP_DISPLAY_ENV=TRUE
   python scripts/test_mcts.py --profiling-enabled
   # Should show OpenMP initialization messages
   ```

---

### Issue: Tensor Creation Slow (> 2ms per batch)

**Symptoms**:
```bash
python scripts/test_mcts.py --profiling-enabled 2>&1 | grep "Tensor creation"
# Output: Tensor creation: 7.5 ms (expected: <2.0 ms)
```

**Diagnosis**:
```bash
# Check if pinned memory buffer allocated
python -c "
from src.core.dlpack_inference_bridge import DLPackInferenceBridge
bridge = DLPackInferenceBridge()
print(f'Pinned: {bridge.pinned_buffer.is_pinned()}')
"
# Expected: True
```

**Fix**:
1. Verify pinned buffer pre-allocation:
   ```python
   # src/core/dlpack_inference_bridge.py
   self.pinned_buffer = torch.zeros(
       (max_batch, max_planes, max_h, max_w),
       dtype=torch.float32,
       pin_memory=True  # ✅ Must be True
   )
   ```

2. Verify non-blocking GPU transfer:
   ```python
   # src/core/dlpack_inference_bridge.py
   self.gpu_buffer[:batch_size, ...].copy_(
       self.pinned_buffer[:batch_size, ...],
       non_blocking=True  # ✅ Must be True
   )
   ```

3. Check for buffer reallocations:
   ```bash
   python scripts/test_mcts.py --profiling-enabled 2>&1 | grep "Pinned buffer reallocations"
   # Expected: 0 (any reallocation indicates bug)
   ```

---

### Issue: GPU Utilization Low (< 60%)

**Symptoms**:
```bash
nvidia-smi dmon -s u
# Output: GPU utilization: 45% (expected: ≥80%)
```

**Diagnosis**:
```bash
# Check batch size
python scripts/test_mcts.py --profiling-enabled 2>&1 | grep "Avg batch size"
# If < 32 → increase simulations or reduce timeout
```

**Fix**:
1. Increase batch size:
   ```bash
   python scripts/test_mcts.py --batch-size 96  # Try larger batches
   ```

2. Reduce batch timeout (collect larger batches):
   ```bash
   python scripts/test_mcts.py --timeout-us 1000  # Wait longer for full batch
   ```

3. Enable mixed precision (Phase 2):
   ```bash
   python scripts/test_mcts.py --fp16  # Use FP16 inference
   ```

---

## Rollback Procedures

### Rollback to Baseline (Before Phase 1)

```bash
# Checkout baseline commit
git log --oneline | grep "Baseline"
git checkout <baseline-commit-hash>

# Rebuild
rm -rf build/
pip install -e . --force-reinstall --no-deps

# Verify baseline performance
python scripts/test_mcts.py --game gomoku --simulations 800 --threads 8
# Expected: ~120 sims/sec
```

### Rollback Phase 2 to Phase 1

```bash
# Use feature flag to disable Phase 2 optimizations
export MCTS_DISABLE_PHASE2=1

python scripts/test_mcts.py --game gomoku --simulations 800 --threads 8
# Should show Phase 1 performance: 1,500-3,000 sims/sec
```

### Rollback Phase 3A to Phase 2

```bash
# Reduce coordinator count to 1
python scripts/test_mcts.py --coordinators 1

# Expected: Phase 2 performance: 7,000-9,000 sims/sec
```

---

## Performance Checklist

Before reporting performance results, verify:

- [ ] **Hardware**: CPU ≥8 cores, GPU ≥6GB VRAM, RAM ≥16GB
- [ ] **Compiler**: g++ ≥9.0 with `-O3 -march=native -fopenmp`
- [ ] **OpenMP**: Linked and enabled (`OMP_NUM_THREADS=8`)
- [ ] **CUDA**: Version ≥11.8, driver ≥520.0
- [ ] **PyTorch**: CUDA-enabled (`torch.cuda.is_available() == True`)
- [ ] **Build**: No warnings during compilation
- [ ] **Tests**: All unit/contract tests pass
- [ ] **Profiling**: Zero state cloning, zero pinned buffer reallocations
- [ ] **Tuning**: Optimal thread count, batch size, timeout configured

---

## References

- [plan.md](plan.md): Detailed implementation plan with code examples
- [spec.md](spec.md): Functional requirements and acceptance criteria
- [data-model.md](data-model.md): Data structures and memory layouts
- [contracts/](contracts/): API interface specifications
- [CLAUDE.md](../../CLAUDE.md): Constitution and performance targets
- [MCTS_OPTIMIZATION_MASTER_PLAN.md](../../MCTS_OPTIMIZATION_MASTER_PLAN.md): High-level optimization strategy

---

## Support

For issues not covered in this guide:

1. **Check existing issues**: https://github.com/cosmosapjw/omoknuni/issues
2. **Profiling results**: Attach output from `scripts/validate_phase*.py`
3. **System info**: Include `python --version`, `nvcc --version`, `g++ --version`
4. **Build logs**: Include full output from `pip install -e . --verbose`

**Common Questions**:

**Q: Can I run this on CPU-only?**
A: Yes, but GPU is strongly recommended. CPU-only performance will be 5-10× slower.

**Q: What if I have an AMD GPU?**
A: ROCm support is experimental. Follow PyTorch ROCm installation guide, but expect lower performance.

**Q: Can I use Windows without WSL?**
A: Not tested. Use WSL2 with Ubuntu 20.04+ for best compatibility.

**Q: How do I know which phase is active?**
A: Check profiling output: `state_clone_count=0` → Phase 1+, `openmp_enabled=True` → Phase 2+, `active_coordinators>1` → Phase 3A+
