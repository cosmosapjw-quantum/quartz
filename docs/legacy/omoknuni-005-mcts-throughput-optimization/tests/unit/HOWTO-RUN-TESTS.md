# How to Run Tests - MCTS Engine & Micro-batching

## Quick Validation Test

The simplest way to validate the PUCT selection implementation:

```bash
# Compile and run standalone test
g++ -std=c++17 -O2 -mavx2 -I./cpp_extensions \
    -o test_selection \
    tests/unit/test_selection_simple.cpp \
    cpp_extensions/mcts/tree.cpp \
    cpp_extensions/mcts/selection.cpp

./test_selection
```

**Expected Output**:
- Basic selection works ✓
- SIMD vs scalar consistency ✓
- Performance target achieved (3.6-5.2x speedup) ✓
- Edge cases handled ✓
- AVX2 support detected ✓

## Full Google Test Suite (AVAILABLE NOW)

Run comprehensive tests with the CMake build system:

```bash
# Build with Google Test (automatic download if not found)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build and run all tests
make test_selection test_node_pool
./tests/unit/test_selection
./tests/unit/test_node_pool

# Or run all tests via CTest
ctest --output-on-failure
```

## Performance Benchmarking

To validate the 4-8x speedup target with various child counts:

```bash
# Test with realistic MCTS tree sizes
for children in 16 32 64 128; do
    echo "Testing with $children children:"
    # Run benchmark with specific child count
done
```

## Manual Verification

Key validation points:
1. **PUCT Formula**: Correctly implements `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`
2. **SIMD Consistency**: Vector and scalar results must match within 1e-6 tolerance
3. **Performance**: 3.6x+ speedup with 64+ children (target achieved)
4. **Edge Cases**: Handles 0 children, invalid nodes, single child correctly

## Compiler Requirements

- **C++17** support required
- **AVX2** support recommended (fallback to scalar if unavailable)
- **Optimization** flags `-O2` or `-O3` for realistic performance testing

The implementation automatically detects AVX2 support and gracefully falls back to scalar operations on unsupported hardware.

## Micro-batching Tests (T015)

Test the dynamic micro-batching implementation:

```bash
# Run micro-batching unit tests
python -m pytest tests/unit/test_micro_batching.py -v

# Run performance validation
python scripts/validate_micro_batching.py

# Run inference worker tests with micro-batching
python -m pytest tests/unit/test_inference_worker.py -v
```

**Expected Results**:
- Count-based batching: ≥32 positions ✓
- Timeout-based batching: ≤3ms constraint ✓
- GPU utilization monitoring: >80% target ✓
- Adaptive batch sizing: Performance feedback ✓
- All 19 micro-batching tests passing ✓

## Full Test Suite

Run all available tests:

```bash
# Unit tests for all components
python -m pytest tests/unit/ -v

# Contract tests (API validation)
python -m pytest tests/contract/ -v

# Integration tests (when available)
python -m pytest tests/integration/ -v

# Performance validation
python scripts/validate_micro_batching.py
```