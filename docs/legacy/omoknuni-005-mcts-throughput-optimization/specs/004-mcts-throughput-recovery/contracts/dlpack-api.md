# DLPack Tensor Bridge API Specification

**Version**: 1.1.0 (Updated with Validation Results)
**Status**: Design Complete, Implementation Validated (2025-10-13)
**Author**: Claude Code
**Date**: 2025-10-08 (Updated: 2025-10-13)

## ⚠️ CRITICAL PERFORMANCE NOTE (2025-10-13 Validation)

**T-VALID-2 Findings:**
- **Measured Overhead**: 7.50 ± 0.20 ms per batch-64 (target: <1.0ms)
- **Root Cause**: Feature extraction loop NOT parallelized with OpenMP
- **Location**: `cpp_extensions/mcts/dlpack_bridge.cpp:431-434`
- **Required Fix**:
  ```cpp
  #pragma omp parallel for schedule(static) if(batch_size > 8)
  for (int i = 0; i < batch_size; ++i) {
      float* state_buffer = data + (i * state_size);
      states[i]->extract_features_to_buffer(state_buffer);
  }
  ```
- **Expected Improvement**: 7.5ms → <1.0ms with 12-thread parallelization

See [validation_report_2025-10-13.md](../../../docs/performance/validation_report_2025-10-13.md) for detailed results.

## Overview

This document specifies the C++ API for zero-copy tensor exchange between C++ MCTS simulation and PyTorch neural network inference using the DLPack protocol.

**Memory Model**: Uses PINNED CPU MEMORY (kDLCUDAHost), NOT direct GPU device memory (kDLCUDA). Tensor resides in pinned host memory for fast H2D GPU transfers (~0.24ms).

## DLPack Background

### What is DLPack?

DLPack is a **protocol for zero-copy tensor exchange** between frameworks (PyTorch, TensorFlow, JAX, etc.). It enables sharing tensor data without copying by passing a standardized `PyCapsule` object containing:

1. **Tensor metadata**: shape, strides, data type, device
2. **Data pointer**: Raw memory address (CPU or GPU)
3. **Deleter function**: Cleanup callback for memory management

### Key Concepts

**Ownership Model**:
- **Producer** (C++): Allocates memory, creates DLPack capsule, transfers ownership
- **Consumer** (PyTorch): Receives capsule via `torch.from_dlpack()`, shares memory
- **Lifetime**: Memory valid until last tensor reference released

**Memory Management**:
```
C++ allocates → DLPack capsule → PyTorch tensor → Shared reference
     ↓                                                    ↓
Deleter registered ← ← ← ← ← ← ← ← ← ← ← Cleanup on last unref
```

**Zero-Copy Guarantee**:
- Same memory address used by C++ and PyTorch
- No memcpy() calls in hot path
- GPU tensors: Direct device pointer sharing

### DLPack Specification Summary

**DLTensor Structure** (from `dlpack.h`):
```c
typedef struct {
    void* data;              // Pointer to data
    DLDevice device;         // CPU, CUDA, etc.
    int32_t ndim;           // Number of dimensions
    DLDataType dtype;       // Data type (float32, int64, etc.)
    int64_t* shape;         // Dimension sizes [batch, channels, height, width]
    int64_t* strides;       // Strides in elements (NULL = row-major)
    uint64_t byte_offset;   // Offset from data pointer
} DLTensor;

typedef struct {
    DLTensor dl_tensor;
    void* manager_ctx;      // Opaque context for deleter
    void (*deleter)(struct DLManagedTensor*);  // Cleanup function
} DLManagedTensor;
```

**Device Types**:
- `kDLCPU = 1`: CPU memory (pageable)
- `kDLCUDA = 2`: CUDA GPU device memory (on-device)
- `kDLCUDAHost = 3`: CUDA pinned host memory (CPU-accessible, fast GPU transfers) ← **WE USE THIS**

**Data Types**:
- `kDLFloat = 2`: Floating-point (float32 for us)
- 32 bits per element

## API Design

### Core Interface

#### 1. Batch Tensor Creation

```cpp
namespace mcts {

/**
 * @brief Create DLPack tensor from batch of game states
 *
 * Creates a 4D tensor [batch_size, num_planes, height, width] containing
 * feature representations of game states. Memory is allocated as CUDA pinned
 * host memory for efficient GPU transfer.
 *
 * @param states Vector of game state pointers (NOT owned, read-only access)
 * @param game_type Game type enum (Gomoku, Chess, Go)
 * @return PyCapsule* containing DLManagedTensor (ownership transferred to Python)
 *
 * Memory ownership:
 * - Caller retains ownership of states
 * - Returned capsule owns allocated tensor memory
 * - PyTorch takes ownership when consuming capsule
 * - Deleter automatically called when last PyTorch tensor reference released
 *
 * Thread safety: Safe to call from multiple threads with different state batches
 * Performance: <0.5ms for batch_size=64 (target)
 *
 * Example:
 *   std::vector<const IGameState*> states = {...};
 *   PyObject* capsule = create_batch_tensor(states, GameType::GOMOKU);
 *   // In Python: tensor = torch.from_dlpack(capsule)
 */
PyObject* create_batch_tensor(
    const std::vector<const IGameState*>& states,
    GameType game_type
);

/**
 * @brief Get tensor shape information for a game type
 *
 * @param game_type Game type enum
 * @param batch_size Number of states in batch
 * @return TensorShape struct with dimensions
 */
struct TensorShape {
    int64_t batch_size;
    int64_t num_planes;   // Gomoku=36, Chess=30, Go=25
    int64_t height;       // Gomoku=15, Chess=8, Go=19
    int64_t width;

    size_t total_elements() const {
        return batch_size * num_planes * height * width;
    }

    size_t size_bytes() const {
        return total_elements() * sizeof(float);
    }
};

TensorShape get_tensor_shape(GameType game_type, size_t batch_size);

} // namespace mcts
```

#### 2. Memory Management

```cpp
namespace mcts {

/**
 * @brief Pinned memory buffer for DLPack tensors
 *
 * Manages CUDA pinned host memory allocation for fast GPU transfers.
 * Implements reference counting for shared ownership between C++ and PyTorch.
 */
class PinnedBuffer {
public:
    /**
     * @brief Allocate pinned memory buffer
     *
     * @param size_bytes Size in bytes
     * @param use_cuda If true, allocate CUDA pinned memory; else regular malloc
     * @throws std::bad_alloc if allocation fails
     */
    PinnedBuffer(size_t size_bytes, bool use_cuda = true);

    /**
     * @brief Get raw data pointer
     * @return Pointer to buffer data (CPU-accessible)
     */
    void* data() { return data_; }
    const void* data() const { return data_; }

    /**
     * @brief Get buffer size
     * @return Size in bytes
     */
    size_t size() const { return size_bytes_; }

    /**
     * @brief Check if using CUDA pinned memory
     */
    bool is_pinned() const { return use_cuda_; }

    /**
     * @brief Increment reference count
     *
     * Called when PyTorch tensor shares this buffer.
     * Thread-safe via atomic operations.
     */
    void add_ref();

    /**
     * @brief Decrement reference count and free if zero
     *
     * Called by DLPack deleter when PyTorch releases tensor.
     * Thread-safe via atomic operations.
     *
     * @return true if buffer was freed
     */
    bool release();

    // Non-copyable, movable
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&&) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&&) noexcept;

    ~PinnedBuffer();

private:
    void* data_{nullptr};
    size_t size_bytes_{0};
    bool use_cuda_{false};
    std::atomic<int> ref_count_{1};  // Starts at 1 (C++ owns initially)
};

/**
 * @brief Buffer pool for reusing pinned memory allocations
 *
 * Reduces allocation overhead by caching buffers of common sizes.
 * Thread-safe with lock-free design where possible.
 */
class BufferPool {
public:
    /**
     * @brief Get singleton instance
     */
    static BufferPool& instance();

    /**
     * @brief Acquire buffer of at least min_size bytes
     *
     * Returns cached buffer if available, otherwise allocates new.
     *
     * @param min_size Minimum size in bytes
     * @param use_cuda Allocate CUDA pinned memory
     * @return Shared pointer to buffer
     */
    std::shared_ptr<PinnedBuffer> acquire(size_t min_size, bool use_cuda = true);

    /**
     * @brief Return buffer to pool for reuse
     *
     * Buffer is cached if within size limits, otherwise freed.
     *
     * @param buffer Buffer to return (must be sole owner)
     */
    void release(std::shared_ptr<PinnedBuffer> buffer);

    /**
     * @brief Clear all cached buffers
     *
     * Frees all buffers in pool. Call during shutdown.
     */
    void clear();

    /**
     * @brief Get pool statistics
     */
    struct Stats {
        size_t num_cached;
        size_t total_bytes_cached;
        size_t num_allocations;
        size_t num_cache_hits;
        size_t num_cache_misses;
    };

    Stats get_stats() const;

private:
    BufferPool() = default;

    // Pool organized by size classes (4KB, 64KB, 1MB, 4MB)
    static constexpr size_t SIZE_CLASSES[] = {4096, 65536, 1048576, 4194304};
    static constexpr size_t MAX_CACHED_PER_CLASS = 8;

    // Lock-free MPSC queue per size class
    // Implementation details omitted for brevity
};

} // namespace mcts
```

#### 3. DLPack Capsule Management

```cpp
namespace mcts {

/**
 * @brief Context for DLPack deleter callback
 *
 * Stores information needed to clean up tensor when PyTorch releases it.
 */
struct DLPackContext {
    std::shared_ptr<PinnedBuffer> buffer;  // Owns the memory
    int64_t* shape_storage;                // Heap-allocated shape array
    int64_t* strides_storage;              // Heap-allocated strides array (if used)

    ~DLPackContext() {
        delete[] shape_storage;
        delete[] strides_storage;
        // buffer automatically freed via shared_ptr
    }
};

/**
 * @brief DLPack deleter callback
 *
 * Called by PyTorch when tensor is destroyed. Frees DLManagedTensor structure
 * and decrements buffer reference count.
 *
 * @param self Pointer to DLManagedTensor being deleted
 */
void dlpack_deleter(DLManagedTensor* self);

/**
 * @brief Create DLManagedTensor from buffer and metadata
 *
 * Internal function used by create_batch_tensor().
 *
 * @param buffer Pinned buffer containing tensor data
 * @param shape Tensor shape (batch, planes, height, width)
 * @param device_type DLPack device type (kDLCPU or kDLCUDAHost)
 * @return DLManagedTensor* (caller owns, must be wrapped in PyCapsule)
 */
DLManagedTensor* create_dlpack_tensor(
    std::shared_ptr<PinnedBuffer> buffer,
    const TensorShape& shape,
    DLDeviceType device_type = kDLCPUPinned
);

/**
 * @brief Wrap DLManagedTensor in PyCapsule
 *
 * Creates Python capsule object for passing to torch.from_dlpack().
 *
 * @param tensor DLManagedTensor (ownership transferred to capsule)
 * @return PyObject* (PyCapsule with "dltensor" name)
 */
PyObject* wrap_dlpack_capsule(DLManagedTensor* tensor);

} // namespace mcts
```

## Memory Ownership Semantics

### Ownership Flow

```
1. C++ allocates PinnedBuffer (ref_count = 1)
       ↓
2. C++ creates DLManagedTensor, stores buffer in context
       ↓
3. C++ wraps in PyCapsule, returns to Python
       ↓
4. Python calls torch.from_dlpack(capsule)
       ↓
5. PyTorch extracts DLManagedTensor, creates tensor view
       ↓
6. Capsule deleter called, frees DLManagedTensor structure
       ↓
7. PyTorch tensor holds buffer reference (ref_count still 1)
       ↓
8. PyTorch tensor destroyed → buffer.release() → ref_count = 0
       ↓
9. Buffer freed (cudaFreeHost or free)
```

### Lifetime Guarantees

| Object | Lifetime | Owner |
|--------|----------|-------|
| `IGameState` | Function call only | Caller (read-only access) |
| `PinnedBuffer` | Until last tensor reference | Shared (ref counted) |
| `DLManagedTensor` | Until capsule consumed | Python capsule |
| `DLPackContext` | Until tensor destroyed | DLManagedTensor |
| `shape/strides` | Until tensor destroyed | DLPackContext |

### Thread Safety

- **`create_batch_tensor()`**: Thread-safe, can be called concurrently with different state batches
- **`PinnedBuffer`**: Thread-safe reference counting via atomics
- **`BufferPool`**: Thread-safe acquire/release operations
- **No shared mutable state** between different tensor creations

## Feature Extraction Interface

Game states must implement direct feature extraction to pre-allocated buffer:

```cpp
namespace alphazero::core {

class IGameState {
public:
    /**
     * @brief Extract features directly to buffer
     *
     * Writes tensor representation to pre-allocated buffer in row-major layout.
     * Layout: [num_planes, height, width] with contiguous memory.
     *
     * @param buffer Float buffer of size (num_planes * height * width)
     *
     * Requirements:
     * - No heap allocations
     * - Thread-safe (read-only state access)
     * - Performance: <10μs per state
     * - Deterministic output (same state → same features)
     *
     * Example for Gomoku (36 planes, 15x15):
     *   float buffer[36 * 15 * 15];
     *   state->extract_features_to_buffer(buffer);
     *   // buffer[0..224] = plane 0 (current player stones)
     *   // buffer[225..449] = plane 1 (opponent stones)
     *   // etc.
     */
    virtual void extract_features_to_buffer(float* buffer) const = 0;

    /**
     * @brief Get number of feature planes for this game type
     */
    virtual int get_num_feature_planes() const = 0;
};

} // namespace alphazero::core
```

## Error Handling

### Error Conditions

1. **Allocation Failure**
   - Throw: `std::bad_alloc`
   - Recovery: Caller catches and uses fallback (numpy conversion)

2. **Invalid Game Type**
   - Throw: `std::invalid_argument`
   - Recovery: Validate game type before calling

3. **Empty Batch**
   - Throw: `std::invalid_argument`
   - Recovery: Check `states.size() > 0` before calling

4. **CUDA Not Available**
   - Fallback: Use regular malloc instead of cudaMallocHost
   - Performance impact: GPU transfers slower but functional

### Error Messages

```cpp
// Example error handling
try {
    PyObject* capsule = create_batch_tensor(states, GameType::GOMOKU);
} catch (const std::bad_alloc& e) {
    // Fallback to numpy conversion
    PyObject* numpy_array = create_numpy_array(states, GameType::GOMOKU);
} catch (const std::invalid_argument& e) {
    // Log error and return null
    std::cerr << "Invalid batch tensor request: " << e.what() << std::endl;
    return nullptr;
}
```

## Performance Targets

| Operation | Target | Current (T-VALID-2) | Status |
|-----------|--------|---------------------|--------|
| `create_batch_tensor(64)` | <1.0ms | 7.50 ± 0.20 ms | ❌ NEEDS FIX |
| Feature extraction per state | <10μs | ~117μs (7.5ms / 64) | ❌ NEEDS OPENMP |
| Feature extraction per state (with OpenMP) | <10μs | <16μs (1ms / 64) | ⏳ PENDING FIX |
| Buffer allocation (cached) | <1μs | Not measured | N/A |
| Buffer allocation (new) | <100μs | Not measured | N/A |
| PyTorch capsule consumption | <50μs | Not measured | ✅ (validated functional) |
| **Total overhead** | <1ms | 7.5ms | ❌ **CRITICAL BLOCKER** |

## Integration with BatchInferenceCoordinator

```python
# Python usage (in BatchInferenceCoordinator callback)
def batch_inference(self, states: List[IGameState]) -> List[Tuple[np.ndarray, float]]:
    # C++ creates DLPack capsule
    capsule = mcts_py.create_batch_tensor(states, game_type)

    # Zero-copy conversion to PyTorch tensor
    tensor = torch.from_dlpack(capsule)  # CPU tensor, shape [N, C, H, W]

    # Transfer to GPU (pinned memory → fast transfer)
    gpu_tensor = tensor.to(self.device, non_blocking=True)

    # Neural network inference
    with torch.no_grad():
        policy_logits, value = self.model(gpu_tensor)

    # Results back to CPU
    policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()
    value = value.cpu().numpy()

    # Return as list of (policy, value) tuples
    return [(policy[i], value[i]) for i in range(len(states))]
```

## Testing Strategy

### Unit Tests

1. **Buffer Management**
   - Test allocation/deallocation
   - Test reference counting correctness
   - Test buffer pool reuse

2. **Capsule Creation**
   - Test metadata correctness (shape, strides, dtype)
   - Test deleter called on destruction
   - Test no memory leaks with valgrind

3. **PyTorch Integration**
   - Test `torch.from_dlpack()` compatibility
   - Test tensor data correctness
   - Test GPU transfer

### Integration Tests

1. **End-to-End Pipeline**
   - Test with real neural network
   - Verify inference results match numpy baseline
   - Measure performance improvement

2. **Error Handling**
   - Test with empty batch
   - Test with invalid game type
   - Test OOM scenarios

### Performance Benchmarks

1. **Creation Time**
   - Measure with batch sizes 1, 16, 32, 64, 128
   - Compare vs numpy conversion baseline

2. **Memory Profiler**
   - Verify zero memcpy() calls in hot path
   - Check buffer pool hit rate
   - Measure total memory usage

## Implementation Phases (T007b-T007g)

This design document supports the following implementation tasks:

- **T007b**: Implement `PinnedBuffer` and `BufferPool` classes
- **T007c**: Implement `DLManagedTensor` creation and deleter
- **T007d**: Implement `create_batch_tensor()` function
- **T007e**: Add `extract_features_to_buffer()` to game states
- **T007f**: Expose to Python via pybind11
- **T007g**: Validation and benchmarking

## References

- [DLPack Specification](https://github.com/dmlc/dlpack)
- [PyTorch DLPack Support](https://pytorch.org/docs/stable/dlpack.html)
- [CUDA Pinned Memory](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

## Assumptions

1. **PyTorch ≥2.0**: Assumes modern DLPack support
2. **Float32 Only**: No support for other dtypes initially
3. **Row-Major Layout**: PyTorch default memory layout
4. **CPU Tensors**: Initial implementation, GPU tensors future work
5. **Single Device**: No multi-GPU support initially

## Open Questions

1. **Buffer Pool Size Limits**: What's the maximum cache size? (Proposed: 64MB per size class)
2. **GPU Direct Support**: Should we support direct GPU memory in Phase 1? (Proposed: No, CPU pinned memory sufficient)
3. **Batch Size Limits**: Maximum batch size to support? (Proposed: 256, covers all use cases)

---

**Document Status**: ✅ Design Complete, Ready for Implementation (T007b)
