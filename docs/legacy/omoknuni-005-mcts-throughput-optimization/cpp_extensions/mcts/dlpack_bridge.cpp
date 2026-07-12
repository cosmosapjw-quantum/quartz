/**
 * @file dlpack_bridge.cpp
 * @brief Implementation of DLPack tensor bridge components
 */

#include "dlpack_bridge.hpp"
#include "profiling/enhanced_profiler.hpp"
#include "utils/igamestate.h"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <optional>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace mcts::profiling;

// CUDA headers (with availability detection)
#ifdef __has_include
#  if __has_include(<cuda_runtime.h>)
#    define HAS_CUDA 1
#    include <cuda_runtime.h>
#  else
#    define HAS_CUDA 0
#  endif
#else
#  define HAS_CUDA 0
#endif

namespace mcts {

// ============================================================================
// CUDA Availability Detection
// ============================================================================

bool is_cuda_available() {
#if HAS_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

// ============================================================================
// PinnedBuffer Implementation
// ============================================================================

PinnedBuffer::PinnedBuffer(size_t size_bytes, bool use_cuda)
    : size_bytes_(size_bytes) {

    if (size_bytes == 0) {
        throw std::invalid_argument("PinnedBuffer: size_bytes must be > 0");
    }

    // Try CUDA pinned memory if requested and available
    bool allocated = false;
    if (use_cuda && is_cuda_available()) {
#if HAS_CUDA
        cudaError_t err = cudaMallocHost(&data_, size_bytes);
        if (err == cudaSuccess) {
            is_cuda_pinned_ = true;
            allocated = true;
        }
        // Fall through to malloc on failure
#endif
    }

    // Fallback to regular malloc
    if (!allocated) {
        data_ = std::malloc(size_bytes);
        if (!data_) {
            throw std::bad_alloc();
        }
        is_cuda_pinned_ = false;
    }
}

PinnedBuffer::~PinnedBuffer() {
    free_memory();
}

void PinnedBuffer::free_memory() {
    if (!data_) {
        return;
    }

    if (is_cuda_pinned_) {
#if HAS_CUDA
        cudaFreeHost(data_);
#endif
    } else {
        std::free(data_);
    }
    data_ = nullptr;
}

// ============================================================================
// BufferPool Implementation
// ============================================================================

BufferPool& BufferPool::instance() {
    static BufferPool pool;
    return pool;
}

std::optional<BufferPool::SizeClass> BufferPool::get_size_class(size_t size) const {
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        if (size <= SIZE_CLASS_BYTES[i]) {
            return static_cast<SizeClass>(i);
        }
    }
    return std::nullopt;  // Too large for pooling
}

size_t BufferPool::get_buffer_size(SizeClass sc) const {
    return SIZE_CLASS_BYTES[static_cast<size_t>(sc)];
}

std::shared_ptr<PinnedBuffer> BufferPool::acquire(size_t min_size, bool use_cuda) {
    auto size_class_opt = get_size_class(min_size);

    // If size is within poolable range, try to reuse from pool
    if (size_class_opt.has_value()) {
        SizeClass sc = size_class_opt.value();
        size_t class_idx = static_cast<size_t>(sc);

        std::lock_guard<std::mutex> lock(mutex_);

        // Check if we have a cached buffer
        if (!pools_[class_idx].empty()) {
            auto buffer = pools_[class_idx].back();
            pools_[class_idx].pop_back();

            total_reused_.fetch_add(1, std::memory_order_relaxed);
            return buffer;
        }
    }

    // Cache miss or too large - allocate new buffer
    size_t alloc_size = size_class_opt.has_value()
        ? get_buffer_size(size_class_opt.value())
        : min_size;

    total_allocated_.fetch_add(1, std::memory_order_relaxed);

    // Simple allocation - pool reuse is manual via release()
    return std::make_shared<PinnedBuffer>(alloc_size, use_cuda);
}

void BufferPool::release(std::shared_ptr<PinnedBuffer> buffer) {
    if (!buffer) {
        return;
    }

    // Check if buffer is poolable size
    auto size_class_opt = get_size_class(buffer->size());
    if (!size_class_opt.has_value()) {
        // Too large for pooling, just let it be deleted
        return;
    }

    SizeClass sc = size_class_opt.value();
    size_t class_idx = static_cast<size_t>(sc);

    std::lock_guard<std::mutex> lock(mutex_);

    // Only cache if pool has space
    if (pools_[class_idx].size() < max_buffers_per_class_) {
        pools_[class_idx].push_back(buffer);
    }
    // Otherwise let buffer be deleted when shared_ptr goes out of scope
}

void BufferPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        pools_[i].clear();
    }
}

BufferPool::Stats BufferPool::get_stats() const {
    Stats stats;
    stats.total_allocated = total_allocated_.load(std::memory_order_relaxed);
    stats.total_reused = total_reused_.load(std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        stats.current_pooled += pools_[i].size();
        for (const auto& buffer : pools_[i]) {
            stats.current_bytes += buffer->size();
        }
    }

    return stats;
}

void BufferPool::set_max_buffers_per_class(size_t max_buffers) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_buffers_per_class_ = max_buffers;

    // Trim pools if they exceed new limit
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        if (pools_[i].size() > max_buffers) {
            pools_[i].resize(max_buffers);
        }
    }
}

// ============================================================================
// DLPack Tensor Capsule Implementation (T007c)
// ============================================================================

// DLPack types and constants (from dlpack v0.8)
extern "C" {

typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLCUDAManaged = 13,
} DLDeviceType;

typedef struct {
    int device_type;
    int device_id;
} DLDevice;

typedef enum {
    kDLFloat = 2,
    kDLUInt = 1,
    kDLInt = 0,
} DLDataTypeCode;

typedef struct {
    uint8_t code;    // DLDataTypeCode
    uint8_t bits;    // Number of bits
    uint16_t lanes;  // Number of lanes (SIMD)
} DLDataType;

typedef struct DLTensor {
    void* data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;  // Can be NULL for row-major
    uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

} // extern "C"

// DLPackContext destructor
DLPackContext::~DLPackContext() {
    delete[] shape_storage;
    delete[] strides_storage;
    // buffer automatically freed via shared_ptr
}

// DLPack deleter callback
void dlpack_deleter(DLManagedTensor* self) {
    if (!self) {
        return;
    }

    // Free context (which frees shape/strides and releases buffer)
    auto* context = static_cast<DLPackContext*>(self->manager_ctx);
    delete context;

    // Free DLManagedTensor structure itself
    delete self;
}

// Create DLManagedTensor from buffer and metadata
DLManagedTensor* create_dlpack_tensor(
    std::shared_ptr<PinnedBuffer> buffer,
    const TensorShape& shape,
    bool use_cuda) {

    if (!buffer) {
        throw std::invalid_argument("create_dlpack_tensor: buffer is null");
    }

    // Allocate context (owns metadata and buffer reference)
    auto* context = new DLPackContext();
    context->buffer = buffer;  // Increment buffer ref count

    // Allocate shape array (4D tensor: batch, planes, height, width)
    context->shape_storage = new int64_t[4];
    context->shape_storage[0] = shape.batch_size;
    context->shape_storage[1] = shape.num_planes;
    context->shape_storage[2] = shape.height;
    context->shape_storage[3] = shape.width;

    // No strides (NULL means row-major)
    context->strides_storage = nullptr;

    // Allocate DLManagedTensor
    auto* managed_tensor = new DLManagedTensor();
    managed_tensor->manager_ctx = context;
    managed_tensor->deleter = dlpack_deleter;

    // Fill in DLTensor metadata
    DLTensor& dl_tensor = managed_tensor->dl_tensor;
    dl_tensor.data = buffer->data();
    dl_tensor.ndim = 4;
    dl_tensor.shape = context->shape_storage;
    dl_tensor.strides = nullptr;  // Row-major
    dl_tensor.byte_offset = 0;

    // Device: CPU (always use kDLCPU for CPU-side memory)
    // NOTE: Even for CUDA pinned memory, use kDLCPU since PyTorch's from_dlpack()
    // doesn't properly support kDLCUDAHost (device_type=3) in some versions.
    // Pinned memory is still on the host (CPU), so kDLCPU is technically correct.
    // PyTorch will detect it's pinned and use fast H2D transfers automatically.
    dl_tensor.device.device_type = kDLCPU;
    dl_tensor.device.device_id = 0;

    // Data type: float32
    dl_tensor.dtype.code = kDLFloat;
    dl_tensor.dtype.bits = 32;
    dl_tensor.dtype.lanes = 1;

    return managed_tensor;
}

// ============================================================================
// Batch Tensor Creation API (T007d)
// ============================================================================

// Helper functions for game type metadata
int get_num_planes(GameType game_type) {
    switch (game_type) {
        case GameType::GOMOKU: return 36;  // Enhanced Gomoku features
        case GameType::CHESS:  return 30;  // Complete Chess state
        case GameType::GO:     return 25;  // Enhanced Go features
        default:
            throw std::invalid_argument("Unknown game type");
    }
}

std::pair<int, int> get_board_size(GameType game_type) {
    switch (game_type) {
        case GameType::GOMOKU: return {15, 15};  // 15×15 board
        case GameType::CHESS:  return {8, 8};    // 8×8 board
        case GameType::GO:     return {19, 19};  // 19×19 board
        default:
            throw std::invalid_argument("Unknown game type");
    }
}

// Create batch tensor from game states
DLManagedTensor* create_batch_tensor(
    int batch_size,
    GameType game_type,
    bool use_cuda) {

    if (batch_size <= 0) {
        throw std::invalid_argument("create_batch_tensor: batch_size must be > 0");
    }

    // Get game-specific dimensions
    int num_planes = get_num_planes(game_type);
    auto [height, width] = get_board_size(game_type);

    // Calculate buffer size: batch × planes × height × width × sizeof(float)
    size_t num_elements = static_cast<size_t>(batch_size) * num_planes * height * width;
    size_t buffer_size = num_elements * sizeof(float);

    // Acquire buffer from pool
    auto buffer = BufferPool::instance().acquire(buffer_size, use_cuda);

    // Initialize buffer to zeros for now (T007e will fill with actual features)
    // NOTE: This is a stub implementation. Real feature extraction happens in T007e.
    std::memset(buffer->data(), 0, buffer_size);

    // Create tensor shape
    TensorShape shape;
    shape.batch_size = batch_size;
    shape.num_planes = num_planes;
    shape.height = height;
    shape.width = width;

    // Create DLPack tensor
    return create_dlpack_tensor(buffer, shape, use_cuda);
}

/**
 * @brief Create batch tensor with actual feature extraction from game states (T007e)
 */
DLManagedTensor* create_batch_tensor_from_states(
    const std::vector<const alphazero::core::IGameState*>& states,
    bool use_cuda) {
    PROFILE_SCOPE(ProfileMetric::TensorCreationOverhead);

    if (states.empty()) {
        throw std::invalid_argument("create_batch_tensor_from_states: states cannot be empty");
    }

    // Get dimensions from first state
    int batch_size = static_cast<int>(states.size());
    int num_planes = states[0]->get_num_feature_planes();
    int height = states[0]->getBoardSize();
    int width = height;  // Assume square boards for now

    // Validate all states are consistent
    for (size_t i = 1; i < states.size(); ++i) {
        if (states[i]->get_num_feature_planes() != num_planes) {
            throw std::runtime_error("create_batch_tensor_from_states: inconsistent num_planes across states");
        }
        if (states[i]->getBoardSize() != height) {
            throw std::runtime_error("create_batch_tensor_from_states: inconsistent board_size across states");
        }
    }

    // Calculate buffer size: batch × planes × height × width × sizeof(float)
    size_t num_elements = static_cast<size_t>(batch_size) * num_planes * height * width;
    size_t buffer_size = num_elements * sizeof(float);

    // Acquire buffer from pool
    auto buffer = BufferPool::instance().acquire(buffer_size, use_cuda);

    // Extract features from each state directly into buffer
    float* data = static_cast<float*>(buffer->data());
    size_t state_size = num_planes * height * width;

    // Parallelize feature extraction with OpenMP
    // CRITICAL: This is a major bottleneck (review.txt lines 22-34)
    // Expected: <1ms, Actual: 7.5ms (OpenMP may not be parallelizing)
    // Use static scheduling for predictable load distribution
    // Only parallelize if batch_size > 8 to avoid threading overhead

    int omp_threads = 1;  // Default if OpenMP disabled
    int omp_max_threads = 1;

#ifdef _OPENMP
    // Track OpenMP configuration BEFORE parallel region
    omp_max_threads = omp_get_max_threads();
    PROFILE_GAUGE(ProfileMetric::OMP_ThreadCount, omp_max_threads);
#else
    // OpenMP disabled at compile time - this is a problem!
    PROFILE_COUNTER(ProfileMetric::FeatureExtractionOpenMP, 0);  // Failed
#endif

    {
        PROFILE_SCOPE(ProfileMetric::FeatureExtractionTotal);

        // Track if parallel execution should happen
        #ifdef _OPENMP
        bool should_parallelize = (batch_size > 8);
        // Reset to 1 before parallel region to detect if it actually ran
        omp_threads = 1;
        #endif

        // T019: Use parallel region with manual loop to detect thread count
        #ifdef _OPENMP
        if (should_parallelize) {
            #pragma omp parallel
            {
                // Get actual thread count FROM INSIDE parallel region (CRITICAL FIX)
                #pragma omp single
                {
                    omp_threads = omp_get_num_threads();  // Actual threads used (NOT max available)
                }

                // Distribute work manually with static scheduling
                #pragma omp for schedule(static)
                for (int i = 0; i < batch_size; ++i) {
                    // T019: REMOVED PROFILE_SCOPE from inside loop - prevents parallelization
                    // Profile the entire batch instead (see line 458 above)
                    float* state_buffer = data + (i * state_size);
                    states[i]->extract_features_to_buffer(state_buffer);
                }
            }
        } else {
        #endif
            // Serial execution (batch_size <= 8 or OpenMP disabled)
            for (int i = 0; i < batch_size; ++i) {
                float* state_buffer = data + (i * state_size);
                states[i]->extract_features_to_buffer(state_buffer);
            }
        #ifdef _OPENMP
        }
        #endif

        // Thread count now correctly set from inside parallel region
        // No need for additional checks here
    }

    // Verify OpenMP actually parallelized (CRITICAL CHECK from review.txt)
#ifdef _OPENMP
    if (batch_size > 8) {
        // Should have used multiple threads
        if (omp_threads == 1) {
            // WARNING: OpenMP NOT parallelizing despite batch_size > 8!
            PROFILE_COUNTER(ProfileMetric::FeatureExtractionOpenMP, 0);  // Failed
            PROFILE_COUNTER(ProfileMetric::FeatureExtractionSerial, batch_size);
        } else {
            // Success: OpenMP parallelized
            PROFILE_COUNTER(ProfileMetric::FeatureExtractionOpenMP, 1);  // Success
            PROFILE_GAUGE(ProfileMetric::OMP_ThreadCount, omp_threads);

            // Track work distribution (simplified: static schedule distributes evenly)
            float actual_work_variance = 0.0f;
            PROFILE_GAUGE(ProfileMetric::OMP_WorkDistribution, actual_work_variance);
        }
    }
#endif

    // Create tensor shape
    TensorShape shape;
    shape.batch_size = batch_size;
    shape.num_planes = num_planes;
    shape.height = height;
    shape.width = width;

    // Create DLPack tensor
    DLManagedTensor* result;
    {
        PROFILE_SCOPE(ProfileMetric::ExpansionDLPackConversion);
        result = create_dlpack_tensor(buffer, shape, use_cuda);
    }

    return result;
}

// Note: wrap_dlpack_capsule() is implemented in python_bindings.cpp
// to avoid Python.h dependency in core library

} // namespace mcts
