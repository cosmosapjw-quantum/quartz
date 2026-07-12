/**
 * @file dlpack_bridge.hpp
 * @brief Zero-copy tensor bridge between C++ and PyTorch via DLPack protocol
 *
 * This module implements the DLPack tensor bridge for eliminating Python data
 * conversion overhead. Key features:
 * - CUDA pinned memory allocation for fast GPU transfers
 * - Reference-counted buffer management with thread-safe operations
 * - Buffer pool with size classes (4KB, 64KB, 1MB, 4MB)
 * - Zero-copy tensor exchange via DLManagedTensor capsules
 *
 * Performance targets (T007):
 * - Batch tensor creation: <0.5ms for batch_size=64
 * - Feature extraction: <10μs per state
 * - Expected speedup: 1.25× vs numpy conversion
 *
 * Memory ownership:
 * - C++ allocates → PyCapsule → torch.from_dlpack() → shared reference
 * - Reference-counted cleanup when both sides release
 * - Thread-safe throughout
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <atomic>
#include <vector>
#include <optional>

// Forward declare IGameState for T007e
namespace alphazero {
namespace core {
class IGameState;
}
}

namespace mcts {

/**
 * @brief CUDA pinned memory buffer with reference counting
 *
 * Provides thread-safe reference-counted memory buffer that can be allocated
 * as CUDA pinned memory (cudaMallocHost) for fast GPU transfers or regular
 * heap memory as fallback.
 *
 * Memory Ownership:
 * - Reference counting starts at 1 (creator owns initial reference)
 * - add_ref() atomically increments count (thread-safe)
 * - release() atomically decrements, frees memory when count reaches 0
 *
 * Thread Safety:
 * - All operations are thread-safe via atomic reference count
 * - Safe to share across multiple threads and Python GIL
 *
 * Performance:
 * - CUDA pinned memory: 2-3× faster GPU transfers vs pageable memory
 * - Reference counting: lock-free atomic operations
 * - Reuse via BufferPool: amortizes allocation overhead
 */
class PinnedBuffer {
public:
    /**
     * @brief Allocate pinned memory buffer
     *
     * Allocates CUDA pinned memory if available, falls back to regular malloc.
     * Initial reference count is 1.
     *
     * @param size_bytes Size of buffer in bytes
     * @param use_cuda Try to allocate CUDA pinned memory (true) or force malloc (false)
     * @throws std::bad_alloc if allocation fails
     */
    PinnedBuffer(size_t size_bytes, bool use_cuda = true);

    /**
     * @brief Destructor - frees memory if reference count is 0
     *
     * WARNING: Should only be called when ref_count_ == 0.
     * Use release() for normal cleanup.
     */
    ~PinnedBuffer();

    // Non-copyable (owns unique memory)
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    // Non-movable (reference counting requires stable address)
    PinnedBuffer(PinnedBuffer&&) = delete;
    PinnedBuffer& operator=(PinnedBuffer&&) = delete;

    /**
     * @brief Get raw pointer to buffer data
     * @return Pointer to buffer memory
     */
    void* data() { return data_; }
    const void* data() const { return data_; }

    /**
     * @brief Get buffer size in bytes
     * @return Size of allocated buffer
     */
    size_t size() const { return size_bytes_; }

    /**
     * @brief Check if buffer is CUDA pinned memory
     * @return true if allocated via cudaMallocHost, false if regular malloc
     */
    bool is_cuda_pinned() const { return is_cuda_pinned_; }

    /**
     * @brief Get current reference count (for debugging)
     *
     * Note: This returns the shared_ptr use count, not internal ref count.
     * Only meaningful when called from Python on buffers managed by shared_ptr.
     *
     * @return Current reference count
     */
    int ref_count() const { return 1; }  // Placeholder for shared_ptr use_count()

private:
    void* data_{nullptr};                      // Buffer memory
    size_t size_bytes_{0};                     // Buffer size
    bool is_cuda_pinned_{false};               // CUDA pinned vs malloc

    /**
     * @brief Free buffer memory
     * Called by destructor or release() when ref_count_ reaches 0
     */
    void free_memory();
};

/**
 * @brief Buffer pool with size classes for efficient reuse
 *
 * Maintains pools of pre-allocated buffers in common sizes to amortize
 * allocation overhead. Thread-safe singleton.
 *
 * Size Classes (power-of-2 aligned):
 * - Tiny:   4 KB   (1 game state)
 * - Small:  64 KB  (16-32 states)
 * - Medium: 1 MB   (64-128 states)
 * - Large:  4 MB   (256+ states)
 *
 * Performance:
 * - acquire() first checks pool cache (O(1) if available)
 * - Cache miss allocates new buffer (~100μs for CUDA pinned memory)
 * - release() returns buffer to pool for reuse
 * - Target: 90%+ cache hit rate during steady state
 *
 * Memory Management:
 * - Buffers returned to pool are kept alive for reuse
 * - Pool shrinks if memory pressure detected
 * - Maximum pool size configurable (default: 16 buffers per size class)
 *
 * Thread Safety:
 * - All operations protected by mutex
 * - Safe to call from multiple threads concurrently
 */
class BufferPool {
public:
    /**
     * @brief Get singleton instance
     * @return Reference to global BufferPool instance
     */
    static BufferPool& instance();

    /**
     * @brief Acquire buffer from pool or allocate new one
     *
     * Searches pool for cached buffer of appropriate size. If found, returns
     * immediately. Otherwise allocates new buffer.
     *
     * @param min_size Minimum required size in bytes
     * @param use_cuda Prefer CUDA pinned memory (true) or regular malloc (false)
     * @return Shared pointer to buffer with reference count 1
     * @throws std::bad_alloc if allocation fails
     */
    std::shared_ptr<PinnedBuffer> acquire(size_t min_size, bool use_cuda = true);

    /**
     * @brief Return buffer to pool for reuse
     *
     * If buffer reference count is 1 (only pool holds reference) and pool
     * has space, caches buffer for reuse. Otherwise lets buffer be freed.
     *
     * @param buffer Buffer to return (must have ref_count == 1)
     */
    void release(std::shared_ptr<PinnedBuffer> buffer);

    /**
     * @brief Clear all cached buffers (for testing/cleanup)
     */
    void clear();

    /**
     * @brief Get pool statistics (for monitoring/debugging)
     */
    struct Stats {
        size_t total_allocated{0};    // Lifetime allocations
        size_t total_reused{0};       // Cache hits
        size_t current_pooled{0};     // Buffers in pool now
        size_t current_bytes{0};      // Total bytes in pool
    };
    Stats get_stats() const;

    /**
     * @brief Configure maximum buffers per size class
     * @param max_buffers Maximum buffers to cache per size (default: 16)
     */
    void set_max_buffers_per_class(size_t max_buffers);

private:
    BufferPool() = default;
    ~BufferPool() = default;

    // Non-copyable, non-movable (singleton)
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    /**
     * @brief Size classes for pooling (power-of-2 aligned)
     */
    enum class SizeClass {
        TINY = 0,    // 4 KB
        SMALL = 1,   // 64 KB
        MEDIUM = 2,  // 1 MB
        LARGE = 3    // 4 MB
    };

    static constexpr size_t NUM_SIZE_CLASSES = 4;
    static constexpr size_t SIZE_CLASS_BYTES[NUM_SIZE_CLASSES] = {
        4 * 1024,       // 4 KB
        64 * 1024,      // 64 KB
        1024 * 1024,    // 1 MB
        4 * 1024 * 1024 // 4 MB
    };

    /**
     * @brief Get size class for requested size
     * @param size Requested size in bytes
     * @return Size class that can hold this size, or std::nullopt if too large
     */
    std::optional<SizeClass> get_size_class(size_t size) const;

    /**
     * @brief Get actual buffer size for a size class
     * @param sc Size class
     * @return Buffer size in bytes
     */
    size_t get_buffer_size(SizeClass sc) const;

    // Pool state (protected by mutex)
    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<PinnedBuffer>> pools_[NUM_SIZE_CLASSES];
    size_t max_buffers_per_class_{16};

    // Statistics
    mutable std::atomic<size_t> total_allocated_{0};
    mutable std::atomic<size_t> total_reused_{0};
};

/**
 * @brief Check if CUDA is available for pinned memory allocation
 * @return true if CUDA runtime is available, false otherwise
 */
bool is_cuda_available();

// ============================================================================
// DLPack Tensor Capsule API (T007c, T007d)
// ============================================================================

// Forward declare DLPack types
struct DLTensor;
struct DLManagedTensor;

/**
 * @brief Game type enumeration (T007d)
 *
 * Defines supported game types with their feature plane counts and board dimensions.
 */
enum class GameType {
    GOMOKU,  // 36 planes, 15×15 board
    CHESS,   // 30 planes, 8×8 board
    GO       // 25 planes, 19×19 board
};

/**
 * @brief Get number of feature planes for a game type
 */
int get_num_planes(GameType game_type);

/**
 * @brief Get board dimensions for a game type
 * @return Pair of (height, width)
 */
std::pair<int, int> get_board_size(GameType game_type);

/**
 * @brief Context for DLPack deleter callback
 *
 * Stores ownership information for tensor memory and metadata.
 * Cleaned up when PyTorch releases the tensor.
 */
struct DLPackContext {
    std::shared_ptr<PinnedBuffer> buffer;  // Owns the data memory
    int64_t* shape_storage{nullptr};       // Heap-allocated shape array
    int64_t* strides_storage{nullptr};     // Heap-allocated strides array (optional)

    ~DLPackContext();
};

/**
 * @brief Tensor shape metadata
 */
struct TensorShape {
    int64_t batch_size;
    int64_t num_planes;
    int64_t height;
    int64_t width;
};

/**
 * @brief DLPack deleter callback
 *
 * Called by PyTorch when tensor is destroyed. Frees DLManagedTensor structure
 * and releases buffer reference.
 *
 * @param self Pointer to DLManagedTensor being deleted
 */
void dlpack_deleter(DLManagedTensor* self);

/**
 * @brief Create DLManagedTensor from buffer and metadata
 *
 * Creates DLPack tensor structure that wraps pre-allocated pinned memory.
 * The tensor takes shared ownership of the buffer via reference counting.
 *
 * @param buffer Pinned buffer containing tensor data
 * @param shape Tensor shape (batch, planes, height, width)
 * @param use_cuda Whether buffer is CUDA pinned memory
 * @return DLManagedTensor* (caller must wrap in PyCapsule and pass to Python)
 * @throws std::bad_alloc if allocation fails
 */
DLManagedTensor* create_dlpack_tensor(
    std::shared_ptr<PinnedBuffer> buffer,
    const TensorShape& shape,
    bool use_cuda = false
);

/**
 * @brief Create batch tensor from game states (T007d)
 *
 * Allocates pinned memory buffer and creates DLPack tensor for a batch of game states.
 * Features are extracted directly into the buffer for zero-copy transfer to PyTorch.
 *
 * NOTE: This is a simplified stub implementation for T007d. Full feature extraction
 * will be implemented in T007e when IGameState::extract_features_to_buffer() is added.
 *
 * @param batch_size Number of states in batch
 * @param game_type Game type (GOMOKU/CHESS/GO)
 * @param use_cuda Use CUDA pinned memory for faster GPU transfers
 * @return DLManagedTensor* ready to wrap in PyCapsule
 * @throws std::bad_alloc if buffer allocation fails
 * @throws std::invalid_argument if batch_size <= 0
 *
 * Example:
 *   auto* tensor = create_batch_tensor(64, GameType::GOMOKU, true);
 *   PyObject* capsule = wrap_dlpack_capsule(tensor);
 *   // Pass capsule to torch.from_dlpack()
 */
DLManagedTensor* create_batch_tensor(
    int batch_size,
    GameType game_type,
    bool use_cuda = false
);

/**
 * @brief Create batch tensor with feature extraction from game states (T007e)
 *
 * Allocates pinned memory buffer and extracts features from game states directly
 * into the buffer for zero-copy transfer to PyTorch.
 *
 * @param states Vector of game state pointers (NOT owned, read-only access)
 * @param use_cuda Use CUDA pinned memory for faster GPU transfers
 * @return DLManagedTensor* ready to wrap in PyCapsule
 * @throws std::bad_alloc if buffer allocation fails
 * @throws std::invalid_argument if states is empty
 * @throws std::runtime_error if states have inconsistent game types
 *
 * Example:
 *   std::vector<const alphazero::core::IGameState*> states = {...};
 *   auto* tensor = create_batch_tensor_from_states(states, true);
 *   PyObject* capsule = wrap_dlpack_capsule(tensor);
 *   // Pass capsule to torch.from_dlpack()
 */
DLManagedTensor* create_batch_tensor_from_states(
    const std::vector<const alphazero::core::IGameState*>& states,
    bool use_cuda = false
);

// Note: wrap_dlpack_capsule() is implemented in python_bindings.cpp
// since it requires Python.h headers. See python_bindings.cpp for usage.

} // namespace mcts
