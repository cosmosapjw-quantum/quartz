#pragma once

#include <cstddef>
#include <cstdint>

namespace alphazero {
namespace core {

/**
 * ThreadLocalArena - High-performance thread-local memory allocator
 *
 * Optimized for MCTS node allocation with:
 * - Bump pointer allocation (O(1), ~1.5ns)
 * - 64-byte alignment (cache-line aligned, prevents false sharing)
 * - Chunk-based memory management (64KB chunks)
 * - Thread-local design (zero contention)
 * - O(1) reset for tree clearing
 *
 * Performance targets:
 * - 33× faster than malloc (1.5ns vs 50ns)
 * - <1% memory overhead
 * - 643MB for 10M nodes
 *
 * NOT thread-safe: Each thread must have its own arena instance.
 */
class ThreadLocalArena {
public:
    /**
     * Allocation statistics for debugging and monitoring
     */
    struct Statistics {
        size_t allocations_from_bump;      // Bump pointer allocations
        size_t allocations_from_freelist;  // Free list reuse (future)
        size_t deallocations;              // Total frees (future)
        size_t chunks_allocated;           // Number of chunks allocated
        size_t bytes_allocated;            // Total bytes allocated
        size_t bytes_in_freelists;         // Bytes in free lists (future)
        size_t fallback_to_malloc;         // Overflow allocations
    };

    /**
     * Constructor
     *
     * @param initial_chunks Number of chunks to pre-allocate (default 2)
     * @param chunk_size Size of each chunk in bytes (default 64KB)
     * @param max_chunks Maximum chunks before fallback to malloc (default 128 = 8MB)
     */
    explicit ThreadLocalArena(
        size_t initial_chunks = 2,
        size_t chunk_size = 64 * 1024,
        size_t max_chunks = 128
    );

    /**
     * Destructor - frees all chunks
     */
    ~ThreadLocalArena();

    /**
     * Allocate memory (64-byte aligned)
     *
     * Fast path: O(1) bump pointer allocation (~1.5ns)
     * Slow path: Allocate new chunk or fallback to malloc
     *
     * @param size Number of bytes to allocate
     * @return Pointer to allocated memory (64-byte aligned), or nullptr on OOM
     */
    void* allocate(size_t size);

    /**
     * Deallocate memory (adds to free list)
     *
     * Adds the freed block to the appropriate size-class free list.
     * Uses intrusive linked list (stores next pointer in freed memory).
     * LIFO ordering for cache locality.
     *
     * @param ptr Pointer returned by allocate()
     * @param size Original allocation size
     */
    void deallocate(void* ptr, size_t size);

    /**
     * Reset arena (invalidates all allocations, resets to initial state)
     *
     * Fast O(1) operation - just resets bump pointers.
     * All previously allocated memory becomes invalid.
     * Chunks are retained for reuse.
     */
    void reset();

    /**
     * Get allocation statistics
     */
    Statistics get_statistics() const { return stats_; }

    // Non-copyable, non-movable
    ThreadLocalArena(const ThreadLocalArena&) = delete;
    ThreadLocalArena& operator=(const ThreadLocalArena&) = delete;

private:
    /**
     * Chunk header (64-byte aligned)
     *
     * Layout:
     * - next: Pointer to next chunk in linked list
     * - chunk_size: Total size of this chunk (including header)
     * - used_bytes: Bytes allocated from this chunk
     * - chunk_id: Unique identifier for debugging
     * - padding: Align to 64 bytes
     */
    struct alignas(64) Chunk {
        Chunk* next;
        size_t chunk_size;
        size_t used_bytes;
        uint32_t chunk_id;
        uint8_t padding[64 - sizeof(Chunk*) - 2*sizeof(size_t) - sizeof(uint32_t)];

        // Data follows immediately after header (flexible array member pattern)
        uint8_t* data() { return reinterpret_cast<uint8_t*>(this) + sizeof(Chunk); }
    };

    /**
     * Free list node (intrusive linked list)
     *
     * When a block is freed, we store the next pointer in the first 8 bytes.
     * This allows zero-overhead free lists.
     */
    struct FreeNode {
        FreeNode* next;
    };

    static constexpr size_t CACHE_LINE_SIZE = 64;  // Cache line alignment
    static constexpr size_t NUM_SIZE_CLASSES = 4;
    static constexpr size_t SIZE_CLASSES[NUM_SIZE_CLASSES] = {64, 128, 192, 256};

    Chunk* current_chunk_;     // Current chunk for allocations
    size_t current_offset_;    // Offset in current chunk
    size_t chunk_size_;        // Size of each chunk (excluding header)
    size_t max_chunks_;        // Maximum chunks before fallback
    size_t num_chunks_;        // Current number of chunks
    uint32_t next_chunk_id_;   // Next chunk ID for debugging

    FreeNode* freelists_[NUM_SIZE_CLASSES];  // Per-size-class free lists

    Statistics stats_;

    /**
     * Allocate a new chunk
     *
     * @param size Size of chunk data (excluding header)
     * @return Pointer to new chunk, or nullptr on OOM
     */
    Chunk* allocate_chunk(size_t size);

    /**
     * Free a chunk
     *
     * @param chunk Chunk to free
     */
    void free_chunk(Chunk* chunk);

    /**
     * Align size up to alignment boundary
     *
     * @param size Size to align
     * @param alignment Alignment boundary (must be power of 2)
     * @return Aligned size
     */
    static size_t align_up(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    /**
     * Allocate from a new chunk (slow path)
     *
     * @param size Size to allocate
     * @return Pointer to allocated memory, or nullptr on OOM
     */
    void* allocate_from_new_chunk(size_t size);

    /**
     * Round size up to next size class
     *
     * @param size Size to round up
     * @return Size class (32, 64, 128, or 256)
     */
    size_t round_up_to_size_class(size_t size) const;

    /**
     * Get size class index for a size
     *
     * @param size Size (must be a valid size class)
     * @return Index into SIZE_CLASSES array (0-3)
     */
    size_t size_to_class_index(size_t size) const;

    /**
     * Pop a node from the free list for a given size
     *
     * @param size Size class to pop from
     * @return Pointer to freed block, or nullptr if list empty
     */
    FreeNode* pop_from_freelist(size_t size);
};

/**
 * Get or create thread-local arena
 *
 * Lazy initialization: Arena is created on first call per thread.
 *
 * @return Pointer to thread-local arena
 */
ThreadLocalArena* get_thread_arena();

/**
 * Destroy thread-local arena
 *
 * Should be called before thread exit to free memory.
 * After calling this, get_thread_arena() will create a new arena.
 */
void destroy_thread_arena();

}} // namespace alphazero::core
