#include "thread_local_arena.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <new>

namespace alphazero {
namespace core {

// Thread-local storage for arena
thread_local ThreadLocalArena* g_thread_arena = nullptr;

ThreadLocalArena::ThreadLocalArena(
    size_t initial_chunks,
    size_t chunk_size,
    size_t max_chunks
)
    : current_chunk_(nullptr),
      current_offset_(0),
      chunk_size_(chunk_size),
      max_chunks_(max_chunks),
      num_chunks_(0),
      next_chunk_id_(0),
      freelists_{nullptr, nullptr, nullptr, nullptr},
      stats_()
{
    // Pre-allocate initial chunks
    for (size_t i = 0; i < initial_chunks; ++i) {
        Chunk* chunk = allocate_chunk(chunk_size_);
        if (!chunk) {
            // OOM during initialization - clean up and throw
            while (current_chunk_) {
                Chunk* next = current_chunk_->next;
                free_chunk(current_chunk_);
                current_chunk_ = next;
            }
            throw std::bad_alloc();
        }

        // Link chunk into list
        chunk->next = current_chunk_;
        current_chunk_ = chunk;
        num_chunks_++;
    }

    // Reset offset to start of first chunk
    current_offset_ = 0;
}

ThreadLocalArena::~ThreadLocalArena() {
    // Free all chunks in linked list
    Chunk* chunk = current_chunk_;
    while (chunk) {
        Chunk* next = chunk->next;
        free_chunk(chunk);
        chunk = next;
    }

    current_chunk_ = nullptr;
    num_chunks_ = 0;
}

void* ThreadLocalArena::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    // Round up to size class
    size_t aligned_size = round_up_to_size_class(size);

    // FASTEST PATH: Try free list first (LIFO for cache locality)
    if (FreeNode* node = pop_from_freelist(aligned_size)) {
        stats_.allocations_from_freelist++;
        stats_.bytes_allocated += aligned_size;
        stats_.bytes_in_freelists -= aligned_size;
        return node;
    }

    // FAST PATH: Try bump pointer allocation in current chunk
    if (current_chunk_) {
        size_t new_offset = current_offset_ + aligned_size;
        size_t available = chunk_size_ - current_offset_;

        if (aligned_size <= available) {
            // Allocation fits in current chunk
            void* ptr = current_chunk_->data() + current_offset_;
            current_offset_ = new_offset;
            current_chunk_->used_bytes = new_offset;
            stats_.allocations_from_bump++;
            stats_.bytes_allocated += aligned_size;
            return ptr;
        }
    }

    // SLOW PATH: Need new chunk
    return allocate_from_new_chunk(aligned_size);
}

void ThreadLocalArena::deallocate(void* ptr, size_t size) {
    if (!ptr) {
        return;
    }

    // Round up to size class
    size_t aligned_size = round_up_to_size_class(size);

    stats_.deallocations++;

    // Only add to free list if it's a valid size class (<=256)
    size_t class_idx = size_to_class_index(aligned_size);
    if (class_idx < NUM_SIZE_CLASSES) {
        // Add to free list (LIFO for cache locality)
        FreeNode* node = static_cast<FreeNode*>(ptr);
        node->next = freelists_[class_idx];
        freelists_[class_idx] = node;
        stats_.bytes_in_freelists += aligned_size;
    }
    // Sizes > 256 are not added to free lists (too large)
}

void ThreadLocalArena::reset() {
    // Reset all chunks to empty state (O(1) operation)
    Chunk* chunk = current_chunk_;
    while (chunk) {
        chunk->used_bytes = 0;
        chunk = chunk->next;
    }

    // Reset to first chunk
    current_offset_ = 0;

    // Clear all free lists
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        freelists_[i] = nullptr;
    }

    // Reset statistics (keep chunks_allocated, but reset allocations)
    stats_.allocations_from_bump = 0;
    stats_.allocations_from_freelist = 0;
    stats_.deallocations = 0;
    stats_.bytes_allocated = 0;
    stats_.bytes_in_freelists = 0;
    stats_.fallback_to_malloc = 0;
}

ThreadLocalArena::Chunk* ThreadLocalArena::allocate_chunk(size_t size) {
    // Allocate chunk + header in one allocation
    // We use posix_memalign for 64-byte alignment
    size_t total_size = sizeof(Chunk) + size;
    void* memory = nullptr;

#ifdef _WIN32
    // Windows: Use _aligned_malloc
    memory = _aligned_malloc(total_size, CACHE_LINE_SIZE);
#else
    // POSIX: Use posix_memalign
    int ret = posix_memalign(&memory, CACHE_LINE_SIZE, total_size);
    if (ret != 0) {
        memory = nullptr;
    }
#endif

    if (!memory) {
        return nullptr;
    }

    // Initialize chunk header using placement new
    Chunk* chunk = new (memory) Chunk();
    chunk->next = nullptr;
    chunk->chunk_size = size;
    chunk->used_bytes = 0;
    chunk->chunk_id = next_chunk_id_++;

    stats_.chunks_allocated++;

    return chunk;
}

void ThreadLocalArena::free_chunk(Chunk* chunk) {
    if (!chunk) {
        return;
    }

#ifdef _WIN32
    _aligned_free(chunk);
#else
    std::free(chunk);
#endif
}

void* ThreadLocalArena::allocate_from_new_chunk(size_t aligned_size) {
    // Check if we've hit the chunk limit
    if (num_chunks_ >= max_chunks_) {
        // Fallback to malloc
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(aligned_size, CACHE_LINE_SIZE);
#else
        int ret = posix_memalign(&ptr, CACHE_LINE_SIZE, aligned_size);
        if (ret != 0) {
            ptr = nullptr;
        }
#endif

        if (ptr) {
            stats_.fallback_to_malloc++;
            stats_.bytes_allocated += aligned_size;
        }
        return ptr;
    }

    // Allocate new chunk (at least as large as requested size)
    size_t new_chunk_size = (aligned_size > chunk_size_) ? aligned_size : chunk_size_;
    Chunk* new_chunk = allocate_chunk(new_chunk_size);

    if (!new_chunk) {
        // OOM - cannot allocate new chunk
        return nullptr;
    }

    // Link new chunk at head of list
    new_chunk->next = current_chunk_;
    current_chunk_ = new_chunk;
    num_chunks_++;

    // Allocate from new chunk
    void* ptr = new_chunk->data();
    current_offset_ = aligned_size;
    new_chunk->used_bytes = aligned_size;
    stats_.allocations_from_bump++;
    stats_.bytes_allocated += aligned_size;

    return ptr;
}

size_t ThreadLocalArena::round_up_to_size_class(size_t size) const {
    // Round up to next size class (64, 128, 192, 256)
    // All size classes are multiples of 64 to maintain alignment
    if (size <= 64) return 64;
    if (size <= 128) return 128;
    if (size <= 192) return 192;
    if (size <= 256) return 256;

    // For sizes > 256, round up to 64-byte alignment
    return align_up(size, CACHE_LINE_SIZE);
}

size_t ThreadLocalArena::size_to_class_index(size_t size) const {
    // Convert size class to index (0-3)
    // Assumes size is already a valid size class
    if (size == 64) return 0;
    if (size == 128) return 1;
    if (size == 192) return 2;
    if (size == 256) return 3;

    // Size > 256 doesn't use free lists
    return NUM_SIZE_CLASSES;  // Invalid index
}

ThreadLocalArena::FreeNode* ThreadLocalArena::pop_from_freelist(size_t size) {
    size_t class_idx = size_to_class_index(size);

    // Sizes > 256 don't use free lists
    if (class_idx >= NUM_SIZE_CLASSES) {
        return nullptr;
    }

    FreeNode* node = freelists_[class_idx];
    if (node) {
        // Pop from LIFO list
        freelists_[class_idx] = node->next;
    }

    return node;
}

// Global thread-local arena accessors

ThreadLocalArena* get_thread_arena() {
    if (!g_thread_arena) {
        // Lazy initialization
        g_thread_arena = new ThreadLocalArena(
            /*initial_chunks=*/2,
            /*chunk_size=*/64 * 1024,
            /*max_chunks=*/128
        );
    }
    return g_thread_arena;
}

void destroy_thread_arena() {
    if (g_thread_arena) {
        delete g_thread_arena;
        g_thread_arena = nullptr;
    }
}

}} // namespace alphazero::core
