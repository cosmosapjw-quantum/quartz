// bump_arena.hpp - Per-thread bump allocator for TinyNode allocation
// Part of T024a: Tiny Node Design & Specification

#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include "tiny_node.hpp"

namespace mcts {

// Per-thread bump allocator for O(1) TinyNode allocation
//
// Design rationale:
// - Allocation speed: O(1) pointer increment (~5ns)
// - No locking: Each thread has its own arena
// - Bulk reclamation: O(1) epoch increment (vs O(N) free)
// - Cache-friendly: Nodes allocated contiguously in blocks
//
// Memory layout:
// - Block size: 65,536 nodes (4MB per block at 64 bytes/node)
// - Pre-allocate blocks on demand (lazy allocation)
// - Epoch-based reclamation for bulk-free
//
// Usage pattern:
//   thread_local BumpArena node_arena;
//
//   // Fast allocation
//   TinyNode* node = node_arena.allocate();
//
//   // Bulk reclamation (after search iteration)
//   node_arena.reclaim_epoch();
class BumpArena {
public:
    // Block size: 64K nodes = 4MB per block (at 64 bytes/node)
    // Tuned for:
    // - L3 cache: 32MB (Ryzen 5900X) - 8 blocks fit in L3
    // - Page size: 4KB - 64 nodes per page, 1024 pages per block
    static constexpr size_t BLOCK_SIZE = 65536;

    BumpArena()
        : current_block_(nullptr),
          offset_(BLOCK_SIZE),  // Trigger allocation on first allocate()
          epoch_marker_idx_(0) {
    }

    ~BumpArena() = default;

    // Allocate a single TinyNode (O(1) fast path)
    //
    // Performance:
    // - Fast path (in-block): ~5ns (offset_++)
    // - Slow path (new block): ~500ns (block allocation)
    // - Expected: 99.99% fast path (1 slow per 64K allocations)
    //
    // Thread safety: NOT thread-safe (thread-local use only)
    TinyNode* allocate() {
        // Fast path: allocate from current block
        if (offset_ < BLOCK_SIZE) {
            return &current_block_[offset_++];
        }

        // Slow path: allocate new block
        allocate_new_block();
        return &current_block_[offset_++];
    }

    // Bulk reclamation via epoch marker
    //
    // Reclaims all blocks allocated before current epoch marker.
    // This is called at the end of each search iteration when all
    // threads have reached a quiescent state (safe to free).
    //
    // Performance: O(1) - erase blocks before marker, reset offset
    //
    // Memory safety:
    // - MUST only be called when no threads hold references to nodes
    // - Use EpochManager to ensure quiescence before calling
    void reclaim_epoch() {
        if (epoch_marker_idx_ > 0) {
            // Erase all blocks before epoch marker
            blocks_.erase(blocks_.begin(), blocks_.begin() + epoch_marker_idx_);

            // Reset epoch marker (now points to first block)
            epoch_marker_idx_ = 0;
        }

        // Reset allocation to start of first block
        if (!blocks_.empty()) {
            current_block_ = blocks_[0].get();
            offset_ = 0;
        } else {
            current_block_ = nullptr;
            offset_ = BLOCK_SIZE;  // Trigger allocation on next allocate()
        }
    }

    // Mark current position as epoch boundary
    //
    // All blocks allocated before this point can be reclaimed
    // after threads reach quiescence.
    void mark_epoch() {
        // Current block becomes the new epoch marker
        if (!blocks_.empty()) {
            epoch_marker_idx_ = blocks_.size() - 1;
        }
    }

    // Get allocation statistics
    struct Stats {
        size_t num_blocks;         // Total blocks allocated
        size_t current_offset;     // Current offset in block
        size_t total_nodes;        // Total nodes allocated
        size_t memory_bytes;       // Total memory allocated (bytes)
        size_t epoch_marker_idx;   // Epoch marker index
    };

    Stats get_stats() const {
        Stats stats;
        stats.num_blocks = blocks_.size();
        stats.current_offset = offset_;
        stats.total_nodes = (blocks_.size() - 1) * BLOCK_SIZE + offset_;
        stats.memory_bytes = blocks_.size() * BLOCK_SIZE * sizeof(TinyNode);
        stats.epoch_marker_idx = epoch_marker_idx_;
        return stats;
    }

    // Reset allocator (for testing)
    void reset() {
        blocks_.clear();
        current_block_ = nullptr;
        offset_ = BLOCK_SIZE;
        epoch_marker_idx_ = 0;
    }

private:
    // Allocate a new block and switch to it
    void allocate_new_block() {
        // Allocate new block (64K nodes × 64 bytes = 4MB)
        auto new_block = std::make_unique<TinyNode[]>(BLOCK_SIZE);

        // Zero-initialize new block (required for atomic variables)
        std::memset(new_block.get(), 0, BLOCK_SIZE * sizeof(TinyNode));

        // Add to block list
        blocks_.push_back(std::move(new_block));

        // Switch to new block
        current_block_ = blocks_.back().get();
        offset_ = 0;
    }

    // Current block for allocation
    TinyNode* current_block_;

    // Current offset in block (next allocation index)
    size_t offset_;

    // Epoch marker index (blocks before this can be reclaimed)
    size_t epoch_marker_idx_;

    // All allocated blocks (owned)
    std::vector<std::unique_ptr<TinyNode[]>> blocks_;
};

// Thread-local bump arena accessor
//
// Each worker thread gets its own arena. No synchronization required.
// Example usage:
//   TinyNode* node = get_thread_arena().allocate();
inline BumpArena& get_thread_arena() {
    thread_local BumpArena arena;
    return arena;
}

} // namespace mcts
