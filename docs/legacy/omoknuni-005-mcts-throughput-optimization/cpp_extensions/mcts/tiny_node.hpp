// tiny_node.hpp - Tiny MCTS node structure for zero-copy architecture
// Part of T024a: Tiny Node Design & Specification

#pragma once

#include <cstdint>
#include <atomic>
#include <cassert>

namespace mcts {

// Tiny MCTS node: 32 bytes core data, aligned to 64 bytes for cache efficiency
//
// Design rationale:
// - Current nodes: 120KB each (contain full game state)
// - Target nodes: 32 bytes (only move, statistics, zobrist, children)
// - Memory reduction: 3,750× per node
// - Cache efficiency: 2 cache lines → 1 cache line
// - Enables elimination of state cloning bottleneck (418μs → 15ns)
//
// Zero-copy pattern:
// - Store only move sequences in tree
// - Reconstruct game states on-demand via make/unmake
// - Thread-local state reconstruction (no cloning)
struct alignas(64) TinyNode {
    //
    // Move and tree structure (16 bytes)
    //

    // Move that led to this node (16 bits, supports all game move encodings)
    // - Gomoku: row*15 + col (0-224)
    // - Chess: from_square*64 + to_square + promotion (0-4095)
    // - Go: row*19 + col + pass (0-362)
    uint16_t move;

    // Parent node index (32 bits, supports 4 billion nodes)
    // Root node has parent_idx = 0 (points to self)
    uint32_t parent_idx;

    // First child index (32 bits, 0 = no children)
    // Children linked as singly-linked list via next_sibling_idx
    uint32_t first_child_idx;

    // Next sibling index (32 bits, 0 = no sibling)
    // Enables efficient child iteration without dynamic allocation
    uint32_t next_sibling_idx;

    //
    // MCTS statistics (16 bytes)
    //

    // Visit count (atomic, 32 bits)
    // Thread-safe increment during backup phase
    std::atomic<uint32_t> visit_count;

    // Total value (atomic, 32 bits as int32 × 1,000,000 for precision)
    // Stored as scaled integer: value_scaled = value * 1,000,000
    // Range: [-2,147, +2,147] with 6 decimal places precision
    // Thread-safe accumulation during backup phase
    std::atomic<int32_t> total_value_scaled;

    // Prior probability (16 bits, scaled 0-65535)
    // Stored as: prior_scaled = prior * 65535
    // Resolution: ~0.0015% (1/65536)
    // Set once during expansion, read-only thereafter
    uint16_t prior_scaled;

    // Virtual loss (8 bits, max 255)
    // WU-UCT algorithm: visit-only virtual loss
    // Thread-safe increment/decrement during selection
    std::atomic<uint8_t> virtual_loss;

    // Node flags (8 bits: terminal, expanded, etc.)
    // Bit layout:
    //   [0]: is_terminal (set if game ended)
    //   [1]: is_expanded (set if children allocated)
    //   [2]: is_root (set for root node)
    //   [3-7]: reserved
    uint8_t flags;

    //
    // Transposition table support (8 bytes)
    //

    // Zobrist hash (64 bits)
    // Incremental hash for position identification
    // Enables transposition table lookups (DAG structure)
    // Updated via XOR during make/unmake moves
    uint64_t zobrist_hash;

    //
    // Total: 2 + 4 + 4 + 4 + 4 + 4 + 2 + 1 + 1 + 8 = 34 bytes
    // Aligned to 64 bytes for cache line efficiency (padding: 30 bytes)
    //

    // Flag bit masks
    static constexpr uint8_t FLAG_TERMINAL = 0x01;
    static constexpr uint8_t FLAG_EXPANDED = 0x02;
    static constexpr uint8_t FLAG_ROOT = 0x04;

    // Scaling constants
    static constexpr int32_t VALUE_SCALE = 1000000;
    static constexpr uint16_t PRIOR_SCALE = 65535;

    // Helper methods (inline for zero overhead)

    inline bool is_terminal() const {
        return (flags & FLAG_TERMINAL) != 0;
    }

    inline bool is_expanded() const {
        return (flags & FLAG_EXPANDED) != 0;
    }

    inline bool is_root() const {
        return (flags & FLAG_ROOT) != 0;
    }

    inline void set_terminal() {
        flags |= FLAG_TERMINAL;
    }

    inline void set_expanded() {
        flags |= FLAG_EXPANDED;
    }

    inline void set_root() {
        flags |= FLAG_ROOT;
    }

    // Convert scaled value to float
    inline float get_value() const {
        return static_cast<float>(total_value_scaled.load(std::memory_order_relaxed)) / VALUE_SCALE;
    }

    // Convert scaled prior to float
    inline float get_prior() const {
        return static_cast<float>(prior_scaled) / PRIOR_SCALE;
    }

    // Get Q value (mean value)
    inline float get_q_value() const {
        uint32_t n = visit_count.load(std::memory_order_relaxed);
        if (n == 0) return 0.0f;
        return get_value() / static_cast<float>(n);
    }
};

// Compile-time assertions
static_assert(sizeof(TinyNode) <= 64, "TinyNode must fit in cache line (64 bytes)");
static_assert(alignof(TinyNode) == 64, "TinyNode must be 64-byte aligned");
static_assert(std::is_trivially_copyable<TinyNode>::value, "TinyNode must be trivially copyable");

// Memory impact analysis:
//   Current architecture: 120 KB per node (contains full game state)
//   New architecture: 32 bytes per node (64 bytes with alignment)
//   Memory reduction: 120,000 / 64 = 1,875× per node
//
//   Example: 10 million nodes
//   - Current: 10M × 120KB = 1.2 GB
//   - New: 10M × 64 bytes = 640 MB (320 MB for 32-byte packed)
//   - Savings: ~600 MB (50% reduction)

} // namespace mcts
