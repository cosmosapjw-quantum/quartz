/**
 * @file tree.hpp
 * @brief High-performance MCTS tree with Structure-of-Arrays memory layout
 *
 * Implements cache-efficient MCTS tree storage using Structure-of-Arrays (SoA)
 * design for optimal memory access patterns and SIMD vectorization support.
 *
 * Key features:
 * - 64-byte aligned arrays for SIMD operations on AMD Ryzen 5900X
 * - <64 bytes per node memory footprint (32-40 bytes typical)
 * - Support for 50M+ nodes (~1GB total memory at 32 bytes/node)
 * - Index-based node references instead of pointers
 * - Cache-friendly memory layout for tree traversal operations
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cassert>
#include <cstring>
#include <mutex>
#include <atomic>
#include <immintrin.h>  // For SIMD intrinsics

namespace mcts {

/**
 * @brief Node index type for referencing nodes in arrays
 *
 * Using int32_t allows for 2+ billion nodes while maintaining
 * cache efficiency. -1 is reserved as NULL_NODE_INDEX.
 */
using NodeIndex = std::int32_t;

/**
 * @brief Special value indicating no node / null reference
 */
constexpr NodeIndex NULL_NODE_INDEX = -1;

/**
 * @brief Node flags packed into a single byte for efficiency
 *
 * Bit layout:
 * - Bit 0: expanded (has children)
 * - Bit 1: terminal (game over)
 * - Bit 2: current_player (0 or 1)
 * - Bits 3-7: reserved for future use
 */
struct NodeFlags {
    std::uint8_t flags;

    // Bit manipulation helpers
    bool is_expanded() const { return flags & 0x01; }
    bool is_terminal() const { return flags & 0x02; }
    std::uint8_t current_player() const { return (flags >> 2) & 0x01; }
    bool is_expanding() const { return flags & 0x08; }

    void set_expanded(bool value) {
        flags = value ? (flags | 0x01) : (flags & ~0x01);
    }

    void set_terminal(bool value) {
        flags = value ? (flags | 0x02) : (flags & ~0x02);
    }

    void set_current_player(std::uint8_t player) {
        flags = (flags & ~0x04) | ((player & 0x01) << 2);
    }

    void set_expanding(bool value) {
        flags = value ? (flags | 0x08) : (flags & static_cast<std::uint8_t>(~0x08));
    }

    NodeFlags() : flags(0) {}
};

/**
 * @brief MCTS tree node information for debugging and validation
 */
struct NodeInfo {
    NodeIndex index;
    float visit_count;
    float total_value;
    float prior_prob;
    float virtual_loss;
    NodeIndex parent_index;
    NodeIndex first_child_index;
    std::uint16_t num_children;
    NodeFlags flags;

    // Derived values
    float q_value() const {
        return visit_count > 0 ? total_value / visit_count : 0.0f;
    }

    bool is_root() const {
        return parent_index == NULL_NODE_INDEX;
    }
};

/**
 * @brief High-performance MCTS tree with Structure-of-Arrays layout
 *
 * This class implements the core MCTS tree data structure using separate
 * aligned arrays for each node attribute. This design provides:
 *
 * 1. Cache efficiency: Related data accessed together during tree traversal
 * 2. SIMD optimization: 64-byte aligned arrays enable vectorized operations
 * 3. Memory efficiency: 32-64 bytes per node vs 200+ bytes with pointers
 * 4. Scalability: Supports 50M+ nodes with <2GB memory usage
 */
class MCTSTree {
public:
    /**
     * @brief Initialize MCTS tree with specified capacity
     *
     * @param max_nodes Maximum number of nodes to support
     */
    explicit MCTSTree(std::size_t max_nodes = 50'000'000);

    /**
     * @brief Destructor - frees aligned memory
     */
    ~MCTSTree();

    // Disable copy/move for now to avoid complexity
    MCTSTree(const MCTSTree&) = delete;
    MCTSTree& operator=(const MCTSTree&) = delete;
    MCTSTree(MCTSTree&&) = delete;
    MCTSTree& operator=(MCTSTree&&) = delete;

    /**
     * @brief Get the root node index (always 0 when tree has nodes)
     */
    NodeIndex get_root_index() const {
        return node_count_ > 0 ? 0 : NULL_NODE_INDEX;
    }

    /**
     * @brief Get current number of nodes in tree
     */
    std::size_t get_node_count() const { return node_count_; }

    /**
     * @brief Get maximum capacity of tree
     */
    std::size_t get_max_nodes() const { return max_nodes_; }

    /**
     * @brief Get memory usage in bytes
     */
    std::size_t get_memory_usage() const;

    /**
     * @brief Get bytes per node (actual memory efficiency)
     */
    double get_bytes_per_node() const {
        return node_count_ > 0 ? static_cast<double>(get_memory_usage()) / node_count_ : 0.0;
    }

    /**
     * @brief Clear all nodes and reset tree
     */
    void clear();

    /**
     * @brief Add root node to empty tree
     *
     * @param prior_prob Prior probability from neural network
     * @param current_player Current player (0 or 1)
     * @return Index of created root node (always 0)
     */
    NodeIndex add_root_node(float prior_prob, std::uint8_t current_player);

    /**
     * @brief Allocate a single node from the pre-allocated pool
     *
     * @return Index of allocated node, or NULL_NODE_INDEX if pool is full
     */
    NodeIndex allocate_node();

    /**
     * @brief Allocate multiple contiguous nodes from the pool
     *
     * This is more efficient than calling allocate_node() multiple times
     * when expanding a parent node with multiple children.
     *
     * @param count Number of contiguous nodes to allocate
     * @return Index of first allocated node, or NULL_NODE_INDEX if insufficient space
     */
    NodeIndex allocate_nodes(std::uint16_t count);

    /**
     * @brief Deallocate a single node back to the pool
     *
     * Note: This marks the node as free but doesn't clear its data.
     * The data will be overwritten when the node is reused.
     *
     * @param index Index of node to deallocate
     */
    void deallocate_node(NodeIndex index);

    /**
     * @brief Deallocate multiple contiguous nodes back to the pool
     *
     * @param first_index Index of first node to deallocate
     * @param count Number of contiguous nodes to deallocate
     */
    void deallocate_nodes(NodeIndex first_index, std::uint16_t count);

    /**
     * @brief Get number of available nodes in the pool
     */
    std::size_t get_available_nodes() const {
        return (max_nodes_ - next_free_index_) + free_nodes_.size();
    }

    /**
     * @brief Check if the tree has space for additional nodes
     *
     * @param count Number of nodes to check for
     * @return true if space is available, false otherwise
     */
    bool has_space_for(std::uint16_t count) const {
        return get_available_nodes() >= count;
    }

    /**
     * @brief Validate tree structure and constraints
     *
     * @return true if tree is valid, false otherwise
     */
    bool validate_tree() const;

    // Node data access methods (inline for performance)

    /**
     * @brief Get visit count for node
     */
    float get_visit_count(NodeIndex index) const {
        assert(is_valid_index(index));
        return visit_counts_[index];
    }

    /**
     * @brief Get total value for node
     */
    float get_total_value(NodeIndex index) const {
        assert(is_valid_index(index));
        return total_values_[index];
    }

    /**
     * @brief Get prior probability for node
     */
    float get_prior_prob(NodeIndex index) const {
        assert(is_valid_index(index));
        return prior_probs_[index];
    }

    /**
     * @brief Get virtual loss for node
     */
    float get_virtual_loss(NodeIndex index) const {
        assert(is_valid_index(index));
        return virtual_losses_[index];
    }

    /**
     * @brief Get parent index for node
     */
    NodeIndex get_parent_index(NodeIndex index) const {
        assert(is_valid_index(index));
        return parent_indices_[index];
    }

    /**
     * @brief Get first child index for node
     */
    NodeIndex get_first_child_index(NodeIndex index) const {
        assert(is_valid_index(index));
        return first_child_indices_[index];
    }

    /**
     * @brief Get number of children for node
     */
    std::uint16_t get_num_children(NodeIndex index) const {
        assert(is_valid_index(index));
        return num_children_[index];
    }

    /**
     * @brief Get flags for node
     */
    NodeFlags get_flags(NodeIndex index) const {
        assert(is_valid_index(index));
        return flags_[index];
    }

    /**
     * @brief Get complete node information for debugging
     */
    NodeInfo get_node_info(NodeIndex index) const;

    /**
     * @brief Set visit count for node
     */
    void set_visit_count(NodeIndex index, float value) {
        assert(is_valid_index(index));
        assert(value >= 0.0f);
        visit_counts_[index] = value;
    }

    /**
     * @brief Set total value for node
     */
    void set_total_value(NodeIndex index, float value) {
        assert(is_valid_index(index));
        total_values_[index] = value;
    }

    /**
     * @brief Set prior probability for node
     */
    void set_prior_prob(NodeIndex index, float value) {
        assert(is_valid_index(index));
        assert(value >= 0.0f && value <= 1.0f);
        prior_probs_[index] = value;
    }

    /**
     * @brief Set virtual loss for node
     */
    void set_virtual_loss(NodeIndex index, float value) {
        assert(is_valid_index(index));
        assert(value >= 0.0f);
        virtual_losses_[index] = value;
    }

    /**
     * @brief Set parent index for node
     */
    void set_parent_index(NodeIndex index, NodeIndex parent) {
        assert(is_valid_index(index));
        assert(parent == NULL_NODE_INDEX || is_valid_index(parent));
        parent_indices_[index] = parent;
    }

    /**
     * @brief Set first child index for node
     */
    void set_first_child_index(NodeIndex index, NodeIndex first_child) {
        assert(is_valid_index(index));
        assert(first_child == NULL_NODE_INDEX || is_valid_index(first_child));
        first_child_indices_[index] = first_child;
    }

    /**
     * @brief Set number of children for node
     */
    void set_num_children(NodeIndex index, std::uint16_t count) {
        assert(is_valid_index(index));
        num_children_[index] = count;
    }

    /**
     * @brief Set flags for node
     */
    void set_flags(NodeIndex index, const NodeFlags& flags) {
        assert(is_valid_index(index));
        flags_[index] = flags;
    }

    /**
     * @brief Atomically try to set expanded flag (for race-free expansion)
     *
     * @param index Node index to update
     * @return true if we successfully set expanded=true (we own expansion)
     *         false if already expanded by another thread
     */
    bool atomic_try_set_expanded(NodeIndex index) {
        assert(is_valid_index(index));

        // Get pointer to flags byte for atomic operations
        std::atomic<std::uint8_t>* atomic_flags =
            reinterpret_cast<std::atomic<std::uint8_t>*>(&flags_[index].flags);

        // Atomically check and set expanded bit (bit 0)
        std::uint8_t expected, desired;
        do {
            expected = atomic_flags->load(std::memory_order_acquire);

            // If already expanded, another thread won the race
            if (expected & 0x01) {
                return false;
            }

            // Set expanded bit while preserving other flags
            desired = expected | 0x01;

        } while (!atomic_flags->compare_exchange_weak(expected, desired,
                                                      std::memory_order_release,
                                                      std::memory_order_acquire));

        return true;  // We successfully set expanded flag
    }

    /**
     * @brief Atomically mark node as being expanded
     *
     * Prevents duplicate expansion requests by ensuring only one thread
     * can submit inference work for a given leaf at a time.
     *
     * @param index Node index to update
     * @return true if we successfully set expanding=true
     */
    bool atomic_try_mark_expanding(NodeIndex index) {
        assert(is_valid_index(index));

        std::atomic<std::uint8_t>* atomic_flags =
            reinterpret_cast<std::atomic<std::uint8_t>*>(&flags_[index].flags);

        std::uint8_t expected, desired;
        do {
            expected = atomic_flags->load(std::memory_order_acquire);

            // If node already expanded or in-flight, do not mark again
            if ((expected & 0x01) || (expected & 0x08)) {
                return false;
            }

            desired = expected | 0x08;  // mark expanding bit (bit 3)

        } while (!atomic_flags->compare_exchange_weak(expected, desired,
                                                      std::memory_order_release,
                                                      std::memory_order_acquire));

        return true;
    }

    /**
     * @brief Clear the expanding flag on a node
     *
     * Called after inference completes (success or failure) to allow
     * other threads to select the node again.
     */
    void clear_expanding_flag(NodeIndex index) {
        assert(is_valid_index(index));

        std::atomic<std::uint8_t>* atomic_flags =
            reinterpret_cast<std::atomic<std::uint8_t>*>(&flags_[index].flags);

        std::uint8_t expected = atomic_flags->load(std::memory_order_acquire);
        while (true) {
            if ((expected & 0x08) == 0) {
                return;  // Already cleared
            }

            std::uint8_t desired = expected & static_cast<std::uint8_t>(~0x08);
            if (atomic_flags->compare_exchange_weak(expected, desired,
                                                    std::memory_order_release,
                                                    std::memory_order_acquire)) {
                return;
            }
        }
    }

    /**
     * @brief Check if node index is valid (within allocated range)
     */
    bool is_valid_index(NodeIndex index) const {
        return index >= 0 && static_cast<std::size_t>(index) < next_free_index_;
    }

    /**
     * @brief Get pointer to visit counts array (for SIMD operations)
     */
    const float* get_visit_counts_ptr() const { return visit_counts_; }
    float* get_visit_counts_ptr() { return visit_counts_; }

    /**
     * @brief Get pointer to total values array (for SIMD operations)
     */
    const float* get_total_values_ptr() const { return total_values_; }
    float* get_total_values_ptr() { return total_values_; }

    /**
     * @brief Get pointer to prior probabilities array (for SIMD operations)
     */
    const float* get_prior_probs_ptr() const { return prior_probs_; }
    float* get_prior_probs_ptr() { return prior_probs_; }

    /**
     * @brief Get pointer to virtual losses array (for SIMD operations)
     */
    const float* get_virtual_losses_ptr() const { return virtual_losses_; }
    float* get_virtual_losses_ptr() { return virtual_losses_; }

    /**
     * @brief Get pointer to flags array (for SIMD operations)
     */
    const NodeFlags* get_flags_ptr() const { return flags_; }
    NodeFlags* get_flags_ptr() { return flags_; }

    /**
     * @brief Get move index associated with a node
     *
     * This stores which move led to this node from its parent.
     * For child nodes, this is the action index (0-361 for Gomoku, 0-4671 for Go).
     * Root node has move index 0 (no parent move).
     *
     * @param index Node index
     * @return Move index (uint16_t, max 65535 actions supported)
     */
    std::uint16_t get_move(NodeIndex index) const {
        assert(is_valid_index(index));
        return moves_[index];
    }

    /**
     * @brief Set move index for a node
     *
     * @param index Node index
     * @param move Move index to store
     */
    void set_move(NodeIndex index, std::uint16_t move) {
        assert(is_valid_index(index));
        moves_[index] = move;
    }

    /**
     * @brief Get thread-local allocation statistics
     *
     * Returns statistics about allocation efficiency for the calling thread.
     * Used for performance analysis and tuning.
     *
     * @return Struct with allocations_from_block, allocations_from_global, allocations_from_freelist
     */
    struct ThreadAllocationStats {
        std::uint64_t allocations_from_block;    // Fast path: thread-local block cache
        std::uint64_t allocations_from_global;   // Slow path: global pool with mutex
        std::uint64_t allocations_from_freelist; // Reuse path: free list
        std::uint32_t block_size;                // Current thread-local block size config

        // Derived metrics
        double fast_path_percentage() const {
            std::uint64_t total = allocations_from_block + allocations_from_global + allocations_from_freelist;
            return total > 0 ? (100.0 * allocations_from_block / total) : 0.0;
        }

        double slow_path_percentage() const {
            std::uint64_t total = allocations_from_block + allocations_from_global + allocations_from_freelist;
            return total > 0 ? (100.0 * allocations_from_global / total) : 0.0;
        }

        double reuse_percentage() const {
            std::uint64_t total = allocations_from_block + allocations_from_global + allocations_from_freelist;
            return total > 0 ? (100.0 * allocations_from_freelist / total) : 0.0;
        }
    };

    ThreadAllocationStats get_thread_allocation_stats() const;

private:
    // Maximum number of nodes this tree can hold
    std::size_t max_nodes_;

    // Current number of nodes in tree (actively used nodes)
    // Atomic to allow lock-free reads from get_node_count()
    std::atomic<std::size_t> node_count_;

    // Free list for efficient node reuse
    std::vector<NodeIndex> free_nodes_;

    // Index of the next contiguous node to allocate (when free list is empty)
    // This tracks the high-water mark of allocated indices
    // Atomic to allow lock-free reads from is_valid_index() and get_available_nodes()
    std::atomic<std::size_t> next_free_index_;

    // Allocation epoch used to invalidate thread-local caches on clear()
    std::atomic<std::uint64_t> allocation_epoch_{0};

    // Unique instance ID to distinguish different tree instances (even at same address)
    std::uint64_t instance_id_;

    // Mutex to protect allocation/deallocation operations
    // Needed for thread-safe concurrent node allocation in MCTS search
    mutable std::mutex allocation_mutex_;

    // Structure-of-Arrays: separate aligned arrays for each node attribute
    // All arrays are 64-byte aligned for SIMD operations

    alignas(64) float* visit_counts_;         // N: visit count for each node
    alignas(64) float* total_values_;         // W: accumulated value sum
    alignas(64) float* prior_probs_;          // P: neural network policy probability
    alignas(64) float* virtual_losses_;       // VL: temporary virtual loss
    alignas(64) NodeIndex* parent_indices_;   // Parent node index (-1 for root)
    alignas(64) NodeIndex* first_child_indices_; // First child index (-1 if no children)
    alignas(64) std::uint16_t* num_children_; // Number of child nodes
    alignas(64) NodeFlags* flags_;            // Packed boolean flags
    alignas(64) std::uint16_t* moves_;        // Move index that led to this node (action space index)

    /**
     * @brief Allocate aligned memory for all arrays
     */
    void allocate_arrays();

    /**
     * @brief Free aligned memory for all arrays
     */
    void deallocate_arrays();

    /**
     * @brief Initialize all arrays to zero/default values
     */
    void initialize_arrays();

    /**
     * @brief Reset a single node to default state.
     */
    void initialize_node(NodeIndex index);

    /**
     * @brief Reset a contiguous set of nodes to default state.
     */
    void initialize_node_range(NodeIndex first_index, std::uint16_t count);
};

/**
 * @brief Memory statistics for MCTS tree
 */
struct TreeMemoryStats {
    std::size_t total_bytes;        // Total memory usage
    std::size_t node_count;         // Number of nodes
    double bytes_per_node;          // Average bytes per node
    std::size_t visit_counts_bytes; // Memory for visit counts
    std::size_t total_values_bytes; // Memory for total values
    std::size_t prior_probs_bytes;  // Memory for prior probabilities
    std::size_t virtual_losses_bytes; // Memory for virtual losses
    std::size_t parent_indices_bytes; // Memory for parent indices
    std::size_t first_child_indices_bytes; // Memory for first child indices
    std::size_t num_children_bytes; // Memory for child counts
    std::size_t flags_bytes;        // Memory for flags
    std::size_t alignment_overhead; // Memory lost to alignment
};

/**
 * @brief Get detailed memory statistics for tree
 */
TreeMemoryStats get_tree_memory_stats(const MCTSTree& tree);

/**
 * @brief Helper function to allocate 64-byte aligned memory
 */
template<typename T>
T* allocate_aligned(std::size_t count) {
    void* ptr = nullptr;
    std::size_t size = count * sizeof(T);

    // Ensure size is multiple of 64 for alignment
    std::size_t aligned_size = ((size + 63) / 64) * 64;

    #ifdef _WIN32
    ptr = _aligned_malloc(aligned_size, 64);
    #else
    if (posix_memalign(&ptr, 64, aligned_size) != 0) {
        ptr = nullptr;
    }
    #endif

    if (!ptr) {
        throw std::bad_alloc();
    }

    return static_cast<T*>(ptr);
}

/**
 * @brief Helper function to free 64-byte aligned memory
 */
template<typename T>
void deallocate_aligned(T* ptr) {
    if (ptr) {
        #ifdef _WIN32
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
    }
}

} // namespace mcts
