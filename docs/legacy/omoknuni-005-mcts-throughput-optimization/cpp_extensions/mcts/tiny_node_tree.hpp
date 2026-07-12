// tiny_node_tree.hpp - Tree storage using TinyNode AoS layout with bump allocator
// Part of T024f-1: TinyNode Storage Layer

#pragma once

#include "tiny_node.hpp"
#include <cstdint>
#include <atomic>
#include <mutex>
#include <vector>
#include <memory>
#include <cassert>

namespace mcts {

/**
 * @brief MCTS tree using TinyNode array-of-structs layout
 *
 * This class implements zero-copy MCTS tree storage where:
 * - Nodes are 64-byte aligned structs (TinyNode)
 * - O(1) bump allocation for fast node creation
 * - Free list for node reuse
 * - Thread-safe allocation via atomics
 * - No state cloning (stores only moves + statistics)
 *
 * Memory layout:
 * - Single contiguous array of TinyNode structs
 * - Each node: 64 bytes (34 bytes data + 30 bytes padding)
 * - 10M nodes = 640 MB (vs 1.2 GB with state cloning)
 */
class TinyNodeTree {
public:
    /**
     * @brief Initialize tree with specified capacity
     *
     * @param max_nodes Maximum number of nodes to support
     */
    explicit TinyNodeTree(std::size_t max_nodes = 50'000'000);

    /**
     * @brief Destructor - frees aligned memory
     */
    ~TinyNodeTree();

    // Disable copy/move to avoid complexity
    TinyNodeTree(const TinyNodeTree&) = delete;
    TinyNodeTree& operator=(const TinyNodeTree&) = delete;
    TinyNodeTree(TinyNodeTree&&) = delete;
    TinyNodeTree& operator=(TinyNodeTree&&) = delete;

    /**
     * @brief Allocate a single node from the pool (O(1) bump allocation)
     *
     * Thread-safe: Uses atomic increment for next_index_
     * Fast path: Bump allocator (no lock)
     * Slow path: Free list (with lock)
     *
     * @return Index of allocated node, or -1 if pool is full
     */
    int32_t allocate_node();

    /**
     * @brief Deallocate a single node back to the pool
     *
     * Note: Node is added to free list for reuse.
     * Node data is NOT cleared (will be overwritten on reuse).
     *
     * @param index Index of node to deallocate
     */
    void deallocate_node(int32_t index);

    /**
     * @brief Clear all nodes and reset tree to empty state
     *
     * O(1) operation: Just resets allocation index and clears free list.
     * Memory is NOT zeroed (nodes will be initialized on allocation).
     */
    void clear();

    /**
     * @brief Get pointer to node by index
     *
     * WARNING: Pointer may be invalidated by allocate_node() if reallocation occurs.
     * Use node index as the canonical reference, not pointers.
     *
     * @param index Node index (0 to node_count_ - 1)
     * @return Pointer to TinyNode, or nullptr if index invalid
     */
    TinyNode* get_node(int32_t index);

    /**
     * @brief Get const pointer to node by index
     */
    const TinyNode* get_node(int32_t index) const;

    /**
     * @brief Check if node index is valid
     */
    bool is_valid_index(int32_t index) const {
        return index >= 0 && static_cast<std::size_t>(index) < next_index_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get current number of allocated nodes
     */
    std::size_t get_node_count() const {
        return next_index_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get maximum capacity
     */
    std::size_t get_max_nodes() const {
        return max_nodes_;
    }

    /**
     * @brief Get memory usage in bytes
     */
    std::size_t get_memory_usage() const {
        return max_nodes_ * sizeof(TinyNode);
    }

    /**
     * @brief Get bytes per node (actual footprint)
     */
    double get_bytes_per_node() const {
        return sizeof(TinyNode);  // Always 64 bytes (aligned)
    }

    /**
     * @brief Get root node index (always 0 if tree has nodes)
     */
    int32_t get_root_index() const {
        return get_node_count() > 0 ? 0 : -1;
    }

    /**
     * @brief Check if tree has space for additional nodes
     *
     * @param count Number of nodes to check for
     * @return true if space available
     */
    bool has_space_for(std::size_t count) const {
        std::size_t current = next_index_.load(std::memory_order_relaxed);
        std::size_t available_bump = (current < max_nodes_) ? (max_nodes_ - current) : 0;

        std::lock_guard<std::mutex> lock(free_list_mutex_);
        std::size_t available_free = free_list_.size();

        return (available_bump + available_free) >= count;
    }

    /**
     * @brief Initialize root node (called once at start of search)
     *
     * @param zobrist_hash Initial zobrist hash for root position
     * @return Index of root node (always 0)
     */
    int32_t init_root(uint64_t zobrist_hash);

    /**
     * @brief Validate tree structure and constraints
     *
     * @return true if tree is valid
     */
    bool validate() const;

    // ====== Child Management (T024f-2) ======

    /**
     * @brief Add a child node to a parent
     *
     * Creates a new child node linked to the parent via sibling pointers.
     * Children are stored as a singly-linked list:
     *   parent.first_child_idx → child1.next_sibling_idx → child2.next_sibling_idx → ...
     *
     * @param parent_idx Parent node index
     * @param move Move that leads to this child (uint16_t action index)
     * @param prior_prob Prior probability from policy network (0.0-1.0)
     * @param zobrist_hash Zobrist hash for this child position
     * @return Index of created child, or -1 if allocation fails
     */
    int32_t add_child(int32_t parent_idx, uint16_t move, float prior_prob, uint64_t zobrist_hash);

    /**
     * @brief Expand a node by adding all legal children
     *
     * Creates child nodes for all legal moves with their prior probabilities.
     * This is the typical MCTS expansion operation after neural network inference.
     *
     * @param parent_idx Parent node to expand
     * @param moves Array of legal moves (uint16_t action indices)
     * @param priors Array of prior probabilities (must sum to ~1.0)
     * @param zobrist_hashes Array of zobrist hashes for each child position
     * @param num_children Number of children to add
     * @return true if all children added successfully, false if allocation fails
     */
    bool expand_node(int32_t parent_idx, const uint16_t* moves, const float* priors,
                     const uint64_t* zobrist_hashes, size_t num_children);

    /**
     * @brief Get number of children for a node
     *
     * Counts children by iterating the sibling-linked list.
     * O(n) where n = number of children (typically small, <100).
     *
     * @param parent_idx Parent node index
     * @return Number of children
     */
    size_t get_child_count(int32_t parent_idx) const;

    /**
     * @brief Iterate children and call a function for each
     *
     * Template function that calls visitor(child_idx) for each child.
     * Useful for PUCT selection, value backup, etc.
     *
     * Example:
     *   tree.for_each_child(parent_idx, [&](int32_t child_idx) {
     *       TinyNode* child = tree.get_node(child_idx);
     *       // ... process child ...
     *   });
     *
     * @param parent_idx Parent node index
     * @param visitor Function to call for each child (receives child_idx)
     */
    template<typename Visitor>
    void for_each_child(int32_t parent_idx, Visitor&& visitor) const {
        if (!is_valid_index(parent_idx)) {
            return;
        }

        const TinyNode* parent = get_node(parent_idx);
        int32_t child_idx = static_cast<int32_t>(parent->first_child_idx);

        while (child_idx != 0) {
            visitor(child_idx);

            const TinyNode* child = get_node(child_idx);
            child_idx = static_cast<int32_t>(child->next_sibling_idx);
        }
    }

    /**
     * @brief Get all child indices as a vector
     *
     * Convenience method that collects all children into a vector.
     * Less efficient than for_each_child for iteration.
     *
     * @param parent_idx Parent node index
     * @return Vector of child indices
     */
    std::vector<int32_t> get_children(int32_t parent_idx) const;

    // ====== Path Traversal (T024f-3) ======

    /**
     * @brief Collect path from root to a node
     *
     * Returns the sequence of node indices from root to target node.
     * Useful for reconstructing game state via make/unmake.
     *
     * @param node_idx Target node index
     * @return Vector of node indices [root, ..., node_idx]
     */
    std::vector<int32_t> get_path_to_node(int32_t node_idx) const;

    /**
     * @brief Select best child using PUCT formula
     *
     * PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N_child + VL_child)
     * where:
     * - Q = W/N (mean value)
     * - P = prior probability
     * - N = visit count
     * - VL = virtual loss
     *
     * @param parent_idx Parent node to select from
     * @param c_puct Exploration constant (typically 1.0-2.0)
     * @return Index of best child, or -1 if no children
     */
    int32_t select_best_child(int32_t parent_idx, float c_puct) const;

    /**
     * @brief Apply virtual loss to a node
     *
     * Thread-safe increment of virtual loss counter.
     * Used during parallel MCTS to prevent multiple threads
     * from exploring the same path.
     *
     * @param node_idx Node to apply virtual loss to
     * @param magnitude Virtual loss magnitude (default 1.0)
     */
    void apply_virtual_loss(int32_t node_idx, uint8_t magnitude = 1);

    /**
     * @brief Remove virtual loss from a node
     *
     * Thread-safe decrement of virtual loss counter.
     * Called after backup completes.
     *
     * @param node_idx Node to remove virtual loss from
     * @param magnitude Virtual loss magnitude (default 1.0)
     */
    void remove_virtual_loss(int32_t node_idx, uint8_t magnitude = 1);

    /**
     * @brief Backup value from leaf to root
     *
     * Propagates value up the tree, updating visit counts and total values.
     * Thread-safe using atomic operations.
     *
     * Value sign flips at each level (negamax framework):
     * - If leaf has value +0.8 for player A
     * - Parent gets -0.8 (from player B's perspective)
     * - Grandparent gets +0.8 (back to player A)
     *
     * @param path Path from root to leaf (from get_path_to_node)
     * @param leaf_value Value at the leaf node (from neural network)
     */
    void backup_value(const std::vector<int32_t>& path, float leaf_value);

private:
    // Maximum capacity
    std::size_t max_nodes_;

    // Next index for bump allocation (atomic for thread-safety)
    std::atomic<std::size_t> next_index_{0};

    // Free list for node reuse
    std::vector<int32_t> free_list_;
    mutable std::mutex free_list_mutex_;

    // Node storage (64-byte aligned array)
    TinyNode* nodes_;

    /**
     * @brief Allocate aligned memory for node array
     */
    void allocate_array();

    /**
     * @brief Free aligned memory
     */
    void deallocate_array();

    /**
     * @brief Initialize a single node to default state
     */
    void init_node(int32_t index);
};

} // namespace mcts
