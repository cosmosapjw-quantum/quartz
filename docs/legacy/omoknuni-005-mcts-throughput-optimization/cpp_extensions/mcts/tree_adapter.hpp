// tree_adapter.hpp - Adapter to expose TinyNodeTree with MCTSTree-compatible API
// Part of T024f-5: Adapter Layer

#pragma once

#include "tiny_node_tree.hpp"
#include "tree.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace mcts {

/**
 * @brief Adapter class that wraps TinyNodeTree to provide MCTSTree-compatible API
 *
 * This adapter enables A/B testing between old MCTSTree (Structure-of-Arrays)
 * and new TinyNodeTree (Array-of-Structs with zero-copy design) without
 * changing simulation runner code.
 *
 * Key differences handled by adapter:
 * - MCTSTree uses float for visit_count/total_value, TinyNode uses atomics
 * - MCTSTree uses array-based children (first_child + num_children)
 * - TinyNode uses sibling-linked children (first_child + next_sibling)
 * - MCTSTree has separate flags, TinyNode packs flags in uint8_t
 * - TinyNode adds zobrist_hash (not in MCTSTree)
 *
 * Performance characteristics:
 * - Same O(1) allocation as MCTSTree
 * - Same O(1) accessor performance
 * - Zero overhead in release builds (inlined methods)
 * - Thread-safe atomics for visit_count and total_value
 *
 * Usage:
 *   TreeAdapter tree(10'000'000);  // Wraps TinyNodeTree
 *   NodeIndex root = tree.add_root_node(1.0f, 0);
 *   tree.set_visit_count(root, 10.0f);
 *   float n = tree.get_visit_count(root);  // Works like MCTSTree
 */
class TreeAdapter {
public:
    /**
     * @brief Initialize adapter with specified capacity
     *
     * @param max_nodes Maximum number of nodes to support
     */
    explicit TreeAdapter(std::size_t max_nodes = 50'000'000);

    /**
     * @brief Destructor
     */
    ~TreeAdapter() = default;

    // Disable copy/move to match MCTSTree behavior
    TreeAdapter(const TreeAdapter&) = delete;
    TreeAdapter& operator=(const TreeAdapter&) = delete;
    TreeAdapter(TreeAdapter&&) = delete;
    TreeAdapter& operator=(TreeAdapter&&) = delete;

    // ====== Tree Management (MCTSTree-compatible) ======

    /**
     * @brief Get the root node index (always 0 when tree has nodes)
     */
    NodeIndex get_root_index() const {
        return tree_->get_root_index();
    }

    /**
     * @brief Get current number of nodes in tree
     */
    std::size_t get_node_count() const {
        return tree_->get_node_count();
    }

    /**
     * @brief Get maximum capacity of tree
     */
    std::size_t get_max_nodes() const {
        return tree_->get_max_nodes();
    }

    /**
     * @brief Get memory usage in bytes
     */
    std::size_t get_memory_usage() const {
        return tree_->get_memory_usage();
    }

    /**
     * @brief Get bytes per node (actual memory efficiency)
     */
    double get_bytes_per_node() const {
        return tree_->get_bytes_per_node();
    }

    /**
     * @brief Clear all nodes and reset tree
     */
    void clear() {
        tree_->clear();
    }

    /**
     * @brief Add root node to empty tree
     *
     * @param prior_prob Prior probability from neural network
     * @param current_player Current player (0 or 1) - stored in flags
     * @param zobrist_hash Initial zobrist hash (default 0)
     * @return Index of created root node (always 0)
     */
    NodeIndex add_root_node(float prior_prob, std::uint8_t current_player, uint64_t zobrist_hash = 0);

    /**
     * @brief Allocate a single node from the pool
     *
     * @return Index of allocated node, or NULL_NODE_INDEX if pool is full
     */
    NodeIndex allocate_node() {
        return tree_->allocate_node();
    }

    /**
     * @brief Allocate multiple contiguous nodes from the pool
     *
     * Note: TinyNodeTree doesn't have contiguous allocation optimization,
     * so this falls back to allocating nodes individually.
     *
     * @param count Number of nodes to allocate
     * @return Index of first allocated node, or NULL_NODE_INDEX if insufficient space
     */
    NodeIndex allocate_nodes(std::uint16_t count);

    /**
     * @brief Deallocate a single node back to the pool
     *
     * @param index Index of node to deallocate
     */
    void deallocate_node(NodeIndex index) {
        tree_->deallocate_node(index);
    }

    /**
     * @brief Deallocate multiple contiguous nodes back to the pool
     *
     * Note: TinyNodeTree doesn't track contiguous blocks, so this
     * deallocates nodes individually.
     *
     * @param first_index Index of first node to deallocate
     * @param count Number of nodes to deallocate
     */
    void deallocate_nodes(NodeIndex first_index, std::uint16_t count);

    /**
     * @brief Get number of available nodes in the pool
     */
    std::size_t get_available_nodes() const;

    /**
     * @brief Check if the tree has space for additional nodes
     *
     * @param count Number of nodes to check for
     * @return true if space is available
     */
    bool has_space_for(std::uint16_t count) const {
        return tree_->has_space_for(count);
    }

    /**
     * @brief Validate tree structure and constraints
     *
     * @return true if tree is valid
     */
    bool validate_tree() const {
        return tree_->validate();
    }

    // ====== Node Data Accessors (MCTSTree-compatible) ======

    /**
     * @brief Get visit count for node
     */
    float get_visit_count(NodeIndex index) const;

    /**
     * @brief Get total value for node
     */
    float get_total_value(NodeIndex index) const;

    /**
     * @brief Get prior probability for node
     */
    float get_prior_prob(NodeIndex index) const;

    /**
     * @brief Get virtual loss for node
     */
    float get_virtual_loss(NodeIndex index) const;

    /**
     * @brief Get parent index for node
     */
    NodeIndex get_parent_index(NodeIndex index) const;

    /**
     * @brief Get first child index for node
     */
    NodeIndex get_first_child_index(NodeIndex index) const;

    /**
     * @brief Get number of children for node
     *
     * Note: This is O(n) in TinyNodeTree (walks sibling list) vs O(1) in MCTSTree.
     * However, n is typically small (<100 children per node).
     */
    std::uint16_t get_num_children(NodeIndex index) const;

    /**
     * @brief Get flags for node
     */
    NodeFlags get_flags(NodeIndex index) const;

    /**
     * @brief Get complete node information for debugging
     */
    NodeInfo get_node_info(NodeIndex index) const;

    // ====== Node Data Mutators (MCTSTree-compatible) ======

    /**
     * @brief Set visit count for node
     */
    void set_visit_count(NodeIndex index, float value);

    /**
     * @brief Set total value for node
     */
    void set_total_value(NodeIndex index, float value);

    /**
     * @brief Set prior probability for node
     */
    void set_prior_prob(NodeIndex index, float value);

    /**
     * @brief Set virtual loss for node
     */
    void set_virtual_loss(NodeIndex index, float value);

    /**
     * @brief Set parent index for node
     */
    void set_parent_index(NodeIndex index, NodeIndex parent);

    /**
     * @brief Set first child index for node
     */
    void set_first_child_index(NodeIndex index, NodeIndex first_child);

    /**
     * @brief Set number of children for node
     *
     * WARNING: This is a NO-OP in TinyNodeTree adapter!
     * TinyNodeTree uses sibling-linked lists, so num_children is derived
     * by counting siblings, not stored directly.
     */
    void set_num_children(NodeIndex index, std::uint16_t count);

    /**
     * @brief Set flags for node
     */
    void set_flags(NodeIndex index, const NodeFlags& flags);

    // ====== Atomic Operations (MCTSTree-compatible) ======

    /**
     * @brief Atomically try to set expanded flag
     *
     * @param index Node index to update
     * @return true if we successfully set expanded=true
     */
    bool atomic_try_set_expanded(NodeIndex index);

    /**
     * @brief Atomically mark node as being expanded
     *
     * @param index Node index to update
     * @return true if we successfully set expanding=true
     */
    bool atomic_try_mark_expanding(NodeIndex index);

    /**
     * @brief Clear the expanding flag on a node
     *
     * @param index Node index to update
     */
    void clear_expanding_flag(NodeIndex index);

    // ====== TinyNodeTree-Specific Extensions ======

    /**
     * @brief Get zobrist hash for node (TinyNode extension)
     */
    uint64_t get_zobrist_hash(NodeIndex index) const;

    /**
     * @brief Set zobrist hash for node (TinyNode extension)
     */
    void set_zobrist_hash(NodeIndex index, uint64_t hash);

    /**
     * @brief Get move that led to this node (TinyNode extension)
     */
    uint16_t get_move(NodeIndex index) const;

    /**
     * @brief Set move that led to this node (TinyNode extension)
     */
    void set_move(NodeIndex index, uint16_t move);

    /**
     * @brief Get underlying TinyNodeTree (for advanced use cases)
     */
    TinyNodeTree* get_tiny_tree() {
        return tree_.get();
    }

    /**
     * @brief Get underlying TinyNodeTree (const version)
     */
    const TinyNodeTree* get_tiny_tree() const {
        return tree_.get();
    }

private:
    // Underlying TinyNodeTree implementation
    std::unique_ptr<TinyNodeTree> tree_;

    // Validate node index
    bool is_valid_index(NodeIndex index) const {
        return tree_->is_valid_index(index);
    }
};

} // namespace mcts
