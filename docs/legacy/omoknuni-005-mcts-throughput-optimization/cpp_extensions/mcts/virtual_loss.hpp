/**
 * @file virtual_loss.hpp
 * @brief Thread-safe virtual loss mechanism for MCTS tree search
 *
 * Virtual loss is a technique to prevent multiple search threads from
 * exploring the same path simultaneously. When a thread traverses down
 * the tree, it applies a temporary "virtual loss" to each node along
 * the path. This makes the path appear less attractive to other threads,
 * encouraging them to explore different branches.
 *
 * Key features:
 * - Thread-safe atomic operations on virtual loss values
 * - Configurable virtual loss magnitude (default 1.0)
 * - Path-based application and removal during tree traversal
 * - Integration with PUCT selection formula
 */

#pragma once

#include "tree.hpp"
#include <vector>
#include <atomic>
#include <cstdint>

namespace mcts {

/**
 * @brief Configuration for virtual loss behavior
 */
struct VirtualLossConfig {
    float magnitude = 1.0f;           // Virtual loss value to apply
    bool enable_virtual_loss = true;  // Enable/disable virtual loss

    VirtualLossConfig() = default;

    VirtualLossConfig(float mag, bool enable = true)
        : magnitude(mag), enable_virtual_loss(enable) {}
};

/**
 * @brief Thread-safe virtual loss manager for MCTS tree
 *
 * This class provides atomic operations for applying and removing
 * virtual loss along search paths. Virtual loss helps coordinate
 * multiple search threads by temporarily penalizing nodes being
 * explored by other threads.
 */
class VirtualLossManager {
public:
    /**
     * @brief Initialize virtual loss manager
     *
     * @param tree Reference to MCTS tree to manage
     * @param config Virtual loss configuration
     */
    explicit VirtualLossManager(MCTSTree& tree, const VirtualLossConfig& config = VirtualLossConfig());

    /**
     * @brief Apply virtual loss along a path from leaf to root
     *
     * This function should be called when a thread starts exploring
     * a path. It applies virtual loss to each node in the path to
     * discourage other threads from following the same route.
     *
     * @param path Vector of node indices from leaf to root
     * @return true if virtual loss was successfully applied to all nodes
     */
    bool apply_virtual_loss_to_path(const std::vector<NodeIndex>& path);

    /**
     * @brief Remove virtual loss along a path from leaf to root
     *
     * This function should be called when a thread finishes exploring
     * a path and is ready to backup the results. It removes the virtual
     * loss that was previously applied.
     *
     * @param path Vector of node indices from leaf to root (same as apply)
     * @return true if virtual loss was successfully removed from all nodes
     */
    bool remove_virtual_loss_from_path(const std::vector<NodeIndex>& path);

    /**
     * @brief Apply virtual loss to a single node atomically
     *
     * @param node_index Index of node to apply virtual loss to
     * @param magnitude Virtual loss value to add (default: config magnitude)
     * @return true if virtual loss was successfully applied
     */
    bool apply_virtual_loss(NodeIndex node_index, float magnitude = -1.0f);

    /**
     * @brief Remove virtual loss from a single node atomically
     *
     * @param node_index Index of node to remove virtual loss from
     * @param magnitude Virtual loss value to remove (default: config magnitude)
     * @return true if virtual loss was successfully removed
     */
    bool remove_virtual_loss(NodeIndex node_index, float magnitude = -1.0f);

    /**
     * @brief Get current virtual loss value for a node
     *
     * This is a non-atomic read for debugging purposes.
     * For thread-safe access during selection, use tree methods directly.
     *
     * @param node_index Index of node to query
     * @return Current virtual loss value
     */
    float get_virtual_loss(NodeIndex node_index) const;

    /**
     * @brief Reset all virtual loss values to zero
     *
     * Useful for debugging and testing. Should not be called
     * during active search operations.
     */
    void reset_all_virtual_loss();

    /**
     * @brief Get virtual loss configuration
     */
    const VirtualLossConfig& get_config() const { return config_; }

    /**
     * @brief Update virtual loss configuration
     *
     * @param new_config New configuration to apply
     */
    void set_config(const VirtualLossConfig& new_config) { config_ = new_config; }

    /**
     * @brief Get statistics about virtual loss usage
     *
     * @return Struct containing virtual loss statistics
     */
    struct VirtualLossStats {
        std::size_t total_applications = 0;    // Total times virtual loss was applied
        std::size_t total_removals = 0;        // Total times virtual loss was removed
        std::size_t current_active_paths = 0;  // Current number of active paths with virtual loss
        float max_virtual_loss = 0.0f;         // Maximum virtual loss value currently in tree
        float avg_virtual_loss = 0.0f;         // Average virtual loss value across all nodes
    };

    VirtualLossStats get_statistics() const;

private:
    MCTSTree& tree_;                    // Reference to MCTS tree
    VirtualLossConfig config_;          // Virtual loss configuration

    // Statistics tracking (atomic for thread safety)
    mutable std::atomic<std::size_t> total_applications_{0};
    mutable std::atomic<std::size_t> total_removals_{0};

    /**
     * @brief Validate node index before virtual loss operations
     */
    bool validate_node_index(NodeIndex node_index) const;

    /**
     * @brief Atomic add operation on virtual loss value
     *
     * Uses compare-and-swap loop to ensure thread-safe updates
     * to the virtual loss array.
     */
    bool atomic_add_virtual_loss(NodeIndex node_index, float delta);
};

/**
 * @brief RAII wrapper for automatic virtual loss management
 *
 * This class automatically applies virtual loss when constructed
 * and removes it when destroyed, ensuring proper cleanup even
 * if exceptions occur during search.
 */
class VirtualLossGuard {
public:
    /**
     * @brief Apply virtual loss to path and store for automatic removal
     *
     * @param manager Reference to virtual loss manager
     * @param path Path to apply virtual loss to
     */
    VirtualLossGuard(VirtualLossManager& manager, const std::vector<NodeIndex>& path);

    /**
     * @brief Remove virtual loss from stored path
     */
    ~VirtualLossGuard();

    // Disable copy/move to prevent double-removal
    VirtualLossGuard(const VirtualLossGuard&) = delete;
    VirtualLossGuard& operator=(const VirtualLossGuard&) = delete;
    VirtualLossGuard(VirtualLossGuard&&) = delete;
    VirtualLossGuard& operator=(VirtualLossGuard&&) = delete;

    /**
     * @brief Check if virtual loss was successfully applied
     */
    bool is_valid() const { return valid_; }

    /**
     * @brief Manually remove virtual loss (called automatically by destructor)
     */
    void release();

private:
    VirtualLossManager& manager_;
    std::vector<NodeIndex> path_;
    bool valid_;
    bool released_;
};

/**
 * @brief WU-UCT style virtual loss manager (visit-only, no Q-value distortion)
 *
 * Unlike classic virtual loss which modifies Q-values during selection,
 * WU-UCT only tracks in-flight simulations and adjusts the exploration
 * term's denominator. This prevents Q-value distortion while still
 * providing effective thread coordination.
 *
 * Formula change:
 * - Classic: Q = (W - VL) / (N + 1), U = P * sqrt(N_parent) / (1 + N)
 * - WU-UCT:  Q = W / N,              U = P * sqrt(N_parent) / (1 + N + VL)
 *
 * Benefits:
 * - Pure Q-values for accurate value estimates
 * - Virtual loss only discourages re-selection via exploration term
 * - More robust to virtual loss magnitude tuning
 * - Lower atomic contention (single counter vs value accumulation)
 */
class WUUCTVirtualLossManager {
public:
    /**
     * @brief Initialize WU-UCT virtual loss manager
     *
     * @param max_nodes Maximum number of tree nodes to support
     * @param virtual_loss_magnitude Scaling factor for virtual loss (default 1.0)
     */
    explicit WUUCTVirtualLossManager(
        std::size_t max_nodes,
        float virtual_loss_magnitude = 1.0f
    );

    /**
     * @brief Destructor - free allocated memory
     */
    ~WUUCTVirtualLossManager();

    // Disable copy/move (contains raw pointers)
    WUUCTVirtualLossManager(const WUUCTVirtualLossManager&) = delete;
    WUUCTVirtualLossManager& operator=(const WUUCTVirtualLossManager&) = delete;
    WUUCTVirtualLossManager(WUUCTVirtualLossManager&&) = delete;
    WUUCTVirtualLossManager& operator=(WUUCTVirtualLossManager&&) = delete;

    /**
     * @brief Apply virtual loss when thread starts exploring a node
     *
     * Thread-safe atomic increment of in-flight counter.
     * Does not modify node's Q-value.
     *
     * @param node_index Index of node being visited
     */
    void add_in_flight(NodeIndex node_index);

    /**
     * @brief Remove virtual loss when simulation completes
     *
     * Thread-safe atomic decrement of in-flight counter.
     *
     * @param node_index Index of node to update
     */
    void remove_in_flight(NodeIndex node_index);

    /**
     * @brief Get exploration term adjustment for PUCT calculation
     *
     * Returns the value to add to visit count in exploration denominator:
     * exploration = c_puct * P * sqrt(N_parent) / (1 + N + adjustment)
     *
     * @param node_index Node to query
     * @return Adjustment value (in_flight_count * magnitude)
     */
    float get_exploration_adjustment(NodeIndex node_index) const;

    /**
     * @brief Check if node is currently being explored (busy-edge masking)
     *
     * @param node_index Node to check
     * @return True if node has active in-flight simulations
     */
    bool is_busy(NodeIndex node_index) const;

    /**
     * @brief Get number of in-flight simulations for a node
     *
     * @param node_index Node to query
     * @return Current in-flight count
     */
    std::uint32_t get_in_flight_count(NodeIndex node_index) const;

    /**
     * @brief Get total selection collision count
     *
     * Tracks how many times threads selected same path
     *
     * @return Total collision count since creation
     */
    std::uint64_t get_collision_count() const {
        return collision_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset collision tracking statistics
     */
    void reset_statistics() {
        collision_count_.store(0, std::memory_order_relaxed);
    }

    /**
     * @brief Update virtual loss magnitude
     *
     * @param new_magnitude New scaling factor
     */
    void set_magnitude(float new_magnitude) {
        magnitude_ = new_magnitude;
    }

    /**
     * @brief Get current virtual loss magnitude
     */
    float get_magnitude() const {
        return magnitude_;
    }

    /**
     * @brief Clear all in-flight counts (for tree reuse)
     */
    void clear_all();

private:
    // In-flight simulation counts (cache-aligned for performance)
    // Note: Using raw array with manual allocation to avoid std::atomic copy issues
    std::atomic<std::uint32_t>* in_flight_;  // One per node
    std::size_t max_nodes_;                   // Size of in_flight_ array
    float magnitude_;                         // Virtual loss scaling factor
    std::atomic<std::uint64_t> collision_count_{0};  // Collision tracking

    /**
     * @brief Validate node index
     */
    bool is_valid_index(NodeIndex node_index) const {
        return node_index >= 0 && static_cast<std::size_t>(node_index) < max_nodes_;
    }
};

/**
 * @brief RAII guard for automatic WU-UCT virtual loss management
 *
 * Applies virtual loss on construction and removes on destruction.
 * Handles paths of nodes for simulation traversal.
 */
class WUUCTVirtualLossGuard {
public:
    /**
     * @brief Apply virtual loss to entire path
     *
     * @param manager WU-UCT manager instance
     * @param path Vector of nodes from leaf to root
     */
    WUUCTVirtualLossGuard(
        WUUCTVirtualLossManager& manager,
        const std::vector<NodeIndex>& path
    );

    /**
     * @brief Remove virtual loss from path
     */
    ~WUUCTVirtualLossGuard();

    // Disable copy/move
    WUUCTVirtualLossGuard(const WUUCTVirtualLossGuard&) = delete;
    WUUCTVirtualLossGuard& operator=(const WUUCTVirtualLossGuard&) = delete;
    WUUCTVirtualLossGuard(WUUCTVirtualLossGuard&&) = delete;
    WUUCTVirtualLossGuard& operator=(WUUCTVirtualLossGuard&&) = delete;

    /**
     * @brief Manually remove virtual loss (called automatically by destructor)
     */
    void release();

private:
    WUUCTVirtualLossManager& manager_;
    std::vector<NodeIndex> path_;
    bool released_;
};

} // namespace mcts