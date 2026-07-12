/**
 * @file backup.hpp
 * @brief Thread-safe value backup mechanism for MCTS with sign flipping
 *
 * This module implements the backup phase of MCTS where leaf evaluation
 * results are propagated back up the tree to the root. The key feature is
 * proper value sign flipping at each tree level to maintain correct
 * perspective from each player's viewpoint.
 *
 * Key features:
 * - Atomic visit count and value updates for thread safety
 * - Proper value sign alternation per ply level
 * - Path traversal from leaf to root
 * - Integration with virtual loss removal
 * - Performance optimized with minimal atomic operations
 */

#pragma once

#include "tree.hpp"
#include "virtual_loss.hpp"
#include <vector>
#include <atomic>
#include <cstdint>

namespace mcts {

/**
 * @brief Configuration for backup behavior
 */
struct BackupConfig {
    bool enable_value_clipping = true;  // Clip values to [-1, 1] range
    bool enable_statistics = true;      // Track backup statistics
    float value_clip_min = -1.0f;       // Minimum value after clipping
    float value_clip_max = 1.0f;        // Maximum value after clipping

    BackupConfig() = default;

    BackupConfig(bool clip, bool stats = true, float min_val = -1.0f, float max_val = 1.0f)
        : enable_value_clipping(clip), enable_statistics(stats),
          value_clip_min(min_val), value_clip_max(max_val) {}
};

/**
 * @brief Result of a backup operation
 */
struct BackupResult {
    bool success = false;               // Whether backup completed successfully
    std::size_t nodes_updated = 0;     // Number of nodes updated in the path
    float final_root_value = 0.0f;     // Final Q-value at root after backup
    float original_leaf_value = 0.0f;  // Original leaf value before sign flipping
};

/**
 * @brief Thread-safe backup manager for MCTS tree
 *
 * This class handles the backup phase of MCTS search, propagating leaf
 * evaluation results back up the tree with proper value sign flipping.
 * Each level of the tree alternates player perspective, so values must
 * be negated at each step up the tree.
 */
class BackupManager {
public:
    /**
     * @brief Initialize backup manager
     *
     * @param tree Reference to MCTS tree
     * @param config Backup configuration
     */
    explicit BackupManager(MCTSTree& tree, const BackupConfig& config = BackupConfig());

    /**
     * @brief Backup a leaf evaluation result along a path to root
     *
     * This is the main backup function. It takes a leaf value and path,
     * then propagates the value up the tree with proper sign flipping
     * at each level. Values are from the current player's perspective
     * at each node.
     *
     * @param path Vector of node indices from leaf to root
     * @param leaf_value Evaluation result at the leaf node [-1, 1]
     * @param virtual_loss_manager Optional VL manager to remove virtual loss
     * @return BackupResult with success status and statistics
     */
    BackupResult backup_value_along_path(
        const std::vector<NodeIndex>& path,
        float leaf_value,
        VirtualLossManager* virtual_loss_manager = nullptr
    );

    /**
     * @brief Backup a terminal game result along a path
     *
     * Special case of backup for terminal positions where the game
     * outcome is known (win/loss/draw). Terminal values are typically
     * +1 for win, 0 for draw, -1 for loss from current player perspective.
     *
     * @param path Vector of node indices from leaf to root
     * @param terminal_value Game outcome value [-1, 1]
     * @param virtual_loss_manager Optional VL manager to remove virtual loss
     * @return BackupResult with success status and statistics
     */
    BackupResult backup_terminal_value(
        const std::vector<NodeIndex>& path,
        float terminal_value,
        VirtualLossManager* virtual_loss_manager = nullptr
    );

    /**
     * @brief Update a single node with value and visit count atomically
     *
     * Low-level function to atomically increment visit count and update
     * total value for a single node. Used internally by path backup.
     *
     * @param node_index Index of node to update
     * @param value_increment Value to add to total_value
     * @param visit_increment Visit count increment (usually 1.0)
     * @return true if update was successful
     */
    bool update_node_atomic(
        NodeIndex node_index,
        float value_increment,
        float visit_increment = 1.0f
    );

    /**
     * @brief Get current Q-value for a node (total_value / visit_count)
     *
     * Thread-safe computation of Q-value using atomic reads.
     * Returns 0 for unvisited nodes.
     *
     * @param node_index Index of node to query
     * @return Current Q-value [-1, 1]
     */
    float get_q_value(NodeIndex node_index) const;

    /**
     * @brief Validate a backup path for correctness
     *
     * Checks that:
     * - All nodes in path are valid
     * - Path represents valid parent-child relationships
     * - Path goes from leaf to root
     *
     * @param path Vector of node indices to validate
     * @return true if path is valid for backup
     */
    bool validate_backup_path(const std::vector<NodeIndex>& path) const;

    /**
     * @brief Get backup configuration
     */
    const BackupConfig& get_config() const { return config_; }

    /**
     * @brief Update backup configuration
     */
    void set_config(const BackupConfig& new_config) { config_ = new_config; }

    /**
     * @brief Statistics about backup operations
     */
    struct BackupStats {
        std::size_t total_backups = 0;          // Total backup operations
        std::size_t successful_backups = 0;     // Successful backup operations
        std::size_t total_nodes_updated = 0;    // Total nodes updated across all backups
        std::size_t path_validation_failures = 0; // Invalid paths encountered
        float avg_path_length = 0.0f;           // Average backup path length
        float avg_absolute_leaf_value = 0.0f;   // Average magnitude of leaf values
    };

    /**
     * @brief Get backup statistics
     */
    BackupStats get_statistics() const;

    /**
     * @brief Reset backup statistics
     */
    void reset_statistics();

private:
    MCTSTree& tree_;                    // Reference to MCTS tree
    BackupConfig config_;               // Backup configuration

    // Statistics tracking (atomic for thread safety)
    mutable std::atomic<std::size_t> total_backups_{0};
    mutable std::atomic<std::size_t> successful_backups_{0};
    mutable std::atomic<std::size_t> total_nodes_updated_{0};
    mutable std::atomic<std::size_t> path_validation_failures_{0};
    mutable std::atomic<float> cumulative_path_length_{0.0f};
    mutable std::atomic<float> cumulative_leaf_value_{0.0f};

    /**
     * @brief Clip value to configured range
     */
    float clip_value(float value) const;

    /**
     * @brief Atomic compare-and-swap update for visit count
     */
    bool atomic_add_visit_count(NodeIndex node_index, float increment);

    /**
     * @brief Atomic compare-and-swap update for total value
     */
    bool atomic_add_total_value(NodeIndex node_index, float increment);

    /**
     * @brief Update statistics after a backup operation
     */
    void update_statistics(const BackupResult& result, std::size_t path_length, float leaf_value);
};

/**
 * @brief RAII wrapper for backup with automatic virtual loss cleanup
 *
 * This class combines backup with virtual loss removal in a single
 * atomic operation, ensuring that virtual loss is always cleaned up
 * even if backup fails or exceptions occur.
 */
class BackupGuard {
public:
    /**
     * @brief Perform backup and remove virtual loss
     *
     * @param backup_manager Reference to backup manager
     * @param virtual_loss_manager Reference to virtual loss manager
     * @param path Path for backup and virtual loss cleanup
     * @param leaf_value Value to backup
     */
    BackupGuard(
        BackupManager& backup_manager,
        VirtualLossManager& virtual_loss_manager,
        const std::vector<NodeIndex>& path,
        float leaf_value
    );

    /**
     * @brief Destructor ensures virtual loss cleanup
     */
    ~BackupGuard();

    // Disable copy/move to prevent double cleanup
    BackupGuard(const BackupGuard&) = delete;
    BackupGuard& operator=(const BackupGuard&) = delete;
    BackupGuard(BackupGuard&&) = delete;
    BackupGuard& operator=(BackupGuard&&) = delete;

    /**
     * @brief Check if backup was successful
     */
    bool was_successful() const { return result_.success; }

    /**
     * @brief Get backup result
     */
    const BackupResult& get_result() const { return result_; }

    /**
     * @brief Manually trigger virtual loss cleanup (called automatically by destructor)
     */
    void cleanup();

private:
    VirtualLossManager& virtual_loss_manager_;
    std::vector<NodeIndex> path_;
    BackupResult result_;
    bool cleaned_up_;
};

} // namespace mcts