/**
 * @file backup.cpp
 * @brief Implementation of thread-safe value backup with sign flipping
 */

#include "backup.hpp"
#include "profiling/enhanced_profiler.hpp"
#include <algorithm>
#include <cmath>
#include <atomic>
#include <cassert>

using namespace mcts::profiling;

namespace mcts {

BackupManager::BackupManager(MCTSTree& tree, const BackupConfig& config)
    : tree_(tree), config_(config) {
}

BackupResult BackupManager::backup_value_along_path(
    const std::vector<NodeIndex>& path,
    float leaf_value,
    VirtualLossManager* virtual_loss_manager
) {
    BackupResult result;
    result.original_leaf_value = leaf_value;

    // Validate path first
    if (!validate_backup_path(path)) {
        if (config_.enable_statistics) {
            path_validation_failures_.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }

    // Clip leaf value if configured
    float current_value = config_.enable_value_clipping ? clip_value(leaf_value) : leaf_value;

    // Remove virtual loss first if manager provided
    if (virtual_loss_manager) {
        virtual_loss_manager->remove_virtual_loss_from_path(path);
    }

    // Backup from leaf to root with sign flipping
    // Values alternate perspective at each level: child -> parent requires negation
    std::size_t nodes_updated = 0;

    for (std::size_t i = 0; i < path.size(); ++i) {
        NodeIndex node_index = path[i];

        // Apply sign flipping: each level up the tree negates the value
        // This ensures values are from the current player's perspective at each node
        float value_for_this_node = (i % 2 == 0) ? current_value : -current_value;

        // Atomic update of visit count and total value
        if (update_node_atomic(node_index, value_for_this_node, 1.0f)) {
            nodes_updated++;
        } else {
            // If any update fails, consider the backup partially successful
            break;
        }
    }

    // Record final root value if we updated the root (last node in path)
    if (nodes_updated > 0 && nodes_updated == path.size()) {
        result.final_root_value = get_q_value(path.back());
        result.success = true;
    }

    result.nodes_updated = nodes_updated;

    // Update statistics
    if (config_.enable_statistics) {
        update_statistics(result, path.size(), leaf_value);
    }

    return result;
}

BackupResult BackupManager::backup_terminal_value(
    const std::vector<NodeIndex>& path,
    float terminal_value,
    VirtualLossManager* virtual_loss_manager
) {
    // Terminal values are exact game outcomes, so we handle them the same way
    // as neural network evaluations but with higher confidence in the value
    return backup_value_along_path(path, terminal_value, virtual_loss_manager);
}

bool BackupManager::update_node_atomic(
    NodeIndex node_index,
    float value_increment,
    float visit_increment
) {
    if (!tree_.is_valid_index(node_index)) {
        return false;
    }

    // Atomic update of visit count
    if (!atomic_add_visit_count(node_index, visit_increment)) {
        return false;
    }

    // Atomic update of total value
    if (!atomic_add_total_value(node_index, value_increment)) {
        // If value update fails, we should ideally rollback visit count
        // but for simplicity and performance, we'll accept the inconsistency
        // as it will be corrected by subsequent operations
        return false;
    }

    return true;
}

float BackupManager::get_q_value(NodeIndex node_index) const {
    if (!tree_.is_valid_index(node_index)) {
        return 0.0f;
    }

    // Thread-safe read of visit count and total value
    // Note: This is not perfectly atomic across both reads, but for
    // Q-value queries it's acceptable to have slight inconsistencies
    float visit_count = tree_.get_visit_count(node_index);
    float total_value = tree_.get_total_value(node_index);

    return (visit_count > 0.0f) ? (total_value / visit_count) : 0.0f;
}

bool BackupManager::validate_backup_path(const std::vector<NodeIndex>& path) const {
    if (path.empty()) {
        return false;
    }

    // Check that all nodes are valid
    for (NodeIndex node_index : path) {
        if (!tree_.is_valid_index(node_index)) {
            return false;
        }
    }

    // Check parent-child relationships
    // Path should go from leaf to root, so each node should be parent of previous
    for (std::size_t i = 1; i < path.size(); ++i) {
        NodeIndex child = path[i - 1];
        NodeIndex parent = path[i];

        // Verify that parent is actually the parent of child
        NodeIndex child_parent = tree_.get_parent_index(child);
        if (child_parent != parent) {
            return false;
        }
    }

    // Last node in path should be root (parent_index == NULL_NODE_INDEX)
    NodeIndex root_candidate = path.back();
    if (tree_.get_parent_index(root_candidate) != NULL_NODE_INDEX) {
        return false;
    }

    return true;
}

BackupManager::BackupStats BackupManager::get_statistics() const {
    BackupStats stats;

    if (!config_.enable_statistics) {
        return stats;
    }

    stats.total_backups = total_backups_.load(std::memory_order_relaxed);
    stats.successful_backups = successful_backups_.load(std::memory_order_relaxed);
    stats.total_nodes_updated = total_nodes_updated_.load(std::memory_order_relaxed);
    stats.path_validation_failures = path_validation_failures_.load(std::memory_order_relaxed);

    float total_path_length = cumulative_path_length_.load(std::memory_order_relaxed);
    float total_leaf_value = cumulative_leaf_value_.load(std::memory_order_relaxed);

    stats.avg_path_length = (stats.total_backups > 0) ?
        total_path_length / stats.total_backups : 0.0f;

    stats.avg_absolute_leaf_value = (stats.total_backups > 0) ?
        total_leaf_value / stats.total_backups : 0.0f;

    return stats;
}

void BackupManager::reset_statistics() {
    total_backups_.store(0, std::memory_order_relaxed);
    successful_backups_.store(0, std::memory_order_relaxed);
    total_nodes_updated_.store(0, std::memory_order_relaxed);
    path_validation_failures_.store(0, std::memory_order_relaxed);
    cumulative_path_length_.store(0.0f, std::memory_order_relaxed);
    cumulative_leaf_value_.store(0.0f, std::memory_order_relaxed);
}

float BackupManager::clip_value(float value) const {
    return std::clamp(value, config_.value_clip_min, config_.value_clip_max);
}

bool BackupManager::atomic_add_visit_count(NodeIndex node_index, float increment) {
    PROFILE_SCOPE(ProfileMetric::BackupAtomicOperations);

    // Get pointer to visit count array for atomic operations
    float* visit_counts_ptr = tree_.get_visit_counts_ptr();

    // Use atomic operations to safely update visit count
    std::atomic<float>* atomic_visit = reinterpret_cast<std::atomic<float>*>(&visit_counts_ptr[node_index]);

    // Track CAS retries (review.txt: atomic contention)
    int retry_count = 0;
    float expected, desired;
    do {
        expected = atomic_visit->load(std::memory_order_acquire);
        desired = expected + increment;

        // Ensure visit count doesn't go negative
        if (desired < 0.0f) {
            desired = 0.0f;
        }

        // Prevent excessive visit count accumulation (safety check)
        if (desired > 1000000.0f) {
            return false;  // Something is wrong - too many visits
        }

        // Track retry on failure
        if (retry_count > 0) {
            PROFILE_COUNTER(ProfileMetric::CAS_RetryCount, 1);
            PROFILE_COUNTER(ProfileMetric::BackupCASRetries, 1);
        }
        retry_count++;

    } while (!atomic_visit->compare_exchange_weak(expected, desired,
                                                  std::memory_order_release,
                                                  std::memory_order_acquire));

    // Track success/failure counts
    if (retry_count == 1) {
        PROFILE_COUNTER(ProfileMetric::CAS_SuccessCount, 1);
    } else {
        PROFILE_COUNTER(ProfileMetric::CAS_FailureCount, retry_count - 1);
        PROFILE_GAUGE(ProfileMetric::CAS_MaxRetriesPerOp, retry_count - 1);
    }

    return true;
}

bool BackupManager::atomic_add_total_value(NodeIndex node_index, float increment) {
    PROFILE_SCOPE(ProfileMetric::BackupValueUpdate);

    // Get pointer to total value array for atomic operations
    float* total_values_ptr = tree_.get_total_values_ptr();

    // Use atomic operations to safely update total value
    std::atomic<float>* atomic_value = reinterpret_cast<std::atomic<float>*>(&total_values_ptr[node_index]);

    // Track CAS retries
    int retry_count = 0;
    float expected, desired;
    do {
        expected = atomic_value->load(std::memory_order_acquire);
        desired = expected + increment;

        // Sanity check for extreme values
        if (std::abs(desired) > 1000000.0f) {
            return false;  // Something is wrong - extreme total value
        }

        // Track retry on failure
        if (retry_count > 0) {
            PROFILE_COUNTER(ProfileMetric::CAS_RetryCount, 1);
            PROFILE_COUNTER(ProfileMetric::BackupCASRetries, 1);
        }
        retry_count++;

    } while (!atomic_value->compare_exchange_weak(expected, desired,
                                                  std::memory_order_release,
                                                  std::memory_order_acquire));

    // Track success/failure counts
    if (retry_count == 1) {
        PROFILE_COUNTER(ProfileMetric::CAS_SuccessCount, 1);
    } else {
        PROFILE_COUNTER(ProfileMetric::CAS_FailureCount, retry_count - 1);
        PROFILE_GAUGE(ProfileMetric::CAS_MaxRetriesPerOp, retry_count - 1);
    }

    return true;
}

void BackupManager::update_statistics(const BackupResult& result, std::size_t path_length, float leaf_value) {
    total_backups_.fetch_add(1, std::memory_order_relaxed);

    if (result.success) {
        successful_backups_.fetch_add(1, std::memory_order_relaxed);
    }

    total_nodes_updated_.fetch_add(result.nodes_updated, std::memory_order_relaxed);

    // Update cumulative statistics for averages
    // Note: std::atomic<float>::fetch_add may not be available in all C++17 implementations
    // Use compare-and-swap loop instead
    float expected_path, desired_path;
    do {
        expected_path = cumulative_path_length_.load(std::memory_order_acquire);
        desired_path = expected_path + static_cast<float>(path_length);
    } while (!cumulative_path_length_.compare_exchange_weak(expected_path, desired_path,
                                                           std::memory_order_release,
                                                           std::memory_order_acquire));

    float expected_value, desired_value;
    do {
        expected_value = cumulative_leaf_value_.load(std::memory_order_acquire);
        desired_value = expected_value + std::abs(leaf_value);
    } while (!cumulative_leaf_value_.compare_exchange_weak(expected_value, desired_value,
                                                          std::memory_order_release,
                                                          std::memory_order_acquire));
}

// BackupGuard implementation

BackupGuard::BackupGuard(
    BackupManager& backup_manager,
    VirtualLossManager& virtual_loss_manager,
    const std::vector<NodeIndex>& path,
    float leaf_value
) : virtual_loss_manager_(virtual_loss_manager), path_(path), cleaned_up_(false) {

    // Perform backup with automatic virtual loss cleanup
    result_ = backup_manager.backup_value_along_path(path_, leaf_value, &virtual_loss_manager_);
}

BackupGuard::~BackupGuard() {
    if (!cleaned_up_) {
        cleanup();
    }
}

void BackupGuard::cleanup() {
    if (!cleaned_up_) {
        // Ensure virtual loss is removed even if backup failed
        virtual_loss_manager_.remove_virtual_loss_from_path(path_);
        cleaned_up_ = true;
    }
}

} // namespace mcts