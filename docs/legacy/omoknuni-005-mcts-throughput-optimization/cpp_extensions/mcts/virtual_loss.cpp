/**
 * @file virtual_loss.cpp
 * @brief Implementation of thread-safe virtual loss mechanism for MCTS
 */

#include "virtual_loss.hpp"
#include "instrumentation.hpp"
#include <algorithm>
#include <cmath>
#include <atomic>
#include <cassert>

namespace mcts {

VirtualLossManager::VirtualLossManager(MCTSTree& tree, const VirtualLossConfig& config)
    : tree_(tree), config_(config) {
}

bool VirtualLossManager::apply_virtual_loss_to_path(const std::vector<NodeIndex>& path) {
    if (!config_.enable_virtual_loss || path.empty()) {
        return true;  // Nothing to do
    }

    // Apply virtual loss to each node in the path
    // Start from leaf (first element) and go towards root
    for (NodeIndex node_index : path) {
        if (!apply_virtual_loss(node_index)) {
            // If any application fails, remove virtual loss from previously processed nodes
            // This ensures consistent state even if something goes wrong
            for (auto it = path.begin(); it != std::find(path.begin(), path.end(), node_index); ++it) {
                remove_virtual_loss(*it);
            }
            return false;
        }
    }

    return true;
}

bool VirtualLossManager::remove_virtual_loss_from_path(const std::vector<NodeIndex>& path) {
    if (!config_.enable_virtual_loss || path.empty()) {
        return true;  // Nothing to do
    }

    bool all_success = true;

    // Remove virtual loss from each node in the path
    // Order doesn't matter for removal, but we'll process in same order as application
    for (NodeIndex node_index : path) {
        if (!remove_virtual_loss(node_index)) {
            all_success = false;
            // Continue processing even if one fails - we want to clean up as much as possible
        }
    }

    return all_success;
}

bool VirtualLossManager::apply_virtual_loss(NodeIndex node_index, float magnitude) {
    // CRITICAL: Check if virtual loss is enabled
    if (!config_.enable_virtual_loss) {
        return true;  // Success but no-op when disabled
    }

    if (!validate_node_index(node_index)) {
        return false;
    }

    float actual_magnitude = (magnitude < 0.0f) ? config_.magnitude : magnitude;

    if (atomic_add_virtual_loss(node_index, actual_magnitude)) {
        Instrumentation::instance().increment_counter(InstrumentationMetric::VirtualLossApply);
        total_applications_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    return false;
}

bool VirtualLossManager::remove_virtual_loss(NodeIndex node_index, float magnitude) {
    // CRITICAL: Check if virtual loss is enabled
    if (!config_.enable_virtual_loss) {
        return true;  // Success but no-op when disabled
    }

    if (!validate_node_index(node_index)) {
        return false;
    }

    float actual_magnitude = (magnitude < 0.0f) ? config_.magnitude : magnitude;

    if (atomic_add_virtual_loss(node_index, -actual_magnitude)) {
        Instrumentation::instance().increment_counter(InstrumentationMetric::VirtualLossRemove);
        total_removals_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    return false;
}

float VirtualLossManager::get_virtual_loss(NodeIndex node_index) const {
    if (!validate_node_index(node_index)) {
        return 0.0f;
    }

    return tree_.get_virtual_loss(node_index);
}

void VirtualLossManager::reset_all_virtual_loss() {
    std::size_t node_count = tree_.get_node_count();

    for (std::size_t i = 0; i < node_count; ++i) {
        NodeIndex node_index = static_cast<NodeIndex>(i);
        if (tree_.is_valid_index(node_index)) {
            tree_.set_virtual_loss(node_index, 0.0f);
        }
    }

    // Reset statistics
    total_applications_.store(0, std::memory_order_relaxed);
    total_removals_.store(0, std::memory_order_relaxed);
}

VirtualLossManager::VirtualLossStats VirtualLossManager::get_statistics() const {
    VirtualLossStats stats;

    stats.total_applications = total_applications_.load(std::memory_order_relaxed);
    stats.total_removals = total_removals_.load(std::memory_order_relaxed);

    // Calculate current active paths (approximation)
    stats.current_active_paths = (stats.total_applications > stats.total_removals) ?
        stats.total_applications - stats.total_removals : 0;

    // Scan tree for virtual loss statistics
    std::size_t node_count = tree_.get_node_count();
    float total_virtual_loss = 0.0f;
    std::size_t nodes_with_virtual_loss = 0;

    for (std::size_t i = 0; i < node_count; ++i) {
        NodeIndex node_index = static_cast<NodeIndex>(i);
        if (tree_.is_valid_index(node_index)) {
            float vl = tree_.get_virtual_loss(node_index);
            if (vl > 0.0f) {
                total_virtual_loss += vl;
                nodes_with_virtual_loss++;
                stats.max_virtual_loss = std::max(stats.max_virtual_loss, vl);
            }
        }
    }

    stats.avg_virtual_loss = (nodes_with_virtual_loss > 0) ?
        total_virtual_loss / nodes_with_virtual_loss : 0.0f;

    return stats;
}

bool VirtualLossManager::validate_node_index(NodeIndex node_index) const {
    return tree_.is_valid_index(node_index);
}

bool VirtualLossManager::atomic_add_virtual_loss(NodeIndex node_index, float delta) {
    // Get pointer to the virtual loss value for atomic operations
    float* virtual_losses_ptr = tree_.get_virtual_losses_ptr();

    // Use atomic operations to safely update virtual loss
    // We'll use compare-and-swap loop to ensure thread safety
    std::atomic<float>* atomic_vl = reinterpret_cast<std::atomic<float>*>(&virtual_losses_ptr[node_index]);

    float expected, desired;
    do {
        expected = atomic_vl->load(std::memory_order_acquire);
        desired = expected + delta;

        // Ensure virtual loss doesn't go negative
        if (desired < 0.0f) {
            desired = 0.0f;
        }

        // Prevent excessive virtual loss accumulation (safety check)
        if (desired > 1000.0f) {
            return false;  // Something is wrong - too much virtual loss
        }

    } while (!atomic_vl->compare_exchange_weak(expected, desired,
                                              std::memory_order_release,
                                              std::memory_order_acquire));

    return true;
}

// VirtualLossGuard implementation

VirtualLossGuard::VirtualLossGuard(VirtualLossManager& manager, const std::vector<NodeIndex>& path)
    : manager_(manager), path_(path), valid_(false), released_(false) {

    valid_ = manager_.apply_virtual_loss_to_path(path_);
}

VirtualLossGuard::~VirtualLossGuard() {
    if (valid_ && !released_) {
        release();
    }
}

void VirtualLossGuard::release() {
    if (valid_ && !released_) {
        manager_.remove_virtual_loss_from_path(path_);
        released_ = true;
    }
}

// ============================================================================
// WU-UCT Virtual Loss Manager Implementation
// ============================================================================

WUUCTVirtualLossManager::WUUCTVirtualLossManager(
    std::size_t max_nodes,
    float virtual_loss_magnitude
) : max_nodes_(max_nodes), magnitude_(virtual_loss_magnitude) {
    // Allocate cache-aligned array of atomic counters
    // Note: We can't use std::vector because std::atomic is not copyable
    in_flight_ = new (std::align_val_t{64}) std::atomic<std::uint32_t>[max_nodes];

    // Initialize all counters to zero
    for (std::size_t i = 0; i < max_nodes_; ++i) {
        in_flight_[i].store(0, std::memory_order_relaxed);
    }
}

void WUUCTVirtualLossManager::add_in_flight(NodeIndex node_index) {
    if (!is_valid_index(node_index)) {
        return;  // Invalid index, silently ignore
    }

    // Atomic increment - wait-free operation
    std::uint32_t old_count = in_flight_[node_index].fetch_add(
        1, std::memory_order_relaxed
    );

    // Track collisions (when multiple threads visit same node)
    if (old_count > 0) {
        collision_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // Instrumentation for performance tracking
    Instrumentation::instance().increment_counter(InstrumentationMetric::VirtualLossApply);
}

void WUUCTVirtualLossManager::remove_in_flight(NodeIndex node_index) {
    if (!is_valid_index(node_index)) {
        return;  // Invalid index, silently ignore
    }

    // Atomic decrement - wait-free operation
    std::uint32_t old_count = in_flight_[node_index].fetch_sub(
        1, std::memory_order_relaxed
    );

    // Safety check: ensure we don't underflow
    if (old_count == 0) {
        // This indicates a bug - removing virtual loss that wasn't applied
        // Re-increment to prevent underflow
        in_flight_[node_index].fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // Instrumentation for performance tracking
    Instrumentation::instance().increment_counter(InstrumentationMetric::VirtualLossRemove);
}

float WUUCTVirtualLossManager::get_exploration_adjustment(NodeIndex node_index) const {
    if (!is_valid_index(node_index)) {
        return 0.0f;
    }

    // Get in-flight count and scale by magnitude
    std::uint32_t count = in_flight_[node_index].load(std::memory_order_relaxed);
    return static_cast<float>(count) * magnitude_;
}

bool WUUCTVirtualLossManager::is_busy(NodeIndex node_index) const {
    if (!is_valid_index(node_index)) {
        return false;
    }

    return in_flight_[node_index].load(std::memory_order_relaxed) > 0;
}

std::uint32_t WUUCTVirtualLossManager::get_in_flight_count(NodeIndex node_index) const {
    if (!is_valid_index(node_index)) {
        return 0;
    }

    return in_flight_[node_index].load(std::memory_order_relaxed);
}

WUUCTVirtualLossManager::~WUUCTVirtualLossManager() {
    // Free aligned memory
    operator delete[](in_flight_, std::align_val_t{64});
}

void WUUCTVirtualLossManager::clear_all() {
    // Reset all in-flight counts to zero
    for (std::size_t i = 0; i < max_nodes_; ++i) {
        in_flight_[i].store(0, std::memory_order_relaxed);
    }

    // Reset collision counter
    collision_count_.store(0, std::memory_order_relaxed);
}

// ============================================================================
// WU-UCT Virtual Loss Guard Implementation
// ============================================================================

WUUCTVirtualLossGuard::WUUCTVirtualLossGuard(
    WUUCTVirtualLossManager& manager,
    const std::vector<NodeIndex>& path
) : manager_(manager), path_(path), released_(false) {
    // Apply virtual loss to entire path
    for (NodeIndex node : path_) {
        manager_.add_in_flight(node);
    }
}

WUUCTVirtualLossGuard::~WUUCTVirtualLossGuard() {
    if (!released_) {
        release();
    }
}

void WUUCTVirtualLossGuard::release() {
    if (released_) {
        return;  // Already released
    }

    // Remove virtual loss from entire path
    for (NodeIndex node : path_) {
        manager_.remove_in_flight(node);
    }

    released_ = true;
}

} // namespace mcts
