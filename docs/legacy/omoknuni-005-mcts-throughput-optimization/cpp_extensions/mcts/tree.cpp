/**
 * @file tree.cpp
 * @brief Implementation of high-performance MCTS tree with SoA layout
 */

#include "tree.hpp"
#include "instrumentation.hpp"
#include "profiling/enhanced_profiler.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <chrono>

using namespace mcts::profiling;

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace mcts {

namespace {

// Increased from 64 to 4096 to reduce contention on global next_free_index_
// With 12 threads, this means 49K nodes allocated without global synchronization
// Per review.pdf recommendation: "give each thread its own block of node indices"
constexpr std::uint32_t kThreadBlockSize = 4096;

// Global counter for generating unique tree instance IDs
std::atomic<std::uint64_t> next_tree_id{1};

struct ThreadLocalBlock {
    MCTSTree* tree = nullptr;
    std::uint64_t tree_id = 0;  // Unique instance ID to detect tree changes
    NodeIndex next = NULL_NODE_INDEX;
    std::uint32_t remaining = 0;
    std::uint64_t epoch = 0;

    // Statistics for tracking thread-local allocation efficiency
    std::uint64_t allocations_from_block = 0;    // Fast path: thread-local block
    std::uint64_t allocations_from_global = 0;   // Slow path: global pool
    std::uint64_t allocations_from_freelist = 0; // Reuse path: free list
};

thread_local ThreadLocalBlock thread_block;

}  // namespace

MCTSTree::MCTSTree(std::size_t max_nodes)
    : max_nodes_(max_nodes)
    , node_count_(0)
    , next_free_index_(0)
    , instance_id_(next_tree_id.fetch_add(1, std::memory_order_relaxed))
    , visit_counts_(nullptr)
    , total_values_(nullptr)
    , prior_probs_(nullptr)
    , virtual_losses_(nullptr)
    , parent_indices_(nullptr)
    , first_child_indices_(nullptr)
    , num_children_(nullptr)
    , flags_(nullptr)
    , moves_(nullptr) {

    if (max_nodes == 0) {
        throw std::invalid_argument("max_nodes must be > 0");
    }

    if (max_nodes > static_cast<std::size_t>(std::numeric_limits<NodeIndex>::max())) {
        throw std::invalid_argument("max_nodes exceeds NodeIndex capacity");
    }

    allocate_arrays();
    initialize_arrays();
}

MCTSTree::~MCTSTree() {
    deallocate_arrays();
}

void MCTSTree::allocate_arrays() {
    try {
        visit_counts_ = allocate_aligned<float>(max_nodes_);
        total_values_ = allocate_aligned<float>(max_nodes_);
        prior_probs_ = allocate_aligned<float>(max_nodes_);
        virtual_losses_ = allocate_aligned<float>(max_nodes_);
        parent_indices_ = allocate_aligned<NodeIndex>(max_nodes_);
        first_child_indices_ = allocate_aligned<NodeIndex>(max_nodes_);
        num_children_ = allocate_aligned<std::uint16_t>(max_nodes_);
        flags_ = allocate_aligned<NodeFlags>(max_nodes_);
        moves_ = allocate_aligned<std::uint16_t>(max_nodes_);
    } catch (...) {
        // Clean up any partially allocated arrays
        deallocate_arrays();
        throw;
    }
}

void MCTSTree::deallocate_arrays() {
    deallocate_aligned(visit_counts_);
    deallocate_aligned(total_values_);
    deallocate_aligned(prior_probs_);
    deallocate_aligned(virtual_losses_);
    deallocate_aligned(parent_indices_);
    deallocate_aligned(first_child_indices_);
    deallocate_aligned(num_children_);
    deallocate_aligned(flags_);
    deallocate_aligned(moves_);

    visit_counts_ = nullptr;
    total_values_ = nullptr;
    prior_probs_ = nullptr;
    virtual_losses_ = nullptr;
    parent_indices_ = nullptr;
    first_child_indices_ = nullptr;
    num_children_ = nullptr;
    flags_ = nullptr;
    moves_ = nullptr;
}

void MCTSTree::initialize_arrays() {
    // Initialize all arrays to zero/default values
    std::memset(visit_counts_, 0, max_nodes_ * sizeof(float));
    std::memset(total_values_, 0, max_nodes_ * sizeof(float));
    std::memset(prior_probs_, 0, max_nodes_ * sizeof(float));
    std::memset(virtual_losses_, 0, max_nodes_ * sizeof(float));

    // Initialize indices to NULL_NODE_INDEX
    std::fill_n(parent_indices_, max_nodes_, NULL_NODE_INDEX);
    std::fill_n(first_child_indices_, max_nodes_, NULL_NODE_INDEX);

    // Initialize counts, flags, and moves to zero
    std::memset(num_children_, 0, max_nodes_ * sizeof(std::uint16_t));
    std::fill_n(flags_, max_nodes_, NodeFlags());
    std::memset(moves_, 0, max_nodes_ * sizeof(std::uint16_t));
}

std::size_t MCTSTree::get_memory_usage() const {
    std::size_t total = 0;

    // Calculate aligned sizes for each array
    auto aligned_size = [](std::size_t size) {
        return ((size + 63) / 64) * 64;
    };

    total += aligned_size(max_nodes_ * sizeof(float));         // visit_counts_
    total += aligned_size(max_nodes_ * sizeof(float));         // total_values_
    total += aligned_size(max_nodes_ * sizeof(float));         // prior_probs_
    total += aligned_size(max_nodes_ * sizeof(float));         // virtual_losses_
    total += aligned_size(max_nodes_ * sizeof(NodeIndex));     // parent_indices_
    total += aligned_size(max_nodes_ * sizeof(NodeIndex));     // first_child_indices_
    total += aligned_size(max_nodes_ * sizeof(std::uint16_t)); // num_children_
    total += aligned_size(max_nodes_ * sizeof(NodeFlags));     // flags_
    total += aligned_size(max_nodes_ * sizeof(std::uint16_t)); // moves_

    return total;
}

void MCTSTree::clear() {
    ScopedMetric metric(InstrumentationMetric::TreeClear);
    node_count_.store(0, std::memory_order_relaxed);
    next_free_index_.store(0, std::memory_order_relaxed);
    free_nodes_.clear();
    allocation_epoch_.fetch_add(1, std::memory_order_acq_rel);
}

NodeIndex MCTSTree::add_root_node(float prior_prob, std::uint8_t current_player) {
    if (node_count_.load(std::memory_order_relaxed) > 0) {
        throw std::logic_error("Root node already exists");
    }

    if (prior_prob < 0.0f || prior_prob > 1.0f) {
        throw std::invalid_argument("prior_prob must be in [0, 1]");
    }

    if (current_player > 1) {
        throw std::invalid_argument("current_player must be 0 or 1");
    }

    // Allocate root node using the pool system
    NodeIndex root_index = allocate_node();
    if (root_index == NULL_NODE_INDEX) {
        throw std::runtime_error("Failed to allocate root node");
    }

    // Initialize root node
    visit_counts_[root_index] = 0.0f;
    total_values_[root_index] = 0.0f;
    prior_probs_[root_index] = prior_prob;
    virtual_losses_[root_index] = 0.0f;
    parent_indices_[root_index] = NULL_NODE_INDEX;  // Root has no parent
    first_child_indices_[root_index] = NULL_NODE_INDEX;  // Not expanded yet
    num_children_[root_index] = 0;

    NodeFlags root_flags;
    root_flags.set_current_player(current_player);
    flags_[root_index] = root_flags;

    return root_index;
}

NodeInfo MCTSTree::get_node_info(NodeIndex index) const {
    if (!is_valid_index(index)) {
        throw std::invalid_argument("Invalid node index");
    }

    NodeInfo info;
    info.index = index;
    info.visit_count = visit_counts_[index];
    info.total_value = total_values_[index];
    info.prior_prob = prior_probs_[index];
    info.virtual_loss = virtual_losses_[index];
    info.parent_index = parent_indices_[index];
    info.first_child_index = first_child_indices_[index];
    info.num_children = num_children_[index];
    info.flags = flags_[index];

    return info;
}

bool MCTSTree::validate_tree() const {
    if (node_count_.load(std::memory_order_relaxed) == 0) {
        return true;  // Empty tree is valid
    }

    if (next_free_index_.load(std::memory_order_relaxed) > max_nodes_) {
        return false;  // Index out of bounds
    }

    // Validate root node
    if (parent_indices_[0] != NULL_NODE_INDEX) {
        return false;  // Root must have no parent
    }

    // Validate all allocated nodes
    for (std::size_t i = 0; i < next_free_index_.load(std::memory_order_relaxed); ++i) {
        NodeIndex index = static_cast<NodeIndex>(i);

        // Check visit count is non-negative
        if (visit_counts_[index] < 0.0f) {
            return false;
        }

        // Check prior probability is in valid range
        if (prior_probs_[index] < 0.0f || prior_probs_[index] > 1.0f) {
            return false;
        }

        // Check virtual loss is non-negative
        if (virtual_losses_[index] < 0.0f) {
            return false;
        }

        // Check parent index validity
        NodeIndex parent = parent_indices_[index];
        if (parent != NULL_NODE_INDEX) {
            if (parent < 0 || static_cast<std::size_t>(parent) >= next_free_index_.load(std::memory_order_relaxed)) {
                return false;  // Invalid parent index
            }
            if (parent >= index) {
                return false;  // Parent must have lower index (DAG property)
            }
        }

        // Check first child index validity
        NodeIndex first_child = first_child_indices_[index];
        if (first_child != NULL_NODE_INDEX) {
            if (first_child < 0 || static_cast<std::size_t>(first_child) >= next_free_index_.load(std::memory_order_relaxed)) {
                return false;  // Invalid child index
            }
            if (first_child <= index) {
                return false;  // Child must have higher index
            }

            // Check that we have the claimed number of children
            std::uint16_t expected_children = num_children_[index];
            for (std::uint16_t j = 0; j < expected_children; ++j) {
                NodeIndex child_index = first_child + j;
                if (static_cast<std::size_t>(child_index) >= next_free_index_.load(std::memory_order_relaxed)) {
                    return false;  // Child index out of range
                }
                if (parent_indices_[child_index] != index) {
                    return false;  // Child doesn't point back to parent
                }
            }
        }

        // Check total value bounds
        float visit_count = visit_counts_[index];
        float total_value = total_values_[index];
        if (std::abs(total_value) > visit_count + 1e-6f) {
            return false;  // Total value exceeds possible bounds
        }
    }

    return true;
}

TreeMemoryStats get_tree_memory_stats(const MCTSTree& tree) {
    TreeMemoryStats stats;

    stats.node_count = tree.get_node_count();
    std::size_t max_nodes = tree.get_max_nodes();

    // Calculate aligned sizes for each array
    auto aligned_size = [](std::size_t size) {
        return ((size + 63) / 64) * 64;
    };

    stats.visit_counts_bytes = aligned_size(max_nodes * sizeof(float));
    stats.total_values_bytes = aligned_size(max_nodes * sizeof(float));
    stats.prior_probs_bytes = aligned_size(max_nodes * sizeof(float));
    stats.virtual_losses_bytes = aligned_size(max_nodes * sizeof(float));
    stats.parent_indices_bytes = aligned_size(max_nodes * sizeof(NodeIndex));
    stats.first_child_indices_bytes = aligned_size(max_nodes * sizeof(NodeIndex));
    stats.num_children_bytes = aligned_size(max_nodes * sizeof(std::uint16_t));
    stats.flags_bytes = aligned_size(max_nodes * sizeof(NodeFlags));

    stats.total_bytes = stats.visit_counts_bytes + stats.total_values_bytes +
                       stats.prior_probs_bytes + stats.virtual_losses_bytes +
                       stats.parent_indices_bytes + stats.first_child_indices_bytes +
                       stats.num_children_bytes + stats.flags_bytes;

    // Calculate raw size without alignment
    std::size_t raw_size = max_nodes * (4 * sizeof(float) + 2 * sizeof(NodeIndex) +
                                       sizeof(std::uint16_t) + sizeof(NodeFlags));

    stats.alignment_overhead = stats.total_bytes - raw_size;
    stats.bytes_per_node = stats.node_count > 0 ?
        static_cast<double>(stats.total_bytes) / stats.node_count : 0.0;

    return stats;
}

NodeIndex MCTSTree::allocate_node() {
    ScopedMetric metric(InstrumentationMetric::TreeAllocateNode);
    const auto current_epoch = allocation_epoch_.load(std::memory_order_acquire);
    auto& block = thread_block;

    // Check if this is a different tree instance or epoch changed
    if (block.tree_id != instance_id_ || block.epoch != current_epoch) {
        block.tree = this;
        block.tree_id = instance_id_;
        block.next = NULL_NODE_INDEX;
        block.remaining = 0;
        block.epoch = current_epoch;
    }

    auto take_from_block = [&]() -> NodeIndex {
        if (block.remaining == 0) {
            return NULL_NODE_INDEX;
        }
        NodeIndex index = block.next;
        ++block.next;
        --block.remaining;
        ++block.allocations_from_block;  // Track fast-path allocations
        return index;
    };

    NodeIndex index = take_from_block();

    if (index == NULL_NODE_INDEX) {
        auto refill_block = [&]() -> NodeIndex {
            std::lock_guard<std::mutex> lock(allocation_mutex_);

            if (!free_nodes_.empty()) {
                NodeIndex reused = free_nodes_.back();
                free_nodes_.pop_back();
                initialize_node(reused);
                block.next = NULL_NODE_INDEX;
                block.remaining = 0;
                block.epoch = current_epoch;
                block.tree = this;
                block.tree_id = instance_id_;
                ++block.allocations_from_freelist;  // Track free list reuse
                return reused;
            }

            std::size_t start = next_free_index_.load(std::memory_order_relaxed);
            if (start >= max_nodes_) {
                block.next = NULL_NODE_INDEX;
                block.remaining = 0;
                block.epoch = current_epoch;
                block.tree = this;
                block.tree_id = instance_id_;
                return NULL_NODE_INDEX;
            }

            const std::size_t capacity = max_nodes_ - start;
            const std::size_t block_size = std::min<std::size_t>(kThreadBlockSize, capacity);

            NodeIndex first_index = static_cast<NodeIndex>(start);
            next_free_index_.fetch_add(block_size, std::memory_order_relaxed);
            initialize_node_range(first_index, static_cast<std::uint16_t>(block_size));

            block.tree = this;
            block.tree_id = instance_id_;
            block.epoch = current_epoch;
            if (block_size > 1) {
                block.next = static_cast<NodeIndex>(first_index + 1);
                block.remaining = static_cast<std::uint32_t>(block_size - 1);
            } else {
                block.next = NULL_NODE_INDEX;
                block.remaining = 0;
            }

            ++block.allocations_from_global;  // Track global pool allocation
            return first_index;
        };

        index = refill_block();
        if (index == NULL_NODE_INDEX) {
            return NULL_NODE_INDEX;
        }
    }

    node_count_.fetch_add(1, std::memory_order_relaxed);
    return index;
}

NodeIndex MCTSTree::allocate_nodes(std::uint16_t count) {
    if (count == 0) {
        return NULL_NODE_INDEX;
    }

    if (count == 1) {
        return allocate_node();
    }

    ScopedMetric metric(InstrumentationMetric::TreeAllocateNodes);
    PROFILE_SCOPE(ProfileMetric::MemoryNodeAllocation);

    // Track mutex wait time (review.txt lines 225-236: allocation contention)
    auto lock_start = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(allocation_mutex_);

    auto lock_acquired = std::chrono::steady_clock::now();
    auto wait_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        lock_acquired - lock_start).count();

    if (wait_time_ns > 0) {
        PROFILE_GAUGE(ProfileMetric::AllocationMutexWait, wait_time_ns);
        PROFILE_GAUGE(ProfileMetric::MutexLockWaitTime, wait_time_ns);
        PROFILE_COUNTER(ProfileMetric::MutexContentionEvents, 1);
    }

    // Slow path allocation (global mutex)
    PROFILE_COUNTER(ProfileMetric::AllocationSlowPath, count);

    // For multiple nodes, we need contiguous allocation
    // Check if we have enough contiguous space from the pool
    if (next_free_index_.load(std::memory_order_relaxed) + count > max_nodes_) {
        return NULL_NODE_INDEX;  // Not enough contiguous space
    }

    NodeIndex first_index = static_cast<NodeIndex>(next_free_index_.load(std::memory_order_relaxed));
    next_free_index_.fetch_add(count, std::memory_order_relaxed);
    node_count_.fetch_add(count, std::memory_order_relaxed);

    initialize_node_range(first_index, count);

    return first_index;
}

void MCTSTree::deallocate_node(NodeIndex index) {
    if (!is_valid_index(index)) {
        return;  // Invalid index, ignore
    }

    std::lock_guard<std::mutex> lock(allocation_mutex_);

    // Add to free list for reuse
    free_nodes_.push_back(index);
    node_count_.fetch_sub(1, std::memory_order_relaxed);  // Decrement active node count
}

void MCTSTree::deallocate_nodes(NodeIndex first_index, std::uint16_t count) {
    if (count == 0) {
        return;
    }

    if (count == 1) {
        deallocate_node(first_index);
        return;
    }

    std::lock_guard<std::mutex> lock(allocation_mutex_);

    // Deallocate multiple contiguous nodes
    std::uint16_t deallocated = 0;
    for (std::uint16_t i = 0; i < count; ++i) {
        NodeIndex index = first_index + i;
        if (is_valid_index(index)) {
            free_nodes_.push_back(index);
            ++deallocated;
        }
    }

    node_count_.fetch_sub(deallocated, std::memory_order_relaxed);  // Decrement active node count
}

void MCTSTree::initialize_node(NodeIndex index) {
    if (index == NULL_NODE_INDEX) {
        return;
    }

    visit_counts_[index] = 0.0f;
    total_values_[index] = 0.0f;
    prior_probs_[index] = 0.0f;
    virtual_losses_[index] = 0.0f;
    parent_indices_[index] = NULL_NODE_INDEX;
    first_child_indices_[index] = NULL_NODE_INDEX;
    num_children_[index] = 0;
    flags_[index] = NodeFlags();
}

void MCTSTree::initialize_node_range(NodeIndex first_index, std::uint16_t count) {
    if (count == 0 || first_index == NULL_NODE_INDEX) {
        return;
    }

    for (std::uint16_t i = 0; i < count; ++i) {
        NodeIndex index = first_index + i;
        if (static_cast<std::size_t>(index) >= max_nodes_) {
            break;
        }
        initialize_node(index);
    }
}

MCTSTree::ThreadAllocationStats MCTSTree::get_thread_allocation_stats() const {
    // Access thread-local block for current thread
    // thread_block is in the anonymous namespace, so we access it directly
    ThreadAllocationStats stats;
    stats.allocations_from_block = thread_block.allocations_from_block;
    stats.allocations_from_global = thread_block.allocations_from_global;
    stats.allocations_from_freelist = thread_block.allocations_from_freelist;
    stats.block_size = kThreadBlockSize;

    return stats;
}

} // namespace mcts
