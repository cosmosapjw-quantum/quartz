// tree_adapter.cpp - Adapter implementation for TinyNodeTree → MCTSTree API
// Part of T024f-5: Adapter Layer

#include "tree_adapter.hpp"
#include <cassert>
#include <algorithm>

namespace mcts {

TreeAdapter::TreeAdapter(std::size_t max_nodes)
    : tree_(std::make_unique<TinyNodeTree>(max_nodes)) {
}

// ====== Tree Management ======

NodeIndex TreeAdapter::add_root_node(float prior_prob, std::uint8_t current_player, uint64_t zobrist_hash) {
    // Initialize root node in TinyNodeTree
    NodeIndex root_idx = tree_->init_root(zobrist_hash);

    if (root_idx < 0) {
        return NULL_NODE_INDEX;
    }

    // Set prior probability
    TinyNode* root = tree_->get_node(root_idx);
    root->prior_scaled = static_cast<uint16_t>(prior_prob * TinyNode::PRIOR_SCALE);

    // Set current_player in flags (bit 2)
    if (current_player != 0) {
        root->flags |= 0x04;  // Set bit 2
    }

    // Mark as root
    root->set_root();

    return root_idx;
}

NodeIndex TreeAdapter::allocate_nodes(std::uint16_t count) {
    // TinyNodeTree doesn't have contiguous allocation, so allocate individually
    // Return first allocated index
    if (count == 0) {
        return NULL_NODE_INDEX;
    }

    NodeIndex first_idx = tree_->allocate_node();
    if (first_idx < 0) {
        return NULL_NODE_INDEX;
    }

    // Allocate remaining nodes (best effort)
    for (std::uint16_t i = 1; i < count; ++i) {
        NodeIndex idx = tree_->allocate_node();
        if (idx < 0) {
            // Failed to allocate all requested nodes
            // Deallocate what we've allocated and return failure
            for (std::uint16_t j = 0; j < i; ++j) {
                tree_->deallocate_node(first_idx + static_cast<int32_t>(j));
            }
            return NULL_NODE_INDEX;
        }
    }

    return first_idx;
}

void TreeAdapter::deallocate_nodes(NodeIndex first_index, std::uint16_t count) {
    // Deallocate nodes individually
    for (std::uint16_t i = 0; i < count; ++i) {
        tree_->deallocate_node(first_index + static_cast<int32_t>(i));
    }
}

std::size_t TreeAdapter::get_available_nodes() const {
    std::size_t current = tree_->get_node_count();
    std::size_t max = tree_->get_max_nodes();
    return (current < max) ? (max - current) : 0;
}

// ====== Node Data Accessors ======

float TreeAdapter::get_visit_count(NodeIndex index) const {
    assert(is_valid_index(index));
    const TinyNode* node = tree_->get_node(index);
    return static_cast<float>(node->visit_count.load(std::memory_order_relaxed));
}

float TreeAdapter::get_total_value(NodeIndex index) const {
    assert(is_valid_index(index));
    const TinyNode* node = tree_->get_node(index);
    return node->get_value();
}

float TreeAdapter::get_prior_prob(NodeIndex index) const {
    assert(is_valid_index(index));
    const TinyNode* node = tree_->get_node(index);
    return node->get_prior();
}

float TreeAdapter::get_virtual_loss(NodeIndex index) const {
    assert(is_valid_index(index));
    const TinyNode* node = tree_->get_node(index);
    return static_cast<float>(node->virtual_loss.load(std::memory_order_relaxed));
}

NodeIndex TreeAdapter::get_parent_index(NodeIndex index) const {
    assert(is_valid_index(index));
    const TinyNode* node = tree_->get_node(index);
    return static_cast<NodeIndex>(node->parent_idx);
}

NodeIndex TreeAdapter::get_first_child_index(NodeIndex index) const {
    assert(is_valid_index(index));
    const TinyNode* node = tree_->get_node(index);
    uint32_t child_idx = node->first_child_idx;
    return (child_idx == 0) ? NULL_NODE_INDEX : static_cast<NodeIndex>(child_idx);
}

std::uint16_t TreeAdapter::get_num_children(NodeIndex index) const {
    assert(is_valid_index(index));
    return static_cast<std::uint16_t>(tree_->get_child_count(index));
}

NodeFlags TreeAdapter::get_flags(NodeIndex index) const {
    assert(is_valid_index(index));
    const TinyNode* node = tree_->get_node(index);

    // Convert TinyNode flags to NodeFlags
    NodeFlags flags;
    flags.flags = 0;

    if (node->is_expanded()) {
        flags.flags |= 0x01;  // expanded bit
    }

    if (node->is_terminal()) {
        flags.flags |= 0x02;  // terminal bit
    }

    // Extract current_player from bit 2 of TinyNode flags
    if (node->flags & 0x04) {
        flags.flags |= 0x04;  // current_player bit
    }

    // Check for expanding flag (bit 3 in TinyNode → bit 3 in NodeFlags)
    if (node->flags & 0x08) {
        flags.flags |= 0x08;  // expanding bit
    }

    return flags;
}

NodeInfo TreeAdapter::get_node_info(NodeIndex index) const {
    assert(is_valid_index(index));

    NodeInfo info;
    info.index = index;
    info.visit_count = get_visit_count(index);
    info.total_value = get_total_value(index);
    info.prior_prob = get_prior_prob(index);
    info.virtual_loss = get_virtual_loss(index);
    info.parent_index = get_parent_index(index);
    info.first_child_index = get_first_child_index(index);
    info.num_children = get_num_children(index);
    info.flags = get_flags(index);

    return info;
}

// ====== Node Data Mutators ======

void TreeAdapter::set_visit_count(NodeIndex index, float value) {
    assert(is_valid_index(index));
    assert(value >= 0.0f);

    TinyNode* node = tree_->get_node(index);
    node->visit_count.store(static_cast<uint32_t>(value), std::memory_order_relaxed);
}

void TreeAdapter::set_total_value(NodeIndex index, float value) {
    assert(is_valid_index(index));

    TinyNode* node = tree_->get_node(index);
    int32_t scaled_value = static_cast<int32_t>(value * TinyNode::VALUE_SCALE);
    node->total_value_scaled.store(scaled_value, std::memory_order_relaxed);
}

void TreeAdapter::set_prior_prob(NodeIndex index, float value) {
    assert(is_valid_index(index));
    assert(value >= 0.0f && value <= 1.0f);

    TinyNode* node = tree_->get_node(index);
    node->prior_scaled = static_cast<uint16_t>(value * TinyNode::PRIOR_SCALE);
}

void TreeAdapter::set_virtual_loss(NodeIndex index, float value) {
    assert(is_valid_index(index));
    assert(value >= 0.0f);

    TinyNode* node = tree_->get_node(index);
    node->virtual_loss.store(static_cast<uint8_t>(std::min(value, 255.0f)), std::memory_order_relaxed);
}

void TreeAdapter::set_parent_index(NodeIndex index, NodeIndex parent) {
    assert(is_valid_index(index));
    assert(parent == NULL_NODE_INDEX || is_valid_index(parent));

    TinyNode* node = tree_->get_node(index);
    node->parent_idx = (parent == NULL_NODE_INDEX) ? 0 : static_cast<uint32_t>(parent);
}

void TreeAdapter::set_first_child_index(NodeIndex index, NodeIndex first_child) {
    assert(is_valid_index(index));
    assert(first_child == NULL_NODE_INDEX || is_valid_index(first_child));

    TinyNode* node = tree_->get_node(index);
    node->first_child_idx = (first_child == NULL_NODE_INDEX) ? 0 : static_cast<uint32_t>(first_child);
}

void TreeAdapter::set_num_children(NodeIndex index, std::uint16_t count) {
    // NO-OP: TinyNodeTree derives num_children by counting siblings
    // This is intentionally a no-op to maintain API compatibility
    (void)index;
    (void)count;
}

void TreeAdapter::set_flags(NodeIndex index, const NodeFlags& flags) {
    assert(is_valid_index(index));

    TinyNode* node = tree_->get_node(index);

    // Convert NodeFlags to TinyNode flags
    node->flags = 0;

    if (flags.is_expanded()) {
        node->flags |= TinyNode::FLAG_EXPANDED;
    }

    if (flags.is_terminal()) {
        node->flags |= TinyNode::FLAG_TERMINAL;
    }

    // Set current_player bit (bit 2)
    if (flags.current_player() != 0) {
        node->flags |= 0x04;
    }

    // Set expanding bit (bit 3)
    if (flags.is_expanding()) {
        node->flags |= 0x08;
    }
}

// ====== Atomic Operations ======

bool TreeAdapter::atomic_try_set_expanded(NodeIndex index) {
    assert(is_valid_index(index));

    TinyNode* node = tree_->get_node(index);

    // Atomically check and set expanded bit
    std::atomic<std::uint8_t>* atomic_flags =
        reinterpret_cast<std::atomic<std::uint8_t>*>(&node->flags);

    std::uint8_t expected, desired;
    do {
        expected = atomic_flags->load(std::memory_order_acquire);

        // If already expanded, another thread won the race
        if (expected & TinyNode::FLAG_EXPANDED) {
            return false;
        }

        // Set expanded bit while preserving other flags
        desired = expected | TinyNode::FLAG_EXPANDED;

    } while (!atomic_flags->compare_exchange_weak(expected, desired,
                                                   std::memory_order_release,
                                                   std::memory_order_acquire));

    return true;
}

bool TreeAdapter::atomic_try_mark_expanding(NodeIndex index) {
    assert(is_valid_index(index));

    TinyNode* node = tree_->get_node(index);

    std::atomic<std::uint8_t>* atomic_flags =
        reinterpret_cast<std::atomic<std::uint8_t>*>(&node->flags);

    std::uint8_t expected, desired;
    do {
        expected = atomic_flags->load(std::memory_order_acquire);

        // If node already expanded or in-flight, do not mark again
        if ((expected & TinyNode::FLAG_EXPANDED) || (expected & 0x08)) {
            return false;
        }

        desired = expected | 0x08;  // mark expanding bit (bit 3)

    } while (!atomic_flags->compare_exchange_weak(expected, desired,
                                                   std::memory_order_release,
                                                   std::memory_order_acquire));

    return true;
}

void TreeAdapter::clear_expanding_flag(NodeIndex index) {
    assert(is_valid_index(index));

    TinyNode* node = tree_->get_node(index);

    std::atomic<std::uint8_t>* atomic_flags =
        reinterpret_cast<std::atomic<std::uint8_t>*>(&node->flags);

    std::uint8_t expected, desired;
    do {
        expected = atomic_flags->load(std::memory_order_acquire);
        desired = expected & static_cast<std::uint8_t>(~0x08);  // clear bit 3
    } while (!atomic_flags->compare_exchange_weak(expected, desired,
                                                   std::memory_order_release,
                                                   std::memory_order_acquire));
}

// ====== TinyNodeTree-Specific Extensions ======

uint64_t TreeAdapter::get_zobrist_hash(NodeIndex index) const {
    assert(is_valid_index(index));
    const TinyNode* node = tree_->get_node(index);
    return node->zobrist_hash;
}

void TreeAdapter::set_zobrist_hash(NodeIndex index, uint64_t hash) {
    assert(is_valid_index(index));
    TinyNode* node = tree_->get_node(index);
    node->zobrist_hash = hash;
}

uint16_t TreeAdapter::get_move(NodeIndex index) const {
    assert(is_valid_index(index));
    const TinyNode* node = tree_->get_node(index);
    return node->move;
}

void TreeAdapter::set_move(NodeIndex index, uint16_t move) {
    assert(is_valid_index(index));
    TinyNode* node = tree_->get_node(index);
    node->move = move;
}

} // namespace mcts
