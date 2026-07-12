// tiny_node_tree.cpp - Implementation of TinyNodeTree
// Part of T024f-1: TinyNode Storage Layer

#include "tiny_node_tree.hpp"
#include <cstring>
#include <stdexcept>
#include <algorithm>  // For std::reverse
#include <limits>     // For std::numeric_limits
#include <cmath>      // For std::sqrt

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace mcts {

TinyNodeTree::TinyNodeTree(std::size_t max_nodes)
    : max_nodes_(max_nodes), nodes_(nullptr) {
    if (max_nodes_ == 0) {
        throw std::invalid_argument("TinyNodeTree: max_nodes must be > 0");
    }

    allocate_array();
}

TinyNodeTree::~TinyNodeTree() {
    deallocate_array();
}

void TinyNodeTree::allocate_array() {
    // Allocate 64-byte aligned memory for node array
    std::size_t size = max_nodes_ * sizeof(TinyNode);

#ifdef _WIN32
    nodes_ = static_cast<TinyNode*>(_aligned_malloc(size, 64));
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, size) != 0) {
        ptr = nullptr;
    }
    nodes_ = static_cast<TinyNode*>(ptr);
#endif

    if (!nodes_) {
        throw std::bad_alloc();
    }

    // Zero-initialize the array
    std::memset(nodes_, 0, size);
}

void TinyNodeTree::deallocate_array() {
    if (nodes_) {
#ifdef _WIN32
        _aligned_free(nodes_);
#else
        free(nodes_);
#endif
        nodes_ = nullptr;
    }
}

int32_t TinyNodeTree::allocate_node() {
    // Fast path: Try free list first (with lock)
    {
        std::lock_guard<std::mutex> lock(free_list_mutex_);
        if (!free_list_.empty()) {
            int32_t index = free_list_.back();
            free_list_.pop_back();
            init_node(index);
            return index;
        }
    }

    // Slow path: Bump allocation (lock-free)
    std::size_t index = next_index_.fetch_add(1, std::memory_order_relaxed);

    if (index >= max_nodes_) {
        // Pool exhausted - rollback
        next_index_.fetch_sub(1, std::memory_order_relaxed);
        return -1;
    }

    init_node(static_cast<int32_t>(index));
    return static_cast<int32_t>(index);
}

void TinyNodeTree::deallocate_node(int32_t index) {
    if (!is_valid_index(index)) {
        return;  // Silently ignore invalid indices
    }

    // Add to free list for reuse
    std::lock_guard<std::mutex> lock(free_list_mutex_);
    free_list_.push_back(index);
}

void TinyNodeTree::clear() {
    // O(1) reset: just reset bump allocator and clear free list
    next_index_.store(0, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(free_list_mutex_);
    free_list_.clear();

    // Note: We do NOT zero the memory - nodes will be re-initialized on allocation
}

TinyNode* TinyNodeTree::get_node(int32_t index) {
    if (!is_valid_index(index)) {
        return nullptr;
    }
    return &nodes_[index];
}

const TinyNode* TinyNodeTree::get_node(int32_t index) const {
    if (!is_valid_index(index)) {
        return nullptr;
    }
    return &nodes_[index];
}

int32_t TinyNodeTree::init_root(uint64_t zobrist_hash) {
    // Clear tree first
    clear();

    // Allocate root node
    int32_t root_idx = allocate_node();
    if (root_idx != 0) {
        throw std::runtime_error("TinyNodeTree::init_root: root index != 0");
    }

    TinyNode* root = get_node(root_idx);
    root->move = 0;  // Root has no move
    root->parent_idx = 0;  // Root points to self
    root->zobrist_hash = zobrist_hash;
    root->flags |= TinyNode::FLAG_ROOT;

    // Root starts with 1 visit (per WU-UCT)
    root->visit_count.store(1, std::memory_order_relaxed);

    return root_idx;
}

void TinyNodeTree::init_node(int32_t index) {
    TinyNode* node = &nodes_[index];

    // Zero-initialize critical fields
    node->move = 0;
    node->parent_idx = 0;
    node->first_child_idx = 0;
    node->next_sibling_idx = 0;
    node->visit_count.store(0, std::memory_order_relaxed);
    node->total_value_scaled.store(0, std::memory_order_relaxed);
    node->prior_scaled = 0;
    node->virtual_loss.store(0, std::memory_order_relaxed);
    node->flags = 0;
    node->zobrist_hash = 0;
}

bool TinyNodeTree::validate() const {
    std::size_t count = get_node_count();

    // Check all allocated nodes
    for (std::size_t i = 0; i < count; ++i) {
        const TinyNode* node = get_node(static_cast<int32_t>(i));
        if (!node) {
            return false;  // Invalid node pointer
        }

        // Validate parent index
        if (i != 0) {  // Non-root
            if (node->parent_idx == 0 && !(node->flags & TinyNode::FLAG_ROOT)) {
                // Only root should have parent_idx = 0 (self-reference)
                // Allow for now - parent will be set during expansion
            }
            if (node->parent_idx >= static_cast<uint32_t>(count)) {
                return false;  // Parent index out of bounds
            }
        } else {  // Root
            if (!(node->flags & TinyNode::FLAG_ROOT)) {
                return false;  // Root must have FLAG_ROOT set
            }
            if (node->parent_idx != 0) {
                return false;  // Root must point to self
            }
        }

        // Validate child indices
        if (node->first_child_idx != 0) {
            if (node->first_child_idx >= static_cast<uint32_t>(count)) {
                return false;  // First child out of bounds
            }
        }

        if (node->next_sibling_idx != 0) {
            if (node->next_sibling_idx >= static_cast<uint32_t>(count)) {
                return false;  // Next sibling out of bounds
            }
        }
    }

    return true;
}

// ====== Child Management (T024f-2) ======

int32_t TinyNodeTree::add_child(int32_t parent_idx, uint16_t move, float prior_prob, uint64_t zobrist_hash) {
    if (!is_valid_index(parent_idx)) {
        return -1;  // Invalid parent
    }

    // Allocate new child node (calls init_node which zeros all fields)
    int32_t child_idx = allocate_node();
    if (child_idx < 0) {
        return -1;  // Allocation failed
    }

    // Get nodes (child is already zero-initialized by allocate_node -> init_node)
    TinyNode* child = &nodes_[child_idx];
    TinyNode* parent = &nodes_[parent_idx];

    // Set child-specific fields (other fields already zero from init_node)
    child->move = move;
    child->parent_idx = static_cast<uint32_t>(parent_idx);
    child->zobrist_hash = zobrist_hash;
    child->prior_scaled = static_cast<uint16_t>(prior_prob * TinyNode::PRIOR_SCALE);

    // Link child into parent's sibling list (add to front for O(1))
    // Note: child->next_sibling_idx is already 0 from init_node
    child->next_sibling_idx = parent->first_child_idx;
    parent->first_child_idx = static_cast<uint32_t>(child_idx);

    // Mark parent as expanded (has children)
    parent->set_expanded();

    return child_idx;
}

bool TinyNodeTree::expand_node(int32_t parent_idx, const uint16_t* moves, const float* priors,
                                const uint64_t* zobrist_hashes, size_t num_children) {
    if (!is_valid_index(parent_idx)) {
        return false;
    }

    if (num_children == 0) {
        return true;  // Nothing to do
    }

    // Check if we have space
    if (!has_space_for(num_children)) {
        return false;
    }

    // Add all children
    for (size_t i = 0; i < num_children; ++i) {
        int32_t child_idx = add_child(parent_idx, moves[i], priors[i], zobrist_hashes[i]);
        if (child_idx < 0) {
            // Allocation failed partway through - this is a partial failure
            // Children already added are still in the tree
            return false;
        }
    }

    return true;
}

size_t TinyNodeTree::get_child_count(int32_t parent_idx) const {
    if (!is_valid_index(parent_idx)) {
        return 0;
    }

    const TinyNode* parent = get_node(parent_idx);
    size_t count = 0;

    int32_t child_idx = static_cast<int32_t>(parent->first_child_idx);
    while (child_idx != 0) {
        ++count;
        const TinyNode* child = get_node(child_idx);
        child_idx = static_cast<int32_t>(child->next_sibling_idx);
    }

    return count;
}

std::vector<int32_t> TinyNodeTree::get_children(int32_t parent_idx) const {
    std::vector<int32_t> children;

    for_each_child(parent_idx, [&children](int32_t child_idx) {
        children.push_back(child_idx);
    });

    return children;
}

// ====== Path Traversal (T024f-3) ======

std::vector<int32_t> TinyNodeTree::get_path_to_node(int32_t node_idx) const {
    if (!is_valid_index(node_idx)) {
        return {};  // Empty path for invalid node
    }

    // Collect path by walking up to root
    std::vector<int32_t> path;
    int32_t current = node_idx;

    while (current >= 0 && is_valid_index(current)) {
        path.push_back(current);

        const TinyNode* node = get_node(current);

        // Stop at root (parent_idx points to self)
        if (node->is_root() || node->parent_idx == static_cast<uint32_t>(current)) {
            break;
        }

        current = static_cast<int32_t>(node->parent_idx);
    }

    // Reverse to get root → node order
    std::reverse(path.begin(), path.end());
    return path;
}

int32_t TinyNodeTree::select_best_child(int32_t parent_idx, float c_puct) const {
    if (!is_valid_index(parent_idx)) {
        return -1;
    }

    const TinyNode* parent = get_node(parent_idx);

    // Check if parent has children
    if (parent->first_child_idx == 0) {
        return -1;  // No children
    }

    // Get parent visit count for PUCT formula
    uint32_t parent_n = parent->visit_count.load(std::memory_order_relaxed);
    float sqrt_parent_n = std::sqrt(static_cast<float>(parent_n));

    // Find best child using PUCT formula
    int32_t best_child = -1;
    float best_score = -std::numeric_limits<float>::infinity();

    for_each_child(parent_idx, [&](int32_t child_idx) {
        const TinyNode* child = get_node(child_idx);

        // Get child statistics (atomic loads)
        uint32_t child_n = child->visit_count.load(std::memory_order_relaxed);
        int32_t child_w_scaled = child->total_value_scaled.load(std::memory_order_relaxed);
        uint8_t child_vl = child->virtual_loss.load(std::memory_order_relaxed);

        // Q-value: mean value (handle division by zero)
        float q_value = 0.0f;
        if (child_n > 0) {
            float child_w = static_cast<float>(child_w_scaled) / TinyNode::VALUE_SCALE;
            q_value = child_w / static_cast<float>(child_n);
        }

        // Prior probability (scaled)
        float prior = child->get_prior();

        // PUCT formula: Q + c_puct * P * sqrt(N_parent) / (1 + N_child + VL_child)
        float exploration = c_puct * prior * sqrt_parent_n / (1.0f + static_cast<float>(child_n) + static_cast<float>(child_vl));
        float puct_score = q_value + exploration;

        if (puct_score > best_score) {
            best_score = puct_score;
            best_child = child_idx;
        }
    });

    return best_child;
}

void TinyNodeTree::apply_virtual_loss(int32_t node_idx, uint8_t magnitude) {
    if (!is_valid_index(node_idx)) {
        return;
    }

    TinyNode* node = &nodes_[node_idx];

    // Atomic increment (thread-safe)
    uint8_t old_vl = node->virtual_loss.load(std::memory_order_relaxed);
    uint8_t new_vl = old_vl + magnitude;

    // Handle overflow (cap at 255)
    if (new_vl < old_vl) {
        new_vl = 255;
    }

    node->virtual_loss.store(new_vl, std::memory_order_relaxed);
}

void TinyNodeTree::remove_virtual_loss(int32_t node_idx, uint8_t magnitude) {
    if (!is_valid_index(node_idx)) {
        return;
    }

    TinyNode* node = &nodes_[node_idx];

    // Atomic decrement (thread-safe)
    uint8_t old_vl = node->virtual_loss.load(std::memory_order_relaxed);
    uint8_t new_vl = (old_vl >= magnitude) ? (old_vl - magnitude) : 0;

    node->virtual_loss.store(new_vl, std::memory_order_relaxed);
}

void TinyNodeTree::backup_value(const std::vector<int32_t>& path, float leaf_value) {
    if (path.empty()) {
        return;
    }

    // Backup from leaf to root, flipping value sign at each level
    // Negamax framework: child's value = -parent's value
    float value = leaf_value;

    // Iterate from leaf to root (reverse order)
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        int32_t node_idx = *it;

        if (!is_valid_index(node_idx)) {
            continue;
        }

        TinyNode* node = &nodes_[node_idx];

        // Atomic increment visit count
        node->visit_count.fetch_add(1, std::memory_order_relaxed);

        // Atomic add to total value (scaled to int32)
        int32_t value_scaled = static_cast<int32_t>(value * TinyNode::VALUE_SCALE);
        node->total_value_scaled.fetch_add(value_scaled, std::memory_order_relaxed);

        // Flip value sign for next level (negamax)
        value = -value;
    }
}

} // namespace mcts
