/**
 * @file test_node_pool.cpp
 * @brief Unit tests for node pool pre-allocation functionality
 */

#include <gtest/gtest.h>
#include "mcts/tree.hpp"
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>

using namespace mcts;

namespace {
constexpr std::size_t kThreadBlockReserve = 64;
}

class NodePoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a small tree for testing
        tree = std::make_unique<MCTSTree>(1000);
    }

    std::unique_ptr<MCTSTree> tree;
};

TEST_F(NodePoolTest, AllocateSingleNode) {
    // Initially no nodes allocated
    EXPECT_EQ(tree->get_node_count(), 0);
    EXPECT_EQ(tree->get_available_nodes(), tree->get_max_nodes());

    // Allocate a single node
    NodeIndex node1 = tree->allocate_node();
    EXPECT_NE(node1, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 1);
    EXPECT_LE(tree->get_available_nodes(), tree->get_max_nodes());
    EXPECT_GE(tree->get_available_nodes(), tree->get_max_nodes() - kThreadBlockReserve);

    // Allocate another node
    NodeIndex node2 = tree->allocate_node();
    EXPECT_NE(node2, NULL_NODE_INDEX);
    EXPECT_NE(node1, node2);
    EXPECT_EQ(tree->get_node_count(), 2);
    EXPECT_LE(tree->get_available_nodes(), tree->get_max_nodes());
    EXPECT_GE(tree->get_available_nodes(), tree->get_max_nodes() - kThreadBlockReserve);
}

TEST_F(NodePoolTest, AllocateMultipleContiguousNodes) {
    // Allocate 5 contiguous nodes
    NodeIndex first_node = tree->allocate_nodes(5);
    EXPECT_NE(first_node, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 5);

    // Verify the nodes are contiguous
    for (int i = 1; i < 5; ++i) {
        EXPECT_EQ(first_node + i, first_node + i);  // Verify arithmetic works
    }

    // Allocate more contiguous nodes
    NodeIndex second_batch = tree->allocate_nodes(3);
    EXPECT_NE(second_batch, NULL_NODE_INDEX);
    EXPECT_EQ(second_batch, first_node + 5);  // Should be immediately after first batch
    EXPECT_EQ(tree->get_node_count(), 8);
}

TEST_F(NodePoolTest, DeallocateAndReuse) {
    // Allocate some nodes
    NodeIndex node1 = tree->allocate_node();
    NodeIndex node2 = tree->allocate_node();
    NodeIndex node3 = tree->allocate_node();

    EXPECT_EQ(tree->get_node_count(), 3);

    // Deallocate middle node
    tree->deallocate_node(node2);
    EXPECT_EQ(tree->get_node_count(), 2);

    // Allocate a new node - should reuse the deallocated one
    NodeIndex node4 = tree->allocate_node();
    EXPECT_NE(node4, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 3);
}

TEST_F(NodePoolTest, DeallocateMultipleNodes) {
    // Allocate a batch of nodes
    NodeIndex first_node = tree->allocate_nodes(10);
    EXPECT_NE(first_node, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 10);

    // Deallocate half of them
    tree->deallocate_nodes(first_node + 5, 5);
    EXPECT_EQ(tree->get_node_count(), 5);

    // Deallocate the rest
    tree->deallocate_nodes(first_node, 5);
    EXPECT_EQ(tree->get_node_count(), 0);
}

TEST_F(NodePoolTest, ExhaustPool) {
    // Create a small tree to easily exhaust
    auto small_tree = std::make_unique<MCTSTree>(3);

    // Allocate all available nodes
    NodeIndex node1 = small_tree->allocate_node();
    NodeIndex node2 = small_tree->allocate_node();
    NodeIndex node3 = small_tree->allocate_node();

    EXPECT_NE(node1, NULL_NODE_INDEX);
    EXPECT_NE(node2, NULL_NODE_INDEX);
    EXPECT_NE(node3, NULL_NODE_INDEX);
    EXPECT_EQ(small_tree->get_node_count(), 3);
    EXPECT_EQ(small_tree->get_available_nodes(), 0);

    // Try to allocate one more - should fail
    NodeIndex node4 = small_tree->allocate_node();
    EXPECT_EQ(node4, NULL_NODE_INDEX);
    EXPECT_EQ(small_tree->get_node_count(), 3);  // Should remain unchanged
}

TEST_F(NodePoolTest, ExhaustPoolMultipleAllocation) {
    // Create a small tree
    auto small_tree = std::make_unique<MCTSTree>(5);

    // Try to allocate more nodes than available
    NodeIndex nodes = small_tree->allocate_nodes(10);
    EXPECT_EQ(nodes, NULL_NODE_INDEX);
    EXPECT_EQ(small_tree->get_node_count(), 0);  // Should remain unchanged

    // Allocate exactly the right amount
    nodes = small_tree->allocate_nodes(5);
    EXPECT_NE(nodes, NULL_NODE_INDEX);
    EXPECT_EQ(small_tree->get_node_count(), 5);
    EXPECT_EQ(small_tree->get_available_nodes(), 0);
}

TEST_F(NodePoolTest, HasSpaceForCheck) {
    EXPECT_TRUE(tree->has_space_for(1));
    EXPECT_TRUE(tree->has_space_for(100));
    EXPECT_TRUE(tree->has_space_for(1000));
    EXPECT_FALSE(tree->has_space_for(1001));

    // Allocate some nodes and check again
    tree->allocate_nodes(500);
    EXPECT_TRUE(tree->has_space_for(1));
    EXPECT_TRUE(tree->has_space_for(500));
    EXPECT_FALSE(tree->has_space_for(501));
}

TEST_F(NodePoolTest, ClearResetsPool) {
    // Allocate some nodes
    tree->allocate_nodes(100);
    EXPECT_EQ(tree->get_node_count(), 100);
    EXPECT_EQ(tree->get_available_nodes(), 900);

    // Clear the tree
    tree->clear();
    EXPECT_EQ(tree->get_node_count(), 0);
    EXPECT_EQ(tree->get_available_nodes(), 1000);

    // Should be able to allocate again
    NodeIndex node = tree->allocate_node();
    EXPECT_NE(node, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 1);
}

TEST_F(NodePoolTest, AddRootNodeUsesPool) {
    // Add root node
    NodeIndex root = tree->add_root_node(0.5f, 0);
    EXPECT_NE(root, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 1);
    EXPECT_LE(tree->get_available_nodes(), tree->get_max_nodes());

    // Verify root is valid
    EXPECT_TRUE(tree->is_valid_index(root));
    EXPECT_EQ(tree->get_prior_prob(root), 0.5f);
    EXPECT_EQ(tree->get_flags(root).current_player(), 0);
}

TEST_F(NodePoolTest, ZeroAllocationHandling) {
    // Test edge cases
    NodeIndex nodes = tree->allocate_nodes(0);
    EXPECT_EQ(nodes, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 0);

    // Deallocate zero nodes should be safe
    tree->deallocate_nodes(0, 0);
    EXPECT_EQ(tree->get_node_count(), 0);
}

TEST_F(NodePoolTest, ClearRemovesStaleData) {
    NodeIndex node = tree->allocate_node();
    ASSERT_NE(node, NULL_NODE_INDEX);

    tree->set_visit_count(node, 42.0f);
    tree->set_total_value(node, -3.5f);
    tree->set_prior_prob(node, 0.75f);
    NodeFlags flags = tree->get_flags(node);
    flags.set_expanded(true);
    tree->set_flags(node, flags);

    tree->clear();

    NodeIndex new_node = tree->allocate_node();
    ASSERT_NE(new_node, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_visit_count(new_node), 0.0f);
    EXPECT_EQ(tree->get_total_value(new_node), 0.0f);
    EXPECT_EQ(tree->get_prior_prob(new_node), 0.0f);
    EXPECT_FALSE(tree->get_flags(new_node).is_expanded());
}

TEST_F(NodePoolTest, ThreadLocalBlockConcurrency) {
    const int thread_count = 6;
    const int allocations_per_thread = 512;
    const int total_allocations = thread_count * allocations_per_thread;

    tree = std::make_unique<MCTSTree>(total_allocations + 128);

    std::vector<NodeIndex> allocated;
    allocated.reserve(total_allocations);
    std::mutex collect_mutex;

    auto worker = [&]() {
        std::vector<NodeIndex> local;
        local.reserve(allocations_per_thread);
        for (int i = 0; i < allocations_per_thread; ++i) {
            NodeIndex index = tree->allocate_node();
            ASSERT_NE(index, NULL_NODE_INDEX);
            local.push_back(index);
        }

        std::lock_guard<std::mutex> lock(collect_mutex);
        allocated.insert(allocated.end(), local.begin(), local.end());
    };

    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (int i = 0; i < thread_count; ++i) {
        threads.emplace_back(worker);
    }
    for (auto& thread : threads) {
        thread.join();
    }

    ASSERT_EQ(static_cast<int>(allocated.size()), total_allocations);
    std::sort(allocated.begin(), allocated.end());
    auto unique_end = std::unique(allocated.begin(), allocated.end());
    EXPECT_EQ(unique_end, allocated.end()) << "Duplicate node indices detected";
    EXPECT_EQ(tree->get_node_count(), static_cast<std::size_t>(total_allocations));
}

TEST_F(NodePoolTest, MemoryEfficiency) {
    // Test that we achieve the target memory efficiency
    const std::size_t target_nodes = 10'000'000;  // 10M nodes
    const std::size_t max_memory_gb = 1;  // 1GB limit
    const std::size_t max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024;

    // Create tree with target capacity
    auto large_tree = std::make_unique<MCTSTree>(target_nodes);

    // Check memory usage
    std::size_t memory_usage = large_tree->get_memory_usage();
    EXPECT_LE(memory_usage, max_memory_bytes)
        << "Memory usage (" << memory_usage << " bytes) exceeds 1GB limit";

    // Check bytes per node
    double bytes_per_node = static_cast<double>(memory_usage) / target_nodes;
    EXPECT_LE(bytes_per_node, 64.0)
        << "Bytes per node (" << bytes_per_node << ") exceeds 64 byte target";

    // Should be much better than 64 bytes actually
    EXPECT_LE(bytes_per_node, 40.0)
        << "Bytes per node (" << bytes_per_node << ") should be closer to 32-40 bytes";
}
