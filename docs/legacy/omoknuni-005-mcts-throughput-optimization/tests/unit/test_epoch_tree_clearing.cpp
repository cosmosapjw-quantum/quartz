/**
 * @file test_epoch_tree_clearing.cpp
 * @brief Unit tests for epoch-based tree clearing (T001b validation)
 *
 * Verifies that tree clearing uses epoch-based approach rather than memset,
 * achieving <1ms clear time instead of 10-50ms for large trees.
 */

#include <gtest/gtest.h>
#include "../../cpp_extensions/mcts/tree.hpp"
#include <chrono>
#include <vector>

using namespace mcts;

class EpochTreeClearingTest : public ::testing::Test {
protected:
    static constexpr std::size_t LARGE_TREE_SIZE = 10'000'000;  // 10M nodes
    static constexpr std::size_t SMALL_TREE_SIZE = 100'000;     // 100k nodes

    void SetUp() override {
        // Tests will create trees as needed
    }
};

// ============================================================================
// Basic Clear Performance Tests
// ============================================================================

TEST_F(EpochTreeClearingTest, ClearIsInstant) {
    MCTSTree tree(LARGE_TREE_SIZE);

    // Add root and allocate some nodes
    tree.add_root_node(0.5f, 0);
    for (int i = 0; i < 1000; ++i) {
        tree.allocate_node();
    }

    // Measure clear time
    auto start = std::chrono::high_resolution_clock::now();
    tree.clear();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Clear time for 10M node tree: " << duration_us.count() << " us\n";

    // Should be <1ms (1000us), much faster than 10-50ms memset would take
    EXPECT_LT(duration_us.count(), 1000);
}

TEST_F(EpochTreeClearingTest, MultipleClearsAreConsistent) {
    MCTSTree tree(SMALL_TREE_SIZE);

    std::vector<long> times;

    for (int iteration = 0; iteration < 100; ++iteration) {
        tree.add_root_node(0.5f, 0);

        // Allocate various numbers of nodes
        int nodes_to_allocate = 100 + (iteration * 50);
        for (int i = 0; i < nodes_to_allocate; ++i) {
            tree.allocate_node();
        }

        // Measure clear
        auto start = std::chrono::high_resolution_clock::now();
        tree.clear();
        auto end = std::chrono::high_resolution_clock::now();

        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    // Calculate statistics
    long sum = 0;
    long max_time = 0;
    for (long t : times) {
        sum += t;
        max_time = std::max(max_time, t);
    }
    long avg_time = sum / times.size();

    std::cout << "Average clear time: " << avg_time << " ns\n";
    std::cout << "Max clear time: " << max_time << " ns\n";

    // All clears should be very fast (<10us)
    EXPECT_LT(max_time, 10'000);  // 10 microseconds
}

// ============================================================================
// Node Initialization Tests
// ============================================================================

TEST_F(EpochTreeClearingTest, NodesAreCleanAfterClear) {
    MCTSTree tree(SMALL_TREE_SIZE);

    // First generation: add root and allocate nodes
    NodeIndex root1 = tree.add_root_node(0.5f, 0);
    ASSERT_EQ(root1, 0);

    NodeIndex node1 = tree.allocate_node();
    ASSERT_NE(node1, NULL_NODE_INDEX);

    // Set some values (simulate MCTS usage)
    tree.set_visit_count(root1, 100);
    tree.set_total_value(root1, 50.0f);
    tree.set_visit_count(node1, 25);
    tree.set_total_value(node1, 10.0f);

    // Clear the tree
    tree.clear();
    EXPECT_EQ(tree.get_node_count(), 0);

    // Second generation: add new root
    NodeIndex root2 = tree.add_root_node(0.3f, 1);
    ASSERT_EQ(root2, 0);  // Same index as before

    // Root should be clean (newly initialized)
    EXPECT_FLOAT_EQ(tree.get_visit_count(root2), 0.0f);
    EXPECT_FLOAT_EQ(tree.get_total_value(root2), 0.0f);
    EXPECT_FLOAT_EQ(tree.get_prior_prob(root2), 0.3f);  // Set by add_root_node

    // Allocate a node - should also be clean
    NodeIndex node2 = tree.allocate_node();
    EXPECT_FLOAT_EQ(tree.get_visit_count(node2), 0.0f);
    EXPECT_FLOAT_EQ(tree.get_total_value(node2), 0.0f);
}

TEST_F(EpochTreeClearingTest, OnlyAllocatedNodesAreInitialized) {
    MCTSTree tree(LARGE_TREE_SIZE);

    // Add root
    tree.add_root_node(0.5f, 0);

    // Allocate only 100 nodes
    for (int i = 0; i < 100; ++i) {
        NodeIndex node = tree.allocate_node();
        EXPECT_NE(node, NULL_NODE_INDEX);
    }

    EXPECT_EQ(tree.get_node_count(), 101);  // root + 100 nodes

    // Clear should be instant even though tree capacity is 10M
    auto start = std::chrono::high_resolution_clock::now();
    tree.clear();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should be <100us (if it were memset'ing 10M nodes, it would take 25ms+)
    EXPECT_LT(duration_us.count(), 100);

    std::cout << "Clear time when using 101/" << LARGE_TREE_SIZE << " nodes: "
              << duration_us.count() << " us\n";
}

// ============================================================================
// Memory Efficiency Tests
// ============================================================================

TEST_F(EpochTreeClearingTest, MemoryFootprintIsConstant) {
    MCTSTree tree(SMALL_TREE_SIZE);

    std::size_t initial_memory = tree.get_memory_usage();

    // Fill tree with nodes
    tree.add_root_node(0.5f, 0);
    for (int i = 0; i < 10'000; ++i) {
        tree.allocate_node();
    }

    std::size_t after_allocation = tree.get_memory_usage();

    // Clear tree
    tree.clear();

    std::size_t after_clear = tree.get_memory_usage();

    // Memory usage should remain constant (epoch-based clearing doesn't free memory)
    EXPECT_EQ(initial_memory, after_allocation);
    EXPECT_EQ(initial_memory, after_clear);

    std::cout << "Memory usage: " << initial_memory / (1024 * 1024) << " MB (constant)\n";
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(EpochTreeClearingTest, RepeatedClearAndAllocate) {
    MCTSTree tree(SMALL_TREE_SIZE);

    for (int iteration = 0; iteration < 1000; ++iteration) {
        tree.add_root_node(0.5f, 0);

        // Allocate different amounts each iteration
        int nodes = 10 + (iteration % 100);
        for (int i = 0; i < nodes; ++i) {
            tree.allocate_node();
        }

        // Clear should be instant every time
        auto start = std::chrono::high_resolution_clock::now();
        tree.clear();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        ASSERT_LT(duration_ns, 10'000);  // <10us
    }
}

TEST_F(EpochTreeClearingTest, LargeTreeStressTest) {
    MCTSTree tree(LARGE_TREE_SIZE);

    // Allocate 1M nodes
    tree.add_root_node(0.5f, 0);
    for (int i = 0; i < 999'999; ++i) {
        NodeIndex node = tree.allocate_node();
        if (node == NULL_NODE_INDEX) {
            break;  // Out of space
        }
    }

    std::size_t nodes_before = tree.get_node_count();
    EXPECT_GT(nodes_before, 100'000);  // Should have allocated many nodes

    // Clear should still be instant
    auto start = std::chrono::high_resolution_clock::now();
    tree.clear();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Clear time for " << nodes_before << " nodes: " << duration_us << " us\n";

    // Should be <1ms even for 1M nodes
    EXPECT_LT(duration_us, 1000);
    EXPECT_EQ(tree.get_node_count(), 0);
}

// ============================================================================
// Comparison with Hypothetical Memset
// ============================================================================

TEST_F(EpochTreeClearingTest, EpochClearingVsMemsetComparison) {
    MCTSTree tree(LARGE_TREE_SIZE);

    tree.add_root_node(0.5f, 0);
    for (int i = 0; i < 1000; ++i) {
        tree.allocate_node();
    }

    // Measure actual epoch-based clear
    auto start = std::chrono::high_resolution_clock::now();
    tree.clear();
    auto end = std::chrono::high_resolution_clock::now();

    auto epoch_clear_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Calculate theoretical memset time
    // 10M nodes * 27 bytes/node = 270MB
    // At 10GB/s memset speed: 270MB / 10GB * 1000ms = 27ms
    double theoretical_memset_ms = (10'000'000.0 * 27.0) / (10.0 * 1024.0 * 1024.0 * 1024.0) * 1000.0;

    std::cout << "Epoch-based clear: " << epoch_clear_us << " us\n";
    std::cout << "Theoretical memset: " << theoretical_memset_ms << " ms ("
              << (theoretical_memset_ms * 1000.0) << " us)\n";
    std::cout << "Speedup: " << (theoretical_memset_ms * 1000.0 / epoch_clear_us) << "x\n";

    // Epoch clearing should be at least 100x faster than memset
    EXPECT_LT(epoch_clear_us, theoretical_memset_ms * 1000.0 / 100.0);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
