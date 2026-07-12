/**
 * @file test_node_pool_simple.cpp
 * @brief Simple standalone test for node pool functionality
 *
 * This can be compiled and run directly without Google Test framework.
 * Compile with: g++ -std=c++17 -O2 -I../../cpp_extensions -o test_node_pool test_node_pool_simple.cpp ../../cpp_extensions/mcts/tree.cpp
 */

#include <iostream>
#include <cassert>
#include <chrono>
#include "mcts/tree.hpp"

using namespace mcts;

void test_basic_allocation() {
    std::cout << "Testing basic allocation..." << std::endl;

    MCTSTree tree(1000);

    // Test single node allocation
    NodeIndex node1 = tree.allocate_node();
    assert(node1 != NULL_NODE_INDEX);
    assert(tree.get_node_count() == 1);
    assert(tree.get_available_nodes() == 999);

    // Test multiple node allocation
    NodeIndex batch = tree.allocate_nodes(10);
    assert(batch != NULL_NODE_INDEX);
    assert(tree.get_node_count() == 11);
    assert(tree.get_available_nodes() == 989);

    std::cout << "✓ Basic allocation tests passed" << std::endl;
}

void test_deallocation_and_reuse() {
    std::cout << "Testing deallocation and reuse..." << std::endl;

    MCTSTree tree(100);

    // Allocate some nodes
    NodeIndex node1 = tree.allocate_node();
    NodeIndex node2 = tree.allocate_node();
    NodeIndex node3 = tree.allocate_node();

    assert(tree.get_node_count() == 3);

    // Deallocate middle node
    tree.deallocate_node(node2);
    assert(tree.get_node_count() == 2);
    assert(tree.get_available_nodes() == 98);  // 97 unused + 1 freed

    // Allocate new node - should reuse freed node
    NodeIndex node4 = tree.allocate_node();
    assert(node4 == node2);  // Should reuse the freed node
    assert(tree.get_node_count() == 3);
    assert(tree.get_available_nodes() == 97);

    std::cout << "✓ Deallocation and reuse tests passed" << std::endl;
}

void test_exhaustion() {
    std::cout << "Testing pool exhaustion..." << std::endl;

    MCTSTree tree(5);

    // Allocate all nodes
    for (int i = 0; i < 5; ++i) {
        NodeIndex node = tree.allocate_node();
        assert(node != NULL_NODE_INDEX);
    }

    assert(tree.get_node_count() == 5);
    assert(tree.get_available_nodes() == 0);

    // Try to allocate one more - should fail
    NodeIndex extra = tree.allocate_node();
    assert(extra == NULL_NODE_INDEX);

    std::cout << "✓ Pool exhaustion tests passed" << std::endl;
}

void test_memory_efficiency() {
    std::cout << "Testing memory efficiency..." << std::endl;

    const std::size_t nodes = 10'000'000;  // 10M nodes
    MCTSTree tree(nodes);

    std::size_t memory_usage = tree.get_memory_usage();
    double bytes_per_node = static_cast<double>(memory_usage) / nodes;

    std::cout << "Memory usage for " << nodes << " nodes: " << memory_usage << " bytes" << std::endl;
    std::cout << "Bytes per node: " << bytes_per_node << std::endl;

    // Check memory efficiency targets
    assert(memory_usage <= 1024 * 1024 * 1024);  // Must be <= 1GB
    assert(bytes_per_node <= 64.0);  // Must be <= 64 bytes per node

    std::cout << "✓ Memory efficiency targets met" << std::endl;
}

void test_performance() {
    std::cout << "Testing allocation performance..." << std::endl;

    MCTSTree tree(1'000'000);

    auto start = std::chrono::high_resolution_clock::now();

    // Allocate 100k nodes
    std::vector<NodeIndex> nodes;
    nodes.reserve(100'000);

    for (int i = 0; i < 100'000; ++i) {
        NodeIndex node = tree.allocate_node();
        assert(node != NULL_NODE_INDEX);
        nodes.push_back(node);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double allocations_per_second = 100'000.0 / (duration.count() / 1'000'000.0);

    std::cout << "Allocated 100k nodes in " << duration.count() << " microseconds" << std::endl;
    std::cout << "Allocation rate: " << static_cast<int>(allocations_per_second) << " nodes/second" << std::endl;

    // Should be very fast (target: > 1M allocations/second)
    assert(allocations_per_second > 1'000'000);

    std::cout << "✓ Performance targets met" << std::endl;
}

void test_root_node_integration() {
    std::cout << "Testing root node integration..." << std::endl;

    MCTSTree tree(1000);

    // Add root node using pool system
    NodeIndex root = tree.add_root_node(0.5f, 0);
    assert(root != NULL_NODE_INDEX);
    assert(tree.get_node_count() == 1);
    assert(tree.is_valid_index(root));

    // Verify root node properties
    assert(tree.get_prior_prob(root) == 0.5f);
    assert(tree.get_flags(root).current_player() == 0);
    assert(tree.get_parent_index(root) == NULL_NODE_INDEX);

    std::cout << "✓ Root node integration tests passed" << std::endl;
}

int main() {
    std::cout << "Running node pool tests..." << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_basic_allocation();
        test_deallocation_and_reuse();
        test_exhaustion();
        test_memory_efficiency();
        test_performance();
        test_root_node_integration();

        std::cout << "========================================" << std::endl;
        std::cout << "✅ All tests passed successfully!" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}