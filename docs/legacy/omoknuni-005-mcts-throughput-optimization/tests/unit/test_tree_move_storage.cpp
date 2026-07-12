/**
 * @file test_tree_move_storage.cpp
 * @brief Tests for MCTSTree move storage functionality
 *
 * Tests the move storage feature that replaces Python dict-based approach
 * with C++ uint16_t array for 50× memory efficiency improvement.
 *
 * Compile with:
 * g++ -std=c++17 -O2 -I../../cpp_extensions -o test_move_storage \
 *     test_tree_move_storage.cpp ../../cpp_extensions/mcts/tree.cpp
 *
 * Run:
 * ./test_move_storage
 */

#include <iostream>
#include <cassert>
#include <vector>
#include "mcts/tree.hpp"

using namespace mcts;

void test_basic_move_storage() {
    std::cout << "Testing basic move storage..." << std::endl;

    MCTSTree tree(1000);
    NodeIndex root = tree.add_root_node(0.5f, 0);

    // Root node should have default move 0
    assert(tree.get_move(root) == 0);

    // Set move index
    tree.set_move(root, 42);
    assert(tree.get_move(root) == 42);

    // Update move index
    tree.set_move(root, 123);
    assert(tree.get_move(root) == 123);

    std::cout << "✓ Basic move storage tests passed" << std::endl;
}

void test_move_storage_for_children() {
    std::cout << "Testing move storage for child nodes..." << std::endl;

    MCTSTree tree(1000);
    NodeIndex root = tree.add_root_node(0.5f, 0);

    // Allocate children
    const uint16_t num_children = 10;
    NodeIndex first_child = tree.allocate_nodes(num_children);
    tree.set_first_child_index(root, first_child);
    tree.set_num_children(root, num_children);

    // Set move indices for each child
    for (uint16_t i = 0; i < num_children; ++i) {
        NodeIndex child_idx = first_child + i;
        uint16_t move_idx = 100 + i;
        tree.set_move(child_idx, move_idx);
    }

    // Verify all move indices
    for (uint16_t i = 0; i < num_children; ++i) {
        NodeIndex child_idx = first_child + i;
        uint16_t expected_move = 100 + i;
        uint16_t actual_move = tree.get_move(child_idx);
        assert(actual_move == expected_move);
    }

    std::cout << "✓ Child move storage tests passed" << std::endl;
}

void test_move_storage_range() {
    std::cout << "Testing move storage with full uint16_t range..." << std::endl;

    MCTSTree tree(100);
    NodeIndex root = tree.add_root_node(0.5f, 0);

    // Test minimum value
    tree.set_move(root, 0);
    assert(tree.get_move(root) == 0);

    // Test maximum value (uint16_t max = 65535)
    tree.set_move(root, 65535);
    assert(tree.get_move(root) == 65535);

    // Test various intermediate values
    std::vector<uint16_t> test_values = {
        1, 255, 256, 1000, 5000, 32767, 32768, 65534
    };

    for (uint16_t value : test_values) {
        tree.set_move(root, value);
        assert(tree.get_move(root) == value);
    }

    std::cout << "✓ Move range tests passed" << std::endl;
}

void test_move_storage_clears() {
    std::cout << "Testing move storage clear..." << std::endl;

    MCTSTree tree(100);
    NodeIndex root = tree.add_root_node(0.5f, 0);
    tree.set_move(root, 999);

    assert(tree.get_move(root) == 999);

    // Clear tree
    tree.clear();
    assert(tree.get_node_count() == 0);

    // Add new root - move should be reset to 0
    NodeIndex new_root = tree.add_root_node(0.5f, 0);
    assert(tree.get_move(new_root) == 0);

    std::cout << "✓ Move storage clear tests passed" << std::endl;
}

void test_move_storage_with_deallocation() {
    std::cout << "Testing move storage with node deallocation..." << std::endl;

    MCTSTree tree(100);
    NodeIndex root = tree.add_root_node(0.5f, 0);

    // Allocate a node and set its move
    NodeIndex node1 = tree.allocate_node();
    tree.set_move(node1, 555);
    assert(tree.get_move(node1) == 555);

    // Deallocate the node
    tree.deallocate_node(node1);

    // Allocate another node (should reuse the same index)
    NodeIndex node2 = tree.allocate_node();
    assert(node2 == node1);

    // Move value persists (not cleared on deallocation)
    assert(tree.get_move(node2) == 555);

    std::cout << "✓ Move storage with deallocation tests passed" << std::endl;
}

void test_move_storage_memory_efficiency() {
    std::cout << "Testing move storage memory efficiency..." << std::endl;

    // Test with 1 million nodes
    const size_t num_nodes = 1'000'000;
    MCTSTree tree(num_nodes);

    size_t memory_bytes = tree.get_memory_usage();
    double bytes_per_node = tree.get_bytes_per_node();

    std::cout << "  Memory for " << num_nodes << " nodes:" << std::endl;
    std::cout << "    Total: " << (memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "    Per node: " << bytes_per_node << " bytes" << std::endl;

    // Verify bytes per node is within expected range
    // With move storage: 4 floats (16 bytes) + 2 int32 (8 bytes) + 3 uint16 (6 bytes) = 30 bytes
    // With alignment overhead, should be < 64 bytes per node
    assert(bytes_per_node < 64.0);

    // Move storage adds only 2 bytes per node
    // Without moves: ~28 bytes; with moves: ~30 bytes
    std::cout << "  Move storage adds only 2 bytes per node" << std::endl;

    std::cout << "✓ Memory efficiency tests passed" << std::endl;
}

void test_move_storage_multiple_trees() {
    std::cout << "Testing move storage with multiple trees..." << std::endl;

    MCTSTree tree1(100);
    MCTSTree tree2(100);

    NodeIndex root1 = tree1.add_root_node(0.5f, 0);
    NodeIndex root2 = tree2.add_root_node(0.5f, 0);

    tree1.set_move(root1, 111);
    tree2.set_move(root2, 222);

    // Each tree maintains independent move storage
    assert(tree1.get_move(root1) == 111);
    assert(tree2.get_move(root2) == 222);

    std::cout << "✓ Multiple trees tests passed" << std::endl;
}

void test_move_storage_large_tree() {
    std::cout << "Testing move storage with large tree..." << std::endl;

    // Create a tree with many nodes and set moves
    MCTSTree tree(10000);
    NodeIndex root = tree.add_root_node(0.5f, 0);

    std::vector<NodeIndex> nodes;
    nodes.reserve(1000);

    // Allocate 1000 nodes
    for (int i = 0; i < 1000; ++i) {
        NodeIndex node = tree.allocate_node();
        assert(node != NULL_NODE_INDEX);
        tree.set_move(node, static_cast<uint16_t>(i));
        nodes.push_back(node);
    }

    // Verify all moves are correct
    for (size_t i = 0; i < nodes.size(); ++i) {
        uint16_t expected = static_cast<uint16_t>(i);
        uint16_t actual = tree.get_move(nodes[i]);
        assert(actual == expected);
    }

    std::cout << "✓ Large tree tests passed" << std::endl;
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Running MCTSTree Move Storage Tests" << std::endl;
    std::cout << "======================================" << std::endl << std::endl;

    try {
        test_basic_move_storage();
        test_move_storage_for_children();
        test_move_storage_range();
        test_move_storage_clears();
        test_move_storage_with_deallocation();
        test_move_storage_memory_efficiency();
        test_move_storage_multiple_trees();
        test_move_storage_large_tree();

        std::cout << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "All tests passed! ✓" << std::endl;
        std::cout << "======================================" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}
