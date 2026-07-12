/**
 * @file test_simulation_backup.cpp
 * @brief Unit tests for SimulationRunner::backup_value() integration
 *
 * Tests the backup phase through SimulationRunner, validating:
 * 1. Correct delegation to BackupManager
 * 2. Virtual loss removal during backup
 * 3. Sign flipping across multiple tree levels
 * 4. Visit count and Q-value correctness
 *
 * HOWTO-RUN-TESTS:
 * ================
 * Compile with:
 * g++ -std=c++17 -O2 -march=native -I../../cpp_extensions -o test_simulation_backup \
 *     test_simulation_backup.cpp \
 *     ../../cpp_extensions/mcts/tree.cpp \
 *     ../../cpp_extensions/mcts/selection.cpp \
 *     ../../cpp_extensions/mcts/backup.cpp \
 *     ../../cpp_extensions/mcts/virtual_loss.cpp \
 *     ../../cpp_extensions/mcts/simulation_runner.cpp \
 *     ../../cpp_extensions/utils/igamestate.cpp
 *
 * Or build with CMake test suite.
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <memory>
#include <cmath>
#include "mcts/simulation_runner.hpp"
#include "mcts/tree.hpp"
#include "mcts/selection.hpp"
#include "mcts/backup.hpp"
#include "mcts/virtual_loss.hpp"

using namespace mcts;

/**
 * @brief Test fixture for backup validation
 */
class BackupTestFixture {
public:
    BackupTestFixture(size_t tree_size = 10000)
        : tree(tree_size)
        , selector()
        , backup(tree)
        , vl_manager(tree)
        , runner(tree, selector, backup, vl_manager)
    {
        root = tree.add_root_node(0.5f, 0);
    }

    MCTSTree tree;
    PUCTSelector selector;
    BackupManager backup;
    VirtualLossManager vl_manager;
    SimulationRunner runner;
    NodeIndex root;
};

void test_single_node_backup() {
    std::cout << "Testing single node backup..." << std::endl;

    BackupTestFixture fixture;

    // Create path with just root
    std::vector<NodeIndex> path = {fixture.root};
    float leaf_value = 0.7f;

    // Perform backup
    fixture.runner.backup_value_public(path, leaf_value);

    // Verify root was updated
    float visit_count = fixture.tree.get_visit_count(fixture.root);
    float total_value = fixture.tree.get_total_value(fixture.root);

    assert(std::abs(visit_count - 1.0f) < 1e-6f && "Root visit count should be 1");
    assert(std::abs(total_value - 0.7f) < 1e-6f && "Root total value should be 0.7");

    float q_value = total_value / visit_count;
    assert(std::abs(q_value - 0.7f) < 1e-6f && "Root Q-value should be 0.7");

    std::cout << "✓ Single node backup test passed" << std::endl;
}

void test_two_level_sign_flip() {
    std::cout << "Testing two-level sign flipping..." << std::endl;

    BackupTestFixture fixture;

    // Create child node
    NodeIndex child = fixture.tree.allocate_node();
    fixture.tree.set_parent_index(child, fixture.root);
    fixture.tree.set_visit_count(child, 0.0f);
    fixture.tree.set_total_value(child, 0.0f);

    // Create path: child → root
    std::vector<NodeIndex> path = {child, fixture.root};
    float leaf_value = 0.8f;

    // Perform backup
    fixture.runner.backup_value_public(path, leaf_value);

    // Child (level 0): should get +0.8
    float child_value = fixture.tree.get_total_value(child);
    float child_visits = fixture.tree.get_visit_count(child);
    assert(std::abs(child_value - 0.8f) < 1e-6f && "Child value should be +0.8");
    assert(std::abs(child_visits - 1.0f) < 1e-6f && "Child visits should be 1");

    // Root (level 1): should get -0.8 (sign flipped)
    float root_value = fixture.tree.get_total_value(fixture.root);
    float root_visits = fixture.tree.get_visit_count(fixture.root);
    assert(std::abs(root_value - (-0.8f)) < 1e-6f && "Root value should be -0.8");
    assert(std::abs(root_visits - 1.0f) < 1e-6f && "Root visits should be 1");

    std::cout << "✓ Two-level sign flip test passed" << std::endl;
}

void test_three_level_sign_flip() {
    std::cout << "Testing three-level sign flipping..." << std::endl;

    BackupTestFixture fixture;

    // Create child and grandchild nodes
    NodeIndex child = fixture.tree.allocate_node();
    NodeIndex grandchild = fixture.tree.allocate_node();

    fixture.tree.set_parent_index(child, fixture.root);
    fixture.tree.set_parent_index(grandchild, child);

    fixture.tree.set_visit_count(child, 0.0f);
    fixture.tree.set_total_value(child, 0.0f);
    fixture.tree.set_visit_count(grandchild, 0.0f);
    fixture.tree.set_total_value(grandchild, 0.0f);

    // Create path: grandchild → child → root
    std::vector<NodeIndex> path = {grandchild, child, fixture.root};
    float leaf_value = 0.6f;

    // Perform backup
    fixture.runner.backup_value_public(path, leaf_value);

    // Grandchild (level 0): should get +0.6
    float gc_value = fixture.tree.get_total_value(grandchild);
    assert(std::abs(gc_value - 0.6f) < 1e-6f && "Grandchild value should be +0.6");

    // Child (level 1): should get -0.6 (sign flipped)
    float child_value = fixture.tree.get_total_value(child);
    assert(std::abs(child_value - (-0.6f)) < 1e-6f && "Child value should be -0.6");

    // Root (level 2): should get +0.6 (sign flipped again)
    float root_value = fixture.tree.get_total_value(fixture.root);
    assert(std::abs(root_value - 0.6f) < 1e-6f && "Root value should be +0.6");

    std::cout << "✓ Three-level sign flip test passed" << std::endl;
}

void test_virtual_loss_removal() {
    std::cout << "Testing virtual loss removal during backup..." << std::endl;

    BackupTestFixture fixture;

    // Create child node
    NodeIndex child = fixture.tree.allocate_node();
    fixture.tree.set_parent_index(child, fixture.root);
    fixture.tree.set_visit_count(child, 0.0f);
    fixture.tree.set_total_value(child, 0.0f);
    fixture.tree.set_virtual_loss(child, 0.0f);

    // Apply virtual loss to child
    fixture.vl_manager.apply_virtual_loss(child);

    // Verify virtual loss was applied
    float vl_before = fixture.tree.get_virtual_loss(child);
    assert(vl_before > 0.0f && "Virtual loss should be applied");

    // Create path and perform backup (should remove virtual loss)
    std::vector<NodeIndex> path = {child, fixture.root};
    float leaf_value = 0.5f;

    fixture.runner.backup_value_public(path, leaf_value);

    // Verify virtual loss was removed
    float vl_after = fixture.tree.get_virtual_loss(child);
    assert(std::abs(vl_after) < 1e-6f && "Virtual loss should be removed after backup");

    std::cout << "✓ Virtual loss removal test passed" << std::endl;
}

void test_multiple_backups() {
    std::cout << "Testing multiple backups accumulate correctly..." << std::endl;

    BackupTestFixture fixture;

    // Create child node
    NodeIndex child = fixture.tree.allocate_node();
    fixture.tree.set_parent_index(child, fixture.root);
    fixture.tree.set_visit_count(child, 0.0f);
    fixture.tree.set_total_value(child, 0.0f);

    // Create path
    std::vector<NodeIndex> path = {child, fixture.root};

    // Perform multiple backups
    fixture.runner.backup_value_public(path, 0.8f);  // First backup
    fixture.runner.backup_value_public(path, 0.6f);  // Second backup
    fixture.runner.backup_value_public(path, 0.4f);  // Third backup

    // Child should accumulate: 0.8 + 0.6 + 0.4 = 1.8
    float child_value = fixture.tree.get_total_value(child);
    float child_visits = fixture.tree.get_visit_count(child);
    assert(std::abs(child_visits - 3.0f) < 1e-6f && "Child should have 3 visits");
    assert(std::abs(child_value - 1.8f) < 1e-6f && "Child value should be 1.8");

    // Root should accumulate: -0.8 + (-0.6) + (-0.4) = -1.8
    float root_value = fixture.tree.get_total_value(fixture.root);
    float root_visits = fixture.tree.get_visit_count(fixture.root);
    assert(std::abs(root_visits - 3.0f) < 1e-6f && "Root should have 3 visits");
    assert(std::abs(root_value - (-1.8f)) < 1e-6f && "Root value should be -1.8");

    // Q-values
    float child_q = child_value / child_visits;
    float root_q = root_value / root_visits;
    assert(std::abs(child_q - 0.6f) < 1e-6f && "Child Q-value should be 0.6");
    assert(std::abs(root_q - (-0.6f)) < 1e-6f && "Root Q-value should be -0.6");

    std::cout << "✓ Multiple backups test passed" << std::endl;
}

void test_terminal_value_backup() {
    std::cout << "Testing terminal value backup..." << std::endl;

    BackupTestFixture fixture;

    // Create a path
    NodeIndex child = fixture.tree.allocate_node();
    fixture.tree.set_parent_index(child, fixture.root);
    fixture.tree.set_visit_count(child, 0.0f);
    fixture.tree.set_total_value(child, 0.0f);

    std::vector<NodeIndex> path = {child, fixture.root};

    // Test win value (+1.0)
    fixture.runner.backup_value_public(path, 1.0f);

    float child_value = fixture.tree.get_total_value(child);
    float root_value = fixture.tree.get_total_value(fixture.root);

    assert(std::abs(child_value - 1.0f) < 1e-6f && "Child should have +1.0 for win");
    assert(std::abs(root_value - (-1.0f)) < 1e-6f && "Root should have -1.0 (opponent loss)");

    std::cout << "✓ Terminal value backup test passed" << std::endl;
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Running SimulationRunner::backup_value Tests" << std::endl;
    std::cout << "======================================" << std::endl << std::endl;

    try {
        test_single_node_backup();
        test_two_level_sign_flip();
        test_three_level_sign_flip();
        test_virtual_loss_removal();
        test_multiple_backups();
        test_terminal_value_backup();

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
