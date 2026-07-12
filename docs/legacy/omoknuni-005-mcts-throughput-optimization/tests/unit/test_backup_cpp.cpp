/**
 * @file test_backup_cpp.cpp
 * @brief C++ unit tests for MCTS value backup mechanism
 *
 * Tests the actual C++ implementation of value backup using Google Test.
 * These tests verify sign flipping, atomic operations, and integration
 * with the MCTS tree structure.
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <chrono>
#include <random>
#include <future>

#include "../../cpp_extensions/mcts/backup.hpp"
#include "../../cpp_extensions/mcts/tree.hpp"
#include "../../cpp_extensions/mcts/virtual_loss.hpp"

namespace mcts {
namespace test {

class BackupTest : public ::testing::Test {
protected:
    void SetUp() override {
        tree_ = std::make_unique<MCTSTree>(1000);
        root_index_ = tree_->add_root_node(0.5f, 0);

        // Create a small tree for testing: root -> child -> grandchild
        child_index_ = tree_->allocate_nodes(1);
        grandchild_index_ = tree_->allocate_nodes(1);

        if (child_index_ != NULL_NODE_INDEX && grandchild_index_ != NULL_NODE_INDEX) {
            tree_->set_parent_index(child_index_, root_index_);
            tree_->set_parent_index(grandchild_index_, child_index_);
        }

        config_ = BackupConfig(true, true, -1.0f, 1.0f);
        manager_ = std::make_unique<BackupManager>(*tree_, config_);
    }

    void TearDown() override {
        manager_.reset();
        tree_.reset();
    }

    std::unique_ptr<MCTSTree> tree_;
    std::unique_ptr<BackupManager> manager_;
    BackupConfig config_;
    NodeIndex root_index_;
    NodeIndex child_index_;
    NodeIndex grandchild_index_;
};

TEST_F(BackupTest, SingleNodeBackup) {
    // Test backup to root only
    std::vector<NodeIndex> path = {root_index_};
    float leaf_value = 0.6f;

    BackupResult result = manager_->backup_value_along_path(path, leaf_value);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.nodes_updated, 1);
    EXPECT_FLOAT_EQ(result.original_leaf_value, leaf_value);

    // Check root was updated correctly
    EXPECT_FLOAT_EQ(tree_->get_visit_count(root_index_), 1.0f);
    EXPECT_FLOAT_EQ(tree_->get_total_value(root_index_), 0.6f);
    EXPECT_FLOAT_EQ(manager_->get_q_value(root_index_), 0.6f);
}

TEST_F(BackupTest, TwoLevelSignFlipping) {
    // Test proper sign flipping across two levels
    std::vector<NodeIndex> path = {child_index_, root_index_};
    float leaf_value = 0.8f;

    BackupResult result = manager_->backup_value_along_path(path, leaf_value);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.nodes_updated, 2);

    // Child (level 0): should get +0.8
    EXPECT_FLOAT_EQ(tree_->get_total_value(child_index_), 0.8f);
    EXPECT_FLOAT_EQ(tree_->get_visit_count(child_index_), 1.0f);

    // Root (level 1): should get -0.8 (sign flipped)
    EXPECT_FLOAT_EQ(tree_->get_total_value(root_index_), -0.8f);
    EXPECT_FLOAT_EQ(tree_->get_visit_count(root_index_), 1.0f);
}

TEST_F(BackupTest, ThreeLevelSignFlipping) {
    // Test proper sign flipping across three levels
    std::vector<NodeIndex> path = {grandchild_index_, child_index_, root_index_};
    float leaf_value = 0.4f;

    BackupResult result = manager_->backup_value_along_path(path, leaf_value);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.nodes_updated, 3);

    // Grandchild (level 0): +0.4
    EXPECT_FLOAT_EQ(tree_->get_total_value(grandchild_index_), 0.4f);
    EXPECT_FLOAT_EQ(manager_->get_q_value(grandchild_index_), 0.4f);

    // Child (level 1): -0.4 (flipped)
    EXPECT_FLOAT_EQ(tree_->get_total_value(child_index_), -0.4f);
    EXPECT_FLOAT_EQ(manager_->get_q_value(child_index_), -0.4f);

    // Root (level 2): +0.4 (flipped back)
    EXPECT_FLOAT_EQ(tree_->get_total_value(root_index_), 0.4f);
    EXPECT_FLOAT_EQ(manager_->get_q_value(root_index_), 0.4f);
}

TEST_F(BackupTest, NegativeLeafValue) {
    // Test backup with negative leaf value
    std::vector<NodeIndex> path = {child_index_, root_index_};
    float leaf_value = -0.7f;

    BackupResult result = manager_->backup_value_along_path(path, leaf_value);

    EXPECT_TRUE(result.success);

    // Child: -0.7
    EXPECT_FLOAT_EQ(tree_->get_total_value(child_index_), -0.7f);

    // Root: +0.7 (sign flipped)
    EXPECT_FLOAT_EQ(tree_->get_total_value(root_index_), 0.7f);
}

TEST_F(BackupTest, MultipleBackupsAccumulate) {
    // Test that multiple backups accumulate correctly
    std::vector<NodeIndex> path = {root_index_};

    // First backup
    BackupResult result1 = manager_->backup_value_along_path(path, 0.3f);
    EXPECT_TRUE(result1.success);

    // Second backup
    BackupResult result2 = manager_->backup_value_along_path(path, 0.4f);
    EXPECT_TRUE(result2.success);

    // Check accumulated values
    EXPECT_FLOAT_EQ(tree_->get_visit_count(root_index_), 2.0f);
    EXPECT_FLOAT_EQ(tree_->get_total_value(root_index_), 0.7f);  // 0.3 + 0.4
    EXPECT_FLOAT_EQ(manager_->get_q_value(root_index_), 0.35f);  // 0.7 / 2.0
}

TEST_F(BackupTest, TerminalValueBackup) {
    // Test terminal value backup (should work same as regular backup)
    std::vector<NodeIndex> path = {child_index_, root_index_};
    float terminal_value = 1.0f;  // Win for current player

    BackupResult result = manager_->backup_terminal_value(path, terminal_value);

    EXPECT_TRUE(result.success);

    // Child gets +1.0 (win)
    EXPECT_FLOAT_EQ(tree_->get_total_value(child_index_), 1.0f);

    // Root gets -1.0 (loss from root's perspective)
    EXPECT_FLOAT_EQ(tree_->get_total_value(root_index_), -1.0f);
}

TEST_F(BackupTest, EmptyPath) {
    // Test backup with empty path
    std::vector<NodeIndex> empty_path;

    BackupResult result = manager_->backup_value_along_path(empty_path, 0.5f);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.nodes_updated, 0);
}

TEST_F(BackupTest, InvalidNodeInPath) {
    // Test backup with invalid node index
    std::vector<NodeIndex> invalid_path = {root_index_, 999};

    BackupResult result = manager_->backup_value_along_path(invalid_path, 0.5f);

    EXPECT_FALSE(result.success);
}

TEST_F(BackupTest, InvalidParentChildRelationship) {
    // Create another child that's not related to first child
    NodeIndex unrelated_child = tree_->allocate_nodes(1);
    tree_->set_parent_index(unrelated_child, root_index_);

    // Try to backup through invalid path
    std::vector<NodeIndex> invalid_path = {unrelated_child, child_index_, root_index_};

    BackupResult result = manager_->backup_value_along_path(invalid_path, 0.5f);

    EXPECT_FALSE(result.success);
}

TEST_F(BackupTest, PathNotEndingAtRoot) {
    // Path that doesn't end at root should fail
    std::vector<NodeIndex> invalid_path = {child_index_};

    BackupResult result = manager_->backup_value_along_path(invalid_path, 0.5f);

    EXPECT_FALSE(result.success);
}

TEST_F(BackupTest, ValueClipping) {
    // Test value clipping
    BackupConfig clipping_config(true, true, -1.0f, 1.0f);
    BackupManager clipping_manager(*tree_, clipping_config);

    std::vector<NodeIndex> path = {root_index_};

    // Test clipping of extreme positive value
    BackupResult result1 = clipping_manager.backup_value_along_path(path, 2.0f);
    EXPECT_TRUE(result1.success);
    EXPECT_FLOAT_EQ(tree_->get_total_value(root_index_), 1.0f);  // Clipped to max

    // Reset tree
    tree_->set_total_value(root_index_, 0.0f);
    tree_->set_visit_count(root_index_, 0.0f);

    // Test clipping of extreme negative value
    BackupResult result2 = clipping_manager.backup_value_along_path(path, -2.0f);
    EXPECT_TRUE(result2.success);
    EXPECT_FLOAT_EQ(tree_->get_total_value(root_index_), -1.0f);  // Clipped to min
}

TEST_F(BackupTest, ValueClippingDisabled) {
    // Test with value clipping disabled
    BackupConfig no_clipping_config(false, true);
    BackupManager no_clipping_manager(*tree_, no_clipping_config);

    std::vector<NodeIndex> path = {root_index_};

    BackupResult result = no_clipping_manager.backup_value_along_path(path, 2.0f);
    EXPECT_TRUE(result.success);
    EXPECT_FLOAT_EQ(tree_->get_total_value(root_index_), 2.0f);  // Not clipped
}

TEST_F(BackupTest, Statistics) {
    auto initial_stats = manager_->get_statistics();
    EXPECT_EQ(initial_stats.total_backups, 0);
    EXPECT_EQ(initial_stats.successful_backups, 0);

    // Perform successful backup
    std::vector<NodeIndex> path = {child_index_, root_index_};
    manager_->backup_value_along_path(path, 0.6f);

    auto stats = manager_->get_statistics();
    EXPECT_EQ(stats.total_backups, 1);
    EXPECT_EQ(stats.successful_backups, 1);
    EXPECT_EQ(stats.total_nodes_updated, 2);
    EXPECT_FLOAT_EQ(stats.avg_path_length, 2.0f);
    EXPECT_FLOAT_EQ(stats.avg_absolute_leaf_value, 0.6f);

    // Perform failed backup
    std::vector<NodeIndex> invalid_path = {999};
    manager_->backup_value_along_path(invalid_path, 0.5f);

    auto final_stats = manager_->get_statistics();
    EXPECT_EQ(final_stats.total_backups, 2);
    EXPECT_EQ(final_stats.successful_backups, 1);
    EXPECT_EQ(final_stats.path_validation_failures, 1);
}

TEST_F(BackupTest, ResetStatistics) {
    // Perform some backups
    std::vector<NodeIndex> path = {root_index_};
    manager_->backup_value_along_path(path, 0.5f);
    manager_->backup_value_along_path(path, 0.3f);

    // Reset statistics
    manager_->reset_statistics();

    auto stats = manager_->get_statistics();
    EXPECT_EQ(stats.total_backups, 0);
    EXPECT_EQ(stats.successful_backups, 0);
    EXPECT_EQ(stats.total_nodes_updated, 0);
    EXPECT_FLOAT_EQ(stats.avg_path_length, 0.0f);
    EXPECT_FLOAT_EQ(stats.avg_absolute_leaf_value, 0.0f);
}

TEST_F(BackupTest, VirtualLossIntegration) {
    VirtualLossConfig vl_config(1.0f, true);
    VirtualLossManager vl_manager(*tree_, vl_config);

    std::vector<NodeIndex> path = {child_index_, root_index_};

    // Apply virtual loss first
    EXPECT_TRUE(vl_manager.apply_virtual_loss_to_path(path));

    // Verify virtual loss is applied
    EXPECT_FLOAT_EQ(vl_manager.get_virtual_loss(child_index_), 1.0f);
    EXPECT_FLOAT_EQ(vl_manager.get_virtual_loss(root_index_), 1.0f);

    // Backup with virtual loss removal
    BackupResult result = manager_->backup_value_along_path(path, 0.5f, &vl_manager);
    EXPECT_TRUE(result.success);

    // Virtual loss should be removed
    EXPECT_FLOAT_EQ(vl_manager.get_virtual_loss(child_index_), 0.0f);
    EXPECT_FLOAT_EQ(vl_manager.get_virtual_loss(root_index_), 0.0f);
}

TEST_F(BackupTest, BackupGuard) {
    VirtualLossConfig vl_config(1.0f, true);
    VirtualLossManager vl_manager(*tree_, vl_config);

    std::vector<NodeIndex> path = {child_index_, root_index_};

    // Apply virtual loss first
    vl_manager.apply_virtual_loss_to_path(path);

    {
        BackupGuard guard(*manager_, vl_manager, path, 0.4f);
        EXPECT_TRUE(guard.was_successful());

        // Virtual loss should still be present during backup
        EXPECT_FLOAT_EQ(vl_manager.get_virtual_loss(child_index_), 0.0f);  // Removed by backup
        EXPECT_FLOAT_EQ(vl_manager.get_virtual_loss(root_index_), 0.0f);
    }  // Guard destructor should ensure cleanup

    // Final check - virtual loss should definitely be removed
    EXPECT_FLOAT_EQ(vl_manager.get_virtual_loss(child_index_), 0.0f);
    EXPECT_FLOAT_EQ(vl_manager.get_virtual_loss(root_index_), 0.0f);
}

class BackupThreadSafetyTest : public BackupTest {
protected:
    static constexpr int NUM_THREADS = 8;
    static constexpr int OPERATIONS_PER_THREAD = 1000;
};

TEST_F(BackupThreadSafetyTest, ConcurrentBackupsToSameNode) {
    std::vector<NodeIndex> path = {root_index_};
    std::vector<std::thread> threads;

    // All threads backup to the same node
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([this, path]() {
            for (int j = 0; j < OPERATIONS_PER_THREAD; ++j) {
                manager_->backup_value_along_path(path, 0.1f);
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Check final state
    float expected_visits = NUM_THREADS * OPERATIONS_PER_THREAD;
    float expected_total_value = expected_visits * 0.1f;

    EXPECT_FLOAT_EQ(tree_->get_visit_count(root_index_), expected_visits);
    EXPECT_FLOAT_EQ(tree_->get_total_value(root_index_), expected_total_value);

    // Check statistics
    auto stats = manager_->get_statistics();
    EXPECT_EQ(stats.total_backups, NUM_THREADS * OPERATIONS_PER_THREAD);
    EXPECT_EQ(stats.successful_backups, NUM_THREADS * OPERATIONS_PER_THREAD);
}

TEST_F(BackupThreadSafetyTest, ConcurrentBackupsDifferentPaths) {
    // Create multiple children for different paths
    std::vector<NodeIndex> children;
    for (int i = 0; i < NUM_THREADS; ++i) {
        NodeIndex child = tree_->allocate_nodes(1);
        tree_->set_parent_index(child, root_index_);
        children.push_back(child);
    }

    std::vector<std::future<void>> futures;

    // Each thread backs up through a different child
    for (int i = 0; i < NUM_THREADS; ++i) {
        futures.push_back(std::async(std::launch::async, [this, children, i]() {
            std::vector<NodeIndex> path = {children[i], root_index_};
            for (int j = 0; j < OPERATIONS_PER_THREAD; ++j) {
                manager_->backup_value_along_path(path, 0.2f);
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        }));
    }

    // Wait for all operations to complete
    for (auto& future : futures) {
        future.wait();
    }

    // Check that all children were updated correctly
    for (NodeIndex child : children) {
        EXPECT_FLOAT_EQ(tree_->get_visit_count(child), OPERATIONS_PER_THREAD);
        EXPECT_FLOAT_EQ(tree_->get_total_value(child), OPERATIONS_PER_THREAD * 0.2f);
    }

    // Root should have been updated by all paths with sign flipping
    float expected_root_visits = NUM_THREADS * OPERATIONS_PER_THREAD;
    float expected_root_value = NUM_THREADS * OPERATIONS_PER_THREAD * (-0.2f);  // Sign flipped

    EXPECT_FLOAT_EQ(tree_->get_visit_count(root_index_), expected_root_visits);
    EXPECT_FLOAT_EQ(tree_->get_total_value(root_index_), expected_root_value);
}

TEST_F(BackupThreadSafetyTest, StressTestWithRandomValues) {
    std::vector<std::thread> threads;
    std::atomic<bool> stop_flag(false);

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([this, &stop_flag]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> value_dist(-1.0, 1.0);

            while (!stop_flag.load()) {
                // Random backup value
                float value = static_cast<float>(value_dist(gen));

                // Sometimes backup to root, sometimes to child
                std::vector<NodeIndex> path;
                if (gen() % 2 == 0) {
                    path = {root_index_};
                } else {
                    path = {child_index_, root_index_};
                }

                manager_->backup_value_along_path(path, value);
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }

    // Run stress test for a short duration
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop_flag.store(true);

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify no corruption - visit counts should be non-negative and reasonable
    EXPECT_GE(tree_->get_visit_count(root_index_), 0.0f);
    EXPECT_GE(tree_->get_visit_count(child_index_), 0.0f);
    EXPECT_LT(tree_->get_visit_count(root_index_), 1000000.0f);  // Sanity check

    // Total value should be finite
    EXPECT_TRUE(std::isfinite(tree_->get_total_value(root_index_)));
    EXPECT_TRUE(std::isfinite(tree_->get_total_value(child_index_)));
}

}  // namespace test
}  // namespace mcts

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}