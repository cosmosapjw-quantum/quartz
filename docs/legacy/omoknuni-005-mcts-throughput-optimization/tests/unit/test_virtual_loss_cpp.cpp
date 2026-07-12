/**
 * @file test_virtual_loss_cpp.cpp
 * @brief C++ unit tests for virtual loss mechanism
 *
 * Tests the actual C++ implementation of virtual loss using Google Test.
 * These tests verify thread safety, atomic operations, and integration
 * with the MCTS tree structure.
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <chrono>
#include <random>
#include <future>

#include "../../cpp_extensions/mcts/virtual_loss.hpp"
#include "../../cpp_extensions/mcts/tree.hpp"

namespace mcts {
namespace test {

class VirtualLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        tree_ = std::make_unique<MCTSTree>(1000);
        root_index_ = tree_->add_root_node(0.5f, 0);

        // Create a small tree for testing
        // Root has 3 children, first child has 2 children
        NodeIndex children = tree_->allocate_nodes(3);
        if (children != NULL_NODE_INDEX) {
            tree_->set_first_child_index(root_index_, children);
            tree_->set_num_children(root_index_, 3);

            for (uint16_t i = 0; i < 3; ++i) {
                NodeIndex child = children + i;
                tree_->set_parent_index(child, root_index_);
                tree_->set_prior_prob(child, 0.33f);
                tree_->set_visit_count(child, 1.0f);
                tree_->set_total_value(child, 0.0f);
            }

            // Add grandchildren to first child
            NodeIndex grandchildren = tree_->allocate_nodes(2);
            if (grandchildren != NULL_NODE_INDEX) {
                tree_->set_first_child_index(children, grandchildren);
                tree_->set_num_children(children, 2);

                for (uint16_t i = 0; i < 2; ++i) {
                    NodeIndex grandchild = grandchildren + i;
                    tree_->set_parent_index(grandchild, children);
                    tree_->set_prior_prob(grandchild, 0.5f);
                    tree_->set_visit_count(grandchild, 0.0f);
                    tree_->set_total_value(grandchild, 0.0f);
                }
            }
        }

        config_ = VirtualLossConfig(1.0f, true);
        manager_ = std::make_unique<VirtualLossManager>(*tree_, config_);
    }

    void TearDown() override {
        manager_.reset();
        tree_.reset();
    }

    std::unique_ptr<MCTSTree> tree_;
    std::unique_ptr<VirtualLossManager> manager_;
    VirtualLossConfig config_;
    NodeIndex root_index_;
};

TEST_F(VirtualLossTest, BasicApplyAndRemove) {
    // Test basic virtual loss application
    EXPECT_TRUE(manager_->apply_virtual_loss(root_index_));
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(root_index_), 1.0f);

    // Test virtual loss removal
    EXPECT_TRUE(manager_->remove_virtual_loss(root_index_));
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(root_index_), 0.0f);
}

TEST_F(VirtualLossTest, CustomMagnitude) {
    float custom_magnitude = 2.5f;

    EXPECT_TRUE(manager_->apply_virtual_loss(root_index_, custom_magnitude));
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(root_index_), custom_magnitude);

    EXPECT_TRUE(manager_->remove_virtual_loss(root_index_, custom_magnitude));
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(root_index_), 0.0f);
}

TEST_F(VirtualLossTest, VirtualLossCannotGoNegative) {
    // Try to remove virtual loss from node with zero virtual loss
    EXPECT_TRUE(manager_->remove_virtual_loss(root_index_));
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(root_index_), 0.0f);

    // Apply some virtual loss, then try to remove more than was applied
    EXPECT_TRUE(manager_->apply_virtual_loss(root_index_, 1.0f));
    EXPECT_TRUE(manager_->remove_virtual_loss(root_index_, 2.0f));
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(root_index_), 0.0f);
}

TEST_F(VirtualLossTest, InvalidNodeIndex) {
    NodeIndex invalid_index = 999;

    EXPECT_FALSE(manager_->apply_virtual_loss(invalid_index));
    EXPECT_FALSE(manager_->remove_virtual_loss(invalid_index));
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(invalid_index), 0.0f);
}

TEST_F(VirtualLossTest, PathOperations) {
    // Create a path from leaf to root
    std::vector<NodeIndex> path = {5, 3, 0};  // grandchild -> child -> root

    // Apply virtual loss to path
    EXPECT_TRUE(manager_->apply_virtual_loss_to_path(path));

    // Check that virtual loss was applied to all nodes in path
    for (NodeIndex node : path) {
        EXPECT_FLOAT_EQ(manager_->get_virtual_loss(node), 1.0f);
    }

    // Remove virtual loss from path
    EXPECT_TRUE(manager_->remove_virtual_loss_from_path(path));

    // Check that virtual loss was removed from all nodes in path
    for (NodeIndex node : path) {
        EXPECT_FLOAT_EQ(manager_->get_virtual_loss(node), 0.0f);
    }
}

TEST_F(VirtualLossTest, EmptyPath) {
    std::vector<NodeIndex> empty_path;

    EXPECT_TRUE(manager_->apply_virtual_loss_to_path(empty_path));
    EXPECT_TRUE(manager_->remove_virtual_loss_from_path(empty_path));
}

TEST_F(VirtualLossTest, PathWithInvalidNode) {
    std::vector<NodeIndex> invalid_path = {0, 999, 3};

    // Should fail due to invalid node
    EXPECT_FALSE(manager_->apply_virtual_loss_to_path(invalid_path));

    // Should have rolled back - no virtual loss applied
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(0), 0.0f);
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(3), 0.0f);
}

TEST_F(VirtualLossTest, VirtualLossGuard) {
    std::vector<NodeIndex> path = {3, 0};

    {
        VirtualLossGuard guard(*manager_, path);
        EXPECT_TRUE(guard.is_valid());

        // Virtual loss should be applied
        for (NodeIndex node : path) {
            EXPECT_FLOAT_EQ(manager_->get_virtual_loss(node), 1.0f);
        }
    }  // Guard destructor should remove virtual loss

    // Virtual loss should be removed
    for (NodeIndex node : path) {
        EXPECT_FLOAT_EQ(manager_->get_virtual_loss(node), 0.0f);
    }
}

TEST_F(VirtualLossTest, VirtualLossGuardManualRelease) {
    std::vector<NodeIndex> path = {3, 0};

    VirtualLossGuard guard(*manager_, path);
    EXPECT_TRUE(guard.is_valid());

    // Virtual loss should be applied
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(3), 1.0f);
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(0), 1.0f);

    // Manually release
    guard.release();

    // Virtual loss should be removed
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(3), 0.0f);
    EXPECT_FLOAT_EQ(manager_->get_virtual_loss(0), 0.0f);
}

TEST_F(VirtualLossTest, DisabledVirtualLoss) {
    VirtualLossConfig disabled_config(1.0f, false);
    VirtualLossManager disabled_manager(*tree_, disabled_config);

    EXPECT_TRUE(disabled_manager.apply_virtual_loss(root_index_));
    EXPECT_FLOAT_EQ(disabled_manager.get_virtual_loss(root_index_), 0.0f);

    std::vector<NodeIndex> path = {3, 0};
    EXPECT_TRUE(disabled_manager.apply_virtual_loss_to_path(path));

    for (NodeIndex node : path) {
        EXPECT_FLOAT_EQ(disabled_manager.get_virtual_loss(node), 0.0f);
    }
}

TEST_F(VirtualLossTest, Statistics) {
    auto initial_stats = manager_->get_statistics();
    EXPECT_EQ(initial_stats.total_applications, 0);
    EXPECT_EQ(initial_stats.total_removals, 0);
    EXPECT_EQ(initial_stats.current_active_paths, 0);

    // Apply virtual loss to multiple nodes
    manager_->apply_virtual_loss(0);
    manager_->apply_virtual_loss(3);
    manager_->apply_virtual_loss(4);

    auto mid_stats = manager_->get_statistics();
    EXPECT_EQ(mid_stats.total_applications, 3);
    EXPECT_EQ(mid_stats.total_removals, 0);
    EXPECT_EQ(mid_stats.current_active_paths, 3);
    EXPECT_FLOAT_EQ(mid_stats.max_virtual_loss, 1.0f);
    EXPECT_FLOAT_EQ(mid_stats.avg_virtual_loss, 1.0f);

    // Remove from some nodes
    manager_->remove_virtual_loss(0);
    manager_->remove_virtual_loss(3);

    auto final_stats = manager_->get_statistics();
    EXPECT_EQ(final_stats.total_applications, 3);
    EXPECT_EQ(final_stats.total_removals, 2);
    EXPECT_EQ(final_stats.current_active_paths, 1);
}

TEST_F(VirtualLossTest, ResetAllVirtualLoss) {
    // Apply virtual loss to multiple nodes
    manager_->apply_virtual_loss(0);
    manager_->apply_virtual_loss(3);
    manager_->apply_virtual_loss(4);

    // Reset all
    manager_->reset_all_virtual_loss();

    // Check all virtual loss is cleared
    for (std::size_t i = 0; i < tree_->get_node_count(); ++i) {
        EXPECT_FLOAT_EQ(manager_->get_virtual_loss(static_cast<NodeIndex>(i)), 0.0f);
    }

    auto stats = manager_->get_statistics();
    EXPECT_EQ(stats.total_applications, 0);
    EXPECT_EQ(stats.total_removals, 0);
}

class VirtualLossThreadSafetyTest : public VirtualLossTest {
protected:
    static constexpr int NUM_THREADS = 10;
    static constexpr int OPERATIONS_PER_THREAD = 1000;
};

TEST_F(VirtualLossThreadSafetyTest, ConcurrentApplyRemove) {
    NodeIndex test_node = root_index_;
    std::vector<std::thread> threads;

    // Half the threads apply virtual loss, half remove it
    for (int i = 0; i < NUM_THREADS; ++i) {
        if (i % 2 == 0) {
            threads.emplace_back([this, test_node]() {
                for (int j = 0; j < OPERATIONS_PER_THREAD; ++j) {
                    manager_->apply_virtual_loss(test_node);
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            });
        } else {
            threads.emplace_back([this, test_node]() {
                for (int j = 0; j < OPERATIONS_PER_THREAD; ++j) {
                    manager_->remove_virtual_loss(test_node);
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            });
        }
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Virtual loss should be non-negative
    float final_vl = manager_->get_virtual_loss(test_node);
    EXPECT_GE(final_vl, 0.0f);

    // Statistics should match expected operations
    auto stats = manager_->get_statistics();
    EXPECT_EQ(stats.total_applications, (NUM_THREADS / 2) * OPERATIONS_PER_THREAD);
    EXPECT_EQ(stats.total_removals, (NUM_THREADS / 2) * OPERATIONS_PER_THREAD);
}

TEST_F(VirtualLossThreadSafetyTest, ConcurrentPathOperations) {
    std::vector<std::vector<NodeIndex>> paths = {
        {3, 0},
        {4, 0},
        {5, 3, 0},
        {6, 3, 0}
    };

    std::vector<std::future<void>> futures;

    for (const auto& path : paths) {
        futures.push_back(std::async(std::launch::async, [this, path]() {
            for (int i = 0; i < 100; ++i) {
                VirtualLossGuard guard(*manager_, path);
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                // Guard destructor will clean up automatically
            }
        }));
    }

    // Wait for all operations to complete
    for (auto& future : futures) {
        future.wait();
    }

    // All virtual loss should be cleaned up
    for (std::size_t i = 0; i < tree_->get_node_count(); ++i) {
        EXPECT_FLOAT_EQ(manager_->get_virtual_loss(static_cast<NodeIndex>(i)), 0.0f);
    }
}

TEST_F(VirtualLossThreadSafetyTest, StressTest) {
    // Create many threads doing random operations
    std::vector<std::thread> threads;
    std::atomic<bool> stop_flag(false);

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([this, &stop_flag]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> node_dist(0, tree_->get_node_count() - 1);
            std::uniform_int_distribution<> op_dist(0, 3);

            while (!stop_flag.load()) {
                NodeIndex node = static_cast<NodeIndex>(node_dist(gen));
                int operation = op_dist(gen);

                switch (operation) {
                    case 0:
                        manager_->apply_virtual_loss(node);
                        break;
                    case 1:
                        manager_->remove_virtual_loss(node);
                        break;
                    case 2: {
                        std::vector<NodeIndex> path = {node, 0};
                        manager_->apply_virtual_loss_to_path(path);
                        break;
                    }
                    case 3: {
                        std::vector<NodeIndex> path = {node, 0};
                        manager_->remove_virtual_loss_from_path(path);
                        break;
                    }
                }

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

    // Verify tree integrity - all virtual loss should be non-negative
    for (std::size_t i = 0; i < tree_->get_node_count(); ++i) {
        float vl = manager_->get_virtual_loss(static_cast<NodeIndex>(i));
        EXPECT_GE(vl, 0.0f);
        EXPECT_LT(vl, 1000.0f);  // Should not exceed safety limit
    }
}

}  // namespace test
}  // namespace mcts

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}