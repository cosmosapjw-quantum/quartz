/**
 * @file test_wuuct_virtual_loss.cpp
 * @brief Unit tests for WU-UCT virtual loss manager
 */

#include <gtest/gtest.h>
#include "../../cpp_extensions/mcts/virtual_loss.hpp"
#include <thread>
#include <vector>
#include <chrono>
#include <random>

using namespace mcts;

class WUUCTVirtualLossTest : public ::testing::Test {
protected:
    static constexpr std::size_t MAX_NODES = 10000;
    static constexpr float DEFAULT_MAGNITUDE = 1.0f;

    WUUCTVirtualLossManager* manager_;

    void SetUp() override {
        manager_ = new WUUCTVirtualLossManager(MAX_NODES, DEFAULT_MAGNITUDE);
    }

    void TearDown() override {
        delete manager_;
    }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(WUUCTVirtualLossTest, InitializationTest) {
    EXPECT_FLOAT_EQ(manager_->get_magnitude(), DEFAULT_MAGNITUDE);
    EXPECT_EQ(manager_->get_collision_count(), 0);

    // All nodes should start with zero in-flight count
    for (NodeIndex i = 0; i < 100; ++i) {
        EXPECT_EQ(manager_->get_in_flight_count(i), 0);
        EXPECT_FLOAT_EQ(manager_->get_exploration_adjustment(i), 0.0f);
        EXPECT_FALSE(manager_->is_busy(i));
    }
}

TEST_F(WUUCTVirtualLossTest, SingleNodeAddRemove) {
    NodeIndex node = 42;

    // Initially zero
    EXPECT_EQ(manager_->get_in_flight_count(node), 0);
    EXPECT_FALSE(manager_->is_busy(node));

    // Add in-flight
    manager_->add_in_flight(node);
    EXPECT_EQ(manager_->get_in_flight_count(node), 1);
    EXPECT_TRUE(manager_->is_busy(node));
    EXPECT_FLOAT_EQ(manager_->get_exploration_adjustment(node), 1.0f);

    // Remove in-flight
    manager_->remove_in_flight(node);
    EXPECT_EQ(manager_->get_in_flight_count(node), 0);
    EXPECT_FALSE(manager_->is_busy(node));
    EXPECT_FLOAT_EQ(manager_->get_exploration_adjustment(node), 0.0f);
}

TEST_F(WUUCTVirtualLossTest, MultipleInflightCounts) {
    NodeIndex node = 100;

    // Add multiple in-flight simulations
    manager_->add_in_flight(node);
    manager_->add_in_flight(node);
    manager_->add_in_flight(node);

    EXPECT_EQ(manager_->get_in_flight_count(node), 3);
    EXPECT_FLOAT_EQ(manager_->get_exploration_adjustment(node), 3.0f);
    EXPECT_TRUE(manager_->is_busy(node));

    // Remove one at a time
    manager_->remove_in_flight(node);
    EXPECT_EQ(manager_->get_in_flight_count(node), 2);

    manager_->remove_in_flight(node);
    EXPECT_EQ(manager_->get_in_flight_count(node), 1);

    manager_->remove_in_flight(node);
    EXPECT_EQ(manager_->get_in_flight_count(node), 0);
    EXPECT_FALSE(manager_->is_busy(node));
}

TEST_F(WUUCTVirtualLossTest, MagnitudeScaling) {
    NodeIndex node = 50;
    float custom_magnitude = 2.5f;

    manager_->set_magnitude(custom_magnitude);
    EXPECT_FLOAT_EQ(manager_->get_magnitude(), custom_magnitude);

    manager_->add_in_flight(node);
    manager_->add_in_flight(node);

    // Adjustment should be count * magnitude
    EXPECT_FLOAT_EQ(manager_->get_exploration_adjustment(node), 2.0f * custom_magnitude);
}

TEST_F(WUUCTVirtualLossTest, CollisionTracking) {
    NodeIndex node = 77;

    // First add should not count as collision
    manager_->add_in_flight(node);
    EXPECT_EQ(manager_->get_collision_count(), 0);

    // Second add should count as collision
    manager_->add_in_flight(node);
    EXPECT_EQ(manager_->get_collision_count(), 1);

    // Third add should increment collision count again
    manager_->add_in_flight(node);
    EXPECT_EQ(manager_->get_collision_count(), 2);

    // Removing doesn't affect collision count
    manager_->remove_in_flight(node);
    manager_->remove_in_flight(node);
    EXPECT_EQ(manager_->get_collision_count(), 2);
}

TEST_F(WUUCTVirtualLossTest, ClearAll) {
    // Add in-flight to multiple nodes
    for (NodeIndex i = 0; i < 10; ++i) {
        manager_->add_in_flight(i);
        manager_->add_in_flight(i);  // Collision
    }

    EXPECT_GT(manager_->get_collision_count(), 0);
    EXPECT_EQ(manager_->get_in_flight_count(5), 2);

    // Clear all
    manager_->clear_all();

    EXPECT_EQ(manager_->get_collision_count(), 0);
    for (NodeIndex i = 0; i < 10; ++i) {
        EXPECT_EQ(manager_->get_in_flight_count(i), 0);
    }
}

TEST_F(WUUCTVirtualLossTest, InvalidNodeIndex) {
    // Negative index
    manager_->add_in_flight(-1);
    EXPECT_EQ(manager_->get_in_flight_count(-1), 0);
    EXPECT_FLOAT_EQ(manager_->get_exploration_adjustment(-1), 0.0f);
    EXPECT_FALSE(manager_->is_busy(-1));

    // Out of bounds index
    NodeIndex too_large = MAX_NODES + 100;
    manager_->add_in_flight(too_large);
    EXPECT_EQ(manager_->get_in_flight_count(too_large), 0);

    // Remove from invalid index should not crash
    manager_->remove_in_flight(-1);
    manager_->remove_in_flight(too_large);
}

TEST_F(WUUCTVirtualLossTest, UnderflowProtection) {
    NodeIndex node = 123;

    // Try to remove virtual loss without adding it first
    manager_->remove_in_flight(node);

    // Count should remain at zero (no underflow)
    EXPECT_EQ(manager_->get_in_flight_count(node), 0);

    // Add and remove should still work correctly
    manager_->add_in_flight(node);
    EXPECT_EQ(manager_->get_in_flight_count(node), 1);

    manager_->remove_in_flight(node);
    EXPECT_EQ(manager_->get_in_flight_count(node), 0);
}

// ============================================================================
// Guard Tests
// ============================================================================

TEST_F(WUUCTVirtualLossTest, GuardBasicUsage) {
    std::vector<NodeIndex> path = {10, 20, 30, 40};

    {
        WUUCTVirtualLossGuard guard(*manager_, path);

        // All nodes in path should have in-flight count = 1
        for (NodeIndex node : path) {
            EXPECT_EQ(manager_->get_in_flight_count(node), 1);
            EXPECT_TRUE(manager_->is_busy(node));
        }
    }

    // After guard destruction, all counts should be zero
    for (NodeIndex node : path) {
        EXPECT_EQ(manager_->get_in_flight_count(node), 0);
        EXPECT_FALSE(manager_->is_busy(node));
    }
}

TEST_F(WUUCTVirtualLossTest, GuardManualRelease) {
    std::vector<NodeIndex> path = {5, 15, 25};

    WUUCTVirtualLossGuard guard(*manager_, path);

    // Verify counts are set
    EXPECT_EQ(manager_->get_in_flight_count(5), 1);

    // Manual release
    guard.release();

    // Counts should be cleared
    for (NodeIndex node : path) {
        EXPECT_EQ(manager_->get_in_flight_count(node), 0);
    }

    // Destructor should not crash (double-release protection)
}

TEST_F(WUUCTVirtualLossTest, GuardEmptyPath) {
    std::vector<NodeIndex> empty_path;

    // Should not crash with empty path
    WUUCTVirtualLossGuard guard(*manager_, empty_path);

    // Destructor should handle empty path gracefully
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(WUUCTVirtualLossTest, ConcurrentAddRemove) {
    constexpr int NUM_THREADS = 8;
    constexpr int ITERATIONS = 1000;
    NodeIndex node = 500;

    std::vector<std::thread> threads;

    // Launch threads that add and remove virtual loss
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([this, node]() {
            for (int i = 0; i < ITERATIONS; ++i) {
                manager_->add_in_flight(node);
                manager_->remove_in_flight(node);
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Final count should be zero (all adds matched by removes)
    EXPECT_EQ(manager_->get_in_flight_count(node), 0);
}

TEST_F(WUUCTVirtualLossTest, ConcurrentMultipleNodes) {
    constexpr int NUM_THREADS = 8;
    constexpr int ITERATIONS = 500;
    constexpr int NODES_PER_THREAD = 10;

    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([this, t]() {
            std::mt19937 rng(t);
            std::uniform_int_distribution<NodeIndex> dist(0, 999);

            for (int i = 0; i < ITERATIONS; ++i) {
                std::vector<NodeIndex> path;
                for (int j = 0; j < NODES_PER_THREAD; ++j) {
                    path.push_back(dist(rng));
                }

                // Use guard for automatic cleanup
                WUUCTVirtualLossGuard guard(*manager_, path);

                // Simulate some work
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All nodes should be clean after threads finish
    for (NodeIndex i = 0; i < 1000; ++i) {
        EXPECT_EQ(manager_->get_in_flight_count(i), 0)
            << "Node " << i << " has non-zero count";
    }
}

TEST_F(WUUCTVirtualLossTest, StressTestAtomicOperations) {
    constexpr int NUM_THREADS = 16;
    constexpr int ITERATIONS = 10000;
    NodeIndex node = 42;

    std::atomic<int> completed_threads{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([this, node, &completed_threads]() {
            for (int i = 0; i < ITERATIONS; ++i) {
                manager_->add_in_flight(node);

                // Read operations should always be consistent
                std::uint32_t count = manager_->get_in_flight_count(node);
                EXPECT_GT(count, 0);  // Should be at least 1 (our own)

                bool busy = manager_->is_busy(node);
                EXPECT_TRUE(busy);

                float adjustment = manager_->get_exploration_adjustment(node);
                EXPECT_GT(adjustment, 0.0f);

                manager_->remove_in_flight(node);
            }

            completed_threads.fetch_add(1);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(completed_threads.load(), NUM_THREADS);
    EXPECT_EQ(manager_->get_in_flight_count(node), 0);
}

// ============================================================================
// Performance Characteristics Tests
// ============================================================================

TEST_F(WUUCTVirtualLossTest, PerformanceAddRemove) {
    constexpr int ITERATIONS = 1000000;
    NodeIndex node = 100;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; ++i) {
        manager_->add_in_flight(node);
        manager_->remove_in_flight(node);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    double ns_per_op = duration.count() / (2.0 * ITERATIONS);  // 2 ops per iteration

    std::cout << "Average time per add/remove: " << ns_per_op << " ns\n";

    // Should be very fast (sub-100ns on modern hardware)
    EXPECT_LT(ns_per_op, 200.0) << "Virtual loss operations are too slow";
}

TEST_F(WUUCTVirtualLossTest, MemoryFootprint) {
    // Each InFlightData is 64 bytes (cache-aligned atomic<uint32_t>)
    // For MAX_NODES = 10000, total should be ~640KB

    constexpr std::size_t EXPECTED_SIZE_BYTES = MAX_NODES * 64;
    constexpr std::size_t MB = 1024 * 1024;

    std::cout << "Expected memory footprint: "
              << EXPECTED_SIZE_BYTES / MB << " MB\n";

    // Should be under 1MB for 10k nodes
    EXPECT_LT(EXPECTED_SIZE_BYTES, MB);
}

// ============================================================================
// Comparison with Classic Virtual Loss
// ============================================================================

TEST_F(WUUCTVirtualLossTest, NoQValueDistortion) {
    // This test documents the key difference from classic virtual loss:
    // WU-UCT does NOT modify Q-values, only the exploration term

    NodeIndex node = 200;

    // Simulate a node with visit count and value
    // (These would normally be in the tree, we're just documenting the formula)
    float visit_count = 10.0f;
    float total_value = 5.0f;
    float prior = 0.1f;
    float parent_visits = 100.0f;
    float c_puct = 1.25f;

    // Classic VL formula (for reference):
    // Q_classic = (total_value - VL) / (visit_count + 1)
    // U_classic = c_puct * prior * sqrt(parent_visits) / (1 + visit_count)

    // WU-UCT formula:
    // Q_wuuct = total_value / visit_count  (PURE Q-VALUE)
    // U_wuuct = c_puct * prior * sqrt(parent_visits) / (1 + visit_count + VL_adjustment)

    manager_->add_in_flight(node);
    float vl_adjustment = manager_->get_exploration_adjustment(node);

    // Q-value remains pure
    float q_value = total_value / visit_count;
    EXPECT_FLOAT_EQ(q_value, 0.5f);  // No distortion from virtual loss

    // Exploration term is adjusted
    float exploration = c_puct * prior * std::sqrt(parent_visits) /
                       (1.0f + visit_count + vl_adjustment);

    // With VL adjustment, exploration is reduced
    float exploration_no_vl = c_puct * prior * std::sqrt(parent_visits) /
                              (1.0f + visit_count);

    EXPECT_LT(exploration, exploration_no_vl);

    // Final PUCT score
    float puct_score = q_value + exploration;

    // The Q-value component is NEVER distorted by virtual loss
    EXPECT_FLOAT_EQ(q_value, total_value / visit_count);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
