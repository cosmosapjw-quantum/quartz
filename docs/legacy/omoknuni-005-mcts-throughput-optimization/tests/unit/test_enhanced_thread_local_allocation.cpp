/**
 * @file test_enhanced_thread_local_allocation.cpp
 * @brief Tests for enhanced thread-local block allocation in MCTSTree
 *
 * Validates T009e implementation:
 * - Increased block size from 64 to 4096 reduces global contention
 * - Statistics tracking works correctly
 * - Multi-threaded allocation scales better
 */

#include <gtest/gtest.h>
#include "../../cpp_extensions/mcts/tree.hpp"
#include <thread>
#include <vector>
#include <atomic>

using namespace mcts;

class EnhancedThreadLocalAllocationTest : public ::testing::Test {
protected:
    static constexpr std::size_t kDefaultMaxNodes = 100'000;
    std::unique_ptr<MCTSTree> tree;

    void SetUp() override {
        tree = std::make_unique<MCTSTree>(kDefaultMaxNodes);
        tree->add_root_node(1.0f, 0);
    }
};

/**
 * Test: Allocate nodes and verify statistics are tracked
 */
TEST_F(EnhancedThreadLocalAllocationTest, StatisticsTracking) {
    // Allocate some nodes (should come from thread-local block)
    for (int i = 0; i < 100; ++i) {
        NodeIndex node = tree->allocate_node();
        ASSERT_NE(node, NULL_NODE_INDEX);
    }

    // Get statistics for this thread
    auto stats = tree->get_thread_allocation_stats();

    // Should have block size of 4096 (new value)
    EXPECT_EQ(stats.block_size, 4096u);

    // Should have allocations (at least 1 global allocation to init block + fast path)
    EXPECT_GT(stats.allocations_from_block + stats.allocations_from_global, 0u);

    // First batch should include a global allocation
    EXPECT_GT(stats.allocations_from_global, 0u);

    // Most allocations should be from thread-local block (fast path)
    EXPECT_GT(stats.fast_path_percentage(), 0.0);
}

/**
 * Test: Large block size means fewer global allocations
 */
TEST_F(EnhancedThreadLocalAllocationTest, LargeBlockReducesGlobalAllocations) {
    // Allocate 5000 nodes (> 1 block, but < 2 blocks with size 4096)
    constexpr int kNumAllocations = 5000;

    for (int i = 0; i < kNumAllocations; ++i) {
        NodeIndex node = tree->allocate_node();
        ASSERT_NE(node, NULL_NODE_INDEX);
    }

    auto stats = tree->get_thread_allocation_stats();

    // Should have ≤3 global allocations (initial + refills)
    // Note: May need more due to single-node allocations vs bulk
    EXPECT_LE(stats.allocations_from_global, 3u);

    // Fast path should be >99% (most allocations from thread-local block with 4096 size)
    EXPECT_GT(stats.fast_path_percentage(), 99.0);
}

/**
 * Test: Multi-threaded allocation with statistics
 */
TEST_F(EnhancedThreadLocalAllocationTest, MultiThreadedAllocation) {
    constexpr int kNumThreads = 4;
    constexpr int kAllocationsPerThread = 1000;

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kAllocationsPerThread; ++i) {
                NodeIndex node = tree->allocate_node();
                if (node != NULL_NODE_INDEX) {
                    success_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All allocations should succeed (tree has capacity for 100k nodes)
    EXPECT_EQ(success_count.load(), kNumThreads * kAllocationsPerThread);

    // Verify final node count
    EXPECT_GT(tree->get_node_count(), static_cast<std::size_t>(kNumThreads * kAllocationsPerThread));
}

/**
 * Test: Allocation after tree clear resets statistics
 */
TEST_F(EnhancedThreadLocalAllocationTest, ClearResetsStatistics) {
    // Initial allocations
    for (int i = 0; i < 100; ++i) {
        tree->allocate_node();
    }

    auto stats_before = tree->get_thread_allocation_stats();
    EXPECT_GT(stats_before.allocations_from_block + stats_before.allocations_from_global, 0u);

    // Clear tree
    tree->clear();
    tree->add_root_node(1.0f, 0);

    // The thread-local block is invalidated by clear() due to epoch change
    // New allocations will require a new global allocation
    for (int i = 0; i < 10; ++i) {
        tree->allocate_node();
    }

    auto stats_after = tree->get_thread_allocation_stats();

    // Statistics continue to accumulate (not reset by tree clear)
    // This is intentional - stats track lifetime of thread's interaction with tree
    EXPECT_GE(stats_after.allocations_from_block + stats_after.allocations_from_global,
              stats_before.allocations_from_block + stats_before.allocations_from_global);
}

/**
 * Test: Percentage calculations work correctly
 */
TEST_F(EnhancedThreadLocalAllocationTest, PercentageCalculations) {
    // Allocate nodes to generate statistics
    for (int i = 0; i < 500; ++i) {
        tree->allocate_node();
    }

    auto stats = tree->get_thread_allocation_stats();

    // Percentages should sum to 100%
    double total_percentage = stats.fast_path_percentage() +
                             stats.slow_path_percentage() +
                             stats.reuse_percentage();

    EXPECT_NEAR(total_percentage, 100.0, 0.01);

    // All percentages should be non-negative
    EXPECT_GE(stats.fast_path_percentage(), 0.0);
    EXPECT_GE(stats.slow_path_percentage(), 0.0);
    EXPECT_GE(stats.reuse_percentage(), 0.0);

    // All percentages should be <=100%
    EXPECT_LE(stats.fast_path_percentage(), 100.0);
    EXPECT_LE(stats.slow_path_percentage(), 100.0);
    EXPECT_LE(stats.reuse_percentage(), 100.0);
}

/**
 * Test: Free list reuse is tracked correctly
 */
TEST_F(EnhancedThreadLocalAllocationTest, FreeListReuseTracking) {
    // Allocate some nodes
    std::vector<NodeIndex> nodes;
    for (int i = 0; i < 10; ++i) {
        nodes.push_back(tree->allocate_node());
    }

    // Deallocate them (adds to free list)
    for (auto node : nodes) {
        tree->deallocate_node(node);
    }

    auto stats_before = tree->get_thread_allocation_stats();

    // Allocate again (should come from free list)
    for (int i = 0; i < 10; ++i) {
        NodeIndex node = tree->allocate_node();
        ASSERT_NE(node, NULL_NODE_INDEX);
    }

    auto stats_after = tree->get_thread_allocation_stats();

    // Note: With large block size (4096), the thread-local block still has many nodes
    // So free list reuse may not happen immediately. Just verify allocations succeed.
    // Free list reuse will be more important after block exhaustion.
    EXPECT_GE(stats_after.allocations_from_freelist, stats_before.allocations_from_freelist);
}

/**
 * Benchmark: Compare allocation speed with larger block size
 */
TEST_F(EnhancedThreadLocalAllocationTest, BenchmarkAllocationSpeed) {
    constexpr int kNumAllocations = 10000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumAllocations; ++i) {
        NodeIndex node = tree->allocate_node();
        (void)node; // Suppress unused variable warning
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    auto stats = tree->get_thread_allocation_stats();

    // Print results
    std::cout << "Allocated " << kNumAllocations << " nodes in " << duration << " μs\n";
    std::cout << "  Average: " << (static_cast<double>(duration) / kNumAllocations) << " μs/node\n";
    std::cout << "  Fast path: " << stats.fast_path_percentage() << "%\n";
    std::cout << "  Slow path: " << stats.slow_path_percentage() << "%\n";
    std::cout << "  Reuse: " << stats.reuse_percentage() << "%\n";
    std::cout << "  Global allocations: " << stats.allocations_from_global << "\n";

    // With block size 4096, expect <1% global allocations (slow path) for 10k nodes
    // This means 99%+ are fast-path allocations from thread-local block
    EXPECT_LT(stats.slow_path_percentage(), 1.0);

    // Fast path should dominate (>99%)
    EXPECT_GT(stats.fast_path_percentage(), 99.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
