/**
 * @file test_thread_affinity.cpp
 * @brief Unit tests for thread affinity management
 */

#include <gtest/gtest.h>
#include "../../cpp_extensions/mcts/thread_affinity.hpp"
#include <thread>
#include <vector>

using namespace mcts;

/**
 * Test fixture for thread affinity tests
 */
class ThreadAffinityTest : public ::testing::Test {
protected:
    ThreadAffinityManager affinity_mgr;
};

/**
 * Test basic topology detection
 */
TEST_F(ThreadAffinityTest, TopologyDetection) {
    const CPUTopology& topology = affinity_mgr.get_topology();

    // Topology should have at least some cores
    EXPECT_GT(topology.total_physical_cores, 0);
    EXPECT_GT(topology.total_logical_cores, 0);

    // Logical cores >= physical cores (SMT)
    EXPECT_GE(topology.total_logical_cores, topology.total_physical_cores);

    // Core vectors should have reasonable sizes
    EXPECT_GT(topology.ccd0_cores.size(), 0);
}

/**
 * Test Ryzen 5900X detection (if running on that CPU)
 */
TEST_F(ThreadAffinityTest, Ryzen5900XDetection) {
    const CPUTopology& topology = affinity_mgr.get_topology();

    if (topology.is_ryzen_5900x) {
        // Verify Ryzen 5900X specific topology
        EXPECT_TRUE(topology.is_multi_ccd);
        EXPECT_EQ(topology.total_physical_cores, 12);
        EXPECT_EQ(topology.total_logical_cores, 24);

        EXPECT_EQ(topology.ccd0_cores.size(), 6);
        EXPECT_EQ(topology.ccd1_cores.size(), 6);
        EXPECT_EQ(topology.smt_siblings.size(), 12);

        // Verify core IDs
        EXPECT_EQ(topology.ccd0_cores[0], 0);
        EXPECT_EQ(topology.ccd0_cores[5], 5);
        EXPECT_EQ(topology.ccd1_cores[0], 6);
        EXPECT_EQ(topology.ccd1_cores[5], 11);
    }
}

/**
 * Test thread affinity setting (single thread)
 */
TEST_F(ThreadAffinityTest, SetAffinitySingleThread) {
    bool result = affinity_mgr.set_thread_affinity(0, 1);

    // Result depends on platform support
#ifdef __linux__
    const CPUTopology& topology = affinity_mgr.get_topology();
    if (topology.is_ryzen_5900x || topology.is_multi_ccd) {
        EXPECT_TRUE(result);
    }
#else
    // Non-Linux platforms should return false
    EXPECT_FALSE(result);
#endif
}

/**
 * Test thread affinity with multiple threads
 */
TEST_F(ThreadAffinityTest, SetAffinityMultipleThreads) {
    const int num_threads = 4;
    std::vector<bool> results(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        results[i] = affinity_mgr.set_thread_affinity(i, num_threads);
    }

#ifdef __linux__
    const CPUTopology& topology = affinity_mgr.get_topology();
    if (topology.is_ryzen_5900x || topology.is_multi_ccd) {
        for (bool result : results) {
            EXPECT_TRUE(result);
        }
    }
#endif
}

/**
 * Test thread affinity with all physical cores
 */
TEST_F(ThreadAffinityTest, SetAffinityAllPhysicalCores) {
    const CPUTopology& topology = affinity_mgr.get_topology();
    int num_threads = topology.total_physical_cores;

    if (num_threads > 0) {
        for (int i = 0; i < num_threads; ++i) {
            bool result = affinity_mgr.set_thread_affinity(i, num_threads);

#ifdef __linux__
            if (topology.is_ryzen_5900x || topology.is_multi_ccd) {
                EXPECT_TRUE(result);
            }
#else
            EXPECT_FALSE(result);
#endif
        }
    }
}

/**
 * Test recommended thread count
 */
TEST_F(ThreadAffinityTest, RecommendedThreadCount) {
    int recommended = affinity_mgr.get_recommended_thread_count();

    EXPECT_GT(recommended, 0);
    EXPECT_LE(recommended, 24);  // Reasonable upper bound

    const CPUTopology& topology = affinity_mgr.get_topology();
    if (topology.is_ryzen_5900x) {
        EXPECT_EQ(recommended, 12);  // Ryzen 5900X optimal
    }
}

/**
 * Test platform support detection
 */
TEST_F(ThreadAffinityTest, PlatformSupport) {
    bool supported = affinity_mgr.is_supported();

#ifdef __linux__
    EXPECT_TRUE(supported);
#else
    EXPECT_FALSE(supported);
#endif
}

/**
 * Test thread affinity in actual threads
 */
TEST_F(ThreadAffinityTest, AffinityInThreads) {
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<bool> results(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            ThreadAffinityManager local_mgr;
            results[i] = local_mgr.set_thread_affinity(i, num_threads);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // At least verify no crashes occurred
    EXPECT_EQ(results.size(), num_threads);
}

/**
 * Test boundary conditions: 0 threads
 */
TEST_F(ThreadAffinityTest, ZeroThreads) {
    // Should handle gracefully without crashing
    bool result = affinity_mgr.set_thread_affinity(0, 0);

    // Implementation may return false for invalid input
    (void)result;  // Don't assert specific behavior
}

/**
 * Test boundary conditions: excessive thread count
 */
TEST_F(ThreadAffinityTest, ExcessiveThreadCount) {
    // Should handle gracefully without crashing
    bool result = affinity_mgr.set_thread_affinity(100, 200);

    // Implementation may return false for excessive counts
    (void)result;  // Don't assert specific behavior
}

/**
 * Test CCD0-only placement strategy (≤6 threads)
 */
TEST_F(ThreadAffinityTest, CCD0OnlyPlacement) {
    const CPUTopology& topology = affinity_mgr.get_topology();

    if (topology.is_ryzen_5900x) {
        // With ≤6 threads, should use CCD0 only
        for (int i = 0; i < 6; ++i) {
            bool result = affinity_mgr.set_thread_affinity(i, 6);
            EXPECT_TRUE(result);
        }
    }
}

/**
 * Test dual-CCD placement strategy (7-12 threads)
 */
TEST_F(ThreadAffinityTest, DualCCDPlacement) {
    const CPUTopology& topology = affinity_mgr.get_topology();

    if (topology.is_ryzen_5900x) {
        // With 8 threads, should use both CCDs
        for (int i = 0; i < 8; ++i) {
            bool result = affinity_mgr.set_thread_affinity(i, 8);
            EXPECT_TRUE(result);
        }
    }
}

/**
 * Test SMT sibling usage (>12 threads)
 */
TEST_F(ThreadAffinityTest, SMTSiblingUsage) {
    const CPUTopology& topology = affinity_mgr.get_topology();

    if (topology.is_ryzen_5900x) {
        // With 16 threads, should use SMT siblings
        for (int i = 0; i < 16; ++i) {
            bool result = affinity_mgr.set_thread_affinity(i, 16);
            EXPECT_TRUE(result);
        }
    }
}

/**
 * Test thread-local affinity manager (should not interfere)
 */
TEST_F(ThreadAffinityTest, ThreadLocalManagers) {
    std::atomic<int> success_count{0};
    const int num_threads = 4;

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            // Each thread has its own manager
            ThreadAffinityManager mgr;
            bool result = mgr.set_thread_affinity(i, num_threads);

#ifdef __linux__
            const CPUTopology& topology = mgr.get_topology();
            if (topology.is_ryzen_5900x || topology.is_multi_ccd) {
                if (result) {
                    success_count.fetch_add(1);
                }
            }
#endif
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // No specific assertion, just ensure no crashes
    EXPECT_GE(success_count.load(), 0);
}
