/**
 * @file test_lock_free_queue.cpp
 * @brief Comprehensive unit tests for lock-free MPMC queue
 */

#include <gtest/gtest.h>
#include "../../cpp_extensions/mcts/lock_free_queue.hpp"
#include <thread>
#include <vector>
#include <atomic>
#include <set>
#include <algorithm>
#include <chrono>

using namespace mcts;

/**
 * Test fixture for lock-free queue tests
 */
class LockFreeQueueTest : public ::testing::Test {
protected:
    MPMCRingBuffer<int, 16> small_queue;
    MPMCRingBuffer<int, 1024> large_queue;
};

/**
 * Test basic enqueue operation
 */
TEST_F(LockFreeQueueTest, BasicEnqueue) {
    int value = 42;
    EXPECT_TRUE(small_queue.try_enqueue(std::move(value)));
    EXPECT_FALSE(small_queue.empty_approx());
    EXPECT_EQ(small_queue.size_approx(), 1);
}

/**
 * Test basic dequeue operation
 */
TEST_F(LockFreeQueueTest, BasicDequeue) {
    int val = 42;
    small_queue.try_enqueue(std::move(val));

    int value = 0;
    EXPECT_TRUE(small_queue.try_dequeue(value));
    EXPECT_EQ(value, 42);
    EXPECT_TRUE(small_queue.empty_approx());
}

/**
 * Test dequeue from empty queue
 */
TEST_F(LockFreeQueueTest, DequeueFromEmpty) {
    int value = 0;
    EXPECT_FALSE(small_queue.try_dequeue(value));
}

/**
 * Test queue full condition
 */
TEST_F(LockFreeQueueTest, QueueFull) {
    // Fill queue to capacity (16 slots)
    for (int i = 0; i < 16; ++i) {
        int val = i;
        EXPECT_TRUE(small_queue.try_enqueue(std::move(val))) << "Failed to enqueue at " << i;
    }

    // Next enqueue should fail (queue full)
    int val = 999;
    EXPECT_FALSE(small_queue.try_enqueue(std::move(val)));
}

/**
 * Test FIFO ordering
 */
TEST_F(LockFreeQueueTest, FIFOOrdering) {
    const int N = 10;

    // Enqueue sequence
    for (int i = 0; i < N; ++i) {
        int val = i;
        EXPECT_TRUE(small_queue.try_enqueue(std::move(val)));
    }

    // Dequeue and verify order
    for (int i = 0; i < N; ++i) {
        int value = -1;
        EXPECT_TRUE(small_queue.try_dequeue(value));
        EXPECT_EQ(value, i) << "FIFO violation at position " << i;
    }
}

/**
 * Test enqueue/dequeue interleaving
 */
TEST_F(LockFreeQueueTest, InterleavedOps) {
    for (int i = 0; i < 100; ++i) {
        int val = i;
        EXPECT_TRUE(small_queue.try_enqueue(std::move(val)));

        int value = -1;
        EXPECT_TRUE(small_queue.try_dequeue(value));
        EXPECT_EQ(value, i);
    }

    EXPECT_TRUE(small_queue.empty_approx());
}

/**
 * Test wrap-around behavior
 */
TEST_F(LockFreeQueueTest, WrapAround) {
    // Fill and empty multiple times to test wrap-around
    for (int round = 0; round < 5; ++round) {
        // Fill queue
        for (int i = 0; i < 16; ++i) {
            int val = round * 100 + i;
            EXPECT_TRUE(small_queue.try_enqueue(std::move(val)));
        }

        // Empty queue
        for (int i = 0; i < 16; ++i) {
            int value = -1;
            EXPECT_TRUE(small_queue.try_dequeue(value));
            EXPECT_EQ(value, round * 100 + i);
        }
    }
}

/**
 * Test bulk enqueue operation
 */
TEST_F(LockFreeQueueTest, BulkEnqueue) {
    std::vector<int> items = {1, 2, 3, 4, 5};

    size_t enqueued = small_queue.try_enqueue_bulk(items.data(), items.size());
    EXPECT_EQ(enqueued, 5);
    EXPECT_EQ(small_queue.size_approx(), 5);
}

/**
 * Test bulk dequeue operation
 */
TEST_F(LockFreeQueueTest, BulkDequeue) {
    // Enqueue sequence
    for (int i = 0; i < 10; ++i) {
        int val = i;
        small_queue.try_enqueue(std::move(val));
    }

    // Bulk dequeue
    std::vector<int> items(10);
    size_t dequeued = small_queue.try_dequeue_bulk(items.data(), items.size());

    EXPECT_EQ(dequeued, 10);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(items[i], i);
    }
}

/**
 * Test bulk enqueue with full queue
 */
TEST_F(LockFreeQueueTest, BulkEnqueuePartial) {
    // Fill queue partially
    for (int i = 0; i < 10; ++i) {
        int val = i;
        small_queue.try_enqueue(std::move(val));
    }

    // Try to enqueue 10 more (should only fit 6)
    std::vector<int> items(10, 42);
    size_t enqueued = small_queue.try_enqueue_bulk(items.data(), items.size());

    EXPECT_EQ(enqueued, 6);  // Only 6 fit (16 - 10 = 6)
}

/**
 * Test concurrent single producer/consumer
 */
TEST_F(LockFreeQueueTest, SPSCConcurrent) {
    std::atomic<bool> done{false};
    const int N = 10000;

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < N; ++i) {
            int val = i;
            while (!large_queue.try_enqueue(std::move(val))) {
                val = i;  // Restore value if enqueue failed
                std::this_thread::yield();
            }
        }
        done.store(true);
    });

    // Consumer thread
    std::thread consumer([&]() {
        int expected = 0;
        while (expected < N) {
            int value = -1;
            if (large_queue.try_dequeue(value)) {
                EXPECT_EQ(value, expected);
                expected++;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_TRUE(large_queue.empty_approx());
}

/**
 * Test concurrent multiple producers/single consumer
 */
TEST_F(LockFreeQueueTest, MPSCConcurrent) {
    const int NUM_PRODUCERS = 4;
    const int ITEMS_PER_PRODUCER = 1000;
    const int TOTAL_ITEMS = NUM_PRODUCERS * ITEMS_PER_PRODUCER;

    std::atomic<int> produced{0};
    std::atomic<bool> producers_done{false};

    // Multiple producer threads
    std::vector<std::thread> producers;
    for (int p = 0; p < NUM_PRODUCERS; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < ITEMS_PER_PRODUCER; ++i) {
                int value = p * 10000 + i;
                while (!large_queue.try_enqueue(std::move(value))) {
                    value = p * 10000 + i;  // Restore value if enqueue failed
                    std::this_thread::yield();
                }
                produced.fetch_add(1);
            }
        });
    }

    // Single consumer thread
    std::set<int> consumed;
    std::thread consumer([&]() {
        while (consumed.size() < TOTAL_ITEMS) {
            int value = -1;
            if (large_queue.try_dequeue(value)) {
                consumed.insert(value);
            } else {
                std::this_thread::yield();
            }
        }
    });

    for (auto& t : producers) {
        t.join();
    }
    producers_done.store(true);
    consumer.join();

    // Verify all items consumed exactly once
    EXPECT_EQ(consumed.size(), TOTAL_ITEMS);
    EXPECT_TRUE(large_queue.empty_approx());
}

/**
 * Test concurrent single producer/multiple consumers
 */
TEST_F(LockFreeQueueTest, SPMCConcurrent) {
    const int NUM_CONSUMERS = 4;
    const int TOTAL_ITEMS = 4000;

    std::atomic<int> consumed{0};

    // Single producer thread
    std::thread producer([&]() {
        for (int i = 0; i < TOTAL_ITEMS; ++i) {
            int val = i;
            while (!large_queue.try_enqueue(std::move(val))) {
                val = i;  // Restore value if enqueue failed
                std::this_thread::yield();
            }
        }
    });

    // Multiple consumer threads
    std::vector<std::set<int>> consumer_values(NUM_CONSUMERS);
    std::vector<std::thread> consumers;

    for (int c = 0; c < NUM_CONSUMERS; ++c) {
        consumers.emplace_back([&, c]() {
            while (consumed.load() < TOTAL_ITEMS) {
                int value = -1;
                if (large_queue.try_dequeue(value)) {
                    consumer_values[c].insert(value);
                    consumed.fetch_add(1);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    producer.join();
    for (auto& t : consumers) {
        t.join();
    }

    // Verify all items consumed exactly once (union of all sets)
    std::set<int> all_consumed;
    for (const auto& values : consumer_values) {
        all_consumed.insert(values.begin(), values.end());
    }

    EXPECT_EQ(all_consumed.size(), TOTAL_ITEMS);
    EXPECT_TRUE(large_queue.empty_approx());
}

/**
 * Test concurrent multiple producers/multiple consumers (MPMC)
 */
TEST_F(LockFreeQueueTest, MPMCConcurrent) {
    const int NUM_PRODUCERS = 4;
    const int NUM_CONSUMERS = 4;
    const int ITEMS_PER_PRODUCER = 500;
    const int TOTAL_ITEMS = NUM_PRODUCERS * ITEMS_PER_PRODUCER;

    std::atomic<int> consumed{0};

    // Multiple producer threads
    std::vector<std::thread> producers;
    for (int p = 0; p < NUM_PRODUCERS; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < ITEMS_PER_PRODUCER; ++i) {
                int value = p * 10000 + i;
                while (!large_queue.try_enqueue(std::move(value))) {
                    value = p * 10000 + i;  // Restore value if enqueue failed
                    std::this_thread::yield();
                }
            }
        });
    }

    // Multiple consumer threads
    std::vector<std::set<int>> consumer_values(NUM_CONSUMERS);
    std::vector<std::thread> consumers;

    for (int c = 0; c < NUM_CONSUMERS; ++c) {
        consumers.emplace_back([&, c]() {
            while (consumed.load() < TOTAL_ITEMS) {
                int value = -1;
                if (large_queue.try_dequeue(value)) {
                    consumer_values[c].insert(value);
                    consumed.fetch_add(1);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    for (auto& t : producers) {
        t.join();
    }
    for (auto& t : consumers) {
        t.join();
    }

    // Verify all items consumed exactly once
    std::set<int> all_consumed;
    for (const auto& values : consumer_values) {
        all_consumed.insert(values.begin(), values.end());
    }

    EXPECT_EQ(all_consumed.size(), TOTAL_ITEMS);
    EXPECT_TRUE(large_queue.empty_approx());
}

/**
 * Test high contention scenario
 */
TEST_F(LockFreeQueueTest, HighContention) {
    const int NUM_THREADS = 8;
    const int OPS_PER_THREAD = 1000;

    std::vector<std::thread> threads;
    std::atomic<int> successful_ops{0};

    // Half producers, half consumers
    for (int t = 0; t < NUM_THREADS; ++t) {
        if (t < NUM_THREADS / 2) {
            // Producer
            threads.emplace_back([&, t]() {
                for (int i = 0; i < OPS_PER_THREAD; ++i) {
                    int val = t * 10000 + i;
                    if (large_queue.try_enqueue(std::move(val))) {
                        successful_ops.fetch_add(1);
                    }
                }
            });
        } else {
            // Consumer
            threads.emplace_back([&]() {
                int value = -1;
                for (int i = 0; i < OPS_PER_THREAD; ++i) {
                    if (large_queue.try_dequeue(value)) {
                        successful_ops.fetch_add(1);
                    }
                }
            });
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    // Just verify no crashes and reasonable operation count
    EXPECT_GT(successful_ops.load(), 0);
}

/**
 * Test performance: enqueue throughput
 */
TEST_F(LockFreeQueueTest, EnqueuePerformance) {
    const int N = 10000;  // Reduced for faster testing
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        int val = i;
        large_queue.try_enqueue(std::move(val));

        // Dequeue to keep queue from filling
        if (i % 64 == 0) {
            int value;
            large_queue.try_dequeue(value);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    double ns_per_op = static_cast<double>(duration.count()) / N;

    // Target: <500ns per enqueue (relaxed for CI)
    EXPECT_LT(ns_per_op, 500) << "Enqueue too slow: " << ns_per_op << "ns per op";

    std::cout << "Enqueue performance: " << ns_per_op << " ns/op" << std::endl;
}

/**
 * Test stress: rapid enqueue/dequeue cycles
 */
TEST_F(LockFreeQueueTest, StressCycles) {
    const int CYCLES = 1000;
    const int BATCH_SIZE = 16;

    for (int cycle = 0; cycle < CYCLES; ++cycle) {
        // Enqueue batch
        for (int i = 0; i < BATCH_SIZE; ++i) {
            int val = cycle * 100 + i;
            EXPECT_TRUE(small_queue.try_enqueue(std::move(val)));
        }

        // Dequeue batch
        for (int i = 0; i < BATCH_SIZE; ++i) {
            int value = -1;
            EXPECT_TRUE(small_queue.try_dequeue(value));
            EXPECT_EQ(value, cycle * 100 + i);
        }

        EXPECT_TRUE(small_queue.empty_approx());
    }
}

/**
 * Test with movable-only type
 */
TEST_F(LockFreeQueueTest, MovableOnlyType) {
    struct MovableOnly {
        int value;
        MovableOnly() : value(0) {}
        MovableOnly(int v) : value(v) {}
        MovableOnly(MovableOnly&&) = default;
        MovableOnly& operator=(MovableOnly&&) = default;
        MovableOnly(const MovableOnly&) = delete;
        MovableOnly& operator=(const MovableOnly&) = delete;
    };

    MPMCRingBuffer<MovableOnly, 16> queue;

    EXPECT_TRUE(queue.try_enqueue(MovableOnly(42)));

    MovableOnly item(0);
    EXPECT_TRUE(queue.try_dequeue(item));
    EXPECT_EQ(item.value, 42);
}

/**
 * Test capacity and size
 */
TEST_F(LockFreeQueueTest, CapacityAndSize) {
    EXPECT_EQ(small_queue.capacity(), 16);
    EXPECT_EQ(large_queue.capacity(), 1024);

    EXPECT_EQ(small_queue.size_approx(), 0);

    int v1 = 1, v2 = 2, v3 = 3;
    small_queue.try_enqueue(std::move(v1));
    small_queue.try_enqueue(std::move(v2));
    small_queue.try_enqueue(std::move(v3));

    EXPECT_EQ(small_queue.size_approx(), 3);

    int value;
    small_queue.try_dequeue(value);

    EXPECT_EQ(small_queue.size_approx(), 2);
}
