/**
 * @file test_async_queue_batching.cpp
 * @brief Unit tests for AsyncInferenceQueue batch collection
 *
 * Tests batch collection with count and timeout triggers.
 * Compile: g++ -std=c++17 -O2 -pthread -I./cpp_extensions -I./cpp_extensions/utils \
 *          -o test_async_queue_batching tests/unit/test_async_queue_batching.cpp \
 *          cpp_extensions/mcts/async_inference_queue.cpp cpp_extensions/mcts/tree.cpp \
 *          cpp_extensions/mcts/virtual_loss.cpp cpp_extensions/games/gomoku/gomoku_state.cpp \
 *          cpp_extensions/games/gomoku/gomoku_rules.cpp cpp_extensions/utils/igamestate.cpp \
 *          cpp_extensions/utils/zobrist_hash.cpp
 */

#include "mcts/async_inference_queue.hpp"
#include "games/gomoku/gomoku_state.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <thread>

using namespace mcts;
using namespace alphazero::core;
using namespace alphazero::games::gomoku;

// Test 1: Batch collection on size trigger
void test_batch_collection_size_trigger() {
    std::cout << "Test 1: Batch collection on size trigger..." << std::flush;

    AsyncInferenceQueue queue;

    // Submit 32 requests
    for (int i = 0; i < 32; ++i) {
        auto state = std::make_unique<GomokuState>();
        std::vector<NodeIndex> path = {static_cast<NodeIndex>(i)};
        queue.submit_request(std::move(state), i, path);
    }

    // Collect batch (min_size=32, timeout=1000ms)
    auto batch = queue.collect_batch(32, 1000.0);

    // Should return immediately with 32 requests
    assert(batch.size() == 32);
    assert(queue.pending_count() == 0);

    std::cout << " PASS" << std::endl;
}

// Test 2: Batch collection on timeout trigger
void test_batch_collection_timeout_trigger() {
    std::cout << "Test 2: Batch collection on timeout trigger..." << std::flush;

    AsyncInferenceQueue queue;

    // Submit only 10 requests (less than min_size=32)
    for (int i = 0; i < 10; ++i) {
        auto state = std::make_unique<GomokuState>();
        std::vector<NodeIndex> path = {static_cast<NodeIndex>(i)};
        queue.submit_request(std::move(state), i, path);
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Collect batch with short timeout (min_size=32, timeout=10ms)
    auto batch = queue.collect_batch(32, 10.0);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);

    // Should return after timeout with all available requests
    assert(batch.size() == 10);
    assert(queue.pending_count() == 0);

    // Duration should be close to timeout (allow wider tolerance for scheduling)
    assert(duration.count() >= 8.0 && duration.count() <= 40.0);

    std::cout << " PASS (timeout: " << duration.count() << "ms)" << std::endl;
}

// Test 3: Empty batch on timeout with no requests
void test_empty_batch_on_timeout() {
    std::cout << "Test 3: Empty batch on timeout with no requests..." << std::flush;

    AsyncInferenceQueue queue;

    // Don't submit any requests

    auto start = std::chrono::high_resolution_clock::now();

    // Collect batch with short timeout
    auto batch = queue.collect_batch(32, 10.0);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);

    // Should return empty batch after timeout
    assert(batch.empty());
    assert(duration.count() >= 8.0 && duration.count() <= 40.0);

    std::cout << " PASS" << std::endl;
}

// Test 4: Partial batch on timeout
void test_partial_batch_on_timeout() {
    std::cout << "Test 4: Partial batch on timeout..." << std::flush;

    AsyncInferenceQueue queue;

    // Submit 20 requests (less than min_size=32)
    for (int i = 0; i < 20; ++i) {
        auto state = std::make_unique<GomokuState>();
        std::vector<NodeIndex> path = {static_cast<NodeIndex>(i)};
        queue.submit_request(std::move(state), i, path);
    }

    // Collect batch
    auto batch = queue.collect_batch(32, 10.0);

    // Should return 20 requests after timeout
    assert(batch.size() == 20);
    assert(queue.pending_count() == 0);

    std::cout << " PASS" << std::endl;
}

// Test 5: Large batch (>min_size)
void test_large_batch() {
    std::cout << "Test 5: Large batch (>min_size)..." << std::flush;

    AsyncInferenceQueue queue;

    // Submit 100 requests (much more than min_size=32)
    for (int i = 0; i < 100; ++i) {
        auto state = std::make_unique<GomokuState>();
        std::vector<NodeIndex> path = {static_cast<NodeIndex>(i)};
        queue.submit_request(std::move(state), i, path);
    }

    // Collect batch
    auto batch = queue.collect_batch(32, 1000.0);

    const size_t expected_batch = 32 + (32 / 2);  // capped at 1.5× min batch size
    assert(batch.size() == expected_batch);
    assert(queue.pending_count() == 0);

    std::cout << " PASS" << std::endl;
}

// Test 6: Multiple consecutive batches
void test_multiple_batches() {
    std::cout << "Test 6: Multiple consecutive batches..." << std::flush;

    AsyncInferenceQueue queue;

    // First batch
    for (int i = 0; i < 32; ++i) {
        auto state = std::make_unique<GomokuState>();
        std::vector<NodeIndex> path = {static_cast<NodeIndex>(i)};
        queue.submit_request(std::move(state), i, path);
    }
    auto batch1 = queue.collect_batch(32, 1000.0);
    const size_t expected_batch = 32 + (32 / 2);
    assert(batch1.size() == expected_batch);

    // Second batch
    for (int i = 0; i < 32; ++i) {
        auto state = std::make_unique<GomokuState>();
        std::vector<NodeIndex> path = {static_cast<NodeIndex>(i + 100)};
        queue.submit_request(std::move(state), i + 100, path);
    }
    auto batch2 = queue.collect_batch(32, 1000.0);
    assert(batch2.size() == expected_batch);

    // Verify different request IDs
    assert(batch1[0].request_id == 0);
    assert(batch2[0].request_id == expected_batch);

    std::cout << " PASS" << std::endl;
}

int main() {
    std::cout << "=== AsyncInferenceQueue Batching Tests ===" << std::endl;

    test_batch_collection_size_trigger();
    test_batch_collection_timeout_trigger();
    test_empty_batch_on_timeout();
    test_partial_batch_on_timeout();
    test_large_batch();
    test_multiple_batches();

    std::cout << "\nAll 6/6 tests passed!" << std::endl;
    return 0;
}
