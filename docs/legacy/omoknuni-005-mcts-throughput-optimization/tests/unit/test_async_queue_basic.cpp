/**
 * @file test_async_queue_basic.cpp
 * @brief Unit tests for AsyncInferenceQueue basic operations
 *
 * Tests request submission, ID generation, and queue growth.
 * Compile: g++ -std=c++17 -O2 -pthread -I./cpp_extensions -o test_async_queue_basic \
 *              tests/unit/test_async_queue_basic.cpp \
 *              cpp_extensions/mcts/async_inference_queue.cpp \
 *              cpp_extensions/mcts/tree.cpp \
 *              cpp_extensions/mcts/virtual_loss.cpp \
 *              cpp_extensions/games/gomoku/gomoku_state.cpp \
 *              cpp_extensions/games/gomoku/gomoku_rules.cpp \
 *              cpp_extensions/utils/random.cpp
 */

#include "mcts/async_inference_queue.hpp"
#include "games/gomoku/gomoku_state.h"
#include <cassert>
#include <iostream>
#include <memory>

using namespace mcts;
using namespace alphazero::core;
using namespace alphazero::games::gomoku;

// Test 1: Queue instantiation
void test_queue_instantiation() {
    std::cout << "Test 1: Queue instantiation..." << std::flush;

    AsyncInferenceQueue queue;

    assert(queue.pending_count() == 0);
    assert(queue.results_count() == 0);
    assert(!queue.has_results());

    std::cout << " PASS" << std::endl;
}

// Test 2: Submit single request
void test_submit_single_request() {
    std::cout << "Test 2: Submit single request..." << std::flush;

    AsyncInferenceQueue queue;

    // Create game state
    auto state = std::make_unique<GomokuState>();

    // Submit request
    std::vector<NodeIndex> path = {0};
    uint64_t req_id = queue.submit_request(std::move(state), 0, path);

    // Verify ID generated
    assert(req_id == 0);  // First request should have ID 0

    // Verify queue size
    assert(queue.pending_count() == 1);

    std::cout << " PASS" << std::endl;
}

// Test 3: Unique ID generation
void test_unique_id_generation() {
    std::cout << "Test 3: Unique ID generation..." << std::flush;

    AsyncInferenceQueue queue;

    // Submit multiple requests
    std::vector<uint64_t> ids;
    for (int i = 0; i < 10; ++i) {
        auto state = std::make_unique<GomokuState>();
        std::vector<NodeIndex> path = {static_cast<NodeIndex>(i)};
        uint64_t req_id = queue.submit_request(std::move(state), i, path);
        ids.push_back(req_id);
    }

    // Verify all IDs are unique and sequential
    for (size_t i = 0; i < ids.size(); ++i) {
        assert(ids[i] == i);
    }

    // Verify queue size
    assert(queue.pending_count() == 10);

    std::cout << " PASS" << std::endl;
}

// Test 4: Request submission is non-blocking (performance)
void test_submission_performance() {
    std::cout << "Test 4: Request submission performance..." << std::flush;

    AsyncInferenceQueue queue;

    auto start = std::chrono::high_resolution_clock::now();

    // Submit 1000 requests
    for (int i = 0; i < 1000; ++i) {
        auto state = std::make_unique<GomokuState>();
        std::vector<NodeIndex> path = {static_cast<NodeIndex>(i)};
        queue.submit_request(std::move(state), i, path);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);

    double avg_ms = duration.count() / 1000.0;

    // Should take <1000ms for 1000 requests (relaxed: includes GomokuState construction)
    // Target: <1ms per request for submission (actual submission is much faster)
    assert(duration.count() < 1000.0);

    std::cout << " PASS (avg: " << avg_ms << "ms per request)" << std::endl;
}

int main() {
    std::cout << "=== AsyncInferenceQueue Basic Tests ===" << std::endl;

    test_queue_instantiation();
    test_submit_single_request();
    test_unique_id_generation();
    test_submission_performance();

    std::cout << "\nAll 4/4 tests passed!" << std::endl;
    return 0;
}
