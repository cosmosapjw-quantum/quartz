/**
 * @file test_async_queue_results.cpp
 * @brief Unit tests for AsyncInferenceQueue result distribution
 *
 * Tests result submission, retrieval, and consumption.
 * Compile: g++ -std=c++17 -O2 -pthread -I./cpp_extensions -o test_async_queue_results \
 *              tests/unit/test_async_queue_results.cpp \
 *              cpp_extensions/mcts/async_inference_queue.cpp \
 *              cpp_extensions/mcts/tree.cpp \
 *              cpp_extensions/mcts/virtual_loss.cpp
 */

#include "mcts/async_inference_queue.hpp"
#include <cassert>
#include <algorithm>
#include <iostream>

using namespace mcts;

// Helper for floating point comparison
bool approx_equal(float a, float b, float epsilon = 0.0001f) {
    return std::abs(a - b) < epsilon;
}

// Test 1: Submit and retrieve single result
void test_submit_retrieve_single_result() {
    std::cout << "Test 1: Submit and retrieve single result..." << std::flush;

    AsyncInferenceQueue queue;

    // Create result
    InferenceResult result;
    result.request_id = 42;
    result.policy = std::vector<float>(225, 0.01f);  // Uniform policy for 15x15
    result.value = 0.5f;

    // Submit result
    queue.submit_results({result});

    // Verify result available
    assert(queue.has_results());
    assert(queue.results_count() == 1);

    // Retrieve result
    auto retrieved = queue.try_get_result(42);
    assert(retrieved.has_value());
    assert(retrieved->request_id == 42);
    assert(retrieved->value == 0.5f);
    assert(retrieved->policy.size() == 225);

    // Verify consumed
    assert(!queue.has_results());
    assert(queue.results_count() == 0);

    std::cout << " PASS" << std::endl;
}

// Test 2: Try get result for non-existent ID
void test_get_nonexistent_result() {
    std::cout << "Test 2: Try get result for non-existent ID..." << std::flush;

    AsyncInferenceQueue queue;

    // Try to get result that doesn't exist
    auto result = queue.try_get_result(999);

    assert(!result.has_value());
    assert(!queue.has_results());

    std::cout << " PASS" << std::endl;
}

// Test 3: Submit multiple results
void test_submit_multiple_results() {
    std::cout << "Test 3: Submit multiple results..." << std::flush;

    AsyncInferenceQueue queue;

    // Create batch of results
    std::vector<InferenceResult> results;
    for (int i = 0; i < 10; ++i) {
        InferenceResult result;
        result.request_id = i;
        result.policy = std::vector<float>(225, 0.01f);
        result.value = static_cast<float>(i) * 0.1f;
        results.push_back(result);
    }

    // Submit batch
    queue.submit_results(results);

    // Verify all results available
    assert(queue.results_count() == 10);

    // Retrieve in different order
    auto r5 = queue.try_get_result(5);
    assert(r5.has_value() && approx_equal(r5->value, 0.5f));

    auto r0 = queue.try_get_result(0);
    assert(r0.has_value() && approx_equal(r0->value, 0.0f));

    auto r9 = queue.try_get_result(9);
    assert(r9.has_value() && approx_equal(r9->value, 0.9f));

    // Verify remaining count
    assert(queue.results_count() == 7);

    std::cout << " PASS" << std::endl;
}

// Test 4: Result consumed after retrieval
void test_result_consumed_after_retrieval() {
    std::cout << "Test 4: Result consumed after retrieval..." << std::flush;

    AsyncInferenceQueue queue;

    // Submit result
    InferenceResult result;
    result.request_id = 100;
    result.policy = std::vector<float>(225, 0.01f);
    result.value = 0.7f;
    queue.submit_results({result});

    // Retrieve once
    auto first = queue.try_get_result(100);
    assert(first.has_value());

    // Try to retrieve again (should fail - consumed)
    auto second = queue.try_get_result(100);
    assert(!second.has_value());

    std::cout << " PASS" << std::endl;
}

// Test 5: Has results check
void test_has_results_check() {
    std::cout << "Test 5: Has results check..." << std::flush;

    AsyncInferenceQueue queue;

    // Initially empty
    assert(!queue.has_results());

    // Submit result
    InferenceResult result;
    result.request_id = 1;
    result.policy = std::vector<float>(225, 0.01f);
    result.value = 0.0f;
    queue.submit_results({result});

    // Should have results
    assert(queue.has_results());

    // Consume result
    queue.try_get_result(1);

    // Should be empty again
    assert(!queue.has_results());

    std::cout << " PASS" << std::endl;
}

// Test 6: Consume ready results in bulk
void test_consume_ready_results() {
    std::cout << "Test 6: Consume ready results..." << std::flush;

    AsyncInferenceQueue queue;

    std::vector<InferenceResult> results;
    for (int i = 0; i < 3; ++i) {
        InferenceResult result;
        result.request_id = i;
        result.policy = std::vector<float>(4, static_cast<float>(i));
        result.value = static_cast<float>(i);
        results.push_back(result);
    }

    queue.submit_results(results);

    auto consumed = queue.consume_ready_results();
    assert(consumed.size() == 3);
    assert(queue.results_count() == 0);
    assert(!queue.has_results());

    // Results should match submitted order (unordered_map -> arbitrary order, so just verify IDs exist)
    std::vector<uint64_t> ids;
    ids.reserve(consumed.size());
    for (const auto& res : consumed) {
        ids.push_back(res.request_id);
    }
    std::sort(ids.begin(), ids.end());
    for (int i = 0; i < 3; ++i) {
        assert(ids[i] == static_cast<uint64_t>(i));
    }

    std::cout << " PASS" << std::endl;
}

int main() {
    std::cout << "=== AsyncInferenceQueue Results Tests ===" << std::endl;

    test_submit_retrieve_single_result();
    test_get_nonexistent_result();
    test_submit_multiple_results();
    test_result_consumed_after_retrieval();
    test_has_results_check();
    test_consume_ready_results();

    std::cout << "\nAll 6/6 tests passed!" << std::endl;
    return 0;
}
