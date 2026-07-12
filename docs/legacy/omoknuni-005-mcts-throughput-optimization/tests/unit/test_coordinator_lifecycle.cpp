/**
 * @file test_coordinator_lifecycle.cpp
 * @brief C++ unit tests for BatchInferenceCoordinator lifecycle management
 *
 * Tests:
 * 1. Thread starts cleanly
 * 2. Thread stops cleanly
 * 3. is_running() reports correct state
 * 4. Can be stopped multiple times (idempotent)
 */

#include "../../cpp_extensions/mcts/batch_inference_coordinator.hpp"
#include "../../cpp_extensions/mcts/async_inference_queue.hpp"
#include "../../cpp_extensions/utils/igamestate.h"
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <vector>

using namespace mcts;

// Mock implementation of IGameState for testing
class MockGameState : public IGameState {
public:
    MockGameState(int id = 0) : id_(id) {}

    std::unique_ptr<IGameState> clone() const override {
        return std::make_unique<MockGameState>(id_);
    }

    // T024c: Stub implementations for make/unmake (not used in coordinator tests)
    uint64_t make_move(uint16_t move) override {
        last_move_ = move;
        return 0;  // Dummy undo token
    }

    void unmake_move(uint16_t move, uint64_t undo_token) override {
        // No-op for this test
    }

    void apply_move(uint16_t move) override { last_move_ = move; }
    bool is_terminal() const override { return false; }
    int get_current_player() const override { return 0; }
    std::vector<uint16_t> get_legal_moves() const override {
        return {0, 1, 2, 3, 4};
    }
    int get_action_space_size() const override { return 225; }
    std::vector<float> get_feature_planes() const override {
        return std::vector<float>(225, 0.0f);
    }
    int get_num_feature_planes() const override { return 1; }
    int get_board_height() const override { return 15; }
    int get_board_width() const override { return 15; }

private:
    int id_;
    uint16_t last_move_{0};
};

// Mock batch inference callback that returns dummy results
class MockBatchCallback : public BatchInferenceCallback {
public:
    std::vector<std::pair<std::vector<float>, float>>
    batch_inference(const std::vector<const IGameState*>& states) override {
        std::vector<std::pair<std::vector<float>, float>> results;
        results.reserve(states.size());

        for (size_t i = 0; i < states.size(); ++i) {
            // Return uniform policy and zero value
            std::vector<float> policy(225, 1.0f / 225.0f);
            float value = 0.0f;
            results.emplace_back(std::move(policy), value);
        }

        return results;
    }
};

void test_coordinator_starts() {
    std::cout << "TEST: Coordinator starts cleanly... ";

    AsyncInferenceQueue queue;
    MockBatchCallback callback;
    BatchInferenceCoordinator coordinator;

    // Initially should not be running
    assert(!coordinator.is_running());

    // Start coordinator
    coordinator.start(queue, callback, 32, 2.0);

    // Should be running now
    assert(coordinator.is_running());

    // Stop coordinator
    coordinator.stop();
    assert(!coordinator.is_running());

    std::cout << "PASSED\n";
}

void test_coordinator_stops_cleanly() {
    std::cout << "TEST: Coordinator stops cleanly... ";

    AsyncInferenceQueue queue;
    MockBatchCallback callback;
    BatchInferenceCoordinator coordinator;

    coordinator.start(queue, callback, 32, 2.0);
    assert(coordinator.is_running());

    // Let it run for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Stop should wait for thread to finish
    coordinator.stop();
    assert(!coordinator.is_running());

    std::cout << "PASSED\n";
}

void test_coordinator_stop_idempotent() {
    std::cout << "TEST: Coordinator stop is idempotent... ";

    AsyncInferenceQueue queue;
    MockBatchCallback callback;
    BatchInferenceCoordinator coordinator;

    coordinator.start(queue, callback, 32, 2.0);

    // Stop multiple times should be safe
    coordinator.stop();
    coordinator.stop();
    coordinator.stop();

    assert(!coordinator.is_running());

    std::cout << "PASSED\n";
}

void test_coordinator_processes_batch() {
    std::cout << "TEST: Coordinator processes batches... ";

    AsyncInferenceQueue queue;
    MockBatchCallback callback;
    BatchInferenceCoordinator coordinator;

    // Start coordinator
    coordinator.start(queue, callback, 2, 100.0);  // Small batch size, long timeout

    // Submit some requests
    MockGameState state1(1);
    MockGameState state2(2);
    MockGameState state3(3);

    uint64_t id1 = queue.submit_request(state1.clone(), 10, {});
    uint64_t id2 = queue.submit_request(state2.clone(), 20, {});
    uint64_t id3 = queue.submit_request(state3.clone(), 30, {});

    // Wait for coordinator to process the batch
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // Results should be available
    auto result1 = queue.try_get_result(id1);
    auto result2 = queue.try_get_result(id2);
    auto result3 = queue.try_get_result(id3);

    assert(result1.has_value());
    assert(result2.has_value());
    assert(result3.has_value());

    assert(result1->request_id == id1);
    assert(result2->request_id == id2);
    assert(result3->request_id == id3);

    assert(result1->policy.size() == 225);
    assert(result2->policy.size() == 225);
    assert(result3->policy.size() == 225);

    coordinator.stop();

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n=== BatchInferenceCoordinator Lifecycle Tests ===\n\n";

    test_coordinator_starts();
    test_coordinator_stops_cleanly();
    test_coordinator_stop_idempotent();
    test_coordinator_processes_batch();

    std::cout << "\n=== All coordinator lifecycle tests passed! ===\n\n";

    return 0;
}
