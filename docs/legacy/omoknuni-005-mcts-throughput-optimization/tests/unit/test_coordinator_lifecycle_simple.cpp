/**
 * @file test_coordinator_lifecycle_simple.cpp
 * @brief Simplified C++ unit tests for BatchInferenceCoordinator lifecycle
 *
 * Tests basic lifecycle without game state dependencies:
 * 1. Thread starts cleanly
 * 2. Thread stops cleanly
 * 3. is_running() reports correct state
 * 4. Can be stopped multiple times (idempotent)
 */

#include "../../cpp_extensions/mcts/batch_inference_coordinator.hpp"
#include "../../cpp_extensions/mcts/async_inference_queue.hpp"
#include "../../cpp_extensions/mcts/batch_inference_callback.hpp"
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>

using namespace mcts;

// Minimal mock callback that doesn't require game states
class MinimalMockCallback : public BatchInferenceCallback {
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
    MinimalMockCallback callback;
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
    MinimalMockCallback callback;
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
    MinimalMockCallback callback;
    BatchInferenceCoordinator coordinator;

    coordinator.start(queue, callback, 32, 2.0);

    // Stop multiple times should be safe
    coordinator.stop();
    coordinator.stop();
    coordinator.stop();

    assert(!coordinator.is_running());

    std::cout << "PASSED\n";
}

void test_coordinator_destructor_cleanup() {
    std::cout << "TEST: Coordinator destructor cleans up... ";

    AsyncInferenceQueue queue;
    MinimalMockCallback callback;

    {
        BatchInferenceCoordinator coordinator;
        coordinator.start(queue, callback, 32, 2.0);
        assert(coordinator.is_running());
        // Destructor should clean up automatically
    }

    // If we get here without hanging, destructor worked
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n=== BatchInferenceCoordinator Lifecycle Tests ===\n\n";

    test_coordinator_starts();
    test_coordinator_stops_cleanly();
    test_coordinator_stop_idempotent();
    test_coordinator_destructor_cleanup();

    std::cout << "\n=== All coordinator lifecycle tests passed! ===\n\n";

    return 0;
}
