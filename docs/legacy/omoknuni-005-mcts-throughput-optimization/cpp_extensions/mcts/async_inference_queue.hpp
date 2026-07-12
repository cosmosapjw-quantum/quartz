/**
 * @file async_inference_queue.hpp
 * @brief Lock-free async inference queue for non-blocking MCTS simulation
 *
 * This module implements a wait-free queue system that decouples MCTS simulation
 * threads from neural network inference. Simulations submit inference requests
 * asynchronously and continue working, while a background coordinator batches
 * requests and calls Python inference once per batch.
 *
 * Performance targets:
 * - Request submission: <0.1ms (wait-free with MPMCRingBuffer)
 * - Batch collection: triggered by count (≥32) OR timeout (≤2ms)
 * - Result retrieval: <0.1ms (lock-free O(1) ring buffer lookup)
 * - Memory: Fixed 8MB allocation (4096 requests + 8192 results)
 *
 * Key design principles:
 * - Wait-free request submission (no locks, no blocking)
 * - Lock-free result retrieval with O(1) ring buffer indexing
 * - Timeout-based batch collection via condition variables (T006c - efficient blocking)
 * - Fixed memory footprint with predictable allocation
 *
 * Architecture (T006b):
 * - Lock-free MPMCRingBuffer for pending requests (capacity 4096)
 * - Ring buffer array for completed results (capacity 8192)
 * - Atomic counters for queue depth monitoring
 * - No mutexes or condition variables in hot paths
 */

#pragma once

#include "tree.hpp"
#include "lock_free_queue.hpp"
#include "../utils/igamestate.h"
#include <vector>
#include <array>
#include <optional>
#include <memory>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <condition_variable>

namespace mcts {

// Forward declaration for game state interface
using IGameState = alphazero::core::IGameState;

/**
 * @brief Request for neural network inference
 *
 * Represents a single position that needs evaluation. Submitted by
 * simulation threads during tree traversal.
 *
 * **Phase 1 Zero-Copy Optimization (T015-T016)**: Changed from game state pointer
 * to pre-extracted features vector. Features are extracted in-place at leaf nodes
 * using thread-local buffers, then moved (not copied) into the request. This
 * eliminates the 418μs state cloning bottleneck (86.6% of execution time).
 */
struct InferenceRequest {
    uint64_t request_id;                        // Unique identifier for this request
    std::vector<float> features;                // OWNED (moved from thread-local buffer)
    int32_t node_index;                         // Tree node to expand
    int32_t action_space_size;                  // Number of legal moves
    int16_t board_size;                         // Board dimension (8, 9, 15, or 19)
    int16_t planes;                             // Feature plane count (25-36)
    std::vector<int16_t> path;                  // Move path from root (for reconstruction fallback)

    // Default constructor for container compatibility
    InferenceRequest() : request_id(0), node_index(0), action_space_size(0), board_size(0), planes(0) {}

    // Move-only type (enforces zero-copy ownership transfer)
    InferenceRequest(InferenceRequest&&) = default;
    InferenceRequest& operator=(InferenceRequest&&) = default;
    InferenceRequest(const InferenceRequest&) = delete;
    InferenceRequest& operator=(const InferenceRequest&) = delete;
};

/**
 * @brief Result from neural network inference
 *
 * Contains policy and value for a previously submitted request.
 */
struct InferenceResult {
    uint64_t request_id;                        // Matches original request
    std::vector<float> policy;                  // Prior probabilities over actions
    float value;                                // Position evaluation [-1, 1]

    // Copyable and movable
    InferenceResult() = default;
    InferenceResult(const InferenceResult&) = default;
    InferenceResult(InferenceResult&&) = default;
    InferenceResult& operator=(const InferenceResult&) = default;
    InferenceResult& operator=(InferenceResult&&) = default;
};

/**
 * @brief Thread-safe async inference queue
 *
 * Decouples MCTS simulation threads from GPU inference by providing:
 * 1. Non-blocking request submission (threads never wait)
 * 2. Batched request collection (by count or timeout)
 * 3. Result distribution back to threads
 *
 * Thread Safety:
 * - Multiple threads can submit requests concurrently
 * - Single coordinator thread collects batches
 * - Multiple threads can retrieve results concurrently
 * - All operations protected by mutexes
 *
 * Performance Characteristics:
 * - Request submission: O(1), <0.1ms
 * - Batch collection: O(batch_size), <2ms timeout
 * - Result retrieval: O(1), <0.1ms
 * - Memory: ~100 bytes per pending request
 */
class AsyncInferenceQueue {
public:
    /**
     * @brief Construct empty inference queue
     */
    AsyncInferenceQueue();

    /**
     * @brief Destructor (cleanup any pending requests)
     */
    ~AsyncInferenceQueue();

    /**
     * @brief Submit inference request with pre-extracted features (non-blocking)
     *
     * Adds request to pending queue and returns immediately. Thread does NOT
     * wait for inference to complete.
     *
     * **Phase 1 Zero-Copy Optimization (T017-T018)**: Accepts rvalue reference
     * to InferenceRequest with pre-extracted features. Features are extracted
     * in-place at leaf nodes using thread-local buffers, then moved (not copied)
     * into the request. This eliminates the 418μs state cloning bottleneck.
     *
     * Thread Safety: Safe to call from multiple threads concurrently
     *
     * @param request Inference request with pre-extracted features (moved, ownership transferred)
     * @return Unique request ID for retrieving result later
     */
    uint64_t submit_request(InferenceRequest&& request);

    /**
     * @brief Submit inference request with backpressure (blocking when full)
     *
     * **Phase 5 Multi-Coordinator Optimization (T060)**: Adds backpressure mechanism
     * to prevent queue overflow when multiple coordinators drain at different rates.
     * Blocks when queue is full (4096 entries) until space becomes available.
     *
     * This is essential for multi-coordinator scenarios where K coordinators may
     * temporarily fall behind submission rate, causing queue buildup. Backpressure
     * prevents memory exhaustion and ensures fair scheduling.
     *
     * Thread Safety: Safe to call from multiple threads concurrently
     *
     * @param request Inference request with pre-extracted features (moved, ownership transferred)
     * @param timeout_ms Maximum wait time in milliseconds (0 = infinite wait)
     * @return Unique request ID for retrieving result later
     * @throws std::runtime_error if timeout expires without space becoming available
     */
    uint64_t submit_request_with_backpressure(InferenceRequest&& request, double timeout_ms = 0.0);

    /**
     * @brief Collect batch of pending requests
     *
     * Returns when EITHER:
     * - Number of pending requests >= min_batch_size
     * - Timeout elapsed (timeout_ms milliseconds)
     *
     * Whichever condition is met first triggers the batch return.
     *
     * Thread Safety: Should only be called by single coordinator thread
     *
     * @param min_batch_size Minimum batch size to wait for (e.g., 32)
     * @param timeout_ms Maximum wait time in milliseconds (e.g., 2.0)
     * @return Vector of requests to process (empty if timeout with no requests)
     */
    std::vector<InferenceRequest> collect_batch(size_t min_batch_size, double timeout_ms);

    /**
     * @brief Submit batch of inference results
     *
     * Called by coordinator thread after GPU inference completes.
     * Makes results available for simulation threads to retrieve.
     *
     * Thread Safety: Should only be called by single coordinator thread
     *
     * @param results Vector of results matching previously collected requests
     */
    void submit_results(const std::vector<InferenceResult>& results);

    /**
     * @brief Try to retrieve result for a request (non-blocking)
     *
     * Checks if result is available for given request ID. If found, returns
     * the result and removes it from the map (consumed).
     *
     * Thread Safety: Safe to call from multiple threads concurrently
     *
     * @param request_id Request ID from submit_request()
     * @return Result if available, std::nullopt otherwise
     */
    std::optional<InferenceResult> try_get_result(uint64_t request_id);

    /**
     * @brief Consume all ready results in a single batch.
     *
     * Moves the completed results into a vector and clears the internal map.
     *
     * Thread Safety: Safe to call from multiple threads; typically used by
     * the async coordinator / simulation runners.
     */
    std::vector<InferenceResult> consume_ready_results();

    /**
     * @brief Check if any results are available
     *
     * Quick check before calling try_get_result() to avoid unnecessary polling.
     *
     * Thread Safety: Safe to call from multiple threads concurrently
     *
     * @return true if results map is non-empty
     */
    bool has_results() const;

    /**
     * @brief Get number of pending requests
     *
     * Useful for monitoring queue depth and detecting backpressure.
     *
     * Thread Safety: Safe to call from any thread
     *
     * @return Number of requests waiting for inference
     */
    size_t pending_count() const;

    /**
     * @brief Get number of completed results waiting for retrieval
     *
     * Useful for monitoring if results are being consumed quickly enough.
     *
     * Thread Safety: Safe to call from any thread
     *
     * @return Number of results available for retrieval
     */
    size_t results_count() const;

    /**
     * @brief Get memory usage estimate in bytes
     *
     * Includes pending requests and completed results.
     *
     * @return Estimated memory usage
     */
    size_t get_memory_usage() const;

    /**
     * @brief Wake up threads waiting in collect_batch()
     *
     * This is called during coordinator shutdown to ensure threads waiting
     * on condition variables are woken up so they can check the running flag
     * and exit cleanly.
     *
     * Thread Safety: Safe to call from any thread
     */
    void shutdown();

    /**
     * @brief Notify waiting threads that queue space is available (T062)
     *
     * **Phase 5 Multi-Coordinator Optimization**: Called after batch dequeue to wake
     * simulation threads blocked in submit_request_with_backpressure(). Enables
     * backpressure mechanism for multi-coordinator scenarios.
     *
     * This should be called immediately after collect_batch() completes to minimize
     * wait time for blocked submission threads.
     *
     * Thread Safety: Safe to call from any thread (typically coordinator thread)
     */
    void notify_dequeued();

    /**
     * @brief Snapshot the request IDs with completed inference results.
     *
     * Thread Safety: Safe to call from any thread.
     *
     * @return Vector of request IDs currently ready for retrieval
     */
    [[deprecated("Use try_get_result() instead - no bulk operations needed")]]
    std::vector<uint64_t> get_ready_request_ids() const;

private:
    // Request ID generation
    std::atomic<uint64_t> next_request_id_{0};

    // Lock-free pending requests queue (T006b)
    MPMCRingBuffer<InferenceRequest, 4096> pending_requests_;
    std::atomic<size_t> pending_count_{0};

    // Condition variable for efficient waiting (T006c)
    std::mutex cv_mutex_;
    std::condition_variable request_ready_;
    std::atomic<bool> shutting_down_{false};

    // Backpressure mechanism for multi-coordinator (T061 - Phase 5)
    std::mutex backpressure_mutex_;
    std::condition_variable space_available_;  // Wakes threads waiting for queue space

    // Lock-free completed results ring buffer (T006b)
    static constexpr size_t RESULTS_BUFFER_CAPACITY = 8192;

    struct alignas(64) ResultSlot {
        std::atomic<bool> occupied{false};
        uint64_t request_id{0};
        InferenceResult data;
    };

    std::array<ResultSlot, RESULTS_BUFFER_CAPACITY> results_buffer_;
    std::atomic<size_t> results_count_{0};
};

} // namespace mcts
