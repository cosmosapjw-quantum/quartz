/**
 * @file async_inference_queue.cpp
 * @brief Lock-free implementation of async inference queue (T006b)
 */

#include "async_inference_queue.hpp"
#include "instrumentation.hpp"
#include "profiling/enhanced_profiler.hpp"
#include <chrono>
#include <iostream>
#include <limits>
#include <thread>
#include <unordered_set>

using namespace mcts::profiling;

namespace mcts {

AsyncInferenceQueue::AsyncInferenceQueue()
    : next_request_id_(0) {
}

AsyncInferenceQueue::~AsyncInferenceQueue() {
    // Cleanup (no mutexes to destroy in lock-free implementation)
}

uint64_t AsyncInferenceQueue::submit_request(InferenceRequest&& request) {
    ScopedMetric metric(InstrumentationMetric::QueueSubmit);
    PROFILE_SCOPE(ProfileMetric::QueueSubmitTotal);

    // Generate unique request ID and assign to request
    uint64_t request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
    request.request_id = request_id;

    // Try to enqueue (wait-free, no locks)
    // **Phase 1 Optimization (T018)**: Request already contains pre-extracted features
    // moved from thread-local buffer. No state cloning occurs here.
    // Queue has 4096 capacity, so full queue should be extremely rare
    {
        PROFILE_SCOPE(ProfileMetric::QueueSubmitEnqueue);
        if (!pending_requests_.try_enqueue(std::move(request))) {
            // Queue full - this should be extremely rare with 4096 capacity
            // If it happens, it means 4096+ requests are pending, which indicates
            // a serious bottleneck in the inference coordinator
            PROFILE_COUNTER(ProfileMetric::CAS_FailureCount, 1);
            throw std::runtime_error("AsyncInferenceQueue: Queue full (4096+ pending requests). "
                                   "Inference coordinator cannot keep up with submission rate.");
        }
        PROFILE_COUNTER(ProfileMetric::CAS_SuccessCount, 1);
    }

    // Successfully enqueued
    pending_count_.fetch_add(1, std::memory_order_relaxed);

    // T020: Notify waiting coordinator threads (condition variable wakeup)
    // Using notify_all() allows OS scheduler to pick the best thread to wake
    // instead of potentially waking a suboptimal thread with notify_one()
    request_ready_.notify_all();

    return request_id;
}

uint64_t AsyncInferenceQueue::submit_request_with_backpressure(InferenceRequest&& request,
                                                                 double timeout_ms) {
    // T060: Phase 5 backpressure mechanism for multi-coordinator scenarios
    ScopedMetric metric(InstrumentationMetric::QueueSubmit);
    PROFILE_SCOPE(ProfileMetric::QueueSubmitTotal);
    using namespace std::chrono;

    // Generate unique request ID
    uint64_t request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
    request.request_id = request_id;

    // Try to enqueue (wait-free, no locks)
    bool enqueued = false;
    auto start_time = steady_clock::now();

    while (!enqueued && !shutting_down_.load(std::memory_order_relaxed)) {
        // Try non-blocking enqueue first
        {
            PROFILE_SCOPE(ProfileMetric::QueueSubmitEnqueue);
            if (pending_requests_.try_enqueue(std::move(request))) {
                enqueued = true;
                PROFILE_COUNTER(ProfileMetric::CAS_SuccessCount, 1);
                break;
            }
            PROFILE_COUNTER(ProfileMetric::CAS_FailureCount, 1);
        }

        // Queue full - wait for space to become available
        // Check timeout
        auto elapsed = steady_clock::now() - start_time;
        if (timeout_ms > 0.0) {
            auto elapsed_ms = duration_cast<milliseconds>(elapsed).count();
            if (elapsed_ms >= timeout_ms) {
                throw std::runtime_error("AsyncInferenceQueue: Timeout waiting for queue space");
            }
        }

        // Wait on condition variable for space to become available
        {
            PROFILE_SCOPE(ProfileMetric::ThreadWaitingForResults);

            auto remaining = (timeout_ms > 0.0)
                ? milliseconds(static_cast<int64_t>(timeout_ms)) - duration_cast<milliseconds>(elapsed)
                : milliseconds(100);  // 100ms poll interval for infinite wait

            std::unique_lock<std::mutex> lock(backpressure_mutex_);
            space_available_.wait_for(lock, remaining, [this] {
                // Wake up if: shutdown requested, or queue has space
                return shutting_down_.load(std::memory_order_relaxed) ||
                       pending_count_.load(std::memory_order_relaxed) < 4000;  // Leave 96-entry margin
            });
        }

        // Check shutdown
        if (shutting_down_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("AsyncInferenceQueue: Shutdown during backpressure wait");
        }
    }

    if (!enqueued) {
        throw std::runtime_error("AsyncInferenceQueue: Failed to enqueue (shutdown or internal error)");
    }

    // Successfully enqueued
    pending_count_.fetch_add(1, std::memory_order_relaxed);

    // Notify waiting coordinator threads
    request_ready_.notify_all();

    return request_id;
}

std::vector<InferenceRequest> AsyncInferenceQueue::collect_batch(size_t min_batch_size,
                                                                   double timeout_ms) {
    ScopedMetric metric(InstrumentationMetric::QueueCollect);
    PROFILE_SCOPE(ProfileMetric::QueueCollectTotal);
    using namespace std::chrono;

    std::vector<InferenceRequest> batch;

    const auto max_batch_size = (min_batch_size > 0)
        ? (min_batch_size + (min_batch_size / 2))
        : 4096;  // Max queue capacity

    const auto timeout_duration = duration<double, std::milli>(timeout_ms);
    const auto deadline = steady_clock::now() + timeout_duration;

    // T006c: Wait for min_batch_size with timeout using condition variable (eliminates CPU waste)
    uint64_t total_wait_ns = 0;
    if (min_batch_size > 0 && timeout_ms > 0.0) {
        while (batch.size() < min_batch_size && !shutting_down_.load(std::memory_order_relaxed)) {
            InferenceRequest request;
            // Lock-free dequeue attempt
            if (pending_requests_.try_dequeue(request)) {
                batch.push_back(std::move(request));
                pending_count_.fetch_sub(1, std::memory_order_relaxed);
            } else {
                // Calculate remaining time
                auto now = steady_clock::now();
                if (now >= deadline) {
                    break;  // Timeout expired
                }
                auto remaining = deadline - now;

                // Block on condition variable instead of polling
                // Track wait time (review.txt lines 71-136: thread idle time)
                auto wait_start = steady_clock::now();
                {
                    // Use single PROFILE_SCOPE to avoid macro name collision
                    PROFILE_SCOPE(ProfileMetric::ThreadWaitingForResults);

                    std::unique_lock<std::mutex> lock(cv_mutex_);
                    request_ready_.wait_for(lock, remaining, [this, &batch, min_batch_size] {
                        // Wake up if: shutdown requested, or queue has data
                        return shutting_down_.load(std::memory_order_relaxed) ||
                               pending_count_.load(std::memory_order_relaxed) > 0;
                    });
                }
                auto wait_elapsed = steady_clock::now() - wait_start;
                total_wait_ns += duration_cast<nanoseconds>(wait_elapsed).count();

                // Re-check timeout after waking up
                if (steady_clock::now() >= deadline) {
                    break;
                }
            }
        }
    }

    // Track idle/wait time for this collect operation
    if (total_wait_ns > 0) {
        PROFILE_GAUGE(ProfileMetric::ThreadIdleTotal, total_wait_ns);
    }

    // Opportunistically grab more up to max_batch_size
    while (batch.size() < max_batch_size) {
        InferenceRequest request;
        if (!pending_requests_.try_dequeue(request)) {
            break;  // Queue empty
        }
        batch.push_back(std::move(request));
        pending_count_.fetch_sub(1, std::memory_order_relaxed);
    }

    // Track unique node indices in batch for diversity metrics
    if (!batch.empty()) {
        std::unordered_set<NodeIndex> unique_nodes;
        for (const auto& request : batch) {
            unique_nodes.insert(request.node_index);
        }

        Instrumentation::instance().increment_counter(
            InstrumentationMetric::UniqueBatchPositions,
            unique_nodes.size()
        );

        // Track batch size
        PROFILE_GAUGE(ProfileMetric::QueueCollectBatchSize, batch.size());
        PROFILE_GAUGE(ProfileMetric::GPUBatchSize, batch.size());
    }

    return batch;
}

void AsyncInferenceQueue::submit_results(const std::vector<InferenceResult>& results) {
    // T006b: Lock-free result submission using ring buffer
    for (const auto& result : results) {
        // Store in ring buffer using request_id for O(1) indexing
        size_t slot_index = result.request_id % RESULTS_BUFFER_CAPACITY;
        ResultSlot& slot = results_buffer_[slot_index];

        // Store data
        slot.request_id = result.request_id;
        slot.data = result;

        // Mark as occupied (release to ensure data is visible)
        slot.occupied.store(true, std::memory_order_release);
        results_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

std::optional<InferenceResult> AsyncInferenceQueue::try_get_result(uint64_t request_id) {
    ScopedMetric metric(InstrumentationMetric::QueueTryGetResult);
    PROFILE_SCOPE(ProfileMetric::QueueTryGetResult);

    // T006b: Lock-free O(1) lookup using ring buffer
    size_t slot_index = request_id % RESULTS_BUFFER_CAPACITY;
    ResultSlot& slot = results_buffer_[slot_index];

    // Check if slot is occupied (acquire to ensure data is visible)
    if (!slot.occupied.load(std::memory_order_acquire)) {
        return std::nullopt;
    }

    // Verify request_id matches (handle collisions)
    if (slot.request_id != request_id) {
        return std::nullopt;
    }

    // Extract result
    InferenceResult result = slot.data;

    // Mark slot as free
    slot.occupied.store(false, std::memory_order_release);
    results_count_.fetch_sub(1, std::memory_order_relaxed);

    return result;
}

bool AsyncInferenceQueue::has_results() const {
    return results_count_.load(std::memory_order_relaxed) > 0;
}

std::vector<InferenceResult> AsyncInferenceQueue::consume_ready_results() {
    // T006b: Scan all slots for occupied entries
    // Note: This is less efficient than individual try_get_result() calls
    std::vector<InferenceResult> results;
    results.reserve(256);  // Pre-allocate reasonable size

    for (size_t i = 0; i < RESULTS_BUFFER_CAPACITY; ++i) {
        ResultSlot& slot = results_buffer_[i];

        // Check if occupied (acquire to ensure data is visible)
        if (slot.occupied.load(std::memory_order_acquire)) {
            // Extract result
            results.push_back(slot.data);

            // Mark as free
            slot.occupied.store(false, std::memory_order_release);
            results_count_.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    return results;
}

size_t AsyncInferenceQueue::pending_count() const {
    return pending_count_.load(std::memory_order_relaxed);
}

size_t AsyncInferenceQueue::results_count() const {
    return results_count_.load(std::memory_order_relaxed);
}

size_t AsyncInferenceQueue::get_memory_usage() const {
    // T006b: Fixed memory allocation
    // - MPMCRingBuffer: 4096 slots * ~100 bytes/request = ~400KB
    // - Results buffer: 8192 slots * ~64 bytes (alignment) = ~524KB
    // Total: ~1MB fixed allocation
    return (4096 * 100) + (RESULTS_BUFFER_CAPACITY * 64);
}

void AsyncInferenceQueue::shutdown() {
    // T006c: Set shutdown flag and wake up all waiting threads
    shutting_down_.store(true, std::memory_order_relaxed);
    request_ready_.notify_all();

    // T062: Also wake threads waiting for queue space (Phase 5 backpressure)
    space_available_.notify_all();
}

void AsyncInferenceQueue::notify_dequeued() {
    // T062: Phase 5 - Wake threads waiting for queue space after batch dequeue
    // Called immediately after collect_batch() to minimize wait time for blocked submissions
    space_available_.notify_all();
}

std::vector<uint64_t> AsyncInferenceQueue::get_ready_request_ids() const {
    // Deprecated: Scan all slots to find occupied entries
    std::vector<uint64_t> ids;
    ids.reserve(256);

    for (size_t i = 0; i < RESULTS_BUFFER_CAPACITY; ++i) {
        const ResultSlot& slot = results_buffer_[i];
        if (slot.occupied.load(std::memory_order_acquire)) {
            ids.push_back(slot.request_id);
        }
    }

    return ids;
}

} // namespace mcts
