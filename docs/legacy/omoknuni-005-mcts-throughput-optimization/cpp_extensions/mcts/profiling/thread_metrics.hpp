/**
 * @file thread_metrics.hpp
 * @brief Thread-local metrics storage with lock-free ring buffers
 * @author MCTS Performance Team
 * @date 2024
 */

#pragma once

#include "enhanced_metrics.hpp"
#include <atomic>
#include <array>
#include <chrono>
#include <cstring>
#include <thread>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace mcts {
namespace profiling {

/**
 * @brief Single timing sample in ring buffer
 */
struct TimingSample {
    ProfileMetric metric;
    uint64_t start_ns;
    uint64_t duration_ns;
    uint32_t thread_id;
    uint32_t depth;  // Call stack depth for hierarchical profiling
};

/**
 * @brief Lock-free single-producer ring buffer for timing samples
 *
 * FIXED 2025-10-16: Increased default capacity from 4096 to 524288
 * - Prevents buffer overflow during profiling (was dropping 88.6% of samples)
 * - Supports 17,476 simulations @ 30 samples/sim
 * - Memory: 16.4 MB per thread (192 MB for 12 threads) - acceptable
 */
template<size_t Capacity = 524288>
class TimingRingBuffer {
public:
    static constexpr size_t kCapacity = Capacity;
    static_assert((kCapacity & (kCapacity - 1)) == 0, "Capacity must be power of 2");

    TimingRingBuffer() : head_(0), tail_(0) {
        // Zero-initialize buffer
        std::memset(buffer_.data(), 0, sizeof(buffer_));
    }

    /**
     * @brief Try to push a sample (lock-free, wait-free)
     * @return true if successful, false if buffer full
     */
    bool try_push(const TimingSample& sample) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & (kCapacity - 1);

        // Check if buffer is full
        if (next_tail == head_.load(std::memory_order_acquire)) {
            overflows_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        // Write sample
        buffer_[current_tail] = sample;

        // Update tail (make visible to consumer)
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    /**
     * @brief Try to pop a sample (lock-free, wait-free)
     * @return true if sample retrieved, false if empty
     */
    bool try_pop(TimingSample& sample) {
        const size_t current_head = head_.load(std::memory_order_relaxed);

        // Check if empty
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }

        // Read sample
        sample = buffer_[current_head];

        // Update head
        head_.store((current_head + 1) & (kCapacity - 1), std::memory_order_release);
        return true;
    }

    /**
     * @brief Get number of samples in buffer
     */
    size_t size() const {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_relaxed);
        return (tail - head) & (kCapacity - 1);
    }

    /**
     * @brief Get overflow count
     */
    uint64_t overflows() const {
        return overflows_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Clear the buffer
     */
    void clear() {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        overflows_.store(0, std::memory_order_relaxed);
    }

private:
    alignas(64) std::atomic<size_t> head_;  // Cache line aligned
    alignas(64) std::atomic<size_t> tail_;  // Separate cache line
    alignas(64) std::array<TimingSample, kCapacity> buffer_;
    std::atomic<uint64_t> overflows_{0};
};

/**
 * @brief Per-thread metrics storage
 */
class ThreadLocalMetrics {
public:
    ThreadLocalMetrics()
        : thread_id_(std::hash<std::thread::id>{}(std::this_thread::get_id())),
          current_depth_(0) {
        // Initialize atomic arrays manually (cannot use fill() with atomics)
        for (size_t i = 0; i < static_cast<size_t>(ProfileMetric::MetricCount); ++i) {
            counters_[i].store(0, std::memory_order_relaxed);
            gauges_[i].store(0, std::memory_order_relaxed);
            total_durations_[i].store(0, std::memory_order_relaxed);
            call_counts_[i].store(0, std::memory_order_relaxed);
        }
    }

    // Destructor intentionally omitted to keep metrics accessible after thread exit
    // Thread-local unique_ptr will still free the memory, but map entry persists for reporting

    /**
     * @brief Record a timing sample
     */
    void record_timing(ProfileMetric metric, uint64_t start_ns, uint64_t duration_ns) {
        TimingSample sample{
            metric,
            start_ns,
            duration_ns,
            thread_id_,
            current_depth_
        };

        timing_buffer_.try_push(sample);

        // Update aggregates
        size_t idx = static_cast<size_t>(metric);
        total_durations_[idx].fetch_add(duration_ns, std::memory_order_relaxed);
        call_counts_[idx].fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Increment a counter metric
     */
    void increment_counter(ProfileMetric metric, uint64_t value = 1) {
        size_t idx = static_cast<size_t>(metric);
        counters_[idx].fetch_add(value, std::memory_order_relaxed);
    }

    /**
     * @brief Update a gauge metric (stores maximum value)
     */
    void update_gauge(ProfileMetric metric, uint64_t value) {
        size_t idx = static_cast<size_t>(metric);

        // Atomic max: only update if value is greater than current
        uint64_t current = gauges_[idx].load(std::memory_order_relaxed);
        while (value > current) {
            if (gauges_[idx].compare_exchange_weak(current, value, std::memory_order_relaxed)) {
                break;  // Successfully updated to new max
            }
            // CAS failed because another thread updated it; current now has the new value
            // Loop continues only if our value is still greater
        }
    }

    /**
     * @brief Enter a profiling scope (increase depth)
     */
    void enter_scope() {
        current_depth_++;
    }

    /**
     * @brief Exit a profiling scope (decrease depth)
     */
    void exit_scope() {
        if (current_depth_ > 0) {
            current_depth_--;
        }
    }

    /**
     * @brief Get timing buffer for analysis
     */
    TimingRingBuffer<524288>& get_timing_buffer() {
        return timing_buffer_;
    }

    /**
     * @brief Get counter value
     */
    uint64_t get_counter(ProfileMetric metric) const {
        size_t idx = static_cast<size_t>(metric);
        return counters_[idx].load(std::memory_order_relaxed);
    }

    /**
     * @brief Get gauge value
     */
    uint64_t get_gauge(ProfileMetric metric) const {
        size_t idx = static_cast<size_t>(metric);
        return gauges_[idx].load(std::memory_order_relaxed);
    }

    /**
     * @brief Get aggregate timing statistics
     */
    void get_timing_stats(ProfileMetric metric, uint64_t& total_ns, uint64_t& count) const {
        size_t idx = static_cast<size_t>(metric);
        total_ns = total_durations_[idx].load(std::memory_order_relaxed);
        count = call_counts_[idx].load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset all metrics
     */
    void reset() {
        timing_buffer_.clear();

        // Manually reset atomic arrays (cannot use fill() with atomics)
        for (size_t i = 0; i < static_cast<size_t>(ProfileMetric::MetricCount); ++i) {
            counters_[i].store(0, std::memory_order_relaxed);
            gauges_[i].store(0, std::memory_order_relaxed);
            total_durations_[i].store(0, std::memory_order_relaxed);
            call_counts_[i].store(0, std::memory_order_relaxed);
        }

        current_depth_ = 0;
    }

    uint32_t thread_id() const { return thread_id_; }

private:
    // Cache-line aligned for performance
    alignas(64) TimingRingBuffer<524288> timing_buffer_;
    alignas(64) std::array<std::atomic<uint64_t>, static_cast<size_t>(ProfileMetric::MetricCount)> counters_;
    alignas(64) std::array<std::atomic<uint64_t>, static_cast<size_t>(ProfileMetric::MetricCount)> gauges_;
    alignas(64) std::array<std::atomic<uint64_t>, static_cast<size_t>(ProfileMetric::MetricCount)> total_durations_;
    alignas(64) std::array<std::atomic<uint64_t>, static_cast<size_t>(ProfileMetric::MetricCount)> call_counts_;

    uint32_t thread_id_;
    uint32_t current_depth_;
};

/**
 * @brief Thread-local storage for metrics
 */
class ThreadMetricsStorage {
public:
    static ThreadMetricsStorage& instance() {
        static ThreadMetricsStorage instance;
        return instance;
    }

    /**
     * @brief Get thread-local metrics (creates if doesn't exist)
     */
    ThreadLocalMetrics& get_thread_metrics() {
        static thread_local std::shared_ptr<ThreadLocalMetrics> tls_metrics;

        if (!tls_metrics) {
            tls_metrics = std::make_shared<ThreadLocalMetrics>();

            // Register with global storage (shared_ptr keeps object alive)
            std::lock_guard<std::mutex> lock(mutex_);
            thread_metrics_[std::this_thread::get_id()] = tls_metrics;
        }

        return *tls_metrics;
    }

    /**
     * @brief Get all thread metrics for aggregation
     * Returns shared_ptr to ensure metrics stay alive during access
     */
    std::vector<std::shared_ptr<ThreadLocalMetrics>> get_all_metrics() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::shared_ptr<ThreadLocalMetrics>> result;
        result.reserve(thread_metrics_.size());

        for (auto& [tid, metrics] : thread_metrics_) {
            if (metrics) {
                result.push_back(metrics);
            }
        }

        return result;
    }

    /**
     * @brief Clear metrics for a thread (called on thread exit)
     */
    void clear_thread(std::thread::id tid) {
        std::lock_guard<std::mutex> lock(mutex_);
        thread_metrics_.erase(tid);
    }

    /**
     * @brief Clear all thread metrics (useful when starting new session)
     */
    void clear_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        thread_metrics_.clear();
    }

private:
    ThreadMetricsStorage() = default;

    std::mutex mutex_;
    std::unordered_map<std::thread::id, std::shared_ptr<ThreadLocalMetrics>> thread_metrics_;
};

} // namespace profiling
} // namespace mcts