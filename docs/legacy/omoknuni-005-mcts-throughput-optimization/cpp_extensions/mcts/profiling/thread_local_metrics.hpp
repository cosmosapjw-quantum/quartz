/**
 * @file thread_local_metrics.hpp
 * @brief Thread-local metric storage for lock-free profiling
 *
 * Each thread maintains its own metric buffer to avoid contention.
 * Lock-free ring buffers for timing samples, atomic counters for events.
 */

#pragma once

#include "metrics.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <chrono>

namespace mcts {
namespace profiling {

/**
 * @brief Hardware performance counter snapshot
 */
struct HardwareCounters {
    std::uint64_t cycles{0};
    std::uint64_t instructions{0};
    std::uint64_t cache_refs{0};
    std::uint64_t cache_misses{0};
    std::uint64_t branch_instructions{0};
    std::uint64_t branch_misses{0};
    std::uint64_t l1d_cache_misses{0};
    std::uint64_t l1i_cache_misses{0};
    std::uint64_t llc_misses{0};
    std::uint64_t tlb_misses{0};

    // Derived metrics
    double ipc() const {
        return cycles > 0 ? static_cast<double>(instructions) / cycles : 0.0;
    }

    double cache_hit_rate() const {
        return cache_refs > 0 ? 1.0 - (static_cast<double>(cache_misses) / cache_refs) : 0.0;
    }

    double branch_miss_rate() const {
        return branch_instructions > 0 ? static_cast<double>(branch_misses) / branch_instructions : 0.0;
    }
};

/**
 * @brief Memory allocation statistics
 */
struct MemoryStats {
    std::atomic<std::uint64_t> allocations{0};
    std::atomic<std::uint64_t> deallocations{0};
    std::atomic<std::uint64_t> bytes_allocated{0};
    std::atomic<std::uint64_t> bytes_freed{0};
    std::atomic<std::uint64_t> arena_fast_path{0};      // Bump pointer
    std::atomic<std::uint64_t> arena_slow_path{0};      // New chunk
    std::atomic<std::uint64_t> arena_freelist_hits{0};  // Free list reuse
    std::atomic<std::uint64_t> cache_line_accesses{0};
};

/**
 * @brief Thread synchronization contention statistics
 */
struct ContentionStats {
    std::atomic<std::uint64_t> atomic_cas_attempts{0};
    std::atomic<std::uint64_t> atomic_cas_failures{0};
    std::atomic<std::uint64_t> spin_wait_cycles{0};
    std::atomic<std::uint64_t> spin_wait_iterations{0};
    std::atomic<std::uint64_t> mutex_lock_waits{0};
    std::atomic<std::uint64_t> condvar_waits{0};
    std::atomic<std::uint64_t> condvar_spurious_wakes{0};
};

/**
 * @brief Single timing sample with context
 */
struct TimingSample {
    std::uint64_t timestamp_ns;     // Absolute timestamp
    std::uint64_t duration_ns;      // Duration of operation
    ProfileMetric metric_id;        // Which metric
    std::uint16_t depth;            // Call stack depth (for hierarchical)
    std::uint8_t thread_id;         // Thread ID (for visualization)

    TimingSample()
        : timestamp_ns(0), duration_ns(0), metric_id(ProfileMetric::Count), depth(0), thread_id(0) {}

    TimingSample(std::uint64_t ts, std::uint64_t dur, ProfileMetric metric, std::uint16_t d = 0, std::uint8_t tid = 0)
        : timestamp_ns(ts), duration_ns(dur), metric_id(metric), depth(d), thread_id(tid) {}
};

/**
 * @brief Thread-local metric storage (lock-free)
 *
 * Cache-line aligned to prevent false sharing between threads.
 * Each thread has its own instance accessed via TLS.
 */
struct alignas(64) ThreadLocalMetrics {
    // Thread identification
    std::uint64_t thread_id;
    std::uint64_t thread_start_ns;

    // Lock-free ring buffer for timing samples
    // FIXED 2025-10-15: Increased from 4096 to 524288 to prevent buffer overflow
    // - Old size: 4,096 samples → captured only 11.4% of data (55,904 samples dropped)
    // - New size: 524,288 samples → supports 17,476 simulations @ 30 samples/sim
    // - Memory cost: 16.4 MB per thread (192 MB for 12 threads) - acceptable for complete profiling
    static constexpr std::size_t SAMPLE_BUFFER_SIZE = 524288;
    std::array<TimingSample, SAMPLE_BUFFER_SIZE> timing_samples;
    std::atomic<std::size_t> timing_head{0};  // Write pointer
    std::atomic<std::size_t> timing_tail{0};  // Read pointer (for consumer)

    // Per-metric counters (lock-free atomic)
    static constexpr std::size_t NUM_METRICS = static_cast<std::size_t>(ProfileMetric::Count);
    std::array<std::atomic<std::uint64_t>, NUM_METRICS> counters;

    // Per-metric gauges (current values, not cumulative)
    std::array<std::atomic<std::int64_t>, NUM_METRICS> gauges;

    // Hardware counter snapshots
    HardwareCounters hw_start;
    HardwareCounters hw_current;

    // Memory profiling
    MemoryStats memory;

    // Contention tracking
    ContentionStats contention;

    // Current profiler depth (for hierarchical timing)
    std::atomic<std::uint16_t> profiler_depth{0};

    // Statistics
    std::atomic<std::uint64_t> samples_written{0};
    std::atomic<std::uint64_t> samples_dropped{0};  // Ring buffer overflow

    ThreadLocalMetrics() : thread_id(0), thread_start_ns(0) {
        // Initialize counters to zero
        for (auto& counter : counters) {
            counter.store(0, std::memory_order_relaxed);
        }
        for (auto& gauge : gauges) {
            gauge.store(0, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Record timing sample (lock-free)
     */
    void record_timing(ProfileMetric metric, std::uint64_t timestamp_ns, std::uint64_t duration_ns) {
        std::size_t head = timing_head.load(std::memory_order_relaxed);
        std::size_t tail = timing_tail.load(std::memory_order_acquire);

        // Check if buffer is full
        std::size_t next_head = (head + 1) % SAMPLE_BUFFER_SIZE;
        if (next_head == tail) {
            samples_dropped.fetch_add(1, std::memory_order_relaxed);
            return;  // Drop sample (buffer full)
        }

        // Write sample
        std::uint16_t depth = profiler_depth.load(std::memory_order_relaxed);
        timing_samples[head] = TimingSample(timestamp_ns, duration_ns, metric, depth, static_cast<std::uint8_t>(thread_id));

        // Advance head pointer
        timing_head.store(next_head, std::memory_order_release);
        samples_written.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Increment counter (lock-free)
     */
    void increment_counter(ProfileMetric metric, std::uint64_t value = 1) {
        std::size_t idx = static_cast<std::size_t>(metric);
        if (idx < NUM_METRICS) {
            counters[idx].fetch_add(value, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Set gauge value (lock-free)
     */
    void set_gauge(ProfileMetric metric, std::int64_t value) {
        std::size_t idx = static_cast<std::size_t>(metric);
        if (idx < NUM_METRICS) {
            gauges[idx].store(value, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Update gauge atomically (add delta)
     */
    void update_gauge(ProfileMetric metric, std::int64_t delta) {
        std::size_t idx = static_cast<std::size_t>(metric);
        if (idx < NUM_METRICS) {
            gauges[idx].fetch_add(delta, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Enter profiler scope (increment depth)
     */
    void enter_scope() {
        profiler_depth.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Exit profiler scope (decrement depth)
     */
    void exit_scope() {
        profiler_depth.fetch_sub(1, std::memory_order_relaxed);
    }

    /**
     * @brief Get current profiler depth
     */
    std::uint16_t get_depth() const {
        return profiler_depth.load(std::memory_order_relaxed);
    }

    /**
     * @brief Read counter value (snapshot)
     */
    std::uint64_t read_counter(ProfileMetric metric) const {
        std::size_t idx = static_cast<std::size_t>(metric);
        if (idx < NUM_METRICS) {
            return counters[idx].load(std::memory_order_relaxed);
        }
        return 0;
    }

    /**
     * @brief Read gauge value (snapshot)
     */
    std::int64_t read_gauge(ProfileMetric metric) const {
        std::size_t idx = static_cast<std::size_t>(metric);
        if (idx < NUM_METRICS) {
            return gauges[idx].load(std::memory_order_relaxed);
        }
        return 0;
    }

    /**
     * @brief Consume timing samples (moves tail pointer)
     *
     * Returns number of samples consumed. Consumer should process
     * timing_samples[tail...new_tail] and then call this.
     */
    std::size_t consume_samples(std::size_t max_samples) {
        std::size_t tail = timing_tail.load(std::memory_order_relaxed);
        std::size_t head = timing_head.load(std::memory_order_acquire);

        if (tail == head) {
            return 0;  // No samples available
        }

        std::size_t available;
        if (head > tail) {
            available = head - tail;
        } else {
            available = (SAMPLE_BUFFER_SIZE - tail) + head;
        }

        std::size_t to_consume = std::min(available, max_samples);
        std::size_t new_tail = (tail + to_consume) % SAMPLE_BUFFER_SIZE;

        timing_tail.store(new_tail, std::memory_order_release);
        return to_consume;
    }

    /**
     * @brief Get number of available samples
     */
    std::size_t available_samples() const {
        std::size_t head = timing_head.load(std::memory_order_acquire);
        std::size_t tail = timing_tail.load(std::memory_order_relaxed);

        if (head >= tail) {
            return head - tail;
        } else {
            return (SAMPLE_BUFFER_SIZE - tail) + head;
        }
    }

    /**
     * @brief Reset all metrics (for new profiling session)
     */
    void reset() {
        timing_head.store(0, std::memory_order_relaxed);
        timing_tail.store(0, std::memory_order_relaxed);
        samples_written.store(0, std::memory_order_relaxed);
        samples_dropped.store(0, std::memory_order_relaxed);
        profiler_depth.store(0, std::memory_order_relaxed);

        for (auto& counter : counters) {
            counter.store(0, std::memory_order_relaxed);
        }
        for (auto& gauge : gauges) {
            gauge.store(0, std::memory_order_relaxed);
        }

        std::memset(&hw_start, 0, sizeof(hw_start));
        std::memset(&hw_current, 0, sizeof(hw_current));
    }
};

/**
 * @brief Get thread-local metrics for current thread
 *
 * Lazily initializes metrics on first access per thread.
 */
ThreadLocalMetrics& get_thread_local_metrics();

/**
 * @brief Destroy thread-local metrics (call before thread exit)
 */
void destroy_thread_local_metrics();

/**
 * @brief Get current timestamp in nanoseconds
 */
inline std::uint64_t get_timestamp_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()
    );
}

/**
 * @brief Get CPU cycle count (x86_64 RDTSC)
 */
inline std::uint64_t get_cpu_cycles() {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
#else
    return 0;  // Not supported on this architecture
#endif
}

/**
 * @brief Get CPU cycle count with serialization (prevents reordering)
 */
inline std::uint64_t get_cpu_cycles_serialized() {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    __asm__ __volatile__(
        "cpuid\n\t"
        "rdtsc\n\t"
        : "=a"(lo), "=d"(hi)
        :: "%rbx", "%rcx"
    );
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
#else
    return 0;
#endif
}

} // namespace profiling
} // namespace mcts
