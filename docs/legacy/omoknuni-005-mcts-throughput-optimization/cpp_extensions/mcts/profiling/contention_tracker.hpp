/**
 * @file contention_tracker.hpp
 * @brief Thread synchronization contention tracking
 *
 * Tracks contention on atomic operations, mutexes, and condition variables
 * to identify synchronization bottlenecks.
 */

#pragma once

#include "metrics.hpp"
#include "thread_local_metrics.hpp"
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace mcts {
namespace profiling {

/**
 * @brief Tracked compare-and-swap operation
 *
 * Wraps std::atomic::compare_exchange_weak with contention tracking.
 */
template<typename T>
class TrackedAtomic {
public:
    TrackedAtomic() : value_(0) {}
    explicit TrackedAtomic(T initial) : value_(initial) {}

    /**
     * @brief Load value
     */
    T load(std::memory_order order = std::memory_order_seq_cst) const {
        return value_.load(order);
    }

    /**
     * @brief Store value
     */
    void store(T val, std::memory_order order = std::memory_order_seq_cst) {
        value_.store(val, order);
    }

    /**
     * @brief Fetch and add
     */
    T fetch_add(T arg, std::memory_order order = std::memory_order_seq_cst) {
        return value_.fetch_add(arg, order);
    }

    /**
     * @brief Fetch and sub
     */
    T fetch_sub(T arg, std::memory_order order = std::memory_order_seq_cst) {
        return value_.fetch_sub(arg, order);
    }

    /**
     * @brief Compare-exchange-weak with contention tracking
     */
    bool compare_exchange_weak(
        T& expected,
        T desired,
        std::memory_order success_order = std::memory_order_seq_cst,
        std::memory_order failure_order = std::memory_order_seq_cst,
        ProfileMetric metric = ProfileMetric::SyncAtomicCASSuccess
    ) {
        auto& tls = get_thread_local_metrics();

        // Track CAS attempt
        tls.contention.atomic_cas_attempts.fetch_add(1, std::memory_order_relaxed);

        // Measure cycles spent in CAS
        std::uint64_t start_cycles = get_cpu_cycles();

        bool success = value_.compare_exchange_weak(
            expected, desired,
            success_order, failure_order
        );

        std::uint64_t end_cycles = get_cpu_cycles();
        std::uint64_t cycles = end_cycles - start_cycles;

        if (success) {
            tls.increment_counter(ProfileMetric::SyncAtomicCASSuccess);
        } else {
            tls.increment_counter(ProfileMetric::SyncAtomicCASFailure);
            tls.contention.atomic_cas_failures.fetch_add(1, std::memory_order_relaxed);
            tls.contention.spin_wait_cycles.fetch_add(cycles, std::memory_order_relaxed);
        }

        return success;
    }

    /**
     * @brief Compare-exchange-strong with contention tracking
     */
    bool compare_exchange_strong(
        T& expected,
        T desired,
        std::memory_order success_order = std::memory_order_seq_cst,
        std::memory_order failure_order = std::memory_order_seq_cst,
        ProfileMetric metric = ProfileMetric::SyncAtomicCASSuccess
    ) {
        auto& tls = get_thread_local_metrics();
        tls.contention.atomic_cas_attempts.fetch_add(1, std::memory_order_relaxed);

        std::uint64_t start_cycles = get_cpu_cycles();

        bool success = value_.compare_exchange_strong(
            expected, desired,
            success_order, failure_order
        );

        std::uint64_t end_cycles = get_cpu_cycles();
        std::uint64_t cycles = end_cycles - start_cycles;

        if (success) {
            tls.increment_counter(ProfileMetric::SyncAtomicCASSuccess);
        } else {
            tls.increment_counter(ProfileMetric::SyncAtomicCASFailure);
            tls.contention.atomic_cas_failures.fetch_add(1, std::memory_order_relaxed);
            tls.contention.spin_wait_cycles.fetch_add(cycles, std::memory_order_relaxed);
        }

        return success;
    }

private:
    std::atomic<T> value_;
};

/**
 * @brief Tracked mutex with lock contention profiling
 */
class TrackedMutex {
public:
    TrackedMutex() = default;

    /**
     * @brief Lock mutex with contention tracking
     */
    void lock() {
        auto& tls = get_thread_local_metrics();

        std::uint64_t start_ns = get_timestamp_ns();

        // Try lock first (fast path)
        if (mutex_.try_lock()) {
            // Got lock immediately (no contention)
            return;
        }

        // Contention detected
        tls.increment_counter(ProfileMetric::SyncMutexLockWait);

        // Blocking lock
        mutex_.lock();

        std::uint64_t end_ns = get_timestamp_ns();
        std::uint64_t wait_ns = end_ns - start_ns;

        // Record wait time
        tls.record_timing(ProfileMetric::SyncMutexLockWait, start_ns, wait_ns);
        tls.contention.mutex_lock_waits.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Try to lock without blocking
     */
    bool try_lock() {
        return mutex_.try_lock();
    }

    /**
     * @brief Unlock mutex
     */
    void unlock() {
        mutex_.unlock();
    }

private:
    std::mutex mutex_;
};

/**
 * @brief Tracked condition variable with wait profiling
 */
class TrackedConditionVariable {
public:
    TrackedConditionVariable() = default;

    /**
     * @brief Wait on condition variable with tracking
     */
    void wait(std::unique_lock<std::mutex>& lock) {
        auto& tls = get_thread_local_metrics();

        std::uint64_t start_ns = get_timestamp_ns();

        cv_.wait(lock);

        std::uint64_t end_ns = get_timestamp_ns();
        std::uint64_t wait_ns = end_ns - start_ns;

        // Record wait time
        tls.record_timing(ProfileMetric::SyncCondVarWait, start_ns, wait_ns);
        tls.contention.condvar_waits.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Wait with predicate (tracks spurious wakeups)
     */
    template<typename Predicate>
    void wait(std::unique_lock<std::mutex>& lock, Predicate pred) {
        auto& tls = get_thread_local_metrics();

        std::uint64_t start_ns = get_timestamp_ns();
        std::uint64_t spurious_wakes = 0;

        // Track spurious wakeups
        while (!pred()) {
            cv_.wait(lock);
            spurious_wakes++;
        }

        std::uint64_t end_ns = get_timestamp_ns();
        std::uint64_t wait_ns = end_ns - start_ns;

        tls.record_timing(ProfileMetric::SyncCondVarWait, start_ns, wait_ns);
        tls.contention.condvar_waits.fetch_add(1, std::memory_order_relaxed);

        if (spurious_wakes > 1) {
            tls.increment_counter(ProfileMetric::QueueCondVarSpuriousWake, spurious_wakes - 1);
            tls.contention.condvar_spurious_wakes.fetch_add(spurious_wakes - 1, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Wait with timeout
     */
    template<typename Rep, typename Period>
    std::cv_status wait_for(
        std::unique_lock<std::mutex>& lock,
        const std::chrono::duration<Rep, Period>& rel_time
    ) {
        auto& tls = get_thread_local_metrics();

        std::uint64_t start_ns = get_timestamp_ns();

        std::cv_status status = cv_.wait_for(lock, rel_time);

        std::uint64_t end_ns = get_timestamp_ns();
        std::uint64_t wait_ns = end_ns - start_ns;

        tls.record_timing(ProfileMetric::SyncCondVarWait, start_ns, wait_ns);
        tls.contention.condvar_waits.fetch_add(1, std::memory_order_relaxed);

        if (status == std::cv_status::timeout) {
            tls.increment_counter(ProfileMetric::QueueCollectTimeout);
        }

        return status;
    }

    /**
     * @brief Wait with timeout and predicate
     */
    template<typename Rep, typename Period, typename Predicate>
    bool wait_for(
        std::unique_lock<std::mutex>& lock,
        const std::chrono::duration<Rep, Period>& rel_time,
        Predicate pred
    ) {
        auto& tls = get_thread_local_metrics();

        std::uint64_t start_ns = get_timestamp_ns();

        bool result = cv_.wait_for(lock, rel_time, pred);

        std::uint64_t end_ns = get_timestamp_ns();
        std::uint64_t wait_ns = end_ns - start_ns;

        tls.record_timing(ProfileMetric::SyncCondVarWait, start_ns, wait_ns);
        tls.contention.condvar_waits.fetch_add(1, std::memory_order_relaxed);

        return result;
    }

    /**
     * @brief Notify one waiting thread
     */
    void notify_one() {
        cv_.notify_one();
    }

    /**
     * @brief Notify all waiting threads
     */
    void notify_all() {
        cv_.notify_all();
    }

private:
    std::condition_variable cv_;
};

/**
 * @brief Spin-wait tracker
 *
 * Tracks cycles spent in spin loops waiting for atomic conditions.
 */
class SpinWaitTracker {
public:
    SpinWaitTracker(ProfileMetric metric)
        : metric_(metric)
        , start_cycles_(get_cpu_cycles())
        , iterations_(0)
    {}

    ~SpinWaitTracker() {
        std::uint64_t end_cycles = get_cpu_cycles();
        std::uint64_t cycles = end_cycles - start_cycles_;

        auto& tls = get_thread_local_metrics();
        tls.contention.spin_wait_cycles.fetch_add(cycles, std::memory_order_relaxed);
        tls.contention.spin_wait_iterations.fetch_add(iterations_, std::memory_order_relaxed);
    }

    void iteration() {
        iterations_++;
    }

private:
    ProfileMetric metric_;
    std::uint64_t start_cycles_;
    std::uint64_t iterations_;
};

/**
 * @brief False sharing detector
 *
 * Tracks cache line accesses to detect false sharing between threads.
 */
class FalseSharingDetector {
public:
    struct CacheLineAccess {
        std::uint64_t address;          // Memory address
        std::uint64_t timestamp_ns;     // Access time
        std::uint8_t thread_id;         // Thread that accessed
        bool is_write;                  // Read or write
    };

    struct FalseSharingReport {
        std::uint64_t cache_line;       // Cache line address (64-byte aligned)
        std::vector<std::uint8_t> threads;  // Threads accessing this line
        std::uint64_t contention_count; // Number of conflicting accesses
        double severity;                // 0.0 to 1.0
    };

    /**
     * @brief Record cache line access
     */
    void record_access(void* ptr, bool is_write);

    /**
     * @brief Analyze accesses and detect false sharing
     */
    std::vector<FalseSharingReport> analyze();

    /**
     * @brief Clear recorded accesses
     */
    void reset();

private:
    std::vector<CacheLineAccess> accesses_;
    std::mutex mutex_;

    static constexpr std::size_t CACHE_LINE_SIZE = 64;

    std::uint64_t get_cache_line(void* ptr) {
        return reinterpret_cast<std::uint64_t>(ptr) & ~(CACHE_LINE_SIZE - 1);
    }
};

/**
 * @brief Convenience macros for tracked operations
 */
#define TRACKED_CAS(atomic, expected, desired, metric) \
    (atomic).compare_exchange_weak(expected, desired, \
        std::memory_order_release, std::memory_order_acquire, metric)

#define TRACKED_SPIN_WAIT(metric) \
    ::mcts::profiling::SpinWaitTracker __spin_tracker_##__LINE__(metric)

#define TRACKED_SPIN_ITERATION() \
    __spin_tracker_##__LINE__.iteration()

} // namespace profiling
} // namespace mcts
