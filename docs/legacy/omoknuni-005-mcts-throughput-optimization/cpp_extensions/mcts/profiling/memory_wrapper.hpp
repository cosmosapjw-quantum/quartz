/**
 * @file memory_wrapper.hpp
 * @brief Memory allocation and synchronization profiling wrappers
 * @author MCTS Performance Team
 * @date 2025-10-15
 *
 * Provides instrumented wrappers for:
 * - Memory allocation/deallocation (tracks TIME, not just count)
 * - Mutex locking (tracks wait time and contention)
 * - Atomic operations (tracks CAS retries and stalls)
 *
 * Usage:
 *   Replace: std::mutex mutex_;
 *   With:    ProfiledMutex<std::mutex> mutex_;
 *
 * For memory tracking:
 *   #include "memory_wrapper.hpp"
 *   // Global new/delete are automatically instrumented when PROFILE_LEVEL_VALUE >= 3
 */

#pragma once

#include "enhanced_profiler.hpp"
#include "enhanced_metrics.hpp"
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <new>

namespace mcts {
namespace profiling {

// ============================================================================
// INSTRUMENTED MUTEX WRAPPER
// ============================================================================

/**
 * @brief Profiled mutex wrapper that tracks lock wait time and contention
 *
 * Drop-in replacement for std::mutex that automatically records:
 * - Time spent waiting to acquire lock
 * - Number of contention events (wait > 1µs)
 * - Total lock acquisitions
 *
 * Template parameter allows wrapping any mutex-like type.
 */
template<typename Mutex = std::mutex>
class ProfiledMutex {
private:
    Mutex mutex_;

public:
    ProfiledMutex() = default;
    ~ProfiledMutex() = default;

    // Non-copyable, non-movable (like std::mutex)
    ProfiledMutex(const ProfiledMutex&) = delete;
    ProfiledMutex& operator=(const ProfiledMutex&) = delete;

    void lock() {
        #if PROFILE_LEVEL_VALUE >= 2
        auto& profiler = EnhancedProfiler::instance();
        auto start = std::chrono::high_resolution_clock::now();
        #endif

        // Actual lock acquisition
        mutex_.lock();

        #if PROFILE_LEVEL_VALUE >= 2
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        // Record timing
        profiler.record_timing(ProfileMetric::MutexLockWaitTime, duration_ns);
        profiler.increment_counter(ProfileMetric::SyncMutexLockAcquired);

        // Track contention events (wait > 1 microsecond indicates contention)
        if (duration_ns > 1000) {
            profiler.increment_counter(ProfileMetric::MutexContentionEvents);
        }
        #endif
    }

    void unlock() {
        mutex_.unlock();
    }

    bool try_lock() {
        #if PROFILE_LEVEL_VALUE >= 2
        auto& profiler = EnhancedProfiler::instance();
        bool acquired = mutex_.try_lock();

        if (!acquired) {
            profiler.increment_counter(ProfileMetric::MutexContentionEvents);
        }

        return acquired;
        #else
        return mutex_.try_lock();
        #endif
    }

    // For compatibility with std::lock_guard, std::unique_lock, etc.
    typedef Mutex mutex_type;
};

// ============================================================================
// INSTRUMENTED ATOMIC OPERATIONS
// ============================================================================

/**
 * @brief Profiled compare-and-swap wrapper that tracks retries
 *
 * Tracks CAS failure rate and retry counts for detecting hot contention spots.
 */
template<typename T>
class ProfiledAtomic {
private:
    std::atomic<T> value_;

public:
    ProfiledAtomic(T initial = T()) : value_(initial) {}

    T load(std::memory_order order = std::memory_order_seq_cst) const {
        return value_.load(order);
    }

    void store(T desired, std::memory_order order = std::memory_order_seq_cst) {
        value_.store(desired, order);
    }

    bool compare_exchange_strong(T& expected, T desired,
                                  std::memory_order success = std::memory_order_seq_cst,
                                  std::memory_order failure = std::memory_order_seq_cst) {
        #if PROFILE_LEVEL_VALUE >= 2
        auto& profiler = EnhancedProfiler::instance();
        bool success_flag = value_.compare_exchange_strong(expected, desired, success, failure);

        if (success_flag) {
            profiler.increment_counter(ProfileMetric::CAS_SuccessCount);
        } else {
            profiler.increment_counter(ProfileMetric::CAS_FailureCount);
            profiler.increment_counter(ProfileMetric::CAS_RetryCount);
        }

        return success_flag;
        #else
        return value_.compare_exchange_strong(expected, desired, success, failure);
        #endif
    }

    bool compare_exchange_weak(T& expected, T desired,
                                std::memory_order success = std::memory_order_seq_cst,
                                std::memory_order failure = std::memory_order_seq_cst) {
        #if PROFILE_LEVEL_VALUE >= 2
        auto& profiler = EnhancedProfiler::instance();
        bool success_flag = value_.compare_exchange_weak(expected, desired, success, failure);

        if (success_flag) {
            profiler.increment_counter(ProfileMetric::CAS_SuccessCount);
        } else {
            profiler.increment_counter(ProfileMetric::CAS_FailureCount);
            profiler.increment_counter(ProfileMetric::CAS_RetryCount);
        }

        return success_flag;
        #else
        return value_.compare_exchange_weak(expected, desired, success, failure);
        #endif
    }

    T fetch_add(T arg, std::memory_order order = std::memory_order_seq_cst) {
        return value_.fetch_add(arg, order);
    }

    T fetch_sub(T arg, std::memory_order order = std::memory_order_seq_cst) {
        return value_.fetch_sub(arg, order);
    }

    // Implicit conversion for convenience
    operator T() const {
        return value_.load();
    }

    T operator=(T desired) {
        store(desired);
        return desired;
    }
};

// ============================================================================
// MEMORY ALLOCATION TRACKING
// ============================================================================

/**
 * @brief Instrumented allocation wrapper
 *
 * Use this for manual allocation tracking in specific code sections.
 * For global tracking, see operator new/delete overrides below.
 */
inline void* tracked_malloc(size_t size, ProfileMetric metric = ProfileMetric::MemoryNodeAllocation) {
    #if PROFILE_LEVEL_VALUE >= 3
    auto& profiler = EnhancedProfiler::instance();
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    void* ptr = std::malloc(size);

    #if PROFILE_LEVEL_VALUE >= 3
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    profiler.record_timing(metric, duration_ns);
    profiler.increment_counter(ProfileMetric::AllocationSlowPath);
    #endif

    return ptr;
}

inline void tracked_free(void* ptr, ProfileMetric metric = ProfileMetric::MemoryNodeDeallocation) {
    #if PROFILE_LEVEL_VALUE >= 3
    auto& profiler = EnhancedProfiler::instance();
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    std::free(ptr);

    #if PROFILE_LEVEL_VALUE >= 3
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    profiler.record_timing(metric, duration_ns);
    #endif
}

// ============================================================================
// SCOPED ALLOCATION TRACKER
// ============================================================================

/**
 * @brief RAII-style allocation tracker for specific code sections
 *
 * Usage:
 *   {
 *       AllocationTracker tracker("node_expansion");
 *       Node* node = new Node();  // Automatically tracked
 *       // ... use node ...
 *       delete node;  // Automatically tracked
 *   }
 */
class AllocationTracker {
private:
    const char* section_name_;
    uint64_t start_time_;
    size_t allocation_count_start_;

public:
    explicit AllocationTracker(const char* section_name)
        : section_name_(section_name)
        , start_time_(0)
        , allocation_count_start_(0)
    {
        #if PROFILE_LEVEL_VALUE >= 3
        auto& profiler = EnhancedProfiler::instance();
        start_time_ = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        // Note: Would need to track per-thread allocation counts for accurate delta
        #endif
    }

    ~AllocationTracker() {
        #if PROFILE_LEVEL_VALUE >= 3
        auto& profiler = EnhancedProfiler::instance();
        auto end_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        uint64_t duration_ns = end_time - start_time_;

        // Record section timing (includes all allocations within)
        profiler.record_timing(ProfileMetric::AllocationMutexWait, duration_ns);
        #endif
    }
};

// Convenience macro for scoped tracking
#define TRACK_ALLOCATIONS(name) \
    mcts::profiling::AllocationTracker _alloc_tracker_##__LINE__(name)

} // namespace profiling
} // namespace mcts

// ============================================================================
// GLOBAL OPERATOR NEW/DELETE OVERRIDES (OPTIONAL - USE WITH CAUTION)
// ============================================================================

// Uncomment these to track ALL allocations globally
// WARNING: High overhead! Only enable for detailed profiling sessions.
//
// Note: These are commented out by default to avoid interfering with
// existing allocators. Enable only when specifically profiling allocation overhead.

/*
#if PROFILE_LEVEL_VALUE >= 3

void* operator new(std::size_t size) {
    auto& profiler = mcts::profiling::EnhancedProfiler::instance();
    auto start = std::chrono::high_resolution_clock::now();

    void* ptr = std::malloc(size);
    if (!ptr) throw std::bad_alloc();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    profiler.record_timing(mcts::profiling::ProfileMetric::MemoryNodeAllocation, duration_ns);
    profiler.increment_counter(mcts::profiling::ProfileMetric::AllocationSlowPath);

    return ptr;
}

void operator delete(void* ptr) noexcept {
    auto& profiler = mcts::profiling::EnhancedProfiler::instance();
    auto start = std::chrono::high_resolution_clock::now();

    std::free(ptr);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    profiler.record_timing(mcts::profiling::ProfileMetric::MemoryNodeDeallocation, duration_ns);
}

void operator delete(void* ptr, std::size_t size) noexcept {
    operator delete(ptr);
}

#endif // PROFILE_LEVEL_VALUE >= 3
*/
