/**
 * @file profiler.hpp
 * @brief Main profiling interface with scoped timing and hierarchical tracking
 *
 * Provides high-level profiling API with minimal overhead:
 * - Scoped timing with RAII
 * - Hierarchical operation tracking
 * - Compile-time enable/disable
 * - Sampling mode for ultra-low overhead
 */

#pragma once

#include "metrics.hpp"
#include "thread_local_metrics.hpp"
#include "hardware_counters.hpp"
#include <atomic>
#include <memory>
#include <vector>

namespace mcts {
namespace profiling {

/**
 * @brief Profiling levels (compile-time configuration)
 */
enum class ProfileLevel : std::uint8_t {
    None = 0,       // Zero overhead, all profiling disabled
    Basic = 1,      // Timers only (~0.1% overhead)
    Detailed = 2,   // + hardware counters (~0.5% overhead)
    Full = 3,       // + memory tracking (~1.0% overhead)
};

// Compile-time profiling level
#ifndef PROFILE_LEVEL
    #define PROFILE_LEVEL 1  // Basic by default
#endif

/**
 * @brief Global profiler singleton
 *
 * Manages profiling sessions, thread registry, and aggregation.
 */
class Profiler {
public:
    /**
     * @brief Get global profiler instance
     */
    static Profiler& instance();

    /**
     * @brief Enable profiling
     */
    void enable();

    /**
     * @brief Disable profiling
     */
    void disable();

    /**
     * @brief Check if profiling is enabled
     */
    bool is_enabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * @brief Set sampling rate (1 = profile all, 100 = profile 1 in 100)
     */
    void set_sampling_rate(std::uint32_t rate) {
        sampling_rate_.store(rate, std::memory_order_relaxed);
    }

    /**
     * @brief Get current sampling rate
     */
    std::uint32_t get_sampling_rate() const {
        return sampling_rate_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Check if current invocation should be sampled
     */
    bool should_sample() const;

    /**
     * @brief Start profiling session
     */
    void start_session(const char* name);

    /**
     * @brief Stop profiling session and generate report
     */
    void stop_session();

    /**
     * @brief Reset all metrics
     */
    void reset();

    /**
     * @brief Register thread for profiling
     */
    void register_thread(std::uint64_t thread_id);

    /**
     * @brief Unregister thread
     */
    void unregister_thread(std::uint64_t thread_id);

    /**
     * @brief Get all thread-local metrics (for aggregation)
     */
    std::vector<ThreadLocalMetrics*> get_all_thread_metrics();

    /**
     * @brief Enable hardware counter tracking
     */
    void enable_hardware_counters();

    /**
     * @brief Disable hardware counter tracking
     */
    void disable_hardware_counters();

    /**
     * @brief Check if hardware counters are enabled
     */
    bool hardware_counters_enabled() const {
        return hw_counters_enabled_.load(std::memory_order_acquire);
    }

    /**
     * @brief Export profiling data to JSON
     */
    void export_json(const char* filename);

    /**
     * @brief Export profiling data to Chrome Trace format
     */
    void export_chrome_trace(const char* filename);

    /**
     * @brief Get profiling overhead estimate (percentage)
     */
    double get_overhead_estimate() const;

private:
    Profiler();
    ~Profiler();

    std::atomic<bool> enabled_{false};
    std::atomic<std::uint32_t> sampling_rate_{1};  // 1 = profile all
    std::atomic<bool> hw_counters_enabled_{false};

    // Session management
    std::string session_name_;
    std::uint64_t session_start_ns_{0};
    std::uint64_t session_end_ns_{0};

    // Thread registry (for aggregation)
    std::mutex thread_registry_mutex_;
    std::vector<ThreadLocalMetrics*> thread_metrics_;
    std::vector<std::uint64_t> thread_ids_;

    // Hardware counters
    std::unique_ptr<HardwareCounterReader> hw_counter_reader_;
};

/**
 * @brief Scoped profiler for automatic timing
 *
 * RAII-based timer that records duration on destruction.
 * Supports pause/resume and hierarchical tracking.
 */
class ScopedProfiler {
public:
    /**
     * @brief Start timing for metric
     */
    explicit ScopedProfiler(ProfileMetric metric)
        : metric_(metric)
        , enabled_(Profiler::instance().is_enabled())
        , should_sample_(enabled_ && Profiler::instance().should_sample())
        , start_ns_(0)
        , pause_ns_(0)
        , paused_(false)
    {
        if (should_sample_) {
            auto& tls = get_thread_local_metrics();
            tls.enter_scope();
            start_ns_ = get_timestamp_ns();
        }
    }

    /**
     * @brief Stop timing and record duration
     */
    ~ScopedProfiler() {
        if (should_sample_ && !paused_) {
            std::uint64_t end_ns = get_timestamp_ns();
            std::uint64_t duration_ns = end_ns - start_ns_;

            auto& tls = get_thread_local_metrics();
            tls.record_timing(metric_, start_ns_, duration_ns);
            tls.exit_scope();
        }
    }

    /**
     * @brief Pause timing (exclude nested operations)
     */
    void pause() {
        if (should_sample_ && !paused_) {
            pause_ns_ = get_timestamp_ns();
            paused_ = true;
        }
    }

    /**
     * @brief Resume timing
     */
    void resume() {
        if (should_sample_ && paused_) {
            std::uint64_t resume_ns = get_timestamp_ns();
            start_ns_ += (resume_ns - pause_ns_);  // Adjust start time
            paused_ = false;
        }
    }

    // Non-copyable, non-movable
    ScopedProfiler(const ScopedProfiler&) = delete;
    ScopedProfiler& operator=(const ScopedProfiler&) = delete;

private:
    ProfileMetric metric_;
    bool enabled_;
    bool should_sample_;
    std::uint64_t start_ns_;
    std::uint64_t pause_ns_;
    bool paused_;
};

/**
 * @brief Scoped profiler with CPU cycle counting
 *
 * More accurate for very short operations (<1μs) but has
 * higher overhead due to CPUID serialization.
 */
class ScopedCycleProfiler {
public:
    explicit ScopedCycleProfiler(ProfileMetric metric)
        : metric_(metric)
        , enabled_(Profiler::instance().is_enabled())
        , should_sample_(enabled_ && Profiler::instance().should_sample())
        , start_cycles_(0)
    {
        if (should_sample_) {
            auto& tls = get_thread_local_metrics();
            tls.enter_scope();
            start_cycles_ = get_cpu_cycles_serialized();
        }
    }

    ~ScopedCycleProfiler() {
        if (should_sample_) {
            std::uint64_t end_cycles = get_cpu_cycles_serialized();
            std::uint64_t cycles = end_cycles - start_cycles_;

            // Convert cycles to nanoseconds (approximate, assumes 3.5 GHz)
            // TODO: Detect CPU frequency dynamically
            constexpr double CPU_GHZ = 3.5;
            std::uint64_t duration_ns = static_cast<std::uint64_t>(cycles / CPU_GHZ);

            auto& tls = get_thread_local_metrics();
            tls.record_timing(metric_, get_timestamp_ns(), duration_ns);
            tls.exit_scope();
        }
    }

    ScopedCycleProfiler(const ScopedCycleProfiler&) = delete;
    ScopedCycleProfiler& operator=(const ScopedCycleProfiler&) = delete;

private:
    ProfileMetric metric_;
    bool enabled_;
    bool should_sample_;
    std::uint64_t start_cycles_;
};

/**
 * @brief Manual profiler for fine-grained control
 */
class ManualProfiler {
public:
    ManualProfiler() : start_ns_(0), metric_(ProfileMetric::Count), running_(false) {}

    void start(ProfileMetric metric) {
        if (Profiler::instance().is_enabled() && Profiler::instance().should_sample()) {
            metric_ = metric;
            start_ns_ = get_timestamp_ns();
            running_ = true;

            auto& tls = get_thread_local_metrics();
            tls.enter_scope();
        }
    }

    void stop() {
        if (running_) {
            std::uint64_t end_ns = get_timestamp_ns();
            std::uint64_t duration_ns = end_ns - start_ns_;

            auto& tls = get_thread_local_metrics();
            tls.record_timing(metric_, start_ns_, duration_ns);
            tls.exit_scope();

            running_ = false;
        }
    }

private:
    std::uint64_t start_ns_;
    ProfileMetric metric_;
    bool running_;
};

/**
 * @brief Record counter increment
 */
inline void record_counter(ProfileMetric metric, std::uint64_t value = 1) {
    if (Profiler::instance().is_enabled()) {
        auto& tls = get_thread_local_metrics();
        tls.increment_counter(metric, value);
    }
}

/**
 * @brief Record gauge value
 */
inline void record_gauge(ProfileMetric metric, std::int64_t value) {
    if (Profiler::instance().is_enabled()) {
        auto& tls = get_thread_local_metrics();
        tls.set_gauge(metric, value);
    }
}

/**
 * @brief Update gauge (add delta)
 */
inline void update_gauge(ProfileMetric metric, std::int64_t delta) {
    if (Profiler::instance().is_enabled()) {
        auto& tls = get_thread_local_metrics();
        tls.update_gauge(metric, delta);
    }
}

// ============================================================
// Macros for conditional compilation
// ============================================================

#if PROFILE_LEVEL >= 1
    #define PROFILE_SCOPE(metric) ::mcts::profiling::ScopedProfiler __profiler_##__LINE__(metric)
    #define PROFILE_SCOPE_CYCLES(metric) ::mcts::profiling::ScopedCycleProfiler __profiler_##__LINE__(metric)
    #define PROFILE_COUNTER(metric, value) ::mcts::profiling::record_counter(metric, value)
    #define PROFILE_GAUGE(metric, value) ::mcts::profiling::record_gauge(metric, value)
    #define PROFILE_UPDATE_GAUGE(metric, delta) ::mcts::profiling::update_gauge(metric, delta)
#else
    #define PROFILE_SCOPE(metric) do {} while(0)
    #define PROFILE_SCOPE_CYCLES(metric) do {} while(0)
    #define PROFILE_COUNTER(metric, value) do {} while(0)
    #define PROFILE_GAUGE(metric, value) do {} while(0)
    #define PROFILE_UPDATE_GAUGE(metric, delta) do {} while(0)
#endif

#if PROFILE_LEVEL >= 2
    #define PROFILE_HW_COUNTER(metric, value) /* TODO: Implement */
#else
    #define PROFILE_HW_COUNTER(metric, value) do {} while(0)
#endif

#if PROFILE_LEVEL >= 3
    #define PROFILE_MEMORY_ALLOC(size) /* TODO: Implement */
    #define PROFILE_MEMORY_FREE(size) /* TODO: Implement */
#else
    #define PROFILE_MEMORY_ALLOC(size) do {} while(0)
    #define PROFILE_MEMORY_FREE(size) do {} while(0)
#endif

/**
 * @brief Named scope for Chrome Trace visualization
 */
#define PROFILE_NAMED_SCOPE(name, metric) \
    ::mcts::profiling::ScopedProfiler __profiler_##__LINE__(metric)

/**
 * @brief Function-level profiling (uses __FUNCTION__)
 */
#define PROFILE_FUNCTION(metric) \
    ::mcts::profiling::ScopedProfiler __profiler_function(metric)

} // namespace profiling
} // namespace mcts
