/**
 * @file enhanced_profiler.hpp
 * @brief Main profiler API with scoped timing and metrics collection
 * @author MCTS Performance Team
 * @date 2024
 */

#pragma once

#include "enhanced_metrics.hpp"
#include "thread_metrics.hpp"
#include "statistical_analyzer.hpp"
#include <chrono>
#include <memory>
#include <fstream>
#include <sstream>

namespace mcts {
namespace profiling {

/**
 * @brief Get current time in nanoseconds
 */
inline uint64_t get_time_ns() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}

/**
 * @brief RAII-based scoped profiler
 */
class ScopedProfiler {
public:
    explicit ScopedProfiler(ProfileMetric metric, bool enabled = true)
        : metric_(metric), enabled_(enabled), start_ns_(0) {
        if (enabled_) {
            auto& metrics = ThreadMetricsStorage::instance().get_thread_metrics();
            metrics.enter_scope();
            start_ns_ = get_time_ns();
        }
    }

    ~ScopedProfiler() {
        if (enabled_ && start_ns_ > 0) {
            uint64_t duration_ns = get_time_ns() - start_ns_;
            auto& metrics = ThreadMetricsStorage::instance().get_thread_metrics();
            metrics.record_timing(metric_, start_ns_, duration_ns);
            metrics.exit_scope();
        }
    }

    // Delete copy/move to ensure RAII semantics
    ScopedProfiler(const ScopedProfiler&) = delete;
    ScopedProfiler& operator=(const ScopedProfiler&) = delete;
    ScopedProfiler(ScopedProfiler&&) = delete;
    ScopedProfiler& operator=(ScopedProfiler&&) = delete;

private:
    ProfileMetric metric_;
    bool enabled_;
    uint64_t start_ns_;
};

/**
 * @brief Main profiler singleton
 */
class EnhancedProfiler {
public:
    static EnhancedProfiler& instance();

    /**
     * @brief Enable/disable profiling
     */
    void set_enabled(bool enabled) {
        enabled_.store(enabled, std::memory_order_release);
    }

    bool is_enabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * @brief Set profiling level
     */
    void set_level(ProfileLevel level) {
        level_ = level;
    }

    ProfileLevel get_level() const {
        return level_;
    }

    /**
     * @brief Start a profiling session
     */
    void start_session(const std::string& name);

    /**
     * @brief Stop current session
     */
    void stop_session();

    /**
     * @brief Reset all metrics
     */
    void reset_metrics() {
        auto all_metrics = ThreadMetricsStorage::instance().get_all_metrics();
        for (const auto& metrics : all_metrics) {
            if (metrics) {
                metrics->reset();
            }
        }
    }

    /**
     * @brief Record a counter increment
     */
    void increment_counter(ProfileMetric metric, uint64_t value = 1) {
        if (!is_enabled()) return;
        auto& metrics = ThreadMetricsStorage::instance().get_thread_metrics();
        metrics.increment_counter(metric, value);
    }

    /**
     * @brief Update a gauge value
     */
    void update_gauge(ProfileMetric metric, uint64_t value) {
        if (!is_enabled()) return;
        auto& metrics = ThreadMetricsStorage::instance().get_thread_metrics();
        metrics.update_gauge(metric, value);
    }

    /**
     * @brief Generate profiling report
     */
    ProfileReport generate_report() const;

    /**
     * @brief Export report to JSON
     */
    void export_json(const std::string& filename) const;

    /**
     * @brief Export to Chrome Trace format
     */
    void export_chrome_trace(const std::string& filename) const;

    /**
     * @brief Export to Markdown report
     */
    void export_markdown(const std::string& filename) const;

    /**
     * @brief Print summary to console
     */
    void print_summary() const;

private:
    EnhancedProfiler() : enabled_(false), level_(ProfileLevel::Basic), session_start_ns_(0), session_end_ns_(0), has_cached_report_(false) {}

    /**
     * @brief Internal method to generate report from current metrics
     */
    ProfileReport generate_report_internal() const;

    std::atomic<bool> enabled_;
    ProfileLevel level_;
    std::string session_name_;
    uint64_t session_start_ns_;
    uint64_t session_end_ns_;
    mutable bool has_cached_report_;
    mutable ProfileReport cached_report_;
};

// Convenience macros for profiling
#if PROFILE_LEVEL_VALUE > 0

    #define PROFILE_SCOPE(metric) \
        ::mcts::profiling::ScopedProfiler _prof_##__LINE__(metric, \
            ::mcts::profiling::EnhancedProfiler::instance().is_enabled())

    #define PROFILE_COUNTER(metric, value) \
        ::mcts::profiling::EnhancedProfiler::instance().increment_counter(metric, value)

    #define PROFILE_GAUGE(metric, value) \
        ::mcts::profiling::EnhancedProfiler::instance().update_gauge(metric, value)

#else
    // No-op when profiling disabled
    #define PROFILE_SCOPE(metric) ((void)0)
    #define PROFILE_COUNTER(metric, value) ((void)0)
    #define PROFILE_GAUGE(metric, value) ((void)0)
#endif

} // namespace profiling
} // namespace mcts