/**
 * @file enhanced_profiler.cpp
 * @brief Implementation of comprehensive MCTS profiler
 */

#include "enhanced_profiler.hpp"
#include "thread_metrics.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>

namespace mcts {
namespace profiling {

// Statistical analyzer implementation
MetricStats StatisticalAnalyzer::compute(std::vector<uint64_t>& samples) {
    MetricStats stats;

    if (samples.empty()) {
        return stats;
    }

    // Sort for percentiles
    std::sort(samples.begin(), samples.end());

    stats.count = samples.size();
    stats.total = std::accumulate(samples.begin(), samples.end(), 0ULL);
    stats.mean = static_cast<double>(stats.total) / stats.count;

    stats.min = static_cast<double>(samples.front());
    stats.max = static_cast<double>(samples.back());

    // Compute percentiles
    auto percentile = [&samples](double p) -> double {
        size_t idx = static_cast<size_t>(p * (samples.size() - 1));
        return static_cast<double>(samples[idx]);
    };

    stats.p50 = percentile(0.50);
    stats.p75 = percentile(0.75);
    stats.p90 = percentile(0.90);
    stats.p95 = percentile(0.95);
    stats.p99 = percentile(0.99);
    stats.p999 = percentile(0.999);

    // Compute standard deviation
    double sum_sq = 0.0;
    for (uint64_t sample : samples) {
        double diff = static_cast<double>(sample) - stats.mean;
        sum_sq += diff * diff;
    }
    stats.stddev = std::sqrt(sum_sq / stats.count);

    return stats;
}

// EnhancedProfiler implementation
EnhancedProfiler& EnhancedProfiler::instance() {
    static EnhancedProfiler instance;
    return instance;
}

void EnhancedProfiler::start_session(const std::string& name) {
    session_name_ = name;
    session_start_ns_ = get_time_ns();
    reset_metrics();
    has_cached_report_ = false;
    set_enabled(true);

    std::cout << "Profiling session started: " << name << std::endl;
}

void EnhancedProfiler::stop_session() {
    session_end_ns_ = get_time_ns();

    // Generate report BEFORE disabling profiler, while threads might still be active
    cached_report_ = generate_report_internal();
    has_cached_report_ = true;

    set_enabled(false);

    double duration_ms = (session_end_ns_ - session_start_ns_) / 1e6;
    std::cout << "Profiling session ended. Duration: "
              << std::fixed << std::setprecision(2) << duration_ms << "ms" << std::endl;
}

ProfileReport EnhancedProfiler::generate_report() const {
    // Return cached report if available
    if (has_cached_report_) {
        return cached_report_;
    }
    return generate_report_internal();
}

ProfileReport EnhancedProfiler::generate_report_internal() const {
    ProfileReport report;
    report.session_name = session_name_;
    report.duration_ns = session_end_ns_ - session_start_ns_;

    // Aggregate metrics from all threads (shared_ptr keeps objects alive)
    auto all_metrics = ThreadMetricsStorage::instance().get_all_metrics();

    // Collect timing samples from all threads
    std::map<ProfileMetric, std::vector<uint64_t>> metric_samples;

    for (const auto& thread_metrics : all_metrics) {
        if (!thread_metrics) continue;  // Defensive check (should never be null)

        // Process timing buffer
        auto& buffer = thread_metrics->get_timing_buffer();
        TimingSample sample;

        while (buffer.try_pop(sample)) {
            report.add_timing_sample(sample);
            metric_samples[sample.metric].push_back(sample.duration_ns);
        }

        // Aggregate counters and gauges
        for (size_t i = 0; i < static_cast<size_t>(ProfileMetric::MetricCount); ++i) {
            auto metric = static_cast<ProfileMetric>(i);

            uint64_t count = thread_metrics->get_counter(metric);
            if (count > 0) {
                report.add_counter(metric, count);
            }

            uint64_t gauge = thread_metrics->get_gauge(metric);
            if (gauge > 0) {
                report.add_gauge(metric, gauge);
            }

            uint64_t total_ns, call_count;
            thread_metrics->get_timing_stats(metric, total_ns, call_count);
            if (call_count > 0) {
                report.add_timing_aggregate(metric, total_ns, call_count);
            }
        }
    }

    // Compute statistics for each metric from samples
    // Note: Only compute from samples if we have them; otherwise keep aggregate stats
    StatisticalAnalyzer analyzer;
    for (auto& [metric, samples] : metric_samples) {
        if (!samples.empty()) {
            // Samples available - compute full statistics (mean, p50, p95, etc.)
            report.timing_stats[metric] = analyzer.compute(samples);
        }
        // If no samples but we have aggregates, they're already in timing_stats via add_timing_aggregate
    }

    // Compute derived metrics
    report.compute_derived_metrics();

    return report;
}

void EnhancedProfiler::export_json(const std::string& filename) const {
    auto report = generate_report();
    std::ofstream out(filename);
    out << report.to_json();
    out.close();
    std::cout << "JSON report exported to: " << filename << std::endl;
}

void EnhancedProfiler::export_chrome_trace(const std::string& filename) const {
    auto report = generate_report();
    std::ofstream out(filename);
    out << report.to_chrome_trace();
    out.close();
    std::cout << "Chrome trace exported to: " << filename << std::endl;
}

void EnhancedProfiler::export_markdown(const std::string& filename) const {
    auto report = generate_report();
    std::ofstream out(filename);
    out << report.to_markdown();
    out.close();
    std::cout << "Markdown report exported to: " << filename << std::endl;
}

void EnhancedProfiler::print_summary() const {
    auto report = generate_report();

    std::cout << "\n========== Profiling Summary ==========" << std::endl;
    std::cout << "Session: " << report.session_name << std::endl;
    std::cout << "Duration: " << std::fixed << std::setprecision(2)
              << (report.duration_ns / 1e9) << "s" << std::endl;

    // Top timing operations
    std::cout << "\nTop Operations by Total Time:" << std::endl;
    std::vector<std::pair<ProfileMetric, MetricStats>> sorted_stats(
        report.timing_stats.begin(), report.timing_stats.end()
    );

    std::sort(sorted_stats.begin(), sorted_stats.end(),
        [](const auto& a, const auto& b) {
            return a.second.total > b.second.total;
        }
    );

    int shown = 0;
    for (const auto& [metric, stats] : sorted_stats) {
        if (shown++ >= 10) break;

        double total_ms = stats.total / 1e6;
        double mean_us = stats.mean / 1e3;
        double pct = 100.0 * stats.total / report.duration_ns;

        std::cout << "  " << std::setw(30) << std::left << metric_to_string(metric)
                  << " | " << std::setw(8) << std::right << stats.count << " calls"
                  << " | " << std::setw(10) << std::fixed << std::setprecision(2)
                  << total_ms << "ms"
                  << " | " << std::setw(8) << std::setprecision(1) << mean_us << "μs"
                  << " | " << std::setw(5) << std::setprecision(1) << pct << "%"
                  << std::endl;
    }

    // Bottlenecks
    if (!report.bottlenecks.empty()) {
        std::cout << "\nDetected Bottlenecks:" << std::endl;
        for (const auto& b : report.bottlenecks) {
            if (b.severity > 10) {
                std::cout << "  [" << std::setw(5) << std::setprecision(1)
                          << b.severity << "%] "
                          << b.description << " -> " << b.recommendation << std::endl;
            }
        }
    }

    std::cout << "=======================================" << std::endl;
}

} // namespace profiling
} // namespace mcts