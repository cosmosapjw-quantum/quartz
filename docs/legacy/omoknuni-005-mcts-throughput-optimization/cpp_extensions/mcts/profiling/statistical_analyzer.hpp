/**
 * @file statistical_analyzer.hpp
 * @brief Statistical analysis of profiling data
 *
 * Computes percentiles, distributions, and advanced statistics
 * from timing samples and counter data.
 */

#pragma once

#include "enhanced_metrics.hpp"
#include <vector>
#include <array>
#include <map>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace mcts {
namespace profiling {

/**
 * @brief Simple statistical summary for profiler
 */
struct MetricStats {
    size_t count = 0;
    uint64_t total = 0;
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    double p50 = 0.0;  // Median
    double p75 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double p999 = 0.0;
    double stddev = 0.0;
};

/**
 * @brief Bottleneck information
 */
struct Bottleneck {
    ProfileMetric metric;
    double severity;  // 0-100 percentage
    std::string description;
    std::string recommendation;
};

/**
 * @brief Profiling report
 */
struct ProfileReport {
    std::string session_name;
    uint64_t duration_ns = 0;
    std::map<ProfileMetric, MetricStats> timing_stats;
    std::map<ProfileMetric, uint64_t> counters;
    std::map<ProfileMetric, uint64_t> gauges;
    std::vector<Bottleneck> bottlenecks;

    void add_timing_sample(const struct TimingSample& sample) {
        (void)sample;  // Used for detailed trace, not needed for aggregate stats
    }

    void add_counter(ProfileMetric metric, uint64_t value) {
        counters[metric] += value;
    }

    void add_gauge(ProfileMetric metric, uint64_t value) {
        gauges[metric] = std::max(gauges[metric], value);
    }

    void add_timing_aggregate(ProfileMetric metric, uint64_t total_ns, uint64_t count) {
        if (count > 0) {
            // Create MetricStats from aggregates
            MetricStats& stats = timing_stats[metric];
            stats.count = count;
            stats.total = total_ns;
            stats.mean = static_cast<double>(total_ns) / count;
            // Note: min/max/percentiles not available from aggregates alone
            // These will be 0 unless populated from samples
        }
    }

    void compute_derived_metrics() {
        // Detect bottlenecks: metrics that take >20% of total time
        for (const auto& [metric, stats] : timing_stats) {
            if (duration_ns == 0) continue;

            double time_pct = 100.0 * stats.total / duration_ns;
            if (time_pct > 20.0) {
                Bottleneck b;
                b.metric = metric;
                b.severity = time_pct;
                b.description = std::string(metric_to_string(metric)) + " consumes " +
                               std::to_string(static_cast<int>(time_pct)) + "% of total time";
                b.recommendation = "Consider optimizing this operation";
                bottlenecks.push_back(b);
            }
        }
    }

    std::string to_json() const {
        std::string json = "{\n";
        json += "  \"session_name\": \"" + session_name + "\",\n";
        json += "  \"duration_ns\": " + std::to_string(duration_ns) + ",\n";

        // Export timing statistics with full details
        json += "  \"timing_stats\": {\n";
        bool first = true;
        for (const auto& [metric, stats] : timing_stats) {
            if (!first) json += ",\n";
            json += "    \"" + std::string(metric_to_string(metric)) + "\": {";
            json += "\"count\": " + std::to_string(stats.count) + ", ";
            json += "\"total\": " + std::to_string(stats.total) + ", ";
            json += "\"mean\": " + std::to_string(stats.mean) + ", ";
            json += "\"min\": " + std::to_string(stats.min) + ", ";
            json += "\"max\": " + std::to_string(stats.max) + ", ";
            json += "\"p50\": " + std::to_string(stats.p50) + ", ";
            json += "\"p75\": " + std::to_string(stats.p75) + ", ";
            json += "\"p90\": " + std::to_string(stats.p90) + ", ";
            json += "\"p95\": " + std::to_string(stats.p95) + ", ";
            json += "\"p99\": " + std::to_string(stats.p99) + ", ";
            json += "\"p999\": " + std::to_string(stats.p999) + ", ";
            json += "\"stddev\": " + std::to_string(stats.stddev) + "}";
            first = false;
        }
        json += "\n  },\n";

        // Export counters (state cloning, OpenMP, CAS retries, etc.)
        json += "  \"counters\": {\n";
        first = true;
        for (const auto& [metric, value] : counters) {
            if (!first) json += ",\n";
            json += "    \"" + std::string(metric_to_string(metric)) + "\": " + std::to_string(value);
            first = false;
        }
        json += "\n  },\n";

        // Export gauges (peak values, thresholds)
        json += "  \"gauges\": {\n";
        first = true;
        for (const auto& [metric, value] : gauges) {
            if (!first) json += ",\n";
            json += "    \"" + std::string(metric_to_string(metric)) + "\": " + std::to_string(value);
            first = false;
        }
        json += "\n  },\n";

        // Export bottlenecks
        json += "  \"bottlenecks\": [\n";
        first = true;
        for (const auto& b : bottlenecks) {
            if (!first) json += ",\n";
            json += "    {";
            json += "\"metric\": \"" + std::string(metric_to_string(b.metric)) + "\", ";
            json += "\"severity\": " + std::to_string(b.severity) + ", ";
            json += "\"description\": \"" + b.description + "\", ";
            json += "\"recommendation\": \"" + b.recommendation + "\"}";
            first = false;
        }
        json += "\n  ]\n";

        json += "}\n";
        return json;
    }

    std::string to_chrome_trace() const {
        std::string trace = "[";
        for (const auto& [metric, stats] : timing_stats) {
            trace += "{\"name\": \"" + std::string(metric_to_string(metric)) + "\", ";
            trace += "\"ph\": \"X\", \"ts\": 0, \"dur\": " + std::to_string(stats.mean / 1000.0) + "}";
            if (metric != timing_stats.rbegin()->first) trace += ",";
        }
        trace += "]";
        return trace;
    }

    std::string to_markdown() const {
        std::string md = "# Profiling Report\n\n";
        md += "**Session:** " + session_name + "\n\n";
        md += "**Duration:** " + std::to_string(duration_ns / 1e9) + "s\n\n";

        md += "## Timing Statistics\n\n";
        md += "| Metric | Count | Mean | Min | Max | P50 | P95 | P99 | StdDev |\n";
        md += "|--------|-------|------|-----|-----|-----|-----|-----|--------|\n";
        for (const auto& [metric, stats] : timing_stats) {
            md += "| " + std::string(metric_to_string(metric)) + " ";
            md += "| " + std::to_string(stats.count) + " ";
            md += "| " + std::to_string(stats.mean / 1e3) + "μs ";
            md += "| " + std::to_string(stats.min / 1e3) + "μs ";
            md += "| " + std::to_string(stats.max / 1e3) + "μs ";
            md += "| " + std::to_string(stats.p50 / 1e3) + "μs ";
            md += "| " + std::to_string(stats.p95 / 1e3) + "μs ";
            md += "| " + std::to_string(stats.p99 / 1e3) + "μs ";
            md += "| " + std::to_string(stats.stddev / 1e3) + "μs |\n";
        }
        md += "\n";

        if (!counters.empty()) {
            md += "## Counters\n\n";
            md += "| Metric | Value |\n";
            md += "|--------|-------|\n";
            for (const auto& [metric, value] : counters) {
                md += "| " + std::string(metric_to_string(metric)) + " | " + std::to_string(value) + " |\n";
            }
            md += "\n";
        }

        if (!gauges.empty()) {
            md += "## Gauges (Peak Values)\n\n";
            md += "| Metric | Value |\n";
            md += "|--------|-------|\n";
            for (const auto& [metric, value] : gauges) {
                md += "| " + std::string(metric_to_string(metric)) + " | " + std::to_string(value) + " |\n";
            }
            md += "\n";
        }

        if (!bottlenecks.empty()) {
            md += "## Detected Bottlenecks\n\n";
            for (const auto& b : bottlenecks) {
                md += "### " + std::string(metric_to_string(b.metric)) + " (" +
                      std::to_string(static_cast<int>(b.severity)) + "%)\n\n";
                md += "**Description:** " + b.description + "\n\n";
                md += "**Recommendation:** " + b.recommendation + "\n\n";
            }
        }

        return md;
    }
};

/**
 * @brief Statistical summary of metric samples
 */
struct MetricStatistics {
    std::uint64_t count;            // Number of samples
    double mean;                     // Mean (average)
    double std_dev;                  // Standard deviation
    double variance;                 // Variance
    std::uint64_t min;               // Minimum value
    std::uint64_t max;               // Maximum value
    double median;                   // Median (50th percentile)
    double p50;                      // 50th percentile
    double p75;                      // 75th percentile
    double p90;                      // 90th percentile
    double p95;                      // 95th percentile
    double p99;                      // 99th percentile
    double p999;                     // 99.9th percentile

    // Histogram (exponential buckets)
    static constexpr std::size_t HISTOGRAM_BUCKETS = 32;
    std::array<std::uint64_t, HISTOGRAM_BUCKETS> histogram;
    std::array<std::uint64_t, HISTOGRAM_BUCKETS> bucket_boundaries;

    // Derived metrics
    double coefficient_of_variation() const {
        return mean > 0 ? std_dev / mean : 0.0;
    }

    double skewness;     // Measure of asymmetry
    double kurtosis;     // Measure of tail heaviness

    MetricStatistics()
        : count(0), mean(0), std_dev(0), variance(0),
          min(0), max(0), median(0),
          p50(0), p75(0), p90(0), p95(0), p99(0), p999(0),
          skewness(0), kurtosis(0)
    {
        histogram.fill(0);
        bucket_boundaries.fill(0);
    }
};

/**
 * @brief Per-thread metric statistics
 */
struct ThreadMetricStatistics {
    std::uint64_t thread_id;
    ProfileMetric metric;
    MetricStatistics stats;

    // Thread-specific metrics
    double samples_per_second;
    double thread_utilization;  // Percentage of time spent in this metric
};

/**
 * @brief Statistical analyzer for profiling data
 *
 * Computes descriptive statistics, percentiles, and distributions
 * from timing samples.
 */
class StatisticalAnalyzer {
public:
    StatisticalAnalyzer() = default;

    /**
     * @brief Analyze samples and compute statistics (for MetricStats)
     * Used by enhanced_profiler.cpp
     */
    MetricStats compute(std::vector<uint64_t>& samples);

    /**
     * @brief Analyze samples and compute statistics
     *
     * @param samples Vector of timing samples (nanoseconds)
     * @return Statistical summary
     */
    MetricStatistics analyze(std::vector<std::uint64_t> samples);

    /**
     * @brief Analyze samples with pre-allocated storage (avoids copy)
     *
     * WARNING: Samples vector will be modified (sorted)
     */
    MetricStatistics analyze_inplace(std::vector<std::uint64_t>& samples);

    /**
     * @brief Compute percentile from sorted samples
     *
     * @param sorted_samples Sorted sample vector
     * @param percentile Percentile to compute (0.0 to 1.0)
     * @return Percentile value
     */
    static double compute_percentile(const std::vector<std::uint64_t>& sorted_samples, double percentile);

    /**
     * @brief Build histogram with exponential buckets
     *
     * Buckets: [0, 1), [1, 2), [2, 4), [4, 8), ..., [2^30, 2^31)
     */
    static void build_histogram(
        const std::vector<std::uint64_t>& samples,
        std::array<std::uint64_t, MetricStatistics::HISTOGRAM_BUCKETS>& histogram,
        std::array<std::uint64_t, MetricStatistics::HISTOGRAM_BUCKETS>& boundaries
    );

    /**
     * @brief Compute skewness (measure of asymmetry)
     *
     * Skewness > 0: Right-skewed (long tail on right)
     * Skewness < 0: Left-skewed (long tail on left)
     * Skewness ≈ 0: Symmetric distribution
     */
    static double compute_skewness(
        const std::vector<std::uint64_t>& samples,
        double mean,
        double std_dev
    );

    /**
     * @brief Compute kurtosis (measure of tail heaviness)
     *
     * Kurtosis > 3: Heavy tails (leptokurtic)
     * Kurtosis < 3: Light tails (platykurtic)
     * Kurtosis ≈ 3: Normal distribution (mesokurtic)
     */
    static double compute_kurtosis(
        const std::vector<std::uint64_t>& samples,
        double mean,
        double std_dev
    );

    /**
     * @brief Detect outliers using IQR method
     *
     * Outliers: Q1 - 1.5*IQR or Q3 + 1.5*IQR
     */
    static std::vector<std::uint64_t> detect_outliers(
        const std::vector<std::uint64_t>& samples,
        double q1,
        double q3
    );
};

/**
 * @brief Online statistics tracker (Welford's algorithm)
 *
 * Computes mean and variance incrementally without storing all samples.
 * Useful for low-memory profiling.
 */
class OnlineStatistics {
public:
    OnlineStatistics() : count_(0), mean_(0.0), m2_(0.0), min_(UINT64_MAX), max_(0) {}

    /**
     * @brief Update statistics with new sample
     */
    void update(std::uint64_t value) {
        count_++;

        // Update min/max
        if (value < min_) min_ = value;
        if (value > max_) max_ = value;

        // Welford's online algorithm for mean and variance
        double delta = static_cast<double>(value) - mean_;
        mean_ += delta / count_;
        double delta2 = static_cast<double>(value) - mean_;
        m2_ += delta * delta2;
    }

    /**
     * @brief Get current mean
     */
    double mean() const { return mean_; }

    /**
     * @brief Get current variance
     */
    double variance() const {
        return count_ > 1 ? m2_ / (count_ - 1) : 0.0;
    }

    /**
     * @brief Get current standard deviation
     */
    double std_dev() const {
        return std::sqrt(variance());
    }

    /**
     * @brief Get sample count
     */
    std::uint64_t count() const { return count_; }

    /**
     * @brief Get minimum value
     */
    std::uint64_t min() const { return min_; }

    /**
     * @brief Get maximum value
     */
    std::uint64_t max() const { return max_; }

    /**
     * @brief Reset statistics
     */
    void reset() {
        count_ = 0;
        mean_ = 0.0;
        m2_ = 0.0;
        min_ = UINT64_MAX;
        max_ = 0;
    }

private:
    std::uint64_t count_;
    double mean_;
    double m2_;  // Sum of squared differences from mean
    std::uint64_t min_;
    std::uint64_t max_;
};

/**
 * @brief Aggregator for thread-local metrics
 *
 * Collects metrics from all threads and computes aggregate statistics.
 */
class MetricAggregator {
public:
    /**
     * @brief Aggregate metrics from all threads
     *
     * @param thread_metrics Vector of thread-local metric storage
     * @return Per-metric aggregate statistics
     */
    std::vector<MetricStatistics> aggregate(
        const std::vector<ThreadLocalMetrics*>& thread_metrics
    );

    /**
     * @brief Aggregate specific metric across threads
     */
    MetricStatistics aggregate_metric(
        const std::vector<ThreadLocalMetrics*>& thread_metrics,
        ProfileMetric metric
    );

    /**
     * @brief Get per-thread statistics for metric
     */
    std::vector<ThreadMetricStatistics> get_per_thread_stats(
        const std::vector<ThreadLocalMetrics*>& thread_metrics,
        ProfileMetric metric
    );

    /**
     * @brief Compute thread imbalance coefficient
     *
     * Measures load imbalance across threads.
     * 0.0 = perfectly balanced, 1.0 = completely imbalanced
     */
    double compute_thread_imbalance(
        const std::vector<ThreadMetricStatistics>& thread_stats
    );

private:
    /**
     * @brief Extract samples for a metric from thread storage
     */
    std::vector<std::uint64_t> extract_samples(
        ThreadLocalMetrics* tls,
        ProfileMetric metric
    );
};

/**
 * @brief Correlation analyzer for metric relationships
 *
 * Computes correlation coefficients between metrics to identify
 * dependencies and bottlenecks.
 */
class CorrelationAnalyzer {
public:
    /**
     * @brief Compute Pearson correlation coefficient
     *
     * @param x First metric samples
     * @param y Second metric samples
     * @return Correlation coefficient [-1, 1]
     *         1: Perfect positive correlation
     *         0: No correlation
     *        -1: Perfect negative correlation
     */
    static double pearson_correlation(
        const std::vector<std::uint64_t>& x,
        const std::vector<std::uint64_t>& y
    );

    /**
     * @brief Compute correlation matrix for all metrics
     *
     * @return Matrix[i][j] = correlation between metric i and metric j
     */
    std::vector<std::vector<double>> compute_correlation_matrix(
        const std::vector<ThreadLocalMetrics*>& thread_metrics
    );

    /**
     * @brief Find highly correlated metric pairs
     *
     * @param threshold Minimum correlation coefficient (e.g., 0.7)
     * @return Vector of (metric1, metric2, correlation) tuples
     */
    std::vector<std::tuple<ProfileMetric, ProfileMetric, double>> find_correlated_pairs(
        const std::vector<ThreadLocalMetrics*>& thread_metrics,
        double threshold = 0.7
    );
};

/**
 * @brief Bottleneck detector
 *
 * Identifies performance bottlenecks using statistical analysis.
 */
class BottleneckDetector {
public:
    struct Bottleneck {
        ProfileMetric metric;
        double severity;        // 0.0 to 1.0 (1.0 = critical)
        double time_percentage; // Percentage of total time
        const char* description;
    };

    /**
     * @brief Detect bottlenecks from profiling data
     *
     * Uses multiple heuristics:
     * - High mean duration
     * - High variance (inconsistent performance)
     * - High percentile (long tail latency)
     * - High thread contention (imbalance)
     */
    std::vector<Bottleneck> detect(
        const std::vector<MetricStatistics>& metrics
    );

    /**
     * @brief Suggest optimizations for bottlenecks
     */
    std::vector<std::string> suggest_optimizations(const Bottleneck& bottleneck);
};

/**
 * @brief Time series analyzer for trend detection
 *
 * Detects performance degradation over time.
 */
class TimeSeriesAnalyzer {
public:
    /**
     * @brief Detect performance regression
     *
     * Compares recent samples to baseline.
     *
     * @param baseline Baseline statistics (e.g., from previous run)
     * @param current Current statistics
     * @return Regression severity (0.0 = no regression, 1.0 = severe)
     */
    double detect_regression(
        const MetricStatistics& baseline,
        const MetricStatistics& current
    );

    /**
     * @brief Detect performance improvement
     */
    double detect_improvement(
        const MetricStatistics& baseline,
        const MetricStatistics& current
    );

    /**
     * @brief Compute moving average
     */
    std::vector<double> moving_average(
        const std::vector<std::uint64_t>& samples,
        std::size_t window_size
    );
};

} // namespace profiling
} // namespace mcts
