/**
 * @file export.hpp
 * @brief Profiling data export utilities
 *
 * Exports profiling data to various formats:
 * - JSON (for programmatic analysis)
 * - Chrome Trace (for chrome://tracing visualization)
 * - CSV (for spreadsheet analysis)
 * - Markdown (for human-readable reports)
 */

#pragma once

#include "metrics.hpp"
#include "thread_local_metrics.hpp"
#include "statistical_analyzer.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace mcts {
namespace profiling {

/**
 * @brief Profiling session metadata
 */
struct SessionMetadata {
    std::string session_name;
    std::string start_time;
    std::string end_time;
    std::uint64_t duration_ns;
    std::uint32_t num_threads;
    std::uint64_t total_samples;
    std::uint64_t total_samples_dropped;

    // System information
    std::string cpu_model;
    std::uint32_t cpu_cores;
    std::uint32_t cpu_threads;
    double cpu_frequency_ghz;
    std::uint64_t total_memory_bytes;
    std::string os_version;
};

/**
 * @brief JSON exporter
 */
class JSONExporter {
public:
    /**
     * @brief Export profiling session to JSON
     *
     * Format:
     * {
     *   "session": {...},
     *   "metrics": [...],
     *   "per_thread": [...],
     *   "hardware_counters": {...},
     *   "contention": {...}
     * }
     */
    static bool export_session(
        const char* filename,
        const SessionMetadata& metadata,
        const std::vector<MetricStatistics>& metrics,
        const std::vector<ThreadLocalMetrics*>& thread_metrics
    );

    /**
     * @brief Export single metric to JSON object
     */
    static std::string export_metric(
        ProfileMetric metric,
        const MetricStatistics& stats
    );

    /**
     * @brief Export thread statistics to JSON array
     */
    static std::string export_thread_stats(
        const std::vector<ThreadMetricStatistics>& thread_stats
    );

private:
    static std::string escape_json_string(const std::string& str);
    static std::string format_timestamp(std::uint64_t timestamp_ns);
};

/**
 * @brief Chrome Trace exporter (chrome://tracing format)
 *
 * Generates trace files compatible with Chrome's built-in profiler.
 */
class ChromeTraceExporter {
public:
    /**
     * @brief Export profiling session to Chrome Trace format
     *
     * Format: JSON array of trace events
     * [
     *   {"name": "Selection", "cat": "MCTS", "ph": "X", "ts": 123, "dur": 456, "pid": 0, "tid": 1},
     *   ...
     * ]
     */
    static bool export_session(
        const char* filename,
        const SessionMetadata& metadata,
        const std::vector<ThreadLocalMetrics*>& thread_metrics
    );

    /**
     * @brief Event phases (Chrome Trace)
     */
    enum class EventPhase : char {
        Complete = 'X',      // Complete event (has duration)
        Begin = 'B',         // Begin event
        End = 'E',           // End event
        Instant = 'i',       // Instant event (no duration)
        Counter = 'C',       // Counter value
        Metadata = 'M',      // Metadata
    };

    /**
     * @brief Trace event structure
     */
    struct TraceEvent {
        const char* name;
        const char* category;
        EventPhase phase;
        std::uint64_t timestamp_us;
        std::uint64_t duration_us;
        std::uint64_t process_id;
        std::uint64_t thread_id;
        std::string args;  // JSON object (optional)
    };

    /**
     * @brief Convert timing sample to trace event
     */
    static TraceEvent sample_to_event(
        const TimingSample& sample,
        std::uint64_t process_id = 0
    );

private:
    static std::string event_to_json(const TraceEvent& event);
};

/**
 * @brief CSV exporter
 */
class CSVExporter {
public:
    /**
     * @brief Export metrics to CSV
     *
     * Columns: metric, count, mean, std_dev, min, max, p50, p95, p99
     */
    static bool export_metrics(
        const char* filename,
        const std::vector<MetricStatistics>& metrics
    );

    /**
     * @brief Export per-thread statistics to CSV
     */
    static bool export_thread_stats(
        const char* filename,
        const std::vector<ThreadMetricStatistics>& thread_stats
    );

    /**
     * @brief Export timing samples to CSV (raw data)
     */
    static bool export_samples(
        const char* filename,
        const std::vector<ThreadLocalMetrics*>& thread_metrics
    );

private:
    static std::string escape_csv_field(const std::string& field);
};

/**
 * @brief Markdown report generator
 *
 * Generates human-readable performance reports in Markdown format.
 */
class MarkdownReporter {
public:
    /**
     * @brief Generate full performance report
     */
    static bool generate_report(
        const char* filename,
        const SessionMetadata& metadata,
        const std::vector<MetricStatistics>& metrics,
        const std::vector<ThreadLocalMetrics*>& thread_metrics
    );

    /**
     * @brief Generate summary section
     */
    static std::string generate_summary(
        const SessionMetadata& metadata,
        const std::vector<MetricStatistics>& metrics
    );

    /**
     * @brief Generate metrics table
     */
    static std::string generate_metrics_table(
        const std::vector<MetricStatistics>& metrics
    );

    /**
     * @brief Generate per-thread breakdown
     */
    static std::string generate_thread_breakdown(
        const std::vector<ThreadMetricStatistics>& thread_stats
    );

    /**
     * @brief Generate bottleneck analysis
     */
    static std::string generate_bottleneck_analysis(
        const std::vector<BottleneckDetector::Bottleneck>& bottlenecks
    );

    /**
     * @brief Generate optimization suggestions
     */
    static std::string generate_optimization_suggestions(
        const std::vector<BottleneckDetector::Bottleneck>& bottlenecks
    );

private:
    static std::string format_duration(std::uint64_t ns);
    static std::string format_percentage(double value);
    static std::string format_size(std::uint64_t bytes);
};

/**
 * @brief HTML report generator with interactive charts
 */
class HTMLReporter {
public:
    /**
     * @brief Generate interactive HTML report
     *
     * Includes:
     * - Summary dashboard
     * - Interactive charts (Chart.js)
     * - Per-metric histograms
     * - Thread timeline visualization
     */
    static bool generate_report(
        const char* filename,
        const SessionMetadata& metadata,
        const std::vector<MetricStatistics>& metrics,
        const std::vector<ThreadLocalMetrics*>& thread_metrics
    );

private:
    static std::string generate_html_header(const SessionMetadata& metadata);
    static std::string generate_summary_section(const std::vector<MetricStatistics>& metrics);
    static std::string generate_charts_section(const std::vector<MetricStatistics>& metrics);
    static std::string generate_histogram_chart(const MetricStatistics& stats);
    static std::string generate_thread_timeline(const std::vector<ThreadLocalMetrics*>& thread_metrics);
};

/**
 * @brief Binary exporter (compact format)
 *
 * For large profiling sessions, binary format is more efficient.
 */
class BinaryExporter {
public:
    /**
     * @brief Export to binary format
     *
     * Format:
     * - Magic number (4 bytes): 0x4D435450 ("MCTP")
     * - Version (2 bytes): 1
     * - Header size (4 bytes)
     * - Header (variable)
     * - Samples (variable)
     */
    static bool export_session(
        const char* filename,
        const SessionMetadata& metadata,
        const std::vector<ThreadLocalMetrics*>& thread_metrics
    );

    /**
     * @brief Import from binary format
     */
    static bool import_session(
        const char* filename,
        SessionMetadata& metadata,
        std::vector<std::vector<TimingSample>>& samples
    );

private:
    static constexpr std::uint32_t MAGIC_NUMBER = 0x4D435450;  // "MCTP"
    static constexpr std::uint16_t FORMAT_VERSION = 1;
};

/**
 * @brief Compare two profiling sessions
 *
 * Generates diff report showing performance changes.
 */
class SessionComparator {
public:
    struct MetricDiff {
        ProfileMetric metric;
        double baseline_mean;
        double current_mean;
        double change_percentage;
        bool is_regression;      // Performance decreased
        bool is_improvement;     // Performance increased
        bool is_significant;     // Change > threshold
    };

    /**
     * @brief Compare two sessions
     *
     * @param baseline Baseline profiling data
     * @param current Current profiling data
     * @param threshold Significance threshold (e.g., 5.0 for 5%)
     * @return Vector of metric diffs
     */
    static std::vector<MetricDiff> compare(
        const std::vector<MetricStatistics>& baseline,
        const std::vector<MetricStatistics>& current,
        double threshold = 5.0
    );

    /**
     * @brief Generate comparison report (Markdown)
     */
    static std::string generate_comparison_report(
        const std::vector<MetricDiff>& diffs
    );

    /**
     * @brief Detect regressions
     */
    static std::vector<MetricDiff> find_regressions(
        const std::vector<MetricDiff>& diffs
    );

    /**
     * @brief Detect improvements
     */
    static std::vector<MetricDiff> find_improvements(
        const std::vector<MetricDiff>& diffs
    );
};

/**
 * @brief Profiling report builder (fluent API)
 *
 * Usage:
 *   ReportBuilder()
 *     .add_session(metadata)
 *     .add_metrics(metrics)
 *     .add_thread_stats(thread_stats)
 *     .export_json("report.json")
 *     .export_chrome_trace("trace.json")
 *     .export_markdown("report.md");
 */
class ReportBuilder {
public:
    ReportBuilder() = default;

    ReportBuilder& add_session(const SessionMetadata& metadata);
    ReportBuilder& add_metrics(const std::vector<MetricStatistics>& metrics);
    ReportBuilder& add_thread_stats(const std::vector<ThreadMetricStatistics>& thread_stats);
    ReportBuilder& add_hardware_counters(const HardwareCounters& hw);
    ReportBuilder& add_bottlenecks(const std::vector<BottleneckDetector::Bottleneck>& bottlenecks);

    bool export_json(const char* filename);
    bool export_chrome_trace(const char* filename);
    bool export_csv(const char* filename);
    bool export_markdown(const char* filename);
    bool export_html(const char* filename);

private:
    SessionMetadata metadata_;
    std::vector<MetricStatistics> metrics_;
    std::vector<ThreadMetricStatistics> thread_stats_;
    HardwareCounters hw_counters_;
    std::vector<BottleneckDetector::Bottleneck> bottlenecks_;
};

/**
 * @brief Profiling data loader
 *
 * Load previously exported profiling data for analysis.
 */
class DataLoader {
public:
    /**
     * @brief Load session from JSON
     */
    static bool load_json(
        const char* filename,
        SessionMetadata& metadata,
        std::vector<MetricStatistics>& metrics
    );

    /**
     * @brief Load session from binary
     */
    static bool load_binary(
        const char* filename,
        SessionMetadata& metadata,
        std::vector<std::vector<TimingSample>>& samples
    );
};

} // namespace profiling
} // namespace mcts
