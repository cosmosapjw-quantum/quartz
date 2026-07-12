/**
 * @file wall_clock_validator.hpp
 * @brief Validates that profiling accounts for all execution time
 * @author MCTS Performance Team
 * @date 2025-10-15
 *
 * Ensures that the sum of all profiled time components approximately equals
 * the total wall-clock duration. If large discrepancies exist, it indicates
 * missing instrumentation.
 *
 * Target: < 10% unaccounted time
 */

#pragma once

#include "profile_report.hpp"
#include "enhanced_metrics.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

namespace mcts {
namespace profiling {

struct TimeAccountingBreakdown {
    std::string category;
    uint64_t total_ns;
    double percentage;
    std::vector<std::string> metrics;
};

class WallClockValidator {
public:
    /**
     * @brief Validate profiling completeness and print detailed breakdown
     *
     * @param report The profile report to validate
     * @return True if < 10% time unaccounted, false otherwise
     */
    static bool validate_and_report(const ProfileReport& report) {
        if (report.duration_ns == 0) {
            std::cerr << "❌ ERROR: Zero duration in profile report!" << std::endl;
            return false;
        }

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "WALL-CLOCK TIME ACCOUNTING VALIDATION" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        // Calculate total accounted time
        uint64_t total_accounted = 0;
        for (const auto& [metric, stats] : report.timing_stats) {
            total_accounted += stats.total;
        }

        uint64_t total_duration = report.duration_ns;
        int64_t unaccounted = total_duration - total_accounted;
        double unaccounted_pct = 100.0 * unaccounted / total_duration;

        // Print high-level summary
        std::cout << "\nSession: " << report.session_name << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total Wall-Clock Time: " << (total_duration / 1e6) << " ms" << std::endl;
        std::cout << "Accounted Time:        " << (total_accounted / 1e6) << " ms ("
                  << (100.0 * total_accounted / total_duration) << "%)" << std::endl;
        std::cout << "Unaccounted Time:      " << (unaccounted / 1e6) << " ms ("
                  << unaccounted_pct << "%)" << std::endl;

        // Categorized breakdown
        auto breakdown = categorize_time(report);
        print_breakdown(breakdown, total_duration);

        // Validation verdict
        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "VERDICT: ";

        bool passed = (unaccounted_pct < 10.0 && unaccounted_pct > -5.0);  // Allow slight over-accounting from overlapping scopes

        if (passed) {
            std::cout << "✅ PASS - Time accounting validated" << std::endl;
            std::cout << "   Unaccounted time: " << unaccounted_pct << "% (< 10% threshold)" << std::endl;
        } else if (unaccounted_pct >= 10.0) {
            std::cout << "❌ FAIL - Missing instrumentation detected" << std::endl;
            std::cout << "   Unaccounted time: " << unaccounted_pct << "% (>= 10% threshold)" << std::endl;
            std::cout << "\n   ACTION REQUIRED:" << std::endl;
            std::cout << "   1. Add PROFILE_SCOPE to more code sections" << std::endl;
            std::cout << "   2. Check for missing Python profiling integration" << std::endl;
            std::cout << "   3. Review hot code paths with 'perf record'" << std::endl;
            std::cout << "   4. See INSTRUMENTATION_CHECKLIST.md for guidance" << std::endl;
        } else {
            std::cout << "⚠️  WARNING - Over-accounting detected" << std::endl;
            std::cout << "   Unaccounted time: " << unaccounted_pct << "% (negative - overlapping scopes)" << std::endl;
            std::cout << "   This indicates nested PROFILE_SCOPE calls (expected for hierarchical profiling)" << std::endl;
            passed = true;  // Not a failure, just a note
        }

        std::cout << std::string(80, '=') << std::endl;

        return passed;
    }

private:
    static std::vector<TimeAccountingBreakdown> categorize_time(const ProfileReport& report) {
        std::map<MetricCategory, TimeAccountingBreakdown> category_map;

        // Initialize categories
        for (int i = 0; i < static_cast<int>(MetricCategory::Network) + 1; ++i) {
            auto cat = static_cast<MetricCategory>(i);
            category_map[cat] = {
                category_to_string(cat),
                0,
                0.0,
                {}
            };
        }

        // Aggregate by category
        for (const auto& [metric, stats] : report.timing_stats) {
            auto metadata = get_metric_metadata(metric);
            auto& cat_breakdown = category_map[metadata.category];

            cat_breakdown.total_ns += stats.total;
            cat_breakdown.metrics.push_back(metadata.name);
        }

        // Convert to vector and calculate percentages
        std::vector<TimeAccountingBreakdown> result;
        for (auto& [cat, breakdown] : category_map) {
            if (breakdown.total_ns > 0) {
                breakdown.percentage = 100.0 * breakdown.total_ns / report.duration_ns;
                result.push_back(breakdown);
            }
        }

        // Sort by total time descending
        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) {
                      return a.total_ns > b.total_ns;
                  });

        return result;
    }

    static void print_breakdown(const std::vector<TimeAccountingBreakdown>& breakdown,
                                 uint64_t total_duration) {
        std::cout << "\n--- Time Breakdown by Category ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);

        for (const auto& item : breakdown) {
            double time_ms = item.total_ns / 1e6;

            std::cout << std::setw(20) << std::left << item.category
                      << " | " << std::setw(10) << std::right << time_ms << " ms"
                      << " | " << std::setw(6) << item.percentage << "%"
                      << std::endl;

            // Show top 3 metrics in this category
            if (item.metrics.size() > 3) {
                std::cout << "     ↳ " << item.metrics[0] << ", "
                          << item.metrics[1] << ", "
                          << item.metrics[2] << " (+" << (item.metrics.size() - 3) << " more)"
                          << std::endl;
            } else if (!item.metrics.empty()) {
                std::cout << "     ↳ ";
                for (size_t i = 0; i < item.metrics.size(); ++i) {
                    std::cout << item.metrics[i];
                    if (i + 1 < item.metrics.size()) std::cout << ", ";
                }
                std::cout << std::endl;
            }
        }
    }
};

/**
 * @brief Quick validation function for use in Python bindings
 *
 * Returns percentage of unaccounted time (positive = missing, negative = over-accounted)
 */
inline double get_unaccounted_percentage(const ProfileReport& report) {
    if (report.duration_ns == 0) return 0.0;

    uint64_t total_accounted = 0;
    for (const auto& [metric, stats] : report.timing_stats) {
        total_accounted += stats.total;
    }

    int64_t unaccounted = report.duration_ns - total_accounted;
    return 100.0 * unaccounted / report.duration_ns;
}

} // namespace profiling
} // namespace mcts
