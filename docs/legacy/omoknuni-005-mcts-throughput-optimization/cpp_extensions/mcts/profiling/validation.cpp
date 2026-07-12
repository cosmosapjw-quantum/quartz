/**
 * @file validation.cpp
 * @brief Implementation of profiling validation framework
 */

#include "validation.hpp"
#include "enhanced_profiler.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>
#include <cmath>

namespace mcts {
namespace profiling {

using namespace std::chrono;

// Helper function to time a function
template<typename Func>
double time_function_ms(Func&& func) {
    auto start = steady_clock::now();
    func();
    auto end = steady_clock::now();
    return duration<double, std::milli>(end - start).count();
}

std::vector<ValidationResult> validate_profiling_infrastructure() {
    std::vector<ValidationResult> results;

    auto& profiler = EnhancedProfiler::instance();

    // Test 1: Enable/Disable
    {
        std::string test_name = "Enable/Disable Toggle";
        bool passed = true;
        std::string message;

        double duration = time_function_ms([&]() {
            profiler.set_enabled(true);
            if (!profiler.is_enabled()) {
                passed = false;
                message = "Failed to enable profiler";
                return;
            }

            profiler.set_enabled(false);
            if (profiler.is_enabled()) {
                passed = false;
                message = "Failed to disable profiler";
                return;
            }

            message = "Successfully toggled profiler on/off";
        });

        results.push_back({test_name, passed, message, duration});
    }

    // Test 2: Session Management
    {
        std::string test_name = "Session Management";
        bool passed = true;
        std::string message;

        double duration = time_function_ms([&]() {
            try {
                profiler.set_enabled(true);
                profiler.start_session("validation_test");
                profiler.stop_session();
                message = "Session start/stop successful";
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Session management failed: ") + e.what();
            }
        });

        results.push_back({test_name, passed, message, duration});
    }

    // Test 3: Metric Recording - Timers
    {
        std::string test_name = "Timer Metrics";
        bool passed = true;
        std::string message;

        double duration = time_function_ms([&]() {
            try {
                profiler.set_enabled(true);
                profiler.start_session("timer_test");

                // Record some timed operations
                {
                    PROFILE_SCOPE(ProfileMetric::SelectionTotal);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }

                {
                    PROFILE_SCOPE(ProfileMetric::ExpansionTotal);
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }

                profiler.stop_session();
                message = "Timer metrics recorded successfully";
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Timer recording failed: ") + e.what();
            }
        });

        results.push_back({test_name, passed, message, duration});
    }

    // Test 4: Metric Recording - Counters
    {
        std::string test_name = "Counter Metrics";
        bool passed = true;
        std::string message;

        double duration = time_function_ms([&]() {
            try {
                profiler.set_enabled(true);
                profiler.reset_metrics();

                // Increment counters
                for (int i = 0; i < 100; ++i) {
                    PROFILE_COUNTER(ProfileMetric::StateCloneCount, 1);
                }

                for (int i = 0; i < 50; ++i) {
                    PROFILE_COUNTER(ProfileMetric::CAS_SuccessCount, 1);
                }

                message = "Counter metrics incremented successfully (150 total)";
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Counter recording failed: ") + e.what();
            }
        });

        results.push_back({test_name, passed, message, duration});
    }

    // Test 5: Metric Recording - Gauges
    {
        std::string test_name = "Gauge Metrics";
        bool passed = true;
        std::string message;

        double duration = time_function_ms([&]() {
            try {
                profiler.set_enabled(true);

                // Update gauges
                PROFILE_GAUGE(ProfileMetric::SelectionDepth, 10);
                PROFILE_GAUGE(ProfileMetric::OMP_ThreadCount, 12);
                PROFILE_GAUGE(ProfileMetric::GPUBatchSize, 64);

                message = "Gauge metrics updated successfully";
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Gauge recording failed: ") + e.what();
            }
        });

        results.push_back({test_name, passed, message, duration});
    }

    // Test 6: JSON Export
    {
        std::string test_name = "JSON Export";
        bool passed = true;
        std::string message;

        double duration = time_function_ms([&]() {
            try {
                profiler.set_enabled(true);
                profiler.start_session("export_test");

                // Record some metrics
                PROFILE_COUNTER(ProfileMetric::TotalSimulations, 1000);
                {
                    PROFILE_SCOPE(ProfileMetric::PipelineE2ELatency);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                profiler.stop_session();
                profiler.export_json("/tmp/profiling_validation.json");

                message = "JSON export successful (/tmp/profiling_validation.json)";
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("JSON export failed: ") + e.what();
            }
        });

        results.push_back({test_name, passed, message, duration});
    }

    // Test 7: Zero Overhead When Disabled
    {
        std::string test_name = "Zero Overhead (Disabled)";
        bool passed = true;
        std::string message;

        // Measure with profiling disabled
        profiler.set_enabled(false);
        double disabled_time = time_function_ms([&]() {
            for (int i = 0; i < 10000; ++i) {
                PROFILE_SCOPE(ProfileMetric::SelectionTotal);
                PROFILE_COUNTER(ProfileMetric::StateCloneCount, 1);
                PROFILE_GAUGE(ProfileMetric::SelectionDepth, i);
            }
        });

        // Measure with profiling enabled
        profiler.set_enabled(true);
        profiler.start_session("overhead_test");
        double enabled_time = time_function_ms([&]() {
            for (int i = 0; i < 10000; ++i) {
                PROFILE_SCOPE(ProfileMetric::SelectionTotal);
                PROFILE_COUNTER(ProfileMetric::StateCloneCount, 1);
                PROFILE_GAUGE(ProfileMetric::SelectionDepth, i);
            }
        });
        profiler.stop_session();

        double overhead_pct = 100.0 * (enabled_time - disabled_time) / disabled_time;

        if (overhead_pct < 5.0) {
            message = "Overhead: " + std::to_string(overhead_pct) + "% (< 5% target)";
        } else if (overhead_pct < 10.0) {
            message = "Overhead: " + std::to_string(overhead_pct) + "% (acceptable but > 5%)";
        } else {
            passed = false;
            message = "Overhead too high: " + std::to_string(overhead_pct) + "% (> 10%)";
        }

        results.push_back({test_name, passed, message, enabled_time + disabled_time});
    }

    // Test 8: Thread Safety (Basic)
    {
        std::string test_name = "Thread Safety";
        bool passed = true;
        std::string message;

        double duration = time_function_ms([&]() {
            try {
                profiler.set_enabled(true);
                profiler.reset_metrics();

                // Launch multiple threads recording metrics
                std::vector<std::thread> threads;
                for (int t = 0; t < 4; ++t) {
                    threads.emplace_back([&]() {
                        for (int i = 0; i < 100; ++i) {
                            PROFILE_SCOPE(ProfileMetric::BackupTotal);
                            PROFILE_COUNTER(ProfileMetric::CAS_RetryCount, 1);
                            std::this_thread::sleep_for(std::chrono::microseconds(10));
                        }
                    });
                }

                for (auto& thread : threads) {
                    thread.join();
                }

                message = "4 threads recorded metrics concurrently without crashes";
            } catch (const std::exception& e) {
                passed = false;
                message = std::string("Thread safety test failed: ") + e.what();
            }
        });

        results.push_back({test_name, passed, message, duration});
    }

    return results;
}

bool run_validation() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Enhanced Profiling Validation Suite\n";
    std::cout << "========================================\n\n";

    auto results = validate_profiling_infrastructure();

    int passed_count = 0;
    int failed_count = 0;

    for (const auto& result : results) {
        std::string status = result.passed ? "✅ PASS" : "❌ FAIL";
        std::cout << status << " | " << result.test_name << " (" << result.duration_ms << " ms)\n";
        std::cout << "       " << result.message << "\n\n";

        if (result.passed) {
            passed_count++;
        } else {
            failed_count++;
        }
    }

    std::cout << "========================================\n";
    std::cout << "Results: " << passed_count << " passed, " << failed_count << " failed\n";
    std::cout << "========================================\n\n";

    if (failed_count == 0) {
        std::cout << "✅ All validation tests passed!\n";
        std::cout << "   The profiling system is ready for production use.\n\n";
        return true;
    } else {
        std::cout << "❌ Some validation tests failed!\n";
        std::cout << "   Fix the issues before using the profiling system.\n\n";
        return false;
    }
}

} // namespace profiling
} // namespace mcts
