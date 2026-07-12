/**
 * @file validation.hpp
 * @brief Validation framework for enhanced profiling system
 * @author MCTS Performance Team
 * @date 2025-10-15
 */

#pragma once

#include <string>
#include <vector>

namespace mcts {
namespace profiling {

/**
 * @brief Validation test result
 */
struct ValidationResult {
    std::string test_name;
    bool passed;
    std::string message;
    double duration_ms;
};

/**
 * @brief Validate the profiling infrastructure
 *
 * Runs a comprehensive test suite to ensure all profiling
 * components work correctly before production use.
 *
 * Tests include:
 * - Enable/disable functionality
 * - Session management
 * - Metric recording (timers, counters, gauges)
 * - Export formats (JSON, Chrome Trace, Markdown)
 * - Thread-local storage
 * - Zero overhead when disabled
 *
 * @return Vector of validation results
 */
std::vector<ValidationResult> validate_profiling_infrastructure();

/**
 * @brief Run validation and print results to console
 *
 * @return true if all tests passed, false otherwise
 */
bool run_validation();

} // namespace profiling
} // namespace mcts
