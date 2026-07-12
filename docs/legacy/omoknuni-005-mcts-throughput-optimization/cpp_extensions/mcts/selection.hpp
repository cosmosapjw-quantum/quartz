/**
 * @file selection.hpp
 * @brief Vectorized PUCT selection for high-performance MCTS tree traversal
 *
 * Implements SIMD-optimized PUCT (Polynomial Upper Confidence Trees) selection
 * using AVX2 instructions for 4-8x performance improvement over naive implementations.
 *
 * PUCT Formula: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
 * Where:
 * - Q = average value (total_value / visit_count)
 * - c_puct = exploration constant (typically 1.25)
 * - P = prior probability from neural network
 * - N_parent = parent visit count
 * - N_child = child visit count
 */

#pragma once

#include "tree.hpp"
#include <immintrin.h>  // AVX2 intrinsics
#include <cmath>
#include <vector>

namespace mcts {

/**
 * @brief Configuration for PUCT selection
 */
struct PUCTConfig {
    float cpuct = 1.25f;           // Exploration constant
    float fpu_value = 0.0f;        // First Play Urgency value for unvisited nodes
    bool use_fpu = true;           // Enable First Play Urgency
    bool enable_simd = true;       // Enable SIMD optimizations
};

/**
 * @brief Result of PUCT selection operation
 */
struct SelectionResult {
    NodeIndex selected_child;      // Index of selected child node
    float best_puct_value;         // PUCT value of selected child
    std::uint16_t child_position;  // Position in children array (0-based)
    bool valid;                    // Whether selection was successful
};

/**
 * @brief High-performance PUCT selection with SIMD optimizations
 *
 * This class provides vectorized implementations of the PUCT selection formula
 * optimized for AMD Ryzen 5900X AVX2 capabilities. Handles variable child counts
 * efficiently and provides fallbacks for edge cases.
 */
class PUCTSelector {
public:
    /**
     * @brief Initialize selector with configuration
     */
    explicit PUCTSelector(const PUCTConfig& config = PUCTConfig{});

    /**
     * @brief Select best child using vectorized PUCT calculation
     *
     * @param tree MCTS tree containing node data
     * @param parent_index Index of parent node to select from
     * @return Selection result with chosen child and PUCT value
     */
    SelectionResult select_child(const MCTSTree& tree, NodeIndex parent_index) const;

    /**
     * @brief Vectorized PUCT calculation for multiple children (AVX2 optimized)
     *
     * Processes up to 8 children simultaneously using SIMD instructions.
     * Handles remainder children with scalar operations.
     * Excludes nodes marked as "expanding" by setting their PUCT to -infinity.
     *
     * @param visit_counts Pointer to visit count array
     * @param total_values Pointer to total value array
     * @param prior_probs Pointer to prior probability array
     * @param virtual_losses Pointer to virtual loss array
     * @param flags Pointer to node flags array
     * @param first_child_index Index of first child
     * @param num_children Number of children to process
     * @param parent_visits Parent node visit count
     * @param exploration_term Pre-computed c_puct * sqrt(parent_visits)
     * @param puct_values Output array for computed PUCT values
     */
    void compute_puct_vectorized(
        const float* visit_counts,
        const float* total_values,
        const float* prior_probs,
        const float* virtual_losses,
        const NodeFlags* flags,
        NodeIndex first_child_index,
        std::uint16_t num_children,
        float exploration_term,
        float* puct_values
    ) const;

    /**
     * @brief Scalar PUCT calculation for single node (fallback)
     */
    float compute_puct_scalar(
        float visit_count,
        float total_value,
        float prior_prob,
        float virtual_loss,
        float exploration_term
    ) const;

    /**
     * @brief Find maximum value and index in array
     *
     * Uses vectorized search for arrays >= 8 elements, scalar for smaller arrays.
     *
     * @param values Array of PUCT values
     * @param count Number of values
     * @return Pair of (max_value, max_index)
     */
    std::pair<float, std::uint16_t> find_max_vectorized(
        const float* values,
        std::uint16_t count
    ) const;

    /**
     * @brief Update configuration
     */
    void set_config(const PUCTConfig& config) { config_ = config; }

    /**
     * @brief Get current configuration
     */
    const PUCTConfig& get_config() const { return config_; }

    /**
     * @brief Check if AVX2 is available on this CPU
     */
    static bool is_avx2_supported();

private:
    PUCTConfig config_;

    /**
     * @brief Compute Q-value with virtual loss adjustment
     */
    float compute_q_value(float visit_count, float total_value, float virtual_loss) const;

    /**
     * @brief Handle First Play Urgency for unvisited nodes
     */
    float get_fpu_value(float prior_prob, float exploration_term) const;
};

/**
 * @brief Performance benchmarking utilities
 */
namespace benchmark {

/**
 * @brief Benchmark PUCT selection performance
 *
 * @param tree Test tree with known structure
 * @param parent_index Parent node to select from
 * @param iterations Number of selection iterations to time
 * @param use_simd Whether to enable SIMD optimizations
 * @return Average time per selection in nanoseconds
 */
double benchmark_selection(
    const MCTSTree& tree,
    NodeIndex parent_index,
    int iterations = 10000,
    bool use_simd = true
);

/**
 * @brief Create test tree for benchmarking with specified structure
 *
 * @param num_children Number of children per parent node
 * @param depth Tree depth
 * @return Configured test tree
 */
std::unique_ptr<MCTSTree> create_benchmark_tree(
    std::uint16_t num_children = 16,
    int depth = 3
);

} // namespace benchmark

} // namespace mcts
