/**
 * @file selection.cpp
 * @brief Implementation of vectorized PUCT selection for MCTS
 */

#include "selection.hpp"
#include "instrumentation.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <cstring>
#include <limits>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace mcts {

PUCTSelector::PUCTSelector(const PUCTConfig& config) : config_(config) {
    // Disable SIMD if not supported
    if (config_.enable_simd && !is_avx2_supported()) {
        config_.enable_simd = false;
    }
}

SelectionResult PUCTSelector::select_child(const MCTSTree& tree, NodeIndex parent_index) const {
    SelectionResult result = {};
    result.valid = false;

    // Validate parent node
    if (!tree.is_valid_index(parent_index)) {
        return result;
    }

    // Check if parent has children
    NodeIndex first_child = tree.get_first_child_index(parent_index);
    std::uint16_t num_children = tree.get_num_children(parent_index);

    if (first_child == NULL_NODE_INDEX || num_children == 0) {
        return result;  // No children to select from
    }

    // Get parent visit count for exploration term
    float parent_visits = tree.get_visit_count(parent_index);
    if (parent_visits <= 0.0f) {
        parent_visits = 1.0f;  // Avoid division by zero
    }

    // Pre-compute exploration term: c_puct * sqrt(parent_visits)
    float exploration_term = config_.cpuct * std::sqrt(parent_visits);

    // Reuse thread-local buffer for PUCT values to avoid reallocations
    thread_local std::vector<float> puct_values_buffer;
    if (puct_values_buffer.size() < num_children) {
        puct_values_buffer.resize(num_children);
    }
    float* puct_values = puct_values_buffer.data();

    // Compute PUCT values using vectorized implementation
    // This will set -infinity for nodes marked as "expanding"
    compute_puct_vectorized(
        tree.get_visit_counts_ptr(),
        tree.get_total_values_ptr(),
        tree.get_prior_probs_ptr(),
        tree.get_virtual_losses_ptr(),
        tree.get_flags_ptr(),
        first_child,
        num_children,
        exploration_term,
        puct_values
    );

    // Find child with maximum PUCT value
    auto [max_value, max_index] = find_max_vectorized(puct_values, num_children);

    // Populate result
    result.selected_child = first_child + max_index;
    result.best_puct_value = max_value;
    result.child_position = max_index;
    result.valid = true;

    return result;
}

void PUCTSelector::compute_puct_vectorized(
    const float* visit_counts,
    const float* total_values,
    const float* prior_probs,
    const float* virtual_losses,
    const NodeFlags* flags,
    NodeIndex first_child_index,
    std::uint16_t num_children,
    float exploration_term,
    float* puct_values
) const {
    const std::uint16_t simd_batch_size = 8;  // AVX2 processes 8 floats
    std::uint16_t processed = 0;

    // Use SIMD for batches of 8 children
    if (config_.enable_simd && num_children >= simd_batch_size) {
        const __m256 exploration_vec = _mm256_set1_ps(exploration_term);
        const __m256 ones_vec = _mm256_set1_ps(1.0f);
        const __m256 fpu_vec = _mm256_set1_ps(config_.fpu_value);
        const __m256 neg_inf_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

        for (; processed + simd_batch_size <= num_children; processed += simd_batch_size) {
            NodeIndex base_index = first_child_index + processed;

            // T013: Prefetch next batch of child data for improved cache performance
            // Prefetch with temporal locality hint (will be used soon)
            if (processed + 2 * simd_batch_size <= num_children) {
                NodeIndex prefetch_index = base_index + simd_batch_size;
                __builtin_prefetch(&visit_counts[prefetch_index], 0, 3);  // read, high temporal locality
                __builtin_prefetch(&total_values[prefetch_index], 0, 3);
                __builtin_prefetch(&prior_probs[prefetch_index], 0, 3);
                __builtin_prefetch(&virtual_losses[prefetch_index], 0, 3);
                __builtin_prefetch(&flags[prefetch_index], 0, 3);
            }

            // Load 8 values simultaneously
            __m256 visits = _mm256_loadu_ps(&visit_counts[base_index]);
            __m256 values = _mm256_loadu_ps(&total_values[base_index]);
            __m256 priors = _mm256_loadu_ps(&prior_probs[base_index]);
            __m256 virtual_loss = _mm256_loadu_ps(&virtual_losses[base_index]);

            // ✅ CRITICAL FIX: Check expanding flags (bit 3 of flags.flags)
            // Load 8 flags and check if bit 3 is set
            alignas(32) float expanding_mask_array[8];
            for (int i = 0; i < 8; ++i) {
                const uint8_t flag_byte = flags[base_index + i].flags;
                const bool is_expanding = (flag_byte & 0x08) != 0;  // Check bit 3
                expanding_mask_array[i] = is_expanding ? -1.0f : 0.0f;  // -1.0f = all bits set
            }
            __m256 expanding_mask = _mm256_load_ps(expanding_mask_array);

            // Adjust visit counts with virtual loss: visit_count + virtual_loss
            __m256 adjusted_visits = _mm256_add_ps(visits, virtual_loss);

            // Compute Q-values: total_value / max(adjusted_visits, 1.0)
            __m256 visit_max = _mm256_max_ps(adjusted_visits, ones_vec);
            __m256 q_values = _mm256_div_ps(values, visit_max);

            // Handle First Play Urgency for unvisited nodes
            if (config_.use_fpu) {
                // Mask for unvisited nodes (visit_count == 0)
                __m256 unvisited_mask = _mm256_cmp_ps(visits, _mm256_setzero_ps(), _CMP_EQ_OQ);
                q_values = _mm256_blendv_ps(q_values, fpu_vec, unvisited_mask);
            }

            // Compute exploration term: prior * exploration_term / (1 + visit_count)
            __m256 denominator = _mm256_add_ps(visits, ones_vec);
            __m256 exploration = _mm256_div_ps(
                _mm256_mul_ps(priors, exploration_vec),
                denominator
            );

            // Final PUCT: Q + exploration
            __m256 puct = _mm256_add_ps(q_values, exploration);

            // ✅ Set -infinity for expanding nodes (they should never be selected)
            puct = _mm256_blendv_ps(puct, neg_inf_vec, expanding_mask);

            // Store results
            _mm256_storeu_ps(&puct_values[processed], puct);
        }
    }

    // Handle remaining children with scalar operations
    for (; processed < num_children; ++processed) {
        NodeIndex child_index = first_child_index + processed;

        // T013: Prefetch next child's data to hide memory latency
        if (processed + 1 < num_children) {
            NodeIndex next_index = child_index + 1;
            __builtin_prefetch(&visit_counts[next_index], 0, 3);
            __builtin_prefetch(&total_values[next_index], 0, 3);
            __builtin_prefetch(&prior_probs[next_index], 0, 3);
            __builtin_prefetch(&virtual_losses[next_index], 0, 3);
            __builtin_prefetch(&flags[next_index], 0, 3);
        }

        // ✅ CRITICAL FIX: Skip expanding nodes (busy-edge masking)
        if (flags[child_index].is_expanding()) {
            Instrumentation::instance().increment_counter(InstrumentationMetric::BusyEdgeMasked);
            puct_values[processed] = -std::numeric_limits<float>::infinity();
            continue;
        }

        float visit_count = visit_counts[child_index];
        float total_value = total_values[child_index];
        float prior_prob = prior_probs[child_index];
        float virtual_loss = virtual_losses[child_index];

        puct_values[processed] = compute_puct_scalar(
            visit_count, total_value, prior_prob, virtual_loss, exploration_term
        );
    }
}

float PUCTSelector::compute_puct_scalar(
    float visit_count,
    float total_value,
    float prior_prob,
    float virtual_loss,
    float exploration_term
) const {
    // Compute Q-value accounting for virtual loss
    float q_value = compute_q_value(visit_count, total_value, virtual_loss);

    // Handle First Play Urgency for unvisited nodes
    if (config_.use_fpu && visit_count == 0.0f) {
        q_value = get_fpu_value(prior_prob, exploration_term);
    }

    // Compute exploration term: prior * exploration_term / (1 + visit_count)
    float exploration = (prior_prob * exploration_term) / (1.0f + visit_count);

    // Final PUCT value
    return q_value + exploration;
}

std::pair<float, std::uint16_t> PUCTSelector::find_max_vectorized(
    const float* values,
    std::uint16_t count
) const {
    if (count == 0) {
        return {0.0f, 0};
    }

    // Simple and reliable approach: use scalar algorithm
    // SIMD optimization for find_max with correct indexing is complex
    // and the performance gain is minimal compared to PUCT computation
    float max_value = values[0];
    std::uint16_t max_index = 0;

    for (std::uint16_t i = 1; i < count; ++i) {
        if (values[i] > max_value) {
            max_value = values[i];
            max_index = i;
        }
    }

    return {max_value, max_index};
}

float PUCTSelector::compute_q_value(float visit_count, float total_value, float virtual_loss) const {
    // Adjust for virtual loss
    float adjusted_visits = visit_count + virtual_loss;
    if (adjusted_visits <= 0.0f) {
        return 0.0f;  // Unvisited node
    }

    return total_value / adjusted_visits;
}

float PUCTSelector::get_fpu_value(float prior_prob, float exploration_term) const {
    if (config_.use_fpu) {
        return config_.fpu_value;
    }
    return 0.0f;
}

bool PUCTSelector::is_avx2_supported() {
#ifdef _MSC_VER
    int cpui[4];
    __cpuid(cpui, 7);
    return (cpui[1] & (1 << 5)) != 0;  // Check AVX2 bit
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, nullptr) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1 << 5)) != 0;  // Check AVX2 bit
    }
    return false;
#endif
}

namespace benchmark {

double benchmark_selection(
    const MCTSTree& tree,
    NodeIndex parent_index,
    int iterations,
    bool use_simd
) {
    PUCTConfig config;
    config.enable_simd = use_simd;
    PUCTSelector selector(config);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        auto result = selector.select_child(tree, parent_index);
        // Prevent optimization from eliminating the call
        volatile bool dummy = result.valid;
        (void)dummy;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    return static_cast<double>(duration.count()) / iterations;
}

std::unique_ptr<MCTSTree> create_benchmark_tree(std::uint16_t num_children, int depth) {
    auto tree = std::make_unique<MCTSTree>(10000);

    // Create root node
    NodeIndex root = tree->add_root_node(0.5f, 0);

    // Add children to root for benchmarking
    if (num_children > 0) {
        NodeIndex first_child = tree->allocate_nodes(num_children);
        if (first_child != NULL_NODE_INDEX) {
            tree->set_first_child_index(root, first_child);
            tree->set_num_children(root, num_children);

            // Initialize children with realistic values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> prob_dist(0.01f, 0.1f);
            std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
            std::uniform_int_distribution<int> visit_dist(0, 100);

            for (std::uint16_t i = 0; i < num_children; ++i) {
                NodeIndex child = first_child + i;

                float visits = static_cast<float>(visit_dist(gen));
                tree->set_visit_count(child, visits);
                tree->set_total_value(child, value_dist(gen) * visits);
                tree->set_prior_prob(child, prob_dist(gen));
                tree->set_virtual_loss(child, 0.0f);
                tree->set_parent_index(child, root);
            }

            // Set root visits to sum of children
            float total_visits = 0.0f;
            for (std::uint16_t i = 0; i < num_children; ++i) {
                total_visits += tree->get_visit_count(first_child + i);
            }
            tree->set_visit_count(root, total_visits + 10.0f);
            tree->set_total_value(root, value_dist(gen) * total_visits);
        }
    }

    return tree;
}

} // namespace benchmark

} // namespace mcts
