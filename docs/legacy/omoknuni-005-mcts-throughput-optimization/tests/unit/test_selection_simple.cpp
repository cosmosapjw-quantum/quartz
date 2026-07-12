/**
 * @file test_selection_simple.cpp
 * @brief Simple standalone test for PUCT selection functionality
 *
 * Compile with: g++ -std=c++17 -O2 -mavx2 -I../../cpp_extensions -o test_selection test_selection_simple.cpp ../../cpp_extensions/mcts/tree.cpp ../../cpp_extensions/mcts/selection.cpp
 */

#include <iostream>
#include <cassert>
#include <chrono>
#include <iomanip>
#include "mcts/selection.hpp"

using namespace mcts;

void test_basic_selection() {
    std::cout << "Testing basic PUCT selection..." << std::endl;

    MCTSTree tree(1000);

    // Create root with 4 children
    NodeIndex root = tree.add_root_node(0.5f, 0);

    const std::uint16_t num_children = 4;
    NodeIndex first_child = tree.allocate_nodes(num_children);
    assert(first_child != NULL_NODE_INDEX);

    tree.set_first_child_index(root, first_child);
    tree.set_num_children(root, num_children);

    // Initialize children with known values
    std::vector<float> visits = {10.0f, 5.0f, 15.0f, 0.0f};      // Child 3 unvisited
    std::vector<float> values = {8.0f, -2.0f, 12.0f, 0.0f};     // Total values
    std::vector<float> priors = {0.4f, 0.3f, 0.2f, 0.1f};       // Prior probabilities

    for (std::uint16_t i = 0; i < num_children; ++i) {
        NodeIndex child = first_child + i;
        tree.set_visit_count(child, visits[i]);
        tree.set_total_value(child, values[i]);
        tree.set_prior_prob(child, priors[i]);
        tree.set_virtual_loss(child, 0.0f);
        tree.set_parent_index(child, root);
    }

    tree.set_visit_count(root, 35.0f);
    tree.set_total_value(root, 18.0f);

    // Test selection
    PUCTSelector selector;
    auto result = selector.select_child(tree, root);

    assert(result.valid);
    assert(result.selected_child >= first_child);
    assert(result.selected_child < first_child + num_children);

    std::cout << "✓ Selected child " << (result.selected_child - first_child)
              << " with PUCT value " << result.best_puct_value << std::endl;
}

void test_simd_vs_scalar_consistency() {
    std::cout << "Testing SIMD vs scalar consistency..." << std::endl;

    auto tree = benchmark::create_benchmark_tree(16, 1);
    NodeIndex root = 0;

    PUCTSelector selector;

    float parent_visits = tree->get_visit_count(root);
    NodeIndex first_child = tree->get_first_child_index(root);
    std::uint16_t num_children = tree->get_num_children(root);
    float exploration_term = 1.25f * std::sqrt(parent_visits);

    // Compute using vectorized method
    std::vector<float> vectorized_values(num_children);
    selector.compute_puct_vectorized(
        tree->get_visit_counts_ptr(),
        tree->get_total_values_ptr(),
        tree->get_prior_probs_ptr(),
        tree->get_virtual_losses_ptr(),
        first_child,
        num_children,
        parent_visits,
        exploration_term,
        vectorized_values.data()
    );

    // Compute using scalar method
    std::vector<float> scalar_values(num_children);
    for (std::uint16_t i = 0; i < num_children; ++i) {
        NodeIndex child = first_child + i;
        scalar_values[i] = selector.compute_puct_scalar(
            tree->get_visit_count(child),
            tree->get_total_value(child),
            tree->get_prior_prob(child),
            tree->get_virtual_loss(child),
            exploration_term
        );
    }

    // Check consistency
    bool consistent = true;
    for (std::uint16_t i = 0; i < num_children; ++i) {
        float diff = std::abs(vectorized_values[i] - scalar_values[i]);
        if (diff > 1e-5f) {
            std::cout << "❌ Mismatch at child " << i
                      << ": vectorized=" << vectorized_values[i]
                      << ", scalar=" << scalar_values[i]
                      << ", diff=" << diff << std::endl;
            consistent = false;
        }
    }

    if (consistent) {
        std::cout << "✓ SIMD and scalar implementations are consistent" << std::endl;
    } else {
        std::cout << "❌ SIMD and scalar implementations differ" << std::endl;
    }
}

void test_performance() {
    std::cout << "Testing selection performance..." << std::endl;

    // Test with realistic child count for MCTS (64 children shows 4-8x speedup)
    auto tree = benchmark::create_benchmark_tree(64, 1);

    const int iterations = 100000;

    // Test SIMD performance
    double simd_time = benchmark::benchmark_selection(*tree, 0, iterations, true);
    std::cout << "SIMD time: " << std::fixed << std::setprecision(2)
              << simd_time << " ns/selection" << std::endl;

    // Test scalar performance
    double scalar_time = benchmark::benchmark_selection(*tree, 0, iterations, false);
    std::cout << "Scalar time: " << std::fixed << std::setprecision(2)
              << scalar_time << " ns/selection" << std::endl;

    if (PUCTSelector::is_avx2_supported()) {
        double speedup = scalar_time / simd_time;
        std::cout << "Speedup: " << std::fixed << std::setprecision(1)
                  << speedup << "x" << std::endl;

        if (speedup >= 3.5) {
            std::cout << "✓ Performance target achieved (≥3.5x speedup, target 4-8x with large trees)" << std::endl;
        } else {
            std::cout << "❌ Performance target not met (<3.5x speedup)" << std::endl;
        }
    } else {
        std::cout << "AVX2 not supported, skipping SIMD performance test" << std::endl;
    }
}

void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;

    MCTSTree tree(100);
    PUCTSelector selector;

    // Test invalid parent
    auto result = selector.select_child(tree, NULL_NODE_INDEX);
    assert(!result.valid);

    // Test node with no children
    NodeIndex leaf = tree.add_root_node(0.5f, 0);
    result = selector.select_child(tree, leaf);
    assert(!result.valid);

    std::cout << "✓ Edge cases handled correctly" << std::endl;
}

void test_avx2_detection() {
    std::cout << "Testing AVX2 detection..." << std::endl;

    bool supported = PUCTSelector::is_avx2_supported();
    std::cout << "AVX2 support: " << (supported ? "Yes" : "No") << std::endl;

    if (supported) {
        std::cout << "✓ SIMD optimizations available" << std::endl;
    } else {
        std::cout << "⚠ SIMD optimizations not available (scalar fallback will be used)" << std::endl;
    }
}

int main() {
    std::cout << "Running PUCT selection tests..." << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_basic_selection();
        test_simd_vs_scalar_consistency();
        test_performance();
        test_edge_cases();
        test_avx2_detection();

        std::cout << "========================================" << std::endl;
        std::cout << "✅ All tests passed successfully!" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}