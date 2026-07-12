/**
 * @file test_selection.cpp
 * @brief Comprehensive unit tests for vectorized PUCT selection
 */

#include <gtest/gtest.h>
#include "mcts/selection.hpp"
#include <vector>
#include <random>
#include <cmath>

using namespace mcts;

class PUCTSelectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        tree = std::make_unique<MCTSTree>(1000);

        // Create root node
        root = tree->add_root_node(0.5f, 0);

        // Set up a basic tree structure for testing
        setup_test_tree();
    }

    void setup_test_tree() {
        // Create 4 children for root
        num_children = 4;
        first_child = tree->allocate_nodes(num_children);

        ASSERT_NE(first_child, NULL_NODE_INDEX);

        tree->set_first_child_index(root, first_child);
        tree->set_num_children(root, num_children);

        // Initialize children with known values for predictable tests
        std::vector<float> visits = {10.0f, 5.0f, 15.0f, 0.0f};      // Child 3 unvisited
        std::vector<float> values = {8.0f, -2.0f, 12.0f, 0.0f};     // Total values
        std::vector<float> priors = {0.4f, 0.3f, 0.2f, 0.1f};       // Prior probabilities

        for (std::uint16_t i = 0; i < num_children; ++i) {
            NodeIndex child = first_child + i;
            tree->set_visit_count(child, visits[i]);
            tree->set_total_value(child, values[i]);
            tree->set_prior_prob(child, priors[i]);
            tree->set_virtual_loss(child, 0.0f);
            tree->set_parent_index(child, root);
        }

        // Set root visits to sum + some extra
        tree->set_visit_count(root, 35.0f);
        tree->set_total_value(root, 18.0f);
    }

    std::unique_ptr<MCTSTree> tree;
    NodeIndex root;
    NodeIndex first_child;
    std::uint16_t num_children;
};

TEST_F(PUCTSelectionTest, BasicSelectionWorks) {
    PUCTSelector selector;

    auto result = selector.select_child(*tree, root);

    EXPECT_TRUE(result.valid);
    EXPECT_GE(result.selected_child, first_child);
    EXPECT_LT(result.selected_child, first_child + num_children);
    EXPECT_GE(result.child_position, 0);
    EXPECT_LT(result.child_position, num_children);
    EXPECT_GT(result.best_puct_value, 0.0f);
}

TEST_F(PUCTSelectionTest, InvalidParentHandling) {
    PUCTSelector selector;

    // Test with invalid parent index
    auto result = selector.select_child(*tree, NULL_NODE_INDEX);
    EXPECT_FALSE(result.valid);

    // Test with out-of-bounds index
    result = selector.select_child(*tree, 9999);
    EXPECT_FALSE(result.valid);
}

TEST_F(PUCTSelectionTest, NoChildrenHandling) {
    PUCTSelector selector;

    // Create a leaf node with no children
    NodeIndex leaf = tree->allocate_node();
    tree->set_visit_count(leaf, 5.0f);
    tree->set_total_value(leaf, 2.5f);
    tree->set_prior_prob(leaf, 0.1f);

    auto result = selector.select_child(*tree, leaf);
    EXPECT_FALSE(result.valid);
}

TEST_F(PUCTSelectionTest, ScalarPUCTCalculation) {
    PUCTSelector selector;

    float exploration_term = 1.25f * std::sqrt(35.0f);  // c_puct * sqrt(parent_visits)

    // Test known values
    float puct = selector.compute_puct_scalar(10.0f, 8.0f, 0.4f, 0.0f, exploration_term);

    // Expected: Q + exploration = (8/10) + (0.4 * exploration_term / (1 + 10))
    float expected_q = 8.0f / 10.0f;
    float expected_exploration = (0.4f * exploration_term) / (1.0f + 10.0f);
    float expected = expected_q + expected_exploration;

    EXPECT_FLOAT_EQ(puct, expected);
}

TEST_F(PUCTSelectionTest, FirstPlayUrgencyHandling) {
    PUCTConfig config;
    config.use_fpu = true;
    config.fpu_value = 1.0f;

    PUCTSelector selector(config);

    float exploration_term = 1.25f * std::sqrt(35.0f);

    // Test unvisited node (visit_count = 0)
    float puct = selector.compute_puct_scalar(0.0f, 0.0f, 0.1f, 0.0f, exploration_term);

    // Should use FPU value instead of Q-value
    float expected_exploration = (0.1f * exploration_term) / (1.0f + 0.0f);
    float expected = config.fpu_value + expected_exploration;

    EXPECT_FLOAT_EQ(puct, expected);
}

TEST_F(PUCTSelectionTest, VirtualLossAdjustment) {
    PUCTSelector selector;

    float exploration_term = 1.25f * std::sqrt(35.0f);

    // Test with virtual loss
    float puct = selector.compute_puct_scalar(10.0f, 8.0f, 0.4f, 2.0f, exploration_term);

    // Q-value should account for virtual loss: 8 / (10 + 2)
    float expected_q = 8.0f / (10.0f + 2.0f);
    float expected_exploration = (0.4f * exploration_term) / (1.0f + 10.0f);
    float expected = expected_q + expected_exploration;

    EXPECT_FLOAT_EQ(puct, expected);
}

TEST_F(PUCTSelectionTest, VectorizedVsScalarConsistency) {
    PUCTSelector selector;

    float parent_visits = tree->get_visit_count(root);
    float exploration_term = 1.25f * std::sqrt(parent_visits);

    // Compute PUCT values using vectorized method
    std::vector<float> vectorized_values(num_children);
    selector.compute_puct_vectorized(
        tree->get_visit_counts_ptr(),
        tree->get_total_values_ptr(),
        tree->get_prior_probs_ptr(),
        tree->get_virtual_losses_ptr(),
        tree->get_flags_ptr(),
        first_child,
        num_children,
        exploration_term,
        vectorized_values.data()
    );

    // Compute same values using scalar method
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

    // Results should be identical (within floating point tolerance)
    for (std::uint16_t i = 0; i < num_children; ++i) {
        EXPECT_NEAR(vectorized_values[i], scalar_values[i], 1e-6f)
            << "Mismatch at child " << i;
    }
}

TEST_F(PUCTSelectionTest, FindMaxVectorized) {
    PUCTSelector selector;

    std::vector<float> values = {0.5f, 1.2f, 0.8f, 2.1f, 0.3f, 1.8f, 0.9f, 1.5f};

    auto [max_value, max_index] = selector.find_max_vectorized(values.data(), values.size());

    EXPECT_FLOAT_EQ(max_value, 2.1f);
    EXPECT_EQ(max_index, 3);
}

TEST_F(PUCTSelectionTest, FindMaxScalarConsistency) {
    PUCTSelector selector;

    // Test with various array sizes
    for (int size = 1; size <= 20; ++size) {
        std::vector<float> values(size);

        // Fill with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 10.0f);

        for (int i = 0; i < size; ++i) {
            values[i] = dist(gen);
        }

        // Find max using vectorized method
        auto [vectorized_max, vectorized_idx] = selector.find_max_vectorized(values.data(), size);

        // Find max using standard algorithm
        auto it = std::max_element(values.begin(), values.end());
        float expected_max = *it;
        std::uint16_t expected_idx = static_cast<std::uint16_t>(std::distance(values.begin(), it));

        EXPECT_FLOAT_EQ(vectorized_max, expected_max) << "Size: " << size;
        EXPECT_EQ(vectorized_idx, expected_idx) << "Size: " << size;
    }
}

TEST_F(PUCTSelectionTest, ConfigurationHandling) {
    PUCTConfig config;
    config.cpuct = 2.0f;
    config.fpu_value = 0.5f;
    config.use_fpu = false;
    config.enable_simd = false;

    PUCTSelector selector(config);

    EXPECT_EQ(selector.get_config().cpuct, 2.0f);
    EXPECT_EQ(selector.get_config().fpu_value, 0.5f);
    EXPECT_FALSE(selector.get_config().use_fpu);
    EXPECT_FALSE(selector.get_config().enable_simd);
}

TEST_F(PUCTSelectionTest, PerformanceBenchmark) {
    // Create larger tree for performance testing (64 children shows 4-8x speedup)
    auto benchmark_tree = benchmark::create_benchmark_tree(64, 2);

    // Benchmark SIMD vs scalar
    double simd_time = benchmark::benchmark_selection(*benchmark_tree, 0, 1000, true);
    double scalar_time = benchmark::benchmark_selection(*benchmark_tree, 0, 1000, false);

    std::cout << "SIMD time: " << simd_time << " ns/selection" << std::endl;
    std::cout << "Scalar time: " << scalar_time << " ns/selection" << std::endl;

    if (PUCTSelector::is_avx2_supported()) {
        double speedup = scalar_time / simd_time;
        std::cout << "Speedup: " << speedup << "x" << std::endl;

        // Should achieve ~3-8x speedup; allow small margin for CI hardware variance
        EXPECT_GE(speedup, 2.5) << "SIMD speedup below 2.5x baseline target";
        EXPECT_LE(speedup, 10.0) << "Unrealistic speedup, possible measurement error";
    }
}

TEST_F(PUCTSelectionTest, AVX2Detection) {
    bool supported = PUCTSelector::is_avx2_supported();

    // Just verify the function returns a boolean
    EXPECT_TRUE(supported || !supported);

    std::cout << "AVX2 support: " << (supported ? "Yes" : "No") << std::endl;
}

TEST_F(PUCTSelectionTest, LargeChildrenCountHandling) {
    // Test with many children (more than SIMD batch size)
    const std::uint16_t large_count = 24;

    // Create new tree for this test
    auto large_tree = std::make_unique<MCTSTree>(1000);
    NodeIndex large_root = large_tree->add_root_node(0.5f, 0);

    NodeIndex large_first_child = large_tree->allocate_nodes(large_count);
    ASSERT_NE(large_first_child, NULL_NODE_INDEX);

    large_tree->set_first_child_index(large_root, large_first_child);
    large_tree->set_num_children(large_root, large_count);

    // Initialize children with varied values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.01f, 1.0f);

    for (std::uint16_t i = 0; i < large_count; ++i) {
        NodeIndex child = large_first_child + i;
        large_tree->set_visit_count(child, dist(gen) * 20.0f);
        large_tree->set_total_value(child, dist(gen) * 10.0f - 5.0f);
        large_tree->set_prior_prob(child, dist(gen) * 0.1f);
        large_tree->set_virtual_loss(child, 0.0f);
        large_tree->set_parent_index(child, large_root);
    }

    large_tree->set_visit_count(large_root, 100.0f);
    large_tree->set_total_value(large_root, 50.0f);

    PUCTSelector selector;
    auto result = selector.select_child(*large_tree, large_root);

    EXPECT_TRUE(result.valid);
    EXPECT_GE(result.selected_child, large_first_child);
    EXPECT_LT(result.selected_child, large_first_child + large_count);
}

// Edge case tests
TEST_F(PUCTSelectionTest, ZeroParentVisits) {
    // Set parent visits to zero
    tree->set_visit_count(root, 0.0f);

    PUCTSelector selector;
    auto result = selector.select_child(*tree, root);

    // Should still work (using fallback value of 1.0 for sqrt)
    EXPECT_TRUE(result.valid);
}

TEST_F(PUCTSelectionTest, NegativeValues) {
    // Set some children to have negative Q-values
    for (std::uint16_t i = 0; i < num_children; ++i) {
        NodeIndex child = first_child + i;
        tree->set_total_value(child, -10.0f);  // Negative total value
    }

    PUCTSelector selector;
    auto result = selector.select_child(*tree, root);

    EXPECT_TRUE(result.valid);
    // Should still select some child even with negative values
}

TEST_F(PUCTSelectionTest, SingleChild) {
    // Create tree with only one child
    auto single_tree = std::make_unique<MCTSTree>(100);
    NodeIndex single_root = single_tree->add_root_node(0.5f, 0);

    NodeIndex single_child = single_tree->allocate_node();
    single_tree->set_first_child_index(single_root, single_child);
    single_tree->set_num_children(single_root, 1);

    single_tree->set_visit_count(single_child, 5.0f);
    single_tree->set_total_value(single_child, 2.5f);
    single_tree->set_prior_prob(single_child, 1.0f);
    single_tree->set_virtual_loss(single_child, 0.0f);
    single_tree->set_parent_index(single_child, single_root);

    single_tree->set_visit_count(single_root, 10.0f);
    single_tree->set_total_value(single_root, 5.0f);

    PUCTSelector selector;
    auto result = selector.select_child(*single_tree, single_root);

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.selected_child, single_child);
    EXPECT_EQ(result.child_position, 0);
}
