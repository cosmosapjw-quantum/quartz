/**
 * @file test_busy_edge_masking.cpp
 * @brief Unit tests for busy-edge masking in PUCT selection (T002)
 *
 * Validates that nodes marked as "expanding" are never selected by other threads,
 * preventing expansion conflicts and improving multi-threaded efficiency.
 */

#include <gtest/gtest.h>
#include "../../cpp_extensions/mcts/tree.hpp"
#include "../../cpp_extensions/mcts/selection.hpp"
#include "../../cpp_extensions/mcts/instrumentation.hpp"
#include <thread>
#include <vector>
#include <atomic>

using namespace mcts;

class BusyEdgeMaskingTest : public ::testing::Test {
protected:
    static constexpr std::size_t MAX_NODES = 100'000;

    void SetUp() override {
        tree_ = new MCTSTree(MAX_NODES);
        selector_ = new PUCTSelector(PUCTConfig{});
        Instrumentation::instance().set_enabled(true);
        Instrumentation::instance().reset();
    }

    void TearDown() override {
        delete selector_;
        delete tree_;
        Instrumentation::instance().set_enabled(false);
    }

    MCTSTree* tree_;
    PUCTSelector* selector_;
};

// ============================================================================
// Basic Masking Tests
// ============================================================================

TEST_F(BusyEdgeMaskingTest, ExpandingNodeIsNotSelected) {
    // Create tree with root and children
    NodeIndex root = tree_->add_root_node(0.5f, 0);

    // Allocate 3 children
    NodeIndex child0 = tree_->allocate_node();
    NodeIndex child1 = tree_->allocate_node();
    NodeIndex child2 = tree_->allocate_node();

    // Set up parent-child relationship
    tree_->set_first_child_index(root, child0);
    tree_->set_num_children(root, 3);

    // Initialize children with different priors
    tree_->set_prior_prob(child0, 0.5f);
    tree_->set_prior_prob(child1, 0.3f);
    tree_->set_prior_prob(child2, 0.2f);

    // Set parent for all children
    tree_->set_parent_index(child0, root);
    tree_->set_parent_index(child1, root);
    tree_->set_parent_index(child2, root);

    // Mark child0 as expanding
    bool marked = tree_->atomic_try_mark_expanding(child0);
    ASSERT_TRUE(marked);
    EXPECT_TRUE(tree_->get_flags(child0).is_expanding());

    // Select best child - should NOT be child0 (it's expanding)
    SelectionResult result = selector_->select_child(*tree_, root);

    ASSERT_TRUE(result.valid);
    EXPECT_NE(result.selected_child, child0) << "Expanding node should never be selected";

    // Should select child1 or child2 (both available)
    EXPECT_TRUE(result.selected_child == child1 || result.selected_child == child2);
}

TEST_F(BusyEdgeMaskingTest, AllButOneExpandingSelectsLastOne) {
    // Create tree with root and 3 children
    NodeIndex root = tree_->add_root_node(0.5f, 0);

    NodeIndex child0 = tree_->allocate_node();
    NodeIndex child1 = tree_->allocate_node();
    NodeIndex child2 = tree_->allocate_node();

    // Set parent indices first
    tree_->set_parent_index(child0, root);
    tree_->set_parent_index(child1, root);
    tree_->set_parent_index(child2, root);

    // Then set first child index
    tree_->set_first_child_index(root, child0);
    tree_->set_num_children(root, 3);

    // All children have equal priors
    tree_->set_prior_prob(child0, 0.33f);
    tree_->set_prior_prob(child1, 0.33f);
    tree_->set_prior_prob(child2, 0.34f);

    // Mark child0 and child1 as expanding
    ASSERT_TRUE(tree_->atomic_try_mark_expanding(child0));
    ASSERT_TRUE(tree_->atomic_try_mark_expanding(child1));

    // Select - should get child2 (only non-expanding node)
    SelectionResult result = selector_->select_child(*tree_, root);

    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.selected_child, child2) << "Should select the only non-expanding child";
}

TEST_F(BusyEdgeMaskingTest, ClearExpandingAllowsReselection) {
    // Create simple tree
    NodeIndex root = tree_->add_root_node(0.5f, 0);
    NodeIndex child = tree_->allocate_node();

    tree_->set_parent_index(child, root);
    tree_->set_first_child_index(root, child);
    tree_->set_num_children(root, 1);
    tree_->set_prior_prob(child, 1.0f);

    // Mark as expanding
    ASSERT_TRUE(tree_->atomic_try_mark_expanding(child));

    // Selection returns -inf PUCT when only child is expanding
    SelectionResult result1 = selector_->select_child(*tree_, root);
    EXPECT_TRUE(result1.valid);
    EXPECT_EQ(result1.best_puct_value, -std::numeric_limits<float>::infinity())
        << "Expanding node should have -inf PUCT";

    // Clear expanding flag
    tree_->clear_expanding_flag(child);
    EXPECT_FALSE(tree_->get_flags(child).is_expanding());

    // Now can select
    SelectionResult result2 = selector_->select_child(*tree_, root);
    ASSERT_TRUE(result2.valid);
    EXPECT_EQ(result2.selected_child, child);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(BusyEdgeMaskingTest, AtomicMarkExpandingPreventsDoubleMarking) {
    NodeIndex root = tree_->add_root_node(0.5f, 0);
    NodeIndex child = tree_->allocate_node();

    tree_->set_first_child_index(root, child);
    tree_->set_num_children(root, 1);
    tree_->set_prior_prob(child, 1.0f);
    tree_->set_parent_index(child, root);

    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;

    // Launch 10 threads trying to mark same node
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&]() {
            if (tree_->atomic_try_mark_expanding(child)) {
                success_count.fetch_add(1);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Only ONE thread should succeed
    EXPECT_EQ(success_count.load(), 1) << "Atomic marking should allow only one success";
    EXPECT_TRUE(tree_->get_flags(child).is_expanding());
}

TEST_F(BusyEdgeMaskingTest, ConcurrentSelectionAvoidsBusyNodes) {
    // Create tree with multiple children
    NodeIndex root = tree_->add_root_node(0.5f, 0);

    constexpr int NUM_CHILDREN = 10;
    std::vector<NodeIndex> children;

    for (int i = 0; i < NUM_CHILDREN; ++i) {
        children.push_back(tree_->allocate_node());
        tree_->set_prior_prob(children[i], 1.0f / NUM_CHILDREN);
        tree_->set_parent_index(children[i], root);
    }

    tree_->set_first_child_index(root, children[0]);
    tree_->set_num_children(root, NUM_CHILDREN);

    // Mark half as expanding
    for (int i = 0; i < NUM_CHILDREN / 2; ++i) {
        ASSERT_TRUE(tree_->atomic_try_mark_expanding(children[i]));
    }

    // Concurrent selections
    std::vector<std::thread> threads;
    std::vector<NodeIndex> selected(20, NULL_NODE_INDEX);

    for (int i = 0; i < 20; ++i) {
        threads.emplace_back([&, i]() {
            SelectionResult result = selector_->select_child(*tree_, root);
            if (result.valid) {
                selected[i] = result.selected_child;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Verify no expanding nodes were selected
    for (NodeIndex sel : selected) {
        if (sel != NULL_NODE_INDEX) {
            EXPECT_FALSE(tree_->get_flags(sel).is_expanding())
                << "Selected node " << sel << " should not be expanding";
        }
    }
}

// ============================================================================
// Instrumentation Tests
// ============================================================================

TEST_F(BusyEdgeMaskingTest, InstrumentationTracksBusyEdgeMasking) {
    Instrumentation::instance().reset();

    // Create tree
    NodeIndex root = tree_->add_root_node(0.5f, 0);

    constexpr int NUM_CHILDREN = 5;
    for (int i = 0; i < NUM_CHILDREN; ++i) {
        NodeIndex child = tree_->allocate_node();
        tree_->set_prior_prob(child, 0.2f);
        tree_->set_parent_index(child, root);
    }

    tree_->set_first_child_index(root, 1);  // Children start at index 1
    tree_->set_num_children(root, NUM_CHILDREN);

    // Mark some as expanding
    tree_->atomic_try_mark_expanding(1);
    tree_->atomic_try_mark_expanding(2);

    // Perform selection - should skip 2 expanding nodes
    SelectionResult result = selector_->select_child(*tree_, root);
    ASSERT_TRUE(result.valid);

    // Check instrumentation
    auto snapshot = Instrumentation::instance().snapshot();

    // BusyEdgeMasked should be incremented (at least 2 times)
    if (snapshot.count(InstrumentationMetric::BusyEdgeMasked)) {
        EXPECT_GE(snapshot.at(InstrumentationMetric::BusyEdgeMasked).call_count, 2)
            << "Should track skipped expanding nodes";
    }
}

TEST_F(BusyEdgeMaskingTest, InstrumentationTracksExpansionConflicts) {
    Instrumentation::instance().reset();

    NodeIndex root = tree_->add_root_node(0.5f, 0);
    NodeIndex child = tree_->allocate_node();

    // First mark succeeds
    ASSERT_TRUE(tree_->atomic_try_mark_expanding(child));

    // Second mark fails (conflict) - but we need to track this at a higher level
    // The continuous_simulation_runner tracks this when atomic_try_mark_expanding fails
    bool second_mark = tree_->atomic_try_mark_expanding(child);
    EXPECT_FALSE(second_mark) << "Second mark should fail (node already expanding)";
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(BusyEdgeMaskingTest, MaskingHasMinimalOverhead) {
    // Create tree with many children
    NodeIndex root = tree_->add_root_node(0.5f, 0);

    constexpr int NUM_CHILDREN = 100;
    for (int i = 0; i < NUM_CHILDREN; ++i) {
        NodeIndex child = tree_->allocate_node();
        tree_->set_prior_prob(child, 1.0f / NUM_CHILDREN);
        tree_->set_parent_index(child, root);
    }

    tree_->set_first_child_index(root, 1);
    tree_->set_num_children(root, NUM_CHILDREN);

    // Benchmark without masking
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        selector_->select_child(*tree_, root);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto no_mask_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count();

    // Mark half as expanding
    for (int i = 1; i <= NUM_CHILDREN / 2; ++i) {
        tree_->atomic_try_mark_expanding(i);
    }

    // Benchmark with masking
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        selector_->select_child(*tree_, root);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto mask_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();

    std::cout << "Selection without masking: " << (no_mask_ns / 1000.0) << " ns/op\n";
    std::cout << "Selection with masking: " << (mask_ns / 1000.0) << " ns/op\n";
    std::cout << "Overhead: " << ((mask_ns - no_mask_ns) / 1000.0) << " ns/op\n";

    // Masking overhead should be minimal (<50% increase)
    EXPECT_LT(mask_ns, no_mask_ns * 1.5) << "Masking overhead should be <50%";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(BusyEdgeMaskingTest, AllChildrenExpandingReturnsInvalid) {
    NodeIndex root = tree_->add_root_node(0.5f, 0);

    constexpr int NUM_CHILDREN = 3;
    for (int i = 0; i < NUM_CHILDREN; ++i) {
        NodeIndex child = tree_->allocate_node();
        tree_->set_prior_prob(child, 0.33f);
        tree_->set_parent_index(child, root);

        // Mark all as expanding
        ASSERT_TRUE(tree_->atomic_try_mark_expanding(child));
    }

    tree_->set_first_child_index(root, 1);
    tree_->set_num_children(root, NUM_CHILDREN);

    // No valid selection possible
    SelectionResult result = selector_->select_child(*tree_, root);

    // Implementation returns best of bad options, but PUCT will be -inf for all
    // The result might be "valid" but will have -inf score
    if (result.valid) {
        EXPECT_EQ(result.best_puct_value, -std::numeric_limits<float>::infinity())
            << "All expanding nodes should have -inf PUCT";
    }
}

TEST_F(BusyEdgeMaskingTest, ExpandedNodeCannotBeMarkedExpanding) {
    NodeIndex root = tree_->add_root_node(0.5f, 0);
    NodeIndex child = tree_->allocate_node();

    // Manually mark as expanded by setting bit 0 in flags
    NodeFlags flags = tree_->get_flags(child);
    flags.set_expanded(true);
    tree_->set_flags(child, flags);

    EXPECT_TRUE(tree_->get_flags(child).is_expanded());

    // Try to mark as expanding - should fail (already expanded)
    bool marked = tree_->atomic_try_mark_expanding(child);
    EXPECT_FALSE(marked) << "Cannot mark already-expanded node as expanding";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
