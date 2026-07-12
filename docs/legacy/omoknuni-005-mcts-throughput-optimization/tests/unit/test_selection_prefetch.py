#!/usr/bin/env python3
"""
Unit tests for T013: Selection Prefetching

Validates that adding __builtin_prefetch() hints does not change
the correctness of PUCT selection. Tests both SIMD and scalar paths.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import mcts_py


class TestSelectionPrefetch:
    """Test suite for selection prefetching optimization (T013)."""

    def test_prefetch_does_not_change_selection_result(self):
        """Verify prefetch hints don't affect which child is selected."""
        tree = mcts_py.MCTSTree(max_nodes=1000)

        # Create root with 20 children (enough for SIMD + scalar paths)
        root = tree.add_root_node(0.5, 0)
        first_child = tree.allocate_nodes(20)
        tree.set_first_child_index(root, first_child)
        tree.set_num_children(root, 20)

        # Initialize children with distinctive values
        for i in range(20):
            child_idx = first_child + i
            tree.set_visit_count(child_idx, float(i + 1))
            tree.set_total_value(child_idx, float(i) * 0.1)
            tree.set_prior_prob(child_idx, 1.0 / 20)
            tree.set_virtual_loss(child_idx, 0.0)
            tree.set_parent_index(child_idx, root)

        # Set root visits
        tree.set_visit_count(root, 210.0)
        tree.set_total_value(root, 19.0)

        # Create selector and select child
        config = mcts_py.PUCTConfig()
        config.cpuct = 1.5
        config.enable_simd = True

        selector = mcts_py.create_puct_selector(config)
        result = selector.select_child(tree, root)

        # Validate result
        assert result.valid, "Selection should succeed"
        assert result.selected_child >= first_child
        assert result.selected_child < first_child + 20

        # Store for comparison with determinism test
        first_selection = result.selected_child

        # Run again - should get same result
        result2 = selector.select_child(tree, root)
        assert result2.selected_child == first_selection, "Selection should be deterministic"

    def test_prefetch_with_various_child_counts(self):
        """Test prefetching with different numbers of children."""
        test_cases = [
            (1, "single child"),
            (7, "less than SIMD batch"),
            (8, "exactly one SIMD batch"),
            (9, "SIMD batch + 1 scalar"),
            (16, "exactly two SIMD batches"),
            (20, "SIMD + scalar mix"),
            (64, "many children (8 SIMD batches)"),
        ]

        for num_children, description in test_cases:
            tree = mcts_py.MCTSTree(max_nodes=1000)
            root = tree.add_root_node(0.5, 0)

            if num_children == 0:
                continue

            first_child = tree.allocate_nodes(num_children)
            tree.set_first_child_index(root, first_child)
            tree.set_num_children(root, num_children)

            # Initialize all children
            for i in range(num_children):
                child_idx = first_child + i
                tree.set_visit_count(child_idx, float(i + 1))
                tree.set_total_value(child_idx, 0.0)
                tree.set_prior_prob(child_idx, 1.0 / num_children)
                tree.set_virtual_loss(child_idx, 0.0)
                tree.set_parent_index(child_idx, root)

            tree.set_visit_count(root, float(num_children * (num_children + 1) // 2))
            tree.set_total_value(root, 0.0)

            # Select child
            config = mcts_py.PUCTConfig()
            config.enable_simd = True
            selector = mcts_py.create_puct_selector(config)
            result = selector.select_child(tree, root)

            assert result.valid, f"Selection failed for {description}"
            assert result.selected_child >= first_child
            assert result.selected_child < first_child + num_children

    def test_prefetch_with_simd_disabled(self):
        """Test that prefetching works when SIMD is disabled (scalar path only)."""
        tree = mcts_py.MCTSTree(max_nodes=1000)
        root = tree.add_root_node(0.5, 0)

        # Create 20 children to test scalar prefetching
        first_child = tree.allocate_nodes(20)
        tree.set_first_child_index(root, first_child)
        tree.set_num_children(root, 20)

        for i in range(20):
            child_idx = first_child + i
            tree.set_visit_count(child_idx, float(i + 1))
            tree.set_total_value(child_idx, float(i) * 0.1)
            tree.set_prior_prob(child_idx, 1.0 / 20)
            tree.set_virtual_loss(child_idx, 0.0)
            tree.set_parent_index(child_idx, root)

        tree.set_visit_count(root, 210.0)
        tree.set_total_value(root, 19.0)

        # Disable SIMD to force scalar path
        config = mcts_py.PUCTConfig()
        config.enable_simd = False
        selector = mcts_py.create_puct_selector(config)
        result = selector.select_child(tree, root)

        assert result.valid, "Scalar selection with prefetch should succeed"
        assert result.selected_child >= first_child
        assert result.selected_child < first_child + 20

    def test_prefetch_with_expanding_nodes(self):
        """Test prefetching when some nodes are marked as expanding."""
        tree = mcts_py.MCTSTree(max_nodes=1000)
        root = tree.add_root_node(0.5, 0)

        # Create 16 children
        first_child = tree.allocate_nodes(16)
        tree.set_first_child_index(root, first_child)
        tree.set_num_children(root, 16)

        for i in range(16):
            child_idx = first_child + i
            tree.set_visit_count(child_idx, float(i + 1))
            tree.set_total_value(child_idx, 0.0)
            tree.set_prior_prob(child_idx, 1.0 / 16)
            tree.set_virtual_loss(child_idx, 0.0)
            tree.set_parent_index(child_idx, root)

            # Mark some nodes as expanding
            if i % 3 == 0:
                tree.set_expanding(child_idx, True)

        tree.set_visit_count(root, 136.0)
        tree.set_total_value(root, 0.0)

        # Select child - should skip expanding nodes
        config = mcts_py.PUCTConfig()
        config.enable_simd = True
        selector = mcts_py.create_puct_selector(config)
        result = selector.select_child(tree, root)

        assert result.valid, "Selection should succeed even with expanding nodes"
        assert not tree.is_expanding(result.selected_child), "Should not select expanding node"

    def test_prefetch_benchmark_correctness(self):
        """Validate prefetch correctness with many children."""
        # Create a tree with 64 children (full SIMD utilization)
        tree = mcts_py.MCTSTree(max_nodes=1000)
        root = tree.add_root_node(0.5, 0)

        # Create 64 children
        first_child = tree.allocate_nodes(64)
        tree.set_first_child_index(root, first_child)
        tree.set_num_children(root, 64)

        for i in range(64):
            child_idx = first_child + i
            tree.set_visit_count(child_idx, float(i + 1))
            tree.set_total_value(child_idx, float(i) * 0.05)
            tree.set_prior_prob(child_idx, 1.0 / 64)
            tree.set_virtual_loss(child_idx, 0.0)
            tree.set_parent_index(child_idx, root)

        tree.set_visit_count(root, 2080.0)
        tree.set_total_value(root, 100.0)

        # Run benchmark with SIMD enabled
        config = mcts_py.PUCTConfig()
        config.enable_simd = True
        selector = mcts_py.create_puct_selector(config)

        # Select multiple times - should be deterministic
        first_result = selector.select_child(tree, root)
        assert first_result.valid

        for _ in range(100):
            result = selector.select_child(tree, root)
            assert result.valid
            assert result.selected_child == first_result.selected_child, \
                "Selection should be deterministic with prefetching"

    def test_prefetch_does_not_affect_puct_values(self):
        """Verify PUCT values are unchanged by prefetching."""
        tree = mcts_py.MCTSTree(max_nodes=1000)
        root = tree.add_root_node(0.5, 0)

        # Create test with known PUCT values
        first_child = tree.allocate_nodes(10)
        tree.set_first_child_index(root, first_child)
        tree.set_num_children(root, 10)

        for i in range(10):
            child_idx = first_child + i
            tree.set_visit_count(child_idx, 10.0)
            tree.set_total_value(child_idx, 5.0)
            tree.set_prior_prob(child_idx, 0.1)
            tree.set_virtual_loss(child_idx, 0.0)
            tree.set_parent_index(child_idx, root)

        tree.set_visit_count(root, 100.0)
        tree.set_total_value(root, 50.0)

        # Select with SIMD
        config_simd = mcts_py.PUCTConfig()
        config_simd.enable_simd = True
        selector_simd = mcts_py.PUCTSelector(config_simd)
        result_simd = selector_simd.select_child(tree, root)

        # Select with scalar (no SIMD)
        config_scalar = mcts_py.PUCTConfig()
        config_scalar.enable_simd = False
        selector_scalar = mcts_py.PUCTSelector(config_scalar)
        result_scalar = selector_scalar.select_child(tree, root)

        # Both should select same child
        assert result_simd.selected_child == result_scalar.selected_child, \
            "SIMD and scalar paths with prefetch should select same child"

        # PUCT values should be very close (within floating point precision)
        assert abs(result_simd.best_puct_value - result_scalar.best_puct_value) < 1e-6, \
            "PUCT values should match between SIMD and scalar"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
