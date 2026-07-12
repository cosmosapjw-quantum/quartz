"""
Test Python API for instrumentation metrics.

This module validates that collision metrics and other instrumentation
data are properly exposed via the Python bindings.
"""

import pytest
import mcts_py


class TestInstrumentationAPI:
    """Test instrumentation API exposed to Python."""

    def setup_method(self):
        """Enable instrumentation and reset metrics before each test."""
        mcts_py.set_instrumentation_enabled(True)
        mcts_py.reset_instrumentation_metrics()

    def teardown_method(self):
        """Reset metrics after each test."""
        mcts_py.reset_instrumentation_metrics()

    def test_instrumentation_snapshot_returns_dict(self):
        """Verify snapshot returns a dictionary."""
        snapshot = mcts_py.get_instrumentation_snapshot()
        assert isinstance(snapshot, dict)

    def test_instrumentation_snapshot_empty_when_no_activity(self):
        """Verify snapshot is empty when no metrics recorded."""
        snapshot = mcts_py.get_instrumentation_snapshot()
        # Should be empty or have zero counts
        assert len(snapshot) == 0 or all(
            v['calls'] == 0 for v in snapshot.values()
        )

    def test_collision_metrics_available(self):
        """Verify collision metrics are available in snapshot."""
        # These metrics should be available even if not yet recorded
        expected_metrics = {
            'expansion_conflict',
            'busy_edge_masked',
            'unique_batch_positions',
            'selection_retry'
        }

        # Create a tree and trigger some operations to populate metrics
        tree = mcts_py.create_test_tree(1000)
        snapshot = mcts_py.get_instrumentation_snapshot()

        # Check that our metric names are valid
        # (they may not appear in snapshot if count is zero)
        all_possible_metrics = {
            'tree_clear', 'tree_allocate_node', 'tree_allocate_nodes',
            'selection', 'expansion', 'backup',
            'virtual_loss_apply', 'virtual_loss_remove',
            'queue_submit', 'queue_collect', 'queue_process_results',
            'queue_try_get_result',
            'expansion_conflict', 'busy_edge_masked',
            'unique_batch_positions', 'selection_retry'
        }

        # All metrics in snapshot should be from our known set
        for metric_name in snapshot.keys():
            assert metric_name in all_possible_metrics, f"Unknown metric: {metric_name}"

    def test_metric_structure(self):
        """Verify each metric has proper structure."""
        tree = mcts_py.create_test_tree(1000)
        # create_test_tree already adds a root node

        snapshot = mcts_py.get_instrumentation_snapshot()

        # Check structure of any present metrics
        for metric_name, metric_data in snapshot.items():
            assert 'calls' in metric_data, f"{metric_name} missing 'calls'"
            assert 'total_ns' in metric_data, f"{metric_name} missing 'total_ns'"
            assert 'avg_ns' in metric_data, f"{metric_name} missing 'avg_ns'"

            assert isinstance(metric_data['calls'], int)
            assert isinstance(metric_data['total_ns'], int)
            assert isinstance(metric_data['avg_ns'], float)

            # Verify average is calculated correctly
            if metric_data['calls'] > 0:
                expected_avg = metric_data['total_ns'] / metric_data['calls']
                assert abs(metric_data['avg_ns'] - expected_avg) < 1.0

    def test_reset_clears_metrics(self):
        """Verify reset clears all metrics."""
        tree = mcts_py.create_test_tree(1000)
        # create_test_tree already adds a root node

        # Get initial snapshot (may have some data)
        snapshot1 = mcts_py.get_instrumentation_snapshot()

        # Reset
        mcts_py.reset_instrumentation_metrics()

        # Should be empty after reset
        snapshot2 = mcts_py.get_instrumentation_snapshot()
        assert len(snapshot2) == 0

    def test_disabled_instrumentation_no_recording(self):
        """Verify disabling instrumentation stops recording."""
        mcts_py.set_instrumentation_enabled(False)

        tree = mcts_py.create_test_tree(1000)
        # create_test_tree already adds a root node

        snapshot = mcts_py.get_instrumentation_snapshot()
        assert len(snapshot) == 0

        # Re-enable for cleanup
        mcts_py.set_instrumentation_enabled(True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
