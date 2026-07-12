"""
Unit tests for MCTS value backup mechanism with sign flipping.

Tests cover:
- Basic backup operations with proper sign flipping
- Atomic operations and thread safety
- Path validation and error handling
- Integration with virtual loss manager
- Terminal value backup
- Statistical tracking and monitoring
"""

import pytest
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

# Mock classes that simulate the C++ implementation for testing


class MockMCTSTree:
    """Mock MCTS tree for testing backup without full C++ implementation."""

    def __init__(self, max_nodes=1000):
        self.max_nodes = max_nodes
        self.node_count = 0
        self.visit_counts = {}
        self.total_values = {}
        self.virtual_losses = {}
        self.parent_indices = {}
        self._lock = threading.Lock()  # Simulate atomic operations

    def add_root_node(self, prior_prob, current_player):
        """Add root node and return its index (always 0)."""
        node_index = 0
        self.node_count = 1
        self.visit_counts[node_index] = 0.0
        self.total_values[node_index] = 0.0
        self.virtual_losses[node_index] = 0.0
        self.parent_indices[node_index] = -1  # NULL_NODE_INDEX
        return node_index

    def allocate_nodes(self, count):
        """Allocate multiple contiguous nodes."""
        if self.node_count + count > self.max_nodes:
            return -1  # NULL_NODE_INDEX

        first_index = self.node_count
        for i in range(count):
            node_index = self.node_count + i
            self.visit_counts[node_index] = 0.0
            self.total_values[node_index] = 0.0
            self.virtual_losses[node_index] = 0.0
            self.parent_indices[node_index] = -1

        self.node_count += count
        return first_index

    def is_valid_index(self, node_index):
        """Check if node index is valid."""
        return 0 <= node_index < self.node_count

    def get_visit_count(self, node_index):
        return self.visit_counts.get(node_index, 0.0)

    def get_total_value(self, node_index):
        return self.total_values.get(node_index, 0.0)

    def get_parent_index(self, node_index):
        return self.parent_indices.get(node_index, -1)

    def set_parent_index(self, node_index, parent):
        if self.is_valid_index(node_index):
            self.parent_indices[node_index] = parent

    def get_visit_counts_ptr(self):
        return self.visit_counts

    def get_total_values_ptr(self):
        return self.total_values


class MockBackupConfig:
    """Mock backup configuration."""

    def __init__(self, enable_value_clipping=True, enable_statistics=True,
                 value_clip_min=-1.0, value_clip_max=1.0):
        self.enable_value_clipping = enable_value_clipping
        self.enable_statistics = enable_statistics
        self.value_clip_min = value_clip_min
        self.value_clip_max = value_clip_max


class MockVirtualLossManager:
    """Mock virtual loss manager for testing integration."""

    def __init__(self):
        self.removed_paths = []

    def remove_virtual_loss_from_path(self, path):
        self.removed_paths.append(path.copy())
        return True


class MockBackupManager:
    """Mock backup manager that simulates the C++ implementation."""

    def __init__(self, tree, config=None):
        self.tree = tree
        self.config = config or MockBackupConfig()
        self.total_backups = 0
        self.successful_backups = 0
        self.total_nodes_updated = 0
        self.path_validation_failures = 0
        self.cumulative_path_length = 0.0
        self.cumulative_leaf_value = 0.0
        self._lock = threading.Lock()

    def backup_value_along_path(self, path, leaf_value, virtual_loss_manager=None):
        """Backup leaf value along path with proper sign flipping."""
        result = type('BackupResult', (), {})()
        result.success = False
        result.nodes_updated = 0
        result.final_root_value = 0.0
        result.original_leaf_value = leaf_value

        with self._lock:
            self.total_backups += 1

            # Validate path
            if not self.validate_backup_path(path):
                self.path_validation_failures += 1
                return result

            # Remove virtual loss if manager provided
            if virtual_loss_manager:
                virtual_loss_manager.remove_virtual_loss_from_path(path)

            # Clip value if configured
            current_value = leaf_value
            if self.config.enable_value_clipping:
                current_value = max(self.config.value_clip_min,
                                   min(self.config.value_clip_max, current_value))

            # Backup with sign flipping
            nodes_updated = 0
            for i, node_index in enumerate(path):
                if not self.tree.is_valid_index(node_index):
                    break

                # Sign flipping: alternate value sign at each level
                value_for_node = current_value if (i % 2 == 0) else -current_value

                # Update visit count and total value
                if self.update_node_atomic(node_index, value_for_node, 1.0):
                    nodes_updated += 1
                else:
                    break

            # Check success
            if nodes_updated > 0 and nodes_updated == len(path):
                result.success = True
                self.successful_backups += 1
                result.final_root_value = self.get_q_value(path[-1])

            result.nodes_updated = nodes_updated
            self.total_nodes_updated += nodes_updated

            # Update statistics
            if self.config.enable_statistics:
                self.cumulative_path_length += len(path)
                self.cumulative_leaf_value += abs(leaf_value)

        return result

    def backup_terminal_value(self, path, terminal_value, virtual_loss_manager=None):
        """Backup terminal value (same as regular backup)."""
        return self.backup_value_along_path(path, terminal_value, virtual_loss_manager)

    def update_node_atomic(self, node_index, value_increment, visit_increment=1.0):
        """Atomically update node visit count and total value."""
        if not self.tree.is_valid_index(node_index):
            return False

        current_visits = self.tree.get_visit_count(node_index)
        current_value = self.tree.get_total_value(node_index)

        new_visits = current_visits + visit_increment
        new_value = current_value + value_increment

        # Safety checks
        if new_visits < 0 or new_visits > 1000000:
            return False
        if abs(new_value) > 1000000:
            return False

        self.tree.visit_counts[node_index] = new_visits
        self.tree.total_values[node_index] = new_value

        return True

    def get_q_value(self, node_index):
        """Get Q-value for node (total_value / visit_count)."""
        if not self.tree.is_valid_index(node_index):
            return 0.0

        visit_count = self.tree.get_visit_count(node_index)
        total_value = self.tree.get_total_value(node_index)

        return total_value / visit_count if visit_count > 0 else 0.0

    def validate_backup_path(self, path):
        """Validate that path is correct for backup."""
        if not path:
            return False

        # Check all nodes are valid
        for node_index in path:
            if not self.tree.is_valid_index(node_index):
                return False

        # Check parent-child relationships
        for i in range(1, len(path)):
            child = path[i - 1]
            parent = path[i]
            child_parent = self.tree.get_parent_index(child)
            if child_parent != parent:
                return False

        # Last node should be root
        root_candidate = path[-1]
        if self.tree.get_parent_index(root_candidate) != -1:
            return False

        return True

    def get_statistics(self):
        """Get backup statistics."""
        stats = type('BackupStats', (), {})()
        stats.total_backups = self.total_backups
        stats.successful_backups = self.successful_backups
        stats.total_nodes_updated = self.total_nodes_updated
        stats.path_validation_failures = self.path_validation_failures

        stats.avg_path_length = (self.cumulative_path_length / self.total_backups
                                if self.total_backups > 0 else 0.0)
        stats.avg_absolute_leaf_value = (self.cumulative_leaf_value / self.total_backups
                                        if self.total_backups > 0 else 0.0)

        return stats

    def reset_statistics(self):
        """Reset all statistics."""
        with self._lock:
            self.total_backups = 0
            self.successful_backups = 0
            self.total_nodes_updated = 0
            self.path_validation_failures = 0
            self.cumulative_path_length = 0.0
            self.cumulative_leaf_value = 0.0


class MockBackupGuard:
    """Mock RAII guard for backup with virtual loss cleanup."""

    def __init__(self, backup_manager, virtual_loss_manager, path, leaf_value):
        self.backup_manager = backup_manager
        self.virtual_loss_manager = virtual_loss_manager
        self.path = path
        self.result = backup_manager.backup_value_along_path(
            path, leaf_value, virtual_loss_manager)
        self.cleaned_up = False

    def was_successful(self):
        return self.result.success

    def get_result(self):
        return self.result

    def cleanup(self):
        if not self.cleaned_up:
            self.virtual_loss_manager.remove_virtual_loss_from_path(self.path)
            self.cleaned_up = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class TestBackupBasic:
    """Test basic backup operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tree = MockMCTSTree()
        self.tree.add_root_node(0.5, 0)
        self.manager = MockBackupManager(self.tree)

    def test_single_node_backup(self):
        """Test backup to a single node (root only)."""
        path = [0]  # Root only
        leaf_value = 0.5

        result = self.manager.backup_value_along_path(path, leaf_value)

        assert result.success is True
        assert result.nodes_updated == 1
        assert result.original_leaf_value == leaf_value

        # Check that root was updated correctly
        assert self.tree.get_visit_count(0) == 1.0
        assert self.tree.get_total_value(0) == 0.5
        assert abs(result.final_root_value - 0.5) < 1e-6

    def test_sign_flipping_two_levels(self):
        """Test proper sign flipping across two levels."""
        # Create child node
        child_index = self.tree.allocate_nodes(1)
        self.tree.set_parent_index(child_index, 0)

        path = [child_index, 0]  # Child to root
        leaf_value = 0.8

        result = self.manager.backup_value_along_path(path, leaf_value)

        assert result.success is True
        assert result.nodes_updated == 2

        # Child (level 0): should get +0.8
        assert abs(self.tree.get_total_value(child_index) - 0.8) < 1e-6
        assert self.tree.get_visit_count(child_index) == 1.0

        # Root (level 1): should get -0.8 (sign flipped)
        assert abs(self.tree.get_total_value(0) - (-0.8)) < 1e-6
        assert self.tree.get_visit_count(0) == 1.0

    def test_sign_flipping_three_levels(self):
        """Test proper sign flipping across three levels."""
        # Create grandchild -> child -> root
        child_index = self.tree.allocate_nodes(1)
        grandchild_index = self.tree.allocate_nodes(1)
        self.tree.set_parent_index(child_index, 0)
        self.tree.set_parent_index(grandchild_index, child_index)

        path = [grandchild_index, child_index, 0]
        leaf_value = 0.6

        result = self.manager.backup_value_along_path(path, leaf_value)

        assert result.success is True
        assert result.nodes_updated == 3

        # Grandchild (level 0): +0.6
        assert abs(self.tree.get_total_value(grandchild_index) - 0.6) < 1e-6

        # Child (level 1): -0.6 (flipped)
        assert abs(self.tree.get_total_value(child_index) - (-0.6)) < 1e-6

        # Root (level 2): +0.6 (flipped back)
        assert abs(self.tree.get_total_value(0) - 0.6) < 1e-6

    def test_negative_leaf_value(self):
        """Test backup with negative leaf value."""
        child_index = self.tree.allocate_nodes(1)
        self.tree.set_parent_index(child_index, 0)

        path = [child_index, 0]
        leaf_value = -0.7

        result = self.manager.backup_value_along_path(path, leaf_value)

        assert result.success is True

        # Child: -0.7
        assert abs(self.tree.get_total_value(child_index) - (-0.7)) < 1e-6

        # Root: +0.7 (sign flipped)
        assert abs(self.tree.get_total_value(0) - 0.7) < 1e-6

    def test_multiple_backups_accumulate(self):
        """Test that multiple backups accumulate correctly."""
        path = [0]  # Root only

        # First backup
        result1 = self.manager.backup_value_along_path(path, 0.3)
        assert result1.success is True

        # Second backup
        result2 = self.manager.backup_value_along_path(path, 0.4)
        assert result2.success is True

        # Check accumulated values
        assert self.tree.get_visit_count(0) == 2.0
        assert abs(self.tree.get_total_value(0) - 0.7) < 1e-6  # 0.3 + 0.4
        assert abs(self.manager.get_q_value(0) - 0.35) < 1e-6  # 0.7 / 2.0


class TestBackupValidation:
    """Test backup path validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tree = MockMCTSTree()
        self.tree.add_root_node(0.5, 0)
        self.manager = MockBackupManager(self.tree)

    def test_empty_path(self):
        """Test backup with empty path."""
        result = self.manager.backup_value_along_path([], 0.5)

        assert result.success is False
        assert result.nodes_updated == 0

    def test_invalid_node_in_path(self):
        """Test backup with invalid node index."""
        path = [0, 999]  # Second node is invalid

        result = self.manager.backup_value_along_path(path, 0.5)

        assert result.success is False

    def test_invalid_parent_child_relationship(self):
        """Test backup with incorrect parent-child relationship."""
        # Create two unrelated nodes
        child1 = self.tree.allocate_nodes(1)
        child2 = self.tree.allocate_nodes(1)
        self.tree.set_parent_index(child1, 0)
        # child2 is not child of child1

        path = [child2, child1, 0]  # Invalid relationship

        result = self.manager.backup_value_along_path(path, 0.5)

        assert result.success is False

    def test_path_not_ending_at_root(self):
        """Test backup with path that doesn't end at root."""
        child_index = self.tree.allocate_nodes(1)
        self.tree.set_parent_index(child_index, 0)

        path = [child_index]  # Doesn't end at root

        result = self.manager.backup_value_along_path(path, 0.5)

        assert result.success is False


class TestBackupConfiguration:
    """Test backup configuration options."""

    def test_value_clipping_enabled(self):
        """Test value clipping when enabled."""
        tree = MockMCTSTree()
        tree.add_root_node(0.5, 0)

        config = MockBackupConfig(enable_value_clipping=True,
                                 value_clip_min=-1.0, value_clip_max=1.0)
        manager = MockBackupManager(tree, config)

        # Test clipping of extreme values
        result = manager.backup_value_along_path([0], 2.0)  # Should clip to 1.0

        assert result.success is True
        assert tree.get_total_value(0) == 1.0  # Clipped value

    def test_value_clipping_disabled(self):
        """Test value clipping when disabled."""
        tree = MockMCTSTree()
        tree.add_root_node(0.5, 0)

        config = MockBackupConfig(enable_value_clipping=False)
        manager = MockBackupManager(tree, config)

        # Extreme value should not be clipped
        result = manager.backup_value_along_path([0], 2.0)

        assert result.success is True
        assert tree.get_total_value(0) == 2.0  # Not clipped

    def test_statistics_disabled(self):
        """Test behavior when statistics are disabled."""
        tree = MockMCTSTree()
        tree.add_root_node(0.5, 0)

        config = MockBackupConfig(enable_statistics=False)
        manager = MockBackupManager(tree, config)

        manager.backup_value_along_path([0], 0.5)

        stats = manager.get_statistics()
        # Should still track basic counters but not cumulative stats
        assert stats.total_backups > 0


class TestBackupTerminalValue:
    """Test terminal value backup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tree = MockMCTSTree()
        self.tree.add_root_node(0.5, 0)
        self.manager = MockBackupManager(self.tree)

    def test_terminal_win_value(self):
        """Test backup of terminal win (+1.0)."""
        child_index = self.tree.allocate_nodes(1)
        self.tree.set_parent_index(child_index, 0)

        path = [child_index, 0]
        terminal_value = 1.0  # Win for current player

        result = self.manager.backup_terminal_value(path, terminal_value)

        assert result.success is True

        # Child gets +1.0 (win)
        assert abs(self.tree.get_total_value(child_index) - 1.0) < 1e-6

        # Root gets -1.0 (loss from root's perspective)
        assert abs(self.tree.get_total_value(0) - (-1.0)) < 1e-6

    def test_terminal_loss_value(self):
        """Test backup of terminal loss (-1.0)."""
        path = [0]  # Root only
        terminal_value = -1.0  # Loss

        result = self.manager.backup_terminal_value(path, terminal_value)

        assert result.success is True
        assert abs(self.tree.get_total_value(0) - (-1.0)) < 1e-6

    def test_terminal_draw_value(self):
        """Test backup of terminal draw (0.0)."""
        path = [0]  # Root only
        terminal_value = 0.0  # Draw

        result = self.manager.backup_terminal_value(path, terminal_value)

        assert result.success is True
        assert abs(self.tree.get_total_value(0) - 0.0) < 1e-6


class TestBackupVirtualLossIntegration:
    """Test integration with virtual loss manager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tree = MockMCTSTree()
        self.tree.add_root_node(0.5, 0)
        self.manager = MockBackupManager(self.tree)
        self.vl_manager = MockVirtualLossManager()

    def test_backup_removes_virtual_loss(self):
        """Test that backup removes virtual loss from path."""
        child_index = self.tree.allocate_nodes(1)
        self.tree.set_parent_index(child_index, 0)

        path = [child_index, 0]

        result = self.manager.backup_value_along_path(path, 0.5, self.vl_manager)

        assert result.success is True
        assert len(self.vl_manager.removed_paths) == 1
        assert self.vl_manager.removed_paths[0] == path

    def test_backup_guard_integration(self):
        """Test BackupGuard with virtual loss cleanup."""
        child_index = self.tree.allocate_nodes(1)
        self.tree.set_parent_index(child_index, 0)

        path = [child_index, 0]

        with MockBackupGuard(self.manager, self.vl_manager, path, 0.4) as guard:
            assert guard.was_successful() is True

        # Virtual loss should be removed twice: once in backup, once in guard cleanup
        assert len(self.vl_manager.removed_paths) == 2


class TestBackupStatistics:
    """Test backup statistics tracking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tree = MockMCTSTree()
        self.tree.add_root_node(0.5, 0)
        self.manager = MockBackupManager(self.tree)

    def test_statistics_tracking(self):
        """Test that statistics are correctly tracked."""
        initial_stats = self.manager.get_statistics()
        assert initial_stats.total_backups == 0

        # Perform successful backup
        child_index = self.tree.allocate_nodes(1)
        self.tree.set_parent_index(child_index, 0)
        path = [child_index, 0]

        self.manager.backup_value_along_path(path, 0.6)

        stats = self.manager.get_statistics()
        assert stats.total_backups == 1
        assert stats.successful_backups == 1
        assert stats.total_nodes_updated == 2
        assert abs(stats.avg_path_length - 2.0) < 1e-6
        assert abs(stats.avg_absolute_leaf_value - 0.6) < 1e-6

    def test_failed_backup_statistics(self):
        """Test statistics for failed backups."""
        # Try backup with invalid path
        self.manager.backup_value_along_path([999], 0.5)

        stats = self.manager.get_statistics()
        assert stats.total_backups == 1
        assert stats.successful_backups == 0
        assert stats.path_validation_failures == 1

    def test_reset_statistics(self):
        """Test resetting statistics."""
        # Perform some backups
        self.manager.backup_value_along_path([0], 0.5)
        self.manager.backup_value_along_path([0], 0.3)

        # Reset
        self.manager.reset_statistics()

        stats = self.manager.get_statistics()
        assert stats.total_backups == 0
        assert stats.successful_backups == 0
        assert stats.total_nodes_updated == 0


class TestBackupThreadSafety:
    """Test thread safety of backup operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tree = MockMCTSTree()
        self.tree.add_root_node(0.5, 0)

        # Create multiple children for concurrent backup
        for i in range(10):
            child_index = self.tree.allocate_nodes(1)
            self.tree.set_parent_index(child_index, 0)

        self.manager = MockBackupManager(self.tree)

    def test_concurrent_backups_same_node(self):
        """Test concurrent backups to the same node."""
        num_threads = 10
        backups_per_thread = 100
        path = [0]  # All threads backup to root

        def worker():
            for _ in range(backups_per_thread):
                self.manager.backup_value_along_path(path, 0.1)
                time.sleep(0.001)

        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Check final state
        expected_visits = num_threads * backups_per_thread
        expected_total_value = expected_visits * 0.1

        assert abs(self.tree.get_visit_count(0) - expected_visits) < 1e-6
        assert abs(self.tree.get_total_value(0) - expected_total_value) < 1e-6

    def test_concurrent_backups_different_paths(self):
        """Test concurrent backups to different paths."""
        paths = [[i + 1, 0] for i in range(5)]  # Different child -> root paths

        def worker(path):
            for _ in range(50):
                self.manager.backup_value_along_path(path, 0.2)
                time.sleep(0.001)

        with ThreadPoolExecutor(max_workers=len(paths)) as executor:
            futures = [executor.submit(worker, path) for path in paths]

            for future in as_completed(futures):
                future.result()

        # Check that all children and root were updated
        assert self.tree.get_visit_count(0) == 5 * 50  # Root visited by all paths

        for i in range(1, 6):  # Children indices 1-5
            assert self.tree.get_visit_count(i) == 50
            assert abs(self.tree.get_total_value(i) - (50 * 0.2)) < 1e-6


@pytest.mark.integration
class TestBackupIntegration:
    """Integration tests for backup with other MCTS components."""

    def test_backup_with_selection_integration(self):
        """Test backup integration with PUCT selection."""
        from concurrent.futures import Future
        from src.core.mcts import AlphaZeroMCTS
        from src.games.game_state import create_game_state

        game = create_game_state('gomoku')
        priority_move = 112

        def inference_fn(state):
            future = Future()
            policy = np.zeros(state.action_space_size, dtype=np.float32)
            legal_moves = state.get_legal_moves()
            if legal_moves:
                for move in legal_moves:
                    policy[move] = 0.01
                if priority_move in legal_moves:
                    policy[priority_move] = 0.9
                policy_sum = policy.sum()
                if policy_sum > 0:
                    policy /= policy_sum
            future.set_result((policy, 0.5))
            return future

        mcts = AlphaZeroMCTS(inference_fn)
        visit_counts = mcts.search(game, simulations=8)

        assert len(visit_counts) > 0
        best_move = max(visit_counts, key=visit_counts.get)
        assert best_move == priority_move

        policy = mcts.get_policy(game, temperature=1.0)
        assert abs(policy.sum() - 1.0) < 1e-6
        assert policy[priority_move] == pytest.approx(policy.max(), rel=1e-2)

    def test_backup_in_full_search_loop(self):
        """Test backup in complete search loop."""
        from concurrent.futures import Future
        from src.core.mcts import AlphaZeroMCTS
        from src.games.game_state import create_game_state

        game = create_game_state('gomoku')

        def inference_fn(state):
            future = Future()
            policy = np.zeros(state.action_space_size, dtype=np.float32)
            legal_moves = state.get_legal_moves()
            if legal_moves:
                uniform = 1.0 / len(legal_moves)
                for move in legal_moves:
                    policy[move] = uniform
            future.set_result((policy, 0.25))
            return future

        mcts = AlphaZeroMCTS(inference_fn)
        mcts.search(game, simulations=6)

        value = mcts.get_value(game)
        assert 0.0 <= value <= 1.0

        policy = mcts.get_policy(game, temperature=1.0)
        assert abs(policy.sum() - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])