"""
Unit tests for virtual loss mechanism in MCTS tree search.

Tests cover:
- Basic virtual loss application and removal using REAL C++ implementation
- Thread safety of atomic operations
- Path-based virtual loss management
- RAII guard functionality
- Configuration and edge cases

All tests now use the real C++ MCTS implementation via mcts_py module.
"""

import pytest
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the real MCTS implementation
import mcts_py


class MockMCTSTree:
    """Mock MCTS tree for testing virtual loss without full C++ implementation."""

    def __init__(self, max_nodes=1000):
        self.max_nodes = max_nodes
        self.node_count = 0
        self.virtual_losses = {}
        self.visit_counts = {}
        self.total_values = {}
        self.prior_probs = {}
        self.parent_indices = {}
        self.first_child_indices = {}
        self.num_children = {}
        self.flags = {}

    def add_root_node(self, prior_prob, current_player):
        """Add root node and return its index (always 0)."""
        node_index = 0
        self.node_count = 1
        self.virtual_losses[node_index] = 0.0
        self.visit_counts[node_index] = 0.0
        self.total_values[node_index] = 0.0
        self.prior_probs[node_index] = prior_prob
        self.parent_indices[node_index] = -1  # NULL_NODE_INDEX
        self.first_child_indices[node_index] = -1
        self.num_children[node_index] = 0
        return node_index

    def allocate_nodes(self, count):
        """Allocate multiple contiguous nodes."""
        if self.node_count + count > self.max_nodes:
            return -1  # NULL_NODE_INDEX

        first_index = self.node_count
        for i in range(count):
            node_index = self.node_count + i
            self.virtual_losses[node_index] = 0.0
            self.visit_counts[node_index] = 0.0
            self.total_values[node_index] = 0.0
            self.prior_probs[node_index] = 0.1
            self.parent_indices[node_index] = -1
            self.first_child_indices[node_index] = -1
            self.num_children[node_index] = 0

        self.node_count += count
        return first_index

    def is_valid_index(self, node_index):
        """Check if node index is valid."""
        return 0 <= node_index < self.node_count

    def get_virtual_loss(self, node_index):
        """Get virtual loss for node."""
        return self.virtual_losses.get(node_index, 0.0)

    def set_virtual_loss(self, node_index, value):
        """Set virtual loss for node."""
        if self.is_valid_index(node_index):
            self.virtual_losses[node_index] = value

    def get_virtual_losses_ptr(self):
        """Return mock pointer for atomic operations."""
        return self.virtual_losses


class MockVirtualLossConfig:
    """Mock virtual loss configuration."""

    def __init__(self, magnitude=1.0, enable_virtual_loss=True):
        self.magnitude = magnitude
        self.enable_virtual_loss = enable_virtual_loss


class MockVirtualLossManager:
    """Mock virtual loss manager that simulates the C++ implementation."""

    def __init__(self, tree, config=None):
        self.tree = tree
        self.config = config or MockVirtualLossConfig()
        self.total_applications = 0
        self.total_removals = 0
        self._lock = threading.Lock()  # Simulate atomic operations

    def apply_virtual_loss_to_path(self, path):
        """Apply virtual loss to all nodes in path."""
        if not self.config.enable_virtual_loss or not path:
            return True

        with self._lock:
            for node_index in path:
                if not self.apply_virtual_loss(node_index):
                    # Rollback on failure
                    for prev_node in path:
                        if prev_node == node_index:
                            break
                        self.remove_virtual_loss(prev_node)
                    return False
            return True

    def remove_virtual_loss_from_path(self, path):
        """Remove virtual loss from all nodes in path."""
        if not self.config.enable_virtual_loss or not path:
            return True

        all_success = True
        with self._lock:
            for node_index in path:
                if not self.remove_virtual_loss(node_index):
                    all_success = False
        return all_success

    def apply_virtual_loss(self, node_index, magnitude=-1.0):
        """Apply virtual loss to single node."""
        if not self.tree.is_valid_index(node_index):
            return False

        actual_magnitude = magnitude if magnitude >= 0 else self.config.magnitude

        with self._lock:
            current_vl = self.tree.get_virtual_loss(node_index)
            new_vl = current_vl + actual_magnitude

            # Safety checks
            if new_vl > 1000.0:
                return False

            self.tree.set_virtual_loss(node_index, new_vl)
            self.total_applications += 1
            return True

    def remove_virtual_loss(self, node_index, magnitude=-1.0):
        """Remove virtual loss from single node."""
        if not self.tree.is_valid_index(node_index):
            return False

        actual_magnitude = magnitude if magnitude >= 0 else self.config.magnitude

        with self._lock:
            current_vl = self.tree.get_virtual_loss(node_index)
            new_vl = max(0.0, current_vl - actual_magnitude)

            self.tree.set_virtual_loss(node_index, new_vl)
            self.total_removals += 1
            return True

    def get_virtual_loss(self, node_index):
        """Get current virtual loss value."""
        return self.tree.get_virtual_loss(node_index)

    def reset_all_virtual_loss(self):
        """Reset all virtual loss values to zero."""
        with self._lock:
            for node_index in range(self.tree.node_count):
                self.tree.set_virtual_loss(node_index, 0.0)
            self.total_applications = 0
            self.total_removals = 0

    def get_statistics(self):
        """Get virtual loss statistics."""
        stats = type('VirtualLossStats', (), {})()
        stats.total_applications = self.total_applications
        stats.total_removals = self.total_removals
        stats.current_active_paths = max(0, self.total_applications - self.total_removals)

        # Calculate max and average virtual loss
        vl_values = [vl for vl in self.tree.virtual_losses.values() if vl > 0]
        stats.max_virtual_loss = max(vl_values) if vl_values else 0.0
        stats.avg_virtual_loss = sum(vl_values) / len(vl_values) if vl_values else 0.0

        return stats


class MockVirtualLossGuard:
    """Mock RAII guard for virtual loss management."""

    def __init__(self, manager, path):
        self.manager = manager
        self.path = path
        self.valid = manager.apply_virtual_loss_to_path(path)
        self.released = False

    def is_valid(self):
        return self.valid

    def release(self):
        if self.valid and not self.released:
            self.manager.remove_virtual_loss_from_path(self.path)
            self.released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class TestVirtualLossBasic:
    """Test basic virtual loss operations."""

    def setup_method(self):
        """Set up test fixtures with real implementation."""
        self.tree = mcts_py.create_test_tree(1000)
        self.manager = mcts_py.create_test_virtual_loss_manager(self.tree)

    def test_apply_single_virtual_loss(self):
        """Test applying virtual loss to a single node."""
        node_index = 0
        initial_vl = self.manager.get_virtual_loss(node_index)

        success = self.manager.apply_virtual_loss(node_index)

        assert success is True
        assert self.manager.get_virtual_loss(node_index) == initial_vl + 1.0

    def test_remove_single_virtual_loss(self):
        """Test removing virtual loss from a single node."""
        node_index = 0

        # Apply first, then remove
        self.manager.apply_virtual_loss(node_index)
        initial_vl = self.manager.get_virtual_loss(node_index)

        success = self.manager.remove_virtual_loss(node_index)

        assert success is True
        assert self.manager.get_virtual_loss(node_index) == initial_vl - 1.0

    def test_virtual_loss_cannot_go_negative(self):
        """Test that virtual loss cannot go below zero."""
        node_index = 0

        # Try to remove virtual loss from a node with zero virtual loss
        success = self.manager.remove_virtual_loss(node_index)

        assert success is True
        assert self.manager.get_virtual_loss(node_index) == 0.0

    def test_custom_magnitude(self):
        """Test applying virtual loss with custom magnitude."""
        node_index = 0
        custom_magnitude = 2.5

        success = self.manager.apply_virtual_loss(node_index, custom_magnitude)

        assert success is True
        assert abs(self.manager.get_virtual_loss(node_index) - custom_magnitude) < 1e-6

    def test_invalid_node_index(self):
        """Test operations on invalid node indices."""
        invalid_index = 999

        apply_success = self.manager.apply_virtual_loss(invalid_index)
        remove_success = self.manager.remove_virtual_loss(invalid_index)

        assert apply_success is False
        assert remove_success is False


class TestVirtualLossPath:
    """Test path-based virtual loss operations."""

    def setup_method(self):
        """Set up test fixtures with real tree."""
        self.tree = mcts_py.create_test_tree(1000)
        root = 0  # Root is always at index 0

        # Create a simple path: root -> child1 -> child2
        child1 = self.tree.allocate_node()
        child2 = self.tree.allocate_node()

        self.path = [child2, child1, root]  # Leaf to root order
        self.manager = mcts_py.create_test_virtual_loss_manager(self.tree)

    def test_apply_virtual_loss_to_path(self):
        """Test applying virtual loss to an entire path."""
        # Record initial virtual loss values
        initial_values = [self.manager.get_virtual_loss(node) for node in self.path]

        success = self.manager.apply_virtual_loss_to_path(self.path)

        assert success is True

        # Check that virtual loss was applied to all nodes in path
        for i, node_index in enumerate(self.path):
            expected = initial_values[i] + 1.0
            assert abs(self.manager.get_virtual_loss(node_index) - expected) < 1e-6

    def test_remove_virtual_loss_from_path(self):
        """Test removing virtual loss from an entire path."""
        # Apply virtual loss first
        self.manager.apply_virtual_loss_to_path(self.path)
        initial_values = [self.manager.get_virtual_loss(node) for node in self.path]

        success = self.manager.remove_virtual_loss_from_path(self.path)

        assert success is True

        # Check that virtual loss was removed from all nodes in path
        for i, node_index in enumerate(self.path):
            expected = initial_values[i] - 1.0
            assert abs(self.manager.get_virtual_loss(node_index) - expected) < 1e-6

    def test_empty_path(self):
        """Test operations on empty paths."""
        empty_path = []

        apply_success = self.manager.apply_virtual_loss_to_path(empty_path)
        remove_success = self.manager.remove_virtual_loss_from_path(empty_path)

        assert apply_success is True
        assert remove_success is True

    def test_path_with_invalid_node(self):
        """Test path operations with invalid node indices."""
        invalid_path = [0, 999, 1]  # Middle node is invalid

        success = self.manager.apply_virtual_loss_to_path(invalid_path)

        assert success is False

        # Check that rollback occurred - no virtual loss should remain
        for node_index in [0, 1]:
            if self.tree.is_valid_index(node_index):
                assert self.manager.get_virtual_loss(node_index) == 0.0


class TestVirtualLossGuard:
    """Test RAII virtual loss guard functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tree = mcts_py.create_test_tree(1000)
        root = 0  # Root node is already created by create_test_tree
        child1 = self.tree.allocate_node()
        child2 = self.tree.allocate_node()

        self.path = [child2, child1, root]
        self.manager = mcts_py.create_test_virtual_loss_manager(self.tree)

    def test_guard_automatic_cleanup(self):
        """Test that virtual loss guard automatically cleans up."""
        initial_values = [self.manager.get_virtual_loss(node) for node in self.path]

        # Create guard in a nested scope to test RAII cleanup
        def use_guard():
            guard = mcts_py.VirtualLossGuard(self.manager, self.path)
            assert guard.is_valid() is True

            # Virtual loss should be applied
            for i, node_index in enumerate(self.path):
                expected = initial_values[i] + 1.0
                assert abs(self.manager.get_virtual_loss(node_index) - expected) < 1e-6
            # Guard goes out of scope here and should auto-cleanup

        use_guard()

        # After guard is destroyed, virtual loss should be removed
        for i, node_index in enumerate(self.path):
            assert abs(self.manager.get_virtual_loss(node_index) - initial_values[i]) < 1e-6

    def test_guard_manual_release(self):
        """Test manually releasing virtual loss guard."""
        initial_values = [self.manager.get_virtual_loss(node) for node in self.path]

        guard = mcts_py.VirtualLossGuard(self.manager, self.path)
        assert guard.is_valid() is True

        # Manually release
        guard.release()

        # Virtual loss should be removed
        for i, node_index in enumerate(self.path):
            assert abs(self.manager.get_virtual_loss(node_index) - initial_values[i]) < 1e-6

    def test_guard_with_invalid_path(self):
        """Test guard behavior with invalid paths."""
        invalid_path = [0, 999, 1]

        guard = mcts_py.VirtualLossGuard(self.manager, invalid_path)

        assert guard.is_valid() is False


class TestVirtualLossConfiguration:
    """Test virtual loss configuration options."""

    def test_disabled_virtual_loss(self):
        """Test behavior when virtual loss is disabled."""
        tree = mcts_py.create_test_tree(1000)

        config = mcts_py.VirtualLossConfig(1.0, False)  # disabled
        manager = mcts_py.create_test_virtual_loss_manager(tree, config)

        # Operations should succeed but not actually apply virtual loss
        success = manager.apply_virtual_loss(0)
        assert success is True
        assert manager.get_virtual_loss(0) == 0.0

    def test_custom_magnitude(self):
        """Test virtual loss with custom magnitude."""
        tree = mcts_py.create_test_tree(1000)

        custom_magnitude = 2.5
        config = mcts_py.VirtualLossConfig(custom_magnitude, True)
        manager = mcts_py.create_test_virtual_loss_manager(tree, config)

        manager.apply_virtual_loss(0)

        assert abs(manager.get_virtual_loss(0) - custom_magnitude) < 1e-6


class TestVirtualLossThreadSafety:
    """Test thread safety of virtual loss operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tree = mcts_py.create_test_tree(1000)

        # Create multiple nodes for concurrent access
        for _ in range(10):
            self.tree.allocate_node()
        self.manager = mcts_py.create_test_virtual_loss_manager(self.tree)

    def test_concurrent_apply_remove(self):
        """Test concurrent virtual loss application and removal."""
        node_index = 0
        num_threads = 10
        operations_per_thread = 100

        def worker_apply():
            for _ in range(operations_per_thread):
                self.manager.apply_virtual_loss(node_index)
                time.sleep(0.001)  # Small delay to encourage race conditions

        def worker_remove():
            for _ in range(operations_per_thread):
                self.manager.remove_virtual_loss(node_index)
                time.sleep(0.001)

        # Start threads
        threads = []
        for _ in range(num_threads // 2):
            threads.append(threading.Thread(target=worker_apply))
            threads.append(threading.Thread(target=worker_remove))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Final virtual loss should be non-negative
        final_vl = self.manager.get_virtual_loss(node_index)
        assert final_vl >= 0.0

        # Statistics should be consistent
        stats = self.manager.get_statistics()
        expected_applications = num_threads // 2 * operations_per_thread
        expected_removals = num_threads // 2 * operations_per_thread

        assert stats.total_applications == expected_applications
        assert stats.total_removals == expected_removals

    def test_concurrent_path_operations(self):
        """Test concurrent path-based virtual loss operations."""
        paths = [
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0]
        ]

        def worker(path):
            guard = mcts_py.VirtualLossGuard(self.manager, path)
            if guard.is_valid():
                time.sleep(0.01)  # Hold virtual loss for a short time
            # Guard automatically cleans up when it goes out of scope

        # Run concurrent path operations
        with ThreadPoolExecutor(max_workers=len(paths)) as executor:
            futures = [executor.submit(worker, path) for path in paths]

            for future in as_completed(futures):
                future.result()  # Wait for completion

        # All virtual loss should be cleaned up
        for i in range(5):
            assert self.manager.get_virtual_loss(i) == 0.0


class TestVirtualLossStatistics:
    """Test virtual loss statistics and monitoring."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tree = mcts_py.create_test_tree(1000)
        for _ in range(5):
            self.tree.allocate_node()
        self.manager = mcts_py.create_test_virtual_loss_manager(self.tree)

    def test_statistics_tracking(self):
        """Test that statistics are correctly tracked."""
        initial_stats = self.manager.get_statistics()
        assert initial_stats.total_applications == 0
        assert initial_stats.total_removals == 0

        # Apply virtual loss to multiple nodes
        for i in range(3):
            self.manager.apply_virtual_loss(i)

        stats = self.manager.get_statistics()
        assert stats.total_applications == 3
        assert stats.total_removals == 0
        assert stats.current_active_paths == 3

        # Remove from some nodes
        for i in range(2):
            self.manager.remove_virtual_loss(i)

        final_stats = self.manager.get_statistics()
        assert final_stats.total_applications == 3
        assert final_stats.total_removals == 2
        assert final_stats.current_active_paths == 1

    def test_reset_statistics(self):
        """Test resetting virtual loss statistics."""
        # Apply some virtual loss
        self.manager.apply_virtual_loss(0)
        self.manager.apply_virtual_loss(1)

        # Reset all
        self.manager.reset_all_virtual_loss()

        # Check that everything is reset
        for i in range(2):
            assert self.manager.get_virtual_loss(i) == 0.0

        stats = self.manager.get_statistics()
        assert stats.total_applications == 0
        assert stats.total_removals == 0
        assert stats.current_active_paths == 0
        assert stats.max_virtual_loss == 0.0
        assert stats.avg_virtual_loss == 0.0


@pytest.mark.integration
class TestVirtualLossIntegration:
    """Integration tests with MCTS components."""

    def test_virtual_loss_with_puct_selection(self):
        """Test that virtual loss affects PUCT selection correctly."""
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
            future.set_result((policy, 0.0))
            return future

        mcts = AlphaZeroMCTS(inference_fn)
        mcts.search(game, simulations=5)

        stats = mcts.virtual_loss_manager.get_statistics()
        assert stats.total_applications >= 5
        assert stats.total_removals >= stats.total_applications
        assert stats.current_active_paths == 0

    def test_virtual_loss_in_search_loop(self):
        """Test virtual loss in a complete search loop."""
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
            future.set_result((policy, 0.1))
            return future

        mcts = AlphaZeroMCTS(inference_fn)
        mcts.search(game, simulations=6)

        policy = mcts.get_policy(game, temperature=1.0)
        assert abs(policy.sum() - 1.0) < 1e-6

        stats = mcts.virtual_loss_manager.get_statistics()
        assert stats.total_applications >= 6
        assert stats.current_active_paths == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])