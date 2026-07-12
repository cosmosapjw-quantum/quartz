"""
Unit tests for TreeAdapter (T024f-5)

Tests the adapter layer that wraps TinyNodeTree to provide MCTSTree-compatible API.
Validates that TreeAdapter correctly translates between the two interfaces.

Test categories:
- Basic tree management (init, clear, capacity)
- Node allocation and deallocation
- Node data accessors (get/set visit_count, total_value, etc.)
- Flags operations (expanded, terminal, current_player)
- Atomic operations (thread-safe expansion)
- TinyNodeTree extensions (zobrist_hash, move)
- API equivalence with MCTSTree
"""

import pytest
import numpy as np
import mcts_py


class TestTreeAdapterBasicManagement:
    """Test basic tree management operations"""

    def test_init_empty_tree(self):
        """Test initialization of empty tree"""
        tree = mcts_py.TreeAdapter(1000)
        assert tree.get_node_count() == 0
        assert tree.get_max_nodes() == 1000
        assert tree.get_root_index() == -1  # NULL_NODE_INDEX when empty

    def test_add_root_node(self):
        """Test adding root node"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0, 0x12345)

        assert root == 0
        assert tree.get_node_count() == 1
        assert tree.get_root_index() == 0

        # Check root properties
        assert abs(tree.get_prior_prob(root) - 1.0) < 0.01
        assert tree.get_visit_count(root) == 1  # init_root sets N=1
        assert tree.get_zobrist_hash(root) == 0x12345

    def test_add_root_with_current_player(self):
        """Test root node with current_player flag"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 1, 0)  # Player 1

        flags = tree.get_flags(root)
        assert flags.current_player() == 1

    def test_clear_tree(self):
        """Test clearing tree"""
        tree = mcts_py.TreeAdapter(1000)
        tree.add_root_node(1.0, 0)

        assert tree.get_node_count() == 1

        tree.clear()

        assert tree.get_node_count() == 0
        assert tree.get_root_index() == -1

    def test_memory_usage(self):
        """Test memory usage reporting"""
        tree = mcts_py.TreeAdapter(1000)
        tree.add_root_node(1.0, 0)

        memory = tree.get_memory_usage()
        bytes_per_node = tree.get_bytes_per_node()

        assert memory > 0
        assert bytes_per_node == 64  # TinyNode is 64 bytes (aligned)


class TestTreeAdapterAllocation:
    """Test node allocation and deallocation"""

    def test_allocate_single_node(self):
        """Test allocating a single node"""
        tree = mcts_py.TreeAdapter(1000)
        tree.add_root_node(1.0, 0)

        node_idx = tree.allocate_node()
        assert node_idx > 0
        assert tree.get_node_count() == 2

    def test_allocate_multiple_nodes(self):
        """Test allocating multiple nodes at once"""
        tree = mcts_py.TreeAdapter(1000)
        tree.add_root_node(1.0, 0)

        first_idx = tree.allocate_nodes(5)
        assert first_idx > 0
        assert tree.get_node_count() == 6  # 1 root + 5 allocated

    def test_deallocate_node(self):
        """Test deallocating a single node"""
        tree = mcts_py.TreeAdapter(1000)
        tree.add_root_node(1.0, 0)

        node_idx = tree.allocate_node()
        tree.deallocate_node(node_idx)

        # Note: TinyNodeTree doesn't decrease node_count on deallocation
        # (nodes go to free list)
        assert tree.get_node_count() == 2

    def test_deallocate_multiple_nodes(self):
        """Test deallocating multiple nodes"""
        tree = mcts_py.TreeAdapter(1000)
        tree.add_root_node(1.0, 0)

        first_idx = tree.allocate_nodes(5)
        tree.deallocate_nodes(first_idx, 5)

        # Nodes should be in free list for reuse
        assert tree.get_node_count() == 6

    def test_has_space_for(self):
        """Test checking available space"""
        tree = mcts_py.TreeAdapter(100)
        tree.add_root_node(1.0, 0)

        assert tree.has_space_for(50)
        assert tree.has_space_for(99)
        assert not tree.has_space_for(1000)

    def test_get_available_nodes(self):
        """Test getting available node count"""
        tree = mcts_py.TreeAdapter(100)
        tree.add_root_node(1.0, 0)

        available = tree.get_available_nodes()
        assert available == 99  # 100 - 1 root


class TestTreeAdapterNodeAccessors:
    """Test node data accessor methods"""

    def test_visit_count(self):
        """Test get/set visit_count"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # Root starts with N=1
        assert tree.get_visit_count(root) == 1

        tree.set_visit_count(root, 10)
        assert tree.get_visit_count(root) == 10

    def test_total_value(self):
        """Test get/set total_value"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # Root starts with W=0
        assert tree.get_total_value(root) == 0.0

        tree.set_total_value(root, 5.5)
        assert abs(tree.get_total_value(root) - 5.5) < 0.001

    def test_prior_prob(self):
        """Test get/set prior_prob"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(0.8, 0)

        assert abs(tree.get_prior_prob(root) - 0.8) < 0.01

        tree.set_prior_prob(root, 0.5)
        assert abs(tree.get_prior_prob(root) - 0.5) < 0.01

    def test_virtual_loss(self):
        """Test get/set virtual_loss"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # Virtual loss starts at 0
        assert tree.get_virtual_loss(root) == 0

        tree.set_virtual_loss(root, 3)
        assert tree.get_virtual_loss(root) == 3

    def test_parent_index(self):
        """Test get/set parent_index"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)
        child = tree.allocate_node()

        tree.set_parent_index(child, root)
        assert tree.get_parent_index(child) == root

    def test_first_child_index(self):
        """Test get/set first_child_index"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)
        child = tree.allocate_node()

        tree.set_first_child_index(root, child)
        assert tree.get_first_child_index(root) == child

    def test_num_children_via_sibling_links(self):
        """Test get_num_children (counts sibling-linked children)"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # Initially no children
        assert tree.get_num_children(root) == 0

        # Use underlying TinyNodeTree to add children properly
        tiny_tree = tree.get_tiny_tree()
        tiny_tree.add_child(root, 1, 0.5, 0x1000)
        tiny_tree.add_child(root, 2, 0.5, 0x2000)

        # Now should count 2 children
        assert tree.get_num_children(root) == 2

    def test_node_info(self):
        """Test get_node_info"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(0.8, 0)

        tree.set_visit_count(root, 10)
        tree.set_total_value(root, 5.0)

        info = tree.get_node_info(root)
        assert info.index == root
        assert info.visit_count == 10
        assert abs(info.total_value - 5.0) < 0.001
        assert abs(info.prior_prob - 0.8) < 0.01
        assert abs(info.q_value() - 0.5) < 0.01  # 5.0 / 10


class TestTreeAdapterFlags:
    """Test node flags operations"""

    def test_expanded_flag(self):
        """Test expanded flag"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        flags = tree.get_flags(root)
        assert not flags.is_expanded()

        # Set expanded via underlying TinyNode
        tiny_tree = tree.get_tiny_tree()
        tiny_tree.add_child(root, 1, 0.5, 0x1000)

        flags = tree.get_flags(root)
        assert flags.is_expanded()

    def test_terminal_flag(self):
        """Test terminal flag"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # Get flags and set terminal
        flags = tree.get_flags(root)
        assert not flags.is_terminal()

        flags.set_terminal(True)
        tree.set_flags(root, flags)

        # Verify terminal is set
        flags = tree.get_flags(root)
        assert flags.is_terminal()

    def test_current_player_flag(self):
        """Test current_player flag"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 1)  # Player 1

        flags = tree.get_flags(root)
        assert flags.current_player() == 1

        # Change to player 0
        flags.set_current_player(0)
        tree.set_flags(root, flags)

        flags = tree.get_flags(root)
        assert flags.current_player() == 0

    def test_set_flags_preserves_all_bits(self):
        """Test that set_flags preserves all flag bits"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 1)

        # Set multiple flags
        flags = tree.get_flags(root)
        flags.set_expanded(True)
        flags.set_terminal(True)
        flags.set_current_player(1)
        tree.set_flags(root, flags)

        # Verify all flags preserved
        flags = tree.get_flags(root)
        assert flags.is_expanded()
        assert flags.is_terminal()
        assert flags.current_player() == 1


class TestTreeAdapterAtomicOperations:
    """Test atomic operations for thread safety"""

    def test_atomic_try_set_expanded(self):
        """Test atomically setting expanded flag"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # First attempt should succeed
        result = tree.atomic_try_set_expanded(root)
        assert result is True

        flags = tree.get_flags(root)
        assert flags.is_expanded()

        # Second attempt should fail (already expanded)
        result = tree.atomic_try_set_expanded(root)
        assert result is False

    def test_atomic_try_mark_expanding(self):
        """Test atomically marking node as expanding"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # First attempt should succeed
        result = tree.atomic_try_mark_expanding(root)
        assert result is True

        flags = tree.get_flags(root)
        assert flags.is_expanding()

        # Second attempt should fail (already expanding)
        result = tree.atomic_try_mark_expanding(root)
        assert result is False

    def test_clear_expanding_flag(self):
        """Test clearing expanding flag"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # Mark as expanding
        tree.atomic_try_mark_expanding(root)
        flags = tree.get_flags(root)
        assert flags.is_expanding()

        # Clear expanding flag
        tree.clear_expanding_flag(root)
        flags = tree.get_flags(root)
        assert not flags.is_expanding()

    def test_expanding_cleared_after_expansion(self):
        """Test that expanding flag doesn't interfere with expanded"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # Mark as expanding
        tree.atomic_try_mark_expanding(root)

        # Set expanded
        tree.atomic_try_set_expanded(root)

        # Both should be set
        flags = tree.get_flags(root)
        assert flags.is_expanding()
        assert flags.is_expanded()

        # Clear expanding
        tree.clear_expanding_flag(root)

        # Expanded should still be set
        flags = tree.get_flags(root)
        assert not flags.is_expanding()
        assert flags.is_expanded()


class TestTreeAdapterTinyNodeExtensions:
    """Test TinyNodeTree-specific extensions"""

    def test_zobrist_hash(self):
        """Test get/set zobrist_hash"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0, 0x123456789ABCDEF0)

        assert tree.get_zobrist_hash(root) == 0x123456789ABCDEF0

        tree.set_zobrist_hash(root, 0xFEDCBA9876543210)
        assert tree.get_zobrist_hash(root) == 0xFEDCBA9876543210

    def test_move(self):
        """Test get/set move"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # Root has no move (default 0)
        assert tree.get_move(root) == 0

        # Set move
        tree.set_move(root, 42)
        assert tree.get_move(root) == 42

    def test_get_tiny_tree(self):
        """Test accessing underlying TinyNodeTree"""
        tree = mcts_py.TreeAdapter(1000)
        tree.add_root_node(1.0, 0)

        tiny_tree = tree.get_tiny_tree()
        assert tiny_tree is not None
        assert tiny_tree.get_node_count() == 1


class TestTreeAdapterMCTSTreeEquivalence:
    """Test equivalence with MCTSTree API patterns"""

    def test_typical_node_expansion_pattern(self):
        """Test typical MCTS expansion pattern"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0, 0)

        # Allocate child nodes (like MCTSTree would)
        child1 = tree.allocate_node()
        child2 = tree.allocate_node()

        # Set child properties
        tree.set_parent_index(child1, root)
        tree.set_parent_index(child2, root)
        tree.set_prior_prob(child1, 0.6)
        tree.set_prior_prob(child2, 0.4)
        tree.set_move(child1, 1)
        tree.set_move(child2, 2)

        # Link children to parent
        tree.set_first_child_index(root, child1)

        # Verify parent-child relationship
        assert tree.get_parent_index(child1) == root
        assert tree.get_parent_index(child2) == root
        assert tree.get_first_child_index(root) == child1

    def test_backup_simulation_pattern(self):
        """Test typical MCTS backup pattern"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        child = tree.allocate_node()
        tree.set_parent_index(child, root)

        # Simulate backup
        tree.set_visit_count(child, 1)
        tree.set_total_value(child, 0.8)

        tree.set_visit_count(root, 2)
        tree.set_total_value(root, -0.8)  # Negamax

        # Verify Q-values
        info_child = tree.get_node_info(child)
        info_root = tree.get_node_info(root)

        assert abs(info_child.q_value() - 0.8) < 0.01
        assert abs(info_root.q_value() - (-0.4)) < 0.01

    def test_virtual_loss_pattern(self):
        """Test virtual loss application pattern"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # Apply virtual loss
        tree.set_virtual_loss(root, 3)
        assert tree.get_virtual_loss(root) == 3

        # Remove virtual loss
        tree.set_virtual_loss(root, 0)
        assert tree.get_virtual_loss(root) == 0

    def test_tree_validation(self):
        """Test tree validation"""
        tree = mcts_py.TreeAdapter(1000)
        tree.add_root_node(1.0, 0)

        # Tree should be valid after root creation
        assert tree.validate_tree()


class TestTreeAdapterEdgeCases:
    """Test edge cases and error handling"""

    def test_allocate_beyond_capacity(self):
        """Test allocating beyond tree capacity"""
        tree = mcts_py.TreeAdapter(10)  # Very small tree
        tree.add_root_node(1.0, 0)

        # Allocate until full
        for i in range(9):
            idx = tree.allocate_node()
            assert idx >= 0

        # Next allocation should fail
        idx = tree.allocate_node()
        assert idx == -1  # NULL_NODE_INDEX

    def test_set_num_children_is_noop(self):
        """Test that set_num_children is a no-op (TinyNode uses sibling links)"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # This should be a no-op
        tree.set_num_children(root, 5)

        # num_children should still be 0 (no actual children added)
        assert tree.get_num_children(root) == 0

    def test_large_value_clamping(self):
        """Test that virtual loss clamps to uint8_t max (255)"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # Set virtual loss beyond uint8_t max
        tree.set_virtual_loss(root, 1000.0)

        # Should clamp to 255
        assert tree.get_virtual_loss(root) == 255

    def test_value_scaling_precision(self):
        """Test that value scaling maintains precision"""
        tree = mcts_py.TreeAdapter(1000)
        root = tree.add_root_node(1.0, 0)

        # Test various values
        test_values = [0.123456, -0.987654, 1.0, -1.0, 0.0]

        for value in test_values:
            tree.set_total_value(root, value)
            retrieved = tree.get_total_value(root)
            assert abs(retrieved - value) < 0.001  # 6 decimal places precision


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
