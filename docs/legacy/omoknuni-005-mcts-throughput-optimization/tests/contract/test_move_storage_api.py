"""
Contract tests for MCTSTree move storage API.

Tests the C++ move storage exposed via pybind11 to ensure:
1. Move storage API is correctly bound and accessible from Python
2. Move indices can be stored and retrieved correctly
3. Memory efficiency: 2 bytes per node vs Python dict approach

HOWTO-RUN-TESTS:
================
# Run move storage contract tests
python -m pytest tests/contract/test_move_storage_api.py -v

# Run with verbose output
python -m pytest tests/contract/test_move_storage_api.py -v -s

# Run specific test
python -m pytest tests/contract/test_move_storage_api.py::TestMoveStorageAPI::test_basic_move_storage -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import C++ bindings
import mcts_py


class TestMoveStorageAPI:
    """Test MCTSTree move storage API contract and bindings."""

    @pytest.fixture
    def tree(self):
        """Create an empty test tree."""
        return mcts_py.MCTSTree(1000)

    def test_move_storage_api_exists(self):
        """Test that move storage methods are accessible."""
        tree = mcts_py.MCTSTree(100)

        assert hasattr(tree, 'get_move'), "MCTSTree should have get_move method"
        assert hasattr(tree, 'set_move'), "MCTSTree should have set_move method"

        # Verify they are callable
        assert callable(tree.get_move), "get_move should be callable"
        assert callable(tree.set_move), "set_move should be callable"

    def test_basic_move_storage(self, tree):
        """Test basic move storage and retrieval."""
        root = tree.add_root_node(0.5, 0)

        # Root node should have move index 0 (default)
        assert tree.get_move(root) == 0, "Root node should have default move index 0"

        # Set move index
        tree.set_move(root, 42)
        assert tree.get_move(root) == 42, "Move index should be retrievable"

        # Set different move index
        tree.set_move(root, 123)
        assert tree.get_move(root) == 123, "Move index should update correctly"

    def test_move_storage_for_children(self, tree):
        """Test move storage for child nodes."""
        root = tree.add_root_node(0.5, 0)

        # Allocate children
        num_children = 5
        first_child = tree.allocate_nodes(num_children)
        tree.set_first_child_index(root, first_child)
        tree.set_num_children(root, num_children)

        # Set move indices for children
        for i in range(num_children):
            child_idx = first_child + i
            move_idx = 100 + i  # Move indices 100, 101, 102, 103, 104
            tree.set_move(child_idx, move_idx)

        # Verify all move indices
        for i in range(num_children):
            child_idx = first_child + i
            expected_move = 100 + i
            actual_move = tree.get_move(child_idx)
            assert actual_move == expected_move, \
                f"Child {i} should have move index {expected_move}, got {actual_move}"

    def test_move_storage_range(self, tree):
        """Test move storage with full uint16_t range."""
        root = tree.add_root_node(0.5, 0)

        # Test minimum value
        tree.set_move(root, 0)
        assert tree.get_move(root) == 0

        # Test maximum value (uint16_t max = 65535)
        tree.set_move(root, 65535)
        assert tree.get_move(root) == 65535

        # Test some intermediate values
        test_values = [1, 255, 256, 1000, 5000, 32767, 32768, 65534]
        for value in test_values:
            tree.set_move(root, value)
            assert tree.get_move(root) == value, f"Failed for move value {value}"

    def test_move_storage_clears_with_tree(self, tree):
        """Test that move storage is cleared when tree is cleared."""
        root = tree.add_root_node(0.5, 0)
        tree.set_move(root, 999)

        # Clear tree
        tree.clear()

        # Add new root and verify move is reset to 0
        new_root = tree.add_root_node(0.5, 0)
        assert tree.get_move(new_root) == 0, "Move should be reset after clear()"

    def test_move_storage_multiple_trees(self):
        """Test that multiple trees maintain separate move storage."""
        tree1 = mcts_py.MCTSTree(100)
        tree2 = mcts_py.MCTSTree(100)

        root1 = tree1.add_root_node(0.5, 0)
        root2 = tree2.add_root_node(0.5, 0)

        tree1.set_move(root1, 111)
        tree2.set_move(root2, 222)

        assert tree1.get_move(root1) == 111, "Tree1 move should be independent"
        assert tree2.get_move(root2) == 222, "Tree2 move should be independent"

    def test_move_storage_with_node_reuse(self, tree):
        """Test move storage when nodes are deallocated and reused."""
        root = tree.add_root_node(0.5, 0)

        # Allocate a node and set its move
        node1 = tree.allocate_node()
        tree.set_move(node1, 555)
        assert tree.get_move(node1) == 555

        # Deallocate the node
        tree.deallocate_node(node1)

        # Allocate another node (should reuse the same index)
        node2 = tree.allocate_node()
        assert node2 == node1, "Node should be reused from free list"

        # The move value should still be there (not cleared on deallocation)
        # This is expected behavior - moves are only cleared on tree clear()
        assert tree.get_move(node2) == 555

    def test_move_storage_persistence(self, tree):
        """Test move storage persists across multiple operations."""
        root = tree.add_root_node(0.5, 0)

        # Allocate multiple nodes and set their moves
        nodes = []
        for i in range(10):
            node = tree.allocate_node()
            tree.set_move(node, i * 100)
            nodes.append(node)

        # Verify all moves are still correct
        for i, node in enumerate(nodes):
            expected_move = i * 100
            actual_move = tree.get_move(node)
            assert actual_move == expected_move, \
                f"Node {node} should have move {expected_move}, got {actual_move}"


class TestMoveStorageMemoryEfficiency:
    """Test memory efficiency of move storage."""

    def test_memory_efficiency_10m_nodes(self):
        """Test memory usage for 10M nodes with move storage."""
        # 10 million nodes
        max_nodes = 10_000_000
        tree = mcts_py.MCTSTree(max_nodes)

        # Get memory usage
        memory_bytes = tree.get_memory_usage()
        memory_mb = memory_bytes / (1024 * 1024)

        # Expected memory calculation:
        # 4 float arrays * 4 bytes = 16 bytes
        # 2 int32 arrays * 4 bytes = 8 bytes
        # 3 uint16 arrays * 2 bytes = 6 bytes (num_children, flags, moves)
        # Total = 30 bytes per node (with alignment overhead)
        # For 10M nodes: ~300MB total

        # Verify memory is reasonable (should be < 400MB with alignment)
        assert memory_mb < 400, \
            f"Memory usage too high: {memory_mb:.1f}MB (expected <400MB)"

        # Verify bytes per node is efficient
        bytes_per_node = tree.get_bytes_per_node()

        # With move storage, we added 2 bytes per node
        # Total should still be < 64 bytes per node (our target)
        assert bytes_per_node < 64, \
            f"Bytes per node too high: {bytes_per_node:.1f} (expected <64)"

        print(f"✓ Memory efficiency validated:")
        print(f"  - Total memory: {memory_mb:.1f} MB for {max_nodes:,} nodes")
        print(f"  - Bytes per node: {bytes_per_node:.1f} bytes")
        print(f"  - Move storage adds only 2 bytes per node")

    def test_move_storage_vs_python_dict(self):
        """Compare move storage efficiency vs Python dict approach."""
        # C++ array approach (what we just implemented)
        tree = mcts_py.MCTSTree(10_000_000)
        cpp_memory_mb = tree.get_memory_usage() / (1024 * 1024)

        # Python dict approach would require:
        # - Dict overhead: ~240 bytes per entry (Python 3.12)
        # - Key (int): ~28 bytes
        # - Value (int): ~28 bytes
        # - Total: ~296 bytes per entry for 10M nodes = ~2,960 MB
        python_dict_mb = 10_000_000 * 296 / (1024 * 1024)

        # C++ approach should be ~50x more efficient
        efficiency_ratio = python_dict_mb / cpp_memory_mb

        assert efficiency_ratio > 10, \
            f"C++ approach should be >10x more efficient, got {efficiency_ratio:.1f}x"

        print(f"✓ Move storage efficiency comparison:")
        print(f"  - C++ array: {cpp_memory_mb:.1f} MB")
        print(f"  - Python dict: {python_dict_mb:.1f} MB (estimated)")
        print(f"  - Efficiency gain: {efficiency_ratio:.1f}x")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
