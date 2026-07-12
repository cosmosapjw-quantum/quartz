"""
Contract tests for SimulationRunner API.

Tests the C++ SimulationRunner class exposed via pybind11 to ensure
the API is correctly bound and accessible from Python.

These tests validate:
1. SimulationRunner can be imported and instantiated
2. Constructor accepts required MCTS components
3. API surface is correctly exposed to Python
4. Methods exist (implementation verified in Phase 2)

Note: SimulationRunner methods are stubs in Phase 1 and will throw
NotImplementedError until Phase 2 implementation. This is expected
behavior for the TDD (Test-Driven Development) approach.

HOWTO-RUN-TESTS:
================
# Run simulation runner contract tests
python -m pytest tests/contract/test_simulation_runner_api.py -v

# Run with verbose output
python -m pytest tests/contract/test_simulation_runner_api.py -v -s

# Run specific test
python -m pytest tests/contract/test_simulation_runner_api.py::TestSimulationRunnerAPI::test_instantiation -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import C++ bindings
import mcts_py


class TestSimulationRunnerAPI:
    """Test SimulationRunner API contract and bindings."""

    @pytest.fixture
    def mcts_components(self):
        """Create MCTS components needed for SimulationRunner."""
        tree = mcts_py.create_test_tree(1000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        vl_manager = mcts_py.create_test_virtual_loss_manager(tree)

        return {
            'tree': tree,
            'selector': selector,
            'backup': backup,
            'vl_manager': vl_manager
        }

    def test_simulation_runner_class_exists(self):
        """Test that SimulationRunner class is accessible from Python."""
        assert hasattr(mcts_py, 'SimulationRunner'), \
            "SimulationRunner should be exported from mcts_py module"

        assert isinstance(mcts_py.SimulationRunner, type), \
            "SimulationRunner should be a class type"

    def test_instantiation(self, mcts_components):
        """Test that SimulationRunner can be instantiated with required components."""
        runner = mcts_py.SimulationRunner(
            mcts_components['tree'],
            mcts_components['selector'],
            mcts_components['backup'],
            mcts_components['vl_manager']
        )

        assert runner is not None, "SimulationRunner should instantiate successfully"
        assert isinstance(runner, mcts_py.SimulationRunner), \
            "Instance should be of type SimulationRunner"

    def test_instantiation_with_kwargs(self, mcts_components):
        """Test that SimulationRunner can be instantiated with keyword arguments."""
        runner = mcts_py.SimulationRunner(
            tree=mcts_components['tree'],
            selector=mcts_components['selector'],
            backup=mcts_components['backup'],
            virtual_loss=mcts_components['vl_manager']
        )

        assert runner is not None
        assert isinstance(runner, mcts_py.SimulationRunner)

    def test_constructor_requires_all_components(self):
        """Test that SimulationRunner constructor validates required arguments."""
        # Test missing arguments
        with pytest.raises(TypeError):
            mcts_py.SimulationRunner()  # No arguments

    def test_constructor_type_validation(self, mcts_components):
        """Test that constructor validates component types."""
        # Try to pass wrong types
        with pytest.raises(TypeError):
            mcts_py.SimulationRunner(
                "not_a_tree",  # Wrong type
                mcts_components['selector'],
                mcts_components['backup'],
                mcts_components['vl_manager']
            )

    def test_runner_class_has_docstring(self):
        """Test that SimulationRunner has documentation."""
        assert mcts_py.SimulationRunner.__doc__ is not None, \
            "SimulationRunner should have docstring"

        # Check docstring mentions it's a C++ implementation
        assert "C++" in mcts_py.SimulationRunner.__doc__ or \
               "simulation" in mcts_py.SimulationRunner.__doc__.lower(), \
            "Docstring should describe the class purpose"

    def test_multiple_instances(self, mcts_components):
        """Test that multiple SimulationRunner instances can coexist."""
        runner1 = mcts_py.SimulationRunner(
            mcts_components['tree'],
            mcts_components['selector'],
            mcts_components['backup'],
            mcts_components['vl_manager']
        )

        runner2 = mcts_py.SimulationRunner(
            mcts_components['tree'],
            mcts_components['selector'],
            mcts_components['backup'],
            mcts_components['vl_manager']
        )

        assert runner1 is not runner2, "Each instantiation should create a new object"
        assert isinstance(runner1, mcts_py.SimulationRunner)
        assert isinstance(runner2, mcts_py.SimulationRunner)

    def test_runner_with_different_components(self):
        """Test that different runners can use different component configurations."""
        # Create first set of components
        tree1 = mcts_py.create_test_tree(1000)
        selector1 = mcts_py.create_puct_selector()
        backup1 = mcts_py.create_backup_manager(tree1)
        vl1 = mcts_py.create_test_virtual_loss_manager(tree1)

        runner1 = mcts_py.SimulationRunner(tree1, selector1, backup1, vl1)

        # Create second set of components with different config
        tree2 = mcts_py.create_test_tree(2000)  # Different size
        config2 = mcts_py.PUCTConfig()
        config2.cpuct = 2.0  # Different exploration constant
        selector2 = mcts_py.create_puct_selector(config2)
        backup2 = mcts_py.create_backup_manager(tree2)
        vl2 = mcts_py.create_test_virtual_loss_manager(tree2)

        runner2 = mcts_py.SimulationRunner(tree2, selector2, backup2, vl2)

        assert runner1 is not runner2
        # Both should be valid instances
        assert isinstance(runner1, mcts_py.SimulationRunner)
        assert isinstance(runner2, mcts_py.SimulationRunner)


class TestSimulationRunnerIntegration:
    """Integration tests for SimulationRunner with MCTS components."""

    def test_runner_shares_tree_with_components(self):
        """Test that SimulationRunner correctly references shared MCTS tree."""
        tree = mcts_py.create_test_tree(1000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        vl_manager = mcts_py.create_test_virtual_loss_manager(tree)

        # Get initial node count
        initial_nodes = tree.get_node_count()

        # Create runner with shared tree
        runner = mcts_py.SimulationRunner(tree, selector, backup, vl_manager)

        # Tree should still have same node count (runner just references it)
        assert tree.get_node_count() == initial_nodes

        # Verify runner was created successfully
        assert runner is not None

    def test_runner_with_custom_puct_config(self):
        """Test SimulationRunner with customized PUCT configuration."""
        tree = mcts_py.create_test_tree(1000)

        # Create custom PUCT config
        puct_config = mcts_py.PUCTConfig()
        puct_config.cpuct = 1.5
        puct_config.fpu_value = -0.5
        puct_config.use_fpu = True

        selector = mcts_py.create_puct_selector(puct_config)
        backup = mcts_py.create_backup_manager(tree)
        vl_manager = mcts_py.create_test_virtual_loss_manager(tree)

        runner = mcts_py.SimulationRunner(tree, selector, backup, vl_manager)

        assert runner is not None
        # Verify selector still has custom config
        retrieved_config = selector.get_config()
        assert retrieved_config.cpuct == 1.5

    def test_runner_with_custom_virtual_loss_config(self):
        """Test SimulationRunner with customized virtual loss configuration."""
        tree = mcts_py.create_test_tree(1000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)

        # Create custom virtual loss config
        vl_config = mcts_py.VirtualLossConfig()
        vl_config.magnitude = 2.0  # Higher virtual loss

        vl_manager = mcts_py.create_test_virtual_loss_manager(tree, vl_config)

        runner = mcts_py.SimulationRunner(tree, selector, backup, vl_manager)

        assert runner is not None

    def test_runner_lifecycle_with_components(self):
        """Test that SimulationRunner handles component lifecycle correctly."""
        tree = mcts_py.create_test_tree(1000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        vl_manager = mcts_py.create_test_virtual_loss_manager(tree)

        # Create runner
        runner = mcts_py.SimulationRunner(tree, selector, backup, vl_manager)
        assert runner is not None

        # Components should still be accessible
        assert tree.get_node_count() >= 0
        assert selector.get_config() is not None

        # Delete runner (Python GC will clean up)
        del runner

        # Components should still be valid (not owned by runner)
        assert tree.get_node_count() >= 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
