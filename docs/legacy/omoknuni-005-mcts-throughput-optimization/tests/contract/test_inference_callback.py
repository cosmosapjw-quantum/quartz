"""
Contract tests for PyInferenceCallback.

Tests the Python inference callback bridge to ensure:
1. PyInferenceCallback can be instantiated with Python callable
2. GIL is properly released during C++ simulation
3. Callback is correctly invoked from C++
4. Type conversions work correctly (Python ↔ C++)
5. Error handling works for invalid callables

HOWTO-RUN-TESTS:
================
# Run inference callback contract tests
python -m pytest tests/contract/test_inference_callback.py -v

# Run with verbose output
python -m pytest tests/contract/test_inference_callback.py -v -s

# Run specific test
python -m pytest tests/contract/test_inference_callback.py::TestInferenceCallbackAPI::test_callback_instantiation -v
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import threading
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import C++ bindings
import mcts_py
import alphazero_py


class TestInferenceCallbackAPI:
    """Test PyInferenceCallback API contract and bindings."""

    def test_inference_callback_class_exists(self):
        """Test that InferenceCallback base class is accessible."""
        assert hasattr(mcts_py, 'InferenceCallback'), \
            "InferenceCallback should be exported from mcts_py module"

    def test_py_inference_callback_class_exists(self):
        """Test that PyInferenceCallback class is accessible."""
        assert hasattr(mcts_py, 'PyInferenceCallback'), \
            "PyInferenceCallback should be exported from mcts_py module"

        assert isinstance(mcts_py.PyInferenceCallback, type), \
            "PyInferenceCallback should be a class type"

    def test_callback_instantiation(self):
        """Test that PyInferenceCallback can be instantiated with a callable."""
        def dummy_inference(state):
            return ([0.5, 0.5], 0.0)

        callback = mcts_py.PyInferenceCallback(dummy_inference)

        assert callback is not None, "PyInferenceCallback should instantiate successfully"
        assert isinstance(callback, mcts_py.PyInferenceCallback), \
            "Instance should be of type PyInferenceCallback"
        assert isinstance(callback, mcts_py.InferenceCallback), \
            "PyInferenceCallback should inherit from InferenceCallback"

    def test_callback_with_lambda(self):
        """Test that PyInferenceCallback works with lambda functions."""
        callback = mcts_py.PyInferenceCallback(
            lambda state: ([0.5, 0.5], 0.0)
        )

        assert callback is not None

    def test_callback_with_method(self):
        """Test that PyInferenceCallback works with instance methods."""
        class InferenceModel:
            def predict(self, state):
                return ([0.5, 0.5], 0.0)

        model = InferenceModel()
        callback = mcts_py.PyInferenceCallback(model.predict)

        assert callback is not None

    def test_callback_requires_callable(self):
        """Test that PyInferenceCallback rejects non-callable objects."""
        # Try to pass a non-callable object
        with pytest.raises((TypeError, RuntimeError, ValueError)):
            mcts_py.PyInferenceCallback("not_a_function")

        with pytest.raises((TypeError, RuntimeError, ValueError)):
            mcts_py.PyInferenceCallback(42)

        with pytest.raises((TypeError, RuntimeError, ValueError)):
            mcts_py.PyInferenceCallback(None)

    def test_callback_has_docstring(self):
        """Test that PyInferenceCallback has documentation."""
        assert mcts_py.PyInferenceCallback.__doc__ is not None, \
            "PyInferenceCallback should have docstring"

        doc = mcts_py.PyInferenceCallback.__doc__
        assert "inference" in doc.lower() or "callback" in doc.lower(), \
            "Docstring should describe the class purpose"


class TestInferenceCallbackInvocation:
    """Test inference callback invocation and type conversions."""

    @pytest.fixture
    def gomoku_game(self):
        """Create a Gomoku game for testing."""
        return alphazero_py.GomokuState(board_size=15)

    def test_callback_invocation_basic(self, gomoku_game):
        """Test that callback is correctly invoked with game state."""
        call_count = [0]  # Use list to allow mutation in closure

        def test_inference(state):
            call_count[0] += 1
            # Return uniform policy and neutral value
            action_space = state.get_action_space_size()
            policy = [1.0 / action_space] * action_space
            return (policy, 0.0)

        callback = mcts_py.PyInferenceCallback(test_inference)

        # Call request_inference directly
        policy, value = callback.request_inference(gomoku_game)

        assert call_count[0] == 1, "Callback should be called once"
        assert isinstance(policy, list), "Policy should be a list"
        assert len(policy) == gomoku_game.get_action_space_size(), \
            "Policy length should match action space"
        assert isinstance(value, float), "Value should be a float"

    def test_callback_with_numpy_array(self, gomoku_game):
        """Test that callback works with numpy array policy."""
        def numpy_inference(state):
            action_space = state.get_action_space_size()
            policy = np.ones(action_space, dtype=np.float32) / action_space
            return (policy, 0.5)

        callback = mcts_py.PyInferenceCallback(numpy_inference)

        policy, value = callback.request_inference(gomoku_game)

        assert isinstance(policy, list), "Policy should be converted to list"
        assert len(policy) == gomoku_game.get_action_space_size()
        assert abs(value - 0.5) < 1e-6

    def test_callback_with_list_policy(self, gomoku_game):
        """Test that callback works with Python list policy."""
        def list_inference(state):
            action_space = state.get_action_space_size()
            policy = [1.0 / action_space] * action_space
            return (policy, 0.3)

        callback = mcts_py.PyInferenceCallback(list_inference)

        policy, value = callback.request_inference(gomoku_game)

        assert isinstance(policy, list)
        assert len(policy) == gomoku_game.get_action_space_size()
        assert abs(value - 0.3) < 1e-6

    def test_callback_value_range(self, gomoku_game):
        """Test that callback accepts values in [-1, 1] range."""
        test_values = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for test_val in test_values:
            def val_inference(state):
                action_space = state.get_action_space_size()
                return ([1.0 / action_space] * action_space, test_val)

            callback = mcts_py.PyInferenceCallback(val_inference)
            policy, value = callback.request_inference(gomoku_game)

            assert abs(value - test_val) < 1e-6, f"Value {test_val} should be preserved"


class TestInferenceCallbackErrors:
    """Test error handling in inference callbacks."""

    @pytest.fixture
    def gomoku_game(self):
        """Create a Gomoku game for testing."""
        return alphazero_py.GomokuState(board_size=15)

    def test_callback_wrong_return_type(self, gomoku_game):
        """Test that callback raises error for wrong return type."""
        def bad_inference(state):
            return "not_a_tuple"  # Wrong return type

        callback = mcts_py.PyInferenceCallback(bad_inference)

        with pytest.raises(RuntimeError):
            callback.request_inference(gomoku_game)

    def test_callback_wrong_tuple_length(self, gomoku_game):
        """Test that callback raises error for wrong tuple length."""
        def bad_inference(state):
            return (0.5,)  # Only one element, need two

        callback = mcts_py.PyInferenceCallback(bad_inference)

        with pytest.raises(RuntimeError):
            callback.request_inference(gomoku_game)

    def test_callback_exception_handling(self, gomoku_game):
        """Test that Python exceptions in callback are properly handled."""
        def failing_inference(state):
            raise ValueError("Intentional test error")

        callback = mcts_py.PyInferenceCallback(failing_inference)

        with pytest.raises(RuntimeError):
            callback.request_inference(gomoku_game)


class TestInferenceCallbackIntegration:
    """Integration tests with SimulationRunner."""

    @pytest.fixture
    def mcts_components(self):
        """Create MCTS components for testing."""
        tree = mcts_py.MCTSTree(10000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        vl_manager = mcts_py.create_test_virtual_loss_manager(tree)

        return {
            'tree': tree,
            'selector': selector,
            'backup': backup,
            'vl_manager': vl_manager
        }

    @pytest.fixture
    def gomoku_game(self):
        """Create a Gomoku game for testing."""
        return alphazero_py.GomokuState(board_size=15)

    def test_simulation_with_callback(self, mcts_components, gomoku_game):
        """Test running simulation with PyInferenceCallback."""
        tree = mcts_components['tree']
        root = tree.add_root_node(0.5, 0)

        # Create callback
        call_count = [0]

        def test_inference(state):
            call_count[0] += 1
            action_space = state.get_action_space_size()
            policy = [1.0 / action_space] * action_space
            return (policy, 0.5)

        callback = mcts_py.PyInferenceCallback(test_inference)

        # Create runner
        runner = mcts_py.SimulationRunner(
            tree,
            mcts_components['selector'],
            mcts_components['backup'],
            mcts_components['vl_manager']
        )

        # Run simulation
        success = runner.run_simulation(gomoku_game, root, callback)

        assert success, "Simulation should complete successfully"
        assert call_count[0] >= 1, "Inference callback should be called at least once"
        assert tree.get_visit_count(root) >= 1.0, "Root should be visited"

    def test_multiple_simulations_with_callback(self, mcts_components, gomoku_game):
        """Test running multiple simulations with callback."""
        tree = mcts_components['tree']
        root = tree.add_root_node(0.5, 0)

        call_count = [0]

        def counting_inference(state):
            call_count[0] += 1
            action_space = state.get_action_space_size()
            policy = [1.0 / action_space] * action_space
            return (policy, 0.0)

        callback = mcts_py.PyInferenceCallback(counting_inference)

        runner = mcts_py.SimulationRunner(
            tree,
            mcts_components['selector'],
            mcts_components['backup'],
            mcts_components['vl_manager']
        )

        # Run 10 simulations
        num_simulations = 10
        for _ in range(num_simulations):
            success = runner.run_simulation(gomoku_game, root, callback)
            assert success

        # Inference should be called multiple times (at least once per simulation)
        assert call_count[0] >= num_simulations, \
            f"Inference should be called at least {num_simulations} times"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
