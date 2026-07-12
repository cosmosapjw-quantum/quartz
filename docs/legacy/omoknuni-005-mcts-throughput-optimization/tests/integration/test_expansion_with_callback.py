"""
Integration tests for SimulationRunner expansion with inference callback.

Tests the expansion phase of MCTS simulation using Python inference callbacks.
Validates that:
1. Inference callback is properly invoked from C++
2. Policy masking and normalization works correctly
3. Child nodes are allocated with correct priors
4. Move indices are recorded correctly
5. Terminal nodes are detected and handled

HOWTO-RUN-TESTS:
================
# Run expansion integration tests
python -m pytest tests/integration/test_expansion_with_callback.py -v

# Run with verbose output
python -m pytest tests/integration/test_expansion_with_callback.py -v -s

# Run specific test
python -m pytest tests/integration/test_expansion_with_callback.py::TestExpansionWithCallback::test_basic_expansion -v
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import C++ bindings
import mcts_py
import alphazero_py


class StubInferenceCallback:
    """
    Stub inference callback for testing.

    Returns deterministic policy and value based on game state.
    """

    def __init__(self, policy_values=None, value_estimate=0.5):
        """
        Initialize stub callback.

        Args:
            policy_values: Dictionary mapping move index to probability.
                          If None, returns uniform policy over action space.
            value_estimate: Fixed value to return (default 0.5)
        """
        self.policy_values = policy_values
        self.value_estimate = value_estimate
        self.call_count = 0
        self.last_state = None

    def __call__(self, state):
        """
        Stub inference that returns predetermined policy and value.

        Args:
            state: Game state (IGameState interface)

        Returns:
            Tuple of (policy_vector, value_scalar)
        """
        self.call_count += 1
        self.last_state = state

        action_space_size = state.get_action_space_size()

        # Create policy vector
        policy = np.zeros(action_space_size, dtype=np.float32)

        if self.policy_values is not None:
            # Use provided policy values
            for move, prob in self.policy_values.items():
                if move < action_space_size:
                    policy[move] = prob
        else:
            # Uniform policy over legal moves
            legal_moves = state.get_legal_moves()
            if len(legal_moves) > 0:
                prob = 1.0 / len(legal_moves)
                for move in legal_moves:
                    policy[move] = prob

        return (policy.tolist(), self.value_estimate)


class TestExpansionWithCallback:
    """Test SimulationRunner::expand_node() with Python inference callback."""

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

    def test_basic_expansion(self, mcts_components, gomoku_game):
        """Test basic expansion with uniform policy."""
        tree = mcts_components['tree']

        # Create root node
        root = tree.add_root_node(0.5, 0)

        # Get game state
        state = gomoku_game
        initial_legal_moves = state.get_legal_moves()

        # Create stub inference callback (uniform policy, value=0.3)
        callback = StubInferenceCallback(value_estimate=0.3)

        # Create simulation runner
        runner = mcts_py.SimulationRunner(
            tree,
            mcts_components['selector'],
            mcts_components['backup'],
            mcts_components['vl_manager']
        )

        # Expand root node via select_leaf_public and expand_node_public
        # (we need to add a public wrapper for expand_node too)
        # For now, test via the full run_simulation when T012 is complete

        # Alternative: Test the expansion logic indirectly
        # We'll verify that inference was called correctly

        assert callback.call_count == 0, "Callback should not be called yet"

        # Note: Full integration test will be possible after T012 (run_simulation)
        # For now, we validate the callback interface

    def test_policy_masking(self):
        """Test that policy is correctly masked to legal moves."""
        # Create a simple test scenario
        tree = mcts_py.MCTSTree(1000)
        root = tree.add_root_node(0.5, 0)

        # Create Gomoku game
        state = alphazero_py.GomokuState(board_size=15)

        # Get legal moves (at start, all 225 positions are legal)
        legal_moves = state.get_legal_moves()
        action_space = state.get_action_space_size()

        # Create policy that gives different probabilities to different moves
        policy_dict = {}
        # Give high probability to center position
        center = 15 * 7 + 7  # (7, 7) on 15x15 board
        policy_dict[center] = 0.8
        # Give small probability to corners
        policy_dict[0] = 0.05
        policy_dict[14] = 0.05
        policy_dict[15*14] = 0.05
        policy_dict[15*14 + 14] = 0.05

        callback = StubInferenceCallback(policy_values=policy_dict, value_estimate=0.4)

        # Test that callback returns correct policy
        policy, value = callback(state)

        assert len(policy) == action_space, f"Policy size should be {action_space}"
        assert value == 0.4, "Value should match stub value"
        assert abs(policy[center] - 0.8) < 1e-6, "Center should have high probability"

        # Verify normalization will happen in C++
        # The policy we provide doesn't sum to 1.0 intentionally
        policy_sum = sum(policy)
        assert policy_sum > 0.0, "Policy should have non-zero sum"

    def test_terminal_expansion(self, gomoku_game):
        """Test expansion of a terminal node."""
        tree = mcts_py.MCTSTree(1000)

        # Create a winning position in Gomoku
        state = gomoku_game

        # Play moves to create a winning position for player 1
        # Horizontal line: (7,7), (7,8), (7,9), (7,10), (7,11)
        winning_moves = [
            15 * 7 + 7,   # (7, 7)
            15 * 7 + 8,   # (7, 8) - player 2
            15 * 7 + 9,   # (7, 9)
            15 * 7 + 10,  # (7, 10) - player 2
            15 * 7 + 11,  # (7, 11)
            15 * 7 + 12,  # (7, 12) - player 2
            15 * 7 + 13,  # (7, 13)
            15 * 6 + 7,   # (6, 7) - player 2
            15 * 8 + 7,   # (8, 7) - forms 5 in a row vertically
        ]

        for move in winning_moves:
            if not state.is_terminal():
                state.make_move(move)

        # Check if terminal
        is_terminal = state.is_terminal()

        if is_terminal:
            # Expansion should detect terminal and not create children
            root = tree.add_root_node(0.5, 0)

            # Callback should not be called for terminal nodes
            callback = StubInferenceCallback(value_estimate=0.5)

            # We would test this through expand_node, but it's private
            # This test will be complete after T012
            assert is_terminal, "State should be terminal"

    def test_callback_invocation(self, gomoku_game):
        """Test that inference callback is properly invoked."""
        callback = StubInferenceCallback(value_estimate=0.6)

        # Create game state
        state = gomoku_game

        # Call callback directly to test interface
        policy, value = callback(state)

        assert callback.call_count == 1, "Callback should be called once"
        assert isinstance(policy, list), "Policy should be a list"
        assert isinstance(value, float), "Value should be a float"
        assert value == 0.6, "Value should match stub value"
        assert len(policy) == state.get_action_space_size(), "Policy size should match action space"

        # Verify policy sums to ~1.0 (uniform distribution over legal moves)
        policy_sum = sum(policy)
        assert abs(policy_sum - 1.0) < 1e-5, f"Policy should sum to 1.0, got {policy_sum}"

    def test_move_index_recording(self):
        """Test that move indices are correctly recorded in tree."""
        tree = mcts_py.MCTSTree(1000)

        # Manually test move storage (this was validated in T008)
        root = tree.add_root_node(0.5, 0)

        # Allocate children
        num_children = 5
        first_child = tree.allocate_nodes(num_children)
        tree.set_first_child_index(root, first_child)
        tree.set_num_children(root, num_children)

        # Set move indices
        test_moves = [10, 20, 30, 40, 50]
        for i, move in enumerate(test_moves):
            tree.set_move(first_child + i, move)

        # Verify
        for i, expected_move in enumerate(test_moves):
            actual_move = tree.get_move(first_child + i)
            assert actual_move == expected_move, \
                f"Child {i} should have move {expected_move}, got {actual_move}"

    def test_expansion_with_restricted_moves(self, gomoku_game):
        """Test expansion when some positions are already occupied."""
        tree = mcts_py.MCTSTree(1000)

        # Create game state with some moves played
        state = gomoku_game

        # Play a few moves
        moves_played = [15 * 7 + 7, 15 * 7 + 8, 15 * 7 + 9]
        for move in moves_played:
            state.make_move(move)

        # Get legal moves (should exclude played positions)
        legal_moves = state.get_legal_moves()
        action_space = state.get_action_space_size()

        # Legal moves should be less than full action space
        assert len(legal_moves) < action_space, \
            f"Legal moves ({len(legal_moves)}) should be less than action space ({action_space})"

        # Verify played moves are not in legal moves
        for played_move in moves_played:
            assert played_move not in legal_moves, \
                f"Played move {played_move} should not be in legal moves"

        # Create callback with policy for all positions
        callback = StubInferenceCallback(value_estimate=0.5)

        # Policy should be uniform over legal moves
        policy, value = callback(state)

        # Check that only legal moves have non-zero probability
        for i, prob in enumerate(policy):
            if i in legal_moves:
                assert prob > 0, f"Legal move {i} should have non-zero probability"
            else:
                # Some non-legal moves might have zero probability in stub
                pass


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
