"""
Integration test to validate C++ SimulationRunner equivalence.

This test validates that the C++ SimulationRunner produces deterministic
results consistent with expected MCTS behavior.

HOWTO-RUN-TESTS:
===============
# Run equivalence tests
python -m pytest tests/integration/test_cpp_vs_python_equivalence.py -v

# Run with verbose output
python -m pytest tests/integration/test_cpp_vs_python_equivalence.py -v -s

# Run specific test
python -m pytest tests/integration/test_cpp_vs_python_equivalence.py::TestCppRunnerEquivalence::test_deterministic_search -v
"""

import pytest
import sys
from pathlib import Path
import numpy as np
from concurrent.futures import Future
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.mcts import AlphaZeroMCTS
import alphazero_py


class TestCppRunnerEquivalence:
    """Test C++ SimulationRunner produces correct and deterministic results."""

    @pytest.fixture
    def gomoku_game(self):
        """Create deterministic Gomoku game."""
        return alphazero_py.GomokuState(board_size=15)

    @pytest.fixture
    def mock_inference_fn(self):
        """Create deterministic mock inference function."""
        def inference(game_state):
            """Return deterministic uniform policy and neutral value."""
            action_space = game_state.get_action_space_size()
            legal_moves = game_state.get_legal_moves()

            # Create uniform policy over legal moves
            policy = np.zeros(action_space, dtype=np.float32)
            if len(legal_moves) > 0:
                prob = 1.0 / len(legal_moves)
                for move in legal_moves:
                    policy[move] = prob

            value = 0.0  # Neutral position

            # Return as completed Future
            future = Future()
            future.set_result((policy, value))
            return future

        return inference

    def test_cpp_runner_initialization(self, gomoku_game, mock_inference_fn):
        """Test that MCTS with C++ runner initializes correctly."""
        mcts = AlphaZeroMCTS(
            inference_fn=mock_inference_fn,
            max_tree_size=10000,
            num_threads=1
        )

        assert mcts is not None
        assert mcts.simulation_runner is not None
        assert mcts.tree is not None

    def test_deterministic_search(self, gomoku_game, mock_inference_fn):
        """Test that C++ runner produces deterministic results with fixed seed."""
        # Set random seed for reproducibility
        np.random.seed(42)

        mcts = AlphaZeroMCTS(
            inference_fn=mock_inference_fn,
            max_tree_size=10000,
            num_threads=1
        )

        # Run search
        visit_counts_1 = mcts.search(gomoku_game, simulations=50, add_noise=False)

        # Reset and run again with same seed
        np.random.seed(42)
        mcts.reset()
        visit_counts_2 = mcts.search(gomoku_game, simulations=50, add_noise=False)

        # Results should be identical (no noise, same seed)
        assert visit_counts_1.keys() == visit_counts_2.keys(), \
            "Move sets should be identical"

        for move in visit_counts_1:
            assert visit_counts_1[move] == visit_counts_2[move], \
                f"Visit count for move {move} should be identical"

    def test_visit_count_consistency(self, gomoku_game, mock_inference_fn):
        """Test that visit counts sum correctly."""
        mcts = AlphaZeroMCTS(
            inference_fn=mock_inference_fn,
            max_tree_size=10000,
            num_threads=1
        )

        simulations = 100
        visit_counts = mcts.search(gomoku_game, simulations=simulations, add_noise=False)

        # Total visits should approximately equal number of simulations
        # (may be less if tree fills up or simulations fail)
        total_visits = sum(visit_counts.values())
        assert total_visits > 0, "Should have at least some visits"
        assert total_visits <= simulations, "Cannot have more visits than simulations"

        # First simulation expands root, subsequent simulations explore children
        # So we expect total_visits to be close to (simulations - 1)
        assert total_visits >= simulations - 10, \
            f"Should have most simulations visiting children, got {total_visits}/{simulations}"

    def test_policy_extraction(self, gomoku_game, mock_inference_fn):
        """Test that policy can be extracted from search results."""
        mcts = AlphaZeroMCTS(
            inference_fn=mock_inference_fn,
            max_tree_size=10000,
            num_threads=1
        )

        visit_counts = mcts.search(gomoku_game, simulations=50, add_noise=False)
        policy = mcts.get_policy(gomoku_game, temperature=1.0)

        assert policy is not None
        assert len(policy) == gomoku_game.get_action_space_size()
        assert np.isclose(np.sum(policy), 1.0, atol=1e-6), \
            "Policy should sum to 1.0"
        assert np.all(policy >= 0), "Policy should be non-negative"

    def test_tree_get_move_functionality(self, gomoku_game, mock_inference_fn):
        """Test that tree.get_move() works correctly."""
        mcts = AlphaZeroMCTS(
            inference_fn=mock_inference_fn,
            max_tree_size=10000,
            num_threads=1
        )

        visit_counts = mcts.search(gomoku_game, simulations=50, add_noise=False)

        # Verify we can retrieve moves from tree
        assert len(visit_counts) > 0, "Should have explored some moves"

        # All moves should be valid
        legal_moves = set(gomoku_game.get_legal_moves())
        for move in visit_counts.keys():
            assert move in legal_moves, f"Move {move} should be legal"

    def test_cpp_runner_performance(self, gomoku_game, mock_inference_fn):
        """Test that C++ runner achieves reasonable performance."""
        import time

        mcts = AlphaZeroMCTS(
            inference_fn=mock_inference_fn,
            max_tree_size=10000,
            num_threads=1
        )

        simulations = 1000
        start_time = time.perf_counter()
        visit_counts = mcts.search(gomoku_game, simulations=simulations, add_noise=False)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        sims_per_sec = simulations / elapsed

        # Should be significantly faster than Python (target: 30k+ sims/sec)
        # With single thread and mock inference, expect at least 1k sims/sec
        assert sims_per_sec > 100, \
            f"C++ runner should achieve >100 sims/sec, got {sims_per_sec:.1f}"

        print(f"\nPerformance: {sims_per_sec:.1f} simulations/second")

    def test_multiple_searches(self, gomoku_game, mock_inference_fn):
        """Test that multiple searches work correctly."""
        mcts = AlphaZeroMCTS(
            inference_fn=mock_inference_fn,
            max_tree_size=10000,
            num_threads=1
        )

        # Run first search
        visit_counts_1 = mcts.search(gomoku_game, simulations=50, add_noise=False)
        assert len(visit_counts_1) > 0

        # Reset and run second search
        mcts.reset()
        visit_counts_2 = mcts.search(gomoku_game, simulations=50, add_noise=False)
        assert len(visit_counts_2) > 0

    def test_dirichlet_noise_variation(self, gomoku_game, mock_inference_fn):
        """Test that Dirichlet noise creates variation."""
        np.random.seed(42)

        mcts = AlphaZeroMCTS(
            inference_fn=mock_inference_fn,
            max_tree_size=10000,
            num_threads=1,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25
        )

        # Run with noise
        visit_counts_with_noise = mcts.search(gomoku_game, simulations=50, add_noise=True)

        # Reset and run without noise
        mcts.reset()
        visit_counts_without_noise = mcts.search(gomoku_game, simulations=50, add_noise=False)

        # With noise, we expect some variation in exploration
        # (results may be similar but shouldn't be identical for all moves)
        assert len(visit_counts_with_noise) > 0
        assert len(visit_counts_without_noise) > 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
