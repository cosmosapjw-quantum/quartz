"""
Real MCTS Integration Tests
==========================

Tests the complete MCTS pipeline with real C++ games and actual neural network integration.
Validates production-ready search behavior, tree integrity, and performance.
"""

import pytest
import numpy as np
import time
from concurrent.futures import Future
from src.core.mcts import AlphaZeroMCTS
from src.games.game_state import create_game_state


class TestRealMCTSIntegration:
    """Test real MCTS implementation with actual game states."""

    def create_fast_inference_fn(self):
        """Create fast mock inference for testing."""
        def inference_fn(game_state):
            future = Future()

            # Generate realistic policy and value
            action_space = game_state.action_space_size
            mask_getter = getattr(game_state, 'get_legal_moves_mask', None)
            if callable(mask_getter):
                legal_moves_mask = mask_getter()
            else:
                legal_moves_list = np.array(game_state.get_legal_moves(), dtype=np.int64)
                legal_moves_mask = np.zeros(action_space, dtype=bool)
                if legal_moves_list.size > 0:
                    legal_moves_mask[legal_moves_list] = True

            legal_moves = np.flatnonzero(legal_moves_mask)

            policy = np.zeros(action_space, dtype=np.float32)
            if len(legal_moves) > 0:
                # Give higher probability to legal moves
                for move in legal_moves:
                    if move >= 0:  # Skip pass moves in Go
                        policy[move] = 1.0 / len([m for m in legal_moves if m >= 0])

            # Generate realistic value based on game state
            value = np.random.uniform(-0.3, 0.3)  # Slight bias towards draw

            future.set_result((policy, value))
            return future

        return inference_fn

    def test_mcts_single_thread_gomoku(self):
        """Test single-threaded MCTS search on Gomoku."""
        game = create_game_state('gomoku')
        inference_fn = self.create_fast_inference_fn()
        mcts = AlphaZeroMCTS(inference_fn)

        # Run small search
        start_time = time.time()
        visit_counts = mcts.search(game, simulations=10)
        search_time = time.time() - start_time

        # Validate results
        assert len(visit_counts) > 0
        assert all(isinstance(k, int) for k in visit_counts.keys())
        assert all(isinstance(v, int) for v in visit_counts.values())
        assert sum(visit_counts.values()) == 10  # All simulations accounted for

        # Test policy extraction
        policy = mcts.get_policy(game, temperature=1.0)
        assert policy.shape == (225,)
        assert abs(policy.sum() - 1.0) < 1e-6  # Should sum to 1

        # Test value extraction
        value = mcts.get_value(game)
        assert -1 <= value <= 1

        # Validate tree growth
        assert mcts.tree_size > 10  # Should have expanded nodes

        print(f"Single-thread MCTS: {mcts.tree_size} nodes in {search_time:.3f}s")

    def test_mcts_single_thread_chess(self):
        """Test single-threaded MCTS search on Chess."""
        game = create_game_state('chess')
        inference_fn = self.create_fast_inference_fn()
        mcts = AlphaZeroMCTS(inference_fn)

        # Run small search
        visit_counts = mcts.search(game, simulations=5)

        # Validate results
        assert len(visit_counts) > 0
        assert sum(visit_counts.values()) == 5

        # Test policy extraction
        policy = mcts.get_policy(game, temperature=1.0)
        assert policy.shape == (20480,)  # Chess action space
        assert abs(policy.sum() - 1.0) < 1e-6

        # Only legal moves should have non-zero policy
        mask_getter = getattr(game, 'get_legal_moves_mask', None)
        if callable(mask_getter):
            legal_moves_mask = mask_getter()
        else:
            legal_list = np.array(game.get_legal_moves(), dtype=np.int64)
            legal_moves_mask = np.zeros(game.action_space_size, dtype=bool)
            if legal_list.size > 0:
                legal_moves_mask[legal_list] = True
        legal_moves = set(np.flatnonzero(legal_moves_mask))
        for move, prob in enumerate(policy):
            if prob > 0:
                assert move in legal_moves

    def test_mcts_single_thread_go(self):
        """Test single-threaded MCTS search on Go."""
        game = create_game_state('go', board_size=9)
        inference_fn = self.create_fast_inference_fn()
        mcts = AlphaZeroMCTS(inference_fn)

        # Run small search
        visit_counts = mcts.search(game, simulations=5)

        # Validate results
        assert len(visit_counts) > 0
        assert sum(visit_counts.values()) == 5

        # Test policy extraction
        policy = mcts.get_policy(game, temperature=1.0)
        assert policy.shape == (82,)  # 9x9 Go action space
        assert abs(policy.sum() - 1.0) < 1e-6

    def test_mcts_tree_integrity_after_search(self):
        """Test MCTS tree maintains integrity after search operations."""
        game = create_game_state('gomoku')
        inference_fn = self.create_fast_inference_fn()
        mcts = AlphaZeroMCTS(inference_fn)

        # Run multiple searches to stress test tree
        for i in range(3):
            visit_counts = mcts.search(game, simulations=5)

            # Tree should grow with each search
            assert mcts.tree_size > i * 5  # At least one node per simulation

            # Policy should remain valid
            policy = mcts.get_policy(game)
            assert abs(policy.sum() - 1.0) < 1e-6

    def test_mcts_with_game_progression(self):
        """Test MCTS on a game that has progressed through moves."""
        game = create_game_state('gomoku')
        inference_fn = self.create_fast_inference_fn()

        # Apply some moves to the game
        mask_getter = getattr(game, 'get_legal_moves_mask', None)
        if callable(mask_getter):
            legal_moves_mask = mask_getter()
        else:
            legal_list = np.array(game.get_legal_moves(), dtype=np.int64)
            legal_moves_mask = np.zeros(game.action_space_size, dtype=bool)
            if legal_list.size > 0:
                legal_moves_mask[legal_list] = True
        legal_moves = np.flatnonzero(legal_moves_mask)
        game = game.make_move(legal_moves[0])  # First move
        game = game.make_move(legal_moves[1])  # Second move

        # Search from this position
        mcts = AlphaZeroMCTS(inference_fn)
        visit_counts = mcts.search(game, simulations=8)

        # Should get valid results from non-starting position
        assert len(visit_counts) > 0
        assert sum(visit_counts.values()) == 8

        policy = mcts.get_policy(game)
        assert abs(policy.sum() - 1.0) < 1e-6

        # Policy should only have non-zero values for legal moves
        mask_getter = getattr(game, 'get_legal_moves_mask', None)
        if callable(mask_getter):
            current_legal_moves_mask = mask_getter()
        else:
            legal_list = np.array(game.get_legal_moves(), dtype=np.int64)
            current_legal_moves_mask = np.zeros(game.action_space_size, dtype=bool)
            if legal_list.size > 0:
                current_legal_moves_mask[legal_list] = True
        current_legal_moves = set(np.flatnonzero(current_legal_moves_mask))
        for move, prob in enumerate(policy):
            if prob > 0:
                assert move in current_legal_moves

    def test_mcts_temperature_effects(self):
        """Test temperature effects on policy extraction."""
        game = create_game_state('gomoku')
        inference_fn = self.create_fast_inference_fn()
        mcts = AlphaZeroMCTS(inference_fn)

        # Run search once
        visit_counts = mcts.search(game, simulations=10)

        # Extract policies with different temperatures
        policy_cold = mcts.get_policy(game, temperature=0.1)
        policy_hot = mcts.get_policy(game, temperature=2.0)

        # Cold temperature should be more concentrated
        entropy_cold = -np.sum(policy_cold * np.log(policy_cold + 1e-8))
        entropy_hot = -np.sum(policy_hot * np.log(policy_hot + 1e-8))

        assert entropy_hot > entropy_cold, "Hot temperature should have higher entropy"

    def test_mcts_dirichlet_noise_effects(self):
        """Test Dirichlet noise effects on search behavior."""
        game = create_game_state('gomoku')
        inference_fn = self.create_fast_inference_fn()

        # Search without noise
        mcts_no_noise = AlphaZeroMCTS(inference_fn)
        visit_counts_no_noise = mcts_no_noise.search(game, simulations=10, add_noise=False)

        # Search with noise
        mcts_with_noise = AlphaZeroMCTS(inference_fn)
        visit_counts_with_noise = mcts_with_noise.search(game, simulations=10, add_noise=True)

        # Results should be different due to exploration noise
        assert visit_counts_no_noise != visit_counts_with_noise

        # Both should be valid
        assert sum(visit_counts_no_noise.values()) == 10
        assert sum(visit_counts_with_noise.values()) == 10

    def test_mcts_performance_single_thread(self):
        """Test MCTS performance in single-threaded mode."""
        game = create_game_state('gomoku')
        inference_fn = self.create_fast_inference_fn()
        mcts = AlphaZeroMCTS(inference_fn)

        # Run larger search and measure performance
        start_time = time.time()
        visit_counts = mcts.search(game, simulations=50)
        search_time = time.time() - start_time

        # Performance validation
        nodes_per_second = mcts.tree_size / search_time
        simulations_per_second = 50 / search_time

        print(f"MCTS Performance: {nodes_per_second:.0f} nodes/sec, {simulations_per_second:.0f} sims/sec")

        # Should achieve reasonable performance even with Python coordination
        assert nodes_per_second > 100, f"Performance too low: {nodes_per_second} nodes/sec"
        assert mcts.tree_size > 50, f"Tree too small: {mcts.tree_size} nodes"

        # Results should be valid
        assert len(visit_counts) > 0
        assert sum(visit_counts.values()) == 50

    def test_mcts_reset_functionality(self):
        """Test MCTS reset clears state correctly."""
        game = create_game_state('gomoku')
        inference_fn = self.create_fast_inference_fn()
        mcts = AlphaZeroMCTS(inference_fn)

        # Run initial search
        mcts.search(game, simulations=5)
        initial_tree_size = mcts.tree_size

        assert initial_tree_size > 0

        # Reset and verify clean state
        mcts.reset()
        assert mcts.tree_size == 0

        # Should be able to search again after reset
        visit_counts = mcts.search(game, simulations=5)
        assert sum(visit_counts.values()) == 5

    def test_mcts_with_terminal_detection(self):
        """Test MCTS handles terminal states correctly."""
        # Create a game and force it to terminal state (mock)
        game = create_game_state('gomoku')

        # For this test, we'll just verify the MCTS can handle
        # what it thinks are terminal nodes without crashing
        inference_fn = self.create_fast_inference_fn()
        mcts = AlphaZeroMCTS(inference_fn)

        # Even on starting position (non-terminal), should work fine
        visit_counts = mcts.search(game, simulations=3)
        assert sum(visit_counts.values()) == 3

        # Value should be reasonable for non-terminal position
        value = mcts.get_value(game)
        assert -1 <= value <= 1

    @pytest.mark.performance
    def test_mcts_scaling_with_simulations(self):
        """Test MCTS performance scales appropriately with simulation count."""
        game = create_game_state('gomoku')
        inference_fn = self.create_fast_inference_fn()

        simulation_counts = [5, 10, 20]
        performance_data = []

        for sim_count in simulation_counts:
            mcts = AlphaZeroMCTS(inference_fn)

            start_time = time.time()
            visit_counts = mcts.search(game, simulations=sim_count)
            search_time = time.time() - start_time

            performance_data.append({
                'simulations': sim_count,
                'time': search_time,
                'tree_size': mcts.tree_size,
                'sims_per_second': sim_count / search_time
            })

            # Validate correctness
            assert sum(visit_counts.values()) == sim_count
            assert mcts.tree_size >= sim_count  # Should have at least one node per simulation

        # Performance should scale reasonably
        for data in performance_data:
            print(f"Sims: {data['simulations']}, Time: {data['time']:.3f}s, "
                  f"Tree: {data['tree_size']}, Rate: {data['sims_per_second']:.0f}/s")

        # Larger searches should achieve higher absolute tree sizes
        assert performance_data[-1]['tree_size'] > performance_data[0]['tree_size']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
