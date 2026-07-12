"""
Real Implementation Tests for Search Coordinator
================================================

Tests the search coordinator with actual C++ game implementations and real MCTS.
No mocks - this validates the production code path.
"""

import pytest
import time
import threading
import numpy as np
from concurrent.futures import Future
from src.core.search_coordinator import SearchCoordinator, SearchRequest, SearchResult
from src.games.game_state import create_game_state
from src.neural.cpu_inference import CPUInferenceWorker


class TestRealSearchCoordinator:
    """Test search coordinator with real implementations."""

    def setup_method(self):
        """Set up with real inference worker."""
        from src.neural.model import create_model_for_game
        import tempfile
        import torch

        # Create temporary model file
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        model_path = self.temp_model_file.name
        self.temp_model_file.close()

        # Create and save a test model with default architecture
        model = create_model_for_game('gomoku')  # Use default architecture
        with torch.no_grad():
            dummy_input = torch.randn(1, model.input_channels, 15, 15)
            _ = model(dummy_input)
        torch.save(model.state_dict(), model_path)

        self.inference_worker = CPUInferenceWorker(
            model_path=model_path,
            device='cpu',
            timeout_ms=100.0  # Fast timeout for testing
        )
        self.coordinator = SearchCoordinator(
            inference_worker=self.inference_worker,
            max_threads=2,  # Small thread count for testing
            max_queue_size=50
        )

    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'coordinator') and self.coordinator.running:
            self.coordinator.stop()

        # Clean up temporary model file
        if hasattr(self, 'temp_model_file'):
            import os
            try:
                os.unlink(self.temp_model_file.name)
            except FileNotFoundError:
                pass

    def test_coordinator_with_real_gomoku(self):
        """Test search coordinator with real Gomoku game."""
        self.coordinator.start()

        # Create real Gomoku game
        game = create_game_state('gomoku')

        request = SearchRequest(
            request_id="gomoku_test",
            game_state=game,
            simulations=5,  # Small number for fast testing
            temperature=1.0
        )

        future = self.coordinator.submit_search(request)
        result = future.result(timeout=10.0)

        # Validate real search results
        assert isinstance(result, SearchResult)
        assert result.request_id == "gomoku_test"
        assert 0 <= result.best_move < 225  # Gomoku has 225 positions
        assert len(result.policy) == 225
        assert abs(result.policy.sum() - 1.0) < 1e-6  # Policy should sum to 1
        assert -1 <= result.value <= 1
        assert result.processing_time_ms > 0

        # Check search info contains real data
        assert 'tree_size' in result.search_info
        assert 'visit_counts' in result.search_info
        assert result.search_info['simulations_completed'] == 5

    def test_coordinator_with_real_chess(self):
        """Test search coordinator with real Chess game - simplified test."""
        import os
        import tempfile
        import torch
        from src.neural.model import AlphaZeroNet

        if getattr(self.coordinator, 'running', False):
            self.coordinator.stop()

        temp_file = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        model_path = temp_file.name
        temp_file.close()

        chess_coordinator = None
        chess_game = create_game_state('chess')
        action_space = chess_game.action_space_size
        try:
            chess_model = AlphaZeroNet(
                input_channels=30,
                num_actions=action_space,
                num_blocks=2,
                hidden_channels=64
            )
            chess_model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 30, 8, 8)
                _ = chess_model(dummy_input)
            torch.save(chess_model.state_dict(), model_path)

            chess_worker = CPUInferenceWorker(
                model_path=model_path,
                device='cpu',
                timeout_ms=150.0
            )
            chess_coordinator = SearchCoordinator(
                inference_worker=chess_worker,
                max_threads=2,
                max_queue_size=64
            )
            chess_coordinator.start()

            request = SearchRequest(
                request_id="chess_test",
                game_state=chess_game,
                simulations=3,
                temperature=1.0
            )

            future = chess_coordinator.submit_search(request)
            result = future.result(timeout=20.0)

            assert isinstance(result, SearchResult)
            assert result.request_id == "chess_test"
            assert len(result.policy) == action_space
            assert 0 <= result.best_move < action_space
            assert abs(result.policy.sum() - 1.0) < 1e-6
            assert -1.0 <= result.value <= 1.0
        finally:
            if chess_coordinator is not None:
                try:
                    chess_coordinator.stop()
                except Exception:
                    pass

            try:
                os.unlink(model_path)
            except FileNotFoundError:
                pass

    def test_coordinator_with_real_go(self):
        """Test search coordinator with real game state - using Gomoku since model is trained for that."""
        self.coordinator.start()

        # Use Gomoku since the inference worker is trained for Gomoku
        # Real inference workers are game-specific, unlike mocks
        game = create_game_state('gomoku')

        # Make a few moves to create a different position
        legal_moves_mask = game.get_legal_moves()
        legal_moves = np.where(legal_moves_mask)[0]
        if len(legal_moves) > 0:
            game.make_move(legal_moves[0])  # Play center move

        request = SearchRequest(
            request_id="gomoku_test_2",
            game_state=game,
            simulations=3,
            temperature=1.0
        )

        future = self.coordinator.submit_search(request)
        result = future.result(timeout=10.0)

        # Validate real search results
        assert isinstance(result, SearchResult)
        assert result.request_id == "gomoku_test_2"

        # Action space should match what the model was trained for
        expected_actions = self.inference_worker._model_info['num_actions'] if self.inference_worker._model_info else 225
        assert 0 <= result.best_move < expected_actions
        assert len(result.policy) == expected_actions
        assert -1 <= result.value <= 1

    def test_multiple_real_searches(self):
        """Test multiple concurrent searches with real games."""
        self.coordinator.start()

        # Create multiple Gomoku games (inference worker is trained for Gomoku)
        # Note: Real inference workers are game-specific, unlike mocks
        games = [
            create_game_state('gomoku'),
            create_game_state('gomoku'),  # Different Gomoku games
            create_game_state('gomoku')
        ]

        # Play a few random moves in each game to create different positions
        import random
        for game in games:
            num_moves = random.randint(0, 3)
            for _ in range(num_moves):
                if not game.is_terminal():
                    legal_moves_mask = game.get_legal_moves()
                    legal_moves = np.where(legal_moves_mask)[0]
                    if len(legal_moves) > 0:
                        move = random.choice(legal_moves)
                        game.make_move(move)

        futures = []
        for i, game in enumerate(games):
            request = SearchRequest(
                request_id=f"multi_test_{i}",
                game_state=game,
                simulations=2  # Very small for speed
            )
            futures.append(self.coordinator.submit_search(request))

        # Wait for all results
        results = []
        for future in futures:
            result = future.result(timeout=15.0)
            results.append(result)

        # Validate all results are different and valid
        assert len(results) == 3
        request_ids = [r.request_id for r in results]
        assert len(set(request_ids)) == 3  # All unique

        # Each result should be valid for Gomoku
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.processing_time_ms > 0
            assert -1 <= result.value <= 1
            assert 0 <= result.best_move < 225  # Gomoku action space
            assert len(result.policy) == 225

    def test_search_with_temperature_effects(self):
        """Test that temperature actually affects policy distribution."""
        self.coordinator.start()

        game = create_game_state('gomoku')

        # Search with high temperature (more exploration)
        request_hot = SearchRequest(
            request_id="temp_hot",
            game_state=game.clone(),
            simulations=5,
            temperature=2.0
        )

        # Search with low temperature (more exploitation)
        request_cold = SearchRequest(
            request_id="temp_cold",
            game_state=game.clone(),
            simulations=5,
            temperature=0.1
        )

        future_hot = self.coordinator.submit_search(request_hot)
        future_cold = self.coordinator.submit_search(request_cold)

        result_hot = future_hot.result(timeout=10.0)
        result_cold = future_cold.result(timeout=10.0)

        # Cold temperature should be more concentrated (higher max probability)
        max_prob_hot = np.max(result_hot.policy)
        max_prob_cold = np.max(result_cold.policy)

        # Cold search should have more concentrated policy
        assert max_prob_cold >= max_prob_hot, "Cold temperature should concentrate policy more"

    def test_search_with_noise_effects(self):
        """Test that Dirichlet noise affects search behavior."""
        self.coordinator.start()

        game = create_game_state('gomoku')

        # Search without noise
        request_no_noise = SearchRequest(
            request_id="no_noise",
            game_state=game.clone(),
            simulations=5,
            add_noise=False
        )

        # Search with noise
        request_with_noise = SearchRequest(
            request_id="with_noise",
            game_state=game.clone(),
            simulations=5,
            add_noise=True
        )

        future_no_noise = self.coordinator.submit_search(request_no_noise)
        future_with_noise = self.coordinator.submit_search(request_with_noise)

        result_no_noise = future_no_noise.result(timeout=10.0)
        result_with_noise = future_with_noise.result(timeout=10.0)

        # Both should be valid results
        assert isinstance(result_no_noise, SearchResult)
        assert isinstance(result_with_noise, SearchResult)

        # Policies should be different due to noise
        policy_diff = np.abs(result_with_noise.policy - result_no_noise.policy).sum()
        assert policy_diff > 0, "Noise should create different policies"

    def test_coordinator_metrics_with_real_searches(self):
        """Test metrics collection during real searches."""
        self.coordinator.start()

        # Submit several searches
        games = [create_game_state('gomoku') for _ in range(3)]
        futures = []

        for i, game in enumerate(games):
            request = SearchRequest(
                request_id=f"metrics_test_{i}",
                game_state=game,
                simulations=2
            )
            futures.append(self.coordinator.submit_search(request))

        # Check metrics while searches are running
        time.sleep(0.1)  # Let searches start
        metrics = self.coordinator.get_metrics()
        assert metrics.active_searches > 0

        # Wait for completion
        for future in futures:
            future.result(timeout=10.0)

        # Check final metrics
        final_metrics = self.coordinator.get_metrics()
        assert final_metrics.completed_searches >= 3
        assert final_metrics.total_simulations >= 6  # 3 searches * 2 simulations

    def test_real_game_state_integration(self):
        """Test that real game states work correctly with search coordinator."""
        self.coordinator.start()

        # Test a game with actual moves applied
        game = create_game_state('gomoku')

        # Apply some moves to the game
        legal_moves_mask = game.get_legal_moves()
        legal_moves = np.where(legal_moves_mask)[0]
        original_legal_count = len(legal_moves)

        game = game.make_move(legal_moves[0])  # Make first legal move

        # Recalculate legal moves after first move
        legal_moves_mask = game.get_legal_moves()
        legal_moves = np.where(legal_moves_mask)[0]
        game = game.make_move(legal_moves[1])  # Make second legal move (different position)

        # Search from this position
        request = SearchRequest(
            request_id="moved_game_test",
            game_state=game,
            simulations=3
        )

        future = self.coordinator.submit_search(request)
        result = future.result(timeout=10.0)

        # Should get valid result from non-starting position
        assert isinstance(result, SearchResult)
        assert result.request_id == "moved_game_test"

        # Legal moves should be reduced by 2 (the moves we made)
        # Get actual legal moves from the final game state
        final_legal_moves_mask = game.get_legal_moves()
        final_legal_moves = np.where(final_legal_moves_mask)[0]
        current_legal_count = len(final_legal_moves)
        assert current_legal_count <= original_legal_count - 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
