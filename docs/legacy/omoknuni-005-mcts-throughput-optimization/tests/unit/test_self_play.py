"""
Unit Tests for Self-Play Game Generator
======================================

Tests the self-play game generation functionality including:
- Temperature scheduling and move selection
- Dirichlet noise application
- Training example creation
- Game outcome determination
- Parallel game generation
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.self_play import (
    SelfPlayGameGenerator, SelfPlayConfig, create_self_play_generator,
    save_games_to_disk, load_games_from_disk
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "specs" / "001-goal-create-spec"))
from contracts.training_api import GameResult, TrainingExample
from src.core.search_coordinator import SearchResult
from src.games.game_state import create_game_state


class TestSelfPlayConfig:
    """Test self-play configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SelfPlayConfig()

        assert config.game_type == "gomoku"
        assert config.mcts_simulations == 800
        assert len(config.temperature_schedule) == 2
        assert config.temperature_schedule[0] == (30, 1.0)
        assert config.temperature_schedule[1] == (1000, 0.1)
        assert config.dirichlet_alpha == 0.3
        assert config.add_dirichlet_noise is True
        assert config.num_threads == 8

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SelfPlayConfig(
            game_type="chess",
            mcts_simulations=1000,
            temperature_schedule=[(20, 1.5), (50, 0.5)],
            dirichlet_alpha=0.2,
            num_threads=12
        )

        assert config.game_type == "chess"
        assert config.mcts_simulations == 1000
        assert config.temperature_schedule == [(20, 1.5), (50, 0.5)]
        assert config.dirichlet_alpha == 0.2
        assert config.num_threads == 12


class TestSelfPlayGameGenerator:
    """Test self-play game generator implementation."""

    @pytest.fixture
    def mock_model_path(self):
        """Mock model path for testing."""
        return "/tmp/mock_model.pth"

    @pytest.fixture
    def generator(self, mock_model_path):
        """Create test generator with mocked dependencies."""
        generator = SelfPlayGameGenerator(
            game_type="gomoku",
            model_path=mock_model_path,
            mcts_simulations=100,  # Reduced for testing
            temperature_schedule=[(10, 1.0), (100, 0.1)],
            num_threads=2  # Reduced for testing
        )

        # Mock the components to avoid launching background threads during unit tests
        generator.inference_worker = Mock()
        generator.search_coordinator = Mock()

        return generator

    def test_initialization(self, mock_model_path):
        """Test generator initialization."""
        generator = SelfPlayGameGenerator(
            game_type="chess",
            model_path=mock_model_path,
            mcts_simulations=1200,
            temperature_schedule=[(25, 1.2), (75, 0.05)],
            add_dirichlet_noise=False,
            num_threads=16
        )

        assert generator.config.game_type == "chess"
        assert generator.config.mcts_simulations == 1200
        assert generator.config.temperature_schedule == [(25, 1.2), (75, 0.05)]
        assert generator.config.add_dirichlet_noise is False
        assert generator.config.num_threads == 16
        assert generator.model_path == mock_model_path

    def test_game_specific_params(self, mock_model_path):
        """Test game-specific parameter setting."""
        # Test Gomoku
        gomoku_gen = SelfPlayGameGenerator("gomoku", mock_model_path)
        assert gomoku_gen.config.dirichlet_alpha == 0.3
        assert gomoku_gen.config.max_game_length == 225

        # Test Chess
        chess_gen = SelfPlayGameGenerator("chess", mock_model_path)
        assert chess_gen.config.dirichlet_alpha == 0.2
        assert chess_gen.config.max_game_length == 512

        # Test Go
        go_gen = SelfPlayGameGenerator("go", mock_model_path)
        assert go_gen.config.dirichlet_alpha == 0.03
        assert go_gen.config.max_game_length == 722

    def test_temperature_scheduling(self, generator):
        """Test temperature calculation based on move count."""
        # Early moves should use high temperature
        assert generator._get_temperature(5) == 1.0
        assert generator._get_temperature(9) == 1.0

        # Later moves should use low temperature
        assert generator._get_temperature(15) == 0.1
        assert generator._get_temperature(50) == 0.1
        assert generator._get_temperature(200) == 0.1

    def test_move_selection_deterministic(self, generator):
        """Test deterministic move selection (temperature = 0)."""
        policy = np.array([0.1, 0.3, 0.6])
        best_move = int(np.argmax(policy))

        # With temperature 0, should always select best move
        for _ in range(10):
            move = generator._select_move_with_temperature(policy, 0.0, best_move)
            assert move == best_move  # Index of maximum value

    def test_move_selection_stochastic(self, generator):
        """Test stochastic move selection with temperature."""
        policy = np.array([0.1, 0.3, 0.6])
        best_move = int(np.argmax(policy))

        # With temperature > 0, should sample probabilistically
        moves = []
        for _ in range(100):
            move = generator._select_move_with_temperature(policy, 1.0, best_move)
            moves.append(move)

        # Should sample all three moves
        unique_moves = set(moves)
        assert len(unique_moves) >= 2  # At least 2 different moves

        # Move 2 should be most frequent (highest probability)
        move_counts = np.bincount(moves)
        assert np.argmax(move_counts) == best_move

    def test_training_example_creation(self, generator):
        """Test creation of training examples."""
        game_state = create_game_state('gomoku')
        policy = np.random.dirichlet(np.ones(game_state.action_space_size))

        example = generator._create_training_example(
            game_state=game_state,
            policy=policy,
            move_number=5,
            game_id="test_game"
        )

        assert isinstance(example, TrainingExample)
        assert example.state.shape == (36, 15, 15)  # Expected feature shape
        assert np.array_equal(example.policy, policy)
        assert example.value == 0.0  # Initial value before outcome
        assert example.game_type == "gomoku"
        assert example.move_number == 5
        assert example.game_id == "test_game"

    def test_outcome_value_assignment(self, generator):
        """Test value assignment based on game outcome."""
        # Create test examples
        examples = []
        for i in range(4):
            example = TrainingExample(
                state=np.zeros((36, 15, 15)),
                policy=np.ones(225) / 225,
                value=0.0,
                game_type="gomoku",
                move_number=i,
                game_id="test"
            )
            examples.append(example)

        # Test player 0 wins
        outcome = {'winner': 0, 'result': 'win_player1'}
        generator._update_examples_with_outcome(examples, outcome, 4)

        # Even moves (0, 2) are player 0 moves -> should be +1
        # Odd moves (1, 3) are player 1 moves -> should be -1
        assert examples[0].value == 1.0   # Player 0 move, player 0 wins
        assert examples[1].value == -1.0  # Player 1 move, player 0 wins
        assert examples[2].value == 1.0   # Player 0 move, player 0 wins
        assert examples[3].value == -1.0  # Player 1 move, player 0 wins

    def test_outcome_value_assignment_draw(self, generator):
        """Test value assignment for drawn games."""
        examples = [
            TrainingExample(
                state=np.zeros((36, 15, 15)),
                policy=np.ones(225) / 225,
                value=0.0,
                game_type="gomoku",
                move_number=0,
                game_id="test"
            )
        ]

        # Test draw
        outcome = {'winner': None, 'result': 'draw'}
        generator._update_examples_with_outcome(examples, outcome, 1)

        assert examples[0].value == 0.0  # Draw value

    def test_generate_game_uses_mcts_policy(self, generator):
        """Test game generation path using provided MCTS policy."""
        action_space = create_game_state('gomoku').action_space_size

        futures = []
        chosen_moves = [12, 48, 96]

        for idx, move in enumerate(chosen_moves):
            policy = np.zeros(action_space, dtype=np.float32)
            policy[move] = 1.0
            search_result = SearchResult(
                request_id=f"req_{idx}",
                best_move=move,
                policy=policy,
                value=0.5,
                processing_time_ms=5.0
            )
            future = Future()
            future.set_result(search_result)
            futures.append(future)

        generator.search_coordinator.submit_search.side_effect = futures

        with patch.object(generator, '_get_temperature', return_value=0.0), \
             patch.object(generator, '_is_game_terminal', side_effect=[False, False, False, True]):
            result = generator.generate_game("test_game")

        assert isinstance(result, GameResult)
        assert result.move_count == len(chosen_moves)
        assert len(result.examples) == len(chosen_moves)
        assert result.metadata['game_id'] == "test_game"
        assert result.metadata['game_type'] == "gomoku"
        assert result.metadata['move_history'][:len(chosen_moves)] == chosen_moves

    def test_generate_games_parallel(self, generator):
        """Test parallel game generation."""
        # Mock the generate_game method
        def mock_generate_game(game_id):
            return GameResult(
                winner=0,
                move_count=10,
                game_length_seconds=1.0,
                examples=[],
                final_board="mock board",
                metadata={'game_id': game_id}
            )

        generator.generate_game = Mock(side_effect=mock_generate_game)

        # Generate games
        games = list(generator.generate_games(num_games=3, parallel_games=2))

        assert len(games) == 3
        assert generator.generate_game.call_count == 3

        # Check all games were generated
        game_ids = {game.metadata['game_id'] for game in games}
        assert len(game_ids) == 3  # All unique IDs

    def test_update_model(self, generator):
        """Test model update functionality."""
        new_model_path = "/tmp/new_model.pth"

        generator.update_model(new_model_path)

        assert generator.model_path == new_model_path
        # Should call update on inference worker if initialized
        if generator.inference_worker:
            generator.inference_worker.update_model.assert_called_with(new_model_path)

    def test_statistics_tracking(self, generator):
        """Test statistics collection."""
        # Initial statistics
        stats = generator.get_statistics()
        assert stats['games_generated'] == 0
        assert stats['total_positions'] == 0

        # Simulate some generated games
        generator.games_generated = 5
        generator.total_positions = 100
        generator.generation_times = [10.0, 12.0, 8.0, 15.0, 11.0]

        stats = generator.get_statistics()
        assert stats['games_generated'] == 5
        assert stats['total_positions'] == 100
        assert stats['average_positions_per_game'] == 20.0
        assert stats['average_generation_time_seconds'] == 11.2
        assert stats['games_per_hour'] == pytest.approx(3600 / 11.2, rel=1e-3)

    def test_board_string_representation(self, generator):
        """Ensure board string is produced from real game state."""
        game_state = create_game_state('gomoku')
        board_str = generator._get_board_string(game_state)

        assert isinstance(board_str, str)
        assert len(board_str) > 0


class TestFactoryFunctions:
    """Test factory functions and utilities."""

    def test_create_self_play_generator(self):
        """Test factory function for generator creation."""
        config = {
            'game_type': 'chess',
            'model_path': '/tmp/test.pth',
            'mcts_simulations': 1000,
            'temperature_schedule': [(50, 0.8)],
            'add_dirichlet_noise': False,
            'num_threads': 4
        }

        generator = create_self_play_generator(config)

        assert generator.config.game_type == 'chess'
        assert generator.model_path == '/tmp/test.pth'
        assert generator.config.mcts_simulations == 1000
        assert generator.config.temperature_schedule == [(50, 0.8)]
        assert generator.config.add_dirichlet_noise is False
        assert generator.config.num_threads == 4

    def test_create_self_play_generator_defaults(self):
        """Test factory function with default values."""
        config = {}

        generator = create_self_play_generator(config)

        assert generator.config.game_type == 'gomoku'
        assert generator.model_path == 'models/latest.pth'
        assert generator.config.mcts_simulations == 800
        assert generator.config.add_dirichlet_noise is True
        assert generator.config.num_threads == 8


class TestGameSerialization:
    """Test game saving and loading functionality."""

    def test_save_and_load_games(self):
        """Test saving games to disk and loading them back."""
        # Create test games
        games = []
        for i in range(2):
            examples = [
                TrainingExample(
                    state=np.random.rand(36, 15, 15).astype(np.float32),
                    policy=np.random.dirichlet([1.0] * 225),
                    value=float(i % 2),  # Alternating values
                    game_type="gomoku",
                    move_number=j,
                    game_id=f"game_{i}"
                )
                for j in range(3)
            ]

            game = GameResult(
                winner=i % 2,
                move_count=3,
                game_length_seconds=10.0 + i,
                examples=examples,
                final_board=f"final_board_{i}",
                metadata={'game_id': f"game_{i}", 'test_data': True}
            )
            games.append(game)

        # Save to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            save_games_to_disk(games, output_path)

            # Check files were created
            game_files = list(output_path.glob("game_*.json"))
            assert len(game_files) == 2

            # Load games back
            loaded_games = load_games_from_disk(output_path)

            assert len(loaded_games) == 2

            # Verify data integrity
            for original, loaded in zip(games, loaded_games):
                assert loaded.winner == original.winner
                assert loaded.move_count == original.move_count
                assert loaded.game_length_seconds == original.game_length_seconds
                assert loaded.final_board == original.final_board
                assert loaded.metadata == original.metadata

                assert len(loaded.examples) == len(original.examples)

                for orig_ex, loaded_ex in zip(original.examples, loaded.examples):
                    assert np.allclose(loaded_ex.state, orig_ex.state)
                    assert np.allclose(loaded_ex.policy, orig_ex.policy)
                    assert loaded_ex.value == orig_ex.value
                    assert loaded_ex.game_type == orig_ex.game_type
                    assert loaded_ex.move_number == orig_ex.move_number
                    assert loaded_ex.game_id == orig_ex.game_id

    def test_save_games_creates_directory(self):
        """Test that save_games_to_disk creates output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "new_subdir" / "games"

            # Directory doesn't exist yet
            assert not output_path.exists()

            # Save empty game list
            save_games_to_disk([], output_path)

            # Directory should be created
            assert output_path.exists()
            assert output_path.is_dir()

    def test_load_games_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)

            loaded_games = load_games_from_disk(output_path)

            assert loaded_games == []


class TestIntegrationWithComponents:
    """Integration tests with other system components."""

    @pytest.mark.integration
    def test_temperature_schedule_integration(self):
        """Test integration of temperature scheduling with real game flow."""
        generator = SelfPlayGameGenerator(
            game_type="gomoku",
            model_path="/tmp/test.pth",
            temperature_schedule=[(5, 1.0), (10, 0.5), (1000, 0.1)]
        )

        # Test temperature progression
        temperatures = [generator._get_temperature(i) for i in range(15)]

        assert all(t == 1.0 for t in temperatures[:5])    # Moves 0-4
        assert all(t == 0.5 for t in temperatures[5:10])  # Moves 5-9
        assert all(t == 0.1 for t in temperatures[10:])   # Moves 10+

    @pytest.mark.integration
    def test_dirichlet_noise_configuration(self):
        """Test Dirichlet noise alpha values for different games."""
        configs = [
            ("gomoku", 0.3),
            ("chess", 0.2),
            ("go", 0.03)
        ]

        for game_type, expected_alpha in configs:
            generator = SelfPlayGameGenerator(
                game_type=game_type,
                model_path="/tmp/test.pth"
            )
            assert generator.config.dirichlet_alpha == expected_alpha


if __name__ == "__main__":
    pytest.main([__file__])
