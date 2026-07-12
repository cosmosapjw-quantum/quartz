"""
Unit tests for training loop orchestration.

Tests the complete training cycle coordination including self-play generation,
experience buffer management, model training, and checkpoint management.
"""

import pytest
import tempfile
import shutil
import json
import time
import signal
import threading
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

import numpy as np

# Import the module under test
from src.training.training_loop import (
    TrainingLoop, TrainingConfig, TrainingMetrics,
    create_training_loop, run_training_session
)

# Import contracts for test data
import sys
sys.path.append('specs/001-goal-create-spec')
from contracts.training_api import GameResult, TrainingExample


class TestTrainingConfig:
    """Test training configuration."""

    def test_default_config_creation(self):
        """Test creating default training configuration."""
        config = TrainingConfig()

        assert config.game_type == "gomoku"
        assert config.self_play_games_per_iteration == 50
        assert config.training_steps_per_iteration == 1000
        assert config.batch_size == 512
        assert config.max_iterations == 1000

    def test_custom_config_creation(self):
        """Test creating custom training configuration."""
        config = TrainingConfig(
            game_type="chess",
            self_play_games_per_iteration=20,
            batch_size=256,
            learning_rate=0.0005
        )

        assert config.game_type == "chess"
        assert config.self_play_games_per_iteration == 20
        assert config.batch_size == 256
        assert config.learning_rate == 0.0005

    def test_config_serialization(self):
        """Test configuration can be serialized to/from dict."""
        config = TrainingConfig(game_type="go", max_iterations=500)
        config_dict = config.__dict__

        assert config_dict['game_type'] == "go"
        assert config_dict['max_iterations'] == 500

        # Test factory function with dict
        new_config = TrainingConfig(**config_dict)
        assert new_config.game_type == "go"
        assert new_config.max_iterations == 500


class TestTrainingMetrics:
    """Test training metrics tracking."""

    def test_default_metrics_creation(self):
        """Test creating default metrics."""
        metrics = TrainingMetrics()

        assert metrics.iteration == 0
        assert metrics.total_games_generated == 0
        assert metrics.total_training_steps == 0
        assert metrics.training_loss == 0.0
        assert metrics.evaluation_history == []

    def test_metrics_update(self):
        """Test updating metrics."""
        metrics = TrainingMetrics()

        metrics.iteration = 5
        metrics.total_games_generated = 250
        metrics.training_loss = 1.5
        metrics.evaluation_history = [0.45, 0.52, 0.58]

        assert metrics.iteration == 5
        assert metrics.total_games_generated == 250
        assert metrics.training_loss == 1.5
        assert len(metrics.evaluation_history) == 3


class TestTrainingLoop:
    """Test training loop orchestration."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock training configuration."""
        return TrainingConfig(
            game_type="gomoku",
            model_path=str(temp_dir / "test_model.pth"),
            self_play_games_per_iteration=5,
            training_steps_per_iteration=10,
            batch_size=32,
            max_iterations=3,
            checkpoint_frequency=2,
            evaluation_frequency=2,
            experience_buffer_path=str(temp_dir / "experience"),
            checkpoint_dir=str(temp_dir / "checkpoints"),
            log_dir=str(temp_dir / "logs"),
            evaluation_dir=str(temp_dir / "eval")
        )

    @pytest.fixture
    def mock_training_examples(self):
        """Create mock training examples."""
        examples = []
        for i in range(10):
            example = TrainingExample(
                state=np.random.rand(36, 15, 15).astype(np.float32),
                policy=np.random.rand(225).astype(np.float32),
                value=np.random.uniform(-1, 1),
                game_type="gomoku",
                move_number=i,
                game_id=f"test_game_{i // 5}"
            )
            examples.append(example)
        return examples

    @pytest.fixture
    def mock_game_results(self, mock_training_examples):
        """Create mock game results."""
        games = []
        for i in range(2):
            game = GameResult(
                winner=i % 2,
                move_count=20 + i * 5,
                game_length_seconds=30.0 + i * 10,
                examples=mock_training_examples[i*5:(i+1)*5],
                final_board=f"Mock board {i}",
                metadata={'game_id': f'test_game_{i}', 'test': True}
            )
            games.append(game)
        return games

    def test_training_loop_initialization(self, mock_config):
        """Test training loop initialization."""
        loop = TrainingLoop(mock_config)

        assert loop.config == mock_config
        assert isinstance(loop.metrics, TrainingMetrics)
        assert not loop.running
        assert not loop.shutdown_requested
        assert loop.current_iteration == 0

        # Check directories are created
        assert Path(mock_config.checkpoint_dir).exists()
        assert Path(mock_config.log_dir).exists()
        assert Path(mock_config.evaluation_dir).exists()

    @patch('src.training.training_loop.SelfPlayGameGenerator')
    @patch('src.training.training_loop.MemoryMappedExperienceBuffer')
    @patch('src.training.training_loop.AlphaZeroTrainer')
    def test_component_initialization(self, mock_trainer_class, mock_buffer_class,
                                    mock_generator_class, mock_config):
        """Test lazy initialization of training components."""
        # Setup mocks
        mock_generator = Mock()
        mock_buffer = Mock()
        mock_trainer = Mock()

        mock_generator_class.return_value = mock_generator
        mock_buffer_class.return_value = mock_buffer
        mock_trainer_class.return_value = mock_trainer

        loop = TrainingLoop(mock_config)

        # Components should not be initialized yet
        assert loop.self_play_generator is None
        assert loop.experience_buffer is None
        assert loop.trainer is None

        # Initialize components
        loop._initialize_components()

        # Check components are created
        assert loop.self_play_generator == mock_generator
        assert loop.experience_buffer == mock_buffer
        assert loop.trainer == mock_trainer

        # Verify constructor calls
        mock_generator_class.assert_called_once_with(
            game_type=mock_config.game_type,
            model_path=mock_config.model_path,
            mcts_simulations=mock_config.mcts_simulations,
            num_threads=8
        )

        mock_buffer_class.assert_called_once_with(
            buffer_path=Path(mock_config.experience_buffer_path),
            max_examples=mock_config.max_experience_examples,
            cache_size_mb=mock_config.cache_size_mb
        )

        mock_trainer_class.assert_called_once_with(
            model_path=mock_config.model_path,
            learning_rate=mock_config.learning_rate,
            weight_decay=mock_config.weight_decay,
            batch_size=mock_config.batch_size,
            use_mixed_precision=True
        )

    def test_generate_self_play_games(self, mock_config, mock_game_results):
        """Test self-play game generation."""
        loop = TrainingLoop(mock_config)

        # Mock self-play generator
        mock_generator = Mock()
        mock_generator.generate_games.return_value = iter(mock_game_results)
        loop.self_play_generator = mock_generator

        # Generate games
        games = loop._generate_self_play_games()

        # Verify results
        assert len(games) == len(mock_game_results)
        assert games == mock_game_results

        # Verify generator was called correctly
        mock_generator.generate_games.assert_called_once_with(
            num_games=mock_config.self_play_games_per_iteration,
            parallel_games=mock_config.parallel_self_play_games
        )

        # Check performance metrics are updated
        assert loop.metrics.games_per_hour > 0

    def test_generate_self_play_games_with_shutdown(self, mock_config, mock_game_results):
        """Test self-play generation with shutdown request."""
        loop = TrainingLoop(mock_config)
        loop.shutdown_requested = True

        # Mock self-play generator that yields games slowly
        mock_generator = Mock()

        def slow_generator():
            for game in mock_game_results:
                yield game
                time.sleep(0.01)  # Small delay to allow shutdown check

        mock_generator.generate_games.return_value = slow_generator()
        loop.self_play_generator = mock_generator

        # Generate games (should stop early due to shutdown)
        games = loop._generate_self_play_games()

        # Should get at least one game before shutdown
        assert len(games) >= 1

    def test_train_model(self, mock_config, mock_training_examples):
        """Test model training."""
        loop = TrainingLoop(mock_config)

        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.train_step.return_value = {
            'total_loss': 1.5,
            'policy_loss': 0.8,
            'value_loss': 0.7,
            'learning_rate': 0.001
        }
        loop.trainer = mock_trainer

        # Mock experience buffer
        mock_buffer = Mock()

        def mock_iterator():
            for i in range(mock_config.training_steps_per_iteration):
                yield mock_training_examples[:mock_config.batch_size]

        mock_buffer.create_training_iterator.return_value = mock_iterator()
        loop.experience_buffer = mock_buffer

        # Train model
        metrics = loop._train_model()

        # Verify training was called
        assert mock_trainer.train_step.call_count == mock_config.training_steps_per_iteration

        # Check metrics
        assert 'total_loss' in metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert metrics['total_loss'] == 1.5

        # Check loop metrics are updated
        assert loop.metrics.training_loss == 1.5
        assert loop.metrics.policy_loss == 0.8
        assert loop.metrics.value_loss == 0.7
        assert abs(loop.metrics.learning_rate - 0.001) < 1e-6
        assert loop.metrics.total_training_steps == mock_config.training_steps_per_iteration

    def test_train_model_with_empty_batches(self, mock_config):
        """Test model training handles empty batches gracefully."""
        loop = TrainingLoop(mock_config)

        # Mock trainer
        mock_trainer = Mock()
        loop.trainer = mock_trainer

        # Mock experience buffer that returns empty batches
        mock_buffer = Mock()
        mock_buffer.create_training_iterator.return_value = iter([[] for _ in range(5)])
        loop.experience_buffer = mock_buffer

        # Train model
        metrics = loop._train_model()

        # Should handle empty batches gracefully
        assert mock_trainer.train_step.call_count == 0
        assert metrics == {}

    def test_save_and_load_training_state(self, mock_config):
        """Test saving and loading training state."""
        loop = TrainingLoop(mock_config)

        # Set some state
        loop.current_iteration = 5
        loop.metrics.total_games_generated = 150
        loop.metrics.total_training_steps = 5000
        loop.metrics.evaluation_history = [0.4, 0.5, 0.6]
        loop.metrics.best_evaluation_win_rate = 0.6

        # Save state
        loop._save_training_state()

        # Verify state file exists
        state_file = Path(mock_config.log_dir) / "training_state.json"
        assert state_file.exists()

        # Create new loop and load state
        new_loop = TrainingLoop(mock_config)
        new_loop._load_training_state()

        # Verify state is restored
        assert new_loop.current_iteration == 5
        assert new_loop.metrics.total_games_generated == 150
        assert new_loop.metrics.total_training_steps == 5000
        assert new_loop.metrics.evaluation_history == [0.4, 0.5, 0.6]
        assert new_loop.metrics.best_evaluation_win_rate == 0.6

    def test_checkpoint_management(self, mock_config):
        """Test checkpoint saving and cleanup."""
        loop = TrainingLoop(mock_config)

        # Mock trainer
        mock_trainer = Mock()
        loop.trainer = mock_trainer

        # Save checkpoint
        loop._save_checkpoint(5)

        # Verify trainer save_checkpoint was called
        expected_path = str(Path(mock_config.checkpoint_dir) / "model_iter_0005.pth")
        mock_trainer.save_checkpoint.assert_called_with(expected_path)

        # Test final checkpoint
        loop._save_checkpoint(10, is_final=True)
        final_path = str(Path(mock_config.checkpoint_dir) / "final_model.pth")
        mock_trainer.save_checkpoint.assert_called_with(final_path)

    def test_early_stopping_detection(self, mock_config):
        """Test early stopping criteria."""
        loop = TrainingLoop(mock_config)
        loop.config.early_stopping_patience = 3

        # Initially no stopping (not enough evaluations)
        loop.metrics.evaluation_history = [0.5, 0.6]
        loop.metrics.best_evaluation_win_rate = 0.6
        assert not loop._should_stop_early()

        # No improvement - should trigger early stopping
        loop.metrics.evaluation_history = [0.6, 0.5, 0.4, 0.3]  # Declining performance
        loop.metrics.best_evaluation_win_rate = 0.6
        assert loop._should_stop_early()

        # Continued improvement - should not stop
        loop.metrics.evaluation_history = [0.5, 0.6, 0.65, 0.7]
        loop.metrics.best_evaluation_win_rate = 0.7
        assert not loop._should_stop_early()

    def test_time_limit_detection(self, mock_config):
        """Test training time limit detection."""
        loop = TrainingLoop(mock_config)
        loop.config.target_training_time_hours = 1.0  # 1 hour limit

        # Just started - should not reach limit
        loop.training_start_time = time.time()
        assert not loop._has_reached_time_limit()

        # Simulate time passing (1.5 hours)
        loop.training_start_time = time.time() - 5400  # 1.5 hours ago
        assert loop._has_reached_time_limit()

        # Check metrics are updated
        assert loop.metrics.total_training_time_hours >= 1.5

    def test_training_statistics(self, mock_config):
        """Test training statistics collection."""
        loop = TrainingLoop(mock_config)

        # Set up some metrics
        loop.metrics.iteration = 10
        loop.metrics.total_games_generated = 500
        loop.metrics.total_training_steps = 10000
        loop.metrics.training_loss = 1.2
        loop.metrics.best_evaluation_win_rate = 0.65

        # Mock components
        mock_buffer = Mock()
        mock_buffer.get_stats.return_value = {'total_examples': 5000}
        loop.experience_buffer = mock_buffer

        mock_trainer = Mock()
        mock_trainer.get_training_stats.return_value = {'step_count': 10000}
        loop.trainer = mock_trainer

        mock_generator = Mock()
        mock_generator.get_statistics.return_value = {'games_generated': 500}
        loop.self_play_generator = mock_generator

        # Get statistics
        stats = loop.get_training_statistics()

        # Verify comprehensive stats
        assert stats['iteration'] == 10
        assert stats['total_games_generated'] == 500
        assert stats['total_training_steps'] == 10000
        assert stats['best_evaluation_win_rate'] == 0.65
        assert 'experience_buffer' in stats
        assert 'trainer' in stats
        assert 'self_play' in stats

    @patch('src.training.training_loop.SelfPlayGameGenerator')
    @patch('src.training.training_loop.MemoryMappedExperienceBuffer')
    @patch('src.training.training_loop.AlphaZeroTrainer')
    def test_full_training_iteration(self, mock_trainer_class, mock_buffer_class,
                                   mock_generator_class, mock_config, mock_game_results,
                                   mock_training_examples):
        """Test complete training iteration."""
        # Setup mocks
        mock_generator = Mock()
        mock_generator.generate_games.return_value = iter(mock_game_results)
        mock_generator.get_statistics.return_value = {'games_generated': len(mock_game_results)}
        mock_generator_class.return_value = mock_generator

        mock_buffer = Mock()
        mock_buffer.create_training_iterator.return_value = iter([
            mock_training_examples[:mock_config.batch_size]
            for _ in range(mock_config.training_steps_per_iteration)
        ])
        mock_buffer_class.return_value = mock_buffer

        mock_trainer = Mock()
        mock_trainer.train_step.return_value = {
            'total_loss': 1.0,
            'policy_loss': 0.5,
            'value_loss': 0.5,
            'learning_rate': 0.001
        }
        mock_trainer_class.return_value = mock_trainer

        loop = TrainingLoop(mock_config)
        loop._initialize_components()

        # Run single training iteration
        metrics = loop._run_training_iteration()

        # Verify all steps were executed
        mock_generator.generate_games.assert_called_once()
        mock_buffer.add_games.assert_called_once_with(mock_game_results)
        assert mock_trainer.train_step.call_count == mock_config.training_steps_per_iteration
        mock_generator.update_model.assert_called_once()

        # Check metrics
        assert metrics['games_generated'] == len(mock_game_results)
        assert 'training_metrics' in metrics
        assert 'iteration_time_seconds' in metrics

    def test_cleanup(self, mock_config):
        """Test resource cleanup."""
        loop = TrainingLoop(mock_config)

        # Mock components
        mock_generator = Mock()
        loop.self_play_generator = mock_generator

        # Cleanup
        loop._cleanup()

        # Verify cleanup was called
        mock_generator.shutdown.assert_called_once()


class TestFactoryFunctions:
    """Test factory functions and utilities."""

    def test_create_training_loop(self):
        """Test training loop factory function."""
        config_dict = {
            'game_type': 'chess',
            'max_iterations': 100,
            'batch_size': 256
        }

        loop = create_training_loop(config_dict)

        assert isinstance(loop, TrainingLoop)
        assert loop.config.game_type == 'chess'
        assert loop.config.max_iterations == 100
        assert loop.config.batch_size == 256

    def test_run_training_session(self, tmp_path):
        """Test running training session from config file."""
        # Create config file
        config_dict = {
            'game_type': 'gomoku',
            'max_iterations': 2,
            'self_play_games_per_iteration': 1,
            'training_steps_per_iteration': 1,
            'checkpoint_frequency': 1,
            'evaluation_frequency': 1,
            'model_path': str(tmp_path / 'test_model.pth'),
            'experience_buffer_path': str(tmp_path / 'experience'),
            'checkpoint_dir': str(tmp_path / 'checkpoints'),
            'log_dir': str(tmp_path / 'logs'),
            'evaluation_dir': str(tmp_path / 'eval')
        }

        config_file = tmp_path / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config_dict, f)

        # Mock the training loop run method to avoid actual training
        with patch('src.training.training_loop.TrainingLoop') as mock_loop_class:
            mock_loop = Mock()
            mock_metrics = TrainingMetrics()
            mock_metrics.iteration = 2
            mock_loop.run_training_loop.return_value = mock_metrics
            mock_loop_class.return_value = mock_loop

            # Run training session
            result = run_training_session(str(config_file))

            # Verify result
            assert isinstance(result, TrainingMetrics)
            assert result.iteration == 2


class TestIntegrationScenarios:
    """Integration test scenarios."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_signal_handling(self, temp_dir):
        """Test graceful shutdown via signal handling."""
        config = TrainingConfig(
            max_iterations=1000,  # Long-running
            checkpoint_dir=str(temp_dir / 'checkpoints'),
            log_dir=str(temp_dir / 'logs'),
            evaluation_dir=str(temp_dir / 'eval')
        )

        loop = TrainingLoop(config)

        # Simulate signal reception
        loop._signal_handler(signal.SIGINT, None)

        assert loop.shutdown_requested

    def test_signal_handler_from_worker_thread(self, temp_dir):
        """Test that TrainingLoop can be instantiated from worker thread without ValueError.

        Signal handlers can only be registered in the main thread. When TrainingLoop
        is created from a worker thread (e.g., in tests), signal registration should
        be skipped to avoid ValueError.
        """
        import threading

        error_occurred = []
        success = []

        def create_loop_in_thread():
            try:
                config = TrainingConfig(
                    checkpoint_dir=str(temp_dir / 'checkpoints'),
                    log_dir=str(temp_dir / 'logs'),
                    evaluation_dir=str(temp_dir / 'eval')
                )
                loop = TrainingLoop(config)
                success.append(True)
                loop.stop()
            except ValueError as e:
                if 'signal only works in main thread' in str(e):
                    error_occurred.append(e)
                else:
                    raise

        thread = threading.Thread(target=create_loop_in_thread)
        thread.start()
        thread.join()

        # Should succeed without ValueError
        assert not error_occurred, f"Signal handler guard failed: {error_occurred[0] if error_occurred else None}"
        assert success, "TrainingLoop should be created successfully in worker thread"

    @patch('src.training.training_loop.SelfPlayGameGenerator')
    @patch('src.training.training_loop.MemoryMappedExperienceBuffer')
    @patch('src.training.training_loop.AlphaZeroTrainer')
    def test_recovery_from_interruption(self, mock_trainer_class, mock_buffer_class,
                                      mock_generator_class, temp_dir):
        """Test recovery from training interruption."""
        config = TrainingConfig(
            max_iterations=10,
            checkpoint_dir=str(temp_dir / 'checkpoints'),
            log_dir=str(temp_dir / 'logs'),
            evaluation_dir=str(temp_dir / 'eval')
        )

        # First training session
        loop1 = TrainingLoop(config)
        loop1.current_iteration = 5
        loop1.metrics.total_games_generated = 250
        loop1._save_training_state()

        # Second training session (recovery)
        loop2 = TrainingLoop(config)
        loop2._load_training_state()

        # Verify state was restored
        assert loop2.current_iteration == 5
        assert loop2.metrics.total_games_generated == 250

    def test_memory_usage_monitoring(self, temp_dir):
        """Test memory usage monitoring."""
        config = TrainingConfig(
            checkpoint_dir=str(temp_dir / 'checkpoints'),
            log_dir=str(temp_dir / 'logs'),
            evaluation_dir=str(temp_dir / 'eval')
        )

        loop = TrainingLoop(config)

        # Update metrics (should include memory usage if psutil available)
        loop._update_metrics({'iteration_time_seconds': 30.0})

        # Memory usage should be tracked (or 0 if psutil not available)
        assert loop.metrics.memory_usage_mb >= 0


if __name__ == '__main__':
    pytest.main([__file__])