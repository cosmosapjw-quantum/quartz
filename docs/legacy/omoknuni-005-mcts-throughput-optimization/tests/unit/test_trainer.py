"""
Unit tests for neural network model trainer.

Tests the AlphaZeroTrainer implementation including:
- Model loading and initialization
- Training step execution with mixed precision
- Validation functionality
- Checkpoint saving and loading
- Learning rate scheduling
- Gradient clipping
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add specs to path for contract imports
sys.path.append('specs/001-goal-create-spec')
from contracts.training_api import TrainingExample

from src.training.trainer import AlphaZeroTrainer
from src.neural.model import create_model_for_game


class TestAlphaZeroTrainer:
    """Test suite for AlphaZeroTrainer."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_model_path(self, temp_dir):
        """Create a sample model file for testing."""
        model_path = temp_dir / "test_model.pth"
        model = create_model_for_game('gomoku')
        torch.save(model, model_path)
        return str(model_path)

    @pytest.fixture
    def sample_training_examples(self):
        """Create sample training examples for testing."""
        examples = []
        for i in range(10):
            state = np.random.rand(36, 15, 15).astype(np.float32)
            policy = np.random.rand(225).astype(np.float32)
            policy = policy / policy.sum()  # Normalize
            value = np.random.uniform(-1, 1)

            example = TrainingExample(
                state=state,
                policy=policy,
                value=value,
                game_type='gomoku',
                move_number=i,
                game_id=f'test_game_{i}'
            )
            examples.append(example)
        return examples

    def test_trainer_initialization_with_existing_model(self, sample_model_path):
        """Test trainer initialization with existing model."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            learning_rate=0.001,
            weight_decay=1e-4,
            batch_size=32,
            use_mixed_precision=False  # Disable for CPU testing
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.step_count == 0
        assert trainer.learning_rate == 0.001
        assert trainer.weight_decay == 1e-4

    def test_trainer_initialization_new_model(self, temp_dir):
        """Test trainer initialization with new model."""
        nonexistent_path = str(temp_dir / "nonexistent.pth")

        trainer = AlphaZeroTrainer(
            model_path=nonexistent_path,
            learning_rate=0.002,
            weight_decay=1e-3,
            batch_size=64,
            use_mixed_precision=False
        )

        assert trainer.model is not None
        assert trainer.learning_rate == 0.002
        assert trainer.weight_decay == 1e-3

    def test_parameter_counting(self, sample_model_path):
        """Test parameter counting functionality."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            use_mixed_precision=False
        )

        param_count = trainer._count_parameters()
        assert param_count > 0

        # Verify count matches manual calculation
        manual_count = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        assert param_count == manual_count

    def test_batch_preparation(self, sample_model_path, sample_training_examples):
        """Test batch preparation from training examples."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            use_mixed_precision=False
        )

        batch = sample_training_examples[:5]
        states, policies, values = trainer._prepare_batch(batch)

        assert states.shape == (5, 36, 15, 15)
        assert policies.shape == (5, 225)
        assert values.shape == (5,)
        assert states.dtype == torch.float32
        assert policies.dtype == torch.float32
        assert values.dtype == torch.float32

    def test_loss_computation(self, sample_model_path):
        """Test loss computation with mock data."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            use_mixed_precision=False
        )

        batch_size = 4
        num_actions = 225

        # Create mock predictions and targets
        policy_pred = torch.randn(batch_size, num_actions)
        value_pred = torch.randn(batch_size, 1)
        policy_target = torch.softmax(torch.randn(batch_size, num_actions), dim=1)
        value_target = torch.randn(batch_size)

        loss, metrics = trainer._compute_loss(policy_pred, value_pred, policy_target, value_target)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'total_loss' in metrics
        assert 'policy_accuracy' in metrics
        assert 'value_mae' in metrics

    def test_train_step(self, sample_model_path, sample_training_examples):
        """Test single training step execution."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            batch_size=32,
            use_mixed_precision=False,
            gradient_clip_norm=1.0
        )

        initial_step_count = trainer.step_count
        batch = sample_training_examples[:5]

        metrics = trainer.train_step(batch)

        # Verify training step executed
        assert trainer.step_count == initial_step_count + 1
        assert len(trainer.loss_history) > 0

        # Verify metrics returned
        required_metrics = ['policy_loss', 'value_loss', 'total_loss',
                          'learning_rate', 'step_time', 'batch_size', 'step_count']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

    def test_train_step_empty_batch(self, sample_model_path):
        """Test train_step with empty batch raises error."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            use_mixed_precision=False
        )

        with pytest.raises(ValueError, match="Empty batch provided"):
            trainer.train_step([])

    def test_validation(self, sample_model_path, sample_training_examples):
        """Test validation functionality."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            batch_size=32,
            use_mixed_precision=False
        )

        validation_data = sample_training_examples[:8]
        val_metrics = trainer.validate(validation_data)

        # Verify validation metrics
        expected_val_metrics = ['val_policy_loss', 'val_value_loss', 'val_total_loss',
                               'val_policy_accuracy', 'val_value_mae']
        for metric in expected_val_metrics:
            assert metric in val_metrics
            assert isinstance(val_metrics[metric], float)

    def test_validation_empty_data(self, sample_model_path):
        """Test validation with empty data returns empty dict."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            use_mixed_precision=False
        )

        val_metrics = trainer.validate([])
        assert val_metrics == {}

    def test_checkpoint_saving_and_loading(self, sample_model_path, temp_dir):
        """Test checkpoint saving and loading functionality."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            use_mixed_precision=False
        )

        # Modify trainer state
        trainer.step_count = 100
        trainer.epoch_count = 5
        trainer.loss_history.append(0.5)

        # Save checkpoint
        checkpoint_path = str(temp_dir / "checkpoint.pth")
        trainer.save_checkpoint(checkpoint_path)

        # Verify files were created
        assert Path(checkpoint_path).exists()
        assert Path(checkpoint_path).with_suffix('.state.pth').exists()

        # Create new trainer and load checkpoint
        new_trainer = AlphaZeroTrainer(
            model_path=checkpoint_path,  # Load from checkpoint
            use_mixed_precision=False
        )
        new_trainer.load_checkpoint(checkpoint_path)

        # Verify state was restored
        assert new_trainer.step_count == 100
        assert new_trainer.epoch_count == 5
        assert len(new_trainer.loss_history) > 0

    def test_learning_rate_scheduling(self, sample_model_path, sample_training_examples):
        """Test learning rate scheduling during training."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            learning_rate=0.01,
            lr_schedule_t_max=10,
            use_mixed_precision=False
        )

        initial_lr = trainer.scheduler.get_last_lr()[0]

        # Run several training steps
        batch = sample_training_examples[:3]
        for _ in range(5):
            trainer.train_step(batch)

        # Learning rate should have changed
        final_lr = trainer.scheduler.get_last_lr()[0]
        assert final_lr != initial_lr

    def test_scheduler_reset(self, sample_model_path):
        """Test learning rate scheduler reset functionality."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            lr_schedule_t_max=100,
            use_mixed_precision=False
        )

        original_t_max = trainer.lr_schedule_t_max
        trainer.reset_scheduler(t_max=50)

        assert trainer.lr_schedule_t_max == 50
        assert trainer.scheduler.T_max == 50

    def test_training_stats(self, sample_model_path, sample_training_examples):
        """Test training statistics collection."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            use_mixed_precision=False
        )

        # Run a few training steps to generate stats
        batch = sample_training_examples[:3]
        trainer.train_step(batch)
        trainer.validate(sample_training_examples[3:6])

        stats = trainer.get_training_stats()

        # Verify basic stats
        required_stats = ['step_count', 'epoch_count', 'current_lr',
                         'total_parameters', 'device', 'mixed_precision']
        for stat in required_stats:
            assert stat in stats

        # Verify loss statistics
        assert 'recent_loss_mean' in stats
        assert 'recent_loss_std' in stats
        assert 'recent_loss_min' in stats
        assert 'recent_loss_max' in stats

    def test_gradient_clipping(self, sample_model_path, sample_training_examples):
        """Test gradient clipping functionality."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            gradient_clip_norm=0.5,
            use_mixed_precision=False
        )

        batch = sample_training_examples[:3]
        metrics = trainer.train_step(batch)

        # Gradient norm should be reported
        assert 'gradient_norm' in metrics
        assert isinstance(metrics['gradient_norm'], float)

    def test_game_type_detection(self, temp_dir):
        """Test game type detection from state dict."""
        # Create models for different games
        gomoku_model = create_model_for_game('gomoku')
        chess_model = create_model_for_game('chess')
        go_model = create_model_for_game('go')

        # Save state dicts
        gomoku_path = temp_dir / "gomoku.pth"
        chess_path = temp_dir / "chess.pth"
        go_path = temp_dir / "go.pth"

        torch.save(gomoku_model.state_dict(), gomoku_path)
        torch.save(chess_model.state_dict(), chess_path)
        torch.save(go_model.state_dict(), go_path)

        # Test detection for each game
        for model_path, expected_game in [(gomoku_path, 'gomoku'),
                                         (chess_path, 'chess'),
                                         (go_path, 'go')]:
            trainer = AlphaZeroTrainer(
                model_path=str(model_path),
                use_mixed_precision=False
            )

            # Verify correct model was loaded
            state_dict = trainer.model.state_dict()
            first_conv_weight = None
            for key, tensor in state_dict.items():
                if 'initial_conv' in key and 'weight' in key:
                    first_conv_weight = tensor
                    break

            assert first_conv_weight is not None
            if expected_game == 'gomoku':
                assert first_conv_weight.shape[1] == 36
            elif expected_game == 'chess':
                assert first_conv_weight.shape[1] == 30
            elif expected_game == 'go':
                assert first_conv_weight.shape[1] == 25

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_training(self, sample_model_path, sample_training_examples):
        """Test mixed precision training (requires CUDA)."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            use_mixed_precision=True
        )

        assert trainer.scaler is not None

        batch = sample_training_examples[:3]
        metrics = trainer.train_step(batch)

        # Should complete without errors
        assert 'total_loss' in metrics
        assert metrics['step_count'] == 1

    def test_device_selection(self, sample_model_path):
        """Test device selection (CPU/CUDA)."""
        trainer = AlphaZeroTrainer(
            model_path=sample_model_path,
            use_mixed_precision=False
        )

        # Should select appropriate device
        if torch.cuda.is_available():
            assert trainer.device.type == 'cuda'
        else:
            assert trainer.device.type == 'cpu'

        # Model should be on the correct device (device type should match)
        model_device = next(trainer.model.parameters()).device
        assert model_device.type == trainer.device.type


if __name__ == '__main__':
    pytest.main([__file__])