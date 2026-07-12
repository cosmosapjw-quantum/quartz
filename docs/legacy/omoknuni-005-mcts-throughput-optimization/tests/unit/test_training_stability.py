"""
Unit tests for training stability monitoring system.

Tests stability monitoring features including NaN detection, gradient norm
monitoring, loss convergence tracking, early stopping, and recovery mechanisms.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Import the modules under test
from src.training.trainer import (
    StabilityConfig, TrainingStabilityMonitor, AlphaZeroTrainer
)
from src.neural.model import AlphaZeroNet, create_model_for_game

# Import contracts for test data
import sys
sys.path.append('specs/001-goal-create-spec')
from contracts.training_api import TrainingExample


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, inject_nan=False, inject_explosion=False):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.inject_nan = inject_nan
        self.inject_explosion = inject_explosion

    def forward(self, x):
        out = self.linear(x)
        if self.inject_nan:
            out = out * float('nan')
        elif self.inject_explosion:
            out = out * 1000.0  # Large values for gradient explosion
        return out, out.mean()


class TestStabilityConfig:
    """Test stability configuration."""

    def test_default_config(self):
        """Test default stability configuration values."""
        config = StabilityConfig()

        assert config.check_nan_frequency == 1
        assert config.gradient_explosion_threshold == 10.0
        assert config.loss_window == 100
        assert config.enable_early_stopping == True
        assert config.enable_automatic_recovery == True
        assert config.max_recovery_attempts == 3

    def test_custom_config(self):
        """Test custom stability configuration."""
        config = StabilityConfig(
            check_nan_frequency=5,
            gradient_explosion_threshold=5.0,
            early_stop_patience=50,
            max_recovery_attempts=5
        )

        assert config.check_nan_frequency == 5
        assert config.gradient_explosion_threshold == 5.0
        assert config.early_stop_patience == 50
        assert config.max_recovery_attempts == 5


class TestTrainingStabilityMonitor:
    """Test training stability monitoring system."""

    @pytest.fixture
    def stability_config(self):
        """Create test stability configuration."""
        return StabilityConfig(
            check_nan_frequency=1,
            gradient_norm_window=10,
            gradient_explosion_threshold=5.0,
            loss_window=20,
            early_stop_patience=10
        )

    @pytest.fixture
    def monitor(self, stability_config):
        """Create test stability monitor."""
        return TrainingStabilityMonitor(stability_config)

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        return MockModel()

    @pytest.fixture
    def mock_optimizer(self, mock_model):
        """Create mock optimizer."""
        return torch.optim.SGD(mock_model.parameters(), lr=0.01)

    def test_monitor_initialization(self, monitor, stability_config):
        """Test stability monitor initialization."""
        assert monitor.config == stability_config
        assert monitor.step_count == 0
        assert monitor.recovery_count == 0
        assert monitor.is_stable == True
        assert len(monitor.gradient_norms) == 0
        assert len(monitor.train_losses) == 0

    def test_nan_detection_in_loss(self, monitor, mock_model):
        """Test NaN detection in loss values."""
        # Create NaN loss
        nan_loss = torch.tensor(float('nan'))

        has_nan = monitor.check_nan_values(mock_model, nan_loss)
        assert has_nan == True

        # Create normal loss
        normal_loss = torch.tensor(1.0)
        has_nan = monitor.check_nan_values(mock_model, normal_loss)
        assert has_nan == False

    def test_nan_detection_in_parameters(self, monitor):
        """Test NaN detection in model parameters."""
        model = MockModel()
        loss = torch.tensor(1.0)

        # Directly inject NaN into parameters
        model.linear.weight.data.fill_(float('nan'))

        # Should detect NaN in parameters
        has_nan = monitor.check_nan_values(model, loss)
        assert has_nan == True

    def test_nan_detection_in_gradients(self, monitor, mock_model):
        """Test NaN detection in gradients."""
        loss = torch.tensor(1.0, requires_grad=True)

        # Create fake NaN gradient
        mock_model.linear.weight.grad = torch.full_like(mock_model.linear.weight, float('nan'))

        has_nan = monitor.check_nan_values(mock_model, loss)
        assert has_nan == True

    def test_gradient_norm_tracking(self, monitor, mock_model):
        """Test gradient norm tracking and explosion detection."""
        # Create normal gradients
        x = torch.randn(5, 10)
        output, _ = mock_model(x)
        loss = output.sum()
        loss.backward()

        grad_norm = monitor.update_gradient_norms(mock_model)
        assert grad_norm > 0.0
        assert len(monitor.gradient_norms) == 1

    def test_gradient_explosion_detection(self, monitor):
        """Test gradient explosion detection."""
        model = MockModel(inject_explosion=True)
        x = torch.randn(5, 10)

        # Create large gradients that will explode
        for _ in range(monitor.config.gradient_norm_patience + 1):
            output, _ = model(x)
            loss = output.sum()
            loss.backward()

            grad_norm = monitor.update_gradient_norms(model)

            # Clear gradients for next iteration
            model.zero_grad()

        # Should detect explosion after patience exceeded
        assert monitor.gradient_explosion_count > monitor.config.gradient_norm_patience

    def test_loss_tracking_and_convergence(self, monitor):
        """Test loss tracking and convergence detection."""
        # Simulate improving losses
        for i in range(20):
            train_loss = 10.0 - i * 0.1  # Decreasing loss
            val_loss = 9.0 - i * 0.1

            should_stop = monitor.update_loss_tracking(train_loss, val_loss)
            assert should_stop == False  # Should not stop while improving

        assert len(monitor.train_losses) == 20
        assert len(monitor.val_losses) == 20
        assert monitor.best_val_loss < 9.0

    def test_loss_divergence_detection(self, monitor):
        """Test loss divergence detection."""
        # First, add some normal losses
        for i in range(15):
            monitor.update_loss_tracking(1.0 + i * 0.01)  # Slowly increasing

        # Then add diverging losses
        for i in range(10):
            train_loss = 2.0 + i * 0.5  # Rapidly increasing
            should_stop = monitor.update_loss_tracking(train_loss)
            if should_stop:
                break

        # Should detect divergence
        assert should_stop == True

    def test_early_stopping_on_validation(self, monitor):
        """Test early stopping based on validation metrics."""
        # Simulate no improvement in validation loss
        best_val_loss = 1.0

        for i in range(monitor.config.early_stop_patience + 5):
            train_loss = 1.0 + i * 0.01
            val_loss = best_val_loss + 0.1  # No improvement

            should_stop = monitor.update_loss_tracking(train_loss, val_loss)
            if should_stop:
                break

        assert should_stop == True
        assert monitor.no_improvement_count >= monitor.config.early_stop_patience

    def test_plateau_detection(self, monitor):
        """Test training plateau detection."""
        # Fill with plateau-like losses (very small changes)
        base_loss = 1.0
        for i in range(monitor.config.loss_window):
            loss = base_loss + np.random.normal(0, 1e-6)  # Very small changes
            monitor.update_loss_tracking(loss)

        is_plateau = monitor.check_plateau()
        assert is_plateau == True

    def test_automatic_recovery(self, monitor, mock_optimizer):
        """Test automatic recovery mechanism."""
        original_lr = mock_optimizer.param_groups[0]['lr']

        # Attempt recovery
        recovery_attempted = monitor.attempt_recovery(mock_optimizer)
        assert recovery_attempted == True
        assert monitor.recovery_count == 1

        # Check learning rate was reduced
        new_lr = mock_optimizer.param_groups[0]['lr']
        assert new_lr < original_lr
        assert new_lr == original_lr * monitor.config.recovery_lr_decay

    def test_recovery_limit(self, monitor, mock_optimizer):
        """Test recovery attempt limit."""
        # Exhaust recovery attempts
        for _ in range(monitor.config.max_recovery_attempts):
            monitor.attempt_recovery(mock_optimizer)

        # Should refuse further recovery
        recovery_attempted = monitor.attempt_recovery(mock_optimizer)
        assert recovery_attempted == False
        assert monitor.recovery_count == monitor.config.max_recovery_attempts

    def test_stability_monitoring_step(self, monitor, mock_model, mock_optimizer):
        """Test complete stability monitoring step."""
        x = torch.randn(5, 10)
        output, _ = mock_model(x)
        loss = output.sum()
        loss.backward()

        results = monitor.step(mock_model, loss, mock_optimizer)

        assert 'step' in results
        assert 'is_stable' in results
        assert 'should_stop' in results
        assert 'recovery_attempted' in results
        assert 'warnings' in results
        assert 'gradient_norm' in results

        assert results['is_stable'] == True
        assert results['should_stop'] == False
        assert isinstance(results['warnings'], list)

    def test_stability_statistics(self, monitor, mock_model, mock_optimizer):
        """Test stability statistics collection."""
        # Run several monitoring steps
        for i in range(10):
            x = torch.randn(5, 10)
            output, _ = mock_model(x)
            loss = output.sum()
            loss.backward()

            monitor.step(mock_model, loss, mock_optimizer, val_loss=1.0 - i * 0.1)
            mock_model.zero_grad()

        stats = monitor.get_statistics()

        assert 'step_count' in stats
        assert 'recovery_count' in stats
        assert 'is_stable' in stats
        assert 'gradient_norm_mean' in stats
        assert 'train_loss_mean' in stats
        assert 'val_loss_mean' in stats

        assert stats['step_count'] == 10
        assert stats['gradient_norm_mean'] > 0


class TestAlphaZeroTrainerWithStability:
    """Test AlphaZero trainer with stability monitoring integration."""

    @pytest.fixture
    def temp_model_path(self):
        """Create temporary model path for non-existent model (to create new)."""
        temp_path = "/tmp/nonexistent_model.pth"
        yield temp_path
        # No cleanup needed since file doesn't exist

    @pytest.fixture
    def trainer_with_stability(self, temp_model_path):
        """Create trainer with stability monitoring."""
        stability_config = StabilityConfig(
            check_nan_frequency=1,
            gradient_explosion_threshold=5.0,
            early_stop_patience=5
        )

        trainer = AlphaZeroTrainer(
            model_path=temp_model_path,
            learning_rate=0.001,
            batch_size=32,
            stability_config=stability_config
        )
        return trainer

    @pytest.fixture
    def mock_training_batch(self):
        """Create mock training batch."""
        batch = []
        for i in range(8):
            # Mock training example
            example = Mock()
            example.state = np.random.random((36, 15, 15)).astype(np.float32)  # Gomoku
            example.policy = np.random.random(225).astype(np.float32)  # 15x15 board
            example.value = np.random.uniform(-1, 1)
            batch.append(example)
        return batch

    def test_trainer_initialization_with_stability(self, trainer_with_stability):
        """Test trainer initialization with stability monitoring."""
        assert hasattr(trainer_with_stability, 'stability_monitor')
        assert isinstance(trainer_with_stability.stability_monitor, TrainingStabilityMonitor)

    def test_train_step_with_stability_monitoring(self, trainer_with_stability, mock_training_batch):
        """Test training step with stability monitoring integration."""
        metrics = trainer_with_stability.train_step(mock_training_batch)

        # Check that stability metrics are included
        assert 'stability_is_stable' in metrics
        assert 'stability_should_stop' in metrics
        assert 'stability_recovery_attempted' in metrics
        assert 'stability_gradient_norm' in metrics
        assert 'stability_warnings' in metrics

        # Note: May not be stable initially due to random model initialization
        # but should not recommend stopping or have attempted recovery on first step
        assert metrics['stability_should_stop'] == False

    def test_training_stats_include_stability(self, trainer_with_stability, mock_training_batch):
        """Test that training statistics include stability metrics."""
        # Run a training step to populate metrics
        trainer_with_stability.train_step(mock_training_batch)

        stats = trainer_with_stability.get_training_stats()

        # Check for stability statistics
        stability_keys = [key for key in stats.keys() if key.startswith('stability_')]
        assert len(stability_keys) > 0

        assert 'stability_step_count' in stats
        assert 'stability_is_stable' in stats

    @patch('src.training.trainer.logger')
    def test_stability_warning_logging(self, mock_logger, trainer_with_stability, mock_training_batch):
        """Test that stability warnings are properly logged."""
        # Simulate a scenario that triggers warnings
        trainer_with_stability.stability_monitor.last_nan_step = 0

        # Mock the stability monitor to return warnings
        with patch.object(trainer_with_stability.stability_monitor, 'step') as mock_step:
            mock_step.return_value = {
                'step': 1,
                'is_stable': False,
                'should_stop': False,
                'recovery_attempted': True,
                'warnings': ['NaN detected', 'Gradient explosion'],
                'gradient_norm': 15.0
            }

            metrics = trainer_with_stability.train_step(mock_training_batch)

            # Check that warnings were logged
            mock_logger.warning.assert_called()
            mock_logger.info.assert_called()

    def test_nan_recovery_in_training(self, trainer_with_stability, mock_training_batch):
        """Test NaN recovery during training."""
        # Inject NaN into model parameters
        for param in trainer_with_stability.model.parameters():
            param.data.fill_(float('nan'))

        # Mock recovery to avoid actual NaN propagation
        with patch.object(trainer_with_stability.stability_monitor, 'attempt_recovery') as mock_recovery:
            mock_recovery.return_value = True

            with patch.object(trainer_with_stability.stability_monitor, 'check_nan_values') as mock_nan_check:
                mock_nan_check.return_value = True

                metrics = trainer_with_stability.train_step(mock_training_batch)

                # Should indicate recovery was attempted
                assert metrics.get('stability_recovery_attempted', False) == True

    def test_validation_loss_tracking(self, trainer_with_stability, mock_training_batch):
        """Test validation loss tracking in stability monitoring."""
        # Create validation data
        val_data = mock_training_batch[:4]  # Smaller validation set

        # Run validation
        val_metrics = trainer_with_stability.validate(val_data)

        # Should include validation loss
        assert 'val_total_loss' in val_metrics

        # Stability monitor should have been updated with validation loss
        assert len(trainer_with_stability.stability_monitor.val_losses) > 0

    def test_early_stopping_integration(self, trainer_with_stability, mock_training_batch):
        """Test early stopping integration with trainer."""
        # Configure for quick early stopping
        trainer_with_stability.stability_monitor.config.early_stop_patience = 2

        # Simulate poor validation performance
        with patch.object(trainer_with_stability.stability_monitor, 'update_loss_tracking') as mock_update:
            mock_update.return_value = True  # Trigger early stopping

            metrics = trainer_with_stability.train_step(mock_training_batch)

            # Should indicate training should stop
            assert metrics.get('stability_stop_training', False) == True


class TestStabilityEdgeCases:
    """Test edge cases and error conditions in stability monitoring."""

    def test_empty_gradient_handling(self):
        """Test handling of models with no gradients."""
        monitor = TrainingStabilityMonitor()
        model = nn.Linear(5, 2)

        # No gradients computed yet
        grad_norm = monitor.update_gradient_norms(model)
        assert grad_norm == 0.0

    def test_stability_with_disabled_features(self):
        """Test stability monitoring with disabled features."""
        config = StabilityConfig(
            enable_early_stopping=False,
            enable_automatic_recovery=False
        )
        monitor = TrainingStabilityMonitor(config)

        # Should not attempt recovery
        mock_optimizer = Mock()
        recovery_attempted = monitor.attempt_recovery(mock_optimizer)
        assert recovery_attempted == False

    def test_single_loss_convergence(self):
        """Test convergence detection with minimal data."""
        monitor = TrainingStabilityMonitor()

        # Add only one loss
        should_stop = monitor.update_loss_tracking(1.0)
        assert should_stop == False

        # Plateau detection with insufficient data
        is_plateau = monitor.check_plateau()
        assert is_plateau == False

    def test_statistics_with_empty_data(self):
        """Test statistics collection with no data."""
        monitor = TrainingStabilityMonitor()

        stats = monitor.get_statistics()
        assert stats['step_count'] == 0
        assert stats['recovery_count'] == 0
        assert stats['is_stable'] == True

        # Should not have gradient/loss statistics
        assert 'gradient_norm_mean' not in stats
        assert 'train_loss_mean' not in stats


if __name__ == '__main__':
    pytest.main([__file__])