"""
Unit tests for checkpoint management system.

Tests checkpoint saving, loading, versioning, best model tracking,
retention policies, and cleanup functionality.
"""

import pytest
import torch
import torch.nn as nn
import json
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules under test
from src.training.checkpoint_manager import (
    CheckpointManager, CheckpointMetadata, RetentionPolicy
)


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.linear = nn.Linear(10, hidden_size)
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, x):
        return self.output(self.linear(x))


class TestCheckpointMetadata:
    """Test checkpoint metadata functionality."""

    def test_metadata_creation(self):
        """Test creating checkpoint metadata."""
        metadata = CheckpointMetadata(
            version=1,
            timestamp=time.time(),
            step_count=1000,
            epoch_count=5,
            train_loss=0.5,
            val_loss=0.3,
            learning_rate=0.001
        )

        assert metadata.version == 1
        assert metadata.step_count == 1000
        assert metadata.train_loss == 0.5
        assert metadata.val_loss == 0.3

    def test_metadata_serialization(self):
        """Test metadata to/from dict conversion."""
        metadata = CheckpointMetadata(
            version=2,
            timestamp=1234567890.0,
            step_count=2000,
            epoch_count=10,
            tags=["milestone", "best"]
        )

        # Test to_dict
        data = metadata.to_dict()
        assert data['version'] == 2
        assert data['timestamp'] == 1234567890.0
        assert data['tags'] == ["milestone", "best"]

        # Test from_dict
        restored = CheckpointMetadata.from_dict(data)
        assert restored.version == metadata.version
        assert restored.timestamp == metadata.timestamp
        assert restored.tags == metadata.tags

    def test_metadata_optional_fields(self):
        """Test metadata with optional fields."""
        metadata = CheckpointMetadata(
            version=1,
            timestamp=time.time(),
            step_count=100,
            epoch_count=1
        )

        assert metadata.train_loss is None
        assert metadata.notes == ""
        assert metadata.tags == []


class TestRetentionPolicy:
    """Test retention policy configuration."""

    def test_default_policy(self):
        """Test default retention policy values."""
        policy = RetentionPolicy()

        assert policy.keep_recent == 10
        assert policy.keep_best == 5
        assert policy.keep_milestone_every == 10000
        assert policy.keep_days == 7
        assert policy.min_free_space_gb == 5.0
        assert policy.max_storage_gb == 50.0

    def test_custom_policy(self):
        """Test custom retention policy."""
        policy = RetentionPolicy(
            keep_recent=20,
            keep_best=10,
            keep_milestone_every=5000,
            keep_days=14
        )

        assert policy.keep_recent == 20
        assert policy.keep_best == 10
        assert policy.keep_milestone_every == 5000
        assert policy.keep_days == 14


class TestCheckpointManager:
    """Test checkpoint manager functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        return MockModel()

    @pytest.fixture
    def mock_optimizer(self, mock_model):
        """Create mock optimizer."""
        return torch.optim.Adam(mock_model.parameters(), lr=0.001)

    @pytest.fixture
    def mock_scheduler(self, mock_optimizer):
        """Create mock scheduler."""
        return torch.optim.lr_scheduler.StepLR(mock_optimizer, step_size=100)

    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create checkpoint manager for testing."""
        return CheckpointManager(
            checkpoint_dir=temp_dir,
            auto_save_every=100,
            enable_best_tracking=True,
            best_metric="val_loss",
            best_mode="min"
        )

    def test_manager_initialization(self, temp_dir):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(temp_dir)

        assert manager.checkpoint_dir == temp_dir
        assert manager.auto_save_every == 1000  # default
        assert manager.enable_best_tracking == True
        assert manager.best_metric == "val_loss"
        assert manager.best_mode == "min"
        assert manager.version_counter == 0

    def test_checkpoint_directory_creation(self, temp_dir):
        """Test automatic directory creation."""
        nested_dir = temp_dir / "nested" / "checkpoints"
        manager = CheckpointManager(nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_save_checkpoint_basic(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test basic checkpoint saving."""
        metrics = {
            'train_loss': 0.5,
            'val_loss': 0.3,
            'learning_rate': 0.001
        }

        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            step_count=100,
            epoch_count=1,
            metrics=metrics,
            force=True
        )

        assert checkpoint_path is not None
        assert checkpoint_path.exists()
        assert checkpoint_path.name == "checkpoint_v000001.pth"

        # Check state file exists
        state_path = checkpoint_path.with_suffix('.state.pth')
        assert state_path.exists()

        # Check metadata was created
        assert 1 in checkpoint_manager.checkpoints_metadata
        metadata = checkpoint_manager.checkpoints_metadata[1]
        assert metadata.version == 1
        assert metadata.step_count == 100
        assert metadata.train_loss == 0.5

    def test_save_checkpoint_with_scheduler(self, checkpoint_manager, mock_model,
                                         mock_optimizer, mock_scheduler):
        """Test checkpoint saving with scheduler."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            step_count=200,
            force=True
        )

        assert checkpoint_path is not None

        # Load and verify state includes scheduler
        state_path = checkpoint_path.with_suffix('.state.pth')
        state = torch.load(state_path, weights_only=False)
        assert 'scheduler_state_dict' in state
        assert state['scheduler_state_dict'] is not None

    def test_auto_save_interval(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test auto-save interval behavior."""
        # Should not save if interval not reached
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            step_count=50,  # Less than auto_save_every=100
            force=False
        )
        assert checkpoint_path is None

        # Should save if interval reached
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            step_count=100,
            force=False
        )
        assert checkpoint_path is not None

    def test_best_model_tracking(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test best model tracking functionality."""
        # Save first checkpoint
        metrics1 = {'val_loss': 0.5}
        checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            step_count=100,
            metrics=metrics1,
            force=True
        )

        # Save better checkpoint
        metrics2 = {'val_loss': 0.3}  # Better (lower)
        checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            step_count=200,
            metrics=metrics2,
            force=True
        )

        # Check best model was updated
        assert checkpoint_manager.best_score == 0.3
        best_path = checkpoint_manager.checkpoint_dir / "best_model.pth"
        assert best_path.exists()

        # Save worse checkpoint
        metrics3 = {'val_loss': 0.7}  # Worse (higher)
        checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            step_count=300,
            metrics=metrics3,
            force=True
        )

        # Best should not change
        assert checkpoint_manager.best_score == 0.3

    def test_best_model_tracking_max_mode(self, temp_dir, mock_model, mock_optimizer):
        """Test best model tracking with max mode."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            best_metric="win_rate",
            best_mode="max"
        )

        # Save first checkpoint
        metrics1 = {'win_rate': 0.6}
        manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            step_count=100,
            metrics=metrics1,
            force=True
        )

        # Save better checkpoint (higher win rate)
        metrics2 = {'win_rate': 0.8}
        manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            step_count=200,
            metrics=metrics2,
            force=True
        )

        assert manager.best_score == 0.8

    def test_checkpoint_versioning(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test checkpoint version numbering."""
        # Save multiple checkpoints
        for i in range(1, 4):
            checkpoint_manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                step_count=i * 100,
                force=True
            )

        # Check versions
        assert checkpoint_manager.version_counter == 3
        assert len(checkpoint_manager.checkpoints_metadata) == 3

        # Check file names
        for i in range(1, 4):
            checkpoint_path = checkpoint_manager.checkpoint_dir / f"checkpoint_v{i:06d}.pth"
            assert checkpoint_path.exists()

    def test_load_checkpoint_latest(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test loading latest checkpoint."""
        # Save multiple checkpoints
        for i in range(1, 4):
            checkpoint_manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                step_count=i * 100,
                force=True
            )

        # Load latest
        result = checkpoint_manager.load_checkpoint()
        assert result is not None

        checkpoint_path, metadata = result
        assert metadata.version == 3  # Latest
        assert metadata.step_count == 300

    def test_load_checkpoint_specific_version(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test loading specific checkpoint version."""
        # Save multiple checkpoints
        for i in range(1, 4):
            checkpoint_manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                step_count=i * 100,
                force=True
            )

        # Load specific version
        result = checkpoint_manager.load_checkpoint(version=2)
        assert result is not None

        checkpoint_path, metadata = result
        assert metadata.version == 2
        assert metadata.step_count == 200

    def test_load_checkpoint_best(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test loading best checkpoint."""
        # Save checkpoints with different scores
        metrics_list = [
            {'val_loss': 0.5},
            {'val_loss': 0.3},  # Best
            {'val_loss': 0.7}
        ]

        for i, metrics in enumerate(metrics_list, 1):
            checkpoint_manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                step_count=i * 100,
                metrics=metrics,
                force=True
            )

        # Load best
        result = checkpoint_manager.load_checkpoint(load_best=True)
        assert result is not None

        checkpoint_path, metadata = result
        assert metadata.version == 2  # Best was version 2
        assert metadata.val_loss == 0.3

    def test_list_checkpoints(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test listing checkpoints with filtering."""
        # Save checkpoints with different properties
        checkpoints_data = [
            (100, {'val_loss': 0.5}, ["train"]),
            (500, {'val_loss': 0.3}, ["milestone"]),
            (1000, {'val_loss': 0.4}, ["milestone", "test"])
        ]

        for step, metrics, tags in checkpoints_data:
            checkpoint_manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                step_count=step,
                metrics=metrics,
                tags=tags,
                force=True
            )

        # List all
        all_checkpoints = checkpoint_manager.list_checkpoints()
        assert len(all_checkpoints) == 3

        # Filter by tags
        milestone_checkpoints = checkpoint_manager.list_checkpoints(tags=["milestone"])
        assert len(milestone_checkpoints) == 2

        # Filter by step range
        filtered_checkpoints = checkpoint_manager.list_checkpoints(min_step=200, max_step=800)
        assert len(filtered_checkpoints) == 1
        assert filtered_checkpoints[0][1].step_count == 500

    def test_delete_checkpoint(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test checkpoint deletion."""
        # Save checkpoints
        for i in range(1, 4):
            checkpoint_manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                step_count=i * 100,
                force=True
            )

        # Delete specific checkpoint
        success = checkpoint_manager.delete_checkpoint(2)
        assert success == True

        # Check it's gone
        assert 2 not in checkpoint_manager.checkpoints_metadata
        checkpoint_path = checkpoint_manager.checkpoint_dir / "checkpoint_v000002.pth"
        assert not checkpoint_path.exists()

        # Other checkpoints should remain
        assert 1 in checkpoint_manager.checkpoints_metadata
        assert 3 in checkpoint_manager.checkpoints_metadata

    def test_delete_best_checkpoint_protection(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test that best checkpoint cannot be deleted."""
        # Save checkpoint that becomes best
        metrics = {'val_loss': 0.3}
        checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            step_count=100,
            metrics=metrics,
            force=True
        )

        # Try to delete best checkpoint
        success = checkpoint_manager.delete_checkpoint(1)
        assert success == False  # Should be protected

        # Checkpoint should still exist
        assert 1 in checkpoint_manager.checkpoints_metadata

    def test_retention_policy_recent(self, temp_dir, mock_model, mock_optimizer):
        """Test retention policy keeping recent checkpoints."""
        policy = RetentionPolicy(keep_recent=2, keep_best=0, keep_days=None)
        manager = CheckpointManager(temp_dir, retention_policy=policy)

        # Save many checkpoints
        for i in range(1, 6):
            manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                step_count=i * 100,
                force=True
            )

        # Should only keep 2 most recent
        remaining = list(manager.checkpoints_metadata.keys())
        assert len(remaining) == 2
        assert 4 in remaining  # Second most recent
        assert 5 in remaining  # Most recent

    def test_retention_policy_best(self, temp_dir, mock_model, mock_optimizer):
        """Test retention policy keeping best checkpoints."""
        policy = RetentionPolicy(keep_recent=0, keep_best=2, keep_days=None)
        manager = CheckpointManager(temp_dir, retention_policy=policy)

        # Save checkpoints with different scores
        scores = [0.8, 0.3, 0.6, 0.2, 0.9]  # Best are 0.2 and 0.3
        for i, score in enumerate(scores, 1):
            manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                step_count=i * 100,
                metrics={'val_loss': score},
                force=True
            )

        # Should keep 2 best (lowest val_loss)
        remaining = list(manager.checkpoints_metadata.keys())
        assert len(remaining) == 2
        assert 4 in remaining  # val_loss = 0.2 (best)
        assert 2 in remaining  # val_loss = 0.3 (second best)

    def test_retention_policy_milestone(self, temp_dir, mock_model, mock_optimizer):
        """Test retention policy keeping milestone checkpoints."""
        policy = RetentionPolicy(keep_recent=0, keep_best=0, keep_milestone_every=1000, keep_days=None)
        manager = CheckpointManager(temp_dir, retention_policy=policy)

        # Save checkpoints at different steps
        steps = [500, 1000, 1500, 2000, 2500]  # Milestones at 1000, 2000
        for i, step in enumerate(steps, 1):
            manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                step_count=step,
                force=True
            )

        # Should keep milestone checkpoints
        remaining = list(manager.checkpoints_metadata.keys())
        assert 2 in remaining  # step 1000
        assert 4 in remaining  # step 2000

    def test_should_auto_save(self, checkpoint_manager):
        """Test auto-save trigger logic."""
        # Initially should save
        assert checkpoint_manager.should_auto_save(100) == True

        # After saving, should not save again immediately
        checkpoint_manager.last_save_step = 100
        assert checkpoint_manager.should_auto_save(150) == False

        # Should save after interval
        assert checkpoint_manager.should_auto_save(200) == True

    def test_storage_stats(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test storage statistics."""
        # Save some checkpoints
        for i in range(1, 4):
            checkpoint_manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                step_count=i * 100,
                metrics={'val_loss': 0.5 - i * 0.1},  # Best is last
                force=True
            )

        stats = checkpoint_manager.get_storage_stats()

        assert stats['checkpoint_count'] == 3
        assert stats['total_size_gb'] > 0
        assert 'free_space_gb' in stats
        assert 'best_checkpoint' in stats
        assert abs(stats['best_score'] - 0.2) < 1e-10  # Best val_loss

    def test_checkpoint_info(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test getting checkpoint information."""
        # Save checkpoint with metadata
        metrics = {
            'train_loss': 0.5,
            'val_loss': 0.3,
            'learning_rate': 0.001
        }

        checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            step_count=100,
            metrics=metrics,
            tags=["test"],
            notes="Test checkpoint",
            force=True
        )

        info = checkpoint_manager.get_checkpoint_info(1)
        assert info is not None
        assert info['version'] == 1
        assert info['step_count'] == 100
        assert info['train_loss'] == 0.5
        assert info['tags'] == ["test"]
        assert info['notes'] == "Test checkpoint"
        assert info['exists'] == True

    def test_metadata_persistence(self, temp_dir, mock_model, mock_optimizer):
        """Test metadata persistence across manager instances."""
        # Create manager and save checkpoints
        manager1 = CheckpointManager(temp_dir)
        manager1.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            step_count=100,
            metrics={'val_loss': 0.3},
            force=True
        )

        # Create new manager instance
        manager2 = CheckpointManager(temp_dir)

        # Should load existing metadata
        assert len(manager2.checkpoints_metadata) == 1
        assert 1 in manager2.checkpoints_metadata
        assert manager2.version_counter == 1
        assert manager2.best_score == 0.3

    def test_error_handling_save_failure(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test error handling during save failure."""
        # Mock torch.save to fail
        with patch('torch.save', side_effect=Exception("Save failed")):
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                step_count=100,
                force=True
            )

            assert checkpoint_path is None
            assert len(checkpoint_manager.checkpoints_metadata) == 0

    def test_error_handling_load_missing(self, checkpoint_manager):
        """Test error handling when loading missing checkpoint."""
        result = checkpoint_manager.load_checkpoint(version=999)
        assert result is None

        result = checkpoint_manager.load_checkpoint(load_best=True)
        assert result is None

    def test_checkpoint_tags_and_notes(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test checkpoint tagging and notes functionality."""
        checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            step_count=100,
            tags=["milestone", "stable"],
            notes="Important checkpoint before major change",
            force=True
        )

        metadata = checkpoint_manager.checkpoints_metadata[1]
        assert "milestone" in metadata.tags
        assert "stable" in metadata.tags
        assert metadata.notes == "Important checkpoint before major change"


if __name__ == '__main__':
    pytest.main([__file__])