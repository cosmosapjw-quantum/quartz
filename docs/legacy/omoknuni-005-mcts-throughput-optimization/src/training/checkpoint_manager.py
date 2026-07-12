"""
Checkpoint Management System
===========================

Comprehensive checkpoint management for training pipelines with automatic saving,
best model tracking, versioning, and cleanup policies.

Features:
- Automatic checkpoint saving with configurable intervals
- Best model tracking based on evaluation metrics
- Version numbering and metadata tracking
- Automatic cleanup of old checkpoints with retention policies
- Comprehensive checkpoint validation and recovery
"""

import os
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint tracking."""

    version: int
    timestamp: float
    step_count: int
    epoch_count: int

    # Training metrics
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None

    # Evaluation metrics
    evaluation_score: Optional[float] = None
    win_rate: Optional[float] = None
    elo_rating: Optional[float] = None

    # Model info
    model_size_mb: Optional[float] = None
    game_type: Optional[str] = None

    # Additional metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary."""
        # Handle tags list properly
        if 'tags' not in data:
            data['tags'] = []
        return cls(**data)


@dataclass
class RetentionPolicy:
    """Configuration for checkpoint retention and cleanup."""

    # Keep N most recent checkpoints
    keep_recent: int = 10

    # Keep N best checkpoints (by evaluation score)
    keep_best: int = 5

    # Keep checkpoints every N steps (for milestone tracking)
    keep_milestone_every: Optional[int] = 10000

    # Keep checkpoints from last N days
    keep_days: Optional[int] = 7

    # Minimum free disk space to maintain (in GB)
    min_free_space_gb: Optional[float] = 5.0

    # Maximum total checkpoint storage (in GB)
    max_storage_gb: Optional[float] = 50.0


class CheckpointManager:
    """
    Comprehensive checkpoint management system.

    Features:
    - Automatic saving with configurable intervals
    - Best model tracking with multiple metrics
    - Version numbering and metadata
    - Retention policies and cleanup
    - Checkpoint validation and recovery
    """

    def __init__(self,
                 checkpoint_dir: Union[str, Path],
                 retention_policy: Optional[RetentionPolicy] = None,
                 auto_save_every: int = 1000,
                 enable_best_tracking: bool = True,
                 best_metric: str = "val_loss",
                 best_mode: str = "min"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            retention_policy: Policy for keeping/cleaning checkpoints
            auto_save_every: Auto-save interval in training steps
            enable_best_tracking: Whether to track best model
            best_metric: Metric name for best model tracking
            best_mode: "min" or "max" for best metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.retention_policy = retention_policy or RetentionPolicy()
        self.auto_save_every = auto_save_every
        self.enable_best_tracking = enable_best_tracking
        self.best_metric = best_metric
        self.best_mode = best_mode

        # State tracking
        self.last_save_step = 0
        self.version_counter = 0
        self.best_score = float('inf') if best_mode == "min" else float('-inf')
        self.best_checkpoint_path = None

        # Metadata tracking
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.checkpoints_metadata: Dict[int, CheckpointMetadata] = {}

        # Load existing metadata
        self._load_metadata()

        logger.info(f"CheckpointManager initialized - Dir: {checkpoint_dir}, "
                   f"Auto-save: every {auto_save_every} steps, "
                   f"Best tracking: {enable_best_tracking} ({best_metric}, {best_mode})")

    def _load_metadata(self) -> None:
        """Load existing checkpoint metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)

                # Load checkpoints metadata
                if 'checkpoints' in data:
                    for version_str, meta_dict in data['checkpoints'].items():
                        version = int(version_str)
                        self.checkpoints_metadata[version] = CheckpointMetadata.from_dict(meta_dict)

                # Load manager state
                self.version_counter = data.get('version_counter', 0)
                self.best_score = data.get('best_score',
                    float('inf') if self.best_mode == "min" else float('-inf'))
                self.best_checkpoint_path = data.get('best_checkpoint_path')

                logger.info(f"Loaded metadata for {len(self.checkpoints_metadata)} checkpoints")

            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")
                self.checkpoints_metadata = {}

    def _save_metadata(self) -> None:
        """Save checkpoint metadata to disk."""
        try:
            data = {
                'version_counter': self.version_counter,
                'best_score': self.best_score,
                'best_checkpoint_path': self.best_checkpoint_path,
                'checkpoints': {
                    str(version): metadata.to_dict()
                    for version, metadata in self.checkpoints_metadata.items()
                }
            }

            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")

    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       step_count: int = 0,
                       epoch_count: int = 0,
                       metrics: Optional[Dict[str, float]] = None,
                       force: bool = False,
                       tags: Optional[List[str]] = None,
                       notes: str = "") -> Optional[Path]:
        """
        Save a checkpoint with full metadata.

        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            scheduler: Optional learning rate scheduler
            step_count: Current training step
            epoch_count: Current training epoch
            metrics: Dictionary of training/validation metrics
            force: Force save even if auto-save interval not reached
            tags: Optional tags for this checkpoint
            notes: Optional notes for this checkpoint

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        # Check if we should save
        if not force and (step_count - self.last_save_step) < self.auto_save_every:
            return None

        # Create new version
        self.version_counter += 1
        version = self.version_counter

        # Create checkpoint path
        checkpoint_path = self.checkpoint_dir / f"checkpoint_v{version:06d}.pth"
        state_path = self.checkpoint_dir / f"checkpoint_v{version:06d}.state.pth"

        try:
            # Save model
            torch.save(model, checkpoint_path)

            # Save training state
            training_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'step_count': step_count,
                'epoch_count': epoch_count,
                'version': version,
                'timestamp': time.time(),
            }
            torch.save(training_state, state_path)

            # Calculate model size
            model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

            # Create metadata
            metrics = metrics or {}
            metadata = CheckpointMetadata(
                version=version,
                timestamp=time.time(),
                step_count=step_count,
                epoch_count=epoch_count,
                train_loss=metrics.get('train_loss'),
                val_loss=metrics.get('val_loss'),
                learning_rate=metrics.get('learning_rate'),
                evaluation_score=metrics.get('evaluation_score'),
                win_rate=metrics.get('win_rate'),
                elo_rating=metrics.get('elo_rating'),
                model_size_mb=model_size_mb,
                game_type=metrics.get('game_type'),
                notes=notes,
                tags=tags or []
            )

            # Store metadata
            self.checkpoints_metadata[version] = metadata

            # Update best model tracking
            if self.enable_best_tracking and self.best_metric in metrics:
                score = metrics[self.best_metric]
                is_better = (
                    (self.best_mode == "min" and score < self.best_score) or
                    (self.best_mode == "max" and score > self.best_score)
                )

                if is_better:
                    self.best_score = score
                    self.best_checkpoint_path = str(checkpoint_path)
                    # Copy to best model path
                    best_path = self.checkpoint_dir / "best_model.pth"
                    shutil.copy2(checkpoint_path, best_path)
                    shutil.copy2(state_path, self.checkpoint_dir / "best_model.state.pth")
                    logger.info(f"New best model saved: {self.best_metric}={score:.6f}")

            # Save metadata
            self._save_metadata()

            # Update tracking
            self.last_save_step = step_count

            logger.info(f"Checkpoint v{version} saved: {checkpoint_path} "
                       f"(step {step_count}, {model_size_mb:.1f}MB)")

            # Run cleanup if needed
            self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint v{version}: {e}")
            # Clean up partial files
            for path in [checkpoint_path, state_path]:
                if path.exists():
                    path.unlink()
            return None

    def load_checkpoint(self,
                       version: Optional[int] = None,
                       load_best: bool = False) -> Optional[Tuple[Path, CheckpointMetadata]]:
        """
        Load a checkpoint.

        Args:
            version: Specific version to load, or None for latest
            load_best: Load best model instead of latest/specific version

        Returns:
            Tuple of (checkpoint_path, metadata) or None if not found
        """
        if load_best:
            if self.best_checkpoint_path and Path(self.best_checkpoint_path).exists():
                best_path = Path(self.best_checkpoint_path)
                version = self._extract_version_from_path(best_path)
                if version and version in self.checkpoints_metadata:
                    return best_path, self.checkpoints_metadata[version]
            return None

        if version is None:
            # Load latest
            if not self.checkpoints_metadata:
                return None
            version = max(self.checkpoints_metadata.keys())

        if version not in self.checkpoints_metadata:
            logger.error(f"Checkpoint version {version} not found")
            return None

        checkpoint_path = self.checkpoint_dir / f"checkpoint_v{version:06d}.pth"
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return None

        metadata = self.checkpoints_metadata[version]
        return checkpoint_path, metadata

    def list_checkpoints(self,
                        tags: Optional[List[str]] = None,
                        min_step: Optional[int] = None,
                        max_step: Optional[int] = None) -> List[Tuple[int, CheckpointMetadata]]:
        """
        List available checkpoints with optional filtering.

        Args:
            tags: Filter by tags
            min_step: Minimum step count
            max_step: Maximum step count

        Returns:
            List of (version, metadata) tuples sorted by version
        """
        results = []

        for version, metadata in self.checkpoints_metadata.items():
            # Filter by tags
            if tags and not any(tag in metadata.tags for tag in tags):
                continue

            # Filter by step range
            if min_step and metadata.step_count < min_step:
                continue
            if max_step and metadata.step_count > max_step:
                continue

            results.append((version, metadata))

        return sorted(results, key=lambda x: x[0])

    def delete_checkpoint(self, version: int) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            version: Version number to delete

        Returns:
            True if successfully deleted
        """
        if version not in self.checkpoints_metadata:
            logger.warning(f"Checkpoint version {version} not found in metadata")
            return False

        # Don't delete best checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_v{version:06d}.pth"
        if self.best_checkpoint_path and str(checkpoint_path) == self.best_checkpoint_path:
            logger.warning(f"Cannot delete best checkpoint v{version}")
            return False

        try:
            # Delete files
            state_path = self.checkpoint_dir / f"checkpoint_v{version:06d}.state.pth"
            for path in [checkpoint_path, state_path]:
                if path.exists():
                    path.unlink()

            # Remove from metadata
            del self.checkpoints_metadata[version]
            self._save_metadata()

            logger.info(f"Deleted checkpoint v{version}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete checkpoint v{version}: {e}")
            return False

    def _cleanup_old_checkpoints(self) -> None:
        """Apply retention policy to clean up old checkpoints."""
        if not self.checkpoints_metadata:
            return

        policy = self.retention_policy
        to_delete = set()

        # Get sorted checkpoints by version
        checkpoints = sorted(self.checkpoints_metadata.items())

        # Keep recent checkpoints
        recent_to_keep = set()
        if policy.keep_recent > 0:
            recent_versions = [v for v, _ in checkpoints[-policy.keep_recent:]]
            recent_to_keep.update(recent_versions)

        # Keep best checkpoints
        best_to_keep = set()
        if policy.keep_best > 0 and self.enable_best_tracking:
            # Sort by best metric score
            metric_checkpoints = [
                (v, m) for v, m in checkpoints
                if getattr(m, self.best_metric.replace('.', '_'), None) is not None
            ]

            if self.best_mode == "min":
                metric_checkpoints.sort(key=lambda x: getattr(x[1], self.best_metric.replace('.', '_')))
            else:
                metric_checkpoints.sort(key=lambda x: getattr(x[1], self.best_metric.replace('.', '_')), reverse=True)

            best_versions = [v for v, _ in metric_checkpoints[:policy.keep_best]]
            best_to_keep.update(best_versions)

        # Keep milestone checkpoints
        milestone_to_keep = set()
        if policy.keep_milestone_every:
            milestone_versions = [
                v for v, m in checkpoints
                if m.step_count % policy.keep_milestone_every == 0
            ]
            milestone_to_keep.update(milestone_versions)

        # Keep recent by days
        days_to_keep = set()
        if policy.keep_days:
            cutoff_time = time.time() - (policy.keep_days * 24 * 3600)
            recent_versions = [
                v for v, m in checkpoints
                if m.timestamp > cutoff_time
            ]
            days_to_keep.update(recent_versions)

        # Combine all keep sets
        keep_versions = recent_to_keep | best_to_keep | milestone_to_keep | days_to_keep

        # Mark others for deletion
        all_versions = set(v for v, _ in checkpoints)
        to_delete = all_versions - keep_versions

        # Delete marked checkpoints
        deleted_count = 0
        for version in to_delete:
            if self.delete_checkpoint(version):
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old checkpoints")

    def _extract_version_from_path(self, path: Path) -> Optional[int]:
        """Extract version number from checkpoint path."""
        try:
            # Format: checkpoint_v000123.pth
            name = path.stem
            if name.startswith('checkpoint_v'):
                return int(name.split('v')[1])
        except:
            pass
        return None

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get checkpoint storage statistics."""
        total_size = 0
        checkpoint_count = 0

        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_v??????.pth"):
            if checkpoint_file.exists() and not checkpoint_file.name.endswith('.state.pth'):
                total_size += checkpoint_file.stat().st_size
                checkpoint_count += 1

        total_size_gb = total_size / (1024 ** 3)

        # Get disk space
        free_space = shutil.disk_usage(self.checkpoint_dir).free / (1024 ** 3)

        return {
            'checkpoint_count': checkpoint_count,
            'total_size_gb': total_size_gb,
            'free_space_gb': free_space,
            'checkpoint_dir': str(self.checkpoint_dir),
            'best_checkpoint': self.best_checkpoint_path,
            'best_score': self.best_score if self.best_score != float('inf') and self.best_score != float('-inf') else None
        }

    def should_auto_save(self, current_step: int) -> bool:
        """Check if auto-save should trigger."""
        return (current_step - self.last_save_step) >= self.auto_save_every

    def get_checkpoint_info(self, version: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific checkpoint."""
        if version not in self.checkpoints_metadata:
            return None

        metadata = self.checkpoints_metadata[version]
        checkpoint_path = self.checkpoint_dir / f"checkpoint_v{version:06d}.pth"

        info = metadata.to_dict()
        info.update({
            'checkpoint_path': str(checkpoint_path),
            'exists': checkpoint_path.exists(),
            'is_best': str(checkpoint_path) == self.best_checkpoint_path
        })

        return info