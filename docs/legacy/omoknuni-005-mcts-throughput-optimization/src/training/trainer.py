"""
Neural Network Model Trainer Implementation
==========================================

Implements ModelTrainer contract with AdamW optimizer, cosine learning rate
scheduling, mixed precision training, and gradient clipping for training stability.

Features:
- AdamW optimizer with configurable weight decay
- Cosine annealing learning rate schedule with warm restarts
- Mixed precision training using PyTorch AMP for RTX 3060 Ti optimization
- Gradient clipping to prevent training instability
- Comprehensive training metrics and validation
- Support for all game types (Gomoku, Chess, Go)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import warnings

# Import contracts and model
import sys
sys.path.append('specs/001-goal-create-spec')
from contracts.training_api import ModelTrainer, TrainingExample

from src.neural.model import AlphaZeroNet, create_model_for_game

logger = logging.getLogger(__name__)


@dataclass
class StabilityConfig:
    """Configuration for training stability monitoring."""

    # NaN detection
    check_nan_frequency: int = 1  # Check every N steps
    nan_recovery_lr_factor: float = 0.1  # LR reduction on NaN recovery

    # Gradient monitoring
    gradient_norm_window: int = 50  # Window for gradient norm tracking
    gradient_explosion_threshold: float = 10.0  # Threshold for gradient explosion
    gradient_norm_patience: int = 20  # Steps to wait before intervention

    # Loss convergence tracking
    loss_window: int = 100  # Window for loss convergence analysis
    plateau_threshold: float = 1e-5  # Minimum improvement to avoid plateau
    plateau_patience: int = 50  # Steps to wait before early stopping
    divergence_threshold: float = 2.0  # Loss increase factor indicating divergence

    # Early stopping
    enable_early_stopping: bool = True
    min_improvement: float = 1e-4  # Minimum validation improvement
    early_stop_patience: int = 100  # Steps to wait before stopping

    # Recovery mechanisms
    enable_automatic_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_lr_decay: float = 0.5  # LR decay on recovery


class TrainingStabilityMonitor:
    """
    Comprehensive training stability monitoring and recovery system.

    Features:
    - NaN detection in gradients, parameters, and losses
    - Gradient norm monitoring and explosion detection
    - Loss convergence tracking and plateau detection
    - Early stopping based on validation metrics
    - Automatic recovery mechanisms
    """

    def __init__(self, config: StabilityConfig = None):
        """Initialize stability monitor with configuration."""
        self.config = config or StabilityConfig()
        self.step_count = 0
        self.recovery_count = 0

        # Gradient monitoring
        self.gradient_norms = deque(maxlen=self.config.gradient_norm_window)
        self.gradient_explosion_count = 0

        # Loss tracking
        self.train_losses = deque(maxlen=self.config.loss_window)
        self.val_losses = deque(maxlen=self.config.loss_window)
        self.best_val_loss = float('inf')
        self.no_improvement_count = 0

        # Stability flags
        self.is_stable = True
        self.last_nan_step = -1
        self.last_explosion_step = -1

        # Recovery state
        self.original_lr = None
        self.recovery_checkpoint = None

        logger.info(f"Training stability monitor initialized with config: {self.config}")

    def check_nan_values(self, model: nn.Module, loss: torch.Tensor,
                        gradients: Optional[List[torch.Tensor]] = None) -> bool:
        """
        Check for NaN values in model parameters, loss, and gradients.

        Args:
            model: PyTorch model to check
            loss: Current loss tensor
            gradients: Optional list of gradient tensors

        Returns:
            bool: True if NaN detected, False otherwise
        """
        # Check loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf detected in loss at step {self.step_count}: {loss.item()}")
            return True

        # Check model parameters
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.warning(f"NaN/Inf detected in parameter {name} at step {self.step_count}")
                return True

        # Check gradients
        if gradients:
            for i, grad in enumerate(gradients):
                if grad is not None and (torch.isnan(grad).any() or torch.isinf(grad).any()):
                    logger.warning(f"NaN/Inf detected in gradient {i} at step {self.step_count}")
                    return True

        # Check gradients from model if not provided
        if gradients is None:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.warning(f"NaN/Inf detected in gradient for {name} at step {self.step_count}")
                        return True

        return False

    def update_gradient_norms(self, model: nn.Module) -> float:
        """
        Update gradient norm tracking and detect explosions.

        Args:
            model: PyTorch model with gradients

        Returns:
            float: Current gradient norm
        """
        total_norm = 0.0
        param_count = 0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)

            # Check for gradient explosion
            if total_norm > self.config.gradient_explosion_threshold:
                self.gradient_explosion_count += 1
                if self.gradient_explosion_count > self.config.gradient_norm_patience:
                    logger.warning(f"Gradient explosion detected at step {self.step_count}: "
                                 f"norm={total_norm:.4f}, threshold={self.config.gradient_explosion_threshold}")
                    self.last_explosion_step = self.step_count
                    return total_norm
            else:
                self.gradient_explosion_count = 0

        return total_norm

    def update_loss_tracking(self, train_loss: float, val_loss: Optional[float] = None) -> bool:
        """
        Update loss tracking and detect convergence issues.

        Args:
            train_loss: Current training loss
            val_loss: Optional validation loss

        Returns:
            bool: True if early stopping should be triggered
        """
        self.train_losses.append(train_loss)

        if val_loss is not None:
            self.val_losses.append(val_loss)

            # Check for best validation loss improvement
            if val_loss < self.best_val_loss - self.config.min_improvement:
                self.best_val_loss = val_loss
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

        # Check for loss divergence
        if len(self.train_losses) >= 10:
            recent_avg = np.mean(list(self.train_losses)[-10:])
            older_avg = np.mean(list(self.train_losses)[-20:-10]) if len(self.train_losses) >= 20 else recent_avg

            if recent_avg > older_avg * self.config.divergence_threshold:
                logger.warning(f"Loss divergence detected at step {self.step_count}: "
                             f"recent={recent_avg:.4f}, older={older_avg:.4f}")
                return True

        # Check for early stopping
        if (self.config.enable_early_stopping and
            self.no_improvement_count >= self.config.early_stop_patience):
            logger.info(f"Early stopping triggered at step {self.step_count}: "
                       f"no improvement for {self.no_improvement_count} steps")
            return True

        return False

    def check_plateau(self) -> bool:
        """
        Check if training has plateaued.

        Returns:
            bool: True if plateau detected
        """
        if len(self.train_losses) < self.config.loss_window:
            return False

        losses = np.array(self.train_losses)

        # Calculate slope of recent losses
        steps = np.arange(len(losses))
        slope = np.polyfit(steps, losses, 1)[0]

        # Check if improvement rate is below threshold
        improvement_rate = abs(slope)
        if improvement_rate < self.config.plateau_threshold:
            logger.info(f"Training plateau detected at step {self.step_count}: "
                       f"improvement_rate={improvement_rate:.6f}")
            return True

        return False

    def attempt_recovery(self, optimizer: torch.optim.Optimizer,
                        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> bool:
        """
        Attempt automatic recovery from training instability.

        Args:
            optimizer: Training optimizer to modify
            scheduler: Optional learning rate scheduler

        Returns:
            bool: True if recovery was attempted
        """
        if not self.config.enable_automatic_recovery:
            return False

        if self.recovery_count >= self.config.max_recovery_attempts:
            logger.error(f"Maximum recovery attempts ({self.config.max_recovery_attempts}) reached")
            return False

        self.recovery_count += 1
        logger.info(f"Attempting training recovery #{self.recovery_count}")

        # Store original learning rate if not already stored
        if self.original_lr is None:
            self.original_lr = optimizer.param_groups[0]['lr']

        # Reduce learning rate
        new_lr = optimizer.param_groups[0]['lr'] * self.config.recovery_lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        logger.info(f"Reduced learning rate to {new_lr:.6f} for recovery")

        # Reset gradient explosion counter
        self.gradient_explosion_count = 0

        # Clear recent unstable history
        if len(self.gradient_norms) > 10:
            # Keep only the first half of gradient norms
            stable_norms = list(self.gradient_norms)[:len(self.gradient_norms)//2]
            self.gradient_norms.clear()
            self.gradient_norms.extend(stable_norms)

        return True

    def step(self, model: nn.Module, loss: torch.Tensor, optimizer: torch.optim.Optimizer,
             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
             val_loss: Optional[float] = None) -> Dict[str, Any]:
        """
        Main stability monitoring step.

        Args:
            model: PyTorch model
            loss: Current training loss
            optimizer: Training optimizer
            scheduler: Optional learning rate scheduler
            val_loss: Optional validation loss

        Returns:
            dict: Stability monitoring results and recommendations
        """
        self.step_count += 1
        results = {
            'step': self.step_count,
            'is_stable': True,
            'should_stop': False,
            'recovery_attempted': False,
            'warnings': []
        }

        # Check for NaN values
        if self.step_count % self.config.check_nan_frequency == 0:
            has_nan = self.check_nan_values(model, loss)
            if has_nan:
                results['is_stable'] = False
                results['warnings'].append('NaN detected')
                self.last_nan_step = self.step_count

                if self.attempt_recovery(optimizer, scheduler):
                    results['recovery_attempted'] = True
                else:
                    results['should_stop'] = True

        # Update gradient norm monitoring
        grad_norm = self.update_gradient_norms(model)
        results['gradient_norm'] = grad_norm

        # Update loss tracking
        should_stop = self.update_loss_tracking(loss.item(), val_loss)
        if should_stop:
            results['should_stop'] = True
            results['warnings'].append('Early stopping triggered')

        # Check for plateau
        if self.check_plateau():
            results['warnings'].append('Training plateau detected')

        # Update stability flag
        self.is_stable = results['is_stable']

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive stability monitoring statistics."""
        stats = {
            'step_count': self.step_count,
            'recovery_count': self.recovery_count,
            'is_stable': self.is_stable,
            'last_nan_step': self.last_nan_step,
            'last_explosion_step': self.last_explosion_step,
        }

        # Gradient statistics
        if self.gradient_norms:
            grad_norms = np.array(self.gradient_norms)
            stats.update({
                'gradient_norm_mean': np.mean(grad_norms),
                'gradient_norm_std': np.std(grad_norms),
                'gradient_norm_max': np.max(grad_norms),
                'gradient_explosion_count': self.gradient_explosion_count,
            })

        # Loss statistics
        if self.train_losses:
            train_losses = np.array(self.train_losses)
            stats.update({
                'train_loss_mean': np.mean(train_losses),
                'train_loss_std': np.std(train_losses),
            })

            # Only calculate trend if we have enough data points and no NaN values
            if len(train_losses) >= 2 and not np.any(np.isnan(train_losses)):
                try:
                    stats['train_loss_trend'] = np.polyfit(range(len(train_losses)), train_losses, 1)[0]
                except np.linalg.LinAlgError:
                    stats['train_loss_trend'] = 0.0
            else:
                stats['train_loss_trend'] = 0.0

        if self.val_losses:
            val_losses = np.array(self.val_losses)
            stats.update({
                'val_loss_mean': np.mean(val_losses),
                'val_loss_std': np.std(val_losses),
                'best_val_loss': self.best_val_loss,
                'no_improvement_count': self.no_improvement_count,
            })

        return stats


class AlphaZeroTrainer(ModelTrainer):
    """
    Neural network trainer for AlphaZero models.

    Implements the ModelTrainer contract with production-ready training features:
    - AdamW optimizer with weight decay regularization
    - Cosine annealing learning rate schedule
    - Mixed precision training for memory efficiency
    - Gradient clipping for training stability
    - Comprehensive metrics tracking
    """

    def __init__(self,
                 model_path: str,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 batch_size: int = 512,
                 use_mixed_precision: bool = True,
                 gradient_clip_norm: float = 1.0,
                 lr_schedule_t_max: int = 1000,
                 lr_min_ratio: float = 0.1,
                 stability_config: Optional[StabilityConfig] = None):
        """
        Initialize AlphaZero model trainer.

        Args:
            model_path: Path to model checkpoint to continue training
            learning_rate: Initial learning rate for AdamW optimizer
            weight_decay: L2 regularization strength
            batch_size: Training batch size (should fit in GPU memory)
            use_mixed_precision: Enable fp16 training with automatic mixed precision
            gradient_clip_norm: Maximum gradient norm for clipping (0 to disable)
            lr_schedule_t_max: Period for cosine annealing schedule
            lr_min_ratio: Minimum learning rate as ratio of initial LR
            stability_config: Configuration for training stability monitoring
        """
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision
        self.gradient_clip_norm = gradient_clip_norm
        self.lr_schedule_t_max = lr_schedule_t_max
        self.lr_min_ratio = lr_min_ratio

        # Device detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training on device: {self.device}")

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler - cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=lr_schedule_t_max,
            eta_min=learning_rate * lr_min_ratio
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if use_mixed_precision else None

        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.loss_history = deque(maxlen=1000)  # Keep last 1000 losses
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))

        # Training stability monitoring
        self.stability_monitor = TrainingStabilityMonitor(stability_config)

        logger.info(f"Trainer initialized - Model: {self.model.__class__.__name__}, "
                   f"Parameters: {self._count_parameters():,}, "
                   f"Mixed Precision: {use_mixed_precision}, "
                   f"Stability Monitoring: Enabled")

    def _load_model(self) -> AlphaZeroNet:
        """Load model from checkpoint or create new model."""
        model_path = Path(self.model_path)

        if model_path.exists():
            logger.info(f"Loading existing model from {model_path}")
            try:
                # Try loading full model first with weights_only=False for our trusted models
                model = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(model, AlphaZeroNet):
                    return model

                # If state_dict, determine game type and create model
                state_dict = model if isinstance(model, dict) else model.state_dict()
                game_type = self._detect_game_type_from_state_dict(state_dict)
                model = create_model_for_game(game_type)

                # Handle PolicyHead lazy initialization
                if 'policy_head.fc.weight' in state_dict:
                    # PolicyHead is already initialized in saved model - initialize ours too
                    input_shape = self._get_input_shape_for_game(game_type)
                    dummy_input = torch.zeros(1, *input_shape)
                    with torch.no_grad():
                        model(dummy_input)  # Force initialization

                model.load_state_dict(state_dict)
                logger.info(f"Loaded model for game type: {game_type}")
                return model

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Could not load model from {model_path}: {e}")
        else:
            # Create new model - default to Gomoku if no existing model
            logger.info("Creating new Gomoku model")
            return create_model_for_game('gomoku')

    def _detect_game_type_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Detect game type from model state dict input channels."""
        # Look for first conv layer to determine input channels
        for key, tensor in state_dict.items():
            if 'initial_conv' in key and 'weight' in key:
                input_channels = tensor.shape[1]
                if input_channels == 36:
                    return 'gomoku'
                elif input_channels == 30:
                    return 'chess'
                elif input_channels == 25:
                    return 'go'
                else:
                    logger.warning(f"Unknown input channels {input_channels}, defaulting to gomoku")
                    return 'gomoku'

        logger.warning("Could not detect game type from state dict, defaulting to gomoku")
        return 'gomoku'

    def _get_input_shape_for_game(self, game_type: str) -> Tuple[int, int, int]:
        """Get input shape for game type."""
        if game_type == "gomoku":
            return (36, 15, 15)
        elif game_type == "chess":
            return (30, 8, 8)
        elif game_type == "go":
            return (25, 19, 19)
        else:
            return (36, 15, 15)  # Default to Gomoku

    def _count_parameters(self) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _prepare_batch(self, batch: List[TrainingExample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert training examples to tensors.

        Args:
            batch: List of training examples

        Returns:
            tuple: (states, policies, values) as tensors
        """
        states = np.stack([example.state for example in batch])
        policies = np.stack([example.policy for example in batch])
        values = np.array([example.value for example in batch], dtype=np.float32)

        # Convert to tensors and move to device
        states_tensor = torch.from_numpy(states).float().to(self.device)
        policies_tensor = torch.from_numpy(policies).float().to(self.device)
        values_tensor = torch.from_numpy(values).float().to(self.device)

        return states_tensor, policies_tensor, values_tensor

    def _compute_loss(self,
                     policy_pred: torch.Tensor,
                     value_pred: torch.Tensor,
                     policy_target: torch.Tensor,
                     value_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training losses.

        Args:
            policy_pred: Predicted policy logits (batch_size, num_actions)
            value_pred: Predicted values (batch_size,)
            policy_target: Target policy distribution (batch_size, num_actions)
            value_target: Target values (batch_size,)

        Returns:
            tuple: (total_loss, metrics_dict)
        """
        # Policy loss: KL divergence between predicted and target distributions
        # Use log_softmax on predictions and batchmean reduction for proper KL divergence
        policy_loss = F.kl_div(F.log_softmax(policy_pred, dim=1), policy_target, reduction='batchmean')

        # Value loss: mean squared error
        value_loss = F.mse_loss(value_pred.squeeze(), value_target, reduction='mean')

        # Total loss: weighted combination
        total_loss = policy_loss + value_loss

        # Additional metrics
        with torch.no_grad():
            # Policy accuracy (top-1)
            policy_pred_classes = torch.argmax(policy_pred, dim=1)
            policy_target_classes = torch.argmax(policy_target, dim=1)
            policy_accuracy = (policy_pred_classes == policy_target_classes).float().mean()

            # Value MAE
            value_mae = F.l1_loss(value_pred.squeeze(), value_target, reduction='mean')

            metrics = {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'total_loss': total_loss.item(),
                'policy_accuracy': policy_accuracy.item(),
                'value_mae': value_mae.item(),
            }

        return total_loss, metrics

    def train_step(self, batch: List[TrainingExample]) -> Dict[str, float]:
        """
        Single training step on batch.

        Args:
            batch: Training examples

        Returns:
            dict: Training metrics including losses and learning rate
        """
        if len(batch) == 0:
            raise ValueError("Empty batch provided to train_step")

        self.model.train()
        start_time = time.time()

        # Prepare batch data
        states, policy_targets, value_targets = self._prepare_batch(batch)

        # Forward pass with mixed precision
        if self.use_mixed_precision and self.scaler is not None:
            with autocast():
                policy_pred, value_pred = self.model(states)
                loss, metrics = self._compute_loss(policy_pred, value_pred,
                                                 policy_targets, value_targets)

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training
            policy_pred, value_pred = self.model(states)
            loss, metrics = self._compute_loss(policy_pred, value_pred,
                                             policy_targets, value_targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            # Optimizer step
            self.optimizer.step()

        # Learning rate scheduling
        self.scheduler.step()

        # Update training state
        self.step_count += 1
        self.loss_history.append(loss.item())
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        # Add additional metrics
        metrics.update({
            'learning_rate': self.scheduler.get_last_lr()[0],
            'step_time': time.time() - start_time,
            'batch_size': len(batch),
            'step_count': self.step_count,
        })

        # Add gradient norm if available
        if self.gradient_clip_norm > 0:
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            metrics['gradient_norm'] = total_norm

        # Run stability monitoring
        stability_results = self.stability_monitor.step(
            model=self.model,
            loss=loss,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )

        # Add stability metrics
        metrics.update({
            'stability_is_stable': stability_results['is_stable'],
            'stability_should_stop': stability_results['should_stop'],
            'stability_recovery_attempted': stability_results['recovery_attempted'],
            'stability_gradient_norm': stability_results.get('gradient_norm', 0.0),
            'stability_warnings': len(stability_results['warnings']),
        })

        # Log stability warnings
        if stability_results['warnings']:
            logger.warning(f"Step {self.step_count}: Stability warnings: {stability_results['warnings']}")

        # Handle early stopping or recovery
        if stability_results['should_stop']:
            logger.error(f"Training should stop due to stability issues at step {self.step_count}")
            metrics['stability_stop_training'] = True

        if stability_results['recovery_attempted']:
            logger.info(f"Training recovery attempted at step {self.step_count}")

        return metrics

    def validate(self, validation_data: List[TrainingExample]) -> Dict[str, float]:
        """
        Validate model on held-out data.

        Args:
            validation_data: Examples for validation

        Returns:
            dict: Validation metrics
        """
        if len(validation_data) == 0:
            return {}

        self.model.eval()
        total_metrics = defaultdict(float)
        num_batches = 0

        with torch.no_grad():
            # Process validation data in batches
            for i in range(0, len(validation_data), self.batch_size):
                batch = validation_data[i:i + self.batch_size]
                states, policy_targets, value_targets = self._prepare_batch(batch)

                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        policy_pred, value_pred = self.model(states)
                        _, batch_metrics = self._compute_loss(policy_pred, value_pred,
                                                            policy_targets, value_targets)
                else:
                    policy_pred, value_pred = self.model(states)
                    _, batch_metrics = self._compute_loss(policy_pred, value_pred,
                                                        policy_targets, value_targets)

                # Accumulate metrics
                for key, value in batch_metrics.items():
                    total_metrics[key] += value
                num_batches += 1

        # Average metrics across batches
        avg_metrics = {f'val_{key}': value / num_batches
                      for key, value in total_metrics.items()}

        # Store validation metrics
        for key, value in avg_metrics.items():
            self.metrics_history[key].append(value)

        # Update stability monitoring with validation loss
        if 'val_total_loss' in avg_metrics:
            # Update the stability monitor step count to match trainer step count
            self.stability_monitor.step_count = self.step_count
            # Update loss tracking with validation loss
            self.stability_monitor.update_loss_tracking(
                self.loss_history[-1] if self.loss_history else 0.0,
                avg_metrics['val_total_loss']
            )

        return avg_metrics

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save model checkpoint.

        Args:
            checkpoint_path: Path for saved checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Save full model for easy loading
        torch.save(self.model, checkpoint_path)

        # Also save training state
        state_dict_path = checkpoint_path.with_suffix('.state.pth')
        training_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'loss_history': list(self.loss_history),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }
        torch.save(training_state, state_dict_path)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        state_dict_path = Path(checkpoint_path).with_suffix('.state.pth')

        if state_dict_path.exists():
            logger.info(f"Loading training state from {state_dict_path}")
            training_state = torch.load(state_dict_path, map_location=self.device, weights_only=False)

            self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
            self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
            self.step_count = training_state.get('step_count', 0)
            self.epoch_count = training_state.get('epoch_count', 0)
            self.loss_history = deque(training_state.get('loss_history', []), maxlen=1000)

            if self.scaler and training_state.get('scaler_state_dict'):
                self.scaler.load_state_dict(training_state['scaler_state_dict'])

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training progress statistics.

        Returns:
            dict: Training stats including iteration count, loss history
        """
        stats = {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'current_lr': self.scheduler.get_last_lr()[0],
            'total_parameters': self._count_parameters(),
            'device': str(self.device),
            'mixed_precision': self.use_mixed_precision,
        }

        # Add recent loss statistics
        if self.loss_history:
            recent_losses = list(self.loss_history)
            stats.update({
                'recent_loss_mean': np.mean(recent_losses),
                'recent_loss_std': np.std(recent_losses),
                'recent_loss_min': np.min(recent_losses),
                'recent_loss_max': np.max(recent_losses),
                'loss_history_length': len(recent_losses),
            })

        # Add metrics history statistics
        for metric_name, history in self.metrics_history.items():
            if history:
                recent_values = list(history)
                stats[f'{metric_name}_mean'] = np.mean(recent_values)
                stats[f'{metric_name}_recent'] = recent_values[-1] if recent_values else 0.0

        # Add stability monitoring statistics
        stability_stats = self.stability_monitor.get_statistics()
        for key, value in stability_stats.items():
            stats[f'stability_{key}'] = value

        return stats

    def reset_scheduler(self, t_max: Optional[int] = None) -> None:
        """
        Reset learning rate scheduler (useful for warm restarts).

        Args:
            t_max: New period for cosine annealing (None to keep current)
        """
        if t_max is not None:
            self.lr_schedule_t_max = t_max

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.lr_schedule_t_max,
            eta_min=self.learning_rate * self.lr_min_ratio
        )
        logger.info(f"Learning rate scheduler reset with T_max={self.lr_schedule_t_max}")