"""
Training Loop Orchestration
===========================

Coordinates the complete training cycle: self-play → experience → training.
Manages checkpoints, validation tracking, and training metrics.

Features:
- Continuous self-play generation and model training
- Automatic checkpoint management and model evaluation
- Configurable training schedules and validation tracking
- Progress monitoring and statistics collection
- Graceful shutdown and recovery capabilities
"""

import os
import time
import signal
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
import json
import yaml
import numpy as np
from datetime import datetime, timedelta

# Import telemetry for logging configuration
from src.telemetry.logger import configure_logging, LogLevel

# Import training components
from src.training.trainer import AlphaZeroTrainer
from src.training.self_play import SelfPlayGameGenerator
from src.training.experience_buffer import MemoryMappedExperienceBuffer
from src.telemetry.metrics import MetricsCollector

# Import contracts
import sys
# Add specs directory to path to import contracts
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "specs" / "001-goal-create-spec"))
from contracts.training_api import GameResult, TrainingExample

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training loop orchestration."""

    # Model and game settings
    game_type: str = "gomoku"
    model_path: str = "models/latest.pth"

    # Self-play settings
    self_play_games_per_iteration: int = 50
    parallel_self_play_games: int = 4
    mcts_simulations: int = 800
    mcts_threads: int = 8
    batch_size_min: int = 32
    batch_size_max: int = 64
    inference_timeout_ms: float = 3.0

    # Training settings
    training_steps_per_iteration: int = 1000
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # Experience buffer settings
    experience_buffer_path: str = "training_data/experience_buffer"
    max_experience_examples: int = 1_000_000
    cache_size_mb: int = 512

    # Checkpoint and evaluation settings
    checkpoint_frequency: int = 5  # Save every N iterations
    evaluation_frequency: int = 10  # Evaluate every N iterations
    evaluation_games: int = 20
    max_checkpoints_to_keep: int = 10

    # Training loop control
    max_iterations: int = 1000
    target_training_time_hours: float = 48.0
    early_stopping_patience: int = 20

    # Validation settings
    validation_frequency: int = 5  # Validate every N iterations
    validation_games: int = 10

    # Performance monitoring
    target_games_per_hour: float = 200.0
    target_training_steps_per_minute: float = 60.0

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "training_logs"
    evaluation_dir: str = "evaluation_results"


@dataclass
class TrainingMetrics:
    """Training progress metrics and statistics."""

    iteration: int = 0
    total_games_generated: int = 0
    total_training_steps: int = 0
    total_training_examples: int = 0

    # Current iteration metrics
    games_this_iteration: int = 0
    training_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    learning_rate: float = 0.0

    # Performance metrics
    games_per_hour: float = 0.0
    training_steps_per_minute: float = 0.0
    memory_usage_mb: float = 0.0

    # Model strength metrics
    last_evaluation_win_rate: float = 0.0
    best_evaluation_win_rate: float = 0.0
    evaluation_history: List[float] = field(default_factory=list)

    # Timing
    iteration_start_time: float = 0.0
    total_training_time_hours: float = 0.0
    estimated_time_remaining_hours: float = 0.0


class TrainingLoop:
    """Main training loop orchestrator."""

    def __init__(self, config: TrainingConfig):
        """Initialize training loop.

        Args:
            config: Training configuration parameters
        """
        self.config = config
        self.metrics = TrainingMetrics()
        self.logger = logging.getLogger(__name__)

        # Create directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.evaluation_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.experience_buffer_path).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.self_play_generator: Optional[SelfPlayGameGenerator] = None
        self.experience_buffer: Optional[MemoryMappedExperienceBuffer] = None
        self.trainer: Optional[AlphaZeroTrainer] = None
        self.telemetry = MetricsCollector()

        # Control state
        self.running = False
        self.shutdown_requested = False
        self.current_iteration = 0
        self.training_start_time = time.time()

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="training_loop")

        # Setup signal handlers for graceful shutdown (only in main thread)
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info(f"Training loop initialized for {config.game_type}")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def _initialize_components(self) -> None:
        """Initialize training components lazily."""
        # Ensure model directory exists and create initial model if needed
        model_path = Path(self.config.model_path)
        model_dir = model_path.parent
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.trainer is None:
            self.logger.info("Initializing model trainer...")
            self.trainer = AlphaZeroTrainer(
                model_path=self.config.model_path,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                batch_size=self.config.batch_size,
                use_mixed_precision=True
            )

            # If model doesn't exist, trainer creates a new one
            # Save it immediately so self-play can use it
            if not model_path.exists():
                self.logger.info(f"Creating initial model at {self.config.model_path}")
                self.trainer.save_checkpoint(str(model_path))

        if self.self_play_generator is None:
            self.logger.info("Initializing self-play generator...")
            self.self_play_generator = SelfPlayGameGenerator(
                game_type=self.config.game_type,
                model_path=self.config.model_path,
                mcts_simulations=self.config.mcts_simulations,
                num_threads=self.config.mcts_threads,
                batch_size_min=self.config.batch_size_min,
                batch_size_max=self.config.batch_size_max,
                inference_timeout_ms=self.config.inference_timeout_ms
            )

        if self.experience_buffer is None:
            self.logger.info("Initializing experience buffer...")
            self.experience_buffer = MemoryMappedExperienceBuffer(
                buffer_path=Path(self.config.experience_buffer_path),
                max_examples=self.config.max_experience_examples,
                cache_size_mb=self.config.cache_size_mb
            )

    def run_training_loop(self) -> TrainingMetrics:
        """Run the complete training loop.

        Returns:
            TrainingMetrics: Final training statistics
        """
        self.logger.info("Starting training loop...")
        self.running = True
        self.training_start_time = time.time()

        try:
            # Initialize all components
            self._initialize_components()

            # Load existing progress if available
            self._load_training_state()

            # Main training loop
            for iteration in range(self.current_iteration, self.config.max_iterations):
                if self.shutdown_requested:
                    self.logger.info("Shutdown requested, stopping training loop")
                    break

                self.current_iteration = iteration
                self.metrics.iteration = iteration
                self.metrics.iteration_start_time = time.time()

                self.logger.info(f"Starting training iteration {iteration + 1}/{self.config.max_iterations}")

                # Run single training iteration
                iteration_metrics = self._run_training_iteration()

                # Update metrics
                self._update_metrics(iteration_metrics)

                # Save checkpoint if needed
                if (iteration + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(iteration + 1)

                # Run evaluation if needed
                if (iteration + 1) % self.config.evaluation_frequency == 0:
                    self._run_evaluation(iteration + 1)

                # Check early stopping conditions
                if self._should_stop_early():
                    self.logger.info("Early stopping criteria met, ending training")
                    break

                # Check time limit
                if self._has_reached_time_limit():
                    self.logger.info("Training time limit reached, ending training")
                    break

                # Log progress
                self._log_iteration_progress(iteration + 1)

                # Save training state
                self._save_training_state()

            # Final checkpoint and evaluation
            self._save_checkpoint(self.current_iteration + 1, is_final=True)
            self._run_evaluation(self.current_iteration + 1, is_final=True)

        except Exception as e:
            self.logger.error(f"Training loop failed: {e}", exc_info=True)
            raise
        finally:
            self.running = False
            self._cleanup()

        self.logger.info("Training loop completed successfully")
        return self.metrics

    def _run_training_iteration(self) -> Dict[str, Any]:
        """Run a single training iteration.

        Returns:
            dict: Iteration metrics
        """
        iteration_start = time.time()

        # Step 1: Generate self-play games
        self.logger.info(f"Generating {self.config.self_play_games_per_iteration} self-play games...")
        games = self._generate_self_play_games()

        # Step 2: Add games to experience buffer
        if games:
            self.logger.info(f"Adding {len(games)} games to experience buffer...")
            self.experience_buffer.add_games(games)

            # Update metrics
            total_examples = sum(len(game.examples) for game in games)
            self.metrics.total_games_generated += len(games)
            self.metrics.games_this_iteration = len(games)
            self.metrics.total_training_examples += total_examples

        # Step 3: Train model on experience buffer
        self.logger.info(f"Training model for {self.config.training_steps_per_iteration} steps...")
        training_metrics = self._train_model()

        # Step 4: Update model for self-play
        if self.self_play_generator:
            self.self_play_generator.update_model(self.config.model_path)

        iteration_time = time.time() - iteration_start

        return {
            'games_generated': len(games) if games else 0,
            'training_examples_added': total_examples if games else 0,
            'training_metrics': training_metrics,
            'iteration_time_seconds': iteration_time
        }

    def _generate_self_play_games(self) -> List[GameResult]:
        """Generate self-play games for training data.

        Returns:
            List of completed games
        """
        if not self.self_play_generator:
            return []

        games = []
        start_time = time.time()

        try:
            for game_result in self.self_play_generator.generate_games(
                num_games=self.config.self_play_games_per_iteration,
                parallel_games=self.config.parallel_self_play_games
            ):
                games.append(game_result)

                # Check for shutdown request during generation
                if self.shutdown_requested:
                    self.logger.info("Shutdown requested during self-play generation")
                    break

            # Update performance metrics
            generation_time = time.time() - start_time
            if generation_time > 0:
                self.metrics.games_per_hour = len(games) * 3600 / generation_time

        except Exception as e:
            self.logger.error(f"Self-play generation failed: {e}", exc_info=True)
            # Return partial results if any

        self.logger.info(f"Generated {len(games)} games in {time.time() - start_time:.1f} seconds")
        return games

    def _train_model(self) -> Dict[str, float]:
        """Train model on experience buffer data.

        Returns:
            dict: Training metrics
        """
        if not self.trainer or not self.experience_buffer:
            return {}

        start_time = time.time()
        total_metrics = {}
        steps_completed = 0

        try:
            # Create training iterator from experience buffer
            training_iterator = self.experience_buffer.create_training_iterator(
                batch_size=self.config.batch_size,
                shuffle_buffer_size=self.config.batch_size * 10
            )

            # Run training steps
            for step in range(self.config.training_steps_per_iteration):
                if self.shutdown_requested:
                    self.logger.info("Shutdown requested during training")
                    break

                try:
                    # Get next training batch
                    batch = next(training_iterator)
                    if not batch:
                        self.logger.warning("No training examples available, skipping training step")
                        continue

                    # Train on batch
                    step_metrics = self.trainer.train_step(batch)

                    # Accumulate metrics
                    for key, value in step_metrics.items():
                        if key not in total_metrics:
                            total_metrics[key] = []
                        total_metrics[key].append(value)

                    steps_completed += 1
                    self.metrics.total_training_steps += 1

                    # Log progress periodically
                    if (step + 1) % 100 == 0:
                        current_loss = step_metrics.get('total_loss', 0.0)
                        current_lr = step_metrics.get('learning_rate', 0.0)
                        self.logger.debug(f"Training step {step + 1}/{self.config.training_steps_per_iteration}: "
                                         f"loss={current_loss:.4f}, lr={current_lr:.6f}")

                except StopIteration:
                    self.logger.warning("Training iterator exhausted, ending training early")
                    break
                except Exception as e:
                    self.logger.error(f"Training step {step + 1} failed: {e}")
                    continue

            # Calculate average metrics
            averaged_metrics = {}
            for key, values in total_metrics.items():
                if values:
                    averaged_metrics[key] = np.mean(values)
                    averaged_metrics[f'{key}_std'] = np.std(values)

            # Update training metrics
            training_time = time.time() - start_time
            if training_time > 0:
                self.metrics.training_steps_per_minute = steps_completed * 60 / training_time

            # Update current iteration metrics
            self.metrics.training_loss = averaged_metrics.get('total_loss', 0.0)
            self.metrics.policy_loss = averaged_metrics.get('policy_loss', 0.0)
            self.metrics.value_loss = averaged_metrics.get('value_loss', 0.0)
            self.metrics.learning_rate = averaged_metrics.get('learning_rate', 0.0)

            self.logger.info(f"Completed {steps_completed} training steps in {training_time:.1f} seconds")

        except Exception as e:
            self.logger.error(f"Model training failed: {e}", exc_info=True)

        return averaged_metrics

    def _run_evaluation(self, iteration: int, is_final: bool = False) -> None:
        """Run model evaluation against baseline.

        Args:
            iteration: Current training iteration
            is_final: Whether this is the final evaluation
        """
        if not self.self_play_generator:
            return

        self.logger.info(f"Running model evaluation at iteration {iteration}...")

        try:
            # For now, run validation games against self (future: implement proper evaluation)
            validation_games = []

            # Generate validation games with current model
            for game_result in self.self_play_generator.generate_games(
                num_games=self.config.evaluation_games,
                parallel_games=2
            ):
                validation_games.append(game_result)

                if self.shutdown_requested:
                    break

            # Calculate evaluation metrics
            if validation_games:
                avg_game_length = np.mean([game.move_count for game in validation_games])
                avg_game_time = np.mean([game.game_length_seconds for game in validation_games])
                total_examples = sum(len(game.examples) for game in validation_games)

                # Simple strength metric: average game complexity
                win_rate = 0.5  # Placeholder - would compare against baseline in real evaluation

                # Update metrics
                self.metrics.last_evaluation_win_rate = win_rate
                if win_rate > self.metrics.best_evaluation_win_rate:
                    self.metrics.best_evaluation_win_rate = win_rate

                self.metrics.evaluation_history.append(win_rate)

                # Save evaluation results
                eval_results = {
                    'iteration': iteration,
                    'win_rate': win_rate,
                    'avg_game_length': avg_game_length,
                    'avg_game_time_seconds': avg_game_time,
                    'total_validation_games': len(validation_games),
                    'total_validation_examples': total_examples,
                    'timestamp': datetime.now().isoformat()
                }

                eval_file = Path(self.config.evaluation_dir) / f"evaluation_iter_{iteration:04d}.json"
                with open(eval_file, 'w') as f:
                    json.dump(eval_results, f, indent=2)

                self.logger.info(f"Evaluation completed: {len(validation_games)} games, "
                                f"avg length: {avg_game_length:.1f} moves, "
                                f"win rate: {win_rate:.3f}")

        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}", exc_info=True)

    def _should_stop_early(self) -> bool:
        """Check if early stopping criteria are met.

        Returns:
            bool: True if training should stop early
        """
        if len(self.metrics.evaluation_history) < self.config.early_stopping_patience:
            return False

        # Check if no improvement in recent evaluations
        recent_evals = self.metrics.evaluation_history[-self.config.early_stopping_patience:]
        best_recent = max(recent_evals)

        if best_recent < self.metrics.best_evaluation_win_rate * 0.95:  # 5% tolerance
            self.logger.info(f"No improvement in last {self.config.early_stopping_patience} evaluations")
            return True

        return False

    def _has_reached_time_limit(self) -> bool:
        """Check if training time limit has been reached.

        Returns:
            bool: True if time limit exceeded
        """
        elapsed_hours = (time.time() - self.training_start_time) / 3600
        self.metrics.total_training_time_hours = elapsed_hours

        if elapsed_hours >= self.config.target_training_time_hours:
            return True

        # Update estimated time remaining
        if self.current_iteration > 0:
            time_per_iteration = elapsed_hours / self.current_iteration
            remaining_iterations = self.config.max_iterations - self.current_iteration
            self.metrics.estimated_time_remaining_hours = time_per_iteration * remaining_iterations

        return False

    def _save_checkpoint(self, iteration: int, is_final: bool = False) -> None:
        """Save model checkpoint.

        Args:
            iteration: Current iteration number
            is_final: Whether this is the final checkpoint
        """
        if not self.trainer:
            return

        try:
            # Save model checkpoint
            if is_final:
                checkpoint_path = Path(self.config.checkpoint_dir) / "final_model.pth"
            else:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"model_iter_{iteration:04d}.pth"

            self.trainer.save_checkpoint(str(checkpoint_path))

            # Update latest model path
            latest_path = Path(self.config.checkpoint_dir) / "latest.pth"
            if checkpoint_path != latest_path:
                # Copy to latest
                import shutil
                shutil.copy2(checkpoint_path, latest_path)
                self.config.model_path = str(latest_path)

            # Clean up old checkpoints
            self._cleanup_old_checkpoints()

            self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to save disk space."""
        try:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            model_files = list(checkpoint_dir.glob("model_iter_*.pth"))

            if len(model_files) > self.config.max_checkpoints_to_keep:
                # Sort by iteration number and keep only the most recent
                model_files.sort(key=lambda p: p.name)
                files_to_remove = model_files[:-self.config.max_checkpoints_to_keep]

                for file_path in files_to_remove:
                    file_path.unlink()
                    # Also remove corresponding state files
                    state_path = file_path.with_suffix('.state.pth')
                    if state_path.exists():
                        state_path.unlink()

                self.logger.info(f"Cleaned up {len(files_to_remove)} old checkpoints")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")

    def _update_metrics(self, iteration_metrics: Dict[str, Any]) -> None:
        """Update training metrics with iteration results.

        Args:
            iteration_metrics: Metrics from current iteration
        """
        # Update memory usage
        try:
            import psutil
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            pass  # psutil not available

        # Update performance metrics from iteration
        if 'iteration_time_seconds' in iteration_metrics:
            iteration_time = iteration_metrics['iteration_time_seconds']
            if iteration_time > 0:
                # Calculate games per hour for this iteration
                games_generated = iteration_metrics.get('games_generated', 0)
                if games_generated > 0:
                    self.metrics.games_per_hour = games_generated * 3600 / iteration_time

    def _log_iteration_progress(self, iteration: int) -> None:
        """Log training progress for current iteration.

        Args:
            iteration: Current iteration number
        """
        elapsed_time = time.time() - self.training_start_time

        self.logger.info(
            f"Iteration {iteration} completed | "
            f"Games: {self.metrics.total_games_generated} total "
            f"({self.metrics.games_per_hour:.1f}/hour) | "
            f"Training steps: {self.metrics.total_training_steps} | "
            f"Loss: {self.metrics.training_loss:.4f} | "
            f"LR: {self.metrics.learning_rate:.6f} | "
            f"Time: {elapsed_time / 3600:.1f}h | "
            f"Memory: {self.metrics.memory_usage_mb:.1f}MB"
        )

    def _save_training_state(self) -> None:
        """Save current training state for recovery."""
        try:
            state = {
                'current_iteration': self.current_iteration,
                'metrics': {
                    'iteration': self.metrics.iteration,
                    'total_games_generated': self.metrics.total_games_generated,
                    'total_training_steps': self.metrics.total_training_steps,
                    'total_training_examples': self.metrics.total_training_examples,
                    'evaluation_history': self.metrics.evaluation_history,
                    'best_evaluation_win_rate': self.metrics.best_evaluation_win_rate,
                    'total_training_time_hours': self.metrics.total_training_time_hours
                },
                'config': self.config.__dict__,
                'timestamp': datetime.now().isoformat()
            }

            state_file = Path(self.config.log_dir) / "training_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save training state: {e}")

    def _load_training_state(self) -> None:
        """Load existing training state for recovery."""
        try:
            state_file = Path(self.config.log_dir) / "training_state.json"
            if not state_file.exists():
                return

            with open(state_file, 'r') as f:
                state = json.load(f)

            # Restore iteration progress
            self.current_iteration = state.get('current_iteration', 0)

            # Restore metrics
            if 'metrics' in state:
                metrics_data = state['metrics']
                self.metrics.iteration = metrics_data.get('iteration', 0)
                self.metrics.total_games_generated = metrics_data.get('total_games_generated', 0)
                self.metrics.total_training_steps = metrics_data.get('total_training_steps', 0)
                self.metrics.total_training_examples = metrics_data.get('total_training_examples', 0)
                self.metrics.evaluation_history = metrics_data.get('evaluation_history', [])
                self.metrics.best_evaluation_win_rate = metrics_data.get('best_evaluation_win_rate', 0.0)
                self.metrics.total_training_time_hours = metrics_data.get('total_training_time_hours', 0.0)

            self.logger.info(f"Restored training state from iteration {self.current_iteration}")

        except Exception as e:
            self.logger.error(f"Failed to load training state: {e}")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the training loop gracefully.

        Args:
            timeout: Maximum time to wait for components to stop
        """
        self.logger.info("Stopping training loop...")
        self.shutdown_requested = True
        self.running = False
        self._cleanup()

    def _cleanup(self) -> None:
        """Cleanup resources and shutdown components."""
        self.logger.info("Cleaning up training loop resources...")

        try:
            # Shutdown self-play generator
            if self.self_play_generator:
                self.self_play_generator.shutdown()

            # Shutdown thread pool
            self.executor.shutdown(wait=True)

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics.

        Returns:
            dict: Complete training statistics
        """
        stats = {
            'iteration': self.metrics.iteration,
            'total_games_generated': self.metrics.total_games_generated,
            'total_training_steps': self.metrics.total_training_steps,
            'total_training_examples': self.metrics.total_training_examples,
            'total_training_time_hours': self.metrics.total_training_time_hours,
            'games_per_hour': self.metrics.games_per_hour,
            'training_steps_per_minute': self.metrics.training_steps_per_minute,
            'memory_usage_mb': self.metrics.memory_usage_mb,
            'last_evaluation_win_rate': self.metrics.last_evaluation_win_rate,
            'best_evaluation_win_rate': self.metrics.best_evaluation_win_rate,
            'evaluation_history': self.metrics.evaluation_history,
            'current_losses': {
                'training_loss': self.metrics.training_loss,
                'policy_loss': self.metrics.policy_loss,
                'value_loss': self.metrics.value_loss
            },
            'learning_rate': self.metrics.learning_rate,
            'estimated_time_remaining_hours': self.metrics.estimated_time_remaining_hours
        }

        # Add component statistics if available
        if self.experience_buffer:
            stats['experience_buffer'] = self.experience_buffer.get_stats()

        if self.trainer:
            stats['trainer'] = self.trainer.get_training_stats()

        if self.self_play_generator:
            stats['self_play'] = self.self_play_generator.get_statistics()

        return stats


def create_training_loop(config_dict: Dict[str, Any]) -> TrainingLoop:
    """Factory function to create training loop from configuration.

    Args:
        config_dict: Configuration parameters (supports both flat and nested YAML structure)

    Returns:
        TrainingLoop: Configured training loop instance
    """
    # Handle nested YAML config structure (mcts, neural_network, training, game, system)
    if 'training' in config_dict and isinstance(config_dict['training'], dict):
        # Extract values from nested structure
        training = config_dict.get('training', {})
        mcts = config_dict.get('mcts', {})
        game = config_dict.get('game', {})
        system = config_dict.get('system', {})

        flat_config = {
            # Game settings
            'game_type': game.get('game_type', 'gomoku'),
            'model_path': system.get('model_dir', 'models/latest.pth') + '/latest.pth',

            # Self-play settings
            'self_play_games_per_iteration': training.get('self_play_games_per_iteration', 50),
            'parallel_self_play_games': training.get('parallel_self_play_games', 4),
            'mcts_simulations': mcts.get('simulations', 800),
            'mcts_threads': mcts.get('threads', 8),
            'batch_size_min': mcts.get('batch_size_min', 32),
            'batch_size_max': mcts.get('batch_size_max', 64),
            'inference_timeout_ms': mcts.get('inference_timeout_ms', 3.0),

            # Training settings
            'training_steps_per_iteration': training.get('training_steps_per_iteration', 1000),
            'batch_size': training.get('batch_size', 512),
            'learning_rate': config_dict.get('neural_network', {}).get('learning_rate', 0.001),
            'weight_decay': config_dict.get('neural_network', {}).get('weight_decay', 1e-4),

            # Experience buffer
            'experience_buffer_path': system.get('data_dir', 'training_data/experience_buffer'),
            'max_experience_examples': training.get('experience_buffer_size', 1_000_000),
            'cache_size_mb': 512,

            # Checkpoints and evaluation
            'checkpoint_frequency': training.get('save_frequency', 5),
            'evaluation_frequency': training.get('evaluation_frequency', 10),
            'evaluation_games': training.get('evaluation_games', 20),
            'max_checkpoints_to_keep': training.get('max_checkpoints', 10),

            # Training loop control
            'max_iterations': 1000,
            'target_training_time_hours': 48.0,
            'early_stopping_patience': training.get('patience', 20),

            # Validation
            'validation_frequency': 5,
            'validation_games': 10,

            # Performance
            'target_games_per_hour': 200.0,
            'target_training_steps_per_minute': 60.0,

            # Paths
            'checkpoint_dir': system.get('checkpoint_dir', 'checkpoints'),
            'log_dir': 'training_logs',
            'evaluation_dir': system.get('results_dir', 'evaluation_results'),
        }
        config_dict = flat_config

    # Filter config_dict to only include valid TrainingConfig fields
    # This prevents TypeError from unknown kwargs when loading from YAML
    from dataclasses import fields
    valid_fields = {f.name for f in fields(TrainingConfig)}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}

    config = TrainingConfig(**filtered_config)
    return TrainingLoop(config)


def run_training_session(config_path: str) -> TrainingMetrics:
    """Run complete training session from configuration file.

    Args:
        config_path: Path to YAML or JSON configuration file

    Returns:
        TrainingMetrics: Final training results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)

    # Create and run training loop
    training_loop = create_training_loop(config_dict)
    return training_loop.run_training_loop()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run AlphaZero training loop")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training configuration JSON file")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Configure telemetry logging - disable JSON formatting for console
    configure_logging(
        level=LogLevel.WARNING,
        enable_console=True,
        enable_file=False,
        structured_format=False  # Disable JSON formatting for console
    )

    # Setup standard logging - only show WARNING and above for most loggers
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set specific loggers to INFO for training-related components only
    logging.getLogger('__main__').setLevel(getattr(logging, args.log_level))
    logging.getLogger('src.training.training_loop').setLevel(logging.INFO)
    logging.getLogger('src.training.trainer').setLevel(logging.INFO)
    logging.getLogger('src.training.self_play').setLevel(logging.INFO)
    logging.getLogger('src.training.experience_buffer').setLevel(logging.INFO)

    # Suppress verbose loggers completely
    logging.getLogger('device_manager').setLevel(logging.CRITICAL)
    logging.getLogger('InferenceWorker').setLevel(logging.CRITICAL)
    logging.getLogger('src.core.search_coordinator').setLevel(logging.CRITICAL)
    logging.getLogger('AlphaZeroMCTS').setLevel(logging.CRITICAL)

    # Run training
    try:
        final_metrics = run_training_session(args.config)
        print(f"Training completed successfully!")
        print(f"Total iterations: {final_metrics.iteration}")
        print(f"Total games generated: {final_metrics.total_games_generated}")
        print(f"Total training time: {final_metrics.total_training_time_hours:.1f} hours")
        print(f"Best evaluation win rate: {final_metrics.best_evaluation_win_rate:.3f}")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        raise