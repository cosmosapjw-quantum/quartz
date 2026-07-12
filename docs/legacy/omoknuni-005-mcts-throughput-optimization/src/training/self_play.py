"""
Self-Play Game Generator
========================

Generates self-play games for training data with temperature scheduling,
Dirichlet noise injection, and proper game outcome determination.

Key features:
- Temperature-based move selection during self-play
- Dirichlet noise at root for exploration
- Position augmentation through game symmetries
- Integration with MCTS search coordinator
- Parallel game generation support
"""

import time
import uuid
import logging
import threading
from typing import List, Dict, Tuple, Optional, Iterator, Any, Callable
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json

# Import training API contracts
import sys
import os
# Add specs directory to path to import contracts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'specs', '001-goal-create-spec'))
from contracts.training_api import (
    SelfPlayGenerator, GameResult, TrainingExample
)

# Import core components
from src.core.search_coordinator import SearchCoordinator, SearchRequest, SearchResult
from src.games.game_state import create_game_state, GameStateWrapper
from src.neural.inference_worker import GPUInferenceWorker
from src.telemetry.metrics import MetricsCollector

# Game bindings
from src.utils.alphazero_py_import import require_alphazero_py

alphazero_py = require_alphazero_py()


@dataclass
class SelfPlayConfig:
    """Configuration for self-play generation."""

    game_type: str = "gomoku"
    mcts_simulations: int = 800
    temperature_schedule: List[Tuple[int, float]] = field(default_factory=lambda: [(30, 1.0), (1000, 0.1)])
    dirichlet_alpha: float = 0.3  # Gomoku default, varies by game
    dirichlet_weight: float = 0.25
    cpuct: float = 1.25
    add_dirichlet_noise: bool = True
    num_threads: int = 8
    max_game_length: int = 512  # Prevent infinite games
    save_positions_from_move: int = 0  # Start saving training data from move N
    rule_variant: str = "standard"


class SelfPlayGameGenerator(SelfPlayGenerator):
    """Self-play game generator implementation."""

    def __init__(self,
                 game_type: str,
                 model_path: str,
                 mcts_simulations: int = 800,
                 temperature_schedule: List[Tuple[int, float]] = None,
                 add_dirichlet_noise: bool = True,
                 num_threads: int = 8,
                 batch_size_min: int = 32,
                 batch_size_max: int = 64,
                 inference_timeout_ms: float = 3.0):
        """Initialize self-play generator.

        Args:
            game_type: Game to play ('gomoku', 'chess', 'go')
            model_path: Path to current neural network model
            mcts_simulations: MCTS simulations per move
            temperature_schedule: [(move_threshold, temperature), ...]
            add_dirichlet_noise: Add exploration noise at root
            num_threads: MCTS search threads
        """
        # Allow environment-driven rule variants for testing and deployment overrides
        variant_override = os.environ.get('SELF_PLAY_RULE_VARIANT')

        self.config = SelfPlayConfig(
            game_type=game_type,
            mcts_simulations=mcts_simulations,
            temperature_schedule=temperature_schedule or [(30, 1.0), (1000, 0.1)],
            add_dirichlet_noise=add_dirichlet_noise,
            num_threads=num_threads,
            rule_variant=variant_override or "standard"
        )

        self.model_path = model_path
        self.logger = logging.getLogger(__name__)

        # Store inference config parameters
        self.batch_size_min = batch_size_min
        self.batch_size_max = batch_size_max
        self.inference_timeout_ms = inference_timeout_ms

        self._game_state_kwargs: Dict[str, Any] = {}

        # Set game-specific parameters
        self._set_game_specific_params()

        # Initialize components (will be set up when first needed)
        self.inference_worker: Optional[GPUInferenceWorker] = None
        self.search_coordinator: Optional[SearchCoordinator] = None
        self.telemetry = MetricsCollector()

        # Statistics tracking
        self.games_generated = 0
        self.total_positions = 0
        self.generation_times = []

        self.logger.info(f"Self-play generator initialized for {game_type}")

    def _set_game_specific_params(self) -> None:
        """Set game-specific parameters like Dirichlet noise."""
        variant = getattr(self.config, 'rule_variant', 'standard') or 'standard'

        # Reset game state configuration for each variant update
        self._game_state_kwargs = {}

        if self.config.game_type == 'gomoku':
            variant_alphas = {
                'standard': 0.3,
                'renju': 0.15,
                'omok': 0.25,
            }
            self.config.dirichlet_alpha = variant_alphas.get(variant, 0.3)
            self.config.max_game_length = 225

            if variant == 'renju':
                self._game_state_kwargs['use_renju'] = True
            elif variant == 'omok':
                self._game_state_kwargs['use_omok'] = True
        elif self.config.game_type == 'chess':
            self.config.dirichlet_alpha = 0.2
            self.config.max_game_length = 512
            if variant == 'chess960':
                self._game_state_kwargs['chess960'] = True
        elif self.config.game_type == 'go':
            self.config.dirichlet_alpha = 0.03
            self.config.max_game_length = 722  # 19x19 + pass moves
            if variant in {'japanese', 'korean', 'chinese'}:
                self._game_state_kwargs['rule_set'] = variant
        else:
            # Fallback to Gomoku defaults for unknown game types
            self.config.dirichlet_alpha = 0.3
            self.config.max_game_length = 225

    def _ensure_components_initialized(self) -> None:
        """Lazy initialization of GPU inference and search coordinator."""
        if self.inference_worker is None:
            # Initialize GPU inference worker
            from src.neural.device_manager import DeviceManager
            device_manager = DeviceManager()
            device_info = device_manager.detect_device()

            if device_info.is_cuda_available:
                from src.neural.inference_worker import GPUInferenceWorker
                self.inference_worker = GPUInferenceWorker(
                    model_path=self.model_path,
                    batch_size=self.batch_size_max,
                    timeout_ms=self.inference_timeout_ms
                )
                self.inference_worker.start()
            else:
                # Fallback to CPU inference
                from src.neural.cpu_inference import CPUInferenceWorker
                self.inference_worker = CPUInferenceWorker(model_path=self.model_path)
                self.inference_worker.start()

            # Initialize search coordinator
            self.search_coordinator = SearchCoordinator(
                inference_worker=self.inference_worker,
                max_threads=self.config.num_threads
            )
            self.search_coordinator.start()

    def generate_game(self, game_id: str) -> GameResult:
        """Generate single self-play game.

        Args:
            game_id: Unique identifier for this game

        Returns:
            GameResult: Complete game with training examples
        """
        self._ensure_components_initialized()

        start_time = time.time()
        self.logger.debug(f"Starting self-play game {game_id}")

        # Create game state using real C++ bindings
        game_state: GameStateWrapper = create_game_state(
            self.config.game_type,
            **self._game_state_kwargs
        )

        # Track game data
        game_examples = []
        move_history = []

        move_count = 0

        try:
            while not self._is_game_terminal(game_state) and move_count < self.config.max_game_length:
                # Get current temperature
                temperature = self._get_temperature(move_count)

                # Perform MCTS search
                search_request = SearchRequest(
                    request_id=f"{game_id}_move_{move_count}",
                    game_state=game_state,
                    simulations=self.config.mcts_simulations,
                    temperature=temperature,
                    add_noise=self.config.add_dirichlet_noise and move_count < 30
                )

                # Submit search request with dynamic timeout based on simulations
                # Rule of thumb: ~0.05s per simulation with GPU batching
                # Add 50% buffer for queueing with parallel games
                timeout_per_move = max(60.0, self.config.mcts_simulations * 0.05 * 1.5)

                if move_count == 0:
                    self.logger.info(f"Game {game_id}: using {self.config.mcts_simulations} simulations, timeout={timeout_per_move:.1f}s per move")

                move_start = time.time()
                search_future = self.search_coordinator.submit_search(search_request)
                search_result = search_future.result(timeout=timeout_per_move)
                move_time = time.time() - move_start

                if move_count % 10 == 0:
                    self.logger.debug(f"Game {game_id}: Move {move_count} took {move_time:.2f}s")

                # Extract training data (if past warmup moves)
                if move_count >= self.config.save_positions_from_move:
                    training_example = self._create_training_example(
                        game_state=game_state,
                        policy=search_result.policy,
                        move_number=move_count,
                        game_id=game_id
                    )
                    game_examples.append(training_example)

                # Apply temperature-based move selection using MCTS policy
                move_action = self._select_move_with_temperature(
                    policy=search_result.policy,
                    temperature=temperature,
                    best_move=search_result.best_move
                )

                # Make the move (C++ wrapper returns new state)
                game_state = game_state.make_move(move_action)

                move_history.append(move_action)
                move_count += 1

                self.logger.debug(f"Game {game_id}: Move {move_count}, action {move_action}, temp {temperature:.2f}")

            # Determine game outcome
            game_outcome = self._determine_game_outcome(game_state)

            # Update training examples with final game value
            self._update_examples_with_outcome(game_examples, game_outcome, move_count)

            # Create game result
            game_result = GameResult(
                winner=game_outcome.get('winner'),
                move_count=move_count,
                game_length_seconds=time.time() - start_time,
                examples=game_examples,
                final_board=self._get_board_string(game_state),
                metadata={
                    'game_id': game_id,
                    'game_type': self.config.game_type,
                    'mcts_simulations': self.config.mcts_simulations,
                    'total_positions': len(game_examples),
                    'final_outcome': game_outcome,
                    'move_history': move_history[:50]  # Limit for storage
                }
            )

            # Update statistics
            self.games_generated += 1
            self.total_positions += len(game_examples)
            self.generation_times.append(time.time() - start_time)

            self.logger.info(f"Game {game_id} completed: {move_count} moves, "
                           f"{len(game_examples)} training examples, "
                           f"outcome: {game_outcome}")

            return game_result

        except Exception as e:
            import traceback
            self.logger.error(f"Error generating game {game_id}: {type(e).__name__}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def generate_games(self,
                      num_games: int,
                      parallel_games: int = 4) -> Iterator[GameResult]:
        """Generate multiple self-play games in parallel.

        Args:
            num_games: Total number of games to generate
            parallel_games: Number of concurrent games

        Yields:
            GameResult: Each completed game as it finishes
        """
        self.logger.info(f"Generating {num_games} self-play games with {parallel_games} parallel")

        with ThreadPoolExecutor(max_workers=parallel_games, thread_name_prefix="selfplay") as executor:
            # Submit all game generation tasks
            futures = []
            for i in range(num_games):
                game_id = f"selfplay_{uuid.uuid4().hex[:8]}_{i}"
                future = executor.submit(self.generate_game, game_id)
                futures.append(future)

            # Yield results as they complete
            for future in as_completed(futures):
                try:
                    game_result = future.result()
                    yield game_result
                except Exception as e:
                    import traceback
                    self.logger.error(f"Failed to generate game: {type(e).__name__}: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    continue

    def update_model(self, model_path: str) -> None:
        """Update neural network model for self-play.

        Args:
            model_path: Path to new model checkpoint
        """
        self.logger.info(f"Updating self-play model: {model_path}")
        self.model_path = model_path

        # If components are initialized, update them
        if self.inference_worker is not None:
            self.inference_worker.update_model(model_path)

    def get_statistics(self) -> Dict[str, Any]:
        """Get self-play generation statistics.

        Returns:
            Dictionary of statistics
        """
        avg_time = np.mean(self.generation_times) if self.generation_times else 0.0
        avg_positions = self.total_positions / max(self.games_generated, 1)

        return {
            'games_generated': self.games_generated,
            'total_positions': self.total_positions,
            'average_positions_per_game': avg_positions,
            'average_generation_time_seconds': avg_time,
            'games_per_hour': 3600 / avg_time if avg_time > 0 else 0.0
        }

    def shutdown(self) -> None:
        """Shutdown self-play generator and cleanup resources."""
        self.logger.info("Shutting down self-play generator")

        if self.search_coordinator:
            self.search_coordinator.stop()

        if self.inference_worker:
            self.inference_worker.stop()

    # Helper methods

    def _get_temperature(self, move_count: int) -> float:
        """Get temperature for current move based on schedule."""
        for move_threshold, temperature in self.config.temperature_schedule:
            if move_count < move_threshold:
                return temperature
        # Return last temperature if past all thresholds
        return self.config.temperature_schedule[-1][1]

    def _select_move_with_temperature(self, policy: np.ndarray, temperature: float,
                                      best_move: int) -> int:
        """Select a move using temperature-scaled MCTS policy."""
        if policy.ndim != 1:
            raise ValueError("Policy must be a 1D probability distribution")

        if temperature <= 1e-5:
            return int(best_move)

        support = np.flatnonzero(policy > 0.0)

        if support.size == 0:
            return int(best_move)

        adjusted_temperature = max(temperature, 1e-3)
        scaled = np.power(policy[support], 1.0 / adjusted_temperature)
        total = scaled.sum()

        if not np.isfinite(total) or total <= 0.0:
            return int(best_move)

        probs = scaled / total
        return int(np.random.choice(support, p=probs))

    def _create_training_example(self,
                                game_state: Any,
                                policy: np.ndarray,
                                move_number: int,
                                game_id: str) -> TrainingExample:
        """Create training example from current position."""
        if hasattr(game_state, 'get_features'):
            features = game_state.get_features()
        elif hasattr(game_state, 'get_tensor_representation'):
            features = game_state.get_tensor_representation()
        else:
            raise RuntimeError("Game state does not expose feature tensors")

        features_array = np.asarray(features, dtype=np.float32)

        return TrainingExample(
            state=features_array,
            policy=policy.copy(),
            value=0.0,  # Will be updated with final game outcome
            game_type=self.config.game_type,
            move_number=move_number,
            game_id=game_id
        )

    def _update_examples_with_outcome(self,
                                     examples: List[TrainingExample],
                                     game_outcome: Dict[str, Any],
                                     final_move_count: int) -> None:
        """Update training examples with final game outcome."""
        winner = game_outcome.get('winner')

        for i, example in enumerate(examples):
            # Calculate value from perspective of player who made the move
            player_to_move = i % 2  # Alternating players

            if winner is None:  # Draw
                example.value = 0.0
            elif winner == player_to_move:  # Win for current player
                example.value = 1.0
            else:  # Loss for current player
                example.value = -1.0

    def _determine_game_outcome(self, game_state: GameStateWrapper) -> Dict[str, Any]:
        """Determine the outcome of a completed game."""
        if not game_state.is_terminal():
            return {'winner': None, 'result': 'max_moves_reached'}

        result = game_state.cpp_state.get_game_result()

        if result == alphazero_py.GameResult.WIN_PLAYER1:
            return {'winner': 0, 'result': 'win_player1'}
        if result == alphazero_py.GameResult.WIN_PLAYER2:
            return {'winner': 1, 'result': 'win_player2'}
        if result == alphazero_py.GameResult.DRAW:
            return {'winner': None, 'result': 'draw'}

        raise RuntimeError(f"Unexpected game result: {result}")

    def _is_game_terminal(self, game_state: GameStateWrapper) -> bool:
        """Check if game is in terminal state."""
        return game_state.is_terminal()

    def _get_board_string(self, game_state: GameStateWrapper) -> str:
        """Get human-readable board representation."""
        return game_state.to_string()


# Factory functions and utilities

def create_self_play_generator(config: Dict[str, Any]) -> SelfPlayGameGenerator:
    """Factory function to create self-play generator from config.

    Args:
        config: Configuration dictionary

    Returns:
        Configured SelfPlayGameGenerator instance
    """
    return SelfPlayGameGenerator(
        game_type=config.get('game_type', 'gomoku'),
        model_path=config.get('model_path', 'models/latest.pth'),
        mcts_simulations=config.get('mcts_simulations', 800),
        temperature_schedule=config.get('temperature_schedule'),
        add_dirichlet_noise=config.get('add_dirichlet_noise', True),
        num_threads=config.get('num_threads', 8)
    )


def save_games_to_disk(games: List[GameResult], output_path: Path) -> None:
    """Save generated games to disk in JSON format.

    Args:
        games: List of completed games
        output_path: Directory to save games
    """
    output_path.mkdir(parents=True, exist_ok=True)

    for i, game in enumerate(games):
        game_file = output_path / f"game_{i:06d}.json"

        # Convert numpy arrays to lists for JSON serialization
        game_data = {
            'winner': game.winner,
            'move_count': game.move_count,
            'game_length_seconds': game.game_length_seconds,
            'final_board': game.final_board,
            'metadata': game.metadata,
            'examples': [
                {
                    'state': example.state.tolist(),
                    'policy': example.policy.tolist(),
                    'value': example.value,
                    'game_type': example.game_type,
                    'move_number': example.move_number,
                    'game_id': example.game_id
                }
                for example in game.examples
            ]
        }

        with open(game_file, 'w') as f:
            json.dump(game_data, f, indent=2)


def load_games_from_disk(input_path: Path) -> List[GameResult]:
    """Load games from disk.

    Args:
        input_path: Directory containing saved games

    Returns:
        List of loaded GameResult objects
    """
    games = []

    for game_file in sorted(input_path.glob("game_*.json")):
        with open(game_file, 'r') as f:
            game_data = json.load(f)

        # Convert back to numpy arrays
        examples = [
            TrainingExample(
                state=np.array(ex['state']),
                policy=np.array(ex['policy']),
                value=ex['value'],
                game_type=ex['game_type'],
                move_number=ex['move_number'],
                game_id=ex['game_id']
            )
            for ex in game_data['examples']
        ]

        game = GameResult(
            winner=game_data['winner'],
            move_count=game_data['move_count'],
            game_length_seconds=game_data['game_length_seconds'],
            examples=examples,
            final_board=game_data['final_board'],
            metadata=game_data['metadata']
        )

        games.append(game)

    return games
