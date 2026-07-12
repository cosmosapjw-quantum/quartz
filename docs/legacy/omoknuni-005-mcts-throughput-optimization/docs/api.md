# AlphaZero Engine API Reference

**Version:** 1.0
**Last Updated:** 2025-09-25
**Documentation Level:** Complete

This document provides comprehensive API reference for the AlphaZero engine, including all public interfaces, parameters, and usage examples.

---

## Table of Contents

1. [MCTS Engine API](#mcts-engine-api)
2. [Neural Network Inference API](#neural-network-inference-api)
3. [Training Pipeline API](#training-pipeline-api)
4. [Game Interface API](#game-interface-api)
5. [Configuration API](#configuration-api)
6. [Telemetry & Monitoring API](#telemetry--monitoring-api)
7. [Usage Examples](#usage-examples)
8. [Error Handling](#error-handling)

---

## MCTS Engine API

The Monte Carlo Tree Search engine provides the core search functionality for board game AI.

### GameState Interface

Abstract base class that must be implemented for each game type.

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

class GameState(ABC):
    """Abstract game state interface."""

    @abstractmethod
    def apply_move_inplace(self, action: int) -> None:
        """Apply move directly to current state (no copy).

        Args:
            action (int): Integer action in game's action space

        Raises:
            ValueError: If action is illegal in current position
        """

    @abstractmethod
    def get_legal_moves(self) -> np.ndarray:
        """Get boolean mask of legal moves.

        Returns:
            np.ndarray: Boolean array where True indicates legal move
        """

    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if game is in terminal state.

        Returns:
            bool: True if game is finished
        """

    @abstractmethod
    def get_terminal_value(self) -> float:
        """Get terminal value from current player's perspective.

        Returns:
            float: Value in [-1.0, 1.0], +1.0 = win, 0.0 = draw, -1.0 = loss

        Raises:
            ValueError: If called on non-terminal state
        """

    @abstractmethod
    def copy(self) -> 'GameState':
        """Create deep copy of current state.

        Returns:
            GameState: Independent copy of current state

        Raises:
            MemoryError: If insufficient memory for copy
        """

    @abstractmethod
    def get_canonical_form(self) -> 'GameState':
        """Get canonical representation (current player as player 1).

        Returns:
            GameState: Canonical form of current state
        """
```

**Usage Example:**

```python
# Implement for Gomoku game
class GomokuState(GameState):
    def __init__(self, board_size=15):
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1
        self.board_size = board_size

    def apply_move_inplace(self, action: int):
        row, col = divmod(action, self.board_size)
        if self.board[row, col] != 0:
            raise ValueError(f"Illegal move: position ({row}, {col}) occupied")
        self.board[row, col] = self.current_player
        self.current_player *= -1

    def get_legal_moves(self) -> np.ndarray:
        return (self.board == 0).flatten()
```

### MCTSEngine Class

Core MCTS search implementation with performance optimizations.

```python
class MCTSEngine:
    """High-performance MCTS engine with GPU inference integration."""

    def __init__(self,
                 inference_worker: InferenceWorker,
                 simulations: int = 800,
                 exploration_constant: float = 1.0,
                 threads: int = 8,
                 virtual_loss: float = 1.0):
        """Initialize MCTS engine.

        Args:
            inference_worker (InferenceWorker): Neural network inference worker
            simulations (int, optional): Number of MCTS simulations. Defaults to 800.
            exploration_constant (float, optional): PUCT exploration parameter. Defaults to 1.0.
            threads (int, optional): Number of search threads. Defaults to 8.
            virtual_loss (float, optional): Virtual loss magnitude for thread coordination. Defaults to 1.0.
        """

    def search(self,
               game_state: GameState,
               temperature: float = 1.0,
               add_noise: bool = False) -> Tuple[np.ndarray, float]:
        """Perform MCTS search from given game state.

        Args:
            game_state (GameState): Initial game state for search
            temperature (float, optional): Temperature for move selection. Defaults to 1.0.
            add_noise (bool, optional): Add Dirichlet noise for exploration. Defaults to False.

        Returns:
            Tuple[np.ndarray, float]: (action probabilities, estimated value)

        Raises:
            ValueError: If game_state is terminal or invalid
            RuntimeError: If inference worker fails
        """

    def get_action_probabilities(self,
                                visit_counts: np.ndarray,
                                temperature: float = 1.0) -> np.ndarray:
        """Convert visit counts to action probabilities.

        Args:
            visit_counts (np.ndarray): Visit counts for each action
            temperature (float, optional): Temperature parameter. Defaults to 1.0.

        Returns:
            np.ndarray: Probability distribution over actions
        """
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `simulations` | int | 800 | Number of MCTS simulations per move (target: 30k+ sims/sec) |
| `exploration_constant` | float | 1.0 | PUCT exploration constant (c_puct) |
| `threads` | int | 8 | Number of parallel search threads (optimal: 8-12) |
| `virtual_loss` | float | 1.0 | Virtual loss magnitude for thread coordination |
| `temperature` | float | 1.0 | Move selection temperature (1.0=exploration, 0.0=exploitation) |

---

## C++ MCTS Components API

High-performance C++ implementation of MCTS core components with Python bindings.

### SimulationRunner Class

C++ simulation runner that performs select → expand → backup pipeline at high speed.

```python
import mcts_py

class SimulationRunner:
    """High-performance C++ MCTS simulation runner."""

    def __init__(self,
                 tree: MCTSTree,
                 selector: PUCTSelector,
                 backup: BackupManager,
                 virtual_loss_manager: VirtualLossManager):
        """Initialize simulation runner with MCTS components.

        Args:
            tree (MCTSTree): Shared MCTS tree for all threads
            selector (PUCTSelector): PUCT-based child selection
            backup (BackupManager): Value backup with sign flipping
            virtual_loss_manager (VirtualLossManager): Thread coordination

        Example:
            tree = mcts_py.MCTSTree(capacity=10000)
            selector = mcts_py.create_puct_selector()
            backup = mcts_py.create_backup_manager(tree)
            vl_manager = mcts_py.create_test_virtual_loss_manager(tree)

            runner = mcts_py.SimulationRunner(tree, selector, backup, vl_manager)
        """

    def run_simulation(self,
                      root_state: IGameState,
                      root_index: int,
                      inference_callback: InferenceCallback) -> bool:
        """Run single MCTS simulation from root with GIL released.

        This method releases the GIL during C++ execution for maximum
        performance. The inference callback is invoked when leaf expansion
        requires neural network evaluation.

        Args:
            root_state (IGameState): Game state at root position
            root_index (int): Root node index in tree
            inference_callback (InferenceCallback): Callback for neural network inference

        Returns:
            bool: True if simulation completed successfully, False if clone fails

        Performance:
            - GIL released during C++ execution
            - Virtual loss applied during select, removed during backup
            - Reuses path buffer for memory efficiency
            - Target: 30,000+ simulations/second with neural network inference
        """
```

### PyInferenceCallback Class

Python/C++ bridge for neural network inference callbacks with automatic GIL management.

```python
import mcts_py

class PyInferenceCallback(InferenceCallback):
    """Wrapper for Python inference callable with GIL management.

    This class bridges Python neural network inference functions to C++ code.
    GIL is automatically acquired when calling Python, then released for C++
    execution. Supports both Python lists and numpy arrays for policy.

    The Python callable should have signature:
        def inference_fn(state: IGameState) -> Tuple[List[float], float]:
            # Neural network inference here
            policy = [0.1, 0.2, ...]  # Probability distribution
            value = 0.5                # Position evaluation [-1, 1]
            return (policy, value)
    """

    def __init__(self, python_fn: callable):
        """Construct callback with Python callable.

        Args:
            python_fn (callable): Python function that takes IGameState
                                 and returns (policy, value) tuple

        Raises:
            ValueError: If python_fn is not callable

        Example:
            def my_inference(state):
                action_space = state.get_action_space_size()
                policy = [1.0 / action_space] * action_space
                value = 0.5
                return (policy, value)

            callback = mcts_py.PyInferenceCallback(my_inference)
        """

    def request_inference(self, state: IGameState) -> Tuple[List[float], float]:
        """Request neural network inference for game state.

        This method is called from C++ during leaf expansion. GIL is
        automatically acquired, Python callable is invoked, and results
        are converted back to C++ types.

        Args:
            state (IGameState): Game state to evaluate

        Returns:
            Tuple[List[float], float]: (policy vector, value scalar)

        Raises:
            RuntimeError: If Python callable fails or returns invalid data

        Supported Policy Formats:
            - Python list: [0.1, 0.2, 0.3, ...]
            - NumPy array: np.array([0.1, 0.2, 0.3, ...])
        """
```

**Integration Example:**

```python
import mcts_py
import alphazero_py
import numpy as np

# Create MCTS components
tree = mcts_py.MCTSTree(10000)
selector = mcts_py.create_puct_selector()
backup = mcts_py.create_backup_manager(tree)
vl_manager = mcts_py.create_test_virtual_loss_manager(tree)

# Create simulation runner
runner = mcts_py.SimulationRunner(tree, selector, backup, vl_manager)

# Create game state
game = alphazero_py.GomokuState(board_size=15)

# Add root node
root = tree.add_root_node(prior=0.5, player=0)

# Define inference callback
def neural_net_inference(state):
    action_space = state.get_action_space_size()
    # In production, this would call your neural network
    policy = np.ones(action_space, dtype=np.float32) / action_space
    value = 0.0
    return (policy, value)

# Wrap Python function for C++
callback = mcts_py.PyInferenceCallback(neural_net_inference)

# Run simulations (GIL released for C++ execution)
for _ in range(800):
    success = runner.run_simulation(game, root, callback)
    if not success:
        print("Simulation failed (tree full or clone error)")
        break

# Get visit counts from tree
visits = [tree.get_visit_count(child) for child in tree.get_children(root)]
print(f"Visit distribution: {visits}")
```

**Performance Characteristics:**

| Component | Performance | Notes |
|-----------|-------------|-------|
| `SimulationRunner` | 30,000+ sims/sec | Including neural network inference |
| `PyInferenceCallback` | GIL managed | Automatic acquire/release for optimal performance |
| Type Conversions | Zero-copy | NumPy arrays converted via pybind11 |
| Thread Safety | Fully thread-safe | Virtual loss coordination for parallel simulations |
| Memory Usage | <1GB | For typical tree sizes (10M nodes) |

---

## Neural Network Inference API

GPU-optimized neural network inference with micro-batching and mixed precision.

### InferenceWorker Class

```python
class InferenceWorker:
    """GPU inference worker with micro-batching optimization."""

    def __init__(self,
                 model_path: str,
                 device: str = 'cuda:0',
                 batch_size: int = 64,
                 timeout_ms: float = 3.0,
                 use_mixed_precision: bool = True):
        """Initialize inference worker.

        Args:
            model_path (str): Path to trained PyTorch model
            device (str, optional): Device for inference. Defaults to 'cuda:0'.
            batch_size (int, optional): Maximum batch size. Defaults to 64.
            timeout_ms (float, optional): Batch timeout in milliseconds. Defaults to 3.0.
            use_mixed_precision (bool, optional): Enable fp16 inference. Defaults to True.
        """

    def warmup(self, input_shape: Tuple[int, int, int]) -> None:
        """Warmup GPU with dummy inference calls.

        Args:
            input_shape (Tuple[int, int, int]): (channels, height, width) for input tensors
        """

    def inference_batch(self,
                       positions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
        """Process batch of positions through neural network.

        Args:
            positions (List[np.ndarray]): List of game position tensors

        Returns:
            Tuple[List[np.ndarray], List[float]]: (policy arrays, value estimates)

        Raises:
            InferenceError: If GPU inference fails
            ValueError: If positions have invalid shape
        """

    def start_worker_thread(self) -> None:
        """Start background worker thread for asynchronous inference."""

    def stop_worker_thread(self) -> None:
        """Stop background worker thread gracefully."""

    def submit_inference_request(self,
                               position: np.ndarray,
                               callback: callable) -> None:
        """Submit asynchronous inference request.

        Args:
            position (np.ndarray): Game position tensor
            callback (callable): Callback function for results
        """
```

**Performance Parameters:**

| Parameter | Type | Default | Target Performance |
|-----------|------|---------|-------------------|
| `batch_size` | int | 64 | Optimal for RTX 3060 Ti (8GB VRAM) |
| `timeout_ms` | float | 3.0 | Batch formation timeout (≤3ms target) |
| `use_mixed_precision` | bool | True | 2x memory efficiency with fp16 |

**Usage Example:**

```python
# Initialize inference worker
worker = InferenceWorker(
    model_path="models/gomoku_latest.pth",
    device="cuda:0",
    batch_size=64,
    timeout_ms=3.0,
    use_mixed_precision=True
)

# Warmup GPU
worker.warmup(input_shape=(36, 15, 15))  # Gomoku: 36 channels, 15x15 board

# Start async worker
worker.start_worker_thread()

# Submit inference request
def handle_result(policy, value):
    print(f"Policy shape: {policy.shape}, Value: {value}")

worker.submit_inference_request(position_tensor, handle_result)
```

---

## Training Pipeline API

Complete training pipeline for self-play generation, experience replay, and model optimization.

### TrainingLoop Class

```python
class TrainingLoop:
    """Complete training pipeline orchestration."""

    def __init__(self, config: AlphaZeroConfig):
        """Initialize training loop.

        Args:
            config (AlphaZeroConfig): Complete training configuration
        """

    def run_training_iteration(self) -> Dict[str, Any]:
        """Execute single training iteration: self-play → experience → training.

        Returns:
            Dict[str, Any]: Training metrics and statistics
        """

    def start_continuous_training(self) -> None:
        """Start continuous training loop with checkpoint management."""

    def stop_training(self) -> None:
        """Stop training gracefully and save final checkpoint."""
```

### SelfPlayGenerator Class

```python
class SelfPlayGenerator:
    """Generate training data through self-play games."""

    def __init__(self,
                 mcts_engine: MCTSEngine,
                 game_class: type,
                 temperature_schedule: List[Tuple[int, float]] = None):
        """Initialize self-play generator.

        Args:
            mcts_engine (MCTSEngine): MCTS engine for move generation
            game_class (type): Game class (e.g., GomokuState)
            temperature_schedule (List[Tuple[int, float]], optional): Temperature schedule by move
        """

    def generate_game(self,
                     add_noise: bool = True,
                     max_moves: int = 450) -> List[Dict[str, Any]]:
        """Generate single self-play game.

        Args:
            add_noise (bool, optional): Add Dirichlet noise for exploration. Defaults to True.
            max_moves (int, optional): Maximum game length. Defaults to 450.

        Returns:
            List[Dict[str, Any]]: List of training examples

        Raises:
            GameError: If game generation fails
            RuntimeError: If MCTS engine fails
        """

    def generate_games_parallel(self,
                               num_games: int,
                               num_workers: int = 4) -> List[List[Dict[str, Any]]]:
        """Generate multiple games in parallel.

        Args:
            num_games (int): Number of games to generate
            num_workers (int, optional): Parallel workers. Defaults to 4.

        Returns:
            List[List[Dict[str, Any]]]: All training examples
        """
```

### ExperienceBuffer Class

```python
class ExperienceBuffer:
    """Memory-mapped experience buffer with efficient sampling."""

    def __init__(self,
                 max_size: int = 1000000,
                 cache_size_mb: int = 512,
                 storage_path: str = "training_data"):
        """Initialize experience buffer.

        Args:
            max_size (int, optional): Maximum number of examples. Defaults to 1000000.
            cache_size_mb (int, optional): LRU cache size in MB. Defaults to 512.
            storage_path (str, optional): Directory for persistent storage. Defaults to "training_data".
        """

    def add_examples(self, examples: List[Dict[str, Any]]) -> None:
        """Add training examples to buffer.

        Args:
            examples (List[Dict[str, Any]]): Training examples from self-play
        """

    def sample_batch(self,
                    batch_size: int,
                    game_balance: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Sample balanced batch for training.

        Args:
            batch_size (int): Size of training batch
            game_balance (Dict[str, float], optional): Game type ratios. Defaults to None.

        Returns:
            List[Dict[str, Any]]: Balanced batch of examples
        """
```

---

## Game Interface API

Unified interface for different board games (Gomoku, Chess, Go).

### Game Adapter Interface

```python
class GameAdapter:
    """Unified interface for all board games."""

    @staticmethod
    def create_game(game_type: str, **kwargs) -> GameState:
        """Factory method to create game instance.

        Args:
            game_type (str): Game type ('gomoku', 'chess', 'go')
            **kwargs: Game-specific parameters

        Returns:
            GameState: Initialized game state
        """

    @staticmethod
    def get_feature_planes(game_state: GameState) -> np.ndarray:
        """Extract neural network input features.

        Args:
            game_state (GameState): Current game state

        Returns:
            np.ndarray: Feature tensor for neural network
        """

    @staticmethod
    def get_action_size(game_type: str) -> int:
        """Get action space size for game.

        Args:
            game_type (str): Game type

        Returns:
            int: Number of possible actions
        """
```

### Game-Specific Parameters

| Game | Action Space | Feature Planes | Board Size |
|------|--------------|----------------|------------|
| Gomoku | 225 (15×15) | 36 channels | 15×15 |
| Chess | 4096 (64×64) | 30 channels | 8×8 |
| Go | 361 (19×19) | 25 channels | 9×9 to 19×19 |

**Usage Example:**

```python
# Create different game types
gomoku = GameAdapter.create_game('gomoku', board_size=15)
chess = GameAdapter.create_game('chess', variant='standard')
go = GameAdapter.create_game('go', board_size=19, rules='chinese')

# Extract features for neural network
features = GameAdapter.get_feature_planes(gomoku)
print(f"Gomoku features shape: {features.shape}")  # (36, 15, 15)

# Get action space size
action_size = GameAdapter.get_action_size('gomoku')
print(f"Gomoku action space: {action_size}")  # 225
```

---

## Configuration API

Type-safe configuration system with environment variable support.

### ConfigManager Class

```python
class ConfigManager:
    """Configuration management with validation and environment overrides."""

    def __init__(self, config_path: str = None):
        """Initialize configuration manager.

        Args:
            config_path (str, optional): Path to configuration file.
                                       Defaults to "config/default.yaml".
        """

    def load_config(self, config_path: str = None) -> AlphaZeroConfig:
        """Load configuration from file with environment overrides.

        Args:
            config_path (str, optional): Optional path override

        Returns:
            AlphaZeroConfig: Loaded and validated configuration

        Raises:
            ConfigurationError: If configuration is invalid
        """

    def save_config(self, config: AlphaZeroConfig, output_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            config (AlphaZeroConfig): Configuration to save
            output_path (str): Output file path
        """
```

### Configuration Structure

```python
@dataclass
class AlphaZeroConfig:
    """Complete AlphaZero engine configuration."""

    mcts: MCTSConfig
    neural_network: NeuralNetworkConfig
    training: TrainingConfig
    game: GameConfig
    system: SystemConfig
```

**Environment Variables:**

Use `ALPHAZERO_<SECTION>_<PARAMETER>` format:

```bash
# MCTS configuration
export ALPHAZERO_MCTS_SIMULATIONS=1600
export ALPHAZERO_MCTS_THREADS=12

# Neural network configuration
export ALPHAZERO_NEURAL_NETWORK_BATCH_SIZE_PREFERRED=128
export ALPHAZERO_NEURAL_NETWORK_USE_MIXED_PRECISION=true

# Training configuration
export ALPHAZERO_TRAINING_BATCH_SIZE=1024
export ALPHAZERO_TRAINING_SELF_PLAY_GAMES_PER_ITERATION=100
```

---

## Telemetry & Monitoring API

Performance monitoring and metrics collection with Prometheus compatibility.

### MetricsCollector Class

```python
class MetricsCollector:
    """Prometheus-compatible metrics collection."""

    def __init__(self, enable_detailed: bool = True):
        """Initialize metrics collector.

        Args:
            enable_detailed (bool, optional): Enable detailed metrics. Defaults to True.
        """

    def record_mcts_performance(self,
                               simulations_per_second: float,
                               avg_tree_size: int) -> None:
        """Record MCTS performance metrics.

        Args:
            simulations_per_second (float): Search performance
            avg_tree_size (int): Average tree size in nodes
        """

    def record_gpu_metrics(self,
                          utilization_percent: float,
                          memory_used_mb: int,
                          batch_size_avg: float) -> None:
        """Record GPU performance metrics.

        Args:
            utilization_percent (float): GPU utilization percentage
            memory_used_mb (int): GPU memory usage in MB
            batch_size_avg (float): Average batch size
        """

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary.

        Returns:
            Dict[str, Any]: All collected metrics
        """
```

### Key Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| `alphazero_simulations_per_second` | 30,000+ | MCTS search performance |
| `alphazero_gpu_utilization_percent` | 80-92% | GPU efficiency |
| `alphazero_memory_usage_gb` | <1GB | Tree memory footprint |
| `alphazero_games_generated_per_hour` | 200+ | Self-play generation rate |
| `alphazero_inference_batch_size_avg` | 32-64 | Batching efficiency |

---

## Usage Examples

### Complete Training Setup

```python
from src.utils.config import load_config
from src.training.training_loop import TrainingLoop
from src.neural.inference_worker import InferenceWorker
from src.core.mcts_engine import MCTSEngine

# Load configuration
config = load_config('config/production.yaml')

# Initialize inference worker
worker = InferenceWorker(
    model_path="models/gomoku_init.pth",
    batch_size=config.neural_network.batch_size_preferred,
    timeout_ms=config.mcts.inference_timeout_ms
)

# Initialize MCTS engine
mcts = MCTSEngine(
    inference_worker=worker,
    simulations=config.mcts.simulations,
    threads=config.mcts.threads
)

# Start training
training_loop = TrainingLoop(config)
training_loop.start_continuous_training()
```

### Single Game Analysis

```python
import numpy as np
from src.games.gomoku import GomokuState
from src.core.mcts_engine import MCTSEngine

# Create game state
game = GomokuState(board_size=15)

# Apply some moves
game.apply_move_inplace(112)  # Center move (7, 7)
game.apply_move_inplace(113)  # Adjacent move

# Get MCTS analysis
policy, value = mcts.search(game, temperature=0.1)

# Find best move
best_action = np.argmax(policy)
print(f"Best move: {divmod(best_action, 15)}")
print(f"Position value: {value:.3f}")
```

### Performance Monitoring

```python
from src.telemetry.metrics import MetricsCollector
import time

metrics = MetricsCollector()

# Benchmark MCTS performance
start_time = time.time()
for _ in range(100):
    policy, value = mcts.search(game)
end_time = time.time()

sims_per_sec = (100 * config.mcts.simulations) / (end_time - start_time)
metrics.record_mcts_performance(sims_per_sec, avg_tree_size=50000)

# Get performance summary
summary = metrics.get_metrics_summary()
print(f"Performance: {summary}")
```

---

## Error Handling

### Common Exceptions

```python
class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

class GameError(Exception):
    """Raised for game-related errors."""

class InferenceError(Exception):
    """Raised for neural network inference errors."""

class TrainingError(Exception):
    """Raised for training pipeline errors."""
```

### Error Handling Best Practices

```python
try:
    # Load configuration
    config = load_config('config/production.yaml')
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Fall back to default configuration
    config = load_config('config/default.yaml')

try:
    # Initialize game
    game = GameAdapter.create_game('gomoku')
    game.apply_move_inplace(invalid_move)
except GameError as e:
    print(f"Game error: {e}")
    # Handle illegal move
    legal_moves = game.get_legal_moves()
    valid_move = np.random.choice(np.where(legal_moves)[0])

try:
    # Neural network inference
    policy, value = worker.inference_batch([position])
except InferenceError as e:
    print(f"Inference error: {e}")
    # Fall back to CPU inference or random policy
    policy = np.ones(action_size) / action_size
    value = 0.0
```

### Debugging and Diagnostics

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system capabilities
def check_system():
    """Diagnostic function for system capabilities."""
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test configuration loading
    try:
        config = load_config()
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration error: {e}")

# Run diagnostics
check_system()
```

---

## Performance Optimization Tips

### MCTS Optimization

```python
# Optimal thread count for Ryzen 5900X
config.mcts.threads = 12

# Optimal batch parameters for RTX 3060 Ti
config.neural_network.batch_size_preferred = 128
config.mcts.inference_timeout_ms = 3.0

# Memory optimization
config.mcts.max_tree_size_mb = 2048
config.system.max_gpu_memory_fraction = 0.95
```

### Hardware-Specific Tuning

```python
# AMD Ryzen 5900X
os.environ['OMP_NUM_THREADS'] = '24'
config.neural_network.cpu_threads = 24

# NVIDIA RTX 3060 Ti
config.neural_network.use_mixed_precision = True
config.neural_network.use_tensorrt = True
config.system.max_gpu_memory_fraction = 0.95
```

---

## API Versioning and Compatibility

- **Current Version:** 1.0
- **Backward Compatibility:** Maintained for all 1.x versions
- **Breaking Changes:** Only in major version updates (2.0+)
- **Deprecation Policy:** 6-month notice for deprecated APIs

### Version Checking

```python
import src
print(f"AlphaZero Engine Version: {src.__version__}")

# Check API compatibility
if src.__version__.startswith('1.'):
    print("✅ API version 1.x compatible")
else:
    print("⚠️  Different API version detected")
```

---

**Document Version:** 1.0
**Last Updated:** 2025-09-25
**Next Review:** 2025-10-25

For operational procedures, see the [Operations Runbook](operations.md).
For training guidance, see the [Training Guide](training_guide.md).