"""
Pytest fixtures for benchmark harness.

Provides reusable fixtures for performance testing according to spec.md v2.0.
"""

from dataclasses import dataclass
from typing import Optional
import pytest
import numpy as np


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    # Game configuration
    game: str = "gomoku"  # gomoku, chess, go
    board_size: int = 15  # 15 for gomoku, 8 for chess, 19 for go

    # MCTS parameters
    num_simulations: int = 10000  # Total simulations to run
    num_threads: int = 4  # MCTS worker threads
    c_puct: float = 1.5  # PUCT exploration constant
    temperature: float = 1.0  # Policy temperature
    dirichlet_alpha: float = 0.3  # Root noise alpha
    dirichlet_epsilon: float = 0.25  # Root noise epsilon
    virtual_loss: float = 1.0  # Virtual loss magnitude

    # Inference batching (spec.md G3, G4)
    batch_size: int = 64  # Target batch size (32-64 optimal)
    batch_timeout_ms: float = 1.0  # Timeout milliseconds (≤3ms)

    # Neural network
    model_path: Optional[str] = None  # Path to .pth model (None = random)
    fp16_enabled: bool = True  # Mixed precision (validated in Phase 5)

    # Feature flags (T003 infrastructure)
    openmp_enabled: bool = True  # OpenMP parallelization (Phase 1 T004-T006)
    state_pooling_enabled: bool = False  # State reuse (Phase 1 T007-T009)
    condition_vars_enabled: bool = False  # CV synchronization (Phase 1 T010-T011)
    node_allocator_optimized: bool = False  # Arena allocation (Phase 1 T012-T013)
    nn_cache_enabled: bool = False  # NN-eval cache (Phase 3 T018-T021)
    root_preexpansion_enabled: bool = True  # Root pre-expansion (T003)
    busy_edge_masking_enabled: bool = True  # Busy-edge masking (T002)

    # Reproducibility
    seed: int = 42  # Random seed for reproducibility

    # Runtime environment
    omp_num_threads: Optional[int] = None  # OMP_NUM_THREADS override
    cuda_device: int = 0  # CUDA device index

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "game": self.game,
            "board_size": self.board_size,
            "num_simulations": self.num_simulations,
            "num_threads": self.num_threads,
            "c_puct": self.c_puct,
            "temperature": self.temperature,
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_epsilon": self.dirichlet_epsilon,
            "virtual_loss": self.virtual_loss,
            "batch_size": self.batch_size,
            "batch_timeout_ms": self.batch_timeout_ms,
            "model_path": self.model_path,
            "fp16_enabled": self.fp16_enabled,
            "openmp_enabled": self.openmp_enabled,
            "state_pooling_enabled": self.state_pooling_enabled,
            "condition_vars_enabled": self.condition_vars_enabled,
            "node_allocator_optimized": self.node_allocator_optimized,
            "nn_cache_enabled": self.nn_cache_enabled,
            "root_preexpansion_enabled": self.root_preexpansion_enabled,
            "busy_edge_masking_enabled": self.busy_edge_masking_enabled,
            "seed": self.seed,
            "omp_num_threads": self.omp_num_threads,
            "cuda_device": self.cuda_device,
        }


@pytest.fixture
def default_benchmark_config():
    """Default benchmark configuration for quick tests."""
    return BenchmarkConfig(
        game="gomoku",
        board_size=15,
        num_simulations=1000,  # Quick test
        num_threads=4,
        batch_size=64,
        batch_timeout_ms=1.0,
        seed=42,
    )


@pytest.fixture
def comprehensive_benchmark_config():
    """Comprehensive benchmark configuration for full validation (T014)."""
    return BenchmarkConfig(
        game="gomoku",
        board_size=15,
        num_simulations=10000,  # Full benchmark
        num_threads=4,
        batch_size=64,
        batch_timeout_ms=1.0,
        seed=42,
    )


@pytest.fixture
def thread_scaling_configs():
    """Generate configs for thread scaling benchmark (T016)."""
    thread_counts = [1, 2, 4, 6, 8, 10, 12]
    configs = []
    for threads in thread_counts:
        config = BenchmarkConfig(
            game="gomoku",
            board_size=15,
            num_simulations=10000,
            num_threads=threads,
            batch_size=64,
            batch_timeout_ms=1.0,
            seed=42,
        )
        configs.append(config)
    return configs


@pytest.fixture
def batch_size_configs():
    """Generate configs for batch size tuning."""
    batch_sizes = [16, 32, 48, 64, 96, 128]
    configs = []
    for batch_size in batch_sizes:
        config = BenchmarkConfig(
            game="gomoku",
            board_size=15,
            num_simulations=10000,
            num_threads=4,
            batch_size=batch_size,
            batch_timeout_ms=1.0,
            seed=42,
        )
        configs.append(config)
    return configs


@pytest.fixture
def timeout_configs():
    """Generate configs for timeout tuning."""
    timeouts = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    configs = []
    for timeout in timeouts:
        config = BenchmarkConfig(
            game="gomoku",
            board_size=15,
            num_simulations=10000,
            num_threads=4,
            batch_size=64,
            batch_timeout_ms=timeout,
            seed=42,
        )
        configs.append(config)
    return configs


@pytest.fixture
def ablation_configs():
    """Generate configs for ablation study (T015)."""
    # Baseline: all optimizations disabled
    baseline = BenchmarkConfig(
        game="gomoku",
        num_simulations=10000,
        num_threads=4,
        seed=42,
        openmp_enabled=False,
        state_pooling_enabled=False,
        condition_vars_enabled=False,
        node_allocator_optimized=False,
        nn_cache_enabled=False,
    )

    # OpenMP only
    openmp = BenchmarkConfig(
        game="gomoku",
        num_simulations=10000,
        num_threads=4,
        seed=42,
        openmp_enabled=True,
        state_pooling_enabled=False,
        condition_vars_enabled=False,
        node_allocator_optimized=False,
        nn_cache_enabled=False,
    )

    # OpenMP + State pooling
    state_pooling = BenchmarkConfig(
        game="gomoku",
        num_simulations=10000,
        num_threads=4,
        seed=42,
        openmp_enabled=True,
        state_pooling_enabled=True,
        condition_vars_enabled=False,
        node_allocator_optimized=False,
        nn_cache_enabled=False,
    )

    # OpenMP + State pooling + CV
    condition_vars = BenchmarkConfig(
        game="gomoku",
        num_simulations=10000,
        num_threads=4,
        seed=42,
        openmp_enabled=True,
        state_pooling_enabled=True,
        condition_vars_enabled=True,
        node_allocator_optimized=False,
        nn_cache_enabled=False,
    )

    # All CPU optimizations
    all_cpu = BenchmarkConfig(
        game="gomoku",
        num_simulations=10000,
        num_threads=4,
        seed=42,
        openmp_enabled=True,
        state_pooling_enabled=True,
        condition_vars_enabled=True,
        node_allocator_optimized=True,
        nn_cache_enabled=False,
    )

    # All optimizations including cache
    all_optimizations = BenchmarkConfig(
        game="gomoku",
        num_simulations=10000,
        num_threads=4,
        seed=42,
        openmp_enabled=True,
        state_pooling_enabled=True,
        condition_vars_enabled=True,
        node_allocator_optimized=True,
        nn_cache_enabled=True,
    )

    return {
        "baseline": baseline,
        "openmp": openmp,
        "state_pooling": state_pooling,
        "condition_vars": condition_vars,
        "all_cpu": all_cpu,
        "all_optimizations": all_optimizations,
    }


@pytest.fixture
def multi_game_configs():
    """Generate configs for multi-game validation."""
    games = [
        ("gomoku", 15),
        ("chess", 8),
        ("go", 19),
    ]
    configs = []
    for game, board_size in games:
        config = BenchmarkConfig(
            game=game,
            board_size=board_size,
            num_simulations=10000,
            num_threads=4,
            batch_size=64,
            batch_timeout_ms=1.0,
            seed=42,
        )
        configs.append(config)
    return configs


@pytest.fixture
def sample_telemetry_data():
    """Sample telemetry data for testing statistics calculation."""
    from tests.performance.telemetry import Telemetry

    # Generate synthetic telemetry with realistic values
    telemetry_list = []
    for i in range(10):
        t = Telemetry(
            throughput=2000 + i * 50 + np.random.normal(0, 20),  # ~2000-2500 sims/sec
            gpu_util_percent=65 + i * 0.5 + np.random.normal(0, 2),  # ~65-70%
            avg_batch_size=60 + np.random.normal(0, 3),  # ~60 batch size
            thread_idle_percent=60 + np.random.normal(0, 5),  # ~60% idle (baseline)
            num_threads=4,
            memory_peak_mb=250 + np.random.normal(0, 10),  # ~250MB
            tree_size_nodes=500000,
        )
        t.compute_derived_metrics()
        telemetry_list.append(t)

    return telemetry_list
