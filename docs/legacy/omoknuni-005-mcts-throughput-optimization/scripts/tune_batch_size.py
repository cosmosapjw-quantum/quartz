#!/usr/bin/env python3
"""
Batch Size Optimization Script
==============================

Systematically tests different batch sizes to find optimal configurations that
maximize GPU utilization while staying within VRAM constraints and maintaining
low latency for neural network inference.

This script performs parameter sweeps across batch sizes 8-512, measuring:
- GPU memory utilization and VRAM consumption
- Inference throughput (inferences/second)
- Latency analysis (time per batch vs batch size)
- GPU utilization percentage during inference
- Memory efficiency metrics and OOM detection

Features:
- Comprehensive batch size testing with GPU memory profiling
- VRAM constraint validation (<85% memory usage)
- Throughput vs latency analysis with optimal point detection
- Multi-game support with game-specific memory requirements
- OOM (Out of Memory) handling with automatic batch size reduction
- Statistical analysis and performance curve fitting
- Detailed reporting with visualization support

Usage:
    python scripts/tune_batch_size.py --game gomoku --iterations 100
    python scripts/tune_batch_size.py --quick-test  # Fast optimization for development
    python scripts/tune_batch_size.py --full-sweep --output results/batch_optimization.json

Target: Optimal batch sizes per game with GPU utilization >80% and memory <85% VRAM.

HOWTO-RUN-TESTS:
================
# Run batch size optimization tests
python -m pytest tests/unit/test_batch_size_optimizer.py -v

# Run quick optimization test (minimal resources)
python scripts/tune_batch_size.py --quick-test --max-batch-size 128 --iterations 50

# Run optimization with specific game and constraints
python scripts/tune_batch_size.py --game gomoku --max-vram-percent 80 --iterations 100

# Run full optimization sweep (comprehensive)
python scripts/tune_batch_size.py --full-sweep --output results/

# Run optimization without plots (headless environments)
python scripts/tune_batch_size.py --quick-test --no-plots

# Example expected output showing optimal batch size determination:
#   Optimal Configuration:
#     Batch Size: 64
#     GPU Utilization: 87.3%
#     VRAM Usage: 6.2GB (77.5% of 8GB)
#     Throughput: 3,847 inferences/sec
#     Latency: 16.6ms per batch
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import multiprocessing
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import statistics
import math

# Scientific computing and visualization
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, plots will be disabled")

# PyTorch for GPU memory profiling
import torch
import torch.nn as nn

# System monitoring
import psutil
try:
    import pynvml as nvml
    nvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: nvidia-ml-py not available, GPU monitoring limited")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import components for testing
from src.neural.inference_worker import GPUInferenceWorker
from src.neural.cpu_inference import CPUInferenceWorker
from src.neural.device_manager import DeviceManager, DummyModel
from src.neural.model import AlphaZeroNet
from src.telemetry.metrics import MetricsCollector

# Game imports (try installed version first, then build version)
try:
    # Try installed version first (from pip install -e .)
    from src import alphazero_py
    GAMES_AVAILABLE = True
    print("C++ game extensions available")
except ImportError:
    try:
        # Fallback to build directory version
        build_path = project_root / "build" / "cpp_extensions" / "games"
        if build_path.exists():
            sys.path.insert(0, str(build_path))
        else:
            # Try current directory relative path as fallback
            cwd_build_path = Path.cwd() / "build" / "cpp_extensions" / "games"
            if cwd_build_path.exists():
                sys.path.insert(0, str(cwd_build_path))

        import alphazero_py
        GAMES_AVAILABLE = True
        print("C++ game extensions available (build version)")
    except ImportError as e:
        GAMES_AVAILABLE = False
        print("Warning: Game extensions not available, using Python fallback implementations")
        # Debug: print actual import error
        if '--verbose' in sys.argv or '-v' in sys.argv:
            print(f"Import error details: {e}")

logger = logging.getLogger(__name__)


@dataclass
class BatchSizeTestConfig:
    """Configuration for batch size testing."""

    batch_size: int
    game_type: str = "gomoku"
    iterations: int = 100
    warmup_iterations: int = 20
    timeout_seconds: float = 60.0
    max_vram_percent: float = 85.0
    measure_latency: bool = True
    monitor_gpu: bool = True


@dataclass
class BatchSizePerformanceResult:
    """Results from testing a specific batch size."""

    batch_size: int
    throughput_inferences_per_sec: float
    average_latency_ms: float
    latency_std_ms: float
    gpu_utilization_percent: float
    vram_usage_mb: float
    vram_usage_percent: float
    memory_efficiency: float  # inferences per MB VRAM
    power_consumption_watts: Optional[float] = None
    temperature_celsius: Optional[float] = None
    success_rate: float = 1.0
    oom_occurred: bool = False
    error_message: Optional[str] = None

    def efficiency_score(self) -> float:
        """Calculate overall efficiency score balancing throughput, memory, and latency."""
        if self.success_rate < 0.8 or self.oom_occurred:
            return 0.0

        # Normalize throughput (higher is better)
        throughput_score = min(1.0, self.throughput_inferences_per_sec / 10000.0)

        # VRAM efficiency (lower usage is better, but not too low)
        vram_target = 75.0  # Target 75% VRAM usage for optimal efficiency
        vram_penalty = abs(self.vram_usage_percent - vram_target) / vram_target
        vram_score = max(0.0, 1.0 - vram_penalty)

        # Memory efficiency (inferences per MB)
        memory_efficiency_score = min(1.0, self.memory_efficiency / 10.0)

        # Latency penalty (lower latency is better)
        latency_score = max(0.0, 1.0 - (self.average_latency_ms - 5.0) / 50.0)

        # GPU utilization (higher is better)
        gpu_score = min(1.0, self.gpu_utilization_percent / 100.0)

        # Combine scores with weights
        return (0.35 * throughput_score + 0.25 * vram_score + 0.20 * memory_efficiency_score +
                0.10 * latency_score + 0.10 * gpu_score)


@dataclass
class BatchSizeOptimizationReport:
    """Complete batch size optimization report with recommendations."""

    test_config: Dict[str, Any]
    results: List[BatchSizePerformanceResult]
    optimal_batch_size: int
    optimal_result: BatchSizePerformanceResult
    performance_curve: List[Tuple[int, float]]  # (batch_size, efficiency_score)
    recommendations: List[str]
    system_info: Dict[str, Any]
    gpu_info: Dict[str, Any]
    test_duration_seconds: float


class GPUMonitor:
    """GPU monitoring utility for memory and utilization tracking."""

    def __init__(self):
        self.nvml_available = NVML_AVAILABLE
        self.device_count = 0

        if self.nvml_available:
            try:
                self.device_count = nvml.nvmlDeviceGetCount()
                logger.info(f"NVML initialized with {self.device_count} GPU(s)")
            except Exception as e:
                logger.warning(f"NVML initialization failed: {e}")
                self.nvml_available = False

    def get_gpu_info(self, device_id: int = 0) -> Dict[str, Any]:
        """Get comprehensive GPU information."""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
        }

        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            # PyTorch GPU info
            props = torch.cuda.get_device_properties(device_id)
            info.update({
                'name': props.name,
                'total_memory_mb': props.total_memory / 1024 / 1024,
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multi_processor_count
            })

            # Current memory usage
            memory_allocated = torch.cuda.memory_allocated(device_id) / 1024 / 1024
            memory_reserved = torch.cuda.memory_reserved(device_id) / 1024 / 1024
            info.update({
                'memory_allocated_mb': memory_allocated,
                'memory_reserved_mb': memory_reserved,
                'memory_free_mb': info['total_memory_mb'] - memory_reserved
            })

        # NVML additional info
        if self.nvml_available and device_id < self.device_count:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(device_id)

                # GPU utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                info['gpu_utilization_percent'] = util.gpu
                info['memory_utilization_percent'] = util.memory

                # Temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                info['temperature_celsius'] = temp

                # Power consumption
                try:
                    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    info['power_consumption_watts'] = power
                except:
                    info['power_consumption_watts'] = None

            except Exception as e:
                logger.warning(f"NVML GPU monitoring failed: {e}")

        return info

    def get_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """Get current GPU memory usage information."""
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            return {'total_mb': 0, 'allocated_mb': 0, 'reserved_mb': 0, 'free_mb': 0}

        total = torch.cuda.get_device_properties(device_id).total_memory / 1024 / 1024
        allocated = torch.cuda.memory_allocated(device_id) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device_id) / 1024 / 1024
        free = total - reserved

        return {
            'total_mb': total,
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'free_mb': free,
            'usage_percent': (reserved / total) * 100 if total > 0 else 0
        }


class RealGameState:
    """Real game state using C++ game modules or Python fallback."""

    def __init__(self, game_type: str = "gomoku"):
        self.game_type = game_type
        self.board_size = self._get_board_size(game_type)
        self.feature_planes = self._get_feature_planes(game_type)
        self.action_space = self._get_action_space(game_type)

        if GAMES_AVAILABLE:
            # Use C++ game implementation
            if game_type == "gomoku":
                self.game = alphazero_py.create_game(alphazero_py.GOMOKU)
            elif game_type == "chess":
                self.game = alphazero_py.create_game(alphazero_py.CHESS)
            elif game_type == "go":
                self.game = alphazero_py.create_game(alphazero_py.GO)
            else:
                raise ValueError(f"Unsupported game type: {game_type}")
        else:
            # Use Python fallback
            self.game = None
            self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
            self.current_player = 1
            self.move_count = 0

    def _get_board_size(self, game_type: str) -> int:
        """Get board size for game type."""
        if game_type == "gomoku":
            return 15
        elif game_type == "chess":
            return 8
        elif game_type == "go":
            return 19
        else:
            return 15

    def _get_feature_planes(self, game_type: str) -> int:
        """Get number of feature planes for game type."""
        if game_type == "gomoku":
            return 36
        elif game_type == "chess":
            return 30
        elif game_type == "go":
            return 25
        else:
            return 36

    def _get_action_space(self, game_type: str) -> int:
        """Get action space size for game type."""
        if game_type == "gomoku":
            return 225  # 15x15
        elif game_type == "chess":
            return 4096  # Chess move encoding
        elif game_type == "go":
            return 362  # 19x19 + pass
        else:
            return 225

    def get_features(self) -> np.ndarray:
        """Get feature representation for neural network."""
        if GAMES_AVAILABLE:
            return self.game.get_enhanced_tensor_representation()
        else:
            # Enhanced fallback - create enhanced feature planes matching C++ implementation
            if self.game_type == "gomoku":
                # Gomoku uses 36 feature planes (8 history + 4 meta features)
                features = np.zeros((36, self.board_size, self.board_size), dtype=np.float32)
            elif self.game_type == "chess":
                # Chess uses 119 feature planes
                features = np.zeros((119, self.board_size, self.board_size), dtype=np.float32)
            elif self.game_type == "go":
                # Go uses 25 feature planes
                features = np.zeros((25, self.board_size, self.board_size), dtype=np.float32)
            else:
                # Default fallback
                features = np.zeros((self.feature_planes, self.board_size, self.board_size), dtype=np.float32)

            # Current position - player 1 pieces
            features[0] = (self.board == 1).astype(np.float32)
            # Current position - player 2 pieces
            features[1] = (self.board == -1).astype(np.float32)

            # Fill remaining features with basic patterns
            # Current player indicator (plane 2)
            features[2] = np.full((self.board_size, self.board_size),
                                self.current_player - 1, dtype=np.float32)  # 0 or 1

            # Move count indicator (plane 3)
            move_count_normalized = min(getattr(self, 'move_count', 0) / 100.0, 1.0)
            features[3] = np.full((self.board_size, self.board_size),
                                move_count_normalized, dtype=np.float32)

            return features

    def get_current_player(self) -> int:
        """Get current player to move."""
        if GAMES_AVAILABLE:
            return self.game.get_current_player()
        else:
            return self.current_player

    @property
    def action_space_size(self) -> int:
        """Get action space size for the game."""
        return self.action_space

    def clone(self) -> 'RealGameState':
        """Create a copy of the game state."""
        if GAMES_AVAILABLE:
            new_state = RealGameState(self.game_type)
            new_state.game = self.game.clone()
            return new_state
        else:
            new_state = RealGameState(self.game_type)
            new_state.board = self.board.copy()
            new_state.current_player = self.current_player
            new_state.move_count = self.move_count
            return new_state

    def get_legal_moves(self) -> list:
        """Get list of legal move indices."""
        if GAMES_AVAILABLE:
            legal_moves_mask = self.game.get_legal_moves()
            return list(np.where(legal_moves_mask)[0])
        else:
            # Simple fallback - all empty positions are legal
            legal_moves = []
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i, j] == 0:
                        legal_moves.append(i * self.board_size + j)
            return legal_moves

    def make_move(self, move: int) -> 'RealGameState':
        """Apply move and return new state."""
        if GAMES_AVAILABLE:
            new_state = self.clone()
            new_state.game.make_move(move)
            return new_state
        else:
            new_state = self.clone()
            row, col = move // self.board_size, move % self.board_size
            new_state.board[row, col] = self.current_player
            new_state.current_player = -self.current_player
            new_state.move_count += 1
            return new_state

    def is_terminal(self) -> bool:
        """Check if game is finished."""
        if GAMES_AVAILABLE:
            return self.game.is_terminal()
        else:
            # Simple fallback - check if board is full
            return not np.any(self.board == 0)


class BatchSizeOptimizer:
    """Batch size optimizer for neural network inference performance."""

    def __init__(self, output_dir: Path = None, enable_plotting: bool = True):
        self.output_dir = output_dir or Path("results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.enable_plotting = enable_plotting and PLOTTING_AVAILABLE

        # System info
        self.system_info = self._get_system_info()
        self.gpu_monitor = GPUMonitor()
        self.gpu_info = self.gpu_monitor.get_gpu_info()

        # Initialize device manager
        self.device_manager = DeviceManager()

        logger.info(f"Batch size optimizer initialized")
        logger.info(f"GPU info: {self.gpu_info.get('name', 'Unknown')} "
                   f"({self.gpu_info.get('total_memory_mb', 0):.0f}MB)")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system hardware information."""
        return {
            'cpu_info': f"{psutil.cpu_count()} cores, {psutil.cpu_count(logical=False)} physical",
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'platform': sys.platform,
            'python_version': sys.version
        }

    def create_test_model(self, game_type: str) -> Tuple[Path, nn.Module]:
        """Create a test model for batch size optimization."""
        game_state = RealGameState(game_type)

        # Create model similar to production size
        model = AlphaZeroNet(
            input_channels=game_state.feature_planes,
            num_actions=game_state.action_space,
            num_blocks=10,  # Moderate size for realistic memory usage
            hidden_channels=128  # Realistic size
        )

        # Initialize model for inference
        model.eval()

        # Save to temporary file
        model_path = self.output_dir / f"test_model_batch_{game_type}.pth"
        torch.save(model, model_path)

        logger.info(f"Created test model for {game_type}: {model_path}")
        return model_path, model

    def run_batch_size_test(self, config: BatchSizeTestConfig) -> BatchSizePerformanceResult:
        """Run performance test for a specific batch size."""
        logger.info(f"Testing batch size {config.batch_size}...")

        try:
            # Check if GPU is available
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available for batch size testing")

            # Clear GPU cache before test
            torch.cuda.empty_cache()

            # Create test model and move to GPU
            model_path, model = self.create_test_model(config.game_type)
            device = torch.device('cuda')
            model = model.to(device)

            # Create test data
            game_state = RealGameState(config.game_type)
            input_shape = (game_state.feature_planes, game_state.board_size, game_state.board_size)

            # Monitor initial GPU memory
            initial_memory = self.gpu_monitor.get_memory_info()

            # Warmup phase
            logger.debug(f"Running {config.warmup_iterations} warmup iterations...")
            self._run_warmup_batch(model, input_shape, config.batch_size, config.warmup_iterations)

            # Main test phase
            latencies = []
            throughputs = []
            gpu_utils = []
            start_time = time.time()

            for i in range(config.iterations):
                if time.time() - start_time > config.timeout_seconds:
                    logger.warning(f"Test timeout reached for batch size {config.batch_size}")
                    break

                try:
                    # Create batch of test data
                    batch_data = torch.randn(
                        config.batch_size, *input_shape,
                        device=device, dtype=torch.float32
                    )

                    # Measure inference time
                    torch.cuda.synchronize()
                    inference_start = time.time()

                    with torch.no_grad():
                        policy, value = model(batch_data)

                    torch.cuda.synchronize()
                    inference_time = time.time() - inference_start

                    # Record metrics
                    latency_ms = inference_time * 1000
                    throughput = config.batch_size / inference_time
                    latencies.append(latency_ms)
                    throughputs.append(throughput)

                    # Monitor GPU utilization
                    if config.monitor_gpu:
                        gpu_info = self.gpu_monitor.get_gpu_info()
                        gpu_utils.append(gpu_info.get('gpu_utilization_percent', 0))

                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM occurred for batch size {config.batch_size}")
                    # Cleanup and return OOM result
                    torch.cuda.empty_cache()
                    if model_path.exists():
                        model_path.unlink()

                    return BatchSizePerformanceResult(
                        batch_size=config.batch_size,
                        throughput_inferences_per_sec=0.0,
                        average_latency_ms=float('inf'),
                        latency_std_ms=0.0,
                        gpu_utilization_percent=0.0,
                        vram_usage_mb=0.0,
                        vram_usage_percent=100.0,
                        memory_efficiency=0.0,
                        success_rate=0.0,
                        oom_occurred=True,
                        error_message="Out of Memory"
                    )

            # Get final GPU memory info
            final_memory = self.gpu_monitor.get_memory_info()

            # Calculate metrics
            if latencies and throughputs:
                avg_latency = statistics.mean(latencies)
                latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
                avg_throughput = statistics.mean(throughputs)
                success_rate = len(latencies) / config.iterations
            else:
                avg_latency = float('inf')
                latency_std = 0.0
                avg_throughput = 0.0
                success_rate = 0.0

            # GPU utilization
            avg_gpu_util = statistics.mean(gpu_utils) if gpu_utils else 0.0

            # Memory calculations
            vram_usage_mb = final_memory['reserved_mb']
            vram_usage_percent = final_memory['usage_percent']
            memory_efficiency = avg_throughput / max(vram_usage_mb, 1.0) if vram_usage_mb > 0 else 0.0

            # Get additional GPU info
            gpu_info = self.gpu_monitor.get_gpu_info()

            result = BatchSizePerformanceResult(
                batch_size=config.batch_size,
                throughput_inferences_per_sec=avg_throughput,
                average_latency_ms=avg_latency,
                latency_std_ms=latency_std,
                gpu_utilization_percent=avg_gpu_util,
                vram_usage_mb=vram_usage_mb,
                vram_usage_percent=vram_usage_percent,
                memory_efficiency=memory_efficiency,
                power_consumption_watts=gpu_info.get('power_consumption_watts'),
                temperature_celsius=gpu_info.get('temperature_celsius'),
                success_rate=success_rate,
                oom_occurred=False
            )

            logger.info(f"Batch {config.batch_size}: "
                       f"{avg_throughput:.0f} inf/sec, "
                       f"{avg_latency:.1f}ms, "
                       f"{vram_usage_percent:.1f}% VRAM, "
                       f"efficiency: {result.efficiency_score():.3f}")

            # Cleanup
            torch.cuda.empty_cache()
            if model_path.exists():
                model_path.unlink()

            return result

        except Exception as e:
            logger.error(f"Error testing batch size {config.batch_size}: {e}")

            # Cleanup on error
            torch.cuda.empty_cache()
            try:
                if 'model_path' in locals() and model_path.exists():
                    model_path.unlink()
            except:
                pass

            return BatchSizePerformanceResult(
                batch_size=config.batch_size,
                throughput_inferences_per_sec=0.0,
                average_latency_ms=float('inf'),
                latency_std_ms=0.0,
                gpu_utilization_percent=0.0,
                vram_usage_mb=0.0,
                vram_usage_percent=0.0,
                memory_efficiency=0.0,
                success_rate=0.0,
                oom_occurred=False,
                error_message=str(e)
            )

    def _run_warmup_batch(self, model: nn.Module, input_shape: Tuple, batch_size: int, iterations: int):
        """Run warmup iterations to stabilize GPU performance."""
        device = next(model.parameters()).device

        for _ in range(iterations):
            try:
                batch_data = torch.randn(batch_size, *input_shape, device=device, dtype=torch.float32)
                with torch.no_grad():
                    _ = model(batch_data)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                # If warmup OOMs, the actual test will too
                break

    def optimize_batch_size(self,
                          game_type: str = "gomoku",
                          iterations: int = 100,
                          min_batch_size: int = 8,
                          max_batch_size: int = 512,
                          max_vram_percent: float = 85.0,
                          quick_test: bool = False) -> BatchSizeOptimizationReport:
        """Run complete batch size optimization."""

        start_time = time.time()
        logger.info(f"Starting batch size optimization for {game_type}")
        logger.info(f"Parameters: {iterations} iterations, batch range {min_batch_size}-{max_batch_size}")
        logger.info(f"Max VRAM: {max_vram_percent}%")

        if quick_test:
            iterations = max(20, iterations // 5)
            max_batch_size = min(max_batch_size, 128)
            logger.info(f"Quick test mode: {iterations} iterations, max batch {max_batch_size}")

        # Test batch sizes (powers of 2 for GPU efficiency)
        batch_sizes = []
        batch_size = min_batch_size
        while batch_size <= max_batch_size:
            batch_sizes.append(batch_size)
            batch_size *= 2

        results = []

        for batch_size in batch_sizes:
            config = BatchSizeTestConfig(
                batch_size=batch_size,
                game_type=game_type,
                iterations=iterations,
                warmup_iterations=max(5, iterations // 10),
                timeout_seconds=120.0 if not quick_test else 60.0,
                max_vram_percent=max_vram_percent
            )

            result = self.run_batch_size_test(config)
            results.append(result)

            # Stop if we hit OOM or exceed VRAM limit
            if result.oom_occurred or result.vram_usage_percent > max_vram_percent:
                logger.info(f"Stopping optimization: "
                           f"{'OOM' if result.oom_occurred else 'VRAM limit exceeded'}")
                break

            # Early stopping if efficiency is clearly decreasing
            if len(results) >= 3 and all(r.efficiency_score() < 0.1 for r in results[-2:]):
                logger.info("Efficiency degraded significantly, stopping early")
                break

        # Find optimal configuration
        valid_results = [r for r in results if r.success_rate >= 0.8 and not r.oom_occurred
                        and r.vram_usage_percent <= max_vram_percent]

        if not valid_results:
            logger.error("No valid results found!")
            valid_results = [r for r in results if r.success_rate > 0]  # Use any working result

        if not valid_results:
            # Fallback to smallest batch size with dummy result
            optimal_result = BatchSizePerformanceResult(
                batch_size=min_batch_size,
                throughput_inferences_per_sec=0.0,
                average_latency_ms=float('inf'),
                latency_std_ms=0.0,
                gpu_utilization_percent=0.0,
                vram_usage_mb=0.0,
                vram_usage_percent=0.0,
                memory_efficiency=0.0,
                success_rate=0.0,
                error_message="No successful runs"
            )
            optimal_batch_size = min_batch_size
        else:
            optimal_result = max(valid_results, key=lambda r: r.efficiency_score())
            optimal_batch_size = optimal_result.batch_size

        # Generate performance curve
        performance_curve = [(r.batch_size, r.efficiency_score()) for r in results]

        # Generate recommendations
        recommendations = self._generate_recommendations(results, optimal_result, max_vram_percent)

        # Create optimization report
        report = BatchSizeOptimizationReport(
            test_config={
                'game_type': game_type,
                'iterations': iterations,
                'min_batch_size': min_batch_size,
                'max_batch_size': max_batch_size,
                'max_vram_percent': max_vram_percent,
                'quick_test': quick_test
            },
            results=results,
            optimal_batch_size=optimal_batch_size,
            optimal_result=optimal_result,
            performance_curve=performance_curve,
            recommendations=recommendations,
            system_info=self.system_info,
            gpu_info=self.gpu_info,
            test_duration_seconds=time.time() - start_time
        )

        logger.info(f"Batch size optimization completed in {report.test_duration_seconds:.1f}s")
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        logger.info(f"Throughput: {optimal_result.throughput_inferences_per_sec:.0f} inferences/sec")
        logger.info(f"VRAM usage: {optimal_result.vram_usage_percent:.1f}%")

        return report

    def _generate_recommendations(self,
                                results: List[BatchSizePerformanceResult],
                                optimal_result: BatchSizePerformanceResult,
                                max_vram_percent: float) -> List[str]:
        """Generate optimization recommendations based on results."""

        recommendations = []

        # Optimal batch size recommendation
        recommendations.append(
            f"Use batch size {optimal_result.batch_size} for optimal performance "
            f"({optimal_result.throughput_inferences_per_sec:.0f} inferences/sec)"
        )

        # VRAM usage analysis
        if optimal_result.vram_usage_percent > max_vram_percent * 0.9:
            recommendations.append(
                f"High VRAM usage ({optimal_result.vram_usage_percent:.1f}%) - "
                f"consider reducing batch size for safety margin"
            )
        elif optimal_result.vram_usage_percent < max_vram_percent * 0.5:
            recommendations.append(
                f"Low VRAM usage ({optimal_result.vram_usage_percent:.1f}%) - "
                f"could potentially use larger batch size for better throughput"
            )
        else:
            recommendations.append(
                f"Good VRAM efficiency ({optimal_result.vram_usage_percent:.1f}%) - "
                f"well-balanced memory usage"
            )

        # GPU utilization analysis
        if optimal_result.gpu_utilization_percent >= 80.0:
            recommendations.append(
                f"Excellent GPU utilization ({optimal_result.gpu_utilization_percent:.1f}%) - "
                f"GPU is well-utilized"
            )
        elif optimal_result.gpu_utilization_percent >= 60.0:
            recommendations.append(
                f"Good GPU utilization ({optimal_result.gpu_utilization_percent:.1f}%) - "
                f"acceptable GPU usage"
            )
        else:
            recommendations.append(
                f"Low GPU utilization ({optimal_result.gpu_utilization_percent:.1f}%) - "
                f"consider increasing batch size or checking for bottlenecks"
            )

        # Memory efficiency analysis
        if optimal_result.memory_efficiency > 5.0:
            recommendations.append(
                f"High memory efficiency ({optimal_result.memory_efficiency:.1f} inf/MB) - "
                f"good balance of throughput and memory usage"
            )
        elif optimal_result.memory_efficiency > 2.0:
            recommendations.append(
                f"Moderate memory efficiency ({optimal_result.memory_efficiency:.1f} inf/MB) - "
                f"acceptable performance per memory unit"
            )
        else:
            recommendations.append(
                f"Low memory efficiency ({optimal_result.memory_efficiency:.1f} inf/MB) - "
                f"consider optimizing memory usage"
            )

        # Latency analysis
        if optimal_result.average_latency_ms <= 20.0:
            recommendations.append(
                f"Low latency ({optimal_result.average_latency_ms:.1f}ms) - "
                f"good responsiveness for real-time applications"
            )
        elif optimal_result.average_latency_ms <= 50.0:
            recommendations.append(
                f"Moderate latency ({optimal_result.average_latency_ms:.1f}ms) - "
                f"acceptable for most applications"
            )
        else:
            recommendations.append(
                f"High latency ({optimal_result.average_latency_ms:.1f}ms) - "
                f"consider smaller batch size for better responsiveness"
            )

        # OOM analysis
        oom_results = [r for r in results if r.oom_occurred]
        if oom_results:
            min_oom_batch = min(r.batch_size for r in oom_results)
            recommendations.append(
                f"Out of memory occurs at batch size {min_oom_batch} - "
                f"stay below this limit to avoid OOM errors"
            )

        # Power consumption (if available)
        if optimal_result.power_consumption_watts:
            recommendations.append(
                f"Power consumption: {optimal_result.power_consumption_watts:.1f}W - "
                f"monitor for thermal limits during extended inference"
            )

        return recommendations

    def save_report(self, report: BatchSizeOptimizationReport, filename: str = None) -> Path:
        """Save optimization report to file."""

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"batch_size_optimization_{timestamp}.json"

        output_path = self.output_dir / filename

        # Convert to serializable format
        report_dict = asdict(report)

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)

        logger.info(f"Batch size optimization report saved to {output_path}")
        return output_path

    def plot_results(self, report: BatchSizeOptimizationReport, save_plots: bool = True) -> Optional[Path]:
        """Create visualization plots of optimization results."""

        if not self.enable_plotting:
            logger.warning("Plotting disabled - matplotlib/seaborn not available")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Batch Size Optimization Results', fontsize=16)

        # Plot 1: Throughput vs Batch Size
        batch_sizes = [r.batch_size for r in report.results]
        throughputs = [r.throughput_inferences_per_sec for r in report.results]

        axes[0, 0].semilogx(batch_sizes, throughputs, 'bo-', linewidth=2, markersize=8, basex=2)
        axes[0, 0].axvline(report.optimal_batch_size, color='r', linestyle='--',
                          label=f'Optimal: {report.optimal_batch_size}')
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Throughput (inferences/sec)')
        axes[0, 0].set_title('Throughput vs Batch Size')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Plot 2: VRAM Usage vs Batch Size
        vram_usages = [r.vram_usage_percent for r in report.results]

        axes[0, 1].semilogx(batch_sizes, vram_usages, 'go-', linewidth=2, markersize=8, basex=2)
        axes[0, 1].axhline(report.test_config['max_vram_percent'], color='r', linestyle='--',
                          label=f"Limit: {report.test_config['max_vram_percent']}%")
        axes[0, 1].axvline(report.optimal_batch_size, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('VRAM Usage (%)')
        axes[0, 1].set_title('VRAM Usage vs Batch Size')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Plot 3: Efficiency Score vs Batch Size
        efficiency_scores = [r.efficiency_score() for r in report.results]

        axes[1, 0].semilogx(batch_sizes, efficiency_scores, 'mo-', linewidth=2, markersize=8, basex=2)
        axes[1, 0].axvline(report.optimal_batch_size, color='r', linestyle='--',
                          label=f'Optimal: {report.optimal_result.efficiency_score():.3f}')
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Efficiency Score')
        axes[1, 0].set_title('Overall Efficiency vs Batch Size')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Plot 4: Latency vs Batch Size
        latencies = [r.average_latency_ms for r in report.results if r.average_latency_ms < float('inf')]
        valid_batch_sizes = [r.batch_size for r in report.results if r.average_latency_ms < float('inf')]

        if latencies and valid_batch_sizes:
            axes[1, 1].semilogx(valid_batch_sizes, latencies, 'ro-', linewidth=2, markersize=8, basex=2)
            axes[1, 1].axvline(report.optimal_batch_size, color='g', linestyle='--', alpha=0.7)
            axes[1, 1].set_xlabel('Batch Size')
            axes[1, 1].set_ylabel('Average Latency (ms)')
            axes[1, 1].set_title('Latency vs Batch Size')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"batch_size_optimization_plots_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Plots saved to {plot_path}")
            return plot_path
        else:
            plt.show()
            return None

    def print_summary(self, report: BatchSizeOptimizationReport):
        """Print optimization summary to console."""

        print("\n" + "="*80)
        print("BATCH SIZE OPTIMIZATION SUMMARY")
        print("="*80)

        print(f"\nTest Configuration:")
        print(f"  Game Type: {report.test_config['game_type']}")
        print(f"  Iterations: {report.test_config['iterations']}")
        print(f"  Batch Size Range: {report.test_config['min_batch_size']}-{report.test_config['max_batch_size']}")
        print(f"  Max VRAM: {report.test_config['max_vram_percent']:.1f}%")
        print(f"  Test Duration: {report.test_duration_seconds:.1f}s")

        print(f"\nGPU Information:")
        print(f"  GPU: {report.gpu_info.get('name', 'Unknown')}")
        print(f"  Total VRAM: {report.gpu_info.get('total_memory_mb', 0):.0f}MB")
        print(f"  Compute Capability: {report.gpu_info.get('compute_capability', 'Unknown')}")

        print(f"\nOptimal Configuration:")
        print(f"  Batch Size: {report.optimal_batch_size}")
        print(f"  Throughput: {report.optimal_result.throughput_inferences_per_sec:.0f} inferences/sec")
        print(f"  Average Latency: {report.optimal_result.average_latency_ms:.1f}ms")
        print(f"  VRAM Usage: {report.optimal_result.vram_usage_mb:.0f}MB ({report.optimal_result.vram_usage_percent:.1f}%)")
        print(f"  GPU Utilization: {report.optimal_result.gpu_utilization_percent:.1f}%")
        print(f"  Memory Efficiency: {report.optimal_result.memory_efficiency:.1f} inferences/MB")
        print(f"  Efficiency Score: {report.optimal_result.efficiency_score():.3f}")

        if report.optimal_result.power_consumption_watts:
            print(f"  Power Consumption: {report.optimal_result.power_consumption_watts:.1f}W")
        if report.optimal_result.temperature_celsius:
            print(f"  GPU Temperature: {report.optimal_result.temperature_celsius:.1f}°C")

        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

        print(f"\nDetailed Results:")
        print(f"{'Batch':<8} {'Throughput':<12} {'Latency':<10} {'VRAM':<8} {'GPU':<8} {'Efficiency':<10}")
        print(f"{'Size':<8} {'(inf/s)':<12} {'(ms)':<10} {'(%)':<8} {'(%)':<8} {'Score':<10}")
        print("-" * 68)

        for result in report.results:
            if result.oom_occurred:
                print(f"{result.batch_size:<8} {'OOM':<12} {'N/A':<10} {'N/A':<8} {'N/A':<8} {'0.000':<10}")
            else:
                print(f"{result.batch_size:<8} "
                      f"{result.throughput_inferences_per_sec:<12.0f} "
                      f"{result.average_latency_ms:<10.1f} "
                      f"{result.vram_usage_percent:<8.1f} "
                      f"{result.gpu_utilization_percent:<8.1f} "
                      f"{result.efficiency_score():<10.3f}")

        print("="*80)


def main():
    """Main function for batch size optimization."""

    parser = argparse.ArgumentParser(
        description="Optimize neural network batch size for GPU memory and throughput",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/tune_batch_size.py --quick-test
  python scripts/tune_batch_size.py --game gomoku --iterations 100
  python scripts/tune_batch_size.py --full-sweep --output results/
  python scripts/tune_batch_size.py --max-batch-size 256 --max-vram 80 --no-plots
        """
    )

    parser.add_argument('--game', default='gomoku', choices=['gomoku', 'chess', 'go'],
                        help='Game type to optimize for (default: gomoku)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of inference iterations per batch size test (default: 100)')
    parser.add_argument('--min-batch-size', type=int, default=8,
                        help='Minimum batch size to test (default: 8)')
    parser.add_argument('--max-batch-size', type=int, default=512,
                        help='Maximum batch size to test (default: 512)')
    parser.add_argument('--max-vram', type=float, default=85.0,
                        help='Maximum VRAM usage percentage (default: 85.0)')
    parser.add_argument('--output', type=Path, default=Path('results'),
                        help='Output directory for results (default: results/)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with reduced parameters')
    parser.add_argument('--full-sweep', action='store_true',
                        help='Run comprehensive test with maximum parameters')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--save-report', type=str,
                        help='Save detailed report to specified filename')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Adjust parameters for full sweep
    if args.full_sweep:
        args.iterations = max(args.iterations, 200)
        args.max_batch_size = min(args.max_batch_size, 1024)
        logger.info("Full sweep mode enabled - using maximum parameters")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Batch size optimization requires GPU.")
        sys.exit(1)

    try:
        # Create optimizer
        optimizer = BatchSizeOptimizer(
            output_dir=args.output,
            enable_plotting=not args.no_plots
        )

        # Run optimization
        report = optimizer.optimize_batch_size(
            game_type=args.game,
            iterations=args.iterations,
            min_batch_size=args.min_batch_size,
            max_batch_size=args.max_batch_size,
            max_vram_percent=args.max_vram,
            quick_test=args.quick_test
        )

        # Print summary
        optimizer.print_summary(report)

        # Save report
        if args.save_report:
            optimizer.save_report(report, args.save_report)
        else:
            optimizer.save_report(report)

        # Generate plots
        if not args.no_plots:
            optimizer.plot_results(report, save_plots=True)

        # Exit with success code if optimization found good results
        optimal_result = report.optimal_result
        target_gpu_util = optimal_result.gpu_utilization_percent >= 80.0
        target_vram_limit = optimal_result.vram_usage_percent <= args.max_vram

        if target_gpu_util and target_vram_limit and optimal_result.success_rate >= 0.8:
            logger.info("Batch size optimization successful - target criteria met")
            sys.exit(0)
        else:
            logger.warning("Optimization completed but target criteria not fully met")
            logger.warning(f"GPU util: {optimal_result.gpu_utilization_percent:.1f}% "
                         f"(target: ≥80%), VRAM: {optimal_result.vram_usage_percent:.1f}% "
                         f"(target: ≤{args.max_vram}%)")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Batch size optimization interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Batch size optimization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()