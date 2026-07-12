#!/usr/bin/env python3
"""
Thread Count Optimization Script
=================================

Systematically tests different thread counts for MCTS search to find the optimal
configuration that maximizes throughput while minimizing contention.

This script performs parameter sweeps across thread counts 1-16, measuring:
- Search throughput (simulations/second)
- Thread utilization efficiency
- Lock contention and synchronization overhead
- CPU core utilization patterns
- Memory access patterns and cache performance

Features:
- Comprehensive performance measurement across thread count range
- Contention detection through timing variance analysis
- CPU and memory monitoring during optimization
- Statistical analysis with confidence intervals
- Optimal recommendation based on multiple criteria
- Detailed reporting with visualization support

Usage:
    python scripts/tune_threads.py --game gomoku --simulations 1000 --iterations 10
    python scripts/tune_threads.py --quick-test  # Fast optimization for development
    python scripts/tune_threads.py --full-sweep --output results/thread_tuning.json

Target: Find optimal thread count (8-10) with <10% contention and peak performance.

HOWTO-RUN-TESTS:
================
# Run thread optimization tests
python -m pytest tests/unit/test_thread_optimizer.py -v

# Run quick optimization test (minimal resources)
python scripts/tune_threads.py --quick-test --max-threads 4 --iterations 5

# Run optimization with specific game and parameters
python scripts/tune_threads.py --game gomoku --simulations 800 --iterations 50

# Run full optimization sweep (comprehensive)
python scripts/tune_threads.py --full-sweep --output results/

# Run optimization without plots (headless environments)
python scripts/tune_threads.py --quick-test --no-plots

# Example expected output showing optimal thread count determination:
#   Optimal Configuration:
#     Thread Count: 8
#     Throughput: 456.2 searches/sec
#     Contention Score: 8.3%
#     Efficiency Score: 0.847
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
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
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

# PyTorch for real models
import torch
import torch.nn as nn

# System monitoring
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import components for testing
from src.core.search_coordinator import SearchCoordinator, SearchRequest
from src.neural.inference_worker import GPUInferenceWorker
from src.neural.cpu_inference import CPUInferenceWorker
from src.neural.device_manager import DeviceManager
from src.neural.model import AlphaZeroNet
from src.telemetry.metrics import MetricsCollector

# Game imports (with fallback for testing)
try:
    import alphazero_py
    GAMES_AVAILABLE = True
    print("C++ game extensions available")
except ImportError:
    GAMES_AVAILABLE = False
    print("Warning: Game extensions not available, using Python fallback implementations")

logger = logging.getLogger(__name__)


@dataclass
class ThreadTestConfig:
    """Configuration for thread count testing."""

    thread_count: int
    game_type: str = "gomoku"
    simulations_per_search: int = 800
    num_searches: int = 100
    warmup_searches: int = 10
    timeout_seconds: float = 60.0
    measure_contention: bool = True
    monitor_system: bool = True


@dataclass
class ThreadPerformanceResult:
    """Results from testing a specific thread count."""

    thread_count: int
    searches_per_second: float
    average_search_time_ms: float
    search_time_std_ms: float
    thread_utilization_percent: float
    contention_score: float
    cpu_utilization_percent: float
    memory_usage_mb: float
    cache_miss_ratio: Optional[float] = None
    success_rate: float = 1.0
    error_message: Optional[str] = None

    def efficiency_score(self) -> float:
        """Calculate overall efficiency score combining throughput and contention."""
        if self.success_rate < 0.8:
            return 0.0

        # Normalize throughput (higher is better)
        throughput_score = min(1.0, self.searches_per_second / 1000.0)

        # Penalize high contention (lower is better)
        contention_penalty = max(0.0, 1.0 - self.contention_score / 50.0)

        # Penalize low thread utilization (target around 80-90%)
        utilization_score = 1.0 - abs(self.thread_utilization_percent - 85.0) / 85.0
        utilization_score = max(0.0, utilization_score)

        # Combine scores with weights
        return (0.5 * throughput_score + 0.3 * contention_penalty + 0.2 * utilization_score)


@dataclass
class OptimizationReport:
    """Complete optimization report with recommendations."""

    test_config: Dict[str, Any]
    results: List[ThreadPerformanceResult]
    optimal_thread_count: int
    optimal_result: ThreadPerformanceResult
    performance_curve: List[Tuple[int, float]]  # (thread_count, efficiency_score)
    recommendations: List[str]
    system_info: Dict[str, Any]
    test_duration_seconds: float


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
                self.game = alphazero_py.GomokuState()
            elif game_type == "chess":
                self.game = alphazero_py.ChessState()
            elif game_type == "go":
                self.game = alphazero_py.GoState()
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

    def copy(self):
        """Create a copy of the game state."""
        new_state = RealGameState(self.game_type)
        if GAMES_AVAILABLE:
            # Game states may be immutable, just assign the current state
            new_state.game = self.game
        else:
            new_state.board = self.board.copy()
            new_state.current_player = self.current_player
            new_state.move_count = self.move_count
        return new_state

    def apply_move(self, move: int):
        """Apply a move to the game state."""
        if GAMES_AVAILABLE:
            # make_move modifies the game in place
            self.game.make_move(move)
        else:
            # Simple fallback implementation
            row, col = divmod(move, self.board_size)
            if row < self.board_size and col < self.board_size and self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.current_player = 3 - self.current_player  # Toggle between 1 and 2
                self.move_count += 1

    def get_legal_moves(self) -> List[int]:
        """Get list of legal moves."""
        if GAMES_AVAILABLE:
            legal_moves_mask = self.game.get_legal_moves()
            return np.where(legal_moves_mask)[0].tolist()
        else:
            # Simple fallback - empty squares
            legal_moves = []
            for move in range(self.action_space):
                if move < self.board_size * self.board_size:
                    row, col = divmod(move, self.board_size)
                    if row < self.board_size and col < self.board_size and self.board[row, col] == 0:
                        legal_moves.append(move)
            return legal_moves

    def make_move(self, move: int) -> 'RealGameState':
        """Apply move and return new state (compatibility method)."""
        new_state = self.clone()
        new_state.apply_move(move)
        return new_state

    def is_terminal(self) -> bool:
        """Check if game is over."""
        if GAMES_AVAILABLE:
            return self.game.is_terminal()
        else:
            # Simple fallback - board full or too many moves
            return len(self.get_legal_moves()) == 0 or self.move_count > 100

    def get_features(self) -> np.ndarray:
        """Get feature representation for neural network."""
        if GAMES_AVAILABLE:
            return self.game.get_enhanced_tensor_representation()
        else:
            # Simple fallback - create basic feature planes
            features = np.zeros((self.feature_planes, self.board_size, self.board_size), dtype=np.float32)

            # Player 1 pieces
            features[0] = (self.board == 1).astype(np.float32)
            # Player 2 pieces
            features[1] = (self.board == -1).astype(np.float32)
            # Current player
            features[2] = np.full((self.board_size, self.board_size),
                                self.current_player, dtype=np.float32)

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


class ThreadOptimizer:
    """Thread count optimizer for MCTS search performance."""

    def __init__(self, output_dir: Path = None, enable_plotting: bool = True):
        self.output_dir = output_dir or Path("results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.enable_plotting = enable_plotting and PLOTTING_AVAILABLE

        # System info
        self.cpu_count = multiprocessing.cpu_count()
        self.system_info = self._get_system_info()

        # Initialize device manager
        self.device_manager = DeviceManager()

        logger.info(f"Thread optimizer initialized - CPU cores: {self.cpu_count}")
        logger.info(f"System info: {self.system_info['cpu_info']}")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system hardware information."""
        return {
            'cpu_info': f"{psutil.cpu_count()} cores, {psutil.cpu_count(logical=False)} physical",
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'platform': sys.platform,
            'python_version': sys.version
        }

    def create_test_model(self, game_type: str) -> Path:
        """Create a test model for threading optimization."""
        game_state = RealGameState(game_type)

        # Create model with minimal size for testing
        model = AlphaZeroNet(
            input_channels=game_state.feature_planes,
            num_actions=game_state.action_space,
            num_blocks=2,  # Minimal for testing
            hidden_channels=32  # Small for testing
        )

        # Initialize model for inference
        model.eval()

        # Save to temporary file
        model_path = self.output_dir / f"test_model_{game_type}.pth"
        torch.save(model, model_path)

        logger.info(f"Created test model for {game_type}: {model_path}")
        return model_path

    def create_real_inference_worker(self, game_type: str, use_gpu: bool = True):
        """Create real inference worker with actual model."""
        # Create test model
        model_path = self.create_test_model(game_type)

        # Try GPU first, fall back to CPU
        if use_gpu and torch.cuda.is_available():
            try:
                worker = GPUInferenceWorker(
                    model_path=str(model_path),
                    device='cuda',
                    batch_size=32,
                    timeout_ms=3.0,
                    use_mixed_precision=True
                )
                setattr(worker, '_model_path', str(model_path))
                logger.info("Created GPU inference worker")
                return worker
            except Exception as e:
                logger.warning(f"Failed to create GPU worker: {e}, falling back to CPU")

        # Use CPU worker
        worker = CPUInferenceWorker(
            model_path=str(model_path),
            device='cpu',
            batch_size=16,  # Smaller batches for CPU
            timeout_ms=10.0
        )
        setattr(worker, '_model_path', str(model_path))
        logger.info("Created CPU inference worker")
        return worker

    def run_thread_count_test(self, config: ThreadTestConfig) -> ThreadPerformanceResult:
        """Run performance test for a specific thread count."""
        logger.info(f"Testing {config.thread_count} threads...")

        inference_worker = None
        coordinator = None
        model_path = None
        failure_reasons: List[str] = []

        try:
            inference_worker = self.create_real_inference_worker(config.game_type, use_gpu=True)
            model_path = getattr(inference_worker, '_model_path', None)

            coordinator = SearchCoordinator(
                inference_worker=inference_worker,
                max_threads=config.thread_count,
                max_queue_size=1000,
                monitoring_interval=0.2
            )
            coordinator.start()

            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            psutil.cpu_percent(interval=None)  # Prime CPU measurement window

            self._run_warmup_searches(coordinator, config)

            search_times: List[float] = []
            cpu_samples: List[float] = []
            parallel_samples: List[int] = []
            inflight: Dict[Any, float] = {}
            start_time = time.time()
            submitted = 0
            completed = 0
            max_parallel = max(1, config.thread_count)

            while completed < config.num_searches:
                elapsed = time.time() - start_time
                remaining = config.timeout_seconds - elapsed
                if remaining <= 0:
                    failure_reasons.append(f"Timeout after {config.timeout_seconds}s")
                    logger.warning(f"Test timeout reached for {config.thread_count} threads")
                    break

                while submitted < config.num_searches and len(inflight) < max_parallel:
                    game_state = RealGameState(config.game_type)
                    legal_moves = game_state.get_legal_moves()
                    if legal_moves and not game_state.is_terminal():
                        num_moves = min(int(np.random.randint(1, 6)), len(legal_moves))
                        for _ in range(num_moves):
                            if game_state.is_terminal():
                                break
                            legal_moves = game_state.get_legal_moves()
                            if legal_moves:
                                move = int(np.random.choice(legal_moves))
                                game_state.apply_move(move)

                    request = SearchRequest(
                        request_id=f"test_{submitted}",
                        game_state=game_state,
                        simulations=config.simulations_per_search,
                        temperature=1.0,
                        add_noise=False
                    )

                    future = coordinator.submit_search(request)
                    inflight[future] = time.time()
                    submitted += 1
                    parallel_samples.append(len(inflight))

                if not inflight:
                    break

                wait_timeout = max(0.0, remaining)
                done, _ = wait(list(inflight.keys()), timeout=wait_timeout, return_when=FIRST_COMPLETED)
                if not done:
                    failure_reasons.append('Search batch timed out')
                    logger.warning(f"Search batch timeout for {config.thread_count} threads")
                    break

                for future in done:
                    search_start = inflight.pop(future)
                    try:
                        future.result(timeout=0)
                        search_time_ms = (time.time() - search_start) * 1000.0
                        search_times.append(search_time_ms)
                        cpu_samples.append(psutil.cpu_percent(interval=None))
                        completed += 1
                    except Exception as exc:
                        failure_reasons.append(str(exc))
                        cpu_samples.append(psutil.cpu_percent(interval=None))

            for future in inflight:
                future.cancel()

            total_time = max(time.time() - start_time, 1e-6)
            final_memory = process.memory_info().rss / 1024 / 1024
            success_rate = (len(search_times) / config.num_searches) if config.num_searches else 0.0

            if search_times:
                avg_search_time = statistics.mean(search_times)
                search_time_std = statistics.stdev(search_times) if len(search_times) > 1 else 0.0
                searches_per_second = len(search_times) / total_time
            else:
                avg_search_time = float('inf')
                search_time_std = 0.0
                searches_per_second = 0.0

            if avg_search_time in (0.0, float('inf')):
                contention_score = 100.0 if avg_search_time == float('inf') else 0.0
            else:
                contention_score = (search_time_std / avg_search_time * 100.0) if avg_search_time > 0 else 100.0

            effective_thread_seconds = sum(st / 1000.0 for st in search_times)
            thread_utilization = 0.0
            if parallel_samples:
                avg_parallelism = sum(parallel_samples) / len(parallel_samples)
                thread_utilization = min(100.0, (avg_parallelism / max(1, config.thread_count)) * 100.0)
            elif config.thread_count > 0 and total_time > 0:
                thread_utilization = (effective_thread_seconds / (config.thread_count * total_time)) * 100.0
                thread_utilization = max(0.0, min(100.0, thread_utilization))

            cpu_utilization = statistics.mean(cpu_samples) if cpu_samples else psutil.cpu_percent(interval=None)
            memory_usage = max(0.0, final_memory - initial_memory)

            if success_rate < 1.0 and not failure_reasons:
                failure_reasons.append('Not all searches completed')

            error_message = None
            if failure_reasons and success_rate < 1.0:
                error_message = '; '.join(dict.fromkeys(failure_reasons))

            result = ThreadPerformanceResult(
                thread_count=config.thread_count,
                searches_per_second=searches_per_second,
                average_search_time_ms=avg_search_time,
                search_time_std_ms=search_time_std,
                thread_utilization_percent=thread_utilization,
                contention_score=contention_score,
                cpu_utilization_percent=cpu_utilization,
                memory_usage_mb=memory_usage,
                success_rate=success_rate,
                error_message=error_message
            )

            logger.info(
                f"Thread {config.thread_count}: {searches_per_second:.1f} searches/sec, "
                f"contention {contention_score:.1f}%, utilization {thread_utilization:.1f}%"
            )

            return result

        except Exception as e:
            logger.error(f"Error testing {config.thread_count} threads: {e}", exc_info=True)
            return ThreadPerformanceResult(
                thread_count=config.thread_count,
                searches_per_second=0.0,
                average_search_time_ms=float('inf'),
                search_time_std_ms=0.0,
                thread_utilization_percent=0.0,
                contention_score=100.0,
                cpu_utilization_percent=0.0,
                memory_usage_mb=0.0,
                success_rate=0.0,
                error_message=str(e)
            )

        finally:
            if coordinator is not None:
                try:
                    coordinator.stop()
                except Exception as stop_error:
                    logger.warning(f"Error stopping coordinator: {stop_error}")

            if inference_worker is not None and hasattr(inference_worker, 'stop'):
                try:
                    inference_worker.stop()
                except Exception:
                    pass

            if model_path:
                try:
                    path_obj = Path(model_path)
                    if path_obj.exists():
                        path_obj.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to delete temporary model {model_path}: {cleanup_error}")

    def _run_warmup_searches(self, coordinator: SearchCoordinator, config: ThreadTestConfig):
        """Run warmup searches to stabilize performance."""
        logger.debug(f"Running {config.warmup_searches} warmup searches...")

        for i in range(config.warmup_searches):
            game_state = RealGameState(config.game_type)

            # Add a few random moves for variety
            legal_moves = game_state.get_legal_moves()
            if legal_moves and not game_state.is_terminal():
                num_moves = min(np.random.randint(1, 4), len(legal_moves))
                for _ in range(num_moves):
                    if game_state.is_terminal():
                        break
                    legal_moves = game_state.get_legal_moves()
                    if legal_moves:
                        move = np.random.choice(legal_moves)
                        game_state.apply_move(move)

            request = SearchRequest(
                request_id=f"warmup_{i}",
                game_state=game_state,
                simulations=config.simulations_per_search // 2,  # Shorter warmup
                temperature=1.0
            )

            future = coordinator.submit_search(request)
            try:
                future.result(timeout=5.0)
            except:
                pass  # Ignore warmup failures

        # Small delay to let system stabilize
        time.sleep(0.5)

    def optimize_thread_count(self,
                            game_type: str = "gomoku",
                            simulations: int = 800,
                            iterations: int = 50,
                            max_threads: int = 16,
                            quick_test: bool = False) -> OptimizationReport:
        """Run complete thread count optimization."""

        start_time = time.time()
        logger.info(f"Starting thread count optimization for {game_type}")
        logger.info(f"Parameters: {simulations} simulations, {iterations} iterations, max {max_threads} threads")

        if quick_test:
            iterations = max(10, iterations // 5)
            simulations = max(100, simulations // 4)
            logger.info(f"Quick test mode: {simulations} simulations, {iterations} iterations")

        # Test thread counts from 1 to max_threads
        thread_counts = list(range(1, min(max_threads + 1, self.cpu_count * 2)))
        results = []

        for thread_count in thread_counts:
            config = ThreadTestConfig(
                thread_count=thread_count,
                game_type=game_type,
                simulations_per_search=simulations,
                num_searches=iterations,
                warmup_searches=max(5, iterations // 10),
                timeout_seconds=120.0 if not quick_test else 30.0
            )

            result = self.run_thread_count_test(config)
            results.append(result)

            # Early stopping if performance is clearly degrading
            if len(results) >= 3 and all(r.efficiency_score() < 0.1 for r in results[-2:]):
                logger.info("Performance degraded significantly, stopping early")
                break

        # Find optimal configuration
        valid_results = [r for r in results if r.success_rate >= 0.8]
        if not valid_results:
            logger.error("No valid results found!")
            valid_results = results  # Use all results as fallback

        optimal_result = max(valid_results, key=lambda r: r.efficiency_score())
        optimal_thread_count = optimal_result.thread_count

        # Generate performance curve
        performance_curve = [(r.thread_count, r.efficiency_score()) for r in results]

        # Generate recommendations
        recommendations = self._generate_recommendations(results, optimal_result)

        # Create optimization report
        report = OptimizationReport(
            test_config={
                'game_type': game_type,
                'simulations': simulations,
                'iterations': iterations,
                'max_threads': max_threads,
                'quick_test': quick_test
            },
            results=results,
            optimal_thread_count=optimal_thread_count,
            optimal_result=optimal_result,
            performance_curve=performance_curve,
            recommendations=recommendations,
            system_info=self.system_info,
            test_duration_seconds=time.time() - start_time
        )

        logger.info(f"Optimization completed in {report.test_duration_seconds:.1f}s")
        logger.info(f"Optimal thread count: {optimal_thread_count}")
        logger.info(f"Peak performance: {optimal_result.searches_per_second:.1f} searches/sec")
        logger.info(f"Contention score: {optimal_result.contention_score:.1f}%")

        try:
            filename = f"thread_optimization_{int(time.time() * 1000)}.json"
            self.save_report(report, filename)
        except Exception as save_error:
            logger.warning(f"Failed to save optimization report: {save_error}")

        return report

    def _generate_recommendations(self,
                                results: List[ThreadPerformanceResult],
                                optimal_result: ThreadPerformanceResult) -> List[str]:
        """Generate optimization recommendations based on results."""

        recommendations = []

        # Optimal thread count recommendation
        recommendations.append(
            f"Use {optimal_result.thread_count} threads for optimal performance "
            f"({optimal_result.searches_per_second:.1f} searches/sec)"
        )

        # Contention analysis
        if optimal_result.contention_score < 10.0:
            recommendations.append(
                f"Low contention detected ({optimal_result.contention_score:.1f}%) - "
                f"good thread synchronization"
            )
        elif optimal_result.contention_score < 25.0:
            recommendations.append(
                f"Moderate contention ({optimal_result.contention_score:.1f}%) - "
                f"consider reducing thread count if stability is more important than throughput"
            )
        else:
            recommendations.append(
                f"High contention detected ({optimal_result.contention_score:.1f}%) - "
                f"consider using fewer threads or optimizing synchronization"
            )

        # Thread utilization analysis
        if optimal_result.thread_utilization_percent < 70.0:
            recommendations.append(
                f"Thread utilization is low ({optimal_result.thread_utilization_percent:.1f}%) - "
                f"consider increasing workload or reducing thread count"
            )
        elif optimal_result.thread_utilization_percent > 95.0:
            recommendations.append(
                f"Thread utilization is very high ({optimal_result.thread_utilization_percent:.1f}%) - "
                f"system may be overloaded"
            )

        # CPU utilization analysis
        if optimal_result.cpu_utilization_percent < 50.0:
            recommendations.append(
                f"CPU utilization is low ({optimal_result.cpu_utilization_percent:.1f}%) - "
                f"could potentially handle more threads"
            )

        # Scaling analysis
        single_thread_result = next((r for r in results if r.thread_count == 1), None)
        if single_thread_result and optimal_result.thread_count > 1:
            scaling_efficiency = (optimal_result.searches_per_second /
                                 (single_thread_result.searches_per_second * optimal_result.thread_count))
            recommendations.append(
                f"Threading efficiency: {scaling_efficiency:.1%} "
                f"({optimal_result.searches_per_second / single_thread_result.searches_per_second:.1f}x speedup "
                f"with {optimal_result.thread_count}x threads)"
            )

        # Hardware-specific recommendations
        if optimal_result.thread_count > self.cpu_count:
            recommendations.append(
                f"Optimal thread count ({optimal_result.thread_count}) exceeds CPU cores ({self.cpu_count}) - "
                f"benefits from hyperthreading or I/O overlap"
            )
        elif optimal_result.thread_count == self.cpu_count:
            recommendations.append("Optimal thread count matches CPU core count - good CPU-bound scaling")

        return recommendations

    def save_report(self, report: OptimizationReport, filename: str = None) -> Path:
        """Save optimization report to file."""

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"thread_optimization_{timestamp}.json"

        output_path = self.output_dir / filename

        # Convert to serializable format
        report_dict = asdict(report)

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)

        logger.info(f"Optimization report saved to {output_path}")
        return output_path

    def plot_results(self, report: OptimizationReport, save_plots: bool = True) -> Optional[Path]:
        """Create visualization plots of optimization results."""

        if not self.enable_plotting:
            logger.warning("Plotting disabled - matplotlib/seaborn not available")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Thread Count Optimization Results', fontsize=16)

        # Plot 1: Throughput vs Thread Count
        thread_counts = [r.thread_count for r in report.results]
        throughputs = [r.searches_per_second for r in report.results]

        axes[0, 0].plot(thread_counts, throughputs, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].axvline(report.optimal_thread_count, color='r', linestyle='--',
                          label=f'Optimal: {report.optimal_thread_count} threads')
        axes[0, 0].set_xlabel('Thread Count')
        axes[0, 0].set_ylabel('Searches/Second')
        axes[0, 0].set_title('Search Throughput vs Thread Count')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Plot 2: Contention Score vs Thread Count
        contention_scores = [r.contention_score for r in report.results]

        axes[0, 1].plot(thread_counts, contention_scores, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].axhline(10.0, color='g', linestyle='--', alpha=0.7, label='Target: <10%')
        axes[0, 1].axvline(report.optimal_thread_count, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Thread Count')
        axes[0, 1].set_ylabel('Contention Score (%)')
        axes[0, 1].set_title('Thread Contention vs Thread Count')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Plot 3: Efficiency Score vs Thread Count
        efficiency_scores = [r.efficiency_score() for r in report.results]

        axes[1, 0].plot(thread_counts, efficiency_scores, 'go-', linewidth=2, markersize=8)
        axes[1, 0].axvline(report.optimal_thread_count, color='r', linestyle='--',
                          label=f'Optimal: {report.optimal_result.efficiency_score():.3f}')
        axes[1, 0].set_xlabel('Thread Count')
        axes[1, 0].set_ylabel('Efficiency Score')
        axes[1, 0].set_title('Overall Efficiency vs Thread Count')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Plot 4: Resource Utilization
        thread_utils = [r.thread_utilization_percent for r in report.results]
        cpu_utils = [r.cpu_utilization_percent for r in report.results]

        axes[1, 1].plot(thread_counts, thread_utils, 'bo-', label='Thread Utilization', linewidth=2)
        axes[1, 1].plot(thread_counts, cpu_utils, 'ro-', label='CPU Utilization', linewidth=2)
        axes[1, 1].axvline(report.optimal_thread_count, color='g', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Thread Count')
        axes[1, 1].set_ylabel('Utilization (%)')
        axes[1, 1].set_title('Resource Utilization vs Thread Count')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()

        if save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"thread_optimization_plots_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Plots saved to {plot_path}")
            return plot_path
        else:
            plt.show()
            return None

    def print_summary(self, report: OptimizationReport):
        """Print optimization summary to console."""

        print("\n" + "="*80)
        print("THREAD COUNT OPTIMIZATION SUMMARY")
        print("="*80)

        print(f"\nTest Configuration:")
        print(f"  Game Type: {report.test_config['game_type']}")
        print(f"  Simulations per Search: {report.test_config['simulations']}")
        print(f"  Iterations: {report.test_config['iterations']}")
        print(f"  Test Duration: {report.test_duration_seconds:.1f}s")

        print(f"\nSystem Information:")
        print(f"  CPU: {report.system_info['cpu_info']}")
        print(f"  Memory: {report.system_info['memory_total_gb']:.1f} GB")

        print(f"\nOptimal Configuration:")
        print(f"  Thread Count: {report.optimal_thread_count}")
        print(f"  Throughput: {report.optimal_result.searches_per_second:.1f} searches/sec")
        print(f"  Average Search Time: {report.optimal_result.average_search_time_ms:.1f}ms")
        print(f"  Contention Score: {report.optimal_result.contention_score:.1f}%")
        print(f"  Thread Utilization: {report.optimal_result.thread_utilization_percent:.1f}%")
        print(f"  Efficiency Score: {report.optimal_result.efficiency_score():.3f}")

        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

        print(f"\nDetailed Results:")
        print(f"{'Threads':<8} {'Throughput':<12} {'Avg Time':<10} {'Contention':<11} {'Efficiency':<10}")
        print(f"{'Count':<8} {'(search/s)':<12} {'(ms)':<10} {'(%)':<11} {'Score':<10}")
        print("-" * 60)

        for result in report.results:
            print(f"{result.thread_count:<8} "
                  f"{result.searches_per_second:<12.1f} "
                  f"{result.average_search_time_ms:<10.1f} "
                  f"{result.contention_score:<11.1f} "
                  f"{result.efficiency_score():<10.3f}")

        print("="*80)


def main():
    """Main function for thread count optimization."""

    parser = argparse.ArgumentParser(
        description="Optimize MCTS thread count for maximum performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/tune_threads.py --quick-test
  python scripts/tune_threads.py --game gomoku --simulations 1000 --iterations 50
  python scripts/tune_threads.py --full-sweep --output results/
  python scripts/tune_threads.py --max-threads 12 --no-plots
        """
    )

    parser.add_argument('--game', default='gomoku', choices=['gomoku', 'chess', 'go'],
                        help='Game type to optimize for (default: gomoku)')
    parser.add_argument('--simulations', type=int, default=800,
                        help='MCTS simulations per search (default: 800)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of searches per thread count test (default: 50)')
    parser.add_argument('--max-threads', type=int, default=16,
                        help='Maximum thread count to test (default: 16)')
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
        args.simulations = max(args.simulations, 1200)
        args.iterations = max(args.iterations, 100)
        args.max_threads = min(args.max_threads, multiprocessing.cpu_count() * 3)
        logger.info("Full sweep mode enabled - using maximum parameters")

    try:
        # Create optimizer
        optimizer = ThreadOptimizer(
            output_dir=args.output,
            enable_plotting=not args.no_plots
        )

        # Run optimization
        report = optimizer.optimize_thread_count(
            game_type=args.game,
            simulations=args.simulations,
            iterations=args.iterations,
            max_threads=args.max_threads,
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
        if report.optimal_result.contention_score < 10.0 and report.optimal_result.searches_per_second > 0:
            logger.info("Optimization successful - target criteria met")
            sys.exit(0)
        else:
            logger.warning("Optimization completed but target criteria not fully met")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
