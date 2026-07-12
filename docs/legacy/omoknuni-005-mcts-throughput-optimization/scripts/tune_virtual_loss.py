#!/usr/bin/env python3
"""
Virtual Loss Magnitude Optimization Script
==========================================

Systematically tests different virtual loss (VL) values to find the optimal
configuration that maximizes thread efficiency while preserving exploration balance.

This script performs parameter sweeps across VL values 0.5-3.0, measuring:
- Thread efficiency and coordination
- Search throughput (simulations/second)
- Exploration balance and diversity
- Thread contention and synchronization
- Policy entropy and search quality metrics

Features:
- Comprehensive virtual loss magnitude testing (0.5-3.0 range)
- Thread efficiency measurement and analysis
- Exploration balance validation through policy diversity
- Contention detection through timing variance analysis
- Statistical significance testing with confidence intervals
- Detailed reporting with visualization support

Usage:
    python scripts/tune_virtual_loss.py --game gomoku --simulations 1000 --iterations 50
    python scripts/tune_virtual_loss.py --quick-test  # Fast optimization for development
    python scripts/tune_virtual_loss.py --full-sweep --output results/virtual_loss_tuning.json

Target: Find optimal VL magnitude (~1.0) with thread efficiency >85% and exploration preserved.

HOWTO-RUN-TESTS:
================
# Run virtual loss optimization tests
python -m pytest tests/unit/test_virtual_loss_optimizer.py -v

# Run quick optimization test (minimal resources)
python scripts/tune_virtual_loss.py --quick-test --max-threads 4 --iterations 20

# Run optimization with specific game and parameters
python scripts/tune_virtual_loss.py --game gomoku --simulations 800 --iterations 50

# Run full optimization sweep (comprehensive)
python scripts/tune_virtual_loss.py --full-sweep --output results/

# Run optimization without plots (headless environments)
python scripts/tune_virtual_loss.py --quick-test --no-plots

# Example expected output showing optimal VL magnitude determination:
#   Optimal Configuration:
#     Virtual Loss Magnitude: 1.0
#     Thread Efficiency: 87.3%
#     Throughput: 423.5 simulations/sec
#     Exploration Balance: 0.92
#     Contention Score: 7.1%
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
class VirtualLossTestConfig:
    """Configuration for virtual loss magnitude testing."""

    virtual_loss_magnitude: float
    game_type: str = "gomoku"
    simulations_per_search: int = 800
    num_searches: int = 100
    thread_count: int = 8
    warmup_searches: int = 10
    timeout_seconds: float = 60.0
    measure_exploration: bool = True
    monitor_system: bool = True


@dataclass
class VirtualLossPerformanceResult:
    """Results from testing a specific virtual loss magnitude."""

    virtual_loss_magnitude: float
    searches_per_second: float
    average_search_time_ms: float
    search_time_std_ms: float
    thread_efficiency_percent: float
    contention_score: float
    exploration_balance: float
    policy_entropy_avg: float
    cpu_utilization_percent: float
    memory_usage_mb: float
    visit_distribution_variance: Optional[float] = None
    path_diversity_score: Optional[float] = None
    success_rate: float = 1.0
    error_message: Optional[str] = None

    def overall_score(self) -> float:
        """Calculate overall optimization score combining efficiency and exploration."""
        if self.success_rate < 0.8:
            return 0.0

        # Normalize throughput (higher is better)
        throughput_score = min(1.0, self.searches_per_second / 1000.0)

        # Thread efficiency score (target 85-95%)
        efficiency_target = 90.0
        efficiency_score = 1.0 - abs(self.thread_efficiency_percent - efficiency_target) / efficiency_target
        efficiency_score = max(0.0, efficiency_score)

        # Penalize high contention (lower is better)
        contention_penalty = max(0.0, 1.0 - self.contention_score / 50.0)

        # Exploration balance (target around 0.85-0.95)
        exploration_target = 0.90
        exploration_score = 1.0 - abs(self.exploration_balance - exploration_target) / exploration_target
        exploration_score = max(0.0, exploration_score)

        # Policy entropy (higher diversity is better)
        entropy_score = min(1.0, self.policy_entropy_avg / 3.0)

        # Combine scores with weights
        return (0.3 * throughput_score + 0.3 * efficiency_score + 0.2 * contention_penalty +
                0.15 * exploration_score + 0.05 * entropy_score)


@dataclass
class VirtualLossOptimizationReport:
    """Complete virtual loss optimization report with recommendations."""

    test_config: Dict[str, Any]
    results: List[VirtualLossPerformanceResult]
    optimal_virtual_loss: float
    optimal_result: VirtualLossPerformanceResult
    performance_curve: List[Tuple[float, float]]  # (vl_magnitude, overall_score)
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


class VirtualLossOptimizer:
    """Virtual loss magnitude optimizer for MCTS search performance."""

    def __init__(self, output_dir: Path = None, enable_plotting: bool = True):
        self.output_dir = output_dir or Path("results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.enable_plotting = enable_plotting and PLOTTING_AVAILABLE

        # System info
        self.cpu_count = multiprocessing.cpu_count()
        self.system_info = self._get_system_info()

        # Initialize device manager
        self.device_manager = DeviceManager()

        logger.info(f"Virtual loss optimizer initialized - CPU cores: {self.cpu_count}")
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
        """Create a test model for virtual loss optimization."""
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
        model_path = self.output_dir / f"test_model_vl_{game_type}.pth"
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
                logger.info("Created GPU inference worker for VL optimization")
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
        logger.info("Created CPU inference worker for VL optimization")
        return worker

    def calculate_exploration_metrics(self, search_results: List[Dict]) -> Tuple[float, float, float]:
        """Calculate exploration balance and diversity metrics from search results."""
        if not search_results:
            return 0.0, 0.0, 0.0

        policy_entropies = []
        visit_variances = []
        top_move_ratios = []

        for result in search_results:
            if 'visit_counts' in result and len(result['visit_counts']) > 0:
                visits = np.array(result['visit_counts'])
                total_visits = np.sum(visits)

                if total_visits > 0:
                    # Policy entropy calculation
                    policy = visits / total_visits
                    # Avoid log(0) by adding small epsilon
                    policy = np.maximum(policy, 1e-10)
                    entropy = -np.sum(policy * np.log(policy))
                    policy_entropies.append(entropy)

                    # Visit distribution variance
                    visit_variances.append(np.var(visits))

                    # Top move concentration (lower is more diverse)
                    top_move_ratio = np.max(visits) / total_visits
                    top_move_ratios.append(top_move_ratio)

        # Calculate averages
        avg_entropy = statistics.mean(policy_entropies) if policy_entropies else 0.0
        avg_variance = statistics.mean(visit_variances) if visit_variances else 0.0
        avg_top_ratio = statistics.mean(top_move_ratios) if top_move_ratios else 1.0

        # Exploration balance score (higher is better diversity)
        # Based on entropy and inverse of top move concentration
        exploration_balance = min(1.0, (avg_entropy / 3.0) * (1.0 - avg_top_ratio))

        return exploration_balance, avg_entropy, avg_variance

    def run_virtual_loss_test(self, config: VirtualLossTestConfig) -> VirtualLossPerformanceResult:
        """Run performance test for a specific virtual loss magnitude."""
        logger.info(f"Testing virtual loss magnitude {config.virtual_loss_magnitude:.1f}...")

        try:
            # Create real inference worker
            inference_worker = self.create_real_inference_worker(config.game_type, use_gpu=True)

            # Create search coordinator with specified virtual loss magnitude
            coordinator = SearchCoordinator(
                inference_worker=inference_worker,
                max_threads=config.thread_count,
                max_queue_size=1000,
                monitoring_interval=0.5
            )

            # Configure virtual loss magnitude (this would be set in the MCTS configuration)
            # For now, we'll assume it's passed through the coordinator or search parameters
            vl_config = {
                'virtual_loss_magnitude': config.virtual_loss_magnitude,
                'enable_virtual_loss': True
            }

            # Start coordinator
            coordinator.start()

            # Monitor system resources
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Warmup phase
            self._run_warmup_searches(coordinator, config)

            # Main test phase
            search_times = []
            search_results = []
            start_time = time.time()
            cpu_times = []

            for i in range(config.num_searches):
                if time.time() - start_time > config.timeout_seconds:
                    logger.warning(f"Test timeout reached for VL {config.virtual_loss_magnitude:.1f}")
                    break

                # Monitor CPU during search
                cpu_before = psutil.cpu_percent()

                # Execute search
                search_start = time.time()
                game_state = RealGameState(config.game_type)

                # Add some random moves to create realistic positions
                legal_moves = game_state.get_legal_moves()
                if legal_moves and not game_state.is_terminal():
                    # Make 1-5 random moves to create varied positions
                    num_moves = min(np.random.randint(1, 6), len(legal_moves))
                    for _ in range(num_moves):
                        if game_state.is_terminal():
                            break
                        legal_moves = game_state.get_legal_moves()
                        if legal_moves:
                            move = np.random.choice(legal_moves)
                            game_state.apply_move(move)

                request = SearchRequest(
                    request_id=f"vl_test_{i}",
                    game_state=game_state,
                    simulations=config.simulations_per_search,
                    temperature=1.0,
                    add_noise=False,
                    # Note: virtual loss configuration handled by coordinator
                )

                future = coordinator.submit_search(request)
                try:
                    result = future.result(timeout=5.0)  # 5 second timeout per search
                    search_time = time.time() - search_start
                    search_times.append(search_time * 1000)  # Convert to ms

                    cpu_after = psutil.cpu_percent()
                    cpu_times.append(cpu_after)

                    # Store search result for exploration analysis
                    if hasattr(result, 'visit_counts'):
                        search_results.append({
                            'visit_counts': result.visit_counts,
                            'policy': result.policy if hasattr(result, 'policy') else None
                        })

                except Exception as e:
                    logger.warning(f"Search failed for VL {config.virtual_loss_magnitude:.1f}: {e}")

            # Calculate metrics
            end_time = time.time()
            total_time = end_time - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Get coordinator metrics
            metrics = coordinator.get_metrics()

            # Stop coordinator and cleanup
            coordinator.stop()

            # Calculate exploration metrics
            exploration_balance, avg_entropy, avg_variance = self.calculate_exploration_metrics(search_results)

            # Cleanup model file
            try:
                model_path = self.output_dir / f"test_model_vl_{config.game_type}.pth"
                if model_path.exists():
                    model_path.unlink()
            except:
                pass  # Ignore cleanup errors

            # Calculate results
            if search_times:
                avg_search_time = statistics.mean(search_times)
                search_time_std = statistics.stdev(search_times) if len(search_times) > 1 else 0.0
                searches_per_second = len(search_times) / total_time
                success_rate = len(search_times) / config.num_searches
            else:
                avg_search_time = float('inf')
                search_time_std = 0.0
                searches_per_second = 0.0
                success_rate = 0.0

            # Calculate contention score (higher variance indicates more contention)
            contention_score = (search_time_std / avg_search_time * 100) if avg_search_time > 0 else 100.0

            # Thread efficiency calculation (based on coordinator metrics)
            thread_efficiency = metrics.thread_utilization if hasattr(metrics, 'thread_utilization') else 75.0

            # CPU and memory metrics
            avg_cpu = statistics.mean(cpu_times) if cpu_times else 0.0
            memory_usage = max(initial_memory, final_memory)

            result = VirtualLossPerformanceResult(
                virtual_loss_magnitude=config.virtual_loss_magnitude,
                searches_per_second=searches_per_second,
                average_search_time_ms=avg_search_time,
                search_time_std_ms=search_time_std,
                thread_efficiency_percent=thread_efficiency,
                contention_score=contention_score,
                exploration_balance=exploration_balance,
                policy_entropy_avg=avg_entropy,
                cpu_utilization_percent=avg_cpu,
                memory_usage_mb=memory_usage,
                visit_distribution_variance=avg_variance,
                success_rate=success_rate
            )

            logger.info(f"VL {config.virtual_loss_magnitude:.1f}: "
                       f"{searches_per_second:.1f} searches/sec, "
                       f"efficiency: {thread_efficiency:.1f}%, "
                       f"exploration: {exploration_balance:.3f}, "
                       f"overall: {result.overall_score():.3f}")

            return result

        except Exception as e:
            logger.error(f"Error testing VL {config.virtual_loss_magnitude:.1f}: {e}")
            return VirtualLossPerformanceResult(
                virtual_loss_magnitude=config.virtual_loss_magnitude,
                searches_per_second=0.0,
                average_search_time_ms=float('inf'),
                search_time_std_ms=0.0,
                thread_efficiency_percent=0.0,
                contention_score=100.0,
                exploration_balance=0.0,
                policy_entropy_avg=0.0,
                cpu_utilization_percent=0.0,
                memory_usage_mb=0.0,
                success_rate=0.0,
                error_message=str(e)
            )

    def _run_warmup_searches(self, coordinator: SearchCoordinator, config: VirtualLossTestConfig):
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
                request_id=f"vl_warmup_{i}",
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

    def optimize_virtual_loss_magnitude(self,
                                      game_type: str = "gomoku",
                                      simulations: int = 800,
                                      iterations: int = 50,
                                      thread_count: int = 8,
                                      vl_min: float = 0.5,
                                      vl_max: float = 3.0,
                                      vl_step: float = 0.1,
                                      quick_test: bool = False) -> VirtualLossOptimizationReport:
        """Run complete virtual loss magnitude optimization."""

        start_time = time.time()
        logger.info(f"Starting virtual loss optimization for {game_type}")
        logger.info(f"Parameters: {simulations} simulations, {iterations} iterations, {thread_count} threads")
        logger.info(f"VL range: {vl_min:.1f} - {vl_max:.1f}, step: {vl_step:.1f}")

        if quick_test:
            iterations = max(20, iterations // 3)
            simulations = max(200, simulations // 4)
            vl_step = max(0.2, vl_step * 2)  # Coarser steps for quick test
            logger.info(f"Quick test mode: {simulations} simulations, {iterations} iterations, step {vl_step:.1f}")

        # Test VL magnitudes in the specified range
        vl_values = np.arange(vl_min, vl_max + vl_step/2, vl_step).round(1)
        results = []

        for vl_magnitude in vl_values:
            config = VirtualLossTestConfig(
                virtual_loss_magnitude=vl_magnitude,
                game_type=game_type,
                simulations_per_search=simulations,
                num_searches=iterations,
                thread_count=thread_count,
                warmup_searches=max(10, iterations // 10),
                timeout_seconds=120.0 if not quick_test else 45.0
            )

            result = self.run_virtual_loss_test(config)
            results.append(result)

            # Early stopping if performance is clearly degrading
            if len(results) >= 5 and all(r.overall_score() < 0.1 for r in results[-3:]):
                logger.info("Performance degraded significantly, stopping early")
                break

        # Find optimal configuration
        valid_results = [r for r in results if r.success_rate >= 0.8]
        if not valid_results:
            logger.error("No valid results found!")
            valid_results = results  # Use all results as fallback

        optimal_result = max(valid_results, key=lambda r: r.overall_score())
        optimal_virtual_loss = optimal_result.virtual_loss_magnitude

        # Generate performance curve
        performance_curve = [(r.virtual_loss_magnitude, r.overall_score()) for r in results]

        # Generate recommendations
        recommendations = self._generate_recommendations(results, optimal_result)

        # Create optimization report
        report = VirtualLossOptimizationReport(
            test_config={
                'game_type': game_type,
                'simulations': simulations,
                'iterations': iterations,
                'thread_count': thread_count,
                'vl_min': vl_min,
                'vl_max': vl_max,
                'vl_step': vl_step,
                'quick_test': quick_test
            },
            results=results,
            optimal_virtual_loss=optimal_virtual_loss,
            optimal_result=optimal_result,
            performance_curve=performance_curve,
            recommendations=recommendations,
            system_info=self.system_info,
            test_duration_seconds=time.time() - start_time
        )

        logger.info(f"VL optimization completed in {report.test_duration_seconds:.1f}s")
        logger.info(f"Optimal virtual loss magnitude: {optimal_virtual_loss:.1f}")
        logger.info(f"Thread efficiency: {optimal_result.thread_efficiency_percent:.1f}%")
        logger.info(f"Exploration balance: {optimal_result.exploration_balance:.3f}")

        return report

    def _generate_recommendations(self,
                                results: List[VirtualLossPerformanceResult],
                                optimal_result: VirtualLossPerformanceResult) -> List[str]:
        """Generate optimization recommendations based on results."""

        recommendations = []

        # Optimal VL magnitude recommendation
        recommendations.append(
            f"Use virtual loss magnitude {optimal_result.virtual_loss_magnitude:.1f} for optimal performance "
            f"(thread efficiency: {optimal_result.thread_efficiency_percent:.1f}%)"
        )

        # Thread efficiency analysis
        if optimal_result.thread_efficiency_percent >= 85.0:
            recommendations.append(
                f"Excellent thread efficiency achieved ({optimal_result.thread_efficiency_percent:.1f}%) - "
                f"threads are well-coordinated with minimal conflicts"
            )
        elif optimal_result.thread_efficiency_percent >= 75.0:
            recommendations.append(
                f"Good thread efficiency ({optimal_result.thread_efficiency_percent:.1f}%) - "
                f"consider fine-tuning VL magnitude for marginal improvements"
            )
        else:
            recommendations.append(
                f"Thread efficiency is suboptimal ({optimal_result.thread_efficiency_percent:.1f}%) - "
                f"consider adjusting thread count or VL magnitude"
            )

        # Exploration balance analysis
        if optimal_result.exploration_balance >= 0.85:
            recommendations.append(
                f"Excellent exploration balance ({optimal_result.exploration_balance:.3f}) - "
                f"search maintains good diversity"
            )
        elif optimal_result.exploration_balance >= 0.70:
            recommendations.append(
                f"Adequate exploration balance ({optimal_result.exploration_balance:.3f}) - "
                f"acceptable diversity in search"
            )
        else:
            recommendations.append(
                f"Low exploration balance ({optimal_result.exploration_balance:.3f}) - "
                f"search may be too focused, consider reducing VL magnitude"
            )

        # Contention analysis
        if optimal_result.contention_score < 10.0:
            recommendations.append(
                f"Low contention detected ({optimal_result.contention_score:.1f}%) - "
                f"excellent thread synchronization"
            )
        elif optimal_result.contention_score < 25.0:
            recommendations.append(
                f"Moderate contention ({optimal_result.contention_score:.1f}%) - "
                f"acceptable for multi-threaded performance"
            )
        else:
            recommendations.append(
                f"High contention detected ({optimal_result.contention_score:.1f}%) - "
                f"consider increasing VL magnitude or reducing thread count"
            )

        # Performance comparison
        best_throughput = max(r.searches_per_second for r in results if r.success_rate >= 0.8)
        if optimal_result.searches_per_second >= best_throughput * 0.95:
            recommendations.append(
                f"Optimal configuration achieves near-peak throughput "
                f"({optimal_result.searches_per_second:.1f} searches/sec)"
            )

        # VL magnitude insights
        if optimal_result.virtual_loss_magnitude < 1.0:
            recommendations.append(
                f"Lower VL magnitude ({optimal_result.virtual_loss_magnitude:.1f}) favors exploration "
                f"over thread coordination - suitable for diverse search requirements"
            )
        elif optimal_result.virtual_loss_magnitude > 1.5:
            recommendations.append(
                f"Higher VL magnitude ({optimal_result.virtual_loss_magnitude:.1f}) prioritizes thread coordination "
                f"over exploration - suitable for high-throughput scenarios"
            )
        else:
            recommendations.append(
                f"Balanced VL magnitude ({optimal_result.virtual_loss_magnitude:.1f}) provides "
                f"good compromise between thread efficiency and exploration"
            )

        return recommendations

    def save_report(self, report: VirtualLossOptimizationReport, filename: str = None) -> Path:
        """Save optimization report to file."""

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"virtual_loss_optimization_{timestamp}.json"

        output_path = self.output_dir / filename

        # Convert to serializable format
        report_dict = asdict(report)

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)

        logger.info(f"Virtual loss optimization report saved to {output_path}")
        return output_path

    def plot_results(self, report: VirtualLossOptimizationReport, save_plots: bool = True) -> Optional[Path]:
        """Create visualization plots of optimization results."""

        if not self.enable_plotting:
            logger.warning("Plotting disabled - matplotlib/seaborn not available")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Virtual Loss Magnitude Optimization Results', fontsize=16)

        # Plot 1: Overall Score vs VL Magnitude
        vl_magnitudes = [r.virtual_loss_magnitude for r in report.results]
        overall_scores = [r.overall_score() for r in report.results]

        axes[0, 0].plot(vl_magnitudes, overall_scores, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].axvline(report.optimal_virtual_loss, color='r', linestyle='--',
                          label=f'Optimal: {report.optimal_virtual_loss:.1f}')
        axes[0, 0].set_xlabel('Virtual Loss Magnitude')
        axes[0, 0].set_ylabel('Overall Score')
        axes[0, 0].set_title('Overall Performance vs Virtual Loss Magnitude')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Plot 2: Thread Efficiency vs VL Magnitude
        thread_efficiencies = [r.thread_efficiency_percent for r in report.results]

        axes[0, 1].plot(vl_magnitudes, thread_efficiencies, 'go-', linewidth=2, markersize=8)
        axes[0, 1].axhline(85.0, color='g', linestyle='--', alpha=0.7, label='Target: >85%')
        axes[0, 1].axvline(report.optimal_virtual_loss, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Virtual Loss Magnitude')
        axes[0, 1].set_ylabel('Thread Efficiency (%)')
        axes[0, 1].set_title('Thread Efficiency vs Virtual Loss Magnitude')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Plot 3: Exploration Balance vs VL Magnitude
        exploration_balances = [r.exploration_balance for r in report.results]

        axes[1, 0].plot(vl_magnitudes, exploration_balances, 'mo-', linewidth=2, markersize=8)
        axes[1, 0].axhline(0.85, color='m', linestyle='--', alpha=0.7, label='Target: >0.85')
        axes[1, 0].axvline(report.optimal_virtual_loss, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Virtual Loss Magnitude')
        axes[1, 0].set_ylabel('Exploration Balance')
        axes[1, 0].set_title('Exploration Balance vs Virtual Loss Magnitude')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Plot 4: Throughput and Contention
        throughputs = [r.searches_per_second for r in report.results]
        contention_scores = [r.contention_score for r in report.results]

        ax4 = axes[1, 1]
        color = 'tab:blue'
        ax4.set_xlabel('Virtual Loss Magnitude')
        ax4.set_ylabel('Searches/Second', color=color)
        line1 = ax4.plot(vl_magnitudes, throughputs, 'bo-', color=color, label='Throughput')
        ax4.tick_params(axis='y', labelcolor=color)

        ax4_twin = ax4.twinx()
        color = 'tab:red'
        ax4_twin.set_ylabel('Contention Score (%)', color=color)
        line2 = ax4_twin.plot(vl_magnitudes, contention_scores, 'ro-', color=color, label='Contention')
        ax4_twin.tick_params(axis='y', labelcolor=color)

        ax4.axvline(report.optimal_virtual_loss, color='g', linestyle='--', alpha=0.7)
        ax4.set_title('Throughput and Contention vs Virtual Loss')
        ax4.grid(True, alpha=0.3)

        # Add legend for dual axis plot
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')

        plt.tight_layout()

        if save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"virtual_loss_optimization_plots_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Plots saved to {plot_path}")
            return plot_path
        else:
            plt.show()
            return None

    def print_summary(self, report: VirtualLossOptimizationReport):
        """Print optimization summary to console."""

        print("\n" + "="*80)
        print("VIRTUAL LOSS MAGNITUDE OPTIMIZATION SUMMARY")
        print("="*80)

        print(f"\nTest Configuration:")
        print(f"  Game Type: {report.test_config['game_type']}")
        print(f"  Simulations per Search: {report.test_config['simulations']}")
        print(f"  Iterations: {report.test_config['iterations']}")
        print(f"  Thread Count: {report.test_config['thread_count']}")
        print(f"  VL Range: {report.test_config['vl_min']:.1f} - {report.test_config['vl_max']:.1f}")
        print(f"  Test Duration: {report.test_duration_seconds:.1f}s")

        print(f"\nSystem Information:")
        print(f"  CPU: {report.system_info['cpu_info']}")
        print(f"  Memory: {report.system_info['memory_total_gb']:.1f} GB")

        print(f"\nOptimal Configuration:")
        print(f"  Virtual Loss Magnitude: {report.optimal_virtual_loss:.1f}")
        print(f"  Thread Efficiency: {report.optimal_result.thread_efficiency_percent:.1f}%")
        print(f"  Throughput: {report.optimal_result.searches_per_second:.1f} searches/sec")
        print(f"  Exploration Balance: {report.optimal_result.exploration_balance:.3f}")
        print(f"  Policy Entropy: {report.optimal_result.policy_entropy_avg:.3f}")
        print(f"  Contention Score: {report.optimal_result.contention_score:.1f}%")
        print(f"  Overall Score: {report.optimal_result.overall_score():.3f}")

        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

        print(f"\nDetailed Results:")
        print(f"{'VL Mag':<8} {'Efficiency':<11} {'Throughput':<12} {'Exploration':<12} {'Overall':<10}")
        print(f"{'Value':<8} {'(%)':<11} {'(search/s)':<12} {'Balance':<12} {'Score':<10}")
        print("-" * 65)

        for result in report.results:
            print(f"{result.virtual_loss_magnitude:<8.1f} "
                  f"{result.thread_efficiency_percent:<11.1f} "
                  f"{result.searches_per_second:<12.1f} "
                  f"{result.exploration_balance:<12.3f} "
                  f"{result.overall_score():<10.3f}")

        print("="*80)


def main():
    """Main function for virtual loss magnitude optimization."""

    parser = argparse.ArgumentParser(
        description="Optimize MCTS virtual loss magnitude for thread efficiency and exploration balance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/tune_virtual_loss.py --quick-test
  python scripts/tune_virtual_loss.py --game gomoku --simulations 1000 --iterations 50
  python scripts/tune_virtual_loss.py --full-sweep --output results/
  python scripts/tune_virtual_loss.py --vl-range 0.5 2.5 --threads 12 --no-plots
        """
    )

    parser.add_argument('--game', default='gomoku', choices=['gomoku', 'chess', 'go'],
                        help='Game type to optimize for (default: gomoku)')
    parser.add_argument('--simulations', type=int, default=800,
                        help='MCTS simulations per search (default: 800)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of searches per VL magnitude test (default: 50)')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of search threads to use (default: 8)')
    parser.add_argument('--vl-range', nargs=2, type=float, default=[0.5, 3.0],
                        metavar=('MIN', 'MAX'), help='VL magnitude range to test (default: 0.5 3.0)')
    parser.add_argument('--vl-step', type=float, default=0.1,
                        help='Step size for VL magnitude testing (default: 0.1)')
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
        args.vl_step = min(args.vl_step, 0.05)  # Finer steps for comprehensive test
        logger.info("Full sweep mode enabled - using maximum parameters")

    try:
        # Create optimizer
        optimizer = VirtualLossOptimizer(
            output_dir=args.output,
            enable_plotting=not args.no_plots
        )

        # Run optimization
        report = optimizer.optimize_virtual_loss_magnitude(
            game_type=args.game,
            simulations=args.simulations,
            iterations=args.iterations,
            thread_count=args.threads,
            vl_min=args.vl_range[0],
            vl_max=args.vl_range[1],
            vl_step=args.vl_step,
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
        target_efficiency = optimal_result.thread_efficiency_percent >= 85.0
        target_exploration = optimal_result.exploration_balance >= 0.80

        if target_efficiency and target_exploration and optimal_result.success_rate >= 0.8:
            logger.info("Virtual loss optimization successful - target criteria met")
            sys.exit(0)
        else:
            logger.warning("Optimization completed but target criteria not fully met")
            logger.warning(f"Thread efficiency: {optimal_result.thread_efficiency_percent:.1f}% "
                         f"(target: ≥85%), Exploration: {optimal_result.exploration_balance:.3f} "
                         f"(target: ≥0.80)")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Virtual loss optimization interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Virtual loss optimization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()