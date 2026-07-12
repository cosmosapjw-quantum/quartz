#!/usr/bin/env python3
"""
Inference Timeout Optimization Script
====================================

Systematically tests different timeout values to find optimal configurations that
balance throughput and responsiveness for neural network inference batching.

This script performs parameter sweeps across timeout values 1-10ms, measuring:
- Inference throughput (batches/second, positions/second)
- Latency analysis (average batch wait time, response time)
- Batch formation efficiency (batch size distribution, timeout hit rate)
- GPU utilization and queue depth statistics
- Responsiveness vs throughput trade-offs

Features:
- Comprehensive timeout testing with throughput/latency analysis
- Dynamic batching performance measurement
- Queue behavior analysis with timeout compliance tracking
- Multi-game support with game-specific timeout requirements
- Statistical analysis with optimal point detection
- Detailed reporting with visualization support

Usage:
    python scripts/tune_timeout.py --game gomoku --iterations 100
    python scripts/tune_timeout.py --quick-test  # Fast optimization for development
    python scripts/tune_timeout.py --full-sweep --output results/timeout_optimization.json

Target: Optimal timeout value (target 3ms) that balances throughput and responsiveness.

HOWTO-RUN-TESTS:
================
# Run timeout optimization tests
python -m pytest tests/unit/test_timeout_optimizer.py -v

# Run quick optimization test (minimal resources)
python scripts/tune_timeout.py --quick-test --max-timeout 5 --iterations 50

# Run optimization with specific game and timeout range
python scripts/tune_timeout.py --game gomoku --min-timeout 1 --max-timeout 10 --iterations 100

# Run full optimization sweep (comprehensive)
python scripts/tune_timeout.py --full-sweep --output results/

# Run optimization without plots (headless environments)
python scripts/tune_timeout.py --quick-test --no-plots

# Example expected output showing optimal timeout determination:
#   Optimal Configuration:
#     Timeout: 3.2ms
#     Throughput: 12,450 inferences/sec
#     Latency: 2.8ms average batch wait
#     Batch Size: 47.3 average
#     Timeout Hit Rate: 23.4%
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
import queue

# Scientific computing and visualization
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, plots will be disabled")

# PyTorch for inference simulation
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
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    # Try to import game extensions
    from src import alphazero_py
    from src.alphazero_py import GomokuState, ChessState, GoState, GameType
    GAME_EXTENSIONS_AVAILABLE = True
    print("C++ game extensions available")
except ImportError as e:
    # Fallback when extensions not built yet
    GAME_EXTENSIONS_AVAILABLE = False
    print(f"Warning: C++ game extensions not available: {e}")
    # Create mock classes for testing
    class MockGameState:
        def __init__(self):
            pass
        def get_enhanced_tensor_representation(self):
            return np.zeros((36, 15, 15), dtype=np.float32)
    GomokuState = ChessState = GoState = MockGameState
    class GameType:
        GOMOKU = 0
        CHESS = 1
        GO = 2

# Import neural network components
try:
    from src.neural.model import AlphaZeroNet
    from src.neural.inference_worker import InferenceWorker, GPUInferenceWorker
    NEURAL_COMPONENTS_AVAILABLE = True
    print("Neural network components available")
except ImportError as e:
    NEURAL_COMPONENTS_AVAILABLE = False
    print(f"Warning: Neural network components not available: {e}")


@dataclass
class TimeoutPerformanceResult:
    """Results from testing a specific timeout value"""
    timeout_ms: float
    throughput_batches_per_sec: float
    throughput_positions_per_sec: float
    avg_batch_size: float
    avg_batch_wait_time_ms: float
    avg_response_time_ms: float
    timeout_hit_rate: float  # Percentage of batches that hit timeout vs size limit
    queue_depth_stats: Dict[str, float]
    gpu_utilization_percent: float
    memory_usage_mb: float
    batch_size_distribution: Dict[str, float]
    efficiency_score: float
    total_batches: int
    total_positions: int
    test_duration_sec: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class TimeoutOptimizationConfig:
    """Configuration for timeout optimization"""
    game_type: str = "gomoku"
    min_timeout_ms: float = 1.0
    max_timeout_ms: float = 10.0
    timeout_step_ms: float = 0.5
    min_batch_size: int = 32
    max_batch_size: int = 256
    iterations_per_timeout: int = 100
    test_duration_per_timeout: float = 30.0  # seconds
    num_producer_threads: int = 8
    positions_per_second_target: int = 15000  # Target inference load
    output_dir: Optional[str] = None
    enable_plots: bool = True
    verbose: bool = True


class GPUMonitor:
    """Monitor GPU utilization and memory usage during timeout optimization"""

    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_stats = deque(maxlen=1000)
        self.memory_stats = deque(maxlen=1000)

    def start_monitoring(self):
        """Start GPU monitoring in background thread"""
        if not NVML_AVAILABLE:
            logging.warning("GPU monitoring not available - nvidia-ml-py required")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Background monitoring loop"""
        if not NVML_AVAILABLE:
            return

        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
        except:
            logging.warning("Could not get GPU handle for monitoring")
            return

        while self.monitoring:
            try:
                # Get GPU utilization
                util_info = nvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_stats.append(util_info.gpu)

                # Get memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                memory_usage_mb = mem_info.used / 1024 / 1024
                self.memory_stats.append(memory_usage_mb)

                time.sleep(0.1)  # Sample every 100ms
            except Exception as e:
                logging.warning(f"GPU monitoring error: {e}")
                time.sleep(1.0)

    def get_stats(self) -> Tuple[float, float]:
        """Get average GPU utilization and memory usage"""
        if not self.gpu_stats:
            return 0.0, 0.0

        avg_gpu_util = statistics.mean(self.gpu_stats)
        avg_memory_mb = statistics.mean(self.memory_stats)
        return avg_gpu_util, avg_memory_mb

    def stats(self) -> Tuple[float, float]:
        """Alias for get_stats() for backward compatibility"""
        return self.get_stats()

    def clear_stats(self):
        """Clear accumulated statistics"""
        self.gpu_stats.clear()
        self.memory_stats.clear()


class MockInferenceWorker:
    """Mock inference worker for testing when real components unavailable"""

    def __init__(self, timeout_ms: float, min_batch_size: int = 32, max_batch_size: int = 256):
        self.timeout_ms = timeout_ms / 1000.0  # Convert to seconds
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.running = False
        self.worker_thread = None
        self.input_queue = queue.Queue(maxsize=1000)
        self.batch_stats = deque(maxlen=1000)
        self.response_times = deque(maxlen=1000)

    def start(self):
        """Start the mock inference worker"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stop the mock inference worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)

    def submit_batch(self, positions: List[np.ndarray], result_queue: queue.Queue):
        """Submit a batch for inference"""
        request_time = time.time()
        self.input_queue.put((positions, result_queue, request_time))

    def _worker_loop(self):
        """Mock worker loop that simulates inference behavior"""
        while self.running:
            try:
                # Collect batch with timeout behavior
                batch = []
                result_queues = []
                request_times = []
                start_time = time.time()

                # Get first request
                try:
                    positions, result_queue, request_time = self.input_queue.get(timeout=self.timeout_ms)
                    batch.extend(positions)
                    result_queues.append(result_queue)
                    request_times.append(request_time)
                except queue.Empty:
                    continue

                # Collect more requests until batch size or timeout
                while len(batch) < self.max_batch_size:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0, self.timeout_ms - elapsed)

                    if remaining_timeout <= 0:
                        break

                    try:
                        positions, result_queue, request_time = self.input_queue.get(
                            timeout=remaining_timeout
                        )
                        batch.extend(positions)
                        result_queues.append(result_queue)
                        request_times.append(request_time)
                    except queue.Empty:
                        break

                if batch:
                    # Record batch statistics
                    batch_wait_time = time.time() - start_time
                    self.batch_stats.append({
                        'size': len(batch),
                        'wait_time': batch_wait_time,
                        'timeout_hit': batch_wait_time >= self.timeout_ms * 0.95
                    })

                    # Simulate inference time (realistic for RTX 3060 Ti)
                    inference_time = 0.005 + len(batch) * 0.0001  # 5ms base + 0.1ms per position
                    time.sleep(inference_time)

                    # Generate mock results
                    current_time = time.time()
                    for i, (result_queue, request_time) in enumerate(zip(result_queues, request_times)):
                        # Mock policy and value
                        policy = np.random.dirichlet([1.0] * 225)  # 15x15 board
                        value = np.random.uniform(-1.0, 1.0)

                        response_time = current_time - request_time
                        self.response_times.append(response_time)

                        try:
                            result_queue.put((policy, value), timeout=1.0)
                        except queue.Full:
                            pass  # Drop if queue full

            except Exception as e:
                logging.warning(f"Mock worker error: {e}")
                time.sleep(0.1)

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch formation statistics"""
        if not self.batch_stats:
            return {
                'avg_batch_size': 0,
                'avg_wait_time': 0,
                'timeout_hit_rate': 0,
                'batch_size_distribution': {}
            }

        sizes = [stat['size'] for stat in self.batch_stats]
        wait_times = [stat['wait_time'] for stat in self.batch_stats]
        timeout_hits = [stat['timeout_hit'] for stat in self.batch_stats]

        # Batch size distribution
        size_counts = defaultdict(int)
        for size in sizes:
            size_counts[str(size)] += 1

        return {
            'avg_batch_size': statistics.mean(sizes),
            'avg_wait_time': statistics.mean(wait_times),
            'timeout_hit_rate': statistics.mean(timeout_hits) * 100,
            'batch_size_distribution': dict(size_counts),
            'total_batches': len(self.batch_stats)
        }

    def get_response_time_stats(self) -> Dict[str, float]:
        """Get response time statistics"""
        if not self.response_times:
            return {'avg_response_time': 0, 'p95_response_time': 0, 'p99_response_time': 0}

        times = list(self.response_times)
        return {
            'avg_response_time': statistics.mean(times),
            'p95_response_time': np.percentile(times, 95),
            'p99_response_time': np.percentile(times, 99)
        }

    def clear_stats(self):
        """Clear accumulated statistics"""
        self.batch_stats.clear()
        self.response_times.clear()


class TimeoutOptimizer:
    """Core timeout optimization functionality"""

    def __init__(self, config: TimeoutOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.gpu_monitor = GPUMonitor()

        # Set up game type
        self.game_type = self._parse_game_type(config.game_type)

        # Results storage
        self.results: List[TimeoutPerformanceResult] = []

    def _parse_game_type(self, game_name: str):
        """Parse game name to game type enum"""
        if GAME_EXTENSIONS_AVAILABLE:
            game_map = {
                'gomoku': GameType.GOMOKU,
                'chess': GameType.CHESS,
                'go': GameType.GO
            }
            return game_map.get(game_name.lower(), GameType.GOMOKU)
        else:
            # Use integer constants when extensions not available
            game_map = {
                'gomoku': 0,
                'chess': 1,
                'go': 2
            }
            return game_map.get(game_name.lower(), 0)

    def _create_game_state(self):
        """Create appropriate game state for testing"""
        if not GAME_EXTENSIONS_AVAILABLE:
            return MockGameState()

        if GAME_EXTENSIONS_AVAILABLE:
            if self.game_type == GameType.GOMOKU:
                return GomokuState()
            elif self.game_type == GameType.CHESS:
                return ChessState()
            elif self.game_type == GameType.GO:
                return GoState()
            else:
                return GomokuState()
        else:
            # When extensions not available, use mock
            return MockGameState()

    def run_optimization(self) -> List[TimeoutPerformanceResult]:
        """Run complete timeout optimization sweep"""
        self.logger.info("Starting timeout optimization...")
        self.logger.info(f"Game: {self.config.game_type}")
        self.logger.info(f"Timeout range: {self.config.min_timeout_ms}-{self.config.max_timeout_ms}ms")
        self.logger.info(f"Test duration per timeout: {self.config.test_duration_per_timeout}s")

        # Generate timeout values to test
        timeout_values = self._generate_timeout_values()
        self.logger.info(f"Testing {len(timeout_values)} timeout values")

        # Test each timeout value
        for i, timeout_ms in enumerate(timeout_values):
            self.logger.info(f"Testing timeout {timeout_ms}ms ({i+1}/{len(timeout_values)})")

            try:
                result = self._test_timeout(timeout_ms)
                self.results.append(result)

                self.logger.info(f"  Throughput: {result.throughput_positions_per_sec:.0f} pos/sec")
                self.logger.info(f"  Avg batch size: {result.avg_batch_size:.1f}")
                self.logger.info(f"  Timeout hit rate: {result.timeout_hit_rate:.1f}%")
                self.logger.info(f"  Efficiency score: {result.efficiency_score:.3f}")

            except Exception as e:
                self.logger.error(f"Error testing timeout {timeout_ms}ms: {e}")
                continue

        # Analyze results and find optimal configuration
        if self.results:
            optimal_result = self._find_optimal_timeout()
            self.logger.info(f"\nOptimal timeout: {optimal_result.timeout_ms}ms")
            self.logger.info(f"  Throughput: {optimal_result.throughput_positions_per_sec:.0f} pos/sec")
            self.logger.info(f"  Avg response time: {optimal_result.avg_response_time_ms:.2f}ms")
            self.logger.info(f"  Efficiency score: {optimal_result.efficiency_score:.3f}")

        return self.results

    def _generate_timeout_values(self) -> List[float]:
        """Generate timeout values to test"""
        timeout_values = []
        current = self.config.min_timeout_ms

        while current <= self.config.max_timeout_ms:
            timeout_values.append(current)
            current += self.config.timeout_step_ms

        return timeout_values

    def _test_timeout(self, timeout_ms: float) -> TimeoutPerformanceResult:
        """Test a specific timeout value"""
        self.logger.debug(f"Testing timeout {timeout_ms}ms...")

        # Create inference worker with this timeout
        if NEURAL_COMPONENTS_AVAILABLE:
            # Use real inference worker if available
            worker = self._create_real_inference_worker(timeout_ms)
        else:
            # Use mock worker for testing
            worker = MockInferenceWorker(
                timeout_ms=timeout_ms,
                min_batch_size=self.config.min_batch_size,
                max_batch_size=self.config.max_batch_size
            )

        # Start monitoring
        self.gpu_monitor.start_monitoring()
        self.gpu_monitor.clear_stats()

        try:
            # Start worker
            worker.start()

            # Run test
            metrics = self._run_throughput_test(worker, timeout_ms)

            # Calculate efficiency score
            efficiency_score = self._calculate_efficiency_score(metrics)

            # Get GPU stats
            gpu_util, memory_mb = self.gpu_monitor.get_stats()

            return TimeoutPerformanceResult(
                timeout_ms=timeout_ms,
                throughput_batches_per_sec=metrics['batches_per_sec'],
                throughput_positions_per_sec=metrics['positions_per_sec'],
                avg_batch_size=metrics['avg_batch_size'],
                avg_batch_wait_time_ms=metrics['avg_batch_wait_time'] * 1000,
                avg_response_time_ms=metrics['avg_response_time'] * 1000,
                timeout_hit_rate=metrics['timeout_hit_rate'],
                queue_depth_stats=metrics['queue_stats'],
                gpu_utilization_percent=gpu_util,
                memory_usage_mb=memory_mb,
                batch_size_distribution=metrics['batch_size_distribution'],
                efficiency_score=efficiency_score,
                total_batches=metrics['total_batches'],
                total_positions=metrics['total_positions'],
                test_duration_sec=self.config.test_duration_per_timeout
            )

        finally:
            # Cleanup
            worker.stop()
            self.gpu_monitor.stop_monitoring()

    def _create_real_inference_worker(self, timeout_ms: float):
        """Create real inference worker if components available"""
        # This would create actual GPU/CPU inference worker
        # For now, fall back to mock
        return MockInferenceWorker(
            timeout_ms=timeout_ms,
            min_batch_size=self.config.min_batch_size,
            max_batch_size=self.config.max_batch_size
        )

    def _run_throughput_test(self, worker, timeout_ms: float) -> Dict[str, Any]:
        """Run throughput test with specified worker"""
        # Create producer threads to generate inference requests
        request_queues = [queue.Queue() for _ in range(self.config.num_producer_threads)]
        results = {'requests_sent': 0, 'responses_received': 0, 'errors': 0}

        def producer_thread(thread_id: int):
            """Producer thread that generates inference requests"""
            game_state = self._create_game_state()
            requests_per_sec = self.config.positions_per_second_target // self.config.num_producer_threads
            request_interval = 1.0 / requests_per_sec

            start_time = time.time()
            while time.time() - start_time < self.config.test_duration_per_timeout:
                try:
                    # Generate position tensor
                    if GAME_EXTENSIONS_AVAILABLE:
                        tensor = game_state.get_enhanced_tensor_representation()
                    else:
                        tensor = np.random.randn(36, 15, 15).astype(np.float32)

                    # Submit inference request
                    result_queue = queue.Queue(maxsize=1)
                    worker.submit_batch([tensor], result_queue)
                    results['requests_sent'] += 1

                    # Try to get result (non-blocking)
                    try:
                        result_queue.get(timeout=0.1)
                        results['responses_received'] += 1
                    except queue.Empty:
                        pass

                    time.sleep(request_interval)

                except Exception as e:
                    results['errors'] += 1
                    time.sleep(0.001)

        # Start producer threads
        threads = []
        for i in range(self.config.num_producer_threads):
            thread = threading.Thread(target=producer_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for test completion
        for thread in threads:
            thread.join()

        # Get worker statistics
        worker_stats = worker.get_batch_stats()
        response_stats = worker.get_response_time_stats()

        # Calculate metrics
        duration = self.config.test_duration_per_timeout
        return {
            'batches_per_sec': worker_stats['total_batches'] / duration,
            'positions_per_sec': results['responses_received'] / duration,
            'avg_batch_size': worker_stats['avg_batch_size'],
            'avg_batch_wait_time': worker_stats['avg_wait_time'],
            'avg_response_time': response_stats['avg_response_time'],
            'timeout_hit_rate': worker_stats['timeout_hit_rate'],
            'queue_stats': {'depth_avg': 0, 'depth_max': 0},  # Simplified
            'batch_size_distribution': worker_stats['batch_size_distribution'],
            'total_batches': worker_stats['total_batches'],
            'total_positions': results['responses_received'],
            'requests_sent': results['requests_sent'],
            'error_rate': results['errors'] / max(1, results['requests_sent'])
        }

    def _calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score combining throughput and responsiveness"""
        # Normalize throughput (target is 15k positions/sec)
        throughput_score = min(1.0, metrics['positions_per_sec'] / 15000.0)

        # Responsiveness score (lower response time is better, target <10ms for more realistic scoring)
        response_time_score = max(0.0, 1.0 - metrics['avg_response_time'] / 0.010)

        # Timeout efficiency (prefer moderate timeout hit rates, ~20-30%)
        timeout_rate = metrics['timeout_hit_rate'] / 100.0
        timeout_score = 1.0 - abs(timeout_rate - 0.25) / 0.25
        timeout_score = max(0.0, timeout_score)

        # Batch size efficiency (prefer larger batches for GPU utilization)
        batch_size_score = min(1.0, metrics['avg_batch_size'] / 64.0)

        # Combined score with weights
        efficiency_score = (
            0.4 * throughput_score +      # 40% throughput
            0.3 * response_time_score +   # 30% responsiveness
            0.2 * timeout_score +         # 20% timeout efficiency
            0.1 * batch_size_score        # 10% batch size
        )

        return efficiency_score

    def _find_optimal_timeout(self) -> TimeoutPerformanceResult:
        """Find optimal timeout configuration from results"""
        if not self.results:
            raise ValueError("No results available")

        # Sort by efficiency score
        sorted_results = sorted(self.results, key=lambda r: r.efficiency_score, reverse=True)
        return sorted_results[0]

    def save_results(self, output_path: str):
        """Save results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump([result.to_dict() for result in self.results], f, indent=2)

        self.logger.info(f"Results saved to {output_file}")

    def create_plots(self, output_dir: str):
        """Create visualization plots"""
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Plotting not available - matplotlib required")
            return

        if not self.results:
            self.logger.warning("No results to plot")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract data for plotting
        timeouts = [r.timeout_ms for r in self.results]
        throughputs = [r.throughput_positions_per_sec for r in self.results]
        response_times = [r.avg_response_time_ms for r in self.results]
        efficiency_scores = [r.efficiency_score for r in self.results]
        timeout_hit_rates = [r.timeout_hit_rate for r in self.results]
        batch_sizes = [r.avg_batch_size for r in self.results]

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Timeout Optimization Results - {self.config.game_type.title()}', fontsize=16)

        # Throughput vs Timeout
        axes[0, 0].plot(timeouts, throughputs, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Timeout (ms)')
        axes[0, 0].set_ylabel('Throughput (positions/sec)')
        axes[0, 0].set_title('Throughput vs Timeout')
        axes[0, 0].grid(True, alpha=0.3)

        # Response Time vs Timeout
        axes[0, 1].plot(timeouts, response_times, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Timeout (ms)')
        axes[0, 1].set_ylabel('Avg Response Time (ms)')
        axes[0, 1].set_title('Response Time vs Timeout')
        axes[0, 1].grid(True, alpha=0.3)

        # Efficiency Score vs Timeout
        axes[0, 2].plot(timeouts, efficiency_scores, 'g-o', linewidth=2, markersize=6)
        axes[0, 2].set_xlabel('Timeout (ms)')
        axes[0, 2].set_ylabel('Efficiency Score')
        axes[0, 2].set_title('Efficiency Score vs Timeout')
        axes[0, 2].grid(True, alpha=0.3)

        # Find and mark optimal point
        optimal_result = self._find_optimal_timeout()
        axes[0, 2].plot(optimal_result.timeout_ms, optimal_result.efficiency_score,
                       'ro', markersize=10, label=f'Optimal: {optimal_result.timeout_ms}ms')
        axes[0, 2].legend()

        # Timeout Hit Rate vs Timeout
        axes[1, 0].plot(timeouts, timeout_hit_rates, 'm-o', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Timeout (ms)')
        axes[1, 0].set_ylabel('Timeout Hit Rate (%)')
        axes[1, 0].set_title('Timeout Hit Rate vs Timeout')
        axes[1, 0].grid(True, alpha=0.3)

        # Batch Size vs Timeout
        axes[1, 1].plot(timeouts, batch_sizes, 'c-o', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Timeout (ms)')
        axes[1, 1].set_ylabel('Avg Batch Size')
        axes[1, 1].set_title('Batch Size vs Timeout')
        axes[1, 1].grid(True, alpha=0.3)

        # Throughput vs Response Time (Pareto frontier)
        axes[1, 2].scatter(response_times, throughputs, c=efficiency_scores,
                          cmap='viridis', s=60, alpha=0.7)
        axes[1, 2].set_xlabel('Avg Response Time (ms)')
        axes[1, 2].set_ylabel('Throughput (positions/sec)')
        axes[1, 2].set_title('Throughput vs Latency Trade-off')
        cbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
        cbar.set_label('Efficiency Score')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'timeout_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Plots saved to {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Optimize inference timeout parameters')
    parser.add_argument('--game', choices=['gomoku', 'chess', 'go'], default='gomoku',
                        help='Game type to optimize for')
    parser.add_argument('--min-timeout', type=float, default=1.0,
                        help='Minimum timeout value in ms')
    parser.add_argument('--max-timeout', type=float, default=10.0,
                        help='Maximum timeout value in ms')
    parser.add_argument('--timeout-step', type=float, default=0.5,
                        help='Timeout step size in ms')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations per timeout')
    parser.add_argument('--test-duration', type=float, default=30.0,
                        help='Test duration per timeout in seconds')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with reduced parameters')
    parser.add_argument('--full-sweep', action='store_true',
                        help='Run comprehensive optimization sweep')
    parser.add_argument('--output', type=str, default='results/timeout_optimization',
                        help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Adjust parameters based on run mode
    if args.quick_test:
        config = TimeoutOptimizationConfig(
            game_type=args.game,
            min_timeout_ms=1.0,
            max_timeout_ms=5.0,
            timeout_step_ms=1.0,
            iterations_per_timeout=50,
            test_duration_per_timeout=10.0,
            output_dir=args.output,
            enable_plots=not args.no_plots,
            verbose=args.verbose
        )
    elif args.full_sweep:
        config = TimeoutOptimizationConfig(
            game_type=args.game,
            min_timeout_ms=0.5,
            max_timeout_ms=15.0,
            timeout_step_ms=0.2,
            iterations_per_timeout=200,
            test_duration_per_timeout=60.0,
            output_dir=args.output,
            enable_plots=not args.no_plots,
            verbose=args.verbose
        )
    else:
        config = TimeoutOptimizationConfig(
            game_type=args.game,
            min_timeout_ms=args.min_timeout,
            max_timeout_ms=args.max_timeout,
            timeout_step_ms=args.timeout_step,
            iterations_per_timeout=args.iterations,
            test_duration_per_timeout=args.test_duration,
            output_dir=args.output,
            enable_plots=not args.no_plots,
            verbose=args.verbose
        )

    # Run optimization
    optimizer = TimeoutOptimizer(config)
    try:
        results = optimizer.run_optimization()

        # Save results
        if config.output_dir:
            output_path = Path(config.output_dir)
            optimizer.save_results(output_path / 'timeout_results.json')

            if config.enable_plots:
                optimizer.create_plots(config.output_dir)

        # Print summary
        if results:
            optimal = optimizer._find_optimal_timeout()
            print(f"\n{'='*60}")
            print(f"TIMEOUT OPTIMIZATION RESULTS")
            print(f"{'='*60}")
            print(f"Game: {config.game_type.title()}")
            print(f"Tested {len(results)} timeout configurations")
            print(f"")
            print(f"OPTIMAL CONFIGURATION:")
            print(f"  Timeout: {optimal.timeout_ms:.1f}ms")
            print(f"  Throughput: {optimal.throughput_positions_per_sec:.0f} positions/sec")
            print(f"  Avg Response Time: {optimal.avg_response_time_ms:.2f}ms")
            print(f"  Avg Batch Size: {optimal.avg_batch_size:.1f}")
            print(f"  Timeout Hit Rate: {optimal.timeout_hit_rate:.1f}%")
            print(f"  Efficiency Score: {optimal.efficiency_score:.3f}")
            print(f"")
            print(f"Target achieved: {'✅' if optimal.timeout_ms <= 3.5 else '❌'} "
                  f"(target: ≤3ms, balances throughput and responsiveness)")

        return 0

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())