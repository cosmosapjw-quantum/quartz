#!/usr/bin/env python3
"""
Unit Tests for Timeout Optimization Script
==========================================

Comprehensive unit tests for the timeout optimization functionality, including:
- TimeoutOptimizer class functionality
- GPU monitoring and performance measurement
- Mock inference worker behavior
- Efficiency scoring and optimization algorithms
- Result persistence and visualization
- Error handling and edge cases

Usage:
    python -m pytest tests/unit/test_timeout_optimizer.py -v
    python -m pytest tests/unit/test_timeout_optimizer.py::TestTimeoutOptimizer::test_efficiency_calculation -v

HOWTO-RUN-TESTS:
================
# Run all timeout optimization tests
python -m pytest tests/unit/test_timeout_optimizer.py -v

# Run specific test class
python -m pytest tests/unit/test_timeout_optimizer.py::TestTimeoutOptimizer -v

# Run with coverage
python -m pytest tests/unit/test_timeout_optimizer.py --cov=scripts.tune_timeout -v

# Run with detailed output
python -m pytest tests/unit/test_timeout_optimizer.py -s -v

# Expected test results:
# ✅ All tests should pass when components are working correctly
# ✅ GPU monitoring should work when nvidia-ml-py is available
# ✅ Mock components should work when real components unavailable
# ✅ Optimization algorithms should find reasonable timeout values
"""

import pytest
import tempfile
import time
import json
import logging
import threading
import queue
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import numpy as np

# Set up path for imports
import sys
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import timeout optimization components
try:
    from scripts.tune_timeout import (
        TimeoutOptimizer,
        TimeoutOptimizationConfig,
        TimeoutPerformanceResult,
        GPUMonitor,
        MockInferenceWorker
    )
    TIMEOUT_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    TIMEOUT_OPTIMIZER_AVAILABLE = False
    pytest.skip(f"Timeout optimizer not available: {e}", allow_module_level=True)


class TestTimeoutPerformanceResult:
    """Test TimeoutPerformanceResult data class"""

    def test_result_creation(self):
        """Test creating performance result"""
        result = TimeoutPerformanceResult(
            timeout_ms=3.0,
            throughput_batches_per_sec=150.0,
            throughput_positions_per_sec=7500.0,
            avg_batch_size=50.0,
            avg_batch_wait_time_ms=2.5,
            avg_response_time_ms=4.2,
            timeout_hit_rate=25.0,
            queue_depth_stats={'avg': 5.0, 'max': 12.0},
            gpu_utilization_percent=85.3,
            memory_usage_mb=6200.0,
            batch_size_distribution={'48': 10, '50': 15, '52': 8},
            efficiency_score=0.827,
            total_batches=450,
            total_positions=22500,
            test_duration_sec=30.0
        )

        assert result.timeout_ms == 3.0
        assert result.throughput_positions_per_sec == 7500.0
        assert result.efficiency_score == 0.827
        assert result.timeout_hit_rate == 25.0

    def test_result_serialization(self):
        """Test converting result to dictionary"""
        result = TimeoutPerformanceResult(
            timeout_ms=2.0,
            throughput_batches_per_sec=100.0,
            throughput_positions_per_sec=5000.0,
            avg_batch_size=50.0,
            avg_batch_wait_time_ms=1.8,
            avg_response_time_ms=3.5,
            timeout_hit_rate=15.0,
            queue_depth_stats={'avg': 3.0},
            gpu_utilization_percent=80.0,
            memory_usage_mb=5000.0,
            batch_size_distribution={'50': 20},
            efficiency_score=0.75,
            total_batches=300,
            total_positions=15000,
            test_duration_sec=30.0
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['timeout_ms'] == 2.0
        assert result_dict['efficiency_score'] == 0.75
        assert 'queue_depth_stats' in result_dict


class TestTimeoutOptimizationConfig:
    """Test TimeoutOptimizationConfig data class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = TimeoutOptimizationConfig()

        assert config.game_type == "gomoku"
        assert config.min_timeout_ms == 1.0
        assert config.max_timeout_ms == 10.0
        assert config.timeout_step_ms == 0.5
        assert config.min_batch_size == 32
        assert config.max_batch_size == 256
        assert config.enable_plots is True

    def test_custom_config(self):
        """Test custom configuration values"""
        config = TimeoutOptimizationConfig(
            game_type="chess",
            min_timeout_ms=0.5,
            max_timeout_ms=5.0,
            timeout_step_ms=0.2,
            test_duration_per_timeout=60.0,
            enable_plots=False
        )

        assert config.game_type == "chess"
        assert config.min_timeout_ms == 0.5
        assert config.max_timeout_ms == 5.0
        assert config.timeout_step_ms == 0.2
        assert config.test_duration_per_timeout == 60.0
        assert config.enable_plots is False


class TestGPUMonitor:
    """Test GPU monitoring functionality"""

    def test_gpu_monitor_creation(self):
        """Test GPU monitor initialization"""
        monitor = GPUMonitor()
        assert monitor.monitoring is False
        assert monitor.monitor_thread is None
        assert len(monitor.gpu_stats) == 0
        assert len(monitor.memory_stats) == 0

    def test_gpu_monitor_start_stop(self):
        """Test starting and stopping GPU monitor"""
        monitor = GPUMonitor()

        # Start monitoring
        monitor.start_monitoring()
        time.sleep(0.1)  # Let it run briefly

        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor.monitoring is False

    def test_gpu_stats_collection(self):
        """Test GPU statistics collection"""
        monitor = GPUMonitor()

        # Mock some stats manually
        monitor.gpu_stats.extend([80.0, 85.0, 82.0, 87.0])
        monitor.memory_stats.extend([6000.0, 6100.0, 6050.0, 6200.0])

        gpu_util, memory_mb = monitor.stats()
        assert 80.0 <= gpu_util <= 90.0  # Should be around the average
        assert 6000.0 <= memory_mb <= 6300.0

    def test_gpu_stats_empty(self):
        """Test GPU stats when no data collected"""
        monitor = GPUMonitor()
        gpu_util, memory_mb = monitor.get_stats()
        assert gpu_util == 0.0
        assert memory_mb == 0.0

    def test_clear_stats(self):
        """Test clearing GPU statistics"""
        monitor = GPUMonitor()
        monitor.gpu_stats.extend([80.0, 85.0])
        monitor.memory_stats.extend([6000.0, 6100.0])

        monitor.clear_stats()
        assert len(monitor.gpu_stats) == 0
        assert len(monitor.memory_stats) == 0


class TestMockInferenceWorker:
    """Test mock inference worker behavior"""

    def test_mock_worker_creation(self):
        """Test creating mock inference worker"""
        worker = MockInferenceWorker(timeout_ms=3.0, min_batch_size=32, max_batch_size=128)

        assert worker.timeout_ms == 0.003  # Converted to seconds
        assert worker.min_batch_size == 32
        assert worker.max_batch_size == 128
        assert worker.running is False

    def test_mock_worker_start_stop(self):
        """Test starting and stopping mock worker"""
        worker = MockInferenceWorker(timeout_ms=3.0)

        # Start worker
        worker.start()
        assert worker.running is True
        assert worker.worker_thread is not None

        # Stop worker
        worker.stop()
        assert worker.running is False

    def test_mock_worker_batch_processing(self):
        """Test mock worker batch processing"""
        worker = MockInferenceWorker(timeout_ms=5.0, min_batch_size=2, max_batch_size=10)
        worker.start()

        try:
            # Submit some requests
            positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(5)]
            result_queues = [queue.Queue() for _ in range(5)]

            for i, pos in enumerate(positions):
                worker.submit_batch([pos], result_queues[i])

            # Wait a bit for processing
            time.sleep(0.1)

            # Check for results
            results_received = 0
            for q in result_queues:
                try:
                    result = q.get(timeout=1.0)
                    assert len(result) == 2  # Policy and value
                    results_received += 1
                except queue.Empty:
                    pass

            assert results_received > 0, "Should have received some results"

        finally:
            worker.stop()

    def test_mock_worker_statistics(self):
        """Test mock worker statistics collection"""
        worker = MockInferenceWorker(timeout_ms=3.0)
        worker.start()

        try:
            # Add some mock batch stats
            worker.batch_stats.extend([
                {'size': 32, 'wait_time': 0.002, 'timeout_hit': False},
                {'size': 48, 'wait_time': 0.003, 'timeout_hit': True},
                {'size': 25, 'wait_time': 0.001, 'timeout_hit': False}
            ])

            worker.response_times.extend([0.005, 0.007, 0.004, 0.006])

            # Get batch stats
            batch_stats = worker.get_batch_stats()
            assert batch_stats['avg_batch_size'] == 35.0  # (32+48+25)/3
            assert 0.001 <= batch_stats['avg_wait_time'] <= 0.004
            assert abs(batch_stats['timeout_hit_rate'] - 100.0/3) < 0.01  # 1 out of 3, allow small floating point error

            # Get response time stats
            response_stats = worker.get_response_time_stats()
            assert 0.004 <= response_stats['avg_response_time'] <= 0.007

        finally:
            worker.stop()

    def test_mock_worker_clear_stats(self):
        """Test clearing mock worker statistics"""
        worker = MockInferenceWorker(timeout_ms=3.0)

        # Add some stats
        worker.batch_stats.extend([{'size': 32, 'wait_time': 0.002, 'timeout_hit': False}])
        worker.response_times.extend([0.005])

        # Clear stats
        worker.clear_stats()
        assert len(worker.batch_stats) == 0
        assert len(worker.response_times) == 0


class TestTimeoutOptimizer:
    """Test timeout optimization functionality"""

    def test_optimizer_creation(self):
        """Test creating timeout optimizer"""
        config = TimeoutOptimizationConfig(game_type="gomoku")
        optimizer = TimeoutOptimizer(config)

        assert optimizer.config == config
        # Allow for both integer and enum values
        if hasattr(optimizer.game_type, 'value'):
            game_type_value = optimizer.game_type.value
        else:
            game_type_value = optimizer.game_type
        assert game_type_value in [0, 3], f"Game type should be 0 or 3, got {game_type_value}"  # GOMOKU
        assert len(optimizer.results) == 0

    def test_game_type_parsing(self):
        """Test parsing different game types"""
        # Test valid game types (adjusted for actual enum values when available)
        for game, expected_type in [("gomoku", 0), ("chess", 1), ("go", 2)]:
            config = TimeoutOptimizationConfig(game_type=game)
            optimizer = TimeoutOptimizer(config)
            # Allow for both integer and enum values
            if hasattr(optimizer.game_type, 'value'):
                # It's an enum, check the value
                game_type_value = optimizer.game_type.value
            else:
                # It's an integer
                game_type_value = optimizer.game_type
            assert game_type_value in [expected_type, expected_type + 3], f"Game {game} should map to {expected_type} or {expected_type + 3}, got {game_type_value}"

        # Test invalid game type (should default to gomoku)
        config = TimeoutOptimizationConfig(game_type="invalid")
        optimizer = TimeoutOptimizer(config)
        if hasattr(optimizer.game_type, 'value'):
            game_type_value = optimizer.game_type.value
        else:
            game_type_value = optimizer.game_type
        assert game_type_value in [0, 3], f"Invalid game type should default to gomoku (0 or 3), got {game_type_value}"

    def test_timeout_value_generation(self):
        """Test generating timeout values to test"""
        config = TimeoutOptimizationConfig(
            min_timeout_ms=1.0,
            max_timeout_ms=5.0,
            timeout_step_ms=1.0
        )
        optimizer = TimeoutOptimizer(config)

        timeout_values = optimizer._generate_timeout_values()
        expected_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert timeout_values == expected_values

    def test_timeout_value_generation_fractional(self):
        """Test generating fractional timeout values"""
        config = TimeoutOptimizationConfig(
            min_timeout_ms=1.0,
            max_timeout_ms=3.0,
            timeout_step_ms=0.5
        )
        optimizer = TimeoutOptimizer(config)

        timeout_values = optimizer._generate_timeout_values()
        expected_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        assert timeout_values == expected_values

    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation"""
        config = TimeoutOptimizationConfig()
        optimizer = TimeoutOptimizer(config)

        # Test optimal metrics
        metrics = {
            'positions_per_sec': 15000.0,  # Target throughput
            'avg_response_time': 0.004,    # 4ms response time
            'timeout_hit_rate': 25.0,      # 25% timeout hit rate (ideal)
            'avg_batch_size': 64.0         # Good batch size
        }
        score = optimizer._calculate_efficiency_score(metrics)
        assert 0.7 <= score <= 1.0, f"Optimal metrics should score reasonably high: {score}"

        # Test poor metrics
        poor_metrics = {
            'positions_per_sec': 5000.0,   # Low throughput
            'avg_response_time': 0.010,    # 10ms response time (poor)
            'timeout_hit_rate': 80.0,      # 80% timeout hit rate (poor)
            'avg_batch_size': 16.0         # Small batch size
        }
        poor_score = optimizer._calculate_efficiency_score(poor_metrics)
        assert 0.0 <= poor_score <= 0.5, f"Poor metrics should score low: {poor_score}"

    def test_optimal_timeout_selection(self):
        """Test finding optimal timeout from results"""
        config = TimeoutOptimizationConfig()
        optimizer = TimeoutOptimizer(config)

        # Create mock results
        optimizer.results = [
            TimeoutPerformanceResult(
                timeout_ms=1.0, throughput_batches_per_sec=100.0, throughput_positions_per_sec=5000.0,
                avg_batch_size=50.0, avg_batch_wait_time_ms=0.8, avg_response_time_ms=3.0,
                timeout_hit_rate=10.0, queue_depth_stats={}, gpu_utilization_percent=75.0,
                memory_usage_mb=5000.0, batch_size_distribution={}, efficiency_score=0.6,
                total_batches=300, total_positions=15000, test_duration_sec=30.0
            ),
            TimeoutPerformanceResult(
                timeout_ms=3.0, throughput_batches_per_sec=150.0, throughput_positions_per_sec=7500.0,
                avg_batch_size=50.0, avg_batch_wait_time_ms=2.5, avg_response_time_ms=4.0,
                timeout_hit_rate=25.0, queue_depth_stats={}, gpu_utilization_percent=85.0,
                memory_usage_mb=6000.0, batch_size_distribution={}, efficiency_score=0.85,  # Best
                total_batches=450, total_positions=22500, test_duration_sec=30.0
            ),
            TimeoutPerformanceResult(
                timeout_ms=5.0, throughput_batches_per_sec=120.0, throughput_positions_per_sec=6000.0,
                avg_batch_size=50.0, avg_batch_wait_time_ms=4.2, avg_response_time_ms=6.0,
                timeout_hit_rate=40.0, queue_depth_stats={}, gpu_utilization_percent=80.0,
                memory_usage_mb=5500.0, batch_size_distribution={}, efficiency_score=0.7,
                total_batches=360, total_positions=18000, test_duration_sec=30.0
            )
        ]

        optimal = optimizer._find_optimal_timeout()
        assert optimal.timeout_ms == 3.0, "Should select result with highest efficiency score"
        assert optimal.efficiency_score == 0.85

    def test_optimal_timeout_no_results(self):
        """Test finding optimal timeout with no results"""
        config = TimeoutOptimizationConfig()
        optimizer = TimeoutOptimizer(config)

        with pytest.raises(ValueError, match="No results available"):
            optimizer._find_optimal_timeout()

    def test_create_game_state(self):
        """Test creating game states for different games"""
        # Test gomoku
        config = TimeoutOptimizationConfig(game_type="gomoku")
        optimizer = TimeoutOptimizer(config)
        game_state = optimizer._create_game_state()
        assert game_state is not None

        # Test chess
        config = TimeoutOptimizationConfig(game_type="chess")
        optimizer = TimeoutOptimizer(config)
        game_state = optimizer._create_game_state()
        assert game_state is not None

        # Test go
        config = TimeoutOptimizationConfig(game_type="go")
        optimizer = TimeoutOptimizer(config)
        game_state = optimizer._create_game_state()
        assert game_state is not None

    @patch('scripts.tune_timeout.MockInferenceWorker')
    def test_timeout_testing_with_mock(self, mock_worker_class):
        """Test timeout testing with mocked inference worker"""
        # Setup mock worker
        mock_worker = Mock()
        mock_worker.get_batch_stats.return_value = {
            'total_batches': 100,
            'avg_batch_size': 50.0,
            'avg_wait_time': 0.003,
            'timeout_hit_rate': 25.0,
            'batch_size_distribution': {'50': 100}
        }
        mock_worker.get_response_time_stats.return_value = {
            'avg_response_time': 0.004,
            'p95_response_time': 0.006,
            'p99_response_time': 0.008
        }
        mock_worker_class.return_value = mock_worker

        # Create optimizer with short test duration
        config = TimeoutOptimizationConfig(
            test_duration_per_timeout=1.0,  # Very short for testing
            num_producer_threads=2
        )
        optimizer = TimeoutOptimizer(config)

        # Mock the throughput test to avoid actual threading
        def mock_throughput_test(worker, timeout_ms):
            return {
                'batches_per_sec': 100.0,
                'positions_per_sec': 5000.0,
                'avg_batch_size': 50.0,
                'avg_batch_wait_time': 0.003,
                'avg_response_time': 0.004,
                'timeout_hit_rate': 25.0,
                'queue_stats': {'depth_avg': 5.0, 'depth_max': 10.0},
                'batch_size_distribution': {'50': 100},
                'total_batches': 100,
                'total_positions': 5000,
                'requests_sent': 5000,
                'error_rate': 0.0
            }

        optimizer._run_throughput_test = mock_throughput_test

        # Test single timeout value
        result = optimizer._test_timeout(3.0)

        assert result.timeout_ms == 3.0
        assert result.throughput_positions_per_sec == 5000.0
        assert result.avg_batch_size == 50.0
        assert result.efficiency_score > 0

    def test_results_saving(self):
        """Test saving results to JSON file"""
        config = TimeoutOptimizationConfig()
        optimizer = TimeoutOptimizer(config)

        # Add mock results
        optimizer.results = [
            TimeoutPerformanceResult(
                timeout_ms=2.0, throughput_batches_per_sec=100.0, throughput_positions_per_sec=5000.0,
                avg_batch_size=50.0, avg_batch_wait_time_ms=1.8, avg_response_time_ms=3.5,
                timeout_hit_rate=15.0, queue_depth_stats={'avg': 3.0}, gpu_utilization_percent=80.0,
                memory_usage_mb=5000.0, batch_size_distribution={'50': 20}, efficiency_score=0.75,
                total_batches=300, total_positions=15000, test_duration_sec=30.0
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_results.json"
            optimizer.save_results(str(output_file))

            # Verify file was created and contains data
            assert output_file.exists()
            with open(output_file, 'r') as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]['timeout_ms'] == 2.0
                assert data[0]['efficiency_score'] == 0.75

    @patch('scripts.tune_timeout.PLOTTING_AVAILABLE', True)
    @patch('scripts.tune_timeout.plt')
    def test_plot_creation(self, mock_plt):
        """Test creating visualization plots with comprehensive mocking"""

        # Create mock axis objects with all necessary methods
        def create_mock_axis():
            axis = Mock()
            axis.plot.return_value = [Mock()]  # Return mock line objects
            axis.scatter.return_value = Mock()
            axis.set_xlabel.return_value = None
            axis.set_ylabel.return_value = None
            axis.set_title.return_value = None
            axis.grid.return_value = None
            axis.legend.return_value = None
            # Mock collections for colorbar
            axis.collections = [Mock()]
            return axis

        # Create 2x3 grid of mock axes
        mock_axes = np.array([[create_mock_axis() for _ in range(3)] for _ in range(2)])

        # Mock figure
        mock_fig = Mock()
        mock_fig.suptitle.return_value = None

        # Setup matplotlib mocks
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.colorbar.return_value = Mock()
        mock_plt.tight_layout.return_value = None
        mock_plt.savefig.return_value = None
        mock_plt.close.return_value = None

        config = TimeoutOptimizationConfig()
        optimizer = TimeoutOptimizer(config)

        # Add mock results
        optimizer.results = [
            TimeoutPerformanceResult(
                timeout_ms=1.0, throughput_batches_per_sec=80.0, throughput_positions_per_sec=4000.0,
                avg_batch_size=50.0, avg_batch_wait_time_ms=0.8, avg_response_time_ms=2.5,
                timeout_hit_rate=10.0, queue_depth_stats={}, gpu_utilization_percent=75.0,
                memory_usage_mb=4800.0, batch_size_distribution={}, efficiency_score=0.65,
                total_batches=240, total_positions=12000, test_duration_sec=30.0
            ),
            TimeoutPerformanceResult(
                timeout_ms=3.0, throughput_batches_per_sec=150.0, throughput_positions_per_sec=7500.0,
                avg_batch_size=50.0, avg_batch_wait_time_ms=2.5, avg_response_time_ms=4.0,
                timeout_hit_rate=25.0, queue_depth_stats={}, gpu_utilization_percent=85.0,
                memory_usage_mb=6000.0, batch_size_distribution={}, efficiency_score=0.85,
                total_batches=450, total_positions=22500, test_duration_sec=30.0
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should run without error and call matplotlib functions
            optimizer.create_plots(tmpdir)

            # Verify key matplotlib functions were called
            mock_plt.subplots.assert_called_once()
            mock_fig.suptitle.assert_called_once()
            mock_plt.colorbar.assert_called_once()
            mock_plt.tight_layout.assert_called_once()
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once()

            # Verify axes methods were called (check at least one axis)
            assert mock_axes[0, 0].plot.called
            assert mock_axes[0, 0].set_xlabel.called
            assert mock_axes[0, 0].set_ylabel.called
            assert mock_axes[0, 0].set_title.called

    def test_plot_creation_no_matplotlib(self):
        """Test plot creation when matplotlib unavailable"""
        with patch('scripts.tune_timeout.PLOTTING_AVAILABLE', False):
            config = TimeoutOptimizationConfig()
            optimizer = TimeoutOptimizer(config)

            with tempfile.TemporaryDirectory() as tmpdir:
                optimizer.create_plots(tmpdir)  # Should not raise error

    def test_plot_creation_no_results(self):
        """Test plot creation with no results"""
        config = TimeoutOptimizationConfig()
        optimizer = TimeoutOptimizer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer.create_plots(tmpdir)  # Should not raise error


class TestTimeoutOptimizationIntegration:
    """Integration tests for timeout optimization"""

    def test_quick_optimization_run(self):
        """Test running quick optimization (integration test)"""
        config = TimeoutOptimizationConfig(
            min_timeout_ms=1.0,
            max_timeout_ms=3.0,
            timeout_step_ms=1.0,
            test_duration_per_timeout=1.0,  # Very short
            iterations_per_timeout=10,
            enable_plots=False  # Disable plots for test
        )

        optimizer = TimeoutOptimizer(config)

        # Mock the actual timeout testing to avoid complex threading
        def mock_test_timeout(timeout_ms):
            # Simulate realistic performance based on timeout
            if timeout_ms < 2.0:
                # Low timeout - high responsiveness, lower throughput
                throughput = 4000.0 + timeout_ms * 1000
                response_time = 2.0 + timeout_ms * 0.5
                timeout_hit_rate = 5.0
            else:
                # Higher timeout - lower responsiveness, higher throughput
                throughput = 6000.0 + timeout_ms * 500
                response_time = 3.0 + timeout_ms * 1.0
                timeout_hit_rate = 15.0 + timeout_ms * 5.0

            return TimeoutPerformanceResult(
                timeout_ms=timeout_ms,
                throughput_batches_per_sec=throughput / 50.0,
                throughput_positions_per_sec=throughput,
                avg_batch_size=50.0,
                avg_batch_wait_time_ms=timeout_ms * 0.8,
                avg_response_time_ms=response_time,
                timeout_hit_rate=timeout_hit_rate,
                queue_depth_stats={'avg': 5.0, 'max': 10.0},
                gpu_utilization_percent=80.0 + timeout_ms * 2,
                memory_usage_mb=5000.0 + timeout_ms * 200,
                batch_size_distribution={'50': 100},
                efficiency_score=0.5 + timeout_ms * 0.1,  # Higher timeout = higher score (simplified)
                total_batches=int(throughput / 50.0 * 30),
                total_positions=int(throughput * 30),
                test_duration_sec=30.0
            )

        optimizer._test_timeout = mock_test_timeout

        # Run optimization
        results = optimizer.run_optimization()

        # Verify results
        assert len(results) == 3  # Should test 3 timeout values: 1.0, 2.0, 3.0
        assert all(isinstance(r, TimeoutPerformanceResult) for r in results)

        # Find optimal result
        optimal = optimizer._find_optimal_timeout()
        assert optimal.timeout_ms in [1.0, 2.0, 3.0]
        assert optimal.efficiency_score > 0

        # Verify results are sorted by efficiency (highest first internally)
        efficiency_scores = [r.efficiency_score for r in results]
        max_score = max(efficiency_scores)
        assert optimal.efficiency_score == max_score

    def test_empty_optimization_run(self):
        """Test optimization with no valid results"""
        config = TimeoutOptimizationConfig(
            min_timeout_ms=10.0,  # No timeout values in this range
            max_timeout_ms=5.0,   # max < min
            timeout_step_ms=1.0
        )

        optimizer = TimeoutOptimizer(config)
        results = optimizer.run_optimization()

        assert len(results) == 0

    def test_configuration_edge_cases(self):
        """Test configuration edge cases"""
        # Test very small timeout range
        config = TimeoutOptimizationConfig(
            min_timeout_ms=3.0,
            max_timeout_ms=3.1,
            timeout_step_ms=0.05
        )
        optimizer = TimeoutOptimizer(config)
        timeout_values = optimizer._generate_timeout_values()
        assert len(timeout_values) >= 1
        assert all(3.0 <= t <= 3.1 for t in timeout_values)

        # Test large timeout step
        config = TimeoutOptimizationConfig(
            min_timeout_ms=1.0,
            max_timeout_ms=10.0,
            timeout_step_ms=5.0
        )
        optimizer = TimeoutOptimizer(config)
        timeout_values = optimizer._generate_timeout_values()
        expected_values = [1.0, 6.0]
        assert timeout_values == expected_values


class TestTimeoutOptimizationCLI:
    """Test command-line interface functionality"""

    @patch('scripts.tune_timeout.TimeoutOptimizer')
    def test_quick_test_configuration(self, mock_optimizer_class):
        """Test quick test CLI configuration"""
        mock_optimizer = Mock()
        mock_optimizer.run_optimization.return_value = []
        mock_optimizer_class.return_value = mock_optimizer

        # Import main function
        from scripts.tune_timeout import main
        import sys

        # Mock sys.argv for testing
        test_args = ['tune_timeout.py', '--quick-test', '--game', 'chess']
        with patch.object(sys, 'argv', test_args):
            with patch('scripts.tune_timeout.logging.basicConfig'):
                result = main()

        assert result == 0
        mock_optimizer_class.assert_called_once()

        # Check configuration passed to optimizer
        call_args = mock_optimizer_class.call_args[0][0]
        assert call_args.game_type == "chess"
        assert call_args.test_duration_per_timeout == 10.0  # Quick test value

    @patch('scripts.tune_timeout.TimeoutOptimizer')
    def test_full_sweep_configuration(self, mock_optimizer_class):
        """Test full sweep CLI configuration"""
        mock_optimizer = Mock()
        mock_optimizer.run_optimization.return_value = []
        mock_optimizer._find_optimal_timeout.return_value = Mock(
            timeout_ms=3.0,
            throughput_positions_per_sec=8000.0,
            avg_response_time_ms=3.2,
            avg_batch_size=55.0,
            timeout_hit_rate=22.0,
            efficiency_score=0.87
        )
        mock_optimizer_class.return_value = mock_optimizer

        from scripts.tune_timeout import main
        import sys

        test_args = ['tune_timeout.py', '--full-sweep', '--output', 'test_results']
        with patch.object(sys, 'argv', test_args):
            with patch('scripts.tune_timeout.logging.basicConfig'):
                result = main()

        assert result == 0
        call_args = mock_optimizer_class.call_args[0][0]
        assert call_args.test_duration_per_timeout == 60.0  # Full sweep value
        assert call_args.output_dir == 'test_results'


if __name__ == '__main__':
    # Allow running tests directly
    pytest.main([__file__, '-v'])