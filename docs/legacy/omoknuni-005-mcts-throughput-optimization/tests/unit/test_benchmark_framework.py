"""
Unit tests for the performance benchmark framework.

Tests the benchmark framework components, result handling,
regression detection, and system metrics collection.
"""

import json
import tempfile
import time
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np

from tests.performance.test_benchmarks import (
    BenchmarkResult, SystemMetrics, BenchmarkFramework,
    AlphaZeroPerformanceBenchmarks
)


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass and methods."""

    def test_benchmark_result_creation(self):
        """Test BenchmarkResult can be created with required fields."""
        result = BenchmarkResult(
            name="test_benchmark",
            metric_name="throughput",
            value=1000.0,
            unit="ops/sec",
            timestamp=time.time(),
            metadata={"iterations": 5}
        )

        assert result.name == "test_benchmark"
        assert result.metric_name == "throughput"
        assert result.value == 1000.0
        assert result.unit == "ops/sec"
        assert isinstance(result.timestamp, float)
        assert result.metadata["iterations"] == 5

    def test_is_within_target_no_limits(self):
        """Test target checking with no limits set."""
        result = BenchmarkResult(
            name="test", metric_name="test", value=100.0,
            unit="units", timestamp=time.time(), metadata={}
        )

        assert result.is_within_target()

    def test_is_within_target_with_limits(self):
        """Test target checking with min/max limits."""
        # Within range
        result = BenchmarkResult(
            name="test", metric_name="test", value=50.0,
            unit="units", timestamp=time.time(), metadata={},
            target_min=10.0, target_max=100.0
        )
        assert result.is_within_target()

        # Below minimum
        result.value = 5.0
        assert not result.is_within_target()

        # Above maximum
        result.value = 150.0
        assert not result.is_within_target()

    def test_regression_score_calculation(self):
        """Test regression score calculation."""
        result = BenchmarkResult(
            name="test", metric_name="test", value=90.0,
            unit="units", timestamp=time.time(), metadata={}
        )

        # 10% regression (90 vs 100)
        score = result.regression_score(100.0)
        assert abs(score - (-0.1)) < 0.001

        # 20% improvement (120 vs 100)
        result.value = 120.0
        score = result.regression_score(100.0)
        assert abs(score - 0.2) < 0.001

        # Handle zero baseline
        score = result.regression_score(0.0)
        assert score == 0.0


class TestSystemMetrics:
    """Test SystemMetrics data collection."""

    @patch('tests.performance.test_benchmarks.PYNVML_AVAILABLE', False)
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.Process')
    def test_system_metrics_capture_basic(self, mock_process, mock_memory, mock_cpu):
        """Test basic system metrics capture."""
        # Setup mocks
        mock_cpu.return_value = 75.5
        mock_memory.return_value = Mock(used=1024 * 1024 * 512)  # 512MB
        mock_process.return_value.num_threads.return_value = 8

        metrics = SystemMetrics.capture()

        assert metrics.cpu_percent == 75.5
        assert metrics.memory_mb == 512.0
        assert metrics.thread_count == 8
        assert metrics.gpu_utilization is None
        assert metrics.gpu_memory_mb is None

    @patch('tests.performance.test_benchmarks.PYNVML_AVAILABLE', True)
    @patch('pynvml.nvmlDeviceGetHandleByIndex')
    @patch('pynvml.nvmlDeviceGetUtilizationRates')
    @patch('pynvml.nvmlDeviceGetMemoryInfo')
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.Process')
    def test_system_metrics_capture_with_gpu(self, mock_process, mock_memory, mock_cpu,
                                           mock_gpu_memory, mock_gpu_util, mock_gpu_handle):
        """Test system metrics capture with GPU monitoring."""
        # Setup mocks
        mock_cpu.return_value = 80.0
        mock_memory.return_value = Mock(used=1024 * 1024 * 1024)  # 1GB
        mock_process.return_value.num_threads.return_value = 12

        mock_gpu_util.return_value = Mock(gpu=85)
        mock_gpu_memory.return_value = Mock(used=2 * 1024 * 1024 * 1024)  # 2GB

        metrics = SystemMetrics.capture()

        assert metrics.cpu_percent == 80.0
        assert metrics.memory_mb == 1024.0
        assert metrics.thread_count == 12
        assert metrics.gpu_utilization == 85
        assert metrics.gpu_memory_mb == 2048.0


class TestBenchmarkFramework:
    """Test BenchmarkFramework functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = BenchmarkFramework(results_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)

    def test_framework_initialization(self):
        """Test benchmark framework initialization."""
        assert self.framework.results_dir.exists()
        assert len(self.framework.current_results) == 0

    def test_run_benchmark_basic(self):
        """Test basic benchmark execution."""
        def simple_benchmark():
            time.sleep(0.01)  # 10ms
            return 42.0

        result = self.framework.run_benchmark(
            name="test_benchmark",
            benchmark_func=simple_benchmark,
            iterations=3,
            warmup=1,
            metric_name="test_value",
            unit="test_units"
        )

        assert result.name == "test_benchmark"
        assert result.metric_name == "test_value"
        assert result.unit == "test_units"
        assert result.value == 42.0
        assert len(self.framework.current_results) == 1
        assert result.metadata["iterations"] == 3

    def test_run_benchmark_with_targets(self):
        """Test benchmark execution with performance targets."""
        def benchmark_func():
            return 100.0

        result = self.framework.run_benchmark(
            name="target_test",
            benchmark_func=benchmark_func,
            target_min=50.0,
            target_max=150.0,
            iterations=2
        )

        assert result.is_within_target()
        assert result.target_min == 50.0
        assert result.target_max == 150.0

    def test_run_benchmark_with_exceptions(self):
        """Test benchmark handling of exceptions."""
        def failing_benchmark():
            raise ValueError("Test exception")

        result = self.framework.run_benchmark(
            name="failing_test",
            benchmark_func=failing_benchmark,
            iterations=2
        )

        assert result.value == 0.0  # Failure indicator
        assert float('inf') in result.metadata["times"]

    def test_save_and_load_results(self):
        """Test saving and loading benchmark results."""
        # Add a test result
        result = BenchmarkResult(
            name="test_save",
            metric_name="value",
            value=123.45,
            unit="units",
            timestamp=time.time(),
            metadata={"test": True}
        )
        self.framework.current_results.append(result)

        # Save results
        saved_file = self.framework.save_results("test_results.json")
        assert saved_file.exists()

        # Load and verify
        with open(saved_file, 'r') as f:
            data = json.load(f)

        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["name"] == "test_save"
        assert data["results"][0]["value"] == 123.45

    def test_load_baseline_nonexistent(self):
        """Test loading baseline from non-existent file."""
        baseline = self.framework.load_baseline("nonexistent.json")
        assert baseline == {}

    def test_load_baseline_existing(self):
        """Test loading baseline from existing file."""
        # Create baseline file
        baseline_data = {
            "results": [
                {"name": "test1", "value": 100.0},
                {"name": "test2", "value": 200.0}
            ]
        }

        baseline_file = self.framework.results_dir / "test_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f)

        baseline = self.framework.load_baseline("test_baseline.json")
        assert baseline["test1"] == 100.0
        assert baseline["test2"] == 200.0

    def test_detect_regressions(self):
        """Test regression detection logic."""
        # Create baseline
        baseline_data = {
            "results": [
                {"name": "stable_test", "value": 100.0},
                {"name": "regression_test", "value": 100.0},
                {"name": "improvement_test", "value": 100.0}
            ]
        }

        baseline_file = self.framework.results_dir / "test_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f)

        # Add current results
        self.framework.current_results = [
            BenchmarkResult("stable_test", "value", 98.0, "units", time.time(), {}),
            BenchmarkResult("regression_test", "value", 90.0, "units", time.time(), {}),
            BenchmarkResult("improvement_test", "value", 110.0, "units", time.time(), {}),
            BenchmarkResult("new_test", "value", 50.0, "units", time.time(), {})
        ]

        regressions = self.framework.detect_regressions("test_baseline.json", threshold=0.05)

        # Should detect regression_test (10% drop)
        assert len(regressions) == 1
        assert regressions[0][0] == "regression_test"
        assert abs(regressions[0][1] - (-10.0)) < 0.1  # -10% regression

    def test_generate_summary(self):
        """Test summary generation."""
        # Add results with targets
        self.framework.current_results = [
            BenchmarkResult("pass_test", "value", 100.0, "units", time.time(), {},
                          target_min=50.0, target_max=150.0),
            BenchmarkResult("fail_test", "value", 200.0, "units", time.time(), {},
                          target_min=50.0, target_max=150.0),
            BenchmarkResult("no_target_test", "value", 75.0, "units", time.time(), {})
        ]

        summary = self.framework._generate_summary()

        assert summary["total_benchmarks"] == 3
        assert summary["targets_met"] == 1
        assert summary["total_targets"] == 2
        assert summary["target_pass_rate"] == 0.5
        assert len(summary["benchmark_names"]) == 3


class TestAlphaZeroPerformanceBenchmarks:
    """Test AlphaZero-specific benchmarks."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = BenchmarkFramework(results_dir=self.temp_dir)
        self.benchmarks = AlphaZeroPerformanceBenchmarks(self.framework)

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)

    def test_mcts_simulation_benchmark(self):
        """Test MCTS simulation rate benchmark."""
        result = self.benchmarks.benchmark_mcts_simulation_rate()

        assert result.name == "mcts_simulations_per_second"
        assert result.metric_name == "simulations/sec"
        assert result.unit == "sims/sec"
        assert result.target_min == 30000.0
        assert result.target_max == 50000.0
        assert result.value > 0

    def test_neural_inference_benchmark(self):
        """Test neural network inference benchmark."""
        result = self.benchmarks.benchmark_neural_inference_throughput()

        assert result.name == "neural_inference_throughput"
        assert result.metric_name == "inferences/sec"
        assert result.unit == "inf/sec"
        assert result.target_min == 1000.0
        assert result.value > 0

    @patch('tests.performance.test_benchmarks.TORCH_AVAILABLE', False)
    def test_neural_inference_cpu_fallback(self):
        """Test neural inference benchmark CPU fallback."""
        result = self.benchmarks.benchmark_neural_inference_throughput()

        assert result.value > 0  # Should still work without PyTorch

    def test_gpu_utilization_benchmark(self):
        """Test GPU utilization benchmark."""
        result = self.benchmarks.benchmark_gpu_utilization()

        assert result.name == "gpu_utilization_percent"
        assert result.metric_name == "GPU utilization"
        assert result.unit == "%"
        assert result.target_min == 80.0
        assert result.target_max == 95.0
        assert result.value > 0

    def test_memory_efficiency_benchmark(self):
        """Test memory efficiency benchmark."""
        result = self.benchmarks.benchmark_memory_efficiency()

        assert result.name == "tree_memory_usage_mb"
        assert result.metric_name == "memory usage"
        assert result.unit == "MB"
        assert result.target_max == 1024.0
        assert result.value > 0

    def test_coordinator_performance_benchmark(self):
        """Test search coordinator performance benchmark."""
        result = self.benchmarks.benchmark_search_coordinator_performance()

        assert result.name == "coordinator_operations_per_second"
        assert result.metric_name == "operations/sec"
        assert result.unit == "ops/sec"
        assert result.target_min == 50000.0
        assert result.value > 0

    def test_all_benchmarks_complete(self):
        """Test that all benchmarks can be run successfully."""
        # Run all benchmarks
        results = [
            self.benchmarks.benchmark_mcts_simulation_rate(),
            self.benchmarks.benchmark_neural_inference_throughput(),
            self.benchmarks.benchmark_gpu_utilization(),
            self.benchmarks.benchmark_memory_efficiency(),
            self.benchmarks.benchmark_search_coordinator_performance()
        ]

        assert len(results) == 5
        assert all(r.value > 0 for r in results)
        assert len(self.framework.current_results) == 5

    def test_benchmark_framework_integration(self):
        """Test integration between benchmark framework and AlphaZero benchmarks."""
        # Run a few benchmarks
        self.benchmarks.benchmark_mcts_simulation_rate()
        self.benchmarks.benchmark_memory_efficiency()

        # Save results
        results_file = self.framework.save_results("integration_test.json")
        assert results_file.exists()

        # Verify results structure
        with open(results_file, 'r') as f:
            data = json.load(f)

        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 2
        assert data["summary"]["total_benchmarks"] == 2


class TestBenchmarkIntegration:
    """Integration tests for the complete benchmark system."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)

    def test_complete_benchmark_workflow(self):
        """Test the complete benchmark workflow."""
        framework = BenchmarkFramework(results_dir=self.temp_dir)
        benchmarks = AlphaZeroPerformanceBenchmarks(framework)

        # Run benchmarks
        benchmarks.benchmark_mcts_simulation_rate()
        benchmarks.benchmark_neural_inference_throughput()

        # Save as baseline
        framework.save_results("baseline.json")

        # Create new framework instance (simulating different run)
        framework2 = BenchmarkFramework(results_dir=self.temp_dir)
        benchmarks2 = AlphaZeroPerformanceBenchmarks(framework2)

        # Run benchmarks again (simulate slight performance change)
        result1 = benchmarks2.benchmark_mcts_simulation_rate()
        result2 = benchmarks2.benchmark_neural_inference_throughput()

        # Artificially create a regression for testing
        result1.value *= 0.85  # 15% regression

        # Check regression detection
        regressions = framework2.detect_regressions("baseline.json", threshold=0.05)

        # Should detect the artificial regression
        assert len(regressions) >= 1

    def test_benchmark_persistence_and_loading(self):
        """Test benchmark result persistence and loading."""
        framework = BenchmarkFramework(results_dir=self.temp_dir)

        # Create test results
        test_results = [
            BenchmarkResult("test1", "metric1", 100.0, "units", time.time(), {}),
            BenchmarkResult("test2", "metric2", 200.0, "units", time.time(), {})
        ]
        framework.current_results = test_results

        # Save and reload
        saved_file = framework.save_results("persistence_test.json")
        loaded_baseline = framework.load_baseline("persistence_test.json")

        assert "test1" in loaded_baseline
        assert "test2" in loaded_baseline
        assert loaded_baseline["test1"] == 100.0
        assert loaded_baseline["test2"] == 200.0

    def test_error_handling_robustness(self):
        """Test error handling in various scenarios."""
        framework = BenchmarkFramework(results_dir=self.temp_dir)

        # Test with failing benchmark function
        def always_fails():
            raise RuntimeError("Intentional failure")

        result = framework.run_benchmark("error_test", always_fails, iterations=3)
        assert result.value == 0.0  # Failure indicator

        # Test loading corrupted baseline
        corrupted_file = framework.results_dir / "corrupted.json"
        with open(corrupted_file, 'w') as f:
            f.write("invalid json content {")

        baseline = framework.load_baseline("corrupted.json")
        assert baseline == {}  # Should handle gracefully

    @patch('tests.performance.test_benchmarks.PYNVML_AVAILABLE', False)
    def test_system_without_gpu_monitoring(self):
        """Test system metrics capture without GPU monitoring."""
        metrics = SystemMetrics.capture()

        assert metrics.cpu_percent >= 0
        assert metrics.memory_mb > 0
        assert metrics.gpu_utilization is None
        assert metrics.gpu_memory_mb is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])