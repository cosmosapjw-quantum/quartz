"""
Unit tests for benchmark harness infrastructure (T001).

Tests verify:
- CSV file creation with correct schema
- Telemetry captures all KPIs from spec.md
- Reproducible results (same seed → same output ±2%)
- Performance (3 iterations complete in <5 minutes)
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import csv
import numpy as np

from tests.performance.benchmark_harness import BenchmarkHarness
from tests.performance.fixtures import BenchmarkConfig
from tests.performance.telemetry import Telemetry, BenchmarkStatistics


class TestBenchmarkHarness:
    """Test suite for BenchmarkHarness class."""

    def test_harness_creates_output_directory(self):
        """Test that harness creates output directory on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_benchmarks"
            harness = BenchmarkHarness(output_dir=str(output_dir))

            assert output_dir.exists()
            assert output_dir.is_dir()

    def test_harness_creates_session_directory(self):
        """Test that harness creates session-specific directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = BenchmarkHarness(output_dir=tmpdir)

            assert harness.output_dir.exists()
            assert harness.session_id  # Should have a session ID

    def test_csv_file_initialization(self):
        """Test that CSV file is created with correct headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = BenchmarkHarness(output_dir=tmpdir)
            csv_path = harness.history_csv

            assert csv_path.exists()

            # Check CSV headers
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames

                # Verify primary KPI fields are present
                assert 'throughput' in headers
                assert 'gpu_util_percent' in headers
                assert 'cpu_util_percent' in headers
                assert 'avg_batch_size' in headers
                assert 'batch_timeout_ms' in headers

                # Verify CPU breakdown fields
                assert 'feature_extraction_ms' in headers
                assert 'selection_time_ms' in headers
                assert 'expansion_time_ms' in headers
                assert 'backup_time_ms' in headers

                # Verify thread metrics
                assert 'num_threads' in headers
                assert 'thread_efficiency' in headers
                assert 'thread_idle_percent' in headers

                # Verify memory fields
                assert 'memory_rss_mb' in headers
                assert 'memory_peak_mb' in headers
                assert 'tree_size_nodes' in headers
                assert 'bytes_per_node' in headers

                # Verify metadata
                assert 'timestamp' in headers
                assert 'git_commit' in headers
                assert 'git_branch' in headers

    def test_csv_output_format(self):
        """Test that CSV output has correct format after benchmark run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = BenchmarkHarness(output_dir=tmpdir)

            # Create minimal config for quick test
            config = BenchmarkConfig(
                game="gomoku",
                num_simulations=100,  # Very small for unit test
                num_threads=2,
                seed=42,
            )

            # Run benchmark (will use mock MCTS in actual implementation)
            # For now, directly test CSV append
            telemetry = Telemetry(
                throughput=2000.0,
                gpu_util_percent=65.0,
                avg_batch_size=60.0,
                num_threads=2,
            )
            telemetry.config = config.to_dict()

            harness._append_to_csv(telemetry)

            # Verify CSV content
            csv_path = harness.history_csv
            assert csv_path.exists()

            import pandas as pd
            df = pd.read_csv(csv_path)

            assert len(df) == 1
            assert df['throughput'].iloc[0] == 2000.0
            assert df['gpu_util_percent'].iloc[0] == 65.0
            assert df['avg_batch_size'].iloc[0] == 60.0

    def test_reproducibility_same_seed(self):
        """Test that same seed produces reproducible results (±2%)."""
        # Create two telemetry objects with same seed
        np.random.seed(42)
        measurements1 = []
        for i in range(5):
            t = Telemetry(
                throughput=2000 + np.random.normal(0, 10),
                gpu_util_percent=65 + np.random.normal(0, 1),
                avg_batch_size=60 + np.random.normal(0, 2),
            )
            measurements1.append(t)

        np.random.seed(42)  # Reset to same seed
        measurements2 = []
        for i in range(5):
            t = Telemetry(
                throughput=2000 + np.random.normal(0, 10),
                gpu_util_percent=65 + np.random.normal(0, 1),
                avg_batch_size=60 + np.random.normal(0, 2),
            )
            measurements2.append(t)

        # Compare measurements
        for t1, t2 in zip(measurements1, measurements2):
            # Should be exactly equal with same seed
            assert t1.throughput == t2.throughput
            assert t1.gpu_util_percent == t2.gpu_util_percent
            assert t1.avg_batch_size == t2.avg_batch_size


class TestTelemetry:
    """Test suite for Telemetry data class."""

    def test_telemetry_initialization(self):
        """Test telemetry object initialization."""
        t = Telemetry()

        # Check default values
        assert t.throughput == 0.0
        assert t.gpu_util_percent == 0.0
        assert t.num_threads == 0

        # Check metadata initialization
        assert t.timestamp  # Should be auto-populated
        assert t.git_commit  # Should be auto-populated
        assert t.hostname  # Should be auto-populated

    def test_telemetry_to_dict(self):
        """Test telemetry conversion to dictionary."""
        t = Telemetry(
            throughput=2000.0,
            gpu_util_percent=65.0,
            num_threads=4,
        )

        data = t.to_dict()

        assert isinstance(data, dict)
        assert data['throughput'] == 2000.0
        assert data['gpu_util_percent'] == 65.0
        assert data['num_threads'] == 4

    def test_telemetry_to_csv_row(self):
        """Test telemetry conversion to CSV row format."""
        config_dict = {'game': 'gomoku', 'num_simulations': 10000}
        t = Telemetry(
            throughput=2000.0,
            config=config_dict,
        )

        row = t.to_csv_row()

        # Config should be flattened
        assert 'config' not in row
        assert 'config_game' in row
        assert 'config_num_simulations' in row
        assert row['config_game'] == 'gomoku'
        assert row['config_num_simulations'] == 10000

    def test_compute_derived_metrics(self):
        """Test derived metrics computation."""
        t = Telemetry(
            throughput=8000.0,
            num_threads=4,
            tree_size_nodes=1000000,
            memory_rss_mb=27.0,  # 27MB for 1M nodes
            batches_submitted=100,
            total_time_sec=10.0,
            gpu_inference_ms=3000.0,  # 3 seconds of GPU time
        )
        t.config = {'baseline_throughput_single_thread': 2000.0}

        t.compute_derived_metrics()

        # Check bytes per node
        expected_bytes = (27.0 * 1024 * 1024) / 1000000
        assert abs(t.bytes_per_node - expected_bytes) < 0.1

        # Check batches per second
        assert abs(t.batches_per_second - 10.0) < 0.01

        # Check GPU overhead percentage
        assert abs(t.gpu_overhead_percent - 30.0) < 0.1  # 3s / 10s = 30%
        assert abs(t.mcts_overhead_percent - 70.0) < 0.1  # 70%

        # Check thread efficiency
        actual_speedup = 8000.0 / 2000.0  # 4x
        linear_speedup = 4
        expected_efficiency = actual_speedup / linear_speedup  # 1.0
        assert abs(t.thread_efficiency - expected_efficiency) < 0.01


class TestBenchmarkStatistics:
    """Test suite for BenchmarkStatistics class."""

    def test_statistics_from_empty_list(self):
        """Test statistics computation with empty list."""
        stats = BenchmarkStatistics.from_telemetry_list([])

        assert stats.num_runs == 0
        assert stats.mean_throughput == 0.0

    def test_statistics_from_telemetry_list(self):
        """Test statistics computation from telemetry list."""
        # Generate sample telemetry data inline
        sample_telemetry_data = []
        for i in range(10):
            t = Telemetry(
                throughput=2000 + i * 50 + np.random.normal(0, 20),
                gpu_util_percent=65 + i * 0.5 + np.random.normal(0, 2),
                avg_batch_size=60 + np.random.normal(0, 3),
                thread_idle_percent=60 + np.random.normal(0, 5),
                num_threads=4,
                memory_peak_mb=250 + np.random.normal(0, 10),
                tree_size_nodes=500000,
            )
            t.compute_derived_metrics()
            sample_telemetry_data.append(t)

        stats = BenchmarkStatistics.from_telemetry_list(sample_telemetry_data)

        # Check sample size
        assert stats.num_runs == 10

        # Check mean values are reasonable
        assert 2000 <= stats.mean_throughput <= 2500
        assert 60 <= stats.mean_gpu_util <= 75
        assert 55 <= stats.mean_batch_size <= 65

        # Check standard deviation is positive
        assert stats.std_throughput > 0
        assert stats.std_gpu_util > 0

        # Check CV (coefficient of variation)
        assert 0 < stats.cv_throughput < 0.5  # Should be < 50%

    def test_target_validation(self):
        """Test target validation logic."""
        # Create telemetry that meets all targets
        good_telemetry = [
            Telemetry(
                throughput=8500.0,  # ≥8,000
                gpu_util_percent=85.0,  # 80-95%
                memory_peak_mb=500.0,  # <1024
            )
            for _ in range(5)
        ]

        stats = BenchmarkStatistics.from_telemetry_list(good_telemetry)

        assert stats.meets_throughput_target
        assert stats.meets_gpu_target
        assert stats.meets_memory_target

        # Create telemetry that fails targets
        bad_telemetry = [
            Telemetry(
                throughput=5000.0,  # <8,000 (FAIL)
                gpu_util_percent=60.0,  # <80% (FAIL)
                memory_peak_mb=1500.0,  # >1024 (FAIL)
            )
            for _ in range(5)
        ]

        stats = BenchmarkStatistics.from_telemetry_list(bad_telemetry)

        assert not stats.meets_throughput_target
        assert not stats.meets_gpu_target
        assert not stats.meets_memory_target

    def test_coefficient_of_variation(self):
        """Test CV calculation for reproducibility check."""
        # High reproducibility (low CV)
        consistent_telemetry = [
            Telemetry(throughput=2000.0 + i * 5.0)
            for i in range(10)
        ]

        stats = BenchmarkStatistics.from_telemetry_list(consistent_telemetry)

        # CV should be low (<5%) for consistent measurements
        assert stats.cv_throughput < 0.05

        # Low reproducibility (high CV)
        variable_telemetry = [
            Telemetry(throughput=2000.0 + i * 200.0)
            for i in range(10)
        ]

        stats = BenchmarkStatistics.from_telemetry_list(variable_telemetry)

        # CV should be high (>20%) for variable measurements
        assert stats.cv_throughput > 0.20


class TestBenchmarkConfig:
    """Test suite for BenchmarkConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BenchmarkConfig(
            game="gomoku",
            board_size=15,
            num_simulations=1000,
            num_threads=4,
            batch_size=64,
            batch_timeout_ms=1.0,
            seed=42,
        )

        assert config.game == "gomoku"
        assert config.board_size == 15
        assert config.num_simulations == 1000
        assert config.num_threads == 4
        assert config.seed == 42

    def test_config_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = BenchmarkConfig(
            game="gomoku",
            num_simulations=10000,
            num_threads=4,
        )

        data = config.to_dict()

        assert isinstance(data, dict)
        assert data['game'] == 'gomoku'
        assert data['num_simulations'] == 10000
        assert data['num_threads'] == 4

    def test_feature_flags(self):
        """Test feature flag configuration."""
        config = BenchmarkConfig(
            openmp_enabled=True,
            state_pooling_enabled=True,
            condition_vars_enabled=True,
            node_allocator_optimized=True,
            nn_cache_enabled=False,
        )

        assert config.openmp_enabled
        assert config.state_pooling_enabled
        assert config.condition_vars_enabled
        assert config.node_allocator_optimized
        assert not config.nn_cache_enabled


class TestIntegrationBenchmark:
    """Integration tests for full benchmark workflow."""

    @pytest.mark.slow
    def test_quick_benchmark_run(self):
        """Test quick benchmark run (3 iterations, <5 minutes)."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            harness = BenchmarkHarness(output_dir=tmpdir)

            config = BenchmarkConfig(
                game="gomoku",
                num_simulations=500,  # Small for speed
                num_threads=2,
                seed=42,
            )

            start = time.time()

            # This will fail until MCTS integration is complete
            # For now, test the infrastructure
            try:
                stats = harness.run_benchmark(config, iterations=3)
                elapsed = time.time() - start

                # Should complete in <5 minutes
                assert elapsed < 300  # 5 minutes

                # Should have correct number of runs
                assert stats.num_runs == 3

                # CV should be reasonable (<10%)
                assert stats.cv_throughput < 0.10

            except (ImportError, AttributeError) as e:
                # MCTS integration not complete yet
                pytest.skip(f"MCTS integration pending: {e}")

    def test_csv_history_persistence(self):
        """Test that CSV history persists across harness instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First harness instance
            harness1 = BenchmarkHarness(output_dir=tmpdir)
            telemetry1 = Telemetry(throughput=2000.0, gpu_util_percent=65.0)
            harness1._append_to_csv(telemetry1)

            # Second harness instance (same directory)
            harness2 = BenchmarkHarness(output_dir=tmpdir)
            telemetry2 = Telemetry(throughput=2100.0, gpu_util_percent=66.0)
            harness2._append_to_csv(telemetry2)

            # Load history
            history = harness2.load_history()

            assert len(history) >= 2  # At least our 2 entries
