"""
Simplified unit tests for the telemetry framework.

Tests core functionality of metrics collection and structured logging.
"""

import time
import json
import threading
from unittest.mock import Mock, patch, MagicMock
import pytest
import logging
from io import StringIO

from src.telemetry.metrics import MetricsCollector, PerformanceMetrics
from src.telemetry.logger import AlphaZeroLogger, LogLevel, LogContext


class TestMetricsCollectorBasic:
    """Basic test cases for MetricsCollector."""

    def test_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(collect_interval=0.5)

        assert collector.collect_interval == 0.5
        assert not collector._running
        assert collector.registry is not None

    def test_record_simulation(self):
        """Test recording MCTS simulations."""
        collector = MetricsCollector()

        # Record some simulations
        collector.record_simulation("gomoku", 0.001)
        collector.record_simulation("chess", 0.002)

        # Check that data was recorded
        assert len(collector._simulation_times) == 2

        # Check Prometheus metrics format
        metrics_output = collector.get_prometheus_metrics()
        assert "alphazero_simulations_total" in metrics_output

    def test_record_inference(self):
        """Test recording neural network inference."""
        collector = MetricsCollector()

        # Record inference requests
        collector.record_inference(32, 0.005)
        collector.record_inference(64, 0.008)

        # Check Prometheus metrics
        metrics_output = collector.get_prometheus_metrics()
        assert "alphazero_inference_requests_total" in metrics_output
        assert "alphazero_inference_duration_seconds" in metrics_output

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    def test_get_current_metrics(self, mock_cpu_percent, mock_virtual_memory):
        """Test getting current performance metrics."""
        # Mock system data
        mock_memory = Mock()
        mock_memory.used = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.total = 32 * 1024 * 1024 * 1024  # 32GB
        mock_virtual_memory.return_value = mock_memory
        mock_cpu_percent.return_value = 45.5

        collector = MetricsCollector()
        metrics = collector.get_current_metrics()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.system_memory_used_mb > 0
        assert metrics.system_memory_total_mb > 0
        assert metrics.cpu_percent == 45.5

    def test_simulation_rate_calculation(self):
        """Test simulations per second calculation."""
        collector = MetricsCollector()

        # Add simulation times manually for testing (within 10-second window)
        current_time = time.time()
        for i in range(50):
            collector._simulation_times.append(
                current_time - 5.0 + i * 0.1
            )  # 5 sims/sec over 10 seconds

        collector._update_simulation_rate()

        # Check rate is approximately correct (50 simulations over 10 seconds = 5 sims/sec)
        rate = collector.simulations_per_second._value._value
        assert 4.0 <= rate <= 6.0

    def test_reset_counters(self):
        """Test resetting performance counters."""
        collector = MetricsCollector()

        # Add some data
        collector.record_simulation("gomoku", 0.001)
        assert len(collector._simulation_times) > 0

        # Reset and verify
        collector.reset_counters()
        assert len(collector._simulation_times) == 0


class TestStructuredLoggingBasic:
    """Basic test cases for structured logging."""

    def test_logger_initialization(self):
        """Test AlphaZeroLogger initialization."""
        logger = AlphaZeroLogger(
            "test_component",
            level=LogLevel.DEBUG,
            enable_console=False,  # Disable console for testing
            enable_file=False,
        )

        assert logger.name == "test_component"
        assert logger.context.component == "test_component"

    def test_context_creation(self):
        """Test logging context functionality."""
        logger = AlphaZeroLogger("test", enable_console=False, enable_file=False)

        context_logger = logger.with_context(
            operation="test_operation", game_type="gomoku"
        )

        assert context_logger.context.operation == "test_operation"
        assert context_logger.context.game_type == "gomoku"

    def test_log_context_dataclass(self):
        """Test LogContext dataclass."""
        context = LogContext(component="test", operation="test_op", game_type="gomoku")

        assert context.component == "test"
        assert context.operation == "test_op"
        assert context.game_type == "gomoku"

    def test_performance_logging_method(self):
        """Test performance logging method exists."""
        logger = AlphaZeroLogger("test", enable_console=False, enable_file=False)

        # Should not raise an exception
        logger.log_performance("test_operation", 0.123)

    def test_simulation_batch_logging_method(self):
        """Test simulation batch logging method exists."""
        logger = AlphaZeroLogger("test", enable_console=False, enable_file=False)

        # Should not raise an exception
        logger.log_simulation_batch(
            game_type="gomoku",
            batch_size=100,
            simulations_per_second=25000.5,
            gpu_utilization=87.3,
            duration=0.004,
        )

    def test_inference_batch_logging_method(self):
        """Test inference batch logging method exists."""
        logger = AlphaZeroLogger("test", enable_console=False, enable_file=False)

        # Should not raise an exception
        logger.log_inference_batch(
            batch_size=64,
            inference_time=0.003,
            queue_wait_time=0.001,
            throughput=21333.3,
        )


class TestPrometheusMetrics:
    """Test Prometheus metrics format and content."""

    def test_prometheus_format_basic(self):
        """Test basic Prometheus metrics format."""
        collector = MetricsCollector()

        # Record some data
        collector.record_simulation("gomoku", 0.001)
        collector.record_inference(32, 0.005)

        prometheus_output = collector.get_prometheus_metrics()

        # Check basic format
        assert "# HELP" in prometheus_output
        assert "# TYPE" in prometheus_output
        assert "alphazero_simulations_total" in prometheus_output
        assert "alphazero_inference_requests_total" in prometheus_output

    def test_metrics_labels(self):
        """Test that metrics include proper labels."""
        collector = MetricsCollector()

        collector.record_simulation("gomoku", 0.001)
        collector.record_simulation("chess", 0.002)

        prometheus_output = collector.get_prometheus_metrics()

        # Should have game type labels
        assert 'game_type="gomoku"' in prometheus_output
        assert 'game_type="chess"' in prometheus_output


class TestThreadSafety:
    """Test thread safety of telemetry components."""

    def test_concurrent_metrics_recording(self):
        """Test thread safety of metrics recording."""
        collector = MetricsCollector()
        errors = []

        def record_metrics(thread_id):
            """Record metrics from a thread."""
            try:
                for i in range(10):  # Reduced iterations for faster test
                    collector.record_simulation("gomoku", 0.001)
                    collector.record_inference(32, 0.005)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Start multiple threads
        threads = []
        for i in range(3):  # Reduced thread count for faster test
            thread = threading.Thread(target=record_metrics, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check for errors
        assert not errors, f"Concurrent access errors: {errors}"

        # Verify data was recorded
        assert len(collector._simulation_times) == 30  # 3 threads * 10 simulations


class TestIntegrationBasic:
    """Basic integration tests."""

    def test_metrics_collector_creation(self):
        """Test that metrics collector can be created and used."""
        from src.telemetry import get_metrics_collector, cleanup_metrics

        # Clean up first
        cleanup_metrics()

        # Get collector
        collector = get_metrics_collector()
        assert collector is not None
        assert collector._running  # Should start automatically

        # Record some data
        collector.record_simulation("gomoku", 0.001)
        metrics = collector.get_current_metrics()
        assert isinstance(metrics, PerformanceMetrics)

        # Clean up
        cleanup_metrics()

    def test_logger_creation(self):
        """Test that logger can be created and used."""
        from src.telemetry import get_logger

        logger = get_logger("test_component")
        assert logger is not None
        assert logger.name == "test_component"

        # Test that multiple calls return same instance
        logger2 = get_logger("test_component")
        assert logger is logger2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
