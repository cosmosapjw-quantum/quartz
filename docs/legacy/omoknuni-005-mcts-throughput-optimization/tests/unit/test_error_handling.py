"""
Unit tests for error handling framework.

Tests comprehensive error handling, thread health monitoring, and
graceful degradation mechanisms.
"""

import pytest
import time
import threading
import queue
import torch
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, Future

from src.utils.errors import (
    AlphaZeroError, ModelError, ModelLoadError, ModelValidationError,
    InferenceError, CriticalInferenceError, MCTSError, TreeCorruptionError,
    MemoryAllocationError, ThreadCoordinationError, TrainingError,
    TrainingStabilityError, ErrorSeverity, ThreadHealthMonitor,
    GPUOperationManager, with_error_handling, validate_gpu_state,
    ErrorReporter, error_reporter
)
from src.neural.model_validator import ModelValidator


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_alphazero_error_base(self):
        """Test base AlphaZeroError functionality."""
        error = AlphaZeroError("Test error", ErrorSeverity.ERROR, {"key": "value"})

        assert str(error) == "Test error (Context: key=value)"
        assert error.severity == ErrorSeverity.ERROR
        assert error.context == {"key": "value"}
        assert error.timestamp > 0

    def test_alphazero_error_no_context(self):
        """Test AlphaZeroError without context."""
        error = AlphaZeroError("Test error")

        assert str(error) == "Test error"
        assert error.severity == ErrorSeverity.ERROR
        assert error.context == {}

    def test_model_load_error(self):
        """Test ModelLoadError with model path context."""
        error = ModelLoadError("Failed to load", model_path="/path/to/model.pth")

        assert "Failed to load" in str(error)
        assert error.context["model_path"] == "/path/to/model.pth"
        assert error.severity == ErrorSeverity.ERROR

    def test_model_validation_error(self):
        """Test ModelValidationError with expected/actual values."""
        error = ModelValidationError("Shape mismatch", expected=(2, 3), actual=(2, 4))

        assert "Shape mismatch" in str(error)
        assert error.context["expected"] == (2, 3)
        assert error.context["actual"] == (2, 4)

    def test_critical_inference_error(self):
        """Test CriticalInferenceError has critical severity."""
        error = CriticalInferenceError("Critical GPU error")

        assert error.severity == ErrorSeverity.CRITICAL
        assert "Critical GPU error" in str(error)

    def test_tree_corruption_error(self):
        """Test TreeCorruptionError with node context."""
        error = TreeCorruptionError("Node corrupted", node_id=12345)

        assert "Node corrupted" in str(error)
        assert error.context["node_id"] == 12345
        assert error.severity == ErrorSeverity.CRITICAL

    def test_thread_coordination_error(self):
        """Test ThreadCoordinationError with thread name."""
        error = ThreadCoordinationError("Thread failed", thread_name="worker_1")

        assert "Thread failed" in str(error)
        assert error.context["thread_name"] == "worker_1"


class TestThreadHealthMonitor:
    """Test thread health monitoring and failure tracking."""

    def test_initialization(self):
        """Test ThreadHealthMonitor initialization."""
        monitor = ThreadHealthMonitor(max_consecutive_failures=5, failure_backoff=0.5)

        assert monitor.max_failures == 5
        assert monitor.base_backoff == 0.5
        assert monitor.failure_counts == {}

    def test_record_success_clears_failures(self):
        """Test recording success clears failure count."""
        monitor = ThreadHealthMonitor()

        # Record some failures first
        monitor.failure_counts["test_thread"] = 3

        # Record success should clear failures
        monitor.record_success("test_thread")

        assert "test_thread" not in monitor.failure_counts

    def test_record_failure_tracking(self):
        """Test failure tracking and counting."""
        monitor = ThreadHealthMonitor(max_consecutive_failures=3, failure_backoff=0.01)

        exception = Exception("Test error")

        # First failure
        should_continue = monitor.record_failure("test_thread", exception)
        assert should_continue is True
        assert monitor.failure_counts["test_thread"] == 1

        # Second failure
        should_continue = monitor.record_failure("test_thread", exception)
        assert should_continue is True
        assert monitor.failure_counts["test_thread"] == 2

        # Third failure - should reach limit
        should_continue = monitor.record_failure("test_thread", exception)
        assert should_continue is False
        assert monitor.failure_counts["test_thread"] == 3

    def test_failure_backoff_timing(self):
        """Test exponential backoff timing."""
        monitor = ThreadHealthMonitor(max_consecutive_failures=5, failure_backoff=0.01)

        exception = Exception("Test error")

        start_time = time.time()
        monitor.record_failure("test_thread", exception)
        first_failure_time = time.time() - start_time

        # Should have minimal backoff for first failure
        assert first_failure_time >= 0.01  # At least the backoff time
        assert first_failure_time < 0.1    # But not too much overhead

    @patch('src.utils.errors.time.sleep')  # Mock sleep to avoid delays in tests
    def test_different_threads_tracked_separately(self, mock_sleep):
        """Test different threads are tracked independently."""
        monitor = ThreadHealthMonitor(max_consecutive_failures=2)

        exception = Exception("Test error")

        # Fail thread1 twice
        assert monitor.record_failure("thread1", exception) is True
        assert monitor.record_failure("thread1", exception) is False

        # thread2 should still be able to fail
        assert monitor.record_failure("thread2", exception) is True
        assert monitor.record_failure("thread2", exception) is False


class TestGPUOperationManager:
    """Test GPU operation management and timeout handling."""

    def test_initialization(self):
        """Test GPUOperationManager initialization."""
        manager = GPUOperationManager(default_timeout=10.0)
        assert manager.default_timeout == 10.0

    def test_successful_operation(self):
        """Test successful operation execution."""
        manager = GPUOperationManager()

        def test_operation():
            return "success"

        result = manager.execute_with_timeout(test_operation, timeout=5.0, operation_name="test")
        assert result == "success"

    def test_operation_with_cuda_oom(self):
        """Test handling of CUDA out of memory errors."""
        manager = GPUOperationManager()

        def failing_operation():
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        with pytest.raises(InferenceError) as exc_info:
            manager.execute_with_timeout(failing_operation, operation_name="test")

        assert "GPU out of memory" in str(exc_info.value)

    def test_operation_with_cuda_runtime_error(self):
        """Test handling of CUDA runtime errors."""
        manager = GPUOperationManager()

        def failing_operation():
            raise RuntimeError("CUDA error: device-side assert triggered")

        with pytest.raises(CriticalInferenceError) as exc_info:
            manager.execute_with_timeout(failing_operation, operation_name="test")

        assert "Critical GPU error" in str(exc_info.value)

    def test_operation_with_unexpected_error(self):
        """Test handling of unexpected errors."""
        manager = GPUOperationManager()

        def failing_operation():
            raise ValueError("Unexpected error")

        with pytest.raises(InferenceError) as exc_info:
            manager.execute_with_timeout(failing_operation, operation_name="test")

        assert "Unexpected error" in str(exc_info.value)


class TestErrorHandlingDecorator:
    """Test error handling decorator."""

    def test_decorator_success(self):
        """Test decorator with successful function."""
        @with_error_handling()
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_decorator_reraise_true(self):
        """Test decorator reraises exceptions by default."""
        @with_error_handling()
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    def test_decorator_reraise_false(self):
        """Test decorator returns default value when reraise=False."""
        @with_error_handling(reraise=False, default_return="default")
        def failing_function():
            raise ValueError("Test error")

        result = failing_function()
        assert result == "default"

    def test_decorator_preserves_custom_exceptions(self):
        """Test decorator preserves AlphaZero custom exceptions."""
        @with_error_handling()
        def function_with_custom_exception():
            raise ModelError("Custom model error")

        with pytest.raises(ModelError):
            function_with_custom_exception()


class TestGPUStateValidation:
    """Test GPU state validation."""

    @patch('torch.cuda.is_available', return_value=False)
    def test_cuda_not_available(self, mock_cuda_available):
        """Test validation when CUDA is not available."""
        device = torch.device('cuda:0')

        with pytest.raises(CriticalInferenceError) as exc_info:
            validate_gpu_state(device)

        assert "CUDA not available" in str(exc_info.value)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    def test_invalid_device_index(self, mock_device_count, mock_cuda_available):
        """Test validation with invalid device index."""
        device = torch.device('cuda:2')  # Only device 0 available

        with pytest.raises(CriticalInferenceError) as exc_info:
            validate_gpu_state(device)

        assert "Invalid CUDA device index" in str(exc_info.value)

    def test_cpu_device_validation(self):
        """Test validation passes for CPU device."""
        device = torch.device('cpu')

        # Should not raise any exception
        validate_gpu_state(device)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.memory_allocated', return_value=1024**3)  # 1GB
    @patch('torch.cuda.memory_reserved', return_value=2*1024**3)  # 2GB
    @patch('torch.cuda.get_device_properties')
    def test_memory_validation(self, mock_get_props, mock_reserved, mock_allocated,
                              mock_device_count, mock_cuda_available):
        """Test GPU memory validation."""
        # Mock device properties
        mock_props = Mock()
        mock_props.total_memory = 8 * 1024**3  # 8GB
        mock_get_props.return_value = mock_props

        device = torch.device('cuda:0')

        # Should not raise exception with reasonable memory usage
        validate_gpu_state(device)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.memory_allocated', return_value=8 * 1024**3)  # 8GB allocated
    @patch('torch.cuda.get_device_properties')
    def test_critical_memory_usage(self, mock_get_props, mock_allocated,
                                  mock_device_count, mock_cuda_available):
        """Test validation fails with critical memory usage."""
        # Mock device properties
        mock_props = Mock()
        mock_props.total_memory = 8 * 1024**3  # 8GB total
        mock_get_props.return_value = mock_props

        device = torch.device('cuda:0')

        with pytest.raises(InferenceError) as exc_info:
            validate_gpu_state(device)

        assert "GPU memory critically low" in str(exc_info.value)


class TestErrorReporter:
    """Test centralized error reporting."""

    def test_error_reporting(self):
        """Test error reporting and tracking."""
        reporter = ErrorReporter()

        error = ModelError("Test model error")
        context = {"model_path": "/test/path"}

        reporter.report_error(error, context)

        # Check error was recorded
        assert reporter.error_counts["ModelError"] == 1
        assert "ModelError" in reporter.last_errors
        assert reporter.last_errors["ModelError"]["message"] == str(error)
        assert reporter.last_errors["ModelError"]["context"] == context

    def test_error_count_accumulation(self):
        """Test error counts accumulate correctly."""
        reporter = ErrorReporter()

        # Report same error type multiple times
        for i in range(3):
            reporter.report_error(ValueError(f"Error {i}"))

        assert reporter.error_counts["ValueError"] == 3

    def test_error_summary(self):
        """Test error summary generation."""
        reporter = ErrorReporter()

        # Report different error types
        reporter.report_error(ModelError("Model error"))
        reporter.report_error(InferenceError("Inference error"))
        reporter.report_error(ModelError("Another model error"))

        summary = reporter.get_error_summary()

        assert summary["total_errors"] == 3
        assert summary["error_counts"]["ModelError"] == 2
        assert summary["error_counts"]["InferenceError"] == 1
        assert len(summary["recent_errors"]) == 2  # Two error types


class TestModelValidator:
    """Test neural network model validation."""

    def test_validator_initialization(self):
        """Test ModelValidator initialization."""
        validator = ModelValidator(
            expected_input_shape=(36, 15, 15),
            expected_output_shapes={"policy": (225,), "value": (1,)},
            device=torch.device('cpu')
        )

        assert validator.expected_input_shape == (36, 15, 15)
        assert validator.expected_output_shapes["policy"] == (225,)
        assert validator.device == torch.device('cpu')

    def test_file_validation_missing_file(self):
        """Test validation fails for missing file."""
        validator = ModelValidator(
            expected_input_shape=(36, 15, 15),
            expected_output_shapes={"policy": (225,), "value": (1,)},
            device=torch.device('cpu')
        )

        with pytest.raises(ModelLoadError) as exc_info:
            validator.validate_model_file("/nonexistent/path.pth")

        assert "does not exist" in str(exc_info.value)

    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_file', return_value=True)
    @patch('pathlib.Path.stat')
    def test_file_validation_empty_file(self, mock_stat, mock_is_file, mock_exists):
        """Test validation fails for empty file."""
        # Mock file stats
        mock_stat_result = Mock()
        mock_stat_result.st_size = 0
        mock_stat.return_value = mock_stat_result

        validator = ModelValidator(
            expected_input_shape=(36, 15, 15),
            expected_output_shapes={"policy": (225,), "value": (1,)},
            device=torch.device('cpu')
        )

        with pytest.raises(ModelLoadError) as exc_info:
            validator.validate_model_file("/test/path.pth")

        assert "empty" in str(exc_info.value)

    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_file', return_value=True)
    @patch('pathlib.Path.stat')
    @patch('builtins.open', create=True)
    def test_file_validation_success(self, mock_open, mock_stat, mock_is_file, mock_exists):
        """Test successful file validation."""
        # Mock file stats
        mock_stat_result = Mock()
        mock_stat_result.st_size = 1024 * 1024  # 1MB
        mock_stat.return_value = mock_stat_result

        # Mock file content for checksum using MagicMock for context manager support
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = False
        # Mock reading chunks for checksum calculation
        mock_file.read.side_effect = [b"test content", b""]  # Return content then EOF
        mock_open.return_value = mock_file

        validator = ModelValidator(
            expected_input_shape=(36, 15, 15),
            expected_output_shapes={"policy": (225,), "value": (1,)},
            device=torch.device('cpu')
        )

        result = validator.validate_model_file("/test/path.pth")

        assert result["size_bytes"] == 1024 * 1024
        assert result["size_mb"] == 1.0
        assert "checksum" in result


class TestIntegrationErrorHandling:
    """Integration tests for error handling across components."""

    def test_thread_health_with_search_coordinator_pattern(self):
        """Test thread health monitoring with search coordinator pattern."""
        monitor = ThreadHealthMonitor(max_consecutive_failures=2, failure_backoff=0.01)

        # Simulate thread coordinator loop with error handling
        thread_name = "test_coordinator"
        error_count = 0

        def simulate_coordinator_loop():
            nonlocal error_count

            for i in range(5):  # Simulate 5 operations
                try:
                    if i >= 2:  # Fail on iterations 2, 3, 4
                        raise InferenceError(f"Simulated error {i}")

                    # Successful operation
                    monitor.record_success(thread_name)

                except InferenceError as e:
                    error_count += 1
                    if not monitor.record_failure(thread_name, e):
                        break  # Thread should terminate

        simulate_coordinator_loop()

        # Should have failed twice and terminated
        assert error_count == 2
        assert monitor.failure_counts[thread_name] == 2

    def test_error_reporter_with_multiple_components(self):
        """Test error reporter with multiple components reporting errors."""
        # Use the global error reporter
        initial_count = error_reporter.get_error_summary()["total_errors"]

        # Simulate errors from different components
        model_error = ModelError("Model loading failed")
        inference_error = InferenceError("GPU inference failed")
        coordination_error = ThreadCoordinationError("Thread coordination failed")

        error_reporter.report_error(model_error, {"component": "model_loader"})
        error_reporter.report_error(inference_error, {"component": "inference_worker"})
        error_reporter.report_error(coordination_error, {"component": "search_coordinator"})

        summary = error_reporter.get_error_summary()

        # Should have 3 new errors
        assert summary["total_errors"] == initial_count + 3
        assert summary["error_counts"]["ModelError"] >= 1
        assert summary["error_counts"]["InferenceError"] >= 1
        assert summary["error_counts"]["ThreadCoordinationError"] >= 1

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.memory_allocated', return_value=512 * 1024**2)  # 512MB
    @patch('torch.cuda.memory_reserved', return_value=1024 * 1024**2)  # 1GB
    @patch('torch.cuda.get_device_properties')
    def test_gpu_operation_manager_integration(self, mock_get_props, mock_reserved,
                                             mock_allocated, mock_device_count, mock_cuda_available):
        """Test GPU operation manager integration with validation."""
        # Mock device properties
        mock_props = Mock()
        mock_props.total_memory = 8 * 1024**3  # 8GB
        mock_get_props.return_value = mock_props

        manager = GPUOperationManager()
        device = torch.device('cuda:0')

        # Validate GPU state first
        validate_gpu_state(device)

        # Then test operation
        def safe_operation():
            return torch.randn(10, 10, device=device)

        result = manager.execute_with_timeout(safe_operation, operation_name="tensor_creation")
        assert result.shape == (10, 10)
        assert result.device == device