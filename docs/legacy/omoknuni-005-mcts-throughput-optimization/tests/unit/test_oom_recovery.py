"""
Unit Tests for OOM Recovery Mechanisms (T050)
=============================================

Tests for CUDA out-of-memory detection, automatic batch size reduction,
and graceful degradation in the GPU inference worker.
"""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple

# Import the inference worker
from src.neural.inference_worker import GPUInferenceWorker


class MockOOMException(RuntimeError):
    """Mock CUDA OOM exception for testing."""
    def __init__(self, message="CUDA out of memory"):
        super().__init__(message)


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a temporary mock model file."""
    model_path = tmp_path / "mock_model.pth"

    # Create a minimal mock model state dict
    mock_state = {
        'conv1.weight': torch.randn(256, 36, 3, 3),  # Gomoku input channels
        'conv1.bias': torch.randn(256),
    }
    torch.save(mock_state, model_path)
    return str(model_path)


@pytest.fixture
def oom_worker(mock_model_path):
    """Create an inference worker configured for OOM testing."""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('pynvml.nvmlInit'), \
         patch('pynvml.nvmlDeviceGetHandleByIndex'), \
         patch('src.neural.inference_worker.create_model_for_game') as mock_create:

        # Mock the model
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        mock_create.return_value = mock_model

        worker = GPUInferenceWorker(
            model_path=mock_model_path,
            device='cuda:0',
            batch_size=64,
            timeout_ms=3.0,
            use_mixed_precision=True
        )

        # Set up model mock
        worker.model = mock_model
        worker.input_shape = (36, 15, 15)

        return worker


@pytest.mark.gpu
class TestOOMDetection:
    """Test OOM error detection functionality."""

    def test_is_oom_error_detection(self, oom_worker):
        """Test OOM error detection with various error messages."""
        # Standard CUDA OOM errors
        assert oom_worker._is_oom_error(RuntimeError("CUDA out of memory"))
        assert oom_worker._is_oom_error(RuntimeError("cuda error: out of memory"))
        assert oom_worker._is_oom_error(RuntimeError("CUDA runtime error: out of memory"))
        assert oom_worker._is_oom_error(RuntimeError("allocation failure"))
        assert oom_worker._is_oom_error(RuntimeError("memory exhausted"))

        # Case insensitive
        assert oom_worker._is_oom_error(RuntimeError("Out Of Memory"))
        assert oom_worker._is_oom_error(RuntimeError("CUDA OUT OF MEMORY"))

        # Non-OOM errors
        assert not oom_worker._is_oom_error(RuntimeError("invalid device"))
        assert not oom_worker._is_oom_error(RuntimeError("model not found"))
        assert not oom_worker._is_oom_error(ValueError("invalid input"))

    def test_oom_error_tracking(self, oom_worker):
        """Test OOM error counting and tracking."""
        # Initial state
        assert oom_worker._oom_count == 0
        assert oom_worker._consecutive_oom_count == 0
        assert oom_worker._last_oom_time == 0.0

        # Simulate OOM recovery
        with patch('torch.cuda.empty_cache'):
            result = oom_worker._handle_oom_recovery()

        # Check tracking
        assert oom_worker._oom_count == 1
        assert oom_worker._consecutive_oom_count == 1
        assert oom_worker._last_oom_time > 0
        assert result is True  # Should attempt recovery

    def test_consecutive_oom_limit(self, oom_worker):
        """Test that too many consecutive OOM errors trigger CPU fallback."""
        with patch('torch.cuda.empty_cache'):
            # Simulate 3 consecutive OOM errors
            assert oom_worker._handle_oom_recovery() is True  # 1st OOM
            assert oom_worker._handle_oom_recovery() is True  # 2nd OOM
            assert oom_worker._handle_oom_recovery() is False  # 3rd OOM - fallback

        assert oom_worker._consecutive_oom_count == 3


@pytest.mark.gpu
class TestBatchSizeReduction:
    """Test automatic batch size reduction on OOM."""

    def test_batch_size_reduction_on_oom(self, oom_worker):
        """Test that batch size is reduced when OOM occurs."""
        original_batch_size = oom_worker.batch_size
        original_optimal = oom_worker._current_optimal_batch

        with patch('torch.cuda.empty_cache'), \
             patch.object(oom_worker, '_setup_pinned_memory_buffers'):

            result = oom_worker._handle_oom_recovery()

        # Batch size should be reduced
        expected_size = max(
            oom_worker._min_batch_size,
            int(original_batch_size * oom_worker._batch_size_reduction_factor)
        )
        assert oom_worker.batch_size == expected_size
        assert oom_worker._current_optimal_batch <= expected_size
        assert result is True

    def test_minimum_batch_size_limit(self, oom_worker):
        """Test that batch size doesn't go below minimum."""
        # Set batch size near minimum
        oom_worker.batch_size = oom_worker._min_batch_size + 1

        with patch('torch.cuda.empty_cache'), \
             patch.object(oom_worker, '_setup_pinned_memory_buffers'):

            result = oom_worker._handle_oom_recovery()

        # Should reach minimum and still attempt recovery
        assert oom_worker.batch_size == oom_worker._min_batch_size
        assert result is True

        # Next OOM at minimum should trigger CPU fallback
        with patch('torch.cuda.empty_cache'):
            result = oom_worker._handle_oom_recovery()

        assert result is False  # Should fallback to CPU

    def test_batch_size_increase_after_success(self, oom_worker):
        """Test gradual batch size increase after successful operations."""
        # Reduce batch size first
        original_size = oom_worker.batch_size
        oom_worker.batch_size = 16
        oom_worker._last_oom_time = time.time() - 100  # Long ago
        oom_worker._consecutive_oom_count = 0

        # Mock favorable conditions
        with patch.object(oom_worker, '_get_memory_usage_fraction', return_value=0.5), \
             patch.object(oom_worker, '_setup_pinned_memory_buffers'):

            oom_worker._attempt_batch_size_increase()

        # Batch size should increase but not exceed original
        assert oom_worker.batch_size > 16
        assert oom_worker.batch_size <= original_size

    def test_no_increase_with_high_memory(self, oom_worker):
        """Test that batch size doesn't increase when memory usage is high."""
        oom_worker.batch_size = 16
        oom_worker._last_oom_time = time.time() - 100

        # Mock high memory usage
        with patch.object(oom_worker, '_get_memory_usage_fraction', return_value=0.95):
            oom_worker._attempt_batch_size_increase()

        # Batch size should not increase
        assert oom_worker.batch_size == 16

    def test_no_increase_after_recent_oom(self, oom_worker):
        """Test that batch size doesn't increase soon after OOM."""
        oom_worker.batch_size = 16
        oom_worker._last_oom_time = time.time()  # Very recent

        with patch.object(oom_worker, '_get_memory_usage_fraction', return_value=0.5):
            oom_worker._attempt_batch_size_increase()

        # Batch size should not increase
        assert oom_worker.batch_size == 16


@pytest.mark.gpu
class TestOOMRecoveryIntegration:
    """Test OOM recovery in the full inference pipeline."""

    def test_oom_recovery_with_retry(self, oom_worker):
        """Test OOM recovery with successful retry."""
        positions = [np.random.randn(36, 15, 15) for _ in range(32)]

        # Mock model inference to fail first time (OOM), succeed second time
        call_count = 0
        def mock_inference_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise MockOOMException("CUDA out of memory")
            else:
                # Return mock successful inference
                batch_size = len(args[0]) if args else 32
                return (
                    torch.randn(batch_size, 225),  # policy logits (15x15)
                    torch.randn(batch_size, 1)     # values
                )

        with patch.object(oom_worker, '_create_batch_tensor_optimized', return_value=torch.randn(32, 36, 15, 15)), \
             patch.object(oom_worker, '_run_inference_with_precision', side_effect=mock_inference_side_effect), \
             patch.object(oom_worker, '_transfer_outputs_optimized', return_value=(np.random.randn(32, 225), np.random.randn(32))), \
             patch('torch.cuda.empty_cache'):

            result = oom_worker.batch_inference(positions)

        # Should succeed after OOM recovery
        assert result is not None
        assert len(result) == 2  # policies, values
        assert oom_worker._oom_count == 1
        assert oom_worker._consecutive_oom_count == 0  # Reset after success

    def test_oom_recovery_with_chunk_processing(self, oom_worker):
        """Test OOM recovery with chunk processing for large batches."""
        # Large batch that requires chunking
        positions = [np.random.randn(36, 15, 15) for _ in range(128)]

        # Set small batch size after OOM
        oom_worker.batch_size = 16

        with patch.object(oom_worker, '_create_batch_tensor_optimized') as mock_tensor, \
             patch.object(oom_worker, '_run_inference_with_precision') as mock_inference, \
             patch.object(oom_worker, '_transfer_outputs_optimized') as mock_transfer:

            # Mock successful chunk processing
            mock_tensor.return_value = torch.randn(16, 36, 15, 15)
            mock_inference.return_value = (torch.randn(16, 225), torch.randn(16, 1))
            mock_transfer.return_value = (np.random.randn(16, 225), np.random.randn(16))

            result = oom_worker._process_batch_chunks(positions)

        assert result is not None
        assert len(result) == 2
        assert result[0].shape[0] == 128  # All positions processed
        assert mock_inference.call_count == 8  # 128 / 16 = 8 chunks

    def test_persistent_oom_fallback_to_cpu(self, oom_worker):
        """Test fallback to CPU when OOM persists after recovery attempts."""
        positions = [np.random.randn(36, 15, 15) for _ in range(16)]

        # Mock persistent OOM errors
        def always_oom(*args, **kwargs):
            raise MockOOMException("CUDA out of memory")

        with patch.object(oom_worker, '_create_batch_tensor_optimized'), \
             patch.object(oom_worker, '_run_inference_with_precision', side_effect=always_oom), \
             patch.object(oom_worker, '_enable_cpu_fallback') as mock_fallback, \
             patch('torch.cuda.empty_cache'):

            try:
                oom_worker.batch_inference(positions)
            except Exception:
                pass  # Expected to fail after fallback attempts

        # Should attempt CPU fallback
        assert mock_fallback.called
        assert oom_worker._consecutive_oom_count >= 1


@pytest.mark.gpu
class TestOOMMetrics:
    """Test OOM-related metrics collection."""

    def test_oom_metrics_in_get_metrics(self, oom_worker):
        """Test that OOM metrics are included in metrics output."""
        # Trigger some OOM events
        with patch('torch.cuda.empty_cache'):
            oom_worker._handle_oom_recovery()
            oom_worker._handle_oom_recovery()

        with patch.object(oom_worker, '_get_memory_usage_fraction', return_value=0.85):
            metrics = oom_worker.get_mixed_precision_metrics()

        # Check OOM metrics are present
        assert 'oom_recovery_enabled' in metrics
        assert 'oom_total_count' in metrics
        assert 'oom_consecutive_count' in metrics
        assert 'original_batch_size' in metrics
        assert 'oom_min_batch_size' in metrics
        assert 'memory_usage_fraction' in metrics
        assert 'memory_usage_high_risk' in metrics

        # Check values
        assert metrics['oom_total_count'] == 2
        assert metrics['oom_consecutive_count'] == 2
        assert metrics['memory_usage_fraction'] == 0.85
        assert metrics['memory_usage_high_risk'] is False  # 0.85 < 0.9 threshold

    def test_memory_usage_fraction_calculation(self, oom_worker):
        """Test memory usage fraction calculation."""
        with patch('torch.cuda.memory_allocated', return_value=1024**3), \
             patch('torch.cuda.get_device_properties') as mock_props:

            mock_device = Mock()
            mock_device.total_memory = 8 * 1024**3  # 8GB
            mock_props.return_value = mock_device

            fraction = oom_worker._get_memory_usage_fraction()

        assert abs(fraction - 0.125) < 0.001  # 1GB / 8GB = 0.125

    def test_oom_recovery_state_reset(self, oom_worker):
        """Test OOM recovery state reset after successful operations."""
        # Set some OOM state
        oom_worker._consecutive_oom_count = 2
        oom_worker._last_successful_batch_size = 32

        # Reset state
        oom_worker._reset_oom_recovery_state()

        assert oom_worker._consecutive_oom_count == 0
        assert oom_worker._last_successful_batch_size == oom_worker.batch_size


@pytest.fixture
def mock_worker_with_cpu_fallback(mock_model_path):
    """Create a worker with CPU fallback enabled."""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('pynvml.nvmlInit'), \
         patch('pynvml.nvmlDeviceGetHandleByIndex'), \
         patch('src.neural.inference_worker.create_model_for_game') as mock_create, \
         patch('src.neural.cpu_inference.CPUInferenceWorker') as mock_cpu:

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        mock_create.return_value = mock_model

        # Mock CPU fallback worker
        mock_cpu_instance = Mock()
        mock_cpu_instance.batch_inference.return_value = (
            np.random.randn(16, 225),  # policies
            np.random.randn(16)        # values
        )
        mock_cpu.return_value = mock_cpu_instance

        worker = GPUInferenceWorker(
            model_path=mock_model_path,
            device='cuda:0',
            batch_size=64
        )
        worker.model = mock_model
        worker.input_shape = (36, 15, 15)

        return worker, mock_cpu_instance


@pytest.mark.integration
class TestOOMRecoveryIntegrationFlow:
    """Integration tests for the complete OOM recovery flow."""

    def test_complete_oom_recovery_flow(self, mock_worker_with_cpu_fallback):
        """Test complete flow from OOM detection to CPU fallback."""
        worker, mock_cpu = mock_worker_with_cpu_fallback
        positions = [np.random.randn(36, 15, 15) for _ in range(16)]

        # Set the CPU fallback worker directly to use the mock
        worker._cpu_fallback_worker = mock_cpu

        # Simulate complete OOM failure leading to CPU fallback
        with patch.object(worker, '_create_batch_tensor_optimized'), \
             patch.object(worker, '_run_inference_with_precision', side_effect=MockOOMException("CUDA out of memory")), \
             patch('torch.cuda.empty_cache'):

            result = worker.batch_inference(positions)

        # Should succeed using CPU fallback
        assert result is not None
        assert len(result) == 2
        assert mock_cpu.batch_inference.called

    def test_oom_recovery_configuration_parameters(self, oom_worker):
        """Test that OOM recovery configuration parameters are set correctly."""
        assert oom_worker._oom_recovery_enabled is True
        assert oom_worker._original_batch_size == 64
        assert oom_worker._min_batch_size == max(1, 64 // 16)  # 4
        assert oom_worker._batch_size_reduction_factor == 0.5
        assert oom_worker._oom_recovery_cooldown == 60.0
        assert oom_worker._oom_memory_threshold == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])