"""
CPU Fallback Mechanism Unit Tests (T018)
========================================

Comprehensive test suite for CPU fallback functionality in neural network inference.
Tests automatic fallback scenarios, error handling, and performance monitoring.
"""

import pytest
import numpy as np
import torch
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock
from queue import Queue

# Import the classes we're testing
import sys
sys.path.append('.')
from src.neural.cpu_inference import CPUInferenceWorker, CPUFallbackInference, should_fallback_to_cpu, detect_gpu_failure
from src.neural.inference_worker import GPUInferenceWorker
from src.neural.model import create_model_for_game


class TestCPUInferenceWorker:
    """Test CPU-only inference worker implementation."""

    @pytest.fixture
    def model_path(self):
        """Create a temporary model file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name

            # Create and save a test model (save full model to avoid state_dict issues)
            model = create_model_for_game('gomoku')
            # Initialize lazy layers
            with torch.no_grad():
                dummy_input = torch.randn(1, 36, 15, 15)
                _ = model(dummy_input)
            torch.save(model, model_path)

        yield model_path

        # Cleanup
        os.unlink(model_path)

    @pytest.fixture
    def cpu_worker(self, model_path):
        """Create a CPU inference worker for testing."""
        worker = CPUInferenceWorker(
            model_path=model_path,
            device='cpu',
            batch_size=2,
            timeout_ms=10.0,
            use_mixed_precision=False
        )
        yield worker
        worker.stop_worker()

    def test_cpu_worker_initialization(self, model_path):
        """Test CPU worker initializes correctly."""
        worker = CPUInferenceWorker(
            model_path=model_path,
            batch_size=8,  # Should be limited to 4
        )

        assert worker.device == 'cpu'
        assert worker.batch_size == 4  # Limited for CPU
        assert worker.use_mixed_precision is False
        assert worker.model is not None
        assert not worker._warmup_completed

        worker.stop_worker()

    def test_cpu_worker_warmup(self, cpu_worker):
        """Test CPU worker warmup process."""
        input_shape = (36, 15, 15)

        # Initial state
        assert not cpu_worker._warmup_completed

        # Warmup
        cpu_worker.warmup(input_shape)

        # Verify warmup completed
        assert cpu_worker._warmup_completed
        assert cpu_worker.input_shape == input_shape

    def test_cpu_batch_inference(self, cpu_worker):
        """Test CPU batch inference functionality."""
        input_shape = (36, 15, 15)
        cpu_worker.warmup(input_shape)

        # Create test positions
        positions = [
            np.random.randn(*input_shape).astype(np.float32)
            for _ in range(3)
        ]

        # Run inference
        policies, values = cpu_worker.batch_inference(positions)

        # Verify outputs (Gomoku 15x15 = 225 actions)
        assert policies.shape == (3, 225)
        assert values.shape == (3,)
        assert np.all(np.isfinite(policies))
        assert np.all(np.isfinite(values))

    def test_cpu_inference_without_warmup(self, cpu_worker):
        """Test CPU inference fails without warmup."""
        positions = [np.random.randn(7, 15, 15).astype(np.float32)]

        with pytest.raises(RuntimeError, match="not warmed up"):
            cpu_worker.batch_inference(positions)

    def test_cpu_inference_loop(self, cpu_worker):
        """Test CPU inference loop with queue processing."""
        input_shape = (36, 15, 15)
        cpu_worker.warmup(input_shape)

        # Setup queues
        input_queue = Queue()
        output_queues = [Queue(), Queue()]

        # Start worker
        cpu_worker.start_worker(input_queue, output_queues)

        try:
            # Create and send test request
            from contracts.inference_api import InferenceRequest

            request = InferenceRequest(
                leaf_node_id=123,
                features=np.random.randn(*input_shape).astype(np.float32),
                thread_id=0,
                path=[1, 2, 3]
            )

            input_queue.put(request)

            # Get result
            result = output_queues[0].get(timeout=2.0)

            # Verify result (Gomoku 15x15 = 225 actions)
            assert result.node_id == 123
            assert result.policy.shape == (225,)
            assert isinstance(result.value, float)
            assert result.path == [1, 2, 3]
            assert result.processing_time_ms > 0

        finally:
            cpu_worker.stop_worker()

    def test_cpu_metrics_collection(self, cpu_worker):
        """Test CPU performance metrics collection."""
        input_shape = (36, 15, 15)
        cpu_worker.warmup(input_shape)

        # Run some inferences to generate metrics
        positions = [np.random.randn(*input_shape).astype(np.float32)]
        cpu_worker.batch_inference(positions)
        cpu_worker.batch_inference(positions)

        # Get metrics
        metrics = cpu_worker.get_metrics()

        # Verify metrics
        assert metrics['device'] == 'cpu'
        assert metrics['inference_type'] == 'cpu_fallback'
        assert metrics['total_inferences'] >= 2
        assert metrics['average_latency_ms'] > 0
        assert metrics['memory_usage_mb'] > 0
        assert metrics['fallback_active'] is True

    def test_cpu_error_handling(self, cpu_worker):
        """Test CPU worker error handling and safe fallback."""
        input_shape = (36, 15, 15)
        cpu_worker.warmup(input_shape)

        # Mock model to raise an error
        original_model = cpu_worker.model
        cpu_worker.model = Mock(side_effect=RuntimeError("Simulated error"))

        try:
            positions = [np.random.randn(*input_shape).astype(np.float32)]
            policies, values = cpu_worker.batch_inference(positions)

            # Should return safe fallback values (Gomoku 15x15 = 225 actions)
            assert policies.shape == (1, 225)
            assert values.shape == (1,)
            assert np.all(policies == 0)  # Safe fallback
            assert values[0] == 0.0

        finally:
            cpu_worker.model = original_model


class TestCPUFallbackInference:
    """Test CPU fallback inference API implementation."""

    @pytest.fixture
    def model_path(self):
        """Create a temporary model file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name

            model = create_model_for_game('gomoku')
            with torch.no_grad():
                dummy_input = torch.randn(1, 36, 15, 15)
                _ = model(dummy_input)
            torch.save(model, model_path)

        yield model_path
        os.unlink(model_path)

    def test_cpu_fallback_initialization(self, model_path):
        """Test CPU fallback inference initialization."""
        fallback = CPUFallbackInference(model_path)

        assert fallback.device == 'cpu'
        assert fallback.model is not None

    def test_cpu_fallback_single_inference(self, model_path):
        """Test single position inference via CPU fallback."""
        fallback = CPUFallbackInference(model_path)

        # Test single inference
        features = np.random.randn(7, 15, 15).astype(np.float32)
        policy, value = fallback.inference(features)

        # Verify output (Gomoku 15x15 = 225 actions)
        assert policy.shape == (225,)
        assert isinstance(value, float)
        assert np.all(np.isfinite(policy))
        assert np.isfinite(value)

    def test_cpu_fallback_error_handling(self, model_path):
        """Test CPU fallback error handling with safe defaults."""
        fallback = CPUFallbackInference(model_path)

        # Mock model to raise error
        fallback.model = Mock(side_effect=RuntimeError("Simulated error"))

        features = np.random.randn(7, 15, 15).astype(np.float32)
        policy, value = fallback.inference(features)

        # Should return safe defaults (Gomoku 15x15 = 225 actions)
        assert policy.shape == (225,)
        assert np.all(policy == 0)
        assert value == 0.0


class TestFallbackDetection:
    """Test GPU failure detection and fallback triggers."""

    def test_detect_gpu_failure_no_cuda(self):
        """Test GPU failure detection when CUDA unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            assert detect_gpu_failure() is True

    def test_detect_gpu_failure_cuda_error(self):
        """Test GPU failure detection on CUDA errors."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.zeros', side_effect=RuntimeError("CUDA error")):
            assert detect_gpu_failure() is True

    def test_detect_gpu_failure_success(self):
        """Test GPU failure detection when GPU works."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.zeros') as mock_zeros, \
             patch('torch.cuda.empty_cache'):

            mock_tensor = Mock()
            mock_zeros.return_value = mock_tensor

            result = detect_gpu_failure()
            assert result is False  # GPU should work

    def test_should_fallback_to_cpu_oom(self):
        """Test fallback detection for out of memory errors."""
        oom_error = RuntimeError("CUDA out of memory")
        assert should_fallback_to_cpu(oom_error) is True

    def test_should_fallback_to_cpu_cuda_errors(self):
        """Test fallback detection for various CUDA errors."""
        cuda_errors = [
            RuntimeError("CUDA error: device-side assert triggered"),
            RuntimeError("CUDA error: illegal memory access"),
            RuntimeError("cublas runtime error"),
            RuntimeError("cuDNN error"),
        ]

        for error in cuda_errors:
            assert should_fallback_to_cpu(error) is True

    def test_should_fallback_to_cpu_non_gpu_error(self):
        """Test that non-GPU errors don't trigger fallback."""
        non_gpu_errors = [
            ValueError("Invalid input shape"),
            RuntimeError("Model not loaded"),
            KeyError("Missing parameter"),
        ]

        for error in non_gpu_errors:
            assert should_fallback_to_cpu(error) is False


class TestGPUWorkerFallback:
    """Test GPU worker automatic fallback to CPU."""

    @pytest.fixture
    def model_path(self):
        """Create a temporary model file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name

            model = create_model_for_game('gomoku')
            with torch.no_grad():
                dummy_input = torch.randn(1, 36, 15, 15)
                _ = model(dummy_input)
            torch.save(model, model_path)

        yield model_path
        os.unlink(model_path)

    def test_gpu_worker_fallback_initialization(self, model_path):
        """Test GPU worker initializes CPU fallback correctly."""
        with patch('src.neural.cpu_inference.detect_gpu_failure', return_value=True):
            worker = GPUInferenceWorker(model_path=model_path, device='cuda:0')

            # Should have fallback enabled
            assert worker._fallback_enabled is True

            # Test that fallback can be manually enabled
            worker._enable_cpu_fallback()
            assert worker._cpu_fallback_worker is not None
            assert worker._fallback_triggered is True

            worker.stop_worker()

    def test_gpu_worker_fallback_on_inference_error(self, model_path):
        """Test GPU worker falls back to CPU on inference errors."""
        worker = GPUInferenceWorker(model_path=model_path, device='cpu')  # Use CPU to avoid real GPU
        worker.warmup((36, 15, 15))

        try:
            # Mock GPU inference to fail
            original_method = worker._run_inference_with_precision
            worker._run_inference_with_precision = Mock(
                side_effect=RuntimeError("CUDA out of memory")
            )

            # Mock CPU fallback worker
            mock_cpu_worker = Mock()
            mock_cpu_worker.batch_inference.return_value = (
                np.ones((2, 225)) / 225,  # Uniform policy (Gomoku 15x15 = 225 actions)
                np.zeros(2)  # Zero values
            )
            worker._cpu_fallback_worker = mock_cpu_worker

            # Run inference - should fallback to CPU
            positions = [
                np.random.randn(7, 15, 15).astype(np.float32),
                np.random.randn(7, 15, 15).astype(np.float32)
            ]

            policies, values = worker.batch_inference(positions)

            # Verify fallback was used
            assert worker._fallback_failure_count > 0
            mock_cpu_worker.batch_inference.assert_called_once_with(positions)
            assert policies.shape == (2, 225)  # Gomoku 15x15 = 225 actions
            assert values.shape == (2,)

        finally:
            worker.stop_worker()

    def test_gpu_worker_fallback_metrics(self, model_path):
        """Test GPU worker includes CPU fallback metrics."""
        worker = GPUInferenceWorker(model_path=model_path, device='cpu')

        try:
            # Enable fallback manually
            worker._fallback_triggered = True
            worker._fallback_failure_count = 3
            worker._cpu_fallback_worker = Mock()
            worker._cpu_fallback_worker.get_metrics.return_value = {
                'device': 'cpu',
                'total_inferences': 10,
                'average_latency_ms': 25.0
            }

            # Get metrics
            metrics = worker.get_metrics()

            # Verify fallback metrics included
            assert metrics['cpu_fallback_enabled'] is True
            assert metrics['cpu_fallback_active'] is True
            assert metrics['cpu_fallback_failure_count'] == 3
            assert metrics['cpu_fallback_available'] is True
            assert metrics['cpu_device'] == 'cpu'
            assert metrics['cpu_total_inferences'] == 10
            assert metrics['cpu_average_latency_ms'] == 25.0

        finally:
            worker.stop_worker()

    def test_gpu_worker_retry_logic(self, model_path):
        """Test GPU worker retry logic after fallback."""
        worker = GPUInferenceWorker(model_path=model_path, device='cpu')

        try:
            # Simulate fallback triggered
            worker._fallback_triggered = True
            worker._last_gpu_attempt = time.time() - 31.0  # 31 seconds ago

            # Should allow retry after 30 seconds
            assert worker._should_attempt_gpu_retry() is True

            # Recent attempt should not allow retry
            worker._last_gpu_attempt = time.time() - 10.0  # 10 seconds ago
            assert worker._should_attempt_gpu_retry() is False

        finally:
            worker.stop_worker()

    def test_gpu_worker_cleanup_fallback(self, model_path):
        """Test GPU worker cleans up CPU fallback on stop."""
        worker = GPUInferenceWorker(model_path=model_path, device='cpu')

        # Mock CPU fallback worker
        mock_cpu_worker = Mock()
        worker._cpu_fallback_worker = mock_cpu_worker

        # Start and then stop worker to test cleanup
        worker._is_running = True  # Simulate running state

        # Stop worker
        worker.stop_worker()

        # Verify cleanup
        mock_cpu_worker.stop_worker.assert_called_once()
        assert worker._cpu_fallback_worker is None

    def test_gpu_worker_safe_fallback_on_double_failure(self, model_path):
        """Test safe fallback when both GPU and CPU fail."""
        worker = GPUInferenceWorker(model_path=model_path, device='cpu')
        worker.warmup((36, 15, 15))

        try:
            # Mock both GPU and CPU to fail
            worker._run_inference_with_precision = Mock(
                side_effect=RuntimeError("CUDA out of memory")
            )

            mock_cpu_worker = Mock()
            mock_cpu_worker.batch_inference.side_effect = RuntimeError("CPU also failed")
            worker._cpu_fallback_worker = mock_cpu_worker

            # Run inference - should return safe defaults
            positions = [np.random.randn(7, 15, 15).astype(np.float32)]
            policies, values = worker.batch_inference(positions)

            # Verify safe defaults returned (Gomoku 15x15 = 225 actions)
            assert policies.shape == (1, 225)
            assert values.shape == (1,)
            assert np.allclose(policies, 1.0 / 225)  # Uniform distribution
            assert values[0] == 0.0

        finally:
            worker.stop_worker()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])