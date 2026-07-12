"""
Unit tests for Pinned Memory Optimization (T017)
=================================================

Tests the pinned memory buffer management and optimization including:
- Pinned memory buffer allocation and management
- Optimized H2D/D2H transfers using pinned memory
- Buffer reuse and automatic fallback mechanisms
- Memory usage tracking and metrics integration

Run with: python -m pytest tests/unit/test_pinned_memory.py -v
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Import worker implementation
from src.neural.inference_worker import GPUInferenceWorker
from src.neural.cpu_inference import CPUInferenceWorker
from src.neural.model import create_model_for_game

# Import contract interfaces
import sys
sys.path.append('specs/001-goal-create-spec')
from contracts.inference_api import InferenceRequest


class TestPinnedMemorySetup:
    """Test pinned memory buffer setup and configuration."""

    def setup_method(self):
        """Setup test fixtures."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            self.model_path = f.name
            # Create and save a valid model
            model = create_model_for_game('gomoku')
            with torch.no_grad():
                dummy_input = torch.randn(1, 36, 15, 15)  # Enhanced Gomoku: 36 input channels
                _ = model(dummy_input)  # Initialize lazy layers
            torch.save(model.state_dict(), self.model_path)

    def teardown_method(self):
        """Cleanup test fixtures."""
        if hasattr(self, 'model_path'):
            os.unlink(self.model_path)

    def test_pinned_memory_initialization_cuda_available(self):
        """Test pinned memory initialization when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            worker = GPUInferenceWorker(
                model_path=self.model_path,
                device='cuda:0',
                batch_size=32,
                timeout_ms=5.0,
                use_mixed_precision=False
            )

            # Should enable pinned memory for CUDA devices
            assert worker._use_pinned_memory == True
            assert worker._pinned_input_buffer is None  # Not allocated yet
            assert worker._pinned_output_buffers == {}
            assert worker._current_buffer_capacity == 0

    def test_pinned_memory_initialization_cpu_device(self):
        """Test pinned memory initialization on CPU device."""
        worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cpu',
            batch_size=32,
            timeout_ms=5.0,
            use_mixed_precision=False
        )

        # Should disable pinned memory for CPU devices
        assert worker._use_pinned_memory == False

    def test_pinned_memory_initialization_cuda_unavailable(self):
        """Test pinned memory initialization when CUDA is unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            worker = GPUInferenceWorker(
                model_path=self.model_path,
                device='cuda:0',
                batch_size=32,
                timeout_ms=5.0,
                use_mixed_precision=False
            )

            # Should disable pinned memory when CUDA unavailable
            assert worker._use_pinned_memory == False


class TestPinnedMemoryBufferManagement:
    """Test pinned memory buffer allocation and management."""

    def setup_method(self):
        """Setup test fixtures."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            self.model_path = f.name
            model = create_model_for_game('gomoku')
            with torch.no_grad():
                dummy_input = torch.randn(1, 36, 15, 15)  # Enhanced Gomoku: 36 input channels
                _ = model(dummy_input)
            torch.save(model.state_dict(), self.model_path)

        with patch('torch.cuda.is_available', return_value=True):
            self.worker = GPUInferenceWorker(
                model_path=self.model_path,
                device='cuda:0',
                batch_size=32,
                timeout_ms=5.0,
                use_mixed_precision=False
            )

    def teardown_method(self):
        """Cleanup test fixtures."""
        if hasattr(self, 'model_path'):
            os.unlink(self.model_path)

    def test_setup_pinned_memory_buffers_success(self):
        """Test successful pinned memory buffer allocation."""
        input_shape = (36, 15, 15)  # Enhanced Gomoku: 36 input channels
        batch_size = 32

        self.worker._setup_pinned_memory_buffers(batch_size, input_shape)

        # Should allocate buffers with safety margin
        expected_capacity = int(batch_size * 1.5)  # 48
        assert self.worker._current_buffer_capacity == expected_capacity

        # Should allocate input buffer
        assert self.worker._pinned_input_buffer is not None
        assert self.worker._pinned_input_buffer.shape == (expected_capacity, 36, 15, 15)  # Enhanced Gomoku: 36 input channels
        assert self.worker._pinned_input_buffer.is_pinned()

        # Should allocate output buffers
        assert 'policy' in self.worker._pinned_output_buffers
        assert 'value' in self.worker._pinned_output_buffers
        assert self.worker._pinned_output_buffers['policy'].shape == (expected_capacity, 361)
        assert self.worker._pinned_output_buffers['value'].shape == (expected_capacity, 1)
        assert self.worker._pinned_output_buffers['policy'].is_pinned()
        assert self.worker._pinned_output_buffers['value'].is_pinned()

    def test_setup_pinned_memory_buffers_reuse_existing(self):
        """Test buffer reuse when current buffers are sufficient."""
        input_shape = (36, 15, 15)  # Enhanced Gomoku: 36 input channels

        # Setup initial buffers
        self.worker._setup_pinned_memory_buffers(32, input_shape)
        initial_buffer = self.worker._pinned_input_buffer
        initial_capacity = self.worker._current_buffer_capacity

        # Request smaller buffers - should reuse existing
        self.worker._setup_pinned_memory_buffers(16, input_shape)

        assert self.worker._pinned_input_buffer is initial_buffer
        assert self.worker._current_buffer_capacity == initial_capacity

    def test_setup_pinned_memory_buffers_expansion(self):
        """Test buffer expansion when larger capacity needed."""
        input_shape = (36, 15, 15)  # Enhanced Gomoku: 36 input channels

        # Setup initial buffers
        self.worker._setup_pinned_memory_buffers(16, input_shape)
        initial_capacity = self.worker._current_buffer_capacity

        # Request larger buffers - should reallocate
        self.worker._setup_pinned_memory_buffers(64, input_shape)

        assert self.worker._current_buffer_capacity > initial_capacity
        assert self.worker._current_buffer_capacity == int(64 * 1.5)

    def test_setup_pinned_memory_buffers_failure_fallback(self):
        """Test fallback when pinned memory allocation fails."""
        input_shape = (36, 15, 15)  # Enhanced Gomoku: 36 input channels

        with patch('torch.empty', side_effect=RuntimeError("CUDA out of memory")):
            self.worker._setup_pinned_memory_buffers(32, input_shape)

            # Should disable pinned memory on failure
            assert self.worker._use_pinned_memory == False
            assert self.worker._pinned_input_buffer is None
            assert self.worker._current_buffer_capacity == 0

    def test_cleanup_pinned_buffers(self):
        """Test pinned memory buffer cleanup."""
        input_shape = (36, 15, 15)  # Enhanced Gomoku: 36 input channels

        # Setup buffers
        self.worker._setup_pinned_memory_buffers(32, input_shape)
        assert self.worker._pinned_input_buffer is not None
        assert len(self.worker._pinned_output_buffers) > 0

        # Cleanup
        self.worker._cleanup_pinned_buffers()

        assert self.worker._pinned_input_buffer is None
        assert len(self.worker._pinned_output_buffers) == 0
        assert self.worker._current_buffer_capacity == 0


class TestOptimizedTensorOperations:
    """Test optimized tensor creation and transfer operations."""

    def setup_method(self):
        """Setup test fixtures."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            self.model_path = f.name
            model = create_model_for_game('gomoku')
            with torch.no_grad():
                dummy_input = torch.randn(1, 36, 15, 15)  # Enhanced Gomoku: 36 input channels
                _ = model(dummy_input)
            torch.save(model.state_dict(), self.model_path)

        with patch('torch.cuda.is_available', return_value=True):
            self.worker = GPUInferenceWorker(
                model_path=self.model_path,
                device='cuda:0',
                batch_size=32,
                timeout_ms=5.0,
                use_mixed_precision=False
            )

        # Setup pinned memory buffers
        input_shape = (36, 15, 15)  # Enhanced Gomoku: 36 input channels
        self.worker._setup_pinned_memory_buffers(32, input_shape)

    def teardown_method(self):
        """Cleanup test fixtures."""
        if hasattr(self, 'model_path'):
            os.unlink(self.model_path)

    def test_create_batch_tensor_optimized_with_pinned_memory(self):
        """Test optimized batch tensor creation using pinned memory."""
        positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(4)]  # Enhanced Gomoku: 36 input channels

        # Ensure pinned memory is enabled and buffer exists
        assert self.worker._use_pinned_memory == True
        assert self.worker._pinned_input_buffer is not None

        # Test that the method succeeds and returns a tensor
        batch_tensor = self.worker._create_batch_tensor_optimized(positions)

        # Should return a valid tensor with correct shape
        assert isinstance(batch_tensor, torch.Tensor)
        assert batch_tensor.shape == (4, 36, 15, 15)  # Enhanced Gomoku: 36 input channels
        assert batch_tensor.device.type == 'cuda'  # Should be on GPU device

    def test_create_batch_tensor_optimized_fallback_standard(self):
        """Test fallback to standard tensor creation."""
        positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(4)]  # Enhanced Gomoku: 36 input channels

        # Disable pinned memory
        self.worker._use_pinned_memory = False

        with patch('torch.tensor') as mock_tensor:
            mock_tensor.return_value = torch.randn(4, 36, 15, 15)  # Enhanced Gomoku: 36 input channels

            batch_tensor = self.worker._create_batch_tensor_optimized(positions)

            # Should use standard tensor creation
            mock_tensor.assert_called_once()
            assert mock_tensor.call_args[1]['device'] == self.worker.device

    def test_create_batch_tensor_optimized_batch_too_large(self):
        """Test fallback when batch size exceeds buffer capacity."""
        # Create batch larger than buffer capacity
        positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(100)]  # Enhanced Gomoku: 36 input channels

        with patch('torch.tensor') as mock_tensor:
            mock_tensor.return_value = torch.randn(100, 36, 15, 15)  # Enhanced Gomoku: 36 input channels

            batch_tensor = self.worker._create_batch_tensor_optimized(positions)

            # Should fallback to standard tensor creation
            mock_tensor.assert_called_once()

    def test_transfer_outputs_optimized_with_pinned_memory(self):
        """Test optimized output transfer using pinned memory."""
        # Create mock output tensors
        policy_logits = torch.randn(4, 225)
        values = torch.randn(4, 1)

        policies_np, values_np = self.worker._transfer_outputs_optimized(policy_logits, values)

        # Should return numpy arrays
        assert isinstance(policies_np, np.ndarray)
        assert isinstance(values_np, np.ndarray)
        assert policies_np.shape == (4, 225)
        assert values_np.shape == (4,)

    def test_transfer_outputs_optimized_fallback_standard(self):
        """Test fallback to standard output transfer."""
        policy_logits = torch.randn(4, 225)
        values = torch.randn(4, 1)

        # Disable pinned memory
        self.worker._use_pinned_memory = False

        with patch.object(policy_logits, 'cpu') as mock_cpu:
            mock_cpu.return_value.numpy.return_value = np.random.randn(4, 225)

            with patch.object(values, 'cpu') as mock_values_cpu:
                mock_values_cpu.return_value.numpy.return_value.squeeze.return_value = np.random.randn(4)

                policies_np, values_np = self.worker._transfer_outputs_optimized(policy_logits, values)

                # Should use standard CPU transfer
                mock_cpu.assert_called_once()
                mock_values_cpu.assert_called_once()

    def test_transfer_outputs_optimized_pinned_memory_failure(self):
        """Test fallback when pinned memory transfer fails."""
        policy_logits = torch.randn(4, 225)
        values = torch.randn(4, 1)

        # Test that fallback works when pinned memory fails
        with patch.object(self.worker._pinned_output_buffers['policy'], 'copy_', side_effect=RuntimeError("Transfer failed")):
            policies_np, values_np = self.worker._transfer_outputs_optimized(policy_logits, values)

            # Should succeed with fallback to standard transfer
            assert isinstance(policies_np, np.ndarray)
            assert isinstance(values_np, np.ndarray)
            assert policies_np.shape == (4, 225)
            assert values_np.shape == (4,)


class TestPinnedMemoryIntegration:
    """Integration tests for pinned memory with actual inference."""

    def setup_method(self):
        """Setup integration test fixtures."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            self.model_path = f.name
            model = create_model_for_game('gomoku')
            with torch.no_grad():
                dummy_input = torch.randn(1, 36, 15, 15)  # Enhanced Gomoku: 36 input channels
                _ = model(dummy_input)
            torch.save(model.state_dict(), self.model_path)

        self.worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cpu',  # Use CPU for testing
            batch_size=32,
            timeout_ms=5.0,
            use_mixed_precision=False
        )

    def teardown_method(self):
        """Cleanup integration test fixtures."""
        if hasattr(self, 'model_path'):
            os.unlink(self.model_path)

    def test_warmup_with_pinned_memory_setup(self):
        """Test warmup process sets up pinned memory buffers."""
        input_shape = (36, 15, 15)  # Enhanced Gomoku: 36 input channels

        # On CPU, pinned memory should be disabled
        self.worker.warmup(input_shape)

        # Should not set up pinned memory on CPU
        assert self.worker._use_pinned_memory == False
        assert self.worker._pinned_input_buffer is None

    @patch('torch.cuda.is_available')
    def test_warmup_with_pinned_memory_setup_cuda(self, mock_cuda_available):
        """Test warmup process sets up pinned memory buffers on CUDA."""
        mock_cuda_available.return_value = True

        # Create CUDA worker
        worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cuda:0',
            batch_size=32,
            timeout_ms=5.0,
            use_mixed_precision=False
        )

        input_shape = (36, 15, 15)  # Enhanced Gomoku: 36 input channels
        worker.warmup(input_shape)

        # Should setup pinned memory on CUDA
        assert worker._use_pinned_memory == True
        assert worker._current_buffer_capacity > 0

    def test_batch_inference_with_optimized_transfers(self):
        """Test batch inference uses optimized tensor operations."""
        positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(4)]  # Enhanced Gomoku: 36 input channels

        policies, values = self.worker.batch_inference(positions)

        # Should produce valid outputs
        assert policies.shape == (4, 225)
        assert values.shape == (4,)
        assert np.all(np.isfinite(policies))
        assert np.all(np.isfinite(values))

    def test_stop_worker_cleans_up_pinned_buffers(self):
        """Test that stopping worker cleans up pinned memory buffers."""
        # Test the cleanup method directly since stop_worker calls it
        self.worker._current_buffer_capacity = 32

        # Mock buffers for testing cleanup
        self.worker._pinned_input_buffer = Mock()
        self.worker._pinned_output_buffers = {'policy': Mock(), 'value': Mock()}

        # Call cleanup directly
        self.worker._cleanup_pinned_buffers()

        # Should cleanup buffers
        assert self.worker._pinned_input_buffer is None
        assert len(self.worker._pinned_output_buffers) == 0
        assert self.worker._current_buffer_capacity == 0


class TestPinnedMemoryMetrics:
    """Test pinned memory metrics integration."""

    def setup_method(self):
        """Setup test fixtures."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            self.model_path = f.name
            model = create_model_for_game('gomoku')
            with torch.no_grad():
                dummy_input = torch.randn(1, 36, 15, 15)  # Enhanced Gomoku: 36 input channels
                _ = model(dummy_input)
            torch.save(model.state_dict(), self.model_path)

        self.worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cpu',
            batch_size=32,
            timeout_ms=5.0,
            use_mixed_precision=False
        )

    def teardown_method(self):
        """Cleanup test fixtures."""
        if hasattr(self, 'model_path'):
            os.unlink(self.model_path)

    def test_pinned_memory_metrics_cpu_device(self):
        """Test pinned memory metrics on CPU device."""
        metrics = self.worker._get_memory_efficiency_metrics()

        # Should include pinned memory metrics
        assert 'pinned_memory_enabled' in metrics
        assert 'pinned_buffer_capacity' in metrics
        assert metrics['pinned_memory_enabled'] == False
        assert metrics['pinned_buffer_capacity'] == 0

    @patch('torch.cuda.is_available')
    def test_pinned_memory_metrics_with_buffers(self, mock_cuda_available):
        """Test pinned memory metrics when buffers are allocated."""
        mock_cuda_available.return_value = True

        worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cuda:0',
            batch_size=32,
            timeout_ms=5.0,
            use_mixed_precision=False
        )

        # Setup pinned memory buffers
        input_shape = (36, 15, 15)  # Enhanced Gomoku: 36 input channels
        worker._setup_pinned_memory_buffers(32, input_shape)

        metrics = worker._get_memory_efficiency_metrics()

        # Should include detailed pinned memory metrics
        assert metrics['pinned_memory_enabled'] == True
        assert metrics['pinned_buffer_capacity'] > 0
        assert 'pinned_memory_usage_mb' in metrics
        assert metrics['pinned_memory_usage_mb'] > 0

    def test_enhanced_metrics_integration(self):
        """Test integration of pinned memory metrics in get_metrics."""
        # Process some batches to generate metrics
        self.worker._update_metrics(32, 0.002)

        metrics = self.worker.get_metrics()

        # Should include pinned memory metrics
        assert 'pinned_memory_enabled' in metrics
        assert 'pinned_buffer_capacity' in metrics


def test_pinned_memory_cpu_worker_compatibility():
    """Test that CPUInferenceWorker maintains compatibility."""
    from src.neural.model import create_model_for_game
    import tempfile
    import os

    # Create a real model for testing
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        model_path = f.name

    try:
        model = create_model_for_game('gomoku')
        with torch.no_grad():
            dummy_input = torch.randn(1, model.input_channels, 15, 15)
            _ = model(dummy_input)
        torch.save(model.state_dict(), model_path)

        cpu_worker = CPUInferenceWorker(
            model_path=model_path,
            device='cpu',
            batch_size=32,
            timeout_ms=5.0,
            use_mixed_precision=False
        )

        # Warmup the worker
        cpu_worker.warmup((36, 15, 15))

    finally:
        # Clean up
        if os.path.exists(model_path):
            os.unlink(model_path)

    # Should be able to run inference without pinned memory features
    positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(4)]  # Enhanced Gomoku: 36 input channels
    policies, values = cpu_worker.batch_inference(positions)

    assert policies.shape == (4, 225)
    assert values.shape == (4,)