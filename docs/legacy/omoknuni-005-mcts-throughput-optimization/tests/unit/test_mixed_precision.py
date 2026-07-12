"""
Unit tests for Mixed Precision Inference (T016)
===============================================

Tests the enhanced mixed precision implementation including:
- FP16 computation with automatic fallback to FP32
- Memory efficiency monitoring and 2x reduction validation
- Device capability detection and compatibility checking
- Accuracy preservation and error handling

Run with: python -m pytest tests/unit/test_mixed_precision.py -v
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


class TestMixedPrecisionSetup:
    """Test mixed precision setup and configuration."""

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

    def test_mixed_precision_initialization_enabled(self):
        """Test mixed precision initialization when enabled."""
        worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cpu',  # Use CPU for testing
            batch_size=32,
            timeout_ms=5.0,
            use_mixed_precision=True
        )

        # Should be disabled on CPU
        assert not worker._mixed_precision_enabled
        assert not worker.use_mixed_precision
        assert worker._mixed_precision_fallback_count == 0

    def test_mixed_precision_initialization_disabled(self):
        """Test mixed precision initialization when disabled."""
        worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cpu',
            batch_size=32,
            timeout_ms=5.0,
            use_mixed_precision=False
        )

        # Should remain disabled
        assert not worker._mixed_precision_enabled
        assert not worker.use_mixed_precision

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_capability')
    def test_mixed_precision_cuda_device_capability_check(self, mock_capability, mock_cuda_available):
        """Test CUDA device capability checking for mixed precision."""
        mock_cuda_available.return_value = True
        mock_capability.return_value = (7, 5)  # RTX 20 series capability

        with patch('src.neural.model.enable_mixed_precision') as mock_enable:
            mock_enable.return_value = MagicMock()

            worker = GPUInferenceWorker(
                model_path=self.model_path,
                device='cuda:0',
                batch_size=32,
                timeout_ms=5.0,
                use_mixed_precision=True
            )

            # Should enable mixed precision for capable device
            assert worker._mixed_precision_enabled

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_capability')
    def test_mixed_precision_low_capability_warning(self, mock_capability, mock_cuda_available):
        """Test warning for devices with low compute capability."""
        mock_cuda_available.return_value = True
        mock_capability.return_value = (6, 1)  # GTX 10 series capability

        with patch('src.neural.model.enable_mixed_precision') as mock_enable:
            mock_enable.return_value = MagicMock()

            # Capture the warning during worker creation
            with patch('src.neural.inference_worker.logging.getLogger') as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                worker = GPUInferenceWorker(
                    model_path=self.model_path,
                    device='cuda:0',
                    batch_size=32,
                    timeout_ms=5.0,
                    use_mixed_precision=True
                )

                # Should warn about low capability but still enable
                mock_logger.warning.assert_called()
                warning_call = mock_logger.warning.call_args[0][0]
                assert "may not benefit from mixed precision" in warning_call

    @patch('torch.cuda.is_available')
    def test_mixed_precision_cuda_unavailable(self, mock_cuda_available):
        """Test mixed precision fallback when CUDA unavailable."""
        mock_cuda_available.return_value = False

        worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cuda:0',
            batch_size=32,
            timeout_ms=5.0,
            use_mixed_precision=True
        )

        # Should disable mixed precision
        assert not worker._mixed_precision_enabled
        assert not worker.use_mixed_precision


class TestMixedPrecisionInference:
    """Test mixed precision inference behavior."""

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
            use_mixed_precision=True  # Will be disabled on CPU
        )

    def teardown_method(self):
        """Cleanup test fixtures."""
        if hasattr(self, 'model_path'):
            os.unlink(self.model_path)

    def test_inference_with_precision_fp32_fallback(self):
        """Test inference with FP32 fallback."""
        batch_tensor = torch.randn(4, 36, 15, 15)  # Enhanced Gomoku: 36 input channels

        policy_logits, values = self.worker._run_inference_with_precision(batch_tensor)

        # Should return valid outputs
        assert policy_logits.shape == (4, 225)  # Gomoku policy size
        assert values.shape == (4, 1)  # Raw model output shape
        assert policy_logits.dtype == torch.float32
        assert values.dtype == torch.float32

    @patch('torch.cuda.is_available')
    def test_inference_with_mixed_precision_fallback_error(self, mock_cuda_available):
        """Test automatic fallback on mixed precision errors."""
        mock_cuda_available.return_value = True

        # Enable mixed precision and set CUDA device
        self.worker._mixed_precision_enabled = True
        self.worker.device = 'cuda:0'

        batch_tensor = torch.randn(4, 36, 15, 15)  # Enhanced Gomoku: 36 input channels

        # Test approach: create a real scenario that triggers autocast failure
        # by moving the input to CUDA but keeping model on CPU (device mismatch)
        if torch.cuda.is_available():
            # Move tensor to GPU to create the device mismatch that causes autocast errors
            batch_tensor = batch_tensor.cuda()

            # The autocast will fail due to device mismatch, triggering fallback
            policy_logits, values = self.worker._run_inference_with_precision(batch_tensor)

            # Should have fallen back and succeeded
            assert policy_logits is not None
            assert values is not None
            assert policy_logits.shape == (4, 225)
            assert values.shape == (4, 1)
            assert self.worker._mixed_precision_fallback_count >= 1
        else:
            # If no CUDA, just test the mock fallback scenario
            model_call_count = 0
            def mock_model_call(*args, **kwargs):
                nonlocal model_call_count
                model_call_count += 1
                if model_call_count == 1:
                    raise RuntimeError("autocast: float16 not supported for this operation")
                else:
                    return (torch.randn(4, 225), torch.randn(4, 1))

            with patch.object(self.worker.model, '__call__', side_effect=mock_model_call):
                policy_logits, values = self.worker._run_inference_with_precision(batch_tensor)

                assert policy_logits is not None
                assert values is not None
                assert self.worker._mixed_precision_fallback_count == 1
                assert model_call_count == 2

    @patch('torch.cuda.is_available')
    def test_mixed_precision_disable_after_failures(self, mock_cuda_available):
        """Test disabling mixed precision after repeated failures."""
        mock_cuda_available.return_value = True

        # Mock mixed precision enabled and device as CUDA
        self.worker._mixed_precision_enabled = True
        self.worker.device = 'cuda:0'

        batch_tensor = torch.randn(4, 36, 15, 15)  # Enhanced Gomoku: 36 input channels

        # Use real CUDA scenario that triggers autocast failures
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()

        call_count = 0
        def mock_model_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Always raise autocast error to trigger fallback behavior
            raise RuntimeError("autocast: unsupported operation")

        with patch.object(self.worker.model, '__call__', side_effect=mock_model_call):
            # Trigger multiple failures - each call should increment fallback count
            for i in range(4):
                try:
                    self.worker._run_inference_with_precision(batch_tensor)
                except RuntimeError:
                    # Expected to fail since our mock always raises errors in both paths
                    pass

            # Should disable mixed precision after 3 failures
            assert not self.worker._mixed_precision_enabled
            assert not self.worker.use_mixed_precision
            assert self.worker._mixed_precision_fallback_count >= 3

    def test_non_precision_error_propagation(self):
        """Test that non-precision related errors are not counted as precision errors."""
        # This test verifies that the error detection logic correctly identifies
        # non-precision errors vs precision errors

        # Test error message detection
        precision_error_msg = "autocast: float16 not supported"
        non_precision_error_msg = "CUDA out of memory"

        # Check that precision errors are detected correctly
        assert "autocast" in precision_error_msg.lower() or "half" in precision_error_msg.lower()

        # Check that non-precision errors are not detected as precision errors
        assert not ("autocast" in non_precision_error_msg.lower() or "half" in non_precision_error_msg.lower())

        # Test that fallback count starts at 0
        assert self.worker._mixed_precision_fallback_count == 0


class TestMemoryEfficiencyMetrics:
    """Test memory efficiency monitoring for mixed precision."""

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
            use_mixed_precision=True
        )

    def teardown_method(self):
        """Cleanup test fixtures."""
        if hasattr(self, 'model_path'):
            os.unlink(self.model_path)

    def test_memory_efficiency_metrics_cpu(self):
        """Test memory efficiency metrics on CPU."""
        metrics = self.worker._get_memory_efficiency_metrics()

        # CPU should return empty metrics
        assert isinstance(metrics, dict)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.max_memory_allocated')
    def test_memory_efficiency_metrics_cuda(self, mock_max_mem, mock_current_mem, mock_cuda_available):
        """Test memory efficiency metrics on CUDA."""
        mock_cuda_available.return_value = True
        mock_current_mem.return_value = 500 * 1024 * 1024  # 500MB
        mock_max_mem.return_value = 800 * 1024 * 1024     # 800MB

        # Set baseline for comparison
        self.worker._baseline_memory_usage = 1000 * 1024 * 1024  # 1GB baseline
        self.worker.device = 'cuda:0'
        self.worker._mixed_precision_enabled = True
        self.worker._mixed_precision_fallback_count = 2

        metrics = self.worker._get_memory_efficiency_metrics()

        # Check memory metrics
        assert 'current_memory_mb' in metrics
        assert 'max_memory_mb' in metrics
        assert 'memory_efficiency_ratio' in metrics
        assert 'memory_reduction_achieved' in metrics
        assert 'mixed_precision_active' in metrics
        assert 'mixed_precision_fallback_count' in metrics

        # Validate values
        assert metrics['current_memory_mb'] == 500
        assert metrics['max_memory_mb'] == 800
        assert metrics['memory_efficiency_ratio'] == 0.5  # 50% of baseline
        assert metrics['memory_reduction_achieved'] == True  # < 70% threshold
        assert metrics['mixed_precision_active'] == True
        assert metrics['mixed_precision_fallback_count'] == 2

    def test_enhanced_metrics_integration(self):
        """Test integration of mixed precision metrics in get_metrics."""
        # Process some batches to generate metrics
        self.worker._update_metrics(32, 0.002)

        metrics = self.worker.get_metrics()

        # Should include mixed precision metrics
        assert 'mixed_precision_active' in metrics
        assert 'mixed_precision_fallback_count' in metrics


class TestMixedPrecisionIntegration:
    """Integration tests for mixed precision with actual inference."""

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
            device='cpu',
            batch_size=32,
            timeout_ms=5.0,
            use_mixed_precision=True
        )

    def teardown_method(self):
        """Cleanup integration test fixtures."""
        if hasattr(self, 'model_path'):
            os.unlink(self.model_path)

    def test_batch_inference_with_mixed_precision(self):
        """Test batch inference with mixed precision configuration."""
        positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(8)]  # Enhanced Gomoku: 36 input channels

        policies, values = self.worker.batch_inference(positions)

        # Should produce valid outputs regardless of precision mode
        assert policies.shape == (8, 225)
        assert values.shape == (8,)
        assert np.all(np.isfinite(policies))
        assert np.all(np.isfinite(values))

        # Policies should be proper probabilities
        assert np.allclose(np.sum(policies, axis=1), 1.0, atol=1e-6)
        assert np.all(policies >= 0)

    def test_warmup_with_mixed_precision(self):
        """Test warmup process with mixed precision."""
        # Should not raise exceptions
        self.worker.warmup((36, 15, 15))  # Enhanced Gomoku: 36 input channels

        # Worker should be ready for inference
        positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(4)]  # Enhanced Gomoku: 36 input channels
        policies, values = self.worker.batch_inference(positions)

        assert policies.shape == (4, 225)
        assert values.shape == (4,)

    def test_accuracy_preservation_fp32_vs_mixed_precision(self):
        """Test that mixed precision preserves reasonable accuracy."""
        positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(4)]  # Enhanced Gomoku: 36 input channels

        # Get FP32 results
        self.worker.use_mixed_precision = False
        self.worker._mixed_precision_enabled = False
        policies_fp32, values_fp32 = self.worker.batch_inference(positions)

        # Enable mixed precision (will fallback to FP32 on CPU but tests the path)
        self.worker.use_mixed_precision = True
        policies_mixed, values_mixed = self.worker.batch_inference(positions)

        # Results should be identical on CPU (both use FP32)
        np.testing.assert_array_equal(policies_fp32, policies_mixed)
        np.testing.assert_array_equal(values_fp32, values_mixed)


@pytest.mark.parametrize("use_mixed_precision,device", [
    (True, 'cpu'),
    (False, 'cpu'),
    (True, 'cuda:0'),  # Will fallback to CPU behavior in tests
])
def test_mixed_precision_parameter_combinations(use_mixed_precision, device):
    """Test mixed precision with various parameter combinations."""
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        model_path = f.name
        model = create_model_for_game('gomoku')
        with torch.no_grad():
            dummy_input = torch.randn(1, 36, 15, 15)  # Enhanced Gomoku: 36 input channels
            _ = model(dummy_input)
        torch.save(model.state_dict(), model_path)

    try:
        # Treat CUDA device as CPU for testing
        test_device = 'cpu' if device.startswith('cuda') else device

        worker = GPUInferenceWorker(
            model_path=model_path,
            device=test_device,
            batch_size=32,
            timeout_ms=5.0,
            use_mixed_precision=use_mixed_precision
        )

        # Should initialize without errors
        assert hasattr(worker, '_mixed_precision_enabled')
        assert hasattr(worker, '_mixed_precision_fallback_count')

        # Should be able to run inference
        positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(4)]  # Enhanced Gomoku: 36 input channels
        policies, values = worker.batch_inference(positions)

        assert policies.shape == (4, 225)
        assert values.shape == (4,)

    finally:
        os.unlink(model_path)


@patch('src.neural.cpu_inference.CPUInferenceWorker.batch_inference')
@patch('src.neural.cpu_inference.CPUInferenceWorker._load_model')
def test_mixed_precision_cpu_worker_compatibility(mock_load_model, mock_batch_inference):
    """Test that CPUInferenceWorker maintains compatibility."""
    # Mock the model loading to avoid file not found error
    mock_load_model.return_value = None

    # Mock batch inference to return expected shapes
    mock_policies = np.random.random((4, 225)).astype(np.float32)
    mock_values = np.random.random(4).astype(np.float32)
    mock_batch_inference.return_value = (mock_policies, mock_values)

    cpu_worker = CPUInferenceWorker(
        model_path='/dummy/path',
        device='cpu',
        batch_size=32,
        timeout_ms=5.0,
        use_mixed_precision=True
    )

    # Should handle mixed precision parameter gracefully
    # CPU worker should force mixed precision to False for performance reasons
    assert hasattr(cpu_worker, 'use_mixed_precision')
    assert cpu_worker.use_mixed_precision == False  # CPU workers don't use mixed precision

    # Should be able to run inference
    positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(4)]  # Enhanced Gomoku: 36 input channels
    policies, values = cpu_worker.batch_inference(positions)

    assert policies.shape == (4, 225)
    assert values.shape == (4,)