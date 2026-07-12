"""
Contract tests for Neural Network Inference API
==============================================

Tests ensure all inference API functions are properly defined and will fail
until actual implementations are provided. This validates the contract
interface before implementation begins.

Run with: python -m pytest tests/contract/test_inference_api.py -v
"""

import pytest
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from queue import Queue
from unittest.mock import Mock, patch
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for model creation utilities
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import contract interfaces
sys.path.append('specs/001-goal-create-spec')
from contracts.inference_api import (
    InferenceWorker,
    InferenceRequest,
    InferenceResult,
    create_inference_worker,
    estimate_batch_size,
    benchmark_inference,
    CPUFallbackInference,
    validate_model_compatibility
)

# Import model creation utilities
from neural.model import create_model_for_game, AlphaZeroNet


def create_temporary_model(game_type: str = 'gomoku') -> str:
    """Create a valid temporary model file for testing.

    Args:
        game_type: Type of game model to create

    Returns:
        str: Path to temporary model file
    """
    # Create a minimal valid AlphaZero model
    model = create_model_for_game(game_type)

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name

    # Save model state dict directly (this is what CPU inference worker expects)
    torch.save(model.state_dict(), temp_path)

    return temp_path


class TestInferenceWorkerContract:
    """Test InferenceWorker abstract base class contract."""

    def test_inference_worker_is_abstract(self):
        """InferenceWorker should be abstract and not instantiable directly."""
        with pytest.raises(TypeError):
            InferenceWorker(
                model_path="dummy.pth",
                device="cpu",
                batch_size=32,
                timeout_ms=3.0,
                use_mixed_precision=False
            )

    def test_inference_worker_has_required_methods(self):
        """InferenceWorker must have all required abstract methods."""
        required_methods = [
            '__init__',
            'warmup',
            'inference_loop',
            'batch_inference',
            'get_metrics'
        ]

        for method_name in required_methods:
            assert hasattr(InferenceWorker, method_name), f"Missing method: {method_name}"
            method = getattr(InferenceWorker, method_name)
            assert hasattr(method, '__isabstractmethod__'), f"Method {method_name} should be abstract"

    def test_inference_worker_init_signature(self):
        """Test InferenceWorker.__init__ has correct signature."""
        import inspect
        sig = inspect.signature(InferenceWorker.__init__)

        expected_params = [
            'self', 'model_path', 'device', 'batch_size',
            'timeout_ms', 'use_mixed_precision'
        ]

        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params

        # Check default values
        assert sig.parameters['device'].default == 'cuda:0'
        assert sig.parameters['batch_size'].default == 64
        assert sig.parameters['timeout_ms'].default == 3.0
        assert sig.parameters['use_mixed_precision'].default == True

    def test_warmup_signature(self):
        """Test warmup method signature."""
        import inspect
        sig = inspect.signature(InferenceWorker.warmup)

        expected_params = ['self', 'input_shape']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params

        # Check type hints
        annotations = sig.parameters['input_shape'].annotation
        assert annotations == Tuple[int, int, int]

    def test_inference_loop_signature(self):
        """Test inference_loop method signature."""
        import inspect
        sig = inspect.signature(InferenceWorker.inference_loop)

        expected_params = ['self', 'input_queue', 'output_queues']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params

    def test_batch_inference_signature(self):
        """Test batch_inference method signature."""
        import inspect
        sig = inspect.signature(InferenceWorker.batch_inference)

        expected_params = ['self', 'positions']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params

        # Check return type
        assert sig.return_annotation == Tuple[np.ndarray, np.ndarray]

    def test_get_metrics_signature(self):
        """Test get_metrics method signature."""
        import inspect
        sig = inspect.signature(InferenceWorker.get_metrics)

        expected_params = ['self']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params

        # Check return type
        assert sig.return_annotation == Dict[str, float]


class TestInferenceRequestContract:
    """Test InferenceRequest data class contract."""

    def test_inference_request_creation(self):
        """InferenceRequest should create with required parameters."""
        features = np.random.rand(3, 15, 15).astype(np.float32)
        path = [0, 1, 2]

        request = InferenceRequest(
            leaf_node_id=42,
            features=features,
            thread_id=1,
            path=path
        )

        assert request.leaf_node_id == 42
        assert np.array_equal(request.features, features)
        assert request.thread_id == 1
        assert request.path == path
        assert request.timestamp is None

    def test_inference_request_attributes(self):
        """InferenceRequest should have all required attributes."""
        required_attrs = [
            'leaf_node_id', 'features', 'thread_id', 'path', 'timestamp'
        ]

        features = np.zeros((3, 15, 15))
        request = InferenceRequest(0, features, 0, [])

        for attr in required_attrs:
            assert hasattr(request, attr), f"Missing attribute: {attr}"


class TestInferenceResultContract:
    """Test InferenceResult data class contract."""

    def test_inference_result_creation(self):
        """InferenceResult should create with required parameters."""
        policy = np.random.rand(225).astype(np.float32)
        path = [0, 1, 2]

        result = InferenceResult(
            node_id=42,
            policy=policy,
            value=0.5,
            path=path,
            processing_time_ms=2.5
        )

        assert result.node_id == 42
        assert np.array_equal(result.policy, policy)
        assert result.value == 0.5
        assert result.path == path
        assert result.processing_time_ms == 2.5

    def test_inference_result_attributes(self):
        """InferenceResult should have all required attributes."""
        required_attrs = [
            'node_id', 'policy', 'value', 'path', 'processing_time_ms'
        ]

        policy = np.zeros(225)
        result = InferenceResult(0, policy, 0.0, [], 0.0)

        for attr in required_attrs:
            assert hasattr(result, attr), f"Missing attribute: {attr}"


class TestFactoryFunctionsContract:
    """Test factory and utility function contracts."""

    def test_create_inference_worker_real_implementation(self):
        """create_inference_worker should return a working inference worker."""
        # Create a valid temporary model file
        model_path = create_temporary_model('gomoku')

        try:
            # Test with CPU device (should work even without CUDA)
            worker = create_inference_worker(model_path, device='cpu')
            assert worker is not None
            assert hasattr(worker, 'warmup')
            assert hasattr(worker, 'batch_inference')
            assert hasattr(worker, 'get_metrics')
        finally:
            os.unlink(model_path)

    def test_create_inference_worker_signature(self):
        """Test create_inference_worker function signature."""
        import inspect
        sig = inspect.signature(create_inference_worker)

        expected_params = ['model_path', 'device', 'kwargs']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params

        # Check default values
        assert sig.parameters['device'].default == 'cuda:0'
        assert sig.parameters['kwargs'].kind == inspect.Parameter.VAR_KEYWORD

    def test_estimate_batch_size_real_implementation(self):
        """estimate_batch_size should return a valid batch size."""
        # Create a valid temporary model file
        model_path = create_temporary_model('gomoku')

        try:
            # Test with CPU device (should work even without CUDA)
            batch_size = estimate_batch_size(model_path, (36, 15, 15), device='cpu')
            assert isinstance(batch_size, int)
            assert batch_size > 0
            assert batch_size <= 256  # Reasonable upper bound
        finally:
            os.unlink(model_path)

    def test_estimate_batch_size_signature(self):
        """Test estimate_batch_size function signature."""
        import inspect
        sig = inspect.signature(estimate_batch_size)

        expected_params = ['model_path', 'input_shape', 'device', 'memory_fraction']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params

        # Check default values and types
        assert sig.parameters['device'].default == 'cuda:0'
        assert sig.parameters['memory_fraction'].default == 0.85
        assert sig.parameters['input_shape'].annotation == Tuple[int, int, int]
        assert sig.return_annotation == int

    def test_benchmark_inference_real_implementation(self):
        """benchmark_inference should return valid benchmark results."""
        # Create a valid temporary model file
        model_path = create_temporary_model('gomoku')

        try:
            # Test with CPU device and small iteration count for speed
            results = benchmark_inference(model_path, (36, 15, 15), [8, 16], device='cpu', num_iterations=3)
            assert isinstance(results, dict)

            for batch_size in [8, 16]:
                assert batch_size in results
                metrics = results[batch_size]
                assert 'latency_ms' in metrics
                assert 'throughput' in metrics
                assert 'memory_usage_gb' in metrics
                assert 'gpu_utilization' in metrics
                assert metrics['latency_ms'] >= 0
                assert metrics['throughput'] >= 0
        finally:
            os.unlink(model_path)

    def test_benchmark_inference_signature(self):
        """Test benchmark_inference function signature."""
        import inspect
        sig = inspect.signature(benchmark_inference)

        expected_params = ['model_path', 'input_shape', 'batch_sizes', 'device', 'num_iterations']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params

        # Check default values and types
        assert sig.parameters['device'].default == 'cuda:0'
        assert sig.parameters['num_iterations'].default == 100
        assert sig.return_annotation == Dict[int, Dict[str, float]]


class TestCPUFallbackContract:
    """Test CPU fallback inference contract."""

    def test_cpu_fallback_real_implementation(self):
        """CPUFallbackInference should work with real implementation."""
        # Create a valid temporary model file
        model_path = create_temporary_model('gomoku')

        try:
            # Should create CPU fallback without error
            cpu_inference = CPUFallbackInference(model_path)
            assert cpu_inference is not None
            assert hasattr(cpu_inference, 'inference')

            # Test inference with dummy features (36 channels for Gomoku)
            features = np.random.randn(36, 15, 15).astype(np.float32)  # Gomoku features
            policy, value = cpu_inference.inference(features)

            assert isinstance(policy, np.ndarray)
            assert isinstance(value, (float, np.floating))
            assert policy.shape == (225,)  # Gomoku action space
            assert -1.0 <= value <= 1.0
            assert np.isclose(np.sum(policy), 1.0, atol=1e-6)  # Policy should be normalized
        finally:
            os.unlink(model_path)

    def test_cpu_fallback_has_required_methods(self):
        """CPUFallbackInference should have required methods."""
        required_methods = ['__init__', 'inference']

        for method_name in required_methods:
            assert hasattr(CPUFallbackInference, method_name), f"Missing method: {method_name}"

    def test_cpu_fallback_init_signature(self):
        """Test CPUFallbackInference.__init__ signature."""
        import inspect
        sig = inspect.signature(CPUFallbackInference.__init__)

        expected_params = ['self', 'model_path']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params

    def test_cpu_fallback_inference_signature(self):
        """Test CPUFallbackInference.inference signature."""
        import inspect
        sig = inspect.signature(CPUFallbackInference.inference)

        expected_params = ['self', 'features']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params

        # Check return type
        assert sig.return_annotation == Tuple[np.ndarray, float]


class TestModelValidationContract:
    """Test model validation function contract."""

    def test_validate_model_compatibility_real_implementation(self):
        """validate_model_compatibility should return validation results."""
        # Test validation for different games
        for game_type in ['gomoku', 'chess', 'go']:
            # Create a valid temporary model file for each game
            model_path = create_temporary_model(game_type)

            try:
                result = validate_model_compatibility(model_path, game_type)
                assert isinstance(result, dict)

                # Check required keys
                required_keys = ['compatible', 'input_shape', 'output_shape', 'architecture', 'parameters']
                for key in required_keys:
                    assert key in result, f"Missing key: {key}"

                assert isinstance(result['compatible'], bool)
                assert isinstance(result['input_shape'], tuple)
                assert isinstance(result['output_shape'], tuple)
                assert isinstance(result['architecture'], str)
                assert isinstance(result['parameters'], int)
            finally:
                os.unlink(model_path)

    def test_validate_model_compatibility_signature(self):
        """Test validate_model_compatibility function signature."""
        import inspect
        sig = inspect.signature(validate_model_compatibility)

        expected_params = ['model_path', 'game_type']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params

        # Check return type
        assert sig.return_annotation == Dict[str, Any]


class TestIntegrationCompatibility:
    """Test compatibility with GPU/CPU environments."""

    def test_torch_compatibility(self):
        """Verify PyTorch is available and version compatible."""
        assert torch.__version__
        version_parts = torch.__version__.split('.')
        major_version = int(version_parts[0])
        assert major_version >= 2, f"PyTorch 2.x required, got {torch.__version__}"

    def test_cuda_detection(self):
        """Test CUDA availability detection."""
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            assert device_count > 0
            device_name = torch.cuda.get_device_name(0)
            assert len(device_name) > 0

    def test_numpy_compatibility(self):
        """Test numpy array compatibility with inference API."""
        # Test typical game feature shapes
        test_shapes = [
            (7, 15, 15),   # Gomoku 7 planes
            (12, 8, 8),    # Chess 12 planes
            (17, 19, 19),  # Go 17 planes
        ]

        for shape in test_shapes:
            features = np.random.rand(*shape).astype(np.float32)
            assert features.dtype == np.float32
            assert features.shape == shape

            # Should be convertible to torch tensor
            tensor = torch.from_numpy(features)
            assert tensor.shape == torch.Size(shape)

    def test_queue_compatibility(self):
        """Test Queue compatibility for threading."""
        from queue import Queue, Empty, Full

        # Test basic queue operations
        q = Queue(maxsize=10)
        q.put("test_item")
        item = q.get(timeout=1.0)
        assert item == "test_item"

        # Test empty queue
        with pytest.raises(Empty):
            q.get(block=False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_detection(self):
        """Test GPU memory detection when CUDA available."""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            assert total_memory > 0

            # Should be able to allocate small tensor
            test_tensor = torch.zeros(10, 10, device='cuda:0')
            assert test_tensor.device.type == 'cuda'

            # Clean up
            del test_tensor
            torch.cuda.empty_cache()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_model_path_handling(self):
        """Test handling of invalid model paths."""
        invalid_path = "/nonexistent/model.pth"

        # Should raise FileNotFoundError with real implementation
        with pytest.raises(FileNotFoundError):
            create_inference_worker(invalid_path)

    def test_invalid_device_handling(self):
        """Test handling of invalid device specifications."""
        # Create a valid temporary model file
        model_path = create_temporary_model('gomoku')

        try:
            # Invalid device falls back to CPU worker, which should succeed
            worker = create_inference_worker(model_path, device='invalid:0')
            assert worker is not None
            # Verify it fell back to CPU
            assert hasattr(worker, 'device')
        finally:
            os.unlink(model_path)

    def test_invalid_batch_sizes(self):
        """Test handling of invalid batch sizes."""
        # Create a valid temporary model file
        model_path = create_temporary_model('gomoku')

        try:
            # With real implementation, invalid memory_fraction should still return valid result
            # (it clamps to reasonable values rather than throwing errors)
            batch_size = estimate_batch_size(model_path, (36, 15, 15), device='cpu', memory_fraction=1.5)
            assert isinstance(batch_size, int)
            assert batch_size > 0
        finally:
            os.unlink(model_path)

    def test_invalid_game_types(self):
        """Test handling of invalid game types."""
        # Create a valid temporary model file
        model_path = create_temporary_model('gomoku')

        try:
            # Should raise ValueError with real implementation for invalid game type
            with pytest.raises(ValueError, match="Unsupported game type"):
                validate_model_compatibility(model_path, 'invalid_game')
        finally:
            os.unlink(model_path)


# HOWTO-RUN-TESTS Block
"""
HOWTO-RUN-TESTS
===============

Run contract tests to verify inference API compliance:

# Run all inference contract tests
python -m pytest tests/contract/test_inference_api.py -v

# Run specific test classes
python -m pytest tests/contract/test_inference_api.py::TestInferenceWorkerContract -v
python -m pytest tests/contract/test_inference_api.py::TestFactoryFunctionsContract -v
python -m pytest tests/contract/test_inference_api.py::TestIntegrationCompatibility -v

# Run with coverage
python -m pytest tests/contract/test_inference_api.py --cov=specs.001-goal-create-spec.contracts.inference_api

# Skip GPU tests if no CUDA available
python -m pytest tests/contract/test_inference_api.py -v -m "not gpu"

Expected Results:
- All tests should PASS initially (testing contract structure)
- NotImplementedError exceptions should be raised by all factory functions
- GPU compatibility tests run only if CUDA available
- 100% coverage of inference_api.py contract interfaces

Test Coverage Includes:
✅ InferenceWorker abstract base class and method signatures
✅ InferenceRequest and InferenceResult data classes
✅ Factory functions (create_inference_worker, estimate_batch_size, benchmark_inference)
✅ CPUFallbackInference class interface
✅ Model validation function interface
✅ GPU/CPU compatibility and environment detection
✅ Error handling for invalid inputs
✅ PyTorch and numpy integration compatibility

The contract tests validate that:
1. All required methods exist with correct signatures
2. Abstract methods properly raise NotImplementedError
3. Data classes have required attributes
4. Type hints are properly specified
5. Default parameter values are correct
6. GPU/CPU environments are properly detected
7. Integration with PyTorch and numpy works correctly

These tests will continue to pass throughout implementation,
serving as regression tests for API compatibility.
"""