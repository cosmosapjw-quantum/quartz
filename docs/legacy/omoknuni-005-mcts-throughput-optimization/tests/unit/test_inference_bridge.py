"""
Unit Tests for C++ Inference Bridge
====================================

Tests the CppInferenceBridge class that bridges C++ MCTS simulation runner
and Python GPU inference worker.

HOWTO-RUN-TESTS:
===============
# Run all inference bridge tests
python -m pytest tests/unit/test_inference_bridge.py -v

# Run with verbose output
python -m pytest tests/unit/test_inference_bridge.py -v -s

# Run specific test
python -m pytest tests/unit/test_inference_bridge.py::TestCppInferenceBridge::test_successful_inference -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import Future
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.cpp_inference_bridge import CppInferenceBridge
from src.utils.errors import InferenceError


class MockGameState:
    """Mock game state for testing."""

    def __init__(self, action_space=225, legal_moves=None):
        self.action_space_size = action_space
        self._legal_moves = legal_moves if legal_moves is not None else list(range(action_space))
        self.features = np.random.randn(17, 15, 15).astype(np.float32)

    def get_tensor_representation(self):
        """Mock tensor representation."""
        return self.features

    def get_legal_moves(self):
        """Mock legal moves."""
        return self._legal_moves


class MockInferenceWorker:
    """Mock GPU inference worker for testing."""

    def __init__(self, should_fail=False, should_timeout=False, should_oom=False):
        self.should_fail = should_fail
        self.should_timeout = should_timeout
        self.should_oom = should_oom
        self._fallback_triggered = False
        self._cpu_fallback_worker = None
        self.call_count = 0

    def batch_inference(self, features_batch):
        """Mock batch inference."""
        self.call_count += 1

        if self.should_timeout:
            raise TimeoutError("Inference timeout")

        if self.should_oom:
            raise RuntimeError("CUDA out of memory")

        if self.should_fail:
            raise RuntimeError("GPU inference failed")

        # Return dummy results
        batch_size = len(features_batch)
        action_space = 225  # Gomoku 15x15
        policy_batch = np.ones((batch_size, action_space), dtype=np.float32) / action_space
        value_batch = np.zeros(batch_size, dtype=np.float32)

        return policy_batch, value_batch


class MockCPUWorker:
    """Mock CPU fallback worker."""

    def __init__(self):
        self.call_count = 0

    def batch_inference(self, features_batch):
        """Mock CPU batch inference."""
        self.call_count += 1
        batch_size = len(features_batch)
        action_space = 225
        policy_batch = np.ones((batch_size, action_space), dtype=np.float32) / action_space
        value_batch = np.zeros(batch_size, dtype=np.float32)
        return policy_batch, value_batch


class TestCppInferenceBridge:
    """Test CppInferenceBridge class."""

    @pytest.fixture
    def mock_worker(self):
        """Create mock inference worker."""
        return MockInferenceWorker()

    @pytest.fixture
    def bridge(self, mock_worker):
        """Create inference bridge."""
        return CppInferenceBridge(mock_worker)

    @pytest.fixture
    def game_state(self):
        """Create mock game state."""
        return MockGameState()

    def test_initialization(self, mock_worker):
        """Test bridge initialization."""
        bridge = CppInferenceBridge(mock_worker, default_timeout=2.0)

        assert bridge.inference_worker is mock_worker
        assert bridge.default_timeout == 2.0
        assert bridge.enable_cpu_fallback is True

        metrics = bridge.get_metrics()
        assert metrics['total_requests'] == 0
        assert metrics['successful_requests'] == 0

    def test_successful_inference(self, bridge, game_state):
        """Test successful GPU inference."""
        # Call bridge
        future = bridge(game_state)

        # Should return a Future
        assert isinstance(future, Future)

        # Get result
        policy, value = future.result(timeout=1.0)

        # Verify result format
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (225,)
        assert np.isclose(np.sum(policy), 1.0, atol=1e-5)
        assert isinstance(value, (float, np.floating))

        # Check metrics
        metrics = bridge.get_metrics()
        assert metrics['total_requests'] == 1
        assert metrics['successful_requests'] == 1
        assert metrics['failed_requests'] == 0
        assert metrics['success_rate'] == 1.0

    def test_batch_inference_uses_worker(self, bridge):
        """batch_inference delegates to the worker and updates metrics."""
        positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(3)]

        policies, values = bridge.batch_inference(positions)

        assert policies.shape == (3, 225)
        assert values.shape == (3,)
        assert bridge.inference_worker.call_count == 1

        metrics = bridge.get_metrics()
        assert metrics['batch_requests'] == 1
        assert metrics['batch_failures'] == 0

    def test_feature_extraction(self, bridge, game_state):
        """Test feature extraction from game state."""
        features = bridge._extract_features(game_state)

        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert features.ndim == 3
        assert features.shape == (17, 15, 15)

    def test_invalid_game_state_no_method(self, bridge):
        """Test error handling for game state without extraction method."""
        invalid_state = Mock(spec=[])  # No methods

        future = bridge(invalid_state)

        with pytest.raises(InferenceError):
            future.result(timeout=1.0)

        metrics = bridge.get_metrics()
        assert metrics['failed_requests'] == 1

    def test_invalid_feature_shape(self, bridge):
        """Test error handling for invalid feature shape."""
        game_state = Mock()
        game_state.get_tensor_representation.return_value = np.zeros((15, 15))  # 2D not 3D

        future = bridge(game_state)

        with pytest.raises(InferenceError):
            future.result(timeout=1.0)

        metrics = bridge.get_metrics()
        assert metrics['failed_requests'] == 1

    def test_timeout_handling(self, game_state):
        """Test inference timeout handling."""
        worker = MockInferenceWorker(should_timeout=True)
        bridge = CppInferenceBridge(worker)

        future = bridge(game_state)

        with pytest.raises(InferenceError) as exc_info:
            future.result(timeout=1.0)

        assert 'timeout' in str(exc_info.value).lower()

        metrics = bridge.get_metrics()
        assert metrics['timeout_requests'] == 1
        assert metrics['failed_requests'] == 1

    def test_gpu_failure_without_fallback(self, game_state):
        """Test GPU failure without CPU fallback enabled."""
        worker = MockInferenceWorker(should_fail=True)
        bridge = CppInferenceBridge(worker, enable_cpu_fallback=False)

        future = bridge(game_state)

        with pytest.raises(InferenceError):
            future.result(timeout=1.0)

        metrics = bridge.get_metrics()
        assert metrics['failed_requests'] == 1
        assert metrics['cpu_fallback_requests'] == 0

    def test_batch_inference_cpu_fallback(self, mock_worker):
        """Batched CPU fallback is used when the GPU path fails."""
        mock_worker.should_fail = True
        mock_worker._cpu_fallback_worker = MockCPUWorker()

        bridge = CppInferenceBridge(mock_worker, enable_cpu_fallback=True)

        positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(2)]
        policies, values = bridge.batch_inference(positions)

        assert policies.shape == (2, 225)
        assert values.shape == (2,)
        assert mock_worker._cpu_fallback_worker.call_count == 1

    def test_cpu_fallback_on_oom(self, game_state):
        """Test CPU fallback triggers on CUDA OOM."""
        worker = MockInferenceWorker(should_oom=True)
        cpu_worker = MockCPUWorker()
        worker._cpu_fallback_worker = cpu_worker

        bridge = CppInferenceBridge(worker, enable_cpu_fallback=True)

        future = bridge(game_state)

        # Should succeed via CPU fallback
        policy, value = future.result(timeout=1.0)

        assert isinstance(policy, np.ndarray)
        assert isinstance(value, (float, np.floating))

        # Verify CPU worker was called
        assert cpu_worker.call_count == 1

        metrics = bridge.get_metrics()
        assert metrics['successful_requests'] == 1
        assert metrics['cpu_fallback_requests'] == 1

    def test_cpu_fallback_on_cuda_error(self, game_state):
        """Test CPU fallback triggers on CUDA errors."""
        worker = MockInferenceWorker()
        worker.batch_inference = Mock(side_effect=RuntimeError("CUDA error: device-side assert"))
        cpu_worker = MockCPUWorker()
        worker._cpu_fallback_worker = cpu_worker

        bridge = CppInferenceBridge(worker, enable_cpu_fallback=True)

        future = bridge(game_state)

        # Should succeed via CPU fallback
        policy, value = future.result(timeout=1.0)

        assert isinstance(policy, np.ndarray)
        assert cpu_worker.call_count == 1

        metrics = bridge.get_metrics()
        assert metrics['cpu_fallback_requests'] == 1

    def test_uniform_policy_fallback(self, bridge, game_state):
        """Test uniform policy fallback when no CPU worker available."""
        # Force failure without CPU fallback worker
        bridge.inference_worker.batch_inference = Mock(side_effect=RuntimeError("GPU error"))
        bridge.inference_worker._cpu_fallback_worker = None

        future = bridge(game_state)

        # Should get uniform policy
        policy, value = future.result(timeout=1.0)

        assert isinstance(policy, np.ndarray)
        assert policy.shape == (225,)
        assert np.isclose(np.sum(policy), 1.0, atol=1e-5)
        assert value == 0.0

        # All legal moves should have equal probability
        legal_moves = game_state.get_legal_moves()
        expected_prob = 1.0 / len(legal_moves)
        for move in legal_moves:
            assert np.isclose(policy[move], expected_prob, atol=1e-5)

        metrics = bridge.get_metrics()
        assert metrics['cpu_fallback_requests'] == 1

    def test_fallback_detection_oom(self, bridge):
        """Test OOM error detection for fallback."""
        oom_error = RuntimeError("CUDA out of memory (OOM)")
        assert bridge._should_use_cpu_fallback(oom_error)

    def test_fallback_detection_cuda(self, bridge):
        """Test CUDA error detection for fallback."""
        cuda_error = RuntimeError("CUDA error: invalid configuration")
        assert bridge._should_use_cpu_fallback(cuda_error)

    def test_fallback_detection_worker_flag(self, bridge):
        """Test fallback detection via worker flag."""
        bridge.inference_worker._fallback_triggered = True
        generic_error = RuntimeError("Some error")
        assert bridge._should_use_cpu_fallback(generic_error)

    def test_fallback_detection_negative(self, bridge):
        """Test non-fallback error detection."""
        generic_error = RuntimeError("Random error")
        bridge.inference_worker._fallback_triggered = False
        assert not bridge._should_use_cpu_fallback(generic_error)

    def test_multiple_requests(self, bridge, game_state):
        """Test handling multiple inference requests."""
        num_requests = 10

        futures = []
        for _ in range(num_requests):
            future = bridge(game_state)
            futures.append(future)

        # All should succeed
        for future in futures:
            policy, value = future.result(timeout=1.0)
            assert isinstance(policy, np.ndarray)

        metrics = bridge.get_metrics()
        assert metrics['total_requests'] == num_requests
        assert metrics['successful_requests'] == num_requests
        assert metrics['success_rate'] == 1.0

    def test_metrics_tracking(self, bridge, game_state):
        """Test metrics tracking across different outcomes."""
        # Successful request
        future1 = bridge(game_state)
        future1.result(timeout=1.0)

        # Failed request
        bridge.inference_worker.should_fail = True
        bridge.enable_cpu_fallback = False
        future2 = bridge(game_state)
        try:
            future2.result(timeout=1.0)
        except InferenceError:
            pass

        metrics = bridge.get_metrics()
        assert metrics['total_requests'] == 2
        assert metrics['successful_requests'] == 1
        assert metrics['failed_requests'] == 1
        assert metrics['success_rate'] == 0.5

    def test_metrics_reset(self, bridge, game_state):
        """Test metrics reset."""
        # Make some requests
        for _ in range(5):
            future = bridge(game_state)
            future.result(timeout=1.0)

        # Verify metrics recorded
        metrics = bridge.get_metrics()
        assert metrics['total_requests'] == 5

        # Reset
        bridge.reset_metrics()

        # Verify cleared
        metrics = bridge.get_metrics()
        assert metrics['total_requests'] == 0
        assert metrics['successful_requests'] == 0
        assert metrics['failed_requests'] == 0

    def test_extract_features_method(self, bridge):
        """Test extract_features() method fallback."""
        game_state = Mock()
        game_state.extract_features.return_value = np.random.randn(17, 15, 15)
        del game_state.get_tensor_representation  # Remove primary method

        features = bridge._extract_features(game_state)

        assert isinstance(features, np.ndarray)
        assert features.ndim == 3

    def test_concurrent_requests(self, bridge, game_state):
        """Test thread safety with concurrent requests."""
        import threading

        num_threads = 10
        futures = []
        lock = threading.Lock()

        def make_request():
            future = bridge(game_state)
            with lock:
                futures.append(future)

        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All requests should succeed
        for future in futures:
            policy, value = future.result(timeout=1.0)
            assert isinstance(policy, np.ndarray)

        metrics = bridge.get_metrics()
        assert metrics['total_requests'] == num_threads
        assert metrics['successful_requests'] == num_threads

    def test_cpu_fallback_failure(self, game_state):
        """Test handling when both GPU and CPU fallback fail."""
        worker = MockInferenceWorker(should_fail=True)
        cpu_worker = Mock()
        cpu_worker.batch_inference.side_effect = RuntimeError("CPU also failed")
        worker._cpu_fallback_worker = cpu_worker

        bridge = CppInferenceBridge(worker, enable_cpu_fallback=True)

        future = bridge(game_state)

        with pytest.raises(InferenceError) as exc_info:
            future.result(timeout=1.0)

        assert 'failed' in str(exc_info.value).lower()

        metrics = bridge.get_metrics()
        assert metrics['failed_requests'] == 1


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
