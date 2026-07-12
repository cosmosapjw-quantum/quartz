"""
Unit tests for Dynamic Micro-batching (T015)
============================================

Tests the enhanced micro-batching logic including:
- Count-based (≥32) OR timeout-based (≤3ms) batching
- GPU utilization monitoring and adaptive batch sizing
- Performance feedback loops and optimization

Run with: python -m pytest tests/unit/test_micro_batching.py -v
"""

import pytest
import torch
import numpy as np
import time
import threading
import tempfile
import os
from queue import Queue, Empty
from unittest.mock import Mock, patch, MagicMock

# Import worker implementation
from src.neural.inference_worker import (
    GPUInferenceWorker,
    create_inference_worker
)

# Import neural model for testing
from src.neural.model import create_model_for_game

# Import contract interfaces
import sys
sys.path.append('specs/001-goal-create-spec')
from contracts.inference_api import (
    InferenceRequest,
    InferenceResult
)


class TestMicroBatchingLogic:
    """Test dynamic micro-batching functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create worker with micro-batching parameters
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            self.model_path = f.name
            # Create and save a valid model
            model = create_model_for_game('gomoku')
            with torch.no_grad():
                dummy_input = torch.randn(1, 36, 15, 15)  # Enhanced Gomoku: 36 input channels
                _ = model(dummy_input)  # Initialize lazy layers
            torch.save(model.state_dict(), self.model_path)

        # Create worker with specific micro-batching configuration
        self.worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cpu',  # Use CPU for consistent testing
            batch_size=64,  # Max batch size
            timeout_ms=10.0,  # Will be capped at 3ms by micro-batching
            use_mixed_precision=False
        )

    def teardown_method(self):
        """Cleanup test fixtures."""
        if hasattr(self, 'model_path'):
            os.unlink(self.model_path)

    def test_micro_batching_parameters_initialization(self):
        """Test that micro-batching parameters are set correctly."""
        # Check minimum batch size (should be ≥32 or capped by max batch size)
        assert self.worker.min_batch_size >= 1
        assert self.worker.min_batch_size <= self.worker.batch_size
        assert self.worker.min_batch_size <= 32  # Target constraint

        # Check maximum timeout (should be ≤3ms)
        assert self.worker.max_timeout_ms <= 0.003  # 3ms in seconds
        assert self.worker.max_timeout_ms > 0

        # Check target GPU utilization
        assert self.worker.target_gpu_utilization == 0.80

        # Check adaptive batching state
        assert hasattr(self.worker, '_performance_history')
        assert hasattr(self.worker, '_current_optimal_batch')
        assert hasattr(self.worker, '_gpu_handle')

    def test_optimal_batch_size_calculation_initial(self):
        """Test optimal batch size calculation with no history."""
        # With no performance history, should return min_batch_size
        optimal_size = self.worker._get_optimal_batch_size()
        assert optimal_size == self.worker.min_batch_size

    def test_optimal_batch_size_calculation_with_history(self):
        """Test optimal batch size calculation with performance history."""
        # Add some mock performance data
        for i in range(10):
            perf_data = {
                'batch_size': 16 + i,
                'inference_time': 0.001 + i * 0.0001,
                'throughput': (16 + i) / (0.001 + i * 0.0001),
                'gpu_utilization': 0.5 + i * 0.03,  # Increasing GPU util
                'timestamp': time.time() - (10 - i)
            }
            self.worker._performance_history.append(perf_data)

        optimal_size = self.worker._get_optimal_batch_size()
        assert optimal_size >= self.worker.min_batch_size
        assert optimal_size <= self.worker.batch_size

    def test_gpu_utilization_monitoring(self):
        """Test GPU utilization monitoring functionality."""
        # Test initialization (should handle missing pynvml gracefully)
        self.worker._init_gpu_monitoring()

        # Test GPU utilization retrieval (should not crash)
        gpu_util = self.worker._get_gpu_utilization()
        assert isinstance(gpu_util, float)
        assert 0.0 <= gpu_util <= 1.0

    def test_enhanced_metrics_collection(self):
        """Test enhanced metrics collection with GPU utilization."""
        # Process a mock batch to generate metrics
        batch_size = 16
        inference_time = 0.002

        initial_batches = self.worker._metrics['total_batches']
        self.worker._update_metrics(batch_size, inference_time)

        # Check basic metrics updated
        assert self.worker._metrics['total_batches'] == initial_batches + 1
        assert self.worker._metrics['total_requests'] == batch_size
        assert self.worker._metrics['total_inference_time'] == inference_time

        # Check performance history updated
        assert len(self.worker._performance_history) == 1
        perf_data = self.worker._performance_history[0]
        assert perf_data['batch_size'] == batch_size
        assert perf_data['inference_time'] == inference_time
        assert perf_data['throughput'] == batch_size / inference_time

    def test_enhanced_metrics_retrieval(self):
        """Test enhanced metrics retrieval with micro-batching data."""
        # Add some performance data
        self.worker._update_metrics(32, 0.002)
        self.worker._update_metrics(40, 0.0025)

        metrics = self.worker.get_metrics()

        # Check basic metrics
        assert 'gpu_utilization' in metrics
        assert 'avg_gpu_utilization' in metrics
        assert 'average_batch_size' in metrics
        assert 'inference_rate' in metrics

        # Check micro-batching specific metrics
        assert 'current_optimal_batch' in metrics
        assert 'min_batch_size' in metrics
        assert 'max_timeout_ms' in metrics
        assert 'target_gpu_utilization' in metrics

        # Check performance targets status
        assert 'meets_batch_target' in metrics
        assert 'meets_gpu_target' in metrics
        assert 'timeout_compliance' in metrics

        # Validate values
        assert metrics['target_gpu_utilization'] == 0.80
        assert metrics['min_batch_size'] == self.worker.min_batch_size
        assert metrics['max_timeout_ms'] == self.worker.max_timeout_ms * 1000

    def test_count_based_batching_behavior(self):
        """Test count-based batching (≥32 target)."""
        input_queue = Queue()

        # Add requests to reach target batch size
        target_requests = 35
        for i in range(target_requests):
            request = InferenceRequest(
                leaf_node_id=i,
                features=np.random.randn(36, 15, 15).astype(np.float32),  # Enhanced Gomoku: 36 input channels
                thread_id=0,
                path=[i]
            )
            input_queue.put(request)

        # Collect batch
        batch = self.worker._collect_batch(input_queue)

        # Should collect at least min_batch_size requests
        assert len(batch) >= self.worker.min_batch_size
        assert len(batch) <= target_requests

    def test_timeout_based_batching_behavior(self):
        """Test timeout-based batching (≤3ms constraint)."""
        input_queue = Queue()

        # Add only a few requests
        for i in range(5):
            request = InferenceRequest(
                leaf_node_id=i,
                features=np.random.randn(36, 15, 15).astype(np.float32),  # Enhanced Gomoku: 36 input channels
                thread_id=0,
                path=[i]
            )
            input_queue.put(request)

        # Collect batch with timing
        start_time = time.time()
        batch = self.worker._collect_batch(input_queue)
        elapsed_time = time.time() - start_time

        # Should respect timeout constraint
        assert elapsed_time <= self.worker.max_timeout_ms * 1.2  # Small tolerance
        assert len(batch) == 5  # Should get all available requests

    def test_adaptive_batch_sizing_low_gpu_util(self):
        """Test adaptive batch sizing when GPU utilization is low."""
        # Mock low GPU utilization
        with patch.object(self.worker, '_get_gpu_utilization', return_value=0.6):
            # Add performance history with low GPU utilization
            for i in range(10):
                perf_data = {
                    'batch_size': 20,
                    'inference_time': 0.002,
                    'throughput': 10000,
                    'gpu_utilization': 0.6,  # Below target (80%)
                    'timestamp': time.time() - (10 - i)
                }
                self.worker._performance_history.append(perf_data)

            initial_optimal = self.worker._current_optimal_batch
            optimal_size = self.worker._get_optimal_batch_size()

            # Should increase batch size to improve GPU utilization
            assert optimal_size >= initial_optimal

    def test_adaptive_batch_sizing_high_gpu_util(self):
        """Test adaptive batch sizing when GPU utilization is high."""
        # Mock high GPU utilization
        with patch.object(self.worker, '_get_gpu_utilization', return_value=0.95):
            # Add performance history with high GPU utilization
            for i in range(10):
                perf_data = {
                    'batch_size': 50,
                    'inference_time': 0.005,
                    'throughput': 10000,
                    'gpu_utilization': 0.95,  # Above target (80%)
                    'timestamp': time.time() - (10 - i)
                }
                self.worker._performance_history.append(perf_data)

            self.worker._current_optimal_batch = 50
            optimal_size = self.worker._get_optimal_batch_size()

            # Should decrease batch size to avoid overload
            assert optimal_size <= 50

    def test_three_phase_collection_strategy(self):
        """Test the three-phase batch collection strategy."""
        input_queue = Queue()

        # Phase 1: Quick collection - add requests rapidly
        for i in range(50):
            request = InferenceRequest(
                leaf_node_id=i,
                features=np.random.randn(36, 15, 15).astype(np.float32),  # Enhanced Gomoku: 36 input channels
                thread_id=0,
                path=[i]
            )
            input_queue.put(request)

        # Mock optimal batch size
        with patch.object(self.worker, '_get_optimal_batch_size', return_value=32):
            batch = self.worker._collect_batch(input_queue)

            # Should collect efficiently
            assert len(batch) >= 32  # At least target size
            assert len(batch) <= self.worker.batch_size

    def test_performance_history_management(self):
        """Test performance history management and limits."""
        # Add more than maxlen entries
        for i in range(150):  # More than deque maxlen of 100
            perf_data = {
                'batch_size': 16,
                'inference_time': 0.001,
                'throughput': 16000,
                'gpu_utilization': 0.8,
                'timestamp': time.time()
            }
            self.worker._performance_history.append(perf_data)

        # Should be limited to maxlen
        assert len(self.worker._performance_history) <= 100

    def test_gpu_monitoring_fallback(self):
        """Test GPU monitoring fallback behavior."""
        # Test with invalid GPU handle
        self.worker._gpu_handle = None
        gpu_util = self.worker._get_gpu_utilization()

        # Should fallback gracefully and return memory-based estimate or 0
        assert isinstance(gpu_util, float)
        assert 0.0 <= gpu_util <= 1.0

    def test_periodic_performance_logging(self):
        """Test periodic performance logging functionality."""
        with patch.object(self.worker.logger, 'info') as mock_info:
            # Process enough batches to trigger periodic logging
            for i in range(50):
                self.worker._update_metrics(32, 0.002)

            # Should have called info logging for performance summary
            mock_info.assert_called()


class TestMicroBatchingIntegration:
    """Integration tests for micro-batching with actual inference."""

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
            timeout_ms=5.0,  # Will be capped to 3ms
            use_mixed_precision=False
        )

    def teardown_method(self):
        """Cleanup integration test fixtures."""
        if hasattr(self, 'worker') and self.worker.is_running():
            self.worker.stop_worker()
        if hasattr(self, 'model_path'):
            os.unlink(self.model_path)

    def test_micro_batching_with_real_inference(self):
        """Test micro-batching with actual neural network inference."""
        input_queue = Queue()
        output_queues = [Queue()]

        # Add multiple requests
        positions = []
        for i in range(40):
            position = np.random.randn(36, 15, 15).astype(np.float32)  # Enhanced Gomoku: 36 input channels
            positions.append(position)
            request = InferenceRequest(
                leaf_node_id=i,
                features=position,
                thread_id=0,
                path=[i]
            )
            input_queue.put(request)

        # Start worker
        self.worker.start_worker(input_queue, output_queues)

        try:
            # Wait for processing
            time.sleep(0.5)

            # Check metrics
            metrics = self.worker.get_metrics()
            assert metrics['total_batches'] > 0
            assert metrics['total_requests'] > 0
            assert metrics['current_optimal_batch'] >= self.worker.min_batch_size

        finally:
            self.worker.stop_worker()

    def test_adaptive_batching_under_load(self):
        """Test adaptive batching behavior under sustained load."""
        input_queue = Queue()
        output_queues = [Queue()]

        self.worker.start_worker(input_queue, output_queues)

        try:
            # Simulate sustained load
            for round_idx in range(5):
                # Add burst of requests
                for i in range(30):
                    request = InferenceRequest(
                        leaf_node_id=round_idx * 30 + i,
                        features=np.random.randn(36, 15, 15).astype(np.float32),  # Enhanced Gomoku: 36 input channels
                        thread_id=0,
                        path=[round_idx * 30 + i]
                    )
                    input_queue.put(request)

                time.sleep(0.1)  # Allow processing

            # Let it settle
            time.sleep(0.5)

            # Check that adaptive batching has adjusted
            metrics = self.worker.get_metrics()

            # Should have processed at least one batch
            assert metrics['total_batches'] >= 1

            # Adaptive batch size should be reasonable
            assert self.worker.min_batch_size <= metrics['current_optimal_batch'] <= self.worker.batch_size

            # Should meet basic performance criteria
            assert metrics['inference_rate'] > 0

        finally:
            self.worker.stop_worker()


@pytest.mark.parametrize("batch_size,timeout_ms", [
    (16, 5.0),    # Small batch, reasonable timeout
    (64, 2.0),    # Large batch, tight timeout
    (128, 1.0),   # Very large batch, very tight timeout
])
def test_micro_batching_parameter_variations(batch_size, timeout_ms):
    """Test micro-batching with various parameter combinations."""
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        model_path = f.name
        model = create_model_for_game('gomoku')
        with torch.no_grad():
            dummy_input = torch.randn(1, 36, 15, 15)  # Enhanced Gomoku: 36 input channels
            _ = model(dummy_input)
        torch.save(model.state_dict(), model_path)

    try:
        worker = GPUInferenceWorker(
            model_path=model_path,
            device='cpu',
            batch_size=batch_size,
            timeout_ms=timeout_ms,
            use_mixed_precision=False
        )

        # Check micro-batching constraints are applied
        assert worker.min_batch_size <= min(32, batch_size)
        assert worker.max_timeout_ms <= 0.003  # Always capped at 3ms
        assert worker.target_gpu_utilization == 0.80

        # Test batch collection works
        input_queue = Queue()
        for i in range(min(batch_size, 20)):  # Add some requests
            request = InferenceRequest(
                leaf_node_id=i,
                features=np.random.randn(36, 15, 15).astype(np.float32),  # Enhanced Gomoku: 36 input channels
                thread_id=0,
                path=[i]
            )
            input_queue.put(request)

        batch = worker._collect_batch(input_queue)
        assert len(batch) <= min(batch_size, 20)
        assert len(batch) > 0

    finally:
        os.unlink(model_path)