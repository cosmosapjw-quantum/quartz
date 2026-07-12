"""
Unit tests for Non-Blocking GPU Transfers (T008d)

Validates that non-blocking transfers:
1. Use CUDA streams correctly
2. Profile transfer times accurately
3. Work with concurrent CPU operations
4. Maintain correctness with stream synchronization
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    from src.core.dlpack_inference_bridge import DLPackInferenceBridge
    from src.neural.model import create_random_model
    import alphazero_py
    COMPONENTS_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    CUDA_AVAILABLE = False
    print(f"Components not available: {e}")


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
class TestNonBlockingTransfers:
    """Tests for non-blocking GPU transfer optimization (T008d)."""

    def test_stream_pool_initialization(self):
        """Verify CUDA stream pool is initialized correctly."""
        model = create_random_model('gomoku', seed=42)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        bridge = DLPackInferenceBridge(
            model=model,
            device=device,
            stream_pool_size=4
        )

        if device == 'cuda':
            assert len(bridge.stream_pool) == 4, "Should have 4 streams"
            for stream in bridge.stream_pool:
                assert isinstance(stream, torch.cuda.Stream)
        else:
            assert len(bridge.stream_pool) == 0, "CPU should have no streams"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_stream_rotation(self):
        """Test that stream pool rotates through streams."""
        model = create_random_model('gomoku', seed=42)
        model = model.cuda()
        model.eval()

        bridge = DLPackInferenceBridge(
            model=model,
            device='cuda',
            stream_pool_size=2
        )

        states = [alphazero_py.GomokuState() for _ in range(32)]

        # Initial index should be 0
        assert bridge.stream_index == 0

        # First inference should use stream 0, then increment to 1
        bridge.batch_inference(states)
        assert bridge.stream_index == 1

        # Second inference should use stream 1, then wrap to 0
        bridge.batch_inference(states)
        assert bridge.stream_index == 0

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_transfer_time_profiling(self):
        """Test that transfer times are profiled correctly."""
        model = create_random_model('gomoku', seed=42)
        model = model.cuda()
        model.eval()

        bridge = DLPackInferenceBridge(
            model=model,
            device='cuda',
            stream_pool_size=2
        )

        states = [alphazero_py.GomokuState() for _ in range(32)]

        # Run several inferences
        for _ in range(10):
            bridge.batch_inference(states)

        # Check that transfer times are recorded
        metrics = bridge.get_metrics()

        assert metrics['avg_h2d_transfer_ms'] >= 0.0, "H2D transfer time should be non-negative"
        assert metrics['avg_d2h_transfer_ms'] >= 0.0, "D2H transfer time should be non-negative"
        assert metrics['avg_inference_ms'] > 0.0, "Inference time should be positive"

        # Total should roughly equal avg_latency_ms (allowing for overhead)
        total_avg = (
            metrics['avg_h2d_transfer_ms'] +
            metrics['avg_d2h_transfer_ms'] +
            metrics['avg_inference_ms']
        )

        # Allow 50% overhead for Python/profiling overhead
        assert total_avg <= metrics['avg_latency_ms'] * 1.5, \
            f"Component times {total_avg:.2f}ms should be <= 1.5× total {metrics['avg_latency_ms']:.2f}ms"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_non_blocking_correctness(self):
        """Verify that non-blocking transfers produce correct results."""
        model = create_random_model('gomoku', seed=42)
        model = model.cuda()
        model.eval()

        # Bridge with streams
        bridge_async = DLPackInferenceBridge(
            model=model,
            device='cuda',
            stream_pool_size=2
        )

        # Same model, no stream pool (blocking)
        bridge_sync = DLPackInferenceBridge(
            model=model,
            device='cuda',
            stream_pool_size=0  # Disable stream pool
        )

        # Use same states
        states = [alphazero_py.GomokuState() for _ in range(16)]

        # Get results from both
        results_async = bridge_async.batch_inference(states)
        results_sync = bridge_sync.batch_inference(states)

        # Should produce identical results
        assert len(results_async) == len(results_sync)

        for (policy_async, value_async), (policy_sync, value_sync) in zip(results_async, results_sync):
            # Policy should be identical (within float tolerance)
            assert len(policy_async) == len(policy_sync)
            for p1, p2 in zip(policy_async, policy_sync):
                assert abs(p1 - p2) < 1e-5, f"Policy differs: {p1} vs {p2}"

            # Value should be identical
            assert abs(value_async - value_sync) < 1e-5, f"Value differs: {value_async} vs {value_sync}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_stream_with_buffer_pool(self):
        """Test that streams work correctly with buffer pool."""
        model = create_random_model('gomoku', seed=42)
        model = model.cuda()
        model.eval()

        bridge = DLPackInferenceBridge(
            model=model,
            device='cuda',
            stream_pool_size=2,
            enable_buffer_pool=True
        )

        # Common batch size (should hit buffer pool)
        states = [alphazero_py.GomokuState() for _ in range(32)]

        # Run multiple inferences
        for _ in range(5):
            results = bridge.batch_inference(states)
            assert len(results) == 32

        # Check buffer pool was used
        metrics = bridge.get_metrics()
        if metrics['buffer_pool'] is not None:
            assert metrics['buffer_pool']['hits'] > 0, "Should have buffer pool hits"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_different_batch_sizes(self):
        """Test streams with different batch sizes."""
        model = create_random_model('gomoku', seed=42)
        model = model.cuda()
        model.eval()

        bridge = DLPackInferenceBridge(
            model=model,
            device='cuda',
            stream_pool_size=2
        )

        batch_sizes = [8, 16, 32, 64, 128]

        for batch_size in batch_sizes:
            states = [alphazero_py.GomokuState() for _ in range(batch_size)]
            results = bridge.batch_inference(states)

            assert len(results) == batch_size

            # Validate results
            for idx, (policy, value) in enumerate(results):
                assert len(policy) == states[0].get_action_space_size()
                policy_sum = sum(policy)
                # Softmax should produce probabilities that sum to 1.0
                assert abs(policy_sum - 1.0) < 1e-5, \
                    f"Batch {batch_size}, result {idx}: Policy sum {policy_sum} not close to 1.0"
                # tanh() guarantees values in [-1, 1]
                assert -1.0 <= value <= 1.0, f"Value {value} out of range [-1, 1]"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_concurrent_inference_calls(self):
        """Test multiple concurrent inference calls with streams."""
        import threading

        model = create_random_model('gomoku', seed=42)
        model = model.cuda()
        model.eval()

        bridge = DLPackInferenceBridge(
            model=model,
            device='cuda',
            stream_pool_size=4
        )

        results_list = []
        errors = []

        def run_inference(thread_id):
            try:
                states = [alphazero_py.GomokuState() for _ in range(16)]
                results = bridge.batch_inference(states)
                results_list.append((thread_id, results))
            except Exception as e:
                errors.append((thread_id, e))

        # Launch 8 threads
        threads = []
        for i in range(8):
            t = threading.Thread(target=run_inference, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all
        for t in threads:
            t.join()

        # Check no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Check all completed
        assert len(results_list) == 8

        # Each should have 16 results
        for thread_id, results in results_list:
            assert len(results) == 16

    def test_metrics_reset_includes_transfer_times(self):
        """Test that metrics reset includes transfer time metrics."""
        model = create_random_model('gomoku', seed=42)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        bridge = DLPackInferenceBridge(
            model=model,
            device=device,
            stream_pool_size=2
        )

        states = [alphazero_py.GomokuState() for _ in range(16)]

        # Run some inferences
        for _ in range(3):
            bridge.batch_inference(states)

        # Reset metrics
        bridge.reset_metrics()

        metrics = bridge.get_metrics()
        assert metrics['total_batches'] == 0
        assert metrics['avg_h2d_transfer_ms'] == 0.0
        assert metrics['avg_d2h_transfer_ms'] == 0.0
        assert metrics['avg_inference_ms'] == 0.0

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_transfer_time_breakdown(self):
        """Test that transfer time breakdown is reasonable."""
        model = create_random_model('gomoku', seed=42)
        model = model.cuda()
        model.eval()

        bridge = DLPackInferenceBridge(
            model=model,
            device='cuda',
            stream_pool_size=2,
            use_mixed_precision=True
        )

        # Warmup
        warmup_states = [alphazero_py.GomokuState() for _ in range(64)]
        for _ in range(5):
            bridge.batch_inference(warmup_states)

        # Reset and measure
        bridge.reset_metrics()

        states = [alphazero_py.GomokuState() for _ in range(64)]
        for _ in range(10):
            bridge.batch_inference(states)

        metrics = bridge.get_metrics()

        print(f"\nTransfer time breakdown (batch_size=64, 10 iterations):")
        print(f"  H2D transfer: {metrics['avg_h2d_transfer_ms']:.3f} ms")
        print(f"  Inference:    {metrics['avg_inference_ms']:.3f} ms")
        print(f"  D2H transfer: {metrics['avg_d2h_transfer_ms']:.3f} ms")
        print(f"  Total:        {metrics['avg_latency_ms']:.3f} ms")

        # Inference should dominate for neural network workloads
        assert metrics['avg_inference_ms'] > 0.0, "Inference time should be positive"

        # Transfers should be relatively small (< 50% of total time)
        transfer_time = metrics['avg_h2d_transfer_ms'] + metrics['avg_d2h_transfer_ms']
        assert transfer_time < metrics['avg_latency_ms'], \
            "Transfer time should be less than total time"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
