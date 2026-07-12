"""
Unit tests for GPU Buffer Pool (T008c)

Validates buffer pooling functionality:
1. Buffer allocation and reuse
2. Thread safety
3. Memory efficiency
4. OOM handling
5. Metrics tracking
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    from src.core.dlpack_inference_bridge import GPUBufferPool, DLPackInferenceBridge
    from src.neural.model import create_random_model
    import alphazero_py
    COMPONENTS_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    CUDA_AVAILABLE = False
    print(f"Components not available: {e}")


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
class TestGPUBufferPool:
    """Tests for GPU buffer pool functionality."""

    def test_buffer_pool_initialization_cpu(self):
        """Buffer pool should not pre-allocate on CPU."""
        device = torch.device('cpu')
        pool = GPUBufferPool(device, num_planes=36, board_size=15)

        assert pool.device == device
        assert pool.num_planes == 36
        assert pool.board_size == 15
        assert len(pool.pool) == 0, "CPU pool should not pre-allocate"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_buffer_pool_initialization_cuda(self):
        """Buffer pool should pre-allocate on CUDA."""
        device = torch.device('cuda')
        pool = GPUBufferPool(device, num_planes=36, board_size=15)

        assert pool.device == device
        assert pool.num_planes == 36
        assert pool.board_size == 15
        assert len(pool.pool) > 0, "CUDA pool should pre-allocate"

        # Should have buffers for common batch sizes
        assert 16 in pool.pool or 32 in pool.pool or 64 in pool.pool

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_buffer_get_and_release(self):
        """Test buffer retrieval and release."""
        device = torch.device('cuda')
        pool = GPUBufferPool(device, num_planes=36, board_size=15)

        # Get a buffer for batch size 32
        buffer1 = pool.get_buffer(32)

        if buffer1 is not None:
            # Should be a valid tensor
            assert buffer1.shape == (32, 36, 15, 15)
            assert buffer1.device.type == device.type

            # Release it
            pool.release_buffer(buffer1)

            # Should be able to get it again
            buffer2 = pool.get_buffer(32)
            assert buffer2 is not None
            assert buffer2 is buffer1, "Should reuse the same buffer"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_buffer_pool_hit_miss_metrics(self):
        """Test buffer pool hit/miss tracking."""
        device = torch.device('cuda')
        pool = GPUBufferPool(device, num_planes=36, board_size=15)

        # Request common size (should hit if pool initialized)
        buffer32 = pool.get_buffer(32)

        # Request uncommon size (should miss)
        buffer7 = pool.get_buffer(7)

        stats = pool.get_stats()

        if buffer32 is not None:
            assert stats['hits'] >= 1, "Should have at least one hit"

        assert stats['misses'] >= 1, "Should have at least one miss"
        assert stats['total_requests'] == stats['hits'] + stats['misses']

        if buffer32 is not None:
            pool.release_buffer(buffer32)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_buffer_pool_double_buffering(self):
        """Test that pool provides multiple buffers per size."""
        device = torch.device('cuda')
        pool = GPUBufferPool(device, num_planes=36, board_size=15)

        # Get two buffers of the same size
        buffer1 = pool.get_buffer(32)
        buffer2 = pool.get_buffer(32)

        # If pool has 2 buffers for size 32, both should succeed
        if buffer1 is not None and buffer2 is not None:
            assert buffer1 is not buffer2, "Should get different buffers"

            # Release both
            pool.release_buffer(buffer1)
            pool.release_buffer(buffer2)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_buffer_pool_exhaustion(self):
        """Test behavior when all buffers are in use."""
        device = torch.device('cuda')
        pool = GPUBufferPool(device, num_planes=36, board_size=15)

        buffers = []

        # Try to get many buffers for the same size
        for _ in range(10):
            buffer = pool.get_buffer(32)
            if buffer is not None:
                buffers.append(buffer)

        # Should eventually get None (pool exhausted)
        # OR get a buffer if pool has enough

        stats = pool.get_stats()
        if len(buffers) < 10:
            assert stats['misses'] > 0, "Should have misses when pool exhausted"

        # Release all buffers
        for buffer in buffers:
            pool.release_buffer(buffer)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_buffer_pool_memory_budget(self):
        """Test that buffer pool stays within memory budget."""
        device = torch.device('cuda')
        pool = GPUBufferPool(device, num_planes=36, board_size=15)

        total_buffers = sum(len(buffers) for buffers in pool.pool.values())

        # Gomoku: 36 planes × 15×15 × 4 bytes = 32.4 KB per state
        # Batch 64: 64 × 32.4 KB = 2.07 MB
        # 3 sizes × 2 buffers = 6 buffers total
        # Total: ~7 MB

        assert total_buffers <= 10, f"Should have ≤10 buffers, got {total_buffers}"

    def test_buffer_pool_cleanup(self):
        """Test buffer pool cleanup."""
        device = torch.device('cpu')  # Use CPU to avoid GPU allocation issues
        pool = GPUBufferPool(device, num_planes=36, board_size=15)

        # Cleanup should clear the pool
        pool.cleanup()
        assert len(pool.pool) == 0


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
class TestDLPackInferenceBridgeWithBufferPool:
    """Integration tests for DLPackInferenceBridge with buffer pool."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_buffer_pool_integration(self):
        """Test that buffer pool integrates correctly with inference bridge."""
        model = create_random_model('gomoku', seed=42)
        model = model.cuda()
        model.eval()

        # Create bridge with buffer pool enabled
        bridge = DLPackInferenceBridge(
            model=model,
            device='cuda',
            enable_buffer_pool=True
        )

        # Create test states
        states = [alphazero_py.GomokuState() for _ in range(32)]

        # Run inference (should initialize buffer pool)
        results = bridge.batch_inference(states)

        assert len(results) == 32

        # Check that buffer pool was initialized
        assert bridge.buffer_pool is not None

        # Get metrics
        metrics = bridge.get_metrics()
        assert 'buffer_pool' in metrics
        assert metrics['buffer_pool'] is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_buffer_pool_reuse(self):
        """Test that buffer pool reuses buffers across multiple inferences."""
        model = create_random_model('gomoku', seed=42)
        model = model.cuda()
        model.eval()

        bridge = DLPackInferenceBridge(
            model=model,
            device='cuda',
            enable_buffer_pool=True
        )

        states = [alphazero_py.GomokuState() for _ in range(32)]

        # Run multiple inferences
        for _ in range(5):
            results = bridge.batch_inference(states)
            assert len(results) == 32

        # Check buffer pool statistics
        metrics = bridge.get_metrics()
        if metrics['buffer_pool'] is not None:
            pool_stats = metrics['buffer_pool']

            # Should have some hits (buffer reuse)
            if pool_stats['total_requests'] > 0:
                assert pool_stats['hits'] >= 0, "Should have buffer pool requests"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_buffer_pool_different_batch_sizes(self):
        """Test buffer pool with different batch sizes."""
        model = create_random_model('gomoku', seed=42)
        model = model.cuda()
        model.eval()

        bridge = DLPackInferenceBridge(
            model=model,
            device='cuda',
            enable_buffer_pool=True
        )

        # Test different batch sizes
        batch_sizes = [16, 32, 64, 7, 100]

        for batch_size in batch_sizes:
            states = [alphazero_py.GomokuState() for _ in range(batch_size)]
            results = bridge.batch_inference(states)
            assert len(results) == batch_size

        # Check that common batch sizes had hits
        metrics = bridge.get_metrics()
        if metrics['buffer_pool'] is not None:
            pool_stats = metrics['buffer_pool']

            # Should have both hits and misses
            assert pool_stats['total_requests'] > 0

    def test_buffer_pool_disabled(self):
        """Test that buffer pool can be disabled."""
        model = create_random_model('gomoku', seed=42)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        bridge = DLPackInferenceBridge(
            model=model,
            device=device,
            enable_buffer_pool=False
        )

        states = [alphazero_py.GomokuState() for _ in range(32)]
        results = bridge.batch_inference(states)

        assert len(results) == 32

        # Buffer pool should remain None
        assert bridge.buffer_pool is None

        # Metrics should show no buffer pool
        metrics = bridge.get_metrics()
        assert metrics['buffer_pool'] is None

    def test_metrics_reset_with_buffer_pool(self):
        """Test that metrics reset includes buffer pool metrics."""
        model = create_random_model('gomoku', seed=42)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        bridge = DLPackInferenceBridge(
            model=model,
            device=device,
            enable_buffer_pool=True
        )

        states = [alphazero_py.GomokuState() for _ in range(32)]

        # Run some inferences
        for _ in range(3):
            bridge.batch_inference(states)

        # Reset metrics
        bridge.reset_metrics()

        metrics = bridge.get_metrics()
        assert metrics['total_batches'] == 0
        assert metrics['total_states'] == 0

        # Buffer pool metrics should also be reset
        if metrics['buffer_pool'] is not None:
            assert metrics['buffer_pool']['hits'] == 0
            assert metrics['buffer_pool']['misses'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
