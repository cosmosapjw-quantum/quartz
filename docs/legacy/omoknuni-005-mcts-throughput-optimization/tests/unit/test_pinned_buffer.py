"""
Unit tests for PinnedBuffer and BufferPool (T007b)

Tests cover:
- PinnedBuffer allocation (CUDA and malloc fallback)
- Reference counting (add_ref/release)
- BufferPool size classes and caching
- Thread safety of pool operations
- Memory leak detection
"""

import pytest
import threading
import time
from typing import List

# Import C++ extension (will be available after build)
try:
    import mcts_py
    HAS_MCTS_PY = True
except ImportError:
    HAS_MCTS_PY = False
    pytestmark = pytest.mark.skip(reason="mcts_py extension not built")


class TestPinnedBuffer:
    """Test PinnedBuffer allocation and reference counting"""

    def test_allocate_small_buffer(self):
        """Test allocating small buffer (should use malloc fallback)"""
        # 4KB buffer
        buffer = mcts_py.PinnedBuffer(4096, use_cuda=False)

        assert buffer.size() == 4096
        assert buffer.ref_count() >= 1  # At least 1
        assert buffer.data() is not None

    def test_allocate_large_buffer(self):
        """Test allocating large buffer (1MB)"""
        # 1MB buffer
        buffer = mcts_py.PinnedBuffer(1024 * 1024, use_cuda=False)

        assert buffer.size() == 1024 * 1024
        assert buffer.ref_count() >= 1  # At least 1

    @pytest.mark.skipif(not mcts_py.is_cuda_available() if HAS_MCTS_PY else True,
                        reason="CUDA not available")
    def test_cuda_pinned_allocation(self):
        """Test CUDA pinned memory allocation"""
        buffer = mcts_py.PinnedBuffer(4096, use_cuda=True)

        assert buffer.size() == 4096
        assert buffer.is_cuda_pinned()
        assert buffer.data() is not None

    def test_reference_counting(self):
        """Test shared_ptr reference counting"""
        buffer = mcts_py.PinnedBuffer(4096, use_cuda=False)

        # Initial ref count from Python should be 1
        assert buffer.ref_count() >= 1

        # Create additional reference
        buffer2 = buffer
        assert buffer.ref_count() >= 2

        # Delete reference
        del buffer2
        assert buffer.ref_count() >= 1

    def test_zero_size_allocation_fails(self):
        """Test that allocating 0 bytes raises error"""
        with pytest.raises(ValueError):
            mcts_py.PinnedBuffer(0, use_cuda=False)

    def test_buffer_data_accessible(self):
        """Test that buffer data can be accessed"""
        buffer = mcts_py.PinnedBuffer(4096, use_cuda=False)
        data_ptr = buffer.data()

        assert data_ptr is not None
        # Should be able to get pointer multiple times
        assert buffer.data() == data_ptr


class TestBufferPool:
    """Test BufferPool caching and reuse"""

    def setup_method(self):
        """Clear pool before each test"""
        if HAS_MCTS_PY:
            mcts_py.BufferPool.instance().clear()

    def test_pool_singleton(self):
        """Test that BufferPool is a singleton"""
        pool1 = mcts_py.BufferPool.instance()
        pool2 = mcts_py.BufferPool.instance()

        # Should be the same instance
        assert pool1 is pool2

    def test_acquire_from_empty_pool(self):
        """Test acquiring buffer when pool is empty (allocates new)"""
        pool = mcts_py.BufferPool.instance()

        buffer = pool.acquire(4096, use_cuda=False)

        assert buffer.size() >= 4096
        assert buffer.ref_count() >= 1  # At least 1

        stats = pool.get_stats()
        assert stats['total_allocated'] >= 1
        assert stats['total_reused'] == 0

    def test_size_class_tiny(self):
        """Test that 4KB requests use TINY size class"""
        pool = mcts_py.BufferPool.instance()

        buffer = pool.acquire(1024, use_cuda=False)  # 1KB request

        # Should get 4KB buffer (TINY size class)
        assert buffer.size() == 4 * 1024

    def test_size_class_small(self):
        """Test that 32KB requests use SMALL size class"""
        pool = mcts_py.BufferPool.instance()

        buffer = pool.acquire(32 * 1024, use_cuda=False)  # 32KB request

        # Should get 64KB buffer (SMALL size class)
        assert buffer.size() == 64 * 1024

    def test_size_class_medium(self):
        """Test that 512KB requests use MEDIUM size class"""
        pool = mcts_py.BufferPool.instance()

        buffer = pool.acquire(512 * 1024, use_cuda=False)  # 512KB request

        # Should get 1MB buffer (MEDIUM size class)
        assert buffer.size() == 1024 * 1024

    def test_size_class_large(self):
        """Test that 2MB requests use LARGE size class"""
        pool = mcts_py.BufferPool.instance()

        buffer = pool.acquire(2 * 1024 * 1024, use_cuda=False)  # 2MB request

        # Should get 4MB buffer (LARGE size class)
        assert buffer.size() == 4 * 1024 * 1024

    def test_buffer_reuse(self):
        """Test that released buffers are reused"""
        pool = mcts_py.BufferPool.instance()

        # Acquire and release buffer
        buffer1 = pool.acquire(4096, use_cuda=False)
        buffer1_size = buffer1.size()
        pool.release(buffer1)
        del buffer1

        # Acquire again - should reuse
        buffer2 = pool.acquire(4096, use_cuda=False)

        stats = pool.get_stats()
        # Should have 1 allocation and 1 reuse
        assert stats['total_allocated'] >= 1
        assert stats['total_reused'] >= 1

    def test_pool_clear(self):
        """Test that clear() removes all cached buffers"""
        pool = mcts_py.BufferPool.instance()

        # Acquire and release some buffers
        buffers = []
        for _ in range(5):
            buf = pool.acquire(4096, use_cuda=False)
            buffers.append(buf)

        for buf in buffers:
            pool.release(buf)
        del buffers

        stats_before = pool.get_stats()
        assert stats_before['current_pooled'] > 0

        # Clear pool
        pool.clear()

        stats_after = pool.get_stats()
        assert stats_after['current_pooled'] == 0
        assert stats_after['current_bytes'] == 0

    def test_max_buffers_per_class(self):
        """Test that pool respects max_buffers_per_class limit"""
        pool = mcts_py.BufferPool.instance()
        pool.set_max_buffers_per_class(3)

        # Acquire and release 5 buffers
        buffers = []
        for _ in range(5):
            buf = pool.acquire(4096, use_cuda=False)
            buffers.append(buf)

        for buf in buffers:
            pool.release(buf)
        del buffers

        stats = pool.get_stats()
        # Should only cache 3 (the limit)
        assert stats['current_pooled'] <= 3

    def test_pool_statistics(self):
        """Test that pool tracks statistics correctly"""
        pool = mcts_py.BufferPool.instance()

        # Allocate 3 new buffers
        buf1 = pool.acquire(4096, use_cuda=False)
        buf2 = pool.acquire(4096, use_cuda=False)
        buf3 = pool.acquire(4096, use_cuda=False)

        stats = pool.get_stats()
        assert stats['total_allocated'] >= 3

        # Release and reuse
        pool.release(buf1)
        del buf1

        buf4 = pool.acquire(4096, use_cuda=False)

        stats = pool.get_stats()
        assert stats['total_reused'] >= 1

    def test_oversized_buffer_not_pooled(self):
        """Test that buffers larger than LARGE class are not pooled"""
        pool = mcts_py.BufferPool.instance()

        # 10MB buffer (larger than LARGE = 4MB)
        buffer = pool.acquire(10 * 1024 * 1024, use_cuda=False)

        # Should get exact size requested
        assert buffer.size() == 10 * 1024 * 1024

        # Release it
        initial_pooled = pool.get_stats()['current_pooled']
        pool.release(buffer)
        del buffer

        # Should NOT be added to pool
        final_pooled = pool.get_stats()['current_pooled']
        assert final_pooled == initial_pooled


class TestThreadSafety:
    """Test thread safety of buffer pool operations"""

    def setup_method(self):
        """Clear pool before each test"""
        if HAS_MCTS_PY:
            mcts_py.BufferPool.instance().clear()

    def test_concurrent_acquire(self):
        """Test that concurrent acquire operations are thread-safe"""
        pool = mcts_py.BufferPool.instance()
        num_threads = 10
        buffers_per_thread = 100

        results: List[List] = [[] for _ in range(num_threads)]

        def worker(thread_id: int):
            for _ in range(buffers_per_thread):
                buffer = pool.acquire(4096, use_cuda=False)
                results[thread_id].append(buffer)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Verify all threads got buffers
        total_buffers = sum(len(r) for r in results)
        assert total_buffers == num_threads * buffers_per_thread

        # All buffers should be valid
        for thread_results in results:
            for buffer in thread_results:
                assert buffer.size() >= 4096
                assert buffer.ref_count() >= 1

    def test_concurrent_release(self):
        """Test that concurrent release operations are thread-safe"""
        pool = mcts_py.BufferPool.instance()
        num_threads = 10
        buffers_per_thread = 100

        # Pre-allocate buffers
        all_buffers = []
        for _ in range(num_threads * buffers_per_thread):
            buffer = pool.acquire(4096, use_cuda=False)
            all_buffers.append(buffer)

        # Distribute to threads
        thread_buffers = []
        for i in range(num_threads):
            start = i * buffers_per_thread
            end = start + buffers_per_thread
            thread_buffers.append(all_buffers[start:end])

        def worker(buffers):
            for buffer in buffers:
                pool.release(buffer)

        threads = []
        for buffers in thread_buffers:
            t = threading.Thread(target=worker, args=(buffers,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Should complete without crashes
        stats = pool.get_stats()
        assert stats['current_pooled'] > 0

    def test_concurrent_acquire_release(self):
        """Test concurrent acquire and release operations"""
        pool = mcts_py.BufferPool.instance()
        num_threads = 8
        iterations = 200

        def worker():
            for _ in range(iterations):
                buffer = pool.acquire(4096, use_cuda=False)
                # Simulate some work
                time.sleep(0.0001)
                pool.release(buffer)

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Should complete without deadlock or crashes
        stats = pool.get_stats()
        # Should have high reuse rate
        if stats['total_allocated'] > 0:
            reuse_rate = stats['total_reused'] / stats['total_allocated']
            assert reuse_rate > 0.5  # At least 50% reuse

    def test_reference_counting_thread_safety(self):
        """Test that shared_ptr reference counting is thread-safe"""
        # Note: Python's shared_ptr ref counting is thread-safe by default
        # This test verifies that concurrent access doesn't crash
        buffer = mcts_py.PinnedBuffer(4096, use_cuda=False)
        num_threads = 10
        iterations = 1000

        def access_buffer():
            for _ in range(iterations):
                # Access buffer properties
                _ = buffer.size()
                _ = buffer.data()
                _ = buffer.ref_count()

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=access_buffer)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Should complete without crashes
        assert buffer.ref_count() >= 1


class TestMemoryLeaks:
    """Test for memory leaks in buffer allocation"""

    def setup_method(self):
        """Clear pool before each test"""
        if HAS_MCTS_PY:
            mcts_py.BufferPool.instance().clear()

    def test_no_leak_with_acquire_release_cycle(self):
        """Test that repeated acquire/release doesn't leak memory"""
        pool = mcts_py.BufferPool.instance()

        initial_stats = pool.get_stats()

        # Perform many acquire/release cycles
        for _ in range(1000):
            buffer = pool.acquire(4096, use_cuda=False)
            pool.release(buffer)
            del buffer

        final_stats = pool.get_stats()

        # Current memory usage should be bounded
        assert final_stats['current_bytes'] < 1024 * 1024  # Less than 1MB

    def test_no_leak_with_reference_counting(self):
        """Test that reference counting doesn't leak"""
        for _ in range(1000):
            buffer = mcts_py.PinnedBuffer(4096, use_cuda=False)

            # Create and clear refs
            refs = [buffer for _ in range(10)]
            refs.clear()

            # Buffer should still be alive
            assert buffer.ref_count() >= 1

            # Delete buffer explicitly
            del buffer

    def test_no_leak_with_oversized_buffers(self):
        """Test that oversized buffers (not pooled) don't leak"""
        pool = mcts_py.BufferPool.instance()

        for _ in range(100):
            # 10MB buffer (not pooled)
            buffer = pool.acquire(10 * 1024 * 1024, use_cuda=False)
            pool.release(buffer)
            del buffer

        # Pool should be empty (oversized not cached)
        stats = pool.get_stats()
        assert stats['current_bytes'] < 5 * 1024 * 1024  # Less than 5MB


class TestCUDAIntegration:
    """Test CUDA-specific functionality"""

    @pytest.mark.skipif(not mcts_py.is_cuda_available() if HAS_MCTS_PY else True,
                        reason="CUDA not available")
    def test_cuda_pinned_buffer_allocation(self):
        """Test allocating CUDA pinned memory"""
        buffer = mcts_py.PinnedBuffer(1024 * 1024, use_cuda=True)

        assert buffer.is_cuda_pinned()
        assert buffer.size() == 1024 * 1024

    @pytest.mark.skipif(not mcts_py.is_cuda_available() if HAS_MCTS_PY else True,
                        reason="CUDA not available")
    def test_pool_with_cuda_pinned_memory(self):
        """Test buffer pool with CUDA pinned memory"""
        pool = mcts_py.BufferPool.instance()

        buffer = pool.acquire(64 * 1024, use_cuda=True)

        assert buffer.is_cuda_pinned()
        assert buffer.size() == 64 * 1024

    def test_fallback_to_malloc_when_cuda_unavailable(self):
        """Test that allocation falls back to malloc if CUDA unavailable"""
        # Force malloc by use_cuda=False
        buffer = mcts_py.PinnedBuffer(4096, use_cuda=False)

        assert not buffer.is_cuda_pinned()
        assert buffer.size() == 4096

    def test_is_cuda_available_returns_bool(self):
        """Test that is_cuda_available returns boolean"""
        result = mcts_py.is_cuda_available()
        assert isinstance(result, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
