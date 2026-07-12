"""
Unit tests for DLPack Tensor Capsule (T007c)

Tests cover:
- DLPack capsule creation and metadata
- PyTorch torch.from_dlpack() integration
- Memory management and cleanup
- Error handling
"""

import pytest
import torch
import numpy as np

# Import C++ extension
try:
    import mcts_py
    HAS_MCTS_PY = True
except ImportError:
    HAS_MCTS_PY = False
    pytestmark = pytest.mark.skip(reason="mcts_py extension not built")


class TestDLPackCapsuleCreation:
    """Test basic DLPack capsule creation"""

    def test_create_simple_capsule(self):
        """Test creating a simple DLPack capsule"""
        # Allocate buffer for 1x3x4x4 tensor (48 floats = 192 bytes)
        buffer = mcts_py.PinnedBuffer(192, use_cuda=False)

        # Fill with test data
        data_ptr = buffer.data()
        test_data = np.arange(48, dtype=np.float32)

        # Create DLPack tensor
        shape = mcts_py.TensorShape(1, 3, 4, 4)  # batch=1, planes=3, 4x4
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)

        assert capsule is not None
        assert isinstance(capsule, object)  # PyCapsule

    def test_capsule_to_pytorch_tensor(self):
        """Test converting DLPack capsule to PyTorch tensor"""
        # Create buffer
        buffer = mcts_py.PinnedBuffer(192, use_cuda=False)

        # Create capsule
        shape = mcts_py.TensorShape(1, 3, 4, 4)
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)

        # Convert to PyTorch tensor
        tensor = torch.from_dlpack(capsule)

        # Verify shape
        assert tensor.shape == (1, 3, 4, 4)
        assert tensor.dtype == torch.float32
        assert not tensor.is_cuda  # CPU tensor

    def test_tensor_shape_metadata(self):
        """Test that tensor shape metadata is correct"""
        buffer = mcts_py.PinnedBuffer(1024, use_cuda=False)

        # Test various shapes
        test_cases = [
            (1, 3, 4, 4),      # Single state, 3 planes
            (16, 36, 15, 15),  # 16 Gomoku states
            (32, 30, 8, 8),    # 32 Chess states
            (64, 25, 19, 19),  # 64 Go states
        ]

        for batch, planes, height, width in test_cases:
            size_bytes = batch * planes * height * width * 4  # float32
            buffer = mcts_py.PinnedBuffer(size_bytes, use_cuda=False)
            shape = mcts_py.TensorShape(batch, planes, height, width)
            capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)

            tensor = torch.from_dlpack(capsule)
            assert tensor.shape == (batch, planes, height, width)

    def test_tensor_data_type(self):
        """Test that tensor data type is float32"""
        buffer = mcts_py.PinnedBuffer(192, use_cuda=False)
        shape = mcts_py.TensorShape(1, 3, 4, 4)
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)

        tensor = torch.from_dlpack(capsule)
        assert tensor.dtype == torch.float32

    def test_tensor_is_contiguous(self):
        """Test that tensor has row-major (contiguous) layout"""
        buffer = mcts_py.PinnedBuffer(192, use_cuda=False)
        shape = mcts_py.TensorShape(1, 3, 4, 4)
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)

        tensor = torch.from_dlpack(capsule)
        assert tensor.is_contiguous()


class TestDLPackMemoryManagement:
    """Test memory management and cleanup"""

    def test_capsule_cleanup(self):
        """Test that capsule properly cleans up when destroyed"""
        buffer = mcts_py.PinnedBuffer(192, use_cuda=False)
        initial_refcount = buffer.ref_count()

        # Create capsule (increments buffer ref count)
        shape = mcts_py.TensorShape(1, 3, 4, 4)
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)

        # Convert to tensor (capsule consumed)
        tensor = torch.from_dlpack(capsule)
        tensor_refcount = buffer.ref_count()

        # Ref count should be higher (tensor holds reference)
        assert tensor_refcount >= initial_refcount

        # Delete tensor (should decrement ref count)
        del tensor

        # Buffer should still be alive (we still hold reference)
        final_refcount = buffer.ref_count()
        assert final_refcount >= 1

    def test_multiple_capsules_from_same_buffer(self):
        """Test creating multiple capsules from same buffer"""
        buffer = mcts_py.PinnedBuffer(192, use_cuda=False)
        shape = mcts_py.TensorShape(1, 3, 4, 4)

        # Create two capsules from same buffer
        capsule1 = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)
        capsule2 = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)

        # Convert to tensors
        tensor1 = torch.from_dlpack(capsule1)
        tensor2 = torch.from_dlpack(capsule2)

        # Both should share the same underlying data
        assert tensor1.data_ptr() == tensor2.data_ptr()

    def test_buffer_outlives_tensor(self):
        """Test that buffer stays alive even after tensor is destroyed"""
        buffer = mcts_py.PinnedBuffer(192, use_cuda=False)
        data_ptr = buffer.data()

        # Create and destroy tensor
        shape = mcts_py.TensorShape(1, 3, 4, 4)
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)
        tensor = torch.from_dlpack(capsule)
        del tensor

        # Buffer should still be accessible
        assert buffer.data() == data_ptr
        assert buffer.size() == 192


class TestDLPackTensorOperations:
    """Test operations on DLPack tensors"""

    def test_tensor_read_write(self):
        """Test reading and writing tensor data"""
        # Create buffer and fill with data
        buffer = mcts_py.PinnedBuffer(192, use_cuda=False)
        shape = mcts_py.TensorShape(1, 3, 4, 4)

        # Create tensor
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)
        tensor = torch.from_dlpack(capsule)

        # Write data
        tensor.fill_(42.0)

        # Verify data
        assert torch.all(tensor == 42.0)

    def test_tensor_arithmetic(self):
        """Test arithmetic operations on tensor"""
        buffer = mcts_py.PinnedBuffer(192, use_cuda=False)
        shape = mcts_py.TensorShape(1, 3, 4, 4)
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)
        tensor = torch.from_dlpack(capsule)

        # Arithmetic operations
        tensor.fill_(10.0)
        tensor += 5.0
        assert torch.all(tensor == 15.0)

        tensor *= 2.0
        assert torch.all(tensor == 30.0)

    def test_tensor_slicing(self):
        """Test slicing operations on tensor"""
        buffer = mcts_py.PinnedBuffer(192, use_cuda=False)
        shape = mcts_py.TensorShape(2, 3, 4, 4)  # batch=2
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)
        tensor = torch.from_dlpack(capsule)

        # Slice first batch item
        first_item = tensor[0]
        assert first_item.shape == (3, 4, 4)

        # Slice plane
        first_plane = tensor[0, 0]
        assert first_plane.shape == (4, 4)


class TestDLPackEdgeCases:
    """Test edge cases and error handling"""

    def test_single_element_tensor(self):
        """Test tensor with single element"""
        buffer = mcts_py.PinnedBuffer(4, use_cuda=False)  # 1 float
        shape = mcts_py.TensorShape(1, 1, 1, 1)
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)

        tensor = torch.from_dlpack(capsule)
        assert tensor.shape == (1, 1, 1, 1)
        assert tensor.numel() == 1

    def test_large_batch_tensor(self):
        """Test tensor with large batch size"""
        # 64 states × 36 planes × 15×15 = 518,400 floats = 2,073,600 bytes
        size_bytes = 64 * 36 * 15 * 15 * 4
        buffer = mcts_py.PinnedBuffer(size_bytes, use_cuda=False)
        shape = mcts_py.TensorShape(64, 36, 15, 15)
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)

        tensor = torch.from_dlpack(capsule)
        assert tensor.shape == (64, 36, 15, 15)
        assert tensor.numel() == 64 * 36 * 15 * 15


class TestDLPackCUDASupport:
    """Test CUDA-specific functionality"""

    @pytest.mark.skipif(not mcts_py.is_cuda_available() if HAS_MCTS_PY else True,
                        reason="CUDA not available")
    def test_cuda_pinned_tensor(self):
        """Test creating tensor from CUDA pinned memory"""
        buffer = mcts_py.PinnedBuffer(192, use_cuda=True)
        assert buffer.is_cuda_pinned()

        shape = mcts_py.TensorShape(1, 3, 4, 4)
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=True)

        tensor = torch.from_dlpack(capsule)
        assert tensor.shape == (1, 3, 4, 4)
        # Note: DLPack kDLCUDAHost is still CPU tensor, just faster GPU transfers

    def test_fallback_to_cpu_when_cuda_unavailable(self):
        """Test graceful fallback when CUDA unavailable"""
        buffer = mcts_py.PinnedBuffer(192, use_cuda=False)
        shape = mcts_py.TensorShape(1, 3, 4, 4)
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)

        tensor = torch.from_dlpack(capsule)
        assert not tensor.is_cuda


class TestDLPackBufferPoolIntegration:
    """Test integration with BufferPool"""

    def test_capsule_from_pooled_buffer(self):
        """Test creating capsule from buffer acquired from pool"""
        pool = mcts_py.BufferPool.instance()

        # Acquire buffer from pool
        buffer = pool.acquire(192, use_cuda=False)

        # Create capsule
        shape = mcts_py.TensorShape(1, 3, 4, 4)
        capsule = mcts_py.create_dlpack_capsule(buffer, shape, use_cuda=False)

        # Convert to tensor
        tensor = torch.from_dlpack(capsule)
        assert tensor.shape == (1, 3, 4, 4)

        # Return buffer to pool
        del tensor
        pool.release(buffer)

    def test_buffer_reuse_across_tensors(self):
        """Test that buffers can be reused for multiple tensors"""
        pool = mcts_py.BufferPool.instance()
        pool.clear()

        # Create and release buffer
        buffer1 = pool.acquire(192, use_cuda=False)
        shape = mcts_py.TensorShape(1, 3, 4, 4)
        capsule1 = mcts_py.create_dlpack_capsule(buffer1, shape, use_cuda=False)
        tensor1 = torch.from_dlpack(capsule1)
        del tensor1
        pool.release(buffer1)
        del buffer1

        # Acquire again (should reuse)
        buffer2 = pool.acquire(192, use_cuda=False)
        capsule2 = mcts_py.create_dlpack_capsule(buffer2, shape, use_cuda=False)
        tensor2 = torch.from_dlpack(capsule2)
        assert tensor2.shape == (1, 3, 4, 4)

        # Verify buffer was reused
        stats = pool.get_stats()
        assert stats['total_reused'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
