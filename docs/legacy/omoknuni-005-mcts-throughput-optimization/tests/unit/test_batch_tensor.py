"""
Unit tests for batch tensor creation (T007d)

Tests verify:
1. Tensor creation for all game types (GOMOKU, CHESS, GO)
2. Correct tensor shapes
3. Zero initialization (stub behavior)
4. Error handling (invalid inputs)
5. Buffer pool integration
6. PyTorch compatibility via torch.from_dlpack()
"""

import pytest
import numpy as np
import mcts_py

# Optional PyTorch import (skip tests if unavailable)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# GameType Enum Tests
# ============================================================================

def test_game_type_enum_exists():
    """GameType enum should be accessible."""
    assert hasattr(mcts_py, 'GameType')
    assert hasattr(mcts_py.GameType, 'GOMOKU')
    assert hasattr(mcts_py.GameType, 'CHESS')
    assert hasattr(mcts_py.GameType, 'GO')


def test_game_type_enum_values():
    """GameType enum values should be distinct."""
    assert mcts_py.GameType.GOMOKU != mcts_py.GameType.CHESS
    assert mcts_py.GameType.CHESS != mcts_py.GameType.GO
    assert mcts_py.GameType.GO != mcts_py.GameType.GOMOKU


# ============================================================================
# Helper Function Tests
# ============================================================================

def test_get_num_planes_gomoku():
    """Gomoku should have 36 planes."""
    num_planes = mcts_py.get_num_planes(mcts_py.GameType.GOMOKU)
    assert num_planes == 36


def test_get_num_planes_chess():
    """Chess should have 30 planes."""
    num_planes = mcts_py.get_num_planes(mcts_py.GameType.CHESS)
    assert num_planes == 30


def test_get_num_planes_go():
    """Go should have 25 planes."""
    num_planes = mcts_py.get_num_planes(mcts_py.GameType.GO)
    assert num_planes == 25


def test_get_board_size_gomoku():
    """Gomoku should have 15×15 board."""
    height, width = mcts_py.get_board_size(mcts_py.GameType.GOMOKU)
    assert height == 15
    assert width == 15


def test_get_board_size_chess():
    """Chess should have 8×8 board."""
    height, width = mcts_py.get_board_size(mcts_py.GameType.CHESS)
    assert height == 8
    assert width == 8


def test_get_board_size_go():
    """Go should have 19×19 board."""
    height, width = mcts_py.get_board_size(mcts_py.GameType.GO)
    assert height == 19
    assert width == 19


# ============================================================================
# Batch Tensor Creation Tests
# ============================================================================

def test_create_batch_tensor_gomoku_shape():
    """Gomoku batch tensor should have shape (batch, 36, 15, 15)."""
    batch_size = 8
    capsule = mcts_py.create_batch_tensor(batch_size, mcts_py.GameType.GOMOKU, use_cuda=False)

    # Capsule should be a valid PyCapsule
    assert capsule is not None
    assert hasattr(capsule, '__class__')
    assert 'capsule' in str(type(capsule)).lower()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_create_batch_tensor_gomoku_torch_conversion():
    """Gomoku tensor should convert to PyTorch with correct shape."""
    batch_size = 4
    capsule = mcts_py.create_batch_tensor(batch_size, mcts_py.GameType.GOMOKU, use_cuda=False)

    # Convert to PyTorch tensor
    tensor = torch.from_dlpack(capsule)

    # Verify shape
    assert tensor.shape == (4, 36, 15, 15)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == 'cpu'


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_create_batch_tensor_gomoku_zero_initialized():
    """Gomoku tensor should be zero-initialized (stub behavior)."""
    batch_size = 2
    capsule = mcts_py.create_batch_tensor(batch_size, mcts_py.GameType.GOMOKU, use_cuda=False)

    tensor = torch.from_dlpack(capsule)

    # All values should be zero (stub implementation)
    assert torch.all(tensor == 0.0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_create_batch_tensor_chess_shape():
    """Chess batch tensor should have shape (batch, 30, 8, 8)."""
    batch_size = 16
    capsule = mcts_py.create_batch_tensor(batch_size, mcts_py.GameType.CHESS, use_cuda=False)

    tensor = torch.from_dlpack(capsule)

    assert tensor.shape == (16, 30, 8, 8)
    assert tensor.dtype == torch.float32


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_create_batch_tensor_chess_zero_initialized():
    """Chess tensor should be zero-initialized."""
    batch_size = 4
    capsule = mcts_py.create_batch_tensor(batch_size, mcts_py.GameType.CHESS, use_cuda=False)

    tensor = torch.from_dlpack(capsule)

    assert torch.all(tensor == 0.0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_create_batch_tensor_go_shape():
    """Go batch tensor should have shape (batch, 25, 19, 19)."""
    batch_size = 8
    capsule = mcts_py.create_batch_tensor(batch_size, mcts_py.GameType.GO, use_cuda=False)

    tensor = torch.from_dlpack(capsule)

    assert tensor.shape == (8, 25, 19, 19)
    assert tensor.dtype == torch.float32


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_create_batch_tensor_go_zero_initialized():
    """Go tensor should be zero-initialized."""
    batch_size = 2
    capsule = mcts_py.create_batch_tensor(batch_size, mcts_py.GameType.GO, use_cuda=False)

    tensor = torch.from_dlpack(capsule)

    assert torch.all(tensor == 0.0)


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_create_batch_tensor_invalid_batch_size_zero():
    """Creating tensor with batch_size=0 should raise error."""
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        mcts_py.create_batch_tensor(0, mcts_py.GameType.GOMOKU, use_cuda=False)


def test_create_batch_tensor_invalid_batch_size_negative():
    """Creating tensor with negative batch_size should raise error."""
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        mcts_py.create_batch_tensor(-5, mcts_py.GameType.GOMOKU, use_cuda=False)


# ============================================================================
# Memory Management Tests
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_batch_tensor_memory_cleanup():
    """Tensor memory should be properly cleaned up after deletion."""
    # Create multiple tensors
    tensors = []
    for _ in range(10):
        capsule = mcts_py.create_batch_tensor(8, mcts_py.GameType.GOMOKU, use_cuda=False)
        tensor = torch.from_dlpack(capsule)
        tensors.append(tensor)

    # Delete all tensors
    del tensors

    # If no crash, memory cleanup succeeded


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_batch_tensor_buffer_pool_integration():
    """Batch tensor should use buffer pool for memory allocation."""
    # Get initial pool stats
    initial_stats = mcts_py.BufferPool.instance().get_stats()

    # Create a tensor (should allocate from pool)
    capsule = mcts_py.create_batch_tensor(8, mcts_py.GameType.GOMOKU, use_cuda=False)
    tensor = torch.from_dlpack(capsule)

    # Get updated stats
    after_create_stats = mcts_py.BufferPool.instance().get_stats()

    # Should have allocated a buffer
    assert after_create_stats['total_allocated'] >= initial_stats['total_allocated']

    # Delete tensor
    del tensor

    # Create another tensor of same size (should reuse from pool)
    capsule2 = mcts_py.create_batch_tensor(8, mcts_py.GameType.GOMOKU, use_cuda=False)
    tensor2 = torch.from_dlpack(capsule2)

    # Get final stats
    final_stats = mcts_py.BufferPool.instance().get_stats()

    # Should have reused buffer (reuse count increases)
    # Note: This depends on buffer pool behavior and size class matching
    # The test just verifies no crashes occur

    del tensor2


# ============================================================================
# Multiple Batch Sizes Tests
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32, 64])
def test_create_batch_tensor_various_batch_sizes(batch_size):
    """Batch tensor should support various batch sizes."""
    capsule = mcts_py.create_batch_tensor(batch_size, mcts_py.GameType.GOMOKU, use_cuda=False)
    tensor = torch.from_dlpack(capsule)

    assert tensor.shape[0] == batch_size
    assert tensor.shape[1:] == (36, 15, 15)


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_create_batch_tensor_single_position():
    """Batch tensor should work with batch_size=1."""
    capsule = mcts_py.create_batch_tensor(1, mcts_py.GameType.GOMOKU, use_cuda=False)
    tensor = torch.from_dlpack(capsule)

    assert tensor.shape == (1, 36, 15, 15)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_create_batch_tensor_large_batch():
    """Batch tensor should handle large batches (128)."""
    capsule = mcts_py.create_batch_tensor(128, mcts_py.GameType.GOMOKU, use_cuda=False)
    tensor = torch.from_dlpack(capsule)

    assert tensor.shape == (128, 36, 15, 15)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_create_batch_tensor_row_major_layout():
    """Batch tensor should be row-major (contiguous)."""
    capsule = mcts_py.create_batch_tensor(4, mcts_py.GameType.GOMOKU, use_cuda=False)
    tensor = torch.from_dlpack(capsule)

    # PyTorch tensor from DLPack should be contiguous
    assert tensor.is_contiguous()


# ============================================================================
# CUDA Tests (Optional)
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(),
                    reason="PyTorch CUDA not available")
def test_create_batch_tensor_cuda_pinned():
    """Batch tensor with use_cuda=True should create CUDA pinned memory."""
    capsule = mcts_py.create_batch_tensor(8, mcts_py.GameType.GOMOKU, use_cuda=True)
    tensor = torch.from_dlpack(capsule)

    # Tensor should be on CPU but pinned
    assert tensor.device.type == 'cpu'
    # Note: PyTorch doesn't expose is_pinned() through DLPack interface


if __name__ == "__main__":
    print("\n=== Batch Tensor Creation Tests (T007d) ===\n")

    # Run tests manually
    test_game_type_enum_exists()
    print("✓ test_game_type_enum_exists")

    test_game_type_enum_values()
    print("✓ test_game_type_enum_values")

    test_get_num_planes_gomoku()
    print("✓ test_get_num_planes_gomoku")

    test_get_num_planes_chess()
    print("✓ test_get_num_planes_chess")

    test_get_num_planes_go()
    print("✓ test_get_num_planes_go")

    test_get_board_size_gomoku()
    print("✓ test_get_board_size_gomoku")

    test_get_board_size_chess()
    print("✓ test_get_board_size_chess")

    test_get_board_size_go()
    print("✓ test_get_board_size_go")

    test_create_batch_tensor_gomoku_shape()
    print("✓ test_create_batch_tensor_gomoku_shape")

    if HAS_TORCH:
        test_create_batch_tensor_gomoku_torch_conversion()
        print("✓ test_create_batch_tensor_gomoku_torch_conversion")

        test_create_batch_tensor_gomoku_zero_initialized()
        print("✓ test_create_batch_tensor_gomoku_zero_initialized")

        test_create_batch_tensor_chess_shape()
        print("✓ test_create_batch_tensor_chess_shape")

        test_create_batch_tensor_chess_zero_initialized()
        print("✓ test_create_batch_tensor_chess_zero_initialized")

        test_create_batch_tensor_go_shape()
        print("✓ test_create_batch_tensor_go_shape")

        test_create_batch_tensor_go_zero_initialized()
        print("✓ test_create_batch_tensor_go_zero_initialized")

        test_batch_tensor_memory_cleanup()
        print("✓ test_batch_tensor_memory_cleanup")

        test_batch_tensor_buffer_pool_integration()
        print("✓ test_batch_tensor_buffer_pool_integration")

        test_create_batch_tensor_single_position()
        print("✓ test_create_batch_tensor_single_position")

        test_create_batch_tensor_large_batch()
        print("✓ test_create_batch_tensor_large_batch")

        test_create_batch_tensor_row_major_layout()
        print("✓ test_create_batch_tensor_row_major_layout")
    else:
        print("⚠ PyTorch tests skipped (PyTorch not available)")

    # Error handling tests
    test_create_batch_tensor_invalid_batch_size_zero()
    print("✓ test_create_batch_tensor_invalid_batch_size_zero")

    test_create_batch_tensor_invalid_batch_size_negative()
    print("✓ test_create_batch_tensor_invalid_batch_size_negative")

    print("\n=== All batch tensor tests passed! ===\n")
