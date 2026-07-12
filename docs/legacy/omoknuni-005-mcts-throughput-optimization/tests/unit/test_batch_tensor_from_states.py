"""
Unit tests for create_batch_tensor_from_states Python binding (T007f)

Tests the Python interface for creating batch tensors with real feature extraction
from game states. Validates torch.from_dlpack() integration and error handling.
"""

import pytest
import numpy as np

# Import game modules
try:
    import alphazero_py
    HAS_ALPHAZERO = True
except ImportError:
    HAS_ALPHAZERO = False
try:
    import mcts_py
    HAS_MCTS = True
except ImportError:
    HAS_MCTS = False


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_ALPHAZERO, reason="alphazero_py not available")
@pytest.mark.skipif(not HAS_MCTS, reason="alphazero_py not available")
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestCreateBatchTensorFromStates:
    """Test suite for create_batch_tensor_from_states binding"""

    def test_gomoku_single_state(self):
        """Test creating batch tensor from single Gomoku state"""
        state = alphazero_py.GomokuState()
        capsule = mcts_py.create_batch_tensor_from_states([state])

        # Convert to PyTorch tensor
        tensor = torch.from_dlpack(capsule)

        # Validate shape
        assert tensor.shape == (1, 36, 15, 15), f"Expected (1, 36, 15, 15), got {tensor.shape}"
        assert tensor.dtype == torch.float32

        # Initial state should have some non-zero features (player indicator)
        assert tensor.sum() > 0, "Tensor should have non-zero features"

    def test_gomoku_batch_with_moves(self):
        """Test creating batch tensor from multiple Gomoku states with moves"""
        batch_size = 8
        states = [alphazero_py.GomokuState() for _ in range(batch_size)]

        # Make different moves in each state
        for i, state in enumerate(states):
            move = 112 + i  # Center and nearby moves
            state.make_move(move)

        capsule = mcts_py.create_batch_tensor_from_states(states)
        tensor = torch.from_dlpack(capsule)

        # Validate shape
        assert tensor.shape == (batch_size, 36, 15, 15)

        # Each state should have different features
        # Check that at least some states differ
        state_sums = tensor.sum(dim=(1, 2, 3))
        assert len(torch.unique(state_sums)) > 1, "States should have different features"

    def test_chess_single_state(self):
        """Test creating batch tensor from single Chess state"""
        state = alphazero_py.ChessState()
        capsule = mcts_py.create_batch_tensor_from_states([state])

        tensor = torch.from_dlpack(capsule)

        # Validate shape (21 planes for Chess)
        assert tensor.shape == (1, 21, 8, 8)
        assert tensor.dtype == torch.float32

        # Initial chess position should have non-zero features (pieces)
        assert tensor.sum() > 0

    def test_go_single_state(self):
        """Test creating batch tensor from single Go state"""
        state = alphazero_py.GoState()
        capsule = mcts_py.create_batch_tensor_from_states([state])

        tensor = torch.from_dlpack(capsule)

        # Validate shape (21 planes for Go)
        assert tensor.shape == (1, 21, 19, 19)
        assert tensor.dtype == torch.float32

        # Empty Go board should still have player indicator features
        assert tensor.sum() > 0

    def test_batch_size_32(self):
        """Test standard batch size of 32"""
        batch_size = 32
        states = [alphazero_py.GomokuState() for _ in range(batch_size)]

        capsule = mcts_py.create_batch_tensor_from_states(states)
        tensor = torch.from_dlpack(capsule)

        assert tensor.shape == (32, 36, 15, 15)
        assert tensor.is_contiguous()

    def test_batch_size_64(self):
        """Test large batch size"""
        batch_size = 64
        states = [alphazero_py.GomokuState() for _ in range(batch_size)]

        capsule = mcts_py.create_batch_tensor_from_states(states)
        tensor = torch.from_dlpack(capsule)

        assert tensor.shape == (64, 36, 15, 15)

    def test_tensor_is_contiguous(self):
        """Test that tensor is memory contiguous"""
        states = [alphazero_py.GomokuState() for _ in range(4)]
        capsule = mcts_py.create_batch_tensor_from_states(states)
        tensor = torch.from_dlpack(capsule)

        assert tensor.is_contiguous(), "Tensor should be contiguous for GPU transfer"

    def test_feature_extraction_correctness(self):
        """Test that features match extract_features_to_buffer"""
        state = alphazero_py.GomokuState()
        state.make_move(112)  # Center move

        # Extract via batch tensor
        capsule = mcts_py.create_batch_tensor_from_states([state])
        batch_tensor = torch.from_dlpack(capsule)

        # Extract via direct method
        buffer = np.zeros(36 * 15 * 15, dtype=np.float32)
        state.extract_features_to_buffer(buffer)
        direct_features = buffer.reshape(36, 15, 15)

        # Compare
        np.testing.assert_allclose(
            batch_tensor[0].cpu().numpy(),
            direct_features,
            rtol=1e-6,
            err_msg="Batch tensor should match direct extraction"
        )

    # Error handling tests

    def test_error_empty_list(self):
        """Test error handling for empty states list"""
        with pytest.raises(Exception, match="empty"):
            mcts_py.create_batch_tensor_from_states([])

    def test_error_mixed_game_types(self):
        """Test error handling for mixed game types"""
        states = [
            alphazero_py.GomokuState(),
            alphazero_py.ChessState()
        ]

        with pytest.raises(Exception, match="same game type"):
            mcts_py.create_batch_tensor_from_states(states)

    def test_error_invalid_state_type(self):
        """Test error handling for invalid state objects"""
        with pytest.raises(Exception, match="not a valid game state|inherit from IGameState"):
            mcts_py.create_batch_tensor_from_states(["not_a_state"])

    def test_error_none_in_list(self):
        """Test error handling for None in states list"""
        with pytest.raises(Exception):
            mcts_py.create_batch_tensor_from_states([None])

    # PyTorch integration tests

    def test_torch_gradient_computation(self):
        """Test that tensor can be used in PyTorch computations"""
        states = [alphazero_py.GomokuState() for _ in range(4)]
        capsule = mcts_py.create_batch_tensor_from_states(states)
        tensor = torch.from_dlpack(capsule)

        # Tensor should be on CPU
        assert tensor.device.type == "cpu"

        # Should be able to compute mean
        mean = tensor.mean()
        assert mean >= 0

    def test_torch_device_transfer(self):
        """Test that tensor can be transferred to GPU if available"""
        states = [alphazero_py.GomokuState() for _ in range(4)]
        capsule = mcts_py.create_batch_tensor_from_states(states)
        tensor = torch.from_dlpack(capsule)

        if torch.cuda.is_available():
            gpu_tensor = tensor.cuda()
            assert gpu_tensor.device.type == "cuda"
            assert gpu_tensor.shape == tensor.shape
        else:
            # Just verify CPU tensor works
            assert tensor.device.type == "cpu"

    def test_batch_diversity(self):
        """Test that different states produce different features"""
        states = []
        np.random.seed(42)

        # Create 16 states with random moves
        for _ in range(16):
            state = alphazero_py.GomokuState()
            # Make 1-3 random moves
            num_moves = np.random.randint(1, 4)
            for _ in range(num_moves):
                legal_moves = state.get_legal_moves()
                if len(legal_moves) > 0:
                    move = legal_moves[np.random.randint(len(legal_moves))]
                    state.make_move(move)
            states.append(state)

        capsule = mcts_py.create_batch_tensor_from_states(states)
        tensor = torch.from_dlpack(capsule)

        # Compute per-state sums
        state_sums = tensor.sum(dim=(1, 2, 3))

        # Should have multiple unique values (different states)
        unique_sums = torch.unique(state_sums)
        assert len(unique_sums) > 4, f"Expected diverse states, got {len(unique_sums)} unique sums"

    def test_use_cuda_false(self):
        """Test with use_cuda=False (default)"""
        states = [alphazero_py.GomokuState() for _ in range(4)]
        capsule = mcts_py.create_batch_tensor_from_states(states, use_cuda=False)
        tensor = torch.from_dlpack(capsule)

        assert tensor.shape == (4, 36, 15, 15)
        # Tensor should be on CPU when use_cuda=False
        assert tensor.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_use_cuda_true(self):
        """Test with use_cuda=True (pinned memory)"""
        states = [alphazero_py.GomokuState() for _ in range(4)]
        capsule = mcts_py.create_batch_tensor_from_states(states, use_cuda=True)
        tensor = torch.from_dlpack(capsule)

        assert tensor.shape == (4, 36, 15, 15)
        # Pinned memory tensor should be on CPU but pinned
        assert tensor.device.type == "cpu"
        # Note: DLPack doesn't expose pinned memory status directly

    def test_api_exists(self):
        """Test that the function is properly exposed"""
        assert hasattr(mcts_py, 'create_batch_tensor_from_states')

        # Check it's callable
        assert callable(mcts_py.create_batch_tensor_from_states)

    def test_docstring_exists(self):
        """Test that function has proper documentation"""
        func = mcts_py.create_batch_tensor_from_states
        assert func.__doc__ is not None
        assert "DLPack" in func.__doc__
        assert "states" in func.__doc__
