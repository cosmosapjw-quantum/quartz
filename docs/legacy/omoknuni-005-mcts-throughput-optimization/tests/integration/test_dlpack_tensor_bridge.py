"""
Integration tests for DLPack tensor bridge (T007g)

Tests PyTorch integration with DLPack tensors created from game states,
including forward/backward passes, gradient computation, and training loops.
"""

import pytest
import numpy as np

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
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_ALPHAZERO, reason="alphazero_py not available")
@pytest.mark.skipif(not HAS_MCTS, reason="mcts_py not available")
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestDLPackPyTorchIntegration:
    """Integration tests for DLPack tensors with PyTorch"""

    def test_forward_pass_simple(self):
        """Test simple forward pass through a neural network"""
        # Create batch of states
        states = [alphazero_py.GomokuState() for _ in range(8)]

        # Get DLPack tensor
        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule)

        # Simple conv layer
        conv = nn.Conv2d(36, 64, kernel_size=3, padding=1)
        output = conv(features)

        assert output.shape == (8, 64, 15, 15)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_backward_pass_gradients(self):
        """Test backward pass computes gradients correctly"""
        states = [alphazero_py.GomokuState() for _ in range(4)]

        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule)

        # Create simple network
        conv = nn.Conv2d(36, 32, kernel_size=3, padding=1)
        pool = nn.AdaptiveAvgPool2d(1)
        fc = nn.Linear(32, 10)

        # Forward pass
        x = F.relu(conv(features))
        x = pool(x).squeeze(-1).squeeze(-1)
        output = fc(x)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert conv.weight.grad is not None
        assert fc.weight.grad is not None
        assert not torch.isnan(conv.weight.grad).any()

    def test_optimizer_step(self):
        """Test optimizer can update weights using DLPack tensors"""
        states = [alphazero_py.GomokuState() for _ in range(8)]

        # Simple model
        model = nn.Sequential(
            nn.Conv2d(36, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Get initial weights
        initial_weight = model[0].weight.data.clone()

        # Training step
        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule)

        output = model(features)
        loss = output.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(initial_weight, model[0].weight.data)

    def test_batch_size_1(self):
        """Test with batch size 1"""
        state = alphazero_py.GomokuState()
        capsule = mcts_py.create_batch_tensor_from_states([state])
        features = torch.from_dlpack(capsule)

        conv = nn.Conv2d(36, 16, kernel_size=3, padding=1)
        output = conv(features)

        assert output.shape == (1, 16, 15, 15)

    def test_batch_size_16(self):
        """Test with batch size 16"""
        states = [alphazero_py.GomokuState() for _ in range(16)]
        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule)

        conv = nn.Conv2d(36, 64, kernel_size=3, padding=1)
        output = conv(features)

        assert output.shape == (16, 64, 15, 15)

    def test_batch_size_32(self):
        """Test with batch size 32"""
        states = [alphazero_py.GomokuState() for _ in range(32)]
        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule)

        conv = nn.Conv2d(36, 64, kernel_size=3, padding=1)
        output = conv(features)

        assert output.shape == (32, 64, 15, 15)

    def test_batch_size_64(self):
        """Test with batch size 64"""
        states = [alphazero_py.GomokuState() for _ in range(64)]
        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule)

        conv = nn.Conv2d(36, 128, kernel_size=3, padding=1)
        output = conv(features)

        assert output.shape == (64, 128, 15, 15)

    def test_batch_size_128(self):
        """Test with batch size 128"""
        states = [alphazero_py.GomokuState() for _ in range(128)]
        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule)

        conv = nn.Conv2d(36, 128, kernel_size=3, padding=1)
        output = conv(features)

        assert output.shape == (128, 128, 15, 15)

    def test_mixed_precision_fp16(self):
        """Test mixed precision training with fp16"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")

        states = [alphazero_py.GomokuState() for _ in range(16)]
        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule).cuda()

        # Convert to fp16
        features_fp16 = features.half()

        conv = nn.Conv2d(36, 64, kernel_size=3, padding=1).cuda().half()
        output = conv(features_fp16)

        assert output.dtype == torch.float16
        assert output.shape == (16, 64, 15, 15)

    def test_training_loop_simulation(self):
        """Test simulated training loop with multiple batches"""
        model = nn.Sequential(
            nn.Conv2d(36, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 225)  # Policy head for 15x15 board
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Simulate 3 training steps
        for step in range(3):
            # Create fresh batch each step
            states = [alphazero_py.GomokuState() for _ in range(16)]
            for i, state in enumerate(states[:8]):
                if i < len(states[:8]):
                    legal_moves = state.get_legal_moves()
                    if len(legal_moves) > 0:
                        state.make_move(legal_moves[i % len(legal_moves)])

            # Get features
            capsule = mcts_py.create_batch_tensor_from_states(states)
            features = torch.from_dlpack(capsule)

            # Forward pass
            policy_logits = model(features)

            # Dummy target (uniform distribution)
            target = torch.ones_like(policy_logits) / 225.0

            # Loss and backward
            loss = F.kl_div(
                F.log_softmax(policy_logits, dim=1),
                target,
                reduction='batchmean'
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            assert loss.item() >= 0
            assert not torch.isnan(loss)

    def test_chess_forward_pass(self):
        """Test forward pass with Chess states"""
        states = [alphazero_py.ChessState() for _ in range(8)]

        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule)

        # Chess has 21 planes, 8x8 board
        conv = nn.Conv2d(21, 64, kernel_size=3, padding=1)
        output = conv(features)

        assert output.shape == (8, 64, 8, 8)

    def test_go_forward_pass(self):
        """Test forward pass with Go states"""
        states = [alphazero_py.GoState() for _ in range(8)]

        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule)

        # Go has 21 planes, 19x19 board
        conv = nn.Conv2d(21, 64, kernel_size=3, padding=1)
        output = conv(features)

        assert output.shape == (8, 64, 19, 19)

    def test_zero_copy_memory_address(self):
        """Verify zero-copy by checking memory addresses"""
        states = [alphazero_py.GomokuState() for _ in range(4)]

        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule)

        # Get data pointer
        data_ptr = features.data_ptr()

        # Creating another view should share the same memory
        features_view = features.view(4, 36, -1)
        assert features_view.data_ptr() == data_ptr

        # Modifying view should modify original
        original_sum = features.sum().item()
        features_view[0, 0, 0] += 1.0
        new_sum = features.sum().item()

        assert new_sum == original_sum + 1.0

    def test_tensor_contiguous_for_gpu(self):
        """Test that tensor is contiguous for efficient GPU transfer"""
        states = [alphazero_py.GomokuState() for _ in range(16)]
        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule)

        assert features.is_contiguous()

        # Should be able to move to GPU if available
        if torch.cuda.is_available():
            features_gpu = features.cuda()
            assert features_gpu.is_cuda
            assert features_gpu.shape == features.shape

    def test_no_intermediate_copies(self):
        """Verify no intermediate numpy arrays are created"""
        import gc
        import sys

        # Clear any cached objects
        gc.collect()

        # Count numpy arrays before
        numpy_before = sum(1 for obj in gc.get_objects()
                          if isinstance(obj, np.ndarray) and obj.size > 1000)

        # Create DLPack tensor
        states = [alphazero_py.GomokuState() for _ in range(32)]
        capsule = mcts_py.create_batch_tensor_from_states(states)
        features = torch.from_dlpack(capsule)

        # Force garbage collection
        gc.collect()

        # Count numpy arrays after
        numpy_after = sum(1 for obj in gc.get_objects()
                         if isinstance(obj, np.ndarray) and obj.size > 1000)

        # Should not have created large numpy arrays
        # (may create small ones for internal bookkeeping)
        large_arrays_created = numpy_after - numpy_before
        assert large_arrays_created <= 1, f"Created {large_arrays_created} large numpy arrays"
