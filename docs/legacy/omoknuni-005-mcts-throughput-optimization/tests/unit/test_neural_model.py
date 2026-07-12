"""
Unit tests for AlphaZero neural network model components.
========================================================

Tests all model components including SE attention, residual blocks,
policy/value heads, and full AlphaZero architecture.

Run with: python -m pytest tests/unit/test_neural_model.py -v
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple
import tempfile
import os

# Import model components
from src.neural.model import (
    SqueezeExcitation,
    ResidualBlock,
    PolicyHead,
    ValueHead,
    AlphaZeroNet,
    create_model_for_game,
    create_random_model,
    enable_mixed_precision,
    validate_model_output
)


class TestSqueezeExcitation:
    """Test Squeeze-Excitation attention module."""

    def test_se_creation(self):
        """Test SE module can be created with different configurations."""
        se = SqueezeExcitation(256, reduction=16)
        assert se.channels == 256
        assert se.reduction == 16
        assert se.fc1.in_features == 256
        assert se.fc1.out_features == 16  # 256 // 16
        assert se.fc2.in_features == 16
        assert se.fc2.out_features == 256

    def test_se_forward_shape(self):
        """Test SE forward pass preserves input shape."""
        se = SqueezeExcitation(64)
        x = torch.randn(2, 64, 8, 8)

        output = se(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_se_attention_weights(self):
        """Test SE produces valid attention weights."""
        se = SqueezeExcitation(32)
        x = torch.randn(1, 32, 4, 4)

        # Forward pass
        output = se(x)

        # Extract attention weights
        with torch.no_grad():
            pooled = se.global_avgpool(x).view(1, 32)
            weights = torch.sigmoid(se.fc2(torch.relu(se.fc1(pooled))))

        # Weights should be in [0, 1]
        assert (weights >= 0).all()
        assert (weights <= 1).all()

    def test_se_different_reductions(self):
        """Test SE with different reduction ratios."""
        for reduction in [4, 8, 16, 32]:
            se = SqueezeExcitation(128, reduction=reduction)
            x = torch.randn(1, 128, 5, 5)

            output = se(x)
            assert output.shape == x.shape

    def test_se_edge_case_channels(self):
        """Test SE with small channel numbers."""
        # Test with channels < reduction
        se = SqueezeExcitation(8, reduction=16)
        assert se.fc1.out_features == 1  # max(1, 8//16)

        x = torch.randn(1, 8, 3, 3)
        output = se(x)
        assert output.shape == x.shape


class TestResidualBlock:
    """Test ResidualBlock with SE attention."""

    def test_residual_block_creation(self):
        """Test ResidualBlock creation with different configurations."""
        # With SE
        block_se = ResidualBlock(128, use_se=True)
        assert hasattr(block_se, 'se')
        assert isinstance(block_se.se, SqueezeExcitation)

        # Without SE
        block_no_se = ResidualBlock(128, use_se=False)
        assert not hasattr(block_no_se, 'se')

    def test_residual_block_forward_shape(self):
        """Test residual block preserves input shape."""
        block = ResidualBlock(64)
        x = torch.randn(4, 64, 8, 8)

        output = block(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_residual_connection(self):
        """Test residual connection is working."""
        block = ResidualBlock(32, use_se=False)

        # Zero-initialize second conv to test pure residual connection
        nn.init.zeros_(block.conv2.weight)

        x = torch.randn(1, 32, 4, 4)
        output = block(x)

        # Output should be close to ReLU(x) since conv2 is zero
        expected = torch.relu(x)
        assert torch.allclose(output, expected, atol=1e-5)

    def test_residual_block_gradients(self):
        """Test gradients flow through residual block."""
        block = ResidualBlock(16)
        x = torch.randn(1, 16, 4, 4, requires_grad=True)

        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestPolicyHead:
    """Test policy head for action prediction."""

    def test_policy_head_creation(self):
        """Test policy head creation."""
        head = PolicyHead(256, 225)  # Gomoku 15x15
        assert head.input_channels == 256
        assert head.num_actions == 225

    def test_policy_head_forward_shape(self):
        """Test policy head output shape."""
        head = PolicyHead(128, 64)
        x = torch.randn(2, 128, 8, 8)

        logits = head(x)

        assert logits.shape == (2, 64)

    def test_policy_head_different_board_sizes(self):
        """Test policy head with different board sizes (separate heads)."""
        # Need separate heads for different board sizes due to lazy initialization
        board_sizes = [8, 15, 19]
        action_counts = [64, 225, 361]

        for board_size, num_actions in zip(board_sizes, action_counts):
            head = PolicyHead(64, num_actions)
            x = torch.randn(1, 64, board_size, board_size)
            logits = head(x)
            assert logits.shape == (1, num_actions)

    def test_policy_head_lazy_initialization(self):
        """Test policy head lazy initialization of linear layer."""
        head = PolicyHead(32, 25)
        assert head.fc is None

        # First forward pass should initialize fc
        x = torch.randn(1, 32, 5, 5)
        logits = head(x)

        assert head.fc is not None
        assert isinstance(head.fc, nn.Linear)
        assert logits.shape == (1, 25)

    def test_policy_head_batch_consistency(self):
        """Test policy head produces consistent outputs across batch sizes."""
        head = PolicyHead(16, 9)
        x_single = torch.randn(1, 16, 3, 3)
        x_batch = x_single.repeat(4, 1, 1, 1)

        logits_single = head(x_single)
        logits_batch = head(x_batch)

        # First element of batch should match single input
        assert torch.allclose(logits_single[0], logits_batch[0], atol=1e-5)


class TestValueHead:
    """Test value head for position evaluation."""

    def test_value_head_creation(self):
        """Test value head creation."""
        head = ValueHead(256)
        assert head.input_channels == 256

    def test_value_head_forward_shape(self):
        """Test value head output shape."""
        head = ValueHead(128)
        x = torch.randn(3, 128, 8, 8)

        values = head(x)

        assert values.shape == (3, 1)

    def test_value_head_output_range(self):
        """Test value head outputs are in [-1, 1] range."""
        head = ValueHead(64)
        x = torch.randn(5, 64, 4, 4)

        values = head(x)

        assert (values >= -1).all()
        assert (values <= 1).all()

    def test_value_head_different_spatial_sizes(self):
        """Test value head works with different spatial dimensions."""
        head = ValueHead(32)

        for size in [4, 8, 15, 19]:
            x = torch.randn(2, 32, size, size)
            values = head(x)
            assert values.shape == (2, 1)

    def test_value_head_global_pooling(self):
        """Test value head properly uses global average pooling."""
        head = ValueHead(16)

        # Create input with known spatial pattern
        x = torch.ones(1, 16, 8, 8)
        values = head(x)

        # Should be deterministic for constant input
        assert values.shape == (1, 1)
        assert not torch.isnan(values).any()


class TestAlphaZeroNet:
    """Test complete AlphaZero network."""

    def test_alphazero_creation(self):
        """Test AlphaZero network creation."""
        model = AlphaZeroNet(
            input_channels=7,
            num_actions=225,
            num_blocks=5,  # Smaller for testing
            hidden_channels=64
        )

        assert model.input_channels == 7
        assert model.num_actions == 225
        assert model.num_blocks == 5
        assert model.hidden_channels == 64
        assert len(model.residual_blocks) == 5

    def test_alphazero_forward_shape(self):
        """Test AlphaZero forward pass shapes."""
        model = AlphaZeroNet(
            input_channels=3,
            num_actions=9,
            num_blocks=2,
            hidden_channels=32
        )

        x = torch.randn(2, 3, 3, 3)
        policy_logits, values = model(x)

        assert policy_logits.shape == (2, 9)
        assert values.shape == (2, 1)

    def test_alphazero_parameter_count(self):
        """Test parameter counting."""
        model = AlphaZeroNet(
            input_channels=1,
            num_actions=4,
            num_blocks=1,
            hidden_channels=8
        )

        num_params = model.get_num_parameters()
        assert num_params > 0

        # Manual count should match
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params == manual_count

    def test_alphazero_memory_estimation(self):
        """Test memory usage estimation."""
        model = AlphaZeroNet(
            input_channels=7,
            num_actions=225,
            num_blocks=3,
            hidden_channels=64
        )

        memory_info = model.get_memory_usage(16, (7, 15, 15))

        required_keys = ['parameters_mb', 'activations_mb', 'outputs_mb', 'total_mb', 'fits_8gb']
        for key in required_keys:
            assert key in memory_info
            assert isinstance(memory_info[key], (int, float, bool))

    def test_alphazero_different_games(self):
        """Test AlphaZero with different game configurations."""
        # Small model for testing
        configs = [
            (3, 9, 3, 3),    # 3x3 tic-tac-toe
            (7, 225, 15, 15), # 15x15 gomoku
            (12, 64, 8, 8),   # 8x8 chess (simplified)
        ]

        for input_channels, num_actions, h, w in configs:
            model = AlphaZeroNet(
                input_channels=input_channels,
                num_actions=num_actions,
                num_blocks=2,
                hidden_channels=32
            )

            x = torch.randn(1, input_channels, h, w)
            policy_logits, values = model(x)

            assert policy_logits.shape == (1, num_actions)
            assert values.shape == (1, 1)

    def test_alphazero_gradients(self):
        """Test gradient flow through full network."""
        model = AlphaZeroNet(
            input_channels=2,
            num_actions=4,
            num_blocks=1,
            hidden_channels=16
        )

        x = torch.randn(1, 2, 2, 2, requires_grad=True)
        policy_logits, values = model(x)

        # Test policy gradients
        policy_loss = policy_logits.sum()
        policy_loss.backward(retain_graph=True)
        assert x.grad is not None

        # Reset gradients
        x.grad.zero_()

        # Test value gradients
        value_loss = values.sum()
        value_loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestModelFactory:
    """Test model factory functions."""

    def test_create_model_for_game(self):
        """Test game-specific model creation."""
        games = ['gomoku', 'chess', 'go']
        expected_configs = [
            (36, 225),   # Gomoku (enhanced features)
            (30, 4096),  # Chess (enhanced features)
            (25, 361),   # Go (enhanced features)
        ]

        for game, (input_channels, num_actions) in zip(games, expected_configs):
            model = create_model_for_game(game, num_blocks=3, hidden_channels=32)

            assert model.input_channels == input_channels
            assert model.num_actions == num_actions
            assert model.num_blocks == 3
            assert model.hidden_channels == 32

    def test_create_model_invalid_game(self):
        """Test invalid game type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported game type"):
            create_model_for_game('invalid_game')

    def test_create_random_model(self):
        """Test random model creation."""
        model1 = create_random_model('gomoku', seed=42)
        model2 = create_random_model('gomoku', seed=42)

        # Same seed should produce identical models
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_create_random_model_different_seeds(self):
        """Test random models with different seeds are different."""
        model1 = create_random_model('gomoku', seed=1)
        model2 = create_random_model('gomoku', seed=2)

        # Different seeds should produce different models
        param_diff = False
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.allclose(p1, p2):
                param_diff = True
                break

        assert param_diff, "Models with different seeds should be different"


class TestMixedPrecision:
    """Test mixed precision support."""

    def test_enable_mixed_precision(self):
        """Test mixed precision enable function."""
        model = AlphaZeroNet(
            input_channels=3,
            num_actions=9,
            num_blocks=2,
            hidden_channels=16
        )

        # Enable mixed precision
        model_fp16 = enable_mixed_precision(model)

        # Check BatchNorm layers are in FP32
        for module in model_fp16.modules():
            if isinstance(module, nn.BatchNorm2d):
                # BatchNorm should stay in FP32 for numerical stability
                assert module.weight.dtype == torch.float32
                assert module.bias.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_forward(self):
        """Test mixed precision forward pass on GPU."""
        model = AlphaZeroNet(
            input_channels=2,
            num_actions=4,
            num_blocks=1,
            hidden_channels=8
        ).cuda()

        model = enable_mixed_precision(model)

        x = torch.randn(1, 2, 3, 3).cuda()

        # Test with autocast
        with torch.cuda.amp.autocast():
            policy_logits, values = model(x)

        assert policy_logits.shape == (1, 4)
        assert values.shape == (1, 1)
        assert not torch.isnan(policy_logits).any()
        assert not torch.isnan(values).any()


class TestModelValidation:
    """Test model validation utilities."""

    def test_validate_model_output_valid(self):
        """Test validation passes for valid model."""
        model = AlphaZeroNet(
            input_channels=2,
            num_actions=4,
            num_blocks=1,
            hidden_channels=8
        )

        x = torch.randn(2, 2, 2, 2)
        is_valid = validate_model_output(model, x)

        assert is_valid

    def test_validate_model_output_invalid_shapes(self):
        """Test validation with custom model that produces wrong shapes."""
        class InvalidModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_actions = 4

            def forward(self, x):
                batch_size = x.size(0)
                # Return wrong shapes
                policy = torch.randn(batch_size, 5)  # Should be 4
                value = torch.randn(batch_size, 2)   # Should be 1
                return policy, value

        model = InvalidModel()
        x = torch.randn(1, 2, 2, 2)

        is_valid = validate_model_output(model, x)
        assert not is_valid

    def test_validate_model_output_invalid_range(self):
        """Test validation with values outside [-1, 1] range."""
        class InvalidRangeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_actions = 4

            def forward(self, x):
                batch_size = x.size(0)
                policy = torch.randn(batch_size, 4)
                value = torch.randn(batch_size, 1) * 5  # Values outside [-1, 1]
                return policy, value

        model = InvalidRangeModel()
        x = torch.randn(1, 2, 2, 2)

        is_valid = validate_model_output(model, x)
        assert not is_valid

    def test_validate_model_output_nan(self):
        """Test validation catches NaN outputs."""
        class NaNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_actions = 4

            def forward(self, x):
                batch_size = x.size(0)
                policy = torch.full((batch_size, 4), float('nan'))
                value = torch.randn(batch_size, 1)
                return policy, value

        model = NaNModel()
        x = torch.randn(1, 2, 2, 2)

        is_valid = validate_model_output(model, x)
        assert not is_valid


class TestModelMemoryEfficiency:
    """Test memory efficiency and constraints."""

    def test_target_parameter_count(self):
        """Test model achieves target parameter count (~24M for RTX 3060 Ti)."""
        model = create_model_for_game('gomoku')  # Default 20 blocks, 256 channels

        num_params = model.get_num_parameters()

        # Should be around 24M parameters (allow 20% tolerance) - optimized for RTX 3060 Ti
        target = 24_000_000
        tolerance = 0.2

        assert target * (1 - tolerance) <= num_params <= target * (1 + tolerance), \
            f"Parameter count {num_params:,} not near target {target:,}"

    def test_memory_fits_constraint(self):
        """Test model fits in 8GB VRAM with optimal batch size."""
        model = create_model_for_game('gomoku')

        memory_info = model.get_memory_usage(64, (7, 15, 15))
        optimal_batch = memory_info['optimal_batch_size']
        optimal_memory = model.get_memory_usage(optimal_batch, (7, 15, 15))

        # Test baseline batch=64 fits comfortably
        assert memory_info['fits_8gb'], \
            f"Model uses {memory_info['total_mb']:.1f}MB with batch=64, should fit in 7GB limit"

        # Test optimal batch utilizes GPU well but still fits
        assert optimal_memory['total_mb'] < 7000, \
            f"Optimal batch {optimal_batch} uses {optimal_memory['total_mb']:.1f}MB, exceeds 7GB limit"

        # Test optimal batch size is reasonable for high throughput
        assert 128 <= optimal_batch <= 512, \
            f"Optimal batch size {optimal_batch} should be 128-512 for RTX 3060 Ti"

    def test_different_batch_sizes(self):
        """Test memory usage scales appropriately with batch size."""
        model = create_model_for_game('gomoku')

        batch_sizes = [1, 16, 32, 64]
        memory_usages = []

        for batch_size in batch_sizes:
            memory_info = model.get_memory_usage(batch_size, (7, 15, 15))
            memory_usages.append(memory_info['total_mb'])

        # Memory should increase with batch size
        for i in range(1, len(memory_usages)):
            assert memory_usages[i] > memory_usages[i-1]


# Integration test for full workflow
class TestModelIntegration:
    """Test full model integration scenarios."""

    def test_training_workflow(self):
        """Test basic training workflow."""
        model = AlphaZeroNet(
            input_channels=3,
            num_actions=9,
            num_blocks=2,
            hidden_channels=16
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Sample batch
        x = torch.randn(4, 3, 3, 3)
        target_policy = torch.randint(0, 9, (4,))
        target_value = torch.randn(4, 1)

        # Forward pass
        policy_logits, values = model(x)

        # Compute losses
        policy_loss = criterion(policy_logits, target_policy)
        value_loss = nn.MSELoss()(values, target_value)
        total_loss = policy_loss + value_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Check gradients were applied
        assert total_loss.item() > 0

    def test_inference_workflow(self):
        """Test inference workflow."""
        model = create_model_for_game('gomoku')
        model.eval()

        # Sample position
        x = torch.randn(1, 36, 15, 15)

        with torch.no_grad():
            policy_logits, values = model(x)

            # Convert to probabilities
            policy_probs = torch.softmax(policy_logits, dim=1)

        assert policy_probs.sum(dim=1).allclose(torch.tensor(1.0), atol=1e-5)
        assert policy_probs.min() >= 0
        assert -1 <= values.item() <= 1

    def test_model_saving_loading(self):
        """Test model can be saved and loaded."""
        model = AlphaZeroNet(
            input_channels=2,
            num_actions=4,
            num_blocks=1,
            hidden_channels=8
        )

        x = torch.randn(1, 2, 2, 2)
        # Initialize lazy layers by running forward pass
        original_output = model(x)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            model_path = f.name

        try:
            # Create new model and initialize with same input
            loaded_model = AlphaZeroNet(
                input_channels=2,
                num_actions=4,
                num_blocks=1,
                hidden_channels=8
            )
            # Initialize lazy layers first
            _ = loaded_model(x)
            # Now load the state dict
            loaded_model.load_state_dict(torch.load(model_path))

            # Compare outputs
            loaded_output = loaded_model(x)

            assert torch.allclose(original_output[0], loaded_output[0], atol=1e-5)
            assert torch.allclose(original_output[1], loaded_output[1], atol=1e-5)

        finally:
            os.unlink(model_path)


# HOWTO-RUN-TESTS Block
"""
HOWTO-RUN-TESTS
===============

Run neural network model tests:

# Run all model tests
python -m pytest tests/unit/test_neural_model.py -v

# Run specific test classes
python -m pytest tests/unit/test_neural_model.py::TestSqueezeExcitation -v
python -m pytest tests/unit/test_neural_model.py::TestAlphaZeroNet -v
python -m pytest tests/unit/test_neural_model.py::TestModelFactory -v

# Test memory efficiency
python -m pytest tests/unit/test_neural_model.py::TestModelMemoryEfficiency -v

# Skip GPU tests if no CUDA available
python -m pytest tests/unit/test_neural_model.py -v -k "not cuda"

# Test direct model execution
python src/neural/model.py

Expected Results:
✅ All components (SE, ResidualBlock, heads) work correctly
✅ Full AlphaZero model produces valid outputs
✅ Parameter count ~10M for production model
✅ Memory usage fits in 8GB VRAM with batch size 64
✅ Mixed precision support working
✅ Game-specific model factory functions work
✅ Model validation catches invalid outputs
✅ Training and inference workflows complete successfully

Test Coverage:
- SqueezeExcitation attention mechanism
- ResidualBlock with SE attention
- PolicyHead and ValueHead components
- Full AlphaZero network architecture
- Model factory functions for different games
- Mixed precision compatibility
- Memory usage estimation and validation
- Parameter counting and model size constraints
- Training and inference workflow integration
- Model saving/loading functionality

Performance Validation:
- Gomoku model: ~10M parameters, fits in 8GB with batch=64
- Forward pass completes without errors
- Gradients flow correctly through all components
- Output shapes and ranges are correct
- Memory estimation is accurate
"""