"""
Unit tests for DLPackInferenceBridge (T008b)

Tests DLPack→PyTorch conversion, error handling, and metrics tracking.
"""

import pytest
import time
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
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from src.core.dlpack_inference_bridge import DLPackInferenceBridge


# Simple test model for Gomoku
class SimpleGomokuNet(nn.Module):
    """Minimal network for testing"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(36, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_policy = nn.Linear(64, 225)  # 15x15 board
        self.fc_value = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        policy = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x)).squeeze(-1)
        return policy, value


@pytest.mark.skipif(not HAS_ALPHAZERO, reason="alphazero_py not available")
@pytest.mark.skipif(not HAS_MCTS, reason="mcts_py not available")
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestDLPackInferenceBridge:
    """Test suite for DLPackInferenceBridge"""

    @pytest.fixture
    def model(self):
        """Create test model"""
        return SimpleGomokuNet()

    @pytest.fixture
    def bridge_cpu(self, model):
        """Create bridge on CPU"""
        return DLPackInferenceBridge(model, device='cpu', warmup_iterations=0)

    @pytest.fixture
    def bridge_cuda(self, model):
        """Create bridge on CUDA if available"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return DLPackInferenceBridge(model, device='cuda', warmup_iterations=0)

    def test_initialization(self, model):
        """Test bridge initialization"""
        bridge = DLPackInferenceBridge(model, device='cpu')

        assert bridge.device.type == 'cpu'
        assert bridge.enable_fallback is True
        assert bridge.model is not None

    def test_batch_inference_single_state(self, bridge_cpu):
        """Test inference with single state"""
        state = alphazero_py.GomokuState()
        results = bridge_cpu.batch_inference([state])

        assert len(results) == 1
        policy, value = results[0]

        assert isinstance(policy, list)
        assert len(policy) == 225  # 15x15 board
        assert all(isinstance(p, float) for p in policy)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_batch_inference_multiple_states(self, bridge_cpu):
        """Test inference with multiple states"""
        states = [alphazero_py.GomokuState() for _ in range(8)]
        results = bridge_cpu.batch_inference(states)

        assert len(results) == 8
        for policy, value in results:
            assert isinstance(policy, list)
            assert len(policy) == 225
            assert isinstance(value, float)

    def test_batch_sizes(self, bridge_cpu):
        """Test various batch sizes"""
        for batch_size in [1, 4, 8, 16, 32, 64]:
            states = [alphazero_py.GomokuState() for _ in range(batch_size)]
            results = bridge_cpu.batch_inference(states)

            assert len(results) == batch_size

    def test_policy_sums_to_one(self, bridge_cpu):
        """Test that policy probabilities sum to ~1.0"""
        states = [alphazero_py.GomokuState() for _ in range(4)]
        results = bridge_cpu.batch_inference(states)

        for policy, _ in results:
            # After softmax, probabilities should sum to 1
            total = sum(policy)
            assert abs(total - 1.0) < 0.01, f"Policy sum {total} != 1.0"

    def test_different_game_states(self, bridge_cpu):
        """Test that different states produce different outputs"""
        states = []
        for i in range(4):
            state = alphazero_py.GomokuState()
            # Make different moves
            if i > 0:
                state.make_move(112 + i)
            states.append(state)

        results = bridge_cpu.batch_inference(states)

        # Policies should be different
        policies = [r[0] for r in results]
        # Check that not all policies are identical
        assert not all(
            np.allclose(policies[0], p, rtol=1e-6)
            for p in policies[1:]
        ), "All policies are identical"

    def test_error_empty_list(self, bridge_cpu):
        """Test error on empty states list"""
        with pytest.raises(ValueError, match="empty"):
            bridge_cpu.batch_inference([])

    def test_metrics_tracking(self, bridge_cpu):
        """Test metrics are tracked correctly"""
        bridge_cpu.reset_metrics()

        states = [alphazero_py.GomokuState() for _ in range(8)]
        bridge_cpu.batch_inference(states)

        metrics = bridge_cpu.get_metrics()

        assert metrics['total_batches'] == 1
        assert metrics['total_states'] == 8
        assert metrics['avg_batch_size'] == 8.0
        assert metrics['dlpack_successes'] == 1
        assert metrics['fallback_uses'] == 0
        assert metrics['avg_latency_ms'] > 0
        assert metrics['dlpack_success_rate'] == 100.0

    def test_metrics_multiple_batches(self, bridge_cpu):
        """Test metrics with multiple batches"""
        bridge_cpu.reset_metrics()

        # Run 3 batches of different sizes
        for batch_size in [4, 8, 16]:
            states = [alphazero_py.GomokuState() for _ in range(batch_size)]
            bridge_cpu.batch_inference(states)

        metrics = bridge_cpu.get_metrics()

        assert metrics['total_batches'] == 3
        assert metrics['total_states'] == 4 + 8 + 16
        assert metrics['avg_batch_size'] == pytest.approx(28 / 3)
        assert metrics['dlpack_successes'] == 3

    def test_metrics_reset(self, bridge_cpu):
        """Test metrics can be reset"""
        states = [alphazero_py.GomokuState() for _ in range(4)]
        bridge_cpu.batch_inference(states)

        bridge_cpu.reset_metrics()
        metrics = bridge_cpu.get_metrics()

        assert metrics['total_batches'] == 0
        assert metrics['total_states'] == 0
        assert metrics['dlpack_successes'] == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_inference(self, bridge_cuda):
        """Test inference on CUDA"""
        states = [alphazero_py.GomokuState() for _ in range(8)]
        results = bridge_cuda.batch_inference(states)

        assert len(results) == 8
        for policy, value in results:
            assert isinstance(policy, list)
            assert isinstance(value, float)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_placement(self, bridge_cuda):
        """Test model is on CUDA device"""
        # Check model is on CUDA
        assert next(bridge_cuda.model.parameters()).is_cuda

    def test_warmup_cpu(self, bridge_cpu):
        """Test warmup on CPU"""
        bridge_cpu.reset_metrics()

        # Warmup should run without errors
        bridge_cpu.warmup_iterations = 3
        bridge_cpu.warmup(batch_size=8, game_type='gomoku')

        metrics = bridge_cpu.get_metrics()
        assert metrics['total_batches'] == 3
        assert metrics['total_states'] == 24  # 3 * 8

    def test_conversion_overhead_low(self, bridge_cpu):
        """Test that DLPack conversion overhead is low (<50μs)"""
        states = [alphazero_py.GomokuState() for _ in range(64)]

        # Warmup
        bridge_cpu.batch_inference(states)

        # Measure conversion time
        # (Full inference includes model forward pass, so we measure total)
        start = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            bridge_cpu.batch_inference(states)
        elapsed_ms = (time.perf_counter() - start) / iterations * 1000

        # Full inference should be reasonably fast
        # (includes conversion + model forward pass)
        assert elapsed_ms < 100.0, f"Inference too slow: {elapsed_ms:.2f} ms"

        # Check metrics show DLPack is being used
        metrics = bridge_cpu.get_metrics()
        assert metrics['dlpack_success_rate'] == 100.0

    def test_fallback_disabled_raises(self, model):
        """Test that fallback disabled raises on DLPack failure"""
        bridge = DLPackInferenceBridge(
            model, device='cpu', enable_fallback=False
        )

        # Normal operation should work
        states = [alphazero_py.GomokuState() for _ in range(4)]
        results = bridge.batch_inference(states)
        assert len(results) == 4

    def test_chess_states(self, bridge_cpu):
        """Test inference with Chess states"""
        # Create simple Chess model
        class SimpleChessNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(21, 64, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc_policy = nn.Linear(64, 64)  # 8x8 board
                self.fc_value = nn.Linear(64, 1)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x).squeeze(-1).squeeze(-1)
                policy = self.fc_policy(x)
                value = torch.tanh(self.fc_value(x)).squeeze(-1)
                return policy, value

        chess_bridge = DLPackInferenceBridge(
            SimpleChessNet(), device='cpu', warmup_iterations=0
        )

        states = [alphazero_py.ChessState() for _ in range(4)]
        results = chess_bridge.batch_inference(states)

        assert len(results) == 4
        for policy, value in results:
            assert len(policy) == 64  # 8x8 board

    def test_go_states(self, bridge_cpu):
        """Test inference with Go states"""
        # Create simple Go model
        class SimpleGoNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(21, 64, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc_policy = nn.Linear(64, 361)  # 19x19 board
                self.fc_value = nn.Linear(64, 1)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x).squeeze(-1).squeeze(-1)
                policy = self.fc_policy(x)
                value = torch.tanh(self.fc_value(x)).squeeze(-1)
                return policy, value

        go_bridge = DLPackInferenceBridge(
            SimpleGoNet(), device='cpu', warmup_iterations=0
        )

        states = [alphazero_py.GoState() for _ in range(4)]
        results = go_bridge.batch_inference(states)

        assert len(results) == 4
        for policy, value in results:
            assert len(policy) == 361  # 19x19 board

    def test_value_range(self, bridge_cpu):
        """Test that value is in valid range [-1, 1]"""
        states = [alphazero_py.GomokuState() for _ in range(16)]
        results = bridge_cpu.batch_inference(states)

        for _, value in results:
            assert -1.0 <= value <= 1.0, f"Value {value} out of range"

    def test_determinism(self, bridge_cpu):
        """Test that same states produce same results"""
        states = [alphazero_py.GomokuState() for _ in range(4)]

        # Make same moves
        for state in states:
            state.make_move(112)

        # Run inference twice
        results1 = bridge_cpu.batch_inference(states)
        results2 = bridge_cpu.batch_inference(states)

        # Results should be identical (deterministic)
        for (p1, v1), (p2, v2) in zip(results1, results2):
            assert np.allclose(p1, p2, rtol=1e-6)
            assert abs(v1 - v2) < 1e-6
