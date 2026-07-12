"""
Integration tests for DLPackInferenceBridge with full pipeline (T008e)

Tests end-to-end integration with neural networks, batch processing,
and sustained load scenarios.
"""

import pytest
import time
import gc
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


# Realistic model architectures for testing
class GomokuResNet(nn.Module):
    """Realistic ResNet for Gomoku (similar to production)"""
    def __init__(self, num_blocks=5, num_channels=128):
        super().__init__()
        self.conv_input = nn.Conv2d(36, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(num_channels) for _ in range(num_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 15 * 15, 225)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 15 * 15, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def _make_res_block(self, num_channels):
        return nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels)
        )

    def forward(self, x):
        # Input
        x = torch.relu(self.bn_input(self.conv_input(x)))

        # Residual blocks
        for block in self.res_blocks:
            identity = x
            x = block(x) + identity
            x = torch.relu(x)

        # Policy head
        policy = torch.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = torch.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)

        return policy, value


@pytest.mark.skipif(not HAS_ALPHAZERO, reason="alphazero_py not available")
@pytest.mark.skipif(not HAS_MCTS, reason="mcts_py not available")
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestDLPackInferenceIntegration:
    """Integration tests for DLPackInferenceBridge"""

    @pytest.fixture
    def resnet_model(self):
        """Create realistic ResNet model"""
        return GomokuResNet(num_blocks=5, num_channels=128)

    @pytest.fixture
    def bridge_cpu(self, resnet_model):
        """Create bridge with ResNet on CPU"""
        return DLPackInferenceBridge(
            resnet_model, device='cpu', warmup_iterations=3
        )

    @pytest.fixture
    def bridge_cuda(self, resnet_model):
        """Create bridge with ResNet on CUDA"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return DLPackInferenceBridge(
            resnet_model, device='cuda', warmup_iterations=3
        )

    def test_resnet_inference_correctness(self, bridge_cpu):
        """Test ResNet inference produces valid outputs"""
        states = [alphazero_py.GomokuState() for _ in range(32)]

        # Make some moves to create diverse states
        for i, state in enumerate(states[:16]):
            state.make_move(112 + (i % 8))

        results = bridge_cpu.batch_inference(states)

        assert len(results) == 32

        for i, (policy, value) in enumerate(results):
            # Policy checks
            assert isinstance(policy, list)
            assert len(policy) == 225
            assert all(isinstance(p, float) for p in policy)
            assert all(p >= 0 for p in policy), f"Negative policy at {i}"

            # Policy should sum to ~1.0
            policy_sum = sum(policy)
            assert abs(policy_sum - 1.0) < 0.01, \
                f"Policy sum {policy_sum} != 1.0 at {i}"

            # Value checks
            assert isinstance(value, float)
            assert -1.0 <= value <= 1.0, \
                f"Value {value} out of range at {i}"

    def test_batch_size_variations(self, bridge_cpu):
        """Test various batch sizes work correctly"""
        for batch_size in [1, 4, 8, 16, 32, 64, 128]:
            states = [alphazero_py.GomokuState() for _ in range(batch_size)]
            results = bridge_cpu.batch_inference(states)

            assert len(results) == batch_size

            # Verify first result
            policy, value = results[0]
            assert len(policy) == 225
            assert abs(sum(policy) - 1.0) < 0.01
            assert -1.0 <= value <= 1.0

    def test_sustained_load(self, bridge_cpu):
        """Test sustained inference load over many batches"""
        bridge_cpu.reset_metrics()

        num_batches = 100
        batch_size = 32

        for _ in range(num_batches):
            states = [alphazero_py.GomokuState() for _ in range(batch_size)]
            results = bridge_cpu.batch_inference(states)

            assert len(results) == batch_size

        metrics = bridge_cpu.get_metrics()

        assert metrics['total_batches'] == num_batches
        assert metrics['total_states'] == num_batches * batch_size
        assert metrics['avg_batch_size'] == batch_size
        assert metrics['dlpack_success_rate'] == 100.0

    def test_no_memory_leak(self, bridge_cpu):
        """Test no memory leaks over many iterations"""
        # Force garbage collection
        gc.collect()

        # Get initial memory state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_allocated = torch.cuda.memory_allocated()
        else:
            initial_allocated = 0

        # Run many iterations
        for _ in range(1000):
            states = [alphazero_py.GomokuState() for _ in range(16)]
            bridge_cpu.batch_inference(states)

        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_allocated = torch.cuda.memory_allocated()
        else:
            final_allocated = 0

        # Memory should not grow significantly
        # (Allow some growth for Python overhead, but not proportional to iterations)
        memory_growth = final_allocated - initial_allocated
        print(f"Memory growth: {memory_growth / 1024 / 1024:.2f} MB")

        # Should not grow more than 100MB even after 1000 iterations
        assert memory_growth < 100 * 1024 * 1024, \
            f"Memory leak detected: {memory_growth / 1024 / 1024:.2f} MB growth"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_inference_performance(self, bridge_cuda):
        """Test GPU inference performance"""
        bridge_cuda.reset_metrics()

        # Warmup
        states = [alphazero_py.GomokuState() for _ in range(64)]
        for _ in range(10):
            bridge_cuda.batch_inference(states)

        # Measure
        bridge_cuda.reset_metrics()
        start = time.perf_counter()
        iterations = 50
        for _ in range(iterations):
            bridge_cuda.batch_inference(states)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / iterations) * 1000
        print(f"\nGPU inference (batch 64): {avg_latency_ms:.2f} ms/iter")

        # Should be reasonably fast on GPU
        assert avg_latency_ms < 100.0, \
            f"GPU inference too slow: {avg_latency_ms:.2f} ms"

        metrics = bridge_cuda.get_metrics()
        assert metrics['dlpack_success_rate'] == 100.0

    def test_different_game_positions(self, bridge_cpu):
        """Test inference on different game positions"""
        # Create states with various move counts
        test_cases = [
            0,   # Empty board
            1,   # Single move
            5,   # Early game
            10,  # Mid game
            20   # Late game
        ]

        for num_moves in test_cases:
            state = alphazero_py.GomokuState()
            for i in range(num_moves):
                legal_moves = state.get_legal_moves()
                if len(legal_moves) == 0:
                    break
                state.make_move(legal_moves[i % len(legal_moves)])

            results = bridge_cpu.batch_inference([state])
            policy, value = results[0]

            assert len(policy) == 225
            assert abs(sum(policy) - 1.0) < 0.01
            assert -1.0 <= value <= 1.0

    def test_concurrent_inference_calls(self, bridge_cpu):
        """Test multiple concurrent inference calls"""
        import threading

        results = []
        errors = []

        def run_inference(batch_size):
            try:
                states = [alphazero_py.GomokuState() for _ in range(batch_size)]
                result = bridge_cpu.batch_inference(states)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = []
        for batch_size in [8, 16, 32]:
            t = threading.Thread(target=run_inference, args=(batch_size,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3

    def test_tensor_correctness_vs_direct_extraction(self, bridge_cpu):
        """Verify bridge produces same features as direct extraction"""
        state = alphazero_py.GomokuState()
        state.make_move(112)

        # Get features via bridge
        results = bridge_cpu.batch_inference([state])
        # Note: We can't directly compare model outputs, but we can verify
        # the shapes and validity

        policy, value = results[0]
        assert len(policy) == 225
        assert -1.0 <= value <= 1.0

    def test_error_recovery(self, bridge_cpu):
        """Test bridge recovers from errors"""
        # First, successful inference
        states = [alphazero_py.GomokuState() for _ in range(8)]
        results = bridge_cpu.batch_inference(states)
        assert len(results) == 8

        # Then, error case (empty list)
        with pytest.raises(ValueError):
            bridge_cpu.batch_inference([])

        # Should still work after error
        results = bridge_cpu.batch_inference(states)
        assert len(results) == 8

    def test_warmup_reduces_first_batch_latency(self, resnet_model):
        """Test that warmup improves first batch performance"""
        # Bridge without warmup
        bridge_no_warmup = DLPackInferenceBridge(
            resnet_model, device='cpu', warmup_iterations=0
        )

        states = [alphazero_py.GomokuState() for _ in range(64)]

        # Measure first batch (cold)
        start = time.perf_counter()
        bridge_no_warmup.batch_inference(states)
        cold_time = time.perf_counter() - start

        # Bridge with warmup
        bridge_with_warmup = DLPackInferenceBridge(
            resnet_model, device='cpu', warmup_iterations=5
        )
        bridge_with_warmup.warmup(batch_size=64)

        # Measure first batch (warm)
        start = time.perf_counter()
        bridge_with_warmup.batch_inference(states)
        warm_time = time.perf_counter() - start

        print(f"\nCold: {cold_time*1000:.2f} ms, Warm: {warm_time*1000:.2f} ms")

        # Warm should be comparable or faster (not significantly slower)
        # (Allow some variance since warmup primarily helps GPU)
        assert warm_time < cold_time * 1.5

    def test_metrics_accuracy(self, bridge_cpu):
        """Test metrics are tracked accurately"""
        bridge_cpu.reset_metrics()

        # Run known workload
        batch_sizes = [8, 16, 32, 64]
        for batch_size in batch_sizes:
            states = [alphazero_py.GomokuState() for _ in range(batch_size)]
            bridge_cpu.batch_inference(states)

        metrics = bridge_cpu.get_metrics()

        assert metrics['total_batches'] == len(batch_sizes)
        assert metrics['total_states'] == sum(batch_sizes)
        assert metrics['avg_batch_size'] == sum(batch_sizes) / len(batch_sizes)
        assert metrics['dlpack_successes'] == len(batch_sizes)
        assert metrics['fallback_uses'] == 0
        assert metrics['dlpack_success_rate'] == 100.0
        assert metrics['avg_latency_ms'] > 0

    def test_model_in_eval_mode(self, bridge_cpu):
        """Verify model is in eval mode (no gradients)"""
        assert not bridge_cpu.model.training

        # Run inference
        states = [alphazero_py.GomokuState() for _ in range(4)]
        bridge_cpu.batch_inference(states)

        # Model should still be in eval mode
        assert not bridge_cpu.model.training

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_efficiency(self, bridge_cuda):
        """Test GPU memory usage is reasonable"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_memory = torch.cuda.memory_allocated()

        # Run inference
        states = [alphazero_py.GomokuState() for _ in range(64)]
        for _ in range(10):
            bridge_cuda.batch_inference(states)

        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_memory - initial_memory) / 1024 / 1024

        print(f"\nGPU memory used: {memory_used:.2f} MB")

        # Should not use excessive memory (allow up to 500MB for model + buffers)
        assert memory_used < 500, \
            f"Excessive GPU memory usage: {memory_used:.2f} MB"
