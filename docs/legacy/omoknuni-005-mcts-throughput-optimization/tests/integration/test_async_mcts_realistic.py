"""
Realistic integration test for async MCTS - NO MOCKS

This test uses:
1. Real neural network (small random network, no training needed)
2. Real game states (Gomoku)
3. Real MCTS search with async batching
4. Actual game play simulation

Goal: Verify 0 errors in realistic usage
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import Future, ThreadPoolExecutor
import time
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from core.mcts import AlphaZeroMCTS
import alphazero_py


class SimpleGomokuNet(nn.Module):
    """Simple neural network for Gomoku (no training needed)."""

    def __init__(self):
        super().__init__()
        # Input: 36 planes x 15x15 (from Gomoku enhanced features)
        self.conv1 = nn.Conv2d(36, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 15 * 15, 225)

        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(15 * 15, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Shared layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Policy head
        policy = torch.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = torch.softmax(policy, dim=1)

        # Value head
        value = torch.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


class RealisticInferenceEngine:
    """Realistic inference engine with neural network."""

    def __init__(self, device='cpu'):
        self.device = device
        self.model = SimpleGomokuNet().to(device)
        self.model.eval()  # Evaluation mode
        self.executor = ThreadPoolExecutor(max_workers=1)  # Simulate async

        # Statistics
        self.total_inferences = 0
        self.total_time = 0.0

    def preprocess_state(self, state) -> torch.Tensor:
        """Convert game state to neural network input."""
        # Get enhanced tensor representation (36 planes x 15x15)
        tensor = state.get_enhanced_tensor_representation()
        # Convert to numpy array then to torch tensor
        tensor_np = np.array(tensor, dtype=np.float32)
        # Shape: (36, 15, 15)
        return torch.from_numpy(tensor_np).unsqueeze(0).to(self.device)

    def inference(self, state) -> Future:
        """Async inference for single state."""
        future = Future()

        def run_inference():
            try:
                start = time.perf_counter()

                # Preprocess
                input_tensor = self.preprocess_state(state)

                # Run neural network
                with torch.no_grad():
                    policy_logits, value = self.model(input_tensor)

                # Convert to numpy
                policy = policy_logits.cpu().numpy()[0]
                value_scalar = float(value.cpu().numpy()[0, 0])

                # Update stats
                self.total_inferences += 1
                self.total_time += time.perf_counter() - start

                future.set_result((policy, value_scalar))
            except Exception as e:
                future.set_exception(e)

        self.executor.submit(run_inference)
        return future

    def shutdown(self):
        """Shutdown executor."""
        self.executor.shutdown(wait=True)

    def get_stats(self):
        """Get performance statistics."""
        avg_time = self.total_time / max(self.total_inferences, 1)
        return {
            'total_inferences': self.total_inferences,
            'total_time_ms': self.total_time * 1000,
            'avg_time_ms': avg_time * 1000,
        }


def test_realistic_async_mcts_single_search():
    """Test realistic async MCTS with single search."""
    print("\n=== Testing realistic async MCTS (single search) ===")

    # Create inference engine with real neural network
    engine = RealisticInferenceEngine()

    try:
        # Create async MCTS
        mcts = AlphaZeroMCTS(
            inference_fn=engine.inference,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=10.0,
            c_puct=1.25
        )

        # Create initial game state
        state = alphazero_py.GomokuState()

        # Run search
        print("Running 100 simulations...")
        start_time = time.perf_counter()
        visit_counts = mcts.search(state, simulations=100)
        search_time = time.perf_counter() - start_time

        # Verify results
        assert len(visit_counts) > 0, "No moves found"
        root_visits = mcts.tree.get_visit_count(mcts.root_index)
        assert root_visits == 100.0, f"Expected 100 visits, got {root_visits}"

        # Get policy
        policy = mcts.get_policy(state, temperature=1.0)
        assert np.isclose(np.sum(policy), 1.0, atol=1e-5), "Policy doesn't sum to 1"
        assert np.all(policy >= 0), "Policy has negative values"

        # Performance metrics
        throughput = 100 / search_time
        stats = engine.get_stats()

        print(f"✓ Search completed successfully")
        print(f"  Throughput: {throughput:.1f} sims/sec")
        print(f"  Total search time: {search_time*1000:.1f}ms")
        print(f"  Neural network stats:")
        print(f"    - Total inferences: {stats['total_inferences']}")
        print(f"    - Total time: {stats['total_time_ms']:.1f}ms")
        print(f"    - Avg per inference: {stats['avg_time_ms']:.2f}ms")
        print(f"  Policy entropy: {-np.sum(policy * np.log(policy + 1e-10)):.2f}")

    finally:
        engine.shutdown()


def test_realistic_async_mcts_game_simulation():
    """Test realistic async MCTS in full game simulation."""
    print("\n=== Testing realistic async MCTS (full game simulation) ===")

    # Create inference engine
    engine = RealisticInferenceEngine()

    try:
        # Create async MCTS
        mcts = AlphaZeroMCTS(
            inference_fn=engine.inference,
            use_async_inference=True,
            async_batch_size=16,
            async_timeout_ms=5.0,
            c_puct=1.25
        )

        # Play a short game
        state = alphazero_py.GomokuState()
        moves_played = 0
        max_moves = 10  # Play 10 moves

        print(f"Simulating {max_moves} moves of Gomoku...")
        total_search_time = 0.0

        while moves_played < max_moves and not state.is_terminal():
            # Run MCTS search
            start = time.perf_counter()
            visit_counts = mcts.search(state, simulations=50, add_noise=False)
            search_time = time.perf_counter() - start
            total_search_time += search_time

            # Get move probabilities
            policy = mcts.get_policy(state, temperature=0.0)  # Greedy

            # Select best move
            legal_moves = state.get_legal_moves()
            legal_policy = [(move, policy[move]) for move in legal_moves]
            best_move = max(legal_policy, key=lambda x: x[1])[0]

            # Apply move
            state.make_move(best_move)
            moves_played += 1

            print(f"  Move {moves_played}: action={best_move}, time={search_time*1000:.1f}ms")

            # Reset tree for next move
            mcts.reset()

        # Final stats
        stats = engine.get_stats()
        avg_time_per_move = total_search_time / moves_played

        print(f"✓ Game simulation completed successfully")
        print(f"  Moves played: {moves_played}")
        print(f"  Total search time: {total_search_time*1000:.1f}ms")
        print(f"  Avg time per move: {avg_time_per_move*1000:.1f}ms")
        print(f"  Total NN inferences: {stats['total_inferences']}")
        print(f"  Inferences per move: {stats['total_inferences'] / moves_played:.1f}")

    finally:
        engine.shutdown()


def test_realistic_sync_vs_async_comparison():
    """Compare sync vs async mode with realistic inference."""
    print("\n=== Testing sync vs async comparison ===")

    # Test parameters
    num_simulations = 50

    # Create inference engines for each mode
    engine_sync = RealisticInferenceEngine()
    engine_async = RealisticInferenceEngine()

    try:
        # Sync mode
        print("Testing sync mode...")
        mcts_sync = AlphaZeroMCTS(
            inference_fn=engine_sync.inference,
            use_async_inference=False,
            c_puct=1.25
        )

        state = alphazero_py.GomokuState()
        start = time.perf_counter()
        mcts_sync.search(state, simulations=num_simulations)
        sync_time = time.perf_counter() - start
        sync_throughput = num_simulations / sync_time

        # Async mode
        print("Testing async mode...")
        mcts_async = AlphaZeroMCTS(
            inference_fn=engine_async.inference,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=10.0,
            c_puct=1.25
        )

        state = alphazero_py.GomokuState()
        start = time.perf_counter()
        mcts_async.search(state, simulations=num_simulations)
        async_time = time.perf_counter() - start
        async_throughput = num_simulations / async_time

        # Compare
        speedup = async_throughput / sync_throughput

        print(f"✓ Comparison completed")
        print(f"  Sync mode:  {sync_throughput:.1f} sims/sec ({sync_time*1000:.1f}ms)")
        print(f"  Async mode: {async_throughput:.1f} sims/sec ({async_time*1000:.1f}ms)")
        print(f"  Speedup: {speedup:.2f}x")

        # Async should be at least as fast (may be slower due to overhead with small batches)
        # But both should complete without errors
        assert sync_throughput > 0
        assert async_throughput > 0

    finally:
        engine_sync.shutdown()
        engine_async.shutdown()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("REALISTIC ASYNC MCTS INTEGRATION TEST (NO MOCKS)")
    print("="*60)

    test_realistic_async_mcts_single_search()
    test_realistic_async_mcts_game_simulation()
    test_realistic_sync_vs_async_comparison()

    print("\n" + "="*60)
    print("ALL REALISTIC TESTS PASSED - 0 ERRORS!")
    print("="*60 + "\n")
