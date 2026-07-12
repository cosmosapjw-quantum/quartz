#!/usr/bin/env python3
"""
Compare synchronous vs asynchronous inference performance.

Tests if async coordination overhead is causing the regression.
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.mcts import AlphaZeroMCTS
from src.core.dlpack_inference_bridge import DLPackInferenceBridge
from src.neural.model import create_model_for_game
import alphazero_py

def test_sync_inference():
    """Test with synchronous inference (no async queue)."""
    print("\n" + "="*70)
    print("SYNCHRONOUS INFERENCE (Baseline Mode)")
    print("="*70)

    device = torch.device('cuda')
    model = create_model_for_game('gomoku').to(device)
    model.eval()

    inference_bridge = DLPackInferenceBridge(
        model=model,
        device=device,
        use_mixed_precision=True
    )

    # Sync mode: use_async_inference=False
    mcts = AlphaZeroMCTS(
        inference_fn=inference_bridge,
        num_threads=4,
        use_async_inference=False  # SYNC MODE
    )

    state = alphazero_py.GomokuState()

    # Warmup
    mcts.search(state, simulations=100)
    mcts.reset()

    # Benchmark
    times = []
    for i in range(5):
        mcts.reset()
        start = time.perf_counter()
        mcts.search(state, simulations=1000)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        throughput = 1000 / elapsed
        print(f"  Iteration {i+1}: {throughput:.0f} sims/sec ({elapsed:.3f}s)")

    avg_throughput = 1000 / (sum(times) / len(times))
    print(f"\n  Average: {avg_throughput:.0f} sims/sec")

    # Get metrics
    metrics = inference_bridge.get_metrics()
    print(f"\n  DLPack success rate: {metrics['dlpack_success_rate']:.1f}%")
    print(f"  Avg H2D transfer: {metrics['avg_h2d_transfer_ms']:.3f} ms")
    print(f"  Avg inference: {metrics['avg_inference_ms']:.3f} ms")
    print(f"  Avg D2H transfer: {metrics['avg_d2h_transfer_ms']:.3f} ms")

    mcts.close()
    return avg_throughput

def test_async_inference():
    """Test with asynchronous inference."""
    print("\n" + "="*70)
    print("ASYNCHRONOUS INFERENCE (Current Implementation)")
    print("="*70)

    device = torch.device('cuda')
    model = create_model_for_game('gomoku').to(device)
    model.eval()

    inference_bridge = DLPackInferenceBridge(
        model=model,
        device=device,
        use_mixed_precision=True
    )

    # Async mode with optimal settings
    mcts = AlphaZeroMCTS(
        inference_fn=inference_bridge,
        num_threads=4,
        use_async_inference=True,  # ASYNC MODE
        async_batch_size=64,
        async_timeout_ms=1.0
    )

    state = alphazero_py.GomokuState()

    # Warmup
    mcts.search(state, simulations=100)
    mcts.reset()

    # Benchmark
    times = []
    for i in range(5):
        mcts.reset()
        start = time.perf_counter()
        mcts.search(state, simulations=1000)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        throughput = 1000 / elapsed
        print(f"  Iteration {i+1}: {throughput:.0f} sims/sec ({elapsed:.3f}s)")

    avg_throughput = 1000 / (sum(times) / len(times))
    print(f"\n  Average: {avg_throughput:.0f} sims/sec")

    # Get metrics
    metrics = inference_bridge.get_metrics()
    print(f"\n  DLPack success rate: {metrics['dlpack_success_rate']:.1f}%")
    print(f"  Avg H2D transfer: {metrics['avg_h2d_transfer_ms']:.3f} ms")
    print(f"  Avg inference: {metrics['avg_inference_ms']:.3f} ms")
    print(f"  Avg D2H transfer: {metrics['avg_d2h_transfer_ms']:.3f} ms")

    mcts.close()
    return avg_throughput

def main():
    """Compare sync vs async performance."""
    print("="*70)
    print("Synchronous vs Asynchronous Inference Comparison")
    print("="*70)
    print("\nGoal: Identify if async overhead is causing regression")
    print(f"Target: 3,831 sims/sec (Spec 003 baseline)")

    sync_throughput = test_sync_inference()
    async_throughput = test_async_inference()

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\nSynchronous:  {sync_throughput:.0f} sims/sec")
    print(f"Asynchronous: {async_throughput:.0f} sims/sec")
    print(f"Ratio: {async_throughput / sync_throughput:.2f}x")

    print(f"\nvs Baseline (3,831 sims/sec):")
    print(f"  Sync: {sync_throughput / 3831 * 100:.1f}%")
    print(f"  Async: {async_throughput / 3831 * 100:.1f}%")

    if sync_throughput > async_throughput * 1.2:
        print("\n⚠️  Async overhead is significant (>20% slower)")
        print("    Consider fixing async coordination or using sync mode")
    elif async_throughput > sync_throughput * 1.2:
        print("\n✅ Async is faster (as expected)")
    else:
        print("\n≈ Similar performance (async overhead acceptable)")

if __name__ == '__main__':
    main()
