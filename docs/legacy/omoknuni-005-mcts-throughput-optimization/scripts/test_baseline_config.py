#!/usr/bin/env python3
"""
Test with exact baseline configuration to reproduce 3,831 sims/sec.

Baseline config (from async_optimization_results.md):
- 4 threads
- Batch size 64
- Timeout 0.5ms
- Result: 3,831 sims/sec
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

def test_configuration(threads, batch_size, timeout_ms, model_desc="Optimized (10.1M)"):
    """Test a specific configuration."""
    print(f"\n{'='*70}")
    print(f"Configuration: {threads} threads, batch {batch_size}, timeout {timeout_ms}ms")
    print(f"Model: {model_desc}")
    print(f"{'='*70}")

    # Create model
    device = torch.device('cuda')
    model = create_model_for_game('gomoku').to(device)
    model.eval()

    # Create inference bridge
    inference_bridge = DLPackInferenceBridge(
        model=model,
        device=device,
        use_mixed_precision=True  # T008f
    )

    # Create MCTS
    mcts = AlphaZeroMCTS(
        inference_fn=inference_bridge,
        num_threads=threads,
        use_async_inference=True,
        async_batch_size=batch_size,
        async_timeout_ms=timeout_ms
    )

    # Create initial state
    state = alphazero_py.GomokuState()

    # Run benchmark
    simulations = 1000
    iterations = 5

    times = []
    for i in range(iterations):
        mcts.reset()
        start = time.perf_counter()
        mcts.search(state, simulations=simulations)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        throughput = simulations / elapsed
        print(f"  Iteration {i+1}: {throughput:.0f} sims/sec ({elapsed:.3f}s)")

    # Calculate average
    avg_time = sum(times) / len(times)
    avg_throughput = simulations / avg_time

    print(f"\n  Average: {avg_throughput:.0f} sims/sec")
    print(f"  vs Baseline: {avg_throughput / 3831 * 100:.1f}%")

    mcts.close()

    return avg_throughput

def main():
    """Test different configurations."""
    print("="*70)
    print("Baseline Configuration Reproduction Test")
    print("="*70)
    print("\nTarget: 3,831 sims/sec (Spec 003 baseline, 4 threads, batch 64)")
    print(f"Model: Optimized 10.1M parameters (vs original 23.8M)")

    # Test exact baseline config
    result_baseline = test_configuration(
        threads=4,
        batch_size=64,
        timeout_ms=0.5
    )

    # Test with optimal batch size from GPU profiling
    result_batch32 = test_configuration(
        threads=4,
        batch_size=32,
        timeout_ms=0.5
    )

    # Test with 2 threads (previously best)
    result_2threads = test_configuration(
        threads=2,
        batch_size=64,
        timeout_ms=0.5
    )

    # Test different timeout
    result_timeout1 = test_configuration(
        threads=4,
        batch_size=64,
        timeout_ms=1.0
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBaseline config (4 threads, batch 64, 0.5ms): {result_baseline:.0f} sims/sec")
    print(f"Optimal GPU batch (4 threads, batch 32, 0.5ms): {result_batch32:.0f} sims/sec")
    print(f"Best from previous (2 threads, batch 64, 0.5ms): {result_2threads:.0f} sims/sec")
    print(f"Higher timeout (4 threads, batch 64, 1.0ms): {result_timeout1:.0f} sims/sec")

    best = max(result_baseline, result_batch32, result_2threads, result_timeout1)
    print(f"\nBest configuration: {best:.0f} sims/sec ({best/3831*100:.1f}% of baseline)")

    if best >= 3831:
        print("✅ MATCHED OR EXCEEDED BASELINE!")
    elif best >= 3831 * 0.9:
        print("⚠️  Within 10% of baseline (acceptable)")
    else:
        print(f"❌ Still {3831/best:.2f}× slower than baseline")

if __name__ == '__main__':
    main()
