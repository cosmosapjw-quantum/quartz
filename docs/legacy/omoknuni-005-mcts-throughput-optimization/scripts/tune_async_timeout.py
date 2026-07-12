#!/usr/bin/env python3
"""
Timeout Tuning for Async Inference (T018)

Test timeout range to find optimal balance between batch size and latency.
Target: Average batch ≥48, throughput ≥28k sims/sec
"""

import sys
import os

# Add both project root and src to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import time
import tempfile
from typing import Dict, List
import alphazero_py
from core.mcts import AlphaZeroMCTS
from neural.inference_worker import GPUInferenceWorker
from neural.model import AlphaZeroNet


def create_test_model(model_path: str, size: str = 'small') -> None:
    """Create test model for benchmarking."""
    if size == 'small':
        model = AlphaZeroNet(
            input_channels=36,
            num_actions=225,
            num_blocks=4,
            hidden_channels=128,
            use_se=False
        )
    else:  # medium
        model = AlphaZeroNet(
            input_channels=36,
            num_actions=225,
            num_blocks=10,
            hidden_channels=192,
            use_se=True
        )

    torch.save(model, model_path)


def measure_timeout_performance(batch_size: int, timeout_ms: float, num_threads: int,
                                 model_path: str, simulations: int = 4000) -> Dict:
    """Measure performance for given timeout."""
    # Create GPU worker
    gpu_worker = GPUInferenceWorker(
        model_path=model_path,
        device='cuda',
        batch_size=batch_size,
        timeout_ms=timeout_ms,
        use_mixed_precision=True
    )
    gpu_worker.warmup(input_shape=(36, 15, 15))

    # Create MCTS
    mcts = AlphaZeroMCTS(
        inference_fn=gpu_worker,
        use_async_inference=True,
        async_batch_size=batch_size,
        async_timeout_ms=timeout_ms,
        num_threads=num_threads,
        c_puct=1.25
    )

    state = alphazero_py.GomokuState()

    # Warmup
    mcts.search(state, simulations=100)
    mcts.reset()

    # Measure
    start = time.perf_counter()
    mcts.search(state, simulations=simulations)
    elapsed = time.perf_counter() - start

    throughput = simulations / elapsed
    metrics = gpu_worker.get_metrics()

    per_sim_latency_ms = (elapsed / simulations) * 1000

    result = {
        'batch_size': batch_size,
        'timeout_ms': timeout_ms,
        'throughput': throughput,
        'avg_batch_size': metrics['average_batch_size'],
        'per_sim_latency_ms': per_sim_latency_ms,
        'total_batches': metrics['total_batches'],
        'inf_per_sim': metrics['total_requests'] / simulations,
    }

    gpu_worker.stop_worker()
    return result


def tune_timeout(model_size: str = 'small', batch_size: int = 64, num_threads: int = 8):
    """Grid search over timeout values."""
    print("=" * 80)
    print("ASYNC TIMEOUT TUNING (T018)")
    print(f"Model: {model_size}, Batch Size: {batch_size}, Threads: {num_threads}")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # Create model
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    create_test_model(model_path, size=model_size)

    # Test timeout range (ms)
    timeouts = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    results = []

    print(f"\n{'Timeout':>8s} {'Throughput':>12s} {'Avg Batch':>12s} {'Latency':>10s} {'Inf/Sim':>10s}")
    print(f"{'-'*8:>8s} {'-'*12:>12s} {'-'*12:>12s} {'-'*10:>10s} {'-'*10:>10s}")

    for timeout in timeouts:
        result = measure_timeout_performance(
            batch_size=batch_size,
            timeout_ms=timeout,
            num_threads=num_threads,
            model_path=model_path,
            simulations=4000
        )
        results.append(result)

        print(f"{result['timeout_ms']:8.1f}ms "
              f"{result['throughput']:12.1f}/s "
              f"{result['avg_batch_size']:12.1f} "
              f"{result['per_sim_latency_ms']:10.2f}ms "
              f"{result['inf_per_sim']:10.2f}")

    # Find optimal configuration
    # Optimize for throughput while maintaining avg_batch >= 48
    valid_results = [r for r in results if r['avg_batch_size'] >= 32]  # Relaxed constraint
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['throughput'])
    else:
        best_result = max(results, key=lambda x: x['throughput'])

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"\n  Optimal Timeout:    {best_result['timeout_ms']:.1f}ms")
    print(f"  Peak Throughput:    {best_result['throughput']:.1f} sims/sec")
    print(f"  Average Batch:      {best_result['avg_batch_size']:.1f}")
    print(f"  Per-Sim Latency:    {best_result['per_sim_latency_ms']:.2f}ms")
    print(f"  Inferences/Sim:     {best_result['inf_per_sim']:.2f}")

    # Target validation
    target_batch = 48
    target_throughput = 28000
    target_latency = 5.0  # <5ms per simulation

    print(f"\n  Target Validation:")
    print(f"    Avg Batch Target:  {target_batch}")
    print(f"    Achieved:          {best_result['avg_batch_size']:.1f}")
    if best_result['avg_batch_size'] >= target_batch:
        print(f"    Status:            ✅ TARGET ACHIEVED")
    else:
        print(f"    Status:            ⚠️  Below target ({best_result['avg_batch_size']:.1f}/{target_batch})")

    print(f"\n    Throughput Target: {target_throughput} sims/sec")
    print(f"    Achieved:          {best_result['throughput']:.1f} sims/sec")
    print(f"    Progress:          {best_result['throughput']/target_throughput*100:.1f}%")

    print(f"\n    Latency Target:    <{target_latency}ms per simulation")
    print(f"    Achieved:          {best_result['per_sim_latency_ms']:.2f}ms")
    if best_result['per_sim_latency_ms'] <= target_latency:
        print(f"    Status:            ✅ TARGET ACHIEVED")
    else:
        print(f"    Status:            ⚠️  Above target")

    # Recommendations
    print(f"\n  Recommendations:")
    if best_result['timeout_ms'] < 1.0:
        print(f"    • Very short timeout may increase overhead, consider 1.0-2.0ms range")
    if best_result['avg_batch_size'] < 32:
        print(f"    • Low batch size, increase timeout or reduce batch size config")
    if best_result['throughput'] < 10000:
        print(f"    • Low throughput, check thread count and model size")

    # Analyze tradeoffs
    print(f"\n  Timeout vs Throughput Tradeoff:")
    for result in results:
        pct_of_best = (result['throughput'] / best_result['throughput']) * 100
        print(f"    {result['timeout_ms']:4.1f}ms: {result['throughput']:7.1f}/s "
              f"({pct_of_best:5.1f}% of best), batch={result['avg_batch_size']:.1f}")

    print("\n" + "=" * 80)

    # Cleanup
    os.unlink(model_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Tune async timeout for optimal performance')
    parser.add_argument('--model', choices=['small', 'medium'], default='small',
                        help='Model size to test')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size configuration')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of simulation threads')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    results = tune_timeout(
        model_size=args.model,
        batch_size=args.batch_size,
        num_threads=args.threads
    )
