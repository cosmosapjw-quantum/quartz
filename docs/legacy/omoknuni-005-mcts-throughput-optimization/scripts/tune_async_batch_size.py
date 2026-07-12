#!/usr/bin/env python3
"""
Batch Size Tuning for Async Inference (T017)

Grid search over batch sizes to find optimal configuration for 30k+ sims/sec.
Measures throughput, GPU utilization, and batch latency.
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
    elif size == 'medium':
        model = AlphaZeroNet(
            input_channels=36,
            num_actions=225,
            num_blocks=10,
            hidden_channels=192,
            use_se=True
        )
    else:  # production
        model = AlphaZeroNet(
            input_channels=36,
            num_actions=225,
            num_blocks=20,
            hidden_channels=256,
            use_se=True
        )

    torch.save(model, model_path)


def measure_throughput(batch_size: int, num_threads: int, timeout_ms: float,
                       model_path: str, simulations: int = 4000) -> Dict:
    """Measure MCTS throughput for given batch size."""
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

    result = {
        'batch_size': batch_size,
        'timeout_ms': timeout_ms,
        'num_threads': num_threads,
        'simulations': simulations,
        'elapsed_time': elapsed,
        'throughput': throughput,
        'avg_batch_size': metrics['average_batch_size'],
        'total_batches': metrics['total_batches'],
        'total_requests': metrics['total_requests'],
        'inf_per_sim': metrics['total_requests'] / simulations,
        'avg_inference_time_ms': (metrics['total_inference_time'] / metrics['total_batches']) * 1000 if metrics['total_batches'] > 0 else 0,
    }

    gpu_worker.stop_worker()
    return result


def tune_batch_size(model_size: str = 'small', num_threads: int = 8, timeout_ms: float = 2.0):
    """Grid search over batch sizes."""
    print("=" * 80)
    print("ASYNC BATCH SIZE TUNING (T017)")
    print(f"Model: {model_size}, Threads: {num_threads}, Timeout: {timeout_ms}ms")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # Create model
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    create_test_model(model_path, size=model_size)

    # Test batch sizes
    batch_sizes = [16, 32, 48, 64, 96, 128]
    results = []

    print(f"\n{'Batch':>6s} {'Throughput':>12s} {'Avg Batch':>12s} {'Inf/Sim':>10s} {'Latency':>10s}")
    print(f"{'-'*6:>6s} {'-'*12:>12s} {'-'*12:>12s} {'-'*10:>10s} {'-'*10:>10s}")

    for batch_size in batch_sizes:
        result = measure_throughput(
            batch_size=batch_size,
            num_threads=num_threads,
            timeout_ms=timeout_ms,
            model_path=model_path,
            simulations=4000
        )
        results.append(result)

        print(f"{result['batch_size']:6d} "
              f"{result['throughput']:12.1f}/s "
              f"{result['avg_batch_size']:12.1f} "
              f"{result['inf_per_sim']:10.2f} "
              f"{result['avg_inference_time_ms']:10.2f}ms")

    # Find optimal configuration
    best_result = max(results, key=lambda x: x['throughput'])

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"\n  Optimal Batch Size: {best_result['batch_size']}")
    print(f"  Peak Throughput:    {best_result['throughput']:.1f} sims/sec")
    print(f"  Average Batch:      {best_result['avg_batch_size']:.1f}")
    print(f"  Inferences/Sim:     {best_result['inf_per_sim']:.2f}")
    print(f"  Inference Latency:  {best_result['avg_inference_time_ms']:.2f}ms")

    # Calculate 90th percentile throughput
    sorted_results = sorted(results, key=lambda x: x['throughput'])
    p90_index = int(len(sorted_results) * 0.9)
    p90_result = sorted_results[p90_index]

    print(f"\n  90th Percentile:")
    print(f"    Batch Size:       {p90_result['batch_size']}")
    print(f"    Throughput:       {p90_result['throughput']:.1f} sims/sec")

    # Target validation
    target_throughput = 25000  # ≥25k target from spec
    target_gpu_util = 60  # ≥60% target from spec

    print(f"\n  Target Validation:")
    print(f"    Throughput Target: {target_throughput} sims/sec")
    print(f"    Achieved:          {best_result['throughput']:.1f} sims/sec")
    print(f"    Progress:          {best_result['throughput']/target_throughput*100:.1f}%")

    if best_result['throughput'] >= target_throughput:
        print(f"    Status:            ✅ TARGET ACHIEVED")
    else:
        print(f"    Status:            ⚠️  Below target (needs further optimization)")

    # Recommendations
    print(f"\n  Recommendations:")
    if best_result['avg_batch_size'] < best_result['batch_size'] * 0.5:
        print(f"    • Reduce batch size to {best_result['batch_size'] // 2} (low utilization)")
    if best_result['inf_per_sim'] < 2.0:
        print(f"    • Tree is shallow (inf/sim={best_result['inf_per_sim']:.2f}), test with deeper positions")
    if best_result['avg_inference_time_ms'] > 5.0:
        print(f"    • High latency ({best_result['avg_inference_time_ms']:.1f}ms), consider smaller model")
    if best_result['throughput'] < 10000:
        print(f"    • Throughput very low, check for bottlenecks in selection/backup")

    print("\n" + "=" * 80)

    # Cleanup
    os.unlink(model_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Tune async batch size for optimal performance')
    parser.add_argument('--model', choices=['small', 'medium', 'production'], default='small',
                        help='Model size to test')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of simulation threads')
    parser.add_argument('--timeout', type=float, default=2.0,
                        help='Batch timeout in milliseconds')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    results = tune_batch_size(
        model_size=args.model,
        num_threads=args.threads,
        timeout_ms=args.timeout
    )
