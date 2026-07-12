#!/usr/bin/env python3
"""
Thread Count Optimization for Async Inference (T019)

Test thread counts to find optimal parallel efficiency.
Target: ≥75% efficiency up to 12 threads, ≥30k sims/sec
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


def measure_thread_performance(num_threads: int, batch_size: int, timeout_ms: float,
                                model_path: str, simulations: int = 4000) -> Dict:
    """Measure performance for given thread count."""
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
        'num_threads': num_threads,
        'throughput': throughput,
        'avg_batch_size': metrics['average_batch_size'],
        'total_batches': metrics['total_batches'],
        'elapsed_time': elapsed,
    }

    gpu_worker.stop_worker()
    return result


def tune_thread_count(model_size: str = 'small', batch_size: int = 64, timeout_ms: float = 2.0):
    """Grid search over thread counts."""
    print("=" * 80)
    print("ASYNC THREAD COUNT OPTIMIZATION (T019)")
    print(f"Model: {model_size}, Batch Size: {batch_size}, Timeout: {timeout_ms}ms")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # Create model
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    create_test_model(model_path, size=model_size)

    # Test thread counts
    thread_counts = [1, 2, 4, 6, 8, 10, 12, 16]
    results = []

    print(f"\n{'Threads':>8s} {'Throughput':>12s} {'Speedup':>10s} {'Efficiency':>12s} {'Avg Batch':>12s}")
    print(f"{'-'*8:>8s} {'-'*12:>12s} {'-'*10:>10s} {'-'*12:>12s} {'-'*12:>12s}")

    baseline_throughput = None

    for num_threads in thread_counts:
        result = measure_thread_performance(
            num_threads=num_threads,
            batch_size=batch_size,
            timeout_ms=timeout_ms,
            model_path=model_path,
            simulations=4000
        )

        if baseline_throughput is None:
            baseline_throughput = result['throughput']
            speedup = 1.0
            efficiency = 1.0
        else:
            speedup = result['throughput'] / baseline_throughput
            efficiency = speedup / num_threads

        result['speedup'] = speedup
        result['efficiency'] = efficiency
        results.append(result)

        print(f"{num_threads:8d} "
              f"{result['throughput']:12.1f}/s "
              f"{speedup:10.2f}x "
              f"{efficiency*100:11.1f}% "
              f"{result['avg_batch_size']:12.1f}")

    # Find optimal configuration
    # Optimize for efficiency >= 75% first, then throughput
    valid_results = [r for r in results if r['efficiency'] >= 0.75]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['throughput'])
    else:
        best_result = max(results, key=lambda x: x['throughput'])

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"\n  Optimal Thread Count: {best_result['num_threads']}")
    print(f"  Peak Throughput:      {best_result['throughput']:.1f} sims/sec")
    print(f"  Speedup:              {best_result['speedup']:.2f}x")
    print(f"  Parallel Efficiency:  {best_result['efficiency']*100:.1f}%")
    print(f"  Average Batch:        {best_result['avg_batch_size']:.1f}")

    # Find saturation point (efficiency < 75%)
    saturation_point = None
    for result in results:
        if result['efficiency'] < 0.75:
            saturation_point = result['num_threads']
            break

    if saturation_point:
        print(f"\n  Saturation Point:     {saturation_point} threads (efficiency drops below 75%)")
        print(f"  Recommendation:       Use ≤{saturation_point-2} threads for optimal efficiency")
    else:
        print(f"\n  Saturation Point:     Not reached (efficiency stays ≥75% throughout)")

    # Target validation
    target_efficiency = 75  # ≥75% at 12 threads
    target_throughput = 30000

    twelve_thread_result = next((r for r in results if r['num_threads'] == 12), None)
    if twelve_thread_result:
        print(f"\n  12-Thread Validation:")
        print(f"    Throughput:         {twelve_thread_result['throughput']:.1f} sims/sec")
        print(f"    Efficiency:         {twelve_thread_result['efficiency']*100:.1f}%")
        if twelve_thread_result['efficiency']*100 >= target_efficiency:
            print(f"    Efficiency Status:  ✅ TARGET ACHIEVED (≥{target_efficiency}%)")
        else:
            print(f"    Efficiency Status:  ⚠️  Below target ({twelve_thread_result['efficiency']*100:.1f}%/{target_efficiency}%)")

    print(f"\n  Throughput Target:    {target_throughput} sims/sec")
    print(f"  Best Achieved:        {best_result['throughput']:.1f} sims/sec")
    print(f"  Progress:             {best_result['throughput']/target_throughput*100:.1f}%")

    if best_result['throughput'] >= target_throughput:
        print(f"  Status:               ✅ TARGET ACHIEVED")
    else:
        print(f"  Status:               ⚠️  Below target (needs further optimization)")

    # Recommendations
    print(f"\n  Recommendations:")
    if best_result['efficiency'] < 0.5:
        print(f"    • Low efficiency ({best_result['efficiency']*100:.1f}%), reduce thread count")
    if saturation_point and saturation_point < 12:
        print(f"    • Saturation at {saturation_point} threads suggests bottleneck (GPU or memory)")
    if best_result['avg_batch_size'] < batch_size * 0.5:
        print(f"    • Low batch utilization, consider reducing batch size or increasing timeout")
    if best_result['throughput'] < 10000:
        print(f"    • Very low throughput, check for bottlenecks beyond threading")

    # Analyze scaling
    print(f"\n  Thread Scaling Analysis:")
    for result in results:
        ideal_speedup = result['num_threads']
        actual_speedup = result['speedup']
        scaling_quality = (actual_speedup / ideal_speedup) * 100 if ideal_speedup > 0 else 0
        print(f"    {result['num_threads']:2d} threads: {result['throughput']:7.1f}/s "
              f"(speedup {actual_speedup:5.2f}x, {scaling_quality:5.1f}% of ideal)")

    print("\n" + "=" * 80)

    # Cleanup
    os.unlink(model_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Optimize thread count for async inference')
    parser.add_argument('--model', choices=['small', 'medium'], default='small',
                        help='Model size to test')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size configuration')
    parser.add_argument('--timeout', type=float, default=2.0,
                        help='Batch timeout in milliseconds')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    results = tune_thread_count(
        model_size=args.model,
        batch_size=args.batch_size,
        timeout_ms=args.timeout
    )
