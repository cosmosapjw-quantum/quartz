#!/usr/bin/env python3
"""
Diagnostic script for thread scaling regression investigation.

This script runs controlled experiments to understand why adding threads
to MCTS causes 5× performance degradation when using a shared GPUInferenceWorker.
"""

import sys
import time
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.mcts import AlphaZeroMCTS
from src.neural.inference_worker import GPUInferenceWorker
from src.neural.model import create_model_for_game
import alphazero_py


def create_worker():
    """Create GPU inference worker with proper initialization."""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Create and initialize model
    model = create_model_for_game('gomoku')
    model = model.to(device)
    model.eval()

    # Initialize lazy layers
    dummy_input = torch.zeros(1, 36, 15, 15, device=device)
    with torch.no_grad():
        _ = model(dummy_input)

    # Save model
    model_path = '/tmp/diagnostic_gomoku_model.pth'
    torch.save(model.state_dict(), model_path)

    # Create worker
    worker = GPUInferenceWorker(
        model_path=model_path,
        device=device,
        batch_size=32,
        timeout_ms=5.0,
        use_mixed_precision=True if device.startswith('cuda') else False
    )

    # Warmup
    worker.warmup(input_shape=(36, 15, 15))

    return worker


def run_experiment(worker, num_threads, num_simulations=800, label=""):
    """Run MCTS with specified thread count and measure performance."""
    print(f"\n{'='*60}")
    print(f"Experiment: {label}")
    print(f"Threads: {num_threads}, Simulations: {num_simulations}")
    print(f"{'='*60}")

    # Reset worker metrics if available
    if hasattr(worker, '_metrics'):
        worker._metrics['batch_sizes'].clear()
        worker._metrics['total_batches'] = 0
        worker._metrics['total_requests'] = 0

    # Create MCTS engine
    mcts = AlphaZeroMCTS(
        inference_fn=worker,
        c_puct=1.25,
        num_threads=num_threads,
        use_async_inference=True,
        async_batch_size=16,
        async_timeout_ms=10.0,
        enable_instrumentation=True
    )

    # Create initial state
    initial_state = alphazero_py.GomokuState(board_size=15)

    # Warmup
    print("Warming up...")
    mcts.search(initial_state, simulations=100)
    mcts.reset()

    # Reset metrics after warmup
    if hasattr(worker, '_metrics'):
        worker._metrics['batch_sizes'].clear()

    # Run benchmark
    print("Running benchmark...")
    start_time = time.perf_counter()
    mcts.search(initial_state, simulations=num_simulations)
    elapsed_time = time.perf_counter() - start_time

    # Calculate metrics
    throughput = num_simulations / elapsed_time

    # Get worker metrics
    metrics = worker.get_metrics()
    avg_batch_size = metrics.get('average_batch_size', 0.0)
    batch_sizes = list(worker._metrics.get('batch_sizes', []))
    total_batches = len(batch_sizes)

    # Get MCTS stats
    stats = mcts.get_statistics()

    # Print results
    print(f"\nResults:")
    print(f"  Elapsed time: {elapsed_time:.3f}s")
    print(f"  Throughput: {throughput:.1f} sims/sec")
    print(f"  Total batches: {total_batches}")
    print(f"  Avg batch size: {avg_batch_size:.1f}")
    if batch_sizes:
        print(f"  Min batch size: {min(batch_sizes)}")
        print(f"  Max batch size: {max(batch_sizes)}")
        print(f"  First 10 batches: {batch_sizes[:10]}")

    # Check for coordinator issues
    if 'coordinator_started' in stats:
        print(f"  Coordinator started: {stats['coordinator_started']}")
        print(f"  Coordinator searches: {stats.get('coordinator_searches', 0)}")

    # Instrumentation metrics
    if 'instrumentation' in stats:
        inst = stats['instrumentation']
        if inst:
            print(f"\nInstrumentation:")
            for key, value in inst.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value}")

    # Cleanup
    mcts.close()

    return {
        'throughput': throughput,
        'elapsed_time': elapsed_time,
        'avg_batch_size': avg_batch_size,
        'total_batches': total_batches,
        'batch_sizes': batch_sizes,
        'stats': stats
    }


def main():
    """Run diagnostic experiments."""
    print("="*80)
    print("THREAD SCALING DIAGNOSTIC - OPTION B INVESTIGATION")
    print("="*80)

    # Create single shared worker (mimics test fixture behavior)
    print("\nCreating shared GPUInferenceWorker...")
    worker = create_worker()
    print("Worker created successfully")

    results = {}

    # Experiment 1: Single thread (baseline)
    results['1_thread'] = run_experiment(
        worker,
        num_threads=1,
        num_simulations=800,
        label="Baseline - 1 Thread"
    )

    # Wait a bit between experiments
    time.sleep(2)

    # Experiment 2: Two threads (first degradation)
    results['2_threads'] = run_experiment(
        worker,
        num_threads=2,
        num_simulations=800,
        label="Degradation Test - 2 Threads (REUSING SAME WORKER)"
    )

    # Wait a bit
    time.sleep(2)

    # Experiment 3: Eight threads
    results['8_threads'] = run_experiment(
        worker,
        num_threads=8,
        num_simulations=800,
        label="Degradation Test - 8 Threads (REUSING SAME WORKER)"
    )

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - WORKER REUSE IMPACT")
    print("="*80)
    print(f"\n1 thread (first use):    {results['1_thread']['throughput']:>8.1f} sims/sec")
    print(f"2 threads (second use):  {results['2_threads']['throughput']:>8.1f} sims/sec")
    print(f"8 threads (third use):   {results['8_threads']['throughput']:>8.1f} sims/sec")

    degradation_2t = results['1_thread']['throughput'] / results['2_threads']['throughput']
    degradation_8t = results['1_thread']['throughput'] / results['8_threads']['throughput']

    print(f"\nDegradation:")
    print(f"  1→2 threads: {degradation_2t:.2f}× SLOWER")
    print(f"  1→8 threads: {degradation_8t:.2f}× SLOWER")

    # Batch size comparison
    print(f"\nBatch Sizes:")
    print(f"  1 thread:  {results['1_thread']['avg_batch_size']:.1f} avg")
    print(f"  2 threads: {results['2_threads']['avg_batch_size']:.1f} avg")
    print(f"  8 threads: {results['8_threads']['avg_batch_size']:.1f} avg")

    # Cleanup
    worker.stop_worker()
    Path('/tmp/diagnostic_gomoku_model.pth').unlink(missing_ok=True)

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
