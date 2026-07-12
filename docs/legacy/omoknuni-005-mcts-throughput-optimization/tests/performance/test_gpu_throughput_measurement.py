"""
GPU Throughput Measurement - Real Hardware Performance

Measures actual MCTS throughput with GPU neural network inference.
This provides ground truth data instead of CPU-based projections.

Hardware: RTX 3060 Ti (8GB VRAM)
Target: Measure real throughput with different thread counts
"""

import pytest
import numpy as np
import torch
import time
import tempfile
import os
from typing import Dict, List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from core.mcts import AlphaZeroMCTS
from neural.inference_worker import GPUInferenceWorker
from neural.model import AlphaZeroNet
import alphazero_py


def create_test_model(model_path: str, size: str = 'small') -> None:
    """Create test model - small for speed, medium for realism."""
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


def measure_gpu_inference_latency(model_path: str, batch_sizes: List[int]) -> Dict:
    """Measure pure GPU inference latency for different batch sizes."""
    print("\n" + "="*80)
    print("GPU INFERENCE LATENCY MEASUREMENT")
    print("="*80)

    # Load model to GPU (weights_only=False since we trust our own model)
    model = torch.load(model_path, weights_only=False)
    model = model.cuda()
    model.eval()

    results = {}

    print(f"\n{'Batch Size':>12s} {'Latency (ms)':>15s} {'Throughput':>15s}")
    print(f"{'-'*12:>12s} {'-'*15:>15s} {'-'*15:>15s}")

    for batch_size in batch_sizes:
        # Create dummy input
        dummy_input = torch.randn(batch_size, 36, 15, 15).cuda()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Measure
        torch.cuda.synchronize()
        times = []
        for _ in range(50):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time_ms = np.mean(times) * 1000
        throughput = batch_size / (avg_time_ms / 1000)

        results[batch_size] = {
            'latency_ms': avg_time_ms,
            'throughput': throughput
        }

        print(f"{batch_size:12d} {avg_time_ms:15.2f} {throughput:15.1f}")

    return results


def test_gpu_throughput_measurement():
    """Measure actual MCTS throughput with GPU inference."""
    print("\n" + "="*80)
    print("ACTUAL GPU THROUGHPUT MEASUREMENT")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("="*80)

    # Check GPU
    assert torch.cuda.is_available(), "CUDA not available"

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    try:
        print("\n[1/5] Creating small test model (4 blocks, 128 channels)...")
        create_test_model(model_path, size='small')

        # First measure pure GPU inference latency
        print("\n[2/5] Measuring pure GPU inference latency...")
        inference_results = measure_gpu_inference_latency(
            model_path,
            batch_sizes=[32, 64, 96, 128, 192, 256]
        )

        print("\n[3/5] Testing MCTS throughput with GPU...")

        # Test configurations
        configs = [
            {'threads': 1, 'batch_size': 32, 'timeout': 2.0},
            {'threads': 4, 'batch_size': 64, 'timeout': 2.0},
            {'threads': 8, 'batch_size': 128, 'timeout': 2.0},
            {'threads': 12, 'batch_size': 192, 'timeout': 1.5},
            {'threads': 16, 'batch_size': 256, 'timeout': 1.0},
        ]

        results = []

        print(f"\n{'Threads':>8s} {'Batch':>8s} {'Timeout':>10s} {'Throughput':>12s} {'Speedup':>10s} {'Avg Batch':>12s}")
        print(f"{'-'*8:>8s} {'-'*8:>8s} {'-'*10:>10s} {'-'*12:>12s} {'-'*10:>10s} {'-'*12:>12s}")

        baseline_throughput = None

        for config in configs:
            num_threads = config['threads']
            batch_size = config['batch_size']
            timeout_ms = config['timeout']

            # Create GPU worker
            gpu_worker = GPUInferenceWorker(
                model_path=model_path,
                device='cuda',  # ← REAL GPU
                batch_size=batch_size,
                timeout_ms=timeout_ms,
                use_mixed_precision=True  # FP16 for speed
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
            mcts.search(state, simulations=50)
            mcts.reset()

            # Measure
            start = time.perf_counter()
            mcts.search(state, simulations=500)
            elapsed = time.perf_counter() - start

            throughput = 500 / elapsed

            # Get metrics
            metrics = gpu_worker.get_metrics()
            avg_batch_size = metrics['average_batch_size']

            if baseline_throughput is None:
                baseline_throughput = throughput
                speedup = 1.0
            else:
                speedup = throughput / baseline_throughput

            results.append({
                'threads': num_threads,
                'batch_size': batch_size,
                'timeout': timeout_ms,
                'throughput': throughput,
                'speedup': speedup,
                'avg_batch_size': avg_batch_size,
                'metrics': metrics
            })

            print(f"{num_threads:8d} {batch_size:8d} {timeout_ms:10.1f} {throughput:12.1f} {speedup:10.2f}x {avg_batch_size:12.1f}")

            gpu_worker.stop_worker()

        print("\n[4/5] Detailed Analysis...")

        best_result = max(results, key=lambda x: x['throughput'])

        print(f"\n  Best Configuration:")
        print(f"    Threads:         {best_result['threads']}")
        print(f"    Batch size:      {best_result['batch_size']}")
        print(f"    Timeout:         {best_result['timeout']:.1f}ms")
        print(f"    Throughput:      {best_result['throughput']:.1f} sims/sec")
        print(f"    Speedup:         {best_result['speedup']:.2f}x")
        print(f"    Avg batch size:  {best_result['avg_batch_size']:.1f}")

        m = best_result['metrics']
        print(f"\n  GPU Worker Metrics:")
        print(f"    Total batches:       {m['total_batches']}")
        print(f"    Total requests:      {m['total_requests']}")
        print(f"    Inference rate:      {m['inference_rate']:.1f} pos/sec")
        print(f"    Avg inference time:  {m['total_inference_time']/m['total_batches']*1000:.2f}ms")

        # Calculate inferences per simulation
        inferences_per_sim = m['total_requests'] / 500
        print(f"    Inferences/sim:      {inferences_per_sim:.2f}")

        print("\n[5/5] GPU vs CPU Comparison...")

        # We know CPU achieved ~820 sims/sec (12 threads)
        cpu_throughput = 820
        gpu_throughput = best_result['throughput']
        gpu_speedup = gpu_throughput / cpu_throughput

        print(f"\n  CPU (12 threads):     {cpu_throughput:8.1f} sims/sec")
        print(f"  GPU (best config):    {gpu_throughput:8.1f} sims/sec")
        print(f"  GPU Speedup:          {gpu_speedup:8.2f}x")

        # Estimate what's possible with larger model
        print(f"\n  Inference Latency Analysis:")
        for bs in [64, 128, 192, 256]:
            if bs in inference_results:
                lat = inference_results[bs]['latency_ms']
                print(f"    Batch {bs:3d}: {lat:5.2f}ms")

        print("\n" + "="*80)
        print("PERFORMANCE VALIDATION")
        print("="*80)

        print(f"  Current (GPU, small model):  {gpu_throughput:8.1f} sims/sec")
        print(f"  Target (30k):               {30000:8.1f} sims/sec")
        print(f"  Progress:                   {gpu_throughput/30000*100:8.1f}%")

        if gpu_throughput >= 30000:
            print(f"\n  ✅ TARGET ACHIEVED!")
        elif gpu_throughput >= 20000:
            print(f"\n  ✅ EXCELLENT - Close to target, tuning will reach 30k")
        elif gpu_throughput >= 10000:
            print(f"\n  ✅ GOOD - On track, need optimization tuning")
        elif gpu_throughput >= 5000:
            print(f"\n  ⚠️  MODERATE - Need investigation + tuning")
        else:
            print(f"\n  ❌ BELOW EXPECTATIONS - Further analysis needed")

        print("\n" + "="*80)

        # Return for further analysis
        return {
            'best_throughput': gpu_throughput,
            'best_config': best_result,
            'inference_latency': inference_results,
            'all_results': results
        }

    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GPU THROUGHPUT MEASUREMENT SUITE")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("="*80)

    results = test_gpu_throughput_measurement()

    print("\n" + "="*80)
    print("MEASUREMENT COMPLETE")
    print(f"Best throughput: {results['best_throughput']:.1f} sims/sec")
    print("="*80 + "\n")
