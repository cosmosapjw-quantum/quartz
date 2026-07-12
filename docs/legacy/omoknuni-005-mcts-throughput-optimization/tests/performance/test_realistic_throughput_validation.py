"""
Realistic Throughput Validation with Real GPUInferenceWorker

This test validates the actual throughput achieved with the direct GPU batching fix.
Uses real components (no mocks) to measure production performance.

Target: ≥10,000 sims/sec with direct GPU batching (baseline before tuning)
Ultimate target: 30,000+ sims/sec after tuning (T017-T020)
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


def create_test_model(model_path: str) -> None:
    """Create a small test model for realistic performance testing."""
    model = AlphaZeroNet(
        input_channels=36,
        num_actions=225,
        num_blocks=4,  # Small but realistic
        hidden_channels=128,
        use_se=False
    )

    # Save full model
    torch.save(model, model_path)


def test_baseline_throughput_with_direct_gpu_batching():
    """Measure baseline throughput with real GPUInferenceWorker and direct GPU batching."""
    print("\n" + "="*80)
    print("REALISTIC THROUGHPUT VALIDATION - Direct GPU Batching")
    print("="*80)

    # Create temporary model
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    try:
        print("\n[1/5] Creating test model...")
        create_test_model(model_path)
        print(f"  ✓ Model created: {model_path}")

        print("\n[2/5] Initializing GPUInferenceWorker...")
        gpu_worker = GPUInferenceWorker(
            model_path=model_path,
            device='cpu',  # Use CPU for testing (GPU would be faster)
            batch_size=32,  # Default batch size
            timeout_ms=2.0,  # Default timeout
            use_mixed_precision=False  # CPU doesn't support FP16
        )

        # Warmup
        print("  ✓ Warming up GPU worker...")
        gpu_worker.warmup(input_shape=(36, 15, 15))
        print(f"  ✓ GPUInferenceWorker ready")

        # Verify it has batch_inference method
        assert hasattr(gpu_worker, 'batch_inference'), \
            "GPUInferenceWorker must have batch_inference() method"
        print(f"  ✓ batch_inference() method detected - will use direct GPU batching mode")

        print("\n[3/5] Creating AlphaZeroMCTS with async mode...")
        mcts = AlphaZeroMCTS(
            inference_fn=gpu_worker,
            use_async_inference=True,
            async_batch_size=32,
            async_timeout_ms=2.0,
            c_puct=1.25
        )
        print(f"  ✓ MCTS initialized with async inference")

        # Initial state
        state = alphazero_py.GomokuState()

        print("\n[4/5] Running throughput benchmarks...")

        # Test configurations
        test_configs = [
            {"name": "Warmup", "sims": 50, "runs": 1},
            {"name": "Small", "sims": 100, "runs": 3},
            {"name": "Medium", "sims": 200, "runs": 3},
            {"name": "Large", "sims": 500, "runs": 3},
        ]

        results = []

        for config in test_configs:
            name = config["name"]
            num_sims = config["sims"]
            num_runs = config["runs"]

            throughputs = []

            for run in range(num_runs):
                # Reset state
                state = alphazero_py.GomokuState()

                # Run search
                start_time = time.perf_counter()
                visit_counts = mcts.search(state, simulations=num_sims)
                elapsed_time = time.perf_counter() - start_time

                # Calculate throughput
                throughput = num_sims / elapsed_time
                throughputs.append(throughput)

                # Verify correctness
                root_visits = mcts.tree.get_visit_count(mcts.root_index)
                assert root_visits == num_sims, \
                    f"Expected {num_sims} visits, got {root_visits}"

                # Reset tree for next run
                mcts.reset()

            # Calculate statistics
            avg_throughput = np.mean(throughputs)
            std_throughput = np.std(throughputs)
            min_throughput = np.min(throughputs)
            max_throughput = np.max(throughputs)

            results.append({
                'name': name,
                'sims': num_sims,
                'avg_throughput': avg_throughput,
                'std': std_throughput,
                'min': min_throughput,
                'max': max_throughput
            })

            if name != "Warmup":
                print(f"  {name:8s} ({num_sims:3d} sims × {num_runs} runs): "
                      f"{avg_throughput:7.1f} ± {std_throughput:5.1f} sims/sec "
                      f"[{min_throughput:.1f} - {max_throughput:.1f}]")

        print("\n[5/5] Performance Analysis...")

        # Get best result (largest configuration)
        best_result = results[-1]  # Large config
        best_throughput = best_result['avg_throughput']

        print(f"\n  Best Configuration:")
        print(f"    Simulations: {best_result['sims']}")
        print(f"    Throughput:  {best_throughput:.1f} ± {best_result['std']:.1f} sims/sec")

        # Get GPU worker metrics
        metrics = gpu_worker.get_metrics()
        print(f"\n  GPU Worker Metrics:")
        print(f"    Total batches:     {metrics['total_batches']}")
        print(f"    Total requests:    {metrics['total_requests']}")
        print(f"    Avg batch size:    {metrics['average_batch_size']:.1f}")
        print(f"    Batch size p90:    {metrics.get('batch_size_p90', 0):.1f}")
        print(f"    Inference rate:    {metrics['inference_rate']:.1f} pos/sec")

        # Performance targets
        print(f"\n" + "="*80)
        print("PERFORMANCE VALIDATION RESULTS")
        print("="*80)

        targets = [
            ("Current (baseline)", best_throughput, "Measured"),
            ("Target (direct batching)", 10000, "T014.5 goal"),
            ("Target (after tuning)", 30000, "T017-T020 goal"),
        ]

        for name, value, note in targets:
            if "Current" in name:
                print(f"  {name:25s}: {value:8.1f} sims/sec ({note})")
            else:
                gap = value - best_throughput
                pct = (best_throughput / value) * 100
                print(f"  {name:25s}: {value:8.1f} sims/sec ({note}) - gap: {gap:.0f} ({pct:.1f}%)")

        print("\n  Performance Analysis:")

        if best_throughput >= 30000:
            print(f"    ✅ EXCELLENT - Exceeded ultimate target!")
        elif best_throughput >= 10000:
            print(f"    ✅ GOOD - Met direct batching target, tuning can reach 30k")
        elif best_throughput >= 5000:
            print(f"    ⚠️  MODERATE - 50%+ of target, needs investigation + tuning")
        else:
            print(f"    ❌ BELOW TARGET - Further optimization needed")

        print(f"\n  Bottleneck Analysis:")

        # Analyze where time is spent
        if metrics['average_batch_size'] < 16:
            print(f"    ⚠️  Low batch size ({metrics['average_batch_size']:.1f}) - increase async_batch_size")

        if metrics.get('timeout_compliance_rate', 1.0) < 0.95:
            print(f"    ⚠️  Timeout violations - adjust async_timeout_ms")

        if metrics['total_batches'] > 0:
            avg_batch_time = metrics['total_inference_time'] / metrics['total_batches']
            if avg_batch_time > 0.005:  # >5ms per batch
                print(f"    ⚠️  Slow inference ({avg_batch_time*1000:.1f}ms/batch) - model size or device issue")

        # Recommendations
        print(f"\n  Recommendations:")
        if best_throughput < 10000:
            print(f"    1. Increase async_batch_size from 32 to 64-128 (T017)")
            print(f"    2. Optimize async_timeout_ms to 1-2ms (T018)")
            print(f"    3. Tune thread count to 8-12 threads (T019)")
            print(f"    4. Use GPU device instead of CPU for 5-10× speedup")

        print("\n" + "="*80)

        # Cleanup
        gpu_worker.stop_worker()

        # Assert we're making progress toward target
        assert best_throughput > 500, \
            f"Expected >500 sims/sec, got {best_throughput:.1f}"

        return best_throughput, metrics

    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_batch_size_scaling():
    """Test how throughput scales with different batch sizes."""
    print("\n" + "="*80)
    print("BATCH SIZE SCALING ANALYSIS")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    try:
        create_test_model(model_path)

        batch_sizes = [8, 16, 32, 64, 128]
        results = []

        print(f"\nTesting batch sizes: {batch_sizes}")
        print(f"{'Batch Size':>12s} {'Throughput':>12s} {'Batch Size (actual)':>20s}")
        print(f"{'-'*12:>12s} {'-'*12:>12s} {'-'*20:>20s}")

        for batch_size in batch_sizes:
            gpu_worker = GPUInferenceWorker(
                model_path=model_path,
                device='cpu',
                batch_size=batch_size,
                timeout_ms=2.0,
                use_mixed_precision=False
            )
            gpu_worker.warmup(input_shape=(36, 15, 15))

            mcts = AlphaZeroMCTS(
                inference_fn=gpu_worker,
                use_async_inference=True,
                async_batch_size=batch_size,
                async_timeout_ms=2.0,
                c_puct=1.25
            )

            state = alphazero_py.GomokuState()

            # Run benchmark
            start = time.perf_counter()
            mcts.search(state, simulations=200)
            elapsed = time.perf_counter() - start

            throughput = 200 / elapsed

            metrics = gpu_worker.get_metrics()
            actual_batch_size = metrics['average_batch_size']

            results.append({
                'batch_size': batch_size,
                'throughput': throughput,
                'actual_batch_size': actual_batch_size
            })

            print(f"{batch_size:12d} {throughput:12.1f} {actual_batch_size:20.1f}")

            gpu_worker.stop_worker()

        # Find optimal
        best = max(results, key=lambda x: x['throughput'])
        print(f"\n  Optimal batch size: {best['batch_size']} → {best['throughput']:.1f} sims/sec")

        print("\n" + "="*80)

        return results

    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("REALISTIC THROUGHPUT VALIDATION SUITE")
    print("Measuring production performance with real GPUInferenceWorker")
    print("="*80)

    # Run baseline test
    throughput, metrics = test_baseline_throughput_with_direct_gpu_batching()

    # Run scaling analysis
    batch_results = test_batch_size_scaling()

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print(f"Baseline throughput: {throughput:.1f} sims/sec")
    print("="*80 + "\n")
