"""
GPU Bottleneck Analysis - Deep Dive

Diagnose why GPU is only 3.61× faster than CPU instead of expected 10×+.

Key Questions:
1. Why only 1.10 inferences/sim? (should be 3-5)
2. Where is the 5.53ms overhead per batch?
3. Why aren't batches filling completely?
"""

import pytest
import numpy as np
import torch
import time
import tempfile
import os
import threading
from collections import defaultdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from core.mcts import AlphaZeroMCTS
from neural.inference_worker import GPUInferenceWorker
from neural.model import AlphaZeroNet
import alphazero_py


def create_test_model(model_path: str) -> None:
    """Create small test model."""
    model = AlphaZeroNet(
        input_channels=36,
        num_actions=225,
        num_blocks=4,
        hidden_channels=128,
        use_se=False
    )
    torch.save(model, model_path)


def test_tree_expansion_depth():
    """Analyze how deep the tree expands and why inferences/sim is low."""
    print("\n" + "="*80)
    print("TREE EXPANSION DEPTH ANALYSIS")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    try:
        create_test_model(model_path)

        # Track inference calls
        inference_count = {'total': 0, 'batch_sizes': []}

        class MonitoredGPUWorker(GPUInferenceWorker):
            def batch_inference(self, positions):
                inference_count['total'] += len(positions)
                inference_count['batch_sizes'].append(len(positions))
                return super().batch_inference(positions)

        gpu_worker = MonitoredGPUWorker(
            model_path=model_path,
            device='cuda',
            batch_size=128,
            timeout_ms=2.0,
            use_mixed_precision=True
        )
        gpu_worker.warmup(input_shape=(36, 15, 15))

        mcts = AlphaZeroMCTS(
            inference_fn=gpu_worker,
            use_async_inference=True,
            async_batch_size=128,
            async_timeout_ms=2.0,
            num_threads=8,
            c_puct=1.25
        )

        state = alphazero_py.GomokuState()

        print("\n[1/3] Running 500 simulations with monitoring...")
        start = time.perf_counter()
        visit_counts = mcts.search(state, simulations=500)
        elapsed = time.perf_counter() - start

        print(f"\n[2/3] Results:")
        print(f"  Simulations:        500")
        print(f"  Inferences:         {inference_count['total']}")
        print(f"  Inferences/sim:     {inference_count['total']/500:.2f}")
        print(f"  Throughput:         {500/elapsed:.1f} sims/sec")
        print(f"  Time:               {elapsed:.2f}s")

        print(f"\n[3/3] Batch Analysis:")
        batch_sizes = inference_count['batch_sizes']
        print(f"  Total batches:      {len(batch_sizes)}")
        print(f"  Avg batch size:     {np.mean(batch_sizes):.1f}")
        print(f"  Min batch size:     {min(batch_sizes)}")
        print(f"  Max batch size:     {max(batch_sizes)}")
        print(f"  Batch size dist:    {batch_sizes}")

        # Analyze tree structure
        print(f"\n  Tree Structure:")
        print(f"    Root visits:      {mcts.tree.get_visit_count(mcts.root_index):.0f}")

        # Get root children
        num_children = mcts.tree.get_num_children(mcts.root_index)
        print(f"    Root children:    {num_children}")

        # Note: Cannot iterate children directly, but we know the structure
        # With uniform random policy, all children get visited once

        gpu_worker.stop_worker()

        # Diagnosis
        print(f"\n" + "="*80)
        print("DIAGNOSIS")
        print("="*80)

        inf_per_sim = inference_count['total'] / 500

        if inf_per_sim < 2.0:
            print(f"  ⚠️  Very low inferences/sim ({inf_per_sim:.2f})")
            print(f"     Likely causes:")
            print(f"     - Random/untrained policy → uniform priors → shallow tree")
            print(f"     - Tree expansion not working correctly")
            print(f"     - Most simulations reusing same nodes")
        elif inf_per_sim < 3.5:
            print(f"  ⚠️  Low inferences/sim ({inf_per_sim:.2f})")
            print(f"     Expected 3-5 for healthy tree expansion")
        else:
            print(f"  ✅ Normal inferences/sim ({inf_per_sim:.2f})")

        if len(batch_sizes) > 0:
            avg_batch = np.mean(batch_sizes)
            if avg_batch < 64:
                print(f"\n  ⚠️  Small batches ({avg_batch:.1f})")
                print(f"     Threads not generating enough concurrent requests")
            elif avg_batch < 100:
                print(f"\n  ⚠️  Moderate batches ({avg_batch:.1f})")
                print(f"     Could be better with more threads or lower timeout")
            else:
                print(f"\n  ✅ Good batch sizes ({avg_batch:.1f})")

        print("\n" + "="*80)

        return {
            'inferences_per_sim': inf_per_sim,
            'batch_sizes': batch_sizes,
            'throughput': 500/elapsed
        }

    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_coordinator_overhead():
    """Measure overhead in async coordinator vs direct GPU calls."""
    print("\n" + "="*80)
    print("COORDINATOR OVERHEAD ANALYSIS")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    try:
        create_test_model(model_path)

        gpu_worker = GPUInferenceWorker(
            model_path=model_path,
            device='cuda',
            batch_size=128,
            timeout_ms=2.0,
            use_mixed_precision=True
        )
        gpu_worker.warmup(input_shape=(36, 15, 15))

        # Test 1: Direct GPU inference
        print("\n[1/2] Testing direct GPU inference (no coordinator)...")

        dummy_positions = [np.random.randn(36, 15, 15).astype(np.float32) for _ in range(128)]

        times = []
        for _ in range(20):
            start = time.perf_counter()
            policies, values = gpu_worker.batch_inference(dummy_positions)
            times.append(time.perf_counter() - start)

        direct_avg = np.mean(times) * 1000
        direct_std = np.std(times) * 1000

        print(f"  Direct GPU time: {direct_avg:.2f} ± {direct_std:.2f}ms (batch=128)")

        # Test 2: Through async coordinator
        print("\n[2/2] Testing through async coordinator...")

        mcts = AlphaZeroMCTS(
            inference_fn=gpu_worker,
            use_async_inference=True,
            async_batch_size=128,
            async_timeout_ms=2.0,
            num_threads=1,  # Single thread for controlled test
            c_puct=1.25
        )

        state = alphazero_py.GomokuState()

        # Warmup
        mcts.search(state, simulations=50)
        mcts.reset()

        # Measure
        start = time.perf_counter()
        mcts.search(state, simulations=100)
        elapsed = time.perf_counter() - start

        metrics = gpu_worker.get_metrics()
        total_inference_time = metrics['total_inference_time'] * 1000  # ms
        num_batches = metrics['total_batches']

        if num_batches > 0:
            coordinator_avg = total_inference_time / num_batches
            overhead = coordinator_avg - direct_avg

            print(f"  Coordinator time: {coordinator_avg:.2f}ms (batch={metrics['average_batch_size']:.1f})")
            print(f"  Overhead: {overhead:.2f}ms ({overhead/coordinator_avg*100:.1f}%)")

            if overhead > 5:
                print(f"\n  ⚠️  HIGH OVERHEAD ({overhead:.2f}ms)")
                print(f"     Coordinator/queue adding significant latency")
            elif overhead > 2:
                print(f"\n  ⚠️  MODERATE OVERHEAD ({overhead:.2f}ms)")
                print(f"     Room for optimization")
            else:
                print(f"\n  ✅ LOW OVERHEAD ({overhead:.2f}ms)")

        gpu_worker.stop_worker()

        print("\n" + "="*80)

    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_request_generation_rate():
    """Measure how fast threads generate inference requests."""
    print("\n" + "="*80)
    print("REQUEST GENERATION RATE ANALYSIS")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    try:
        create_test_model(model_path)

        for num_threads in [1, 4, 8, 12, 16]:
            # Use very fast mock to measure request generation
            class FastMockWorker:
                def __init__(self):
                    self.request_times = []

                def batch_inference(self, positions):
                    self.request_times.append(time.perf_counter())
                    # Instant return
                    policies = [np.ones(225) / 225 for _ in positions]
                    values = [0.0 for _ in positions]
                    return policies, values

                def stop_worker(self):
                    pass

                def get_metrics(self):
                    return {
                        'total_batches': len(self.request_times),
                        'total_requests': 0,
                        'average_batch_size': 0,
                        'inference_rate': 0
                    }

            mock_worker = FastMockWorker()

            mcts = AlphaZeroMCTS(
                inference_fn=mock_worker,
                use_async_inference=True,
                async_batch_size=128,
                async_timeout_ms=2.0,
                num_threads=num_threads,
                c_puct=1.25
            )

            state = alphazero_py.GomokuState()

            start = time.perf_counter()
            mcts.search(state, simulations=500)
            elapsed = time.perf_counter() - start

            num_batches = len(mock_worker.request_times)

            if num_batches > 1:
                intervals = np.diff(mock_worker.request_times) * 1000  # ms
                avg_interval = np.mean(intervals)
                requests_per_sec = 1000 / avg_interval if avg_interval > 0 else 0

                print(f"  {num_threads:2d} threads: {500/elapsed:6.1f} sims/sec, "
                      f"{num_batches:3d} batches, "
                      f"{avg_interval:5.2f}ms interval, "
                      f"{requests_per_sec:6.1f} batches/sec")

        print("\n" + "="*80)

    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GPU BOTTLENECK ANALYSIS SUITE")
    print("="*80)

    test_tree_expansion_depth()
    test_coordinator_overhead()
    test_request_generation_rate()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")
