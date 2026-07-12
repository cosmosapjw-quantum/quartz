"""
GPU Throughput with Deep Tree Expansion

Test GPU performance with high simulation counts that force deep tree expansion.
This represents realistic conditions where the tree is explored deeply.

Target: With deep trees (3-5 inferences/sim), measure true GPU throughput.
"""

import pytest
import numpy as np
import torch
import time
import tempfile
import os

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


def test_deep_tree_gpu_throughput():
    """Test GPU throughput with deep tree expansion (high simulation count)."""
    print("\n" + "="*80)
    print("GPU THROUGHPUT WITH DEEP TREE EXPANSION")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("="*80)

    assert torch.cuda.is_available(), "CUDA not available"

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    try:
        print("\n[1/4] Creating test model...")
        create_test_model(model_path)

        # Track inference calls
        inference_count = {'total': 0}

        class MonitoredGPUWorker(GPUInferenceWorker):
            def batch_inference(self, positions):
                inference_count['total'] += len(positions)
                return super().batch_inference(positions)

        print("\n[2/4] Testing with increasing simulation counts...")

        configs = [
            {'name': 'Warmup', 'sims': 500, 'threads': 8},
            {'name': 'Medium', 'sims': 2000, 'threads': 12},
            {'name': 'Deep', 'sims': 5000, 'threads': 16},
            {'name': 'Very Deep', 'sims': 10000, 'threads': 16},
        ]

        results = []

        print(f"\n{'Config':>12s} {'Sims':>8s} {'Threads':>8s} {'Throughput':>12s} {'Inf/Sim':>10s} {'Inf Rate':>12s}")
        print(f"{'-'*12:>12s} {'-'*8:>8s} {'-'*8:>8s} {'-'*12:>12s} {'-'*10:>10s} {'-'*12:>12s}")

        for config in configs:
            name = config['name']
            num_sims = config['sims']
            num_threads = config['threads']

            # Reset counter
            inference_count['total'] = 0

            gpu_worker = MonitoredGPUWorker(
                model_path=model_path,
                device='cuda',
                batch_size=256,
                timeout_ms=1.0,  # Lower timeout for deep trees
                use_mixed_precision=True
            )
            gpu_worker.warmup(input_shape=(36, 15, 15))

            mcts = AlphaZeroMCTS(
                inference_fn=gpu_worker,
                use_async_inference=True,
                async_batch_size=256,
                async_timeout_ms=1.0,
                num_threads=num_threads,
                c_puct=1.25
            )

            state = alphazero_py.GomokuState()

            # Run
            start = time.perf_counter()
            visit_counts = mcts.search(state, simulations=num_sims)
            elapsed = time.perf_counter() - start

            throughput = num_sims / elapsed
            inferences = inference_count['total']
            inf_per_sim = inferences / num_sims if num_sims > 0 else 0
            inf_rate = inferences / elapsed

            if name != 'Warmup':
                results.append({
                    'name': name,
                    'sims': num_sims,
                    'threads': num_threads,
                    'throughput': throughput,
                    'inf_per_sim': inf_per_sim,
                    'inf_rate': inf_rate,
                    'elapsed': elapsed
                })

                print(f"{name:>12s} {num_sims:8d} {num_threads:8d} {throughput:12.1f} {inf_per_sim:10.2f} {inf_rate:12.1f}")

            gpu_worker.stop_worker()

        print("\n[3/4] Analysis...")

        # Find configuration with best inf/sim (deepest tree)
        deepest = max(results, key=lambda x: x['inf_per_sim'])
        fastest = max(results, key=lambda x: x['throughput'])

        print(f"\n  Deepest Tree Configuration:")
        print(f"    Simulations:     {deepest['sims']}")
        print(f"    Inferences/sim:  {deepest['inf_per_sim']:.2f}")
        print(f"    Throughput:      {deepest['throughput']:.1f} sims/sec")
        print(f"    Inference rate:  {deepest['inf_rate']:.1f} inf/sec")

        print(f"\n  Fastest Configuration:")
        print(f"    Simulations:     {fastest['sims']}")
        print(f"    Inferences/sim:  {fastest['inf_per_sim']:.2f}")
        print(f"    Throughput:      {fastest['throughput']:.1f} sims/sec")
        print(f"    Inference rate:  {fastest['inf_rate']:.1f} inf/sec")

        print("\n[4/4] Scaling Analysis...")

        print(f"\n  Inf/Sim Progression:")
        for r in results:
            print(f"    {r['sims']:5d} sims → {r['inf_per_sim']:.2f} inf/sim")

        print(f"\n  Throughput vs Tree Depth:")
        for r in results:
            print(f"    {r['inf_per_sim']:.2f} inf/sim → {r['throughput']:8.1f} sims/sec")

        print("\n" + "="*80)
        print("PERFORMANCE VALIDATION")
        print("="*80)

        # Best realistic performance (deepest tree)
        best_throughput = deepest['throughput']
        best_inf_per_sim = deepest['inf_per_sim']

        print(f"\n  Deep Tree Performance:")
        print(f"    Throughput:      {best_throughput:8.1f} sims/sec")
        print(f"    Inferences/sim:  {best_inf_per_sim:8.2f}")
        print(f"    Inference rate:  {deepest['inf_rate']:8.1f} inf/sec")

        # Extrapolate to trained policy (higher inf/sim)
        if best_inf_per_sim > 1.5:
            # With trained policy, inf/sim would be higher (3-5)
            trained_inf_per_sim = 3.5
            extrapolated = best_throughput * (best_inf_per_sim / trained_inf_per_sim)
            print(f"\n  Extrapolated (trained policy, {trained_inf_per_sim} inf/sim):")
            print(f"    Throughput:      {extrapolated:8.1f} sims/sec")

        print(f"\n  vs Target:")
        print(f"    Current:         {best_throughput:8.1f} sims/sec")
        print(f"    Target:          {30000:8.1f} sims/sec")
        print(f"    Progress:        {best_throughput/30000*100:8.1f}%")

        if best_throughput >= 30000:
            print(f"\n  ✅ TARGET ACHIEVED!")
        elif best_throughput >= 15000:
            print(f"\n  ✅ EXCELLENT - Close to target!")
        elif best_throughput >= 8000:
            print(f"\n  ✅ GOOD - Substantial progress")
        elif best_throughput >= 4000:
            print(f"\n  ⚠️  MODERATE - Need more optimization")
        else:
            print(f"\n  ❌ BELOW EXPECTATIONS")

        print("\n" + "="*80)

        return results

    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEEP TREE GPU THROUGHPUT TEST")
    print("="*80)

    results = test_deep_tree_gpu_throughput()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")
