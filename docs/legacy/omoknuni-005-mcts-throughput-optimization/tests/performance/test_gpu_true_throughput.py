"""
GPU True Throughput - Multiple Searches Without Reset

Strategy: Run multiple searches on the same tree (no reset) to force depth.
As tree deepens, more inferences/sim → higher request rate → true GPU performance.

This simulates a real game scenario where the tree is reused and deepened.
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


def test_gpu_true_throughput_multiple_searches():
    """Measure GPU throughput with tree reuse (multiple searches, no reset)."""
    print("\n" + "="*80)
    print("GPU TRUE THROUGHPUT - MULTIPLE SEARCHES (NO RESET)")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("="*80)

    assert torch.cuda.is_available(), "CUDA not available"

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    try:
        print("\n[1/3] Creating test model...")
        create_test_model(model_path)

        # Track across all searches
        total_inferences = {'count': 0, 'by_search': []}

        class MonitoredGPUWorker(GPUInferenceWorker):
            def batch_inference(self, positions):
                total_inferences['count'] += len(positions)
                return super().batch_inference(positions)

        gpu_worker = MonitoredGPUWorker(
            model_path=model_path,
            device='cuda',
            batch_size=256,
            timeout_ms=0.5,  # Very low timeout for high request rate
            use_mixed_precision=True
        )
        gpu_worker.warmup(input_shape=(36, 15, 15))

        mcts = AlphaZeroMCTS(
            inference_fn=gpu_worker,
            use_async_inference=True,
            async_batch_size=256,
            async_timeout_ms=0.5,
            num_threads=20,  # Max threads for maximum request rate
            c_puct=1.25
        )

        state = alphazero_py.GomokuState()

        print("\n[2/3] Running multiple searches without reset...")
        print(f"\n{'Search':>8s} {'Sims':>8s} {'Throughput':>12s} {'Inf/Sim':>10s} {'Cumul Inf':>12s} {'Tree Nodes':>12s}")
        print(f"{'-'*8:>8s} {'-'*8:>8s} {'-'*12:>12s} {'-'*10:>10s} {'-'*12:>12s} {'-'*12:>12s}")

        results = []
        cumulative_sims = 0

        for search_num in range(1, 6):  # 5 searches
            # Increase sims each search to deepen tree
            num_sims = 1000 * search_num

            start_inf = total_inferences['count']
            start_time = time.perf_counter()

            visit_counts = mcts.search(state, simulations=num_sims)

            elapsed = time.perf_counter() - start_time
            search_inferences = total_inferences['count'] - start_inf
            inf_per_sim = search_inferences / num_sims if num_sims > 0 else 0
            throughput = num_sims / elapsed

            cumulative_sims += num_sims
            tree_nodes = mcts.tree.get_node_count()

            total_inferences['by_search'].append(search_inferences)

            results.append({
                'search': search_num,
                'sims': num_sims,
                'throughput': throughput,
                'inf_per_sim': inf_per_sim,
                'inferences': search_inferences,
                'tree_nodes': tree_nodes,
                'elapsed': elapsed
            })

            print(f"{search_num:8d} {num_sims:8d} {throughput:12.1f} {inf_per_sim:10.2f} {total_inferences['count']:12d} {tree_nodes:12d}")

            # DON'T reset - let tree deepen

        print("\n[3/3] Analysis...")

        print(f"\n  Search Progression:")
        for r in results:
            print(f"    Search {r['search']}: {r['inf_per_sim']:.2f} inf/sim, {r['throughput']:8.1f} sims/sec")

        # Find search with highest inf/sim (deepest tree)
        deepest = max(results, key=lambda x: x['inf_per_sim'])
        fastest = max(results, key=lambda x: x['throughput'])

        print(f"\n  Deepest Tree (Search {deepest['search']}):")
        print(f"    Inferences/sim:  {deepest['inf_per_sim']:.2f}")
        print(f"    Throughput:      {deepest['throughput']:.1f} sims/sec")
        print(f"    Inference rate:  {deepest['inferences']/deepest['elapsed']:.1f} inf/sec")
        print(f"    Tree nodes:      {deepest['tree_nodes']}")

        print(f"\n  Fastest (Search {fastest['search']}):")
        print(f"    Inferences/sim:  {fastest['inf_per_sim']:.2f}")
        print(f"    Throughput:      {fastest['throughput']:.1f} sims/sec")
        print(f"    Inference rate:  {fastest['inferences']/fastest['elapsed']:.1f} inf/sec")

        # Get GPU metrics
        metrics = gpu_worker.get_metrics()
        print(f"\n  GPU Worker Metrics (Cumulative):")
        print(f"    Total batches:      {metrics['total_batches']}")
        print(f"    Total requests:     {metrics['total_requests']}")
        print(f"    Avg batch size:     {metrics['average_batch_size']:.1f}")
        print(f"    Inference rate:     {metrics['inference_rate']:.1f} pos/sec")

        gpu_worker.stop_worker()

        print("\n" + "="*80)
        print("PERFORMANCE VALIDATION")
        print("="*80)

        # Use deepest tree performance as best estimate
        best_throughput = deepest['throughput']
        best_inf_per_sim = deepest['inf_per_sim']
        best_inf_rate = deepest['inferences'] / deepest['elapsed']

        print(f"\n  Best Performance (Deep Tree):")
        print(f"    Throughput:        {best_throughput:8.1f} sims/sec")
        print(f"    Inferences/sim:    {best_inf_per_sim:8.2f}")
        print(f"    Inference rate:    {best_inf_rate:8.1f} inf/sec")

        # Extrapolate to trained policy
        if best_inf_per_sim > 1.5:
            trained_inf_per_sim = 3.5  # Typical for trained policy
            # If we can sustain this inference rate with trained policy:
            extrapolated_sims = best_inf_rate / trained_inf_per_sim
            print(f"\n  Extrapolated (Trained Policy, {trained_inf_per_sim} inf/sim):")
            print(f"    Sustained inf rate: {best_inf_rate:8.1f} inf/sec")
            print(f"    Throughput:         {extrapolated_sims:8.1f} sims/sec")

        print(f"\n  vs Target:")
        print(f"    Current:           {best_throughput:8.1f} sims/sec")
        print(f"    Target:            {30000:8.1f} sims/sec")
        print(f"    Progress:          {best_throughput/30000*100:8.1f}%")

        if best_throughput >= 30000:
            print(f"\n  ✅ TARGET ACHIEVED!")
        elif best_throughput >= 20000:
            print(f"\n  ✅ EXCELLENT - Very close to target!")
        elif best_throughput >= 10000:
            print(f"\n  ✅ GOOD - Significant progress")
        elif best_throughput >= 5000:
            print(f"\n  ⚠️  MODERATE - Need optimization")
        else:
            print(f"\n  ❌ BELOW EXPECTATIONS")

        # Key insight
        if best_inf_rate > 10000:
            print(f"\n  💡 KEY INSIGHT:")
            print(f"     GPU can sustain {best_inf_rate:.0f} inf/sec")
            print(f"     With trained policy (3-5 inf/sim), this enables:")
            print(f"       • 3 inf/sim → {best_inf_rate/3:.0f} sims/sec")
            print(f"       • 4 inf/sim → {best_inf_rate/4:.0f} sims/sec")
            print(f"       • 5 inf/sim → {best_inf_rate/5:.0f} sims/sec")

        print("\n" + "="*80)

        return results

    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GPU TRUE THROUGHPUT TEST")
    print("="*80)

    results = test_gpu_true_throughput_multiple_searches()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")
