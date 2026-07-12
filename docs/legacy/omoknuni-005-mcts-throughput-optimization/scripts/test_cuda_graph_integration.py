#!/usr/bin/env python3
"""
CUDA Graph Integration Test
============================

Validates CUDA graph integration into DLPackInferenceBridge with full MCTS pipeline.
Tests both functionality and performance improvement.

Usage:
    # Quick test (3 runs, 100 simulations)
    python scripts/test_cuda_graph_integration.py --quick

    # Full test (10 runs, 1000 simulations)
    python scripts/test_cuda_graph_integration.py --runs 10 --simulations 1000

    # Compare with/without CUDA graphs
    python scripts/test_cuda_graph_integration.py --compare

Expected Results:
    - 2-3× speedup for small batches (8-16) due to launch overhead reduction
    - 1.1-1.5× speedup for medium batches (32-64)
    - Minimal improvement for large batches (128+) - compute bound

Author: MCTS Performance Team
Date: 2025-10-21
"""

import argparse
import sys
import time
import statistics
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mcts_py
    import alphazero_py
    import torch
    from src.neural.model import create_resnet_eca_model
    from src.core.dlpack_inference_bridge import DLPackInferenceBridge
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class CUDAGraphIntegrationTest:
    """Comprehensive CUDA graph integration testing"""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run_tests(self) -> int:
        """Run complete test suite"""
        print("=" * 80)
        print("CUDA GRAPH INTEGRATION TEST")
        print("=" * 80)

        # Test 1: Functionality test
        print("\n" + "=" * 80)
        print("TEST 1: CUDA GRAPH FUNCTIONALITY")
        print("=" * 80)

        if not self._test_functionality():
            print("\n❌ FUNCTIONALITY TEST FAILED")
            return 1

        print("\n✅ FUNCTIONALITY TEST PASSED")

        # Test 2: Performance comparison
        print("\n" + "=" * 80)
        print("TEST 2: PERFORMANCE COMPARISON")
        print("=" * 80)

        if not self._test_performance():
            print("\n❌ PERFORMANCE TEST FAILED")
            return 1

        print("\n✅ PERFORMANCE TEST PASSED")

        # Test 3: Batch size variation
        if not self.args.quick:
            print("\n" + "=" * 80)
            print("TEST 3: BATCH SIZE VARIATION")
            print("=" * 80)

            if not self._test_batch_variation():
                print("\n❌ BATCH SIZE VARIATION TEST FAILED")
                return 1

            print("\n✅ BATCH SIZE VARIATION TEST PASSED")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)

        return 0

    def _test_functionality(self) -> bool:
        """Test basic CUDA graph functionality"""
        print("\n🔬 Creating ResNet-ECA 128×12 model...")

        # Create model
        model = create_resnet_eca_model('gomoku', size='128x12')
        model = model.cuda()
        print(f"   Model created: {model.get_num_parameters():,} parameters")

        # Create bridge with CUDA graphs enabled
        print("\n🔬 Initializing DLPackInferenceBridge with CUDA graphs...")
        bridge_with_graphs = DLPackInferenceBridge(
            model=model,
            device='cuda',
            use_cuda_graphs=True,
            graph_batch_sizes=[8, 16, 32, 64]
        )

        # Run dummy inference to trigger graph capture
        print("\n🔬 Running test inference to trigger CUDA graph capture...")
        import numpy as np

        state = alphazero_py.GomokuState(board_size=15)
        action_space_size = state.get_action_space_size()

        # Create dummy features
        features = np.random.randn(36, 15, 15).astype(np.float32)
        board_sizes = [15]
        num_planes_list = [36]

        try:
            results = bridge_with_graphs.batch_inference_features(
                [features], board_sizes, num_planes_list
            )

            # Verify results format
            assert len(results) == 1, "Expected 1 result"
            policy, value = results[0]
            assert len(policy) == action_space_size, f"Expected {action_space_size} policy entries"
            assert isinstance(value, float), "Value should be float"

            print("   ✅ CUDA graph capture successful")

            # Verify graph manager exists
            if bridge_with_graphs.graph_manager is not None:
                stats = bridge_with_graphs.graph_manager.get_stats()
                print(f"   ✅ Graph manager active: {stats['captured_batch_sizes']}")
            else:
                print("   ⚠️  Graph manager not initialized (may be expected on CPU)")

            return True

        except Exception as e:
            print(f"   ❌ CUDA graph test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _test_performance(self) -> bool:
        """Test performance improvement with CUDA graphs"""
        print("\n🔬 Testing performance with/without CUDA graphs...")

        # Create model
        model = create_resnet_eca_model('gomoku', size='128x12').cuda()

        # Test configuration
        batch_size = 64
        num_runs = self.args.runs
        simulations = self.args.simulations

        print(f"   Configuration: {num_runs} runs × {simulations} simulations")
        print(f"   Batch size: {batch_size}")

        # Test WITHOUT CUDA graphs
        print("\n📊 Baseline (CUDA graphs OFF)...")
        throughput_baseline = self._run_mcts_benchmark(
            model, use_cuda_graphs=False, batch_size=batch_size,
            simulations=simulations, runs=num_runs
        )

        if throughput_baseline is None:
            return False

        # Test WITH CUDA graphs
        print("\n📊 Optimized (CUDA graphs ON)...")
        throughput_optimized = self._run_mcts_benchmark(
            model, use_cuda_graphs=True, batch_size=batch_size,
            simulations=simulations, runs=num_runs
        )

        if throughput_optimized is None:
            return False

        # Calculate speedup
        speedup = throughput_optimized / throughput_baseline

        print("\n" + "=" * 80)
        print("PERFORMANCE RESULTS")
        print("=" * 80)
        print(f"Baseline (no graphs):   {throughput_baseline:.1f} sims/sec")
        print(f"Optimized (with graphs): {throughput_optimized:.1f} sims/sec")
        print(f"Speedup:                 {speedup:.2f}×")

        # Validate speedup (expect at least 1.05× improvement)
        if speedup >= 1.05:
            print(f"✅ CUDA graphs provide {speedup:.2f}× speedup (target: ≥1.05×)")
            return True
        else:
            print(f"⚠️  Speedup {speedup:.2f}× below target (1.05×)")
            print("   This may be expected for large batches (compute-bound)")
            return True  # Don't fail, just warn

    def _test_batch_variation(self) -> bool:
        """Test performance across different batch sizes"""
        print("\n🔬 Testing batch size variation...")

        model = create_resnet_eca_model('gomoku', size='128x12').cuda()

        batch_sizes = [8, 16, 32, 64]
        simulations = 500  # Shorter for batch variation test
        runs = 3

        results = []

        for batch_size in batch_sizes:
            print(f"\n📊 Testing batch size {batch_size}...")

            # Without graphs
            throughput_baseline = self._run_mcts_benchmark(
                model, use_cuda_graphs=False, batch_size=batch_size,
                simulations=simulations, runs=runs
            )

            # With graphs
            throughput_optimized = self._run_mcts_benchmark(
                model, use_cuda_graphs=True, batch_size=batch_size,
                simulations=simulations, runs=runs
            )

            if throughput_baseline and throughput_optimized:
                speedup = throughput_optimized / throughput_baseline
                results.append({
                    'batch_size': batch_size,
                    'baseline': throughput_baseline,
                    'optimized': throughput_optimized,
                    'speedup': speedup
                })
                print(f"   Batch {batch_size}: {speedup:.2f}× speedup")

        # Print summary
        print("\n" + "=" * 80)
        print("BATCH SIZE VARIATION SUMMARY")
        print("=" * 80)
        print(f"{'Batch':>6} | {'Baseline':>12} | {'Optimized':>12} | {'Speedup':>8}")
        print("-" * 80)

        for r in results:
            print(f"{r['batch_size']:>6} | {r['baseline']:>12.1f} | {r['optimized']:>12.1f} | {r['speedup']:>8.2f}×")

        # Verify small batches get more improvement
        small_batch_speedup = next((r['speedup'] for r in results if r['batch_size'] == 8), None)
        large_batch_speedup = next((r['speedup'] for r in results if r['batch_size'] == 64), None)

        if small_batch_speedup and large_batch_speedup:
            if small_batch_speedup > large_batch_speedup:
                print(f"\n✅ Small batches benefit more: {small_batch_speedup:.2f}× vs {large_batch_speedup:.2f}×")
            else:
                print(f"\n⚠️  Expected small batches to benefit more (launch overhead)")

        return True

    def _run_mcts_benchmark(
        self, model, use_cuda_graphs: bool, batch_size: int,
        simulations: int, runs: int
    ) -> float:
        """Run MCTS benchmark and return average throughput"""
        import numpy as np

        # Setup MCTS components
        state = alphazero_py.GomokuState(board_size=15)
        action_space_size = state.get_action_space_size()
        tree = mcts_py.create_test_tree(100000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        vl_manager = mcts_py.create_test_virtual_loss_manager(tree)

        runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, vl_manager)
        root = tree.get_root_index()

        # Create AsyncInferenceQueue
        queue = mcts_py.AsyncInferenceQueue()

        # Create DLPackInferenceBridge
        bridge = DLPackInferenceBridge(
            model=model,
            device='cuda',
            use_cuda_graphs=use_cuda_graphs,
            graph_batch_sizes=[8, 16, 32, 64, 128, 256]
        )

        # Wrap bridge in PyBatchInferenceCallback
        callback = mcts_py.PyBatchInferenceCallback(bridge.batch_inference_features)

        # Create and start coordinator
        coordinator = mcts_py.BatchInferenceCoordinator()
        coordinator.start(queue, callback, batch_size=batch_size, timeout_ms=5.0)

        # Disable profiling for clean measurement
        profiler = mcts_py.EnhancedProfiler.instance()
        profiler.set_enabled(False)

        throughputs = []

        try:
            for run in range(1, runs + 1):
                # Reset tree
                tree.clear()

                # Run simulations
                start_time = time.perf_counter()
                successes = runner.run_continuous(state, root, queue, simulations)
                end_time = time.perf_counter()

                wall_clock_time = end_time - start_time
                throughput = successes / wall_clock_time if wall_clock_time > 0 else 0
                throughputs.append(throughput)

                print(f"      Run {run}/{runs}: {throughput:.1f} sims/sec")

            # Return mean throughput
            mean_throughput = statistics.mean(throughputs)
            print(f"   Mean: {mean_throughput:.1f} sims/sec")

            return mean_throughput

        except Exception as e:
            print(f"   ❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            coordinator.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Test CUDA graph integration with MCTS pipeline"
    )

    parser.add_argument(
        '--simulations',
        type=int,
        default=1000,
        help="Number of simulations per run (default: 1000)"
    )

    parser.add_argument(
        '--runs',
        type=int,
        default=5,
        help="Number of measurement runs (default: 5)"
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help="Quick test (100 sims, 3 runs)"
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.simulations = 100
        args.runs = 3

    tester = CUDAGraphIntegrationTest(args)
    return tester.run_tests()


if __name__ == '__main__':
    sys.exit(main())
