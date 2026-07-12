#!/usr/bin/env python3
"""
Wall-Clock Validation - End-to-End Performance Measurement

Measures actual wall-clock time for MCTS operations without profiling overhead.
This provides ground-truth performance baselines for comparison.

Usage:
    # Quick validation
    python scripts/wall_clock_validation.py --quick

    # Full validation with multiple runs
    python scripts/wall_clock_validation.py --runs 10 --simulations 1600

    # Compare with/without profiling
    python scripts/wall_clock_validation.py --compare-profiling

Author: MCTS Performance Team
Date: 2025-10-15
"""

import argparse
import sys
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mcts_py
    import alphazero_py
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class WallClockValidator:
    """Wall-clock performance validation"""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run_validation(self) -> int:
        """Run complete validation"""
        print("=" * 80)
        print("WALL-CLOCK VALIDATION")
        print("=" * 80)

        results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'simulations': self.args.simulations,
                'runs': self.args.runs,
                'warmup': self.args.warmup,
            },
            'measurements': [],
        }

        # Warmup run
        if self.args.warmup:
            print("\n🔥 Warmup run...")
            self._single_run(warmup=True)
            print("   ✅ Warmup complete")

        # Main measurements
        print(f"\n📊 Running {self.args.runs} measurement runs...")
        print(f"   Simulations per run: {self.args.simulations}")

        wall_times = []
        throughputs = []

        for run in range(1, self.args.runs + 1):
            print(f"\n[{run}/{self.args.runs}] ", end='', flush=True)

            result = self._single_run(warmup=False)

            wall_times.append(result['wall_clock_time'])
            throughputs.append(result['throughput'])
            results['measurements'].append(result)

            print(f"⏱️  {result['wall_clock_time']:.3f}s → {result['throughput']:.1f} sims/sec")

        # Statistical analysis
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS")
        print("=" * 80)

        print(f"\n📈 Wall-Clock Time ({self.args.runs} runs):")
        print(f"   Mean:   {statistics.mean(wall_times):.3f}s")
        print(f"   Median: {statistics.median(wall_times):.3f}s")
        print(f"   StdDev: {statistics.stdev(wall_times) if len(wall_times) > 1 else 0:.3f}s")
        print(f"   Min:    {min(wall_times):.3f}s")
        print(f"   Max:    {max(wall_times):.3f}s")

        print(f"\n🚀 Throughput ({self.args.runs} runs):")
        print(f"   Mean:   {statistics.mean(throughputs):.1f} sims/sec")
        print(f"   Median: {statistics.median(throughputs):.1f} sims/sec")
        print(f"   StdDev: {statistics.stdev(throughputs) if len(throughputs) > 1 else 0:.1f} sims/sec")
        print(f"   Min:    {min(throughputs):.1f} sims/sec")
        print(f"   Max:    {max(throughputs):.1f} sims/sec")

        # Variability analysis
        cv_time = (statistics.stdev(wall_times) / statistics.mean(wall_times) * 100) if len(wall_times) > 1 else 0
        cv_throughput = (statistics.stdev(throughputs) / statistics.mean(throughputs) * 100) if len(throughputs) > 1 else 0

        print(f"\n📊 Coefficient of Variation:")
        print(f"   Time:       {cv_time:.2f}%")
        print(f"   Throughput: {cv_throughput:.2f}%")

        if cv_time < 5.0:
            print("   ✅ Low variability - results are stable")
        elif cv_time < 10.0:
            print("   ⚠️  Moderate variability - consider more runs")
        else:
            print("   ❌ High variability - results may be unreliable")

        # Target analysis
        target_throughput = 8000  # From CLAUDE.md
        mean_throughput = statistics.mean(throughputs)
        target_pct = (mean_throughput / target_throughput) * 100

        print(f"\n🎯 Target Analysis:")
        print(f"   Target:  {target_throughput} sims/sec")
        print(f"   Current: {mean_throughput:.1f} sims/sec")
        print(f"   Progress: {target_pct:.1f}% of target")

        if target_pct >= 100:
            print("   ✅ Target achieved!")
        elif target_pct >= 75:
            print("   ⚠️  Close to target - optimization needed")
        else:
            print("   ❌ Below target - significant optimization needed")

        # Export results
        results['statistics'] = {
            'wall_clock_time': {
                'mean': statistics.mean(wall_times),
                'median': statistics.median(wall_times),
                'stdev': statistics.stdev(wall_times) if len(wall_times) > 1 else 0,
                'min': min(wall_times),
                'max': max(wall_times),
                'cv_percent': cv_time,
            },
            'throughput': {
                'mean': statistics.mean(throughputs),
                'median': statistics.median(throughputs),
                'stdev': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                'min': min(throughputs),
                'max': max(throughputs),
                'cv_percent': cv_throughput,
            },
            'target_analysis': {
                'target_throughput': target_throughput,
                'current_throughput': mean_throughput,
                'target_percent': target_pct,
            }
        }

        output_file = f"wall_clock_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        # Compare with profiling if requested
        if self.args.compare_profiling:
            print("\n" + "=" * 80)
            print("PROFILING OVERHEAD COMPARISON")
            print("=" * 80)
            self._compare_with_profiling(statistics.mean(throughputs))

        print("\n" + "=" * 80)
        print("✅ VALIDATION COMPLETE")
        print("=" * 80)

        return 0

    def _single_run(self, warmup: bool = False) -> Dict[str, Any]:
        """Run a single measurement"""
        import numpy as np
        import torch
        from src.neural.model import create_random_model
        from src.core.dlpack_inference_bridge import DLPackInferenceBridge

        # Setup MCTS components
        state = alphazero_py.GomokuState(board_size=15)
        action_space_size = state.get_action_space_size()
        tree = mcts_py.create_test_tree(100000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        vl_manager = mcts_py.create_test_virtual_loss_manager(tree)

        # Create ContinuousSimulationRunner (correct runner with zero-copy optimization)
        runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, vl_manager)
        root = tree.get_root_index()

        # Create AsyncInferenceQueue
        queue = mcts_py.AsyncInferenceQueue()

        # REAL GPU inference (like unified profiler)
        model = create_random_model('gomoku', seed=42)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        inference_bridge = DLPackInferenceBridge(
            model=model,
            device=device,
            use_mixed_precision=True,
            use_cuda_graphs=False  # Disable to avoid fallback on partial batches
        )

        if not warmup:
            # Warmup GPU on first run
            inference_bridge.warmup(batch_size=64, game_type='gomoku')

        def batch_inference_fn(features_batch, board_sizes, num_planes_list):
            """Real GPU batch callback"""
            return inference_bridge.batch_inference_features(
                features_batch, board_sizes, num_planes_list
            )

        callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)

        # Create and start coordinator
        coordinator = mcts_py.BatchInferenceCoordinator()
        coordinator.start(queue, callback, batch_size=64, timeout_ms=5.0)

        # Disable profiling for pure wall-clock measurement
        profiler = mcts_py.EnhancedProfiler.instance()
        profiler.set_enabled(False)

        try:
            # Wall-clock measurement
            start_time = time.perf_counter()

            # Run continuous simulations (uses make/unmake pattern)
            successes = runner.run_continuous(state, root, queue, self.args.simulations)

            end_time = time.perf_counter()
            wall_clock_time = end_time - start_time

            throughput = successes / wall_clock_time if wall_clock_time > 0 else 0

            result = {
                'simulations': self.args.simulations,
                'successful': successes,
                'wall_clock_time': wall_clock_time,
                'throughput': throughput,
                'tree_nodes': tree.get_node_count(),
                'avg_time_per_sim_ms': (wall_clock_time / successes * 1000) if successes > 0 else 0,
            }

            return result

        finally:
            # Always stop coordinator
            coordinator.stop()

    def _compare_with_profiling(self, baseline_throughput: float):
        """Compare performance with profiling enabled"""
        import numpy as np

        print("\n🔬 Running with profiling enabled...")

        # Setup
        state = alphazero_py.GomokuState(board_size=15)
        action_space_size = state.get_action_space_size()
        tree = mcts_py.create_test_tree(100000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        vl_manager = mcts_py.create_test_virtual_loss_manager(tree)

        # Create ContinuousSimulationRunner
        runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, vl_manager)
        root = tree.get_root_index()

        # Create AsyncInferenceQueue
        queue = mcts_py.AsyncInferenceQueue()

        # Dummy batch inference
        def batch_inference_fn(features_batch, board_sizes, num_planes_list):
            """Dummy batch callback"""
            results = []
            for _ in features_batch:
                policy = np.ones(action_space_size, dtype=np.float32) / action_space_size
                value = 0.0
                results.append((policy.tolist(), value))
            return results

        callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)

        # Create and start coordinator
        coordinator = mcts_py.BatchInferenceCoordinator()
        coordinator.start(queue, callback, batch_size=64, timeout_ms=5.0)

        # Enable profiling
        profiler = mcts_py.EnhancedProfiler.instance()
        profiler.set_enabled(True)
        profiler.start_session("overhead_test")

        try:
            start_time = time.perf_counter()

            # Run continuous simulations
            successes = runner.run_continuous(state, root, queue, self.args.simulations)

            end_time = time.perf_counter()
            wall_clock_time = end_time - start_time

            profiler.stop_session()

            throughput_with_profiling = successes / wall_clock_time if wall_clock_time > 0 else 0
            overhead_percent = ((baseline_throughput - throughput_with_profiling) / baseline_throughput) * 100

            print(f"\n📊 Profiling Overhead:")
            print(f"   Baseline:         {baseline_throughput:.1f} sims/sec (profiling OFF)")
            print(f"   With profiling:   {throughput_with_profiling:.1f} sims/sec (profiling ON)")
            print(f"   Overhead:         {overhead_percent:.2f}%")

            if overhead_percent < 5.0:
                print("   ✅ Low overhead - profiling is efficient")
            elif overhead_percent < 10.0:
                print("   ⚠️  Moderate overhead - acceptable for debugging")
            else:
                print("   ❌ High overhead - consider reducing profiling level")

        finally:
            # Always stop coordinator
            coordinator.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Wall-clock validation for MCTS performance"
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
        '--warmup',
        action='store_true',
        default=True,
        help="Run warmup before measurements (default: True)"
    )

    parser.add_argument(
        '--no-warmup',
        dest='warmup',
        action='store_false',
        help="Skip warmup run"
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help="Quick test (100 sims, 3 runs)"
    )

    parser.add_argument(
        '--compare-profiling',
        action='store_true',
        help="Compare performance with/without profiling"
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.simulations = 100
        args.runs = 3

    validator = WallClockValidator(args)
    return validator.run_validation()


if __name__ == '__main__':
    sys.exit(main())
