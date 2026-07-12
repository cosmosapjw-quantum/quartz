#!/usr/bin/env python3
"""
Profiling Campaign - Parameter Sweep and Analysis

Runs comprehensive profiling with parameter sweeps using UnifiedProfiler
to get full coverage: 295 C++ metrics + Python profiling + bottleneck analysis.

Usage:
    # Quick test (few trials)
    python scripts/profiling_campaign.py --quick-test

    # Full campaign (all parameter combinations)
    python scripts/profiling_campaign.py --output results/campaign_001

    # Custom parameter ranges
    python scripts/profiling_campaign.py \
        --simulations 800,1600,3200 \
        --threads 2,4,8,12 \
        --output results/custom

Features:
    - Uses UnifiedProfiler for comprehensive metrics (295 C++, Python profiling)
    - Automated bottleneck detection (state cloning, OpenMP, thread idle, CAS)
    - Exports JSON, Chrome Trace, Markdown for each trial
    - Parameter sweep to find optimal configuration

Author: MCTS Performance Team
Date: 2025-10-15
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from itertools import product
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Import UnifiedProfiler to leverage all profiling features
    from scripts.unified_profiler import UnifiedProfiler
    import mcts_py
    import alphazero_py
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you've built the C++ extensions:")
    print("   pip install -e .")
    sys.exit(1)


class ProfilingCampaign:
    """Orchestrates parameter sweep profiling campaign"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Parse parameter ranges
        self.simulations = self._parse_range(args.simulations)
        self.threads = self._parse_range(args.threads)
        self.batch_sizes = self._parse_range(args.batch_sizes)

        # Results storage
        self.results = []
        self.campaign_start = datetime.now()

    def _parse_range(self, range_str: str) -> List[int]:
        """Parse comma-separated range string to list of ints"""
        return [int(x.strip()) for x in range_str.split(',')]

    def run_campaign(self) -> int:
        """Run complete profiling campaign using UnifiedProfiler"""
        print("=" * 80)
        print("COMPREHENSIVE PROFILING CAMPAIGN")
        print("Using UnifiedProfiler for Full Metrics Coverage")
        print("=" * 80)

        # Generate all parameter combinations with repetitions
        param_combinations = list(product(
            self.simulations,
            self.threads,
            self.batch_sizes,
            range(1, self.args.repetitions + 1)  # Repetition numbers
        ))

        total_trials = len(param_combinations)
        print(f"\n📊 Campaign Configuration:")
        print(f"   Simulations: {self.simulations}")
        print(f"   Threads: {self.threads}")
        print(f"   Batch sizes: {self.batch_sizes}")
        print(f"   Repetitions: {self.args.repetitions} per config")
        print(f"   Total trials: {total_trials}")
        print(f"   Output: {self.output_dir}")

        print(f"\n   ✅ Comprehensive profiling enabled (via UnifiedProfiler):")
        print(f"      - 295 C++ metrics (state cloning, OpenMP, thread idle, CAS, mutex)")
        print(f"      - Python profiling (GIL, inference, thread coordination)")
        print(f"      - Automated bottleneck detection")
        print(f"      - Chrome Trace for timeline visualization")
        print(f"      - JSON + Markdown reports per trial")

        if not self.args.yes:
            response = input(f"\n⚠️  Run {total_trials} comprehensive profiling trials? (y/N): ")
            if response.lower() != 'y':
                print("❌ Campaign cancelled")
                return 1

        # Run each trial
        print("\n" + "=" * 80)
        print("RUNNING TRIALS")
        print("=" * 80)

        for i, (sims, threads, batch, rep) in enumerate(param_combinations, 1):
            rep_label = f" (rep {rep}/{self.args.repetitions})" if self.args.repetitions > 1 else ""
            print(f"\n[{i}/{total_trials}] Trial: sims={sims}, threads={threads}, batch={batch}{rep_label}")

            result = self.run_single_trial(
                trial_id=i,
                simulations=sims,
                num_threads=threads,
                batch_size=batch
            )
            result['repetition'] = rep  # Track which repetition this is

            self.results.append(result)

            # Print summary
            if result['success']:
                print(f"   ✅ Completed in {result['wall_clock_time']:.2f}s")
                print(f"      Throughput: {result['throughput']:.1f} sims/sec")
                print(f"      Tree nodes: {result['tree_nodes']}")
            else:
                print(f"   ❌ Failed: {result['error']}")

        # Analyze results
        print("\n" + "=" * 80)
        print("ANALYZING RESULTS")
        print("=" * 80)

        self.analyze_results()
        self.export_results()

        print("\n" + "=" * 80)
        print("✅ CAMPAIGN COMPLETE")
        print("=" * 80)
        print(f"   Results saved to: {self.output_dir}")
        print(f"   Total time: {(datetime.now() - self.campaign_start).total_seconds():.1f}s")

        return 0

    def run_single_trial(
        self,
        trial_id: int,
        simulations: int,
        num_threads: int,
        batch_size: int
    ) -> Dict[str, Any]:
        """
        Run single trial using UnifiedProfiler for comprehensive metrics.

        This leverages UnifiedProfiler to get:
        - 295 C++ metrics (state cloning, OpenMP, thread idle, CAS, mutex)
        - Python profiling (GIL, inference, thread coordination)
        - Automated bottleneck analysis
        - Full export suite (JSON, Chrome Trace, Markdown)
        """

        trial_dir = self.output_dir / f"trial_{trial_id:03d}"
        trial_dir.mkdir(exist_ok=True)

        result = {
            'trial_id': trial_id,
            'simulations': simulations,
            'threads': num_threads,
            'batch_size': batch_size,
            'timestamp': datetime.now().isoformat(),
            'success': False,
        }

        try:
            # Create mock args for UnifiedProfiler
            class MockArgs:
                def __init__(self, simulations, threads, batch_size, output):
                    self.simulations = simulations
                    self.threads = threads
                    self.batch_size = batch_size
                    self.output = str(output)
                    self.validate = False  # Skip validation for speed in campaigns
                    self.runner_type = "continuous"  # Use T024f-6 make/unmake + T019 OpenMP batch extraction

            mock_args = MockArgs(simulations, num_threads, batch_size, trial_dir)

            # Use UnifiedProfiler for complete profiling coverage
            unified = UnifiedProfiler(mock_args)

            # Wall-clock timing
            start_time = time.perf_counter()

            # Run unified profiling workflow
            # This automatically:
            # - Enables C++ profiler (295 metrics)
            # - Enables Python profiler (GIL, inference, threads)
            # - Runs MCTS workload
            # - Exports all results (JSON, Chrome Trace, Markdown)
            # - Performs bottleneck analysis
            unified.run()

            end_time = time.perf_counter()
            wall_clock_time = end_time - start_time

            # Load comprehensive results from UnifiedProfiler exports
            json_path = trial_dir / "cpp_profiling.json"
            python_path = trial_dir / "python_profiling.json"

            if json_path.exists():
                with open(json_path, 'r') as f:
                    cpp_metrics = json.load(f)

                # Extract key metrics from UnifiedProfiler output
                # JSON structure: {timing_stats: {...}, counters: {...}, gauges: {...}, bottlenecks: [...]}

                counters = cpp_metrics.get('counters', {})
                gauges = cpp_metrics.get('gauges', {})
                timing_stats = cpp_metrics.get('timing_stats', {})
                bottlenecks = cpp_metrics.get('bottlenecks', [])

                # Extract state cloning metrics
                state_clone_count = counters.get('state_clone_count', 0)
                state_clone_time = timing_stats.get('state_clone_total', {}).get('total', 0)

                # Extract OpenMP metrics
                omp_success = counters.get('feature_extract_omp', 0)

                # Extract thread idle time
                thread_idle_time = timing_stats.get('thread_idle_total', {}).get('total', 0)

                # Extract CAS retry count
                cas_retries = counters.get('cas_retry', 0)

                result.update({
                    'success': True,
                    'wall_clock_time': wall_clock_time,
                    'successful_simulations': simulations,
                    'throughput': simulations / wall_clock_time if wall_clock_time > 0 else 0,
                    'tree_nodes': gauges.get('tree_node_count', 0),
                    'avg_time_per_sim': wall_clock_time / simulations if simulations > 0 else 0,

                    # Bottleneck indicators from counters
                    'state_clone_count': state_clone_count,
                    'state_clone_time_ms': state_clone_time / 1e6 if state_clone_time > 0 else 0,
                    'omp_parallel_success': omp_success,
                    'thread_idle_time_ms': thread_idle_time / 1e6 if thread_idle_time > 0 else 0,
                    'cas_retries': cas_retries,

                    # Bottleneck analysis summary
                    'bottleneck_count': len(bottlenecks),
                    'primary_bottleneck': bottlenecks[0]['metric'] if bottlenecks else None,
                    'bottleneck_severity': bottlenecks[0]['severity'] if bottlenecks else 0,

                    # Export availability
                    'python_metrics_available': python_path.exists(),
                    'chrome_trace_available': (trial_dir / "cpp_trace.json").exists(),
                    'markdown_report_available': (trial_dir / "cpp_report.md").exists(),
                })

        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()

        # Save trial result
        with open(trial_dir / "result.json", 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def analyze_results(self):
        """Analyze results with bottleneck insights from UnifiedProfiler"""

        if not self.results:
            print("⚠️  No results to analyze")
            return

        # Filter successful trials
        successful = [r for r in self.results if r['success']]

        if not successful:
            print("❌ No successful trials")
            return

        print(f"\n✅ Successful trials: {len(successful)}/{len(self.results)}")

        # Find best throughput
        best = max(successful, key=lambda x: x['throughput'])

        print("\n🏆 Best Configuration:")
        print(f"   Simulations: {best['simulations']}")
        print(f"   Threads: {best['threads']}")
        print(f"   Batch size: {best['batch_size']}")
        print(f"   Throughput: {best['throughput']:.1f} sims/sec")
        print(f"   Wall-clock: {best['wall_clock_time']:.2f}s")

        # Thread scaling analysis
        print("\n📈 Thread Scaling Analysis:")
        for threads in sorted(set(r['threads'] for r in successful)):
            thread_results = [r for r in successful if r['threads'] == threads]
            avg_throughput = np.mean([r['throughput'] for r in thread_results])
            print(f"   {threads:2d} threads: {avg_throughput:7.1f} sims/sec (avg)")

        # Batch size analysis
        print("\n📦 Batch Size Analysis:")
        for batch in sorted(set(r['batch_size'] for r in successful)):
            batch_results = [r for r in successful if r['batch_size'] == batch]
            avg_throughput = np.mean([r['throughput'] for r in batch_results])
            print(f"   Batch {batch:3d}: {avg_throughput:7.1f} sims/sec (avg)")

        # Bottleneck analysis from UnifiedProfiler data
        print("\n🔍 Bottleneck Analysis (from UnifiedProfiler metrics):")

        # State cloning analysis
        avg_clones = np.mean([r.get('state_clone_count', 0) for r in successful if 'state_clone_count' in r])
        if avg_clones > 0:
            avg_per_sim = avg_clones / np.mean([r['simulations'] for r in successful])
            print(f"   🔴 State Cloning: {avg_clones:.0f} avg clones ({avg_per_sim:.1f}× per simulation)")
            if avg_per_sim > 1.5:
                print(f"      → HIGH PRIORITY: Implement state pooling (review.txt line 37-54)")

        # OpenMP parallelization
        omp_failures = sum(1 for r in successful if r.get('omp_parallel_success', 0) == 0)
        if omp_failures > 0:
            print(f"   🔴 OpenMP: {omp_failures}/{len(successful)} trials NOT parallelizing")
            print(f"      → HIGH PRIORITY: Fix feature extraction (dlpack_bridge.cpp:431-434)")

        # Thread idle time
        avg_idle_ms = np.mean([r.get('thread_idle_time_ms', 0) for r in successful if 'thread_idle_time_ms' in r])
        if avg_idle_ms > 10:
            print(f"   ⚠️  Thread Idle: {avg_idle_ms:.1f}ms avg idle time")
            print(f"      → Reduce coordination overhead (review.txt line 71-136)")

        # CAS contention
        avg_cas = np.mean([r.get('cas_retries', 0) for r in successful if 'cas_retries' in r])
        if avg_cas > 100:
            print(f"   ⚠️  CAS Contention: {avg_cas:.0f} avg retries")
            print(f"      → Consider lock-free alternatives")

        # Python profiling coverage
        python_enabled = sum(1 for r in successful if r.get('python_metrics_available', False))
        chrome_enabled = sum(1 for r in successful if r.get('chrome_trace_available', False))

        print(f"\n   ✅ UnifiedProfiler Coverage:")
        print(f"      - Python profiling: {python_enabled}/{len(successful)} trials")
        print(f"      - Chrome traces: {chrome_enabled}/{len(successful)} trials")
        print(f"      - 295 C++ metrics per trial")

    def export_results(self):
        """Export campaign results to files"""

        # Export summary JSON
        summary = {
            'campaign': {
                'start_time': self.campaign_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - self.campaign_start).total_seconds(),
                'total_trials': len(self.results),
                'successful_trials': sum(1 for r in self.results if r['success']),
            },
            'parameters': {
                'simulations': self.simulations,
                'threads': self.threads,
                'batch_sizes': self.batch_sizes,
            },
            'results': self.results,
        }

        summary_path = self.output_dir / "campaign_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📄 Summary exported: {summary_path}")

        # Export CSV with comprehensive bottleneck metrics for easy analysis
        csv_path = self.output_dir / "results.csv"
        with open(csv_path, 'w') as f:
            # Header - includes all bottleneck metrics from UnifiedProfiler
            # Note: Using 'state_clone_count' to match JSON field name (not 'state_clones')
            f.write("trial_id,simulations,threads,batch_size,success,throughput,wall_clock_time,")
            f.write("tree_nodes,state_clone_count,state_clone_time_ms,omp_success,thread_idle_ms,cas_retries,")
            f.write("bottleneck_count,primary_bottleneck,bottleneck_severity,")
            f.write("python_metrics,chrome_trace\n")

            # Data rows
            for r in self.results:
                f.write(f"{r['trial_id']},{r['simulations']},{r['threads']},{r['batch_size']},")
                if r['success']:
                    f.write(f"1,{r['throughput']:.2f},{r['wall_clock_time']:.4f},")
                    f.write(f"{r.get('tree_nodes', 0)},")
                    f.write(f"{r.get('state_clone_count', 0)},")  # Matches JSON field name
                    f.write(f"{r.get('state_clone_time_ms', 0):.2f},")
                    f.write(f"{r.get('omp_parallel_success', 0)},")
                    f.write(f"{r.get('thread_idle_time_ms', 0):.2f},")
                    f.write(f"{r.get('cas_retries', 0)},")
                    f.write(f"{r.get('bottleneck_count', 0)},")
                    f.write(f"{r.get('primary_bottleneck', 'none')},")
                    f.write(f"{r.get('bottleneck_severity', 0):.1f},")
                    f.write(f"{1 if r.get('python_metrics_available', False) else 0},")
                    f.write(f"{1 if r.get('chrome_trace_available', False) else 0}\n")
                else:
                    f.write("0,0,0,0,0,0,0,0,0,0,none,0,0,0\n")

        print(f"📊 CSV exported: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive profiling campaign with parameter sweeps"
    )

    # Quick test mode
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help="Run quick test with few trials (2 sims × 2 threads × 1 batch = 4 trials)"
    )

    # Parameter ranges
    parser.add_argument(
        '--simulations',
        type=str,
        default=None,
        help="Comma-separated simulation counts (default: 100,400,800,1600)"
    )

    parser.add_argument(
        '--threads',
        type=str,
        default=None,
        help="Comma-separated thread counts (default: 1,2,4,8)"
    )

    parser.add_argument(
        '--batch-sizes',
        type=str,
        default=None,
        help="Comma-separated batch sizes (default: 32,64)"
    )

    parser.add_argument(
        '--repetitions',
        type=int,
        default=1,
        help="Number of repetitions per configuration for statistical significance (default: 1)"
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        default=f"profiling_campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory for results"
    )

    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Apply defaults
    if args.quick_test:
        args.simulations = args.simulations or "100,200"
        args.threads = args.threads or "1,4"
        args.batch_sizes = args.batch_sizes or "64"
    else:
        args.simulations = args.simulations or "100,400,800,1600"
        args.threads = args.threads or "1,2,4,8"
        args.batch_sizes = args.batch_sizes or "32,64"

    # Run campaign
    campaign = ProfilingCampaign(args)
    return campaign.run_campaign()


if __name__ == '__main__':
    sys.exit(main())
