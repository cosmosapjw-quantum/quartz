#!/usr/bin/env python3
"""
Profiling Results Analysis

Analyzes and compares profiling results from multiple campaigns to identify
performance bottlenecks and optimization opportunities.

Usage:
    # Analyze single campaign
    python scripts/analyze_profiling_results.py profiling_campaign_*/campaign_summary.json

    # Compare multiple campaigns
    python scripts/analyze_profiling_results.py campaign1/campaign_summary.json campaign2/campaign_summary.json

    # Generate detailed report
    python scripts/analyze_profiling_results.py --detailed results/*/campaign_summary.json

Author: MCTS Performance Team
Date: 2025-10-15
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import glob

import numpy as np
import pandas as pd


class ResultsAnalyzer:
    """Analyzes profiling campaign results"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.campaigns = []

    def load_campaigns(self, paths: List[str]) -> bool:
        """Load campaign results from files"""

        print("=" * 80)
        print("LOADING CAMPAIGN RESULTS")
        print("=" * 80)

        for path_pattern in paths:
            # Expand glob patterns
            files = glob.glob(path_pattern)

            for file_path in files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    self.campaigns.append({
                        'path': file_path,
                        'data': data,
                    })

                    print(f"✅ Loaded: {file_path}")
                    print(f"   Trials: {data['campaign']['total_trials']}")
                    print(f"   Success: {data['campaign']['successful_trials']}")

                except Exception as e:
                    print(f"❌ Failed to load {file_path}: {e}")

        if not self.campaigns:
            print("\n❌ No campaigns loaded")
            return False

        print(f"\n✅ Loaded {len(self.campaigns)} campaign(s)")
        return True

    def analyze(self) -> int:
        """Run complete analysis"""

        if not self.campaigns:
            print("❌ No campaigns to analyze")
            return 1

        print("\n" + "=" * 80)
        print("PERFORMANCE ANALYSIS")
        print("=" * 80)

        # Aggregate all results
        all_results = []
        for campaign in self.campaigns:
            for result in campaign['data']['results']:
                if result['success']:
                    all_results.append(result)

        if not all_results:
            print("❌ No successful results to analyze")
            return 1

        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_results)

        print(f"\n📊 Dataset: {len(df)} successful trials")

        # Overall statistics
        self._print_overall_stats(df)

        # Thread scaling analysis
        self._analyze_thread_scaling(df)

        # Batch size analysis
        self._analyze_batch_sizes(df)

        # Simulation count analysis
        self._analyze_simulation_counts(df)

        # Find optimal configuration
        self._find_optimal_config(df)

        # Bottleneck identification
        if self.args.detailed:
            self._identify_bottlenecks(df)

        return 0

    def _print_overall_stats(self, df: pd.DataFrame):
        """Print overall statistics"""

        print("\n" + "=" * 80)
        print("OVERALL STATISTICS")
        print("=" * 80)

        print(f"\n⏱️  Wall-Clock Time:")
        print(f"   Mean:   {df['wall_clock_time'].mean():.3f}s")
        print(f"   Median: {df['wall_clock_time'].median():.3f}s")
        print(f"   StdDev: {df['wall_clock_time'].std():.3f}s")
        print(f"   Min:    {df['wall_clock_time'].min():.3f}s")
        print(f"   Max:    {df['wall_clock_time'].max():.3f}s")

        print(f"\n🚀 Throughput:")
        print(f"   Mean:   {df['throughput'].mean():.1f} sims/sec")
        print(f"   Median: {df['throughput'].median():.1f} sims/sec")
        print(f"   StdDev: {df['throughput'].std():.1f} sims/sec")
        print(f"   Min:    {df['throughput'].min():.1f} sims/sec")
        print(f"   Max:    {df['throughput'].max():.1f} sims/sec")

        print(f"\n🌳 Tree Nodes Created:")
        print(f"   Mean:   {df['tree_nodes'].mean():.0f}")
        print(f"   Median: {df['tree_nodes'].median():.0f}")
        print(f"   StdDev: {df['tree_nodes'].std():.0f}")
        print(f"   Min:    {df['tree_nodes'].min():.0f}")
        print(f"   Max:    {df['tree_nodes'].max():.0f}")

    def _analyze_thread_scaling(self, df: pd.DataFrame):
        """Analyze thread scaling efficiency"""

        print("\n" + "=" * 80)
        print("THREAD SCALING ANALYSIS")
        print("=" * 80)

        thread_groups = df.groupby('threads')

        print("\n📊 Throughput by Thread Count:")
        for threads, group in thread_groups:
            mean_tp = group['throughput'].mean()
            std_tp = group['throughput'].std()
            count = len(group)

            print(f"   {threads:2d} threads: {mean_tp:7.1f} ± {std_tp:5.1f} sims/sec (n={count})")

        # Calculate scaling efficiency
        if 1 in df['threads'].values:
            baseline = df[df['threads'] == 1]['throughput'].mean()

            print("\n📈 Scaling Efficiency (vs 1 thread):")
            for threads in sorted(df['threads'].unique()):
                if threads == 1:
                    continue

                mean_tp = df[df['threads'] == threads]['throughput'].mean()
                ideal_speedup = threads
                actual_speedup = mean_tp / baseline
                efficiency = (actual_speedup / ideal_speedup) * 100

                print(f"   {threads:2d} threads: {actual_speedup:.2f}× speedup ({efficiency:.1f}% efficiency)")

                if efficiency >= 90:
                    status = "✅ Excellent"
                elif efficiency >= 70:
                    status = "✓  Good"
                elif efficiency >= 50:
                    status = "⚠️  Moderate"
                else:
                    status = "❌ Poor"

                print(f"              {status} - {efficiency:.1f}% of ideal")

    def _analyze_batch_sizes(self, df: pd.DataFrame):
        """Analyze batch size impact"""

        print("\n" + "=" * 80)
        print("BATCH SIZE ANALYSIS")
        print("=" * 80)

        batch_groups = df.groupby('batch_size')

        print("\n📦 Throughput by Batch Size:")
        for batch, group in batch_groups:
            mean_tp = group['throughput'].mean()
            std_tp = group['throughput'].std()
            count = len(group)

            print(f"   Batch {batch:3d}: {mean_tp:7.1f} ± {std_tp:5.1f} sims/sec (n={count})")

    def _analyze_simulation_counts(self, df: pd.DataFrame):
        """Analyze simulation count impact"""

        print("\n" + "=" * 80)
        print("SIMULATION COUNT ANALYSIS")
        print("=" * 80)

        sim_groups = df.groupby('simulations')

        print("\n🎯 Throughput by Simulation Count:")
        for sims, group in sim_groups:
            mean_tp = group['throughput'].mean()
            std_tp = group['throughput'].std()
            count = len(group)

            print(f"   {sims:5d} sims: {mean_tp:7.1f} ± {std_tp:5.1f} sims/sec (n={count})")

    def _find_optimal_config(self, df: pd.DataFrame):
        """Find optimal configuration"""

        print("\n" + "=" * 80)
        print("OPTIMAL CONFIGURATION")
        print("=" * 80)

        # Find highest throughput
        best_idx = df['throughput'].idxmax()
        best = df.loc[best_idx]

        print("\n🏆 Best Configuration:")
        print(f"   Simulations:  {best['simulations']}")
        print(f"   Threads:      {best['threads']}")
        print(f"   Batch size:   {best['batch_size']}")
        print(f"   Throughput:   {best['throughput']:.1f} sims/sec")
        print(f"   Wall-clock:   {best['wall_clock_time']:.3f}s")
        print(f"   Tree nodes:   {best['tree_nodes']}")

        # Find most consistent configuration
        config_groups = df.groupby(['threads', 'batch_size'])
        most_consistent = None
        lowest_cv = float('inf')

        for config, group in config_groups:
            if len(group) < 2:
                continue

            cv = (group['throughput'].std() / group['throughput'].mean()) * 100

            if cv < lowest_cv:
                lowest_cv = cv
                most_consistent = (config, group)

        if most_consistent:
            (threads, batch), group = most_consistent

            print("\n🎯 Most Consistent Configuration:")
            print(f"   Threads:      {threads}")
            print(f"   Batch size:   {batch}")
            print(f"   Throughput:   {group['throughput'].mean():.1f} ± {group['throughput'].std():.1f} sims/sec")
            print(f"   Variability:  {lowest_cv:.2f}% CV")

    def _identify_bottlenecks(self, df: pd.DataFrame):
        """Identify performance bottlenecks"""

        print("\n" + "=" * 80)
        print("BOTTLENECK IDENTIFICATION")
        print("=" * 80)

        # Check for thread scaling issues
        if 1 in df['threads'].values and 8 in df['threads'].values:
            single_thread = df[df['threads'] == 1]['throughput'].mean()
            eight_threads = df[df['threads'] == 8]['throughput'].mean()
            scaling_ratio = eight_threads / single_thread

            print("\n🔍 Thread Scaling:")
            if scaling_ratio < 4.0:
                print(f"   ❌ Poor scaling: {scaling_ratio:.2f}× with 8 threads (expect ≥4×)")
                print("   → Likely bottleneck: Thread contention or synchronization overhead")
            elif scaling_ratio < 6.0:
                print(f"   ⚠️  Moderate scaling: {scaling_ratio:.2f}× with 8 threads")
                print("   → Possible bottleneck: Shared resource contention")
            else:
                print(f"   ✅ Good scaling: {scaling_ratio:.2f}× with 8 threads")

        # Check for throughput ceiling
        max_throughput = df['throughput'].max()
        target_throughput = 8000  # From CLAUDE.md

        print("\n🎯 Target Analysis:")
        print(f"   Target:   {target_throughput} sims/sec")
        print(f"   Achieved: {max_throughput:.1f} sims/sec")
        print(f"   Progress: {(max_throughput / target_throughput) * 100:.1f}%")

        if max_throughput < target_throughput * 0.25:
            print("   ❌ Far from target - major bottlenecks present")
            print("   → Check: OpenMP parallelization, state cloning, inference overhead")
        elif max_throughput < target_throughput * 0.75:
            print("   ⚠️  Below target - optimization needed")
            print("   → Check: Thread efficiency, memory allocation, virtual loss")
        else:
            print("   ✅ Near or at target")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze profiling campaign results"
    )

    parser.add_argument(
        'files',
        nargs='+',
        help="Campaign summary JSON files (supports glob patterns)"
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help="Show detailed bottleneck analysis"
    )

    args = parser.parse_args()

    analyzer = ResultsAnalyzer(args)

    if not analyzer.load_campaigns(args.files):
        return 1

    return analyzer.analyze()


if __name__ == '__main__':
    sys.exit(main())
