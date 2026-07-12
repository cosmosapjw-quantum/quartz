#!/usr/bin/env python3
"""
Performance regression detection script for CI/CD pipeline.
Compares current benchmark results with baseline to detect regressions.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


def load_benchmark_results(filepath: str) -> Optional[Dict[str, Any]]:
    """Load benchmark results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def compare_benchmarks(current_file: str, baseline_file: str = ".benchmarks/baseline.json") -> bool:
    """
    Compare current benchmark results with baseline.

    Returns:
        bool: True if no significant regression detected, False otherwise
    """
    current = load_benchmark_results(current_file)
    baseline = load_benchmark_results(baseline_file)

    if not current:
        print("❌ No current benchmark results found")
        return False

    if not baseline:
        print("ℹ️  No baseline found - creating new baseline")
        return True

    # Extract benchmark data
    current_benchmarks = current.get('benchmarks', [])
    baseline_benchmarks = baseline.get('benchmarks', [])

    if not current_benchmarks:
        print("❌ No benchmark data in current results")
        return False

    if not baseline_benchmarks:
        print("ℹ️  No baseline benchmark data - using current as baseline")
        return True

    # Create lookup for baseline benchmarks
    baseline_lookup = {bench['name']: bench for bench in baseline_benchmarks}

    regressions = []
    improvements = []
    new_benchmarks = []

    # Regression threshold (e.g., 20% slower)
    REGRESSION_THRESHOLD = 1.20
    IMPROVEMENT_THRESHOLD = 0.80

    for current_bench in current_benchmarks:
        name = current_bench['name']
        current_time = current_bench['stats']['mean']

        if name in baseline_lookup:
            baseline_time = baseline_lookup[name]['stats']['mean']
            ratio = current_time / baseline_time

            if ratio > REGRESSION_THRESHOLD:
                regressions.append({
                    'name': name,
                    'current': current_time,
                    'baseline': baseline_time,
                    'ratio': ratio,
                    'percent_change': (ratio - 1) * 100
                })
            elif ratio < IMPROVEMENT_THRESHOLD:
                improvements.append({
                    'name': name,
                    'current': current_time,
                    'baseline': baseline_time,
                    'ratio': ratio,
                    'percent_change': (1 - ratio) * 100
                })
        else:
            new_benchmarks.append({
                'name': name,
                'time': current_time
            })

    # Report results
    print("\n📊 Benchmark Comparison Results")
    print("=" * 50)

    if improvements:
        print(f"\n✅ Performance Improvements ({len(improvements)}):")
        for imp in improvements:
            print(f"  • {imp['name']}: {imp['percent_change']:.1f}% faster")
            print(f"    {imp['baseline']:.6f}s → {imp['current']:.6f}s")

    if new_benchmarks:
        print(f"\n🆕 New Benchmarks ({len(new_benchmarks)}):")
        for new in new_benchmarks:
            print(f"  • {new['name']}: {new['time']:.6f}s")

    if regressions:
        print(f"\n⚠️  Performance Regressions ({len(regressions)}):")
        for reg in regressions:
            print(f"  • {reg['name']}: {reg['percent_change']:.1f}% slower")
            print(f"    {reg['baseline']:.6f}s → {reg['current']:.6f}s")

    total_current = len(current_benchmarks)
    total_baseline = len(baseline_benchmarks)
    print(f"\n📈 Summary: {total_current} current benchmarks vs {total_baseline} baseline")

    # Determine overall result
    if regressions:
        print("\n❌ PERFORMANCE REGRESSION DETECTED!")
        print("Review the regressions above before merging.")
        return False
    else:
        print("\n✅ No significant performance regressions detected.")
        return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Compare benchmark results for regression detection')
    parser.add_argument('current_file', help='Path to current benchmark results JSON file')
    parser.add_argument('--baseline', default='.benchmarks/baseline.json',
                        help='Path to baseline benchmark results (default: .benchmarks/baseline.json)')
    parser.add_argument('--fail-on-regression', action='store_true',
                        help='Exit with non-zero code if regression detected')

    args = parser.parse_args()

    # Ensure baseline directory exists
    Path('.benchmarks').mkdir(exist_ok=True)

    success = compare_benchmarks(args.current_file, args.baseline)

    if args.fail_on_regression and not success:
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()