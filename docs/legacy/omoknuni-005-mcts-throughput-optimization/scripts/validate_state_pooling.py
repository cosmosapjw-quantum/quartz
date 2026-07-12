#!/usr/bin/env python3
"""
Validate state pooling optimization with profiling.

Acceptance criteria (T018h):
- alloc_slow_path counter <20,000 for 2,000 sims (<10 per sim)
- state_clone_total <50ms (<5% of time)
- throughput ≥7,500 sims/sec (3.0× minimum improvement)
"""

import subprocess
import json
import sys
import os
from pathlib import Path

def run_profiling_benchmark():
    """Run profiling benchmark with state pooling enabled."""
    print("🔬 Running profiling benchmark (2000 sims, 8 threads, 10 iterations)...")
    print("This will take ~2-3 minutes...\n")

    result = subprocess.run([
        sys.executable, 'scripts/benchmark_throughput.py',
        '--game', 'gomoku',
        '--threads', '8',
        '--simulations', '2000',
        '--seed', '42',
        '--iterations', '10',
        '--enable-profiling'
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Benchmark failed:\n{result.stderr}")
        sys.exit(1)

    # Parse JSON output (skip C++ profiler messages)
    try:
        # Find JSON start (first '{')
        stdout = result.stdout
        json_start = stdout.find('{')
        if json_start == -1:
            raise ValueError("No JSON found in output")

        json_str = stdout[json_start:]
        data = json.loads(json_str)
        return data
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Failed to parse benchmark output: {e}\n{result.stdout}")
        sys.exit(1)

def validate_results(data):
    """Validate profiling results against acceptance criteria."""
    print("=" * 70)
    print("T018h: State Pooling Validation Results")
    print("=" * 70)

    errors = []

    # Extract metrics
    cpp_prof = data.get('cpp_profiling', {})
    alloc_count = cpp_prof.get('counters', {}).get('alloc_slow_path', 0)
    state_clone_ms = cpp_prof.get('timings', {}).get('state_clone_total', 0)
    total_time_ms = cpp_prof.get('session_duration_ms', 1)
    throughput = data.get('mean_throughput_sims_per_sec', 0)

    num_sims = 2000

    print(f"\n📊 Raw Metrics:")
    print(f"  Allocations (total):     {alloc_count:,}")
    print(f"  State clone time:        {state_clone_ms:.2f} ms")
    print(f"  Total runtime:           {total_time_ms:.2f} ms")
    print(f"  Throughput:              {throughput:.0f} sims/sec")

    # Criterion 1: Allocations <10 per simulation
    print(f"\n✅ Criterion 1: Allocation Reduction")
    alloc_per_sim = alloc_count / num_sims
    if alloc_per_sim >= 10:
        errors.append(
            f"  ❌ Allocations per sim: {alloc_per_sim:.1f} (target: <10)"
        )
    else:
        print(f"  ✅ Allocations per sim: {alloc_per_sim:.1f} (target: <10)")

    # Criterion 2: State cloning <5% of time
    print(f"\n✅ Criterion 2: State Clone Time Reduction")
    clone_pct = (state_clone_ms / total_time_ms) * 100 if total_time_ms > 0 else 0
    if clone_pct >= 5.0:
        errors.append(
            f"  ❌ State cloning: {clone_pct:.1f}% of time (target: <5%)"
        )
    else:
        print(f"  ✅ State cloning: {clone_pct:.1f}% of time (target: <5%)")

    # Criterion 3: Throughput ≥7,500 sims/sec
    print(f"\n✅ Criterion 3: Throughput Improvement")
    if throughput < 7500:
        errors.append(
            f"  ❌ Throughput: {throughput:.0f} sims/sec (target: ≥7,500)"
        )
    else:
        print(f"  ✅ Throughput: {throughput:.0f} sims/sec (target: ≥7,500)")

    # Performance comparison
    baseline_throughput = 2659  # From profiling campaign
    improvement = throughput / baseline_throughput
    print(f"\n📈 Performance Improvement:")
    print(f"  Baseline:     {baseline_throughput} sims/sec")
    print(f"  Optimized:    {throughput:.0f} sims/sec")
    print(f"  Improvement:  {improvement:.2f}× ({(improvement - 1) * 100:.1f}% faster)")

    # Summary
    print("\n" + "=" * 70)
    if errors:
        print("❌ VALIDATION FAILED")
        print("=" * 70)
        for error in errors:
            print(error)
        print("\n⚠️  Opening 'Needs Decision' note in TASKS.md...")
        return False
    else:
        print("✅ VALIDATION PASSED - All criteria met!")
        print("=" * 70)
        return True

def save_profiling_report(data):
    """Save profiling data to artifacts directory."""
    artifacts_dir = Path("artifacts/profiling/state_pooling_validation")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    report_file = artifacts_dir / "validation_results.json"
    with open(report_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n💾 Profiling report saved: {report_file}")
    return report_file

def main():
    """Main validation workflow."""
    print("🚀 State Pooling Validation (T018h)\n")

    # Run benchmark
    data = run_profiling_benchmark()

    # Validate results
    success = validate_results(data)

    # Save artifacts
    report_file = save_profiling_report(data)

    # Exit with appropriate code
    if success:
        print("\n✅ T018h complete - ready for T018i (performance benchmarking)")
        sys.exit(0)
    else:
        print("\n❌ T018h failed - needs decision before proceeding")
        sys.exit(1)

if __name__ == '__main__':
    main()
