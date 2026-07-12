#!/usr/bin/env python3
"""
Profiling Campaign Analysis Script
Purpose: Analyze profiling results from Phase 1 and Phase 2 campaigns
Usage: python scripts/profiling/analyze_campaign.py <profiling_dir> [options]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import statistics

def load_trial_data(trial_dir: Path) -> Optional[Dict]:
    """Load profiling data from a single trial directory."""
    cpp_profiling = trial_dir / "cpp_profiling.json"
    python_profiling = trial_dir / "python_profiling.json"

    data = {}

    try:
        if cpp_profiling.exists():
            with open(cpp_profiling) as f:
                data['cpp'] = json.load(f)

        if python_profiling.exists():
            with open(python_profiling) as f:
                data['python'] = json.load(f)

        return data if data else None
    except Exception as e:
        print(f"Warning: Failed to load trial {trial_dir.name}: {e}", file=sys.stderr)
        return None


def analyze_phase1(trials: List[Dict]) -> Dict:
    """Analyze Phase 1 specific metrics (state cloning elimination)."""

    # Extract Phase 1 metrics
    state_cloning_times = []
    state_clone_counts = []
    feature_move_counts = []
    throughputs = []

    for trial in trials:
        cpp = trial.get('cpp', {})

        # State cloning metrics
        if 'state_cloning' in cpp:
            state_cloning_times.append(cpp['state_cloning']['total_elapsed_ns'] / 1000.0)  # Convert to μs

        if 'state_clone_count' in cpp:
            state_clone_counts.append(cpp['state_clone_count']['call_count'])

        if 'feature_move_count' in cpp:
            feature_move_counts.append(cpp['feature_move_count']['call_count'])

        # Throughput
        if 'simulations_per_second' in cpp:
            throughputs.append(cpp['simulations_per_second'])

    results = {
        'phase': 1,
        'metrics': {}
    }

    if state_cloning_times:
        results['metrics']['state_cloning_us'] = {
            'mean': statistics.mean(state_cloning_times),
            'median': statistics.median(state_cloning_times),
            'stdev': statistics.stdev(state_cloning_times) if len(state_cloning_times) > 1 else 0,
            'p95': sorted(state_cloning_times)[int(len(state_cloning_times) * 0.95)]
        }

    if state_clone_counts:
        results['metrics']['state_clone_count'] = {
            'mean': statistics.mean(state_clone_counts),
            'total': sum(state_clone_counts),
            'max': max(state_clone_counts)
        }

    if feature_move_counts:
        results['metrics']['feature_move_count'] = {
            'mean': statistics.mean(feature_move_counts),
            'total': sum(feature_move_counts)
        }

    if throughputs:
        results['metrics']['throughput_sims_sec'] = {
            'mean': statistics.mean(throughputs),
            'median': statistics.median(throughputs),
            'stdev': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
            'min': min(throughputs),
            'max': max(throughputs)
        }

    return results


def analyze_phase2(trials: List[Dict]) -> Dict:
    """Analyze Phase 2 specific metrics (tensor pipeline + OpenMP)."""

    # Extract Phase 2 metrics
    tensor_creation_times = []
    h2d_transfer_times = []
    openmp_thread_counts = []
    openmp_enabled_count = 0
    pinned_buffer_reuse_counts = []
    pinned_buffer_alloc_counts = []
    throughputs = []

    for trial in trials:
        cpp = trial.get('cpp', {})
        python = trial.get('python', {})

        # Tensor creation
        if 'tensor_creation' in cpp:
            tensor_creation_times.append(cpp['tensor_creation']['total_elapsed_ns'] / 1e6)  # Convert to ms
        elif 'tensor_creation' in python:
            tensor_creation_times.append(python['tensor_creation'] * 1000.0)  # Convert s to ms

        # H2D transfer
        if 'h2d_transfer' in cpp:
            h2d_transfer_times.append(cpp['h2d_transfer']['total_elapsed_ns'] / 1e6)  # Convert to ms

        # OpenMP
        if 'openmp_thread_count' in cpp:
            openmp_thread_counts.append(cpp['openmp_thread_count']['call_count'])

        if 'openmp_enabled' in cpp and cpp['openmp_enabled']['call_count'] > 0:
            openmp_enabled_count += 1

        # Pinned buffer
        if 'pinned_buffer_reuse' in cpp:
            pinned_buffer_reuse_counts.append(cpp['pinned_buffer_reuse']['call_count'])

        if 'pinned_buffer_allocation' in cpp:
            pinned_buffer_alloc_counts.append(cpp['pinned_buffer_allocation']['call_count'])

        # Throughput
        if 'simulations_per_second' in cpp:
            throughputs.append(cpp['simulations_per_second'])

    results = {
        'phase': 2,
        'metrics': {}
    }

    if tensor_creation_times:
        results['metrics']['tensor_creation_ms'] = {
            'mean': statistics.mean(tensor_creation_times),
            'median': statistics.median(tensor_creation_times),
            'p95': sorted(tensor_creation_times)[int(len(tensor_creation_times) * 0.95)],
            'max': max(tensor_creation_times)
        }

    if h2d_transfer_times:
        results['metrics']['h2d_transfer_ms'] = {
            'mean': statistics.mean(h2d_transfer_times),
            'p95': sorted(h2d_transfer_times)[int(len(h2d_transfer_times) * 0.95)]
        }

    if openmp_thread_counts:
        results['metrics']['openmp_thread_count'] = {
            'mean': statistics.mean(openmp_thread_counts),
            'median': statistics.median(openmp_thread_counts)
        }

    results['metrics']['openmp_enabled_pct'] = (openmp_enabled_count / len(trials) * 100) if trials else 0

    if pinned_buffer_reuse_counts and pinned_buffer_alloc_counts:
        total_reuse = sum(pinned_buffer_reuse_counts)
        total_alloc = sum(pinned_buffer_alloc_counts)
        results['metrics']['pinned_buffer_reuse_pct'] = (total_reuse / (total_reuse + total_alloc) * 100) if (total_reuse + total_alloc) > 0 else 0

    if throughputs:
        results['metrics']['throughput_sims_sec'] = {
            'mean': statistics.mean(throughputs),
            'median': statistics.median(throughputs),
            'stdev': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
            'min': min(throughputs),
            'max': max(throughputs)
        }

    return results


def compare_to_baseline(results: Dict, baseline_throughput: float = 120.4):
    """Compare results to documented baseline (120.4 sims/sec)."""
    throughput = results['metrics'].get('throughput_sims_sec', {}).get('mean', 0)
    if throughput > 0:
        speedup = throughput / baseline_throughput
        results['comparison'] = {
            'baseline_sims_sec': baseline_throughput,
            'current_sims_sec': throughput,
            'speedup': speedup,
            'improvement_pct': (speedup - 1) * 100
        }


def main():
    parser = argparse.ArgumentParser(description='Analyze MCTS profiling campaign results')
    parser.add_argument('profiling_dir', type=Path, help='Directory containing profiling campaign results')
    parser.add_argument('--phase', type=int, choices=[1, 2], help='Analyze phase-specific metrics (1 or 2)')
    parser.add_argument('--compare-to-baseline', action='store_true', help='Compare to 120.4 sims/sec baseline')
    parser.add_argument('--compare-to-phase1', action='store_true', help='Compare to Phase 1 results')
    parser.add_argument('--output', type=Path, help='Write results to JSON file')
    parser.add_argument('--get-throughput', action='store_true', help='Output only mean throughput (for scripting)')

    args = parser.parse_args()

    if not args.profiling_dir.exists():
        print(f"Error: Profiling directory not found: {args.profiling_dir}", file=sys.stderr)
        sys.exit(1)

    # Load all trial data
    campaign_dir = args.profiling_dir / 'campaign' if (args.profiling_dir / 'campaign').exists() else args.profiling_dir

    trials = []
    for trial_dir in sorted(campaign_dir.glob('trial_*')):
        if trial_dir.is_dir():
            data = load_trial_data(trial_dir)
            if data:
                trials.append(data)

    if not trials:
        print(f"Error: No valid trial data found in {campaign_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(trials)} trials from {args.profiling_dir}")
    print()

    # Analyze based on phase
    if args.phase == 1:
        results = analyze_phase1(trials)
    elif args.phase == 2:
        results = analyze_phase2(trials)
    else:
        # Auto-detect phase based on metrics presence
        sample = trials[0]
        if 'state_cloning' in sample.get('cpp', {}) or 'state_clone_count' in sample.get('cpp', {}):
            print("Detected Phase 1 metrics, analyzing Phase 1...")
            results = analyze_phase1(trials)
        elif 'tensor_creation' in sample.get('cpp', {}) or 'openmp_thread_count' in sample.get('cpp', {}):
            print("Detected Phase 2 metrics, analyzing Phase 2...")
            results = analyze_phase2(trials)
        else:
            print("Could not auto-detect phase, defaulting to Phase 1 analysis...")
            results = analyze_phase1(trials)

    # Add comparison if requested
    if args.compare_to_baseline:
        compare_to_baseline(results)

    # Output for scripting (throughput only)
    if args.get_throughput:
        throughput = results['metrics'].get('throughput_sims_sec', {}).get('mean', 0)
        print(int(throughput))
        sys.exit(0)

    # Pretty print results
    print(f"=== Phase {results['phase']} Analysis Results ===")
    print()

    for metric_name, metric_data in results['metrics'].items():
        if isinstance(metric_data, dict):
            print(f"{metric_name}:")
            for key, value in metric_data.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"{metric_name}: {metric_data:.2f}")
        print()

    if 'comparison' in results:
        print("=== Baseline Comparison ===")
        comp = results['comparison']
        print(f"Baseline:     {comp['baseline_sims_sec']:.1f} sims/sec")
        print(f"Current:      {comp['current_sims_sec']:.1f} sims/sec")
        print(f"Speedup:      {comp['speedup']:.1f}×")
        print(f"Improvement:  {comp['improvement_pct']:.1f}%")
        print()

    # Write to file if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == '__main__':
    main()
