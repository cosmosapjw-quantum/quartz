#!/usr/bin/env python3
"""
Phase 2 Benchmark Harness: Tensor Pipeline + OpenMP
Purpose: Validate Phase 2 acceptance criteria (7,000-9,000 sims/sec, tensor <2ms, OpenMP >1)
Usage: python scripts/benchmark_phase2.py --trials 100 [--output profiling_phase2]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_single_trial(trial_num: int, args: argparse.Namespace) -> Dict:
    """Run a single MCTS trial with Phase 2 optimizations and collect profiling data."""
    try:
        # Import mcts_py
        import mcts_py

        # Enable instrumentation
        mcts_py.set_instrumentation_enabled(True)

        # Reset metrics
        mcts_py.reset_instrumentation()

        # Run MCTS search
        start_time = time.time()

        # Placeholder for actual MCTS run
        # This will be implemented during Phase 4 (User Story 2)
        # For now, return mock data structure

        elapsed_time = time.time() - start_time

        # Collect profiling metrics
        metrics = mcts_py.get_instrumentation_snapshot() if hasattr(mcts_py, 'get_instrumentation_snapshot') else {}

        # Calculate throughput
        simulations_per_second = args.simulations / elapsed_time if elapsed_time > 0 else 0

        return {
            'trial_num': trial_num,
            'elapsed_time': elapsed_time,
            'simulations': args.simulations,
            'simulations_per_second': simulations_per_second,
            'metrics': metrics,
            'batch_size': args.batch_size,
            'timestamp': datetime.now().isoformat()
        }

    except ImportError:
        print(f"Warning: mcts_py not available, skipping trial {trial_num}", file=sys.stderr)
        return {
            'trial_num': trial_num,
            'error': 'mcts_py not available',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error in trial {trial_num}: {e}", file=sys.stderr)
        return {
            'trial_num': trial_num,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def validate_phase2_acceptance(results: List[Dict]) -> Dict:
    """Validate Phase 2 acceptance criteria."""
    successful_trials = [r for r in results if 'error' not in r]

    if not successful_trials:
        return {
            'accepted': False,
            'reason': 'No successful trials'
        }

    # Calculate metrics
    throughputs = [r['simulations_per_second'] for r in successful_trials]
    avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0

    tensor_creation_times = []
    h2d_transfer_times = []
    openmp_thread_counts = []
    openmp_enabled_count = 0
    pinned_buffer_reuse_counts = []

    for result in successful_trials:
        metrics = result.get('metrics', {})

        if 'tensor_creation' in metrics:
            # Convert from ns to ms
            tensor_creation_times.append(metrics['tensor_creation'].get('total_elapsed_ns', 0) / 1e6)

        if 'h2d_transfer' in metrics:
            h2d_transfer_times.append(metrics['h2d_transfer'].get('total_elapsed_ns', 0) / 1e6)

        if 'openmp_thread_count' in metrics:
            openmp_thread_counts.append(metrics['openmp_thread_count'].get('call_count', 0))

        if 'openmp_enabled' in metrics and metrics['openmp_enabled'].get('call_count', 0) > 0:
            openmp_enabled_count += 1

        if 'pinned_buffer_reuse' in metrics:
            pinned_buffer_reuse_counts.append(metrics['pinned_buffer_reuse'].get('call_count', 0))

    # Calculate averages and percentiles
    avg_tensor_creation = sum(tensor_creation_times) / len(tensor_creation_times) if tensor_creation_times else 0
    p95_tensor_creation = sorted(tensor_creation_times)[int(len(tensor_creation_times) * 0.95)] if tensor_creation_times else 0

    avg_h2d_transfer = sum(h2d_transfer_times) / len(h2d_transfer_times) if h2d_transfer_times else 0

    avg_openmp_threads = sum(openmp_thread_counts) / len(openmp_thread_counts) if openmp_thread_counts else 0
    openmp_enabled_pct = (openmp_enabled_count / len(successful_trials) * 100) if successful_trials else 0

    total_pinned_reuse = sum(pinned_buffer_reuse_counts) if pinned_buffer_reuse_counts else 0

    # Check acceptance criteria
    criteria = {
        'throughput_target': (7000, 9000),
        'tensor_creation_p95_max_ms': 2.0,
        'h2d_transfer_max_ms': 1.0,
        'openmp_threads_min': 2,
        'openmp_enabled_min_pct': 95.0,
        'pinned_buffer_reuse_pct_min': 99.0
    }

    acceptance_checks = {
        'throughput_in_range': criteria['throughput_target'][0] <= avg_throughput <= criteria['throughput_target'][1],
        'tensor_creation_fast': p95_tensor_creation <= criteria['tensor_creation_p95_max_ms'],
        'h2d_transfer_fast': avg_h2d_transfer <= criteria['h2d_transfer_max_ms'],
        'openmp_threads_sufficient': avg_openmp_threads >= criteria['openmp_threads_min'],
        'openmp_enabled': openmp_enabled_pct >= criteria['openmp_enabled_min_pct']
    }

    all_passed = all(acceptance_checks.values())

    return {
        'accepted': all_passed,
        'metrics': {
            'avg_throughput_sims_sec': avg_throughput,
            'avg_tensor_creation_ms': avg_tensor_creation,
            'p95_tensor_creation_ms': p95_tensor_creation,
            'avg_h2d_transfer_ms': avg_h2d_transfer,
            'avg_openmp_thread_count': avg_openmp_threads,
            'openmp_enabled_pct': openmp_enabled_pct,
            'total_pinned_buffer_reuse': total_pinned_reuse
        },
        'criteria': criteria,
        'checks': acceptance_checks
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 2 benchmark harness (Tensor Pipeline + OpenMP)')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials to run (default: 100)')
    parser.add_argument('--simulations', type=int, default=800, help='Simulations per trial (default: 800)')
    parser.add_argument('--threads', type=int, default=8, help='Number of simulation threads (default: 8)')
    parser.add_argument('--batch-size', type=int, default=64, help='Inference batch size (default: 64)')
    parser.add_argument('--game', choices=['gomoku', 'chess', 'go'], default='gomoku', help='Game to benchmark (default: gomoku)')
    parser.add_argument('--output', type=Path, help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')

    args = parser.parse_args()

    print("=" * 80)
    print("Phase 2 Benchmark: Tensor Pipeline + OpenMP")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Trials:      {args.trials}")
    print(f"  Simulations: {args.simulations}")
    print(f"  Threads:     {args.threads}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Game:        {args.game}")
    print()

    # Setup output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"profiling_phase2_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_dir = output_dir / "campaign"
    campaign_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Run trials
    results = []
    start_time = time.time()

    for trial_num in range(1, args.trials + 1):
        if args.verbose or trial_num % 10 == 0:
            print(f"Running trial {trial_num}/{args.trials}...", end='')
            sys.stdout.flush()

        result = run_single_trial(trial_num, args)
        results.append(result)

        # Save trial result
        trial_dir = campaign_dir / f"trial_{trial_num:03d}"
        trial_dir.mkdir(exist_ok=True)

        with open(trial_dir / "cpp_profiling.json", 'w') as f:
            json.dump(result, f, indent=2)

        if args.verbose or trial_num % 10 == 0:
            throughput = result.get('simulations_per_second', 0)
            print(f" {throughput:.0f} sims/sec")

    elapsed = time.time() - start_time
    print()
    print(f"Completed {args.trials} trials in {elapsed:.1f} seconds")
    print()

    # Validate acceptance criteria
    print("=" * 80)
    print("Phase 2 Acceptance Validation")
    print("=" * 80)

    validation = validate_phase2_acceptance(results)

    print(f"Metrics:")
    for metric, value in validation['metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")
    print()

    print(f"Acceptance Criteria:")
    for check, passed in validation['checks'].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    print()

    if validation['accepted']:
        print("🎯 ✅ Phase 2 ACCEPTED - PRIMARY TARGET ACHIEVED!")
        print(f"   Throughput: {validation['metrics']['avg_throughput_sims_sec']:.0f} sims/sec (target: 7,000-9,000)")
        print(f"   Tensor creation (p95): {validation['metrics']['p95_tensor_creation_ms']:.2f}ms (target: ≤2.0ms)")
        print(f"   OpenMP threads: {validation['metrics']['avg_openmp_thread_count']:.1f} (target: ≥2)")
    else:
        print("❌ Phase 2 REJECTED")
        print("   Rollback procedure:")
        print("     git revert HEAD~15..HEAD  # Revert Phase 2 commits")
        print("     pip install -e . --force-reinstall --no-deps")
        print("     scripts/validate_all_phases.sh --verify-phase1")

    # Save summary
    summary = {
        'phase': 2,
        'trials': args.trials,
        'configuration': vars(args),
        'validation': validation,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Results saved to {output_dir}/")

    # Exit with appropriate code
    sys.exit(0 if validation['accepted'] else 1)


if __name__ == '__main__':
    main()
