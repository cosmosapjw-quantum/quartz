#!/usr/bin/env python3
"""
Phase 1 Benchmark Harness: State Cloning Elimination
Purpose: Validate Phase 1 acceptance criteria (1,500-3,000 sims/sec, <1% state cloning)
Usage: python scripts/benchmark_phase1.py --trials 100 [--output profiling_phase1]
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
    """Run a single MCTS trial and collect profiling data."""
    try:
        # Import mcts_py (will be available after Phase 1 implementation)
        import mcts_py

        # Enable instrumentation
        mcts_py.set_instrumentation_enabled(True)

        # Reset metrics
        mcts_py.reset_instrumentation()

        # Run MCTS search
        start_time = time.time()

        # Placeholder for actual MCTS run
        # This will be implemented during Phase 3 (User Story 1)
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


def validate_phase1_acceptance(results: List[Dict]) -> Dict:
    """Validate Phase 1 acceptance criteria."""
    successful_trials = [r for r in results if 'error' not in r]

    if not successful_trials:
        return {
            'accepted': False,
            'reason': 'No successful trials'
        }

    # Calculate metrics
    throughputs = [r['simulations_per_second'] for r in successful_trials]
    avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0

    state_clone_counts = []
    state_cloning_times = []
    total_times = []

    for result in successful_trials:
        metrics = result.get('metrics', {})

        if 'state_clone_count' in metrics:
            state_clone_counts.append(metrics['state_clone_count'].get('call_count', 0))

        if 'state_cloning' in metrics:
            state_cloning_times.append(metrics['state_cloning'].get('total_elapsed_ns', 0))

        total_times.append(result['elapsed_time'] * 1e9)  # Convert to ns

    avg_state_clones = sum(state_clone_counts) / len(state_clone_counts) if state_clone_counts else 0
    total_state_cloning_time = sum(state_cloning_times)
    total_execution_time = sum(total_times)

    state_cloning_pct = (total_state_cloning_time / total_execution_time * 100) if total_execution_time > 0 else 0

    # Check acceptance criteria
    criteria = {
        'throughput_target': (1500, 3000),
        'state_cloning_pct_max': 1.0,
        'state_clone_count_max': 0
    }

    acceptance_checks = {
        'throughput_in_range': criteria['throughput_target'][0] <= avg_throughput <= criteria['throughput_target'][1],
        'state_cloning_below_threshold': state_cloning_pct < criteria['state_cloning_pct_max'],
        'zero_state_clones': avg_state_clones == 0
    }

    all_passed = all(acceptance_checks.values())

    return {
        'accepted': all_passed,
        'metrics': {
            'avg_throughput_sims_sec': avg_throughput,
            'state_cloning_pct': state_cloning_pct,
            'avg_state_clone_count': avg_state_clones
        },
        'criteria': criteria,
        'checks': acceptance_checks
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 1 benchmark harness (State Cloning Elimination)')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials to run (default: 100)')
    parser.add_argument('--simulations', type=int, default=800, help='Simulations per trial (default: 800)')
    parser.add_argument('--threads', type=int, default=8, help='Number of simulation threads (default: 8)')
    parser.add_argument('--game', choices=['gomoku', 'chess', 'go'], default='gomoku', help='Game to benchmark (default: gomoku)')
    parser.add_argument('--output', type=Path, help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')

    args = parser.parse_args()

    print("=" * 80)
    print("Phase 1 Benchmark: State Cloning Elimination")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Trials:      {args.trials}")
    print(f"  Simulations: {args.simulations}")
    print(f"  Threads:     {args.threads}")
    print(f"  Game:        {args.game}")
    print()

    # Setup output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"profiling_phase1_{timestamp}")

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
    print("Phase 1 Acceptance Validation")
    print("=" * 80)

    validation = validate_phase1_acceptance(results)

    print(f"Metrics:")
    for metric, value in validation['metrics'].items():
        print(f"  {metric}: {value:.2f}")
    print()

    print(f"Acceptance Criteria:")
    for check, passed in validation['checks'].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    print()

    if validation['accepted']:
        print("🎉 ✅ Phase 1 ACCEPTED")
        print(f"   Throughput: {validation['metrics']['avg_throughput_sims_sec']:.0f} sims/sec (target: 1,500-3,000)")
        print(f"   State cloning: {validation['metrics']['state_cloning_pct']:.2f}% (target: <1%)")
    else:
        print("❌ Phase 1 REJECTED")
        print("   Rollback procedure:")
        print("     git revert HEAD~10..HEAD")
        print("     pip install -e . --force-reinstall --no-deps")
        print("     scripts/validate_all_phases.sh --verify-baseline")

    # Save summary
    summary = {
        'phase': 1,
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
