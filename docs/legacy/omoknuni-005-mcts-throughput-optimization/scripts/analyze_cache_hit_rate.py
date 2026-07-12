#!/usr/bin/env python3
"""
Cache Hit Rate Analysis Script (Phase 1A)
==========================================

Measures the ratio of MCTS simulations to NN evaluations to identify
if low GPU utilization is caused by high cache hit rate.

Usage:
    python scripts/analyze_cache_hit_rate.py --simulations 800 --threads 8

Output:
    - Total MCTS simulations
    - Total NN evaluations
    - Cache hit rate (%)
    - Analysis: Is cache hit rate the bottleneck?
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.core.mcts import AlphaZeroMCTS
from src.neural.model import create_model_for_game
from src.neural.inference_worker import GPUInferenceWorker
import alphazero_py


def analyze_cache_hit_rate(game_name='gomoku', simulations=800, num_threads=8):
    """Analyze cache hit rate in MCTS search."""
    print("=" * 70)
    print("CACHE HIT RATE ANALYSIS (Phase 1A)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Game:        {game_name}")
    print(f"  Simulations: {simulations}")
    print(f"  Threads:     {num_threads}")
    print()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create initial state
    if game_name == 'gomoku':
        initial_state = alphazero_py.GomokuState()
    elif game_name == 'chess':
        initial_state = alphazero_py.ChessState()
    elif game_name == 'go':
        initial_state = alphazero_py.GoState()
    else:
        raise ValueError(f"Unknown game: {game_name}")

    # Create GPU inference worker (will load model internally)
    print("\nCreating GPUInferenceWorker...")
    worker = GPUInferenceWorker(
        model_path=None,  # Use default random weights
        device=device.type,
        batch_size=64,
        timeout_ms=3.0,
        use_mixed_precision=False  # FP32 for consistent comparison
    )

    # Create MCTS engine
    print("Creating MCTS engine...")
    mcts = AlphaZeroMCTS(
        inference_fn=worker,
        num_threads=num_threads,
        use_async_inference=True,
        async_batch_size=16,
        async_timeout_ms=10.0
    )

    try:
        # Run search
        print(f"\nRunning search with {simulations} simulations...")
        start_time = time.perf_counter()
        mcts.search(initial_state, simulations=simulations)
        elapsed_time = time.perf_counter() - start_time

        # Get statistics
        mcts_stats = mcts.get_statistics()
        worker_metrics = worker.get_metrics()

        # Calculate metrics
        total_simulations = mcts_stats['simulations_completed']
        nn_evaluations = worker_metrics['nn_evaluations']
        states_evaluated = worker_metrics['states_evaluated']

        # Cache hit rate calculation
        # If cache hit rate is high, many simulations reuse existing evaluations
        # Formula: cache_hit_rate = 1 - (states_evaluated / total_simulations)
        # Note: states_evaluated counts the number of unique states sent to NN
        cache_hit_rate = 1.0 - (states_evaluated / total_simulations) if total_simulations > 0 else 0.0
        cache_hit_rate_pct = cache_hit_rate * 100

        # Throughput
        throughput = total_simulations / elapsed_time

        # Results
        print("\n" + "=" * 70)
        print("ANALYSIS RESULTS")
        print("=" * 70)

        print("\nPerformance Metrics:")
        print(f"  Total time:           {elapsed_time:.3f}s")
        print(f"  Throughput:           {throughput:.1f} sims/sec")
        print(f"  GPU utilization:      {worker_metrics['avg_gpu_utilization']*100:.1f}%")
        print(f"  Avg batch size:       {worker_metrics['average_batch_size']:.1f}")

        print("\nMCTS Statistics:")
        print(f"  Total simulations:    {total_simulations}")
        print(f"  Tree size (nodes):    {mcts_stats['tree_size']}")

        print("\nNN Evaluation Statistics:")
        print(f"  Total NN evaluations: {nn_evaluations}")
        print(f"  States evaluated:     {states_evaluated}")
        print(f"  Avg states/batch:     {states_evaluated/nn_evaluations if nn_evaluations > 0 else 0:.1f}")

        print("\nCache Hit Rate Analysis:")
        print(f"  States evaluated:     {states_evaluated}")
        print(f"  Total simulations:    {total_simulations}")
        print(f"  Evaluation ratio:     {states_evaluated/total_simulations if total_simulations > 0 else 0:.3f}")
        print(f"  Cache hit rate:       {cache_hit_rate_pct:.1f}%")

        # Interpretation
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)

        if cache_hit_rate_pct > 50:
            print(f"🔴 HIGH CACHE HIT RATE ({cache_hit_rate_pct:.1f}%)")
            print("\nThis explains low GPU utilization!")
            print(f"- Only {100-cache_hit_rate_pct:.1f}% of simulations require new NN evaluations")
            print(f"- {cache_hit_rate_pct:.1f}% of simulations reuse existing values")
            print("\nRoot cause: MCTS is highly efficient at reusing tree values.")
            print("This is EXPECTED BEHAVIOR for MCTS, not a bug.")
            print("\nIMPLICATION: Low GPU utilization (30%) is CORRECT.")
            print("The GPU is idle because threads don't need constant NN evaluations.")
        elif cache_hit_rate_pct > 20:
            print(f"⚠️  MODERATE CACHE HIT RATE ({cache_hit_rate_pct:.1f}%)")
            print("\nCache hits contribute to low GPU utilization, but not the primary cause.")
            print(f"- {100-cache_hit_rate_pct:.1f}% of simulations require NN evaluations")
            print(f"- {cache_hit_rate_pct:.1f}% reuse existing values")
            print("\nNeed further investigation: Thread coordination or tree search overhead.")
        else:
            print(f"✅ LOW CACHE HIT RATE ({cache_hit_rate_pct:.1f}%)")
            print("\nCache hits are NOT the bottleneck.")
            print(f"- {100-cache_hit_rate_pct:.1f}% of simulations require NN evaluations")
            print("\nLow GPU utilization is caused by:")
            print("  1. Thread coordination overhead (virtual loss, contention)")
            print("  2. Tree search overhead (PUCT, state cloning)")
            print("  3. Coordinator timing issues")
            print("\nNext step: Enable C++ profiling (Phase 1B) to identify specific bottleneck.")

        print("\n" + "=" * 70)

        return {
            'throughput': throughput,
            'cache_hit_rate': cache_hit_rate,
            'total_simulations': total_simulations,
            'nn_evaluations': nn_evaluations,
            'states_evaluated': states_evaluated,
            'gpu_utilization': worker_metrics['avg_gpu_utilization'],
        }

    finally:
        mcts.close()
        worker.stop_worker()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze cache hit rate in MCTS search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--game', type=str, default='gomoku',
                       choices=['gomoku', 'chess', 'go'],
                       help='Game type (default: gomoku)')
    parser.add_argument('--simulations', type=int, default=800,
                       help='Number of simulations (default: 800)')
    parser.add_argument('--threads', type=int, default=8,
                       help='Number of threads (default: 8)')

    args = parser.parse_args()

    # Run analysis
    results = analyze_cache_hit_rate(
        game_name=args.game,
        simulations=args.simulations,
        num_threads=args.threads
    )

    # Save results
    output_file = Path(__file__).parent.parent / "cache_hit_analysis_results.txt"
    with open(output_file, 'w') as f:
        f.write("Cache Hit Rate Analysis Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Throughput:        {results['throughput']:.1f} sims/sec\n")
        f.write(f"Cache hit rate:    {results['cache_hit_rate']*100:.1f}%\n")
        f.write(f"Total simulations: {results['total_simulations']}\n")
        f.write(f"NN evaluations:    {results['nn_evaluations']}\n")
        f.write(f"States evaluated:  {results['states_evaluated']}\n")
        f.write(f"GPU utilization:   {results['gpu_utilization']*100:.1f}%\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
