#!/usr/bin/env python3
"""
Benchmark Neural Network Architectures (comments.md validation)
================================================================

Validates the expected throughput targets for ResNet-ECA architectures:
- ResNet-ECA 128×12: 28-40k pps (3.7M params)
- Ghost-ResNet-ECA 96×12: 49-70k pps (2.2M params)
- Baseline AlphaZeroNet 192×15: 10-14k pps (10M params)

Reference: specs/005-mcts-throughput-optimization/comments.md
"""

import torch
import torch.nn as nn
import time
import argparse
from typing import Dict, List, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.neural.model import (
    create_resnet_eca_model,
    create_ghost_resnet_eca_model,
    create_model_for_game,
)


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int],
    batch_sizes: List[int] = [16, 32, 64, 96, 128, 256],
    num_warmup: int = 10,
    num_iterations: int = 100,
    use_amp: bool = True,
    device: str = 'cuda'
) -> Dict:
    """Benchmark a model across different batch sizes.

    Args:
        model: Neural network to benchmark
        input_shape: Input tensor shape (batch, channels, height, width)
        batch_sizes: List of batch sizes to test
        num_warmup: Number of warmup iterations
        num_iterations: Number of timing iterations
        use_amp: Whether to use FP16 mixed precision
        device: Device to run on

    Returns:
        Dictionary with benchmark results
    """
    model = model.to(device)
    model.eval()

    results = {}

    for batch_size in batch_sizes:
        # Create input tensor
        x = torch.randn(batch_size, *input_shape[1:], device=device)

        # Warmup
        print(f"  Warming up batch size {batch_size}...", end='', flush=True)
        for _ in range(num_warmup):
            with torch.no_grad():
                if use_amp and device == 'cuda':
                    with torch.amp.autocast('cuda'):
                        policy, value = model(x)
                else:
                    policy, value = model(x)

        if device == 'cuda':
            torch.cuda.synchronize()
        print(" done")

        # Benchmark
        print(f"  Benchmarking batch size {batch_size}...", end='', flush=True)
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            with torch.no_grad():
                if use_amp and device == 'cuda':
                    with torch.amp.autocast('cuda'):
                        policy, value = model(x)
                else:
                    policy, value = model(x)

        if device == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        print(" done")

        # Calculate metrics
        time_per_batch_ms = (elapsed / num_iterations) * 1000
        positions_per_second = (batch_size * num_iterations) / elapsed

        results[batch_size] = {
            'time_per_batch_ms': time_per_batch_ms,
            'positions_per_second': positions_per_second,
        }

    return results


def print_results_table(
    results: Dict[str, Dict],
    model_name: str,
    expected_pps: Tuple[int, int],
    params: int
):
    """Print formatted results table.

    Args:
        results: Benchmark results dictionary
        model_name: Name of the model
        expected_pps: Expected (min, max) positions per second
        params: Number of parameters
    """
    print(f"\n{'='*80}")
    print(f"{model_name}")
    print(f"Parameters: {params:,} (~{params/1e6:.1f}M)")
    print(f"Expected throughput: {expected_pps[0]/1000:.0f}-{expected_pps[1]/1000:.0f}k pps")
    print(f"{'='*80}")
    print(f"{'Batch':>6} | {'Time/Batch (ms)':>16} | {'Throughput (pps)':>18} | Status")
    print(f"{'-'*6}-+-{'-'*16}-+-{'-'*18}-+{'-'*10}")

    best_pps = 0
    for batch_size, metrics in sorted(results.items()):
        time_ms = metrics['time_per_batch_ms']
        pps = metrics['positions_per_second']
        best_pps = max(best_pps, pps)

        # Check if within expected range
        if expected_pps[0] <= pps <= expected_pps[1]:
            status = "✅ TARGET"
        elif pps > expected_pps[1]:
            status = "✅ EXCEED"
        elif pps >= expected_pps[0] * 0.8:
            status = "⚠️  NEAR"
        else:
            status = "❌ BELOW"

        print(f"{batch_size:6d} | {time_ms:16.2f} | {pps:18.1f} | {status}")

    print(f"{'-'*80}")
    print(f"Peak throughput: {best_pps:,.1f} pps ({best_pps/1000:.1f}k pps)")

    # Overall verdict
    if best_pps >= expected_pps[0]:
        print(f"✅ PASS: Achieved {best_pps/expected_pps[0]:.2f}× minimum target")
    else:
        print(f"❌ FAIL: Only {best_pps/expected_pps[0]:.2f}× minimum target")
    print()


def main():
    parser = argparse.ArgumentParser(description='Benchmark NN architectures')
    parser.add_argument('--game', type=str, default='gomoku',
                       choices=['gomoku', 'chess', 'go', 'go9', 'go19'],
                       help='Game type')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       default=[16, 32, 64, 96, 128, 256],
                       help='Batch sizes to test')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision (FP16)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['all'],
                       choices=['all', 'resnet-eca-128', 'resnet-eca-96', 'ghost-eca', 'baseline'],
                       help='Models to benchmark')

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == 'cuda':
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Determine input shape based on game
    if args.game == 'gomoku':
        input_shape = (1, 36, 15, 15)
    elif args.game == 'chess':
        input_shape = (1, 30, 8, 8)
    elif args.game == 'go' or args.game == 'go9':
        input_shape = (1, 25, 9, 9)
    elif args.game == 'go19':
        input_shape = (1, 25, 19, 19)
    else:
        raise ValueError(f"Unsupported game: {args.game}")

    print(f"\n{'='*80}")
    print(f"Neural Network Architecture Benchmark")
    print(f"{'='*80}")
    print(f"Game: {args.game}")
    print(f"Input shape: {input_shape}")
    print(f"Device: {args.device}")
    print(f"Mixed precision: {not args.no_amp and args.device == 'cuda'}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.iterations}")
    print(f"Batch sizes: {args.batch_sizes}")
    print()

    models_to_test = args.models if 'all' not in args.models else ['resnet-eca-128', 'resnet-eca-96', 'ghost-eca', 'baseline']

    # Benchmark ResNet-ECA 128×12 (TOP RECOMMENDATION)
    if 'resnet-eca-128' in models_to_test:
        print("\n" + "="*80)
        print("RESNET-ECA 128×12 (TOP RECOMMENDATION - comments.md)")
        print("="*80)
        model = create_resnet_eca_model(args.game, size='128x12')
        params = model.get_num_parameters()
        print(f"Created model with {params:,} parameters (~{params/1e6:.2f}M)")

        results = benchmark_model(
            model, input_shape,
            batch_sizes=args.batch_sizes,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
            use_amp=not args.no_amp,
            device=args.device
        )

        print_results_table(
            results,
            "ResNet-ECA 128×12 (comments.md top pick)",
            expected_pps=(28000, 40000),  # 28-40k pps target
            params=params
        )

    # Benchmark ResNet-ECA 96×12 (FAST VARIANT)
    if 'resnet-eca-96' in models_to_test:
        print("\n" + "="*80)
        print("RESNET-ECA 96×12 (FAST VARIANT)")
        print("="*80)
        model = create_resnet_eca_model(args.game, size='96x12')
        params = model.get_num_parameters()
        print(f"Created model with {params:,} parameters (~{params/1e6:.2f}M)")

        results = benchmark_model(
            model, input_shape,
            batch_sizes=args.batch_sizes,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
            use_amp=not args.no_amp,
            device=args.device
        )

        print_results_table(
            results,
            "ResNet-ECA 96×12 (fast variant)",
            expected_pps=(35000, 50000),  # Interpolated from table
            params=params
        )

    # Benchmark Ghost-ResNet-ECA 96×12 (ULTRA-LIGHT)
    if 'ghost-eca' in models_to_test:
        print("\n" + "="*80)
        print("GHOST-RESNET-ECA 96×12 (ULTRA-LIGHT)")
        print("="*80)
        model = create_ghost_resnet_eca_model(args.game)
        params = model.get_num_parameters()
        print(f"Created model with {params:,} parameters (~{params/1e6:.2f}M)")

        results = benchmark_model(
            model, input_shape,
            batch_sizes=args.batch_sizes,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
            use_amp=not args.no_amp,
            device=args.device
        )

        print_results_table(
            results,
            "Ghost-ResNet-ECA 96×12 (ultra-light)",
            expected_pps=(49000, 70000),  # 49-70k pps target
            params=params
        )

    # Benchmark baseline AlphaZeroNet 192×15
    if 'baseline' in models_to_test:
        print("\n" + "="*80)
        print("BASELINE: AlphaZeroNet 192×15 (CURRENT)")
        print("="*80)
        model = create_model_for_game(args.game)
        params = model.get_num_parameters()
        print(f"Created model with {params:,} parameters (~{params/1e6:.2f}M)")

        results = benchmark_model(
            model, input_shape,
            batch_sizes=args.batch_sizes,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
            use_amp=not args.no_amp,
            device=args.device
        )

        print_results_table(
            results,
            "AlphaZeroNet 192×15 (baseline)",
            expected_pps=(10000, 14000),  # 10-14k pps expected
            params=params
        )

    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    print()
    print("Based on comments.md analysis:")
    print("  - ResNet-ECA 128×12 is the TOP RECOMMENDATION for Gomoku")
    print("  - Expected 3× speedup vs 192×15 baseline with minimal Elo loss")
    print("  - Ghost-ResNet-ECA 96×12 provides 5-7× speedup for maximum throughput")
    print()
    print("Next steps:")
    print("  1. If targets achieved: Update create_model_for_game() to use ResNet-ECA by default")
    print("  2. Update DLPackInferenceBridge with FP16 I/O, CUDA graphs, adaptive batching")
    print("  3. Re-run MCTS profiling to validate end-to-end throughput improvements")
    print()


if __name__ == '__main__':
    main()
