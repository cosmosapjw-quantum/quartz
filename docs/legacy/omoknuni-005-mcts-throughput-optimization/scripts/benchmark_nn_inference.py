#!/usr/bin/env python3
"""
Neural Network Inference Speed Benchmark
=========================================

Comprehensive benchmarking of AlphaZeroNet vs FastMCTSNet inference speed.

Tests:
- AlphaZeroNet (baseline)
- FastMCTSNet (training mode - multi-branch)
- FastMCTSNet (deploy mode - fused)
- FastMCTSNet (with early exits)
- Various batch sizes (1, 16, 32, 64, 128)
- CPU and GPU (if available)

Usage:
    python scripts/benchmark_nn_inference.py --game gomoku --iterations 1000
    python scripts/benchmark_nn_inference.py --game chess --batch-sizes 32,64 --device cuda
"""

import torch
import torch.nn as nn
import time
import numpy as np
import argparse
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Add project root to path
sys.path.insert(0, '.')

from src.neural.model import (
    create_model_for_game,
    create_fast_model_for_game,
    AlphaZeroNet,
    FastMCTSNet
)


@dataclass
class BenchmarkResult:
    """Benchmark result for a single configuration."""
    model_name: str
    mode: str  # 'training', 'deploy', 'early_exit'
    batch_size: int
    iterations: int
    total_time: float
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: float  # samples/sec
    device: str


class NeuralNetworkBenchmark:
    """Comprehensive neural network inference benchmark."""

    def __init__(self,
                 game: str = 'gomoku',
                 device: str = 'cuda',
                 warmup_iterations: int = 10,
                 verbose: bool = True):
        """
        Args:
            game: Game type ('gomoku', 'chess', 'go')
            device: Device to run on ('cuda', 'cpu')
            warmup_iterations: Number of warmup iterations
            verbose: Print progress messages
        """
        self.game = game
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        self.warmup_iterations = warmup_iterations
        self.verbose = verbose

        # Determine input shape based on game
        self.input_shapes = {
            'gomoku': (36, 15, 15),
            'chess': (30, 8, 8),
            'go': (25, 19, 19),
        }
        self.input_shape = self.input_shapes.get(game, (36, 15, 15))

        if self.verbose:
            print(f"Neural Network Inference Benchmark")
            print(f"=" * 60)
            print(f"Game:          {game}")
            print(f"Device:        {self.device}")
            print(f"Input shape:   {self.input_shape}")
            if self.device == 'cuda':
                print(f"GPU:           {torch.cuda.get_device_name(0)}")
            print(f"=" * 60)

    def _warmup(self, model: nn.Module, batch_size: int):
        """Warmup GPU with dummy inference calls."""
        if self.verbose:
            print(f"  Warming up (batch_size={batch_size})...", end='', flush=True)

        dummy_input = torch.randn(batch_size, *self.input_shape, device=self.device)

        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(dummy_input)
                if self.device == 'cuda':
                    torch.cuda.synchronize()

        if self.verbose:
            print(" done")

    def _benchmark_single(self,
                         model: nn.Module,
                         batch_size: int,
                         iterations: int,
                         inference_mode: bool = False) -> Tuple[float, List[float]]:
        """
        Benchmark a single configuration.

        Args:
            model: Model to benchmark
            batch_size: Batch size
            iterations: Number of iterations
            inference_mode: Whether to use inference_mode (for early exits)

        Returns:
            (total_time, list of per-iteration times)
        """
        # Warmup
        self._warmup(model, batch_size)

        # Generate test input
        test_input = torch.randn(batch_size, *self.input_shape, device=self.device)

        # Benchmark
        times = []
        model.eval()

        with torch.no_grad():
            for i in range(iterations):
                if self.device == 'cuda':
                    torch.cuda.synchronize()

                start = time.perf_counter()

                # Forward pass
                if isinstance(model, FastMCTSNet):
                    _ = model(test_input, inference_mode=inference_mode)
                else:
                    _ = model(test_input)

                if self.device == 'cuda':
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start
                times.append(elapsed)

                # Progress indicator
                if self.verbose and i % max(1, iterations // 10) == 0:
                    print(f"    Progress: {i}/{iterations}", end='\r', flush=True)

        if self.verbose:
            print(f"    Progress: {iterations}/{iterations} - Complete!")

        return sum(times), times

    def benchmark_model(self,
                       model_name: str,
                       model: nn.Module,
                       batch_sizes: List[int],
                       iterations: int,
                       modes: Optional[List[str]] = None) -> List[BenchmarkResult]:
        """
        Benchmark a model across multiple configurations.

        Args:
            model_name: Name of the model
            model: Model instance
            batch_sizes: List of batch sizes to test
            iterations: Number of iterations per config
            modes: List of modes to test (for FastMCTSNet)

        Returns:
            List of BenchmarkResult
        """
        results = []

        if modes is None:
            modes = ['training']  # Default mode for AlphaZeroNet

        for mode in modes:
            if self.verbose:
                print(f"\n{model_name} - Mode: {mode}")
                print("-" * 60)

            # Configure model for mode
            if mode == 'deploy' and isinstance(model, FastMCTSNet):
                model.switch_to_deploy()
                if self.verbose:
                    print("  Switched to deploy mode (fused convs)")

            for batch_size in batch_sizes:
                if self.verbose:
                    print(f"\n  Batch size: {batch_size}")

                # Run benchmark
                inference_mode = (mode == 'early_exit')
                total_time, times = self._benchmark_single(
                    model, batch_size, iterations, inference_mode
                )

                # Compute statistics
                times_array = np.array(times)
                mean_time = np.mean(times_array)
                std_time = np.std(times_array)
                min_time = np.min(times_array)
                max_time = np.max(times_array)
                throughput = (batch_size * iterations) / total_time

                result = BenchmarkResult(
                    model_name=model_name,
                    mode=mode,
                    batch_size=batch_size,
                    iterations=iterations,
                    total_time=total_time,
                    mean_time=mean_time,
                    std_time=std_time,
                    min_time=min_time,
                    max_time=max_time,
                    throughput=throughput,
                    device=self.device
                )
                results.append(result)

                if self.verbose:
                    print(f"    Mean time:   {mean_time*1000:.3f} ms")
                    print(f"    Std dev:     {std_time*1000:.3f} ms")
                    print(f"    Min time:    {min_time*1000:.3f} ms")
                    print(f"    Max time:    {max_time*1000:.3f} ms")
                    print(f"    Throughput:  {throughput:.1f} samples/sec")

        return results

    def run_comparison(self,
                      batch_sizes: List[int],
                      iterations: int) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comprehensive comparison between AlphaZeroNet and FastMCTSNet.

        Args:
            batch_sizes: List of batch sizes to test
            iterations: Number of iterations per config

        Returns:
            Dictionary mapping model names to results
        """
        all_results = {}

        # 1. Benchmark AlphaZeroNet (baseline)
        if self.verbose:
            print(f"\n{'='*60}")
            print("BENCHMARK 1/3: AlphaZeroNet (Baseline)")
            print(f"{'='*60}")

        alphazero_model = create_model_for_game(self.game, use_fast_model=False)
        alphazero_model = alphazero_model.to(self.device)
        alphazero_model.eval()

        alphazero_results = self.benchmark_model(
            model_name='AlphaZeroNet',
            model=alphazero_model,
            batch_sizes=batch_sizes,
            iterations=iterations,
            modes=['training']
        )
        all_results['AlphaZeroNet'] = alphazero_results

        # 2. Benchmark FastMCTSNet (training mode)
        if self.verbose:
            print(f"\n{'='*60}")
            print("BENCHMARK 2/3: FastMCTSNet (Training Mode)")
            print(f"{'='*60}")

        fast_model_train = create_fast_model_for_game(self.game)
        fast_model_train = fast_model_train.to(self.device)
        fast_model_train.eval()

        fast_train_results = self.benchmark_model(
            model_name='FastMCTSNet',
            model=fast_model_train,
            batch_sizes=batch_sizes,
            iterations=iterations,
            modes=['training']
        )
        all_results['FastMCTSNet-Training'] = fast_train_results

        # 3. Benchmark FastMCTSNet (deploy mode + early exits)
        if self.verbose:
            print(f"\n{'='*60}")
            print("BENCHMARK 3/3: FastMCTSNet (Deploy + Early Exit)")
            print(f"{'='*60}")

        fast_model_deploy = create_fast_model_for_game(self.game)
        fast_model_deploy = fast_model_deploy.to(self.device)
        fast_model_deploy.eval()

        fast_deploy_results = self.benchmark_model(
            model_name='FastMCTSNet',
            model=fast_model_deploy,
            batch_sizes=batch_sizes,
            iterations=iterations,
            modes=['deploy', 'early_exit']
        )
        all_results['FastMCTSNet-Deploy'] = fast_deploy_results[:len(batch_sizes)]
        all_results['FastMCTSNet-EarlyExit'] = fast_deploy_results[len(batch_sizes):]

        return all_results

    def print_comparison_table(self, all_results: Dict[str, List[BenchmarkResult]]):
        """Print formatted comparison table."""
        print(f"\n{'='*80}")
        print("INFERENCE SPEED COMPARISON")
        print(f"{'='*80}")

        # Group by batch size
        batch_sizes = sorted(set(r.batch_size for results in all_results.values() for r in results))

        for batch_size in batch_sizes:
            print(f"\nBatch Size: {batch_size}")
            print("-" * 80)
            print(f"{'Model':<25} {'Mode':<15} {'Time (ms)':<12} {'Throughput':<15} {'Speedup':<10}")
            print("-" * 80)

            # Get baseline (AlphaZeroNet)
            baseline_result = next(
                (r for r in all_results.get('AlphaZeroNet', []) if r.batch_size == batch_size),
                None
            )
            baseline_throughput = baseline_result.throughput if baseline_result else 1.0

            # Print all results for this batch size
            for model_key, results in all_results.items():
                for result in results:
                    if result.batch_size == batch_size:
                        speedup = result.throughput / baseline_throughput
                        speedup_str = f"{speedup:.2f}×" if speedup != 1.0 else "baseline"

                        print(f"{result.model_name:<25} {result.mode:<15} "
                              f"{result.mean_time*1000:<12.3f} "
                              f"{result.throughput:<15.1f} "
                              f"{speedup_str:<10}")

        print("=" * 80)

    def print_summary(self, all_results: Dict[str, List[BenchmarkResult]]):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")

        # Calculate average speedups across all batch sizes
        alphazero_results = all_results.get('AlphaZeroNet', [])

        if not alphazero_results:
            print("No baseline results available")
            return

        avg_baseline_throughput = np.mean([r.throughput for r in alphazero_results])

        print(f"\nAverage Throughput (samples/sec):")
        print("-" * 80)

        for model_key, results in all_results.items():
            if not results:
                continue
            avg_throughput = np.mean([r.throughput for r in results])
            speedup = avg_throughput / avg_baseline_throughput

            print(f"  {model_key:<30} {avg_throughput:>12.1f}  ({speedup:.2f}× baseline)")

        # Best configuration
        print(f"\nBest Configuration:")
        print("-" * 80)

        best_result = max(
            (r for results in all_results.values() for r in results),
            key=lambda r: r.throughput
        )
        baseline_best = max(alphazero_results, key=lambda r: r.throughput)

        print(f"  Model:       {best_result.model_name}")
        print(f"  Mode:        {best_result.mode}")
        print(f"  Batch size:  {best_result.batch_size}")
        print(f"  Throughput:  {best_result.throughput:.1f} samples/sec")
        print(f"  Speedup:     {best_result.throughput/baseline_best.throughput:.2f}× over best AlphaZeroNet")

        print("=" * 80)

    def export_results(self, all_results: Dict[str, List[BenchmarkResult]], filename: str):
        """Export results to JSON file."""
        output = {
            'game': self.game,
            'device': self.device,
            'input_shape': self.input_shape,
            'results': {}
        }

        for model_key, results in all_results.items():
            output['results'][model_key] = [
                {
                    'model_name': r.model_name,
                    'mode': r.mode,
                    'batch_size': r.batch_size,
                    'iterations': r.iterations,
                    'mean_time_ms': r.mean_time * 1000,
                    'std_time_ms': r.std_time * 1000,
                    'throughput': r.throughput,
                }
                for r in results
            ]

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults exported to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark neural network inference speed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark (Gomoku, GPU, default batch sizes)
  python scripts/benchmark_nn_inference.py

  # Chess with specific batch sizes
  python scripts/benchmark_nn_inference.py --game chess --batch-sizes 16,32,64

  # CPU benchmark with more iterations
  python scripts/benchmark_nn_inference.py --device cpu --iterations 500

  # Quick test
  python scripts/benchmark_nn_inference.py --quick
        """
    )

    parser.add_argument(
        '--game',
        type=str,
        default='gomoku',
        choices=['gomoku', 'chess', 'go'],
        help='Game type (default: gomoku)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    parser.add_argument(
        '--batch-sizes',
        type=str,
        default='1,16,32,64,128',
        help='Comma-separated batch sizes (default: 1,16,32,64,128)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Number of iterations per config (default: 1000)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup iterations (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test (fewer iterations, smaller batches)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    # Quick mode overrides
    if args.quick:
        batch_sizes = [32, 64]
        iterations = 100
        print("Quick mode: batch_sizes=[32,64], iterations=100")
    else:
        iterations = args.iterations

    # Run benchmark
    benchmark = NeuralNetworkBenchmark(
        game=args.game,
        device=args.device,
        warmup_iterations=args.warmup,
        verbose=not args.quiet
    )

    all_results = benchmark.run_comparison(
        batch_sizes=batch_sizes,
        iterations=iterations
    )

    # Print results
    benchmark.print_comparison_table(all_results)
    benchmark.print_summary(all_results)

    # Export if requested
    if args.output:
        benchmark.export_results(all_results, args.output)

    print("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()
