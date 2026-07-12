#!/usr/bin/env python3
"""
Quick benchmark of all FastMCTSNet size configurations.
"""

import torch
import time
import sys
sys.path.insert(0, '.')

from src.neural.model import create_model_for_game, create_fast_model_for_game

def benchmark_model(model, batch_size, iterations, device):
    """Benchmark a single model configuration."""
    model = model.to(device).eval()

    # Create input
    input_tensor = torch.randn(batch_size, 36, 15, 15, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    mean_time = sum(times) / len(times)
    throughput = (batch_size * iterations) / sum(times)

    return mean_time, throughput

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    iterations = 100

    print(f"FastMCTSNet Size Benchmark")
    print(f"=" * 80)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {iterations}")
    print(f"=" * 80)

    # Benchmark baseline
    print(f"\nBenchmarking AlphaZeroNet (baseline)...")
    model_az = create_model_for_game('gomoku', use_fast_model=False)
    time_az, throughput_az = benchmark_model(model_az, batch_size, iterations, device)
    params_az = model_az.get_num_parameters()

    print(f"  Parameters: {params_az:,} ({params_az/1e6:.2f}M)")
    print(f"  Time: {time_az*1000:.3f} ms")
    print(f"  Throughput: {throughput_az:.1f} samples/sec")

    # Benchmark all sizes
    results = {}
    for size in ['nano', 'small', 'medium', 'large']:
        print(f"\nBenchmarking FastMCTSNet-{size.upper()}...")

        # Training mode
        model = create_fast_model_for_game('gomoku', size=size)
        params = model.get_num_parameters()
        time_train, throughput_train = benchmark_model(model, batch_size, iterations, device)

        # Deploy mode
        model.switch_to_deploy()
        time_deploy, throughput_deploy = benchmark_model(model, batch_size, iterations, device)

        speedup_train = throughput_train / throughput_az
        speedup_deploy = throughput_deploy / throughput_az

        results[size] = {
            'params': params,
            'time_train': time_train,
            'throughput_train': throughput_train,
            'speedup_train': speedup_train,
            'time_deploy': time_deploy,
            'throughput_deploy': throughput_deploy,
            'speedup_deploy': speedup_deploy,
        }

        print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
        print(f"  Training mode:")
        print(f"    Time: {time_train*1000:.3f} ms")
        print(f"    Throughput: {throughput_train:.1f} samples/sec ({speedup_train:.2f}× baseline)")
        print(f"  Deploy mode:")
        print(f"    Time: {time_deploy*1000:.3f} ms")
        print(f"    Throughput: {throughput_deploy:.1f} samples/sec ({speedup_deploy:.2f}× baseline)")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"={'=' * 80}")
    print(f"\n{'Size':<10} {'Params':<12} {'Deploy Time':<15} {'Throughput':<15} {'Speedup':<10}")
    print(f"{'-' * 80}")
    print(f"{'AlphaZero':<10} {params_az/1e6:<12.2f} {time_az*1000:<15.3f} {throughput_az:<15.1f} {'1.00×':<10}")

    for size in ['nano', 'small', 'medium', 'large']:
        r = results[size]
        print(f"{size.upper():<10} {r['params']/1e6:<12.2f} {r['time_deploy']*1000:<15.3f} "
              f"{r['throughput_deploy']:<15.1f} {r['speedup_deploy']:<10.2f}×")

    print(f"=" * 80)

    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print(f"-" * 80)
    print(f"For 48h superhuman training on RTX 3060 Ti:")
    print(f"  SMALL: {results['small']['params']/1e6:.2f}M params, {results['small']['speedup_deploy']:.2f}× speedup - RECOMMENDED")
    print(f"For maximum strength (7+ days training):")
    print(f"  LARGE: {results['large']['params']/1e6:.2f}M params, {results['large']['speedup_deploy']:.2f}× speedup")
    print(f"=" * 80)

if __name__ == '__main__':
    main()
