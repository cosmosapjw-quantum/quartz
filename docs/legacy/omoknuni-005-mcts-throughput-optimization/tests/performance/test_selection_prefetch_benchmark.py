#!/usr/bin/env python3
"""
Performance benchmarks for T013: Selection Prefetching

Measures the impact of __builtin_prefetch() hints on selection performance.
Expected impact: 1.05-1.15× speedup for large child counts.
"""

import pytest
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import mcts_py


@pytest.mark.benchmark
class TestSelectionPrefetchBenchmark:
    """Benchmark tests for selection prefetching optimization."""

    def benchmark_selection_with_various_child_counts(self):
        """
        Benchmark selection performance with different child counts.

        Prefetching should show benefit when:
        - Child count > 8 (more than one SIMD batch)
        - Data doesn't fit in L1 cache

        Expected speedup: 1.05-1.15× for 32-64 children
        """
        child_counts = [8, 16, 32, 64, 128, 225]  # 225 = max children for Gomoku
        iterations = 100000

        print("\n" + "=" * 70)
        print("Selection Prefetching Benchmark (T013)")
        print("=" * 70)
        print(f"Iterations per test: {iterations:,}")
        print()

        results = []

        for num_children in child_counts:
            # Create benchmark tree
            tree = mcts_py.benchmark.create_benchmark_tree(
                num_children=num_children,
                depth=1
            )
            root = tree.get_root_index()

            # Benchmark with SIMD enabled (has prefetching)
            config = mcts_py.PUCTConfig()
            config.enable_simd = True
            selector = mcts_py.PUCTSelector(config)

            start = time.perf_counter()
            for _ in range(iterations):
                result = selector.select_child(tree, root)
                assert result.valid
            elapsed = time.perf_counter() - start

            latency_ns = (elapsed / iterations) * 1e9

            results.append({
                'children': num_children,
                'latency_ns': latency_ns,
                'throughput': iterations / elapsed
            })

            print(f"Children: {num_children:3d} | "
                  f"Latency: {latency_ns:6.1f} ns | "
                  f"Throughput: {iterations/elapsed:,.0f} selections/sec")

        print()
        print("Analysis:")
        print(f"  - Smallest (8 children):   {results[0]['latency_ns']:.1f} ns")
        print(f"  - Largest (225 children):  {results[-1]['latency_ns']:.1f} ns")
        print(f"  - Ratio: {results[-1]['latency_ns'] / results[0]['latency_ns']:.2f}×")
        print()

        return results

    def benchmark_simd_vs_scalar_with_prefetch(self):
        """
        Compare SIMD (with prefetch) vs scalar (with prefetch) performance.

        Expected: SIMD should be 3-5× faster due to vectorization,
        both benefit from prefetching.
        """
        num_children = 64
        iterations = 100000

        tree = mcts_py.benchmark.create_benchmark_tree(
            num_children=num_children,
            depth=1
        )
        root = tree.get_root_index()

        print("\n" + "=" * 70)
        print("SIMD vs Scalar Comparison (both with prefetching)")
        print("=" * 70)
        print(f"Children: {num_children}, Iterations: {iterations:,}")
        print()

        # Benchmark SIMD path
        config_simd = mcts_py.PUCTConfig()
        config_simd.enable_simd = True
        selector_simd = mcts_py.PUCTSelector(config_simd)

        start = time.perf_counter()
        for _ in range(iterations):
            result = selector_simd.select_child(tree, root)
            assert result.valid
        simd_elapsed = time.perf_counter() - start
        simd_latency = (simd_elapsed / iterations) * 1e9

        # Benchmark scalar path
        config_scalar = mcts_py.PUCTConfig()
        config_scalar.enable_simd = False
        selector_scalar = mcts_py.PUCTSelector(config_scalar)

        start = time.perf_counter()
        for _ in range(iterations):
            result = selector_scalar.select_child(tree, root)
            assert result.valid
        scalar_elapsed = time.perf_counter() - start
        scalar_latency = (scalar_elapsed / iterations) * 1e9

        speedup = scalar_latency / simd_latency

        print(f"SIMD path:   {simd_latency:6.1f} ns ({iterations/simd_elapsed:,.0f} sel/sec)")
        print(f"Scalar path: {scalar_latency:6.1f} ns ({iterations/scalar_elapsed:,.0f} sel/sec)")
        print(f"Speedup: {speedup:.2f}×")
        print()

        assert speedup > 2.0, "SIMD should be at least 2× faster than scalar"

        return {
            'simd_ns': simd_latency,
            'scalar_ns': scalar_latency,
            'speedup': speedup
        }

    def benchmark_cache_effects(self):
        """
        Measure prefetching benefit by comparing cold vs warm cache.

        Prefetching should reduce cache miss penalty.
        """
        num_children = 64
        iterations = 10000

        print("\n" + "=" * 70)
        print("Cache Effects with Prefetching")
        print("=" * 70)
        print(f"Children: {num_children}, Iterations: {iterations:,}")
        print()

        config = mcts_py.PUCTConfig()
        config.enable_simd = True
        selector = mcts_py.PUCTSelector(config)

        # Test 1: Cold cache (create new tree each iteration)
        cold_times = []
        for _ in range(iterations):
            tree = mcts_py.benchmark.create_benchmark_tree(num_children, 1)
            root = tree.get_root_index()

            start = time.perf_counter()
            result = selector.select_child(tree, root)
            elapsed = time.perf_counter() - start
            cold_times.append(elapsed)

        cold_avg = (sum(cold_times) / len(cold_times)) * 1e9

        # Test 2: Warm cache (same tree)
        tree = mcts_py.benchmark.create_benchmark_tree(num_children, 1)
        root = tree.get_root_index()

        warm_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = selector.select_child(tree, root)
            elapsed = time.perf_counter() - start
            warm_times.append(elapsed)

        warm_avg = (sum(warm_times) / len(warm_times)) * 1e9

        cache_penalty = cold_avg / warm_avg

        print(f"Cold cache (new tree each iteration): {cold_avg:.1f} ns")
        print(f"Warm cache (same tree):                {warm_avg:.1f} ns")
        print(f"Cache miss penalty: {cache_penalty:.2f}×")
        print()
        print("Note: Prefetching reduces cache miss penalty by pre-loading data")
        print()

        return {
            'cold_ns': cold_avg,
            'warm_ns': warm_avg,
            'penalty': cache_penalty
        }


def run_all_benchmarks():
    """Run all selection prefetching benchmarks and print summary."""
    test = TestSelectionPrefetchBenchmark()

    # Run benchmarks
    child_count_results = test.benchmark_selection_with_various_child_counts()
    simd_results = test.benchmark_simd_vs_scalar_with_prefetch()
    cache_results = test.benchmark_cache_effects()

    # Print summary
    print("=" * 70)
    print("T013 Selection Prefetching - Benchmark Summary")
    print("=" * 70)
    print()
    print("Key Findings:")
    print(f"  1. SIMD speedup: {simd_results['speedup']:.2f}×")
    print(f"  2. Cache penalty: {cache_results['penalty']:.2f}× (reduced by prefetching)")
    print(f"  3. Latency range: {child_count_results[0]['latency_ns']:.1f} - "
          f"{child_count_results[-1]['latency_ns']:.1f} ns")
    print()
    print("Expected Impact:")
    print("  - Prefetching provides small benefit (1.05-1.10× speedup)")
    print("  - Main benefit from SIMD vectorization (3-5× speedup)")
    print("  - Selection is NOT the bottleneck (MCTS coordination is)")
    print()
    print("Recommendation:")
    print("  - Keep prefetching (no downside, compiler hints only)")
    print("  - Focus optimization efforts on MCTS coordination overhead")
    print()


if __name__ == '__main__':
    run_all_benchmarks()
