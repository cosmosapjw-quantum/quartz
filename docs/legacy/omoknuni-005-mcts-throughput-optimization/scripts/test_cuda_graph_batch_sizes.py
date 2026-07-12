#!/usr/bin/env python3
"""
CUDA Graph Batch Size Comparison
=================================

Tests CUDA graph speedup across different batch sizes to validate that
small batches benefit more (launch-bound) than large batches (compute-bound).

Usage:
    python scripts/test_cuda_graph_batch_sizes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import time
from src.neural.model import create_resnet_eca_model
from src.core.dlpack_inference_bridge import DLPackInferenceBridge

print("=" * 80)
print("CUDA GRAPH BATCH SIZE COMPARISON")
print("=" * 80)

# Create model
print("\n[1/2] Creating ResNet-ECA 128×12 model...")
model = create_resnet_eca_model('gomoku', size='128x12').cuda()
print(f"   ✅ Model created: {model.get_num_parameters():,} parameters")

# Test different batch sizes
batch_sizes = [8, 16, 32, 64]
num_iterations = 200

print(f"\n[2/2] Testing batch sizes: {batch_sizes}")
print(f"   Iterations per batch: {num_iterations}")

results = []

for batch_size in batch_sizes:
    print(f"\n{'='*80}")
    print(f"BATCH SIZE: {batch_size}")
    print(f"{'='*80}")

    # Create dummy features
    features = np.random.randn(36, 15, 15).astype(np.float32)
    features_batch = [features for _ in range(batch_size)]
    board_sizes = [15] * batch_size
    num_planes_list = [36] * batch_size

    # Bridge WITHOUT CUDA graphs
    bridge_no_graphs = DLPackInferenceBridge(
        model=model, device='cuda', use_cuda_graphs=False
    )

    # Bridge WITH CUDA graphs
    bridge_with_graphs = DLPackInferenceBridge(
        model=model, device='cuda', use_cuda_graphs=True,
        graph_batch_sizes=[8, 16, 32, 64]
    )

    # Warmup
    for _ in range(10):
        bridge_no_graphs.batch_inference_features(features_batch, board_sizes, num_planes_list)
        bridge_with_graphs.batch_inference_features(features_batch, board_sizes, num_planes_list)

    # Benchmark WITHOUT graphs
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = bridge_no_graphs.batch_inference_features(features_batch, board_sizes, num_planes_list)
    torch.cuda.synchronize()
    time_no_graphs = time.perf_counter() - start

    # Benchmark WITH graphs
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = bridge_with_graphs.batch_inference_features(features_batch, board_sizes, num_planes_list)
    torch.cuda.synchronize()
    time_with_graphs = time.perf_counter() - start

    speedup = time_no_graphs / time_with_graphs

    # Calculate throughput (positions/second)
    throughput_baseline = (batch_size * num_iterations) / time_no_graphs
    throughput_optimized = (batch_size * num_iterations) / time_with_graphs

    print(f"\n📊 Results:")
    print(f"   Without graphs: {time_no_graphs*1000:.2f} ms total ({throughput_baseline:.1f} pos/sec)")
    print(f"   With graphs:    {time_with_graphs*1000:.2f} ms total ({throughput_optimized:.1f} pos/sec)")
    print(f"   Speedup:        {speedup:.2f}×")

    results.append({
        'batch_size': batch_size,
        'time_no_graphs': time_no_graphs,
        'time_with_graphs': time_with_graphs,
        'speedup': speedup,
        'throughput_baseline': throughput_baseline,
        'throughput_optimized': throughput_optimized
    })

# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n{'Batch':>6} | {'No Graphs':>15} | {'With Graphs':>15} | {'Speedup':>8}")
print(f"{'Size':>6} | {'(pos/sec)':>15} | {'(pos/sec)':>15} | {' ':>8}")
print("-" * 80)

for r in results:
    print(
        f"{r['batch_size']:>6} | "
        f"{r['throughput_baseline']:>15,.1f} | "
        f"{r['throughput_optimized']:>15,.1f} | "
        f"{r['speedup']:>8.2f}×"
    )

# Validate that small batches benefit more
small_batch_speedup = results[0]['speedup']  # Batch 8
large_batch_speedup = results[-1]['speedup']  # Batch 64

print(f"\n📈 Analysis:")
print(f"   Smallest batch ({results[0]['batch_size']}): {small_batch_speedup:.2f}× speedup")
print(f"   Largest batch ({results[-1]['batch_size']}):  {large_batch_speedup:.2f}× speedup")

if small_batch_speedup > large_batch_speedup:
    print(f"   ✅ Small batches benefit more from CUDA graphs (launch overhead reduction)")
else:
    print(f"   ⚠️  Unexpected: large batches showing equal or better speedup")

# Find best configuration
best = max(results, key=lambda x: x['throughput_optimized'])
print(f"\n🏆 Best throughput: Batch {best['batch_size']} with {best['throughput_optimized']:.1f} pos/sec")

print("\n" + "=" * 80)
print("✅ TEST COMPLETE")
print("=" * 80)
