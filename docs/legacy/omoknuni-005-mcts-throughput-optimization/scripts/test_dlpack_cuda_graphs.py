#!/usr/bin/env python3
"""
DLPack CUDA Graph Integration Test
===================================

Tests CUDA graph integration into DLPackInferenceBridge without full MCTS pipeline.
This is faster than full integration test and validates the core functionality.

Usage:
    python scripts/test_dlpack_cuda_graphs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from src.neural.model import create_resnet_eca_model
from src.core.dlpack_inference_bridge import DLPackInferenceBridge

print("=" * 80)
print("DLPACK CUDA GRAPH INTEGRATION TEST")
print("=" * 80)

# Test 1: Create model
print("\n[1/5] Creating ResNet-ECA 128×12 model...")
model = create_resnet_eca_model('gomoku', size='128x12')
model = model.cuda()
print(f"   ✅ Model created: {model.get_num_parameters():,} parameters")

# Test 2: Create bridge WITH CUDA graphs
print("\n[2/5] Creating DLPackInferenceBridge with CUDA graphs...")
bridge_with_graphs = DLPackInferenceBridge(
    model=model,
    device='cuda',
    use_cuda_graphs=True,
    graph_batch_sizes=[8, 16, 32, 64]
)
print(f"   ✅ Bridge created with CUDA graphs enabled")

# Test 3: Run inference to trigger graph capture
print("\n[3/5] Running inference to trigger CUDA graph capture...")

# Create dummy features (Gomoku 15×15 with 36 planes)
batch_size = 64
features = np.random.randn(36, 15, 15).astype(np.float32)
features_batch = [features for _ in range(batch_size)]
board_sizes = [15] * batch_size
num_planes_list = [36] * batch_size

results = bridge_with_graphs.batch_inference_features(
    features_batch, board_sizes, num_planes_list
)

print(f"   ✅ Inference successful: {len(results)} results returned")
print(f"      Policy length: {len(results[0][0])}")
print(f"      Value: {results[0][1]:.4f}")

# Verify graph manager was initialized
if bridge_with_graphs.graph_manager is not None:
    stats = bridge_with_graphs.graph_manager.get_stats()
    print(f"   ✅ Graph manager initialized:")
    print(f"      Captured batches: {stats['captured_batch_sizes']}")
    print(f"      Graph hits: {stats['graph_hits']}")
else:
    print(f"   ❌ Graph manager NOT initialized!")
    sys.exit(1)

# Test 4: Run multiple inferences to verify graph reuse
print("\n[4/5] Testing graph reuse (100 iterations)...")

for i in range(100):
    results = bridge_with_graphs.batch_inference_features(
        features_batch, board_sizes, num_planes_list
    )

stats_after = bridge_with_graphs.graph_manager.get_stats()
print(f"   ✅ 100 inferences completed")
print(f"      Total graph hits: {stats_after['graph_hits']}")
print(f"      Hit rate: {stats_after['hit_rate_percent']:.1f}%")

# Test 5: Compare with baseline (no CUDA graphs)
print("\n[5/5] Comparing performance with/without CUDA graphs...")

# Create bridge WITHOUT CUDA graphs
bridge_no_graphs = DLPackInferenceBridge(
    model=model,
    device='cuda',
    use_cuda_graphs=False
)

import time

# Warmup
for _ in range(10):
    bridge_no_graphs.batch_inference_features(features_batch, board_sizes, num_planes_list)

# Benchmark WITHOUT graphs
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    results_baseline = bridge_no_graphs.batch_inference_features(
        features_batch, board_sizes, num_planes_list
    )
torch.cuda.synchronize()
time_no_graphs = time.perf_counter() - start

# Benchmark WITH graphs
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    results_optimized = bridge_with_graphs.batch_inference_features(
        features_batch, board_sizes, num_planes_list
    )
torch.cuda.synchronize()
time_with_graphs = time.perf_counter() - start

speedup = time_no_graphs / time_with_graphs

print(f"\n📊 Performance Results (batch size {batch_size}, 100 iterations):")
print(f"   Without graphs: {time_no_graphs*1000:.2f} ms")
print(f"   With graphs:    {time_with_graphs*1000:.2f} ms")
print(f"   Speedup:        {speedup:.2f}×")

if speedup >= 1.05:
    print(f"   ✅ CUDA graphs provide measurable speedup ({speedup:.2f}×)")
elif speedup >= 1.0:
    print(f"   ⚠️  Minimal speedup ({speedup:.2f}×) - expected for large batches (compute-bound)")
else:
    print(f"   ❌ Performance regression ({speedup:.2f}×) - investigate!")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED")
print("=" * 80)
