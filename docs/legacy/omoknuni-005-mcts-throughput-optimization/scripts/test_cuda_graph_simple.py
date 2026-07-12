#!/usr/bin/env python3
"""
Simple CUDA Graph Integration Test
===================================

Minimal test to verify CUDA graph manager can be created and used.
This is a quick sanity check before running full MCTS integration tests.

Usage:
    python scripts/test_cuda_graph_simple.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.neural.model import create_resnet_eca_model
from src.core.cuda_graph_manager import CUDAGraphManager, create_graph_manager_for_model

print("=" * 80)
print("SIMPLE CUDA GRAPH TEST")
print("=" * 80)

# Test 1: Create model
print("\n[1/4] Creating ResNet-ECA 128×12 model...")
model = create_resnet_eca_model('gomoku', size='128x12')
model = model.cuda()
print(f"   ✅ Model created: {model.get_num_parameters():,} parameters")

# Test 2: Create CUDA graph manager
print("\n[2/4] Creating CUDA graph manager...")
graph_mgr = create_graph_manager_for_model(
    model, 'gomoku', batch_sizes=[8, 16, 32, 64]
)
print(f"   ✅ Graph manager created for batch sizes: {graph_mgr.batch_sizes}")

# Test 3: Warmup and capture
print("\n[3/4] Warming up and capturing CUDA graphs...")
graph_mgr.warmup_and_capture()
print(f"   ✅ Graphs captured: {list(graph_mgr.graphs.keys())}")

# Test 4: Run inference
print("\n[4/4] Testing graph replay...")
batch_size = 64
test_input = torch.randn(batch_size, 36, 15, 15, device='cuda', dtype=torch.float16)

policy, value = graph_mgr.infer(test_input, return_logits=False)

print(f"   ✅ Inference successful:")
print(f"      Policy shape: {policy.shape}")
print(f"      Value shape:  {value.shape}")
print(f"      Policy sum:   {policy.sum(dim=1).mean():.4f} (should be ~1.0)")

# Check stats
stats = graph_mgr.get_stats()
print(f"\n📊 Graph Manager Stats:")
print(f"   Graph hits:     {stats['graph_hits']}")
print(f"   Fallback count: {stats['fallback_count']}")
print(f"   Hit rate:       {stats['hit_rate_percent']:.1f}%")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED")
print("=" * 80)
