#!/usr/bin/env python3
"""
Verify that DLPack zero-copy is actually working.

Checks:
1. DLPack tensors created directly on GPU
2. H2D transfer time is zero
3. Features tensor on correct device
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.dlpack_inference_bridge import DLPackInferenceBridge
from src.neural.model import create_model_for_game
import alphazero_py
import mcts_py

def test_zero_copy():
    """Test that zero-copy is working."""
    print("="*70)
    print("DLPack Zero-Copy Verification")
    print("="*70)

    device = torch.device('cuda')
    model = create_model_for_game('gomoku').to(device)
    model.eval()

    # Create bridge
    bridge = DLPackInferenceBridge(
        model=model,
        device=device,
        use_mixed_precision=True
    )

    # Create test states
    states = [alphazero_py.GomokuState() for _ in range(64)]

    print(f"\n1. Testing DLPack tensor creation...")
    print(f"   Target device: {device}")

    # Test CPU tensor creation
    print(f"\n   A. CPU tensor creation (use_cuda=False):")
    try:
        capsule_cpu = mcts_py.create_batch_tensor_from_states(states, use_cuda=False)
        features_cpu = torch.from_dlpack(capsule_cpu)
        print(f"      Tensor device: {features_cpu.device}")
        print(f"      Is on CUDA: {features_cpu.is_cuda}")
        print(f"      Matches target: {features_cpu.device == device}")
        if features_cpu.device == device:
            print(f"      ✅ Zero-copy would work")
        else:
            print(f"      ❌ Would require H2D transfer")
    except Exception as e:
        print(f"      Error: {e}")

    # Test GPU tensor creation
    print(f"\n   B. GPU tensor creation (use_cuda=True):")
    try:
        capsule_gpu = mcts_py.create_batch_tensor_from_states(states, use_cuda=True)
        features_gpu = torch.from_dlpack(capsule_gpu)
        print(f"      Tensor device: {features_gpu.device}")
        print(f"      Is on CUDA: {features_gpu.is_cuda}")
        print(f"      Matches target: {features_gpu.device == device}")
        if features_gpu.device == device:
            print(f"      ✅ Zero-copy achieved!")
        else:
            print(f"      ❌ Device mismatch")
    except Exception as e:
        print(f"      Error: {e}")

    print(f"\n2. Testing inference bridge...")

    # Reset metrics
    bridge.reset_metrics()

    # Run inference
    print(f"   Running 20 batches of 64 states...")
    for i in range(20):
        bridge.batch_inference(states)

    # Get metrics
    metrics = bridge.get_metrics()

    print(f"\n3. Metrics Analysis:")
    print(f"   Total batches: {metrics['total_batches']}")
    print(f"   DLPack success rate: {metrics['dlpack_success_rate']:.1f}%")
    print(f"   Fallback uses: {metrics['fallback_uses']}")
    print(f"\n   Transfer times:")
    print(f"   - Avg H2D transfer: {metrics['avg_h2d_transfer_ms']:.4f} ms")
    print(f"   - Avg inference: {metrics['avg_inference_ms']:.2f} ms")
    print(f"   - Avg D2H transfer: {metrics['avg_d2h_transfer_ms']:.4f} ms")

    # Verify zero-copy
    print(f"\n4. Zero-Copy Verification:")

    if metrics['dlpack_success_rate'] < 100.0:
        print(f"   ❌ DLPack not always succeeding ({metrics['dlpack_success_rate']:.1f}%)")
        print(f"      Fallback uses: {metrics['fallback_uses']}")
        return False

    if metrics['avg_h2d_transfer_ms'] > 0.1:
        print(f"   ⚠️  H2D transfer time > 0.1ms ({metrics['avg_h2d_transfer_ms']:.4f} ms)")
        print(f"      Zero-copy may not be working properly")
        print(f"      Expected: ~0.0ms (no transfer needed)")
        return False

    if metrics['avg_h2d_transfer_ms'] < 0.01:
        print(f"   ✅ H2D transfer time ≈ 0ms ({metrics['avg_h2d_transfer_ms']:.4f} ms)")
        print(f"      Zero-copy is working!")
        return True
    else:
        print(f"   ✅ H2D transfer time very low ({metrics['avg_h2d_transfer_ms']:.4f} ms)")
        print(f"      Zero-copy likely working (small measurement overhead)")
        return True

def main():
    """Run zero-copy verification."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test zero-copy")
        return 1

    try:
        success = test_zero_copy()

        print(f"\n" + "="*70)
        print("RESULT")
        print("="*70)

        if success:
            print(f"\n✅ Zero-copy is working correctly!")
            print(f"   - DLPack tensors created on GPU")
            print(f"   - No CPU→GPU transfer needed")
            print(f"   - H2D time ≈ 0ms")
            return 0
        else:
            print(f"\n❌ Zero-copy is NOT working!")
            print(f"   - Check DLPack tensor creation")
            print(f"   - Check device matching")
            return 1

    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
