#!/usr/bin/env python3
"""
Diagnose GPU inference bottleneck.

This script performs detailed profiling of the GPU inference pipeline to identify
why performance is catastrophically slow (41ms/batch vs expected 6-8ms).

Checks:
1. FP16 is actually being used (not just enabled)
2. Model size and architecture
3. Memory transfer overhead vs compute time
4. Batch size scaling characteristics
5. CUDA stream synchronization issues
"""

import torch
import torch.cuda
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.neural.model import create_model_for_game
from src.core.dlpack_inference_bridge import DLPackInferenceBridge

def print_section(title):
    """Print section header."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")

def diagnose_model_architecture():
    """Analyze model architecture and parameter count."""
    print_section("1. Model Architecture Analysis")

    model = create_model_for_game('gomoku')

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: AlphaZeroNet (Gomoku)")
    print(f"  Residual blocks: {model.num_blocks}")
    print(f"  Channels: {model.hidden_channels}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (FP32): {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  Model size (FP16): {total_params * 2 / 1024 / 1024:.2f} MB")

    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Check actual model dtype
    first_param_dtype = next(model.parameters()).dtype
    print(f"\n  Model parameter dtype: {first_param_dtype}")
    print(f"  Model device: {next(model.parameters()).device}")

    return model, device

def diagnose_fp16_usage(model, device):
    """Verify FP16 is actually being used during inference."""
    print_section("2. FP16 Mixed Precision Verification")

    batch_size = 64
    input_shape = (batch_size, 36, 15, 15)  # Gomoku features

    # Create test input
    test_input = torch.randn(input_shape, device=device, dtype=torch.float32)

    print(f"Test input shape: {test_input.shape}")
    print(f"Test input dtype: {test_input.dtype}")
    print(f"Test input device: {test_input.device}")

    # Test without autocast
    print("\n--- Without autocast (FP32) ---")
    with torch.no_grad():
        start = time.perf_counter()
        policy_logits, value = model(test_input)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    print(f"  Policy dtype: {policy_logits.dtype}")
    print(f"  Value dtype: {value.dtype}")
    print(f"  Inference time: {elapsed*1000:.2f} ms")

    # Test with autocast
    print("\n--- With autocast (FP16) ---")
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            start = time.perf_counter()
            policy_logits_fp16, value_fp16 = model(test_input)
            torch.cuda.synchronize()
            elapsed_fp16 = time.perf_counter() - start

    print(f"  Policy dtype: {policy_logits_fp16.dtype}")
    print(f"  Value dtype: {value_fp16.dtype}")
    print(f"  Inference time: {elapsed_fp16*1000:.2f} ms")
    print(f"  Speedup: {elapsed/elapsed_fp16:.2f}x")

    # Check if FP16 is actually faster
    if elapsed_fp16 < elapsed * 0.9:
        print(f"\n  ✅ FP16 is working ({elapsed/elapsed_fp16:.2f}x faster)")
    else:
        print(f"\n  ⚠️  FP16 not providing speedup (only {elapsed/elapsed_fp16:.2f}x)")
        print(f"  This suggests tensor cores may not be utilized")

def diagnose_memory_transfers(model, device):
    """Profile memory transfer overhead."""
    print_section("3. Memory Transfer Overhead Analysis")

    batch_size = 64
    input_shape = (batch_size, 36, 15, 15)

    # Create CPU input
    cpu_input = np.random.randn(*input_shape).astype(np.float32)

    # Measure H2D transfer
    print("Host-to-Device (H2D) transfer:")
    start = time.perf_counter()
    gpu_input = torch.from_numpy(cpu_input).to(device)
    torch.cuda.synchronize()
    h2d_time = time.perf_counter() - start
    print(f"  Time: {h2d_time*1000:.2f} ms")
    print(f"  Bandwidth: {cpu_input.nbytes / h2d_time / 1e9:.2f} GB/s")

    # Measure inference
    print("\nInference (with autocast):")
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            start = time.perf_counter()
            policy_logits, value = model(gpu_input)
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - start
    print(f"  Time: {inference_time*1000:.2f} ms")

    # Measure D2H transfer
    print("\nDevice-to-Host (D2H) transfer:")
    start = time.perf_counter()
    policy_cpu = policy_logits.cpu()
    value_cpu = value.cpu()
    torch.cuda.synchronize()
    d2h_time = time.perf_counter() - start
    output_bytes = policy_logits.numel() * 4 + value.numel() * 4
    print(f"  Time: {d2h_time*1000:.2f} ms")
    print(f"  Bandwidth: {output_bytes / d2h_time / 1e9:.2f} GB/s")

    # Total pipeline time
    total_time = h2d_time + inference_time + d2h_time
    print(f"\nTotal pipeline time: {total_time*1000:.2f} ms")
    print(f"  H2D: {h2d_time/total_time*100:.1f}%")
    print(f"  Inference: {inference_time/total_time*100:.1f}%")
    print(f"  D2H: {d2h_time/total_time*100:.1f}%")

def diagnose_dlpack_bridge():
    """Profile DLPackInferenceBridge end-to-end."""
    print_section("4. DLPackInferenceBridge End-to-End Profiling")

    device = torch.device('cuda')
    model = create_model_for_game('gomoku').to(device)
    model.eval()

    # Create bridge with FP16
    bridge = DLPackInferenceBridge(
        model=model,
        device=device,
        use_mixed_precision=True
    )

    print(f"Bridge configuration:")
    print(f"  Device: {bridge.device}")
    print(f"  Mixed precision: {bridge.use_mixed_precision}")

    # Profile with different batch sizes
    batch_sizes = [1, 8, 16, 32, 64]

    print(f"\nBatch size profiling:")
    for batch_size in batch_sizes:
        # Create test features (Gomoku: 36 channels, 15x15)
        features = np.random.randn(batch_size, 36, 15, 15).astype(np.float32)

        # Warmup
        for _ in range(3):
            bridge.batch_inference(features)

        # Profile
        times = []
        for _ in range(20):
            start = time.perf_counter()
            policy, value = bridge.batch_inference(features)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size / avg_time

        print(f"  Batch {batch_size:3d}: {avg_time*1000:6.2f} ± {std_time*1000:4.2f} ms, "
              f"{throughput:7.1f} states/sec")

def diagnose_cuda_streams():
    """Check CUDA stream behavior."""
    print_section("5. CUDA Stream Analysis")

    device = torch.device('cuda')

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")

    # Check if tensor cores are available
    major, minor = torch.cuda.get_device_capability(0)
    has_tensor_cores = major >= 7  # Volta (7.0) and newer
    print(f"\nTensor cores available: {has_tensor_cores}")

    if has_tensor_cores:
        print(f"  ✅ RTX 3060 Ti (Ampere, compute 8.6) has tensor cores")
        print(f"  FP16 should provide significant speedup")
    else:
        print(f"  ⚠️  No tensor cores detected (compute {major}.{minor})")

def main():
    """Run all diagnostics."""
    print("="*80)
    print("GPU Inference Bottleneck Diagnosis")
    print("="*80)
    print(f"\nTarget: Identify why GPU inference is slow (41ms/batch vs 6-8ms expected)")

    if not torch.cuda.is_available():
        print("\n❌ ERROR: CUDA not available")
        return 1

    # Run diagnostics
    model, device = diagnose_model_architecture()
    diagnose_fp16_usage(model, device)
    diagnose_memory_transfers(model, device)
    diagnose_dlpack_bridge()
    diagnose_cuda_streams()

    print("\n" + "="*80)
    print("Diagnosis Complete")
    print("="*80)

    return 0

if __name__ == '__main__':
    sys.exit(main())
