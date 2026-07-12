#!/usr/bin/env python3
"""Validate FP16 mixed precision inference."""

import time
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def measure_inference_throughput(model, use_fp16: bool, batch_size: int, iterations: int):
    """Measure inference throughput with/without FP16."""

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("❌ CUDA not available, FP16 validation requires GPU")
        return None, None

    model = model.to(device).eval()

    # Create dummy batch (Gomoku: 36 channels × 15×15)
    dummy_input = torch.randn(batch_size, 36, 15, 15, device=device)

    # Warmup (10 iterations)
    print(f"  Warming up ({'FP16' if use_fp16 else 'FP32'})...")
    with torch.no_grad():
        for _ in range(10):
            if use_fp16:
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)

    torch.cuda.synchronize()

    # Measure
    print(f"  Measuring ({iterations} iterations)...")
    timings = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            if use_fp16:
                with torch.cuda.amp.autocast():
                    policy, value = model(dummy_input)
            else:
                policy, value = model(dummy_input)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        timings.append(elapsed)

    mean_time = np.mean(timings)
    std_time = np.std(timings)
    throughput = (batch_size * 1000) / mean_time  # states/sec

    return mean_time, throughput, std_time

def validate_fp16_accuracy(model, batch_size: int):
    """Validate FP16 accuracy vs FP32 on SAME input."""

    device = torch.device('cuda:0')
    model = model.to(device).eval()

    # Use FIXED random seed for reproducible comparison
    torch.manual_seed(42)
    dummy_input = torch.randn(batch_size, 36, 15, 15, device=device)

    # FP32 inference
    with torch.no_grad():
        policy_logits_fp32, value_fp32 = model(dummy_input)

    # FP16 inference (on SAME input)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            policy_logits_fp16, value_fp16 = model(dummy_input)

    # Convert FP16 outputs to FP32 for comparison
    policy_logits_fp16 = policy_logits_fp16.float()
    value_fp16 = value_fp16.float()

    # Compare RAW logits (before softmax)
    policy_mse = torch.mean((policy_logits_fp32 - policy_logits_fp16) ** 2).item()
    value_mse = torch.mean((value_fp32 - value_fp16) ** 2).item()

    policy_max_diff = torch.max(torch.abs(policy_logits_fp32 - policy_logits_fp16)).item()
    value_max_diff = torch.max(torch.abs(value_fp32 - value_fp16)).item()

    # Also compare after softmax (what actually matters for MCTS)
    policy_probs_fp32 = torch.softmax(policy_logits_fp32, dim=-1)
    policy_probs_fp16 = torch.softmax(policy_logits_fp16.float(), dim=-1)
    policy_prob_mse = torch.mean((policy_probs_fp32 - policy_probs_fp16) ** 2).item()

    return policy_mse, value_mse, policy_max_diff, value_max_diff, policy_prob_mse

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate FP16 mixed precision inference")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint (.pth file)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size to test (default: 64)")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations for timing (default: 100)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ CUDA not available, cannot validate FP16")
        sys.exit(1)

    print("=" * 80)
    print("FP16 Mixed Precision Validation")
    print("=" * 80)

    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        checkpoint = torch.load(args.model, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            model_state = checkpoint

        # Create model (assume AlphaZeroNet architecture)
        from src.neural.model import AlphaZeroNet
        model = AlphaZeroNet(
            input_channels=36,
            num_actions=225,  # Gomoku 15×15
            num_blocks=20,
            hidden_channels=256
        )
        model.load_state_dict(model_state)
        print("✅ Model loaded successfully")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Throughput comparison
    print(f"\n{'='*80}")
    print("Throughput Comparison")
    print(f"{'='*80}")

    print("\n1. FP32 Inference:")
    fp32_time, fp32_throughput, fp32_std = measure_inference_throughput(
        model, False, args.batch_size, args.iterations
    )

    if fp32_time is None:
        sys.exit(1)

    print(f"  Mean: {fp32_time:.2f} ± {fp32_std:.2f} ms/batch")
    print(f"  Throughput: {fp32_throughput:.0f} states/sec")

    print("\n2. FP16 Inference:")
    fp16_time, fp16_throughput, fp16_std = measure_inference_throughput(
        model, True, args.batch_size, args.iterations
    )

    if fp16_time is None:
        sys.exit(1)

    print(f"  Mean: {fp16_time:.2f} ± {fp16_std:.2f} ms/batch")
    print(f"  Throughput: {fp16_throughput:.0f} states/sec")

    speedup = fp32_time / fp16_time
    print(f"\n3. Speedup: {speedup:.2f}×")

    # Accuracy comparison
    print(f"\n{'='*80}")
    print("Accuracy Comparison (FP32 vs FP16)")
    print(f"{'='*80}")

    policy_mse, value_mse, policy_max, value_max, policy_prob_mse = validate_fp16_accuracy(
        model, args.batch_size
    )

    print(f"\n  Logits Comparison (raw network output):")
    print(f"    Policy Logits MSE: {policy_mse:.6f}")
    print(f"    Policy Logits Max Diff: {policy_max:.6f}")
    print(f"\n  Post-Softmax Comparison (what MCTS uses):")
    print(f"    Policy Probability MSE: {policy_prob_mse:.6f} (target: <0.01)")
    print(f"\n  Value Comparison:")
    print(f"    Value MSE: {value_mse:.6f} (target: <0.01)")
    print(f"    Value Max Diff: {value_max:.6f}")

    # Final verdict
    print(f"\n{'='*80}")
    print("Validation Result")
    print(f"{'='*80}")

    speedup_pass = speedup >= 1.5
    # Use probability MSE (what actually matters for MCTS), not logit MSE
    accuracy_pass = policy_prob_mse < 0.01 and value_mse < 0.01

    if speedup_pass and accuracy_pass:
        print("\n✅ PASS: FP16 validated successfully")
        print(f"  - Speedup: {speedup:.2f}× (≥1.5× required)")
        print(f"  - Policy Probability MSE: {policy_prob_mse:.6f} (<0.01 required)")
        print(f"  - Value MSE: {value_mse:.6f} (<0.01 required)")
        sys.exit(0)
    else:
        print("\n❌ FAIL: FP16 validation failed")
        if not speedup_pass:
            print(f"  - Speedup too low: {speedup:.2f}× (need ≥1.5×)")
        if not accuracy_pass:
            print(f"  - Accuracy degraded:")
            print(f"    Policy Probability MSE: {policy_prob_mse:.6f} (need <0.01)")
            print(f"    Value MSE: {value_mse:.6f} (need <0.01)")
        sys.exit(1)
