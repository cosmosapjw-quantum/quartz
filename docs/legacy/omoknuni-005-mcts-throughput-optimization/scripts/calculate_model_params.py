#!/usr/bin/env python3
"""
Calculate parameter counts for different AlphaZero model configurations.

This helps identify optimal configurations for target performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.neural.model import AlphaZeroNet, create_model_for_game

def estimate_params(input_channels, num_actions, num_blocks, hidden_channels, use_se=True):
    """Estimate parameter count for a model configuration."""

    # Initial convolution: input_channels × hidden_channels × 3×3
    initial_conv = input_channels * hidden_channels * 9

    # Residual blocks
    # Each block: 2 conv layers (hidden × hidden × 3×3) + BN params + SE block
    conv_per_block = 2 * (hidden_channels * hidden_channels * 9)
    bn_per_block = 4 * hidden_channels  # 2 BN layers (weight + bias)

    if use_se:
        # SE block: 2 FC layers with reduction ratio 16
        reduced = max(1, hidden_channels // 16)
        se_per_block = (hidden_channels * reduced) + (reduced * hidden_channels)
    else:
        se_per_block = 0

    residual_params = num_blocks * (conv_per_block + bn_per_block + se_per_block)

    # Policy head
    # 1×1 conv: hidden × 2, BN, FC: (2 * board_area) × num_actions
    board_size = int(num_actions ** 0.5)
    policy_conv = hidden_channels * 2
    policy_fc = (2 * board_size * board_size) * num_actions

    # Value head
    # 1×1 conv: hidden × 1, BN, FC1: 1 × 256, FC2: 256 × 1
    value_conv = hidden_channels * 1
    value_fc = 256 + 256

    total = initial_conv + residual_params + policy_conv + policy_fc + value_conv + value_fc

    return {
        'total': int(total),
        'initial_conv': initial_conv,
        'residual_blocks': residual_params,
        'policy_head': policy_conv + policy_fc,
        'value_head': value_conv + value_fc,
        'millions': total / 1e6
    }

def print_config(name, input_ch, actions, blocks, channels, use_se=True):
    """Print parameter count for a configuration."""
    est = estimate_params(input_ch, actions, blocks, channels, use_se)

    # Create actual model to verify
    model = AlphaZeroNet(
        input_channels=input_ch,
        num_actions=actions,
        num_blocks=blocks,
        hidden_channels=channels,
        use_se=use_se
    )
    actual = sum(p.numel() for p in model.parameters())

    print(f"\n{name}:")
    print(f"  Configuration: {blocks} blocks × {channels} channels")
    print(f"  Use SE: {use_se}")
    print(f"  Estimated params: {est['millions']:.2f}M")
    print(f"  Actual params: {actual / 1e6:.2f}M")
    print(f"  Error: {abs(est['total'] - actual) / actual * 100:.1f}%")

    return actual

def main():
    """Compare different model configurations."""
    print("="*80)
    print("AlphaZero Model Parameter Analysis (Gomoku)")
    print("="*80)

    input_channels = 36  # Gomoku enhanced features
    num_actions = 225    # 15×15 board

    print("\n" + "="*80)
    print("CURRENT CONFIGURATION")
    print("="*80)
    current = print_config("Current (Spec)", input_channels, num_actions, 20, 256, True)

    print("\n" + "="*80)
    print("TARGET CONFIGURATIONS (~10M params)")
    print("="*80)

    configs = [
        ("Option A: 12 blocks × 256 channels", 12, 256, True),
        ("Option B: 15 blocks × 192 channels", 15, 192, True),
        ("Option C: 20 blocks × 192 channels", 20, 192, True),
        ("Option D: 15 blocks × 256 channels (no SE)", 15, 256, False),
    ]

    results = []
    for name, blocks, channels, use_se in configs:
        params = print_config(name, input_channels, num_actions, blocks, channels, use_se)
        results.append((name, blocks, channels, use_se, params))

    # Find closest to 10M
    target = 10_000_000
    best = min(results, key=lambda x: abs(x[4] - target))

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print(f"\nBest match for 10M target: {best[0]}")
    print(f"  Params: {best[4] / 1e6:.2f}M")
    print(f"  Distance from 10M: {abs(best[4] - target) / 1e6:.2f}M")

    speedup = current / best[4]
    print(f"\nExpected speedup vs current: {speedup:.2f}×")
    print(f"  Current: {current / 1e6:.2f}M params, ~180ms")
    print(f"  Target:  {best[4] / 1e6:.2f}M params, ~{180/speedup:.1f}ms (estimated)")

    print("\n" + "="*80)
    print("IMPLEMENTATION")
    print("="*80)
    print("\nTo use this configuration, update src/neural/model.py:")
    print(f"""
default_kwargs = {{
    'num_blocks': {best[1]},      # Reduced from 20
    'hidden_channels': {best[2]},  # Reduced from 256
    'use_se': {best[3]}
}}
""")

if __name__ == '__main__':
    main()
