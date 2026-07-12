#!/usr/bin/env python3
"""Create a test model checkpoint for validation purposes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.neural.model import AlphaZeroNet

def create_test_model(output_path: str):
    """Create and save a randomly initialized model."""

    print("Creating test model...")
    print("  Architecture: AlphaZeroNet")
    print("  Input channels: 36 (Gomoku)")
    print("  Blocks: 20")
    print("  Filters: 256")
    print("  Action size: 225 (15×15)")

    model = AlphaZeroNet(
        input_channels=36,
        num_actions=225,
        num_blocks=20,
        hidden_channels=256
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Save checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_channels': 36,
            'num_blocks': 20,
            'hidden_channels': 256,
            'num_actions': 225,
        },
        'note': 'Test model for FP16 validation - randomly initialized'
    }, output_path)

    print(f"\n✅ Model saved to: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create test model checkpoint")
    parser.add_argument("--output", type=str, default="models/test_gomoku.pth",
                       help="Output path for model checkpoint")
    args = parser.parse_args()

    create_test_model(args.output)
