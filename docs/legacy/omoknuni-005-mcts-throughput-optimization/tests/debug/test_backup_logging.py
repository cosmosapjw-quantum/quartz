"""
Backup Logging Test - Trace actual backup calls
"""

import pytest
import torch
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from core.mcts import AlphaZeroMCTS
from neural.inference_worker import GPUInferenceWorker
from neural.model import AlphaZeroNet
import alphazero_py


def create_test_model(model_path: str) -> None:
    model = AlphaZeroNet(
        input_channels=36,
        num_actions=225,
        num_blocks=4,
        hidden_channels=128,
        use_se=False
    )
    torch.save(model, model_path)


def test_backup_with_logging():
    """Test with logging to see what's actually happening."""
    print("\n" + "="*80)
    print("BACKUP LOGGING TEST")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    try:
        create_test_model(model_path)

        gpu_worker = GPUInferenceWorker(
            model_path=model_path,
            device='cuda',
            batch_size=32,
            timeout_ms=2.0,
            use_mixed_precision=True
        )
        gpu_worker.warmup(input_shape=(36, 15, 15))

        mcts = AlphaZeroMCTS(
            inference_fn=gpu_worker,
            use_async_inference=True,
            async_batch_size=32,
            async_timeout_ms=2.0,
            num_threads=1,  # Single thread for debugging
            c_puct=1.25
        )

        state = alphazero_py.GomokuState()

        print("\n[1/3] Before search:")
        print(f"  Root visits: {mcts.tree.get_visit_count(0)}")

        print("\n[2/3] Running 10 simulations (single thread)...")
        mcts.search(state, simulations=10)

        print("\n[3/3] After search:")
        root_visits = mcts.tree.get_visit_count(0)
        first_child = mcts.tree.get_first_child_index(0)

        print(f"  Root visits: {root_visits}")
        print(f"  First child index: {first_child}")

        if first_child != 4294967295:  # NULL_NODE_INDEX
            num_children = mcts.tree.get_num_children(0)
            print(f"  Num children: {num_children}")

            # Check first 5 children
            for i in range(min(5, num_children)):
                child_idx = first_child + i
                visits = mcts.tree.get_visit_count(child_idx)
                expanded = mcts.tree.get_flags(child_idx).is_expanded()
                print(f"  Child {i} (idx={child_idx}): visits={visits:.0f}, expanded={expanded}")

        # Get backup stats
        backup_stats = mcts.backup_manager.get_statistics()
        print(f"\n  Backup Stats:")
        print(f"    Total backups: {backup_stats.total_backups}")
        print(f"    Successful: {backup_stats.successful_backups}")
        print(f"    Nodes updated: {backup_stats.total_nodes_updated}")
        print(f"    Validation failures: {backup_stats.path_validation_failures}")

        print("\n" + "="*80)

        gpu_worker.stop_worker()

        # Assertions
        assert root_visits == 10, f"Root should have 10 visits, got {root_visits}"

        # At least one child should have visits
        has_child_visits = False
        if first_child != 4294967295:
            for i in range(min(10, num_children)):
                if mcts.tree.get_visit_count(first_child + i) > 0:
                    has_child_visits = True
                    break

        assert has_child_visits, "At least one child should have visits"

    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


if __name__ == "__main__":
    test_backup_with_logging()
