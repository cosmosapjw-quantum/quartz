"""
Tree Structure Debugging - Critical Analysis

Diagnose why MCTS tree stays flat (1.00 inf/sim) even with 10,000 simulations.

Questions:
1. Are root children actually marked as expanded?
2. Does selection descend past level 1?
3. Are there any level-2 nodes in the tree?
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


def test_tree_structure_analysis():
    """Analyze actual tree structure after search."""
    print("\n" + "="*80)
    print("TREE STRUCTURE DEBUG ANALYSIS")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
        model_path = f.name

    try:
        create_test_model(model_path)

        gpu_worker = GPUInferenceWorker(
            model_path=model_path,
            device='cuda',
            batch_size=128,
            timeout_ms=2.0,
            use_mixed_precision=True
        )
        gpu_worker.warmup(input_shape=(36, 15, 15))

        mcts = AlphaZeroMCTS(
            inference_fn=gpu_worker,
            use_async_inference=True,
            async_batch_size=128,
            async_timeout_ms=2.0,
            num_threads=8,
            c_puct=1.25
        )

        state = alphazero_py.GomokuState()

        print("\n[1/4] Running 1000 simulations...")
        mcts.search(state, simulations=1000)

        print("\n[2/4] Analyzing tree structure...")

        # Root analysis
        root_idx = mcts.root_index
        root_visits = mcts.tree.get_visit_count(root_idx)
        root_children_count = mcts.tree.get_num_children(root_idx)
        first_child = mcts.tree.get_first_child_index(root_idx)
        root_flags = mcts.tree.get_flags(root_idx)

        print(f"\n  Root Node:")
        print(f"    Index: {root_idx}")
        print(f"    Visits: {root_visits:.0f}")
        print(f"    Num children: {root_children_count}")
        print(f"    First child index: {first_child}")
        print(f"    Is expanded: {root_flags.is_expanded()}")

        # Analyze root children
        if root_children_count > 0 and first_child != 4294967295:  # NULL_NODE_INDEX
            print(f"\n  Root Children Analysis (first 10):")
            print(f"    {'Index':>8s} {'Visits':>8s} {'Expanded':>10s} {'Children':>10s}")
            print(f"    {'-'*8:>8s} {'-'*8:>8s} {'-'*10:>10s} {'-'*10:>10s}")

            expanded_count = 0
            total_level2_nodes = 0

            for i in range(min(10, root_children_count)):
                child_idx = first_child + i
                child_visits = mcts.tree.get_visit_count(child_idx)
                child_flags = mcts.tree.get_flags(child_idx)
                is_expanded = child_flags.is_expanded()
                num_children = mcts.tree.get_num_children(child_idx)

                print(f"    {child_idx:8d} {child_visits:8.0f} {str(is_expanded):>10s} {num_children:10d}")

                if is_expanded:
                    expanded_count += 1
                    total_level2_nodes += num_children

            # Check all root children
            total_expanded = 0
            total_level2 = 0
            for i in range(root_children_count):
                child_idx = first_child + i
                child_flags = mcts.tree.get_flags(child_idx)
                if child_flags.is_expanded():
                    total_expanded += 1
                    total_level2 += mcts.tree.get_num_children(child_idx)

            print(f"\n  Level 1 Summary (all {root_children_count} children):")
            print(f"    Total expanded: {total_expanded}")
            print(f"    Total level-2 nodes: {total_level2}")

        # Tree-wide analysis
        total_nodes = mcts.tree.get_node_count()
        print(f"\n[3/4] Tree-wide Analysis:")
        print(f"  Total nodes allocated: {total_nodes}")
        print(f"  Expected structure:")
        print(f"    - 1 root")
        print(f"    - {root_children_count} level-1 nodes")
        print(f"    - Should have level-2 nodes if depth > 1")

        # Calculate expected vs actual
        expected_min_level2 = max(0, 1000 - 225)  # Sims beyond first 225
        print(f"\n  Expected minimum level-2 nodes: {expected_min_level2}")
        print(f"  Actual level-2 nodes: {total_level2}")

        # Diagnosis
        print(f"\n[4/4] Diagnosis:")

        if total_level2 == 0:
            print(f"  ❌ CRITICAL BUG: No level-2 nodes!")
            print(f"     Tree is completely flat (depth=1)")
            print(f"     Possible causes:")
            print(f"       1. Expansion not marking nodes as expanded")
            print(f"       2. Selection not descending to expanded nodes")
            print(f"       3. Race condition causing expansion failures")

        elif total_level2 < expected_min_level2 * 0.5:
            print(f"  ⚠️  WARNING: Very few level-2 nodes")
            print(f"     Tree expansion is working but inefficient")

        else:
            print(f"  ✅ Tree structure looks normal")
            print(f"     Depth > 1 confirmed")

        print("\n" + "="*80)

        gpu_worker.stop_worker()

        return {
            'total_level2_nodes': total_level2,
            'expanded_count': total_expanded,
            'total_nodes': total_nodes
        }

    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


if __name__ == "__main__":
    test_tree_structure_analysis()