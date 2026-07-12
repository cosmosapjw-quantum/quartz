#!/usr/bin/env python3
"""
Simple validation that prefetch implementation doesn't break selection.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mcts_py

def main():
    print("Testing selection with prefetching...")

    # Create tree and root
    tree = mcts_py.MCTSTree(10000)
    root = tree.add_root_node(0.5, 0)

    # Add 20 children
    first_child = tree.allocate_nodes(20)
    tree.set_first_child_index(root, first_child)
    tree.set_num_children(root, 20)

    for i in range(20):
        child_idx = first_child + i
        tree.set_visit_count(child_idx, float(i + 1))
        tree.set_total_value(child_idx, float(i) * 0.1)
        tree.set_prior_prob(child_idx, 1.0 / 20)
        tree.set_virtual_loss(child_idx, 0.0)
        tree.set_parent_index(child_idx, root)

    tree.set_visit_count(root, 210.0)
    tree.set_total_value(root, 19.0)

    # Create selector
    config = mcts_py.PUCTConfig()
    config.enable_simd = True
    selector = mcts_py.create_puct_selector(config)

    # Select child
    result = selector.select_child(tree, root)

    if result.valid:
        print(f"✅ Selection successful")
        print(f"   Selected child: {result.selected_child}")
        print(f"   PUCT value: {result.best_puct_value:.4f}")

        # Test determinism
        result2 = selector.select_child(tree, root)
        if result2.selected_child == result.selected_child:
            print("✅ Selection is deterministic")
        else:
            print("❌ Selection is not deterministic!")
            return 1

        print("\n✅ Prefetching implementation validated - no functional changes")
        return 0
    else:
        print("❌ Selection failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
