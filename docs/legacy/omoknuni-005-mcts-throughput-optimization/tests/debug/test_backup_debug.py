"""
Backup Mechanism Debug

Test if backup is actually updating visit counts on path nodes.
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import alphazero_py
import mcts_py


def test_backup_direct():
    """Test backup directly without async complexity."""
    print("\n" + "="*80)
    print("DIRECT BACKUP TEST")
    print("="*80)

    # Create tree
    tree = mcts_py.MCTSTree(10000)
    root = tree.add_root_node(0.5, 0)

    # Allocate children manually
    first_child = tree.allocate_nodes(3)
    tree.set_first_child_index(root, first_child)
    tree.set_num_children(root, 3)

    for i in range(3):
        child_idx = first_child + i
        tree.set_prior_prob(child_idx, 1.0/3.0)
        tree.set_parent_index(child_idx, root)
        tree.set_visit_count(child_idx, 0.0)
        tree.set_total_value(child_idx, 0.0)

        flags = mcts_py.NodeFlags()
        flags.set_expanded(True)
        tree.set_flags(child_idx, flags)

    root_flags = mcts_py.NodeFlags()
    root_flags.set_expanded(True)
    tree.set_flags(root, root_flags)

    print(f"\n[1/3] Initial state:")
    print(f"  Root visits: {tree.get_visit_count(root):.0f}")
    print(f"  Child 0 visits: {tree.get_visit_count(first_child):.0f}")
    print(f"  Child 1 visits: {tree.get_visit_count(first_child+1):.0f}")
    print(f"  Child 2 visits: {tree.get_visit_count(first_child+2):.0f}")

    # Create backup manager
    backup_mgr = mcts_py.BackupManager(tree)

    # Test backup along path: child_0 → root
    path = [first_child, root]  # Leaf to root order
    print(f"\n[2/3] Backing up path: {path}")
    print(f"  Leaf value: 0.5")

    result = backup_mgr.backup_value_along_path(path, 0.5, None)

    print(f"  Backup success: {result.success}")
    print(f"  Nodes updated: {result.nodes_updated}")

    print(f"\n[3/3] After backup:")
    print(f"  Root visits: {tree.get_visit_count(root):.0f}")
    print(f"  Child 0 visits: {tree.get_visit_count(first_child):.0f}")
    print(f"  Child 1 visits: {tree.get_visit_count(first_child+1):.0f}")
    print(f"  Child 2 visits: {tree.get_visit_count(first_child+2):.0f}")

    root_visits_after = tree.get_visit_count(root)
    child0_visits_after = tree.get_visit_count(first_child)

    print(f"\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    if root_visits_after == 1.0 and child0_visits_after == 1.0:
        print(f"  ✅ Backup working correctly")
        print(f"     Both root and child have 1 visit")
    elif root_visits_after == 1.0 and child0_visits_after == 0.0:
        print(f"  ❌ BUG: Only root updated, child NOT updated")
        print(f"     Backup is skipping intermediate nodes!")
    elif root_visits_after == 0.0:
        print(f"  ❌ BUG: Backup completely failed")
        print(f"     No nodes were updated")
    else:
        print(f"  ⚠️  UNEXPECTED: Root={root_visits_after}, Child={child0_visits_after}")

    print("="*80 + "\n")

    assert root_visits_after >= 1.0, "Root should have at least 1 visit"
    assert child0_visits_after >= 1.0, "Child should have at least 1 visit"


if __name__ == "__main__":
    test_backup_direct()
