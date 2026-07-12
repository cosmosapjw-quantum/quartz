"""Tests verifying MCTSTree clear/reset semantics via Python bindings."""

import pytest
import mcts_py


def test_tree_clear_resets_state():
    tree = mcts_py.MCTSTree(128)

    root = tree.add_root_node(0.5, 0)
    tree.set_visit_count(root, 10.0)
    tree.set_total_value(root, -2.0)
    tree.set_prior_prob(root, 0.3)
    flags = tree.get_flags(root)
    flags.set_expanded(True)
    tree.set_flags(root, flags)

    tree.clear()

    prior = 0.25
    new_root = tree.add_root_node(prior, 1)
    assert tree.get_visit_count(new_root) == pytest.approx(0.0)
    assert tree.get_total_value(new_root) == pytest.approx(0.0)
    assert tree.get_prior_prob(new_root) == pytest.approx(prior)
    assert not tree.get_flags(new_root).is_expanded()
