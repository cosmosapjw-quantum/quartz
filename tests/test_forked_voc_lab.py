"""Tests for quartz.experiments.forked_voc — realized-VOC labeling of frozen traces."""

import numpy as np
import pytest

from quartz.experiments.forked_voc import (
    FORKED_VOC_SCHEMA_VERSION,
    computation_step_labels,
    decision_movement_curve,
    discrimination,
    label_trace_bundle,
    measure_tightness,
    screen_bundles,
    shallow_deep_margin_swing,
    top2_margin,
    total_variation,
    voc_proxy,
)


def _bundle(budgets, policies):
    return {"trace_budgets": budgets, "trace_policies": [np.asarray(p, float) for p in policies]}


# frozen traces of distinct decision character
STABLE = _bundle([8, 16, 32, 64], [[0.7, 0.2, 0.1]] * 4)  # never moves
LATE_FLIP = _bundle(  # deep search overturns the shallow pick
    [8, 16, 32, 64],
    [[0.45, 0.44, 0.11], [0.44, 0.45, 0.11], [0.4, 0.5, 0.1], [0.3, 0.62, 0.08]],
)
EARLY_SETTLE = _bundle(  # moves once early then locks in
    [8, 16, 32, 64],
    [[0.4, 0.5, 0.1], [0.7, 0.2, 0.1], [0.72, 0.19, 0.09], [0.73, 0.18, 0.09]],
)


def test_top2_margin_and_tv():
    assert top2_margin([0.7, 0.2, 0.1]) == pytest.approx(0.5)
    assert top2_margin([1.0]) == pytest.approx(1.0)
    assert top2_margin([]) == 0.0
    assert total_variation([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert total_variation([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_step_labels_detect_argmax_flip():
    steps = computation_step_labels(*[LATE_FLIP["trace_budgets"], LATE_FLIP["trace_policies"]])
    assert len(steps) == 3
    # the flip happens on the 8->16 step (argmax 0 -> 1)
    assert steps[0]["argmax_flipped"] is True
    assert all(s["decision_movement"] >= 0.0 for s in steps)
    # stable trace: no flips, zero movement
    stable_steps = computation_step_labels(STABLE["trace_budgets"], STABLE["trace_policies"])
    assert all(not s["argmax_flipped"] for s in stable_steps)
    assert sum(s["decision_movement"] for s in stable_steps) == pytest.approx(0.0)


def test_voc_proxy_orders_positions_by_computation_value():
    stable = voc_proxy(STABLE["trace_budgets"], STABLE["trace_policies"])
    late = voc_proxy(LATE_FLIP["trace_budgets"], LATE_FLIP["trace_policies"])
    # a settled position scores ~0; a churning one scores clearly higher
    assert stable == pytest.approx(0.0)
    assert late > 0.05
    assert late > stable


def test_voc_proxy_not_zero_when_churn_ends_at_equal_margin():
    # Stage 2 real-trace finding: a position with argmax flips must NOT score 0
    # just because its end margin equals its start margin. The primary proxy
    # (total decision movement) stays positive; the margin-swing diagnostic is
    # the one that can be 0 here.
    churn = _bundle([8, 16, 32], [[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.5, 0.3, 0.2]])
    assert voc_proxy(churn["trace_budgets"], churn["trace_policies"]) > 0.1
    lab = label_trace_bundle(churn)
    assert lab["n_argmax_flips"] == 2
    assert lab["voc_proxy"] > 0.1


def test_label_bundle_fields():
    lab = label_trace_bundle(LATE_FLIP)
    assert lab["forked_voc_schema_version"] == FORKED_VOC_SCHEMA_VERSION
    assert lab["n_steps"] == 3
    assert lab["n_argmax_flips"] >= 1
    assert lab["final_overturns_shallow"] is True
    assert lab["total_decision_movement"] > 0.0
    stable = label_trace_bundle(STABLE)
    assert stable["n_argmax_flips"] == 0
    assert stable["final_overturns_shallow"] is False


def test_screen_degeneracy_kill_check():
    # all-stable positions => proxies all 0 => degenerate (kill).
    deg = screen_bundles([STABLE, STABLE, STABLE])
    assert deg["degenerate"] is True
    assert deg["voc_proxy_std"] == pytest.approx(0.0)
    # a mix of decision characters => informative (not degenerate).
    mix = screen_bundles([STABLE, LATE_FLIP, EARLY_SETTLE])
    assert mix["degenerate"] is False
    assert mix["voc_proxy_std"] > 0.0
    assert mix["n_positions"] == 3
    assert 0.0 <= mix["overturn_rate"] <= 1.0


def test_measure_tightness_needs_varying_budget():
    bundles = [STABLE, LATE_FLIP, EARLY_SETTLE]
    # fixed shared ladder => constant per-move budget => tightness undefined.
    assert measure_tightness(bundles, [64, 64, 64]) is None
    # a controller that spent MORE where VOC is higher => positive tightness.
    proxies = [label_trace_bundle(b)["voc_proxy"] for b in bundles]
    order = np.argsort(proxies)  # ascending VOC
    budgets = [0, 0, 0]
    for rank, idx in enumerate(order):
        budgets[idx] = 8 * (rank + 1)  # more budget to higher VOC
    assert measure_tightness(bundles, budgets) == pytest.approx(1.0)


def test_discrimination_weak_vs_strong():
    # strong checkpoint: sharper, more overturns resolved => higher mean proxy
    weak = [STABLE, STABLE]
    strong = [LATE_FLIP, EARLY_SETTLE]
    d = discrimination(weak, strong)
    assert d["weak_degenerate"] is True
    assert d["strong_degenerate"] is False
    assert d["delta_strong_minus_weak"] > 0.0


def test_empty_and_short_traces_safe():
    assert voc_proxy([8], [[0.5, 0.5]]) == 0.0
    assert shallow_deep_margin_swing([8], [[0.5, 0.5]]) == 0.0
    assert decision_movement_curve([], []) == []
    s = screen_bundles([])
    assert s["n_positions"] == 0
    assert s["degenerate"] is True
