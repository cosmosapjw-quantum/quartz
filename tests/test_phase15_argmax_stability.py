"""Tests for quartz.phase15_argmax_stability — H1 bootstrap argmax-stability.

Pins the Dirichlet argmax-posterior math and the two falsifiable contracts:
(1) stability increases with N at a fixed gap (more data → more confident);
(2) the pre-experiment discrimination gate rejects a saturated signal.
"""

import numpy as np
import pytest

from quartz.phase15_argmax_stability import (
    ARGMAX_STABILITY_SCHEMA_VERSION,
    argmax_stability,
    counts_from_policy,
    dirichlet_argmax_posterior,
    should_stop_by_argmax_stability,
    stability_discrimination_gate,
)


def test_posterior_is_probability_vector():
    post = dirichlet_argmax_posterior([10, 5, 2], seed=0)
    assert post.shape == (3,)
    assert post.sum() == pytest.approx(1.0)
    assert np.all(post >= 0.0)
    # the most-visited arm has the highest argmax-probability
    assert int(np.argmax(post)) == 0


def test_blowout_is_stable_tight_race_is_not():
    # A blowout (40 vs 1 vs 1) → stability ~1.
    assert argmax_stability([40, 1, 1], seed=0) > 0.98
    # A near-tie (10 vs 9) → stability well below 1 (real flip risk).
    tie = argmax_stability([10, 9], seed=0)
    assert 0.5 < tie < 0.85


def test_stability_increases_with_n_at_fixed_gap():
    # Same 60/40 split, growing N → monotonically more stable. This is the
    # signal's core discriminating property; a flat response would be
    # useless (and would fail the gate below). Base [6,4] (N=10) starts
    # genuinely uncertain so there is room to grow toward 1.
    prev = 0.0
    seen = []
    for scale in (1, 2, 4, 8, 16):
        counts = [6 * scale, 4 * scale]
        s = argmax_stability(counts, seed=0)
        seen.append(s)
        assert s >= prev - 0.02, seen  # non-decreasing (small MC slack)
        prev = s
    assert seen[0] < 0.85, seen  # small-N is genuinely uncertain
    assert seen[-1] > seen[0] + 0.1, seen  # clearly grows overall


def test_single_and_empty_arm():
    assert argmax_stability([5]) == 1.0
    assert argmax_stability([]) == 0.0
    assert dirichlet_argmax_posterior([]).size == 0
    assert dirichlet_argmax_posterior([7]).tolist() == [1.0]


def test_all_zero_counts_is_uniform_posterior():
    # Symmetric Dirichlet(alpha,...) → uniform argmax posterior in
    # expectation; assert within MC slack (finite n_boot).
    post = dirichlet_argmax_posterior([0, 0, 0, 0], seed=0)
    assert post == pytest.approx([0.25, 0.25, 0.25, 0.25], abs=0.03)
    assert argmax_stability([0, 0, 0]) == pytest.approx(1.0 / 3.0, abs=0.03)


def test_should_stop_respects_threshold_and_min_visits():
    # Blowout above min_visits → stop.
    stop, meta = should_stop_by_argmax_stability([40, 1, 1], threshold=0.9, min_visits=8, seed=0)
    assert stop is True
    assert meta["argmax_index"] == 0
    assert meta["argmax_stability"] > 0.9
    # Same shape but below min_visits → never stop even if "stable".
    stop2, meta2 = should_stop_by_argmax_stability([4, 0, 0], threshold=0.9, min_visits=8, seed=0)
    assert stop2 is False
    assert meta2["total_visits"] == 4.0
    # Tight race above min_visits → do not stop.
    stop3, _ = should_stop_by_argmax_stability([10, 9], threshold=0.9, min_visits=8, seed=0)
    assert stop3 is False


def test_discrimination_gate_passes_on_varied_positions():
    # Mix of blowout, tight, and mid positions → stability varies → passes.
    vectors = [[40, 1, 1], [10, 9], [20, 12, 3], [5, 5, 5]]
    gate = stability_discrimination_gate(vectors, seed=0)
    assert gate["n_positions"] == 4
    assert gate["discriminates"] is True
    assert gate["stability_std"] > 1e-3
    assert gate["stability_min"] < gate["stability_max"]


def test_discrimination_gate_fails_on_saturated_signal():
    # All positions are extreme blowouts → stability ~1 everywhere →
    # zero-variance → the gate correctly reports the signal as useless.
    vectors = [[200, 1], [300, 1], [250, 2]]
    gate = stability_discrimination_gate(vectors, seed=0, trivial_std_eps=1e-3)
    assert gate["discriminates"] is False
    assert gate["stability_std"] <= 1e-3


def test_discrimination_gate_empty_input():
    gate = stability_discrimination_gate([])
    assert gate["n_positions"] == 0
    assert gate["discriminates"] is False
    assert gate["stability_mean"] is None


def test_counts_from_policy_roundtrip():
    counts = counts_from_policy([0.5, 0.3, 0.2], 100)
    assert counts.tolist() == [50, 30, 20]
    assert int(counts.sum()) == 100
    # rounding remainder handed to the largest-share arm, sum preserved
    counts2 = counts_from_policy([0.34, 0.33, 0.33], 100)
    assert int(counts2.sum()) == 100
    assert counts2[0] >= counts2[1]
    # degenerate inputs
    assert counts_from_policy([0.0, 0.0], 50).tolist() == [0, 0]
    assert counts_from_policy([0.6, 0.4], 0).tolist() == [0, 0]


def test_schema_version_surfaced():
    _, meta = should_stop_by_argmax_stability([40, 1, 1], seed=0)
    assert meta["argmax_stability_schema_version"] == ARGMAX_STABILITY_SCHEMA_VERSION
