"""Tests for bqpp_prototype.reservoir — nested-reservoir live-set."""

from bqpp_prototype.reservoir import Reservoir, lambda_score, quantile


def test_quantile_single_value():
    assert quantile([5.0], 0.25) == 5.0


def test_quantile_two_values():
    """linear interpolation between data points."""
    assert quantile([0.0, 10.0], 0.5) == 5.0


def test_quantile_clean_25th():
    """[1, 2, 3, 4]: 25th percentile = 1 + 0.75 * (2 - 1) = 1.75."""
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.25) == 1.75


def test_lambda_score_components():
    """Lambda = U + ρ * KG + τ * log_prior_smoothed."""
    s = lambda_score(
        upper_ci_a=0.6, kg_a=0.05, log_prior_smoothed_a=-1.0,
        rho=1.0, tau=0.1,
    )
    # = 0.6 + 1.0 * 0.05 + 0.1 * (-1.0) = 0.55
    assert abs(s - 0.55) < 1e-9


def test_reservoir_add_respects_max_size():
    res = Reservoir(max_size=2)
    assert res.add(0, current_iter=0)
    assert res.add(1, current_iter=0)
    # full; cannot add a 3rd
    assert not res.add(2, current_iter=0)


def test_reservoir_remove_starts_cooldown():
    res = Reservoir(max_size=4, cooldown_iters=10)
    res.add(0, current_iter=0)
    res.remove(0, current_iter=5)
    # arm 0 cannot re-enter until iter >= 5 + 10 = 15
    assert not res.is_eligible(0, current_iter=10)
    assert not res.is_eligible(0, current_iter=14)
    assert res.is_eligible(0, current_iter=15)


def test_reservoir_quantile_pruning():
    """Live arms with score below 25th-percentile are removed."""
    res = Reservoir(max_size=10, cooldown_iters=5)
    for i in range(4):
        res.add(i, current_iter=0)
    scores = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3}
    # 25th percentile of [0.3, 0.5, 0.7, 0.9] = 0.45
    # Strict <: arm with score 0.3 is removed.
    removed = res.prune_below_quantile(scores, q=0.25, current_iter=10)
    assert removed == [3]
    assert res.live == [0, 1, 2]


def test_reservoir_no_thrashing():
    """An arm just-removed cannot be added back within cooldown_iters."""
    res = Reservoir(max_size=4, cooldown_iters=200)
    res.add(0, current_iter=0)
    res.remove(0, current_iter=100)
    # 50 iters later, should still be in cooldown
    assert not res.add(0, current_iter=150)
    # After cooldown expires, can re-enter
    assert res.add(0, current_iter=301)
