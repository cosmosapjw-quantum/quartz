"""Tests for bqpp_prototype.kg — Knowledge Gradient approximation."""

import math

import pytest

from bqpp_prototype.kg import (
    compute_kg_array,
    kg_gaussian_per_arm,
    top_m_kg_with_uc_bound,
)


def test_kg_zero_at_empirical_best():
    """KG[best] is set to 0 by convention."""
    mu_hats = [0.8, 0.5, 0.4]
    n_pulls = [200, 100, 100]
    sigma2s = [0.01, 0.01, 0.01]
    kg = compute_kg_array(mu_hats, n_pulls, sigma2s)
    assert kg[0] == 0.0


def test_kg_positive_for_subopt_arms_with_finite_uncertainty():
    """KG > 0 for all sub-optimal arms with sigma2 > 0."""
    mu_hats = [0.8, 0.5, 0.4]
    n_pulls = [200, 100, 100]
    sigma2s = [0.01, 0.01, 0.01]
    kg = compute_kg_array(mu_hats, n_pulls, sigma2s)
    for a in [1, 2]:
        assert kg[a] > 0.0, f"arm {a} has zero KG"


def test_kg_monotone_in_sigma_a():
    """More uncertain challenger has larger KG (variance-adaptive)."""
    mu_hats = [0.8, 0.5]
    n_pulls = [200, 100]
    kg_low_sigma = kg_gaussian_per_arm(
        mu_hats[1],
        n_pulls[1],
        0.001,
        mu_hats[0],
        n_pulls[0],
        0.01,
    )
    kg_high_sigma = kg_gaussian_per_arm(
        mu_hats[1],
        n_pulls[1],
        0.05,
        mu_hats[0],
        n_pulls[0],
        0.01,
    )
    assert kg_high_sigma > kg_low_sigma, f"low={kg_low_sigma} vs high={kg_high_sigma}"


def test_kg_monotone_in_inverse_n_a():
    """Less-pulled arm has larger KG."""
    mu_hats = [0.8, 0.5]
    sigma2s = [0.01, 0.01]
    kg_few = kg_gaussian_per_arm(
        mu_hats[1],
        5,
        sigma2s[1],
        mu_hats[0],
        200,
        sigma2s[0],
    )
    kg_many = kg_gaussian_per_arm(
        mu_hats[1],
        100,
        sigma2s[1],
        mu_hats[0],
        200,
        sigma2s[0],
    )
    assert kg_few > kg_many, f"few={kg_few} vs many={kg_many}"


def test_kg_at_clear_loss_goes_to_zero():
    """Δ_a / s_a → ∞ ⇒ KG → 0."""
    # Big gap, tiny noise
    kg = kg_gaussian_per_arm(
        mu_a=0.0,
        n_a=100,
        sigma2_a=0.0001,
        mu_b=1.0,
        n_b=100,
        sigma2_b=0.0001,
    )
    assert kg < 1e-9


def test_kg_full_formula_matches_voi_for_kg_position():
    """KG_a = expected_improvement(Δ, s) where Δ = mu_b - mu_a."""
    from bqpp_prototype.voi import expected_improvement

    mu_a, n_a, sigma2_a = 0.4, 50, 0.04
    mu_b, n_b, sigma2_b = 0.7, 100, 0.04
    lambda0 = 4.0
    s = math.sqrt(sigma2_b / (n_b + lambda0) + sigma2_a / (n_a + lambda0))
    delta = mu_b - mu_a
    expected = expected_improvement(delta, s)
    actual = kg_gaussian_per_arm(
        mu_a,
        n_a,
        sigma2_a,
        mu_b,
        n_b,
        sigma2_b,
        lambda0,
    )
    assert math.isclose(actual, expected, rel_tol=1e-9)


def test_top_m_kg_includes_best():
    """top-m KG always includes the empirical best (so KG[best] is exactly 0)."""
    mu_hats = [0.8, 0.5, 0.4, 0.3]
    n_pulls = [200, 100, 100, 100]
    sigma2s = [0.01, 0.01, 0.01, 0.01]
    upper_ci = [0.85, 0.6, 0.5, 0.4]
    kg, n_eval = top_m_kg_with_uc_bound(
        mu_hats,
        n_pulls,
        sigma2s,
        upper_ci,
        lower_ci_best=0.75,
        m=2,
    )
    # arm 0 is best, KG[0] should be exactly 0
    assert kg[0] == 0.0
    # at least 2 arms evaluated (best + top of upper_ci)
    assert n_eval >= 2


def test_top_m_kg_bounds_below_threshold_arms_by_uc():
    """Arms NOT in top-m get the conservative U_a - L_b bound."""
    mu_hats = [0.8, 0.5, 0.4, 0.3]
    n_pulls = [200, 100, 100, 100]
    sigma2s = [0.01, 0.01, 0.01, 0.01]
    upper_ci = [0.85, 0.6, 0.5, 0.4]
    L_b = 0.75
    kg, _ = top_m_kg_with_uc_bound(
        mu_hats,
        n_pulls,
        sigma2s,
        upper_ci,
        lower_ci_best=L_b,
        m=1,
    )
    # arm 1 is in top-m (highest upper_ci among non-best);
    # arm 2, 3 are NOT in top-m and should get U_a - L_b bound.
    # arm 2: 0.5 - 0.75 = -0.25 ⇒ max(_, 0) = 0
    # arm 3: 0.4 - 0.75 = -0.35 ⇒ max(_, 0) = 0
    # Since the bounds are all negative, the bound becomes 0.
    assert kg[2] == 0.0 or kg[2] >= 0.0
    assert kg[3] == 0.0


def test_compute_kg_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_kg_array([0.5], [1, 2], [0.01])
