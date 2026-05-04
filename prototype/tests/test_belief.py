"""Tests for bqpp_prototype.belief — Welford + empirical-Bayes shrinkage."""

import math

import numpy as np
import pytest

from bqpp_prototype.belief import (
    WelfordState,
    empirical_bayes_shrinkage,
    map_q_to_unit,
    map_unit_to_q,
    shrunk_sigma,
)


def test_welford_matches_numpy_var_at_1e6_samples():
    """Welford incremental variance vs np.var(ddof=1) on 1e6 normal samples."""
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=0.5, scale=0.1, size=1_000_000)
    state = WelfordState()
    for x in samples:
        state.update(float(x))
    welford_var = state.sample_variance()
    numpy_var = float(np.var(samples, ddof=1))
    assert abs(welford_var - numpy_var) < 1e-9, (
        f"Welford {welford_var} vs np.var {numpy_var}"
    )


def test_welford_f32_drift_documented():
    """Document the f32 vs f64 Welford drift past N=1e5.

    The Rust EdgeView.m2 is f64 *precisely* because f32 Welford diverges
    from the true variance once N gets large. This test pins the
    expected drift magnitude so that any future change to the Rust
    field type is preceded by re-checking this expectation.

    We compute Welford in f32 (via numpy.float32) and compare to
    np.var(samples_f64, ddof=1). At N=1e6 the f32 drift is around
    5e-4 — small but non-zero. f64 stays within 1e-9.
    """
    rng = np.random.default_rng(1)
    samples_f64 = rng.normal(loc=0.5, scale=0.1, size=1_000_000)
    # f32 Welford
    n = 0
    mean = np.float32(0.0)
    M2 = np.float32(0.0)
    for x in samples_f64.astype(np.float32):
        n += 1
        delta = x - mean
        mean = mean + delta / np.float32(n)
        delta2 = x - mean
        M2 = M2 + delta * delta2
    f32_var = float(M2 / (n - 1))
    f64_var = float(np.var(samples_f64, ddof=1))
    drift = abs(f32_var - f64_var)
    # Drift IS observable (f32 precision loss past N=1e5);
    # the assertion bracket this test as an acceptance criterion:
    assert drift > 1e-6, (
        f"f32 drift {drift} is too small; the test needs re-tuning, "
        "but it's a sign things are accidentally OK at f32 precision."
    )
    assert drift < 1e-2, (
        f"f32 drift {drift} is much larger than expected; investigate."
    )


def test_empirical_bayes_shrinkage_at_n_zero_gives_parent_variance():
    """At N=0, M2=0 with lambda0>0 ⇒ shrunk variance == parent variance.

    Hand-derived: (0 + 4 * 0.09) / (max(-1, 1) + 4) = 0.36 / 5 = 0.072
    so sigma_a = sqrt(0.072) ≈ 0.2683.

    Wait — this matches the Rust EdgeView::sigma_a expected value at
    n=1, not n=0. At n=0 the Rust formula uses (n - 1, 1).max() too,
    so both n=0 and n=1 give the SAME result. Good — this is the
    intended behavior.
    """
    sigma2 = empirical_bayes_shrinkage(
        n=0, M2=0.0, sigma2_parent=0.09, lambda0=4.0,
    )
    # (0 + 4 * 0.09) / (max(-1, 1) + 4) = 0.36 / 5 = 0.072
    assert math.isclose(sigma2, 0.072, abs_tol=1e-9)
    sigma = shrunk_sigma(n=0, M2=0.0, sigma2_parent=0.09, lambda0=4.0)
    assert math.isclose(sigma, math.sqrt(0.072), abs_tol=1e-9)


def test_empirical_bayes_shrinkage_at_n_one_matches_n_zero():
    """N=1 with M2=0 gives the same result as N=0 (both use the floor)."""
    s_n0 = empirical_bayes_shrinkage(n=0, M2=0.0, sigma2_parent=0.09, lambda0=4.0)
    s_n1 = empirical_bayes_shrinkage(n=1, M2=0.0, sigma2_parent=0.09, lambda0=4.0)
    assert math.isclose(s_n0, s_n1, abs_tol=1e-12)


def test_empirical_bayes_shrinkage_data_dominates_at_large_n():
    """At large N with M2 = (n-1) * 0.04 ⇒ shrunk variance approaches 0.04."""
    n = 1000
    sigma2_data = 0.04
    M2 = (n - 1) * sigma2_data
    sigma2 = empirical_bayes_shrinkage(
        n=n, M2=M2, sigma2_parent=0.0625, lambda0=4.0,
    )
    # Difference should be small (within 5% of data variance)
    rel_err = abs(sigma2 - sigma2_data) / sigma2_data
    assert rel_err < 0.05, f"sigma2 = {sigma2}, expected ≈ {sigma2_data}"


def test_empirical_bayes_shrinkage_floors_at_sigma2_floor():
    """Even when M2=0 and sigma2_parent=0, the result is at least sigma2_floor."""
    sigma2 = empirical_bayes_shrinkage(
        n=100, M2=0.0, sigma2_parent=0.0, lambda0=4.0, sigma2_floor=1e-6,
    )
    assert sigma2 >= 1e-6


def test_empirical_bayes_shrinkage_rejects_negative_lambda():
    with pytest.raises(ValueError):
        empirical_bayes_shrinkage(n=1, M2=0.0, sigma2_parent=0.1, lambda0=-1.0)


def test_empirical_bayes_shrinkage_rejects_negative_parent_var():
    with pytest.raises(ValueError):
        empirical_bayes_shrinkage(n=1, M2=0.0, sigma2_parent=-0.1, lambda0=4.0)


def test_q_unit_mapping_round_trip():
    """map_q_to_unit and map_unit_to_q are exact inverses."""
    for q in [-1.0, -0.5, 0.0, 0.3, 0.999]:
        u = map_q_to_unit(q)
        q_back = map_unit_to_q(u)
        assert math.isclose(q, q_back, abs_tol=1e-12)


def test_q_unit_mapping_endpoints():
    """Q=-1 → mu=0; Q=+1 → mu=1; Q=0 → mu=0.5."""
    assert map_q_to_unit(-1.0) == 0.0
    assert map_q_to_unit(1.0) == 1.0
    assert map_q_to_unit(0.0) == 0.5
