"""Tests for bqpp_prototype.voi — full E[max(X,0)] expected improvement.

Audit §1.2 regression: my original ``s * phi(z)`` formula overestimates
EI in clear-lead regimes. This test file pins the correct formula and
explicitly compares against the wrong one.
"""

import math

import numpy as np
import pytest
from scipy.stats import truncnorm

from bqpp_prototype.voi import (
    expected_improvement,
    standard_normal_cdf,
    standard_normal_pdf,
    wrong_voi_phi_only,
)


def test_standard_normal_pdf_at_zero():
    """phi(0) = 1 / sqrt(2 pi) ≈ 0.3989."""
    assert math.isclose(standard_normal_pdf(0.0), 1.0 / math.sqrt(2 * math.pi))


def test_standard_normal_cdf_symmetric():
    """Phi(-z) + Phi(z) = 1 for all z."""
    for z in [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]:
        assert math.isclose(
            standard_normal_cdf(-z) + standard_normal_cdf(z),
            1.0,
            abs_tol=1e-9,
        )


def test_ei_at_delta_zero():
    """Δ=0 ⇒ EI = s * phi(0) = s / sqrt(2 pi)."""
    s = 0.1
    ei = expected_improvement(0.0, s)
    expected = s / math.sqrt(2 * math.pi)
    assert math.isclose(ei, expected, abs_tol=1e-9)


def test_ei_decreases_with_delta():
    """EI is strictly decreasing in delta for fixed s."""
    s = 0.1
    eis = [expected_improvement(d, s) for d in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]]
    for a, b in zip(eis, eis[1:]):
        assert b < a, f"non-monotone: {eis}"


def test_ei_at_large_delta_goes_to_zero():
    """As Δ → ∞, EI → 0 (clear-loss arm)."""
    s = 0.1
    ei = expected_improvement(10.0, s)  # 100 sigmas away
    assert ei < 1e-9


def test_ei_at_s_zero_is_zero():
    """No uncertainty ⇒ no expected improvement."""
    assert expected_improvement(0.5, 0.0) == 0.0
    assert expected_improvement(0.0, 0.0) == 0.0


def test_ei_matches_scipy_truncnorm():
    """Cross-check with scipy.stats.truncnorm.expect for several cases.

    For X ~ N(-Δ, s²), E[max(X, 0)] = E[X | X > 0] * P(X > 0).
    Using scipy: lower = (0 - mean) / s, upper = inf, mean = -Δ:
        a = -mean / s = Δ / s
        E[max(X, 0)] = trunc(a, inf, loc=-Δ, scale=s).expect(lambda x: x) * (1 - Phi(a))

    Wait — that's not quite right. The truncated mean assumes X > 0,
    but we want E[max(X, 0)] which is integrated over the FULL real
    line of X. Equivalent: E[max(X, 0)] = ∫_0^∞ x f(x) dx where
    f is N(-Δ, s²). Use scipy.stats.norm directly with numerical
    integration via .expect(lambda x: max(x, 0)).
    """
    from scipy.stats import norm

    for delta, s in [(0.0, 0.1), (0.05, 0.1), (0.1, 0.1), (0.3, 0.1)]:
        # Sample-based reference using a wide range
        rv = norm(loc=-delta, scale=s)
        # numerical integration via .expect
        ref = rv.expect(lambda x: max(x, 0.0), lb=-10 * s - delta, ub=10 * s)
        ours = expected_improvement(delta, s)
        assert math.isclose(ours, ref, rel_tol=1e-3, abs_tol=1e-6), (
            f"delta={delta} s={s}: ours={ours} ref={ref}"
        )


def test_voi_phi_only_overestimates_in_clear_lead():
    """Audit §1.2 regression.

    My original ``s * phi(Δ/s)`` formula was supposed to be an
    "underestimate" but in clear-lead regimes (Δ > 0) the correct
    formula subtracts ``Δ * Phi(-Δ/s)`` — which is positive — so
    the wrong formula is in fact larger than the correct one.

    This test pins the wrong-direction error so it can never be
    silently re-introduced.
    """
    # Cases chosen so that both values are non-trivially positive AND
    # the clear-lead gap is well above f64 noise. (delta=0.3, s=0.05)
    # was excluded because z=6 makes both values ~1e-10 where f64
    # rounding dominates.
    cases = [(0.05, 0.1), (0.1, 0.1), (0.2, 0.1), (0.15, 0.1)]
    for delta, s in cases:
        wrong = wrong_voi_phi_only(delta, s)
        correct = expected_improvement(delta, s)
        assert wrong >= correct, (
            f"audit §1.2: 'wrong formula' should be >= correct in clear-lead. "
            f"delta={delta} s={s}: wrong={wrong} correct={correct}"
        )
        # Use relative gap rather than absolute so cases with small
        # absolute EI still pin the over-estimate direction.
        rel_gap = (wrong - correct) / max(correct, 1e-12)
        assert rel_gap > 1e-3, (
            f"audit §1.2: relative over-estimate too small. "
            f"delta={delta} s={s}: wrong={wrong} correct={correct} "
            f"rel_gap={rel_gap}"
        )


def test_voi_phi_only_equals_full_at_delta_zero():
    """At Δ=0, both formulas give the same value (s · phi(0))."""
    for s in [0.05, 0.1, 0.2, 0.5]:
        wrong = wrong_voi_phi_only(0.0, s)
        correct = expected_improvement(0.0, s)
        assert math.isclose(wrong, correct, abs_tol=1e-12)


def test_ei_negative_delta_clamped():
    """If caller passes Δ<0 (arm a > arm b), EI should still be sensible.

    Convention: the function clamps delta to 0 (treats as "tied").
    """
    ei = expected_improvement(-0.1, 0.1)
    expected = expected_improvement(0.0, 0.1)
    assert math.isclose(ei, expected, abs_tol=1e-9)
