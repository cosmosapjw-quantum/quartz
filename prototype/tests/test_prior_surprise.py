"""Tests for bqpp_prototype.prior_surprise — χ² *statistic*, no p-value."""

import math

import pytest

from bqpp_prototype.prior_surprise import (
    expand_active_at,
    prior_surprise_statistic,
)


def test_zero_when_visits_match_prior():
    """exact match ⇒ statistic = 0."""
    visits = [25, 25, 25, 25]
    prior = [0.25, 0.25, 0.25, 0.25]
    s = prior_surprise_statistic(visits, prior)
    assert abs(s) < 1e-9


def test_positive_when_visits_diverge():
    """concentrated visits ⇒ large statistic."""
    visits = [80, 10, 10, 0]
    prior = [0.25, 0.25, 0.25, 0.25]
    s = prior_surprise_statistic(visits, prior)
    # (80 - 25)^2/25 + (10 - 25)^2/25 + (10 - 25)^2/25 + (0 - 25)^2/25
    # = 121 + 9 + 9 + 25 = 164
    assert math.isclose(s, 164.0, abs_tol=1.0)


def test_invariant_to_permutation():
    """χ² statistic is invariant to permuting (N_a, π_a) jointly."""
    visits = [80, 10, 10, 0]
    prior = [0.25, 0.25, 0.25, 0.25]
    s1 = prior_surprise_statistic(visits, prior)
    # Swap arm 0 and arm 3 in BOTH visits and prior
    visits2 = [0, 10, 10, 80]
    prior2 = [0.25, 0.25, 0.25, 0.25]
    s2 = prior_surprise_statistic(visits2, prior2)
    assert math.isclose(s1, s2, abs_tol=1e-9)


def test_zero_visits_returns_zero():
    """N=0 ⇒ statistic = 0 regardless of prior."""
    s = prior_surprise_statistic([0, 0, 0], [0.5, 0.3, 0.2])
    assert s == 0.0


def test_does_not_emit_p_value():
    """Audit §1.6 regression: this module does NOT actually IMPORT or USE
    scipy.stats. Inspect runtime globals (not source text — the docstring
    intentionally MENTIONS scipy.stats to warn callers, which would be
    a false-positive on a string-grep test).
    """
    import bqpp_prototype.prior_surprise as mod
    # Module's runtime globals must not contain scipy.stats imports.
    assert "scipy" not in [name.split(".", 1)[0] for name in mod.__dict__]
    # And the module's __dict__ must not contain a function named like
    # ``chi2_pvalue`` or ``ppf``.
    forbidden_attrs = {"chi2_pvalue", "p_value", "ppf", "chi2"}
    for attr in mod.__dict__:
        assert attr not in forbidden_attrs, (
            f"prior_surprise.{attr} re-introduces the formal-test framing"
        )


def test_expand_active_threshold_is_caller_supplied():
    """expand_active_at takes a threshold — not a degrees-of-freedom-derived value."""
    s = prior_surprise_statistic(
        visit_counts=[80, 10, 10, 0],
        prior=[0.25, 0.25, 0.25, 0.25],
    )
    # caller-supplied threshold; the function does NOT consult chi2 CDF
    assert expand_active_at(s, threshold=10.0) is True
    assert expand_active_at(s, threshold=200.0) is False


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        prior_surprise_statistic([1, 2], [0.5, 0.3, 0.2])


def test_handles_zero_prior_via_eps_clamp():
    """A zero-prior action with positive visits doesn't blow up."""
    visits = [10, 10, 10, 0]
    prior = [0.0, 0.5, 0.3, 0.2]  # arm 0 has zero prior!
    s = prior_surprise_statistic(visits, prior, eps=1e-6)
    # Doesn't raise; statistic is large but finite.
    assert math.isfinite(s)
    assert s > 0
