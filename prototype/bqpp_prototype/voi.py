"""Full expected improvement E[max(X, 0)] for a Gaussian-posterior arm.

This module corrects the audit's §1.2 finding. My original formula was

    VOI(a) = phi(-Delta / s_a) * s_a

and I described dropping the second truncated-normal term as a
"conservative underestimate." This was wrong on two counts:

1. The proper expected improvement under
    X ~ N(mu_a - mu_b, s_a^2)  =  N(-Delta, s_a^2)  with Delta >= 0
   is

    E[max(X, 0)] = s_a * phi(Delta / s_a) - Delta * Phi(-Delta / s_a)

   Note the second term is **subtracted**, not added.

2. Dropping the second term overestimates EI in the clear-lead regime
   (where the two terms nearly cancel for arms with mu_a < mu_b).
   Overestimating EI delays halt — search runs longer than necessary —
   which is the **opposite** of "safer."

The correct prescription is the full expected-improvement formula, and
the test ``test_voi_phi_only_overestimates`` regression-tests against
the wrong formula to make sure we never silently revert.

References:
    Russo, D. & Van Roy, B. (2018). "Satisficing in Time-Sensitive
    Bandit Learning." Mathematics of Operations Research.

    The truncated-normal expected-improvement formula is a textbook
    Bayesian-optimization result; see e.g. Frazier et al. 2009,
    "The Knowledge-Gradient Policy for Correlated Normal Beliefs."
"""

from __future__ import annotations

import math


_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def standard_normal_pdf(z: float) -> float:
    """Standard-normal density phi(z) = (1/sqrt(2 pi)) exp(-z^2/2)."""
    return _INV_SQRT_2PI * math.exp(-0.5 * z * z)


def standard_normal_cdf(z: float) -> float:
    """Standard-normal CDF Phi(z). Uses math.erf for accuracy."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def expected_improvement(delta: float, s: float) -> float:
    """Full expected improvement E[max(X, 0)] under X ~ N(-delta, s^2).

    Formula:
        E[max(X, 0)] = s * phi(delta / s) - delta * Phi(-delta / s)

    where ``delta = mu_b - mu_a >= 0`` (the gap between the empirical
    best and the challenger arm `a`).

    Edge cases:
        delta = 0           ⇒ s * phi(0) - 0 * Phi(0) = s / sqrt(2 pi)
                              ≈ 0.3989 * s.
        delta -> +∞         ⇒ phi(inf) = 0, Phi(-inf) = 0, EI -> 0.
        s = 0               ⇒ undefined (no uncertainty); we return 0.0
                              by convention.

    Returns a non-negative float.
    """
    if s <= 0.0:
        return 0.0
    if delta < 0.0:
        # Caller bug; arm `a` has higher mean than the empirical best.
        # Treat as delta = 0 (arm `a` is at least as good).
        delta = 0.0
    z = delta / s
    return s * standard_normal_pdf(z) - delta * standard_normal_cdf(-z)


def wrong_voi_phi_only(delta: float, s: float) -> float:
    """The *wrong* formula my P09 plan had. Kept here purely for
    regression testing — see test_voi_phi_only_overestimates.

    DO NOT USE in production code. Always use :func:`expected_improvement`.
    """
    if s <= 0.0:
        return 0.0
    z = delta / s
    return s * standard_normal_pdf(z)
