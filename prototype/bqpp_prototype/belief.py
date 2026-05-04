"""Welford online variance + empirical-Bayes variance shrinkage.

The audit (``../report.md`` §6.1) re-named the legacy "Beta-Binomial
posterior" framing to **empirical-Bayes variance shrinkage**, which is
the more precise statistical term: we shrink the per-arm Welford
sample variance toward a parent (root-level) variance using a
weak-prior pseudo-count ``lambda0``.

Formula (canonical, on the [0, 1] scale):

    sigma2_shrunk = (M2 + lambda0 * sigma2_parent)
                  / (max(n - 1, 1) + lambda0)

Reference:
    Welford, B. P. (1962). "Note on a method for calculating corrected
    sums of squares and products." Technometrics 4(3): 419-420.

    Murphy, K. P. (2012). "Machine Learning: A Probabilistic
    Perspective", §4.6 (Normal-inverse-Gamma conjugate; equivalent
    formulation).

Hand-derived sanity values (regression-tested in tests/test_belief.py):

    n = 0,  M2 = 0,  sigma_parent = 0.3,  lambda0 = 4
        ⇒ sigma2 = (0 + 4 * 0.09) / (max(-1, 1) + 4) = 0.36 / 5 = 0.072
          sigma  ≈ 0.2683

    n = 1,  M2 = 0,  sigma_parent = 0.3,  lambda0 = 4
        ⇒ sigma2 = 0.36 / (max(0, 1) + 4) = 0.36 / 5 = 0.072
          sigma  ≈ 0.2683

    n large,  M2 = (n-1) * 0.04,  ...  ⇒ sigma2 → 0.04 (data dominates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class WelfordState:
    """Online running mean + sum-of-squared-deviations.

    ``mean`` and ``M2`` together let us compute the unbiased sample
    variance as ``M2 / (n - 1)``; the prototype keeps both as plain
    floats (the Rust ``EdgeView.m2`` field is f64 to avoid f32 drift
    past N=1e5 — verified empirically with the test
    ``test_welford_f32_vs_f64_drift``).
    """

    n: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def update(self, x: float) -> None:
        """Welford incremental update."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def sample_variance(self) -> float:
        """Unbiased sample variance ``M2 / (n - 1)``. Returns 0 at n<2."""
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)


def empirical_bayes_shrinkage(
    n: int,
    M2: float,
    sigma2_parent: float,
    lambda0: float = 4.0,
    sigma2_floor: float = 1e-6,
) -> float:
    """Empirical-Bayes shrunk variance.

    Formula:
        sigma2_shrunk = (M2 + lambda0 * sigma2_parent)
                      / (max(n - 1, 1) + lambda0)

    The denominator's ``max(n - 1, 1)`` (instead of just ``n - 1``)
    handles the n ∈ {0, 1} case without division by zero. The audit
    (§6.1) specifies this as the canonical form to align with the
    bias-corrected sample-variance denominator.

    The result is floored at ``sigma2_floor`` to prevent downstream
    divisions by zero in the Bernstein width formula.

    Returns the shrunk **variance**, not std-dev. Caller takes ``sqrt``
    when the std-dev is needed.
    """
    if lambda0 < 0:
        raise ValueError(f"lambda0 must be non-negative, got {lambda0}")
    if sigma2_parent < 0:
        raise ValueError(f"sigma2_parent must be non-negative, got {sigma2_parent}")
    n_eff = max(n - 1, 1) + lambda0
    numerator = M2 + lambda0 * sigma2_parent
    sigma2 = numerator / n_eff
    return max(sigma2, sigma2_floor)


def shrunk_sigma(
    n: int,
    M2: float,
    sigma2_parent: float,
    lambda0: float = 4.0,
    sigma2_floor: float = 1e-6,
) -> float:
    """Convenience wrapper returning sqrt of the shrunk variance."""
    return math.sqrt(
        empirical_bayes_shrinkage(n, M2, sigma2_parent, lambda0, sigma2_floor)
    )


def map_q_to_unit(q: float) -> float:
    """Map Q ∈ [-1, 1] to mu ∈ [0, 1] via mu = (Q + 1) / 2.

    Rank-preserving by construction. Required for Bernoulli-KL ops in
    the certificate / KL-LUCB modules. The audit (§1.4) flagged scale
    mixing as a real bug class; this single canonical mapping function
    is the prevention.
    """
    return 0.5 * (q + 1.0)


def map_unit_to_q(mu: float) -> float:
    """Inverse of :func:`map_q_to_unit`. Returns Q ∈ [-1, 1]."""
    return 2.0 * mu - 1.0
