"""Empirical Bernstein confidence intervals + L_b > max_{a≠b} U_a certificate.

This module corrects the audit's §1.3 finding: my original
``eb_gap = best_mu - 2 * EB_b`` formula uses only the best-arm bound and
is **not** a best-vs-runner-up certificate. The correct formula is

    g_EB = L_b − max_{a ≠ b} U_a

with per-arm widths from the Maurer-Pontil 2009 empirical Bernstein
bound:

    w_a = sqrt(2 * sigma2_a * log(3 K t^alpha / delta) / max(n_a, 1))
        + 7 * R * log(3 K t^alpha / delta) / (3 * max(n_a - 1, 1))

where:
    R = range of the random variable (1 if values in [0, 1], 2 if in
        [-1, 1]). The prototype canonicalizes on [0, 1] (R = 1).
    K = number of arms / candidates.
    t = current iteration (used in the union-bound time scaling).
    alpha = 1.1 (matching the Kaufmann-Kalyanakrishnan 2013 KL-LUCB
        threshold form).

References:
    Maurer, A. & Pontil, M. (2009). "Empirical Bernstein Bounds and
    Sample Variance Penalization." COLT 2009.

A1-b audit fix: the original ``3 * K`` factor in the log argument only
covers a union bound over the K arms times a doubling factor for
upper/lower bounds *at one fixed t*. Because the width is recomputed
and acted on at every observe() call (anytime use, t=1,2,3,...), the
per-t failure probabilities must themselves sum to at most delta
across all t — a plain constant-times-K factor does not provide that,
so the realized anytime failure probability was roughly 14*delta, not
delta. ``eb_log_term`` now reuses ``kl_lucb.kl_lucb_beta``'s
already-correct time-uniform threshold (Kaufmann-Kalyanakrishnan
2013 Theorem 8's k1=405.5, alpha=1.1 peeling constant, which is
exactly built to make sum_t delta_t <= delta) instead of
reimplementing a similar-looking but uncorrected formula. Both
certificate families (KL-LUCB and this Maurer-Pontil EB bound) now
share one audited anytime-valid threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .kl_lucb import kl_lucb_beta


@dataclass
class EBInterval:
    """Per-arm empirical-Bernstein confidence interval on [0, 1] scale."""

    mu_hat: float
    width: float

    @property
    def lower(self) -> float:
        return max(self.mu_hat - self.width, 0.0)

    @property
    def upper(self) -> float:
        return min(self.mu_hat + self.width, 1.0)


def eb_log_term(K: int, t: int, delta: float, alpha: float = 1.1) -> float:
    """Compute the anytime-valid union-bound log term
    log(k1 * K * t^alpha / delta), k1=405.5 for alpha=1.1.

    This is the per-arm threshold that appears inside both the
    variance-scaled and the constant terms of the Maurer-Pontil
    width. Delegates to ``kl_lucb.kl_lucb_beta`` (see A1-b note in
    this module's docstring): the same KK13 peeling constant that
    makes the KL-LUCB certificate anytime-valid also fixes this one.

    ``t`` is clamped to >= 1 to avoid log(0) at the very first step.
    """
    t_safe = max(t, 1)
    K_safe = max(K, 1)
    return kl_lucb_beta(t=float(t_safe), K=float(K_safe), delta=delta, alpha=alpha)


def empirical_bernstein_width(
    n: int,
    sigma2: float,
    K: int,
    t: int,
    delta: float = 0.05,
    R: float = 1.0,
    alpha: float = 1.1,
) -> float:
    """Maurer-Pontil 2009 empirical Bernstein width for one arm.

    Formula:
        w = sqrt(2 * sigma2 * L / max(n, 1)) + 7 * R * L / (3 * max(n - 1, 1))

    where L = log(3 K t^alpha / delta).

    Returns the **width** (half-width of the CI), not the upper or
    lower bound. Caller forms the bounds via mu_hat ± width.

    The R parameter encodes the random-variable range:
        R = 1 if Q values mapped to [0, 1] (canonical for BQ++)
        R = 2 if Q values are raw [-1, 1]
    The audit (§6.1) recommends R=1 with mu_hat = (Q + 1) / 2.
    """
    if n <= 0:
        # No data ⇒ width is the maximum possible (1.0 on [0,1] scale).
        return 1.0
    L = eb_log_term(K, t, delta, alpha)
    var_term = math.sqrt(2.0 * sigma2 * L / max(n, 1))
    const_term = 7.0 * R * L / (3.0 * max(n - 1, 1))
    return var_term + const_term


def eb_interval_from_arm(
    mu_hat: float,
    n: int,
    sigma2: float,
    K: int,
    t: int,
    delta: float = 0.05,
    R: float = 1.0,
) -> EBInterval:
    """Build an EBInterval for one arm from its empirical statistics."""
    width = empirical_bernstein_width(n, sigma2, K, t, delta, R)
    return EBInterval(mu_hat=mu_hat, width=width)


@dataclass
class EBCertificate:
    """Result of evaluating the L_b > max_{a≠b} U_a certificate.

    ``gap`` is the signed certificate value:
        gap = L_b - max_{a ≠ b} U_a

    Stop is allowed when ``gap > 0``. Negative ``gap`` values are
    diagnostic: they tell us how far from PAC certainty we are.
    """

    best_pos: int
    runner_up_pos: int
    L_b: float
    max_U_runner: float
    gap: float

    @property
    def fired(self) -> bool:
        return self.gap > 0.0


def best_vs_runner_certificate(
    intervals: list[EBInterval],
) -> EBCertificate:
    """Compute the Empirical-Bernstein L_b > max_{a≠b} U_a certificate.

    The "best" arm is taken as ``argmax mu_hat``; the runner-up is
    ``argmax_{a ≠ best} U_a`` (the arm whose upper bound is closest to
    the best's mean — the toughest competitor under uncertainty).

    Raises ``ValueError`` if fewer than 2 arms are provided.
    """
    if len(intervals) < 2:
        raise ValueError(
            f"Need at least 2 arms for the certificate; got {len(intervals)}"
        )

    # Empirical best by mu_hat
    best_pos = max(range(len(intervals)), key=lambda i: intervals[i].mu_hat)
    L_b = intervals[best_pos].lower

    # Runner-up by upper bound (the toughest competitor)
    best_U_other = -math.inf
    runner_up_pos = -1
    for i, iv in enumerate(intervals):
        if i == best_pos:
            continue
        if iv.upper > best_U_other:
            best_U_other = iv.upper
            runner_up_pos = i

    gap = L_b - best_U_other
    return EBCertificate(
        best_pos=best_pos,
        runner_up_pos=runner_up_pos,
        L_b=L_b,
        max_U_runner=best_U_other,
        gap=gap,
    )
