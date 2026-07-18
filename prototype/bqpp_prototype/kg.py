"""Knowledge Gradient (KG) approximation for arm allocation.

The Knowledge Gradient policy (Frazier-Powell-Dayanik 2009) computes
the **value of one more pull** of arm a as

    KG_a = E[max_j mu_j_post] - max_j mu_j_prior

i.e. the expected improvement in the identified-best mean after one
hypothetical observation. For Gaussian-posterior arms, this reduces to
a closed-form involving phi(z) and Phi(z) — the same expected-
improvement integral as :func:`bqpp_prototype.voi.expected_improvement`,
but evaluated for a different "delta" because the comparison is
against the empirical best (not against the arm itself).

For BQ++ Phase 4's KG-per-cost stop rule, we approximate KG_a using
the gap between the empirical best and arm a, scaled by the
standard-error of the difference. This is the same structure as the
EI formula but applied to "value of one more pull" rather than "value
of selecting this arm now."

Top-m approximation:
    For CPU-friendly hot-path, we evaluate KG only on a top-m
    candidate set. The remaining arms are bounded above by
    U_a - L_b (using the empirical-Bernstein widths from
    :mod:`certificate`). This bound is conservative — true KG is
    almost always smaller than U_a - L_b — so any arm bounded below
    the top-m's minimum KG is safe to skip.

Reference:
    Frazier, P. I., Powell, W. B., & Dayanik, S. (2009). "The
    Knowledge-Gradient Policy for Correlated Normal Beliefs."
    INFORMS Journal on Computing 21(4): 599-613.
"""

from __future__ import annotations

import math
from typing import Sequence

from .voi import expected_improvement


def kg_gaussian_per_arm(
    mu_a: float,
    n_a: int,
    sigma2_a: float,
    mu_b: float,
    n_b: int,
    sigma2_b: float,
    lambda0: float = 4.0,
) -> float:
    """Knowledge Gradient for arm `a` against the empirical best `b`.

    Formula (Gaussian posterior, one hypothetical extra pull):
        s_a = sqrt(sigma2_b / (n_b + lambda0) + sigma2_a / (n_a + lambda0))
        Delta_a = max(mu_b - mu_a, 0)
        KG_a = s_a * phi(Delta_a / s_a) - Delta_a * Phi(-Delta_a / s_a)

    Note: at the empirical-best arm (mu_a == mu_b, Delta_a == 0), KG
    reduces to ``s_a / sqrt(2 * pi)`` — non-zero but small. The audit
    convention is that KG of the **identified-best arm** itself is
    set to 0 (no value in pulling the leader; it can't improve
    decision quality further). The :func:`compute_kg_array` wrapper
    applies this convention.

    s_a denominators use ``n + lambda0`` (the Bayesian effective
    sample size) rather than just ``n`` to avoid division by zero at
    n_a = 0 and to match the empirical-Bayes shrinkage in
    :mod:`belief`.
    """
    s2_a = sigma2_a / (n_a + lambda0) if (n_a + lambda0) > 0 else 0.0
    s2_b = sigma2_b / (n_b + lambda0) if (n_b + lambda0) > 0 else 0.0
    s = math.sqrt(s2_b + s2_a)
    delta = max(mu_b - mu_a, 0.0)
    return expected_improvement(delta, s)


def compute_kg_array(
    mu_hats: Sequence[float],
    n_pulls: Sequence[int],
    sigma2s: Sequence[float],
    best_pos: int | None = None,
    lambda0: float = 4.0,
) -> list[float]:
    """Compute KG_a for every arm, with KG[best] = 0 by convention.

    If ``best_pos`` is None, it defaults to ``argmax mu_hats``.
    """
    if not (len(mu_hats) == len(n_pulls) == len(sigma2s)):
        raise ValueError("input sequences must have equal length")
    K = len(mu_hats)
    if K == 0:
        return []
    if best_pos is None:
        best_pos = max(range(K), key=lambda i: mu_hats[i])

    kg = [0.0] * K
    mu_b = mu_hats[best_pos]
    n_b = n_pulls[best_pos]
    sigma2_b = sigma2s[best_pos]

    for a in range(K):
        if a == best_pos:
            kg[a] = 0.0  # convention: leader itself has zero KG
            continue
        kg[a] = kg_gaussian_per_arm(
            mu_a=mu_hats[a],
            n_a=n_pulls[a],
            sigma2_a=sigma2s[a],
            mu_b=mu_b,
            n_b=n_b,
            sigma2_b=sigma2_b,
            lambda0=lambda0,
        )
    return kg


def top_m_kg_with_uc_bound(
    mu_hats: Sequence[float],
    n_pulls: Sequence[int],
    sigma2s: Sequence[float],
    upper_ci: Sequence[float],
    lower_ci_best: float,
    m: int,
    best_pos: int | None = None,
    lambda0: float = 4.0,
) -> tuple[list[float], int]:
    """Compute KG only on top-m candidates; bound the rest by U_a - L_b.

    Returns ``(kg_array, n_evaluated)`` where ``n_evaluated`` is the
    number of arms whose KG was computed exactly (the remaining
    arms have ``kg = max(upper_ci[a] - lower_ci_best, 0)`` — a
    conservative upper bound on their true KG that is "free" to
    compute since the EB bounds are already cached).

    This is the CPU-friendly form used by the Rust hot path.
    """
    if not (len(mu_hats) == len(n_pulls) == len(sigma2s) == len(upper_ci)):
        raise ValueError("input sequences must have equal length")
    K = len(mu_hats)
    if best_pos is None and K > 0:
        best_pos = max(range(K), key=lambda i: mu_hats[i])

    # Pick top-m candidates by upper_ci (the proxy for "this arm could
    # plausibly become the best"). Always include the empirical best
    # so the leader-vs-runner-up comparison stays calibrated.
    m_eff = max(min(m, K), 1)
    sorted_indices = sorted(range(K), key=lambda i: upper_ci[i], reverse=True)
    top_m = set(sorted_indices[:m_eff])
    if best_pos is not None:
        top_m.add(best_pos)

    kg = [0.0] * K
    mu_b = mu_hats[best_pos] if best_pos is not None else 0.0
    n_b = n_pulls[best_pos] if best_pos is not None else 0
    sigma2_b = sigma2s[best_pos] if best_pos is not None else 0.0

    for a in range(K):
        if a == best_pos:
            kg[a] = 0.0
            continue
        if a in top_m:
            kg[a] = kg_gaussian_per_arm(
                mu_hats[a],
                n_pulls[a],
                sigma2s[a],
                mu_b,
                n_b,
                sigma2_b,
                lambda0,
            )
        else:
            # Conservative bound: U_a - L_b is an over-estimate of true KG
            # for arms below the top-m threshold. Used only for "is this
            # arm definitely below the top-m's minimum?" pruning.
            kg[a] = max(upper_ci[a] - lower_ci_best, 0.0)
    return kg, len(top_m)
