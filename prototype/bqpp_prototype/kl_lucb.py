"""Kaufmann-Kalyanakrishnan 2013 KL-LUCB stopping rule.

Reference implementation matching the Rust ``src/mcts/policy/kl_helpers.rs``
(commit ``3370f95``). Used for cross-language regression: the Python and
Rust ``kl_lucb_beta`` must agree to f32 precision.

Reference:
    Kaufmann, E. & Kalyanakrishnan, S. (2013). "Information Complexity
    in Bandit Subset Selection." COLT 2013, Theorem 8.

PAC scope (audit §1.5): the δ-PAC guarantee assumes iid Bernoulli
samples per arm. For NN-driven value backups in AlphaZero MCTS, this
assumption is violated by shared subtree backups, virtual loss
coupling, and value bias. Use the empirical-Bernstein certificate
(:mod:`certificate`) as the primary halt rule for NN backups; reserve
KL-LUCB for terminal Bernoulli win/loss backups.
"""

from __future__ import annotations

import math


def bernoulli_kl(p: float, q: float) -> float:
    """Bernoulli KL divergence d(p ‖ q), clamped for stability."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    q = min(max(q, 1e-6), 1.0 - 1e-6)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def kl_upper(mu: float, n: int, beta: float) -> float:
    """Solve sup{q ∈ [mu, 1) : n * KL(mu, q) ≤ beta} via 32-iter bisection."""
    if n <= 0:
        return 1.0 - 1e-6
    lo, hi = mu, 1.0 - 1e-6
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if n * bernoulli_kl(mu, mid) > beta:
            hi = mid
        else:
            lo = mid
    return hi


def kl_lower(mu: float, n: int, beta: float) -> float:
    """Solve inf{q ∈ [0, mu] : n * KL(mu, q) ≤ beta} via 32-iter bisection."""
    if n <= 0:
        return 1e-6
    lo, hi = 1e-6, mu
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if n * bernoulli_kl(mu, mid) > beta:
            lo = mid
        else:
            hi = mid
    return lo


def kl_lucb_beta(t: float, K: float, delta: float, alpha: float = 1.1) -> float:
    """KK13 Theorem 8 stopping threshold.

    β(t, δ) = log(k₁ · K · t^α / δ),  k₁ = 405.5, α = 1.1

    Hand sanity: β(151, 3, 0.05) ≈ 15.618 (regression test against
    the Rust impl in commit ``3370f95``).
    """
    k1 = 405.5
    return math.log(k1 * K * (t ** alpha) / delta)


def kl_lucb_gap_bits(
    mu_hats: list[float],
    n_pulls: list[int],
    K: int,
    t: float,
    delta: float = 0.05,
    min_pulls: int = 30,
) -> tuple[float, int, int]:
    """Compute KL-LUCB stopping gap_bits from empirical statistics.

    Inputs are on the [0, 1] Bernoulli scale (caller maps Q via
    ``mu = (Q + 1) / 2``).

    Returns ``(gap_bits, best_pos, runner_up_pos)``. Stop is allowed
    when ``gap_bits > 0``. ``runner_up_pos`` returns ``best_pos`` when
    fewer than 2 arms have ``min_pulls`` (i.e. the comparison is
    ill-defined; caller should treat this as "continue").
    """
    n_arms = len(mu_hats)
    if n_arms < 2 or K < 2:
        return -1.0, 0, 0

    beta = kl_lucb_beta(t, K, delta)

    # Empirical best (restricted to arms with sufficient pulls)
    best_pos = -1
    best_mu = -1.0
    for i in range(n_arms):
        if n_pulls[i] >= min_pulls and mu_hats[i] > best_mu:
            best_mu = mu_hats[i]
            best_pos = i
    if best_pos < 0:
        return -1.0, 0, 0

    # Runner-up by upper bound
    runner_up_pos = best_pos
    second_ucb = -1.0
    for i in range(n_arms):
        if i == best_pos or n_pulls[i] < min_pulls:
            continue
        u = kl_upper(mu_hats[i], n_pulls[i], beta)
        if u > second_ucb:
            second_ucb = u
            runner_up_pos = i
    if runner_up_pos == best_pos:
        return -1.0, best_pos, best_pos

    # gap_bits = L_best - U_runner
    n_best = n_pulls[best_pos]
    l_best = kl_lower(best_mu, n_best, beta)
    return l_best - second_ucb, best_pos, runner_up_pos
