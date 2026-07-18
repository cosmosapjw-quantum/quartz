"""Tests for bqpp_prototype.kl_lucb — KK13 reference matching Rust kl_helpers."""

import math

from bqpp_prototype.kl_lucb import (
    bernoulli_kl,
    kl_lower,
    kl_lucb_beta,
    kl_lucb_gap_bits,
    kl_upper,
)


def test_bernoulli_kl_zero_on_diagonal():
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert abs(bernoulli_kl(p, p)) < 1e-5


def test_bernoulli_kl_non_negative():
    for p in [0.1, 0.5, 0.9]:
        for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
            kl = bernoulli_kl(p, q)
            assert kl >= -1e-5


def test_kl_lucb_beta_kk13_sanity():
    """β(151, 3, 0.05) ≈ 15.618 — matches Rust kl_helpers test
    (commit 3370f95 in Rust; that test was the audit-corrected
    hand-derivation of the audit's §1 setup)."""
    beta = kl_lucb_beta(151, 3, 0.05)
    assert abs(beta - 15.618) < 0.05


def test_kl_upper_inverts_bisection():
    mu, n, beta = 0.6, 100, 5.0
    q = kl_upper(mu, n, beta)
    target = n * bernoulli_kl(mu, q)
    assert abs(target - beta) < 0.1
    assert q > mu


def test_kl_lower_inverts_bisection():
    mu, n, beta = 0.6, 100, 5.0
    q = kl_lower(mu, n, beta)
    target = n * bernoulli_kl(mu, q)
    assert abs(target - beta) < 0.1
    assert q < mu


def test_kl_lucb_gap_tight_does_not_fire():
    """Audit-aligned tight-gap test: N=[100,50,1], Q→μ=[0.8,0.75,0.7].
    min_pulls=30 still gates the best-arm side (arm 0 wins). A1-a: the
    runner-up side no longer filters by min_pulls, so arm 2 (n=1) now
    correctly wins the runner-up race — its 1-pull upper bound is
    wider than arm 1's (n=50) — and gap stays negative."""
    mu_hats = [0.8, 0.75, 0.7]
    n_pulls = [100, 50, 1]
    gap, best, runner = kl_lucb_gap_bits(
        mu_hats,
        n_pulls,
        K=3,
        t=151,
        delta=0.05,
        min_pulls=30,
    )
    assert best == 0
    assert runner == 2
    assert gap < 0.0, f"expected negative gap, got {gap}"


def test_kl_lucb_gap_wide_fires():
    """Audit-aligned wide-gap test: huge separation between two
    adequately-sampled arms ⇒ gap > 0. See
    test_kl_lucb_gap_underpulled_arm_blocks_wide_gap_fire below for
    the case where a third, barely-visited candidate is live."""
    mu_hats = [0.95, 0.5]  # Q=[0.9, 0.0] mapped
    n_pulls = [10000, 500]
    gap, best, runner = kl_lucb_gap_bits(
        mu_hats,
        n_pulls,
        K=3,
        t=10501,
        delta=0.05,
        min_pulls=30,
    )
    assert best == 0
    assert runner == 1
    assert gap > 0.0, f"expected positive gap, got {gap}"


def test_kl_lucb_gap_underpulled_arm_blocks_wide_gap_fire():
    """A1-a regression: the whole point of removing min_pulls from the
    runner-up side. Same wide, well-resolved gap as
    test_kl_lucb_gap_wide_fires, plus a third candidate live at only 1
    pull. Before the fix that arm was silently excluded from the
    runner-up race and the cert fired anyway (anti-conservative — the
    barely-sampled arm was never actually ruled out). After the fix,
    its near-1.0 upper bound correctly blocks the stop."""
    mu_hats = [0.95, 0.5, 0.25]  # Q=[0.9, 0.0, -0.5] mapped
    n_pulls = [10000, 500, 1]
    gap, best, runner = kl_lucb_gap_bits(
        mu_hats,
        n_pulls,
        K=3,
        t=10501,
        delta=0.05,
        min_pulls=30,
    )
    assert best == 0
    assert runner == 2, "the under-sampled arm must win the runner-up slot"
    assert gap < 0.0, (
        f"an unresolved 1-pull candidate must block certification, got {gap}"
    )


def test_kl_lucb_gap_decreases_with_t_at_fixed_n():
    """Audit §1.7 regression: β grows with t, so a stale cache from
    earlier t can show a positive gap that is no longer valid at
    the current t.

    Hold n fixed at the wide-gap configuration and advance t. Gap
    must monotonically decrease.
    """
    mu_hats = [0.95, 0.5]
    n_pulls = [10000, 500]
    gaps = []
    for t in [10501, 50000, 200000, 1_000_000]:
        gap, _, _ = kl_lucb_gap_bits(mu_hats, n_pulls, K=2, t=t, delta=0.05)
        gaps.append(gap)
    for a, b in zip(gaps, gaps[1:]):
        assert b <= a + 1e-9, f"non-monotone in t: {gaps}"


def test_kl_lucb_gap_runner_up_ignores_min_pulls():
    """A1-a regression: arms below min_pulls must still participate in
    the runner-up race (only the best-arm side is gated by
    min_pulls). With two tied under-pulled arms (n=5 each), the first
    one seen wins the tie and its wide bound keeps gap negative —
    this used to return the ill-defined runner==best sentinel because
    BOTH non-best arms were wrongly excluded from the comparison."""
    mu_hats = [0.8, 0.5, 0.5]
    n_pulls = [100, 5, 5]
    gap, best, runner = kl_lucb_gap_bits(
        mu_hats,
        n_pulls,
        K=3,
        t=110,
        delta=0.05,
        min_pulls=30,
    )
    assert best == 0
    assert runner != best, "an under-pulled arm must now win the runner-up slot"
    assert gap < 0.0, f"expected negative gap, got {gap}"


def test_kl_lucb_gap_single_arm_is_invalid():
    """The only remaining ill-defined case: fewer than 2 candidate
    arms at all (the n_arms < 2 guard), not merely under-pulled ones."""
    gap, best, runner = kl_lucb_gap_bits(
        [0.8],
        [100],
        K=1,
        t=110,
        delta=0.05,
        min_pulls=30,
    )
    assert runner == best  # invalid comparison sentinel
