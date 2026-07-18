"""Tests for bqpp_prototype.gumbel_sh — Gumbel + Sequential Halving."""

import math
import random

from bqpp_prototype.gumbel_sh import (
    SequentialHalvingBracket,
    advance_round,
    gumbel_top_m,
    initial_bracket,
    sample_gumbel,
    select_winner,
)


def test_sample_gumbel_distribution_mean():
    """Gumbel(0, 1) has mean γ ≈ 0.5772 (Euler-Mascheroni constant)."""
    rng = random.Random(0)
    samples = [sample_gumbel(rng) for _ in range(200_000)]
    mean = sum(samples) / len(samples)
    # Euler-Mascheroni constant ≈ 0.5772
    assert abs(mean - 0.5772) < 0.02, f"sample mean = {mean}"


def test_gumbel_top_m_returns_correct_count():
    """gumbel_top_m returns at most m indices."""
    rng = random.Random(0)
    log_priors = [math.log(p) for p in [0.5, 0.3, 0.1, 0.05, 0.05]]
    out = gumbel_top_m(log_priors, m=3, rng=rng)
    assert len(out) == 3
    assert len(set(out)) == 3  # no duplicates


def test_gumbel_top_m_concentration_on_strong_prior():
    """For a strongly-peaked prior, top-1 picks the mode with high prob."""
    log_priors = [math.log(0.97), math.log(0.01), math.log(0.01), math.log(0.01)]
    rng = random.Random(0)
    n_runs = 1000
    n_correct = 0
    for _ in range(n_runs):
        top1 = gumbel_top_m(log_priors, m=1, rng=rng)
        if top1[0] == 0:
            n_correct += 1
    # Strong prior 0.97 → expected pick rate ~97%
    rate = n_correct / n_runs
    assert rate > 0.93, f"rate = {rate}"


def test_gumbel_top_m_uniform_prior_distributes():
    """Uniform prior ⇒ all arms picked roughly equally as top-1."""
    K = 4
    log_priors = [math.log(1.0 / K)] * K
    rng = random.Random(42)
    counts = [0] * K
    n_runs = 4000
    for _ in range(n_runs):
        top1 = gumbel_top_m(log_priors, m=1, rng=rng)
        counts[top1[0]] += 1
    # each arm should be picked ~25% of the time, allow ±5pp
    for c in counts:
        rate = c / n_runs
        assert 0.20 < rate < 0.30, f"counts = {counts}"


def test_gumbel_top_m_empty_input():
    rng = random.Random(0)
    assert gumbel_top_m([], m=3, rng=rng) == []
    log_priors = [math.log(0.5), math.log(0.5)]
    assert gumbel_top_m(log_priors, m=0, rng=rng) == []


def test_sh_bracket_n_total_rounds():
    """log_2(m_0) rounds for m_0 = 2, 4, 8."""
    for m, expected_rounds in [(2, 1), (4, 2), (8, 3), (5, 3)]:
        bracket = SequentialHalvingBracket(
            candidates=list(range(m)),
            budget=100,
            n_initial_candidates=m,
        )
        assert bracket.n_total_rounds == expected_rounds


def test_sh_advance_halves_candidate_set():
    """Each advance_round drops the bottom half by mean."""
    bracket = SequentialHalvingBracket(
        candidates=[0, 1, 2, 3],
        budget=64,
        n_initial_candidates=4,
    )
    # Means: arm 0 best, arm 3 worst
    arm_means = [0.9, 0.7, 0.5, 0.3]
    nb = advance_round(bracket, arm_means)
    # After round 1: top half = arms 0, 1
    assert sorted(nb.candidates) == [0, 1]
    nb2 = advance_round(nb, arm_means)
    # After round 2: top half = arm 0
    assert nb2.candidates == [0]


def test_sh_select_winner_argmax_mean():
    """final winner is argmax mean over live candidates."""
    bracket = SequentialHalvingBracket(
        candidates=[0, 2],
        budget=64,
        n_initial_candidates=4,
    )
    arm_means = [0.9, 0.5, 0.7, 0.3]
    assert select_winner(bracket, arm_means) == 0


def test_sh_resumable_property():
    """Pause + resume must not change the winner (deterministic given seed)."""
    rng = random.Random(0)
    log_priors = [math.log(0.1)] * 8
    arm_means = [0.5, 0.6, 0.55, 0.7, 0.4, 0.3, 0.65, 0.45]

    # Run 1: full bracket
    b = initial_bracket(log_priors, m_initial=8, budget=64, rng=rng)
    full_b = b
    while not full_b.is_done():
        full_b = advance_round(full_b, arm_means)
    full_winner = select_winner(full_b, arm_means)

    # Run 2: same seed, pause after round 1
    rng2 = random.Random(0)
    b2 = initial_bracket(log_priors, m_initial=8, budget=64, rng=rng2)
    paused = advance_round(b2, arm_means)  # one round
    while not paused.is_done():
        paused = advance_round(paused, arm_means)
    resumed_winner = select_winner(paused, arm_means)

    assert full_winner == resumed_winner


def test_sh_advance_round_keeps_ceiling_half_on_odd_count():
    """A3-c regression: Karnin-Koren-Somekh keeps ceil(m_r/2)
    survivors, not floor(m_r/2) — an odd live-set size (5) must keep
    3, not 2. budget=100 is generous enough that shrink-to-affordable
    is a no-op here, isolating the keep_n formula itself."""
    bracket = SequentialHalvingBracket(
        candidates=[0, 1, 2, 3, 4],
        budget=100,
        n_initial_candidates=5,
    )
    arm_means = [0.9, 0.7, 0.5, 0.3, 0.1]
    nb = advance_round(bracket, arm_means)
    assert len(nb.candidates) == 3, (
        f"5 live arms must keep ceil(5/2)=3, got {nb.candidates}"
    )
    assert sorted(nb.candidates) == [0, 1, 2]


def test_sh_new_shrinks_initial_candidates_to_avoid_budget_overshoot():
    """A3-c regression: the whole point of _shrink_to_affordable. At
    budget=8 with 8 initial candidates (3 rounds), the unshrunk
    bracket's round_budget would floor to 0 and get forced to 1 by
    the max(...,1) safety net every round — consuming
    8+4+2=14 visits against a declared budget of 8. After the fix,
    the initial candidate count is shrunk so visits_consumed never
    exceeds the declared budget."""
    bracket = SequentialHalvingBracket(
        candidates=list(range(8)), budget=8, n_initial_candidates=8
    )
    assert len(bracket.candidates) < 8, (
        f"8 candidates at budget=8 must be shrunk, got {bracket.candidates}"
    )
    assert bracket.candidates[0] == 0  # highest-scoring prefix preserved

    arm_means = [0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01]
    current = bracket
    while not current.is_done():
        current = advance_round(current, arm_means)
    assert current.visits_consumed <= 8, (
        f"must not exceed declared budget, got {current.visits_consumed}"
    )


def test_sh_new_does_not_shrink_when_budget_is_affordable():
    """A budget that already comfortably affords the requested
    candidate count must NOT be shrunk."""
    bracket = SequentialHalvingBracket(
        candidates=list(range(8)), budget=100, n_initial_candidates=8
    )
    assert len(bracket.candidates) == 8
    assert bracket.n_initial_candidates == 8
