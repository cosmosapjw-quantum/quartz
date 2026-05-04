"""Synthetic bandit + tree fixtures for prototype validation.

Light-weight fixtures used by ``controller.py`` and the integration
tests. Not a substitute for the real MCTS engine — these are pure
Python, no PUCT, no virtual loss, no NN — they exist to validate that
BQ++'s numerical primitives compose correctly under known ground
truth.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class GaussianBandit:
    """K-armed Gaussian bandit with known true means and variances.

    Each arm's reward distribution is N(true_mean, true_sigma^2).
    The :func:`pull` method draws one observation; :func:`pull_n`
    is a vectorized version.

    Reward range is documented but **not enforced** at the boundary —
    the controller sees raw values and is responsible for clamping if
    needed (the EB width formula has an R parameter for the range).
    """

    true_means: list[float]
    true_sigmas: list[float]
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if len(self.true_means) != len(self.true_sigmas):
            raise ValueError(
                f"true_means and true_sigmas must have equal length; "
                f"got {len(self.true_means)} vs {len(self.true_sigmas)}"
            )

    @property
    def K(self) -> int:
        return len(self.true_means)

    @property
    def best_arm(self) -> int:
        """Ground-truth best arm by true mean (ties broken by lowest index)."""
        return max(range(self.K), key=lambda i: self.true_means[i])

    def pull(self, arm: int) -> float:
        """Single sample from arm ``arm``."""
        return self.rng.gauss(self.true_means[arm], self.true_sigmas[arm])

    def pull_n(self, arm: int, n: int) -> list[float]:
        return [self.pull(arm) for _ in range(n)]


def make_clear_lead_bandit(seed: int = 0) -> GaussianBandit:
    """3-arm fixture where arm 0 dominates (large gap, small noise)."""
    return GaussianBandit(
        true_means=[0.8, 0.4, 0.3],
        true_sigmas=[0.05, 0.05, 0.05],
        rng=random.Random(seed),
    )


def make_tight_gap_bandit(seed: int = 0) -> GaussianBandit:
    """3-arm fixture where the top two arms are nearly tied (hard problem)."""
    return GaussianBandit(
        true_means=[0.55, 0.5, 0.3],
        true_sigmas=[0.1, 0.1, 0.1],
        rng=random.Random(seed),
    )


def make_hidden_best_bandit(seed: int = 0, K: int = 20) -> GaussianBandit:
    """K-arm fixture where the prior favors arms 0-2 but arm K-1 is true-best.

    Used by the nested-reservoir escape regression test
    (test_phase6_reservoir_recovers_low_prior_best). The prior is
    intentionally misaligned with the true means to create the
    "hidden win" scenario.
    """
    means = [0.4 + 0.01 * i for i in range(K)]  # rising, but small
    means[K - 1] = 0.85                         # arm K-1 is true best
    sigmas = [0.1] * K
    return GaussianBandit(
        true_means=means,
        true_sigmas=sigmas,
        rng=random.Random(seed),
    )
