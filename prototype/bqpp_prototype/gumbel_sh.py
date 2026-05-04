"""Gumbel without-replacement sampling + Sequential Halving root scheduler.

Reference:
    Danihelka, I., Guez, A., Schrittwieser, J., & Silver, D. (2022).
    "Policy Improvement by Planning with Gumbel." ICLR 2022.

Direct match for the user's primary objective: reduce the number of
NN evals per move while preserving (or improving) play quality. Pure
PUCT at root visits all candidates roughly proportional to prior;
Gumbel SH replaces this with a candidate-set selection (top-m by
``log π̃₀(a) + g_a``, g_a ~ Gumbel(0, 1)) followed by Sequential
Halving over the candidate set.

The sampling-without-replacement equivalence (Yellott 1977,
"The relationship of probabilistic choice models to the
distribution of choices and the Plackett-Luce model"): drawing the
top-m via ``argmax(log p_i + g_i)`` produces samples drawn from the
Plackett-Luce distribution over the prior — i.e. proportional to
prior, but as a *set* rather than independent picks. This is
provably the right thing to do for AlphaZero policy improvement
when the simulation budget is small.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def sample_gumbel(rng: random.Random) -> float:
    """Sample one Gumbel(0, 1) variate via inverse-CDF.

    Formula: ``g = -log(-log(U))`` for U ~ Uniform(0, 1). Standard
    technique; see Maddison-Mnih-Teh 2017 "The Concrete Distribution"
    appendix for derivation.
    """
    u = rng.random()
    # Avoid log(0) at the extreme tails of the uniform.
    u = max(min(u, 1.0 - 1e-12), 1e-12)
    return -math.log(-math.log(u))


def gumbel_top_m(
    log_priors: list[float],
    m: int,
    rng: random.Random,
) -> list[int]:
    """Select top-m indices by ``log π_i + g_i`` (Gumbel-top-m sampling).

    Returns indices of the m largest perturbed log-prior values, in
    descending order. This is the without-replacement Plackett-Luce
    sample (Yellott 1977 equivalence).

    For m = 1 this reduces to ``argmax(log π_i + g_i)`` which is
    a single Gumbel-Max sample of the categorical distribution
    (the textbook reparameterization trick).
    """
    if m <= 0 or len(log_priors) == 0:
        return []
    perturbed = [log_priors[i] + sample_gumbel(rng) for i in range(len(log_priors))]
    sorted_indices = sorted(
        range(len(log_priors)),
        key=lambda i: perturbed[i],
        reverse=True,
    )
    return sorted_indices[: min(m, len(log_priors))]


@dataclass
class SequentialHalvingBracket:
    """Anytime-resumable Sequential Halving bracket state.

    Sequential Halving (Karnin-Koren-Somekh 2013) divides a fixed
    budget B into log_2(m_0) rounds; each round halves the live-set
    by dropping the bottom half by mean-reward. Total visits per
    round = ⌊B / (m_r * log_2(m_0))⌋ where m_r is the live-set size
    in round r.

    Anytime property: the bracket can be paused after any complete
    round and resumed without changing the eventual selection (given
    the same RNG state).
    """

    candidates: list[int]
    budget: int
    n_initial_candidates: int
    rounds_completed: int = 0
    visits_consumed: int = 0
    visit_history: list[int] | None = None  # per-arm cumulative visits

    def __post_init__(self):
        if self.visit_history is None:
            self.visit_history = [0] * len(self.candidates)

    @property
    def n_total_rounds(self) -> int:
        """Number of halving rounds in the bracket."""
        m0 = max(self.n_initial_candidates, 1)
        return max(1, int(math.ceil(math.log2(m0))) if m0 > 1 else 1)

    @property
    def round_budget(self) -> int:
        """Per-arm visits in each round.

        Formula: ⌊B / (m_r * log_2(m_0))⌋ where m_r is the current
        live-set size.
        """
        m_r = len(self.candidates)
        rounds = self.n_total_rounds
        if m_r == 0 or rounds == 0:
            return 0
        per_arm = self.budget // (m_r * rounds)
        return max(per_arm, 1)

    def is_done(self) -> bool:
        return len(self.candidates) <= 1 or self.rounds_completed >= self.n_total_rounds


def initial_bracket(
    log_priors: list[float],
    m_initial: int,
    budget: int,
    rng: random.Random,
) -> SequentialHalvingBracket:
    """Construct the initial SH bracket from a Gumbel-top-m candidate set."""
    candidates = gumbel_top_m(log_priors, m_initial, rng)
    return SequentialHalvingBracket(
        candidates=candidates,
        budget=budget,
        n_initial_candidates=len(candidates),
    )


def advance_round(
    bracket: SequentialHalvingBracket,
    arm_means: list[float],
) -> SequentialHalvingBracket:
    """Advance the SH bracket by one halving round.

    ``arm_means`` is indexed by the original log_prior position
    (not by candidate position within the bracket). The bracket
    drops the bottom half of its current candidates by ``arm_means``.

    Returns a new bracket; does not mutate the input.
    """
    if bracket.is_done():
        return bracket
    candidates = bracket.candidates
    if len(candidates) <= 1:
        return bracket
    # Sort candidates by mean (descending)
    sorted_by_mean = sorted(candidates, key=lambda i: arm_means[i], reverse=True)
    # Keep top half
    keep_n = max(1, len(sorted_by_mean) // 2)
    new_candidates = sorted_by_mean[:keep_n]
    return SequentialHalvingBracket(
        candidates=new_candidates,
        budget=bracket.budget,
        n_initial_candidates=bracket.n_initial_candidates,
        rounds_completed=bracket.rounds_completed + 1,
        visits_consumed=bracket.visits_consumed + bracket.round_budget * len(candidates),
        visit_history=list(bracket.visit_history),
    )


def select_winner(bracket: SequentialHalvingBracket, arm_means: list[float]) -> int:
    """Final selection from a finished bracket: argmax mean over live candidates."""
    if not bracket.candidates:
        raise ValueError("empty bracket; no winner")
    return max(bracket.candidates, key=lambda i: arm_means[i])
