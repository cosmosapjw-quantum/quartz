"""End-to-end BQ++ controller on a synthetic bandit.

Ties together the modules in this directory to validate that the
single-principle objective ("expected decision-loss reduction per
compute cost; halt on certificate or VOI < cost") composes correctly
on a known-ground-truth bandit fixture.

This is **not** a port of the actual MCTS engine — it skips PUCT,
virtual loss, NN evaluation, and tree expansion. Per the audit's
recommendation (Phase 1: Python prototype) the goal is math
validation, not engine integration. The actual controller lives
in Rust under ``src/mcts/policy/`` (BQ++ Phase 2+).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .belief import WelfordState, empirical_bayes_shrinkage
from .certificate import (
    EBCertificate,
    EBInterval,
    best_vs_runner_certificate,
    eb_interval_from_arm,
)
from .kg import compute_kg_array
from .synthetic import GaussianBandit


@dataclass
class HaltDecision:
    """Decision returned by :func:`maybe_halt`."""

    halt: bool
    reason: str
    iteration: int


@dataclass
class ControllerRun:
    """Results of running the BQ++ prototype controller end-to-end."""

    halted_at: int
    halt_reason: str
    pulls_per_arm: list[int]
    means_per_arm: list[float]
    cert_history: list[float]   # gap_bits at each check
    selected_arm: int


def _maybe_halt(
    intervals: list[EBInterval],
    kg: list[float],
    iteration: int,
    min_total: int,
    min_pulls_per_arm: int,
    pulls_per_arm: list[int],
    cost_per_pull: float,
    kg_threshold: float,
) -> HaltDecision:
    """Composite halt rule: certificate OR KG < cost.

    Returns ``halt=False`` until ``iteration >= min_total`` and at
    least 2 arms have ``min_pulls_per_arm`` pulls.

    Then:
    - If the EB certificate fires (gap > 0): halt with reason
      ``EmpBernsteinCertified``.
    - If the maximum KG across challenger arms is below the
      ``kg_threshold * cost_per_pull`` floor: halt with reason
      ``PolicyConverged`` (no remaining computation has positive
      expected improvement per cost).
    - Otherwise continue.
    """
    if iteration < min_total:
        return HaltDecision(False, "BelowMinTotal", iteration)
    armed_count = sum(1 for n in pulls_per_arm if n >= min_pulls_per_arm)
    if armed_count < 2:
        return HaltDecision(False, "InsufficientArms", iteration)

    cert = best_vs_runner_certificate(intervals)
    if cert.fired:
        return HaltDecision(True, "EmpBernsteinCertified", iteration)

    max_kg = max(kg)
    if max_kg < kg_threshold * cost_per_pull:
        return HaltDecision(True, "PolicyConverged", iteration)

    return HaltDecision(False, "Continue", iteration)


def _pick_next_arm(
    kg: list[float],
    pulls_per_arm: list[int],
    mu_hats: list[float],
) -> int:
    """LUCB-style allocation: alternate between empirical-best and the
    arm with maximum KG (the runner-up).

    Round-robin first until every arm has at least one pull. Then on
    each step:
        - if pulls_best < pulls_runner_kg, pull best
        - otherwise pull argmax KG

    This keeps the empirical-best arm well-pulled (so its lower CI
    tightens), which is necessary for the certificate to fire. A pure
    "argmax KG" allocation never re-pulls the best arm (KG[best] = 0
    by convention), and the certificate never fires.

    Real BQ++ uses Gumbel SH (see :mod:`gumbel_sh`) for the candidate
    set; the prototype's allocation rule is deliberately simple to
    keep this controller test focused on the halt behavior rather
    than the allocation behavior.
    """
    for a, n in enumerate(pulls_per_arm):
        if n == 0:
            return a
    best_pos = max(range(len(mu_hats)), key=lambda i: mu_hats[i])
    # Runner-up by KG, EXCLUDING the empirical best (whose KG is 0 by
    # convention but whose argmax-tie at 0 would otherwise pollute the
    # selection). When all non-best KGs are ~0 the runner-up falls back
    # to "the non-best arm with fewest pulls" — keeps round-robin
    # progress instead of stalling on the best arm.
    non_best = [i for i in range(len(kg)) if i != best_pos]
    if not non_best:
        return best_pos
    runner_kg_pos = max(non_best, key=lambda i: (kg[i], -pulls_per_arm[i]))
    if pulls_per_arm[best_pos] <= pulls_per_arm[runner_kg_pos]:
        return best_pos
    return runner_kg_pos


def run_controller(
    bandit: GaussianBandit,
    *,
    delta: float = 0.05,
    lambda0: float = 4.0,
    min_total: int = 100,
    min_pulls_per_arm: int = 30,
    max_iters: int = 5000,
    cost_per_pull: float = 1.0,
    kg_threshold: float = 1e-3,
    sigma2_parent: float = 0.0625,
) -> ControllerRun:
    """Run the BQ++ prototype on a synthetic bandit until halt.

    Returns the full ControllerRun history. The expected behavior:
    - On clear-lead bandits, the EB certificate fires within a few
      hundred pulls.
    - On tight-gap bandits, the KG-stop fires later and the chosen
      arm matches the true best with high probability (≥ 90% on
      the test fixture).
    """
    K = bandit.K
    welford = [WelfordState() for _ in range(K)]
    cert_history: list[float] = []

    for iteration in range(max_iters):
        # Build per-arm intervals on the [0, 1] scale
        mu_hats: list[float] = []
        sigma2s: list[float] = []
        n_pulls: list[int] = []
        intervals: list[EBInterval] = []

        for a in range(K):
            n = welford[a].n
            # The synthetic bandit emits values directly on the [0, 1]
            # scale (true_means in [0, 1]) so we use the empirical mean
            # as-is. The map_q_to_unit utility is only for the real
            # MCTS engine where backed-up Q values are in [-1, 1]; the
            # prototype's GaussianBandit fixture is already on the
            # canonical scale per the audit §6.1 recommendation.
            mu_unit = welford[a].mean if n > 0 else 0.5
            sigma2 = empirical_bayes_shrinkage(
                n=n, M2=welford[a].M2,
                sigma2_parent=sigma2_parent, lambda0=lambda0,
            )
            mu_hats.append(mu_unit)
            sigma2s.append(sigma2)
            n_pulls.append(n)
            intervals.append(
                eb_interval_from_arm(
                    mu_hat=mu_unit, n=n, sigma2=sigma2,
                    K=K, t=max(iteration, 1), delta=delta,
                )
            )

        # KG per arm
        kg = compute_kg_array(
            mu_hats=mu_hats, n_pulls=n_pulls, sigma2s=sigma2s,
            lambda0=lambda0,
        )

        # Halt check
        decision = _maybe_halt(
            intervals=intervals, kg=kg, iteration=iteration,
            min_total=min_total, min_pulls_per_arm=min_pulls_per_arm,
            pulls_per_arm=n_pulls, cost_per_pull=cost_per_pull,
            kg_threshold=kg_threshold,
        )
        if iteration >= min_total:
            try:
                cert = best_vs_runner_certificate(intervals)
                cert_history.append(cert.gap)
            except ValueError:
                pass
        if decision.halt:
            best_pos = max(range(K), key=lambda i: mu_hats[i])
            return ControllerRun(
                halted_at=iteration,
                halt_reason=decision.reason,
                pulls_per_arm=n_pulls,
                means_per_arm=mu_hats,
                cert_history=cert_history,
                selected_arm=best_pos,
            )

        # Pick next arm and pull
        arm = _pick_next_arm(kg, n_pulls, mu_hats)
        x = bandit.pull(arm)
        welford[arm].update(x)

    # Reached max_iters
    mu_hats_final = [w.mean if w.n > 0 else 0.5 for w in welford]
    best_pos = max(range(K), key=lambda i: mu_hats_final[i])
    return ControllerRun(
        halted_at=max_iters,
        halt_reason="MaxIters",
        pulls_per_arm=[w.n for w in welford],
        means_per_arm=mu_hats_final,
        cert_history=cert_history,
        selected_arm=best_pos,
    )
