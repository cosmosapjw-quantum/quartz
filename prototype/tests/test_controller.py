"""Tests for bqpp_prototype.controller — end-to-end on synthetic bandits."""

from bqpp_prototype.controller import run_controller
from bqpp_prototype.synthetic import (
    make_clear_lead_bandit,
    make_hidden_best_bandit,
    make_tight_gap_bandit,
)


def test_clear_lead_halts_quickly():
    """3-arm clear-lead bandit halts well within max_iters with the
    correct best arm. Either halt reason is valid for clear-lead;
    the qualitative property is "halts AND picks the right arm."
    """
    bandit = make_clear_lead_bandit(seed=42)
    run = run_controller(
        bandit,
        delta=0.05,
        lambda0=4.0,
        min_total=100,
        min_pulls_per_arm=30,
        max_iters=2000,
    )
    assert run.halt_reason in {"EmpBernsteinCertified", "PolicyConverged"}, (
        f"unexpected halt reason: {run.halt_reason} at iter {run.halted_at}"
    )
    assert run.selected_arm == bandit.best_arm
    assert run.halted_at < 2000


def test_clear_lead_certificate_fires_with_higher_threshold():
    """Force the EB certificate path by setting kg_threshold = 0.

    With kg_threshold=0, the KG-stop never fires; only the certificate
    can halt the controller. This test verifies the certificate path
    is reachable and works on a clear-lead bandit.
    """
    bandit = make_clear_lead_bandit(seed=42)
    run = run_controller(
        bandit,
        delta=0.05,
        lambda0=4.0,
        min_total=100,
        min_pulls_per_arm=30,
        max_iters=5000,
        kg_threshold=0.0,  # disable KG-stop; certificate must fire
    )
    assert run.halt_reason == "EmpBernsteinCertified", (
        f"unexpected halt reason: {run.halt_reason} at iter {run.halted_at}"
    )
    assert run.selected_arm == bandit.best_arm


def test_tight_gap_can_halt_via_kg():
    """Tight-gap bandit may run out the KG budget without certificate.

    Either the certificate fires (if luck favors), or the KG-stop
    fires when no more computation has positive expected improvement.
    Either way, the chosen arm should match the true best with
    reasonable probability.
    """
    n_correct = 0
    for seed in range(10):
        bandit = make_tight_gap_bandit(seed=seed)
        run = run_controller(
            bandit,
            delta=0.05,
            lambda0=4.0,
            min_total=100,
            min_pulls_per_arm=30,
            max_iters=2000,
        )
        # either reason is acceptable
        assert run.halt_reason in {
            "EmpBernsteinCertified",
            "PolicyConverged",
            "MaxIters",
        }
        if run.selected_arm == bandit.best_arm:
            n_correct += 1
    # tight gap: ≥ 50% correct over 10 seeds.
    assert n_correct >= 5, f"tight-gap fixture: {n_correct}/10 correct"


def test_halt_reasons_are_canonical():
    """Every halt reason is one of the canonical set."""
    bandit = make_clear_lead_bandit(seed=0)
    run = run_controller(bandit, max_iters=2000)
    assert run.halt_reason in {
        "EmpBernsteinCertified",
        "PolicyConverged",
        "MaxIters",
    }


def test_pulls_per_arm_recorded():
    """The controller emits per-arm pull counts in its result."""
    bandit = make_clear_lead_bandit(seed=0)
    run = run_controller(bandit, max_iters=500)
    assert len(run.pulls_per_arm) == bandit.K
    assert sum(run.pulls_per_arm) >= 100  # at least min_total iters


def test_cert_history_monotone_for_clear_lead():
    """For a clear-lead bandit the certificate gap should *eventually*
    become positive. We don't enforce strict monotonicity (Brownian
    fluctuations on early visits) but the final value should be > 0
    if the run halted via certificate."""
    bandit = make_clear_lead_bandit(seed=42)
    run = run_controller(bandit, max_iters=2000)
    if run.halt_reason == "EmpBernsteinCertified":
        # last gap value should be > 0
        assert run.cert_history[-1] > 0
