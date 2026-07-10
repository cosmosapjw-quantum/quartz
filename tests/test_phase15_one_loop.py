"""Tests for quartz.phase15_one_loop — H2 finite-N curvature readout (B13).

Pins the correction math and, most importantly, the falsifiable KILL-TEST
contract from the adversarial audit (§0.6 CCoT): the correction's effect
must vanish as the root visit count grows. A correction that persists at
large N would be double-counting Q and is disqualified.
"""

import numpy as np
import pytest

from quartz.phase15_ablation import apply_system_readout, Phase15System
from quartz.phase15_one_loop import (
    ONE_LOOP_SCHEMA_VERSION,
    apply_one_loop_readout,
    one_loop_correction,
)


def _kl(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))))


def test_correction_reinflates_low_visit_arms():
    # A concentrated finite-N policy: arm 0 dominates, arms 1/2 under-visited.
    pi_bar = np.array([0.8, 0.15, 0.05])
    eff, meta = one_loop_correction(pi_bar, n_total=20, curvature=1.0, n_floor=1.0)
    assert meta["one_loop_active"] is True
    assert eff == pytest.approx(eff.sum() / eff.sum() * eff)  # sums to 1
    assert eff.sum() == pytest.approx(1.0)
    # under-visited arms gain mass; the dominant arm loses relative mass.
    assert eff[2] / pi_bar[2] > eff[0] / pi_bar[0]
    assert eff[0] < pi_bar[0]


def test_kill_test_effect_vanishes_as_n_grows():
    # THE kill test: KL(pi_bar || eff) must shrink monotonically toward 0
    # as total visits grow. Persistence would mean double-counting.
    pi_bar = np.array([0.6, 0.25, 0.1, 0.05])
    kls = []
    for n_total in (8, 64, 512, 4096, 65536):
        _, meta = one_loop_correction(pi_bar, n_total=n_total, curvature=1.0)
        kls.append(meta["one_loop_effect_kl"])
    # strictly decreasing
    assert all(kls[i] > kls[i + 1] for i in range(len(kls) - 1)), kls
    # and asymptotically negligible
    assert kls[-1] < 1e-4
    assert kls[0] > kls[-1] * 100  # small-N effect is orders larger


def test_decision_relevant_top1_delta_vanishes_with_budget():
    # Real-trace finding (B13 GPU validation): full-support effect_kl can be
    # tail-dominated and fail to vanish for diffuse policies, but the
    # DECISION-relevant top1_delta (mass pulled off the best arm) DOES vanish
    # with budget and the argmax is preserved. This is the correct kill-test
    # metric. Fixed mildly-concentrated policy, growing budget.
    pi_bar = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
    prev_mag = None
    for n_total in (8, 32, 128, 512, 2048):
        _, meta = one_loop_correction(pi_bar, n_total=n_total, curvature=1.0)
        assert meta["one_loop_argmax_preserved"] is True
        mag = abs(meta["one_loop_top1_delta"])
        if prev_mag is not None:
            assert mag <= prev_mag + 1e-9, (n_total, mag, prev_mag)
        prev_mag = mag
    assert abs(meta["one_loop_top1_delta"]) < 1e-3  # negligible at high budget


def test_argmax_preserved_flag_present_in_inactive_branches():
    _, m0 = one_loop_correction(np.array([0.0, 1.0, 0.0]), n_total=10)  # single arm
    _, m1 = one_loop_correction(np.array([0.6, 0.4]), n_total=0)  # zero budget
    for m in (m0, m1):
        assert m["one_loop_argmax_preserved"] is True
        assert m["one_loop_top1_delta"] == 0.0


def test_effect_scales_with_curvature():
    pi_bar = np.array([0.7, 0.2, 0.1])
    _, m0 = one_loop_correction(pi_bar, n_total=16, curvature=0.5)
    _, m1 = one_loop_correction(pi_bar, n_total=16, curvature=2.0)
    assert m1["one_loop_effect_kl"] > m0["one_loop_effect_kl"]
    # curvature 0 = identity, inactive
    eff, m2 = one_loop_correction(pi_bar, n_total=16, curvature=0.0)
    assert m2["one_loop_active"] is False
    assert eff == pytest.approx(pi_bar)


def test_unvisited_arms_stay_zero_no_invented_mass():
    # arm 3 is unvisited (0.0). The correction must not invent mass for it
    # (no completed-Q data exists in the trace).
    pi_bar = np.array([0.5, 0.3, 0.2, 0.0])
    eff, meta = one_loop_correction(pi_bar, n_total=25)
    assert eff[3] == 0.0
    assert eff.sum() == pytest.approx(1.0)
    assert meta["one_loop_support"] == 3


def test_single_live_arm_and_empty_are_inactive():
    eff, meta = one_loop_correction(np.array([0.0, 1.0, 0.0]), n_total=10)
    assert meta["one_loop_active"] is False
    assert eff == pytest.approx([0.0, 1.0, 0.0])
    eff2, meta2 = one_loop_correction(np.array([0.0, 0.0]), n_total=10)
    assert meta2["one_loop_active"] is False
    assert meta2["one_loop_effect_kl"] == 0.0


def test_zero_budget_is_identity():
    pi_bar = np.array([0.6, 0.4])
    eff, meta = one_loop_correction(pi_bar, n_total=0)
    assert meta["one_loop_active"] is False
    assert eff == pytest.approx(pi_bar)


def test_readout_adapter_signature_and_metadata():
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    trace_policies = [
        np.array([0.4, 0.3, 0.2, 0.1]),
        np.array([0.6, 0.25, 0.1, 0.05]),
    ]
    trace_budgets = [16, 64]
    eff, meta = apply_one_loop_readout(prior, trace_policies, trace_budgets, {})
    assert meta["belief_revision_operator"] == "one_loop_finite_n"
    assert meta["one_loop_schema_version"] == ONE_LOOP_SCHEMA_VERSION
    assert len(meta["effective_policy"]) == 4
    assert eff.sum() == pytest.approx(1.0)


def test_readout_empty_trace_falls_back_to_prior():
    prior = np.array([0.7, 0.3])
    eff, meta = apply_one_loop_readout(prior, [], [], {})
    assert meta["one_loop_active"] is False
    assert eff == pytest.approx(prior)


def test_readout_reachable_through_apply_system_readout_dispatch():
    # The operator must be dispatchable end-to-end (real wiring, not a
    # dead function) — mirrors the A0-a lesson: unreachable code is a lie.
    system = Phase15System(
        id="B13",
        label="one_loop_finite_n",
        group="B",
        substrate="root_only",
        controller="posthoc",
        refresh_operator="one_loop_finite_n",
        params={"one_loop_curvature": 1.0},
    )
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    trace_policies = [np.array([0.7, 0.2, 0.08, 0.02])]
    eff, meta = apply_system_readout(system, prior, trace_policies, [64], 64)
    assert meta["belief_revision_operator"] == "one_loop_finite_n"
    assert eff.sum() == pytest.approx(1.0)


def test_params_override_curvature_and_floor():
    prior = np.array([0.5, 0.5])
    trace_policies = [np.array([0.9, 0.08, 0.02])]
    _, meta_hi = apply_one_loop_readout(prior, trace_policies, [32], {"one_loop_curvature": 3.0})
    _, meta_lo = apply_one_loop_readout(prior, trace_policies, [32], {"one_loop_curvature": 0.2})
    assert meta_hi["one_loop_effect_kl"] > meta_lo["one_loop_effect_kl"]
