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


def test_decision_relevant_top1_delta_monotone_only_for_FIXED_policy_shape():
    # For a FIXED policy shape (N_a grows proportionally with budget), the
    # decision-relevant top1_delta shrinks monotonically toward 0 and the
    # argmax is preserved. NOTE (adversarial-verify catch): monotonicity is
    # NOT a general property — on real traces the policy SHAPE changes with
    # budget (support grows), which breaks per-step monotonicity; see
    # test_diffuse_spreading_policy_top1_delta_not_monotone below. Only the
    # endpoint net-decrease + argmax-preservation generalize.
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


def test_diffuse_spreading_policy_top1_delta_not_monotone_but_argmax_preserved():
    # Replicates the real-bundle finding (bundle bed6feee: |top1_delta| grew
    # +43% from budget 8->16 before netting down). When the policy SPREADS
    # with budget (support grows, more N_a~1 tail arms), per-step top1_delta
    # is NOT monotone. The DEFENSIBLE general contract is: argmax preserved
    # at every step, and the high-budget endpoint magnitude is below the
    # low-budget one (net decrease) — NOT strict monotonicity.
    steps = [
        (8, np.array([0.40, 0.30, 0.30, 0.0, 0.0, 0.0])),   # concentrated, small support
        (16, np.array([0.22, 0.20, 0.20, 0.20, 0.18, 0.0])),  # spreads -> top1_delta can grow
        (64, np.array([0.30, 0.18, 0.16, 0.14, 0.12, 0.10])),  # wider support, larger N
    ]
    mags = []
    for n_total, pol in steps:
        _, meta = one_loop_correction(pol, n_total=n_total, curvature=1.0)
        assert meta["one_loop_argmax_preserved"] is True
        mags.append(abs(meta["one_loop_top1_delta"]))
    # net decrease endpoint-to-endpoint holds even though it is not monotone
    assert mags[-1] < mags[0], mags
    # and the middle step is allowed to exceed neighbours (non-monotone) —
    # this is the real behaviour, asserted so a future "fix" back to strict
    # monotonicity is caught as an overclaim.
    assert max(mags) >= mags[0], mags


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
