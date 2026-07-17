import numpy as np
import pytest

from quartz.idea_foundry.analysis import (
    B13CurvatureReadoutSkeleton,
    PhysicsFalsifierSkeleton,
    SymmetryOrbitAuditSkeleton,
)
from quartz.idea_foundry.candidates import (
    DynamicLiveSetParticleSkeleton,
    GumbelSequentialHalvingSkeleton,
    ResidualEvidenceWideningSkeleton,
)
from quartz.idea_foundry.contracts import ActionEvidence, RootSnapshot, RuntimeEvidence
from quartz.idea_foundry.decision import StaticAnchorRpoSkeleton, StopCouncilSkeleton
from quartz.idea_foundry.registry import AXIS_SPECS, axis_ids, get_axis_spec
from quartz.idea_foundry.representation import (
    CpuIncrementalStudentSpec,
    DiffusionRegularizedEvaluatorSpec,
    RegretStateArchiveSkeleton,
    RwRestLiteSpec,
)
from quartz.idea_foundry.systems import ServiceCurvePoint, ServiceCurveSchedulerSkeleton


def snapshot(*, hidden=False):
    actions = [
        ActionEvidence(
            edge_pos=0,
            action_id=10,
            visible=True,
            prior_anchor=0.5,
            current_prior=0.5,
            visits=20,
            mean_q=0.6,
            mc_radius=0.02,
            epistemic_radius=0.01,
            cost_ms=0.2,
            policy_signature=(0.7, 0.2, 0.1),
        ),
        ActionEvidence(
            edge_pos=1,
            action_id=11,
            visible=True,
            prior_anchor=0.3,
            current_prior=0.3,
            visits=10,
            mean_q=0.3,
            mc_radius=0.03,
            epistemic_radius=0.01,
            cost_ms=0.2,
            policy_signature=(0.6, 0.3, 0.1),
        ),
        ActionEvidence(
            edge_pos=2,
            action_id=12,
            visible=not hidden,
            prior_anchor=0.2,
            current_prior=0.2,
            visits=0,
            mean_q=0.0,
            upper_hint=0.8,
            mc_radius=0.05,
            epistemic_radius=0.05,
            cost_ms=0.2,
            policy_signature=(0.2, 0.4, 0.4),
        ),
    ]
    return RootSnapshot(
        root_hash="root",
        checkpoint_id="ckpt",
        position_id="pos",
        search_epoch=1,
        root_visits=30,
        iteration=30,
        elapsed_ms=10.0,
        remaining_visits=34,
        actions=tuple(actions),
        runtime=RuntimeEvidence(threads=8, semantic_path_overlap=0.7),
        policy_entropy=0.8,
        effective_branching=2.2,
        top2_margin=0.3,
        margin_slope=0.0,
        entropy_slope=0.0,
        h1_stability=0.99,
        p_flip=0.01,
        candidate_omission_risk=0.0,
        fresh=True,
    )


def test_registry_has_all_axes_and_unique_ids():
    assert len(AXIS_SPECS) >= 16
    assert len(AXIS_SPECS) == 24
    assert len(set(axis_ids())) == len(AXIS_SPECS)
    assert get_axis_spec("A01_stop_council").title.startswith("Calibrated")


def test_snapshot_validates_and_separates_action_id_from_edge_pos():
    snap = snapshot()
    snap.validate()
    assert snap.actions[0].edge_pos == 0
    assert snap.actions[0].action_id == 10


def test_static_anchor_rpo_preserves_probability_mass():
    snap = snapshot(hidden=True)
    policy = StaticAnchorRpoSkeleton().improved_policy(snap.actions)
    assert set(policy) == {10, 11, 12}
    assert sum(policy.values()) == pytest.approx(1.0)
    assert policy[12] == pytest.approx(0.2)


def test_stop_council_emits_stop_only_on_fresh_low_risk_snapshot():
    proposals = StopCouncilSkeleton(max_wrong_risk=0.05).propose(snapshot())
    assert len(proposals) == 1
    assert proposals[0].kind.value == "stop"


def test_residual_widening_emits_hidden_candidate():
    snap = snapshot(hidden=True)
    proposals = ResidualEvidenceWideningSkeleton(max_residual_ratio=0.01).propose(snap)
    assert proposals
    assert proposals[0].kind.value == "widen"
    assert 12 in proposals[0].target_action_ids


def test_gumbel_candidates_are_deterministic_for_same_snapshot():
    op = GumbelSequentialHalvingSkeleton(candidate_count=3)
    assert op.root_candidates(snapshot()) == op.root_candidates(snapshot())


def test_live_set_weights_normalize():
    weights = DynamicLiveSetParticleSkeleton().weights(snapshot(hidden=True))
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) == {10, 11, 12}


def test_service_curve_selects_feasible_throughput_point():
    points = [
        ServiceCurvePoint(32, 1, 2, 800.0, 20.0),
        ServiceCurvePoint(64, 2, 4, 1200.0, 40.0),
        ServiceCurvePoint(128, 4, 8, 1500.0, 140.0),
    ]
    selected = ServiceCurveSchedulerSkeleton(latency_cap_ms=100.0).choose(points)
    assert selected is not None
    assert selected.batch_size == 64


def test_b13_readout_is_normalized_and_decision_neutral_surface():
    readout = B13CurvatureReadoutSkeleton().readout(snapshot())
    assert sum(readout.values()) == pytest.approx(1.0)


def test_physics_beta_fit_reports_residual():
    result = PhysicsFalsifierSkeleton(beta_grid=(0.0, 1.0, 2.0)).fit_effective_beta(
        [0.7, 0.2, 0.1], [1.0, 0.0, -0.5]
    )
    assert result["beta_eff"] in {0.0, 1.0, 2.0}
    assert result["residual_kl"] >= 0.0


def test_symmetry_audit_detects_equivariant_permutation():
    audit = SymmetryOrbitAuditSkeleton().audit_policy(
        [0.1, 0.2, 0.7], [0.7, 0.1, 0.2], [1, 2, 0]
    )
    assert audit["equivariant"] is True


def test_representation_specs_validate():
    DiffusionRegularizedEvaluatorSpec().validate()
    RwRestLiteSpec().validate()
    CpuIncrementalStudentSpec().validate()
    assert RegretStateArchiveSkeleton().uniform_mix > 0.0
