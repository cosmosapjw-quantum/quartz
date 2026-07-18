import json
from pathlib import Path

import pytest

from quartz.idea_foundry import (
    ALL_AXIS_TYPES,
    A01StopCouncil,
    A02StaticAnchorRPO,
    A03UncertaintyDecomposition,
    A07ResidualEvidenceWidening,
    ConservativeArbiter,
    CostVector,
    EdgeObservation,
    MetaActionKind,
    RootObservation,
    RuntimeObservation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "configs" / "idea_foundry.axes.v1.json"


def observation() -> RootObservation:
    edges = (
        EdgeObservation(
            edge_pos=0,
            action_id=10,
            visible=True,
            prior_anchor=0.6,
            prior_current=0.6,
            visits=16,
            virtual_visits=0,
            pending=0,
            q_mean=0.4,
            q_sum=6.4,
            m2=0.8,
            mc_radius=0.04,
            epistemic_radius=0.02,
            drift_radius=0.01,
            bias_radius=0.01,
            lower=0.32,
            upper=0.48,
        ),
        EdgeObservation(
            edge_pos=1,
            action_id=481,
            visible=True,
            prior_anchor=0.3,
            prior_current=0.3,
            visits=8,
            virtual_visits=1,
            pending=1,
            q_mean=0.2,
            q_sum=1.6,
            m2=0.6,
            mc_radius=0.08,
            epistemic_radius=0.03,
            drift_radius=0.02,
            bias_radius=0.01,
            lower=0.06,
            upper=0.34,
        ),
        EdgeObservation(
            edge_pos=2,
            action_id=999,
            visible=False,
            prior_anchor=0.1,
            prior_current=0.1,
            visits=0,
            virtual_visits=0,
            pending=0,
            q_mean=0.0,
            q_sum=0.0,
            m2=0.0,
            mc_radius=0.2,
            epistemic_radius=0.1,
            drift_radius=0.0,
            bias_radius=0.05,
            lower=-0.35,
            upper=0.35,
        ),
    )
    return RootObservation(
        root_hash=123,
        checkpoint_id="seed_1/gen_1",
        position_id="p1",
        game="gomoku7",
        root_visits=24,
        iteration=25,
        elapsed_ms=12,
        remaining_visits=40,
        n_children=3,
        n_visible=2,
        entropy=0.9,
        effective_branching=2.4,
        top2_margin=0.2,
        margin_slope=-0.01,
        entropy_slope=0.02,
        h1_stability=0.99,
        p_flip=0.01,
        prior_visit_js=0.2,
        candidate_omission_bound=0.01,
        revision_count=0,
        edges=edges,
        runtime=RuntimeObservation(threads=8, batch_size=32, inflight=4),
    )


def test_registry_has_all_26_axes_and_valid_skeleton_paths():
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    axes = payload["axes"]
    assert len(axes) == 26
    assert [row["id"] for row in axes] == [f"A{i:02d}" for i in range(1, 27)]
    for row in axes:
        assert (REPO_ROOT / row["documentation"]).exists()
        assert (REPO_ROOT / row["rust_skeleton"]["module"]).exists()
        assert (REPO_ROOT / row["python_skeleton"]["module"]).exists()


def test_python_axis_catalog_is_complete_and_unique():
    assert len(ALL_AXIS_TYPES) == 26
    instances = [axis_type() for axis_type in ALL_AXIS_TYPES]
    ids = [item.axis_id for item in instances]
    assert len(set(ids)) == 26
    assert {axis_id.split(".", 1)[0] for axis_id in ids} == {
        f"A{i:02d}" for i in range(1, 27)
    }


def test_all_python_skeletons_accept_one_observation():
    obs = observation()
    for axis_type in ALL_AXIS_TYPES:
        axis = axis_type()
        proposals = axis.propose(obs)
        assert proposals is not None, axis.axis_id
        for proposal in proposals:
            assert proposal.axis_id == axis.axis_id


def test_static_anchor_rpo_is_normalized_and_does_not_use_action_id_as_index():
    policy = A02StaticAnchorRPO().solve(observation().edges)
    assert set(policy) == {0, 1, 2}
    assert sum(policy.values()) == pytest.approx(1.0)
    assert 481 not in policy


def test_uncertainty_default_is_conservative_sum():
    edge = observation().edges[0]
    axis = A03UncertaintyDecomposition()
    assert axis.radius(edge) == pytest.approx(0.08)
    lo, hi = axis.bounds(edge)
    assert lo == pytest.approx(0.32)
    assert hi == pytest.approx(0.48)


def test_stop_council_emits_stop_only_for_low_combined_risk():
    proposals = A01StopCouncil(risk_limit=0.05).propose(observation())
    assert len(proposals) == 1
    assert proposals[0].action.kind is MetaActionKind.STOP


def test_residual_widening_emits_explicit_widen_action_when_bound_is_large():
    proposals = A07ResidualEvidenceWidening(max_tail_mass=0.0).propose(observation())
    assert proposals
    assert proposals[0].action.kind is MetaActionKind.WIDEN


def test_arbiter_rejects_negative_non_stop_net_value():
    obs = observation()
    proposal = list(A07ResidualEvidenceWidening(max_tail_mass=0.0).propose(obs))[0]
    choice = ConservativeArbiter().choose(
        [proposal],
        {"nn_eval": 100.0, "cpu_ms": 0.0, "gpu_ms": 0.0, "energy_proxy": 0.0},
    )
    assert choice is None


def test_cost_vector_weighting_is_explicit():
    cost = CostVector(nn_evals=2, cpu_ms=3, gpu_ms=4, energy_proxy=5)
    assert (
        cost.weighted({"nn_evals": 1, "cpu_ms": 2, "gpu_ms": 3, "energy_proxy": 4})
        == 40
    )
