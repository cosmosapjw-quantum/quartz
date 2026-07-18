"""Dependency-light first contract gates for all 26 foundry axes.

These gates are executable specifications for wiring and invariants.  They are
not efficacy experiments: a passing result always remains
``completed_no_promotion`` until the trace/live/training lane named in the
atlas is run with its own evidence contract.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import replace
from typing import Any

from .contracts import (
    AxisStatus,
    CostVector,
    EdgeObservation,
    FreshnessIdentity,
    MetaAction,
    MetaActionKind,
    RootObservation,
    RuntimeObservation,
)
from .control import (
    A01StopCouncil,
    A02StaticAnchorRPO,
    A03UncertaintyDecomposition,
    A04KgVocAllocator,
    A05CounterfactualMetaTeacher,
    A24LearnedBudgetGate,
)
from .learning import (
    A17B13CurvatureReadout,
    A18DiffusionRegularizedEvaluator,
    A19RwRestLiteEvaluator,
    A20RegretStateArchive,
    A21CoherenceSignedPathShadow,
    A22PhysicsFalsificationDashboard,
    A23CpuIncrementalPatternStudent,
)
from .search import (
    A06GumbelSequentialHalving,
    A07ResidualEvidenceWidening,
    A08TacticalProofBackend,
    A09H3ChangePointRouter,
    A10PriorRefreshSpecialist,
    A11DynamicLiveSetParticles,
    A12JsdLocallyBalancedSampler,
    A13PendingFlowWuUct,
    A14SemanticPathLsh,
    A15ServiceCurveScheduler,
    A16MonteCarloGraphSharing,
    A25MentsSoftBackup,
    A26NestedContourExactLab,
)
from .serialization import (
    canonical_sha256,
    proposal_to_payload,
    root_observation_to_payload,
    validate_proposal,
)


AXIS_TYPES = (
    A01StopCouncil,
    A02StaticAnchorRPO,
    A03UncertaintyDecomposition,
    A04KgVocAllocator,
    A05CounterfactualMetaTeacher,
    A06GumbelSequentialHalving,
    A07ResidualEvidenceWidening,
    A08TacticalProofBackend,
    A09H3ChangePointRouter,
    A10PriorRefreshSpecialist,
    A11DynamicLiveSetParticles,
    A12JsdLocallyBalancedSampler,
    A13PendingFlowWuUct,
    A14SemanticPathLsh,
    A15ServiceCurveScheduler,
    A16MonteCarloGraphSharing,
    A17B13CurvatureReadout,
    A18DiffusionRegularizedEvaluator,
    A19RwRestLiteEvaluator,
    A20RegretStateArchive,
    A21CoherenceSignedPathShadow,
    A22PhysicsFalsificationDashboard,
    A23CpuIncrementalPatternStudent,
    A24LearnedBudgetGate,
    A25MentsSoftBackup,
    A26NestedContourExactLab,
)

AXIS_TYPE_BY_ID = {
    axis_type().axis_id.split(".", 1)[0]: axis_type for axis_type in AXIS_TYPES
}

_EVIDENCE_STATUS_BY_AXIS_STATUS = {
    AxisStatus.SEED: "skeleton_only",
    AxisStatus.MECHANISM_VALID: "mechanism_valid",
    AxisStatus.SHADOW: "shadow_only",
    AxisStatus.CONDITIONAL: "conditional_only",
    AxisStatus.ACTIVE_EXPERIMENTAL: "active_experimental",
    AxisStatus.DEPLOYMENT_CANDIDATE: "deployment_candidate",
    AxisStatus.DORMANT: "skeleton_only",
    AxisStatus.ANALYSIS_ONLY: "analysis_only",
}


def contract_observation(seed: int = 20260718) -> RootObservation:
    """Build a deterministic bank member that exercises every proposal family."""

    rng = random.Random(seed)
    edges = (
        EdgeObservation(
            edge_pos=0,
            action_id=10,
            visible=True,
            prior_anchor=0.55,
            prior_current=0.50,
            visits=16,
            virtual_visits=0,
            pending=0,
            q_mean=0.40,
            q_sum=6.40,
            m2=0.80,
            last_value=0.45,
            mc_radius=0.04,
            epistemic_radius=0.02,
            drift_radius=0.01,
            bias_radius=0.01,
            lower=0.32,
            upper=0.48,
            tactical_flags=("forced_win",),
        ),
        EdgeObservation(
            edge_pos=1,
            action_id=481,
            visible=True,
            prior_anchor=0.30,
            prior_current=0.32,
            visits=8,
            virtual_visits=1,
            pending=1,
            q_mean=0.20,
            q_sum=1.60,
            m2=0.60,
            last_value=0.10,
            mc_radius=0.08,
            epistemic_radius=0.03,
            drift_radius=0.02,
            bias_radius=0.01,
            lower=0.06,
            upper=0.34,
            tactical_flags=("candidate_win",),
        ),
        EdgeObservation(
            edge_pos=2,
            action_id=999,
            visible=False,
            prior_anchor=0.10,
            prior_current=0.10,
            visits=0,
            virtual_visits=0,
            pending=0,
            q_mean=0.00,
            q_sum=0.00,
            m2=0.00,
            mc_radius=0.20,
            epistemic_radius=0.10,
            drift_radius=0.00,
            bias_radius=0.05,
            lower=-0.35,
            upper=0.55,
            tactical_flags=("forced_block",),
        ),
        EdgeObservation(
            edge_pos=3,
            action_id=7,
            visible=False,
            prior_anchor=0.05,
            prior_current=0.08,
            visits=0,
            virtual_visits=0,
            pending=0,
            q_mean=-0.05 + rng.uniform(-1e-9, 1e-9),
            q_sum=0.00,
            m2=0.00,
            mc_radius=0.25,
            epistemic_radius=0.12,
            drift_radius=0.02,
            bias_radius=0.05,
            lower=-0.49,
            upper=0.42,
        ),
    )
    edge_set_hash = hashlib.sha256(
        ",".join(f"{edge.edge_pos}:{edge.action_id}" for edge in edges).encode("ascii")
    ).hexdigest()
    freshness = FreshnessIdentity(
        root_hash=123456789,
        checkpoint_id="seed_1/gen_8",
        evaluator_id="gomoku7-resnet-96x6:seed_1/gen_8",
        edge_set_hash=edge_set_hash,
        candidate_epoch=3,
        tt_identity_policy="state_eval_cache_only",
        cache_schema_version=1,
        root_visits=24,
        iteration=25,
    )
    return RootObservation(
        root_hash=freshness.root_hash,
        checkpoint_id=freshness.checkpoint_id,
        position_id="contract_fixture/near_tie",
        game="gomoku7",
        root_visits=freshness.root_visits,
        iteration=freshness.iteration,
        elapsed_ms=12,
        remaining_visits=40,
        n_children=4,
        n_visible=2,
        entropy=0.90,
        effective_branching=3.40,
        top2_margin=0.20,
        margin_slope=-0.01,
        entropy_slope=0.02,
        h1_stability=0.99,
        p_flip=0.01,
        prior_visit_js=0.80,
        candidate_omission_bound=0.01,
        revision_count=1,
        edges=edges,
        runtime=RuntimeObservation(
            threads=16,
            batch_size=8,
            inflight=4,
            queue_wait_ms=0.25,
            eval_latency_ms=1.25,
            nps=1500.0,
            edge_duplicate_rate=0.10,
            semantic_path_overlap=0.75,
            max_pending=2,
            tt_wait_ns=500,
        ),
        extras={
            "evaluator_id": freshness.evaluator_id,
            "edge_set_hash": edge_set_hash,
            "candidate_epoch": freshness.candidate_epoch,
            "tt_identity_policy": freshness.tt_identity_policy,
            "cache_schema_version": freshness.cache_schema_version,
            "shareable_transpositions": 2,
            "shareable_state_key": 0xA160001,
        },
        freshness=freshness,
    )


def contract_fixture_bank(seed: int = 20260718) -> tuple[RootObservation, ...]:
    """Return the preregistered hidden-best, near-tie, and multimodal bank."""

    near_tie = contract_observation(seed)

    hidden_edges = list(near_tie.edges)
    hidden_edges[2] = replace(
        hidden_edges[2],
        q_mean=0.65,
        q_sum=0.0,
        lower=0.40,
        upper=0.85,
    )
    hidden_root_hash = near_tie.root_hash ^ 0x48494444
    hidden_best = replace(
        near_tie,
        root_hash=hidden_root_hash,
        position_id="contract_fixture/hidden_best",
        top2_margin=0.25,
        candidate_omission_bound=0.30,
        edges=tuple(hidden_edges),
        freshness=replace(
            near_tie.freshness_identity(),
            root_hash=hidden_root_hash,
            candidate_epoch=4,
        ),
    )

    modal_edges = (
        replace(near_tie.edges[0], q_mean=0.35, lower=0.25, upper=0.45),
        replace(near_tie.edges[1], q_mean=0.34, lower=0.20, upper=0.48),
        replace(near_tie.edges[2], q_mean=0.33, lower=0.05, upper=0.61),
        replace(near_tie.edges[3], q_mean=-0.30, lower=-0.60, upper=0.05),
    )
    multimodal_root_hash = near_tie.root_hash ^ 0x4D4F4445
    multimodal = replace(
        near_tie,
        root_hash=multimodal_root_hash,
        position_id="contract_fixture/multimodal",
        top2_margin=0.01,
        entropy=1.25,
        candidate_omission_bound=0.15,
        edges=modal_edges,
        freshness=replace(
            near_tie.freshness_identity(),
            root_hash=multimodal_root_hash,
            candidate_epoch=5,
        ),
    )
    return hidden_best, near_tie, multimodal


def _axis_instance(axis_id: str):
    try:
        axis_type = AXIS_TYPE_BY_ID[axis_id]
    except KeyError as exc:
        raise ValueError(f"unknown foundry axis: {axis_id}") from exc
    if axis_id == "A15":
        return A15ServiceCurveScheduler(table={(8, 4): 100.0, (16, 2): 125.0})
    if axis_id == "A24":
        return A24LearnedBudgetGate(predict_gain=lambda _obs, budget: 1.0 / budget)
    return axis_type()


def _check(condition: bool, label: str, rows: list[dict[str, Any]]) -> None:
    rows.append({"metric": label, "value": bool(condition)})
    if not condition:
        raise AssertionError(label)


def run_axis_contract_gate(
    axis_id: str,
    *,
    role: str,
    seed: int = 20260718,
) -> dict[str, Any]:
    """Run deterministic invariants and return a manifest-ready result."""

    if not role.strip():
        raise ValueError("role must be non-empty")
    axis = _axis_instance(axis_id)
    observation = contract_observation(seed)
    before = root_observation_to_payload(observation)
    proposals = tuple(axis.propose(observation))
    second = tuple(axis.propose(copy.deepcopy(observation)))
    rows: list[dict[str, Any]] = []

    encoded = [proposal_to_payload(item) for item in proposals]
    encoded_second = [proposal_to_payload(item) for item in second]
    for proposal in proposals:
        validate_proposal(proposal)
        _check(proposal.axis_id == axis.axis_id, "proposal_axis_identity", rows)
    _check(encoded == encoded_second, "deterministic_proposals", rows)
    _check(
        root_observation_to_payload(observation) == before,
        "observation_immutable",
        rows,
    )
    _check(len(proposals) > 0, "nonempty_contract_output", rows)

    action_kinds = {proposal.action.kind for proposal in proposals}
    if axis.status is AxisStatus.ANALYSIS_ONLY:
        _check(action_kinds == {MetaActionKind.NOOP}, "analysis_only_no_control", rows)
    if axis_id == "A01":
        _check(action_kinds == {MetaActionKind.STOP}, "calibrated_stop_shape", rows)
    elif axis_id == "A02":
        policy = axis.solve(observation.edges)
        _check(set(policy) == {0, 1, 2, 3}, "edge_pos_indexing", rows)
        _check(math.isclose(sum(policy.values()), 1.0), "policy_normalized", rows)
    elif axis_id == "A03":
        edge = observation.edges[0]
        _check(
            axis.radius(edge)
            >= A03UncertaintyDecomposition("A03.test", combine="rss").radius(edge),
            "conservative_sum_ge_rss",
            rows,
        )
    elif axis_id == "A04":
        _check(
            MetaActionKind.STOP not in action_kinds, "kg_allocation_never_stop", rows
        )
    elif axis_id == "A05":
        fork_specs = (
            (MetaAction(MetaActionKind.STOP, primary=0), 0.40, 0.32, CostVector()),
            (
                MetaAction(MetaActionKind.SAMPLE, primary=0, amount=8),
                0.40,
                0.18,
                CostVector(nn_evals=8, cpu_ms=2.0),
            ),
            (
                MetaAction(MetaActionKind.SAMPLE, primary=1, amount=8),
                0.40,
                0.16,
                CostVector(nn_evals=8, cpu_ms=2.2),
            ),
            (
                MetaAction(MetaActionKind.WIDEN, amount=4),
                0.40,
                0.21,
                CostVector(nn_evals=4, cpu_ms=1.5),
            ),
        )
        labels = tuple(
            axis.build_label(
                obs=observation,
                action=action,
                loss_before=loss_before,
                loss_after=loss_after,
                cost=cost,
                oracle_action_id=10,
            )
            for action, loss_before, loss_after, cost in fork_specs
        )
        _check(
            [label.action.kind for label in labels]
            == [
                MetaActionKind.STOP,
                MetaActionKind.SAMPLE,
                MetaActionKind.SAMPLE,
                MetaActionKind.WIDEN,
            ],
            "counterfactual_action_set",
            rows,
        )
        _check(
            labels[1].action.primary == observation.best_edge().edge_pos
            and labels[2].action.primary == observation.runner_up().edge_pos,
            "incumbent_challenger_roles",
            rows,
        )
        _check(
            all(
                label.checkpoint_id == observation.checkpoint_id
                and label.position_id == observation.position_id
                and label.budget == observation.root_visits
                for label in labels
            ),
            "resident_root_frozen_across_forks",
            rows,
        )
        _check(
            all(
                math.isclose(
                    label.regret_reduction,
                    loss_before - loss_after,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and label.realized_cost == cost
                for label, (_, loss_before, loss_after, cost) in zip(labels, fork_specs)
            ),
            "raw_regret_and_cost_vectors_preserved",
            rows,
        )
        freshness = observation.freshness_identity()
        _check(
            all(label.freshness_identity == freshness for label in labels),
            "root_evaluator_candidate_tt_cache_identity_preserved",
            rows,
        )
    elif axis_id == "A06":
        candidates = axis.initial_candidates(observation)
        _check(len(candidates) == len(set(candidates)), "without_replacement", rows)
        _check(
            sum(p.action.amount for p in proposals) <= observation.remaining_visits,
            "budget_conserved",
            rows,
        )
    elif axis_id == "A07":
        bound = axis.bound(observation)
        _check(0.0 <= bound <= 1.0, "residual_mass_is_probability", rows)
        _check(action_kinds == {MetaActionKind.WIDEN}, "explicit_widen", rows)
    elif axis_id == "A08":
        _check(
            action_kinds <= {MetaActionKind.STOP, MetaActionKind.PROVE},
            "proof_actions_only",
            rows,
        )
    elif axis_id == "A09":
        _check(axis.score(observation) > 0.0, "continuous_change_score", rows)
    elif axis_id == "A10":
        _check(
            action_kinds == {MetaActionKind.NOOP}, "dormant_audit_no_activation", rows
        )
    elif axis_id == "A11":
        _check(
            action_kinds == {MetaActionKind.RESAMPLE_MODE},
            "root_particle_actions",
            rows,
        )
    elif axis_id == "A12":
        p, q = [0.7, 0.3], [0.2, 0.8]
        _check(math.isclose(axis.jsd(p, q), axis.jsd(q, p)), "jsd_symmetric", rows)
        _check(math.isclose(axis.jsd(p, p), 0.0, abs_tol=1e-12), "jsd_identity", rows)
    elif axis_id == "A13":
        edge = observation.edges[1]
        _check(
            axis.effective_visits(edge)
            == edge.visits + edge.pending + edge.virtual_visits,
            "pending_accounted_separately",
            rows,
        )
    elif axis_id == "A14":
        _check(
            axis.signature(["b", "a", "a"]) == axis.signature(["a", "b"]),
            "signature_set_invariant",
            rows,
        )
    elif axis_id == "A15":
        _check(axis.best() == (16, 2), "service_curve_argmax", rows)
    elif axis_id == "A16":
        _check(
            action_kinds == {MetaActionKind.MERGE_OR_SHARE}, "cache_share_action", rows
        )
        _check(
            all(p.action.label == "state_cache_only" for p in proposals),
            "parent_stats_not_merged",
            rows,
        )
    elif axis_id == "A17":
        policy = axis.transform(observation)
        _check(math.isclose(sum(policy.values()), 1.0), "readout_normalized", rows)
        _check(
            max(policy, key=policy.get) == observation.best_edge().edge_pos,
            "decision_neutral_fixture",
            rows,
        )
    elif axis_id == "A18":
        _check(
            axis.loss_contract()["inference"]
            == "direct_deterministic_policy_value_only",
            "deterministic_inference_contract",
            rows,
        )
    elif axis_id == "A19":
        _check(
            axis.architecture_contract()["routing"] == "soft_train_then_static_prune",
            "static_deployment_routing",
            rows,
        )
    elif axis_id == "A20":
        _check(
            action_kinds == {MetaActionKind.ARCHIVE_STATE}, "archive_only_action", rows
        )
    elif axis_id == "A21":
        _check(0.0 <= axis.coherence(observation) <= 1.0, "bounded_coherence", rows)
    elif axis_id == "A22":
        _check(
            len(axis.beta_fit_inputs(observation)["policy"]) == len(observation.edges),
            "dashboard_support_complete",
            rows,
        )
    elif axis_id == "A23":
        _check(
            bool(axis.deployment_contract()["incremental_update"]),
            "incremental_contract_explicit",
            rows,
        )
    elif axis_id == "A24":
        _check(action_kinds == {MetaActionKind.NOOP}, "budget_gate_offline_only", rows)
        _check(
            all(
                "candidate_root_budget" in proposal.telemetry for proposal in proposals
            ),
            "budget_telemetry_present",
            rows,
        )
        _check(
            all(p.estimate.regret_reduction_lcb <= 0.0 for p in proposals),
            "unearned_gain_not_positive_lcb",
            rows,
        )
    elif axis_id == "A25":
        cold = A25MentsSoftBackup(temperature=1e-6).soft_value([0.2, 0.7], [0.5, 0.5])
        _check(math.isclose(cold, 0.7, abs_tol=1e-5), "temperature_zero_limit", rows)
    elif axis_id == "A26":
        _check(action_kinds == {MetaActionKind.NOOP}, "exact_lab_offline", rows)
        likelihoods = (0.05, 0.20, 0.75, 0.90)
        prior = (0.10, 0.20, 0.30, 0.40)
        enumerated = axis.enumerated_evidence(likelihoods, prior)
        contour = axis.finite_contour_evidence(likelihoods, prior)
        _check(
            math.isclose(enumerated, contour, rel_tol=0.0, abs_tol=1e-12),
            "finite_contour_matches_enumeration",
            rows,
        )

    bank_payloads: list[dict[str, Any]] = []
    if axis_id in {"A06", "A07", "A12", "A25", "A26"}:
        for fixture in contract_fixture_bank(seed):
            bank_axis = _axis_instance(axis_id)
            first = [proposal_to_payload(item) for item in bank_axis.propose(fixture)]
            replay = [
                proposal_to_payload(item)
                for item in _axis_instance(axis_id).propose(copy.deepcopy(fixture))
            ]
            _check(
                first == replay,
                f"bank_deterministic:{fixture.position_id}",
                rows,
            )
            _check(
                bool(first),
                f"bank_nonempty:{fixture.position_id}",
                rows,
            )
            if axis_id == "A06":
                _check(
                    sum(item["proposal"]["action"]["amount"] for item in first)
                    <= fixture.remaining_visits,
                    f"bank_budget_conserved:{fixture.position_id}",
                    rows,
                )
            elif axis_id == "A07":
                bound = bank_axis.bound(fixture)
                _check(
                    0.0 <= bound <= 1.0,
                    f"bank_omission_bound:{fixture.position_id}",
                    rows,
                )
            bank_payloads.append(
                {
                    "fixture_id": fixture.position_id,
                    "observation_hash": canonical_sha256(
                        root_observation_to_payload(fixture)
                    ),
                    "proposals": first,
                }
            )
        _check(len(bank_payloads) == 3, "preregistered_fixture_bank_complete", rows)
    else:
        bank_payloads.append(
            {
                "fixture_id": observation.position_id,
                "observation_hash": canonical_sha256(before),
                "proposals": encoded,
            }
        )

    outcome_detail = (
        "DORMANT_NO_ELIGIBLE_SLICE"
        if axis_id == "A10"
        else "CONTRACT_GATE_PASSED_NO_EFFICACY_PROMOTION"
    )
    return {
        "axis_id": axis_id,
        "axis_symbol": axis.axis_id,
        "role": role,
        "seed": seed,
        "fixture_id": observation.position_id,
        "fixture_ids": [item["fixture_id"] for item in bank_payloads],
        "fixture_hash": canonical_sha256(before),
        "fixture_bank_hash": canonical_sha256(bank_payloads),
        "proposal_hash": canonical_sha256(
            [item["proposals"] for item in bank_payloads]
        ),
        "proposal_count": len(proposals),
        "evidence_status": _EVIDENCE_STATUS_BY_AXIS_STATUS[axis.status],
        "axis_registry_status": axis.status.value,
        "execution_status": "completed_no_promotion",
        "promotion": {"auto": False, "eligible": False, "reason": outcome_detail},
        "outcome_detail": outcome_detail,
        "rows": rows,
        "proposals": encoded,
    }
