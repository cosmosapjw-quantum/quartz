import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from quartz.idea_foundry import (
    ContractValidationError,
    CostVector,
    EdgeObservation,
    FoundryRootExtras,
    FreshnessIdentity,
    MetaAction,
    MetaActionKind,
    MetaProposal,
    ProposalEstimate,
    RootObservation,
    RuntimeObservation,
    foundry_root_extras_from_payload,
    foundry_root_extras_to_payload,
    proposal_from_payload,
    proposal_to_payload,
    root_observation_from_payload,
    root_observation_to_payload,
)
from quartz.idea_foundry.gates import (
    AXIS_TYPE_BY_ID,
    contract_fixture_bank,
    contract_observation,
    run_axis_contract_gate,
)
from quartz.idea_foundry.search import A26NestedContourExactLab
from quartz.idea_foundry.serialization import (
    freshness_from_payload,
    freshness_to_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "idea_foundry_contract_v1.json"


def _proposal(action, *, evidence_scope="contract_fixture_only"):
    return MetaProposal(
        axis_id="A04.kg_voc_allocator",
        action=action,
        estimate=ProposalEstimate(confidence=0.5),
        activation_guard="fresh root identity required",
        explanation="wire parity fixture",
        evidence_scope=evidence_scope,
    )


def _replace_root_wire_field(observation, location, field_name, value):
    if location == "root":
        return replace(observation, **{field_name: value})
    if location == "edge":
        edge = replace(observation.edges[0], **{field_name: value})
        return replace(observation, edges=(edge, *observation.edges[1:]))
    if location == "runtime":
        runtime = replace(observation.runtime, **{field_name: value})
        return replace(observation, runtime=runtime)
    raise AssertionError(f"unknown location: {location}")


def _set_root_payload_field(payload, location, field_name, value):
    row = payload["observation"]
    if location == "root":
        row[field_name] = value
    elif location == "edge":
        row["edges"][0][field_name] = value
    elif location == "runtime":
        row["runtime"][field_name] = value
    else:
        raise AssertionError(f"unknown location: {location}")


def _golden_root_observation():
    freshness = FreshnessIdentity(
        root_hash=123,
        checkpoint_id="seed_1/gen_1",
        evaluator_id="model-sha256:abc",
        edge_set_hash="edge-sha256:def",
        candidate_epoch=1,
        tt_identity_policy="state_eval_cache_only",
        cache_schema_version=1,
        root_visits=24,
        iteration=25,
    )
    runtime = RuntimeObservation()
    observation = RootObservation(
        root_hash=123,
        checkpoint_id="seed_1/gen_1",
        position_id="p1",
        game="gomoku7",
        root_visits=24,
        iteration=25,
        elapsed_ms=12,
        remaining_visits=40,
        n_children=1,
        n_visible=1,
        entropy=0.1,
        effective_branching=1.0,
        top2_margin=0.2,
        margin_slope=-0.01,
        entropy_slope=0.02,
        h1_stability=0.99,
        p_flip=0.01,
        prior_visit_js=0.2,
        candidate_omission_bound=0.01,
        revision_count=0,
        edges=(
            EdgeObservation(
                edge_pos=0,
                action_id=481,
                visible=True,
                prior_anchor=1.0,
                prior_current=1.0,
                visits=24,
                virtual_visits=0,
                pending=0,
                q_mean=0.2,
                q_sum=4.8,
                m2=0.6,
                last_value=0.1,
            ),
        ),
        runtime=runtime,
        freshness=freshness,
    )
    extras = FoundryRootExtras(
        freshness=freshness,
        entropy=0.1,
        effective_branching=1.0,
        top2_margin=0.2,
        margin_slope=-0.01,
        entropy_slope=0.02,
        h1_stability=0.99,
        p_flip=0.01,
        prior_visit_js=0.2,
        omission_bound=0.01,
        revision_count=0,
        runtime=runtime,
    )
    return observation, extras


def test_root_observation_roundtrip_is_canonical_for_seed_bank():
    """Decode/encode is stable across a deterministic generated input bank."""

    for seed in range(32):
        original = contract_observation(seed)
        payload = root_observation_to_payload(original)
        restored = root_observation_from_payload(payload)
        assert root_observation_to_payload(restored) == payload


def test_every_axis_proposal_roundtrip_is_canonical():
    observation = contract_observation()
    for axis_id, axis_type in sorted(AXIS_TYPE_BY_ID.items()):
        axis = axis_type()
        if axis_id == "A15":
            axis.table = {(8, 4): 10.0, (16, 2): 11.0}
        for proposal in axis.propose(observation):
            payload = proposal_to_payload(proposal)
            assert proposal_to_payload(proposal_from_payload(payload)) == payload


def test_cross_language_golden_fixture_matches_canonical_shape():
    fixture = json.loads(GOLDEN.read_text(encoding="utf-8"))
    proposal = MetaProposal(
        axis_id="A04.kg_voc_allocator",
        action=MetaAction(
            MetaActionKind.SAMPLE,
            primary=1,
            amount=8,
            label="golden_sample",
        ),
        estimate=ProposalEstimate(
            regret_reduction_mean=0.25,
            regret_reduction_lcb=0.1,
            confidence=0.5,
            cost=CostVector(nn_evals=8, cpu_ms=2),
        ),
        activation_guard="fresh root identity required",
        explanation="cross-language golden SAMPLE proposal",
        evidence_scope="contract_fixture_only",
        telemetry={"fixture": True},
    )
    assert proposal_to_payload(proposal) == fixture["meta_proposal_payload"]
    assert (
        freshness_to_payload(contract_observation().freshness_identity())[
            "tt_identity_policy"
        ]
        == fixture["freshness"]["tt_identity_policy"]
    )


def test_cross_language_root_and_foundry_extras_match_golden_json_exactly():
    fixture = json.loads(GOLDEN.read_text(encoding="utf-8"))
    observation, extras = _golden_root_observation()

    root_payload = root_observation_to_payload(observation)
    extras_payload = foundry_root_extras_to_payload(extras)
    assert root_payload == fixture["root_observation_payload"]
    assert extras_payload == fixture["foundry_root_extras"]
    assert root_observation_from_payload(root_payload) == observation
    assert foundry_root_extras_from_payload(extras_payload) == extras


@pytest.mark.parametrize("schema_version", [True, 0, 2, 1.0, 1 << 16])
def test_foundry_root_extras_rejects_noncanonical_schema_versions(schema_version):
    _, extras = _golden_root_observation()
    with pytest.raises(ContractValidationError, match="schema_version"):
        foundry_root_extras_to_payload(replace(extras, schema_version=schema_version))

    payload = foundry_root_extras_to_payload(extras)
    payload["schema_version"] = schema_version
    with pytest.raises(ContractValidationError, match="schema_version"):
        foundry_root_extras_from_payload(payload)


@pytest.mark.parametrize(
    ("field_name", "malformed_value"),
    [
        ("checkpoint_id", ""),
        ("evaluator_id", " \t"),
        ("edge_set_hash", ""),
        ("tt_identity_policy", "\n"),
        ("root_hash", -1),
        ("candidate_epoch", -1),
        ("cache_schema_version", 0),
        ("root_visits", -1),
        ("iteration", -1),
        ("root_hash", 1 << 64),
        ("candidate_epoch", 1 << 64),
        ("cache_schema_version", 1 << 16),
        ("root_visits", 1 << 32),
        ("iteration", 1 << 64),
        ("root_hash", True),
        ("candidate_epoch", 1.0),
        ("root_visits", "1"),
    ],
)
def test_freshness_identity_rejects_malformed_rust_parity_values(
    field_name, malformed_value
):
    valid = contract_observation().freshness_identity()
    malformed = replace(valid, **{field_name: malformed_value})

    with pytest.raises(ContractValidationError, match=field_name):
        freshness_to_payload(malformed)

    payload = freshness_to_payload(valid)
    payload[field_name] = malformed_value
    with pytest.raises(ContractValidationError, match=field_name):
        freshness_from_payload(payload)


def test_root_observation_validation_rejects_malformed_freshness_identity():
    observation = contract_observation()
    malformed = replace(observation.freshness_identity(), evaluator_id=" ")

    with pytest.raises(ContractValidationError, match="freshness.evaluator_id"):
        root_observation_to_payload(replace(observation, freshness=malformed))


@pytest.mark.parametrize(
    ("location", "field_name", "bits"),
    [
        ("root", "root_hash", 64),
        ("root", "root_visits", 32),
        ("root", "iteration", 64),
        ("root", "elapsed_ms", 64),
        ("root", "remaining_visits", 32),
        ("root", "n_children", 16),
        ("root", "n_visible", 16),
        ("root", "revision_count", 16),
        ("edge", "edge_pos", 16),
        ("edge", "action_id", 32),
        ("edge", "visits", 32),
        ("edge", "virtual_visits", 32),
        ("edge", "pending", 32),
        ("runtime", "threads", 16),
        ("runtime", "batch_size", 16),
        ("runtime", "inflight", 16),
        ("runtime", "max_pending", 16),
        ("runtime", "tt_wait_ns", 64),
    ],
)
def test_root_wire_unsigned_fields_reject_noncanonical_values(
    location, field_name, bits
):
    observation = contract_observation()
    for malformed in (-1, True, 1.0, "1", 1 << bits):
        with pytest.raises(ContractValidationError, match=field_name):
            root_observation_to_payload(
                _replace_root_wire_field(observation, location, field_name, malformed)
            )

        payload = root_observation_to_payload(observation)
        _set_root_payload_field(payload, location, field_name, malformed)
        with pytest.raises(ContractValidationError, match=field_name):
            root_observation_from_payload(payload)


def test_edge_position_must_be_inside_root_child_domain_on_encode_and_decode():
    observation = contract_observation()
    out_of_range = observation.n_children
    malformed_edge = replace(observation.edges[0], edge_pos=out_of_range)

    with pytest.raises(ContractValidationError, match="edge_pos"):
        root_observation_to_payload(
            replace(observation, edges=(malformed_edge, *observation.edges[1:]))
        )

    payload = root_observation_to_payload(observation)
    payload["observation"]["edges"][0]["edge_pos"] = out_of_range
    with pytest.raises(ContractValidationError, match="edge_pos"):
        root_observation_from_payload(payload)


@pytest.mark.parametrize(
    "action",
    [
        MetaAction(MetaActionKind.STOP, primary=0),
        MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1),
        MetaAction(MetaActionKind.CHALLENGE, primary=0, secondary=1, amount=2),
        MetaAction(MetaActionKind.WIDEN, amount=1),
        MetaAction(MetaActionKind.DEEPEN, primary=0, amount=1),
        MetaAction(MetaActionKind.PROVE, primary=0, amount=1),
        MetaAction(MetaActionKind.RESAMPLE_MODE, primary=0, amount=1),
        MetaAction(MetaActionKind.MERGE_OR_SHARE, primary=1),
        MetaAction(MetaActionKind.SET_BATCH, amount=1),
        MetaAction(MetaActionKind.SET_INFLIGHT, amount=1),
        MetaAction(MetaActionKind.SET_THREADS, amount=1),
        MetaAction(MetaActionKind.REANALYSE, primary=1),
        MetaAction(MetaActionKind.ARCHIVE_STATE, value=0.5),
        MetaAction(MetaActionKind.NOOP),
    ],
)
def test_every_meta_action_kind_roundtrips_in_its_canonical_shape(action):
    payload = proposal_to_payload(_proposal(action))
    restored = proposal_from_payload(payload)
    assert restored.action == action
    assert proposal_to_payload(restored) == payload


@pytest.mark.parametrize(
    ("valid_action", "field_name", "malformed", "remove_on_decode"),
    [
        (MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1), "primary", None, True),
        (
            MetaAction(
                MetaActionKind.CHALLENGE,
                primary=0,
                secondary=1,
                amount=1,
            ),
            "secondary",
            None,
            True,
        ),
        (MetaAction(MetaActionKind.ARCHIVE_STATE, value=0.5), "value", None, True),
        (MetaAction(MetaActionKind.NOOP), "primary", 0, False),
        (MetaAction(MetaActionKind.STOP), "amount", 1, False),
        (MetaAction(MetaActionKind.WIDEN, amount=1), "primary", 0, False),
        (
            MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1),
            "secondary",
            0,
            False,
        ),
        (
            MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1),
            "value",
            0.0,
            False,
        ),
        (MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1), "primary", -1, False),
        (
            MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1),
            "primary",
            1 << 16,
            False,
        ),
        (
            MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1),
            "primary",
            True,
            False,
        ),
        (MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1), "primary", 1.0, False),
        (MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1), "amount", -1, False),
        (
            MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1),
            "amount",
            1 << 32,
            False,
        ),
        (MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1), "amount", True, False),
        (MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1), "amount", 1.0, False),
        (
            MetaAction(MetaActionKind.RESAMPLE_MODE, primary=0, amount=1),
            "amount",
            1 << 16,
            False,
        ),
        (
            MetaAction(MetaActionKind.MERGE_OR_SHARE, primary=1),
            "primary",
            1 << 64,
            False,
        ),
    ],
)
def test_meta_action_rejects_missing_forbidden_and_out_of_range_fields(
    valid_action, field_name, malformed, remove_on_decode
):
    with pytest.raises(ContractValidationError, match=field_name):
        proposal_to_payload(_proposal(replace(valid_action, **{field_name: malformed})))

    payload = proposal_to_payload(_proposal(valid_action))
    action_payload = payload["proposal"]["action"]
    if remove_on_decode:
        action_payload.pop(field_name)
    else:
        action_payload[field_name] = malformed
    with pytest.raises(ContractValidationError, match=field_name):
        proposal_from_payload(payload)


@pytest.mark.parametrize("malformed", ["", " \t", True, 1])
def test_optional_action_label_must_be_a_nonempty_string(malformed):
    valid_action = MetaAction(MetaActionKind.NOOP, label="audit_only")
    with pytest.raises(ContractValidationError, match="action.label"):
        proposal_to_payload(_proposal(replace(valid_action, label=malformed)))

    payload = proposal_to_payload(_proposal(valid_action))
    payload["proposal"]["action"]["label"] = malformed
    with pytest.raises(ContractValidationError, match="action.label"):
        proposal_from_payload(payload)


@pytest.mark.parametrize("malformed", [None, "", " \n", True, 1])
def test_proposal_evidence_scope_must_be_a_nonempty_string(malformed):
    action = MetaAction(MetaActionKind.NOOP)
    with pytest.raises(ContractValidationError, match="evidence_scope"):
        proposal_to_payload(_proposal(action, evidence_scope=malformed))

    payload = proposal_to_payload(_proposal(action))
    if malformed is None:
        payload["proposal"].pop("evidence_scope")
    else:
        payload["proposal"]["evidence_scope"] = malformed
    with pytest.raises(ContractValidationError, match="evidence_scope"):
        proposal_from_payload(payload)


@pytest.mark.parametrize(
    ("location", "field_name"),
    [
        ("root", "entropy"),
        ("root", "effective_branching"),
        ("root", "top2_margin"),
        ("root", "margin_slope"),
        ("root", "entropy_slope"),
        ("root", "h1_stability"),
        ("root", "p_flip"),
        ("root", "prior_visit_js"),
        ("root", "candidate_omission_bound"),
        ("edge", "prior_anchor"),
        ("edge", "prior_current"),
        ("edge", "q_mean"),
        ("edge", "q_sum"),
        ("edge", "m2"),
        ("edge", "last_value"),
        ("edge", "mc_radius"),
        ("edge", "epistemic_radius"),
        ("edge", "drift_radius"),
        ("edge", "bias_radius"),
        ("edge", "lower"),
        ("edge", "upper"),
        ("runtime", "queue_wait_ms"),
        ("runtime", "eval_latency_ms"),
        ("runtime", "nps"),
        ("runtime", "edge_duplicate_rate"),
        ("runtime", "semantic_path_overlap"),
    ],
)
def test_root_wire_floats_reject_bool_and_numeric_strings(location, field_name):
    observation = contract_observation()
    for malformed in (True, "1.0"):
        with pytest.raises(ContractValidationError, match=field_name):
            root_observation_to_payload(
                _replace_root_wire_field(observation, location, field_name, malformed)
            )

        payload = root_observation_to_payload(observation)
        _set_root_payload_field(payload, location, field_name, malformed)
        with pytest.raises(ContractValidationError, match=field_name):
            root_observation_from_payload(payload)


@pytest.mark.parametrize(
    ("section", "field_name"),
    [
        ("estimate", "regret_reduction_mean"),
        ("estimate", "regret_reduction_lcb"),
        ("estimate", "confidence"),
        ("cost", "nn_evals"),
        ("cost", "cpu_ms"),
        ("cost", "gpu_ms"),
        ("cost", "energy_proxy"),
    ],
)
def test_proposal_wire_floats_reject_bool_and_numeric_strings(section, field_name):
    action = MetaAction(MetaActionKind.NOOP)
    valid = _proposal(action)
    for malformed in (True, "1.0"):
        if section == "estimate":
            estimate = replace(valid.estimate, **{field_name: malformed})
        else:
            cost = replace(valid.estimate.cost, **{field_name: malformed})
            estimate = replace(valid.estimate, cost=cost)
        with pytest.raises(ContractValidationError, match=field_name):
            proposal_to_payload(replace(valid, estimate=estimate))

        payload = proposal_to_payload(valid)
        estimate_payload = payload["proposal"]["estimate"]
        target = estimate_payload if section == "estimate" else estimate_payload["cost"]
        target[field_name] = malformed
        with pytest.raises(ContractValidationError, match=field_name):
            proposal_from_payload(payload)


@pytest.mark.parametrize("malformed", [True, "1.0"])
def test_archive_action_value_rejects_bool_and_numeric_strings(malformed):
    valid_action = MetaAction(MetaActionKind.ARCHIVE_STATE, value=0.5)
    with pytest.raises(ContractValidationError, match="action.value"):
        proposal_to_payload(_proposal(replace(valid_action, value=malformed)))

    payload = proposal_to_payload(_proposal(valid_action))
    payload["proposal"]["action"]["value"] = malformed
    with pytest.raises(ContractValidationError, match="action.value"):
        proposal_from_payload(payload)


@pytest.mark.parametrize("field_name", ["checkpoint_id", "position_id", "game"])
@pytest.mark.parametrize("malformed", [True, 1, 1.0])
def test_root_string_fields_reject_non_strings(field_name, malformed):
    observation = contract_observation()
    with pytest.raises(ContractValidationError, match=field_name):
        root_observation_to_payload(replace(observation, **{field_name: malformed}))

    payload = root_observation_to_payload(observation)
    payload["observation"][field_name] = malformed
    with pytest.raises(ContractValidationError, match=field_name):
        root_observation_from_payload(payload)


@pytest.mark.parametrize("malformed", ["false", 0, 1])
def test_edge_visible_rejects_truthiness_coercion(malformed):
    observation = contract_observation()
    malformed_edge = replace(observation.edges[0], visible=malformed)
    with pytest.raises(ContractValidationError, match="edge.visible"):
        root_observation_to_payload(
            replace(observation, edges=(malformed_edge, *observation.edges[1:]))
        )

    payload = root_observation_to_payload(observation)
    payload["observation"]["edges"][0]["visible"] = malformed
    with pytest.raises(ContractValidationError, match="edge.visible"):
        root_observation_from_payload(payload)


@pytest.mark.parametrize(
    ("encoded_flags", "wire_flags"),
    [
        ((1,), [1]),
        (("forced_win", 2), ["forced_win", 2]),
        (["forced_win"], "forced_win"),
    ],
)
def test_edge_tactical_flags_require_a_string_array(encoded_flags, wire_flags):
    observation = contract_observation()
    malformed_edge = replace(observation.edges[0], tactical_flags=encoded_flags)
    with pytest.raises(ContractValidationError, match="tactical_flags"):
        root_observation_to_payload(
            replace(observation, edges=(malformed_edge, *observation.edges[1:]))
        )

    payload = root_observation_to_payload(observation)
    payload["observation"]["edges"][0]["tactical_flags"] = wire_flags
    with pytest.raises(ContractValidationError, match="tactical_flags"):
        root_observation_from_payload(payload)


@pytest.mark.parametrize("field_name", ["activation_guard", "explanation"])
@pytest.mark.parametrize("malformed", [True, 1, 1.0])
def test_proposal_required_string_fields_reject_coercion(field_name, malformed):
    proposal = _proposal(MetaAction(MetaActionKind.NOOP))
    with pytest.raises(ContractValidationError, match=field_name):
        proposal_to_payload(replace(proposal, **{field_name: malformed}))

    payload = proposal_to_payload(proposal)
    payload["proposal"][field_name] = malformed
    with pytest.raises(ContractValidationError, match=field_name):
        proposal_from_payload(payload)


def test_telemetry_mapping_keys_are_not_coerced_to_strings():
    proposal = replace(
        _proposal(MetaAction(MetaActionKind.NOOP)), telemetry={1: "value"}
    )
    with pytest.raises(ContractValidationError, match="keys must be strings"):
        proposal_to_payload(proposal)


@pytest.mark.parametrize(
    ("section", "field_name"),
    [
        ("estimate", "regret_reduction_mean"),
        ("estimate", "regret_reduction_lcb"),
        ("estimate", "confidence"),
        ("estimate", "cost"),
        ("cost", "nn_evals"),
        ("cost", "cpu_ms"),
        ("cost", "gpu_ms"),
        ("cost", "energy_proxy"),
    ],
)
def test_proposal_estimate_and_cost_members_are_required(section, field_name):
    payload = proposal_to_payload(_proposal(MetaAction(MetaActionKind.NOOP)))
    estimate = payload["proposal"]["estimate"]
    target = estimate if section == "estimate" else estimate["cost"]
    target.pop(field_name)

    with pytest.raises(ContractValidationError, match=field_name):
        proposal_from_payload(payload)


@pytest.mark.parametrize(
    "field_name",
    [
        "threads",
        "batch_size",
        "inflight",
        "queue_wait_ms",
        "eval_latency_ms",
        "nps",
        "edge_duplicate_rate",
        "semantic_path_overlap",
        "max_pending",
        "tt_wait_ns",
    ],
)
def test_present_runtime_object_requires_every_rust_member(field_name):
    payload = root_observation_to_payload(contract_observation())
    payload["observation"]["runtime"].pop(field_name)

    with pytest.raises(ContractValidationError, match=field_name):
        root_observation_from_payload(payload)


def test_absent_runtime_object_uses_the_rust_default_snapshot():
    payload = root_observation_to_payload(contract_observation())
    payload["observation"].pop("runtime")

    restored = root_observation_from_payload(payload)
    assert restored.runtime.threads == 1
    assert restored.runtime.batch_size == 1
    assert restored.runtime.inflight == 1
    assert restored.runtime.max_pending == 0
    assert restored.runtime.tt_wait_ns == 0


def test_rust_floating_wire_fields_encode_as_canonical_json_floats():
    observation = contract_observation()
    for location, field_name in (
        *[
            ("root", name)
            for name in (
                "entropy",
                "effective_branching",
                "top2_margin",
                "margin_slope",
                "entropy_slope",
                "h1_stability",
                "p_flip",
                "prior_visit_js",
                "candidate_omission_bound",
            )
        ],
        *[
            ("edge", name)
            for name in (
                "prior_anchor",
                "prior_current",
                "q_mean",
                "q_sum",
                "m2",
                "last_value",
                "mc_radius",
                "epistemic_radius",
                "drift_radius",
                "bias_radius",
                "lower",
                "upper",
            )
        ],
        *[
            ("runtime", name)
            for name in (
                "queue_wait_ms",
                "eval_latency_ms",
                "nps",
                "edge_duplicate_rate",
                "semantic_path_overlap",
            )
        ],
    ):
        canonical_int = -1 if field_name == "lower" else 1
        payload = root_observation_to_payload(
            _replace_root_wire_field(observation, location, field_name, canonical_int)
        )
        row = payload["observation"]
        if location == "edge":
            row = row["edges"][0]
        elif location == "runtime":
            row = row["runtime"]
        assert type(row[field_name]) is float

    proposal = _proposal(MetaAction(MetaActionKind.ARCHIVE_STATE, value=1))
    proposal = replace(
        proposal,
        estimate=ProposalEstimate(
            regret_reduction_mean=1,
            regret_reduction_lcb=1,
            confidence=1,
            cost=CostVector(nn_evals=1, cpu_ms=1, gpu_ms=1, energy_proxy=1),
        ),
    )
    payload = proposal_to_payload(proposal)["proposal"]
    assert type(payload["action"]["value"]) is float
    assert all(
        type(payload["estimate"][name]) is float
        for name in ("regret_reduction_mean", "regret_reduction_lcb", "confidence")
    )
    assert all(
        type(payload["estimate"]["cost"][name]) is float
        for name in ("nn_evals", "cpu_ms", "gpu_ms", "energy_proxy")
    )


def test_validator_rejects_duplicate_edge_positions_and_nonfinite_cost():
    observation = contract_observation()
    duplicate = replace(observation.edges[1], edge_pos=observation.edges[0].edge_pos)
    with pytest.raises(ContractValidationError, match="edge_pos"):
        root_observation_to_payload(
            replace(observation, edges=(observation.edges[0], duplicate))
        )

    proposal = MetaProposal(
        axis_id="A04.kg_voc_allocator",
        action=MetaAction(MetaActionKind.SAMPLE, primary=0, amount=1),
        estimate=ProposalEstimate(cost=CostVector(nn_evals=math.nan)),
        activation_guard="guard",
        explanation="explanation",
    )
    with pytest.raises(ContractValidationError, match="finite"):
        proposal_to_payload(proposal)


def test_cost_price_key_is_canonical_with_v1_read_compatibility():
    cost = CostVector(nn_evals=2)
    assert cost.weighted({"nn_evals": 3}) == 6
    assert cost.weighted({"nn_eval": 3}) == 6
    assert cost.weighted({"nn_evals": 4, "nn_eval": 100}) == 8


def test_all_26_contract_gates_complete_without_auto_promotion():
    assert sorted(AXIS_TYPE_BY_ID) == [f"A{i:02d}" for i in range(1, 27)]
    for axis_id in sorted(AXIS_TYPE_BY_ID):
        result = run_axis_contract_gate(axis_id, role="contract")
        assert result["execution_status"] == "completed_no_promotion"
        assert result["promotion"] == {
            "auto": False,
            "eligible": False,
            "reason": result["outcome_detail"],
        }
        assert result["rows"]
    assert (
        run_axis_contract_gate("A10", role="conditional")["outcome_detail"]
        == "DORMANT_NO_ELIGIBLE_SLICE"
    )


def test_candidate_axes_use_the_preregistered_three_fixture_bank():
    assert [item.position_id for item in contract_fixture_bank()] == [
        "contract_fixture/hidden_best",
        "contract_fixture/near_tie",
        "contract_fixture/multimodal",
    ]
    for axis_id in ("A06", "A07", "A12", "A25", "A26"):
        result = run_axis_contract_gate(axis_id, role="synthetic")
        assert result["fixture_ids"] == [
            "contract_fixture/hidden_best",
            "contract_fixture/near_tie",
            "contract_fixture/multimodal",
        ]


def test_a26_finite_contour_identity_matches_enumeration_for_generated_bank():
    lab = A26NestedContourExactLab()
    for seed in range(1, 33):
        likelihoods = [((seed * (idx + 3)) % 17) / 16 for idx in range(6)]
        prior = [1 + ((seed + idx * 5) % 11) for idx in range(6)]
        assert math.isclose(
            lab.enumerated_evidence(likelihoods, prior),
            lab.finite_contour_evidence(likelihoods, prior),
            rel_tol=0.0,
            abs_tol=1e-12,
        )


@pytest.mark.parametrize(
    ("likelihoods", "prior"),
    [
        ([math.nan, 0.5], [1.0, 1.0]),
        ([math.inf, 0.5], [1.0, 1.0]),
        ([0.25, 0.5], [math.nan, 1.0]),
        ([0.25, 0.5], [math.inf, 1.0]),
        ([0.25, 0.5], [float.fromhex("0x1.fffffffffffffp+1023")] * 2),
    ],
    ids=[
        "nan-likelihood",
        "positive-inf-likelihood",
        "nan-prior",
        "positive-inf-prior",
        "overflow-prior-total",
    ],
)
def test_a26_rejects_nonfinite_inputs_and_overflowed_prior_mass(likelihoods, prior):
    lab = A26NestedContourExactLab()

    for evidence in (lab.enumerated_evidence, lab.finite_contour_evidence):
        with pytest.raises(ValueError, match="finite"):
            evidence(likelihoods, prior)


def test_axis_gate_cli_writes_validated_artifact_trio(tmp_path):
    output = tmp_path / "A01.trace"
    proc = subprocess.run(
        [
            str(REPO_ROOT / "venv" / "bin" / "python"),
            str(REPO_ROOT / "scripts" / "idea_foundry_axis_gate.py"),
            "--axis",
            "A01",
            "--role",
            "trace",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (output / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["auto_promoted"] is False
    assert manifest["status"] == "completed_no_promotion"
    assert summary["promotion"]["auto"] is False
    assert rows and all(row["axis_id"] == "A01" for row in rows)
