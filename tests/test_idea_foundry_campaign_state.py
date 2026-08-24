"""Focused contracts for identity-bound initial campaign publication."""

from __future__ import annotations

import copy

import pytest

from quartz.experiment_manifest import canonical_sha256
from quartz.idea_foundry.axis_workflow import load_workflow_specs
from quartz.idea_foundry.campaign_state import (
    CampaignStateError,
    build_initial_state_plan,
    publish_initial_run,
    validate_campaign_state_v2,
)
from quartz.idea_foundry.execution_identity import (
    ExecutionCapture,
    ExecutionIdentity,
    FileIdentity,
    RuntimeIdentity,
)


def _capture() -> ExecutionCapture:
    digest = "a" * 64
    identity = ExecutionIdentity(
        schema_version=1,
        git_head=digest,
        git_dirty=False,
        axis_registry=FileIdentity("configs/idea_foundry.axes.v1.json", 1, digest),
        lab_registry=FileIdentity("configs/idea_lab.local.v2.json", 1, digest),
        source_files=(
            FileIdentity("quartz/idea_foundry/status_schema.py", 1, digest),
            FileIdentity("quartz/idea_foundry/axis_workflow.py", 1, digest),
            FileIdentity("quartz/idea_foundry/sequential.py", 1, digest),
            FileIdentity("scripts/idea_foundry_run_all.py", 1, digest),
        ),
        runtime=RuntimeIdentity("python", "3", "platform", ("foundry",)),
    )
    return ExecutionCapture(
        identity=identity,
        identity_sha256=canonical_sha256(identity.to_payload()),
        axis_registry_bytes=b"captured-axes",
        lab_registry_bytes=b"captured-lanes",
        specs=load_workflow_specs(),
    )


def _plan():
    return build_initial_state_plan(
        "initial-state", 23, _capture(), "2026-08-25T00:00:00Z"
    )


def test_initial_state_is_derived_only_from_capture_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quartz.idea_foundry.sequential as sequential

    capture = _capture()

    def fail_live_registry_read():
        raise AssertionError("execution descriptors must not reread the registry")

    monkeypatch.setattr(sequential, "load_workflow_specs", fail_live_registry_read)
    plan = build_initial_state_plan(
        "initial-state", 23, capture, "2026-08-25T00:00:00Z"
    )
    state = plan["campaign_state"]

    assert state["schema_version"] == 2
    assert state["run_id"] == "initial-state"
    assert state["axes"] == [
        {
            "order_index": spec.order_index,
            "axis_id": spec.axis_id,
            "slug": spec.slug,
            "lane_id": spec.lane_id,
            "role": spec.role,
            "status": {
                "schema_version": 2,
                "execution": "planned",
                "contract": "not_evaluated",
                "effect": "not_evaluated",
                "evidence_maturity": "contract_only",
                "promotion": "ineligible",
            },
            "attempts": [],
        }
        for spec in capture.specs
    ]
    assert sequential._captured_execution_descriptors(
        capture.specs
    ) == sequential._persisted_execution_descriptors(state)


def test_initial_publication_orders_identity_before_state_and_attempts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quartz.idea_foundry.campaign_state as campaign_state

    written: list[str] = []
    original_dump = campaign_state.atomic_json_dump

    def recording_dump(path, payload) -> None:
        written.append(path.name)
        original_dump(path, payload)

    monkeypatch.setattr(campaign_state, "atomic_json_dump", recording_dump)
    root = tmp_path / "campaign"
    publish_initial_run(root, _plan())

    assert written == ["execution_identity.json", "campaign_state.json"]
    assert (root / "execution_identity.json").is_file()
    assert (root / "campaign_state.json").is_file()
    assert not (root / ".writer.lock").exists()
    assert (
        validate_campaign_state_v2(
            _plan()["campaign_state"], expected_run_id="initial-state"
        )["axes"][0]["attempts"]
        == []
    )


def test_duplicate_new_run_is_rejected_without_overwrite(tmp_path) -> None:
    root = tmp_path / "campaign"
    publish_initial_run(root, _plan())
    state_before = (root / "campaign_state.json").read_bytes()
    identity_before = (root / "execution_identity.json").read_bytes()

    with pytest.raises(CampaignStateError, match="already exists"):
        publish_initial_run(root, _plan())

    assert (root / "campaign_state.json").read_bytes() == state_before
    assert (root / "execution_identity.json").read_bytes() == identity_before


def test_invalid_state_is_rejected_before_atomic_write(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quartz.idea_foundry.campaign_state as campaign_state

    plan = _plan()
    plan["campaign_state"]["schema_version"] = 1
    writes: list[object] = []
    monkeypatch.setattr(
        campaign_state, "atomic_json_dump", lambda *args: writes.append(args)
    )

    with pytest.raises(CampaignStateError, match="schema_version"):
        publish_initial_run(tmp_path / "campaign", plan)

    assert writes == []
    assert not (tmp_path / "campaign").exists()


def test_partial_identity_only_root_is_preserved_and_not_reused(tmp_path) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    identity_only = b'{"identity":"captured"}\n'
    (root / "execution_identity.json").write_bytes(identity_only)

    with pytest.raises(CampaignStateError, match="already exists"):
        publish_initial_run(root, _plan())

    assert (root / "execution_identity.json").read_bytes() == identity_only
    assert not (root / "campaign_state.json").exists()


def test_state_v2_rejects_alias_types_extra_fields_and_wrong_axis_order() -> None:
    state = _plan()["campaign_state"]
    alias = copy.deepcopy(state)
    alias["seed"] = True
    extra = copy.deepcopy(state)
    extra["unexpected"] = None
    reversed_axes = copy.deepcopy(state)
    reversed_axes["axes"] = list(reversed(reversed_axes["axes"]))

    for invalid in (alias, extra, reversed_axes):
        with pytest.raises(CampaignStateError):
            validate_campaign_state_v2(invalid)
