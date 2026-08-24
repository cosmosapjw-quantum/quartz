"""Behavioral contracts for Idea Foundry status-schema-v2 writers."""

import pytest

from quartz.idea_foundry import status_schema


def _canonical_rows() -> list[dict[str, object]]:
    return [
        {
            "schema_version": 2,
            "execution": "planned",
            "contract": "not_evaluated",
            "effect": "not_evaluated",
            "evidence_maturity": "contract_only",
            "promotion": "ineligible",
        },
        {
            "schema_version": 2,
            "execution": "running",
            "contract": "not_evaluated",
            "effect": "not_evaluated",
            "evidence_maturity": "contract_only",
            "promotion": "ineligible",
        },
        {
            "schema_version": 2,
            "execution": "failed",
            "contract": "failed",
            "effect": "invalidated",
            "evidence_maturity": "contract_only",
            "promotion": "ineligible",
        },
        {
            "schema_version": 2,
            "execution": "success",
            "contract": "passed",
            "effect": "non_estimable",
            "evidence_maturity": "contract_only",
            "promotion": "ineligible",
        },
        {
            "schema_version": 2,
            "execution": "skipped",
            "contract": "not_applicable",
            "effect": "non_estimable",
            "evidence_maturity": "contract_only",
            "promotion": "ineligible",
        },
    ]


def test_status_v2_round_trip_for_all_canonical_rows() -> None:
    expected = _canonical_rows()
    observed = [
        status_schema.planned_status(),
        status_schema.running_status(),
        status_schema.failed_status(),
        status_schema.succeeded_status(),
        status_schema.skipped_status(),
    ]
    assert observed == expected
    for row in observed:
        assert status_schema.validate_status_v2(row) == row
    estimable_success = {
        "schema_version": 2,
        "execution": "success",
        "contract": "passed",
        "effect": "estimable",
        "evidence_maturity": "ablation",
        "promotion": "eligible",
    }
    for writer in (
        status_schema.StatusWriter.ANALYSIS,
        status_schema.StatusWriter.META_ANALYSIS,
    ):
        assert (
            status_schema.validate_writer_status(
                status_schema.succeeded_status(),
                writer=writer,
            )
            == status_schema.succeeded_status()
        )
        for rejected in (
            status_schema.planned_status(),
            status_schema.running_status(),
            status_schema.failed_status(),
            status_schema.skipped_status(),
            estimable_success,
        ):
            with pytest.raises(status_schema.StatusSchemaError):
                status_schema.validate_writer_status(rejected, writer=writer)


def test_status_v2_rejects_bool_float_alias_and_extra_fields() -> None:
    canonical = _canonical_rows()[3]
    missing_promotion = dict(canonical)
    missing_promotion.pop("promotion")
    invalid_rows = [
        {**canonical, "schema_version": True},
        {**canonical, "schema_version": 2.0},
        {**canonical, "schema_version": "2"},
        {**canonical, "execution": status_schema.ExecutionStatus.SUCCESS},
        {**canonical, "execution": "SUCCESS"},
        {**canonical, "unexpected": "field"},
        missing_promotion,
    ]
    for invalid in invalid_rows:
        with pytest.raises(status_schema.StatusSchemaError):
            status_schema.validate_status_v2(invalid)


def test_campaign_writer_rejects_planned_and_skipped() -> None:
    accepted = (
        status_schema.running_status(),
        status_schema.failed_status(),
        status_schema.succeeded_status(),
    )
    for value in accepted:
        assert (
            status_schema.validate_writer_status(
                value, writer=status_schema.StatusWriter.CAMPAIGN
            )
            == value
        )
    for rejected in (status_schema.planned_status(), status_schema.skipped_status()):
        with pytest.raises(status_schema.StatusSchemaError):
            status_schema.validate_writer_status(
                rejected, writer=status_schema.StatusWriter.CAMPAIGN
            )


def test_a10_writer_accepts_skipped_and_rejects_success() -> None:
    accepted = (
        status_schema.planned_status(),
        status_schema.running_status(),
        status_schema.failed_status(),
        status_schema.skipped_status(),
    )
    for value in accepted:
        assert (
            status_schema.validate_writer_status(
                value,
                writer=status_schema.StatusWriter.AXIS,
                axis_id="A10",
            )
            == value
        )
    with pytest.raises(status_schema.StatusSchemaError):
        status_schema.validate_writer_status(
            status_schema.succeeded_status(),
            writer=status_schema.StatusWriter.AXIS,
            axis_id="A10",
        )


def test_normal_axis_writer_accepts_success_and_rejects_skipped() -> None:
    accepted = (
        status_schema.planned_status(),
        status_schema.running_status(),
        status_schema.failed_status(),
        status_schema.succeeded_status(),
    )
    for value in accepted:
        assert (
            status_schema.validate_writer_status(
                value,
                writer=status_schema.StatusWriter.AXIS,
                axis_id="A01",
            )
            == value
        )
    with pytest.raises(status_schema.StatusSchemaError):
        status_schema.validate_writer_status(
            status_schema.skipped_status(),
            writer=status_schema.StatusWriter.AXIS,
            axis_id="A01",
        )


def test_resume_eligible_campaign_is_failed_only() -> None:
    assert (
        status_schema.is_resume_eligible_campaign_status(status_schema.failed_status())
        is True
    )
    for value in (status_schema.running_status(), status_schema.succeeded_status()):
        assert status_schema.is_resume_eligible_campaign_status(value) is False


def test_legacy_helpers_preserve_existing_axis_artifact_semantics() -> None:
    normal = status_schema.first_gate_status("A01")
    dormant = status_schema.first_gate_status("A10")
    assert normal == status_schema.succeeded_status()
    assert dormant == status_schema.skipped_status()
    assert status_schema.validate_axis_status("A01", normal) == normal
    assert status_schema.validate_axis_status("A10", dormant) == dormant
    assert status_schema.validate_campaign_status(normal) == normal
    with pytest.raises(status_schema.StatusSchemaError, match="campaign status"):
        status_schema.validate_campaign_status(status_schema.running_status())
    with pytest.raises(status_schema.StatusSchemaError, match="campaign status"):
        status_schema.validate_campaign_status(status_schema.failed_status())
    with pytest.raises(status_schema.StatusSchemaError, match="axis status"):
        status_schema.validate_axis_status("A01", status_schema.running_status())
    with pytest.raises(status_schema.StatusSchemaError, match="axis status"):
        status_schema.validate_axis_status("A10", status_schema.failed_status())
    assert status_schema.is_success(normal) is True
    assert status_schema.is_success(dormant) is False
    assert status_schema.is_resumable(normal) is True
    assert status_schema.is_resumable(dormant) is True
    assert status_schema.is_resumable(status_schema.failed_status()) is False
    assert status_schema.is_terminal_axis_status(normal, axis_id="A01") is True
    assert status_schema.is_terminal_axis_status(dormant, axis_id="A10") is True
    assert status_schema.is_poolable(normal) is False
    assert status_schema.is_legacy_status("completed_no_promotion") is True
    assert status_schema.is_legacy_status({"schema_version": 1}) is True
    assert status_schema.is_legacy_status(normal) is False
    assert (
        status_schema.transition_status(status_schema.ExecutionStatus.PLANNED)
        == status_schema.planned_status()
    )
    assert (
        status_schema.transition_status(status_schema.ExecutionStatus.RUNNING)
        == status_schema.running_status()
    )
    assert (
        status_schema.transition_status(status_schema.ExecutionStatus.FAILED)
        == status_schema.failed_status()
    )
