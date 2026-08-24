"""Canonical status-schema-v2 contract for Idea Foundry artifacts."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping


STATUS_SCHEMA_VERSION = 2
STATUS_SCHEMA_PATH = Path(__file__).resolve()


def _enum(name: str, values: str) -> type[StrEnum]:
    return StrEnum(name, {value.upper(): value for value in values.split()})


class StatusWriter(StrEnum):
    CAMPAIGN = "campaign"
    AXIS = "axis"
    ANALYSIS = "analysis"
    META_ANALYSIS = "meta_analysis"


ExecutionStatus = _enum("ExecutionStatus", "planned running success failed skipped")
ContractStatus = _enum("ContractStatus", "not_evaluated passed failed not_applicable")
EffectStatus = _enum(
    "EffectStatus", "not_evaluated estimable non_estimable invalidated"
)
EvidenceMaturity = _enum(
    "EvidenceMaturity", "contract_only diagnostic ablation confirmatory"
)
PromotionStatus = _enum("PromotionStatus", "ineligible eligible promoted")


class StatusSchemaError(ValueError):
    """Raised when a status cannot be represented by its schema contract."""


_FIELDS: dict[str, type[StrEnum]] = {
    "execution": ExecutionStatus,
    "contract": ContractStatus,
    "effect": EffectStatus,
    "evidence_maturity": EvidenceMaturity,
    "promotion": PromotionStatus,
}
_STATUS_KEYS = frozenset({"schema_version", *_FIELDS})


def status_v2(*values: StrEnum) -> dict[str, Any]:
    """Build a strict schema-v2 status from its canonical enum members."""

    try:
        items = tuple(zip(_FIELDS.items(), values, strict=True))
    except ValueError as exc:
        raise StatusSchemaError("status schema-v2 requires every status field") from exc
    serialized: dict[str, object] = {"schema_version": STATUS_SCHEMA_VERSION}
    for (field, enum), value in items:
        if type(value) is not enum:
            raise StatusSchemaError(f"{field} must use its canonical enum member")
        serialized[field] = value.value
    return validate_status_v2(serialized)


def planned_status() -> dict[str, Any]:
    return status_v2(
        ExecutionStatus.PLANNED,
        ContractStatus.NOT_EVALUATED,
        EffectStatus.NOT_EVALUATED,
        EvidenceMaturity.CONTRACT_ONLY,
        PromotionStatus.INELIGIBLE,
    )


def running_status() -> dict[str, Any]:
    return status_v2(
        ExecutionStatus.RUNNING,
        ContractStatus.NOT_EVALUATED,
        EffectStatus.NOT_EVALUATED,
        EvidenceMaturity.CONTRACT_ONLY,
        PromotionStatus.INELIGIBLE,
    )


def failed_status() -> dict[str, Any]:
    return status_v2(
        ExecutionStatus.FAILED,
        ContractStatus.FAILED,
        EffectStatus.INVALIDATED,
        EvidenceMaturity.CONTRACT_ONLY,
        PromotionStatus.INELIGIBLE,
    )


def succeeded_status() -> dict[str, Any]:
    return status_v2(
        ExecutionStatus.SUCCESS,
        ContractStatus.PASSED,
        EffectStatus.NON_ESTIMABLE,
        EvidenceMaturity.CONTRACT_ONLY,
        PromotionStatus.INELIGIBLE,
    )


def skipped_status() -> dict[str, Any]:
    return status_v2(
        ExecutionStatus.SKIPPED,
        ContractStatus.NOT_APPLICABLE,
        EffectStatus.NON_ESTIMABLE,
        EvidenceMaturity.CONTRACT_ONLY,
        PromotionStatus.INELIGIBLE,
    )


def first_gate_status(axis_id: str) -> dict[str, Any]:
    return skipped_status() if axis_id == "A10" else succeeded_status()


def is_legacy_status(value: Any) -> bool:
    return isinstance(value, str) or (
        isinstance(value, Mapping)
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
    )


def validate_status_v2(value: Any) -> dict[str, Any]:
    if is_legacy_status(value):
        raise StatusSchemaError("legacy status schema v1 is inspectable only")
    if not isinstance(value, Mapping) or set(value) != _STATUS_KEYS:
        raise StatusSchemaError("status schema-v2 fields are malformed")
    if type(value.get("schema_version")) is not int:
        raise StatusSchemaError("status schema-v2 version must be an exact integer")
    if value["schema_version"] != STATUS_SCHEMA_VERSION:
        raise StatusSchemaError("status schema-v2 version is unsupported")
    parsed: dict[str, StrEnum] = {}
    for field, enum in _FIELDS.items():
        raw = value[field]
        if type(raw) is not str:
            raise StatusSchemaError("status schema-v2 fields must be exact strings")
        try:
            parsed[field] = enum(raw)
        except ValueError as exc:
            raise StatusSchemaError(
                "status schema-v2 contains an unknown enum"
            ) from exc
    execution = parsed["execution"]
    contract = parsed["contract"]
    effect = parsed["effect"]
    maturity = parsed["evidence_maturity"]
    promotion = parsed["promotion"]
    pending = execution in {ExecutionStatus.PLANNED, ExecutionStatus.RUNNING}
    invalid = pending != (
        contract is ContractStatus.NOT_EVALUATED
        and effect is EffectStatus.NOT_EVALUATED
    )
    invalid |= execution is ExecutionStatus.SUCCESS and (
        contract is not ContractStatus.PASSED
        or effect not in {EffectStatus.NON_ESTIMABLE, EffectStatus.ESTIMABLE}
    )
    invalid |= execution is ExecutionStatus.FAILED and (
        contract is not ContractStatus.FAILED or effect is not EffectStatus.INVALIDATED
    )
    invalid |= execution is ExecutionStatus.SKIPPED and (
        contract is not ContractStatus.NOT_APPLICABLE
        or effect is not EffectStatus.NON_ESTIMABLE
    )
    invalid |= maturity is EvidenceMaturity.CONTRACT_ONLY and (
        effect is EffectStatus.ESTIMABLE
    )
    invalid |= (
        promotion is PromotionStatus.PROMOTED
        and maturity is not EvidenceMaturity.CONFIRMATORY
    )
    invalid |= (
        promotion is not PromotionStatus.INELIGIBLE
        and effect is not EffectStatus.ESTIMABLE
    )
    invalid |= (
        execution is not ExecutionStatus.SUCCESS
        and promotion is not PromotionStatus.INELIGIBLE
    )
    if invalid:
        raise StatusSchemaError("status combination violates the evidence lattice")
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        **{field: item.value for field, item in parsed.items()},
    }


def validate_writer_status(
    value: Any,
    *,
    writer: StatusWriter,
    axis_id: str | None = None,
) -> dict[str, Any]:
    """Validate a status against the execution states its writer may emit."""

    if type(writer) is not StatusWriter:
        raise StatusSchemaError("status writer must be a StatusWriter member")
    status = validate_status_v2(value)
    execution = status["execution"]
    if writer is StatusWriter.CAMPAIGN:
        permitted = {"running", "failed", "success"}
    elif writer is StatusWriter.AXIS:
        if not isinstance(axis_id, str) or not axis_id:
            raise StatusSchemaError("axis writer requires a non-empty axis_id")
        permitted = (
            {"planned", "running", "failed", "skipped"}
            if axis_id == "A10"
            else {"planned", "running", "failed", "success"}
        )
    elif writer in {StatusWriter.ANALYSIS, StatusWriter.META_ANALYSIS}:
        if execution != "success" or status["effect"] != "non_estimable":
            raise StatusSchemaError(
                f"{writer.value} writer requires a non-estimable success status"
            )
        return status
    else:
        raise StatusSchemaError("status writer is unsupported")
    if execution not in permitted:
        raise StatusSchemaError(
            f"{writer.value} status writer cannot emit execution status {execution!r}"
        )
    return status


def validate_axis_status(axis_id: str, value: Any) -> dict[str, Any]:
    status = validate_status_v2(value)
    if status != first_gate_status(axis_id):
        raise StatusSchemaError(f"axis status does not match {axis_id}")
    return status


def validate_campaign_status(value: Any) -> dict[str, Any]:
    status = validate_status_v2(value)
    if status != first_gate_status("campaign"):
        raise StatusSchemaError("campaign status is not exact success")
    return status


def transition_status(execution: StrEnum) -> dict[str, Any]:
    if type(execution) is not ExecutionStatus:
        raise StatusSchemaError(
            "transition execution must be an ExecutionStatus member"
        )
    if execution is ExecutionStatus.PLANNED:
        return planned_status()
    if execution is ExecutionStatus.RUNNING:
        return running_status()
    if execution is ExecutionStatus.FAILED:
        return failed_status()
    if execution is ExecutionStatus.SUCCESS:
        return succeeded_status()
    if execution is ExecutionStatus.SKIPPED:
        return skipped_status()
    raise StatusSchemaError("transition execution is unsupported")


def is_success(value: Any) -> bool:
    return validate_status_v2(value)["execution"] == ExecutionStatus.SUCCESS


def is_terminal_axis_status(value: Any, *, axis_id: str) -> bool:
    status = validate_writer_status(
        value,
        writer=StatusWriter.AXIS,
        axis_id=axis_id,
    )
    return status["execution"] in {"success", "skipped"}


def is_resume_eligible_campaign_status(value: Any) -> bool:
    return (
        validate_writer_status(value, writer=StatusWriter.CAMPAIGN)["execution"]
        == "failed"
    )


def is_resumable(value: Any, *, axis_id: str | None = None) -> bool:
    """Compatibility alias for terminal axis artifact semantics."""

    status = validate_status_v2(value)
    resolved_axis_id = axis_id
    if resolved_axis_id is None:
        resolved_axis_id = "A10" if status["execution"] == "skipped" else "A01"
    return is_terminal_axis_status(status, axis_id=resolved_axis_id)


def is_poolable(value: Any) -> bool:
    status = validate_status_v2(value)
    return (
        is_success(status)
        and status["effect"] == EffectStatus.ESTIMABLE
        and status["evidence_maturity"]
        in {EvidenceMaturity.ABLATION, EvidenceMaturity.CONFIRMATORY}
    )
