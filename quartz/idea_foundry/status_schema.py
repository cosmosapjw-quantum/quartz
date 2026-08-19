"""Canonical status-schema-v2 contract for Idea Foundry artifacts."""

from enum import StrEnum
from typing import Any, Mapping

# fmt: off
STATUS_SCHEMA_VERSION = 2

def _enum(name: str, values: str) -> type[StrEnum]:
    return StrEnum(name, {value.upper(): value for value in values.split()})

ExecutionStatus = _enum("ExecutionStatus", "planned running success failed skipped")
ContractStatus = _enum("ContractStatus", "not_evaluated passed failed not_applicable")
EffectStatus = _enum("EffectStatus", "not_evaluated estimable non_estimable invalidated")
EvidenceMaturity = _enum("EvidenceMaturity", "contract_only diagnostic ablation confirmatory")
PromotionStatus = _enum("PromotionStatus", "ineligible eligible promoted")

class StatusSchemaError(ValueError):
    pass

_FIELDS = {
    "execution": ExecutionStatus,
    "contract": ContractStatus,
    "effect": EffectStatus,
    "evidence_maturity": EvidenceMaturity,
    "promotion": PromotionStatus,
}

def status_v2(*values: StrEnum) -> dict[str, Any]:
    return validate_status_v2({"schema_version": 2, **dict(zip(_FIELDS, map(str, values), strict=True))})

def first_gate_status(axis_id: str) -> dict[str, Any]:
    dormant = axis_id == "A10"
    return status_v2(
        ExecutionStatus.SKIPPED if dormant else ExecutionStatus.SUCCESS,
        ContractStatus.NOT_APPLICABLE if dormant else ContractStatus.PASSED,
        EffectStatus.NON_ESTIMABLE, EvidenceMaturity.CONTRACT_ONLY,
        PromotionStatus.INELIGIBLE,
    )

def transition_status(execution: StrEnum) -> dict[str, Any]:
    failed = execution is ExecutionStatus.FAILED
    return status_v2(
        execution, ContractStatus.FAILED if failed else ContractStatus.NOT_EVALUATED,
        EffectStatus.INVALIDATED if failed else EffectStatus.NOT_EVALUATED,
        EvidenceMaturity.CONTRACT_ONLY, PromotionStatus.INELIGIBLE,
    )

def is_legacy_status(value: Any) -> bool:
    return isinstance(value, str) or (isinstance(value, Mapping) and value.get("schema_version") == 1)

def validate_status_v2(value: Any) -> dict[str, Any]:
    if is_legacy_status(value):
        raise StatusSchemaError("legacy status schema v1 is inspectable only")
    if not isinstance(value, Mapping) or set(value) != {"schema_version", *_FIELDS} or value.get("schema_version") != STATUS_SCHEMA_VERSION:
        raise StatusSchemaError("status schema-v2 fields are malformed")
    try:
        parsed = {key: enum(value[key]) for key, enum in _FIELDS.items()}
    except (TypeError, ValueError) as exc:
        raise StatusSchemaError("status schema-v2 contains an unknown enum") from exc
    execution, contract, effect, maturity, promotion = parsed.values()
    pending = execution in {ExecutionStatus.PLANNED, ExecutionStatus.RUNNING}
    invalid = pending != (contract is ContractStatus.NOT_EVALUATED and effect is EffectStatus.NOT_EVALUATED)
    invalid |= execution is ExecutionStatus.SUCCESS and (contract is not ContractStatus.PASSED or effect not in {EffectStatus.NON_ESTIMABLE, EffectStatus.ESTIMABLE})
    invalid |= execution is ExecutionStatus.FAILED and (contract is not ContractStatus.FAILED or effect is not EffectStatus.INVALIDATED)
    invalid |= execution is ExecutionStatus.SKIPPED and (contract is not ContractStatus.NOT_APPLICABLE or effect is not EffectStatus.NON_ESTIMABLE)
    invalid |= maturity is EvidenceMaturity.CONTRACT_ONLY and effect is EffectStatus.ESTIMABLE
    invalid |= promotion is PromotionStatus.PROMOTED and maturity is not EvidenceMaturity.CONFIRMATORY
    invalid |= promotion is not PromotionStatus.INELIGIBLE and effect is not EffectStatus.ESTIMABLE
    invalid |= execution is not ExecutionStatus.SUCCESS and promotion is not PromotionStatus.INELIGIBLE
    if invalid:
        raise StatusSchemaError("status combination violates the evidence lattice")
    return {"schema_version": 2, **{key: item.value for key, item in parsed.items()}}

def is_success(value: Any) -> bool:
    return validate_status_v2(value)["execution"] == ExecutionStatus.SUCCESS

def is_resumable(value: Any) -> bool:
    return validate_status_v2(value)["execution"] in {ExecutionStatus.SUCCESS, ExecutionStatus.SKIPPED}

def is_poolable(value: Any) -> bool:
    status = validate_status_v2(value)
    return is_success(status) and status["effect"] == EffectStatus.ESTIMABLE and status["evidence_maturity"] in {EvidenceMaturity.ABLATION, EvidenceMaturity.CONFIRMATORY}
# fmt: on
