"""Strict initial-state publication for sequential Idea Foundry campaigns."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quartz.experiment_manifest import atomic_json_dump, canonical_sha256
from quartz.idea_foundry.execution_identity import (
    ExecutionCapture,
    ExecutionIdentityError,
    validate_execution_identity_payload,
)
from quartz.idea_foundry.status_schema import (
    ExecutionStatus,
    StatusSchemaError,
    first_gate_status,
    planned_status,
    running_status,
    validate_writer_status,
    StatusWriter,
)


CAMPAIGN_STATE_SCHEMA_VERSION = 2
CAMPAIGN_SUITE = "first-gate-all-sequential"
CLAIM_SCOPE = "synthetic_contract_execution_only"
IDENTITY_FILENAME = "execution_identity.json"
STATE_FILENAME = "campaign_state.json"
_STATE_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "suite",
        "status",
        "seed",
        "created_at",
        "updated_at",
        "execution_identity",
        "claim_scope",
        "axes",
    }
)
_IDENTITY_REFERENCE_KEYS = frozenset({"path", "sha256"})
_STATE_OPTIONAL_KEYS = frozenset({"resumed_at", "completed_at"})
_AXIS_REQUIRED_KEYS = frozenset(
    {"order_index", "axis_id", "slug", "lane_id", "role", "status", "attempts"}
)
_AXIS_OPTIONAL_KEYS = frozenset({"current_attempt", "failure_reason", "resume_action"})
_PLAN_KEYS = frozenset({"campaign_state", "execution_identity"})


class CampaignStateError(RuntimeError):
    """Raised when initial campaign state cannot be published safely."""


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CampaignStateError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], keys: frozenset[str], label: str) -> None:
    if set(value) != keys:
        raise CampaignStateError(f"{label} fields are malformed")


def _allowed_keys(
    value: Mapping[str, object],
    required: frozenset[str],
    optional: frozenset[str],
    label: str,
) -> None:
    actual = set(value)
    if not required <= actual or actual - (required | optional):
        raise CampaignStateError(f"{label} fields are malformed")


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise CampaignStateError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise CampaignStateError(f"{label} must be an exact integer")
    return value


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CampaignStateError(f"{label} must be lowercase SHA-256")
    return digest


def _axis_from_spec(spec: Any) -> dict[str, object]:
    return {
        "order_index": spec.order_index,
        "axis_id": spec.axis_id,
        "slug": spec.slug,
        "lane_id": spec.lane_id,
        "role": spec.role,
        "status": planned_status(),
        "attempts": [],
    }


def build_initial_state_plan(
    run_id: str, seed: int, capture: ExecutionCapture, now: str
) -> dict[str, object]:
    """Build a state plan entirely from one already-captured execution identity."""

    if type(capture) is not ExecutionCapture:
        raise CampaignStateError("capture must be an ExecutionCapture")
    run_id = _string(run_id, "run_id")
    seed = _integer(seed, "seed")
    now = _string(now, "now")
    try:
        identity = validate_execution_identity_payload(capture.identity.to_payload())
    except ExecutionIdentityError as exc:
        raise CampaignStateError(f"invalid captured execution identity: {exc}") from exc
    identity_sha256 = _sha256(capture.identity_sha256, "capture.identity_sha256")
    if identity_sha256 != canonical_sha256(identity):
        raise CampaignStateError("capture identity digest does not match identity")
    specs = tuple(capture.specs)
    if not specs:
        raise CampaignStateError("capture specs must not be empty")
    state = {
        "schema_version": CAMPAIGN_STATE_SCHEMA_VERSION,
        "run_id": run_id,
        "suite": CAMPAIGN_SUITE,
        "status": running_status(),
        "seed": seed,
        "created_at": now,
        "updated_at": now,
        "execution_identity": {
            "path": IDENTITY_FILENAME,
            "sha256": identity_sha256,
        },
        "claim_scope": CLAIM_SCOPE,
        "axes": [_axis_from_spec(spec) for spec in specs],
    }
    return {
        "campaign_state": validate_campaign_state_v2(
            state, expected_run_id=run_id, expected_identity_sha256=identity_sha256
        ),
        "execution_identity": identity,
    }


def validate_campaign_state_v2(
    value: object,
    expected_run_id: str | None = None,
    expected_identity_sha256: str | None = None,
) -> dict[str, object]:
    """Return a copied, exact v2 state or reject it before any write occurs."""

    state = _mapping(value, "campaign state")
    _allowed_keys(state, _STATE_REQUIRED_KEYS, _STATE_OPTIONAL_KEYS, "campaign state")
    if _integer(state["schema_version"], "campaign state.schema_version") != 2:
        raise CampaignStateError("campaign state.schema_version is unsupported")
    run_id = _string(state["run_id"], "campaign state.run_id")
    if expected_run_id is not None and run_id != expected_run_id:
        raise CampaignStateError(
            "campaign state.run_id does not match expected identity"
        )
    if _string(state["suite"], "campaign state.suite") != CAMPAIGN_SUITE:
        raise CampaignStateError("campaign state.suite is unsupported")
    try:
        status = validate_writer_status(state["status"], writer=StatusWriter.CAMPAIGN)
    except StatusSchemaError as exc:
        raise CampaignStateError(str(exc)) from exc
    seed = _integer(state["seed"], "campaign state.seed")
    if seed < 0:
        raise CampaignStateError("campaign state.seed must be nonnegative")
    created_at = _string(state["created_at"], "campaign state.created_at")
    updated_at = _string(state["updated_at"], "campaign state.updated_at")
    reference = _mapping(state["execution_identity"], "execution identity reference")
    _exact_keys(reference, _IDENTITY_REFERENCE_KEYS, "execution identity reference")
    if (
        _string(reference["path"], "execution identity reference.path")
        != IDENTITY_FILENAME
    ):
        raise CampaignStateError("execution identity reference.path is unsupported")
    identity_sha256 = _sha256(
        reference["sha256"], "execution identity reference.sha256"
    )
    if (
        expected_identity_sha256 is not None
        and identity_sha256 != expected_identity_sha256
    ):
        raise CampaignStateError(
            "execution identity digest does not match expected identity"
        )
    if _string(state["claim_scope"], "campaign state.claim_scope") != CLAIM_SCOPE:
        raise CampaignStateError("campaign state.claim_scope is unsupported")
    axes = state["axes"]
    if type(axes) is not list or not axes:
        raise CampaignStateError("campaign state.axes must be a non-empty array")
    normalized_axes: list[dict[str, object]] = []
    expected_index = 0
    axis_ids: set[str] = set()
    for axis in axes:
        row = _mapping(axis, "campaign axis")
        _allowed_keys(row, _AXIS_REQUIRED_KEYS, _AXIS_OPTIONAL_KEYS, "campaign axis")
        order_index = _integer(row["order_index"], "campaign axis.order_index")
        if order_index != expected_index:
            raise CampaignStateError("campaign axis order is malformed")
        expected_index += 1
        axis_id = _string(row["axis_id"], "campaign axis.axis_id")
        if axis_id in axis_ids:
            raise CampaignStateError("campaign axis order is malformed")
        axis_ids.add(axis_id)
        try:
            axis_status = validate_writer_status(
                row["status"], writer=StatusWriter.AXIS, axis_id=axis_id
            )
        except StatusSchemaError as exc:
            raise CampaignStateError(str(exc)) from exc
        attempts = row["attempts"]
        if type(attempts) is not list or not all(
            isinstance(item, Mapping) for item in attempts
        ):
            raise CampaignStateError("campaign axis attempts must be arrays of objects")
        normalized_axis: dict[str, object] = {
            "order_index": order_index,
            "axis_id": axis_id,
            "slug": _string(row["slug"], "campaign axis.slug"),
            "lane_id": _string(row["lane_id"], "campaign axis.lane_id"),
            "role": _string(row["role"], "campaign axis.role"),
            "status": axis_status,
            "attempts": copy.deepcopy(attempts),
        }
        for key in _AXIS_OPTIONAL_KEYS:
            if key in row:
                normalized_axis[key] = _string(row[key], f"campaign axis.{key}")
        normalized_axes.append(normalized_axis)
    normalized: dict[str, object] = {
        "schema_version": CAMPAIGN_STATE_SCHEMA_VERSION,
        "run_id": run_id,
        "suite": CAMPAIGN_SUITE,
        "status": status,
        "seed": seed,
        "created_at": created_at,
        "updated_at": updated_at,
        "execution_identity": {"path": IDENTITY_FILENAME, "sha256": identity_sha256},
        "claim_scope": CLAIM_SCOPE,
        "axes": normalized_axes,
    }
    for key in _STATE_OPTIONAL_KEYS:
        if key in state:
            normalized[key] = _string(state[key], f"campaign state.{key}")
    executions = [row["status"]["execution"] for row in normalized_axes]
    if status["execution"] == ExecutionStatus.SUCCESS.value and any(
        row["status"] != first_gate_status(row["axis_id"]) for row in normalized_axes
    ):
        raise CampaignStateError("successful campaign contains incomplete axes")
    if (
        status["execution"] == ExecutionStatus.FAILED.value
        and "failed" not in executions
    ):
        raise CampaignStateError("failed campaign requires a failed axis")
    return normalized


def _validated_plan(value: object) -> tuple[dict[str, object], dict[str, object]]:
    plan = _mapping(value, "initial state plan")
    _exact_keys(plan, _PLAN_KEYS, "initial state plan")
    try:
        identity = validate_execution_identity_payload(plan["execution_identity"])
    except ExecutionIdentityError as exc:
        raise CampaignStateError(f"invalid captured execution identity: {exc}") from exc
    digest = canonical_sha256(identity)
    state = validate_campaign_state_v2(
        plan["campaign_state"], expected_identity_sha256=digest
    )
    if state["status"] != running_status() or any(
        row["status"] != planned_status() or row["attempts"] for row in state["axes"]
    ):
        raise CampaignStateError(
            "initial publication requires an untouched initial state"
        )
    return state, identity


def publish_initial_run(run_root: Path, plan: object) -> dict[str, object]:
    """Publish identity then state into a never-before-used campaign root."""

    state, identity = _validated_plan(copy.deepcopy(plan))
    root = Path(run_root)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise CampaignStateError(f"new campaign root already exists: {root}") from exc
    lock_path = root / ".writer.lock"
    try:
        lock_path.touch(exist_ok=False)
        atomic_json_dump(root / IDENTITY_FILENAME, identity)
        atomic_json_dump(root / STATE_FILENAME, state)
    finally:
        if lock_path.exists():
            lock_path.unlink()
    return state


__all__ = [
    "CAMPAIGN_STATE_SCHEMA_VERSION",
    "CampaignStateError",
    "build_initial_state_plan",
    "publish_initial_run",
    "validate_campaign_state_v2",
]
