"""Validation-first public resume planning for Idea Foundry campaigns."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from quartz.experiment_manifest import atomic_json_dump, canonical_sha256
from quartz.idea_foundry import axis_workflow as _axis_workflow
from quartz.idea_foundry.axis_workflow import (
    AxisWorkflowError,
    AxisWorkflowSpec,
    decode_json_strict,
    parse_workflow_specs,
    validate_axis_analysis,
)
from quartz.idea_foundry.campaign_state import (
    CAMPAIGN_STATE_SCHEMA_VERSION,
    CampaignStateError,
    validate_campaign_state_v2,
)
from quartz.idea_foundry.execution_identity import (
    EXECUTION_IDENTITY_SCHEMA_VERSION,
    ExecutionIdentity,
    FileIdentity,
    RuntimeIdentity,
    validate_execution_identity_payload,
)
from quartz.idea_foundry.status_schema import (
    failed_status,
    first_gate_status,
    running_status,
)


class ResumeError(RuntimeError):
    """Raised when a resume cannot be admitted without changing evidence."""


@dataclass(frozen=True)
class RetryAttemptPlan:
    axis_index: int
    axis_id: str
    attempt_number: int
    output_dir: str
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ResumePlan:
    schema_version: int
    run_id: str
    seed: int
    source_state_sha256: str
    terminal_prefix_axis_ids: tuple[str, ...]
    retry: RetryAttemptPlan
    next_state: dict[str, object]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def execution_identity_from_payload(value: object) -> ExecutionIdentity:
    """Reconstruct the strict dataclass used by a persisted identity payload."""

    try:
        payload = validate_execution_identity_payload(value)
    except Exception as exc:  # the public error boundary is ResumeError
        raise ResumeError(f"invalid persisted execution identity: {exc}") from exc
    try:

        def file_identity(row: Mapping[str, object]) -> FileIdentity:
            return FileIdentity(
                str(row["path"]), int(row["size_bytes"]), str(row["sha256"])
            )

        runtime_payload = payload["runtime"]
        assert isinstance(runtime_payload, Mapping)
        return ExecutionIdentity(
            schema_version=int(payload["schema_version"]),
            git_head=str(payload["git_head"]),
            git_dirty=bool(payload["git_dirty"]),
            axis_registry=file_identity(payload["axis_registry"]),
            lab_registry=file_identity(payload["lab_registry"]),
            source_files=tuple(
                file_identity(row)
                for row in payload["source_files"]  # type: ignore[arg-type]
            ),
            runtime=RuntimeIdentity(
                python_executable=str(runtime_payload["python_executable"]),
                python_version=str(runtime_payload["python_version"]),
                platform=str(runtime_payload["platform"]),
                argv=tuple(str(item) for item in runtime_payload["argv"]),  # type: ignore[arg-type]
            ),
        )
    except (KeyError, TypeError, ValueError, AssertionError) as exc:
        raise ResumeError("persisted execution identity fields are malformed") from exc


def _validate_captured_identity(
    identity: ExecutionIdentity,
    axis_payload: bytes,
    lab_payload: bytes,
) -> tuple[Any, ...]:
    if type(identity) is not ExecutionIdentity:
        raise ResumeError("resume identity must be an ExecutionIdentity")
    if (
        identity.schema_version != EXECUTION_IDENTITY_SCHEMA_VERSION
        or identity.git_dirty
    ):
        raise ResumeError("resume identity is not an accepted clean identity")
    payload = validate_execution_identity_payload(identity.to_payload())
    if canonical_sha256(payload) != _identity_digest(identity):
        raise ResumeError("resume identity digest is internally inconsistent")
    if identity.axis_registry.size_bytes != len(axis_payload):
        raise ResumeError("captured axis registry size drifted")
    if identity.lab_registry.size_bytes != len(lab_payload):
        raise ResumeError("captured lab registry size drifted")
    if identity.axis_registry.sha256 != _sha256_bytes(axis_payload):
        raise ResumeError("captured axis registry hash drifted")
    if identity.lab_registry.sha256 != _sha256_bytes(lab_payload):
        raise ResumeError("captured lab registry hash drifted")
    try:
        axes = decode_json_strict(axis_payload, label="captured axis registry")
        lab = decode_json_strict(lab_payload, label="captured lab registry")
        return parse_workflow_specs(axes, lab)
    except AxisWorkflowError as exc:
        raise ResumeError(f"captured workflow registry is invalid: {exc}") from exc


def _identity_digest(identity: ExecutionIdentity) -> str:
    return canonical_sha256(validate_execution_identity_payload(identity.to_payload()))


@contextmanager
def _captured_workflow_spec(spec: AxisWorkflowSpec) -> Iterator[None]:
    original = _axis_workflow.workflow_spec

    def captured(axis_id: str) -> AxisWorkflowSpec:
        if axis_id.upper() != spec.axis_id:
            raise AxisWorkflowError(f"captured registry does not contain {axis_id}")
        return spec

    _axis_workflow.workflow_spec = captured
    try:
        yield
    finally:
        _axis_workflow.workflow_spec = original


def _validate_axis_analysis_with_spec(
    spec: AxisWorkflowSpec, *, input_dir: Path, analysis_dir: Path
) -> dict[str, Any]:
    with _captured_workflow_spec(spec):
        return validate_axis_analysis(
            spec.axis_id, input_dir=input_dir, analysis_dir=analysis_dir
        )


def validate_captured_axis_analysis(
    axis_id: str,
    *,
    input_dir: Path,
    analysis_dir: Path,
    captured_axes_payload: bytes,
    captured_lab_payload: bytes,
) -> dict[str, Any]:
    """Validate an analysis against specs parsed from the captured registries."""

    try:
        specs = parse_workflow_specs(
            decode_json_strict(captured_axes_payload, label="captured axis registry"),
            decode_json_strict(captured_lab_payload, label="captured lab registry"),
        )
    except AxisWorkflowError as exc:
        raise ResumeError(f"captured workflow registry is invalid: {exc}") from exc
    normalized_axis_id = axis_id.upper()
    try:
        spec = next(item for item in specs if item.axis_id == normalized_axis_id)
    except StopIteration as exc:
        raise ResumeError(f"captured registry does not contain {axis_id}") from exc
    try:
        return _validate_axis_analysis_with_spec(
            spec, input_dir=input_dir, analysis_dir=analysis_dir
        )
    except AxisWorkflowError as exc:
        raise ResumeError(f"captured terminal analysis is invalid: {exc}") from exc


def _require_attempts(
    axis_id: str, row: Mapping[str, object]
) -> list[dict[str, object]]:
    attempts = row.get("attempts")
    if type(attempts) is not list or not attempts:
        raise ResumeError(f"{axis_id} resume requires prior attempts")
    normalized: list[dict[str, object]] = []
    for number, raw in enumerate(attempts, 1):
        if not isinstance(raw, Mapping):
            raise ResumeError(f"{axis_id} attempt {number} is malformed")
        expected_output = f"axes/{axis_id}/attempt-{number:03d}"
        expected_stdout = f"logs/{axis_id}.attempt-{number:03d}.stdout.log"
        expected_stderr = f"logs/{axis_id}.attempt-{number:03d}.stderr.log"
        if (
            raw.get("attempt_number") != number
            or raw.get("output_dir") != expected_output
            or raw.get("stdout") != expected_stdout
            or raw.get("stderr") != expected_stderr
            or type(raw.get("started_at")) is not str
            or not raw.get("started_at")
            or type(raw.get("completed_at")) is not str
            or not raw.get("completed_at")
            or type(raw.get("returncode")) is not int
            or raw.get("process_outcome") == "running"
        ):
            raise ResumeError(f"{axis_id} attempt {number} is not terminal")
        normalized.append(copy.deepcopy(dict(raw)))
    return normalized


def _check_axis_metadata(state: Mapping[str, object], specs: tuple[Any, ...]) -> None:
    axes = state.get("axes")
    if type(axes) is not list or len(axes) != len(specs):
        raise ResumeError("campaign axes do not match captured workflow registry")
    for index, (row, spec) in enumerate(zip(axes, specs, strict=True)):
        if not isinstance(row, Mapping) or {
            row.get("order_index"),
            row.get("axis_id"),
            row.get("slug"),
            row.get("lane_id"),
            row.get("role"),
        } != {index, spec.axis_id, spec.slug, spec.lane_id, spec.role}:
            raise ResumeError("campaign axis metadata does not match captured registry")


def _terminal_prefix_and_retry(
    *, state: dict[str, object], run_root: Path, specs: tuple[AxisWorkflowSpec, ...]
) -> tuple[tuple[str, ...], int, dict[str, object]]:
    axes = state["axes"]
    assert isinstance(axes, list)
    specs_by_id = {spec.axis_id: spec for spec in specs}
    terminal: list[str] = []
    failed: list[tuple[int, dict[str, object]]] = []
    failure_seen = False
    for index, raw in enumerate(axes):
        if not isinstance(raw, dict):
            raise ResumeError("campaign axis row is malformed")
        axis_id = raw.get("axis_id")
        status = raw.get("status")
        if not isinstance(axis_id, str) or not isinstance(status, Mapping):
            raise ResumeError("campaign axis identity is malformed")
        execution = status.get("execution")
        if execution in {"success", "skipped"}:
            if failure_seen:
                raise ResumeError("terminal axes must form a prefix")
            attempts = _require_attempts(axis_id, raw)
            current_attempt = raw.get("current_attempt")
            if current_attempt != attempts[-1]["output_dir"]:
                raise ResumeError(f"{axis_id} current attempt is malformed")
            try:
                analysis = _validate_axis_analysis_with_spec(
                    specs_by_id[axis_id],
                    input_dir=run_root / str(current_attempt),
                    analysis_dir=run_root / str(current_attempt) / "analysis",
                )
            except KeyError as exc:
                raise ResumeError(
                    f"captured registry does not contain {axis_id}"
                ) from exc
            except AxisWorkflowError as exc:
                raise ResumeError(
                    f"terminal prefix artifact validation failed: {exc}"
                ) from exc
            if analysis.get("status") != first_gate_status(axis_id):
                raise ResumeError("terminal prefix artifact status is invalid")
            terminal.append(axis_id)
        elif execution == "failed":
            failure_seen = True
            failed.append((index, raw))
            _require_attempts(axis_id, raw)
            if type(raw.get("failure_reason")) is not str or not raw["failure_reason"]:
                raise ResumeError("failed axis requires failure_reason")
        elif execution == "planned":
            if not failure_seen or raw.get("attempts"):
                raise ResumeError("planned axes must follow the failed axis")
            if "failure_reason" in raw:
                raise ResumeError("planned axis contains stale failure_reason")
        else:
            raise ResumeError("resume requires an inactive failed campaign")
    if len(failed) != 1:
        raise ResumeError("resume requires exactly one failed axis")
    return tuple(terminal), failed[0][0], failed[0][1]


def plan_resume(
    *,
    run_root: Path,
    state_bytes: bytes,
    identity: ExecutionIdentity,
    captured_axes_payload: bytes,
    captured_lab_payload: bytes,
    run_id: str,
    seed: int,
) -> ResumePlan:
    """Build one complete resume transition without filesystem mutation."""

    try:
        raw_state = decode_json_strict(state_bytes, label="campaign state")
        current = validate_campaign_state_v2(
            raw_state,
            expected_run_id=run_id,
            expected_identity_sha256=_identity_digest(identity),
        )
    except (AxisWorkflowError, CampaignStateError) as exc:
        raise ResumeError(f"resume state is invalid: {exc}") from exc
    if current["seed"] != seed or current["status"] != failed_status():
        raise ResumeError("only an inactive FAILED campaign may resume")
    specs = _validate_captured_identity(
        identity, captured_axes_payload, captured_lab_payload
    )
    _check_axis_metadata(current, specs)
    terminal_ids, failed_index, failed_axis = _terminal_prefix_and_retry(
        state=current, run_root=Path(run_root), specs=specs
    )
    axis_id = str(failed_axis["axis_id"])
    attempts = failed_axis["attempts"]
    assert isinstance(attempts, list)
    attempt_number = len(attempts) + 1
    output_dir = f"axes/{axis_id}/attempt-{attempt_number:03d}"
    stdout = f"logs/{axis_id}.attempt-{attempt_number:03d}.stdout.log"
    stderr = f"logs/{axis_id}.attempt-{attempt_number:03d}.stderr.log"
    now = _now()
    next_state = copy.deepcopy(current)
    next_state["status"] = running_status()
    next_state["resumed_at"] = now
    next_state["updated_at"] = now
    for row in next_state["axes"]:  # type: ignore[index]
        if row["axis_id"] in terminal_ids:
            row["resume_action"] = "verified_skip"
    retry_row = next_state["axes"][failed_index]  # type: ignore[index]
    retry_row["status"] = running_status()
    retry_row.pop("failure_reason", None)
    retry_row.pop("resume_action", None)
    retry_row["current_attempt"] = output_dir
    retry_row["attempts"].append(  # type: ignore[union-attr]
        {
            "attempt_number": attempt_number,
            "started_at": now,
            "output_dir": output_dir,
            "stdout": stdout,
            "stderr": stderr,
            "process_outcome": "running",
        }
    )
    try:
        validated_next = validate_campaign_state_v2(
            next_state,
            expected_run_id=run_id,
            expected_identity_sha256=_identity_digest(identity),
        )
    except CampaignStateError as exc:
        raise ResumeError(f"planned resume state is invalid: {exc}") from exc
    return ResumePlan(
        schema_version=CAMPAIGN_STATE_SCHEMA_VERSION,
        run_id=run_id,
        seed=seed,
        source_state_sha256=_sha256_bytes(state_bytes),
        terminal_prefix_axis_ids=terminal_ids,
        retry=RetryAttemptPlan(
            axis_index=failed_index,
            axis_id=axis_id,
            attempt_number=attempt_number,
            output_dir=output_dir,
            stdout=stdout,
            stderr=stderr,
        ),
        next_state=validated_next,
    )


@contextmanager
def _state_lock(state_path: Path) -> Iterator[Any]:
    try:
        fd = os.open(str(Path(state_path).parent), os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise ResumeError("campaign run root cannot be locked for resume") from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResumeError("campaign writer is already active") from exc
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _validate_resume_delta(
    current: Mapping[str, object],
    planned: Mapping[str, object],
    plan: ResumePlan,
) -> None:
    if current.get("status") != failed_status():
        raise ResumeError("resume source campaign is no longer FAILED")
    if planned.get("status") != running_status():
        raise ResumeError("resume plan must publish a RUNNING campaign")
    if current.get("seed") != plan.seed or planned.get("seed") != plan.seed:
        raise ResumeError("resume plan seed does not match the source campaign")
    if current.get("execution_identity") != planned.get("execution_identity"):
        raise ResumeError("resume plan execution identity is not source-bound")
    current_axes = current.get("axes")
    planned_axes = planned.get("axes")
    if not isinstance(current_axes, list) or not isinstance(planned_axes, list):
        raise ResumeError("resume plan axes are malformed")
    if len(current_axes) != len(planned_axes):
        raise ResumeError("resume plan changed campaign axis count")
    if not 0 <= plan.retry.axis_index < len(current_axes):
        raise ResumeError("resume plan retry index is out of range")
    terminal_ids = set(plan.terminal_prefix_axis_ids)
    if len(terminal_ids) != len(plan.terminal_prefix_axis_ids):
        raise ResumeError("resume plan terminal prefix contains duplicates")
    actual_terminal_prefix: list[str] = []
    for row in current_axes:
        if not isinstance(row, Mapping):
            raise ResumeError("resume source axis row is malformed")
        status = row.get("status")
        if not isinstance(status, Mapping):
            raise ResumeError("resume source axis status is malformed")
        if status.get("execution") in {"success", "skipped"}:
            actual_terminal_prefix.append(str(row["axis_id"]))
        else:
            break
    if tuple(actual_terminal_prefix) != plan.terminal_prefix_axis_ids:
        raise ResumeError("resume plan terminal prefix is not source-bound")
    appended_count = 0
    for index, (current_row, planned_row) in enumerate(
        zip(current_axes, planned_axes, strict=True)
    ):
        if not isinstance(current_row, Mapping) or not isinstance(planned_row, Mapping):
            raise ResumeError("resume plan axis row is malformed")
        if current_row.get("axis_id") != planned_row.get("axis_id"):
            raise ResumeError("resume plan changed axis identity")
        if (
            index == plan.retry.axis_index
            and current_row.get("axis_id") != plan.retry.axis_id
        ):
            raise ResumeError("resume plan retry axis identity is stale")
        if index != plan.retry.axis_index:
            if current_row.get("attempts") != planned_row.get("attempts"):
                raise ResumeError("resume plan changed a prior attempt")
            if current_row.get("axis_id") in terminal_ids:
                if planned_row.get(
                    "resume_action"
                ) != "verified_skip" or planned_row.get("status") != current_row.get(
                    "status"
                ):
                    raise ResumeError("resume plan lost a verified terminal skip")
            elif "resume_action" in planned_row:
                raise ResumeError("resume plan added an unexpected skip marker")
            continue
        old_attempts = current_row.get("attempts")
        new_attempts = planned_row.get("attempts")
        if (
            not isinstance(old_attempts, list)
            or not isinstance(new_attempts, list)
            or len(new_attempts) != len(old_attempts) + 1
            or new_attempts[:-1] != old_attempts
        ):
            raise ResumeError("resume plan does not preserve prior attempts")
        appended = new_attempts[-1]
        if not isinstance(appended, Mapping) or (
            appended.get("attempt_number") != len(old_attempts) + 1
            or appended.get("attempt_number") != plan.retry.attempt_number
            or appended.get("output_dir") != plan.retry.output_dir
            or appended.get("stdout") != plan.retry.stdout
            or appended.get("stderr") != plan.retry.stderr
            or appended.get("process_outcome") != "running"
        ):
            raise ResumeError("resume plan appended attempt is malformed")
        if (
            planned_row.get("status") != running_status()
            or "failure_reason" in planned_row
            or planned_row.get("current_attempt") != plan.retry.output_dir
        ):
            raise ResumeError("resume plan retry row is not canonical")
        appended_count += 1
    if appended_count != 1:
        raise ResumeError("resume plan must append exactly one retry")


def apply_resume_plan(*, state_path: Path, plan: ResumePlan) -> dict[str, object]:
    """Publish exactly one already-validated transition, or publish nothing."""

    if (
        type(plan) is not ResumePlan
        or plan.schema_version != CAMPAIGN_STATE_SCHEMA_VERSION
    ):
        raise ResumeError("resume plan schema is invalid")
    try:
        validated_next = validate_campaign_state_v2(
            copy.deepcopy(plan.next_state), expected_run_id=plan.run_id
        )
    except CampaignStateError as exc:
        raise ResumeError(f"resume plan state is invalid: {exc}") from exc
    with _state_lock(Path(state_path)):
        current_bytes = Path(state_path).read_bytes()
        if _sha256_bytes(current_bytes) != plan.source_state_sha256:
            raise ResumeError("campaign state changed before resume admission")
        try:
            current = decode_json_strict(current_bytes, label="campaign state")
            validate_campaign_state_v2(current, expected_run_id=plan.run_id)
        except (AxisWorkflowError, CampaignStateError) as exc:
            raise ResumeError(
                f"campaign state changed before resume admission: {exc}"
            ) from exc
        _validate_resume_delta(current, validated_next, plan)
        atomic_json_dump(Path(state_path), validated_next)
    return validated_next


__all__ = [
    "ResumeError",
    "RetryAttemptPlan",
    "ResumePlan",
    "execution_identity_from_payload",
    "plan_resume",
    "apply_resume_plan",
    "validate_captured_axis_analysis",
]
