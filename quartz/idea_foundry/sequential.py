"""Fail-closed sequential runner for all 26 idea-foundry first gates."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from quartz.experiment_manifest import atomic_json_dump, file_sha256
from quartz.idea_foundry.axis_workflow import (
    REPO_ROOT,
    AxisWorkflowError,
    load_json_strict,
    load_workflow_specs,
    validate_axis_analysis,
)
from quartz.idea_foundry.campaign_state import (
    CAMPAIGN_STATE_SCHEMA_VERSION,
    build_initial_state_plan,
    publish_initial_run,
    validate_campaign_state_v2,
)
from quartz.idea_foundry.execution_identity import (
    ExecutionIdentityError,
    capture_execution_identity,
)
from quartz.idea_foundry.resume import (
    ResumeError,
    apply_resume_plan,
    execution_identity_from_payload,
    plan_resume,
    validate_captured_axis_analysis,
)
from quartz.idea_foundry.status_schema import (
    STATUS_SCHEMA_PATH,
    ExecutionStatus,
    StatusSchemaError,
    first_gate_status,
    is_resumable,
    transition_status,
    validate_axis_status,
    validate_status_v2,
)

SEQUENTIAL_SCHEMA_VERSION = 1
DEFAULT_CAMPAIGN_ROOT = REPO_ROOT / "results" / "idea_foundry_sequential"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")


class SequentialCampaignError(RuntimeError):
    """Raised when a campaign cannot proceed without violating its contract."""


@dataclass(frozen=True)
class _ExecutionAxisDescriptor:
    axis_id: str
    script_path: Path


def _captured_execution_descriptors(
    specs: Sequence[Any],
) -> tuple[_ExecutionAxisDescriptor, ...]:
    """Freeze executable descriptors from one captured workflow specification set."""

    descriptors = tuple(
        _ExecutionAxisDescriptor(spec.axis_id, spec.script_path.resolve())
        for spec in specs
    )
    if not descriptors or len({item.axis_id for item in descriptors}) != len(
        descriptors
    ):
        raise SequentialCampaignError("captured workflow descriptors are malformed")
    return descriptors


def _persisted_execution_descriptors(
    state: Mapping[str, Any],
) -> tuple[_ExecutionAxisDescriptor, ...]:
    """Reconstruct execution descriptors only from persisted campaign axes."""

    axes = state.get("axes")
    if not isinstance(axes, list):
        raise SequentialCampaignError("campaign axes must be a list")
    descriptors: list[_ExecutionAxisDescriptor] = []
    for row in axes:
        if not isinstance(row, Mapping):
            raise SequentialCampaignError("campaign axis descriptor is malformed")
        axis_id, slug = row.get("axis_id"), row.get("slug")
        if (
            type(axis_id) is not str
            or not re.fullmatch(r"A[0-9]{2}", axis_id)
            or type(slug) is not str
            or not re.fullmatch(r"[a-z0-9_]+", slug)
        ):
            raise SequentialCampaignError("campaign axis descriptor is malformed")
        descriptors.append(
            _ExecutionAxisDescriptor(
                axis_id,
                REPO_ROOT / "scripts" / "idea_foundry" / f"{axis_id.lower()}_{slug}.py",
            )
        )
    if not descriptors or len({item.axis_id for item in descriptors}) != len(
        descriptors
    ):
        raise SequentialCampaignError("campaign axis descriptors are malformed")
    return tuple(descriptors)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_campaign_root(raw_root: Path) -> Path:
    root = raw_root.resolve()
    allowed = (REPO_ROOT / "results").resolve()
    try:
        root.relative_to(allowed)
    except ValueError as exc:
        raise SequentialCampaignError(
            f"campaign root must remain under {allowed}: {root}"
        ) from exc
    return root


def resolve_run_root(campaign_root: Path, run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise SequentialCampaignError(f"unsafe run id: {run_id!r}")
    root = resolve_campaign_root(campaign_root)
    run_root = (root / run_id).resolve()
    try:
        run_root.relative_to(root)
    except ValueError as exc:
        raise SequentialCampaignError("run id escapes the campaign root") from exc
    return run_root


def _fingerprint(entrypoint: Path) -> dict[str, Any]:
    specs = load_workflow_specs()
    paths = [
        REPO_ROOT / "configs" / "idea_foundry.axes.v1.json",
        REPO_ROOT / "configs" / "idea_lab.local.v2.json",
        REPO_ROOT / "quartz" / "idea_foundry" / "axis_workflow.py",
        STATUS_SCHEMA_PATH,
        Path(__file__).resolve(),
        entrypoint.resolve(),
        *(spec.script_path for spec in specs),
    ]
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "axis_order": [spec.axis_id for spec in specs],
        "sources": [
            {
                "path": str(path.resolve().relative_to(REPO_ROOT)),
                "sha256": file_sha256(path),
            }
            for path in sorted(set(paths))
        ],
    }


def campaign_plan(entrypoint: Path) -> dict[str, Any]:
    specs = load_workflow_specs()
    return {
        "schema_version": SEQUENTIAL_SCHEMA_VERSION,
        "suite": "first-gate-all-sequential",
        "claim_scope": "synthetic_contract_execution_only",
        "axis_count": len(specs),
        "axes": [
            {
                "order_index": spec.order_index,
                "axis_id": spec.axis_id,
                "slug": spec.slug,
                "lane_id": spec.lane_id,
                "role": spec.role,
                "script": str(spec.script_path.relative_to(REPO_ROOT)),
                "analysis": "per-axis contract diagnostics only",
            }
            for spec in specs
        ],
        "fingerprint": _fingerprint(entrypoint),
    }


def _new_state(run_id: str, seed: int, entrypoint: Path) -> dict[str, Any]:
    now = utc_now()
    specs = load_workflow_specs()
    return {
        "schema_version": SEQUENTIAL_SCHEMA_VERSION,
        "run_id": run_id,
        "suite": "first-gate-all-sequential",
        "status": transition_status(ExecutionStatus.RUNNING),
        "seed": seed,
        "created_at": now,
        "updated_at": now,
        "fingerprint": _fingerprint(entrypoint),
        "claim_scope": "synthetic_contract_execution_only",
        "axes": [
            {
                "order_index": spec.order_index,
                "axis_id": spec.axis_id,
                "slug": spec.slug,
                "lane_id": spec.lane_id,
                "role": spec.role,
                "status": transition_status(ExecutionStatus.PLANNED),
                "attempts": [],
            }
            for spec in specs
        ],
    }


def _save_state(path: Path, state: dict[str, Any]) -> None:
    candidate = copy.deepcopy(state)
    candidate["updated_at"] = utc_now()
    if candidate.get("schema_version") == CAMPAIGN_STATE_SCHEMA_VERSION:
        normalized = validate_campaign_state_v2(candidate)
        state.clear()
        state.update(normalized)
    else:
        state["updated_at"] = candidate["updated_at"]
    atomic_json_dump(path, state)


def _validate_state(
    state: Any, run_id: str, seed: int, entrypoint: Path
) -> dict[str, Any]:
    if (
        isinstance(state, dict)
        and state.get("schema_version") == CAMPAIGN_STATE_SCHEMA_VERSION
    ):
        validated = validate_campaign_state_v2(state, expected_run_id=run_id)
        if validated.get("seed") != seed:
            raise SequentialCampaignError("run identity or seed changed")
        return validated
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != SEQUENTIAL_SCHEMA_VERSION
    ):
        raise SequentialCampaignError("campaign state schema mismatch")
    axes = state.get("axes")
    expected = [spec.axis_id for spec in load_workflow_specs()]
    if not isinstance(axes, list) or not all(isinstance(row, Mapping) for row in axes):
        raise SequentialCampaignError("campaign axes must be a list of objects")
    if [row.get("axis_id") for row in axes] != expected:
        raise SequentialCampaignError("campaign axis order changed")
    if not all(isinstance(row.get("attempts"), list) for row in axes):
        raise SequentialCampaignError("campaign axis attempts must be lists")
    try:
        # fmt: off
        partial = [transition_status(item) for item in (ExecutionStatus.PLANNED, ExecutionStatus.RUNNING, ExecutionStatus.FAILED)]
        campaign_status = validate_status_v2(state.get("status"))
        if campaign_status not in (*partial[1:], first_gate_status("campaign")):
            raise StatusSchemaError("campaign status is not writer-representable")
        for axis_id, row in zip(expected, axes, strict=True):
            if validate_status_v2(row.get("status")) not in (*partial, first_gate_status(axis_id)):
                raise StatusSchemaError("axis status is not writer-representable")
        if campaign_status == first_gate_status("campaign") and any(row["status"] != first_gate_status(axis_id) for axis_id, row in zip(expected, axes, strict=True)):
            raise StatusSchemaError("successful campaign contains incomplete axes")
        # fmt: on
    except StatusSchemaError as exc:
        raise SequentialCampaignError(str(exc)) from exc
    if state.get("run_id") != run_id or state.get("seed") != seed:
        raise SequentialCampaignError("resume run identity or seed changed")
    if state.get("fingerprint") != _fingerprint(entrypoint):
        raise SequentialCampaignError(
            "resume refused: registry, source, or interpreter hash changed"
        )
    return state


def _terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def _run_attempt(
    *,
    script_path: Path,
    output_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    seed: int,
    timeout_seconds: float | None,
) -> tuple[int, str]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script_path),
        "run-and-analyze",
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
    ]
    with (
        stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_handle,
        stderr_path.open("w", encoding="utf-8", buffering=1) as stderr_handle,
    ):
        previous_handlers: dict[int, Any] = {}
        proc: subprocess.Popen[Any] | None = None

        def _interrupt_handler(signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt(f"received signal {signum}")

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, _interrupt_handler)
            proc = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
            )
            return proc.wait(timeout=timeout_seconds), "completed"
        except OSError as exc:
            stderr_handle.write(
                f"subprocess launch failed: {type(exc).__name__}: {exc}\n"
            )
            return 126, "failed"
        except subprocess.TimeoutExpired:
            assert proc is not None
            _terminate_process_group(proc)
            return 124, "timeout"
        except KeyboardInterrupt:
            if proc is not None:
                _terminate_process_group(proc)
            return 130, "interrupted"
        except BaseException:
            if proc is not None:
                _terminate_process_group(proc)
            raise
        finally:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)


def _validated_attempt(
    axis_id: str, run_root: Path, axis_row: Mapping[str, Any]
) -> bool:
    try:
        status = validate_axis_status(axis_id, axis_row.get("status"))
        resumable = is_resumable(status)
    except StatusSchemaError:
        return False
    if not resumable:
        return False
    raw_attempt = axis_row.get("current_attempt")
    if not isinstance(raw_attempt, str):
        return False
    attempt_dir = (run_root / raw_attempt).resolve()
    try:
        attempt_dir.relative_to(run_root.resolve())
    except ValueError:
        return False
    try:
        analysis = validate_axis_analysis(
            axis_id, input_dir=attempt_dir, analysis_dir=attempt_dir / "analysis"
        )
    except AxisWorkflowError:
        return False
    return analysis["status"] == status


def _campaign_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    axes = state["axes"]
    status_counts = {
        status: sum(row["status"]["execution"] == status for row in axes)
        for status in sorted({row["status"]["execution"] for row in axes})
    }
    return {
        "schema_version": SEQUENTIAL_SCHEMA_VERSION,
        "run_id": state["run_id"],
        "suite": state["suite"],
        "status": state["status"],
        "axis_count": len(axes),
        "status_counts": status_counts,
        "axes": [
            {
                "axis_id": row["axis_id"],
                "lane_id": row["lane_id"],
                "role": row["role"],
                "status": row["status"],
                "current_attempt": row.get("current_attempt"),
                "attempt_count": len(row.get("attempts", [])),
            }
            for row in axes
        ],
        "claim_scope": "synthetic_contract_execution_only",
        "prohibited_inferences": [
            "play_strength",
            "efficacy",
            "production_readiness",
            "cross_axis_effect_pooling_without_a_shared_estimand",
        ],
    }


def run_campaign(
    *,
    campaign_root: Path,
    run_id: str,
    seed: int,
    timeout_seconds: float | None,
    resume: bool,
    entrypoint: Path,
) -> dict[str, Any]:
    run_root = resolve_run_root(campaign_root, run_id)
    state_path = run_root / "campaign_state.json"
    if resume:
        try:
            state_bytes = state_path.read_bytes()
            identity_payload = load_json_strict(run_root / "execution_identity.json")
            identity = execution_identity_from_payload(identity_payload)
            captured_axes_payload = (
                REPO_ROOT / "configs" / "idea_foundry.axes.v1.json"
            ).read_bytes()
            captured_lab_payload = (
                REPO_ROOT / "configs" / "idea_lab.local.v2.json"
            ).read_bytes()
        except (OSError, AxisWorkflowError, ResumeError) as exc:
            raise SequentialCampaignError(f"resume capture failed: {exc}") from exc
        try:
            current_capture = capture_execution_identity(
                repo_root=REPO_ROOT,
                entrypoint=entrypoint,
                argv=identity.runtime.argv,
            )
        except ExecutionIdentityError as exc:
            raise SequentialCampaignError(
                f"resume source capture failed: {exc}"
            ) from exc
        if (
            current_capture.identity.git_head != identity.git_head
            or current_capture.identity.source_files != identity.source_files
            or current_capture.axis_registry_bytes != captured_axes_payload
            or current_capture.lab_registry_bytes != captured_lab_payload
        ):
            raise SequentialCampaignError("resume source or registry identity drifted")
        try:
            resume_plan = plan_resume(
                run_root=run_root,
                state_bytes=state_bytes,
                identity=identity,
                captured_axes_payload=captured_axes_payload,
                captured_lab_payload=captured_lab_payload,
                run_id=run_id,
                seed=seed,
            )
            state = apply_resume_plan(state_path=state_path, plan=resume_plan)
        except ResumeError as exc:
            raise SequentialCampaignError(f"resume blocked: {exc}") from exc
        descriptors = _persisted_execution_descriptors(state)
    else:
        capture = capture_execution_identity(
            repo_root=REPO_ROOT,
            entrypoint=entrypoint,
            argv=("run", "--run-id", run_id, "--seed", str(seed)),
        )
        plan = build_initial_state_plan(run_id, seed, capture, utc_now())
        state = publish_initial_run(run_root, plan)
        descriptors = _captured_execution_descriptors(capture.specs)
    for axis_index, descriptor in enumerate(descriptors):
        axis_rows = state.get("axes")
        if not isinstance(axis_rows, list) or axis_index >= len(axis_rows):
            raise SequentialCampaignError(
                "campaign axis state changed during execution"
            )
        axis_row = axis_rows[axis_index]
        if not isinstance(axis_row, dict):
            raise SequentialCampaignError("campaign axis row is malformed")
        if resume and axis_row.get("resume_action") == "verified_skip":
            continue

        active_retry = bool(
            resume
            and axis_row.get("status", {}).get("execution") == "running"
            and axis_row.get("attempts")
            and axis_row.get("current_attempt")
            == axis_row["attempts"][-1].get("output_dir")
            and axis_row["attempts"][-1].get("process_outcome") == "running"
        )
        if active_retry:
            attempt = axis_row["attempts"][-1]
            attempt_number = int(attempt["attempt_number"])
            relative_attempt = Path(str(attempt["output_dir"]))
            attempt_dir = run_root / relative_attempt
            stdout_path = run_root / str(attempt["stdout"])
            stderr_path = run_root / str(attempt["stderr"])
        else:
            attempt_number = len(axis_row.get("attempts", [])) + 1
            relative_attempt = (
                Path("axes") / descriptor.axis_id / f"attempt-{attempt_number:03d}"
            )
            attempt_dir = run_root / relative_attempt
            stdout_path = (
                run_root
                / "logs"
                / f"{descriptor.axis_id}.attempt-{attempt_number:03d}.stdout.log"
            )
            stderr_path = (
                run_root
                / "logs"
                / f"{descriptor.axis_id}.attempt-{attempt_number:03d}.stderr.log"
            )
            attempt = {
                "attempt_number": attempt_number,
                "started_at": utc_now(),
                "output_dir": str(relative_attempt),
                "stdout": str(stdout_path.relative_to(run_root)),
                "stderr": str(stderr_path.relative_to(run_root)),
                "process_outcome": "running",
            }
            axis_row.setdefault("attempts", []).append(attempt)
            axis_row["status"] = transition_status(ExecutionStatus.RUNNING)
            axis_row["current_attempt"] = str(relative_attempt)
            axis_row.pop("resume_action", None)
            _save_state(state_path, state)
        axis_rows = state.get("axes")
        if not isinstance(axis_rows, list) or axis_index >= len(axis_rows):
            raise SequentialCampaignError(
                "campaign axis state changed during execution"
            )
        axis_row = axis_rows[axis_index]
        if not isinstance(axis_row, dict) or not axis_row.get("attempts"):
            raise SequentialCampaignError("campaign attempt state is malformed")
        attempt = axis_row["attempts"][-1]
        if not isinstance(attempt, dict):
            raise SequentialCampaignError("campaign attempt record is malformed")
        returncode, process_status = _run_attempt(
            script_path=descriptor.script_path,
            output_dir=attempt_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
        attempt["completed_at"] = utc_now()
        attempt["returncode"] = returncode
        attempt["process_outcome"] = process_status
        if returncode == 0:
            try:
                if resume:
                    analysis = validate_captured_axis_analysis(
                        descriptor.axis_id,
                        input_dir=attempt_dir,
                        analysis_dir=attempt_dir / "analysis",
                        captured_axes_payload=captured_axes_payload,
                        captured_lab_payload=captured_lab_payload,
                    )
                else:
                    analysis = validate_axis_analysis(
                        descriptor.axis_id,
                        input_dir=attempt_dir,
                        analysis_dir=attempt_dir / "analysis",
                    )
            except (AxisWorkflowError, ResumeError) as exc:
                returncode = 2
                attempt["returncode"] = returncode
                attempt["process_outcome"] = "failed"
                attempt["failure_reason"] = str(exc)
            else:
                axis_row["status"] = analysis["status"]
        if returncode != 0:
            axis_row["status"] = transition_status(ExecutionStatus.FAILED)
            axis_row["failure_reason"] = attempt.get(
                "failure_reason", f"axis subprocess exited with {returncode}"
            )
            state["status"] = transition_status(ExecutionStatus.FAILED)
            _save_state(state_path, state)
            atomic_json_dump(
                run_root / "campaign_summary.json", _campaign_summary(state)
            )
            raise SequentialCampaignError(
                f"{descriptor.axis_id} stopped campaign: {axis_row['failure_reason']}"
            )
        _save_state(state_path, state)

    state["status"] = first_gate_status("campaign")
    state["completed_at"] = utc_now()
    _save_state(state_path, state)
    summary = _campaign_summary(state)
    atomic_json_dump(run_root / "campaign_summary.json", summary)
    return summary


def sequential_main(argv: Sequence[str] | None = None, *, entrypoint: Path) -> int:
    parser = argparse.ArgumentParser(
        description="Run A01--A26 first gates sequentially"
    )
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--json", action="store_true")
    for name in ("run", "resume"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--run-id", required=True)
        command_parser.add_argument("--seed", type=int, default=20260718)
        command_parser.add_argument("--timeout-seconds", type=float, default=None)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            payload = campaign_plan(entrypoint)
        elif args.command == "status":
            run_root = resolve_run_root(args.campaign_root, args.run_id)
            payload = load_json_strict(run_root / "campaign_state.json")
        else:
            if args.timeout_seconds is not None and args.timeout_seconds <= 0:
                raise SequentialCampaignError("timeout must be positive")
            payload = run_campaign(
                campaign_root=args.campaign_root,
                run_id=args.run_id,
                seed=args.seed,
                timeout_seconds=args.timeout_seconds,
                resume=args.command == "resume",
                entrypoint=entrypoint,
            )
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (AxisWorkflowError, ResumeError, SequentialCampaignError) as exc:
        print(f"SEQUENTIAL CAMPAIGN BLOCKED: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "DEFAULT_CAMPAIGN_ROOT",
    "SEQUENTIAL_SCHEMA_VERSION",
    "SequentialCampaignError",
    "campaign_plan",
    "resolve_run_root",
    "run_campaign",
    "sequential_main",
]
