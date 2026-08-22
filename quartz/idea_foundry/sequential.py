"""Fail-closed sequential runner for all 26 idea-foundry first gates."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
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
from quartz.idea_foundry.execution_seal import (
    ExecutionSealError,
    _exact_json,
    canonical_json_bytes,
    capture_execution_seal,
    claim_run_root,
    load_canonical_json,
    publish_execution_seal,
    read_execution_seal,
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

SEQUENTIAL_SCHEMA_VERSION = 2
DEFAULT_CAMPAIGN_ROOT = REPO_ROOT / "results" / "idea_foundry_sequential"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")


class SequentialCampaignError(RuntimeError):
    """Raised when a campaign cannot proceed without violating its contract."""


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


def _new_state(run_id: str, seed: int, execution_seal_sha256: str) -> dict[str, Any]:
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
        "execution_seal_sha256": execution_seal_sha256,
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
    state["updated_at"] = utc_now()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(state))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_state(
    state: Any, run_id: str, seed: int, seal_sha256: str
) -> dict[str, Any]:
    _exact_json(state)
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
    if state.get("execution_seal_sha256") != seal_sha256:
        raise SequentialCampaignError("resume refused: execution seal changed")
    _validate_state_shape(state)
    return state


# fmt: off
def _validate_state_shape(state: dict[str, Any]) -> None:
    def strict_status(value: Any) -> dict[str, Any]:
        if type(value) is not dict or set(value) != {"schema_version", "execution", "contract", "effect", "evidence_maturity", "promotion"} or type(value["schema_version"]) is not int or any(type(value[key]) is not str for key in set(value) - {"schema_version"}): raise SequentialCampaignError("status fields are malformed")
        return validate_status_v2(value)
    base = {"schema_version", "run_id", "suite", "status", "seed", "created_at", "updated_at", "execution_seal_sha256", "claim_scope", "axes"}
    optional = set(state) - base
    if set(state) - {"resumed_at", "completed_at"} != base or type(state["schema_version"]) is not int or state["schema_version"] != 2 or type(state["run_id"]) is not str or not state["run_id"] or type(state["seed"]) is not int or state["suite"] != "first-gate-all-sequential" or state["claim_scope"] != "synthetic_contract_execution_only" or type(state["execution_seal_sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", state["execution_seal_sha256"]):
        raise SequentialCampaignError("campaign state fields are malformed")
    if any(type(state.get(key)) is not str or not state[key] for key in ("created_at", "updated_at", *optional)):
        raise SequentialCampaignError("campaign lifecycle timestamp is malformed")
    running, failed = (transition_status(item) for item in (ExecutionStatus.RUNNING, ExecutionStatus.FAILED))
    status = strict_status(state["status"])
    expected_optional = {"completed_at"} if status == first_gate_status("campaign") else ({"resumed_at"} if "resumed_at" in state else set())
    if status not in (running, failed, first_gate_status("campaign")) or optional != expected_optional | ({"resumed_at"} if status == first_gate_status("campaign") and "resumed_at" in state else set()):
        raise SequentialCampaignError("campaign lifecycle is invalid")
    specs = load_workflow_specs(); phases = []
    for spec, row in zip(specs, state["axes"], strict=True):
        row_base = {"order_index", "axis_id", "slug", "lane_id", "role", "status", "attempts"}
        if type(row) is not dict or not row_base <= set(row) or set(row) - row_base - {"current_attempt", "resume_action", "failure_reason"} or [row.get(key) for key in ("order_index", "axis_id", "slug", "lane_id", "role")] != [spec.order_index, spec.axis_id, spec.slug, spec.lane_id, spec.role] or type(row["attempts"]) is not list:
            raise SequentialCampaignError("axis row fields are malformed")
        axis_status = strict_status(row["status"]); execution = axis_status["execution"]
        terminal = first_gate_status(spec.axis_id); allowed = (transition_status(ExecutionStatus.PLANNED), running, failed, terminal)
        if axis_status not in allowed: raise SequentialCampaignError("axis status is not writer-representable")
        attempts = row["attempts"]
        for number, attempt in enumerate(attempts, 1):
            fixed = {"attempt_number", "started_at", "output_dir", "stdout", "stderr", "process_outcome"}; extra = set(attempt) - fixed
            stem = f"{spec.axis_id}.attempt-{number:03d}"; output = f"axes/{spec.axis_id}/attempt-{number:03d}"
            if type(attempt) is not dict or not fixed <= set(attempt) or attempt["attempt_number"] != number or type(attempt["attempt_number"]) is not int or any(type(attempt[key]) is not str or not attempt[key] for key in fixed - {"attempt_number"}) or (attempt["output_dir"], attempt["stdout"], attempt["stderr"]) != (output, f"logs/{stem}.stdout.log", f"logs/{stem}.stderr.log"):
                raise SequentialCampaignError("attempt fields are malformed")
            outcome = attempt["process_outcome"]; active = outcome == "running"
            if active:
                if extra or number != len(attempts) or execution != "running": raise SequentialCampaignError("active attempt is invalid")
            else:
                reason = {"failure_reason"} if outcome == "failed" and attempt.get("returncode") == 2 else set()
                relations = {"completed": lambda code: code == 0 or code not in {0, 2, 124, 126, 130}, "failed": lambda code: code in {2, 126}, "timeout": lambda code: code == 124, "interrupted": lambda code: code == 130}
                if outcome not in relations or extra != {"completed_at", "returncode"} | reason or type(attempt.get("completed_at")) is not str or not attempt["completed_at"] or type(attempt.get("returncode")) is not int or not relations[outcome](attempt["returncode"]) or (reason and (type(attempt.get("failure_reason")) is not str or not attempt["failure_reason"])) or (number < len(attempts) and attempt["returncode"] == 0): raise SequentialCampaignError("finished attempt is invalid")
        current = f"axes/{spec.axis_id}/attempt-{len(attempts):03d}"
        if execution == "planned" and (attempts or set(row) != row_base): raise SequentialCampaignError("planned axis is invalid")
        if execution != "planned" and (not attempts or row.get("current_attempt") != current) or execution == "running" and attempts[-1].get("process_outcome") != "running": raise SequentialCampaignError("axis current attempt is invalid")
        if axis_status == terminal and attempts[-1].get("returncode") != 0 or execution == "failed" and attempts[-1].get("returncode") == 0: raise SequentialCampaignError("axis terminal attempt is invalid")
        if execution == "failed" and (type(row.get("failure_reason")) is not str or not row["failure_reason"]): raise SequentialCampaignError("failed axis lacks reason")
        if execution != "failed" and "failure_reason" in row: raise SequentialCampaignError("axis failure reason is misplaced")
        if "resume_action" in row and (row["resume_action"] != "verified_skip" or "resumed_at" not in state or axis_status != terminal): raise SequentialCampaignError("resume action is invalid")
        phases.append("terminal" if axis_status == terminal else execution)
    patterns = {"running": r"(?:terminal)*(?:running)?(?:planned)*", "failed": r"(?:terminal)*failed(?:planned)*", "success": r"(?:terminal)*"}
    if not re.fullmatch(patterns[status["execution"]], "".join(phases)):
        raise SequentialCampaignError("campaign axis sequence is invalid")
# fmt: on


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
    try:
        if resume:
            seal, digest = read_execution_seal(run_root)
            state = _validate_state(
                load_canonical_json(state_path), run_id, seed, digest
            )
            if state["status"] != transition_status(ExecutionStatus.FAILED):
                raise SequentialCampaignError(
                    "only an inactive failed campaign may resume"
                )
            if canonical_json_bytes(
                capture_execution_seal(
                    REPO_ROOT, run_id=run_id, seed=seed, executable=Path(sys.executable)
                )
            ) != canonical_json_bytes(seal):
                raise SequentialCampaignError(
                    "live workspace changed since execution seal"
                )
            state["status"] = transition_status(ExecutionStatus.RUNNING)
            state["resumed_at"] = utc_now()
        else:
            claim_run_root(run_root)
            seal = capture_execution_seal(
                REPO_ROOT, run_id=run_id, seed=seed, executable=Path(sys.executable)
            )
            raw = publish_execution_seal(run_root, seal)
            if (
                canonical_json_bytes(
                    capture_execution_seal(
                        REPO_ROOT,
                        run_id=run_id,
                        seed=seed,
                        executable=Path(sys.executable),
                    )
                )
                != raw
            ):
                raise SequentialCampaignError(
                    "live workspace changed after execution seal publication"
                )
            state = _new_state(
                run_id, seed, file_sha256(run_root / "campaign_execution_seal.json")
            )
            _save_state(state_path, state)
    except ExecutionSealError as exc:
        raise SequentialCampaignError(str(exc)) from exc

    specs = load_workflow_specs()
    for spec, axis_row in zip(specs, state["axes"], strict=True):
        if resume and is_resumable(axis_row.get("status")):
            if not _validated_attempt(spec.axis_id, run_root, axis_row):
                state["status"] = transition_status(ExecutionStatus.FAILED)
                axis_row["status"] = transition_status(ExecutionStatus.FAILED)
                axis_row["failure_reason"] = (
                    "previously successful artifact failed validation"
                )
                _save_state(state_path, state)
                atomic_json_dump(
                    run_root / "campaign_summary.json", _campaign_summary(state)
                )
                raise SequentialCampaignError(axis_row["failure_reason"])
            axis_row["resume_action"] = "verified_skip"
            continue

        attempt_number = len(axis_row.get("attempts", [])) + 1
        relative_attempt = Path("axes") / spec.axis_id / f"attempt-{attempt_number:03d}"
        attempt_dir = run_root / relative_attempt
        stdout_path = (
            run_root
            / "logs"
            / f"{spec.axis_id}.attempt-{attempt_number:03d}.stdout.log"
        )
        stderr_path = (
            run_root
            / "logs"
            / f"{spec.axis_id}.attempt-{attempt_number:03d}.stderr.log"
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
        returncode, process_status = _run_attempt(
            script_path=spec.script_path,
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
                analysis = validate_axis_analysis(
                    spec.axis_id,
                    input_dir=attempt_dir,
                    analysis_dir=attempt_dir / "analysis",
                )
            except AxisWorkflowError as exc:
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
                f"{spec.axis_id} stopped campaign: {axis_row['failure_reason']}"
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
    except (AxisWorkflowError, SequentialCampaignError) as exc:
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
