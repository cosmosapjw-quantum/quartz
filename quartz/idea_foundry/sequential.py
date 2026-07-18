"""Fail-closed sequential runner for all 26 idea-foundry first gates."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
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

SEQUENTIAL_SCHEMA_VERSION = 1
DEFAULT_CAMPAIGN_ROOT = REPO_ROOT / "results" / "idea_foundry_sequential"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
TERMINAL_SUCCESS = "completed_no_promotion"


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
        "promotion": {"auto": False, "eligible": False},
    }


def _new_state(run_id: str, seed: int, entrypoint: Path) -> dict[str, Any]:
    now = utc_now()
    specs = load_workflow_specs()
    return {
        "schema_version": SEQUENTIAL_SCHEMA_VERSION,
        "run_id": run_id,
        "suite": "first-gate-all-sequential",
        "status": "running",
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
                "status": "planned",
                "attempts": [],
            }
            for spec in specs
        ],
        "promotion": {"auto": False, "eligible": False},
    }


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_json_dump(path, state)


def _validate_state(
    state: Any, run_id: str, seed: int, entrypoint: Path
) -> dict[str, Any]:
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != SEQUENTIAL_SCHEMA_VERSION
    ):
        raise SequentialCampaignError("campaign state schema mismatch")
    if state.get("run_id") != run_id or state.get("seed") != seed:
        raise SequentialCampaignError("resume run identity or seed changed")
    if state.get("fingerprint") != _fingerprint(entrypoint):
        raise SequentialCampaignError(
            "resume refused: registry, source, or interpreter hash changed"
        )
    axes = state.get("axes")
    expected = [spec.axis_id for spec in load_workflow_specs()]
    if not isinstance(axes, list) or [row.get("axis_id") for row in axes] != expected:
        raise SequentialCampaignError("campaign axis order changed")
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
    if axis_row.get("status") != TERMINAL_SUCCESS:
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
        validate_axis_analysis(
            axis_id, input_dir=attempt_dir, analysis_dir=attempt_dir / "analysis"
        )
    except AxisWorkflowError:
        return False
    return True


def _campaign_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    axes = state["axes"]
    status_counts = {
        status: sum(row.get("status") == status for row in axes)
        for status in sorted({str(row.get("status")) for row in axes})
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
                "promotion_eligible": False,
            }
            for row in axes
        ],
        "claim_scope": "synthetic_contract_execution_only",
        "promotion": {"auto": False, "eligible": False},
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
        state = _validate_state(load_json_strict(state_path), run_id, seed, entrypoint)
        state["status"] = "running"
        state["resumed_at"] = utc_now()
    else:
        if run_root.exists():
            raise SequentialCampaignError(f"new run root already exists: {run_root}")
        run_root.mkdir(parents=True)
        state = _new_state(run_id, seed, entrypoint)
    _save_state(state_path, state)

    specs = load_workflow_specs()
    for spec, axis_row in zip(specs, state["axes"], strict=True):
        if resume and axis_row.get("status") == TERMINAL_SUCCESS:
            if not _validated_attempt(spec.axis_id, run_root, axis_row):
                state["status"] = "failed"
                axis_row["status"] = "failed"
                axis_row["failure_reason"] = (
                    "previously successful artifact failed validation"
                )
                _save_state(state_path, state)
                atomic_json_dump(
                    run_root / "campaign_summary.json", _campaign_summary(state)
                )
                raise SequentialCampaignError(axis_row["failure_reason"])
            axis_row["resume_action"] = "verified_skip"
            _save_state(state_path, state)
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
            "status": "running",
        }
        axis_row.setdefault("attempts", []).append(attempt)
        axis_row["status"] = "running"
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
        attempt["status"] = process_status
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
                attempt["status"] = "failed"
                attempt["failure_reason"] = str(exc)
            else:
                axis_row["status"] = TERMINAL_SUCCESS
                axis_row["analysis_status"] = analysis["analysis_status"]
                axis_row["evidence_status"] = analysis["source_evidence_status"]
                axis_row["promotion_eligible"] = False
        if returncode != 0:
            axis_row["status"] = attempt["status"]
            axis_row["failure_reason"] = attempt.get(
                "failure_reason", f"axis subprocess exited with {returncode}"
            )
            state["status"] = axis_row["status"]
            _save_state(state_path, state)
            atomic_json_dump(
                run_root / "campaign_summary.json", _campaign_summary(state)
            )
            raise SequentialCampaignError(
                f"{spec.axis_id} stopped campaign: {axis_row['failure_reason']}"
            )
        _save_state(state_path, state)

    state["status"] = "completed_no_promotion"
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
