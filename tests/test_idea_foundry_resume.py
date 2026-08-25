"""Public resume contract tests for PR05-R1.

These tests use synthetic state and monkeypatched runner boundaries.  They do
not launch an Idea Foundry axis subprocess or create scientific artifacts.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from quartz.experiment_manifest import atomic_json_dump, canonical_sha256
from quartz.idea_foundry.axis_workflow import (
    decode_json_strict,
    parse_workflow_specs,
)
from quartz.idea_foundry.campaign_state import (
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
from quartz.idea_foundry.resume import (
    ResumeError,
    apply_resume_plan,
    execution_identity_from_payload,
    plan_resume,
)
from quartz.idea_foundry.status_schema import (
    failed_status,
    first_gate_status,
    running_status,
    succeeded_status,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
AXIS_PATH = REPO_ROOT / "configs" / "idea_foundry.axes.v1.json"
LAB_PATH = REPO_ROOT / "configs" / "idea_lab.local.v2.json"


def _capture() -> ExecutionCapture:
    axis_bytes = AXIS_PATH.read_bytes()
    lab_bytes = LAB_PATH.read_bytes()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    def file_identity(path: str) -> FileIdentity:
        content = (REPO_ROOT / path).read_bytes()
        return FileIdentity(path, len(content), hashlib.sha256(content).hexdigest())

    identity = ExecutionIdentity(
        schema_version=1,
        git_head=git_head,
        git_dirty=False,
        axis_registry=FileIdentity(
            "configs/idea_foundry.axes.v1.json",
            len(axis_bytes),
            hashlib.sha256(axis_bytes).hexdigest(),
        ),
        lab_registry=FileIdentity(
            "configs/idea_lab.local.v2.json",
            len(lab_bytes),
            hashlib.sha256(lab_bytes).hexdigest(),
        ),
        source_files=tuple(
            file_identity(path)
            for path in (
                "quartz/idea_foundry/status_schema.py",
                "quartz/idea_foundry/axis_workflow.py",
                "quartz/idea_foundry/sequential.py",
                "scripts/idea_foundry_run_all.py",
            )
        ),
        runtime=RuntimeIdentity(
            sys.executable,
            platform.python_version(),
            platform.platform(),
            ("run", "--run-id", "resume-test", "--seed", "23"),
        ),
    )
    specs = parse_workflow_specs(
        decode_json_strict(axis_bytes, label=str(AXIS_PATH)),
        decode_json_strict(lab_bytes, label=str(LAB_PATH)),
        repo_root=REPO_ROOT,
    )
    return ExecutionCapture(
        identity=identity,
        identity_sha256=canonical_sha256(identity.to_payload()),
        axis_registry_bytes=axis_bytes,
        lab_registry_bytes=lab_bytes,
        specs=specs,
    )


def _initial_plan(run_id: str = "resume-test") -> dict[str, object]:
    return build_initial_state_plan(run_id, 23, _capture(), "2026-08-25T00:00:00Z")


def _attempt(
    axis_id: str, number: int = 1, *, outcome: str = "completed"
) -> dict[str, object]:
    return {
        "attempt_number": number,
        "started_at": "2026-08-25T00:00:01Z",
        "completed_at": "2026-08-25T00:00:02Z",
        "output_dir": f"axes/{axis_id}/attempt-{number:03d}",
        "stdout": f"logs/{axis_id}.attempt-{number:03d}.stdout.log",
        "stderr": f"logs/{axis_id}.attempt-{number:03d}.stderr.log",
        "process_outcome": outcome,
        "returncode": 1,
    }


def _prepare_failed_run(
    tmp_path: Path,
    *,
    terminal_prefix: bool = False,
    campaign_status: dict[str, object] | None = None,
    failed_axis_index: int = 0,
) -> tuple[Path, dict[str, object]]:
    plan = _initial_plan()
    root = tmp_path / "run-root"
    publish_initial_run(root, plan)
    state = copy.deepcopy(plan["campaign_state"])
    assert isinstance(state, dict)
    axes = state["axes"]
    assert isinstance(axes, list)
    if terminal_prefix:
        terminal = axes[0]
        assert isinstance(terminal, dict)
        terminal["status"] = succeeded_status()
        terminal_axis_id = str(terminal["axis_id"])
        terminal["current_attempt"] = f"axes/{terminal_axis_id}/attempt-001"
        terminal["attempts"] = [_attempt(terminal_axis_id)]
        failed_axis_index = 1
    failed = axes[failed_axis_index]
    assert isinstance(failed, dict)
    failed["status"] = failed_status()
    failed["failure_reason"] = "synthetic prior failure"
    failed["current_attempt"] = f"axes/{failed['axis_id']}/attempt-001"
    failed["attempts"] = [_attempt(str(failed["axis_id"]))]
    state["status"] = campaign_status or failed_status()
    atomic_json_dump(root / "campaign_state.json", state)
    return root, state


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _patch_synthetic_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    import quartz.idea_foundry.sequential as sequential

    monkeypatch.setattr(
        sequential, "capture_execution_identity", lambda **_: _capture()
    )
    monkeypatch.setattr(
        sequential,
        "_run_attempt",
        lambda **_: (0, "completed"),
    )
    monkeypatch.setattr(
        sequential,
        "validate_axis_analysis",
        lambda axis_id, **_: {"status": first_gate_status(axis_id)},
    )
    monkeypatch.setattr(
        sequential,
        "validate_captured_axis_analysis",
        lambda axis_id, **_: {"status": first_gate_status(axis_id)},
    )


def _state_with_retry(state: dict[str, object]) -> dict[str, object]:
    next_state = copy.deepcopy(state)
    next_state["status"] = running_status()
    next_state["resumed_at"] = "2026-08-25T00:00:03Z"
    rows = next_state["axes"]
    assert isinstance(rows, list)
    retry = rows[0]
    assert isinstance(retry, dict)
    axis_id = str(retry["axis_id"])
    retry["status"] = running_status()
    retry.pop("failure_reason", None)
    retry["current_attempt"] = f"axes/{axis_id}/attempt-002"
    retry["attempts"].append(
        {
            "attempt_number": 2,
            "started_at": "2026-08-25T00:00:03Z",
            "output_dir": f"axes/{axis_id}/attempt-002",
            "stdout": f"logs/{axis_id}.attempt-002.stdout.log",
            "stderr": f"logs/{axis_id}.attempt-002.stderr.log",
            "process_outcome": "running",
        }
    )
    return next_state


def _run_public(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    import quartz.idea_foundry.sequential as sequential

    monkeypatch.setattr(sequential, "resolve_run_root", lambda *_: root)
    return sequential.run_campaign(
        campaign_root=root.parent,
        run_id="resume-test",
        seed=23,
        timeout_seconds=None,
        resume=resume,
        entrypoint=REPO_ROOT / "scripts" / "idea_foundry_run_all.py",
    )


def test_public_resume_rejection_is_byte_identical_for_invalid_prior_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state = _prepare_failed_run(tmp_path, terminal_prefix=True)
    _patch_synthetic_runner(monkeypatch)
    axes = state["axes"]
    assert isinstance(axes, list)
    failed = axes[1]
    assert isinstance(failed, dict)
    before = _snapshot(root)

    import quartz.idea_foundry.sequential as sequential

    monkeypatch.setattr(sequential, "resolve_run_root", lambda *_: root)
    with pytest.raises(Exception):
        _run_public(monkeypatch, root)

    assert _snapshot(root) == before


@pytest.mark.parametrize("kind", ["running", "success", "legacy", "identity_drift"])
def test_public_resume_rejection_is_byte_identical_for_terminal_running_legacy_and_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    _patch_synthetic_runner(monkeypatch)
    if kind == "running":
        root, state = _prepare_failed_run(tmp_path)
        axes = state["axes"]
        assert isinstance(axes, list)
        row = axes[0]
        assert isinstance(row, dict)
        row["status"] = running_status()
        atomic_json_dump(root / "campaign_state.json", state)
    elif kind == "success":
        root, state = _prepare_failed_run(tmp_path)
        axes = state["axes"]
        assert isinstance(axes, list)
        for row in axes:
            assert isinstance(row, dict)
            row["status"] = first_gate_status(str(row["axis_id"]))
        state["status"] = succeeded_status()
        atomic_json_dump(root / "campaign_state.json", state)
    elif kind == "legacy":
        root, state = _prepare_failed_run(tmp_path)
        state["schema_version"] = 1
        atomic_json_dump(root / "campaign_state.json", state)
    else:
        root, state = _prepare_failed_run(tmp_path)
        state["execution_identity"] = {
            "path": "execution_identity.json",
            "sha256": "c" * 64,
        }
        atomic_json_dump(root / "campaign_state.json", state)
    before = _snapshot(root)
    with pytest.raises(Exception):
        _run_public(monkeypatch, root)
    assert _snapshot(root) == before


def test_public_resume_uses_captured_registry_bytes_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _prepare_failed_run(tmp_path)
    _patch_synthetic_runner(monkeypatch)
    import quartz.idea_foundry.sequential as sequential

    monkeypatch.setattr(
        sequential,
        "load_workflow_specs",
        lambda: (_ for _ in ()).throw(AssertionError("live registry reread")),
    )
    _run_public(monkeypatch, root)


def test_public_resume_rejects_source_drift_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _prepare_failed_run(tmp_path)
    _patch_synthetic_runner(monkeypatch)
    import quartz.idea_foundry.sequential as sequential

    baseline = _capture()
    drifted_identity = replace(baseline.identity, git_head="f" * 40)
    drifted_capture = replace(
        baseline,
        identity=drifted_identity,
        identity_sha256=canonical_sha256(drifted_identity.to_payload()),
    )
    monkeypatch.setattr(
        sequential, "capture_execution_identity", lambda **_: drifted_capture
    )
    before = _snapshot(root)
    with pytest.raises(Exception):
        _run_public(monkeypatch, root)
    assert _snapshot(root) == before


def test_public_resume_failed_to_success_preserves_attempt_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, before_state = _prepare_failed_run(tmp_path)
    before_axes = before_state["axes"]
    assert isinstance(before_axes, list)
    before_attempt = copy.deepcopy(before_axes[0]["attempts"])
    _patch_synthetic_runner(monkeypatch)

    _run_public(monkeypatch, root)

    state = json.loads((root / "campaign_state.json").read_text())
    first = state["axes"][0]
    assert first["attempts"][0] == before_attempt[0]
    assert len(first["attempts"]) == 2
    assert "failure_reason" not in first
    assert validate_campaign_state_v2(state)["status"]["execution"] == "success"


def test_retry_running_and_success_states_pass_canonical_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _prepare_failed_run(tmp_path)
    _patch_synthetic_runner(monkeypatch)
    _run_public(monkeypatch, root)
    state = json.loads((root / "campaign_state.json").read_text())
    assert validate_campaign_state_v2(state)["status"]["execution"] == "success"
    for row in state["axes"]:
        assert "failure_reason" not in row


def test_apply_resume_plan_rejects_direct_seed_tamper_without_mutation(
    tmp_path: Path,
) -> None:
    root, _ = _prepare_failed_run(tmp_path)
    identity = execution_identity_from_payload(
        json.loads((root / "execution_identity.json").read_text())
    )
    state_path = root / "campaign_state.json"
    state_bytes = state_path.read_bytes()
    plan = plan_resume(
        run_root=root,
        state_bytes=state_bytes,
        identity=identity,
        captured_axes_payload=AXIS_PATH.read_bytes(),
        captured_lab_payload=LAB_PATH.read_bytes(),
        run_id="resume-test",
        seed=23,
    )
    tampered = replace(plan, seed=24)
    before = _snapshot(root)
    with pytest.raises(ResumeError):
        apply_resume_plan(state_path=state_path, plan=tampered)
    assert _snapshot(root) == before


def test_public_run_campaign_resume_uses_resume_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state = _prepare_failed_run(tmp_path)
    _patch_synthetic_runner(monkeypatch)
    import quartz.idea_foundry.sequential as sequential

    calls: list[str] = []
    planned_state = _state_with_retry(state)

    def fake_plan(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append("plan")
        return SimpleNamespace(state=copy.deepcopy(planned_state))

    def fake_apply(*args: Any, **kwargs: Any) -> dict[str, object]:
        calls.append("apply")
        return kwargs["plan"].state

    monkeypatch.setattr(sequential, "plan_resume", fake_plan, raising=False)
    monkeypatch.setattr(sequential, "apply_resume_plan", fake_apply, raising=False)
    _run_public(monkeypatch, root)
    assert calls == ["plan", "apply"]


def test_cli_resume_and_python_api_share_the_same_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = []
    for index in range(2):
        root, state = _prepare_failed_run(tmp_path / str(index))
        roots.append((root, state))
    _patch_synthetic_runner(monkeypatch)
    import quartz.idea_foundry.sequential as sequential

    calls: list[str] = []
    planned_states = [_state_with_retry(state) for _, state in roots]

    def fake_plan(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append("plan")
        return SimpleNamespace(
            state=copy.deepcopy(planned_states[calls.count("plan") - 1])
        )

    def fake_apply(*args: Any, **kwargs: Any) -> dict[str, object]:
        calls.append("apply")
        return kwargs["plan"].state

    monkeypatch.setattr(sequential, "plan_resume", fake_plan, raising=False)
    monkeypatch.setattr(sequential, "apply_resume_plan", fake_apply, raising=False)
    monkeypatch.setattr(
        sequential,
        "resolve_run_root",
        lambda campaign_root, run_id: roots[len(calls) // 2][0],
    )
    _run_public(monkeypatch, roots[0][0])
    sequential.sequential_main(
        [
            "--campaign-root",
            str(roots[1][0].parent),
            "resume",
            "--run-id",
            "resume-test",
        ],
        entrypoint=REPO_ROOT / "scripts" / "idea_foundry_run_all.py",
    )
    assert calls == ["plan", "apply", "plan", "apply"]
