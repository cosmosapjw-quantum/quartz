from __future__ import annotations

import copy
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from quartz.idea_foundry.axis_workflow import (
    AXIS_REGISTRY_PATH,
    LAB_REGISTRY_PATH,
    decode_json_strict,
    parse_workflow_specs,
)
from quartz.idea_foundry.execution_identity import (
    ExecutionIdentityError,
    capture_execution_identity,
    validate_execution_identity_payload,
)
from quartz.experiment_manifest import canonical_sha256


SOURCE_PATHS = (
    "quartz/idea_foundry/status_schema.py",
    "quartz/idea_foundry/axis_workflow.py",
    "quartz/idea_foundry/sequential.py",
    "scripts/idea_foundry_run_all.py",
)


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _ordinary_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "ordinary-repo"
    repo.mkdir()
    for relative in SOURCE_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    (repo / "configs").mkdir()
    shutil.copyfile(AXIS_REGISTRY_PATH, repo / "configs" / AXIS_REGISTRY_PATH.name)
    shutil.copyfile(LAB_REGISTRY_PATH, repo / "configs" / LAB_REGISTRY_PATH.name)
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "test@example.invalid")
    _run_git(repo, "config", "user.name", "PR-03 test")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "fixture")
    return repo


def _capture(repo: Path):
    return capture_execution_identity(
        repo_root=repo,
        entrypoint=repo / "scripts" / "idea_foundry_run_all.py",
        argv=("--seed", "17"),
    )


def test_capture_execution_identity_records_clean_commit_configs_sources_runtime(
    tmp_path: Path,
) -> None:
    repo = _ordinary_repo(tmp_path)

    capture = _capture(repo)

    assert (
        capture.identity.git_head
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    )
    assert capture.identity.git_dirty is False
    assert capture.identity.axis_registry.path == "configs/idea_foundry.axes.v1.json"
    assert capture.identity.axis_registry.size_bytes == len(capture.axis_registry_bytes)
    assert (
        capture.identity.axis_registry.sha256
        == hashlib.sha256(capture.axis_registry_bytes).hexdigest()
    )
    assert capture.identity.lab_registry.path == "configs/idea_lab.local.v2.json"
    assert [source.path for source in capture.identity.source_files] == list(
        SOURCE_PATHS
    )
    assert capture.identity.runtime.python_executable == sys.executable
    assert capture.identity.runtime.argv == ("--seed", "17")
    assert capture.identity_sha256 == canonical_sha256(capture.identity.to_payload())


def test_capture_execution_identity_rejects_tracked_and_untracked_drift(
    tmp_path: Path,
) -> None:
    repo = _ordinary_repo(tmp_path)
    tracked = repo / SOURCE_PATHS[0]
    original = tracked.read_bytes()
    tracked.write_bytes(original + b"# drift\n")
    with pytest.raises(ExecutionIdentityError, match="working tree is dirty"):
        _capture(repo)
    tracked.write_bytes(original)
    (repo / "untracked.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ExecutionIdentityError, match="working tree is dirty"):
        _capture(repo)


def test_capture_execution_identity_fails_when_git_identity_is_unavailable(
    tmp_path: Path,
) -> None:
    repo = _ordinary_repo(tmp_path)
    git_dir = repo / ".git"
    unavailable = repo / ".git-unavailable"
    git_dir.rename(unavailable)
    try:
        with pytest.raises(ExecutionIdentityError, match="Git"):
            _capture(repo)
    finally:
        unavailable.rename(git_dir)


def test_capture_uses_one_config_read_and_builds_specs_from_captured_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _ordinary_repo(tmp_path)
    axis_path = repo / "configs" / AXIS_REGISTRY_PATH.name
    lab_path = repo / "configs" / LAB_REGISTRY_PATH.name
    reads = {axis_path: 0, lab_path: 0}
    original_read_bytes = Path.read_bytes

    def counting_read_bytes(path: Path) -> bytes:
        if path in reads:
            reads[path] += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    capture = _capture(repo)

    assert reads == {axis_path: 1, lab_path: 1}
    assert capture.specs == parse_workflow_specs(
        decode_json_strict(capture.axis_registry_bytes, label="axis registry"),
        decode_json_strict(capture.lab_registry_bytes, label="lab registry"),
        repo_root=repo,
    )


def test_identity_hash_is_stable_under_dict_serialization_order(tmp_path: Path) -> None:
    payload = _capture(_ordinary_repo(tmp_path)).identity.to_payload()
    reordered = {
        "runtime": {
            "argv": payload["runtime"]["argv"],
            "platform": payload["runtime"]["platform"],
            "python_version": payload["runtime"]["python_version"],
            "python_executable": payload["runtime"]["python_executable"],
        },
        "source_files": list(reversed(payload["source_files"])),
        "lab_registry": payload["lab_registry"],
        "axis_registry": payload["axis_registry"],
        "git_dirty": payload["git_dirty"],
        "git_head": payload["git_head"],
        "schema_version": payload["schema_version"],
    }
    reordered["source_files"] = payload["source_files"]

    assert canonical_sha256(
        validate_execution_identity_payload(payload)
    ) == canonical_sha256(validate_execution_identity_payload(reordered))


def test_identity_payload_rejects_missing_extra_and_wrong_types(tmp_path: Path) -> None:
    payload = _capture(_ordinary_repo(tmp_path)).identity.to_payload()
    missing = dict(payload)
    missing.pop("runtime")
    extra = {**payload, "unexpected": None}
    wrong_type = dict(payload)
    wrong_type["git_dirty"] = 0
    wrong_schema = {**payload, "schema_version": 2}
    dirty = {**payload, "git_dirty": True}
    wrong_axis_path = copy.deepcopy(payload)
    wrong_axis_path["axis_registry"]["path"] = "configs/other.json"
    wrong_lab_path = copy.deepcopy(payload)
    wrong_lab_path["lab_registry"]["path"] = "configs/other.json"
    short_sources = {**payload, "source_files": payload["source_files"][:-1]}
    wrong_source_path = copy.deepcopy(payload)
    wrong_source_path["source_files"][0]["path"] = "quartz/other.py"
    negative_size = copy.deepcopy(payload)
    negative_size["source_files"][0]["size_bytes"] = -1
    invalid_sha256 = copy.deepcopy(payload)
    invalid_sha256["source_files"][0]["sha256"] = "A" * 64

    for invalid in (
        missing,
        extra,
        wrong_type,
        wrong_schema,
        dirty,
        wrong_axis_path,
        wrong_lab_path,
        short_sources,
        wrong_source_path,
        negative_size,
        invalid_sha256,
    ):
        with pytest.raises(ExecutionIdentityError):
            validate_execution_identity_payload(invalid)
