"""Trusted-local execution identity capture for Idea Foundry runs."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from quartz.experiment_manifest import canonical_sha256
from quartz.idea_foundry.axis_workflow import (
    AxisWorkflowSpec,
    decode_json_strict,
    parse_workflow_specs,
)


EXECUTION_IDENTITY_SCHEMA_VERSION = 1
AXIS_REGISTRY_RELATIVE_PATH = "configs/idea_foundry.axes.v1.json"
LAB_REGISTRY_RELATIVE_PATH = "configs/idea_lab.local.v2.json"
SOURCE_RELATIVE_PATHS = (
    "quartz/idea_foundry/status_schema.py",
    "quartz/idea_foundry/axis_workflow.py",
    "quartz/idea_foundry/sequential.py",
    "scripts/idea_foundry_run_all.py",
)


class ExecutionIdentityError(RuntimeError):
    """Raised when a trusted-local execution identity cannot be captured."""


@dataclass(frozen=True)
class FileIdentity:
    path: str
    size_bytes: int
    sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class RuntimeIdentity:
    python_executable: str
    python_version: str
    platform: str
    argv: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "platform": self.platform,
            "argv": list(self.argv),
        }


@dataclass(frozen=True)
class ExecutionIdentity:
    schema_version: int
    git_head: str
    git_dirty: bool
    axis_registry: FileIdentity
    lab_registry: FileIdentity
    source_files: tuple[FileIdentity, ...]
    runtime: RuntimeIdentity

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "git_head": self.git_head,
            "git_dirty": self.git_dirty,
            "axis_registry": self.axis_registry.to_payload(),
            "lab_registry": self.lab_registry.to_payload(),
            "source_files": [source.to_payload() for source in self.source_files],
            "runtime": self.runtime.to_payload(),
        }


@dataclass(frozen=True)
class ExecutionCapture:
    identity: ExecutionIdentity
    identity_sha256: str
    axis_registry_bytes: bytes
    lab_registry_bytes: bytes
    specs: tuple[AxisWorkflowSpec, ...]


def _git_output(repo_root: Path, arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ExecutionIdentityError("unable to obtain Git identity") from exc
    if completed.returncode != 0:
        raise ExecutionIdentityError("unable to obtain Git identity")
    return completed.stdout.strip()


def _file_identity(repo_root: Path, relative_path: str) -> FileIdentity:
    content = (repo_root / relative_path).read_bytes()
    return FileIdentity(
        path=relative_path,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _file_identity_from_bytes(relative_path: str, content: bytes) -> FileIdentity:
    return FileIdentity(
        path=relative_path,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def capture_execution_identity(
    *, repo_root: Path, entrypoint: Path, argv: Sequence[str]
) -> ExecutionCapture:
    """Capture one clean local workspace and the bytes that define its workflow."""

    root = repo_root.resolve()
    git_head = _git_output(root, ("rev-parse", "HEAD"))
    status = _git_output(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    if status:
        raise ExecutionIdentityError("working tree is dirty")

    axis_registry_bytes = (root / AXIS_REGISTRY_RELATIVE_PATH).read_bytes()
    lab_registry_bytes = (root / LAB_REGISTRY_RELATIVE_PATH).read_bytes()
    specs = parse_workflow_specs(
        decode_json_strict(axis_registry_bytes, label=AXIS_REGISTRY_RELATIVE_PATH),
        decode_json_strict(lab_registry_bytes, label=LAB_REGISTRY_RELATIVE_PATH),
        repo_root=root,
    )
    entrypoint_relative = str(entrypoint.resolve().relative_to(root))
    if entrypoint_relative != SOURCE_RELATIVE_PATHS[-1]:
        raise ExecutionIdentityError("entrypoint does not match the execution source")
    identity = ExecutionIdentity(
        schema_version=EXECUTION_IDENTITY_SCHEMA_VERSION,
        git_head=git_head,
        git_dirty=False,
        axis_registry=_file_identity_from_bytes(
            AXIS_REGISTRY_RELATIVE_PATH, axis_registry_bytes
        ),
        lab_registry=_file_identity_from_bytes(
            LAB_REGISTRY_RELATIVE_PATH, lab_registry_bytes
        ),
        source_files=tuple(
            _file_identity(root, relative_path)
            for relative_path in SOURCE_RELATIVE_PATHS
        ),
        runtime=RuntimeIdentity(
            python_executable=sys.executable,
            python_version=platform.python_version(),
            platform=platform.platform(),
            argv=tuple(argv),
        ),
    )
    payload = validate_execution_identity_payload(identity.to_payload())
    return ExecutionCapture(
        identity=identity,
        identity_sha256=canonical_sha256(payload),
        axis_registry_bytes=axis_registry_bytes,
        lab_registry_bytes=lab_registry_bytes,
        specs=specs,
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExecutionIdentityError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], *, keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        raise ExecutionIdentityError(
            f"{label} fields are invalid: missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )


def _string(value: object, label: str) -> str:
    if type(value) is not str:
        raise ExecutionIdentityError(f"{label} must be a string")
    return value


def _file_payload(
    value: object, label: str, *, expected_path: str | None = None
) -> dict[str, object]:
    row = _mapping(value, label)
    _exact_keys(row, keys={"path", "size_bytes", "sha256"}, label=label)
    size_bytes = row["size_bytes"]
    if type(size_bytes) is not int or size_bytes < 0:
        raise ExecutionIdentityError(
            f"{label}.size_bytes must be a nonnegative integer"
        )
    path = _string(row["path"], f"{label}.path")
    if expected_path is not None and path != expected_path:
        raise ExecutionIdentityError(
            f"{label}.path does not match the execution contract"
        )
    sha256 = _string(row["sha256"], f"{label}.sha256")
    if len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        raise ExecutionIdentityError(f"{label}.sha256 must be lowercase SHA-256")
    return {
        "path": path,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _runtime_payload(value: object) -> dict[str, object]:
    row = _mapping(value, "runtime")
    _exact_keys(
        row,
        keys={"python_executable", "python_version", "platform", "argv"},
        label="runtime",
    )
    argv = row["argv"]
    if type(argv) is not list or any(type(item) is not str for item in argv):
        raise ExecutionIdentityError("runtime.argv must be an array of strings")
    return {
        "python_executable": _string(
            row["python_executable"], "runtime.python_executable"
        ),
        "python_version": _string(row["python_version"], "runtime.python_version"),
        "platform": _string(row["platform"], "runtime.platform"),
        "argv": list(argv),
    }


def validate_execution_identity_payload(value: object) -> dict[str, object]:
    """Validate and normalize an exact execution-identity JSON payload."""

    row = _mapping(value, "execution identity")
    _exact_keys(
        row,
        keys={
            "schema_version",
            "git_head",
            "git_dirty",
            "axis_registry",
            "lab_registry",
            "source_files",
            "runtime",
        },
        label="execution identity",
    )
    schema_version = row["schema_version"]
    if type(schema_version) is not int:
        raise ExecutionIdentityError(
            "execution identity.schema_version must be an integer"
        )
    if schema_version != EXECUTION_IDENTITY_SCHEMA_VERSION:
        raise ExecutionIdentityError("execution identity.schema_version is unsupported")
    git_dirty = row["git_dirty"]
    if type(git_dirty) is not bool:
        raise ExecutionIdentityError("execution identity.git_dirty must be a boolean")
    if git_dirty:
        raise ExecutionIdentityError("execution identity.git_dirty must be false")
    source_files = row["source_files"]
    if type(source_files) is not list:
        raise ExecutionIdentityError("execution identity.source_files must be an array")
    if len(source_files) != len(SOURCE_RELATIVE_PATHS):
        raise ExecutionIdentityError(
            "execution identity.source_files has the wrong length"
        )
    return {
        "schema_version": schema_version,
        "git_head": _string(row["git_head"], "execution identity.git_head"),
        "git_dirty": git_dirty,
        "axis_registry": _file_payload(
            row["axis_registry"],
            "axis_registry",
            expected_path=AXIS_REGISTRY_RELATIVE_PATH,
        ),
        "lab_registry": _file_payload(
            row["lab_registry"],
            "lab_registry",
            expected_path=LAB_REGISTRY_RELATIVE_PATH,
        ),
        "source_files": [
            _file_payload(
                source,
                f"source_files[{index}]",
                expected_path=SOURCE_RELATIVE_PATHS[index],
            )
            for index, source in enumerate(source_files)
        ],
        "runtime": _runtime_payload(row["runtime"]),
    }
