#!/usr/bin/env python3
"""Verify Idea Foundry receipts against bytes in their declared Git trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPTS_DIR = REPO_ROOT / "docs" / "idea_foundry" / "receipts"
RECEIPT_SCHEMA_VERSION = 2
INVENTORY_NAMES = tuple(
    f"{name}_inventory" for name in "source input binary artifact".split()
)
_IDENTITY_KEYS = {"commit", "dirty", *INVENTORY_NAMES}
_RUN_KEYS = set(
    "run_id execution_identity analysis_identity transformation_link".split()
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FULL_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def file_sha256(path: Path) -> str:
    """Return a file digest for API compatibility, not receipt binding."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str, input: bytes | None = None) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            input=input,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Git command failed: {' '.join(args[:2])}") from exc


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if set(value) != keys:
        missing, unknown = sorted(keys - set(value)), sorted(set(value) - keys)
        raise ValueError(f"{label} keys invalid; missing={missing}, unknown={unknown}")
    return value


def validate_repo_path(value: Any, label: str) -> str:
    """Return a canonical relative POSIX path or fail before Git invocation."""
    parts = value.split("/") if isinstance(value, str) else []
    invalid = (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\0" in value
        or value.startswith(("/", "-"))
        or value.endswith("/")
        or any(part in {"", ".", ".."} for part in parts)
    )
    if invalid:
        raise ValueError(f"{label}.path must be a normalized in-repository POSIX path")
    return value


def _descriptor(value: Any, label: str) -> dict[str, str]:
    item = _exact(value, {"path", "sha256"}, label)
    path, digest = validate_repo_path(item["path"], label), item["sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label}.sha256 must be 64 lowercase hex characters")
    return {"path": path, "sha256": digest}


def validate_identity_contract(value: Any, label: str) -> dict[str, Any]:
    """Validate an identity without reading Git or the worktree."""
    identity = _exact(value, _IDENTITY_KEYS, label)
    if identity["dirty"] is not False:
        raise ValueError(f"{label}.dirty must be false")
    commit = identity["commit"]
    if not isinstance(commit, str) or _FULL_OID.fullmatch(commit) is None:
        raise ValueError(f"{label}.commit cannot be resolved as commit")
    normalized: dict[str, Any] = {"commit": commit, "dirty": False}
    for name in INVENTORY_NAMES:
        values = identity[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{label}.{name} must be nonempty")
        items = [_descriptor(item, f"{label}.{name}") for item in values]
        paths = [item["path"] for item in items]
        if len(paths) != len(set(paths)):
            raise ValueError(f"duplicate path in {label}.{name}")
        normalized[name] = items
    return normalized


def validate_raw_manifest_link(value: Any, execution: dict, analysis: dict) -> None:
    """Bind one raw-manifest descriptor across execution and analysis."""
    link = _exact(value, {"raw_execution_manifest"}, "transformation_link")
    raw = _descriptor(
        link["raw_execution_manifest"], "transformation_link.raw_execution_manifest"
    )
    pairs = (
        (execution["artifact_inventory"], "execution_identity.artifact_inventory"),
        (analysis["input_inventory"], "analysis_identity.input_inventory"),
    )
    for inventory, label in pairs:
        matches = [item for item in inventory if item["path"] == raw["path"]]
        if not matches:
            raise ValueError(f"raw execution manifest is absent from {label}")
        if len(matches) != 1 or matches[0] != raw:
            raise ValueError("raw execution manifest descriptor mismatch")


def _resolve_commit(repo_root: Path, oid: str, label: str) -> None:
    try:
        object_format = (
            _git(repo_root, "rev-parse", "--show-object-format").decode().strip()
        )
        if len(oid) != {"sha1": 40, "sha256": 64}[object_format]:
            raise ValueError
        if _git(repo_root, "cat-file", "-t", oid).strip() != b"commit":
            raise ValueError
    except (KeyError, UnicodeDecodeError, ValueError):
        raise ValueError(f"{label}.commit cannot be resolved as commit") from None


def _blob_sha256(repo_root: Path, oid: str) -> str:
    digest = hashlib.sha256()
    try:
        process = subprocess.Popen(
            ["git", "cat-file", "blob", oid],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if process.stdout is None:
            raise ValueError("Git blob stream unavailable")
        while chunk := process.stdout.read(65536):
            digest.update(chunk)
        if process.wait() != 0:
            raise ValueError("Git blob read failed")
    except OSError as exc:
        raise ValueError("Git blob read failed") from exc
    return digest.hexdigest()


def _verify_tree_entry(repo_root: Path, commit: str, item: dict, label: str) -> None:
    path = item["path"]
    try:
        output = _git(
            repo_root, "ls-tree", "-z", "--full-tree", commit, "--", f":(literal){path}"
        )
    except ValueError:
        raise ValueError(f"Git-tree path missing for {label}: {path}") from None
    records = [record for record in output.split(b"\0") if record]
    if not records:
        raise ValueError(f"Git-tree path missing for {label}: {path}")
    try:
        metadata, found = records[0].split(b"\t", 1)
        mode, kind, blob = metadata.split(b" ", 2)
    except ValueError:
        raise ValueError(f"Git-tree record malformed for {label}: {path}") from None
    if len(records) != 1 or found != path.encode():
        raise ValueError(f"Git-tree path lookup is ambiguous for {label}: {path}")
    if mode not in {b"100644", b"100755"} or kind != b"blob":
        raise ValueError(f"Git-tree path is not a regular file for {label}: {path}")
    actual = _blob_sha256(repo_root, blob.decode())
    if actual != item["sha256"]:
        raise ValueError(
            f"Git-tree hash drift for {label}: {path}; expected {item['sha256']}, got {actual}"
        )


def _verify_identity(repo_root: Path, value: Any, label: str) -> tuple[dict, int]:
    identity = validate_identity_contract(value, label)
    _resolve_commit(repo_root, identity["commit"], label)
    for name in INVENTORY_NAMES:
        for item in identity[name]:
            _verify_tree_entry(repo_root, identity["commit"], item, f"{label}.{name}")
    return identity, sum(len(identity[name]) for name in INVENTORY_NAMES)


def verify_receipt(
    receipt_path: Path, *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Validate one schema-v2 receipt against its declared Git objects."""
    receipt_path, repo_root = Path(receipt_path), Path(repo_root)
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError(f"receipt is missing or symlink: {receipt_path}")
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    version = data.get("schema_version") if isinstance(data, dict) else None
    if type(version) is int and version == 1:
        raise ValueError(
            "legacy receipt schema v1 is invalid/unbound; read-only inspection only"
        )
    if type(version) is not int or version != RECEIPT_SCHEMA_VERSION:
        raise ValueError("receipt schema_version must be integer 2")
    runs = _exact(data, {"schema_version", "runs"}, "receipt")["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"receipt contains no run entries: {receipt_path}")
    seen: set[str] = set()
    entries = artifacts = 0
    for index, value in enumerate(runs):
        run = _exact(value, _RUN_KEYS, f"runs[{index}]")
        run_id = run["run_id"]
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"runs[{index}].run_id must be a nonempty string")
        if run_id in seen:
            raise ValueError(f"duplicate run_id: {run_id}")
        seen.add(run_id)
        execution, left = _verify_identity(
            repo_root, run["execution_identity"], "execution_identity"
        )
        analysis, right = _verify_identity(
            repo_root, run["analysis_identity"], "analysis_identity"
        )
        validate_raw_manifest_link(run["transformation_link"], execution, analysis)
        entries += left + right
        artifacts += len(execution["artifact_inventory"]) + len(
            analysis["artifact_inventory"]
        )
    try:
        name = str(receipt_path.relative_to(repo_root))
    except ValueError:
        name = str(receipt_path)
    return {
        "receipt": name,
        "status": "VERIFIED",
        "verified_runs": len(runs),
        "verified_artifacts": artifacts,
        "verified_entries": entries,
    }


def _empty_result(message: str, allow: bool) -> int:
    if allow:
        print(f"[DIAGNOSTIC] {message}; allow-empty accepted")
        return 0
    print(f"[FAIL] {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Idea Foundry evidence receipts"
    )
    parser.add_argument("--receipts-dir", type=Path, default=RECEIPTS_DIR)
    parser.add_argument("--allow-empty-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    directory = args.receipts_dir
    if not directory.is_dir() or directory.is_symlink():
        return _empty_result(
            f"receipts directory missing: {directory}", args.allow_empty_diagnostic
        )
    receipt_files = sorted(directory.glob("*.receipt.json"))
    if not receipt_files:
        return _empty_result(
            f"no receipt files found: {directory}", args.allow_empty_diagnostic
        )
    passed = True
    for receipt_path in receipt_files:
        try:
            result = verify_receipt(receipt_path)
            print(
                f"[OK] {result['receipt']}: {result['verified_runs']} runs, "
                f"{result['verified_artifacts']} artifacts verified"
            )
        except Exception as exc:
            print(f"[FAIL] {receipt_path}: {exc}", file=sys.stderr)
            passed = False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
