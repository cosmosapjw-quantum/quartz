"""Small provenance helpers for mechanism-level research experiments.

This module is intentionally independent of the training/evaluation manifests.
It records enough information to reproduce a local synthetic assay without
promoting that assay to evidence about the live Rust MCTS engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quartz.contract_summary import stable_json_hash


MANIFEST_FORMAT_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repo_root: Path, args: Sequence[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_provenance(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    head = _git_output(root, ["rev-parse", "HEAD"])
    status = _git_output(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    diff = _git_output(root, ["diff", "--binary", "HEAD"])
    return {
        "head": head,
        "dirty": None if status is None else bool(status),
        "status_sha256": None if status is None else hashlib.sha256(status.encode()).hexdigest(),
        "tracked_diff_sha256": None if diff is None else hashlib.sha256(diff.encode()).hexdigest(),
    }


def source_fingerprints(paths: Sequence[str | Path], repo_root: str | Path) -> list[dict[str, str]]:
    root = Path(repo_root).resolve()
    rows = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        try:
            display = str(path.relative_to(root))
        except ValueError:
            display = str(path)
        rows.append({"path": display, "sha256": file_sha256(path)})
    return sorted(rows, key=lambda row: row["path"])


def build_run_manifest(
    *,
    experiment_id: str,
    execution_mode: str,
    resolved_config: Mapping[str, Any],
    repo_root: str | Path,
    source_paths: Sequence[str | Path],
    argv: Sequence[str],
    started_at: str,
    assumptions: Sequence[str],
    prohibited_inferences: Sequence[str],
) -> dict[str, Any]:
    contract = {
        "experiment_id": str(experiment_id),
        "execution_mode": str(execution_mode),
        "resolved_config": dict(resolved_config),
        "assumptions": list(assumptions),
        "prohibited_inferences": list(prohibited_inferences),
    }
    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "experiment_id": str(experiment_id),
        "execution_mode": str(execution_mode),
        "claim_status": "synthetic_screening_only",
        "started_at": str(started_at),
        "completed_at": None,
        "status": "running",
        "resolved_config": dict(resolved_config),
        "run_contract": contract,
        "run_contract_hash": canonical_sha256(contract),
        "run_contract_short_hash": stable_json_hash(contract),
        "git": git_provenance(repo_root),
        "sources": source_fingerprints(source_paths, repo_root),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "argv": list(argv),
            "cwd": os.getcwd(),
        },
        "assumptions": list(assumptions),
        "prohibited_inferences": list(prohibited_inferences),
        "artifacts": [],
    }


def finalize_run_manifest(
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
    artifact_paths: Sequence[str | Path],
    status: str = "completed",
) -> dict[str, Any]:
    output_root = Path(output_dir).resolve()
    payload = dict(manifest)
    artifacts = []
    for raw_path in artifact_paths:
        path = Path(raw_path).resolve()
        artifacts.append(
            {
                "path": str(path.relative_to(output_root)),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    payload["artifacts"] = sorted(artifacts, key=lambda row: row["path"])
    payload["completed_at"] = utc_now()
    payload["status"] = str(status)
    return payload


def atomic_json_dump(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
