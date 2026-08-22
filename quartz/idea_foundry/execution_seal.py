"""Prospective, fail-closed identity seal for sequential Idea Foundry runs."""

# fmt: off
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from quartz.idea_foundry.axis_workflow import decode_json_strict

SOURCES = (
    "scripts/idea_foundry/a03_uncertainty_decomposition.py", "scripts/idea_foundry/a09_h3_change_point_router.py",
    "scripts/idea_foundry/a01_calibrated_stop_council.py", "scripts/idea_foundry/a02_static_anchor_rpo.py",
    "scripts/idea_foundry/a17_b13_curvature_readout.py", "scripts/idea_foundry/a21_coherence_signed_path_shadow.py",
    "scripts/idea_foundry/a22_physics_falsification_dashboard.py", "scripts/idea_foundry/a06_gumbel_sequential_halving.py",
    "scripts/idea_foundry/a07_residual_evidence_widening.py", "scripts/idea_foundry/a12_jsd_locally_balanced_sampler.py",
    "scripts/idea_foundry/a25_ments_soft_backup.py", "scripts/idea_foundry/a26_nested_contour_exact_lab.py",
    "scripts/idea_foundry/a05_counterfactual_meta_teacher.py", "scripts/idea_foundry/a04_kg_voc_allocator.py",
    "scripts/idea_foundry/a08_tactical_proof_backend.py", "scripts/idea_foundry/a13_pending_flow_wu_uct.py",
    "scripts/idea_foundry/a15_service_curve_scheduler.py", "scripts/idea_foundry/a14_semantic_path_lsh.py",
    "scripts/idea_foundry/a16_monte_carlo_graph_sharing.py", "scripts/idea_foundry/a11_dynamic_live_set_particles.py",
    "scripts/idea_foundry/a18_diffusion_regularized_evaluator.py", "scripts/idea_foundry/a19_rw_rest_lite_evaluator.py",
    "scripts/idea_foundry/a23_cpu_incremental_pattern_student.py", "scripts/idea_foundry/a20_regret_state_archive.py",
    "scripts/idea_foundry/a24_learned_budget_gate.py", "scripts/idea_foundry/a10_prior_refresh_specialist.py",
    "quartz/experiment_manifest.py", "quartz/idea_foundry/__init__.py", "quartz/idea_foundry/axis_workflow.py",
    "quartz/idea_foundry/contracts.py", "quartz/idea_foundry/control.py", "quartz/idea_foundry/execution_seal.py",
    "quartz/idea_foundry/gates.py", "quartz/idea_foundry/learning.py", "quartz/idea_foundry/search.py",
    "quartz/idea_foundry/sequential.py", "quartz/idea_foundry/serialization.py", "quartz/idea_foundry/status_schema.py",
    "scripts/idea_foundry_axis_gate.py", "scripts/idea_foundry_run_all.py",
)
INPUTS = ("configs/idea_foundry.axes.v1.json", "configs/idea_lab.local.v2.json")
AXES = ("A03", "A09", "A01", "A02", "A17", "A21", "A22", "A06", "A07", "A12", "A25", "A26", "A05", "A04", "A08", "A13", "A15", "A14", "A16", "A11", "A18", "A19", "A23", "A20", "A24", "A10")
CAMPAIGN_ARTIFACTS = ("campaign_execution_seal.json", "campaign_state.json", "campaign_summary.json")
ATTEMPT_ARTIFACTS = ("run_manifest.json", "rows.jsonl", "summary.json", "analysis/analysis_manifest.json", "analysis/analysis.json", "analysis/analysis_rows.jsonl", "analysis/diagnostic.png")
_TOP = ("schema_version", "kind", "suite", "claim_scope", "run_id", "seed", "git", "axis_order", "sources", "inputs", "binaries", "campaign_artifacts", "terminal_attempt_artifacts")
_HEX = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}\Z")
class ExecutionSealError(RuntimeError):
    """Raised when an execution identity cannot be proven exactly."""


def _walk(path: Path, *, allow_final_symlink: bool = False) -> os.stat_result:
    path = path.absolute()
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) and not (allow_final_symlink and index == len(path.parts) - 2):
            raise ExecutionSealError(f"symlinked path component: {current}")
    return path.lstat()


def _stable_read(path: Path, *, links: int = 1) -> bytes:
    before = _walk(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != links:
        raise ExecutionSealError(f"not a unique regular file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        chunks = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    final = path.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_nlink, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(opened) or identity(opened) != identity(after) or identity(after) != identity(final):
        raise ExecutionSealError(f"file changed while read: {path}")
    return b"".join(chunks)


def _git(root: Path, *args: str) -> bytes:
    env = {"PATH": os.defpath, "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_NO_REPLACE_OBJECTS": "1"}
    command = ["/usr/bin/git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null", "--literal-pathspecs", *args]
    try:
        return subprocess.run(command, cwd=root, env=env, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExecutionSealError(f"git observation failed: {' '.join(args)}") from exc


def _record(path: str, data: bytes) -> dict[str, Any]:
    return {"path": path, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _repo_path(value: str) -> None:
    pure = PurePosixPath(value)
    if type(value) is not str or not value or pure.is_absolute() or str(pure) != value or any(part in {"", ".", ".."} for part in pure.parts):
        raise ExecutionSealError(f"non-canonical repository path: {value!r}")


def _exact_json(value: Any, seen: set[int] | None = None) -> None:
    if type(value) in {str, int, bool, type(None)}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ExecutionSealError("non-finite JSON value")
        return
    if type(value) not in {list, dict}:
        raise ExecutionSealError("non-built-in JSON value")
    seen = seen or set()
    if id(value) in seen:
        raise ExecutionSealError("cyclic JSON value")
    seen.add(id(value))
    items = value.items() if type(value) is dict else enumerate(value)
    for key, item in items:
        if type(value) is dict and type(key) is not str:
            raise ExecutionSealError("non-string JSON key")
        _exact_json(item, seen)
    seen.remove(id(value))


def canonical_json_bytes(value: Any) -> bytes:
    _exact_json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() + b"\n"


def validate_execution_seal(value: Any) -> dict[str, Any]:
    _exact_json(value)
    if type(value) is not dict or set(value) != set(_TOP):
        raise ExecutionSealError("execution seal fields are malformed")
    fixed = {"schema_version": 2, "kind": "idea_foundry_campaign_execution_seal", "suite": "first-gate-all-sequential", "claim_scope": "synthetic_contract_execution_only"}
    if any(type(value[key]) is not type(expected) or value[key] != expected for key, expected in fixed.items()):
        raise ExecutionSealError("execution seal literals are invalid")
    if type(value["run_id"]) is not str or not value["run_id"] or type(value["seed"]) is not int:
        raise ExecutionSealError("execution seal identity types are invalid")
    git = value["git"]
    if type(git) is not dict or set(git) != {"commit", "tree"} or any(type(item) is not str or not _HEX.fullmatch(item) for item in git.values()):
        raise ExecutionSealError("git identity is invalid")
    if value["axis_order"] != list(AXES) or type(value["axis_order"]) is not list:
        raise ExecutionSealError("axis order is invalid")
    for key, expected in (("sources", SOURCES), ("inputs", INPUTS)):
        rows = value[key]
        if type(rows) is not list or [row.get("path") if type(row) is dict else None for row in rows] != list(expected):
            raise ExecutionSealError(f"{key} inventory is invalid")
        for row in rows:
            _repo_path(row["path"])
            if set(row) != {"path", "size_bytes", "sha256"} or type(row["size_bytes"]) is not int or row["size_bytes"] < 0 or type(row["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
                raise ExecutionSealError(f"{key} record is invalid")
    binaries = value["binaries"]
    keys = {"invocation_path", "resolved_target_path", "size_bytes", "sha256"}
    if type(binaries) is not list or len(binaries) != 1 or type(binaries[0]) is not dict or set(binaries[0]) != keys:
        raise ExecutionSealError("binary descriptor is invalid")
    binary = binaries[0]
    if any(type(binary[key]) is not str or not Path(binary[key]).is_absolute() for key in ("invocation_path", "resolved_target_path")) or type(binary["size_bytes"]) is not int or type(binary["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", binary["sha256"]):
        raise ExecutionSealError("binary descriptor values are invalid")
    if value["campaign_artifacts"] != list(CAMPAIGN_ARTIFACTS) or value["terminal_attempt_artifacts"] != list(ATTEMPT_ARTIFACTS):
        raise ExecutionSealError("artifact inventory is invalid")
    return value


def load_canonical_json(path: Path) -> Any:
    raw = _stable_read(path)
    try:
        value = decode_json_strict(raw)
    except Exception as exc:
        raise ExecutionSealError(f"invalid strict JSON: {path}") from exc
    if canonical_json_bytes(value) != raw:
        raise ExecutionSealError(f"non-canonical JSON bytes: {path}")
    return value


def capture_execution_seal(repo_root: Path, *, run_id: str, seed: int, executable: Path) -> dict[str, Any]:
    root = repo_root.absolute()
    root_info = _walk(root)
    if not stat.S_ISDIR(root_info.st_mode) or root.resolve() != root:
        raise ExecutionSealError("repository root is not a stable directory")
    if executable != Path(sys.executable) or not executable.is_absolute():
        raise ExecutionSealError("interpreter invocation does not equal sys.executable")
    if _git(root, "for-each-ref", "--format=%(refname)", "refs/replace").strip():
        raise ExecutionSealError("replacement refs are forbidden")
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    tree = _git(root, "rev-parse", "--verify", "HEAD^{tree}").decode().strip()
    if not _HEX.fullmatch(commit) or not _HEX.fullmatch(tree) or _git(root, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ExecutionSealError("repository identity is malformed or dirty")
    inventories = []
    for paths in (SOURCES, INPUTS):
        rows = []
        for relative in paths:
            _repo_path(relative)
            if _git(root, "ls-files", "-v", "--", relative).decode()[:1] != "H":
                raise ExecutionSealError(f"index flags are forbidden: {relative}")
            metadata = _git(root, "ls-tree", commit, "--", relative).decode().rstrip().split(None, 3)
            if len(metadata) != 4 or metadata[1] != "blob" or metadata[3].removeprefix("\t") != relative:
                raise ExecutionSealError(f"commit blob is absent: {relative}")
            live = _stable_read(root / relative)
            if _git(root, "cat-file", "blob", f"{commit}:{relative}") != live:
                raise ExecutionSealError(f"live source differs from commit: {relative}")
            rows.append(_record(relative, live))
        inventories.append(rows)
    invocation_info = _walk(executable, allow_final_symlink=True)
    target = executable.resolve(strict=True)
    target_bytes = _stable_read(target)
    if invocation_info != executable.lstat() or executable.resolve(strict=True) != target:
        raise ExecutionSealError("interpreter invocation changed")
    binary = {"invocation_path": str(executable), "resolved_target_path": str(target), "size_bytes": len(target_bytes), "sha256": hashlib.sha256(target_bytes).hexdigest()}
    if commit != _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip() or tree != _git(root, "rev-parse", "--verify", "HEAD^{tree}").decode().strip() or _git(root, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ExecutionSealError("repository identity changed during capture")
    seal = {"schema_version": 2, "kind": "idea_foundry_campaign_execution_seal", "suite": "first-gate-all-sequential", "claim_scope": "synthetic_contract_execution_only", "run_id": run_id, "seed": seed, "git": {"commit": commit, "tree": tree}, "axis_order": list(AXES), "sources": inventories[0], "inputs": inventories[1], "binaries": [binary], "campaign_artifacts": list(CAMPAIGN_ARTIFACTS), "terminal_attempt_artifacts": list(ATTEMPT_ARTIFACTS)}
    return validate_execution_seal(seal)


def claim_run_root(run_root: Path) -> None:
    run_root.parent.mkdir(parents=True, exist_ok=True)
    _walk(run_root.parent)
    try:
        os.mkdir(run_root)
    except OSError as exc:
        raise ExecutionSealError(f"run root is not absent: {run_root}") from exc
    _walk(run_root)


def publish_execution_seal(run_root: Path, seal: Any) -> bytes:
    raw = canonical_json_bytes(validate_execution_seal(seal))
    _walk(run_root)
    fd, temporary = tempfile.mkstemp(prefix=".campaign_execution_seal.", dir=run_root)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    final = run_root / CAMPAIGN_ARTIFACTS[0]
    try:
        os.link(temporary, final, follow_symlinks=False)
    except OSError as exc:
        raise ExecutionSealError("execution seal publication collided") from exc
    os.unlink(temporary)
    directory_fd = os.open(run_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if final.lstat().st_nlink != 1 or final.read_bytes() != raw:
        raise ExecutionSealError("published execution seal is not unique")
    return raw


def read_execution_seal(run_root: Path) -> tuple[dict[str, Any], str]:
    _walk(run_root)
    path = run_root / CAMPAIGN_ARTIFACTS[0]
    value = validate_execution_seal(load_canonical_json(path))
    return value, hashlib.sha256(canonical_json_bytes(value)).hexdigest()
