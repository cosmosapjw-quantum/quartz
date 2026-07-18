#!/usr/bin/env python3
"""Local experiment orchestrator for the QUARTZ idea lab.

The script is intentionally small: it discovers registered experiment lanes,
checks local requirements, prints executable plans, and optionally runs the
available commands into isolated output directories.
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "idea_lab.local.v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "idea_lab_local"
PROFILE_DEVICE = {"cpu": "cpu", "cuda": "cuda", "rocm": "cuda"}
RUNTIME_STATUSES = {
    "planned",
    "blocked",
    "running",
    "succeeded",
    "completed_no_promotion",
    "failed",
    "timeout",
    "interrupted",
    "skipped",
}
CONFIG_EXECUTION_STATUSES = {"available", "planned", "blocked", "dormant"}
EVIDENCE_STATUSES = {
    "skeleton_only",
    "mechanism_valid",
    "shadow_only",
    "conditional_only",
    "analysis_only",
}
DEPENDENCY_CONDITIONS = {
    "all_succeeded",
    "terminal_without_technical_failure",
    "all_terminal_non_technical_failure",
}
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
TEMPLATE_TOKEN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
RESERVED_TEMPLATE_KEYS = {
    "repo_root",
    "run_root",
    "output_root",
    "run_id",
    "lane_id",
    "output",
    "python",
    "profile",
    "device",
}


class ConfigError(ValueError):
    pass


class ArtifactError(RuntimeError):
    pass


class CampaignInterrupted(BaseException):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"campaign interrupted by signal {signum}")


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_executable_path(command: str) -> str:
    """Normalize command spelling without resolving virtualenv symlinks.

    A virtualenv launcher and its base interpreter may resolve to the same inode
    while selecting different environments.  Preserve that launcher identity,
    but remove the irrelevant relative-versus-absolute spelling difference so
    resume compares the interpreter that was actually selected.
    """
    if os.path.isabs(command):
        return os.path.normpath(command)
    if os.sep in command or (os.altsep is not None and os.altsep in command):
        return os.path.abspath(command)
    return shutil.which(command) or command


def sanitize_id(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)


def parse_sets(values: Sequence[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(f"--set expects key=value, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError("--set key must not be empty")
        out[key] = value
    return out


def format_obj(value: Any, ctx: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return TEMPLATE_TOKEN.sub(
            lambda match: str(ctx.get(match.group(1), match.group(0))),
            value,
        )
    if isinstance(value, list):
        return [format_obj(item, ctx) for item in value]
    if isinstance(value, dict):
        return {key: format_obj(val, ctx) for key, val in value.items()}
    return value


def _contains_template(value: Any) -> bool:
    if isinstance(value, str):
        return TEMPLATE_TOKEN.search(value) is not None
    if isinstance(value, list):
        return any(_contains_template(item) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_template(item) for item in value.values())
    return False


def validate_run_id(run_id: str) -> str:
    if not SAFE_RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ConfigError(
            "run id must be 1-128 safe characters, start with an alphanumeric, "
            "and contain only alphanumerics, '.', '_' or '-'"
        )
    return run_id


def confined_path(path: Path, root: Path, *, label: str) -> Path:
    candidate = path.resolve(strict=False)
    boundary = root.resolve(strict=False)
    try:
        candidate.relative_to(boundary)
    except ValueError as exc:
        raise ConfigError(f"{label} escapes campaign root: {path}") from exc
    return candidate


def hash_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    h = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        h.update(child.relative_to(path).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(child).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _resolve_declared_path(raw: str, *, config_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    return config_path.parent / path


def _validate_suites(cfg: Mapping[str, Any], seen: set[str]) -> None:
    suites = cfg.get("suites") or {}
    if not isinstance(suites, dict):
        raise ConfigError("suites must be an object")
    for suite, ids in suites.items():
        if not isinstance(ids, list):
            raise ConfigError(f"suite {suite!r} must be a list")
        unknown = sorted(set(map(str, ids)) - seen)
        if unknown:
            raise ConfigError(f"suite {suite!r} references unknown lanes: {unknown}")


def _topological_ids(lanes: Sequence[Mapping[str, Any]]) -> list[str]:
    ids = [str(lane["id"]) for lane in lanes]
    known = set(ids)
    dependencies: dict[str, list[str]] = {}
    for lane in lanes:
        lane_id = str(lane["id"])
        raw_dependencies = lane.get("depends_on") or []
        if not isinstance(raw_dependencies, list) or not all(
            isinstance(item, str) for item in raw_dependencies
        ):
            raise ConfigError(f"lane {lane_id} depends_on must be a string list")
        unknown = sorted(set(raw_dependencies) - known)
        if unknown:
            raise ConfigError(
                f"lane {lane_id} references unknown dependencies: {unknown}"
            )
        if lane_id in raw_dependencies:
            raise ConfigError(f"lane {lane_id} cannot depend on itself")
        dependencies[lane_id] = list(raw_dependencies)

    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []

    def visit(lane_id: str, stack: list[str]) -> None:
        if lane_id in visited:
            return
        if lane_id in visiting:
            start = stack.index(lane_id) if lane_id in stack else 0
            cycle = stack[start:] + [lane_id]
            raise ConfigError("lane dependency cycle: " + " -> ".join(cycle))
        visiting.add(lane_id)
        stack.append(lane_id)
        for dependency in dependencies[lane_id]:
            visit(dependency, stack)
        stack.pop()
        visiting.remove(lane_id)
        visited.add(lane_id)
        order.append(lane_id)

    for lane_id in ids:
        visit(lane_id, [])
    return order


def _validate_v2_config(cfg: dict[str, Any], path: Path) -> None:
    required_top = {
        "experiment_id",
        "axis_registry",
        "default_suite",
        "suites",
        "lanes",
        "source_dependencies",
    }
    missing_top = sorted(required_top - set(cfg))
    if missing_top:
        raise ConfigError(f"v2 registry missing top-level fields: {missing_top}")
    source_dependencies = cfg.get("source_dependencies")
    if (
        not isinstance(source_dependencies, list)
        or not source_dependencies
        or not all(isinstance(item, str) and item for item in source_dependencies)
    ):
        raise ConfigError("v2 source_dependencies must be a non-empty string list")
    resolved_source_dependencies: list[str] = []
    for raw_path in source_dependencies:
        source_path = _resolve_declared_path(raw_path, config_path=path)
        if not source_path.exists():
            raise ConfigError(f"missing campaign source dependency: {source_path}")
        resolved_source_dependencies.append(str(source_path.resolve()))
    lanes = cfg.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        raise ConfigError("registry requires a non-empty lanes list")
    required_lane = {
        "id",
        "axis_id",
        "role",
        "execution_status",
        "evidence_status",
        "claim_scope",
        "depends_on",
        "dependency_condition",
        "resource_profile",
        "inputs",
        "seed_contract",
        "steps",
        "expected_artifacts",
        "promotion_gate",
        "resume_policy",
        "prohibited_inferences",
    }
    seen: set[str] = set()
    axis_to_lanes: dict[str, list[str]] = {}
    for lane in lanes:
        if not isinstance(lane, dict):
            raise ConfigError("each v2 lane must be an object")
        missing = sorted(required_lane - set(lane))
        lane_id = str(lane.get("id", ""))
        if missing:
            raise ConfigError(
                f"lane {lane_id or '<missing>'} missing v2 fields: {missing}"
            )
        if not lane_id or lane_id in seen:
            raise ConfigError(f"missing or duplicate lane id: {lane_id!r}")
        if not SAFE_RUN_ID.fullmatch(lane_id) or lane_id in {".", ".."}:
            raise ConfigError(f"v2 lane id is not path-safe: {lane_id!r}")
        seen.add(lane_id)
        axis_id = str(lane["axis_id"])
        if not re.fullmatch(r"A\d{2}", axis_id):
            raise ConfigError(f"lane {lane_id} has invalid axis_id {axis_id!r}")
        axis_to_lanes.setdefault(axis_id, []).append(lane_id)
        if lane["execution_status"] not in CONFIG_EXECUTION_STATUSES:
            raise ConfigError(
                f"lane {lane_id} has invalid execution_status {lane['execution_status']!r}"
            )
        if lane["evidence_status"] not in EVIDENCE_STATUSES:
            raise ConfigError(
                f"lane {lane_id} has invalid evidence_status {lane['evidence_status']!r}"
            )
        condition = lane["dependency_condition"]
        if condition not in DEPENDENCY_CONDITIONS:
            raise ConfigError(
                f"lane {lane_id} has invalid dependency_condition {condition!r}"
            )
        resource = lane["resource_profile"]
        if (
            not isinstance(resource, dict)
            or not isinstance(resource.get("allowed"), list)
            or not resource["allowed"]
        ):
            raise ConfigError(
                f"lane {lane_id} resource_profile.allowed must be a non-empty list"
            )
        bad_profiles = sorted(set(map(str, resource["allowed"])) - set(PROFILE_DEVICE))
        if bad_profiles:
            raise ConfigError(
                f"lane {lane_id} has invalid resource profiles: {bad_profiles}"
            )
        hardware_gate = resource.get("hardware_gate")
        if hardware_gate not in {None, "cuda", "rocm"}:
            raise ConfigError(
                f"lane {lane_id} has invalid hardware_gate {hardware_gate!r}"
            )
        if not isinstance(lane["inputs"], list):
            raise ConfigError(f"lane {lane_id} inputs must be a list")
        for input_spec in lane["inputs"]:
            if not isinstance(input_spec, dict):
                raise ConfigError(f"lane {lane_id} input declarations must be objects")
            if "required" in input_spec and not isinstance(
                input_spec["required"], bool
            ):
                raise ConfigError(f"lane {lane_id} input required must be boolean")
            if "path" in input_spec and not isinstance(input_spec["path"], str):
                raise ConfigError(f"lane {lane_id} input path must be a string")
        if not isinstance(lane["seed_contract"], dict):
            raise ConfigError(f"lane {lane_id} seed_contract must be an object")
        if not isinstance(lane["steps"], list):
            raise ConfigError(f"lane {lane_id} steps must be a list")
        if lane["execution_status"] == "available" and not lane["steps"]:
            raise ConfigError(f"available lane {lane_id} requires at least one step")
        if lane["execution_status"] in {"blocked", "dormant"} and lane["steps"]:
            raise ConfigError(
                f"{lane['execution_status']} lane {lane_id} must not register executable steps"
            )
        step_names: list[str] = []
        for step in lane["steps"]:
            if (
                not isinstance(step, dict)
                or not isinstance(step.get("command"), list)
                or not step["command"]
            ):
                raise ConfigError(
                    f"lane {lane_id} has a step without a non-empty command list"
                )
            if not all(isinstance(token, str) for token in step["command"]):
                raise ConfigError(f"lane {lane_id} step command tokens must be strings")
            step_name = step.get("name")
            if not isinstance(step_name, str) or not step_name:
                raise ConfigError(f"lane {lane_id} has a step without a non-empty name")
            step_names.append(step_name)
            if "timeout_s" in step and (
                not isinstance(step["timeout_s"], int)
                or isinstance(step["timeout_s"], bool)
                or step["timeout_s"] <= 0
            ):
                raise ConfigError(
                    f"lane {lane_id} step {step_name} timeout_s must be a positive integer"
                )
            nested_artifacts = step.get("expected_artifacts") or []
            if not isinstance(nested_artifacts, list):
                raise ConfigError(
                    f"lane {lane_id} step {step_name} expected_artifacts must be a list"
                )
            for artifact in nested_artifacts:
                if not isinstance(artifact, dict) or not isinstance(
                    artifact.get("path"), str
                ):
                    raise ConfigError(
                        f"lane {lane_id} step {step_name} artifact requires a string path"
                    )
                if artifact.get("kind", "file") not in {
                    "file",
                    "json",
                    "jsonl",
                    "directory",
                }:
                    raise ConfigError(
                        f"lane {lane_id} step {step_name} has invalid artifact kind "
                        f"{artifact.get('kind')!r}"
                    )
                if "required" in artifact and not isinstance(
                    artifact["required"], bool
                ):
                    raise ConfigError(
                        f"lane {lane_id} step {step_name} artifact required must be boolean"
                    )
                explicit_owners = [
                    str(artifact[key])
                    for key in ("step", "after_step")
                    if key in artifact
                ]
                if any(owner != step_name for owner in explicit_owners):
                    raise ConfigError(
                        f"lane {lane_id} step {step_name} artifact has a conflicting owner"
                    )
        if len(step_names) != len(set(step_names)):
            raise ConfigError(f"lane {lane_id} has duplicate step names")
        if not isinstance(lane["expected_artifacts"], list):
            raise ConfigError(f"lane {lane_id} expected_artifacts must be a list")
        for artifact in lane["expected_artifacts"]:
            if not isinstance(artifact, dict) or not isinstance(
                artifact.get("path"), str
            ):
                raise ConfigError(
                    f"lane {lane_id} expected artifact requires a string path"
                )
            if artifact.get("kind", "file") not in {
                "file",
                "json",
                "jsonl",
                "directory",
            }:
                raise ConfigError(
                    f"lane {lane_id} has invalid artifact kind {artifact.get('kind')!r}"
                )
            if "required" in artifact and not isinstance(artifact["required"], bool):
                raise ConfigError(f"lane {lane_id} artifact required must be boolean")
            if (
                "step" in artifact
                and "after_step" in artifact
                and (str(artifact["step"]) != str(artifact["after_step"]))
            ):
                raise ConfigError(
                    f"lane {lane_id} artifact has conflicting step and after_step owners"
                )
            owner = artifact.get("step", artifact.get("after_step"))
            if owner is not None and str(owner) not in set(step_names):
                raise ConfigError(
                    f"lane {lane_id} artifact references unknown step owner {owner!r}"
                )
        if not isinstance(lane["promotion_gate"], dict):
            raise ConfigError(f"lane {lane_id} promotion_gate must be an object")
        if lane["promotion_gate"].get("allow_auto_promotion") is not False:
            raise ConfigError(f"lane {lane_id} must set allow_auto_promotion=false")
        if not isinstance(lane["resume_policy"], dict):
            raise ConfigError(f"lane {lane_id} resume_policy must be an object")
        resume_aliases = {
            "registry": ("require_registry_hash", "require_registry_hash_match"),
            "git": ("require_git_commit", "require_git_commit_match"),
            "source": ("require_source_hashes", "require_source_hash_match"),
            "input": ("require_input_hashes", "require_input_hash_match"),
        }
        for label, aliases in resume_aliases.items():
            values = [
                lane["resume_policy"][key]
                for key in aliases
                if key in lane["resume_policy"]
            ]
            if not values or not all(value is True for value in values):
                raise ConfigError(
                    f"lane {lane_id} v2 resume policy must require {label} hash matching"
                )
        if lane["resume_policy"].get("overwrite_is_resume", False) is not False:
            raise ConfigError(f"lane {lane_id} cannot enable overwrite as resume")
        if not isinstance(lane["prohibited_inferences"], list) or not all(
            isinstance(item, str) for item in lane["prohibited_inferences"]
        ):
            raise ConfigError(
                f"lane {lane_id} prohibited_inferences must be a string list"
            )
        if lane["role"] == "live" and (
            lane["evidence_status"] == "analysis_only"
            or lane["claim_scope"] == "analysis_only"
        ):
            raise ConfigError(f"analysis-only lane {lane_id} cannot declare role=live")

    _validate_suites(cfg, seen)
    _topological_ids(lanes)

    axis_registry = _resolve_declared_path(str(cfg["axis_registry"]), config_path=path)
    try:
        axis_payload = _strict_json_loads(axis_registry.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"missing axis registry: {axis_registry}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ConfigError(
            f"invalid axis registry JSON: {axis_registry}: {exc}"
        ) from exc
    registry_axes = [str(row.get("id", "")) for row in axis_payload.get("axes", [])]
    if len(registry_axes) != len(set(registry_axes)) or not registry_axes:
        raise ConfigError("axis registry contains missing or duplicate axis ids")
    missing_axes = sorted(set(registry_axes) - set(axis_to_lanes))
    extra_axes = sorted(set(axis_to_lanes) - set(registry_axes))
    if missing_axes or extra_axes:
        raise ConfigError(
            f"v2 axis coverage mismatch; missing={missing_axes}, extra={extra_axes}"
        )
    expected_axes = {f"A{idx:02d}" for idx in range(1, 27)}
    if set(registry_axes) != expected_axes:
        raise ConfigError("idea-foundry axis registry must contain exactly A01-A26")
    axis_contract = cfg.get("axis_registry_contract") or {}
    expected_registry_hash = (
        axis_contract.get("sha256") if isinstance(axis_contract, Mapping) else None
    )
    if expected_registry_hash and str(expected_registry_hash).lower() != sha256_file(
        axis_registry
    ):
        raise ConfigError("axis registry sha256 does not match axis_registry_contract")
    required_axis_ids = (
        axis_contract.get("required_axis_ids")
        if isinstance(axis_contract, Mapping)
        else None
    )
    if required_axis_ids is not None and set(map(str, required_axis_ids)) != set(
        registry_axes
    ):
        raise ConfigError(
            "axis_registry_contract.required_axis_ids does not match the registry"
        )
    policy = cfg.get("campaign_policy") or {}
    configured_runtime = (
        policy.get("runtime_statuses") if isinstance(policy, Mapping) else None
    )
    if (
        configured_runtime is not None
        and set(map(str, configured_runtime)) != RUNTIME_STATUSES
    ):
        raise ConfigError(
            "campaign_policy.runtime_statuses must match the fixed v2 runtime status vocabulary"
        )
    cfg["_axis_registry_path"] = str(axis_registry.resolve())
    cfg["_axis_ids"] = registry_axes
    cfg["_source_dependencies"] = resolved_source_dependencies


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        cfg = _strict_json_loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"missing registry: {path}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ConfigError(f"invalid registry JSON: {path}: {exc}") from exc
    version = cfg.get("format_version")
    if version not in {1, 2}:
        raise ConfigError("unsupported registry format_version")
    if version == 2:
        _validate_v2_config(cfg, path.resolve())
        cfg["_config_path"] = str(path.resolve())
        return cfg
    lanes = cfg.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        raise ConfigError("registry requires a non-empty lanes list")
    seen: set[str] = set()
    for lane in lanes:
        lane_id = str(lane.get("id", ""))
        if not lane_id or lane_id in seen:
            raise ConfigError(f"missing or duplicate lane id: {lane_id!r}")
        seen.add(lane_id)
        if lane.get("status") not in {"available", "planned", "deprecated", "dormant"}:
            raise ConfigError(
                f"lane {lane_id} has invalid status {lane.get('status')!r}"
            )
    _validate_suites(cfg, seen)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def lane_index(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(lane["id"]): dict(lane) for lane in cfg["lanes"]}


def expand_lanes(
    cfg: Mapping[str, Any], suites: Sequence[str] | None, lane_ids: Sequence[str] | None
) -> list[dict[str, Any]]:
    idx = lane_index(cfg)
    selected: list[str] = []
    if lane_ids:
        selected.extend(lane_ids)
    if suites:
        for suite in suites:
            if suite not in cfg.get("suites", {}):
                raise ConfigError(f"unknown suite: {suite}")
            selected.extend(str(x) for x in cfg["suites"][suite])
    if not selected:
        default_suite = str(cfg.get("default_suite", "smoke"))
        if default_suite not in cfg.get("suites", {}):
            raise ConfigError(f"default suite {default_suite!r} not registered")
        selected.extend(str(x) for x in cfg["suites"][default_suite])
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_with_dependencies(lane_id: str) -> None:
        if lane_id not in idx:
            raise ConfigError(f"unknown lane: {lane_id}")
        if lane_id in seen:
            return
        if cfg.get("format_version") == 2:
            for dependency in idx[lane_id].get("depends_on", []) or []:
                add_with_dependencies(str(dependency))
        out.append(idx[lane_id])
        seen.add(lane_id)

    for lane_id in selected:
        add_with_dependencies(str(lane_id))
    return out


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@functools.lru_cache(maxsize=256)
def target_module_exists(python: str, name: str) -> tuple[bool, str | None]:
    code = (
        "import importlib.util,json; "
        f"s=importlib.util.find_spec({name!r}); "
        "print(json.dumps({'found': s is not None, 'origin': getattr(s, 'origin', None)}))"
    )
    try:
        proc = subprocess.run(
            [python, "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, repr(exc)
    if proc.returncode != 0:
        return (
            False,
            proc.stderr.strip() or f"target interpreter exited {proc.returncode}",
        )
    try:
        payload = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return False, "target interpreter returned invalid module-probe JSON"
    return bool(payload.get("found")), payload.get("origin")


def profile_supported(lane: Mapping[str, Any], profile: str) -> bool:
    resource = lane.get("resource_profile")
    if isinstance(resource, Mapping):
        allowed = resource.get("allowed")
        return not allowed or profile in allowed
    profiles = lane.get("profiles")
    return not profiles or profile in profiles


def lane_requirements(lane: Mapping[str, Any]) -> Mapping[str, Any]:
    req = lane.get("requirements") or {}
    if not isinstance(req, Mapping):
        return {}
    return req


@dataclass
class LanePlan:
    lane: dict[str, Any]
    status: str
    blockers: list[str]
    commands: list[dict[str, Any]]
    output_dir: str
    artifact_globs: list[str]
    expected_artifacts: list[dict[str, Any]] = field(default_factory=list)
    input_records: list[dict[str, Any]] = field(default_factory=list)
    source_paths: list[str] = field(default_factory=list)
    format_version: int = 1

    @property
    def ready(self) -> bool:
        return self.status == "READY"


def build_context(
    *,
    repo_root: Path,
    profile: str,
    output_root: Path,
    run_id: str,
    lane_id: str,
    python: str,
    device: str | None,
    variables: Mapping[str, str],
) -> dict[str, Any]:
    validate_run_id(run_id)
    reserved = sorted(set(variables) & RESERVED_TEMPLATE_KEYS)
    if reserved:
        raise ConfigError(f"--set cannot override reserved template keys: {reserved}")
    run_root = output_root / run_id
    lane_output = run_root / sanitize_id(lane_id)
    ctx: dict[str, Any] = {
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "output_root": str(output_root),
        "run_id": run_id,
        "lane_id": lane_id,
        "output": str(lane_output),
        "python": python,
        "profile": profile,
        "device": device or PROFILE_DEVICE[profile],
    }
    ctx.update(variables)
    return ctx


def _v2_output_path(raw: str, *, run_root: Path, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = run_root / path
    return confined_path(path, run_root, label=label)


def _command_output_path_errors(
    command: Sequence[Any], *, run_root: Path, lane_id: str
) -> list[str]:
    output_flags = {
        "--output",
        "--output-dir",
        "--out",
        "--results-dir",
        "--artifact-dir",
    }
    tokens = [str(item) for item in command]
    errors: list[str] = []
    for idx, token in enumerate(tokens):
        raw_path: str | None = None
        if token in output_flags and idx + 1 < len(tokens):
            raw_path = tokens[idx + 1]
        else:
            for flag in output_flags:
                prefix = flag + "="
                if token.startswith(prefix):
                    raw_path = token[len(prefix) :]
                    break
        if raw_path is None or _contains_template(raw_path):
            continue
        try:
            _v2_output_path(
                raw_path, run_root=run_root, label=f"lane {lane_id} command output"
            )
        except ConfigError as exc:
            errors.append(str(exc))
    return errors


def _bind_artifact_lane_identity(
    spec: dict[str, Any], lane: Mapping[str, Any]
) -> dict[str, Any]:
    schema = spec.get("schema")
    if not isinstance(schema, Mapping):
        return spec
    schema = dict(schema)
    required = {str(key) for key in schema.get("required_keys", []) or []}
    equals = dict(schema.get("equals") or {})
    identity = {
        "axis_id": str(lane["axis_id"]),
        "role": str(lane["role"]),
        "evidence_status": str(lane["evidence_status"]),
        "claim_scope": str(lane["claim_scope"]),
    }
    artifact_name = Path(str(spec.get("path", ""))).name
    enforced = {"axis_id", "role"}
    if artifact_name in {"run_manifest.json", "summary.json"}:
        enforced.add("claim_scope")
    if artifact_name == "summary.json":
        enforced.add("evidence_status")
    for key, expected in identity.items():
        if key in equals and equals[key] != expected:
            raise ConfigError(
                f"lane {lane['id']} artifact schema identity conflict for {key}: "
                f"configured {equals[key]!r}, lane declares {expected!r}"
            )
        if key in required or key in enforced:
            equals[key] = expected
    if "auto_promoted" in required:
        configured = equals.get("auto_promoted", False)
        if configured is not False:
            raise ConfigError(
                f"lane {lane['id']} artifact schema cannot allow auto_promoted=true"
            )
        equals["auto_promoted"] = False
    if "promotion" in required:
        configured = equals.get("promotion.auto", False)
        if configured is not False:
            raise ConfigError(
                f"lane {lane['id']} artifact schema cannot allow promotion.auto=true"
            )
        equals["promotion.auto"] = False
    if equals:
        schema["equals"] = equals
    spec["schema"] = schema
    return spec


def _input_record(
    spec: Mapping[str, Any], *, ctx: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, Any], str | None]:
    formatted = dict(format_obj(dict(spec), ctx))
    record: dict[str, Any] = {
        "name": str(formatted.get("name", formatted.get("path", "input"))),
        "kind": str(formatted.get("kind", "file")),
        "required": bool(formatted.get("required", True)),
    }
    raw_path = formatted.get("path")
    if raw_path is None:
        record["semantic_sha256"] = hashlib.sha256(
            json.dumps(formatted, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return record, None
    if not isinstance(raw_path, str) or _contains_template(raw_path):
        return record, f"unresolved input path template: {raw_path}"
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve(strict=False)
    record["path"] = str(path)
    if not path.exists():
        record["exists"] = False
        if record["required"]:
            return record, f"missing required input: {path}"
        return record, None
    record["exists"] = True
    record["sha256"] = hash_path(path)
    expected_hash = formatted.get("sha256")
    if expected_hash and str(expected_hash).lower() != record["sha256"]:
        return record, f"input sha256 mismatch: {path}"
    return record, None


def _source_paths_for_plan(
    lane: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    ctx: Mapping[str, Any],
    campaign_source_paths: Sequence[str] = (),
) -> list[str]:
    candidates: set[Path] = {Path(__file__).resolve()}
    for raw in campaign_source_paths:
        path = Path(str(raw))
        if not path.is_absolute():
            path = repo_root / path
        if path.exists():
            candidates.add(path.resolve())
    for raw in lane_requirements(lane).get("paths", []) or []:
        formatted = str(format_obj(raw, ctx))
        if _contains_template(formatted):
            continue
        path = Path(formatted)
        if not path.is_absolute():
            path = repo_root / path
        if path.exists():
            candidates.add(path.resolve())
    for step in commands:
        for token in step.get("command", []) or []:
            path = Path(str(token))
            if not path.is_absolute():
                path = repo_root / path
            if path.is_file():
                candidates.add(path.resolve())
    return [str(path) for path in sorted(candidates, key=str)]


def plan_lane(
    lane: Mapping[str, Any],
    *,
    repo_root: Path,
    profile: str,
    output_root: Path,
    run_id: str,
    python: str,
    device: str | None,
    variables: Mapping[str, str],
    campaign_source_paths: Sequence[str] = (),
) -> LanePlan:
    lane_id = str(lane["id"])
    version = 2 if "execution_status" in lane else 1
    if version == 2:
        python = canonical_executable_path(python)
    ctx = build_context(
        repo_root=repo_root,
        profile=profile,
        output_root=output_root,
        run_id=run_id,
        lane_id=lane_id,
        python=python,
        device=device,
        variables=variables,
    )
    blockers: list[str] = []
    if not profile_supported(lane, profile):
        blockers.append(f"profile {profile!r} not supported by lane")

    status = str(lane.get("execution_status", lane.get("status", "available")))
    if status != "available":
        for item in lane.get("blocked_by", []) or []:
            prefix = "planned" if version == 1 else status
            blockers.append(f"{prefix}: {item}")
        if version == 2 and not (lane.get("blocked_by") or []):
            blockers.append(f"execution_status={status}")

    if version == 2:
        resource = lane.get("resource_profile") or {}
        hardware_gate = resource.get("hardware_gate")
        if hardware_gate and profile != hardware_gate:
            blockers.append(
                f"hardware gate {hardware_gate!r} requires matching profile"
            )

    req = lane_requirements(lane)
    for var in req.get("variables", []) or []:
        if str(var) not in ctx or not str(ctx[str(var)]):
            blockers.append(f"missing --set {var}=...")
    for cmd in req.get("commands", []) or []:
        cmd_name = str(format_obj(cmd, ctx))
        if not command_exists(cmd_name):
            blockers.append(f"missing command: {cmd_name}")
    for module in req.get("python_modules", []) or []:
        module_name = str(format_obj(module, ctx))
        found, detail = target_module_exists(python, module_name)
        if not found:
            blockers.append(f"missing python module: {module_name}")
    for raw_path in req.get("paths", []) or []:
        formatted = str(format_obj(raw_path, ctx))
        if "{" in formatted or "}" in formatted:
            blockers.append(f"unresolved path template: {formatted}")
            continue
        path = Path(formatted)
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            blockers.append(f"missing path: {formatted}")

    commands: list[dict[str, Any]] = []
    for step in lane.get("steps", []) or []:
        step_payload = dict(format_obj(dict(step), ctx))
        if version == 2 and _contains_template(step_payload):
            blockers.append(
                f"unresolved step template: {step_payload.get('name', 'unnamed')}"
            )
        commands.append(step_payload)
    artifact_globs = [
        str(format_obj(item, ctx)) for item in lane.get("artifact_globs", []) or []
    ]
    expected_artifacts: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    if version == 2:
        run_root = Path(str(ctx["run_root"]))
        for idx, step in enumerate(commands, start=1):
            blockers.extend(
                _command_output_path_errors(
                    step.get("command", []),
                    run_root=run_root,
                    lane_id=lane_id,
                )
            )
            stdout_target = step.get("stdout")
            if stdout_target:
                try:
                    step["stdout"] = str(
                        _v2_output_path(
                            str(stdout_target),
                            run_root=run_root,
                            label=f"lane {lane_id} step {idx} stdout",
                        )
                    )
                except ConfigError as exc:
                    blockers.append(str(exc))
            for artifact in step.get("expected_artifacts", []) or []:
                formatted = dict(format_obj(dict(artifact), ctx))
                formatted["step"] = str(step.get("name", f"step{idx}"))
                expected_artifacts.append(_bind_artifact_lane_identity(formatted, lane))
        for artifact in lane.get("expected_artifacts", []) or []:
            formatted = dict(format_obj(dict(artifact), ctx))
            expected_artifacts.append(_bind_artifact_lane_identity(formatted, lane))
        for artifact in expected_artifacts:
            raw_path = artifact.get("path")
            if not isinstance(raw_path, str) or _contains_template(raw_path):
                blockers.append(f"unresolved expected artifact path: {raw_path}")
                continue
            try:
                artifact["path"] = str(
                    _v2_output_path(
                        raw_path,
                        run_root=run_root,
                        label=f"lane {lane_id} expected artifact",
                    )
                )
            except ConfigError as exc:
                blockers.append(str(exc))
        for spec in lane.get("inputs", []) or []:
            if not isinstance(spec, Mapping):
                blockers.append(f"invalid input declaration: {spec!r}")
                continue
            record, error = _input_record(spec, ctx=ctx, repo_root=repo_root)
            input_records.append(record)
            if error:
                blockers.append(error)
    plan_status = "READY" if not blockers else "BLOCKED"
    return LanePlan(
        dict(lane),
        plan_status,
        blockers,
        commands,
        str(ctx["output"]),
        artifact_globs,
        expected_artifacts=expected_artifacts,
        input_records=input_records,
        source_paths=_source_paths_for_plan(
            lane,
            commands,
            repo_root=repo_root,
            ctx=ctx,
            campaign_source_paths=campaign_source_paths,
        ),
        format_version=version,
    )


def make_run_id(prefix: str | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return validate_run_id(prefix or f"run-{stamp}")


def _target_python_report(
    python: str, modules: Sequence[str]
) -> tuple[dict[str, Any] | None, str | None]:
    code = """
import importlib.util, json, platform, sys
mods = {name: importlib.util.find_spec(name) is not None for name in %r}
payload = {
    "executable": sys.executable,
    "version": platform.python_version(),
    "modules": mods,
    "torch": None,
}
try:
    import torch
    names = []
    properties_ok = True
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(idx)
                names.append(str(getattr(props, "name", torch.cuda.get_device_name(idx))))
            except Exception:
                properties_ok = False
    payload["torch"] = {
        "version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_names": names,
        "device_properties_ok": properties_ok,
        "cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
        "hip_build": getattr(getattr(torch, "version", None), "hip", None),
    }
except Exception as exc:
    payload["torch_error"] = repr(exc)
print(json.dumps(payload, sort_keys=True))
""" % (list(modules),)
    try:
        proc = subprocess.run(
            [python, "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, repr(exc)
    if proc.returncode != 0:
        return (
            None,
            proc.stderr.strip() or f"target interpreter exited {proc.returncode}",
        )
    try:
        return json.loads(proc.stdout.strip()), None
    except json.JSONDecodeError:
        return None, "target interpreter returned invalid doctor JSON"


def _probe_driver(commands: Sequence[Sequence[str]]) -> tuple[bool, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for command in commands:
        executable = shutil.which(command[0])
        if executable is None:
            attempts.append({"command": list(command), "found": False})
            continue
        try:
            proc = subprocess.run(
                list(command),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            attempts.append(
                {"command": list(command), "found": True, "error": repr(exc)}
            )
            continue
        attempt = {
            "command": list(command),
            "found": True,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[:2000],
            "stderr": proc.stderr.strip()[:2000],
        }
        attempts.append(attempt)
        if proc.returncode == 0 and bool(proc.stdout.strip()):
            return True, {"successful_probe": attempt, "attempts": attempts}
    return False, {"successful_probe": None, "attempts": attempts}


def collect_doctor(
    repo_root: Path, *, profile: str, python: str, strict: bool
) -> dict[str, Any]:
    python = canonical_executable_path(python)
    checks: list[dict[str, Any]] = []

    def add(kind: str, name: str, ok: bool, detail: str | None = None) -> None:
        checks.append({"kind": kind, "name": name, "ok": bool(ok), "detail": detail})

    python_ok = Path(python).exists() or command_exists(python)
    modules = ["numpy", "pytest"] if strict else ["numpy"]
    target, target_error = (
        _target_python_report(python, modules)
        if python_ok
        else (None, "interpreter not found")
    )
    add(
        "python",
        python,
        bool(target),
        json.dumps({"error": target_error}, sort_keys=True)
        if target is None
        else json.dumps(
            {"executable": target.get("executable"), "version": target.get("version")},
            sort_keys=True,
        ),
    )
    for cmd in ["git", "cargo", "rustc"]:
        add("command", cmd, command_exists(cmd), shutil.which(cmd))
    for module in modules:
        found = bool(target and target.get("modules", {}).get(module))
        add("python_module", module, found, f"probed by {python}")
    if strict:
        rust_bin = repo_root / "target" / "release" / "mcts_demo"
        add("path", str(rust_bin.relative_to(repo_root)), rust_bin.exists())
    torch_detail = target.get("torch") if target else None
    if profile == "cpu":
        add(
            "python_module",
            "torch",
            bool(torch_detail),
            json.dumps(torch_detail or {}, sort_keys=True),
        )
    elif profile == "cuda":
        names = [
            str(item).lower() for item in (torch_detail or {}).get("device_names", [])
        ]
        nvidia_tokens = (
            "nvidia",
            "geforce",
            "quadro",
            "tesla",
            "rtx",
            "gtx",
            "a100",
            "h100",
            "b100",
            "b200",
            "l40",
            "v100",
            "t4",
        )
        vendor_ok = bool(
            names
            and all(any(token in name for token in nvidia_tokens) for name in names)
        )
        build_ok = bool(
            torch_detail
            and torch_detail.get("cuda_build")
            and not torch_detail.get("hip_build")
        )
        device_ok = bool(
            torch_detail
            and torch_detail.get("cuda_available")
            and torch_detail.get("device_count", 0) > 0
            and torch_detail.get("device_properties_ok")
            and vendor_ok
        )
        driver_ok, driver_detail = _probe_driver((("nvidia-smi", "-L"),))
        detail = dict(torch_detail or {})
        detail.update(
            {
                "nvidia_device_visible": device_ok,
                "nvidia_name_contract": vendor_ok,
                "driver_runtime_probe_ok": driver_ok,
                "driver_probe": driver_detail,
            }
        )
        add(
            "accelerator",
            "torch[cuda]",
            build_ok and device_ok and driver_ok,
            json.dumps(detail, sort_keys=True),
        )
    else:
        names = [
            str(item).lower() for item in (torch_detail or {}).get("device_names", [])
        ]
        vendor_ok = bool(
            names
            and any(
                any(tag in name for tag in ("amd", "radeon", "instinct", "mi"))
                for name in names
            )
        )
        build_ok = bool(torch_detail and torch_detail.get("hip_build"))
        device_ok = bool(
            torch_detail
            and torch_detail.get("cuda_available")
            and torch_detail.get("device_count", 0) > 0
            and torch_detail.get("device_properties_ok")
            and vendor_ok
        )
        driver_ok, driver_detail = _probe_driver(
            (
                ("rocminfo",),
                ("rocm-smi", "--showproductname"),
            )
        )
        detail = dict(torch_detail or {})
        detail.update(
            {
                "amd_device_visible": device_ok,
                "amd_name_contract": vendor_ok,
                "driver_runtime_probe_ok": driver_ok,
                "driver_probe": driver_detail,
            }
        )
        add(
            "accelerator",
            "torch[rocm]",
            build_ok and device_ok and driver_ok,
            json.dumps(detail, sort_keys=True),
        )
    return {
        "profile": profile,
        "platform": platform.platform(),
        "python": python,
        "strict": strict,
        "checks": checks,
        "target_python": target,
        "ok": all(item["ok"] for item in checks if strict or item["kind"] != "path"),
    }


def print_text_plan(plans: Sequence[LanePlan]) -> None:
    for plan in plans:
        lane = plan.lane
        print(f"[{plan.status}] {lane['id']} — {lane.get('title', '')}")
        if lane.get("description"):
            print(f"  {lane['description']}")
        print(f"  output: {plan.output_dir}")
        for blocker in plan.blockers:
            print(f"  blocker: {blocker}")
        for idx, step in enumerate(plan.commands, start=1):
            print(f"  step {idx}: {step.get('name', f'step{idx}')}")
            print("    " + " ".join(map(str, step.get("command", []))))
        if plan.artifact_globs:
            print("  artifacts:")
            for item in plan.artifact_globs:
                print(f"    {item}")
        print()


def git_info(repo_root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                args, cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    return {
        "branch": run(["git", "branch", "--show-current"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _json_lookup(payload: Any, dotted: str) -> tuple[bool, Any]:
    current = payload
    if not dotted:
        return True, current
    for part in dotted.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
            continue
        return False, None
    return True, current


def _schema_type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        ),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, False)


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {token}")


def _strict_json_loads(raw: str) -> Any:
    payload = json.loads(raw, parse_constant=_reject_json_constant)

    def visit(value: Any, location: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {location}")
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                visit(child, f"{location}[{idx}]")

    visit(payload, "$")
    return payload


def _validate_inline_schema(
    value: Any, schema: Mapping[str, Any], *, location: str = "$"
) -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not _schema_type_matches(value, str(expected_type)):
        return [
            f"{location}: expected type {expected_type}, got {type(value).__name__}"
        ]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{location}: expected const {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{location}: {value!r} is not in enum")
    required = schema.get("required") or []
    if isinstance(value, Mapping):
        for key in required:
            if key not in value:
                errors.append(f"{location}: missing required key {key!r}")
        properties = schema.get("properties") or {}
        if isinstance(properties, Mapping):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, Mapping):
                    errors.extend(
                        _validate_inline_schema(
                            value[key], child_schema, location=f"{location}.{key}"
                        )
                    )
    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            errors.append(f"{location}: expected at least {schema['minItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for idx, item in enumerate(value):
                errors.extend(
                    _validate_inline_schema(
                        item, item_schema, location=f"{location}[{idx}]"
                    )
                )
    return errors


def _validate_contract_schema(
    payload: Any, schema: Mapping[str, Any], *, location: str
) -> list[str]:
    errors = _validate_inline_schema(payload, schema, location=location)
    required_keys = schema.get("required_keys") or []
    for key in required_keys:
        found, _ = _json_lookup(payload, str(key))
        if not found:
            errors.append(f"{location}: missing required key path {key!r}")
    equals = schema.get("equals") or {}
    if isinstance(equals, Mapping):
        for key, expected in equals.items():
            found, actual = _json_lookup(payload, str(key))
            if not found:
                errors.append(f"{location}: equals path {key!r} is missing")
            elif actual != expected:
                errors.append(
                    f"{location}: {key!r} expected {expected!r}, got {actual!r}"
                )
    return errors


def _validate_embedded_source_hashes(
    payload: Any,
    *,
    artifact_path: Path,
    source_root: Path,
) -> None:
    if not isinstance(payload, Mapping) or "source_hashes" not in payload:
        return
    records = payload["source_hashes"]
    if isinstance(records, Mapping):
        records = [{"path": path, "sha256": digest} for path, digest in records.items()]
    if not isinstance(records, list) or not records:
        raise ArtifactError(f"invalid source_hashes contract in {artifact_path}")
    root = source_root.resolve()
    for record in records:
        if not isinstance(record, Mapping):
            raise ArtifactError(
                f"invalid source hash row in {artifact_path}: {record!r}"
            )
        raw_path = record.get("path")
        expected = record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ArtifactError(
                f"source hash row requires path and sha256 in {artifact_path}"
            )
        source_path = Path(raw_path)
        if not source_path.is_absolute():
            source_path = root / source_path
        source_path = source_path.resolve(strict=False)
        try:
            source_path.relative_to(root)
        except ValueError as exc:
            raise ArtifactError(
                f"embedded source path escapes repository root in {artifact_path}: {raw_path}"
            ) from exc
        if not source_path.exists():
            raise ArtifactError(
                f"embedded source is missing for {artifact_path}: {source_path}"
            )
        actual = hash_path(source_path)
        if actual != expected.lower():
            raise ArtifactError(
                f"embedded source hash drift for {artifact_path}: {source_path}"
            )


def validate_expected_artifact(
    spec: Mapping[str, Any],
    *,
    campaign_root: Path,
    previous_sha256: str | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    raw_path = spec.get("path")
    if not isinstance(raw_path, str):
        raise ArtifactError("expected artifact has no string path")
    try:
        path = confined_path(Path(raw_path), campaign_root, label="expected artifact")
    except ConfigError as exc:
        raise ArtifactError(str(exc)) from exc
    raw_required = spec.get("required", True)
    if not isinstance(raw_required, bool):
        raise ArtifactError(f"expected artifact required flag must be boolean: {path}")
    required = raw_required
    if not path.exists():
        if required:
            raise ArtifactError(f"required artifact is missing: {path}")
        return {
            "path": str(path),
            "kind": spec.get("kind", "file"),
            "required": False,
            "exists": False,
        }
    kind = str(spec.get("kind", "file"))
    if kind not in {"file", "json", "jsonl", "directory"}:
        raise ArtifactError(f"unsupported expected artifact kind {kind!r}: {path}")
    if kind == "directory":
        if not path.is_dir():
            raise ArtifactError(
                f"expected directory artifact, got non-directory: {path}"
            )
        for descendant in path.rglob("*"):
            if descendant.is_symlink():
                raise ArtifactError(
                    f"directory artifact contains a forbidden symlink: {descendant}"
                )
            try:
                descendant.resolve(strict=False).relative_to(campaign_root.resolve())
            except ValueError as exc:
                raise ArtifactError(
                    f"directory artifact descendant escapes campaign root: {descendant}"
                ) from exc
        actual_hash = hash_path(path)
        payload: Any = None
    else:
        if not path.is_file():
            raise ArtifactError(f"expected file artifact, got non-file: {path}")
        actual_hash = sha256_file(path)
        payload = None
        if kind == "json":
            try:
                payload = _strict_json_loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ArtifactError(f"invalid JSON artifact {path}: {exc}") from exc
        elif kind == "jsonl":
            rows: list[Any] = []
            try:
                for number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if line.strip():
                        rows.append(_strict_json_loads(line))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ArtifactError(
                    f"invalid JSONL artifact {path} at line {number}: {exc}"
                ) from exc
            payload = rows
    expected_hash = spec.get("sha256")
    if expected_hash and str(expected_hash).lower() != actual_hash:
        raise ArtifactError(f"configured sha256 mismatch for {path}")
    if previous_sha256 and previous_sha256 != actual_hash:
        raise ArtifactError(f"artifact hash drift since prior successful step: {path}")
    schema = spec.get("schema")
    if schema:
        if kind not in {"json", "jsonl"}:
            raise ArtifactError(
                f"schema validation requires json/jsonl artifact: {path}"
            )
        if not isinstance(schema, Mapping):
            raise ArtifactError(f"artifact schema must be an object: {path}")
        if kind == "jsonl":
            schema_errors = []
            min_rows = int(schema.get("min_rows", schema.get("minItems", 0)) or 0)
            if len(payload) < min_rows:
                schema_errors.append(f"{path}: expected at least {min_rows} JSONL rows")
            row_schema = dict(schema.get("items") or {})
            for key in ("required_keys", "equals"):
                if key in schema:
                    row_schema[key] = schema[key]
            for idx, row in enumerate(payload):
                schema_errors.extend(
                    _validate_contract_schema(
                        row, row_schema, location=f"{path}:{idx + 1}"
                    )
                )
        else:
            schema_errors = _validate_contract_schema(
                payload, schema, location=str(path)
            )
        if schema_errors:
            raise ArtifactError("; ".join(schema_errors))
    if source_root is not None and kind == "json":
        _validate_embedded_source_hashes(
            payload,
            artifact_path=path,
            source_root=source_root,
        )
    return {
        "path": str(path),
        "kind": kind,
        "required": required,
        "exists": True,
        "sha256": actual_hash,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _expected_for_step(
    plan: LanePlan, *, step_name: str, index: int
) -> list[dict[str, Any]]:
    last = index == len(plan.commands)
    selected: list[dict[str, Any]] = []
    for spec in plan.expected_artifacts:
        owner = spec.get("step", spec.get("after_step"))
        if owner is not None and str(owner) == step_name:
            selected.append(spec)
        elif owner is None and last:
            selected.append(spec)
    return selected


def _hash_sources(plans: Sequence[LanePlan]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw_path in sorted({item for plan in plans for item in plan.source_paths}):
        path = Path(raw_path)
        if path.exists():
            hashes[str(path.resolve())] = hash_path(path)
    return hashes


def _expanded_plan_sha256(plans: Sequence[LanePlan]) -> str:
    payload = [
        {
            "lane_id": str(plan.lane["id"]),
            "status": plan.status,
            "commands": plan.commands,
            "output_dir": plan.output_dir,
            "expected_artifacts": plan.expected_artifacts,
            "input_records": plan.input_records,
            "source_paths": plan.source_paths,
            "format_version": plan.format_version,
        }
        for plan in plans
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def campaign_fingerprint(
    plans: Sequence[LanePlan],
    *,
    cfg_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    inputs: dict[str, list[dict[str, Any]]] = {}
    for plan in plans:
        inputs[str(plan.lane["id"])] = plan.input_records
    git = git_info(repo_root)
    return {
        "registry_sha256": sha256_file(cfg_path),
        "git_commit": git.get("commit"),
        "source_hashes": _hash_sources(plans),
        "input_records": inputs,
        "expanded_plan_sha256": _expanded_plan_sha256(plans),
    }


def verify_resume_fingerprint(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    plans: Sequence[LanePlan],
) -> None:
    if previous.get("expanded_plan_sha256") != current.get("expanded_plan_sha256"):
        raise ConfigError("resume refused: expanded_plan_sha256 changed")
    comparisons = {
        "require_registry_hash": "registry_sha256",
        "require_git_commit": "git_commit",
        "require_source_hashes": "source_hashes",
        "require_input_hashes": "input_records",
    }
    for key in comparisons.values():
        if previous.get(key) != current.get(key):
            raise ConfigError(f"resume refused: {key} changed")


def _terminate_process_group(
    proc: subprocess.Popen[bytes], *, grace_s: float = 1.0
) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _run_streaming_process(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    stdout_path: Path | None,
    stdout_target_path: Path | None,
    timeout_s: int | None,
) -> tuple[int, float, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stdout_target_path is not None:
        stdout_target_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # Attempt paths are immutable evidence.  Exclusive creation turns any
    # bookkeeping/path collision into a technical failure instead of silently
    # truncating an earlier attempt.
    with log_path.open("xb") as log_file:
        log_file.write(
            ("$ " + " ".join(command) + "\n\n").encode("utf-8", errors="replace")
        )
        stdout_file = None
        stdout_target_file = None
        try:
            stdout_file = stdout_path.open("xb") if stdout_path is not None else None
            stdout_target_file = (
                stdout_target_path.open("wb")
                if stdout_target_path is not None and stdout_target_path != stdout_path
                else None
            )
            try:
                proc = subprocess.Popen(
                    list(command),
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    bufsize=0,
                )
            except OSError as exc:
                log_file.write(
                    (f"[launch_error]\n{exc!r}\n").encode("utf-8", errors="replace")
                )
                return 127, time.perf_counter() - t0, "failed"
            lock = threading.Lock()

            def pump(stream: Any, label: bytes, mirrors: Sequence[Any] = ()) -> None:
                with lock:
                    log_file.write(b"\n[" + label + b"]\n")
                    log_file.flush()
                while True:
                    chunk = stream.read(65536)
                    if not chunk:
                        break
                    with lock:
                        log_file.write(chunk)
                        log_file.flush()
                    for mirror in mirrors:
                        mirror.write(chunk)
                        mirror.flush()

            stdout_mirrors = tuple(
                item for item in (stdout_file, stdout_target_file) if item is not None
            )
            stdout_thread = threading.Thread(
                target=pump,
                args=(proc.stdout, b"stdout", stdout_mirrors),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=pump, args=(proc.stderr, b"stderr"), daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            status = "succeeded"
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                status = "timeout"
                _terminate_process_group(proc)
            except (KeyboardInterrupt, CampaignInterrupted):
                status = "interrupted"
                _terminate_process_group(proc)
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            rc = int(proc.returncode if proc.returncode is not None else 1)
            if status == "timeout":
                rc = 124
                log_file.write(f"\n[TIMEOUT after {timeout_s}s]\n".encode("utf-8"))
            elif status == "interrupted":
                rc = 130
                log_file.write(b"\n[INTERRUPTED]\n")
            elif rc != 0:
                status = "failed"
            return rc, time.perf_counter() - t0, status
        finally:
            if stdout_file is not None:
                stdout_file.close()
            if stdout_target_file is not None:
                stdout_target_file.close()


def _attempt_output_path(path: Path, attempt: int) -> Path:
    """Return an immutable sibling path for one process attempt."""
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}.attempt-{attempt:03d}{suffix}")


def _promotion_result(
    plan: LanePlan,
) -> tuple[str, str | None, Mapping[str, Any] | None]:
    gate = plan.lane.get("promotion_gate") or {}
    negative_status = str(gate.get("negative_result_status", "completed_no_promotion"))
    for spec in plan.expected_artifacts:
        if spec.get("kind") != "json" or not str(spec.get("path", "")).endswith(
            "summary.json"
        ):
            continue
        path = Path(str(spec["path"]))
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        found_execution, execution_value = _json_lookup(payload, "execution_status")
        if not found_execution:
            raise ArtifactError(f"summary artifact lacks execution_status: {path}")
        execution_status = str(execution_value).lower()
        if execution_status not in {"succeeded", "completed_no_promotion"}:
            raise ArtifactError(
                f"summary artifact reports non-success execution_status={execution_value!r}: {path}"
            )
        if execution_status == "completed_no_promotion":
            return negative_status, "execution_status=completed_no_promotion", payload
        for key in ("promotion_eligible", "promotion_passed", "gate_passed"):
            found, value = _json_lookup(payload, key)
            if found and value is False:
                return negative_status, f"{key}=false", payload
        for key in ("promotion.eligible", "promotion.auto"):
            found, value = _json_lookup(payload, key)
            if found and value is False:
                return negative_status, f"{key}=false", payload
        found, value = _json_lookup(payload, "status")
        if found and str(value).upper() in {
            "DORMANT_NO_ELIGIBLE_SLICE",
            "NO_PROMOTION",
            "NEGATIVE",
            "COMPLETED_NO_PROMOTION",
        }:
            return negative_status, str(value), payload
        return "succeeded", None, payload
    return "succeeded", None, None


def _dependency_satisfied(condition: str, statuses: Sequence[str]) -> bool:
    if condition == "all_succeeded":
        return all(status == "succeeded" for status in statuses)
    if condition in {
        "terminal_without_technical_failure",
        "all_terminal_non_technical_failure",
    }:
        return all(
            status in {"succeeded", "completed_no_promotion"} for status in statuses
        )
    return False


def _write_campaign_summary(state: Mapping[str, Any], path: Path) -> None:
    lanes = state.get("lanes", {})
    axis_rows: dict[str, list[dict[str, Any]]] = {}
    for lane_id in state.get("lane_order", []):
        row = lanes.get(lane_id, {})
        axis_rows.setdefault(str(row.get("axis_id", "unknown")), []).append(
            {
                "lane_id": lane_id,
                "status": row.get("status"),
                "evidence_status": row.get("evidence_status"),
                "evidence_status_origin": row.get(
                    "evidence_status_origin", "registry_preexisting"
                ),
                "gate_evidence_status": row.get("gate_evidence_status", "unexecuted"),
                "execution_mode": row.get("execution_mode"),
                "claim_scope": row.get("claim_scope"),
                "outcome_detail": row.get("outcome_detail"),
                "blockers": row.get("blockers", []),
                "promotion_eligible": row.get("promotion_eligible", False),
                "promotion": row.get("promotion"),
                "reason": row.get("reason"),
                "prohibited_inferences": row.get("prohibited_inferences", []),
            }
        )
    write_json(
        path,
        {
            "format_version": 2,
            "run_id": state.get("run_id"),
            "status": state.get("status"),
            "campaign_scope": state.get("campaign_scope", "contract_preflight_only"),
            "claim_audit": {
                "campaign_substrate": "IMPLEMENTED",
                "deterministic_contract_preflight": "VALIDATED",
                "actual_trace_shadow_live_training_gates": "SPECIFIED",
                "efficacy_play_strength_production_readiness": "FORBIDDEN",
            },
            "updated_at": state.get("updated_at"),
            "selected_lane_count": len(state.get("lane_order", [])),
            "axes": axis_rows,
            "lanes": {
                lane_id: lanes.get(lane_id, {}).get("status")
                for lane_id in state.get("lane_order", [])
            },
            "registered_lanes": state.get("registered_lanes", {}),
        },
    )


def _persist_campaign(
    state: dict[str, Any], *, state_path: Path, summary_path: Path
) -> None:
    state["updated_at"] = utc_now()
    write_json(state_path, state)
    _write_campaign_summary(state, summary_path)


def _acquire_campaign_lock(run_root: Path) -> Any:
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / ".campaign.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "owner metadata unavailable"
        handle.close()
        raise ConfigError(f"campaign is locked by another controller: {owner}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps({"pid": os.getpid(), "acquired_at": utc_now()}, sort_keys=True)
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def _release_campaign_lock(handle: Any) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def run_v2_campaign(
    plans: Sequence[LanePlan],
    *,
    repo_root: Path,
    output_root: Path,
    run_id: str,
    cfg_path: Path,
    cfg: Mapping[str, Any],
    profile: str,
    python: str,
    doctor: Mapping[str, Any],
    resume: bool,
) -> int:
    validate_run_id(run_id)
    python = canonical_executable_path(python)
    run_root = confined_path(output_root / run_id, output_root, label="campaign root")
    lock_handle = _acquire_campaign_lock(run_root)
    try:
        return _run_v2_campaign_locked(
            plans,
            repo_root=repo_root,
            output_root=output_root,
            run_id=run_id,
            cfg_path=cfg_path,
            cfg=cfg,
            profile=profile,
            python=python,
            doctor=doctor,
            resume=resume,
        )
    finally:
        _release_campaign_lock(lock_handle)


def _run_v2_campaign_locked(
    plans: Sequence[LanePlan],
    *,
    repo_root: Path,
    output_root: Path,
    run_id: str,
    cfg_path: Path,
    cfg: Mapping[str, Any],
    profile: str,
    python: str,
    doctor: Mapping[str, Any],
    resume: bool,
) -> int:
    validate_run_id(run_id)
    run_root = confined_path(output_root / run_id, output_root, label="campaign root")
    state_path = run_root / "campaign_state.json"
    summary_path = run_root / "campaign_summary.json"
    commands_log = run_root / "commands.jsonl"
    fingerprint = campaign_fingerprint(plans, cfg_path=cfg_path, repo_root=repo_root)

    if resume:
        if not state_path.is_file():
            raise ConfigError(f"cannot resume; campaign state is missing: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("format_version") != 2 or state.get("run_id") != run_id:
            raise ConfigError("cannot resume incompatible campaign state")
        if state.get("profile") != profile or state.get("python") != python:
            raise ConfigError("resume refused: profile or target interpreter changed")
        if list(state.get("lane_order", [])) != [
            str(plan.lane["id"]) for plan in plans
        ]:
            raise ConfigError("resume refused: selected lane order changed")
        verify_resume_fingerprint(state.get("fingerprint", {}), fingerprint, plans)
        state["status"] = "running"
        state["resumed_at"] = utc_now()
    else:
        material_entries = (
            [entry for entry in run_root.iterdir() if entry.name != ".campaign.lock"]
            if run_root.exists()
            else []
        )
        if material_entries:
            raise ConfigError(
                f"v2 campaign directory must be new and empty: {run_root}"
            )
        run_root.mkdir(parents=True, exist_ok=True)
        selected_lane_ids = {str(plan.lane["id"]) for plan in plans}
        registered_lanes = {
            str(lane["id"]): {
                "axis_id": str(lane["axis_id"]),
                "role": str(lane["role"]),
                "execution_status": str(lane["execution_status"]),
                "evidence_status": str(lane["evidence_status"]),
                "claim_scope": str(lane["claim_scope"]),
                "blocked_by": list(lane.get("blocked_by") or []),
                "selected": str(lane["id"]) in selected_lane_ids,
            }
            for lane in cfg.get("lanes", [])
        }
        lane_state: dict[str, Any] = {}
        for plan in plans:
            lane_id = str(plan.lane["id"])
            lane_state[lane_id] = {
                "axis_id": plan.lane["axis_id"],
                "role": plan.lane["role"],
                "status": "planned" if plan.ready else "blocked",
                "evidence_status": plan.lane["evidence_status"],
                "evidence_status_origin": "registry_preexisting",
                "gate_evidence_status": "unexecuted",
                "claim_scope": plan.lane["claim_scope"],
                "depends_on": list(plan.lane.get("depends_on") or []),
                "dependency_condition": plan.lane["dependency_condition"],
                "blockers": list(plan.blockers),
                "steps": [],
                "attempts": [],
                "artifacts": [],
                "promotion_eligible": False,
                "prohibited_inferences": list(plan.lane["prohibited_inferences"]),
            }
        state = {
            "format_version": 2,
            "experiment_id": cfg.get("experiment_id"),
            "run_id": run_id,
            "status": "running",
            "campaign_scope": "contract_preflight_only",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "profile": profile,
            "python": python,
            "registry": str(cfg_path),
            "fingerprint": fingerprint,
            "git": git_info(repo_root),
            "doctor": doctor,
            "lane_order": [str(plan.lane["id"]) for plan in plans],
            "lanes": lane_state,
            "registered_lanes": registered_lanes,
        }
        write_json(
            run_root / "lab_manifest.json",
            {
                "format_version": 2,
                "experiment_id": cfg.get("experiment_id"),
                "run_id": run_id,
                "started_at": state["started_at"],
                "profile": profile,
                "python": python,
                "registry": str(cfg_path),
                "fingerprint": fingerprint,
                "git": state["git"],
                "doctor": doctor,
                "lanes": state["lane_order"],
            },
        )
    _persist_campaign(state, state_path=state_path, summary_path=summary_path)

    plan_by_id = {str(plan.lane["id"]): plan for plan in plans}
    exit_code = 0
    interrupted = False
    old_handlers: dict[int, Any] = {}

    def campaign_signal(signum: int, _frame: Any) -> None:
        raise CampaignInterrupted(signum)

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, campaign_signal)
    try:
        for lane_id in state["lane_order"]:
            plan = plan_by_id[lane_id]
            lane_row = state["lanes"][lane_id]
            if not plan.ready:
                lane_row["status"] = "blocked"
                exit_code = max(exit_code, 2)
                _persist_campaign(
                    state, state_path=state_path, summary_path=summary_path
                )
                continue

            dependency_ids = list(plan.lane.get("depends_on") or [])
            dependency_statuses = [
                state["lanes"][item]["status"] for item in dependency_ids
            ]
            condition = str(plan.lane["dependency_condition"])
            if dependency_ids and not _dependency_satisfied(
                condition, dependency_statuses
            ):
                lane_row["status"] = "skipped"
                lane_row["reason"] = (
                    f"dependency condition {condition} not met: "
                    + ", ".join(
                        f"{dep}={state['lanes'][dep]['status']}"
                        for dep in dependency_ids
                    )
                )
                _persist_campaign(
                    state, state_path=state_path, summary_path=summary_path
                )
                continue

            prior_steps: dict[str, Mapping[str, Any]] = {
                str(row.get("name")): row for row in lane_row.get("steps", [])
            }
            attempt_history = lane_row.setdefault("attempts", [])
            if not isinstance(attempt_history, list):
                raise ConfigError(
                    f"campaign state has invalid attempt history for lane {lane_id}"
                )
            # Migrate a pre-attempt-history v2 state without altering its
            # retained row or log path.  Such a state can only expose its most
            # recent attempt because older versions did not preserve more.
            if resume and not attempt_history:
                for old_row in lane_row.get("steps", []):
                    migrated = dict(old_row)
                    migrated.pop("resume_action", None)
                    migrated["attempt"] = 1
                    migrated["attempt_id"] = f"{lane_id}:{old_row.get('name')}:001"
                    attempt_history.append(migrated)
            next_steps: list[dict[str, Any]] = []
            lane_row["status"] = "running"
            lane_row.pop("reason", None)
            lane_row["artifacts"] = []
            _persist_campaign(state, state_path=state_path, summary_path=summary_path)
            lane_failed = False
            upstream_reexecuted = False
            lane_out = confined_path(
                Path(plan.output_dir), run_root, label=f"lane {lane_id} output"
            )
            lane_out.mkdir(parents=True, exist_ok=True)

            for idx, step in enumerate(plan.commands, start=1):
                name = str(step.get("name", f"step{idx}"))
                command = [str(item) for item in step.get("command", [])]
                expected = _expected_for_step(plan, step_name=name, index=idx)
                previous = prior_steps.get(name)
                if (
                    resume
                    and not upstream_reexecuted
                    and previous
                    and previous.get("status") == "succeeded"
                    and expected
                ):
                    previous_artifacts = {
                        str(item.get("path")): str(item.get("sha256"))
                        for item in previous.get("artifacts", [])
                        if item.get("sha256")
                    }
                    try:
                        verified = [
                            validate_expected_artifact(
                                spec,
                                campaign_root=run_root,
                                previous_sha256=previous_artifacts.get(
                                    str(spec["path"])
                                ),
                                source_root=repo_root,
                            )
                            for spec in expected
                        ]
                    except ArtifactError as exc:
                        previous = None
                        append_jsonl(
                            commands_log,
                            {
                                "event": "resume_reexecute",
                                "at": utc_now(),
                                "lane": lane_id,
                                "step": name,
                                "reason": str(exc),
                            },
                        )
                    else:
                        if all(
                            str(spec["path"]) in previous_artifacts for spec in expected
                        ):
                            skipped_row = dict(previous)
                            skipped_row["resume_action"] = "verified_skip"
                            skipped_row["artifacts"] = verified
                            next_steps.append(skipped_row)
                            lane_row["steps"] = next_steps
                            lane_row["artifacts"] = [
                                artifact
                                for row in next_steps
                                for artifact in row.get("artifacts", [])
                            ]
                            _persist_campaign(
                                state, state_path=state_path, summary_path=summary_path
                            )
                            continue

                if resume and upstream_reexecuted and previous:
                    append_jsonl(
                        commands_log,
                        {
                            "event": "resume_reexecute",
                            "at": utc_now(),
                            "lane": lane_id,
                            "step": name,
                            "reason": "an upstream step was reexecuted",
                        },
                    )
                upstream_reexecuted = True

                timeout = int(step.get("timeout_s", 0) or 0) or None
                prior_attempts = [
                    int(row.get("attempt", 0))
                    for row in attempt_history
                    if row.get("name") == name and isinstance(row.get("attempt"), int)
                ]
                attempt = max(prior_attempts, default=0) + 1
                attempt_id = f"{lane_id}:{name}:{attempt:03d}"
                log_path = confined_path(
                    lane_out
                    / (f"{idx:02d}_{sanitize_id(name)}.attempt-{attempt:03d}.log"),
                    run_root,
                    label=f"lane {lane_id} log",
                )
                stdout_target_path = (
                    Path(str(step["stdout"])) if step.get("stdout") else None
                )
                stdout_path = None
                if stdout_target_path is not None:
                    stdout_target_path = confined_path(
                        stdout_target_path,
                        run_root,
                        label=f"lane {lane_id} stdout target",
                    )
                    stdout_path = confined_path(
                        _attempt_output_path(stdout_target_path, attempt),
                        run_root,
                        label=f"lane {lane_id} attempt stdout",
                    )
                step_row: dict[str, Any] = {
                    "name": name,
                    "status": "running",
                    "command": command,
                    "attempt": attempt,
                    "attempt_id": attempt_id,
                    "started_at": utc_now(),
                    "log": str(log_path),
                    "stdout": str(stdout_path) if stdout_path else None,
                    "stdout_target": str(stdout_target_path)
                    if stdout_target_path
                    else None,
                    "artifacts": [],
                }
                next_steps.append(step_row)
                attempt_history.append(step_row)
                lane_row["steps"] = next_steps
                _persist_campaign(
                    state, state_path=state_path, summary_path=summary_path
                )
                append_jsonl(
                    commands_log,
                    {
                        "event": "start",
                        "at": step_row["started_at"],
                        "lane": lane_id,
                        "step": name,
                        "attempt": attempt,
                        "attempt_id": attempt_id,
                        "command": command,
                        "log": str(log_path),
                    },
                )
                try:
                    rc, elapsed, process_status = _run_streaming_process(
                        command,
                        cwd=repo_root,
                        log_path=log_path,
                        stdout_path=stdout_path,
                        stdout_target_path=stdout_target_path,
                        timeout_s=timeout,
                    )
                except CampaignInterrupted:
                    rc, elapsed, process_status = 130, 0.0, "interrupted"
                except OSError as exc:
                    rc, elapsed, process_status = 126, 0.0, "failed"
                    step_row["runner_error"] = repr(exc)
                step_row.update(
                    {
                        "returncode": rc,
                        "elapsed_s": elapsed,
                        "finished_at": utc_now(),
                        "status": process_status,
                    }
                )
                append_jsonl(
                    commands_log,
                    {
                        "event": "finish",
                        "at": step_row["finished_at"],
                        "lane": lane_id,
                        "step": name,
                        "attempt": attempt,
                        "attempt_id": attempt_id,
                        "returncode": rc,
                        "elapsed_s": elapsed,
                        "status": process_status,
                        "log": str(log_path),
                    },
                )
                if process_status != "succeeded":
                    lane_row["status"] = process_status
                    lane_row["reason"] = (
                        f"step {name} {process_status} with returncode {rc}"
                    )
                    exit_code = rc or 1
                    lane_failed = True
                    interrupted = process_status == "interrupted"
                    _persist_campaign(
                        state, state_path=state_path, summary_path=summary_path
                    )
                    break
                try:
                    artifacts = [
                        validate_expected_artifact(
                            spec,
                            campaign_root=run_root,
                            source_root=repo_root,
                        )
                        for spec in expected
                    ]
                except ArtifactError as exc:
                    step_row["status"] = "failed"
                    step_row["artifact_error"] = str(exc)
                    lane_row["status"] = "failed"
                    lane_row["reason"] = (
                        f"artifact contract failed after step {name}: {exc}"
                    )
                    exit_code = 3
                    lane_failed = True
                    _persist_campaign(
                        state, state_path=state_path, summary_path=summary_path
                    )
                    break
                step_row["artifacts"] = artifacts
                lane_row["artifacts"] = [
                    artifact
                    for row in next_steps
                    for artifact in row.get("artifacts", [])
                ]
                _persist_campaign(
                    state, state_path=state_path, summary_path=summary_path
                )

            if lane_failed:
                # Technical execution and artifact-contract failures are fail-closed.
                break
            try:
                lane_status, reason, summary_payload = _promotion_result(plan)
            except ArtifactError as exc:
                lane_row["status"] = "failed"
                lane_row["reason"] = f"summary contract failed: {exc}"
                exit_code = 3
                _persist_campaign(
                    state, state_path=state_path, summary_path=summary_path
                )
                break
            if lane_status not in {"succeeded", "completed_no_promotion"}:
                lane_status = "completed_no_promotion"
            lane_row["status"] = lane_status
            lane_row["promotion_eligible"] = False
            if summary_payload is not None:
                lane_row["execution_mode"] = summary_payload.get("execution_mode")
                lane_row["gate_evidence_status"] = summary_payload.get(
                    "gate_evidence_status",
                    "contract_only",
                )
                lane_row["evidence_status_origin"] = summary_payload.get(
                    "evidence_status_origin",
                    "registry_preexisting",
                )
                lane_row["outcome_detail"] = summary_payload.get("outcome_detail")
                lane_row["promotion"] = summary_payload.get("promotion")
            if reason:
                lane_row["reason"] = reason
            _persist_campaign(state, state_path=state_path, summary_path=summary_path)
    except (KeyboardInterrupt, CampaignInterrupted) as exc:
        interrupted = True
        exit_code = 130
        state["status"] = "interrupted"
        state["reason"] = str(exc)
        for lane_id in state["lane_order"]:
            if state["lanes"][lane_id]["status"] == "running":
                state["lanes"][lane_id]["status"] = "interrupted"
        _persist_campaign(state, state_path=state_path, summary_path=summary_path)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    statuses = [state["lanes"][lane_id]["status"] for lane_id in state["lane_order"]]
    if interrupted:
        state["status"] = "interrupted"
    elif any(status in {"failed", "timeout"} for status in statuses):
        state["status"] = "failed"
    elif any(status == "blocked" for status in statuses):
        state["status"] = "blocked"
        exit_code = max(exit_code, 2)
    elif any(status in {"planned", "running"} for status in statuses):
        state["status"] = "interrupted"
        exit_code = exit_code or 1
    else:
        state["status"] = "succeeded"
    state["finished_at"] = utc_now()
    state["exit_code"] = exit_code
    _persist_campaign(state, state_path=state_path, summary_path=summary_path)
    manifest_path = run_root / "lab_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "finished_at": state["finished_at"],
            "status": state["status"],
            "exit_code": exit_code,
        }
    )
    write_json(manifest_path, manifest)
    return exit_code


def run_plans(
    plans: Sequence[LanePlan],
    *,
    repo_root: Path,
    output_root: Path,
    run_id: str,
    cfg_path: Path,
    cfg: Mapping[str, Any],
    profile: str,
    python: str,
    doctor: Mapping[str, Any],
    keep_going: bool,
    overwrite: bool,
    resume: bool = False,
) -> int:
    if cfg.get("format_version") == 2:
        if overwrite:
            raise ConfigError(
                "--overwrite is forbidden for v2 campaigns; use resume --run-id"
            )
        return run_v2_campaign(
            plans,
            repo_root=repo_root,
            output_root=output_root,
            run_id=run_id,
            cfg_path=cfg_path,
            cfg=cfg,
            profile=profile,
            python=python,
            doctor=doctor,
            resume=resume,
        )
    run_root = output_root / run_id
    if run_root.exists() and any(run_root.iterdir()) and not overwrite:
        raise SystemExit(f"output directory is not empty; pass --overwrite: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    commands_log = run_root / "commands.jsonl"
    summary_path = run_root / "summary.json"
    manifest = {
        "format_version": 1,
        "experiment_id": cfg.get("experiment_id"),
        "started_at": utc_now(),
        "profile": profile,
        "python": python,
        "registry": str(cfg_path),
        "registry_sha256": sha256_file(cfg_path) if cfg_path.exists() else None,
        "git": git_info(repo_root),
        "doctor": doctor,
        "lanes": [plan.lane["id"] for plan in plans],
    }
    write_json(run_root / "lab_manifest.json", manifest)
    results: list[dict[str, Any]] = []
    exit_code = 0
    for plan in plans:
        lane_id = str(plan.lane["id"])
        lane_out = Path(plan.output_dir)
        lane_out.mkdir(parents=True, exist_ok=True)
        if not plan.ready:
            row = {"lane": lane_id, "status": "blocked", "blockers": plan.blockers}
            results.append(row)
            write_json(summary_path, {"results": results})
            exit_code = 2
            if not keep_going:
                break
            continue
        lane_status = "ok"
        for idx, step in enumerate(plan.commands, start=1):
            name = str(step.get("name", f"step{idx}"))
            command = [str(x) for x in step.get("command", [])]
            timeout = int(step.get("timeout_s", 0) or 0) or None
            log_path = lane_out / f"{idx:02d}_{sanitize_id(name)}.log"
            event = {
                "event": "start",
                "at": utc_now(),
                "lane": lane_id,
                "step": name,
                "command": command,
                "log": str(log_path),
            }
            append_jsonl(commands_log, event)
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    command,
                    cwd=repo_root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
                rc = int(proc.returncode)
                stdout = proc.stdout
                stderr = proc.stderr
            except subprocess.TimeoutExpired as exc:
                rc = 124
                stdout = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr = exc.stderr if isinstance(exc.stderr, str) else ""
                stderr += f"\nTIMEOUT after {timeout}s\n"
            elapsed = time.perf_counter() - t0
            stdout_target = step.get("stdout")
            if stdout_target:
                stdout_path = Path(str(stdout_target))
                if not stdout_path.is_absolute():
                    stdout_path = repo_root / stdout_path
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_path.write_text(stdout, encoding="utf-8")
            with log_path.open("w", encoding="utf-8") as f:
                f.write("$ " + " ".join(command) + "\n\n")
                f.write("[stdout]\n")
                f.write(stdout)
                if stdout and not stdout.endswith("\n"):
                    f.write("\n")
                f.write("\n[stderr]\n")
                f.write(stderr)
                if stderr and not stderr.endswith("\n"):
                    f.write("\n")
                if stdout_target:
                    f.write(f"\n[stdout_saved_to]\n{stdout_target}\n")
            append_jsonl(
                commands_log,
                {
                    "event": "finish",
                    "at": utc_now(),
                    "lane": lane_id,
                    "step": name,
                    "returncode": rc,
                    "elapsed_s": elapsed,
                    "log": str(log_path),
                    "stdout": stdout_target,
                },
            )
            if rc != 0:
                lane_status = "failed"
                exit_code = rc or 1
                if not keep_going:
                    break
        results.append(
            {"lane": lane_id, "status": lane_status, "output_dir": str(lane_out)}
        )
        write_json(
            summary_path,
            {
                "format_version": 1,
                "run_id": run_id,
                "updated_at": utc_now(),
                "results": results,
            },
        )
        if lane_status != "ok" and not keep_going:
            break
    manifest["finished_at"] = utc_now()
    manifest["exit_code"] = exit_code
    write_json(run_root / "lab_manifest.json", manifest)
    return exit_code


def build_common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--profile", choices=sorted(PROFILE_DEVICE), default="cpu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--device",
        default=None,
        help="Override device template; default derives from profile",
    )
    parser.add_argument(
        "--set", dest="sets", action="append", default=[], metavar="KEY=VALUE"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List registered lanes")
    build_common_parser(p_list)
    p_list.add_argument("--status", default=None)
    p_list.add_argument("--group", action="append", default=[])
    p_list.add_argument("--json", action="store_true")

    p_doctor = sub.add_parser("doctor", help="Check local environment")
    build_common_parser(p_doctor)
    p_doctor.add_argument("--strict", action="store_true")
    p_doctor.add_argument("--json", action="store_true")

    for name in ("plan", "run", "resume"):
        p = sub.add_parser(name, help=f"{name.capitalize()} experiment lanes")
        build_common_parser(p)
        if name != "resume":
            p.add_argument("--suite", action="append", default=[])
            p.add_argument("--lane", action="append", default=[])
        p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        p.add_argument("--run-id", required=name == "resume", default=None)
        p.add_argument("--json", action="store_true")
        if name == "run":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--keep-going", action="store_true")
            p.add_argument("--overwrite", action="store_true")

    p_status = sub.add_parser("status", help="Show persisted v2 campaign status")
    p_status.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p_status.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_status.add_argument("--run-id", required=True)
    p_status.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.cmd == "status":
        run_id = validate_run_id(args.run_id)
        output_root = (
            args.output_root
            if args.output_root.is_absolute()
            else repo_root / args.output_root
        )
        run_root = confined_path(
            output_root / run_id, output_root, label="campaign root"
        )
        state_path = run_root / "campaign_state.json"
        if not state_path.is_file():
            raise ConfigError(f"campaign state is missing: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if args.json:
            print(json.dumps(state, indent=2, sort_keys=True))
        else:
            print(
                f"run_id={state.get('run_id')} status={state.get('status')} updated_at={state.get('updated_at')}"
            )
            for lane_id in state.get("lane_order", []):
                row = state.get("lanes", {}).get(lane_id, {})
                reason = f" ({row['reason']})" if row.get("reason") else ""
                print(f"{lane_id:32s} {str(row.get('status', 'unknown')):24s}{reason}")
        return 0

    cfg_path = args.config if args.config.is_absolute() else repo_root / args.config
    cfg = load_config(cfg_path)
    variables = parse_sets(args.sets)

    if args.cmd == "list":
        lanes = [dict(lane) for lane in cfg["lanes"]]
        if args.status:
            lanes = [
                lane
                for lane in lanes
                if lane.get("execution_status", lane.get("status")) == args.status
            ]
        for group in args.group:
            lanes = [lane for lane in lanes if group in (lane.get("groups") or [])]
        if args.json:
            print(json.dumps(lanes, indent=2, sort_keys=True))
        else:
            for lane in lanes:
                groups = ",".join(lane.get("groups") or [])
                execution_status = lane.get("execution_status", lane.get("status", ""))
                print(f"{lane['id']:42s} {execution_status:10s} {groups}")
        return 0

    if args.cmd == "doctor":
        report = collect_doctor(
            repo_root, profile=args.profile, python=args.python, strict=args.strict
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"profile={report['profile']} ok={report['ok']}")
            for item in report["checks"]:
                mark = "OK" if item["ok"] else "FAIL"
                detail = f" ({item['detail']})" if item.get("detail") else ""
                print(f"[{mark}] {item['kind']}: {item['name']}{detail}")
        return 0 if report["ok"] else 2

    run_id = make_run_id(args.run_id)
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else repo_root / args.output_root
    )
    if args.cmd == "resume":
        if cfg.get("format_version") != 2:
            raise ConfigError("resume is supported only for v2 campaigns")
        state_path = (
            confined_path(output_root / run_id, output_root, label="campaign root")
            / "campaign_state.json"
        )
        if not state_path.is_file():
            raise ConfigError(f"cannot resume; campaign state is missing: {state_path}")
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        lane_ids = [str(item) for item in previous_state.get("lane_order", [])]
        lanes = expand_lanes(cfg, [], lane_ids)
    else:
        lanes = expand_lanes(cfg, args.suite, args.lane)
    plans = [
        plan_lane(
            lane,
            repo_root=repo_root,
            profile=args.profile,
            output_root=output_root,
            run_id=run_id,
            python=args.python,
            device=args.device,
            variables=variables,
            campaign_source_paths=cfg.get("_source_dependencies", []),
        )
        for lane in lanes
    ]
    if args.json:
        payload = [
            {
                "id": plan.lane["id"],
                "status": plan.status,
                "blockers": plan.blockers,
                "commands": plan.commands,
                "output_dir": plan.output_dir,
                "artifact_globs": plan.artifact_globs,
            }
            for plan in plans
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_text_plan(plans)
    blocked = [plan for plan in plans if not plan.ready]
    if args.cmd == "plan" or getattr(args, "dry_run", False):
        return 2 if blocked else 0
    if cfg.get("format_version") == 2 and getattr(args, "overwrite", False):
        raise ConfigError(
            "--overwrite is forbidden for v2 campaigns; use resume --run-id"
        )
    doctor = collect_doctor(
        repo_root, profile=args.profile, python=args.python, strict=False
    )
    if cfg.get("format_version") == 2:
        for plan in plans:
            gate = (plan.lane.get("resource_profile") or {}).get("hardware_gate")
            if gate:
                accelerator = next(
                    (
                        item
                        for item in doctor["checks"]
                        if item["kind"] == "accelerator"
                        and item["name"] == f"torch[{gate}]"
                    ),
                    None,
                )
                if accelerator is None or not accelerator["ok"]:
                    raise ConfigError(
                        f"hardware doctor refused lane {plan.lane['id']}: no consistent {gate} device proof"
                    )
    return run_plans(
        plans,
        repo_root=repo_root,
        output_root=output_root,
        run_id=run_id,
        cfg_path=cfg_path,
        cfg=cfg,
        profile=args.profile,
        python=args.python,
        doctor=doctor,
        keep_going=getattr(args, "keep_going", False),
        overwrite=getattr(args, "overwrite", False),
        resume=args.cmd == "resume",
    )


if __name__ == "__main__":
    raise SystemExit(main())
