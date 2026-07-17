#!/usr/bin/env python3
"""Local experiment orchestrator for the QUARTZ idea lab.

The script is intentionally small: it discovers registered experiment lanes,
checks local requirements, prints executable plans, and optionally runs the
available commands into isolated output directories.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "idea_lab.local.v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "idea_lab_local"
PROFILE_DEVICE = {"cpu": "cpu", "cuda": "cuda", "rocm": "cuda"}


class ConfigError(ValueError):
    pass


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
    safe = SafeDict({k: str(v) for k, v in ctx.items()})
    if isinstance(value, str):
        return value.format_map(safe)
    if isinstance(value, list):
        return [format_obj(item, ctx) for item in value]
    if isinstance(value, dict):
        return {key: format_obj(val, ctx) for key, val in value.items()}
    return value


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"missing registry: {path}") from exc
    if cfg.get("format_version") != 1:
        raise ConfigError("unsupported registry format_version")
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
            raise ConfigError(f"lane {lane_id} has invalid status {lane.get('status')!r}")
    suites = cfg.get("suites") or {}
    if not isinstance(suites, dict):
        raise ConfigError("suites must be an object")
    for suite, ids in suites.items():
        if not isinstance(ids, list):
            raise ConfigError(f"suite {suite!r} must be a list")
        unknown = sorted(set(map(str, ids)) - seen)
        if unknown:
            raise ConfigError(f"suite {suite!r} references unknown lanes: {unknown}")
    return cfg


def lane_index(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(lane["id"]): dict(lane) for lane in cfg["lanes"]}


def expand_lanes(cfg: Mapping[str, Any], suites: Sequence[str] | None, lane_ids: Sequence[str] | None) -> list[dict[str, Any]]:
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
    for lane_id in selected:
        if lane_id not in idx:
            raise ConfigError(f"unknown lane: {lane_id}")
        if lane_id not in seen:
            out.append(idx[lane_id])
            seen.add(lane_id)
    return out


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def profile_supported(lane: Mapping[str, Any], profile: str) -> bool:
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

    @property
    def ready(self) -> bool:
        return self.status == "READY"


def build_context(*, repo_root: Path, profile: str, output_root: Path, run_id: str, lane_id: str,
                  python: str, device: str | None, variables: Mapping[str, str]) -> dict[str, Any]:
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


def plan_lane(lane: Mapping[str, Any], *, repo_root: Path, profile: str, output_root: Path,
              run_id: str, python: str, device: str | None, variables: Mapping[str, str]) -> LanePlan:
    lane_id = str(lane["id"])
    ctx = build_context(
        repo_root=repo_root, profile=profile, output_root=output_root, run_id=run_id,
        lane_id=lane_id, python=python, device=device, variables=variables,
    )
    blockers: list[str] = []
    if not profile_supported(lane, profile):
        blockers.append(f"profile {profile!r} not supported by lane")

    status = str(lane.get("status", "available"))
    if status != "available":
        for item in lane.get("blocked_by", []) or []:
            blockers.append(f"planned: {item}")

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
        if not module_exists(module_name):
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
        command = format_obj(step.get("command", []), ctx)
        step_payload = dict(step)
        step_payload["command"] = command
        if "stdout" in step_payload:
            step_payload["stdout"] = format_obj(step_payload["stdout"], ctx)
        commands.append(step_payload)
    artifact_globs = [str(format_obj(item, ctx)) for item in lane.get("artifact_globs", []) or []]
    plan_status = "READY" if not blockers else "BLOCKED"
    return LanePlan(dict(lane), plan_status, blockers, commands, str(ctx["output"]), artifact_globs)


def make_run_id(prefix: str | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return sanitize_id(prefix or f"run-{stamp}")


def collect_doctor(repo_root: Path, *, profile: str, python: str, strict: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(kind: str, name: str, ok: bool, detail: str | None = None) -> None:
        checks.append({"kind": kind, "name": name, "ok": bool(ok), "detail": detail})

    add("python", python, Path(python).exists() or command_exists(python), sys.version.split()[0])
    for cmd in ["git", "cargo", "rustc"]:
        add("command", cmd, command_exists(cmd), shutil.which(cmd))
    for module in (["numpy", "pytest"] if strict else ["numpy"]):
        add("python_module", module, module_exists(module))
    if strict:
        rust_bin = repo_root / "target" / "release" / "mcts_demo"
        add("path", str(rust_bin.relative_to(repo_root)), rust_bin.exists())
    try:
        import torch  # type: ignore
        torch_detail = {"version": getattr(torch, "__version__", None)}
        cuda_ok = bool(torch.cuda.is_available())
        torch_detail["cuda_available"] = cuda_ok
        torch_detail["cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
        torch_detail["hip_build"] = getattr(getattr(torch, "version", None), "hip", None)
        if profile == "cpu":
            add("python_module", "torch", True, json.dumps(torch_detail, sort_keys=True))
        else:
            add("python_module", f"torch[{profile}]", cuda_ok, json.dumps(torch_detail, sort_keys=True))
    except Exception as exc:  # pragma: no cover - depends on local environment
        add("python_module", "torch", False, repr(exc))
    return {
        "profile": profile,
        "platform": platform.platform(),
        "python": python,
        "strict": strict,
        "checks": checks,
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
            return subprocess.check_output(args, cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
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
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def run_plans(plans: Sequence[LanePlan], *, repo_root: Path, output_root: Path, run_id: str,
              cfg_path: Path, cfg: Mapping[str, Any], profile: str, python: str,
              doctor: Mapping[str, Any], keep_going: bool, overwrite: bool) -> int:
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
            event = {"event": "start", "at": utc_now(), "lane": lane_id, "step": name, "command": command, "log": str(log_path)}
            append_jsonl(commands_log, event)
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    command, cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
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
            append_jsonl(commands_log, {
                "event": "finish", "at": utc_now(), "lane": lane_id, "step": name,
                "returncode": rc, "elapsed_s": elapsed, "log": str(log_path), "stdout": stdout_target,
            })
            if rc != 0:
                lane_status = "failed"
                exit_code = rc or 1
                if not keep_going:
                    break
        results.append({"lane": lane_id, "status": lane_status, "output_dir": str(lane_out)})
        write_json(summary_path, {"format_version": 1, "run_id": run_id, "updated_at": utc_now(), "results": results})
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
    parser.add_argument("--device", default=None, help="Override device template; default derives from profile")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")


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

    for name in ("plan", "run"):
        p = sub.add_parser(name, help=f"{name.capitalize()} experiment lanes")
        build_common_parser(p)
        p.add_argument("--suite", action="append", default=[])
        p.add_argument("--lane", action="append", default=[])
        p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        p.add_argument("--run-id", default=None)
        p.add_argument("--json", action="store_true")
        if name == "run":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--keep-going", action="store_true")
            p.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    cfg_path = args.config if args.config.is_absolute() else repo_root / args.config
    cfg = load_config(cfg_path)
    variables = parse_sets(args.sets)

    if args.cmd == "list":
        lanes = [dict(lane) for lane in cfg["lanes"]]
        if args.status:
            lanes = [lane for lane in lanes if lane.get("status") == args.status]
        for group in args.group:
            lanes = [lane for lane in lanes if group in (lane.get("groups") or [])]
        if args.json:
            print(json.dumps(lanes, indent=2, sort_keys=True))
        else:
            for lane in lanes:
                groups = ",".join(lane.get("groups") or [])
                print(f"{lane['id']:42s} {lane.get('status',''):10s} {groups}")
        return 0

    if args.cmd == "doctor":
        report = collect_doctor(repo_root, profile=args.profile, python=args.python, strict=args.strict)
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
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    lanes = expand_lanes(cfg, args.suite, args.lane)
    plans = [
        plan_lane(
            lane, repo_root=repo_root, profile=args.profile, output_root=output_root,
            run_id=run_id, python=args.python, device=args.device, variables=variables,
        )
        for lane in lanes
    ]
    if args.json:
        payload = [
            {
                "id": plan.lane["id"], "status": plan.status, "blockers": plan.blockers,
                "commands": plan.commands, "output_dir": plan.output_dir, "artifact_globs": plan.artifact_globs,
            }
            for plan in plans
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_text_plan(plans)
    blocked = [plan for plan in plans if not plan.ready]
    if args.cmd == "plan" or getattr(args, "dry_run", False):
        return 2 if blocked else 0
    doctor = collect_doctor(repo_root, profile=args.profile, python=args.python, strict=False)
    return run_plans(
        plans, repo_root=repo_root, output_root=output_root, run_id=run_id,
        cfg_path=cfg_path, cfg=cfg, profile=args.profile, python=args.python,
        doctor=doctor, keep_going=args.keep_going, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
