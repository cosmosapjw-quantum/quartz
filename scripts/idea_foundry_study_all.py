#!/usr/bin/env python3
"""Run, resume, or inspect all 26 first scientific gates sequentially."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quartz.experiment_manifest import atomic_json_dump, file_sha256, utc_now  # noqa: E402
from quartz.idea_foundry.studies import (  # noqa: E402
    StudyError,
    fingerprint,
    load_study_specs,
    study_plan,
)


CAMPAIGN_ROOT = REPO_ROOT / "results" / "idea_foundry_studies"
TERMINAL = {"completed_no_promotion", "skipped"}


def _safe_campaign_root(raw_root: Path) -> Path:
    root = raw_root.resolve()
    allowed = (REPO_ROOT / "results").resolve()
    try:
        root.relative_to(allowed)
    except ValueError as exc:
        raise StudyError(f"campaign root must remain under {allowed}: {root}") from exc
    return root


def _safe_run_root(run_id: str, campaign_root: Path = CAMPAIGN_ROOT) -> Path:
    if (
        not run_id
        or len(run_id) > 96
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in run_id
        )
    ):
        raise StudyError(f"unsafe run id: {run_id!r}")
    base = _safe_campaign_root(campaign_root)
    root = (base / run_id).resolve()
    root.relative_to(base)
    return root


def _campaign_sources() -> tuple[Path, ...]:
    return (
        REPO_ROOT / "configs" / "idea_foundry.studies.v1.json",
        REPO_ROOT / "quartz" / "idea_foundry" / "studies.py",
        REPO_ROOT / "quartz" / "idea_foundry" / "a19_proxy.py",
        REPO_ROOT / "scripts" / "idea_foundry_study.py",
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "a15_matched_service_curve.py",
        REPO_ROOT / "configs" / "a15_matched_service_curve.v1.json",
        REPO_ROOT / "quartz" / "experiments" / "a15_matched_service_curve.py",
        REPO_ROOT / "quartz" / "host_resources.py",
        REPO_ROOT / "scripts" / "a18_evaluator_ablation.py",
        REPO_ROOT / "scripts" / "a19_proxy_screen.py",
    )


def _git_head() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise StudyError("cannot resolve the campaign Git HEAD")
    return process.stdout.strip()


def _identity(profile: str, seed: int) -> dict[str, Any]:
    return {
        "profile": profile,
        "seed": seed,
        "git_head": _git_head(),
        "python_executable": str(Path(sys.executable).resolve()),
        "source_fingerprint": fingerprint(_campaign_sources()),
    }


def _new_state(run_id: str, profile: str, seed: int) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "suite": "first-scientific-gate-all",
        "profile": profile,
        "seed": seed,
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "identity": _identity(profile, seed),
        "axes": [
            {
                "order_index": index,
                "axis_id": spec.axis_id,
                "gate_kind": spec.gate_kind,
                "runner": spec.runner,
                "status": "planned",
                "attempts": [],
            }
            for index, spec in enumerate(load_study_specs())
        ],
        "claim_scope": "first_scientific_gate_diagnostic_only",
        "promotion": {"auto": False, "eligible": False},
    }


def _save(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_json_dump(path, state)


def _load_state(path: Path, profile: str, seed: int) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"invalid campaign state: {exc}") from exc
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise StudyError("campaign state schema mismatch")
    if state.get("identity") != _identity(profile, seed):
        raise StudyError(
            "resume refused: profile, seed, Git, source, or interpreter drift"
        )
    expected = [spec.axis_id for spec in load_study_specs()]
    observed = [row.get("axis_id") for row in state.get("axes", [])]
    if observed != expected:
        raise StudyError("resume refused: axis order drift")
    return state


def _validate_artifact_set(axis_dir: Path) -> str | None:
    def records_with_hash(value: Any):
        if isinstance(value, Mapping):
            if isinstance(value.get("path"), str) and isinstance(
                value.get("sha256"), str
            ):
                yield value
            for child in value.values():
                yield from records_with_hash(child)
        elif isinstance(value, list):
            for child in value:
                yield from records_with_hash(child)

    def validate_a18_inputs(
        manifest: Mapping[str, Any], inventory: Mapping[str, Any]
    ) -> bool:
        if manifest.get("axis_id") != "A18":
            return False
        rows = manifest.get("input_inventory")
        if not isinstance(rows, list):
            return False
        expected_spec = inventory.get("experiment_spec")
        if not isinstance(expected_spec, str):
            return False
        matching_specs = [
            path
            for path in (REPO_ROOT / "configs").glob("a18_evaluator_ablation.*.v1.json")
            if path.is_file() and file_sha256(path) == expected_spec
        ]
        if len(matching_specs) != 1:
            return False
        by_seed = {
            int(row["seed"]): row
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("seed"), int)
        }
        for key, expected_hash in inventory.items():
            if key == "experiment_spec":
                continue
            if not isinstance(key, str) or not isinstance(expected_hash, str):
                return False
            prefix, separator, field = key.partition(".")
            if not separator or not prefix.startswith("seed_"):
                return False
            try:
                seed = int(prefix.removeprefix("seed_"))
            except ValueError:
                return False
            row = by_seed.get(seed)
            if row is None or row.get(f"{field}_sha256") != expected_hash:
                return False
            raw_path = row.get(field)
            if not isinstance(raw_path, str):
                return False
            path = Path(raw_path).resolve()
            try:
                path.relative_to(REPO_ROOT.resolve())
            except ValueError:
                return False
            if not path.is_file() or file_sha256(path) != expected_hash:
                return False
        return True

    def validate_manifest(manifest_path: Path, seen: set[Path]) -> bool:
        resolved_manifest = manifest_path.resolve()
        if resolved_manifest in seen:
            return True
        seen.add(resolved_manifest)
        try:
            manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(manifest, dict):
            return False
        for inventory_name in ("source_hashes", "input_hashes"):
            inventory = manifest.get(inventory_name, [])
            if inventory_name == "input_hashes" and isinstance(inventory, Mapping):
                if not validate_a18_inputs(manifest, inventory):
                    return False
                continue
            if not isinstance(inventory, list):
                return False
            for record in inventory:
                if not isinstance(record, Mapping):
                    return False
                raw_path = record.get("path")
                expected_hash = record.get("sha256")
                if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
                    return False
                candidate = Path(raw_path)
                path = (
                    candidate.resolve()
                    if candidate.is_absolute()
                    else (REPO_ROOT / candidate).resolve()
                )
                try:
                    path.relative_to(REPO_ROOT.resolve())
                except ValueError:
                    return False
                if not path.is_file() or file_sha256(path) != expected_hash:
                    return False
                if (
                    inventory_name == "input_hashes"
                    and path.name == "run_manifest.json"
                ):
                    if not validate_manifest(path, seen):
                        return False
        artifacts = manifest.get("artifacts", [])
        if not isinstance(artifacts, (list, Mapping)):
            return False
        for record in records_with_hash(artifacts):
            raw_path = record["path"]
            expected_hash = record["sha256"]
            candidate = Path(raw_path)
            path = (
                candidate.resolve()
                if candidate.is_absolute()
                else (resolved_manifest.parent / candidate).resolve()
            )
            try:
                path.relative_to(resolved_manifest.parent)
            except ValueError:
                return False
            if not path.is_file() or file_sha256(path) != expected_hash:
                return False
        return True

    manifest_path = axis_dir / "run_manifest.json"
    summary_path = axis_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict) or not validate_manifest(manifest_path, set()):
        return None
    status = summary.get("execution_status")
    return str(status) if status in TERMINAL else None


def _recover_axis(axis: dict[str, Any], validated_status: str) -> bool:
    changed = axis.get("status") != validated_status or "blocker" in axis
    axis["status"] = validated_status
    axis.pop("blocker", None)
    if changed:
        axis["recovered_published_artifact_at"] = utc_now()
    return changed


def _terminate(proc: subprocess.Popen[Any]) -> None:
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


def _run_axis(
    *,
    axis_id: str,
    profile: str,
    seed: int,
    output_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float | None,
) -> tuple[int, str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "idea_foundry_study.py"),
        "run",
        "--axis",
        axis_id,
        "--profile",
        profile,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("w", encoding="utf-8", buffering=1) as stdout,
        stderr_path.open("w", encoding="utf-8", buffering=1) as stderr,
    ):
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout_seconds), "completed"
        except subprocess.TimeoutExpired:
            _terminate(proc)
            return 124, "timeout"
        except KeyboardInterrupt:
            _terminate(proc)
            return 130, "interrupted"


def _summary(state: Mapping[str, Any]) -> dict[str, Any]:
    axes = state["axes"]
    return {
        "schema_version": 1,
        "run_id": state["run_id"],
        "suite": state["suite"],
        "profile": state["profile"],
        "status": state["status"],
        "axis_count": len(axes),
        "status_counts": dict(sorted(Counter(row["status"] for row in axes).items())),
        "axes": [
            {
                "axis_id": row["axis_id"],
                "gate_kind": row["gate_kind"],
                "status": row["status"],
                "attempt_count": len(row["attempts"]),
                "output_dir": row.get("output_dir"),
                "blocker": row.get("blocker"),
                "promotion_eligible": False,
            }
            for row in axes
        ],
        "claim_scope": "first_scientific_gate_diagnostic_only",
        "promotion": {"auto": False, "eligible": False},
    }


def run_campaign(
    *,
    run_id: str,
    profile: str,
    seed: int,
    resume: bool,
    timeout_multiplier: float,
    campaign_root: Path = CAMPAIGN_ROOT,
) -> dict[str, Any]:
    run_root = _safe_run_root(run_id, campaign_root)
    state_path = run_root / "campaign_state.json"
    if resume:
        state = _load_state(state_path, profile, seed)
        state["status"] = "running"
        state["resumed_at"] = utc_now()
    else:
        if run_root.exists():
            raise StudyError(f"new campaign root already exists: {run_root}")
        run_root.mkdir(parents=True)
        state = _new_state(run_id, profile, seed)
    _save(state_path, state)
    spec_by_axis = {spec.axis_id: spec for spec in load_study_specs()}
    for axis in state["axes"]:
        axis_id = axis["axis_id"]
        output_dir = run_root / "axes" / axis_id
        validated_status = _validate_artifact_set(output_dir) if resume else None
        if validated_status in TERMINAL:
            if _recover_axis(axis, validated_status):
                _save(state_path, state)
            continue
        if output_dir.exists():
            raise StudyError(
                f"resume found an unverified existing axis directory: {output_dir}"
            )
        attempt = len(axis["attempts"]) + 1
        axis["status"] = "running"
        axis["started_at"] = utc_now()
        axis["output_dir"] = str(output_dir.relative_to(run_root))
        _save(state_path, state)
        spec = spec_by_axis[axis_id]
        estimate = spec.pilot_seconds if profile == "pilot" else spec.full_seconds
        timeout = max(60.0, estimate * timeout_multiplier)
        monotonic_start = time.monotonic()
        returncode, process_status = _run_axis(
            axis_id=axis_id,
            profile=profile,
            seed=seed,
            output_dir=output_dir,
            stdout_path=run_root / "logs" / f"{axis_id}.attempt-{attempt}.stdout.log",
            stderr_path=run_root / "logs" / f"{axis_id}.attempt-{attempt}.stderr.log",
            timeout_seconds=timeout,
        )
        validated_status = _validate_artifact_set(output_dir)
        axis["attempts"].append(
            {
                "attempt": attempt,
                "completed_at": utc_now(),
                "returncode": returncode,
                "process_status": process_status,
                "artifact_status": validated_status,
                "elapsed_seconds": time.monotonic() - monotonic_start,
            }
        )
        if returncode == 0 and validated_status in TERMINAL:
            axis["status"] = validated_status
            axis.pop("blocker", None)
        else:
            axis["status"] = (
                process_status if process_status != "completed" else "failed"
            )
            axis["blocker"] = (
                f"returncode={returncode}; artifact_status={validated_status!r}"
            )
            state["status"] = axis["status"]
            _save(state_path, state)
            atomic_json_dump(run_root / "campaign_summary.json", _summary(state))
            raise StudyError(f"campaign stopped at {axis_id}: {axis['blocker']}")
        _save(state_path, state)
    state["status"] = "completed_no_promotion"
    _save(state_path, state)
    summary = _summary(state)
    atomic_json_dump(run_root / "campaign_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--json", action="store_true")
    for command in ("run", "resume"):
        run_parser = subparsers.add_parser(command)
        run_parser.add_argument("--run-id", required=True)
        run_parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
        run_parser.add_argument("--seed", type=int, default=20260719)
        run_parser.add_argument("--timeout-multiplier", type=float, default=3.0)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            print(json.dumps(study_plan(), indent=2, sort_keys=True))
            return 0
        if args.command == "status":
            run_root = _safe_run_root(args.run_id, args.campaign_root)
            state = json.loads(
                (run_root / "campaign_state.json").read_text(encoding="utf-8")
            )
            print(json.dumps(_summary(state), indent=2, sort_keys=True))
            return 0
        if args.timeout_multiplier <= 0:
            raise StudyError("timeout multiplier must be positive")
        summary = run_campaign(
            run_id=args.run_id,
            profile=args.profile,
            seed=args.seed,
            resume=args.command == "resume",
            timeout_multiplier=args.timeout_multiplier,
            campaign_root=args.campaign_root,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, StudyError) as exc:
        print(f"IDEA FOUNDRY CAMPAIGN BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
