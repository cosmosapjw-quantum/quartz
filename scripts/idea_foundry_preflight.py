#!/usr/bin/env python3
"""Run the fail-closed pre-ablation Idea Foundry verification bundle."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quartz.experiment_manifest import (  # noqa: E402
    atomic_json_dump,
    file_sha256,
    git_provenance,
)
from quartz.idea_foundry.sequential import (  # noqa: E402
    RUN_ID_PATTERN,
    SequentialCampaignError,
)

PREFLIGHT_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "idea_foundry_preflight"


class PreflightError(RuntimeError):
    """Raised when pre-ablation verification cannot finish cleanly."""


@dataclass(frozen=True)
class PreflightStep:
    name: str
    command: tuple[str, ...]
    expected_returncodes: tuple[int, ...] = (0,)
    environment: tuple[tuple[str, str], ...] = ()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _confined_output_root(raw_root: Path) -> Path:
    root = raw_root.resolve()
    allowed = (REPO_ROOT / "results").resolve()
    try:
        root.relative_to(allowed)
    except ValueError as exc:
        raise PreflightError(
            f"preflight output must remain under {allowed}: {root}"
        ) from exc
    return root


def _run_root(output_root: Path, run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise PreflightError(f"unsafe preflight run id: {run_id!r}")
    root = _confined_output_root(output_root)
    target = (root / run_id).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise PreflightError("preflight run id escapes its output root") from exc
    return target


def _changed_source_hashes() -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise PreflightError(f"git status failed: {proc.stderr.strip()}")
    records: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        raw_path = line[3:]
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        path = REPO_ROOT / raw_path
        record: dict[str, Any] = {"path": raw_path}
        if path.is_file() and not path.is_symlink():
            record.update(sha256=file_sha256(path), size_bytes=path.stat().st_size)
        else:
            record["state"] = "missing_or_non_regular"
        records.append(record)
    return sorted(records, key=lambda row: str(row["path"]))


def verify_import_receipt() -> dict[str, Any]:
    receipt_path = REPO_ROOT / "docs" / "idea_foundry" / "IMPORT_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    archive_rows = receipt.get("archives")
    if not isinstance(archive_rows, list) or len(archive_rows) != 2:
        raise PreflightError("import receipt must contain the ZIP and patch")
    for row in archive_rows:
        path = REPO_ROOT / str(row.get("path"))
        if not path.is_file() or path.is_symlink():
            raise PreflightError(f"import archive is missing or non-regular: {path}")
        if file_sha256(path) != row.get("sha256"):
            raise PreflightError(f"import archive hash mismatch: {path}")
    payload_manifest = REPO_ROOT / receipt["baseline_verification"]["payload_manifest"]
    payloads = [
        line.strip()
        for line in payload_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_count = int(receipt["baseline_verification"]["payload_file_count"])
    if len(payloads) != expected_count or len(payloads) != len(set(payloads)):
        raise PreflightError("payload manifest count or uniqueness mismatch")
    zip_path = REPO_ROOT / "quartz_idea_foundry_skeleton.zip"
    with zipfile.ZipFile(zip_path) as archive:
        zip_payloads = sorted(
            name for name in archive.namelist() if not name.endswith("/")
        )
        if zip_payloads != sorted(payloads):
            raise PreflightError("ZIP payloads differ from BUNDLE_FILE_LIST.txt")
        commit = str(receipt["applied_commit"]["commit"])
        for payload_path in payloads:
            proc = subprocess.run(
                ["git", "show", f"{commit}:{payload_path}"],
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0 or proc.stdout != archive.read(payload_path):
                raise PreflightError(
                    f"applied commit payload differs from ZIP: {payload_path}"
                )
    return {
        "status": "VALIDATED_IMPORT_PROVENANCE_ONLY",
        "receipt": str(receipt_path.relative_to(REPO_ROOT)),
        "receipt_sha256": file_sha256(receipt_path),
        "payload_count": len(payloads),
        "archives": archive_rows,
        "claim_scope": "import provenance only",
    }


def build_steps(
    *, python: Path, run_root: Path, mode: str
) -> tuple[PreflightStep, ...]:
    python_cmd = str(python)
    venv_ruff = str(python.parent / "ruff")
    changed_python_paths = tuple(
        str(row["path"])
        for row in _changed_source_hashes()
        if str(row["path"]).endswith(".py") and "sha256" in row
    )
    if not changed_python_paths:
        raise PreflightError("preflight found no changed Python files to lint")
    artifacts = run_root / "artifacts"
    sequential_root = artifacts / "sequential"
    sequential_run = sequential_root / "first-gate-all"
    idea_lab_root = artifacts / "idea_lab"
    phase15_root = artifacts / "phase15_ci_gate"
    a18_inspect_root = artifacts / "a18_study_inspect"
    targeted_tests = (
        "tests/test_idea_foundry_preflight_matrix.py",
        "tests/test_idea_foundry_axis_workflows.py",
        "tests/test_idea_foundry_meta_analysis.py",
        "tests/test_idea_foundry_contracts_v2.py",
        "tests/test_idea_foundry_registry_v2.py",
        "tests/test_idea_foundry_skeletons.py",
        "tests/test_idea_lab.py",
        "tests/test_idea_lab_v2.py",
        "tests/test_a15_matched_service_curve.py",
        "tests/test_a18_evaluator_ablation.py",
        "tests/test_a19_ablation_readiness.py",
    )
    quick_steps = (
        PreflightStep(
            "ruff-check",
            (venv_ruff, "check", *changed_python_paths),
        ),
        PreflightStep(
            "targeted-preflight-matrix",
            (python_cmd, "-m", "pytest", "-q", *targeted_tests),
        ),
        PreflightStep(
            "idea-foundry-plan",
            (python_cmd, "scripts/idea_foundry_run_all.py", "plan", "--json"),
        ),
        PreflightStep(
            "idea-lab-first-gate-plan",
            (
                python_cmd,
                "scripts/idea_lab.py",
                "plan",
                "--config",
                "configs/idea_lab.local.v2.json",
                "--suite",
                "first-gate-all",
                "--profile",
                "cpu",
                "--python",
                python_cmd,
                "--json",
            ),
        ),
        PreflightStep(
            "cuda-doctor",
            (
                python_cmd,
                "scripts/idea_lab.py",
                "doctor",
                "--config",
                "configs/idea_lab.local.v2.json",
                "--profile",
                "cuda",
                "--python",
                python_cmd,
                "--strict",
                "--json",
            ),
        ),
        PreflightStep(
            "ablation-readiness-plan",
            (
                python_cmd,
                "scripts/idea_lab.py",
                "plan",
                "--config",
                "configs/idea_lab.local.v2.json",
                "--suite",
                "a15-a19-ablation-readiness",
                "--profile",
                "cuda",
                "--python",
                python_cmd,
                "--json",
            ),
        ),
        PreflightStep(
            "live-promotion-remains-blocked",
            (
                python_cmd,
                "scripts/idea_lab.py",
                "plan",
                "--config",
                "configs/idea_lab.local.v2.json",
                "--suite",
                "live-promotion-blocked",
                "--profile",
                "cpu",
                "--python",
                python_cmd,
                "--json",
            ),
            expected_returncodes=(2,),
        ),
        PreflightStep(
            "accelerator-promotion-remains-blocked",
            (
                python_cmd,
                "scripts/idea_lab.py",
                "plan",
                "--config",
                "configs/idea_lab.local.v2.json",
                "--suite",
                "accelerator-promotion-blocked",
                "--profile",
                "cuda",
                "--python",
                python_cmd,
                "--json",
            ),
            expected_returncodes=(2,),
        ),
    )
    if mode == "quick":
        return quick_steps
    release_steps = (
        PreflightStep("cargo-default", ("cargo", "test", "--release", "--locked")),
        PreflightStep(
            "cargo-idea-foundry",
            ("cargo", "test", "--release", "--locked", "--features", "idea-foundry"),
        ),
        PreflightStep(
            "cargo-mcts-demo",
            ("cargo", "build", "--release", "--locked", "--bin", "mcts_demo"),
        ),
        PreflightStep(
            "full-python-regression",
            (
                python_cmd,
                "-m",
                "pytest",
                "-q",
                "tests/",
                "--ignore=tests/test_play_gui.py",
            ),
        ),
        PreflightStep(
            "eager-real-loop",
            (
                python_cmd,
                "-m",
                "pytest",
                "-q",
                "-m",
                "real_loop",
                "tests/test_real_loop_e2e.py",
            ),
            environment=(("QUARTZ_FORCE_EAGER_EVAL", "1"),),
        ),
        PreflightStep(
            "phase15-ci-smoke",
            (
                python_cmd,
                "scripts/phase15_benchmark_ci_smoke.py",
                "--output",
                str(phase15_root),
                "--rust-binary",
                str(REPO_ROOT / "target" / "release" / "mcts_demo"),
            ),
        ),
        PreflightStep(
            "first-gate-sequential-run",
            (
                python_cmd,
                "scripts/idea_foundry_run_all.py",
                "--campaign-root",
                str(sequential_root),
                "run",
                "--run-id",
                "first-gate-all",
                "--seed",
                "20260718",
                "--timeout-seconds",
                "120",
            ),
        ),
        PreflightStep(
            "first-gate-sequential-resume",
            (
                python_cmd,
                "scripts/idea_foundry_run_all.py",
                "--campaign-root",
                str(sequential_root),
                "resume",
                "--run-id",
                "first-gate-all",
                "--seed",
                "20260718",
                "--timeout-seconds",
                "120",
            ),
        ),
        PreflightStep(
            "first-gate-campaign-analysis",
            (
                python_cmd,
                "scripts/idea_foundry_analyze_campaign.py",
                "--campaign-dir",
                str(sequential_run),
            ),
        ),
        PreflightStep(
            "first-gate-meta-analysis",
            (
                python_cmd,
                "scripts/idea_foundry_meta_analyze.py",
                "--input",
                str(sequential_run / "campaign_analysis" / "campaign_analysis.json"),
                "--output-dir",
                str(sequential_run / "meta_analysis"),
            ),
        ),
        PreflightStep(
            "a15-a19-readiness-run",
            (
                python_cmd,
                "scripts/idea_lab.py",
                "run",
                "--config",
                "configs/idea_lab.local.v2.json",
                "--suite",
                "a15-a19-ablation-readiness",
                "--profile",
                "cuda",
                "--python",
                python_cmd,
                "--output-root",
                str(idea_lab_root),
                "--run-id",
                "a15-a19-readiness",
            ),
        ),
        PreflightStep(
            "a15-a19-readiness-resume",
            (
                python_cmd,
                "scripts/idea_lab.py",
                "resume",
                "--config",
                "configs/idea_lab.local.v2.json",
                "--profile",
                "cuda",
                "--python",
                python_cmd,
                "--output-root",
                str(idea_lab_root),
                "--run-id",
                "a15-a19-readiness",
            ),
        ),
        PreflightStep(
            "a18-study-input-inspect",
            (
                python_cmd,
                "scripts/a18_evaluator_ablation.py",
                "--spec",
                "configs/a18_evaluator_ablation.study.v1.json",
                "--output-dir",
                str(a18_inspect_root),
                "--device",
                "cuda",
                "inspect",
            ),
        ),
    )
    return (*quick_steps, *release_steps)


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


def _run_step(
    step: PreflightStep,
    *,
    logs_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    stdout_path = logs_dir / f"{step.name}.stdout.log"
    stderr_path = logs_dir / f"{step.name}.stderr.log"
    started = utc_now()
    monotonic_start = time.monotonic()
    environment = os.environ.copy()
    environment.update(dict(step.environment))
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        proc = subprocess.Popen(
            step.command,
            cwd=REPO_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=environment,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=timeout_seconds)
            process_status = "completed"
        except subprocess.TimeoutExpired:
            _terminate(proc)
            returncode = 124
            process_status = "timeout"
        except KeyboardInterrupt:
            _terminate(proc)
            returncode = 130
            process_status = "interrupted"
    return {
        "name": step.name,
        "command": list(step.command),
        "expected_returncodes": list(step.expected_returncodes),
        "environment_overrides": dict(step.environment),
        "started_at": started,
        "completed_at": utc_now(),
        "duration_seconds": time.monotonic() - monotonic_start,
        "returncode": returncode,
        "process_status": process_status,
        "status": "passed" if returncode in step.expected_returncodes else "failed",
        "stdout": {
            "path": str(stdout_path.relative_to(logs_dir.parent)),
            "sha256": file_sha256(stdout_path),
        },
        "stderr": {
            "path": str(stderr_path.relative_to(logs_dir.parent)),
            "sha256": file_sha256(stderr_path),
        },
    }


def run_preflight(
    *,
    run_id: str,
    output_root: Path,
    python: Path,
    mode: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    run_root = _run_root(output_root, run_id)
    if run_root.exists():
        raise PreflightError(f"preflight output already exists: {run_root}")
    if not python.is_file() or not python.resolve().is_file():
        raise PreflightError(f"target Python launcher is missing: {python}")
    run_root.mkdir(parents=True)
    logs_dir = run_root / "logs"
    logs_dir.mkdir()
    state_path = run_root / "preflight_state.json"
    before_sources = _changed_source_hashes()
    state: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": mode,
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "claim_scope": "pre_ablation_execution_readiness_only",
        "automatic_claim_promotion": False,
        "git_before": git_provenance(REPO_ROOT),
        "changed_source_hashes_before": before_sources,
        "import_receipt": verify_import_receipt(),
        "steps": [],
    }
    atomic_json_dump(state_path, state)
    for step in build_steps(python=python, run_root=run_root, mode=mode):
        row = _run_step(step, logs_dir=logs_dir, timeout_seconds=timeout_seconds)
        state["steps"].append(row)
        state["updated_at"] = utc_now()
        atomic_json_dump(state_path, state)
        if row["status"] != "passed":
            state["status"] = "failed"
            state["failed_step"] = step.name
            state["completed_at"] = utc_now()
            atomic_json_dump(state_path, state)
            raise PreflightError(
                f"preflight step failed: {step.name} (rc={row['returncode']})"
            )
    after_sources = _changed_source_hashes()
    if after_sources != before_sources:
        state["status"] = "failed"
        state["failed_step"] = "worktree-source-stability"
        state["changed_source_hashes_after"] = after_sources
        state["completed_at"] = utc_now()
        atomic_json_dump(state_path, state)
        raise PreflightError(
            "tracked or untracked source files changed during preflight"
        )
    state.update(
        status="passed",
        completed_at=utc_now(),
        changed_source_hashes_after=after_sources,
        git_after=git_provenance(REPO_ROOT),
        readiness={
            "ablation_execution_preflight": "READY",
            "scientific_efficacy": "NOT_EVALUATED",
            "claim_promotion": "FORBIDDEN_AUTOMATICALLY",
        },
    )
    atomic_json_dump(state_path, state)
    report = dict(state)
    report["preflight_state_sha256_before_report"] = file_sha256(state_path)
    atomic_json_dump(run_root / "preflight_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mode", choices=("quick", "release"), default="quick")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)
    try:
        if args.timeout_seconds <= 0:
            raise PreflightError("timeout must be positive")
        report = run_preflight(
            run_id=args.run_id,
            output_root=args.output_root,
            python=args.python,
            mode=args.mode,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (PreflightError, SequentialCampaignError) as exc:
        print(f"IDEA FOUNDRY PREFLIGHT BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
