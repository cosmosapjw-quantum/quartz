"""Executable adapters for the metacontroller 2x2 factorial harness.

The design/analysis contracts live in :mod:`metacontroller_factorial`.  This
module owns the mutating work: a feature-isolated Rust build, paired training
treatments, fixed-opening arena games, and atomically persisted campaign state.
All paths remain below the repository and every successful step carries hashes.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from quartz.experiment_manifest import (
    atomic_json_dump,
    canonical_sha256,
    file_sha256,
    utc_now,
)
from quartz.idea_foundry.axis_workflow import atomic_jsonl_dump
from quartz.idea_foundry.metacontroller_factorial import (
    FactorialHarnessError,
    build_plan,
    load_config,
    load_json_strict,
    run_preflight,
    validate_raw_rows,
    write_analysis,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_SCHEMA_VERSION = 1
RUNTIME_AXIS_ID = "A01"
RUNTIME_BINARY_RELATIVE = "target/idea-foundry-release/release/mcts_demo"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
EXECUTION_SOURCE_PATHS = (
    "quartz/cli_main.py",
    "quartz/evaluator_runtime.py",
    "quartz/runtime_support.py",
    "quartz/selfplay_runtime.py",
    "quartz/idea_foundry/metacontroller_factorial.py",
    "quartz/idea_foundry/metacontroller_factorial_execution.py",
    "scripts/metacontroller_factorial_study.py",
)


def _safe_path(root: Path, raw: str | Path, *, label: str) -> Path:
    candidate = Path(raw)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise FactorialHarnessError(f"{label} escapes repository root: {raw}") from exc
    return resolved


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise FactorialHarnessError(
            "run id must use 1-96 ASCII letters, digits, dot, underscore, or dash"
        )
    return run_id


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise FactorialHarnessError("unable to resolve Git HEAD for campaign contract")
    return value


def _execution_source_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in EXECUTION_SOURCE_PATHS:
        path = _safe_path(root, relative, label="execution source")
        if not path.is_file() or path.is_symlink():
            raise FactorialHarnessError(f"execution source is missing: {relative}")
        hashes[relative] = file_sha256(path)
    return hashes


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FactorialHarnessError(
                f"invalid JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise FactorialHarnessError(
                f"JSONL row is not an object: {path}:{line_number}"
            )
        rows.append(value)
    return rows


def probe_foundry_binary(
    binary: str | Path, *, repo_root: str | Path = REPO_ROOT
) -> dict[str, Any]:
    """Query the server without invoking an NN evaluator."""

    root = Path(repo_root).resolve()
    binary_path = _safe_path(root, binary, label="Rust binary")
    if not binary_path.is_file() or binary_path.is_symlink():
        raise FactorialHarnessError(f"feature Rust binary is missing: {binary_path}")
    env = os.environ.copy()
    env["QUARTZ_DISABLE_QIPC_SHM"] = "1"
    process = subprocess.Popen(
        [str(binary_path), "--server"],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=env,
    )
    try:
        stdout, stderr = process.communicate(
            '{"cmd":"foundry_capabilities"}\n{"cmd":"quit"}\n', timeout=10
        )
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        raise FactorialHarnessError(
            "feature Rust binary capability probe timed out"
        ) from exc
    if process.returncode != 0:
        raise FactorialHarnessError(
            f"feature Rust binary capability probe failed ({process.returncode}): {stderr[-400:]}"
        )
    payload = None
    for line in stdout.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "idea_foundry_feature" in candidate:
            payload = candidate
            break
    if payload is None:
        raise FactorialHarnessError(
            "feature Rust binary emitted no capability document"
        )
    expected = {
        "idea_foundry_feature": True,
        "runtime_adapter": "foundry_search_policy_v1",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise FactorialHarnessError(
                f"feature Rust binary capability {key}={payload.get(key)!r}, expected {value!r}"
            )
    if RUNTIME_AXIS_ID not in payload.get("supported_axis_ids", []):
        raise FactorialHarnessError("feature Rust binary does not advertise A01")
    if set(payload.get("supported_modes", [])) != {"shadow_noop", "active"}:
        raise FactorialHarnessError("feature Rust binary mode inventory is incomplete")
    return {
        **payload,
        "path": _relative(root, binary_path),
        "sha256": file_sha256(binary_path),
    }


def build_feature_binary(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    command = [
        "cargo",
        "build",
        "--release",
        "--locked",
        "--features",
        "idea-foundry",
        "--target-dir",
        "target/idea-foundry-release",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise FactorialHarnessError(
            f"feature Rust build failed with return code {completed.returncode}"
        )
    return probe_foundry_binary(RUNTIME_BINARY_RELATIVE, repo_root=root)


def _feature_search_config(
    base: Mapping[str, Any], *, mode: str, identity: str, risk_limit: float = 0.05
) -> dict[str, Any]:
    if mode not in {"shadow_noop", "active"}:
        raise FactorialHarnessError(f"unsupported Foundry mode: {mode}")
    config = dict(base)
    config.update(
        {
            "check_interval": 1,
            "foundry_mode": mode,
            "foundry_axis_id": RUNTIME_AXIS_ID,
            "foundry_checkpoint_id": identity,
            "foundry_evaluator_id": identity,
            "foundry_risk_limit": float(risk_limit),
            "foundry_min_visits": 16,
        }
    )
    return config


def smoke_runtime_adapter(
    *,
    checkpoint: str | Path,
    rust_binary: str | Path = RUNTIME_BINARY_RELATIVE,
    device: str = "cpu",
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Exercise one identical Gomoku7 root in shadow and active modes.

    The permissive risk limit is intentional and confined to this mechanism
    smoke: it guarantees an A01 STOP proposal after 16 visits.  The output is
    implementation evidence only and is never eligible for scientific use.
    """

    root = Path(repo_root).resolve()
    checkpoint_path = _safe_path(root, checkpoint, label="smoke checkpoint")
    binary = _safe_path(root, rust_binary, label="feature Rust binary")
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise FactorialHarnessError(f"smoke checkpoint is missing: {checkpoint_path}")
    capability = probe_foundry_binary(binary, repo_root=root)

    import torch

    from quartz.backend import load_torch_state_dict
    from quartz.evaluator_runtime import RustNNEvaluatorEngine
    from quartz.models_torch import AlphaZeroNet
    from quartz.runtime_support import build_training_game_adapter
    from quartz.training_catalog import GAME_CONFIGS

    torch_device = torch.device(device)
    cfg_base = dict(GAME_CONFIGS["gomoku7"])
    cfg_base.update(
        {
            "_name": "gomoku7",
            "iters": 64,
            "search_profile": "quartz",
            "vl_mode": "adaptive",
            "halt_mode": "fixed",
            "n_threads": 1,
            "batch_size": 1,
            "batch_timeout_us": 1,
            "check_interval": 1,
            "seed": 20260722,
        }
    )
    model = AlphaZeroNet(cfg_base).to(torch_device)
    model.load_state_dict(
        load_torch_state_dict(checkpoint_path, torch, map_location=torch_device)
    )
    model.eval()
    identity = file_sha256(checkpoint_path)
    rows = []
    try:
        for mode in ("shadow_noop", "active"):
            cfg = dict(cfg_base)
            cfg.update(
                _feature_search_config({}, mode=mode, identity=identity, risk_limit=1.0)
            )
            engine = RustNNEvaluatorEngine(
                f"mechanism-smoke-{mode}", cfg, model, torch_device, str(binary)
            )
            try:
                game = build_training_game_adapter(dict(cfg_base))
                move, metadata = engine.select_move(game)
            finally:
                engine.reset()
            policy = (metadata.get("controller_summary") or {}).get(
                "search_policy"
            ) or {}
            realized = metadata.get("realized_budget") or {}
            row = {
                "mode": mode,
                "move": int(move),
                "decisions": int(policy.get("metacontroller_decisions") or 0),
                "actions": int(policy.get("metacontroller_actions") or 0),
                "coordination_errors": int(
                    policy.get("metacontroller_coordination_errors") or 0
                ),
                "realized_iterations": int(realized.get("realized_iterations") or 0),
                "evaluator_calls": int(realized.get("evaluator_calls") or 0),
                "stop_reason": str(realized.get("stop_reason") or ""),
                "policy": policy,
            }
            rows.append(row)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    shadow, active = rows
    if shadow["decisions"] <= 0 or shadow["actions"] != 0:
        raise FactorialHarnessError(
            f"shadow mechanism smoke violated observation/no-op contract: {shadow}"
        )
    if active["decisions"] <= 0 or active["actions"] <= 0:
        raise FactorialHarnessError(
            f"active mechanism smoke did not execute A01 STOP: {active}"
        )
    if active["coordination_errors"] or shadow["coordination_errors"]:
        raise FactorialHarnessError("mechanism smoke recorded coordination errors")
    if active["realized_iterations"] >= shadow["realized_iterations"]:
        raise FactorialHarnessError(
            "active mechanism smoke did not reduce root iterations"
        )
    return {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "SMOKE_VALIDATED_NOT_SCIENTIFIC_EVIDENCE",
        "claim_scope": "runtime_adapter_mechanism_only",
        "checkpoint": {
            "path": _relative(root, checkpoint_path),
            "sha256": identity,
        },
        "feature_binary": capability,
        "rows": rows,
        "promotion": {"auto": False, "eligible": False},
    }


def materialize_execution_config(
    *,
    base_config_path: str | Path,
    profile_name: str,
    run_id: str,
    anchor_checkpoint: str | Path,
    repo_root: str | Path = REPO_ROOT,
) -> tuple[Path, dict[str, Any]]:
    """Freeze run-local output templates and anchor identity."""

    root = Path(repo_root).resolve()
    run_id = _validate_run_id(run_id)
    base_path = _safe_path(root, base_config_path, label="base factorial config")
    config = load_config(base_path)
    if profile_name not in config["profiles"]:
        raise FactorialHarnessError(f"unknown profile: {profile_name}")
    anchor = _safe_path(root, anchor_checkpoint, label="anchor checkpoint")
    if not anchor.is_file() or anchor.is_symlink():
        raise FactorialHarnessError(
            f"anchor checkpoint is not a regular file: {anchor}"
        )
    run_root = root / "results" / "metacontroller_factorial_runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    for name in ("off", "on"):
        arm_root = run_root / "training" / name / "seed_{seed}"
        config["training_arms"][name]["checkpoint_template"] = _relative(
            root, arm_root / "latest.pt"
        )
        config["training_arms"][name]["manifest_template"] = _relative(
            root, arm_root / "training_treatment_manifest.json"
        )
    controller = config["controller_contract"]
    controller.update(
        {
            "training_adapter_status": "implemented",
            "runtime_adapter_status": "implemented",
            "production_loop_wired": True,
            "execution_ready": True,
            "active_axis_ids": [RUNTIME_AXIS_ID],
        }
    )
    config["runtime_arms"]["on"]["authorized_axis_ids"] = [RUNTIME_AXIS_ID]
    for name in ("off", "on"):
        config["runtime_arms"][name]["search_config"]["check_interval"] = 1
    config["anchor"].update(
        {
            "checkpoint": _relative(root, anchor),
            "checkpoint_sha256": file_sha256(anchor),
        }
    )
    config["anchor"]["search_config"]["check_interval"] = 1
    resolved_path = run_root / "factorial_config.resolved.json"
    if resolved_path.exists():
        observed = load_json_strict(resolved_path)
        if canonical_sha256(observed) != canonical_sha256(config):
            raise FactorialHarnessError(
                "resolved factorial config drifted for this run id"
            )
    else:
        atomic_json_dump(resolved_path, config)
    # Re-validate the serialized object, not the in-memory draft.
    return resolved_path, load_config(resolved_path)


def _terminate_group(process: subprocess.Popen[Any], sig: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


def _run_logged_process(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
) -> tuple[int, float, str]:
    started = time.monotonic()
    status = "failed"
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_s)
            status = "succeeded" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            _terminate_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _terminate_group(process, signal.SIGKILL)
                process.wait(timeout=5)
            returncode = 124
            status = "timeout"
        except KeyboardInterrupt:
            _terminate_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _terminate_group(process, signal.SIGKILL)
            status = "interrupted"
            returncode = 130
    return returncode, time.monotonic() - started, status


def _training_telemetry(output_dir: Path, *, batch_size: int) -> dict[str, Any]:
    log_rows = _read_json_lines(output_dir / "train_log.jsonl")
    learner_updates = sum(int(row.get("train_steps") or 0) for row in log_rows)
    selfplay_wallclock = sum(
        float(row.get("selfplay_wallclock_s") or 0.0) for row in log_rows
    )
    exact_log_telemetry = bool(log_rows) and all(
        all(
            key in row
            for key in (
                "selfplay_nn_evals",
                "metacontroller_observations",
                "metacontroller_actions",
            )
        )
        for row in log_rows
    )
    if exact_log_telemetry:
        evaluator_calls = sum(int(row["selfplay_nn_evals"]) for row in log_rows)
        decisions = sum(int(row["metacontroller_observations"]) for row in log_rows)
        actions = sum(int(row["metacontroller_actions"]) for row in log_rows)
    else:
        # Backward-compatible fallback for older training outputs.  Replay is
        # capacity bounded, so this branch is not accepted as exact campaign
        # accounting by newly materialized runs.
        evaluator_calls = decisions = actions = 0
        replay_path = output_dir / "replay.npz"
        if replay_path.is_file():
            evaluator_calls, decisions, actions = _telemetry_from_replay(replay_path)
    return {
        "learner_updates": learner_updates,
        "sgd_examples": learner_updates * batch_size,
        "selfplay_nn_evals": evaluator_calls,
        "selfplay_wallclock_s": selfplay_wallclock,
        "metacontroller_observations": decisions,
        "metacontroller_actions": actions,
        "telemetry_source": (
            "iteration_trace_totals"
            if exact_log_telemetry
            else "bounded_replay_fallback"
        ),
    }


def _telemetry_from_replay(replay_path: Path) -> tuple[int, int, int]:
    evaluator_calls = decisions = actions = 0
    if replay_path.is_file():
        import numpy as np

        with np.load(replay_path, allow_pickle=False) as archive:
            metadata_rows = archive.get("metadata_json")
            if metadata_rows is not None:
                for raw in metadata_rows:
                    try:
                        metadata = json.loads(str(raw))
                    except json.JSONDecodeError:
                        continue
                    realized = metadata.get("realized_budget") or {}
                    policy = (metadata.get("controller_summary") or {}).get(
                        "search_policy"
                    ) or {}
                    evaluator_calls += int(realized.get("evaluator_calls") or 0)
                    decisions += int(policy.get("metacontroller_decisions") or 0)
                    actions += int(policy.get("metacontroller_actions") or 0)
    return evaluator_calls, decisions, actions


def run_training_treatments(
    *,
    config_path: str | Path,
    profile_name: str,
    initial_checkpoint_template: str,
    rust_binary: str | Path = RUNTIME_BINARY_RELATIVE,
    iterations: int,
    games_per_iteration: int,
    device: str,
    timeout_s: float,
    repo_root: str | Path = REPO_ROOT,
    state_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train OFF then ON from the same seed-specific initial checkpoint."""

    root = Path(repo_root).resolve()
    config_file = _safe_path(root, config_path, label="factorial config")
    config = load_config(config_file)
    profile = config["profiles"][profile_name]
    controller = config["controller_contract"]
    if not (
        controller["training_adapter_status"] == "implemented"
        and controller["runtime_adapter_status"] == "implemented"
        and controller["production_loop_wired"]
        and controller["execution_ready"]
        and controller["active_axis_ids"] == [RUNTIME_AXIS_ID]
    ):
        raise FactorialHarnessError(
            "training requires a materialized A01 execution-ready config"
        )
    expected_results_root = (
        root / "results" / "metacontroller_factorial_runs"
    ).resolve()
    try:
        config_file.relative_to(expected_results_root)
    except ValueError as exc:
        raise FactorialHarnessError(
            "training config must be materialized under results/metacontroller_factorial_runs"
        ) from exc
    binary = _safe_path(root, rust_binary, label="feature Rust binary")
    capability = probe_foundry_binary(binary, repo_root=root)
    if "{seed}" not in initial_checkpoint_template:
        raise FactorialHarnessError("initial checkpoint template must contain {seed}")
    if iterations < 1 or games_per_iteration < 1 or timeout_s <= 0:
        raise FactorialHarnessError(
            "training iterations, games, and timeout must be positive"
        )

    state_file = (
        _safe_path(root, state_path, label="campaign state")
        if state_path is not None
        else config_file.parent / "campaign_state.json"
    )
    try:
        state_file.relative_to(config_file.parent)
    except ValueError as exc:
        raise FactorialHarnessError(
            "campaign state must remain inside the resolved run directory"
        ) from exc
    state = (
        dict(load_json_strict(state_file))
        if state_file.is_file()
        else {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "status": "running",
            "created_at": utc_now(),
            "config_sha256": file_sha256(config_file),
            "git_head": _git_head(root),
            "execution_source_hashes": _execution_source_hashes(root),
            "feature_binary": capability,
            "steps": {},
        }
    )
    if state.get("config_sha256") != file_sha256(config_file):
        raise FactorialHarnessError("campaign state/config hash mismatch")
    if state.get("git_head") != _git_head(root):
        raise FactorialHarnessError("campaign state/Git HEAD mismatch")
    if state.get("execution_source_hashes") != _execution_source_hashes(root):
        raise FactorialHarnessError("campaign state/source hash mismatch")

    from quartz.training_catalog import GAME_CONFIGS

    base_game_cfg = dict(GAME_CONFIGS[config["game"]])
    batch_size = int(base_game_cfg["batch"])
    base_source = {
        "config_sha256": file_sha256(config_file),
        "rust_binary_sha256": capability["sha256"],
        "git_head": _git_head(root),
        "execution_source_hashes": _execution_source_hashes(root),
        "game": config["game"],
        "profile": profile_name,
        "iterations": iterations,
        "games_per_iteration": games_per_iteration,
        "device": device,
        "no_pipeline": True,
        "no_autotune": True,
    }

    for seed in profile["seeds"]:
        initial = _safe_path(
            root,
            initial_checkpoint_template.format(seed=seed),
            label=f"seed {seed} initial checkpoint",
        )
        if not initial.is_file() or initial.is_symlink():
            raise FactorialHarnessError(f"initial checkpoint is missing: {initial}")
        initial_hash = file_sha256(initial)
        source_fingerprint = canonical_sha256(
            {**base_source, "seed": seed, "initial": initial_hash}
        )
        for arm_name in ("off", "on"):
            step_id = f"train_{arm_name}_s{seed}"
            arm = config["training_arms"][arm_name]
            output_checkpoint = _safe_path(
                root,
                arm["checkpoint_template"].format(seed=seed),
                label=f"{step_id} checkpoint",
            )
            output_dir = output_checkpoint.parent
            manifest_path = _safe_path(
                root,
                arm["manifest_template"].format(seed=seed),
                label=f"{step_id} manifest",
            )
            existing = state["steps"].get(step_id) or {}
            if (
                existing.get("status") in {"succeeded", "completed_no_promotion"}
                and output_checkpoint.is_file()
                and manifest_path.is_file()
                and existing.get("checkpoint_sha256") == file_sha256(output_checkpoint)
                and existing.get("manifest_sha256") == file_sha256(manifest_path)
            ):
                existing_manifest = load_json_strict(manifest_path)
                if existing_manifest.get("source_fingerprint") != source_fingerprint:
                    raise FactorialHarnessError(
                        f"training step {step_id} source fingerprint drift"
                    )
                continue
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FactorialHarnessError(
                    f"refusing to overwrite non-empty training directory: {output_dir}"
                )
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            attempts = list(existing.get("attempts") or [])
            attempt_number = len(attempts) + 1
            attempt_dir = output_dir.with_name(
                f".{output_dir.name}.attempt-{attempt_number:03d}"
            )
            if attempt_dir.exists():
                raise FactorialHarnessError(
                    f"training attempt directory already exists: {attempt_dir}"
                )
            attempt_dir.mkdir(parents=True)
            attempt_checkpoint = attempt_dir / output_checkpoint.name
            attempt_manifest = attempt_dir / manifest_path.name
            identity = f"{config['experiment_id']}:s{seed}:{arm_name}:{initial_hash}"
            override = _feature_search_config(
                config["runtime_arms"][arm_name]["search_config"],
                mode=arm["controller_mode"],
                identity=identity,
            )
            override_path = attempt_dir / "training_overrides.json"
            atomic_json_dump(override_path, override)
            command = [
                sys.executable,
                "-m",
                "quartz.train",
                "--game",
                config["game"],
                "--iterations",
                str(iterations),
                "--games",
                str(games_per_iteration),
                "--device",
                device,
                "--backend",
                "torch",
                "--model",
                str(initial),
                "--output",
                str(attempt_dir),
                "--config",
                str(override_path),
                "--rust-binary",
                str(binary),
                "--seed",
                str(seed),
                "--eval-interval",
                str(iterations + 1),
                "--no-autotune",
                "--no-pipeline",
            ]
            state["steps"][step_id] = {
                **existing,
                "status": "running",
                "started_at": utc_now(),
                "command": command,
                "attempts": attempts,
                "active_attempt": _relative(root, attempt_dir),
            }
            atomic_json_dump(state_file, state)
            returncode, elapsed_s, process_status = _run_logged_process(
                command,
                cwd=root,
                stdout_path=attempt_dir / "stdout.log",
                stderr_path=attempt_dir / "stderr.log",
                timeout_s=timeout_s,
            )
            attempt_record = {
                "attempt": attempt_number,
                "path": _relative(root, attempt_dir),
                "status": process_status,
                "returncode": returncode,
                "elapsed_s": elapsed_s,
                "finished_at": utc_now(),
            }
            attempts.append(attempt_record)
            state["steps"][step_id].update(attempt_record)
            state["steps"][step_id]["attempts"] = attempts
            state["steps"][step_id].pop("active_attempt", None)
            atomic_json_dump(state_file, state)
            if process_status == "interrupted":
                state["status"] = "interrupted"
                atomic_json_dump(state_file, state)
                raise KeyboardInterrupt
            if returncode != 0:
                state["status"] = "blocked"
                atomic_json_dump(state_file, state)
                raise FactorialHarnessError(
                    f"training step {step_id} failed with return code {returncode}"
                )
            if (
                not attempt_checkpoint.is_file()
                or not (attempt_dir / "replay.npz").is_file()
            ):
                raise FactorialHarnessError(
                    f"training step {step_id} omitted required artifacts"
                )
            telemetry = _training_telemetry(attempt_dir, batch_size=batch_size)
            if telemetry["telemetry_source"] != "iteration_trace_totals":
                raise FactorialHarnessError(
                    f"training step {step_id} lacks exact iteration telemetry"
                )
            if telemetry["learner_updates"] <= 0:
                raise FactorialHarnessError(
                    f"training step {step_id} executed no SGD update"
                )
            if telemetry["metacontroller_observations"] <= 0:
                raise FactorialHarnessError(
                    f"training step {step_id} recorded no metacontroller observation"
                )
            if arm_name == "off" and telemetry["metacontroller_actions"] != 0:
                raise FactorialHarnessError(
                    "shadow training arm executed a live action"
                )
            treatment_status = (
                "completed_no_promotion"
                if arm_name == "on" and telemetry["metacontroller_actions"] <= 0
                else "succeeded"
            )
            trajectory_files = [
                attempt_dir / "train_log.jsonl",
                attempt_dir / "replay.npz",
            ]
            manifest = {
                "schema_version": 1,
                "experiment_id": config["experiment_id"],
                "training_seed": seed,
                "training_controller_mode": arm["controller_mode"],
                "checkpoint_path": _relative(root, output_checkpoint),
                "checkpoint_sha256": file_sha256(attempt_checkpoint),
                "initial_checkpoint_sha256": initial_hash,
                "controller_contract_hash": canonical_sha256(
                    {
                        "controller_mode": arm["controller_mode"],
                        "implementation": config["controller_contract"][
                            "implementation"
                        ],
                        "active_axis_ids": config["controller_contract"][
                            "active_axis_ids"
                        ],
                    }
                ),
                "source_fingerprint": source_fingerprint,
                "training_trajectory_sha256": canonical_sha256(
                    {
                        "files": [
                            {"path": path.name, "sha256": file_sha256(path)}
                            for path in trajectory_files
                        ],
                        "treatment": override,
                    }
                ),
                **telemetry,
                "status": treatment_status,
            }
            atomic_json_dump(attempt_manifest, manifest)
            os.replace(attempt_dir, output_dir)
            state["steps"][step_id].update(
                {
                    "status": treatment_status,
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "manifest_sha256": file_sha256(manifest_path),
                    "output_dir": _relative(root, output_dir),
                }
            )
            atomic_json_dump(state_file, state)
    state["status"] = (
        "training_completed_no_promotion"
        if any(
            step.get("status") == "completed_no_promotion"
            for step in state["steps"].values()
        )
        else "training_succeeded"
    )
    state["updated_at"] = utc_now()
    atomic_json_dump(state_file, state)
    return state


def _telemetry_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise FactorialHarnessError("game side has no move telemetry")
    return {
        "nn_evals_per_move": mean(sample["evaluator_calls"] for sample in samples),
        "root_visits_per_move": mean(sample["root_visits"] for sample in samples),
        "wallclock_ms_per_move": mean(sample["wallclock_ms"] for sample in samples),
        "metacontroller_decisions": sum(sample["decisions"] for sample in samples),
        "metacontroller_actions": sum(sample["actions"] for sample in samples),
        "selection_trace_covered": all(sample["trace_covered"] for sample in samples),
    }


def run_arena(
    *,
    config_path: str | Path,
    profile_name: str,
    preflight_path: str | Path,
    output_path: str | Path,
    rust_binary: str | Path = RUNTIME_BINARY_RELATIVE,
    device: str,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Execute every planned game sequentially and checkpoint JSONL rows."""

    root = Path(repo_root).resolve()
    config_file = _safe_path(root, config_path, label="factorial config")
    preflight_file = _safe_path(root, preflight_path, label="preflight")
    output = _safe_path(root, output_path, label="raw row output")
    config = load_config(config_file)
    plan = build_plan(config_file, profile_name, repo_root=root)
    preflight = load_json_strict(preflight_file)
    if preflight.get("ready") is not True or preflight.get("status") != "READY":
        raise FactorialHarnessError("arena requires a READY preflight")
    if preflight.get("plan_contract_hash") != plan["plan_contract_hash"]:
        raise FactorialHarnessError("arena preflight/plan hash mismatch")
    binary = _safe_path(root, rust_binary, label="feature Rust binary")
    probe_foundry_binary(binary, repo_root=root)

    from quartz.backend import load_torch_state_dict
    from quartz.evaluator_runtime import RustNNEvaluatorEngine
    from quartz.models_torch import AlphaZeroNet
    from quartz.runtime_support import build_training_game_adapter
    from quartz.training_catalog import GAME_CONFIGS
    import torch

    torch_device = torch.device(device)
    agents = preflight["resolved_agent_contracts"]
    base_cfg = dict(GAME_CONFIGS[config["game"]])
    base_cfg["_name"] = config["game"]
    model_cache: dict[str, Any] = {}
    engines: dict[str, Any] = {}
    for agent_id, agent in agents.items():
        checkpoint = _safe_path(
            root, agent["checkpoint_path"], label=f"{agent_id} checkpoint"
        )
        checkpoint_hash = agent["checkpoint_sha256"]
        model = model_cache.get(checkpoint_hash)
        if model is None:
            model = AlphaZeroNet(base_cfg).to(torch_device)
            model.load_state_dict(
                load_torch_state_dict(checkpoint, torch, map_location=torch_device)
            )
            model.eval()
            model_cache[checkpoint_hash] = model
        runtime = agent["runtime_contract"]
        cfg = dict(base_cfg)
        cfg.update(runtime["search_config"])
        cfg.update(
            _feature_search_config(
                {}, mode=runtime["controller_mode"], identity=checkpoint_hash
            )
        )
        engines[agent_id] = RustNNEvaluatorEngine(
            agent_id, cfg, model, torch_device, str(binary)
        )

    existing_rows = _read_json_lines(output)
    if existing_rows:
        validated_prefix = validate_raw_rows(
            existing_rows,
            plan=plan,
            preflight=preflight,
            require_complete=False,
            require_treatment_delivery=False,
        )
        existing_rows = validated_prefix["rows"]
    by_id = {row["game_id"]: row for row in existing_rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for game_spec in plan["games"]:
            if game_spec["game_id"] in by_id:
                continue
            game = build_training_game_adapter(dict(base_cfg))
            for move in game_spec["opening_moves"]:
                if game.is_terminal() or move not in game.legal_moves():
                    raise FactorialHarnessError(
                        f"opening {game_spec['opening_id']} is illegal at move {move}"
                    )
                game.apply_move(move)
            telemetry: dict[str, list[dict[str, Any]]] = {"black": [], "white": []}
            while not game.is_terminal():
                side = "black" if game.current_player() == 0 else "white"
                agent_id = game_spec[f"{side}_agent_id"]
                engine = engines[agent_id]
                engine._cfg["seed"] = int(game_spec["eval_seed"]) + sum(
                    len(values) for values in telemetry.values()
                )
                started = time.perf_counter()
                move, metadata = engine.select_move(game)
                wallclock_ms = (time.perf_counter() - started) * 1000.0
                if move not in game.legal_moves():
                    raise FactorialHarnessError(
                        f"engine {agent_id} returned illegal move {move}"
                    )
                realized = metadata.get("realized_budget") or {}
                policy = (metadata.get("controller_summary") or {}).get("search_policy")
                expected_mode = agents[agent_id]["runtime_contract"]["controller_mode"]
                trace_covered = (
                    isinstance(policy, Mapping)
                    and policy.get("foundry_mode") == expected_mode
                    and policy.get("foundry_axis_ids") == ["A01.stop_council"]
                )
                telemetry[side].append(
                    {
                        "evaluator_calls": float(
                            realized.get("evaluator_calls") or 0.0
                        ),
                        "root_visits": float(
                            realized.get("realized_iterations") or 0.0
                        ),
                        "wallclock_ms": wallclock_ms,
                        "decisions": int(
                            (policy or {}).get("metacontroller_decisions") or 0
                        ),
                        "actions": int(
                            (policy or {}).get("metacontroller_actions") or 0
                        ),
                        "trace_covered": trace_covered,
                    }
                )
                game.apply_move(move)
            outcome = float(game.outcome_for_black() or 0.0)
            row = {
                "schema_version": 1,
                **{
                    key: game_spec[key]
                    for key in (
                        "game_id",
                        "block_id",
                        "design_role",
                        "training_seed",
                        "opening_id",
                        "opening_group_id",
                        "eval_seed",
                        "black_agent_id",
                        "white_agent_id",
                    )
                },
                "black_agent_contract_hash": agents[game_spec["black_agent_id"]][
                    "agent_contract_hash"
                ],
                "white_agent_contract_hash": agents[game_spec["white_agent_id"]][
                    "agent_contract_hash"
                ],
                "status": "scored",
                "score_black": 0.5 * (outcome + 1.0),
                "black_telemetry": _telemetry_summary(telemetry["black"]),
                "white_telemetry": _telemetry_summary(telemetry["white"]),
            }
            existing_rows.append(row)
            by_id[row["game_id"]] = row
            atomic_jsonl_dump(output, existing_rows)
    finally:
        for engine in engines.values():
            engine.reset()
        model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    validated = validate_raw_rows(
        existing_rows,
        plan=plan,
        preflight=preflight,
        require_treatment_delivery=False,
    )
    treatment_status = (
        "succeeded" if validated["treatment_delivered"] else "completed_no_promotion"
    )
    return {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": treatment_status,
        "game_count": len(existing_rows),
        "expected_game_count": len(plan["games"]),
        "treatment_delivered": validated["treatment_delivered"],
        "active_action_counts": validated["active_action_counts"],
        "output": _relative(root, output),
        "sha256": file_sha256(output),
    }


def prepare_preflight_artifact(
    *,
    config_path: str | Path,
    profile_name: str,
    output_path: str | Path,
    rust_binary: str | Path = RUNTIME_BINARY_RELATIVE,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    payload = run_preflight(config_path, profile_name, repo_root=root)
    try:
        payload["feature_binary"] = probe_foundry_binary(rust_binary, repo_root=root)
    except FactorialHarnessError as exc:
        payload["blockers"].append(
            {"code": "FEATURE_BINARY_INVALID", "message": str(exc), "details": {}}
        )
        payload["ready"] = False
        payload["status"] = "BLOCKED"
        payload["counts"]["blocker_count"] = len(payload["blockers"])
    output = _safe_path(root, output_path, label="preflight output")
    atomic_json_dump(output, payload)
    return payload


def _load_or_prepare_preflight(
    *,
    config_path: Path,
    profile_name: str,
    output_path: Path,
    rust_binary: str | Path,
    root: Path,
) -> dict[str, Any]:
    if not output_path.exists():
        return prepare_preflight_artifact(
            config_path=config_path,
            profile_name=profile_name,
            output_path=output_path,
            rust_binary=rust_binary,
            repo_root=root,
        )
    existing = dict(load_json_strict(output_path))
    current = run_preflight(config_path, profile_name, repo_root=root)
    try:
        current["feature_binary"] = probe_foundry_binary(rust_binary, repo_root=root)
    except FactorialHarnessError as exc:
        current["blockers"].append(
            {"code": "FEATURE_BINARY_INVALID", "message": str(exc), "details": {}}
        )
        current["ready"] = False
        current["status"] = "BLOCKED"
        current["counts"]["blocker_count"] = len(current["blockers"])
    existing_contract = {
        key: value for key, value in existing.items() if key != "created_at"
    }
    current_contract = {
        key: value for key, value in current.items() if key != "created_at"
    }
    if canonical_sha256(existing_contract) != canonical_sha256(current_contract):
        raise FactorialHarnessError(
            "existing preflight artifact drifted from live inputs"
        )
    return existing


def _campaign_root(root: Path, run_id: str) -> Path:
    return root / "results" / "metacontroller_factorial_runs" / _validate_run_id(run_id)


def _campaign_paths(root: Path, run_id: str) -> dict[str, Path]:
    run_root = _campaign_root(root, run_id)
    return {
        "root": run_root,
        "invocation": run_root / "campaign_invocation.json",
        "config": run_root / "factorial_config.resolved.json",
        "state": run_root / "campaign_state.json",
        "preflight": run_root / "preflight.json",
        "rows": run_root / "games.jsonl",
        "analysis": run_root / "analysis",
        "summary": run_root / "campaign_summary.json",
    }


def _artifact_descriptor(root: Path, path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": _relative(root, path),
        "exists": path.is_file() and not path.is_symlink(),
    }
    if payload["exists"]:
        payload.update({"sha256": file_sha256(path), "size_bytes": path.stat().st_size})
    return payload


def campaign_status(
    run_id: str, *, repo_root: str | Path = REPO_ROOT
) -> dict[str, Any]:
    """Return a read-only, hash-bearing campaign snapshot."""

    root = Path(repo_root).resolve()
    paths = _campaign_paths(root, run_id)
    if not paths["root"].is_dir() or paths["root"].is_symlink():
        raise FactorialHarnessError(f"campaign run does not exist: {run_id}")
    state = load_json_strict(paths["state"]) if paths["state"].is_file() else None
    summary = load_json_strict(paths["summary"]) if paths["summary"].is_file() else None
    artifacts = {
        key: _artifact_descriptor(root, path)
        for key, path in paths.items()
        if key not in {"root", "analysis"}
    }
    analysis_manifest = paths["analysis"] / "analysis_manifest.json"
    artifacts["analysis_manifest"] = _artifact_descriptor(root, analysis_manifest)
    return {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "run_id": run_id,
        "status": (summary or state or {}).get("status", "planned"),
        "run_root": _relative(root, paths["root"]),
        "state": state,
        "summary": summary,
        "artifacts": artifacts,
    }


def _freeze_invocation(
    *,
    run_id: str,
    base_config_path: Path,
    profile_name: str,
    anchor_checkpoint: Path,
    initial_checkpoint_template: str,
    rust_binary: Path,
    iterations: int,
    games_per_iteration: int,
    device: str,
    timeout_s: float,
    root: Path,
) -> dict[str, Any]:
    if "{seed}" not in initial_checkpoint_template:
        raise FactorialHarnessError("initial checkpoint template must contain {seed}")
    invocation = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "run_id": run_id,
        "base_config": _relative(root, base_config_path),
        "base_config_sha256": file_sha256(base_config_path),
        "profile": profile_name,
        "anchor_checkpoint": _relative(root, anchor_checkpoint),
        "anchor_checkpoint_sha256": file_sha256(anchor_checkpoint),
        "initial_checkpoint_template": initial_checkpoint_template,
        "rust_binary": _relative(root, rust_binary),
        "rust_binary_sha256": file_sha256(rust_binary),
        "iterations": iterations,
        "games_per_iteration": games_per_iteration,
        "device": device,
        "timeout_s": timeout_s,
        "python_executable": str(Path(sys.executable).resolve()),
        "git_head": _git_head(root),
        "execution_source_hashes": _execution_source_hashes(root),
    }
    invocation["contract_sha256"] = canonical_sha256(invocation)
    return invocation


def _validate_invocation(invocation: Mapping[str, Any], root: Path) -> dict[str, Any]:
    required = {
        "schema_version",
        "run_id",
        "base_config",
        "base_config_sha256",
        "profile",
        "anchor_checkpoint",
        "anchor_checkpoint_sha256",
        "initial_checkpoint_template",
        "rust_binary",
        "rust_binary_sha256",
        "iterations",
        "games_per_iteration",
        "device",
        "timeout_s",
        "python_executable",
        "git_head",
        "execution_source_hashes",
        "contract_sha256",
    }
    if set(invocation) != required:
        raise FactorialHarnessError("campaign invocation schema mismatch")
    payload = dict(invocation)
    expected_contract = payload.pop("contract_sha256")
    if canonical_sha256(payload) != expected_contract:
        raise FactorialHarnessError("campaign invocation contract hash mismatch")
    if payload["schema_version"] != EXECUTION_SCHEMA_VERSION:
        raise FactorialHarnessError("campaign invocation version mismatch")
    _validate_run_id(str(payload["run_id"]))
    for path_key, hash_key in (
        ("base_config", "base_config_sha256"),
        ("anchor_checkpoint", "anchor_checkpoint_sha256"),
        ("rust_binary", "rust_binary_sha256"),
    ):
        path = _safe_path(root, payload[path_key], label=path_key)
        if not path.is_file() or path.is_symlink():
            raise FactorialHarnessError(f"campaign input is missing: {path_key}")
        if file_sha256(path) != payload[hash_key]:
            raise FactorialHarnessError(f"campaign input hash drift: {path_key}")
    if payload["python_executable"] != str(Path(sys.executable).resolve()):
        raise FactorialHarnessError("campaign Python interpreter drift")
    if payload["git_head"] != _git_head(root):
        raise FactorialHarnessError("campaign Git HEAD drift")
    if payload["execution_source_hashes"] != _execution_source_hashes(root):
        raise FactorialHarnessError("campaign execution source hash drift")
    return dict(invocation)


def _load_existing_analysis(*, analysis_dir: Path, root: Path) -> dict[str, Any] | None:
    manifest_path = analysis_dir / "analysis_manifest.json"
    analysis_path = analysis_dir / "analysis.json"
    if not analysis_dir.exists():
        return None
    if not manifest_path.is_file() or not analysis_path.is_file():
        raise FactorialHarnessError("partial analysis directory cannot be resumed")
    manifest = load_json_strict(manifest_path)
    for artifact in manifest.get("artifacts", []):
        path = _safe_path(analysis_dir, artifact["path"], label="analysis artifact")
        if not path.is_file() or file_sha256(path) != artifact["sha256"]:
            raise FactorialHarnessError("analysis artifact hash mismatch")
    for source in manifest.get("inputs", []):
        path = Path(source["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise FactorialHarnessError("analysis input escapes repository") from exc
        if not path.is_file() or file_sha256(path) != source["sha256"]:
            raise FactorialHarnessError("analysis input hash mismatch")
    return dict(load_json_strict(analysis_path))


def _campaign_summary(
    *,
    root: Path,
    paths: Mapping[str, Path],
    status: str,
    stage: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    summary = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": status,
        "stage": stage,
        "updated_at": utc_now(),
        "claim_scope": "factorial_execution_and_analysis_diagnostic_only",
        "details": dict(details),
        "artifacts": {
            key: _artifact_descriptor(root, path)
            for key, path in paths.items()
            if key not in {"root", "analysis", "summary"}
        },
        "promotion": {"auto": False, "eligible": False},
    }
    analysis_manifest = paths["analysis"] / "analysis_manifest.json"
    summary["artifacts"]["analysis_manifest"] = _artifact_descriptor(
        root, analysis_manifest
    )
    atomic_json_dump(paths["summary"], summary)
    return summary


def _execute_campaign(
    invocation: Mapping[str, Any], *, repo_root: str | Path = REPO_ROOT
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    frozen = _validate_invocation(invocation, root)
    run_id = str(frozen["run_id"])
    paths = _campaign_paths(root, run_id)
    config_path = paths["config"]

    try:
        training = run_training_treatments(
            config_path=config_path,
            profile_name=str(frozen["profile"]),
            initial_checkpoint_template=str(frozen["initial_checkpoint_template"]),
            rust_binary=str(frozen["rust_binary"]),
            iterations=int(frozen["iterations"]),
            games_per_iteration=int(frozen["games_per_iteration"]),
            device=str(frozen["device"]),
            timeout_s=float(frozen["timeout_s"]),
            repo_root=root,
            state_path=paths["state"],
        )
        preflight = _load_or_prepare_preflight(
            config_path=config_path,
            profile_name=str(frozen["profile"]),
            output_path=paths["preflight"],
            rust_binary=str(frozen["rust_binary"]),
            root=root,
        )
        if not preflight["ready"]:
            return _campaign_summary(
                root=root,
                paths=paths,
                status="COMPLETED_NO_PROMOTION",
                stage="preflight",
                details={
                    "training_status": training["status"],
                    "blockers": preflight["blockers"],
                },
            )
        arena = run_arena(
            config_path=config_path,
            profile_name=str(frozen["profile"]),
            preflight_path=paths["preflight"],
            output_path=paths["rows"],
            rust_binary=str(frozen["rust_binary"]),
            device=str(frozen["device"]),
            repo_root=root,
        )
        if arena["status"] == "completed_no_promotion":
            return _campaign_summary(
                root=root,
                paths=paths,
                status="COMPLETED_NO_PROMOTION",
                stage="arena",
                details={"training_status": training["status"], "arena": arena},
            )
        analysis = _load_existing_analysis(analysis_dir=paths["analysis"], root=root)
        if analysis is None:
            analysis = write_analysis(
                config_path=config_path,
                profile_name=str(frozen["profile"]),
                preflight_path=paths["preflight"],
                raw_rows_path=paths["rows"],
                output_dir=paths["analysis"],
                repo_root=root,
            )
        return _campaign_summary(
            root=root,
            paths=paths,
            status=str(analysis["status"]),
            stage="analysis",
            details={
                "training_status": training["status"],
                "arena": arena,
                "analysis_status": analysis["status"],
            },
        )
    except KeyboardInterrupt:
        return _campaign_summary(
            root=root,
            paths=paths,
            status="interrupted",
            stage="campaign",
            details={"reason": "keyboard_interrupt"},
        )
    except FactorialHarnessError as exc:
        _campaign_summary(
            root=root,
            paths=paths,
            status="blocked",
            stage="campaign",
            details={"reason": str(exc)},
        )
        raise


def start_factorial_campaign(
    *,
    base_config_path: str | Path,
    profile_name: str,
    run_id: str,
    anchor_checkpoint: str | Path,
    initial_checkpoint_template: str,
    rust_binary: str | Path = RUNTIME_BINARY_RELATIVE,
    iterations: int,
    games_per_iteration: int,
    device: str,
    timeout_s: float,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    paths = _campaign_paths(root, run_id)
    base_config = _safe_path(root, base_config_path, label="base factorial config")
    anchor = _safe_path(root, anchor_checkpoint, label="anchor checkpoint")
    binary = _safe_path(root, rust_binary, label="feature Rust binary")
    for label, path in (
        ("base factorial config", base_config),
        ("anchor checkpoint", anchor),
        ("feature Rust binary", binary),
    ):
        if not path.is_file() or path.is_symlink():
            raise FactorialHarnessError(f"{label} is not a regular file: {path}")
    if iterations < 1 or games_per_iteration < 1 or timeout_s <= 0:
        raise FactorialHarnessError("campaign numeric limits must be positive")
    materialize_execution_config(
        base_config_path=base_config,
        profile_name=profile_name,
        run_id=run_id,
        anchor_checkpoint=anchor,
        repo_root=root,
    )
    invocation = _freeze_invocation(
        run_id=run_id,
        base_config_path=base_config,
        profile_name=profile_name,
        anchor_checkpoint=anchor,
        initial_checkpoint_template=initial_checkpoint_template,
        rust_binary=binary,
        iterations=iterations,
        games_per_iteration=games_per_iteration,
        device=device,
        timeout_s=timeout_s,
        root=root,
    )
    if paths["invocation"].exists():
        observed = load_json_strict(paths["invocation"])
        if canonical_sha256(observed) != canonical_sha256(invocation):
            raise FactorialHarnessError("campaign invocation drifted for this run id")
    else:
        atomic_json_dump(paths["invocation"], invocation)
    return _execute_campaign(invocation, repo_root=root)


def resume_factorial_campaign(
    run_id: str, *, repo_root: str | Path = REPO_ROOT
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    paths = _campaign_paths(root, run_id)
    if not paths["invocation"].is_file():
        raise FactorialHarnessError("campaign invocation is missing")
    invocation = load_json_strict(paths["invocation"])
    if invocation.get("run_id") != run_id:
        raise FactorialHarnessError("campaign invocation run id mismatch")
    return _execute_campaign(invocation, repo_root=root)
